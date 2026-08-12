"""Tests for the FastAPI service.

The invariants worth testing here are architectural, not CRUD:

  - the perception boundary is enforced by the INTERFACE, not by client discipline
    (there must be no endpoint that accepts pixels)
  - `/report` cannot emit an unverified claim, even when the generator lies
  - `/report` refuses to invent a report for a day that was never analysed

Everything else (row counts, status codes) is incidental. Run:
    python tests/test_service.py
"""

from __future__ import annotations

import sys
import tempfile
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    print("SKIP: fastapi not installed (pip install -r requirements-serve.txt)")
    sys.exit(0)

from behaviorsense.data.simulator import BehaviourSimulator, PERSONAS  # noqa: E402
from behaviorsense.schemas import Role  # noqa: E402
from behaviorsense.service import api  # noqa: E402

D0 = date(2026, 2, 1)


def fresh_client(tmp: Path) -> TestClient:
    """Point the service at a throwaway DB and clear in-memory analyser state.

    The redirect is ASSERTED, not assumed. `connect()` used to bind DB_PATH as a default
    argument, so this reassignment did nothing and every test wrote to the real
    data/behaviorsense.db - undetected for as long as row counts stayed under the page
    size. A fixture that silently fails to isolate makes every test above it vacuous.
    """
    api.DB_PATH = tmp / "test.db"
    api._analyzers.clear()
    api._states.clear()
    with closing(api.connect()) as conn:
        landed = Path(conn.execute("PRAGMA database_list").fetchone()["file"]).resolve()
    assert landed == api.DB_PATH.resolve(), (
        f"service opened {landed}, not the temp DB {api.DB_PATH.resolve()} - tests are "
        "writing to the real database and are not isolated")
    return TestClient(api.app)


def seed_days(client: TestClient, n: int = 24) -> list[date]:
    """Analyse n simulated days through the HTTP layer."""
    sim = BehaviourSimulator(PERSONAS["regular_margaret"], seed=0)
    result = sim.generate(start=D0, n_days=n, anomalies=())
    days = []
    for feats in result.days:
        r = client.post(
            f"/analyze/day?day={feats.day.isoformat()}&role=resident",
            json=feats.model_dump(mode="json"),
        )
        assert r.status_code == 200, r.text
        days.append(feats.day)
    return days


def test_x1_health_and_no_pixel_endpoint():
    """No endpoint may accept image or video data.

    The "LLM never sees pixels" claim must hold because the interface makes it
    impossible, not because callers are careful. If a /ingest/video route ever appears,
    the claim silently becomes a convention.
    """
    with tempfile.TemporaryDirectory() as td:
        client = fresh_client(Path(td))
        assert client.get("/health").json()["status"] == "ok"

        paths = client.get("/openapi.json").json()["paths"]
        banned = ("video", "image", "frame_bytes", "upload", "pixels")
        for p in paths:
            for word in banned:
                assert word not in p.lower(), f"endpoint {p} accepts raw media"

        # /ingest/frames must take FrameObservation (structured), not bytes.
        body = paths["/ingest/frames"]["post"]["requestBody"]["content"]
        assert "application/json" in body, "frames endpoint accepts non-JSON payloads"
        assert "multipart/form-data" not in body
        print(f"  X1 {len(paths)} endpoints, none accept raw media; "
              f"/ingest/frames is JSON-only")


def test_x2_analyze_then_report_is_verified():
    with tempfile.TemporaryDirectory() as td:
        client = fresh_client(Path(td))
        days = seed_days(client, 24)

        r = client.post(f"/report?day={days[-1].isoformat()}&role=resident")
        assert r.status_code == 200, r.text
        rep = r.json()
        assert rep["hallucination_rate"] == 0.0, rep
        assert rep["withheld_claims"] == 0
        assert rep["verified_claims"], "no claims returned"
        for c in rep["verified_claims"]:
            assert c["evidence_ref"].startswith("feat:")
        print(f"  X2 24 days analysed -> report with {len(rep['verified_claims'])} "
              f"verified claims, {rep['withheld_claims']} withheld, "
              f"rate {rep['hallucination_rate']:.1%}")


def test_x3_report_never_returns_unfaithful_claims():
    """Swap in a lying generator: the endpoint must withhold, not warn.

    This is the test that matters. A response that included false claims with a
    `verified: false` flag would be worse than useless - a caveat beside confident prose
    is not protection, because the prose is what gets remembered.
    """
    from behaviorsense.agents.reasoning.reporter import (
        CaregiverReporter,
        HallucinatingStubLLM,
        ReporterConfig,
    )

    with tempfile.TemporaryDirectory() as td:
        client = fresh_client(Path(td))
        days = seed_days(client, 24)

        original = api._reporter
        try:
            api._reporter = CaregiverReporter(
                HallucinatingStubLLM(rate=1.0, seed=2), config=ReporterConfig()
            )
            r = client.post(f"/report?day={days[-1].isoformat()}&role=resident")
            assert r.status_code == 200, r.text
            rep = r.json()
            assert rep["verified_claims"] == [], (
                f"unverified claims leaked to the caller: {rep['verified_claims']}"
            )
            assert rep["withheld_claims"] > 0
            assert rep["hallucination_rate"] == 1.0
            print(f"  X3 fully-corrupted generator -> 0 claims returned, "
                  f"{rep['withheld_claims']} withheld, rate 100%")
        finally:
            api._reporter = original


def test_x4_report_refuses_unanalysed_day():
    with tempfile.TemporaryDirectory() as td:
        client = fresh_client(Path(td))
        seed_days(client, 10)
        future = (D0 + timedelta(days=99)).isoformat()
        r = client.post(f"/report?day={future}&role=resident")
        assert r.status_code == 404, f"invented a report for an unanalysed day: {r.text}"
        assert "analyze/day" in r.json()["detail"]
        print("  X4 unanalysed day -> 404 with a pointer to /analyze/day")


def test_x5_alerts_and_stats_round_trip():
    with tempfile.TemporaryDirectory() as td:
        client = fresh_client(Path(td))
        sim = BehaviourSimulator(PERSONAS["regular_margaret"], seed=1)
        from behaviorsense.data.simulator import AnomalyKind, InjectedAnomaly

        result = sim.generate(
            start=D0, n_days=40,
            anomalies=(InjectedAnomaly(kind=AnomalyKind.FALL_EVENT,
                                       onset_day=D0 + timedelta(days=30)),),
        )
        for feats in result.days:
            client.post(
                f"/analyze/day?day={feats.day.isoformat()}&role=resident",
                json=feats.model_dump(mode="json"),
            )

        alerts = client.get("/alerts?limit=50").json()
        assert alerts, "injected fall produced no alerts"
        assert all("kind" in a and "evidence" in a for a in alerts)

        s = client.get("/stats").json()
        assert s["days_analysed"] == 40, s
        # Compare against an unpaginated read: /alerts?limit=50 is a PAGE, and asserting
        # a total equals a page silently passes only while the total stays under it.
        every = client.get("/alerts?limit=10000").json()
        assert s["alerts"] == len(every), (
            f"stats says {s['alerts']} alerts, full listing has {len(every)}")
        assert len(alerts) == min(50, len(every)), (
            f"limit=50 returned {len(alerts)}, expected {min(50, len(every))}")
        # A fresh DB must hold only this test's rows.
        assert len(every) < 200, (
            f"{len(every)} alerts from 40 simulated days - implausible, and the signature "
            "of a leaked database carrying prior runs")
        print(f"  X5 40 days -> {s['alerts']} alerts persisted "
              f"(kinds: {sorted({a['kind'] for a in alerts})})")


def test_x6_segments_ingest_and_aggregate():
    """Segments in -> features derived server-side, no client-supplied features."""
    from datetime import datetime

    from behaviorsense.schemas import ActivitySegment

    with tempfile.TemporaryDirectory() as td:
        client = fresh_client(Path(td))
        day = date(2026, 3, 2)
        base = datetime(2026, 3, 2, 8, 0, 0)
        segs = []
        for i, (name, aid, mins) in enumerate([
            ("walking", 0, 10), ("eating", 9, 30), ("sitting", 2, 60),
            ("walking", 0, 12), ("eating", 9, 25),
        ]):
            start = base + timedelta(minutes=i * 90)
            segs.append(ActivitySegment(
                segment_id=f"s{i}", track_id=0, role=Role.RESIDENT,
                activity_id=aid, activity_name=name,
                start_time=start, end_time=start + timedelta(minutes=mins),
                confidence=0.9, room="kitchen", n_windows=mins,
            ))
        r = client.post("/ingest/segments",
                        json=[s.model_dump(mode="json") for s in segs])
        assert r.status_code == 200 and r.json()["stored"] == 5, r.text

        r = client.post(f"/analyze/day?day={day.isoformat()}&role=resident")
        assert r.status_code == 200, r.text
        feats = r.json()["features"]
        assert feats["meal_events"] == 2, feats["meal_events"]
        assert feats["walking_duration_s"] == 22 * 60, feats["walking_duration_s"]
        print(f"  X6 5 segments -> meals={feats['meal_events']}, "
              f"walking={feats['walking_duration_s']:.0f}s aggregated server-side")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            print(f"{fn.__name__}:")
            fn()
            print("  PASS")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
