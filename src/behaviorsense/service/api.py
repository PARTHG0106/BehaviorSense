"""FastAPI service: schema objects in, verified reports out.

Design constraints carried over from the agent architecture:

  - Agents communicate through versioned Pydantic schemas persisted to SQLite, not
    function calls. The API surface is therefore just those schemas over HTTP, and every
    stage stays independently replayable.
  - The perception boundary is enforced at the API level: `/ingest/frame` accepts
    `FrameObservation` (already-structured), never image bytes. There is deliberately no
    endpoint that accepts video - if one existed, the "LLM never sees pixels" claim would
    depend on client discipline rather than on the interface.
  - `/report` always returns the VERIFIED view. There is no endpoint that returns raw
    unverified LLM output, because an endpoint like that would inevitably end up wired
    into a UI.

Run:
    PYTHONPATH=src uvicorn behaviorsense.service.api:app --reload
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import date, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from behaviorsense.agents.behaviour import BehaviourAnalyzer, aggregate_daily_features
from behaviorsense.agents.reasoning.reporter import (
    CaregiverReporter,
    FaithfulStubLLM,
    ReporterConfig,
)
from behaviorsense.schemas import (
    ActivitySegment,
    Alert,
    BehaviourState,
    DailyFeatures,
    FrameObservation,
    Role,
)

DB_PATH = Path("data/behaviorsense.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS frames (
    frame_idx INTEGER, ts TEXT, payload TEXT
);
CREATE TABLE IF NOT EXISTS segments (
    segment_id TEXT PRIMARY KEY, track_id INTEGER, role TEXT,
    activity_id INTEGER, start_time TEXT, end_time TEXT, payload TEXT
);
CREATE TABLE IF NOT EXISTS daily_features (
    day TEXT, role TEXT, payload TEXT, PRIMARY KEY (day, role)
);
CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY, day TEXT, kind TEXT, severity TEXT, payload TEXT
);
CREATE TABLE IF NOT EXISTS reports (
    report_id TEXT PRIMARY KEY, day TEXT, hallucination_rate REAL, payload TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_day ON alerts(day);
CREATE INDEX IF NOT EXISTS idx_segments_time ON segments(start_time);
"""


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    """Open the service DB, resolving DB_PATH at CALL time.

    The default used to be `path: Path = DB_PATH`, which binds when the function is
    defined - so `api.DB_PATH = tmp / "test.db"` had no effect and every test run wrote
    to the real `data/behaviorsense.db` instead of its temp directory. That is why the
    committed database had accumulated 66 alert rows, and why `test_x5` finally failed:
    the count grew by 2 per run until it crossed the `?limit=50` page size. The tests
    were passing for 40 runs while proving nothing about isolation.
    """
    path = Path(path if path is not None else DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    with closing(conn.cursor()) as cur:
        cur.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


app = FastAPI(
    title="BehaviorSense AI",
    version="1.0.0",
    description="Identity-aware behaviour monitoring with verified LLM reporting.",
)

_analyzers: dict[str, BehaviourAnalyzer] = {}
_states: dict[tuple[str, str], BehaviourState] = {}
"""Most recent analysed state per (role, day).

Held in memory because a `BehaviourState` carries the baselines and deviations the
verifier needs to resolve evidence refs, and re-deriving them from the SQLite rows would
duplicate Agent 3's logic in the service layer - two implementations of the same
statistics is exactly how a report and its verification drift apart. The stored rows
remain the durable record; this is a cache keyed to what was actually analysed.
"""
_reporter = CaregiverReporter(FaithfulStubLLM(), config=ReporterConfig())


def analyzer_for(role: str) -> BehaviourAnalyzer:
    if role not in _analyzers:
        _analyzers[role] = BehaviourAnalyzer()
    return _analyzers[role]


class IngestAck(BaseModel):
    stored: int
    kind: str


class DayResponse(BaseModel):
    day: date
    features: DailyFeatures | None = None
    alerts: list[Alert] = []
    deviations: dict[str, float] = {}
    drift_signals: dict[str, float] = {}


class ReportResponse(BaseModel):
    report_day: date
    summary: str
    verified_claims: list[dict[str, Any]]
    withheld_claims: int
    hallucination_rate: float
    recommendation: str
    escalate: bool
    model_name: str
    constrained_decoding: bool


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "behaviorsense", "version": app.version}


@app.post("/ingest/frames", response_model=IngestAck)
def ingest_frames(frames: list[FrameObservation]) -> IngestAck:
    """Accept Agent 1 output. Structured observations only - never image data."""
    with closing(connect()) as conn:
        conn.executemany(
            "INSERT INTO frames (frame_idx, ts, payload) VALUES (?, ?, ?)",
            [(f.frame_idx, f.timestamp.isoformat(), f.model_dump_json()) for f in frames],
        )
        conn.commit()
    return IngestAck(stored=len(frames), kind="frame_observation")


@app.post("/ingest/segments", response_model=IngestAck)
def ingest_segments(segments: list[ActivitySegment]) -> IngestAck:
    """Accept Agent 2 output."""
    with closing(connect()) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO segments "
            "(segment_id, track_id, role, activity_id, start_time, end_time, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(s.segment_id, s.track_id, s.role.value, s.activity_id,
              s.start_time.isoformat(), s.end_time.isoformat(), s.model_dump_json())
             for s in segments],
        )
        conn.commit()
    return IngestAck(stored=len(segments), kind="activity_segment")


@app.post("/analyze/day", response_model=DayResponse)
def analyze_day(
    day: date, role: Role = Role.RESIDENT, features: DailyFeatures | None = None
) -> DayResponse:
    """Run Agent 3 for one day.

    If `features` is omitted, they are aggregated from stored segments for that day -
    which is the normal path; passing features directly exists for replay and testing.
    """
    if features is None:
        with closing(connect()) as conn:
            rows = conn.execute(
                "SELECT payload FROM segments WHERE date(start_time) = ? AND role = ?",
                (day.isoformat(), role.value),
            ).fetchall()
        if not rows:
            raise HTTPException(404, f"no segments stored for {day} / {role.value}")
        segments = [ActivitySegment.model_validate_json(r["payload"]) for r in rows]
        features = aggregate_daily_features(segments, day=day, subject_role=role)

    state = analyzer_for(role.value).analyze_day(features)
    _states[(role.value, day.isoformat())] = state
    _persist_state(state)
    return DayResponse(
        day=state.report_day,
        features=state.today,
        alerts=state.alerts,
        deviations=state.deviations,
        drift_signals=state.drift_signals,
    )


def _persist_state(state: BehaviourState) -> None:
    with closing(connect()) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO daily_features (day, role, payload) VALUES (?, ?, ?)",
            (state.report_day.isoformat(), state.subject_role.value,
             state.today.model_dump_json()),
        )
        conn.executemany(
            "INSERT OR REPLACE INTO alerts (alert_id, day, kind, severity, payload) "
            "VALUES (?, ?, ?, ?, ?)",
            [(a.alert_id, a.day.isoformat(), a.kind.value, a.severity.value,
              a.model_dump_json()) for a in state.alerts],
        )
        conn.commit()


@app.get("/alerts", response_model=list[Alert])
def list_alerts(since: date | None = None, limit: int = 100) -> list[Alert]:
    q = "SELECT payload FROM alerts"
    params: tuple = ()
    if since:
        q += " WHERE day >= ?"
        params = (since.isoformat(),)
    q += " ORDER BY day DESC LIMIT ?"
    params = (*params, limit)
    with closing(connect()) as conn:
        rows = conn.execute(q, params).fetchall()
    return [Alert.model_validate_json(r["payload"]) for r in rows]


@app.post("/report", response_model=ReportResponse)
def generate_report(day: date, role: Role = Role.RESIDENT) -> ReportResponse:
    """Generate and VERIFY a caregiver report. Unverified claims never leave this call."""
    state = _states.get((role.value, day.isoformat()))
    if state is None:
        raise HTTPException(
            404,
            f"no analysed state for {day}. POST /analyze/day first - reports are "
            "generated from stored state, never from raw input.",
        )

    outcome = _reporter.report(state)
    faithful = {v.claim_id for v in outcome.report.verifications if v.is_faithful}
    with closing(connect()) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO reports (report_id, day, hallucination_rate, payload) "
            "VALUES (?, ?, ?, ?)",
            (outcome.report.report_id, day.isoformat(), outcome.hallucination_rate,
             outcome.report.model_dump_json()),
        )
        conn.commit()

    return ReportResponse(
        report_day=day,
        summary=outcome.report.summary,
        verified_claims=[
            json.loads(c.model_dump_json()) for c in outcome.report.claims
            if c.claim_id in faithful
        ],
        withheld_claims=len(outcome.report.claims) - len(faithful),
        hallucination_rate=outcome.hallucination_rate,
        recommendation=outcome.report.recommendation,
        escalate=outcome.report.escalate,
        model_name=outcome.report.model_name,
        constrained_decoding=outcome.report.constrained_decoding,
    )


@app.get("/stats")
def stats() -> dict[str, Any]:
    with closing(connect()) as conn:
        def count(t: str) -> int:
            return conn.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"]
        return {
            "frames": count("frames"),
            "segments": count("segments"),
            "days_analysed": count("daily_features"),
            "alerts": count("alerts"),
            "reports": count("reports"),
            "generated_at": datetime.now().isoformat(),
        }
