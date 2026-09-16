"""Local backend: Agents 3 and 4 on this machine, Agents 1 and 2 on a Kaggle GPU.

    python web/local_backend.py --kaggle https://<something>.trycloudflare.com
    # then point the front end at http://127.0.0.1:8899

Why the split is here and not somewhere else
--------------------------------------------
The OpenRouter keys live on this machine, so Agent 4 belongs on this machine - and the cut lands
on the **Agent 2 -> Agent 3** seam, which is where this project already claims pixels stop.
`state_to_payload` exists precisely so Agent 4 sees numbers only; splitting here enforces a
boundary the design already asserts rather than inventing one.

    browser  ->  this process              ->  Kaggle P100
                 Agent 3  numpy                Agent 1  RTMO
                 Agent 4  hosted LLM           Agent 2  ST-GCN++
                 C1-C5    pure Python

Three things that fall out of it, beyond speed:

* **No credential ever reaches Kaggle.** Not in Secrets, not in a cell, not in a Save Version.
* **The verifier runs where you can debug it** - in seconds, not behind a 12-hour session.
* **The browser talks only to localhost**, so there is no CORS-over-tunnel and no mixed content.

What it deliberately does NOT do
--------------------------------
It does not reimplement Agents 3 and 4. `behaviorsense.service.staging.agents_3_and_4` is the one
implementation and both this file and notebook 05 call it. Two implementations of one payload is
the exact failure `tests/test_web_contract.py` was written after: the dev stub and the notebook
drifted, five key-comparison tests passed, and every upload 500'd.

It also does not repair the model's output. `CaregiverReporter` owns parsing and `repair_claim`,
and a proxy that quietly fixed malformed JSON would hide the thing the hallucination measurement
is about.
"""

from __future__ import annotations

import argparse
import http.server
import json
import socketserver
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behaviorsense.agents.reasoning.openrouter import (  # noqa: E402
    DEFAULT_MODEL,
    OpenRouterLLM,
    discover_keys,
)
from behaviorsense.agents.reasoning.reporter import CaregiverReporter, ReporterConfig  # noqa: E402
from behaviorsense.schemas import ActivitySegment, Role  # noqa: E402
from behaviorsense.service.staging import agents_3_and_4, assert_subject  # noqa: E402

PORT = 8899
BASE = datetime(2026, 1, 1, 9, 0, 0)


class GalleryStore:
    """The operator's enrolled residents, as one plain JSON file.

    Deliberately NOT a Kaggle dataset and NOT under web/ (the static-server root - a gallery
    there would be downloadable by anyone with the page URL). These are biometric templates
    of a real person: the store is a deletable file on the machine that owns the keys, the
    write path is this class alone, and `gallery.json` is gitignored.

    The record is the centroid the matcher uses, not the raw crops: enrol() normalises then
    averages, and one normalised vector is its own mean, so the round-trip through the Kaggle
    half (which rebuilds its gallery from this JSON) is exact.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.residents: dict = {}
        if path.is_file():
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
                self.residents = doc.get("residents") or {}
            except (ValueError, OSError) as exc:
                print(f"  WARNING gallery unreadable ({exc}); starting empty at {path}")

    def as_request_json(self) -> str:
        return json.dumps({"residents": self.residents})

    def enrol(self, name: str, embeddings: list, *, role: str = "resident") -> int:
        """Add or refresh a resident from the clip's best crops. Returns crops used."""
        if not name.strip() or not embeddings:
            return 0
        import numpy as np
        vecs = [np.asarray(e, dtype=np.float32).ravel() for e in embeddings]
        dims = {v.size for v in vecs}
        if len(dims) != 1 or not dims or dims.pop() < 8:
            return 0
        # Normalise BEFORE averaging (see OpenSetGallery.enrol): only direction carries
        # identity under cosine distance, so a high-magnitude crop cannot dominate.
        unit = [v / (np.linalg.norm(v) or 1.0) for v in vecs]
        centroid = np.mean(unit, axis=0)
        centroid /= np.linalg.norm(centroid) or 1.0
        prev = self.residents.get(name.strip())
        self.residents[name.strip()] = {
            "role": role,
            "centroid": [round(float(x), 6) for x in centroid],
            "n_updates": int((prev or {}).get("n_updates", 0)) + len(vecs),
            "enrolled_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.path.write_text(json.dumps({"residents": self.residents}, indent=1),
                             encoding="utf-8")
        return len(vecs)


def rebuild_segments(tracks: list[dict], *, base: datetime = BASE) -> list[ActivitySegment]:
    """Kaggle's wire `tracks` -> `ActivitySegment` objects.

    The wire form carries `t0`/`t1` as seconds from the clip start because the browser draws
    timelines with them; `ActivitySegment` wants absolute datetimes. This is the only place that
    conversion happens, so the two cannot drift into different beliefs about the origin.
    """
    out: list[ActivitySegment] = []
    for t in tracks:
        role = Role(t.get("role", "unknown"))
        for s in t.get("segments", []):
            t0, t1 = float(s["t0"]), float(s["t1"])
            if t1 <= t0:
                # A zero-length segment fails the schema's own interval check. Nudging it by a
                # millisecond is honest here: the duration is a rounding artefact of the wire
                # format, not a claim that the activity took no time.
                t1 = t0 + 0.001
            out.append(ActivitySegment(
                segment_id=f"t{t['track_id']}-{s['label']}-{t0:.2f}",
                track_id=int(t["track_id"]), role=role,
                activity_id=int(s["label"]), activity_name=s["name"],
                start_time=base + timedelta(seconds=t0),
                end_time=base + timedelta(seconds=t1),
                confidence=float(s.get("confidence", 0.5)),
                room=s.get("room")))
    return out


class Upstream:
    """The Kaggle half. Forwards the raw multipart body and yields its NDJSON lines."""

    def __init__(self, base: str, token: str = "", timeout: float = 180.0) -> None:
        self.base = base.rstrip("/")
        self.token = token
        # A SILENCE BUDGET, NOT A DURATION CAP - and that is a property of urllib, not a choice
        # made here. `urlopen(timeout=N)` bounds each individual read, so it resets every time a
        # line arrives: measured, three lines 0.5 s apart streamed fine under `timeout=2.0` and the
        # error came only from the silent gap after them. So this never kills a long-but-streaming
        # analysis, however many minutes of pose it takes, which is what makes raising MAX_FRAMES
        # to ten minutes of footage safe.
        #
        # 180 s rather than the old 600 s because the GPU half beats every 10 s, so three minutes
        # of nothing is eighteen missed heartbeats - decisively dead rather than merely slow. It
        # matches `stall_s` in video.py, so both halves give up on silence at the same point.
        self.timeout = timeout

    def health(self) -> dict:
        req = urllib.request.Request(f"{self.base}/health", headers=self._headers())
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode("utf-8"))

    def _headers(self, extra: dict | None = None) -> dict:
        h = {"X-BS-Token": self.token} if self.token else {}
        return {**h, **(extra or {})}

    def video_stages_12(self, body: bytes, content_type: str, gallery_json: str = ""):
        """POST the upload to `?stages=12` and yield parsed NDJSON events.

        The body is passed through byte-for-byte rather than re-encoded: the upload is already a
        multipart stream and re-building it here would be a second place for the boundary and the
        field name to be wrong. The gallery is APPENDED by boundary surgery - strip the closing
        marker, add one more part, re-close - so the video bytes are never parsed, buffered in a
        second copy, or re-encoded here. A browser-chosen boundary colliding with video content
        is the design assumption multipart itself rests on.

        READLINE, NOT read(65536). `read(n)` blocks until n bytes accumulate or the stream ends, so
        with an ~80-byte heartbeat every 10 s it would sit silent for over two hours before
        forwarding anything. Measured: four 1 s-spaced heartbeats all arrived together at t+4.0 s
        under `read(65536)`, and at t+0.0/1.0/2.0/3.0 s under `readline()`. The GPU half was
        streaming correctly; this loop was the thing hoarding it, which is why the page reported
        "No data from the backend for 90 s" while the session was perfectly healthy. It only became
        visible once pose ran longer than the watchdog - at 35 s the whole response landed first.
        """
        if gallery_json:
            # Boundary surgery, described in the docstring above. The closing marker is
            # `--<boundary>--`; bodies end with it, with or without a trailing CRLF.
            import re as _re
            m = _re.search(r'boundary=(?:"([^"]+)"|([^;]+))', content_type or "")
            if not m:
                raise ValueError("cannot append gallery: upload is not multipart")
            boundary = (m.group(1) or m.group(2)).strip()
            # CUT AT the closing delimiter, do not keep it. The first version appended after
            # `--boundary--`, which is the END-OF-MULTIPART marker: a parser stops there and
            # everything following is epilogue it must ignore. The bytes looked right - the
            # gallery part was present, the body closed properly - and W10 asserted exactly
            # that, so the whole thing shipped while `gallery` never reached FastAPI and
            # re-identification silently stayed off. Bytes present is not the same as field
            # parsed, which is why W10 now runs a real multipart parser over the result.
            close = ("--" + boundary + "--").encode()
            cut = body.rfind(close)
            if cut < 0:
                raise ValueError("cannot append gallery: no closing multipart delimiter")
            part = ("--" + boundary + "\r\n"
                    'Content-Disposition: form-data; name="gallery"\r\n\r\n'
                    + gallery_json + "\r\n--" + boundary + "--\r\n").encode("utf-8")
            body = body[:cut] + part
        req = urllib.request.Request(
            f"{self.base}/video?stages=12", data=body, method="POST",
            headers=self._headers({"Content-Type": content_type}))
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            for raw in iter(r.readline, b""):
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue            # a partial line is never guessed at


def finish_locally(upstream_result: dict, reporter) -> tuple[list[dict], list[dict], dict]:
    """Agents 3 and 4 here. Returns (stage records, checks, timing additions).

    The subject assertion is redone locally from `frames_per_track` rather than trusted from the
    wire, because the whole point of this split is that the half holding the keys owns the
    reasoning. Kaggle's own verdict is used when its re-identifier actually matched a gallery.
    """
    tracks = upstream_result.get("tracks", [])
    segments = rebuild_segments(tracks)
    fpt = {int(k): int(v) for k, v in (upstream_result.get("frames_per_track") or {}).items()}
    # Spans come over the wire because only the GPU half has the per-frame observations, and they
    # are what lets one person's fragmented ids be reunited: a track that leaves detection for more
    # than 1.5 s returns with a new id, and keeping only the most-present fragment discarded the
    # rest of the same person's activity.
    spans = {int(k): (int(v[0]), int(v[1]))
             for k, v in (upstream_result.get("track_spans") or {}).items() if len(v) == 2}
    # First/last box per track, for the reachability constraint: disjoint spans alone would merge a
    # television in the corner into the resident, since RTMO detects the people on the screen.
    boxes = {int(k): (v[0], v[1])
             for k, v in (upstream_result.get("track_boxes") or {}).items()
             if isinstance(v, (list, tuple)) and len(v) == 2}
    if upstream_result.get("subject_provenance") == "reid_matched" and segments:
        kept, subj, prov = segments, upstream_result.get("subject_track"), "reid_matched"
        subj_tracks: list[int] = []
    else:
        kept, subj, prov, subj_tracks = assert_subject(
            segments, fpt, spans=spans or None, boxes=boxes or None,
            fps=float(upstream_result.get("fps") or 20.0))

    day_s = upstream_result.get("day")
    day = date.fromisoformat(day_s) if day_s else BASE.date()
    obs = float(upstream_result.get("observed_hours") or 0.0)

    records, checks = [], []
    for kind, obj in agents_3_and_4(kept, day=day, observed_hours=obs, subject_track=subj,
                                    subject_provenance=prov, reporter=reporter,
                                    subject_tracks=subj_tracks):
        if kind == "stage":
            records.append(obj)
        else:
            checks = obj
    timing = {"behaviour_s": records[0]["elapsed_s"] if records else 0.0,
              "report_s": records[1]["elapsed_s"] if len(records) > 1 else 0.0}
    return records, checks, timing


def demo_payload(reporter, scenario: int = 0) -> dict:
    """One stored-day report, in the shape the ledger's `renderLive` consumes.

    THE LOCAL HALF OF /demo. The notebook has had this endpoint since the page grew a
    "Generate live" chip; the local backend never did, so the chip appeared whenever the page
    was live against THIS process and then failed with "no route /demo" - a broken advertised
    feature that only fired in the split configuration the page is actually used in. Same
    simulator, same reporter, same serialisation as the notebook's `_serialise`, because the
    ledger re-verifies the claims in the browser and the two implementations must not drift.
    """
    from behaviorsense.agents.behaviour import BehaviourAnalyzer
    from behaviorsense.agents.reasoning.verifier import FaithfulnessVerifier
    from behaviorsense.data.simulator import standard_scenarios

    scenarios = list(standard_scenarios())
    if not 0 <= scenario < len(scenarios):
        raise ValueError(f"scenario {scenario} of {len(scenarios)}")
    analyzer = BehaviourAnalyzer()
    picked = None
    for day in scenarios[scenario].run().days:
        st = analyzer.analyze_day(day)
        if st.baselines and st.alerts:
            picked = st               # last alerting day: the decline is developed
    if picked is None:
        raise ValueError("simulator produced no alerting day")

    out = reporter.report(picked)
    index = FaithfulnessVerifier().build_index(picked)
    return {
        "day": picked.report_day.isoformat(),
        "subject": picked.subject_role.value,
        "summary": out.report.summary,
        "recommendation": out.report.recommendation,
        "escalate": out.report.escalate,
        "claims": [c.model_dump(mode="json") for c in out.report.claims],
        "verifications": [v.model_dump(mode="json") for v in out.report.verifications],
        "evidence": {k: v.model_dump(mode="json") for k, v in index.items()},
        "alerts": [{"kind": a.kind.value, "severity": a.severity.value}
                   for a in picked.alerts],
        "hallucination_rate": out.hallucination_rate,
        "emitted": out.n_emitted_claims,
        "schema_rejected": out.dropped_claims,
        "model": out.report.model_name,
    }


class Handler(http.server.BaseHTTPRequestHandler):
    # Chunked transfer needs HTTP/1.1, or the framing arrives as body bytes.
    protocol_version = "HTTP/1.1"
    upstream: Upstream
    reporter: object
    gallery: GalleryStore
    # Set once the client's socket is gone. Every write checks it, because the alternative is what
    # this file used to do: a `ConnectionAbortedError` inside `_emit`, then a second one from the
    # error handler trying to report the first, then a third from `_close_stream` in `finally` -
    # three stacked tracebacks on the console for the entirely ordinary event of a browser tab
    # being closed mid-analysis.
    _dead = False

    def _cors(self) -> None:
        # The page is served from :5175 and this from :8899, so CORS is required for the same
        # reason it is on Kaggle - but only across localhost, never across a tunnel.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        # GUARDED, like `_emit`. The page polls /health every 15 s and navigates away mid-poll
        # whenever it is reloaded, so a client disconnect here is routine - but this path wrote
        # unguarded and printed a full socketserver traceback for it, which reads as a crash in
        # a log where real failures also live. Only `_emit` got the guard when the streaming
        # path was fixed; /health was left behind.
        self._write(body)

    def _open_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self._cors()
        self.send_header("Cache-Control", "no-store")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _write(self, raw: bytes) -> bool:                            # noqa: D401
        """One chunked write. Returns False once the client has gone, and never raises for it.

        A disconnect is not an error condition for this process: Agents 1 and 2 already ran on
        Kaggle and the work is finished either way. It is recorded so the loop can stop early
        instead of computing a report nobody is listening for.
        """
        if self._dead:
            return False
        try:
            self.wfile.write(raw)
            self.wfile.flush()
            return True
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
            self._dead = True
            return False

    def _emit(self, obj: dict) -> None:
        line = (json.dumps(obj, default=str) + "\n").encode()
        self._write(f"{len(line):X}\r\n".encode() + line + b"\r\n")

    def _close_stream(self) -> None:
        self._write(b"0\r\n\r\n")

    def _blocking(self, fn, *, step: str, every: float = 5.0):
        """Run `fn()` on a worker thread, heartbeating until it returns.

        The page's watchdog is on SILENCE, not on total duration - 90 s of it. Agent 4 is now a
        network call that can legitimately spend tens of seconds sweeping a provider chain when the
        free endpoints are saturated, and a silent stretch that long makes the browser abort a run
        that was working. Per-stage lines are not enough on their own; the silence *inside* one
        stage is what has to be bounded, which is the same fix notebook 05 needed.
        """
        out: dict = {}

        def run() -> None:
            try:
                out["value"] = fn()
            except BaseException as exc:                             # noqa: BLE001
                out["error"] = exc                  # re-raised on this thread, traceback intact

        th = threading.Thread(target=run, daemon=True)
        th.start()
        t0 = time.time()
        while th.is_alive():
            th.join(every)
            if th.is_alive():
                self._emit({"event": "heartbeat", "step": step,
                            "elapsed_s": round(time.time() - t0, 1)})
        if "error" in out:
            raise out["error"]
        return out.get("value")

    def do_OPTIONS(self) -> None:                                   # noqa: N802
        self._json({})

    def do_GET(self) -> None:                                       # noqa: N802
        # /demo, streamed like /video: the ledger's "Generate live" chip is shown whenever
        # the page is live against THIS process, and before this route existed it failed
        # with "no route /demo" - a broken advertised feature in the split configuration
        # the page is actually used in. Same NDJSON envelope as the notebook's /demo.
        if self.path.startswith("/demo"):
            self._demo()
            return
        if not self.path.startswith("/health"):
            self._json({"detail": f"no route {self.path}"}, 404)
            return
        # /health reflects BOTH halves. A local process that reported itself healthy while the
        # GPU half was gone would put the page in `Live` with nothing behind it.
        try:
            up = self.upstream.health()
        except Exception as exc:                                    # noqa: BLE001
            self._json({"service": "behaviorsense", "ready": False,
                        "error": f"upstream {self.upstream.base}: {type(exc).__name__}: {exc}",
                        "split": "local agents 3-4, kaggle agents 1-2"}, 200)
            return
        self._json({**up, "service": "behaviorsense", "split": "local agents 3-4",
                    "agent4": getattr(self.reporter, "name", None) or "unset",
                    "upstream": self.upstream.base,
                    # The GPU half no longer writes reports, so its own `agent4` would be stale.
                    "ready": bool(up.get("ready"))})

    def do_POST(self) -> None:                                      # noqa: N802
        if not self.path.startswith("/video"):
            self._json({"detail": f"no route {self.path}"}, 404)
            return
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            self._json({"detail": "empty upload"}, 422)
            return
        body = self.rfile.read(n)
        ctype = self.headers.get("Content-Type", "application/octet-stream")

        self._open_stream()
        t0 = time.time()
        # The enrol request travels twice on purpose. The FORM field goes to the Kaggle half
        # (it enables OSNet and crop collection); the header tells THIS process the name to
        # persist under, which is a local decision the notebook has no business making.
        enrol_name = (self.headers.get("X-BS-Enrol") or "").strip()
        gallery_json = self.gallery.as_request_json() if len(self.gallery.residents) else ""
        try:
            result12 = None
            for ev in self.upstream.video_stages_12(body, ctype, gallery_json=gallery_json):
                kind = ev.get("event")
                if kind == "result":
                    result12 = ev["payload"]
                elif kind == "error":
                    self._emit(ev)
                    return
                else:
                    # `accepted`, `stage` and `heartbeat` pass straight through, so the page draws
                    # Agents 1 and 2 while they happen rather than after everything finishes.
                    self._emit(ev)
                if self._dead:
                    return          # nobody is reading; Agents 1-2 already ran, so just stop
            if result12 is None:
                self._emit({"event": "error", "kind": "upstream",
                            "detail": "the GPU half closed the stream without a result - the "
                                      "Kaggle session may have ended mid-run"})
                return

            # ENROLMENT, persisted where the gallery lives. Only when the operator asked
            # (the header) AND the GPU half actually collected crops for the subject - a
            # name with no embeddings would enrol a resident who cannot ever be matched.
            if enrol_name:
                gu = (result12 or {}).get("gallery_updates") or {}
                used = self.gallery.enrol(enrol_name, gu.get("embeddings") or [])
                if used:
                    self._emit({"event": "stage", "stage": {
                        "agent": 1, "name": "enrolment", "status": "done",
                        "elapsed_s": 0.0,
                        "payload": {"enrolled": enrol_name, "crops": used,
                                    "matched_this_clip": bool(gu.get("matched")),
                                    "caveats": [
                                        f"Enrolled from the best {used} crops of this clip's "
                                        "subject track. The gallery is a local file on this "
                                        "machine and never leaves it; delete the file to "
                                        "forget the enrolment."]}}})
                else:
                    self._emit({"event": "stage", "stage": {
                        "agent": 1, "name": "enrolment", "status": "failed",
                        "elapsed_s": 0.0,
                        "payload": {"error": f"could not enrol {enrol_name!r}: the GPU half "
                                              "returned no usable crops for the subject "
                                              "track (was OSNet attached?)"}}})

            self._emit({"event": "heartbeat", "step": "agents_3_4_local",
                        "elapsed_s": round(time.time() - t0, 1)})
            records, checks, timing = self._blocking(
                lambda: finish_locally(result12, self.reporter), step="agents_3_4_local")
            for rec in records:
                self._emit({"event": "stage", "stage": rec})

            payload = dict(result12)
            payload["stages"] = list(result12.get("stages", [])) + records
            payload["checks"] = checks
            payload["stages_run"] = "1234"
            payload["timing"] = {**result12.get("timing", {}), **timing}
            s4 = records[1]["payload"] if len(records) > 1 else {}
            payload["report"] = s4.get("summary")
            payload["report_provenance"] = (
                None if not s4.get("summary") else
                "Claims are verified against features measured from THIS video. The baseline they "
                "are compared to is a declared simulated reference, because a 14-day rolling "
                "median cannot come from one clip - see stages[2].caveats.")
            self._emit({"event": "result", "payload": payload})
        except urllib.error.HTTPError as exc:
            self._emit({"event": "error", "kind": "upstream",
                        "detail": f"GPU half returned {exc.code}: "
                                  f"{exc.read().decode('utf-8', 'replace')[:300]}"})
        except Exception as exc:                                    # noqa: BLE001
            import traceback
            self._emit({"event": "error", "kind": "local",
                        "detail": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-1200:]})
        finally:
            self._close_stream()

    def _demo(self) -> None:
        """GET /demo?scenario=N -> NDJSON: accepted, heartbeats, result."""
        import urllib.parse

        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        scenario = int((query.get("scenario") or ["0"])[0])
        self._open_stream()
        try:
            self._emit({"event": "accepted", "scenario": scenario})
            payload = self._blocking(
                lambda: demo_payload(self.reporter, scenario), step="generating")
            self._emit({"event": "result", "payload": payload})
        except Exception as exc:                                    # noqa: BLE001
            import traceback
            self._emit({"event": "error", "kind": "local",
                        "detail": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-1200:]})
        finally:
            self._close_stream()

    def log_message(self, *args) -> None:                           # noqa: D102
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kaggle", required=True, help="the trycloudflare address notebook 05 printed")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--token", default="", help="X-BS-Token, if the notebook set BS_TOKEN")
    ap.add_argument("--gallery", default=str(ROOT / "gallery.json"),
                    help="the enrolled-residents file this process owns (gitignored; delete "
                         "it to forget every enrolment)")
    args = ap.parse_args()

    keys = discover_keys()
    if not keys:
        print("no OpenRouter keys in the environment. Set OPENROUTER_API_KEY_1..N (or "
              "OPENROUTER_API_KEY_LIST, comma-separated). They stay on this machine - that is the "
              "reason Agent 4 runs here.")
        return 1
    Handler.upstream = Upstream(args.kaggle, token=args.token)
    Handler.gallery = GalleryStore(Path(args.gallery))
    llm = OpenRouterLLM(args.model, keys=keys)
    Handler.reporter = CaregiverReporter(llm, config=ReporterConfig(constrained=True))

    print(f"local backend on http://127.0.0.1:{args.port}")
    print(f"  agents 1-2 (GPU)  -> {args.kaggle}")
    print(f"  agents 3-4 + C1-C5 here, {len(keys)} OpenRouter key(s)")
    # The chain is printed rather than just the primary because the failure this replaced looked
    # like a key problem and was a provider problem: thirteen keys, thirteen one-each 429s, and
    # `rate-limited upstream` in every body. Seeing the providers listed is the reminder that the
    # second axis exists.
    print(f"  provider chain ({len(llm.plan)}): {' -> '.join(llm.plan)}")
    if len(Handler.gallery.residents):
        print(f"  re-id gallery  : {len(Handler.gallery.residents)} enrolled "
              f"({', '.join(Handler.gallery.residents)}) - uploads will be matched")
    else:
        print("  re-id gallery  : empty - identity is asserted, not recognised. Enrol from "
              "the page (a name in the Enrol field) to turn it on")
    print("  point the front end at this address, not at the tunnel")
    try:
        up = Handler.upstream.health()
        print(f"  upstream ready={up.get('ready')} gpu={up.get('gpu')} "
              f"streams={up.get('streams')}")
    except Exception as exc:                                        # noqa: BLE001
        print(f"  WARNING upstream unreachable: {type(exc).__name__}: {exc}")

    class Server(socketserver.ThreadingTCPServer):
        # Threaded: a stream holds its connection for minutes and the page polls /health every
        # 15 s. Single-threaded, that poll queues behind the upload and the indicator drops to
        # "No answer" mid-run - the page reporting a dead backend while talking to a live one.
        daemon_threads = True

        def server_bind(self) -> None:
            # `allow_reuse_address = True` was here and it is a PORT-SHARING BUG ON WINDOWS.
            # SO_REUSEADDR means "reuse TIME_WAIT" on POSIX, but "bind a port another live process
            # is already listening on" on Windows - so a stray dev stub on 8899 let this server
            # print its banner, bind 'successfully', and then silently lose the browser's requests
            # to the older socket. MEASURED: two PIDs listening on 8899 at once, the page showing
            # the dev stub's fixture health ("no GPU reported, 4 ADL streams") over a terminal
            # whose own probe had printed "gpu=Tesla P100, streams=['bone']". The Windows-correct
            # option is SO_EXCLUSIVEADDRUSE: a taken port fails LOUDLY at startup, which is the
            # only acceptable behaviour for two processes claiming one port.
            import socket as _socket
            if hasattr(_socket, "SO_EXCLUSIVEADDRUSE"):
                self.socket.setsockopt(_socket.SOL_SOCKET, _socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                self.socket.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
            super().server_bind()

    with Server(("127.0.0.1", args.port), Handler) as srv:
        srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())

