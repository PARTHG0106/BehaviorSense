"""A stand-in for the Kaggle backend, so the live path can be developed without a GPU.

`notebooks/05_serve_inference_online.ipynb` is the real thing: Qwen2.5-7B and the four ADL
streams on a T4, behind a Cloudflare tunnel. Booking a GPU session to check that a banner
renders is absurd, so this serves the same routes with the same shapes and no model at all.

It is deliberately NOT importable from `src/` and does not pretend to classify anything. The
one thing it is careful about is the fixture: claim `c3` carries a right number told
backwards, so the page can be seen *rejecting* something. A stand-in where every claim passes
would prove only that the happy path renders, which is the half that was never in doubt.

    python web/dev_backend.py
    # then paste http://127.0.0.1:8899 into the page's Address field
"""

from __future__ import annotations

import http.server
import json
import math
import os
import re
import socketserver
import time

PORT = 8899
# Fake per-agent latency so the progressive reveal is visible locally. The real chain takes
# minutes; 0.6 s a line is enough to watch cards appear in order without making the suite slow.
# `BS_STREAM_DELAY=0` for tests, which assert content and should not pay for theatre.
STREAM_DELAY = float(os.environ.get("BS_STREAM_DELAY", "0.6"))
DAY = "2026-03-04"


def ev(name: str, observed: float, baseline: float, pct: float, z: float) -> dict:
    return {
        "ref": f"feat:{name}:{DAY}",
        "feature_name": name,
        "observed_value": observed,
        "baseline_median": baseline,
        "delta": observed - baseline,
        "pct_change": pct,
        "robust_z": z,
        "baseline_n_days": 30,
        "day": DAY,
    }


EVIDENCE = {e["ref"]: e for e in (
    ev("walking_duration_s", 820, 1810, -54.7, -4.31),
    ev("meal_events", 1, 3, -66.7, -2.88),
    ev("social_interaction_duration_s", 260, 1180, -78.0, -3.19),
    ev("medication_events", 0, 1, -100.0, -2.24),
)}

CLAIMS = [
    {"claim_id": "c1",
     "text": "Walking duration fell to 820 s against a baseline of 1810 s.",
     "evidence_ref": f"feat:walking_duration_s:{DAY}",
     "claimed_value": 820.0, "claimed_pct_change": -54.7, "direction": "decrease"},
    # Prose echoes the digits ("1 meal ... 3"), not words ("One ... three") — C5 requires
    # the quoted figure to appear numerically, which is exactly what the real prompt
    # demands of the model. A stub whose faithful claims fail C5 would misrender the demo.
    {"claim_id": "c2",
     "text": "1 meal was recorded where 3 are usual.",
     "evidence_ref": f"feat:meal_events:{DAY}",
     "claimed_value": 1.0, "claimed_pct_change": -66.7, "direction": "decrease"},
    # The fault. C1-C4 all pass: the citation resolves, the value is right, the prose
    # echoes it, and the magnitude is right. Only C4 catches that the story is inverted.
    {"claim_id": "c3",
     "text": "Time in conversation improved markedly, rising to 260 s.",
     "evidence_ref": f"feat:social_interaction_duration_s:{DAY}",
     "claimed_value": 260.0, "claimed_pct_change": 78.0, "direction": "increase"},
    {"claim_id": "c4",
     "text": "0 medication events were recorded today.",
     "evidence_ref": f"feat:medication_events:{DAY}",
     "claimed_value": 0.0, "claimed_pct_change": None, "direction": "decrease"},
]

CHECKS = ("ref_exists", "value_matches", "pct_matches", "direction_consistent",
          "prose_quoted_value")

_NUM_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:[kKmM](?![A-Za-z0-9]))?"
                     r"|-?\d+(?:\.\d+)?(?:[kKmM](?![A-Za-z0-9]))?")
_SCALE = {"k": 1e3, "K": 1e3, "m": 1e6, "M": 1e6}


def _numbers_in_text(text: str) -> list[float]:
    """The C5 tokenizer, mirroring verifier.py. The grouped-comma alternative must come
    first or "1,800" scans as 1 then 800 — the exact bug the Python verifier hit."""
    out = []
    for m in _NUM_RE.finditer(text):
        s = m.group(0).replace(",", "")
        scale = _SCALE.get(s[-1])
        if scale:
            s = s[:-1]
        try:
            out.append(float(s) * (scale or 1.0))
        except ValueError:
            pass
    return out


def _prose_quotes_value(text: str, value: float) -> bool:
    """Same 2% relative tolerance as the real C5; vacuous when no value is quoted."""
    for n in _numbers_in_text(text):
        if abs(value) < 1e-6:
            if abs(n) < 1e-3:
                return True
            continue
        # Same display half-step floor as the Python C2/C5: the reporter publishes
        # 2-decimal values, and a faithful copy of one must verify.
        if abs(n - value) <= max(0.02 * abs(value), 0.005):
            return True
    return False


def verdict(claim: dict) -> dict:
    """The same five checks, computed the same way, so the page's agreement banner is a
    real comparison rather than an echo of whatever this file asserts.

    The notes are formatted exactly as `verifier.py` formats them - "pct mismatch: claimed
    +78.0%, actual -78.0%" - because the whole promise of the withheld margin is that it
    prints THE ARITHMETIC THAT FAILED. A stub that returned an empty note list would render
    the generic fallback and quietly demonstrate the weaker version of the feature.
    """
    e = EVIDENCE[claim["evidence_ref"]]
    actual = "increase" if e["delta"] > 0 else "decrease" if e["delta"] < 0 else "unchanged"
    pct = claim["claimed_pct_change"]
    # Display half-step floor, mirroring verifier.py: a claim quoting the published
    # round(x, 2) must pass even when 2% of a small value is tighter than the rounding.
    value_matches = (abs(claim["claimed_value"] - e["observed_value"])
                     <= max(0.02 * abs(e["observed_value"]), 0.005))
    pct_matches = pct is None or abs(pct - e["pct_change"]) <= 2.0
    direction_ok = claim["direction"] == actual
    prose_ok = (claim["claimed_value"] is None
                or _prose_quotes_value(claim["text"], claim["claimed_value"]))

    notes = []
    if not value_matches:
        notes.append(f"value mismatch: claimed {claim['claimed_value']:g}, "
                     f"actual {e['observed_value']:g}")
    if pct is None:
        notes.append("no percentage quoted; C3 vacuous")
    elif not pct_matches:
        notes.append(f"pct mismatch: claimed {pct:+.1f}%, actual {e['pct_change']:+.1f}%")
    if not direction_ok:
        notes.append(f"direction field wrong: claimed {claim['direction']}, "
                     f"actual {actual} (delta {e['delta']:+g})")
    if claim["claimed_value"] is not None and not prose_ok:
        notes.append(f"prose does not echo claimed_value {claim['claimed_value']:g}: a "
                     "caregiver would read the prose number, not the row - flagged")

    return {
        "claim_id": claim["claim_id"],
        "ref_exists": True,
        "value_matches": value_matches,
        "pct_matches": pct_matches,
        "direction_consistent": direction_ok,
        "prose_quoted_value": prose_ok,
        "notes": notes,
    }


VERDICTS = [verdict(c) for c in CLAIMS]
FAITHFUL = sum(1 for v in VERDICTS if all(v[k] for k in CHECKS))

DEMO = {
    "day": DAY,
    "subject": "resident",
    "summary": "Mobility, meals and social contact are all below this resident's own "
               "baseline for a third consecutive day, across independent features.",
    "recommendation": "Call today; arrange a review if walking has not recovered tomorrow.",
    "escalate": True,
    "claims": CLAIMS,
    "verifications": VERDICTS,
    "evidence": EVIDENCE,
    "alerts": [{"kind": "acute_mobility_drop", "severity": "urgent"}],
    "hallucination_rate": 1 - FAITHFUL / len(CLAIMS),
    "emitted": len(CLAIMS),
    "schema_rejected": 0,
    "model": "dev-stub (not a model)",
}

HEALTH = {
    "service": "behaviorsense",
    "ready": True,
    "error": None,
    "gpu": None,                 # honest: there is no GPU here
    "streams": ["joint", "bone", "joint_motion", "bone_motion"],
    "tau": 0.25,
    "auth": False,
    "video": True,               # this stub fakes /video too
    "reid": True,
    "max_upload_mb": 60,
    "max_frames": 900,
}

# ---------------------------------------------------------------------------
# /video — the full four-agent chain over a synthetic two-person clip.
#
# Shape parity with notebook 05's /video is the contract this file exists to hold: same
# top-level keys, same `stages` records (agent, name, status, elapsed_s, payload), same
# per-claim C1-C4 rows. `tests/test_web_contract.py` asserts it, because a stub that drifts
# is worse than no stub - the page would be developed against a shape the GPU never sends.
#
# Two people, and one of them falls, so the multi-person path, the fall styling and the
# role distinction are all exercised. Agents 3 and 4 report against a DECLARED simulated
# baseline, exactly as the real endpoint does; the provenance field is not decoration, it
# is what stops a reference-relative robust-z reading as this person's own decline.
# ---------------------------------------------------------------------------

VIDEO_W, VIDEO_H, VIDEO_FPS, VIDEO_FRAMES = 1280, 720, 15.0, 150

# A standing COCO-17 figure in a 0..1 box, walked across the frame. Rough on purpose: this
# exists to prove the overlay maps source pixels to CSS pixels and distinguishes two people,
# not to look like a person.
_POSE = [
    (0.50, 0.06), (0.47, 0.04), (0.53, 0.04), (0.44, 0.05), (0.56, 0.05),
    (0.38, 0.18), (0.62, 0.18), (0.32, 0.36), (0.68, 0.36),
    (0.28, 0.52), (0.72, 0.52), (0.42, 0.54), (0.58, 0.54),
    (0.40, 0.76), (0.60, 0.76), (0.39, 0.97), (0.61, 0.97),
]


def _figure(cx: float, cy: float, h: float, sway: float) -> list[list[float]]:
    w = h * 0.42
    out = []
    for i, (px, py) in enumerate(_POSE):
        x = cx + (px - 0.5) * w + (sway if i in (9, 10) else 0.0)
        y = cy + (py - 0.5) * h
        out.append([round(x, 1), round(y, 1), 0.92])
    return out


def _video_payload() -> dict:
    frames = []
    for i in range(VIDEO_FRAMES):
        t = i / VIDEO_FPS
        sway = 18.0 * math.sin(t * 4.0)
        people = [
            {"track_id": 0, "role": "resident", "role_confidence": 0.86,
             "box": [0, 0, 1, 1], "kp": _figure(380 + i * 2.2, 400, 480, sway)},
            {"track_id": 1, "role": "visitor", "role_confidence": 0.71,
             "box": [0, 0, 1, 1], "kp": _figure(900 - i * 1.4, 415, 440, -sway)},
        ]
        for p in people:                       # box from the joints, as RTMO's would be
            xs = [j[0] for j in p["kp"]]
            ys = [j[1] for j in p["kp"]]
            p["box"] = [round(min(xs) - 18, 1), round(min(ys) - 24, 1),
                        round(max(xs) + 18, 1), round(max(ys) + 24, 1)]
        # One stretch where the resident's pose is unusable, so the overlay's dashed
        # "tracked, pose unusable" state is exercised rather than assumed.
        if 60 <= i < 72:
            people[0]["kp"] = None
        frames.append({"i": i, "t": round(t, 3), "people": people})

    def seg(name, label, t0, t1, conf, room="living_room"):
        return {"label": label, "name": name, "t0": t0, "t1": t1, "confidence": conf,
                "room": room}

    tracks = [
        {"track_id": 0, "role": "resident", "is_subject": True,
         "matched_name": "Mary",
         "n_segments": 4, "fall_segments": 2,
         "segments": [seg("walking", 0, 0.0, 3.0, 0.44),
                      seg("standing", 1, 3.0, 5.0, 0.31),
                      seg("falling", 7, 5.0, 6.0, 0.58),
                      seg("fallen_on_ground", 8, 6.0, 9.9, 0.49)]},
        {"track_id": 1, "role": "visitor", "is_subject": False,
         "matched_name": None,
         "n_segments": 2, "fall_segments": 0,
         "segments": [seg("walking", 0, 0.0, 4.5, 0.39),
                      seg("interacting_with_person", 18, 4.5, 9.9, 0.27)]},
    ]

    # Agent 3's payload. The feature values are what a 10-second clip would actually
    # support - two of them, small - and `baseline_provenance` says in the payload itself
    # that the comparison point is not this person. The real backend says the same thing
    # from the same field, which is the only reason the front end can render one component
    # for both.
    stage3 = {
        "baseline_provenance": "reference_cohort_simulated",
        # This stub's track 0 is `resident`, so the gallery "matched" and no assertion was
        # needed. The real backend sends "asserted_most_present_track" on any uploaded clip,
        # because a stranger has no enrolled gallery and every track comes back `unknown`.
        "subject_provenance": "reid_matched",
        "subject_track": 0,
        # Empty here because this stub's gallery "matched". On an uploaded clip it carries
        # every track merged into the subject: one person who leaves detection for more than
        # 1.5 s returns with a new id, and keeping one fragment discarded the rest.
        "subject_tracks": [],
        "baseline_days": 21,
        "real_days_from_this_video": 1,
        "observed_hours": 23.4,
        "is_reliable": True,
        # A FULL DAY, because the stub's own claims (c1-c4) cite baselines and percentage
        # changes, and those cells only exist on a reliable day - a stub that declared a
        # partial window while showing pct-claims would model a state the real system cannot
        # produce. The partial-window path (deviations withheld) is exercised by W6 against
        # the notebook's real staging code instead.
        "deviations_withheld": False,
        "features_from_video": {"walking_duration_s": 820.0, "fall_events": 0.0,
                                "social_interaction_duration_s": 260.0,
                                "sit_to_stand_count": 14.0, "meal_events": 1.0},
        "deviations": [{"feature": "walking_duration_s", "robust_z": -4.31},
                       {"feature": "social_interaction_duration_s", "robust_z": -3.19},
                       {"feature": "meal_events", "robust_z": -2.88},
                       {"feature": "medication_events", "robust_z": -2.24}],
        "alerts": [{"kind": "acute_mobility_drop", "severity": "urgent",
                    "rule": "robust_z <= -3 on walking_duration_s"},
                   {"kind": "fall_detected", "severity": "critical",
                    "rule": "fall segment present"}],
        "caveats": [
            "The BASELINE is a simulated reference persona, not this person's history - "
            "one clip cannot supply 14 days.",
            "Feature VALUES are measured from the uploaded video and are real.",
            "Robust-z and alerts are therefore reference-relative: read them as 'unlike "
            "the reference', never as 'this person has declined'.",
        ],
    }

    stage4 = {
        "model": "dev-stub (not a model)",
        "constrained_decoding": True,
        "summary": DEMO["summary"],
        "recommendation": DEMO["recommendation"],
        "escalate": True,
        "claims_emitted": len(CLAIMS),
        "claims_scorable": len(CLAIMS),
        "hallucination_rate": round(1 - FAITHFUL / len(CLAIMS), 3),
        "parse_failed": False,
        # Carried so the stub and the notebook keep the same shape. Real runs put the reason a
        # generation failed to parse here - "no JSON object" (the model answered in prose) versus
        # "unterminated JSON object" (max_tokens too low) - because one boolean cannot tell a
        # compliance failure from a budget one, and they need different fixes.
        "notes": [],
        "caveats": ["Claims cite features measured from this video; the baseline they are "
                    "compared against is the declared reference above."],
    }

    # C1-C4 per claim, computed by the same `verdict()` the /demo route uses - so the
    # deliberately-inverted c3 fails here too and the stage view can be seen REJECTING
    # something rather than only rendering passes.
    by_id = {c["claim_id"]: c for c in CLAIMS}
    checks = []
    for v in VERDICTS:
        c = by_id[v["claim_id"]]
        faithful = all(v[k] for k in CHECKS)
        checks.append({
            "claim_id": v["claim_id"], "text": c["text"],
            "evidence_ref": c["evidence_ref"], "claimed_value": c["claimed_value"],
            "claimed_pct_change": c["claimed_pct_change"], "direction": c["direction"],
            "C1_ref_exists": v["ref_exists"], "C2_value_matches": v["value_matches"],
            "C3_pct_matches": v["pct_matches"],
            "C4_direction_consistent": v["direction_consistent"],
            "C5_prose_quoted_value": v["prose_quoted_value"],
            "faithful": faithful, "notes": v["notes"], "shown_to_caregiver": faithful,
        })

    def stage(n, name, status, elapsed, payload):
        return {"agent": n, "name": name, "status": status,
                "elapsed_s": elapsed, "payload": payload}

    stages = [
        stage(1, "perception", "done", 0.0, {
            "fps": VIDEO_FPS, "frames_kept": VIDEO_FRAMES, "truncated": False,
            "providers": ["CPUExecutionProvider (dev stub)"], "n_people": len(tracks),
            "reid": True,
            # Parity with notebook 05: why re-id did or did not run. The stub's fixture
            # gallery "matched", so its note says the request was honoured.
            "reid_note": {"requested": True, "gallery_attached": True,
                          "weights_present": True},
            "tracks": [{"track_id": t["track_id"], "role": t["role"]} for t in tracks],
            # Pose-quality diagnostics. The stub reports a HEALTHY clip: high mean confidence
            # and one weak record (the twelve-frame unusable stretch), so the ">25% weak" and
            # "container disagrees" banners stay off here and can only fire on real footage.
            "mean_kp_score": 0.71,
            "person_records": 240,
            "weak_person_records": 12,
            "container_size": [1280, 720],
            "decoded_size": [1280, 720],
            # A 25 fps source resampled to the 20 Hz the Toyota shards were built at, so
            # the window is 1.50 s and the rate banner stays off - matching the served
            # configuration, not the retired 15 Hz Charades one. The banner can therefore
            # only fire on real footage from a camera the training corpus never matched.
            "source_fps": 25.0,
            "window_seconds": 1.5,
            "rate_matches_shards": True,
            "trained_fps": 20.0,
        }),
        stage(2, "activity", "done", 0.0, {
            "streams_served": HEALTH["streams"], "tau": 0.25, "temperature": 0.76,
            "decoding": "calibrate -> object fusion -> Viterbi -> abstain -> segments",
            "n_segments": sum(t["n_segments"] for t in tracks), "per_track": tracks,
        }),
        stage(3, "behaviour", "done", 0.0, stage3),
        stage(4, "report", "done", 0.0, stage4),
    ]

    return {
        "fps": VIDEO_FPS, "width": VIDEO_W, "height": VIDEO_H,
        "frames_kept": VIDEO_FRAMES, "truncated": False, "reid": True,
        "providers": ["CPUExecutionProvider (dev stub)"],
        "frames": frames,
        "tracks": tracks,
        "n_people": len(tracks),
        "stages": stages,
        "checks": checks,
        # Seam metadata. A caller running Agents 3-4 itself needs the day, the observed span and
        # the per-track frame counts to assert a subject; `stages_run` stops a truncated response
        # being mistaken for a complete one that produced no claims.
        "stages_run": "1234",
        "subject_provenance": "reid_matched",
        "subject_track": 0,
        "subject_tracks": [],
        "frames_per_track": {"0": 240, "1": 180},
        # First/last frame per track. Only the pose half has per-frame observations, and the
        # local half needs them to reunite one person's fragmented track ids.
        "track_spans": {"0": [0, 239], "1": [40, 219]},
        # First/last box per track, for the reachability constraint: disjoint spans alone
        # would merge a television in shot into the resident.
        "track_boxes": {"0": [[300, 200, 360, 400], [300, 200, 360, 400]],
                        "1": [[310, 205, 368, 402], [310, 205, 368, 402]]},
        # Parity with notebook 05: what the local half needs to persist an enrolment. The
        # stub's track 0 "matched", so its gallery_updates say so - with fake embeddings,
        # since this stub has no OSNet. Enrolment against the stub writes garbage; it is a
        # rendering fixture, not an identity system.
        "gallery_updates": {"subject_track": 0, "matched": True,
                            "embeddings": [[0.1, 0.2, 0.3, 0.4]]},
        "day": "2026-01-22",
        "observed_hours": 0.0083,
        "rss_mb": {"start": 900.0, "after_pose": 900.0,
                   "after_classify": 900.0, "after_report": 900.0},
        "timing": {"pose_s": 0.0, "classify_s": 0.0, "behaviour_s": 0.0, "report_s": 0.0},
        "tau": 0.25, "temperature": 0.76,
        "report": stage4["summary"],
        "report_provenance": (
            "Claims are verified against features measured from THIS video. The baseline "
            "they are compared to is a declared simulated reference, because a 14-day "
            "rolling median cannot come from one clip - see stages[2].caveats."),
    }


class Handler(http.server.BaseHTTPRequestHandler):
    # Chunked transfer needs HTTP/1.1; the default HTTP/1.0 makes the client read to EOF and
    # the explicit chunk headers would be delivered as body bytes. Every `_send` reply sets
    # Content-Length, which is what keep-alive requires in return.
    protocol_version = "HTTP/1.1"

    def _send(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        # The page is served from :5173 and this from :8899, so CORS is required here for
        # the same reason it is required on Kaggle.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send({})

    def _stream(self, events, delay: float = 0.0) -> None:
        """NDJSON, chunked, flushed per line — the shape notebook 05 sends.

        Not a nicety on either side. The Cloudflare quick tunnel drops a request whose origin
        has not answered in about two minutes (measured: /demo cut at 125.8 s with a 524 while
        the T4 was still generating), so the real backend streams and answers immediately.

        This stub must stream too, or the page gets developed against a shape the GPU never
        sends — which is the whole reason this file exists. `delay` fakes the per-agent wait so
        the progressive reveal can actually be seen locally; without it every line arrives in
        the same tick and the streaming path looks identical to the old blocking one.
        """
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            for i, ev in enumerate(events):
                if delay and i:
                    time.sleep(delay)
                line = (json.dumps(ev) + "\n").encode()
                self.wfile.write(f"{len(line):X}\r\n".encode() + line + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass                      # the page navigated away mid-stream; nothing to report

    def do_POST(self) -> None:
        if not self.path.startswith("/video"):
            self._send({"detail": f"no route {self.path}"}, 404)
            return
        # Read and discard the upload. The stub cannot decode video, and pretending to would
        # make the response depend on the file — which is precisely the thing it must not
        # imply. Reading it fully still exercises the real upload path in the browser.
        n = int(self.headers.get("Content-Length") or 0)
        remaining = n
        while remaining > 0:
            remaining -= len(self.rfile.read(min(1 << 20, remaining)) or b"")
        # Rejections happen BEFORE the stream, while an HTTP status is still available. Once
        # the first chunk is out, 413/422 are gone and an error has to travel in-band.
        if n > 60 << 20:
            self._send({"detail": "upload exceeds 60 MB"}, 413)
            return
        if n == 0:
            self._send({"detail": "empty upload"}, 422)
            return
        payload = _video_payload()
        events = [{"event": "accepted", "max_frames": 900}]
        for s in payload["stages"]:
            # A heartbeat before Agent 4, mirroring the real backend. Generation is the one
            # stage that outlasts the client's silence watchdog, so the notebook beats through
            # it on a worker thread; a stub that never beat would let the page be built as if
            # `heartbeat` did not exist.
            if s["agent"] == 4:
                events.append({"event": "heartbeat", "step": "qwen_generating",
                               "elapsed_s": 10.0})
            events.append({"event": "stage", "stage": s})
        events.append({"event": "result", "payload": payload})
        self._stream(events, delay=STREAM_DELAY)

    def do_GET(self) -> None:
        if self.path.startswith("/health"):
            self._send(HEALTH)
        elif self.path.startswith("/demo"):
            self._stream([
                {"event": "accepted", "scenario": 0, "cached": False},
                {"event": "progress", "step": "state_ready",
                 "day": DEMO.get("day"), "alerts": len(DEMO.get("alerts") or []),
                 "note": "writing the report - this is the slow stage"},
                {"event": "heartbeat", "step": "qwen_generating", "elapsed_s": 10.0},
                {"event": "result", "payload": DEMO},
            ], delay=STREAM_DELAY)
        else:
            self._send({"detail": f"no route {self.path}"}, 404)

    def log_message(self, *args) -> None:      # noqa: D102 - quiet by default
        pass


if __name__ == "__main__":
    # BS_PORT exists so a dev stub can run WITHOUT sharing the production port. Both this file
    # and `local_backend.py` default to 8899, and on Windows `allow_reuse_address` let both bind
    # it at once - measured as two PIDs listening simultaneously, with the browser served by the
    # wrong one. Anyone running the stub alongside the real backend should set BS_PORT=8898.
    PORT = int(os.environ.get("BS_PORT", PORT))

    # THREADED. A stream now holds its connection open for seconds, and the page polls
    # /health every 15 s throughout — on a single-threaded server that poll blocks behind the
    # stream, the indicator drops to "No answer" mid-upload, and the page reports a dead
    # backend while it is busily talking to one.
    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True

        def server_bind(self) -> None:
            # NOT `allow_reuse_address`: on Windows SO_REUSEADDR means "share a live port with
            # another process", not POSIX's "reuse TIME_WAIT" - which is how two backends ended
            # up listening on 8899 together with the browser talking to the wrong one.
            # SO_EXCLUSIVEADDRUSE makes a taken port fail at startup instead, loudly.
            import socket as _socket
            if hasattr(_socket, "SO_EXCLUSIVEADDRUSE"):
                self.socket.setsockopt(_socket.SOL_SOCKET, _socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                self.socket.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
            super().server_bind()

    with Server(("127.0.0.1", PORT), Handler) as srv:
        print(f"dev backend on http://127.0.0.1:{PORT}")
        print(f"  {len(CLAIMS)} claims, {len(CLAIMS) - FAITHFUL} of them unfaithful by design")
        print(f"  NDJSON streaming, {STREAM_DELAY:g}s between lines "
              f"(BS_STREAM_DELAY=0 to disable)")
        srv.serve_forever()
