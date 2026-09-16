"""Agents 3 and 4 as a library function, so the Kaggle notebook and the local backend share one.

Why this exists
---------------
Agent 4's keys live on the operator's machine, so Agent 4 should run there - and the split lands
exactly on the **Agent 2 -> Agent 3** seam, which is where this project already claims pixels
stop. `state_to_payload` exists precisely so Agent 4 sees numbers only; cutting here enforces a
boundary the design already asserts rather than inventing one.

    browser  ->  localhost                ->  Kaggle tunnel
                 Agent 3  (numpy)             Agent 1  RTMO      (GPU)
                 Agent 4  (hosted LLM)        Agent 2  ST-GCN++  (GPU)
                 C1-C5    (pure Python)

That means two programs need identical Agent 3/4 staging, and two implementations of one payload
is the exact failure `tests/test_web_contract.py` was written after: the dev stub and the notebook
drifted, five key-comparison tests passed, and every upload 500'd. So the staging is here, once,
and both callers import it.

Three decisions carried over from the served version, each one paid for
----------------------------------------------------------------------
**The clip is dated AFTER the primed reference window.** Baselines come from
`compute(before=day)`, and an uploaded clip's timestamps start at the decoder's origin -
2026-01-01, the same date the simulator's first reference day carries. `before=` then matched
nothing, so every baseline was empty, every deviation was empty, Agent 4 wrote "0 features
reviewed", and the C1-C5 table rendered blank while all four stages reported `done`.

**Identity is asserted and declared.** OSNet matches against an enrolled gallery; a stranger's
clip has none, so every track is `unknown` and `aggregate_daily_features`' subject filter matched
nobody - all 27 features read 0.0 while Agent 2 reported real segments. The most-present track is
treated as the subject and `subject_provenance` says so.

**The baseline is a declared reference, not this person.** One upload cannot supply a 14-day
rolling median. Feature VALUES are measured and real; every robust-z is "unlike the reference".
"""

from __future__ import annotations

import time
from datetime import timedelta
from typing import Any, Iterator, Sequence

from behaviorsense.schemas import ActivitySegment, Role

REFERENCE_DAYS = 21


def primed_reference(days: int = REFERENCE_DAYS):
    """A `BehaviourAnalyzer` primed on `days` simulated days, plus the day after the last.

    Returns `(analyzer, next_day, n_primed)`. The date matters as much as the analyzer: it is what
    stops `compute(before=day)` matching nothing.
    """
    from behaviorsense.agents.behaviour import BehaviourAnalyzer
    from behaviorsense.data.simulator import standard_scenarios

    analyzer = BehaviourAnalyzer()
    ref = next(iter(standard_scenarios())).run()
    primed = 0
    for day in ref.days[:days]:
        analyzer.analyze_day(day)
        primed += 1
    return analyzer, ref.days[primed - 1].day + timedelta(days=1), primed


def track_spans(frames: Sequence[Any]) -> dict[int, tuple[int, int]]:
    """First and last frame index each `track_id` appears in.

    Spans rather than per-frame sets: fragmentation from detection dropout produces tracks that
    are SEQUENTIAL in time, so first/last is enough to tell "the same person, re-detected" from
    "two people in the room", and it is one integer pair per track instead of 900 sets.
    """
    spans: dict[int, list[int]] = {}
    for f in frames:
        i = int(getattr(f, "frame_idx", 0))
        for p in getattr(f, "persons", ()) or ():
            t = int(p.track_id)
            cur = spans.get(t)
            if cur is None:
                spans[t] = [i, i]
            else:
                cur[0] = min(cur[0], i)
                cur[1] = max(cur[1], i)
    return {t: (a, b) for t, (a, b) in spans.items()}


def box_centroid(box: Sequence[float]) -> tuple[float, float]:
    """Centre of a box. `box` is [x1, y1, x2, y2] in the decoder's pixel space."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_height(box: Sequence[float]) -> float:
    return max(1.0, float(box[3]) - float(box[1]))


def reachable(prev: Sequence[float], nxt: Sequence[float], gap_s: float, *,
              max_body_heights_per_s: float = 1.5, slack_body_heights: float = 1.0) -> bool:
    """Could one walking person have got from `prev` to `nxt` in `gap_s` seconds?

    A body-height is the unit because a bounding box's height tracks how tall the person looks,
    so distance measured in box heights needs no camera calibration and no pixel-per-metre guess.
    A person walks at roughly 1.2 body-heights per second, and this defaults to 1.5 with one
    further body-height of slack for tracker handoff jitter and a wide first step.

    NO DETECTOR NEEDED, which is the point. `tv_duration_s` and `drinking_events` both need
    RT-DETR, but this guard does not: it rejects a candidate fragment by ARITHMETIC rather than by
    recognising a television. Measured against the failure it exists for - a nine-minute kitchen
    clip at 640x480 whose two screenshots both contain a television showing people - a TV person's
    box is fixed in a corner while the resident moves around the counter, so the walk between the
    resident's last box and the TV person's first box exceeds what any person could cover in the
    gap. The reverse mistake matters more: wrongly splitting one person re-creates the discarded
    activity this merge was built to stop, so the slack is deliberately generous.
    """
    if gap_s <= 0:
        return True                 # same instant: an overlap check, not a distance check
    px, py = box_centroid(prev)
    nx, ny = box_centroid(nxt)
    allowed = (max_body_heights_per_s * gap_s + slack_body_heights) * box_height(prev)
    return ((nx - px) ** 2 + (ny - py) ** 2) ** 0.5 <= allowed


def disjoint_chain(spans: dict[int, tuple[int, int]],
                   weights: dict[int, int],
                   *, boxes: dict[int, tuple[Sequence[float], Sequence[float]]] | None = None,
                   fps: float = 1.0) -> list[int]:
    """The heaviest chain of tracks that is consistently one person. Longest path over a DAG.

    Two constraints, and they are different in kind:

    * **Spans must not overlap.** Decisive and not a heuristic: two tracks seen in the SAME frame
      are two people, full stop. Tracks that never co-occur cannot be separated without a gallery,
      and with re-identification off there is none.
    * **Consecutive fragments must be REACHABLE.** Disjoint spans alone merge anything sequential -
      including a television in the corner, since RTMO detects the people on the screen and a
      screen does not share frames with the room. `boxes` carries each track's first and last box,
      so consecutive fragments are also required to be a walk a person could have made.

    Heaviest rather than longest-single: measured on a 45 s upload, one person produced three
    tracks and keeping only the most-present one discarded 10.5 s of housework and ALL 8.6 s of
    cooking, so `cooking_duration_s` read 0.00 on the same page that showed a
    `cooking_food_prep 8.6s` segment. The weight is frames observed, so the chain maximises
    evidence kept rather than the count of fragments.

    Longest path rather than interval scheduling because the reachability constraint is not
    transitive: A reachable from B and B from C does not imply A from C, so the best chain ending
    at `i` must examine every compatible predecessor rather than the nearest one. n is the number
    of tracks - single digits here - so the quadratic scan costs nothing.
    """
    items = sorted(((a, b, t) for t, (a, b) in spans.items()), key=lambda x: (x[1], x[0], x[2]))
    n = len(items)
    if n <= 1:
        return [t for _, _, t in items]

    def compatible(i: int, j: int) -> bool:
        """Can item j (earlier) precede item i in one person's chain?"""
        if items[j][1] >= items[i][0]:
            return False                        # spans overlap: two people in one frame
        if not boxes:
            return True
        bj, bi = boxes.get(items[j][2]), boxes.get(items[i][2])
        if not bj or not bi:
            return True
        gap_s = (items[i][0] - items[j][1]) / max(fps, 1e-6)
        return reachable(bj[1], bi[0], gap_s)

    best = [int(weights.get(t, 0)) for _, _, t in items]     # best chain ENDING at i
    prev: list[int | None] = [None] * n
    for i in range(n):
        for j in range(i):
            if not compatible(i, j):
                continue
            cand = best[j] + int(weights.get(items[i][2], 0))
            if cand > best[i]:
                best[i], prev[i] = cand, j
    end = max(range(n), key=lambda i: (best[i], -i))
    out: list[int] = []
    while end is not None:
        out.append(items[end][2])
        end = prev[end]
    return sorted(out)


def track_boxes(frames: Sequence[Any],
                spans: dict[int, tuple[int, int]]) -> dict[int, tuple[list[float], list[float]]]:
    """Each track's FIRST and LAST box, for the reachability constraint in `disjoint_chain`.

    Only the two ends are kept. The constraint is about consecutive fragments - where a person
    stopped and where they resumed - so a track that wandered around in between costs nothing to
    the decision and would just be more to carry over the wire.
    """
    out: dict[int, tuple[list[float], list[float]]] = {}
    for f in frames:
        i = int(getattr(f, "frame_idx", 0))
        for p in getattr(f, "persons", ()) or ():
            t = int(p.track_id)
            span = spans.get(t)
            if span is None:
                continue
            box = [float(p.box.x1), float(p.box.y1), float(p.box.x2), float(p.box.y2)]
            if i == span[0] and t not in out:
                out[t] = (box, box)             # frames arrive in order, so this is the first
            elif i == span[1]:
                out[t] = (out[t][0] if t in out else box, box)
    return out


def assert_subject(segments: Sequence[ActivitySegment],
                   frames_per_track: dict[int, int],
                   *, spans: dict[int, tuple[int, int]] | None = None,
                   boxes: dict[int, tuple[Sequence[float], Sequence[float]]] | None = None,
                   fps: float = 20.0,
                   ) -> tuple[list[ActivitySegment], int | None, str, list[int]]:
    """Decide which track(s) are the subject when re-identification found nobody.

    Returns `(segments, primary_track, provenance, subject_tracks)`.

    Without `spans` this is the old behaviour: the most-present track, and every other fragment's
    activity discarded. With them, tracks whose spans never overlap - and whose boxes are a walk a
    person could actually have made - are merged, because a person who leaves detection for more
    than the tracker's bridge horizon (15 s at 20 Hz for a confirmed track) comes back with a NEW
id and there is otherwise nothing
    to reunite them. A real ReID match is never overridden: if any segment already carries
    RESIDENT, the gallery spoke and this stays out of the way.
    """
    if not segments or Role.RESIDENT in {s.role for s in segments}:
        return list(segments), None, "reid_matched", []
    with_segs = {s.track_id for s in segments}
    cands = {k: v for k, v in frames_per_track.items() if k in with_segs}
    if not cands:
        return list(segments), None, "reid_matched", []

    if spans:
        # Only tracks that actually produced a segment can be the subject: an id that never filled
        # a 30-frame window contributes nothing to merge and would widen the chain for free.
        chain = disjoint_chain(
            {t: s for t, s in spans.items() if t in cands}, cands,
            boxes={t: b for t, b in (boxes or {}).items() if t in cands}, fps=fps)
    else:
        chain = []
    if len(chain) > 1:
        subject_tracks = chain
        provenance = "asserted_disjoint_track_chain"
    else:
        # Most-present, not first-seen: the longest-observed person is who the clip is about, and
        # first-seen would hand the subject slot to whoever walked past the lens.
        subject_tracks = [max(cands, key=lambda k: (cands[k], -k))]
        provenance = "asserted_most_present_track"
    chosen = set(subject_tracks)
    primary = max(subject_tracks, key=lambda k: (cands.get(k, 0), -k))
    return ([s.model_copy(update={"role": Role.RESIDENT}) if s.track_id in chosen else s
             for s in segments], primary, provenance, subject_tracks)


def behaviour_stage(segments: Sequence[ActivitySegment], *, day, observed_hours: float,
                    subject_track: int | None, subject_provenance: str,
                    subject_tracks: Sequence[int] = ()) -> tuple[dict, Any]:
    """Agent 3: segments -> `(stage_payload, BehaviourState)`. Numpy only, no GPU, no model."""
    from behaviorsense.agents.behaviour import aggregate_daily_features

    analyzer, ref_day, primed = primed_reference()
    today = aggregate_daily_features(segments, day=day, observed_hours=observed_hours)
    # The clip is re-dated to sit after the primed window. Without this every baseline is empty.
    today = today.model_copy(update={"day": ref_day})
    state = analyzer.analyze_day(today)

    merged = [int(t) for t in subject_tracks]
    reliable = bool(today.is_reliable())
    payload = {
        "baseline_provenance": "reference_cohort_simulated",
        "subject_provenance": subject_provenance,
        "subject_track": subject_track,
        "subject_tracks": merged,
        "baseline_days": primed,
        "real_days_from_this_video": 1,
        "observed_hours": round(observed_hours, 3),
        # THE SAME GATE THE MODEL'S PROMPT USES, applied to the page. Agent 3's deviation table
        # showed "walking_duration_s -10.00" on a 9m56s clip whose walking RATE was normal
        # (62.7 s/h against the baseline's ~67 s/h): a ten-minute total against a 22.9 h median
        # is a unit error, and it was withheld from Agent 4's prompt for exactly that reason.
        # A caregiver reads this card too, so the numbers are withheld here as well and the
        # reason is a field the page renders rather than a number it has to discount mentally.
        "deviations_withheld": not reliable,
        # CALL IT. `is_reliable` is a METHOD, so `bool(getattr(today, "is_reliable", False))`
        # evaluates a bound method object - always truthy - and reported a 29-second clip
        # (observed_hours 0.01) as a reliable day. The page suppresses its own "far below the 8 h a
        # day needs to be called reliable" warning on that field, so the one sentence telling a
        # caregiver not to act on this was the sentence the bug removed.
        "is_reliable": bool(today.is_reliable()),
        "features_from_video": {k: round(float(v), 2) for k, v in today.numeric_items().items()},
        "deviations": [] if not reliable else [
            {"feature": k, "robust_z": round(float(v), 2)}
            for k, v in sorted(state.deviations.items(),
                               key=lambda kv: -abs(kv[1]))[:10]],
        "alerts": [{"kind": a.kind.value, "severity": a.severity.value, "rule": a.rule_name}
                   for a in state.alerts],
        "caveats": [
            "The BASELINE is a simulated reference persona, not this person's history - one clip "
            "cannot supply 14 days.",
            "Feature VALUES are measured from the uploaded video and are real.",
            "Robust-z and alerts are therefore reference-relative: read them as 'unlike the "
            "reference', never as 'this person has declined'.",
        ] + ([
            "IDENTITY IS ASSERTED, not re-identified: no resident is enrolled, so the "
            f"most-present track ({subject_track}) was treated as the subject. OSNet reports "
            "every track as unidentified, which is correct - it has no gallery to match a "
            "stranger against. Without this the subject filter matches nothing and all 27 "
            "features read 0.",
        ] if subject_provenance == "asserted_most_present_track" else []) + ([
            f"TRACKS {', '.join(str(t) for t in merged)} WERE MERGED into one subject: their "
            "frame spans never overlap AND each fragment begins within walking distance of where "
            "the previous one ended, so they cannot be people present at the same time. A person "
            "who leaves detection for longer than the tracker's ~15 s bridge returns with a "
            "new track id, and counting "
            "only one fragment discarded the rest of their activity - measured once as "
            "cooking_duration_s reading 0.00 beside a cooking segment of 8.6 s. Two tracks seen "
            "in the SAME frame, or separated by more than a person could walk in the gap, are "
            "never merged.",
        ] if subject_provenance == "asserted_disjoint_track_chain" else []),
    }
    return payload, state


def report_stage(state, reporter) -> tuple[dict, list[dict]]:
    """Agent 4 + C1-C5: `(stage_payload, checks)`.

    `checks` carries the five verdicts SEPARATELY per claim. "verified" as one boolean hides which
    guarantee is doing the work, and C5 - does the prose actually quote the figure the field
    records - is the one the latest measured run says catches the most.
    """
    out = reporter.report(state)
    rate = out.hallucination_rate
    payload = {
        "model": out.report.model_name,
        "constrained_decoding": out.report.constrained_decoding,
        "summary": out.report.summary,
        "recommendation": out.report.recommendation,
        "escalate": out.report.escalate,
        "claims_emitted": out.n_emitted_claims,
        "claims_scorable": len(out.report.claims),
        "hallucination_rate": None if rate != rate else round(rate, 3),
        "parse_failed": out.parse_failed,
        # WHY it did not parse, not just that it did not. `_extract_json` raises two deliberately
        # different messages - "no JSON object in response" is a compliance failure (the model
        # answered in prose) and "unterminated JSON object (truncated generation)" is a budget
        # failure (max_tokens too low for a model that reasons before it answers). Dropping the
        # message reduced both to one boolean on the card, so a fallback model failing every clip
        # gave no way to tell which of the two it was.
        "notes": list(out.notes),
        "caveats": ["Claims cite features measured from this video; the baseline they are "
                    "compared against is the declared reference above."],
    }
    by_id = {c.claim_id: c for c in out.report.claims}
    checks = []
    for v in out.report.verifications:
        c = by_id.get(v.claim_id)
        checks.append({
            "claim_id": v.claim_id,
            "text": None if c is None else c.text,
            "evidence_ref": None if c is None else c.evidence_ref,
            "claimed_value": None if c is None else c.claimed_value,
            "claimed_pct_change": None if c is None else c.claimed_pct_change,
            "direction": None if c is None else c.direction,
            "C1_ref_exists": v.ref_exists,
            "C2_value_matches": v.value_matches,
            "C3_pct_matches": v.pct_matches,
            "C4_direction_consistent": v.direction_consistent,
            "C5_prose_quoted_value": v.prose_quoted_value,
            "faithful": v.is_faithful,
            "notes": list(v.notes),
            "shown_to_caregiver": v.is_faithful,
        })
    return payload, checks


def stage(n: int, name: str, status: str, elapsed: float, payload: dict) -> dict:
    """One stage record, in the shape `stages.js` reads."""
    return {"agent": n, "name": name, "status": status,
            "elapsed_s": round(elapsed, 2), "payload": payload}


def agents_3_and_4(segments: Sequence[ActivitySegment], *, day, observed_hours: float,
                   subject_track: int | None, subject_provenance: str,
                   reporter, t0: float | None = None,
                   subject_tracks: Sequence[int] = ()) -> Iterator[tuple[str, Any]]:
    """Yield ("stage", agent-3), ("stage", agent-4), then ("checks", [...]).

    A generator so the caller can emit Agent 3 BEFORE the LLM is called - generation is the slow
    stage and a page that fills in progressively beats one that shows nothing for a minute.

    Each agent is wrapped SEPARATELY. One try block around both meant an Agent 4 exception left
    stage 4 reading "no reporter loaded (weights absent)" - blaming a missing model for a crash in
    one that had loaded. A failure must be attributed to the agent that produced it.
    """
    t = t0 or time.time()
    s3 = None
    state = None
    try:
        s3, state = behaviour_stage(segments, day=day, observed_hours=observed_hours,
                                    subject_track=subject_track,
                                    subject_provenance=subject_provenance,
                                    subject_tracks=subject_tracks)
        t3 = time.time()
    except Exception as exc:                                        # noqa: BLE001
        s3, t3 = {"error": f"{type(exc).__name__}: {exc}"}, time.time()
    yield "stage", stage(3, "behaviour",
                         "done" if "error" not in s3 else "failed", t3 - t, s3)

    checks: list[dict] = []
    if state is None:
        s4 = {"why_skipped": "Agent 3 did not complete, so there is no verified state to write a "
                             "report from."}
    elif reporter is None:
        s4 = {"why_skipped": "no reporter configured - Agents 1-3 ran and the report was not "
                             "written."}
    else:
        try:
            s4, checks = report_stage(state, reporter)
        except Exception as exc:                                    # noqa: BLE001
            s4 = {"error": f"{type(exc).__name__}: {exc}"}
    t4 = time.time()
    yield "stage", stage(4, "report",
                         "done" if s4 and "error" not in s4 and "why_skipped" not in s4
                         else ("failed" if "error" in s4 else "skipped"), t4 - t3, s4)
    yield "checks", checks


__all__ = ["REFERENCE_DAYS", "agents_3_and_4", "assert_subject", "behaviour_stage",
           "box_centroid", "box_height", "disjoint_chain", "primed_reference",
           "reachable", "report_stage", "stage", "track_boxes", "track_spans"]
