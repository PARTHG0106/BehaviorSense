"""Tests for Agent 1 (perception): tracking, open-set ReID, role assignment.

Anti-vacuous discipline (see results/tuning_log.md, "Two vacuous tests")
----------------------------------------------------------------------
The failure mode this file is written against is a test suite that proves nothing:

  - A gallery whose threshold is set absurdly tight REJECTS EVERYTHING, and would pass
    every "stranger must be UNKNOWN" assertion. So every rejection test is paired with a
    positive control - a query that MUST match, on the same gallery, with the same
    thresholds. Rejection is only meaningful if acceptance is also demonstrated.
  - A tracker that never deletes tracks would pass every "ID survives occlusion" test.
    So P3 also asserts the IoU between the last observed box and the re-appearance box is
    exactly 0.0, which proves the recovery could ONLY have come from motion prediction.
  - Threshold tests place queries at measured distances either side of the documented
    operating point, rather than at distances so extreme that any threshold passes.

Every test prints the quantity it is really measuring, so a passing test can be audited.

Run: python tests/test_perception.py
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behaviorsense.agents.perception import (  # noqa: E402
    GalleryEntry,
    OpenSetGallery,
    PerceptionAgent,
    PerceptionConfig,
    RawDetection,
    SimpleTracker,
    Track,
    cosine_distance,
    iou,
    role_durations,
    torso_vertical_ratio,
)
from behaviorsense.schemas import (  # noqa: E402
    BoundingBox,
    DetectedObject,
    Keypoints,
    Role,
)

T0 = datetime(2026, 3, 1, 9, 0, 0)
DIM = 128


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def box(x1: float, y1: float, w: float = 60.0, h: float = 160.0) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x1 + w, y2=y1 + h)


def kp(n_visible: int = 17, fallen: bool = False) -> Keypoints:
    """COCO-17 keypoints with `n_visible` scored above the 0.3 visibility threshold."""
    xy = [(100.0 + i, 100.0 + i) for i in range(17)]
    if fallen:
        # Shoulders and hips side by side at equal height -> horizontal torso.
        xy[5], xy[6] = (100.0, 150.0), (110.0, 150.0)
        xy[11], xy[12] = (200.0, 150.0), (210.0, 150.0)
    else:
        xy[5], xy[6] = (100.0, 100.0), (110.0, 100.0)
        xy[11], xy[12] = (100.0, 200.0), (110.0, 200.0)
    scores = [0.9] * n_visible + [0.0] * (17 - n_visible)
    return Keypoints(xy=xy, scores=scores)


class ScriptedDetector:
    """Replays a per-frame list of detections. Stands in for RTMO."""

    def __init__(self, script: list[list[RawDetection]]) -> None:
        self.script = script
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[RawDetection]:
        out = self.script[self.calls] if self.calls < len(self.script) else []
        self.calls += 1
        return list(out)


class ZoneEmbedder:
    """Returns an embedding chosen by which horizontal zone the box centre falls in.

    Lets a test place a specific identity at a specific screen position, which is how
    role-assignment behaviour is driven without a real OSNet checkpoint.
    """

    def __init__(self, zones: list[tuple[float, np.ndarray]]) -> None:
        self.zones = zones  # (x_upper_bound, embedding)
        self.calls = 0

    @property
    def dim(self) -> int:
        return DIM

    def embed(self, frame: np.ndarray, b: BoundingBox) -> np.ndarray:
        self.calls += 1
        cx, _ = b.centroid
        for upper, emb in self.zones:
            if cx < upper:
                return emb
        return self.zones[-1][1]


class CountingObjectDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[DetectedObject]:
        self.calls += 1
        return [DetectedObject(label="pill_bottle", confidence=0.8, box=box(10, 10, 20, 20))]


def frames(n: int) -> list[np.ndarray]:
    return [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(n)]


def stamps(n: int) -> list[datetime]:
    return [T0 + timedelta(seconds=i / 15.0) for i in range(n)]


# -- embedding geometry -----------------------------------------------------


def basis(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """An orthonormal pair (u, w) in R^DIM."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=DIM)
    u /= np.linalg.norm(u)
    w = rng.normal(size=DIM)
    w -= np.dot(w, u) * u
    w /= np.linalg.norm(w)
    return u, w


def at_angle(u: np.ndarray, w: np.ndarray, theta: float) -> np.ndarray:
    """Unit vector at angle `theta` from u, inside the plane spanned by (u, w).

    Gives exact control over cosine distance: d = 1 - cos(theta). This is what makes the
    threshold tests real - a query can be placed at 0.29 and 0.31 either side of the
    documented 0.30 operating point instead of at 0.01 and 1.0, where any threshold
    whatsoever would pass.
    """
    return math.cos(theta) * u + math.sin(theta) * w


def at_distance(u: np.ndarray, w: np.ndarray, d: float) -> np.ndarray:
    return at_angle(u, w, math.acos(max(-1.0, min(1.0, 1.0 - d))))


# ---------------------------------------------------------------------------
# P1 - geometry primitives
# ---------------------------------------------------------------------------


def test_p1_iou_and_cosine():
    a = box(0, 0, 100, 100)
    assert abs(iou(a, a) - 1.0) < 1e-9
    assert iou(a, box(200, 200, 100, 100)) == 0.0
    half = iou(a, box(50, 0, 100, 100))
    assert abs(half - (5000 / 15000)) < 1e-9, half

    u, w = basis(1)
    assert cosine_distance(u, u) < 1e-9
    assert abs(cosine_distance(u, w) - 1.0) < 1e-9, cosine_distance(u, w)
    assert abs(cosine_distance(u, -u) - 2.0) < 1e-9
    # A zero vector must not silently look identical to everything.
    assert cosine_distance(u, np.zeros(DIM)) == 2.0
    for d in (0.05, 0.30, 0.75):
        got = cosine_distance(u, at_distance(u, w, d))
        assert abs(got - d) < 1e-9, f"at_distance({d}) gave {got}"
    print(f"  P1 iou(half-overlap)={half:.4f}, at_distance exact to 1e-9")


# ---------------------------------------------------------------------------
# P2/P3 - tracking
# ---------------------------------------------------------------------------


def test_p2_track_id_persists_and_min_hits_gates():
    """A walking person keeps one ID; a one-frame blip never becomes a person."""
    cfg = PerceptionConfig()
    tracker = SimpleTracker(cfg)

    ids: list[int] = []
    reported_at: list[int] = []
    for f in range(10):
        dets = [RawDetection(box=box(100 + 8 * f, 100), confidence=0.9)]
        if f == 4:
            # Single-frame spurious detection somewhere else in the room.
            dets.append(RawDetection(box=box(600, 300), confidence=0.9))
        active = tracker.update(dets)
        for t in active:
            ids.append(t.track_id)
            reported_at.append(f)

    assert set(ids) == {0}, f"expected one persistent ID, got {sorted(set(ids))}"
    first = min(reported_at)
    assert first == cfg.min_hits - 1, f"confirmed at frame {first}, expected {cfg.min_hits-1}"
    print(f"  P2 one ID across 10 frames, confirmed at frame {first}; "
          f"1-frame blip never reported ({len(set(ids))} ID total)")


def test_p3_occlusion_recovery_requires_motion_prediction():
    """ID survives an 8-frame occlusion - and could not have without coasting.

    The proof that this is not vacuous: IoU between the last OBSERVED box and the
    re-appearance box is exactly 0.0. Only the velocity-coasted prediction can bridge it.
    The control run (short max_age) loses the ID on the same input.
    """
    # 15 px/frame across a 60 px-wide box. At 15 fps this is a person crossing a quarter
    # of their own width per frame - ordinary indoor walking. The association ceiling for
    # motion-only tracking is ~0.54 box widths/frame; it is measured in P3b rather than
    # assumed here, because a speed above the ceiling never establishes velocity at all
    # and this test would then fail for a reason unrelated to occlusion.
    speed, gap = 15.0, 8
    seen, hidden, back = 6, 8, 3

    def sequence() -> list[list[RawDetection]]:
        script: list[list[RawDetection]] = []
        for f in range(seen):
            script.append([RawDetection(box=box(100 + speed * f, 100), confidence=0.9)])
        script.extend([[] for _ in range(hidden)])
        for f in range(seen + hidden, seen + hidden + back):
            script.append([RawDetection(box=box(100 + speed * f, 100), confidence=0.9)])
        return script

    script = sequence()
    last_observed = script[seen - 1][0].box
    reappear = script[seen + hidden][0].box
    bridge_iou = iou(last_observed, reappear)
    assert bridge_iou == 0.0, f"test is vacuous: boxes overlap by IoU {bridge_iou}"

    def run(max_age: int) -> list[int]:
        tracker = SimpleTracker(PerceptionConfig(max_age_frames=max_age))
        out: list[int] = []
        for f, dets in enumerate(script):
            active = tracker.update(dets)
            if f >= seen + hidden:
                out.extend(t.track_id for t in active)
        return out

    kept = run(30)
    lost = run(gap - 3)  # deletes the track before it can be recovered

    assert kept and set(kept) == {0}, f"ID not preserved through occlusion: {kept}"
    assert 0 not in lost, f"control failed to lose the ID (max_age too generous): {lost}"
    print(f"  P3 gap={gap} frames, IoU(last_seen, reappear)={bridge_iou:.1f} -> "
          f"same ID {sorted(set(kept))} with max_age=30; control (max_age={gap-3}) "
          f"gave {sorted(set(lost)) or 'no confirmed track'}")


def test_p3b_association_ceiling_is_measured_not_assumed():
    """Find the speed at which motion-only association breaks, and state it.

    This is a capability limit, not a bug: with no appearance gating, association is
    IoU-only, so a person moving more than ~0.54 of their own box width per frame cannot
    be linked frame-to-frame and re-enters as a new identity. Real BoT-SORT closes this
    with embedding-gated association; SimpleTracker does not, and a limit that is
    measured and documented is honest where an undiscovered one is a latent bug.

    Note what prediction does and does not buy. Coasting bridges MISSED frames (P3); it
    cannot rescue a speed that never associated in the first place, because velocity is
    only learned from a successful match. That asymmetry is asserted here.
    """
    width = 60.0

    def survives(px_per_frame: float, n: int = 12) -> bool:
        tracker = SimpleTracker(PerceptionConfig())
        ids: set[int] = set()
        for f in range(n):
            for t in tracker.update(
                [RawDetection(box=box(100 + px_per_frame * f, 100, width), confidence=0.9)]
            ):
                ids.add(t.track_id)
        return ids == {0}

    # Analytic ceiling for two equal boxes overlapping at IoU = t:
    #   IoU = (w - d) / (w + d)  =>  d/w = (1 - t) / (1 + t)
    thr = PerceptionConfig().iou_threshold
    predicted = (1 - thr) / (1 + thr)

    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = (lo + hi) / 2
        if survives(mid * width):
            lo = mid
        else:
            hi = mid

    assert abs(lo - predicted) < 0.02, (
        f"measured ceiling {lo:.3f} box widths/frame disagrees with the analytic "
        f"{predicted:.3f} - the association rule is not doing what the docstring says"
    )
    assert survives(predicted * width * 0.9), "failed below the ceiling"
    assert not survives(predicted * width * 1.1), "succeeded above the ceiling"
    print(f"  P3b association ceiling = {lo:.3f} box widths/frame "
          f"(analytic {predicted:.3f} at IoU>{thr}); above it, ID switches. "
          f"At 15 fps / 60 px box that is ~{lo * width * 15:.0f} px/s.")


def test_p4_two_threshold_association():
    """A weak detection may EXTEND a track but must never CREATE one."""
    cfg = PerceptionConfig()
    weak = (cfg.det_low_confidence + cfg.det_confidence) / 2.0
    assert cfg.det_low_confidence < weak < cfg.det_confidence

    # (a) weak-only input creates nothing, no matter how long it persists.
    t1 = SimpleTracker(cfg)
    created = 0
    for f in range(12):
        t1.update([RawDetection(box=box(100 + 5 * f, 100), confidence=weak)])
        created = max(created, len(t1.tracks))
    assert created == 0, f"weak detections created {created} track(s)"

    # (b) same weak score, but after the track is established: it must survive.
    t2 = SimpleTracker(cfg)
    for f in range(4):
        t2.update([RawDetection(box=box(100 + 5 * f, 100), confidence=0.9)])
    survived = []
    for f in range(4, 12):
        active = t2.update([RawDetection(box=box(100 + 5 * f, 100), confidence=weak)])
        survived.extend(t.track_id for t in active)

    assert set(survived) == {0}, f"weak detections failed to extend the track: {survived}"
    assert len(survived) == 8, f"expected 8 extended frames, got {len(survived)}"
    print(f"  P4 conf={weak:.2f}: created 0 tracks from scratch, "
          f"extended an existing track for {len(survived)}/8 frames")


# ---------------------------------------------------------------------------
# P5-P8 - open-set gallery
# ---------------------------------------------------------------------------


def test_p5_open_set_accepts_and_rejects_at_the_documented_threshold():
    """The gallery must accept just inside the threshold and reject just outside it.

    A gallery that rejected everything would pass a stranger-only test. Both directions
    are asserted on the same gallery, with queries placed 0.01 either side of 0.30.
    """
    cfg = PerceptionConfig()
    u, w = basis(2)
    gallery = OpenSetGallery(cfg)
    gallery.enrol("margaret", Role.RESIDENT, [u])

    thr = cfg.reid_match_threshold
    inside = at_distance(u, w, thr - 0.01)
    outside = at_distance(u, w, thr + 0.01)
    stranger = basis(99)[0]

    hit, d_in = gallery.match(inside, box_area=20000)
    miss, d_out = gallery.match(outside, box_area=20000)
    none_, d_str = gallery.match(stranger, box_area=20000)

    assert hit is not None and hit.role is Role.RESIDENT, f"rejected a d={d_in:.3f} match"
    assert miss is None, f"accepted d={d_out:.3f} above threshold {thr}"
    assert none_ is None, f"accepted a stranger at d={d_str:.3f}"
    assert d_str > thr, f"stranger landed inside the threshold ({d_str:.3f})"
    print(f"  P5 thr={thr}: accept@{d_in:.3f}, reject@{d_out:.3f}, "
          f"stranger@{d_str:.3f} -> UNKNOWN")


def test_p6_ambiguous_match_is_refused():
    """Two enrolled people, near-tied distances -> refuse, with a positive control."""
    cfg = PerceptionConfig()
    u, w = basis(3)
    gallery = OpenSetGallery(cfg)

    theta_q = math.acos(0.80)          # query sits at d=0.20 from A
    theta_b = theta_q + math.acos(0.77)  # ...and d=0.23 from B
    gallery.enrol("margaret", Role.RESIDENT, [u])
    gallery.enrol("nora", Role.VISITOR, [at_angle(u, w, theta_b)])

    ambiguous = at_angle(u, w, theta_q)
    clear = at_distance(u, w, 0.10)

    d_a = cosine_distance(ambiguous, u)
    d_b = cosine_distance(ambiguous, at_angle(u, w, theta_b))
    assert d_a < cfg.reid_match_threshold and d_b < cfg.reid_match_threshold, (
        f"test is vacuous: only one candidate is inside the threshold "
        f"(d_a={d_a:.3f}, d_b={d_b:.3f})"
    )
    assert abs(d_b - d_a) < cfg.reid_margin

    refused, _ = gallery.match(ambiguous, box_area=20000)
    accepted, d_acc = gallery.match(clear, box_area=20000)

    assert refused is None, f"accepted an ambiguous match (gap {abs(d_b-d_a):.3f})"
    assert accepted is not None and accepted.name == "margaret", (
        "positive control failed: an unambiguous query was also refused, so the "
        "rejection above proves nothing"
    )
    print(f"  P6 d_A={d_a:.3f} d_B={d_b:.3f} gap={abs(d_b-d_a):.3f}<{cfg.reid_margin} "
          f"-> refused; control d={d_acc:.3f} -> matched {accepted.name}")


def test_p7_tiny_crops_are_not_trusted():
    cfg = PerceptionConfig()
    u, _ = basis(4)
    gallery = OpenSetGallery(cfg)
    gallery.enrol("margaret", Role.RESIDENT, [u])

    small = cfg.min_embed_box_area * 0.5
    large = cfg.min_embed_box_area * 2.0
    assert gallery.match(u, box_area=small)[0] is None, "trusted a tiny crop"
    hit, _ = gallery.match(u, box_area=large)
    assert hit is not None, "rejected a perfect match on a large crop"
    print(f"  P7 identical embedding: area={small:.0f}px -> UNKNOWN, "
          f"area={large:.0f}px -> {hit.name}")


def test_p8_centroid_adapts_without_being_poisoned():
    """One bad crop must move the centroid only slightly, and clean data must undo it."""
    cfg = PerceptionConfig()
    u, _ = basis(5)
    stranger = basis(77)[0]
    gallery = OpenSetGallery(cfg)
    entry = gallery.enrol("margaret", Role.RESIDENT, [u])

    gallery.update_centroid("margaret", stranger)
    drift = cosine_distance(entry.centroid, u)
    assert drift < cfg.reid_match_threshold / 3.0, (
        f"a single bad crop moved the centroid by {drift:.4f}"
    )
    assert gallery.match(u, box_area=20000)[0] is not None, "poisoned out of a true match"

    for _ in range(20):
        gallery.update_centroid("margaret", u)
    recovered = cosine_distance(entry.centroid, u)
    assert recovered < drift / 10.0, f"no recovery: {drift:.5f} -> {recovered:.5f}"

    # Adaptation must actually happen, or "resists poisoning" is just "ignores input".
    fresh = OpenSetGallery(cfg)
    e2 = fresh.enrol("arthur", Role.RESIDENT, [u])
    target = at_distance(u, basis(5)[1], 0.20)
    for _ in range(20):
        fresh.update_centroid("arthur", target)
    moved = cosine_distance(e2.centroid, u)
    assert moved > 0.05, f"centroid did not adapt at all (moved {moved:.4f})"
    print(f"  P8 poison drift={drift:.5f} -> after 20 clean updates {recovered:.5f}; "
          f"sustained new appearance moved centroid {moved:.3f}")


def test_p8b_gallery_rejects_unknown_enrolment():
    gallery = OpenSetGallery()
    u, _ = basis(6)
    for bad, why in ((Role.UNKNOWN, "UNKNOWN is the rejection outcome"), (None, "no embeddings")):
        try:
            if bad is Role.UNKNOWN:
                gallery.enrol("x", Role.UNKNOWN, [u])
            else:
                gallery.enrol("x", Role.RESIDENT, [])
        except ValueError:
            continue
        raise AssertionError(f"enrol accepted an invalid case: {why}")
    assert len(gallery) == 0
    print("  P8b enrol rejects UNKNOWN role and empty embedding list")


# ---------------------------------------------------------------------------
# P9-P10 - role assignment
# ---------------------------------------------------------------------------


def agent_for_voting(cfg: PerceptionConfig) -> PerceptionAgent:
    return PerceptionAgent(detector=ScriptedDetector([]), config=cfg)


def test_p9_role_needs_evidence_before_it_is_claimed():
    cfg = PerceptionConfig()
    agent = agent_for_voting(cfg)
    track = Track(track_id=0, box=box(100, 100), confidence=0.9)

    history: list[tuple[int, Role, float]] = []
    for i in range(1, cfg.role_min_votes + 3):
        agent._vote_role(track, Role.RESIDENT)
        history.append((i, track.role, track.role_confidence))

    for n, role, conf in history:
        if n < cfg.role_min_votes:
            assert role is Role.UNKNOWN, f"claimed {role} on only {n} vote(s)"
            assert conf == 0.0, f"UNKNOWN carried confidence {conf}"
        else:
            assert role is Role.RESIDENT, f"still UNKNOWN after {n} consistent votes"
    assert track.role_confidence > 0.5
    print(f"  P9 UNKNOWN through {cfg.role_min_votes-1} votes, RESIDENT at "
          f"{cfg.role_min_votes} (conf {track.role_confidence:.2f})")


def test_p10_hysteresis_delays_switches_without_blocking_them():
    """A brief blip must not switch the role; a sustained change must.

    Measured differentially: the same vote stream is replayed with the margin disabled,
    and the switch must happen strictly earlier there. Without that comparison, a config
    that simply never switches would also pass "the blip caused no switch".
    """
    cfg = PerceptionConfig()

    # (a) a 2-frame blip during a settled RESIDENT track
    agent = agent_for_voting(cfg)
    track = Track(track_id=0, box=box(100, 100), confidence=0.9)
    for _ in range(10):
        agent._vote_role(track, Role.RESIDENT)
    assert track.role is Role.RESIDENT
    for _ in range(2):
        agent._vote_role(track, Role.VISITOR)
    assert track.role is Role.RESIDENT, "a 2-frame blip flipped the role"

    # (b) a sustained change, with and without the margin
    def switch_frame(margin: int) -> int | None:
        a = agent_for_voting(PerceptionConfig(role_switch_margin=margin))
        t = Track(track_id=0, box=box(100, 100), confidence=0.9)
        for _ in range(10):
            a._vote_role(t, Role.RESIDENT)
        assert t.role is Role.RESIDENT
        for f in range(1, 31):
            a._vote_role(t, Role.VISITOR)
            if t.role is Role.VISITOR:
                return f
        return None

    with_margin = switch_frame(cfg.role_switch_margin)
    without = switch_frame(0)

    assert without is not None, "control never switched - the stream cannot switch at all"
    assert with_margin is not None, "hysteresis blocked a sustained switch entirely"
    assert with_margin > without, (
        f"margin had no effect: switched at {with_margin} with margin "
        f"{cfg.role_switch_margin}, {without} without"
    )
    print(f"  P10 2-frame blip -> no switch; sustained switch at frame {with_margin} "
          f"(margin={cfg.role_switch_margin}) vs {without} (margin=0)")


def test_p10b_unknown_never_carries_confidence():
    """The schema forbids confident UNKNOWN; the voter must never construct one."""
    cfg = PerceptionConfig()
    agent = agent_for_voting(cfg)
    track = Track(track_id=0, box=box(100, 100), confidence=0.9)
    rng = np.random.default_rng(11)
    roles = [Role.UNKNOWN, Role.RESIDENT, Role.VISITOR, Role.CARER]
    worst = 0.0
    for _ in range(300):
        agent._vote_role(track, roles[int(rng.integers(0, 4))])
        if track.role is Role.UNKNOWN:
            worst = max(worst, track.role_confidence)
        assert 0.0 <= track.role_confidence <= 1.0, track.role_confidence
        assert len(track.role_votes) <= cfg.role_vote_window
    assert worst == 0.0, f"UNKNOWN reported confidence {worst}"
    print(f"  P10b 300 random votes: max confidence while UNKNOWN = {worst:.1f}, "
          f"window capped at {cfg.role_vote_window}")


# ---------------------------------------------------------------------------
# P11-P14 - agent integration
# ---------------------------------------------------------------------------


def test_p11_end_to_end_resident_and_stranger():
    """Two people in frame: an enrolled resident and an unknown visitor."""
    cfg = PerceptionConfig()
    u, w = basis(7)
    resident_emb = u
    stranger_emb = basis(88)[0]

    gallery = OpenSetGallery(cfg)
    gallery.enrol("margaret", Role.RESIDENT, [resident_emb])

    n = 20
    script = [
        [
            RawDetection(box=box(100, 100), confidence=0.9, keypoints=kp()),
            RawDetection(box=box(500, 100), confidence=0.9, keypoints=kp()),
        ]
        for _ in range(n)
    ]
    agent = PerceptionAgent(
        detector=ScriptedDetector(script),
        embedder=ZoneEmbedder([(300.0, resident_emb), (10_000.0, stranger_emb)]),
        object_detector=CountingObjectDetector(),
        gallery=gallery,
        config=cfg,
    )
    obs = agent.process_sequence(frames(n), stamps(n))

    final = obs[-1]
    assert final.n_persons == 2, final.n_persons
    by_x = sorted(final.persons, key=lambda p: p.box.x1)
    left, right = by_x
    assert left.role is Role.RESIDENT, f"enrolled resident labelled {left.role}"
    assert left.role_confidence > 0.5
    assert agent.track_name(left.track_id) == "margaret"
    assert right.role is Role.UNKNOWN, f"stranger labelled {right.role}"
    assert right.role_confidence == 0.0
    assert agent.track_name(right.track_id) is None
    assert right.reid_distance is not None and right.reid_distance > cfg.reid_match_threshold
    assert left.track_id != right.track_id

    durations = role_durations(obs, fps=15.0)
    assert durations[Role.RESIDENT] > 0 and durations[Role.UNKNOWN] > 0
    print(f"  P11 resident d={left.reid_distance:.3f} conf={left.role_confidence:.2f}; "
          f"stranger d={right.reid_distance:.3f} -> UNKNOWN conf=0.0; "
          f"presence {durations[Role.RESIDENT]:.2f}s / {durations[Role.UNKNOWN]:.2f}s")


def test_p12_unusable_pose_is_dropped_but_the_person_is_kept():
    """Occluded pose must not reach Agent 2; presence must still be reported."""
    cfg = PerceptionConfig()
    n = 10

    def run(n_visible: int):
        script = [
            [RawDetection(box=box(100, 100), confidence=0.9, keypoints=kp(n_visible))]
            for _ in range(n)
        ]
        agent = PerceptionAgent(detector=ScriptedDetector(script), config=cfg)
        return agent.process_sequence(frames(n), stamps(n))

    occluded = run(cfg.min_visible_keypoints - 1)[-1].persons
    usable = run(cfg.min_visible_keypoints)[-1].persons

    assert len(occluded) == 1 and occluded[0].keypoints is None, (
        "unusable pose was passed downstream"
    )
    assert len(usable) == 1 and usable[0].keypoints is not None, (
        "positive control failed: usable pose was also dropped"
    )
    print(f"  P12 {cfg.min_visible_keypoints-1} visible -> person kept, keypoints=None; "
          f"{cfg.min_visible_keypoints} visible -> keypoints passed")


def test_p13_objects_sampled_at_1hz_but_available_every_frame():
    cfg = PerceptionConfig()
    fps, n = 15.0, 45
    script = [[RawDetection(box=box(100, 100), confidence=0.9, keypoints=kp())] for _ in range(n)]
    objects = CountingObjectDetector()
    agent = PerceptionAgent(
        detector=ScriptedDetector(script),
        object_detector=objects,
        config=cfg,
        fps=fps,
    )
    obs = agent.process_sequence(frames(n), stamps(n))

    expected = math.ceil(n / (fps / cfg.object_sample_hz))
    assert objects.calls == expected, f"{objects.calls} object calls, expected {expected}"
    assert all(o.objects for o in obs), "cached objects missing on non-sample frames"
    saving = 1.0 - objects.calls / n
    print(f"  P13 {n} frames -> {objects.calls} object-detector calls "
          f"({saving:.0%} saved), objects present on {sum(1 for o in obs if o.objects)}/{n}")


def test_p14_reset_clears_tracks_and_keeps_enrolment():
    cfg = PerceptionConfig()
    u, _ = basis(8)
    gallery = OpenSetGallery(cfg)
    gallery.enrol("margaret", Role.RESIDENT, [u])

    n = 12
    script = [
        [RawDetection(box=box(100, 100), confidence=0.9, keypoints=kp())] for _ in range(n * 2)
    ]
    agent = PerceptionAgent(
        detector=ScriptedDetector(script),
        embedder=ZoneEmbedder([(10_000.0, u)]),
        gallery=gallery,
        config=cfg,
    )
    first = agent.process_sequence(frames(n), stamps(n))
    assert first[-1].persons[0].role is Role.RESIDENT
    assert first[-1].frame_idx == n - 1

    agent.reset()
    assert agent._frame_idx == 0
    assert agent.tracker.tracks == [], "tracks survived reset"
    assert agent.track_name(0) is None, "track->name binding survived reset"
    assert len(agent.gallery) == 1, "reset wiped enrolment"

    second = agent.process_sequence(frames(n), stamps(n))
    assert second[0].frame_idx == 0, "frame index did not restart"
    assert second[-1].persons[0].role is Role.RESIDENT, "identity lost after reset"
    print(f"  P14 reset: frame_idx {n-1} -> 0, tracks cleared, "
          f"{len(agent.gallery)} enrolment kept, role re-acquired")


def test_p15_torso_ratio_separates_upright_from_horizontal():
    up = torso_vertical_ratio(kp())
    down = torso_vertical_ratio(kp(fallen=True))
    blind = torso_vertical_ratio(kp(n_visible=5))

    assert up is not None and down is not None
    assert up > 0.9, f"upright torso scored {up:.3f}"
    assert down < 0.1, f"horizontal torso scored {down:.3f}"
    assert blind is None, "reported a ratio from invisible keypoints"
    print(f"  P15 upright={up:.3f} fallen={down:.3f} occluded=None "
          f"(aspect_ratio cue: {box(0,0,60,160).aspect_ratio:.2f} standing vs "
          f"{box(0,0,160,60).aspect_ratio:.2f} lying)")


def test_p16_no_gallery_means_no_identity_claims():
    """With no embedder, every person must stay UNKNOWN rather than default to RESIDENT.

    This is the degenerate deployment (ReID checkpoint missing). Silently labelling
    everyone RESIDENT would let Agent 3 aggregate a visitor's activity into the
    resident's daily features, so the honest failure mode is "no identity at all".
    """
    n = 15
    script = [[RawDetection(box=box(100, 100), confidence=0.9, keypoints=kp())] for _ in range(n)]
    agent = PerceptionAgent(detector=ScriptedDetector(script))
    obs = agent.process_sequence(frames(n), stamps(n))
    people = [p for o in obs for p in o.persons]

    assert people, "no persons reported at all - test proves nothing"
    assert all(p.role is Role.UNKNOWN for p in people)
    assert all(p.role_confidence == 0.0 for p in people)
    assert all(p.reid_distance is None for p in people), "invented a ReID distance"
    print(f"  P16 no embedder: {len(people)} observations, all UNKNOWN, "
          f"reid_distance=None (no fabricated identity)")


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
