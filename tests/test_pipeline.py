"""Tests for the ensemble classifier and the frames -> segments pipeline.

These test the two seams that had no coverage because they had no implementation: nothing
loaded a trained checkpoint into Agent 2, and nothing converted Agent 1's per-frame output
into Agent 2's per-track windows. Both are places where a bug produces *plausible* output
rather than an error, which is the hardest kind to notice:

  - a tensor permuted M-for-V has exactly the right shape and is nonsense (E1)
  - a checkpoint trained on `bone` loads into a `joint` slot without complaint (E3)
  - windows pooled across two people classify a visitor's motion as the resident's (P2)
  - a padded window ends in zeroed joints, which looks like a body collapsing (P3)

Run: python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behaviorsense.agents.activity import CLASS_NAMES, FALLING, N_CLASSES  # noqa: E402
from behaviorsense.models.ensemble import (  # noqa: E402
    EnsembleClassifier,
    SyntheticClassifier,
    windows_to_tensor,
)
from behaviorsense.models.stgcnpp import STGCNpp  # noqa: E402
from behaviorsense.pipeline import (  # noqa: E402
    WINDOW_FRAMES,
    ActivityPipeline,
    PipelineStats,
    frames_to_windows,
    observed_hours_from_frames,
)
from behaviorsense.schemas import (  # noqa: E402
    BoundingBox,
    DetectedObject,
    FrameObservation,
    Keypoints,
    PersonObservation,
    Role,
)

T0 = datetime(2026, 3, 1, 9, 0, 0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def pose(y_offset: float = 0.0, x: float = 0.0, visible: int = 17) -> Keypoints:
    xy = [(x + i * 0.01, 1.0 - y_offset + i * 0.01) for i in range(17)]
    xy[5], xy[6] = (x - 0.2, 1.5 - y_offset), (x + 0.2, 1.5 - y_offset)
    xy[11], xy[12] = (x - 0.15, 0.5 - y_offset), (x + 0.15, 0.5 - y_offset)
    scores = [0.9] * visible + [0.0] * (17 - visible)
    return Keypoints(xy=xy, scores=scores)


def person(track_id: int, frame_idx: int, ts: datetime, *, role=Role.RESIDENT,
           y: float = 0.0, x: float = 0.0, room: str | None = "lounge",
           kp: Keypoints | None = None) -> PersonObservation:
    return PersonObservation(
        frame_idx=frame_idx,
        timestamp=ts,
        track_id=track_id,
        box=BoundingBox(x1=100 + x, y1=100 + y, x2=160 + x, y2=260 + y),
        detection_confidence=0.9,
        keypoints=pose(y_offset=y, x=x) if kp is None else kp,
        role=role,
        role_confidence=0.9 if role is not Role.UNKNOWN else 0.0,
        room=room,
    )


def sequence(n: int, *, fps: float = 15.0, builder=None,
             objects: tuple[str, ...] = ()) -> list[FrameObservation]:
    frames = []
    for i in range(n):
        ts = T0 + timedelta(seconds=i / fps)
        people = builder(i, ts) if builder else [person(0, i, ts)]
        frames.append(FrameObservation(
            frame_idx=i, timestamp=ts, persons=people,
            objects=[DetectedObject(label=o, confidence=0.8,
                                    box=BoundingBox(x1=1, y1=1, x2=20, y2=20))
                     for o in objects],
        ))
    return frames


# ---------------------------------------------------------------------------
# E: ensemble classifier
# ---------------------------------------------------------------------------


def test_e1_tensor_conversion_puts_axes_where_the_model_expects():
    """[N,T,M,17,3] -> [N,3,T,17,M], verified element-wise, not just by shape.

    A permutation that swaps M and V yields the right shape and total nonsense. This is
    checked with distinguishable values so a transposition cannot pass.
    """
    N, T, M = 2, 4, 2
    w = np.zeros((N, T, M, 17, 3), dtype=np.float32)
    for n in range(N):
        for t in range(T):
            for m in range(M):
                for v in range(17):
                    for c in range(3):
                        w[n, t, m, v, c] = n * 1000 + t * 100 + m * 50 + v + c * 0.1

    x = windows_to_tensor(w)
    assert tuple(x.shape) == (N, 3, T, 17, M), x.shape
    for n in range(N):
        for t in range(T):
            for m in range(M):
                for v in range(17):
                    for c in range(3):
                        assert abs(x[n, c, t, v, m].item() - w[n, t, m, v, c]) < 1e-5, (
                            f"axis mismatch at n={n} c={c} t={t} v={v} m={m}"
                        )
    try:
        windows_to_tensor(np.zeros((2, 4, 17, 3), dtype=np.float32))
        raise AssertionError("accepted a wrongly-ranked array")
    except ValueError:
        pass
    print(f"  E1 [{N},{T},{M},17,3] -> {tuple(x.shape)} verified element-wise")


def save_ckpt(path: Path, stream: str, seed: int = 0, n_classes: int = N_CLASSES) -> Path:
    torch.manual_seed(seed)
    model = STGCNpp(n_classes=n_classes)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "ema": model.state_dict(), "ema_step": 1,
         "epoch": 0, "global_step": 1, "best": 0.0,
         "args": {"stream": stream, "epochs": 1}, "metrics": {}},
        path,
    )
    return path


def test_e2_ensemble_loads_and_averages_streams():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, s in enumerate(("joint", "bone", "joint_motion", "bone_motion")):
            save_ckpt(td / f"adl_{s}" / "best.pt", s, seed=i)

        clf = EnsembleClassifier.from_run_dir(td)
        assert len(clf.streams) == 4, clf.streams

        w = np.random.default_rng(0).normal(0, 0.3, (5, WINDOW_FRAMES, 2, 17, 3)).astype(np.float32)
        w[..., 2] = 0.9
        out = clf.logits(w)
        assert out.shape == (5, N_CLASSES), out.shape
        assert np.isfinite(out).all()

        # The average must actually differ from every member, or averaging is a no-op.
        per = clf.per_stream_logits(w)
        assert set(per) == set(clf.streams)
        mean = np.mean([per[s] for s in clf.streams], axis=0)
        assert np.abs(out - mean).max() < 1e-4, "logits() is not the per-stream mean"
        for s in clf.streams:
            assert np.abs(out - per[s]).max() > 1e-3, f"ensemble == stream {s} alone"

        single = EnsembleClassifier({"joint": td / "adl_joint" / "best.pt"})
        assert np.abs(single.logits(w) - per["joint"]).max() < 1e-4
        print(f"  E2 4 streams loaded, logits == per-stream mean, differs from each "
              f"member (max dev {max(np.abs(out - per[s]).max() for s in clf.streams):.3f})")


def test_e3_mismatched_stream_checkpoint_is_refused():
    """A bone-trained checkpoint in a joint slot must fail loudly.

    The architecture is stream-agnostic, so this loads cleanly and produces confident
    garbage - exactly the failure that is invisible without an explicit check.
    """
    with tempfile.TemporaryDirectory() as td:
        p = save_ckpt(Path(td) / "adl_joint" / "best.pt", "bone")
        try:
            EnsembleClassifier({"joint": p})
        except ValueError as exc:
            assert "trained on stream" in str(exc), exc
        else:
            raise AssertionError("loaded a bone checkpoint as the joint stream")

        # Control: the matching stream loads fine, so the guard is not blanket-rejecting.
        ok = save_ckpt(Path(td) / "adl_bone" / "best.pt", "bone")
        assert EnsembleClassifier({"bone": ok}).streams == ["bone"]
        print("  E3 stream mismatch refused; matching stream accepted")


def test_e3b_partial_checkpoint_is_refused():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "bad.pt"
        model = STGCNpp(n_classes=N_CLASSES)
        state = {k: v for k, v in model.state_dict().items() if "blocks.0" not in k}
        torch.save({"model": state, "args": {"stream": "joint"}}, p)
        try:
            EnsembleClassifier({"joint": p})
        except RuntimeError as exc:
            assert "missing" in str(exc), exc
            print(f"  E3b partial checkpoint refused: {str(exc)[:64]}...")
            return
    raise AssertionError("accepted a checkpoint with missing weights")


def test_e4_logit_vs_prob_combination_differ():
    """The two combination modes must not be secretly the same code path.

    Logit averaging is a product of experts (a stream can veto); probability averaging is
    a mixture (it cannot). If these produced identical output the ablation would be
    measuring nothing.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, s in enumerate(("joint", "bone")):
            save_ckpt(td / f"adl_{s}" / "best.pt", s, seed=i + 7)
        w = np.random.default_rng(1).normal(0, 0.5, (4, WINDOW_FRAMES, 2, 17, 3)).astype(np.float32)
        w[..., 2] = 0.9

        a = EnsembleClassifier.from_run_dir(td, combine="logit").logits(w)
        b = EnsembleClassifier.from_run_dir(td, combine="prob").logits(w)
        from behaviorsense.agents.activity import softmax

        d = np.abs(softmax(a) - softmax(b)).max()
        assert d > 1e-4, f"logit and prob combination gave identical posteriors ({d:.2e})"
        print(f"  E4 logit vs prob combination: max posterior difference {d:.4f}")


def test_e5_synthetic_classifier_responds_to_geometry():
    """The stub must be driven by the input, not return a constant."""
    clf = SyntheticClassifier(seed=0)
    T = WINDOW_FRAMES

    def window(drop: float, jitter: float = 0.0) -> np.ndarray:
        w = np.zeros((1, T, 2, 17, 3), dtype=np.float32)
        for t in range(T):
            frac = t / (T - 1)
            w[0, t, 0, :, 1] = 1.0 - drop * frac
            w[0, t, 0, :, 0] = jitter * frac
        w[..., 2] = 0.9
        return w

    falling = clf.logits(window(drop=1.5)).argmax(axis=1)[0]
    walking = clf.logits(window(drop=0.0, jitter=2.0)).argmax(axis=1)[0]
    still = clf.logits(window(drop=0.0)).argmax(axis=1)[0]
    assert falling == FALLING, f"descent -> {CLASS_NAMES[falling]}"
    assert walking == 0, f"motion -> {CLASS_NAMES[walking]}"
    assert still == 2, f"stillness -> {CLASS_NAMES[still]}"
    assert clf.logits(np.zeros((0, T, 2, 17, 3), dtype=np.float32)).shape == (0, N_CLASSES)
    print(f"  E5 synthetic: descent->{CLASS_NAMES[falling]}, motion->{CLASS_NAMES[walking]}, "
          f"still->{CLASS_NAMES[still]}")


# ---------------------------------------------------------------------------
# P: frames -> windows -> segments
# ---------------------------------------------------------------------------


def test_p1_windows_are_built_per_track_with_correct_timing():
    stats = PipelineStats()
    frames = sequence(90)
    tws = frames_to_windows(frames, stats=stats)

    assert len(tws) == 1 and tws[0].track_id == 0
    tw = tws[0]
    expected = 1 + (90 - WINDOW_FRAMES) // 15
    assert len(tw) == expected, f"{len(tw)} windows, expected {expected}"
    for i in range(len(tw)):
        assert tw.ends[i] > tw.starts[i]
        span = (tw.ends[i] - tw.starts[i]).total_seconds()
        assert abs(span - (WINDOW_FRAMES - 1) / 15.0) < 1e-6, span
        assert tw.windows[i].shape == (WINDOW_FRAMES, 2, 17, 3)
    print(f"  P1 90 frames -> {len(tw)} windows of {WINDOW_FRAMES}, "
          f"{(tw.ends[0]-tw.starts[0]).total_seconds():.2f}s each; {stats.summary()}")


def test_p2_two_people_never_share_a_window():
    """A visitor's motion must not be attributable to the resident.

    Agent 3 aggregates per role, so pooling two tracks into one window would merge a
    visitor's activity into the resident's daily features with no error anywhere.
    """
    def builder(i, ts):
        return [
            person(0, i, ts, role=Role.RESIDENT, x=0.0),
            person(1, i, ts, role=Role.VISITOR, x=300.0),
        ]

    tws = frames_to_windows(sequence(60, builder=builder))
    assert len(tws) == 2, f"expected 2 per-track groups, got {len(tws)}"
    by_id = {t.track_id: t for t in tws}
    assert by_id[0].role is Role.RESIDENT and by_id[1].role is Role.VISITOR

    # Slot 1 must be empty: one window holds one person, not two.
    for tw in tws:
        for w in tw.windows:
            assert np.abs(w[:, 1]).max() == 0.0, "a second person leaked into the window"
    # And the two tracks must carry different geometry.
    assert np.abs(by_id[0].windows[0][:, 0] - by_id[1].windows[0][:, 0]).max() > 1.0
    print(f"  P2 2 tracks -> {len(tws)} separate window groups, roles "
          f"{[t.role.value for t in tws]}, no cross-contamination")


def test_p3_short_and_gappy_tracks_are_dropped_not_padded():
    """Padding would invent skeletons; zeroed trailing joints look exactly like a fall."""
    stats = PipelineStats()
    short = frames_to_windows(sequence(WINDOW_FRAMES - 1), stats=stats)
    assert short == [], "built a window from too few frames"
    assert stats.dropped_short_track == 1

    # A long gap must break the run rather than bridge it.
    def builder(i, ts):
        return [] if 30 <= i < 50 else [person(0, i, ts)]

    stats2 = PipelineStats()
    tws = frames_to_windows(sequence(90, builder=builder), stats=stats2)
    assert stats2.gaps_broken >= 1, "a 20-frame gap was silently bridged"
    for tw in tws:
        for w in tw.windows:
            # No window may contain an all-zero frame in slot 0.
            assert (np.abs(w[:, 0]).sum(axis=(1, 2)) > 0).all(), "window contains padding"
    print(f"  P3 {WINDOW_FRAMES-1} frames -> 0 windows; 20-frame gap -> "
          f"{stats2.gaps_broken} break(s), {stats2.windows_built} clean windows")


def test_p4_unusable_pose_frames_are_excluded():
    """Agent 1 sets keypoints=None when pose is unusable; those frames carry no skeleton."""
    def builder(i, ts):
        p = person(0, i, ts)
        if 10 <= i < 20:
            p = p.model_copy(update={"keypoints": None})
        return [p]

    stats = PipelineStats()
    frames_to_windows(sequence(80, builder=builder), stats=stats)
    assert stats.dropped_no_pose == 10, stats.dropped_no_pose
    print(f"  P4 10 keypoint-less frames excluded ({stats.summary()})")


def test_p5_pipeline_produces_segments_with_provenance():
    clf = SyntheticClassifier(seed=0)
    pipe = ActivityPipeline(clf)
    stats = PipelineStats()
    frames = sequence(150, objects=("cup", "couch"))
    segments = pipe.run(frames, stats=stats)

    assert segments, "no segments produced"
    for s in segments:
        assert s.track_id == 0 and s.role is Role.RESIDENT and s.room == "lounge"
        assert s.end_time > s.start_time
        assert 0 <= s.activity_id < N_CLASSES
        assert s.activity_name == CLASS_NAMES[s.activity_id]
        assert set(s.supporting_objects) <= {"cup", "couch"}
    assert segments == sorted(segments, key=lambda s: (s.start_time, s.track_id))
    total = sum(s.duration_s for s in segments)
    print(f"  P5 150 frames -> {len(segments)} segment(s), {total:.1f}s, "
          f"objects+room+role threaded through")


def test_p6_room_change_splits_segments():
    """A continuous activity crossing rooms must not swallow the room transition."""
    def builder(i, ts):
        return [person(0, i, ts, room="kitchen" if i < 75 else "lounge")]

    pipe = ActivityPipeline(SyntheticClassifier(seed=0))
    segments = pipe.run(sequence(150, builder=builder))
    rooms = [s.room for s in segments]
    assert "kitchen" in rooms and "lounge" in rooms, rooms
    assert len(segments) >= 2
    for s in segments:
        assert s.room in ("kitchen", "lounge")
    print(f"  P6 room change -> {len(segments)} segments across {sorted(set(rooms))}")


def test_p7_full_path_to_daily_features():
    """frames -> segments -> DailyFeatures, the exact input Agent 3 consumes."""
    pipe = ActivityPipeline(SyntheticClassifier(seed=0))
    stats = PipelineStats()

    def builder(i, ts):
        # Moving for the first half, still for the second: both classes should appear.
        return [person(0, i, ts, x=float(i) * 3.0 if i < 150 else 450.0)]

    frames = sequence(300, builder=builder)
    feats, segments = pipe.run_to_features(frames, subject_role=Role.RESIDENT, stats=stats)

    assert feats.day == T0.date()
    assert feats.subject_role is Role.RESIDENT
    assert segments
    total_activity = feats.walking_duration_s + feats.sitting_duration_s
    assert total_activity > 0, "no duration features derived from segments"

    # observed_hours must come from the camera schedule, not be inferred as a full day.
    hours = observed_hours_from_frames(frames)
    assert 0 < hours < 0.01, hours  # 300 frames @ 15fps = 20s
    feats2, _ = pipe.run_to_features(frames, observed_hours=hours)
    assert abs(feats2.observed_hours - hours) < 1e-9
    assert not feats2.is_reliable(), "a 20-second day must not be treated as reliable"
    print(f"  P7 300 frames -> {len(segments)} segments -> features "
          f"(walk {feats.walking_duration_s:.0f}s, sit {feats.sitting_duration_s:.0f}s), "
          f"observed_hours={hours:.4f} -> is_reliable=False")


def test_p8_observed_hours_ignores_gaps():
    """Two ten-minute bursts twelve hours apart is 20 minutes observed, not 12 hours."""
    frames = []
    for burst, offset in enumerate((0, 12 * 3600)):
        for i in range(150):
            ts = T0 + timedelta(seconds=offset + i / 15.0)
            frames.append(FrameObservation(
                frame_idx=burst * 1000 + i, timestamp=ts,
                persons=[person(0, burst * 1000 + i, ts)],
            ))
    hours = observed_hours_from_frames(frames)
    naive = (frames[-1].timestamp - frames[0].timestamp).total_seconds() / 3600.0
    assert hours < 0.02, f"gap was counted as observation ({hours:.3f} h)"
    assert naive > 11.9
    print(f"  P8 two 10s bursts 12h apart: observed {hours*3600:.0f}s "
          f"(naive first-to-last would say {naive:.1f} h)")


def test_p9_trained_ensemble_drives_the_pipeline():
    """The whole seam, with a real checkpoint rather than the synthetic stub."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, s in enumerate(("joint", "bone")):
            save_ckpt(td / f"adl_{s}" / "best.pt", s, seed=i + 3)
        clf = EnsembleClassifier.from_run_dir(td)
        pipe = ActivityPipeline(clf)
        segments = pipe.run(sequence(120))
        assert segments, "trained ensemble produced no segments"
        for s in segments:
            assert 0 <= s.activity_id < N_CLASSES
            assert 0.0 <= s.confidence <= 1.0
        print(f"  P9 {clf.name} -> {len(segments)} segment(s) "
              f"({', '.join(sorted({s.activity_name for s in segments}))})")


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
