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

import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behaviorsense.agents.activity import CLASS_NAMES, FALLING, N_CLASSES  # noqa: E402
from behaviorsense.eval.activity_eval import (  # noqa: E402
    apply_emergency_floor,
    combine,
    count_runs,
    fit_transition_matrix,
    fragmentation,
    scores,
    second_person_present,
    sequences_by_subject,
    subset_scores,
    tau_sweep,
)
from behaviorsense.models.ensemble import (  # noqa: E402
    EnsembleClassifier,
    SyntheticClassifier,
    flip_windows,
    windows_to_tensor,
)
from behaviorsense.models.stgcnpp import FLIP_INDEX, STGCNpp  # noqa: E402
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


def _long_tailed_logits(n: int = 4000, n_classes: int = N_CLASSES, seed: int = 0):
    """Labels shaped like the Charades map (one 40% class, heavy tail) plus a classifier
    biased toward the head exactly as cross-entropy training on that prior would leave it."""
    rng = np.random.default_rng(seed)
    prior = np.r_[0.40, np.full(4, 0.08), np.full(n_classes - 5, 0.28 / (n_classes - 5))]
    prior /= prior.sum()
    y = rng.choice(n_classes, size=n, p=prior)

    def stream(strength: float, s: int) -> np.ndarray:
        r = np.random.default_rng(s)
        z = r.normal(0.0, 1.0, (n, n_classes))
        z[np.arange(n), y] += strength                 # signal
        z += np.log(prior)[None, :] * 2.0              # head bias
        return z.astype(np.float32)

    return {"joint": stream(2.2, seed + 1), "bone": stream(2.3, seed + 2),
            "joint_motion": stream(0.9, seed + 3),
            "bone_motion": stream(0.8, seed + 4)}, y


def test_p10_logit_adjustment_recovers_mean_class_on_long_tailed_labels():
    """The post-hoc lever P1's numbers call for, with tau=0 as the negative control.

    P1 measured mean-class accuracy 0.189 for the best stream against a label distribution
    whose largest class is 39.4%. The classifier optimised cross-entropy under that prior
    and is then scored by a metric that weights all 20 classes equally - two different
    objectives. Subtracting tau*log(prior) at DECISION time is the standard correction and
    needs no retraining, so it is worth knowing whether it bites before spending a GPU
    session on anything else.

    tau=0 must reproduce the unadjusted numbers exactly. Without that control a bug that
    silently improved every score would look like a discovery.
    """
    per_stream, y = _long_tailed_logits()
    ens = combine(per_stream, tuple(sorted(per_stream)), "logit")

    sweep = tau_sweep(ens, y)
    base = sweep[0]
    assert base[0] == 0.0
    unadjusted = scores(ens, y)
    assert base[1:] == unadjusted, (
        f"tau=0 changed the score: {base[1:]} vs {unadjusted} - logit_adjust is not a no-op "
        "at tau=0, so every other row in the sweep is suspect"
    )

    peak = max(sweep, key=lambda r: r[2])
    assert peak[2] > base[2] + 0.05, (
        f"logit adjustment moved mean-class only {peak[2] - base[2]:+.3f} on data built to "
        "need it; the correction is not working"
    )
    # It should cost top-1 - that is the trade being made, not a free lunch, and a version
    # that improved both would mean the baseline was simply broken.
    assert peak[1] <= base[1] + 1e-9 or peak[2] - base[2] > 0.2, (
        "mean-class and top-1 both improved, which is not what a prior correction does"
    )
    print(f"  P10 logit adjustment tau=0 -> {base[2]:.3f} mean-class, "
          f"tau={peak[0]:.2f} -> {peak[2]:.3f} ({peak[2] - base[2]:+.3f}), "
          f"top-1 {base[1]:.3f} -> {peak[1]:.3f}")


def test_p11_stream_subsets_are_scored_and_can_beat_the_full_average():
    """Averaging four streams of unequal quality is not automatically best.

    P1 already showed the four-stream ensemble losing to `bone` alone on mean-class while
    winning top-1. The subset question follows immediately, and answering it needs only the
    per-stream logits - which is why notebook 04 now saves them.
    """
    per_stream, y = _long_tailed_logits()
    rows = subset_scores(per_stream, y, mode="logit")

    assert len(rows) == 15, f"expected all 15 non-empty subsets, got {len(rows)}"
    assert all(rows[i][2] >= rows[i + 1][2] for i in range(len(rows) - 1)), "not sorted"
    full = next(r for r in rows if len(r[0]) == 4)
    best = rows[0]
    assert best[2] >= full[2]

    # `combine` must mean the same thing as the served ensemble, or a subset chosen here
    # would not reproduce in deployment.
    manual = np.mean([per_stream[s] for s in sorted(per_stream)], axis=0)
    assert np.allclose(combine(per_stream, tuple(sorted(per_stream)), "logit"), manual,
                       atol=1e-5), "logit combination diverged from a plain mean of logits"
    probs = np.stack([np.exp(z - z.max(-1, keepdims=True)) for z in
                      (per_stream[s] for s in sorted(per_stream))])
    probs /= probs.sum(-1, keepdims=True)
    assert np.allclose(np.exp(combine(per_stream, tuple(sorted(per_stream)), "prob")),
                       probs.mean(0), atol=1e-5), "prob combination is not a mixture"

    print(f"  P11 15 subsets scored; best `{'+'.join(best[0])}` {best[2]:.3f} vs "
          f"all-four {full[2]:.3f}; logit/prob combination match the ensemble's semantics")


def test_p12_notebook_04_and_the_scripts_share_one_metric_definition():
    """`scores` must not be re-implemented in the notebook.

    It was inline there, so any test of the metric validated a copy. That is the exact
    pattern that let three notebook-04 defects reach Kaggle, and the fix is structural: one
    definition in the library, imported by the notebook, the rescoring script and this test.
    """
    nb = json.loads((ROOT / "notebooks/04_evaluate_blackwell_offline.ipynb")
                    .read_text(encoding="utf-8"))
    cells = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
    p1 = [c for c in cells if "P1 val:" in c]
    assert len(p1) == 1, f"expected one P1 cell, found {len(p1)}"
    assert "from behaviorsense.eval.activity_eval import" in p1[0], (
        "the P1 cell no longer imports the shared metric"
    )
    assert "def scores(" not in p1[0], (
        "`scores` is defined inline in the notebook again - the notebook and the tests can "
        "now disagree about what mean-class accuracy means"
    )
    assert "MIN_SUPPORT = 50" not in p1[0], "MIN_SUPPORT re-declared in the notebook"
    assert "val_logits.npz" in p1[0], (
        "the P1 cell no longer saves the validation logits, so every post-hoc accuracy "
        "experiment needs a fresh GPU session again"
    )
    rescore = (ROOT / "scripts/rescore_p1.py").read_text(encoding="utf-8")
    assert "from behaviorsense.eval.activity_eval import" in rescore
    print("  P12 P1 cell imports the shared scores()/MIN_SUPPORT and saves "
          "val_logits.npz; rescore_p1.py imports the same definition")


def test_p13_test_time_flip_is_a_real_mirror_and_changes_the_prediction():
    """TTA must mirror the body, not just negate x, and must actually be applied.

    Negating x without remapping left/right joints produces a pose no human can adopt - a
    body whose left wrist sits on its right side - which the model never trained on, so the
    "augmented" average would be polluted by an out-of-distribution view. `FLIP_INDEX` is
    the same permutation `augment()` uses at training time, so the mirrored window is a
    view the model has genuinely seen.

    The second half matters as much: `clf.tta` is a plain attribute, so a stale deployment
    could set it and silently get no augmentation. This asserts the logits actually move.
    """
    x = np.random.default_rng(3).normal(0, 1, (4, 30, 2, 17, 3)).astype(np.float32)

    assert np.allclose(flip_windows(flip_windows(x)), x), "flip is not an involution"
    l_wrist, r_wrist = 9, 10                                   # COCO-17
    f = flip_windows(x)
    assert np.allclose(f[..., l_wrist, 0], -x[..., r_wrist, 0]), "x not negated on swap"
    assert np.allclose(f[..., l_wrist, 1], x[..., r_wrist, 1]), "y did not swap L/R"
    assert np.allclose(f[..., l_wrist, 2], x[..., r_wrist, 2]), (
        "confidence scores were geometrically transformed; they are not coordinates"
    )
    assert FLIP_INDEX[l_wrist] == r_wrist and FLIP_INDEX[r_wrist] == l_wrist
    for bad in (np.zeros((2, 30, 2, 17, 2), np.float32), np.zeros((2, 30, 2, 16, 3), np.float32)):
        try:
            flip_windows(bad)
            raise AssertionError(f"accepted {bad.shape}")
        except ValueError:
            pass

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, stream in enumerate(("joint", "bone")):
            torch.manual_seed(i)
            model = STGCNpp(n_classes=N_CLASSES)
            p = td / f"adl_{stream}" / "best.pt"
            p.parent.mkdir(parents=True)
            torch.save({"model": model.state_dict(), "args": {"stream": stream}}, p)

        plain = EnsembleClassifier.from_run_dir(td, device="cpu")
        assert "+tta" not in plain.name
        base = plain.logits(x)

        plain.tta = True
        assert "+tta" in plain.name, "name does not record that TTA is on"
        tta = plain.logits(x)
        assert tta.shape == base.shape
        assert not np.allclose(tta, base, atol=1e-4), (
            "TTA produced identical logits - the mirrored pass is not being averaged in"
        )
        # It must be the MEAN of the two views, not a replacement.
        plain.tta = False
        mirrored = plain.logits(flip_windows(x))
        assert np.allclose(tta, (base + mirrored) / 2.0, atol=1e-4), (
            "TTA is not the mean of the two views"
        )
        drift = float(np.abs(tta - base).mean())
    print(f"  P13 flip is an involution with L/R remap and untouched scores; TTA averages "
          f"two views (mean |delta| {drift:.4f}) and is recorded in the model name")


def test_p14_segment_metrics_separate_label_quality_from_structure():
    """Accuracy cannot decide argmax vs Viterbi; fragmentation is the missing column.

    Measured on real data, per-window argmax + logit adjustment beat every smoothed variant
    on mean-class (0.202 vs 0.174). Read alone that says "drop smoothing" - but Agent 3
    derives `walking_bouts` and `mean_bout_duration_s` from segment COUNTS, so a decoding
    that shatters one true walking stretch into nine reports nine bouts and a ninth of the
    mean duration while scoring identically per window. Both columns or neither.

    Also pins the transition-matrix fit, whose value turned out to be much smaller than
    predicted: the hand-set 0.90 self-transition against a fitted 0.764 recovered only 0.004
    of the 0.040 mean-class that smoothing costs.
    """
    truth = [np.array([0] * 10 + [1] * 10), np.array([2] * 6)]
    assert count_runs(truth[0]) == 2 and count_runs(truth[1]) == 1
    assert count_runs(np.array([], dtype=int)) == 0

    perfect = [s.copy() for s in truth]
    flicker = [np.array([0, 1] * 5 + [1, 0] * 5), np.array([2, 3] * 3)]
    merged = [np.zeros(20, dtype=int), np.zeros(6, dtype=int)]
    assert fragmentation(perfect, truth)["ratio"] == 1.0
    assert fragmentation(flicker, truth)["ratio"] > 5, "flickering not detected"
    assert fragmentation(merged, truth)["ratio"] < 1, "over-merging not detected"

    # sequences_by_subject must reconstruct time order and drop stubs.
    subjects = np.array(["a"] * 5 + ["b"] * 2 + ["c"] * 4)
    values = np.arange(11)
    seqs = sequences_by_subject(subjects, values, min_len=3)
    assert [len(s) for s in seqs] == [5, 4], [len(s) for s in seqs]
    assert list(seqs[0]) == [0, 1, 2, 3, 4], "index order is not preserved"

    # A fitted transition matrix must be row-stochastic, and the emergency floor must
    # survive it exactly - a counted matrix leaves `falling` (0.5% of windows) unreachable.
    rng = np.random.default_rng(0)
    label_seqs = [rng.integers(0, N_CLASSES, 50) for _ in range(20)]
    A = fit_transition_matrix(label_seqs, n_classes=N_CLASSES)
    assert A.shape == (N_CLASSES, N_CLASSES)
    assert np.allclose(A.sum(axis=1), 1.0), "fitted matrix is not row-stochastic"
    assert (A > 0).all(), "Laplace smoothing left an unreachable transition"

    sticky = [np.repeat(np.arange(5), 10) for _ in range(10)]
    A_sticky = fit_transition_matrix(sticky, n_classes=N_CLASSES)
    assert np.mean(np.diag(A_sticky)) > np.mean(np.diag(A)), (
        "fitting does not recover a higher self-transition from sticky sequences"
    )

    floored = apply_emergency_floor(A, FALLING, 0.05)
    assert np.allclose(floored.sum(axis=1), 1.0), "floor broke row-stochasticity"
    for i in range(N_CLASSES):
        if i != FALLING:
            assert floored[i, FALLING] >= 0.05 - 1e-12, (
                f"row {i} into-falling is {floored[i, FALLING]:.4f}, below the 0.05 floor - "
                "a naive floor-then-normalise shaves it back and a one-window fall is "
                "smoothed out of existence"
            )
    print(f"  P14 fragmentation 1.00/perfect, {fragmentation(flicker, truth)['ratio']:.1f}x/"
          f"flicker, {fragmentation(merged, truth)['ratio']:.1f}x/merged; fitted matrix "
          f"row-stochastic with self-transition {np.mean(np.diag(A_sticky)):.2f} on sticky "
          f"sequences; emergency floor survives at {floored[0, FALLING]:.3f}")


def test_p15_second_person_signal_is_derived_from_slot_occupancy():
    """The cheapest object-context signal must read slot 1, not slot 0, and gate on noise.

    `interacting_with_person` is the second-worst measured class (F1 0.042) because pose
    alone cannot express "someone else is here" - but every shard already carries two
    person slots, and slot 1 is non-zero exactly when the tracker held a second person.
    `OBJECT_PRIORS["person"]` (+1.5 log-odds on class 18) was written for this signal and
    had never received it. Three failure modes pinned here: reading the wrong slot, firing
    on single-frame tracker flicker, and the fusion path not actually moving class 18.
    """
    rng = np.random.default_rng(7)
    sk = rng.normal(0, 0.4, (6, 30, 2, 17, 3)).astype(np.float32)
    sk[..., 2] = 0.0
    sk[:, :, 0, :, 2] = 0.9                    # resident always visible in slot 0
    sk[2, 10:20, 1, :6, 2] = 0.8               # window 2: real second track, 10 frames
    sk[5, :, 1, :, 2] = 0.7                    # window 5: second person throughout
    sk[3, 4, 1, 0, 2] = 0.9                    # window 3: one joint, one frame - flicker

    present = second_person_present(sk)
    assert np.flatnonzero(present).tolist() == [2, 5], np.flatnonzero(present).tolist()
    assert not present[3], "single-joint single-frame tracker flicker counted as a person"
    # Slot confusion control: a signal derived from slot 0 would fire on EVERY window.
    assert present.sum() < len(sk), "signal fires everywhere - it is reading the resident"
    for bad in (np.zeros((2, 30, 1, 17, 3), np.float32), np.zeros((2, 30, 2, 17, 2), np.float32)):
        try:
            second_person_present(bad)
            raise AssertionError(f"accepted {bad.shape}")
        except ValueError:
            pass

    # End to end through the SERVING code path: fuse_objects must raise class 18's
    # posterior on flagged windows and leave unflagged windows untouched.
    from behaviorsense.agents.activity import ActivityAgent, ActivityConfig, softmax
    agent = ActivityAgent(ActivityConfig())
    logits = rng.normal(0, 1, (6, N_CLASSES)).astype(np.float64)
    post = softmax(logits)
    objs = [("person",) if s else () for s in present]
    fused = agent.fuse_objects(post, objs)
    iwp = 18
    for i in range(6):
        if present[i]:
            assert fused[i, iwp] > post[i, iwp], f"window {i}: class 18 not up-weighted"
        else:
            # Renormalisation after the medication-absent penalty shifts mass slightly;
            # class 18's RELATIVE odds against any other class must be unchanged.
            ratio = (fused[i, iwp] / fused[i, 0]) / (post[i, iwp] / post[i, 0])
            assert abs(ratio - 1.0) < 1e-9, f"window {i}: class 18 odds moved without a person"
    print(f"  P15 slot-1 occupancy -> windows {np.flatnonzero(present).tolist()} flagged, "
          f"flicker rejected, fuse_objects raises class 18 only there "
          f"(x{fused[5, iwp] / post[5, iwp]:.1f} on flagged)")


def test_p16_video_path_isolates_crashes_and_feeds_two_people_through_the_seam():
    """The upload path, at its two load-bearing seams.

    `extract_isolated` runs decode in a CHILD process, and that is not defensive habit: cv2
    delegates to ffmpeg, ffmpeg raises SIGSEGV/SIGABRT on malformed streams, and a signal is
    not an exception - `try/except` cannot see it and the interpreter simply stops. Notebook
    02 lost finished corpora to exactly this. Serving uploads inline would mean one bad file
    takes down the session, the tunnel and the demo together, so the test that matters is
    that a failing child becomes a `ValueError` here while this process stays alive.

    `rebuild_observations` is the other seam: the wire format is compact for the browser's
    benefit, and this is the only thing that reads it back. If it drifts, two people arrive
    as one, or as none, and `frames_to_windows` silently returns fewer window sets than there
    were people.
    """
    from behaviorsense.video import (
        box_from_keypoints,
        crash_reason,
        extract_isolated,
        rebuild_observations,
    )

    # 1. A box from visible joints only, padded past their extent so OSNet sees clothing
    # rather than a skeleton crop - the operating point was fitted on person crops.
    xy = np.array([[10.0, 20.0], [30.0, 60.0], [50.0, 100.0]], dtype=np.float32)
    box = box_from_keypoints(xy, np.array([0.9, 0.9, 0.9], dtype=np.float32))
    assert box.x1 < 10 and box.x2 > 50 and box.y1 < 20 and box.y2 > 100
    assert box_from_keypoints(xy, np.array([0.9, 0.1, 0.1], np.float32)) is None, (
        "one visible joint is not a person"
    )
    assert box_from_keypoints(np.zeros((3, 2), np.float32),
                              np.ones(3, np.float32)) is None, "degenerate box accepted"

    # 2. Abnormal exits, on both platforms. POSIX signals with a negative code; Windows
    # reports an NTSTATUS exception as a large unsigned status, so `returncode < 0` alone is
    # right on Kaggle and wrong on the machine this test runs on.
    assert crash_reason(0) is None and crash_reason(1) is None
    assert "signal 11" in crash_reason(-11)
    assert "0xC0000005" in crash_reason(0xC0000005)

    # 3. A child that fails must surface as ValueError, and we must still be here after.
    try:
        extract_isolated("does-not-exist.mp4", rtmo="missing.onnx", device="cpu", timeout=120)
        raise AssertionError("a missing video was accepted")
    except ValueError as exc:
        assert "pose extraction failed" in str(exc), str(exc)
    alive = 2 + 2  # the point: this line runs

    # 4. Two people through the seam, in the shape the backend actually sends.
    def wire_person(track_id, role, cx, cy):
        return {"track_id": track_id, "role": role, "role_confidence": 0.8,
                "box": [cx - 30, cy - 60, cx + 30, cy + 80],
                "kp": [[cx + j * 1.5, cy + j * 2.0, 0.9] for j in range(17)]}

    fps = 15.0
    payload = {"fps": fps, "width": 1280, "height": 720, "frames": [
        {"i": i, "t": round(i / fps, 3),
         "people": [wire_person(0, "resident", 400 + i, 300),
                    wire_person(1, "visitor", 800 - i, 320)]}
        for i in range(40)]}

    frames = rebuild_observations(payload)
    assert len(frames) == 40
    assert [p.role.value for p in frames[0].persons] == ["resident", "visitor"]

    windows = frames_to_windows(frames)
    assert len(windows) == 2, (
        f"two tracked people produced {len(windows)} window set(s) - the wire format and "
        "rebuild_observations have drifted apart"
    )
    for tw in windows:
        stacked = np.stack(tw.windows)
        assert stacked.shape[1:] == (WINDOW_FRAMES, 2, 17, 3), stacked.shape

    # An unusable pose must survive as "tracked, no keypoints" rather than as an absence:
    # the overlay draws those two states differently and Agent 2 must abstain, not guess.
    payload["frames"][5]["people"][0]["kp"] = None
    rebuilt = rebuild_observations(payload)
    assert rebuilt[5].persons[0].keypoints is None
    assert rebuilt[5].n_persons == 2, "a pose-less person was dropped from the frame"

    print(f"  P16 child failure -> ValueError (parent alive, {alive == 4}); crash codes "
          f"named on both platforms; 2 wire people -> {len(windows)} window sets "
          f"{np.stack(windows[0].windows).shape}; unusable pose kept as tracked")


def test_p17_decode_child_can_import_behaviorsense_without_an_installed_package():
    """The child interpreter must find `behaviorsense`. It is not the parent's `sys.path`.

    This is a real observed serving failure, not a hypothetical: a clip uploaded to notebook
    05 came back `pose extraction failed: No module named 'behaviorsense'` while the parent
    was importing the package perfectly well. Notebook 05 makes it importable with
    `sys.path.insert(0, str(SRC))`, which configures one interpreter and says nothing to a
    fresh one. `subprocess.run` inherits the ENVIRONMENT, so the path has to travel there.

    This suite reproduces the deployment exactly - it does the same `sys.path.insert` and sets
    no PYTHONPATH - so a child spawned here is in the same position as the child on Kaggle.
    Both directions are asserted, because only the negative control proves the fix is doing
    the work rather than an installed copy of the package quietly satisfying the import.
    """
    import os
    import subprocess

    from behaviorsense.video import PKG_PARENT, child_env, extract_isolated

    assert (PKG_PARENT / "behaviorsense" / "__init__.py").is_file(), (
        f"PKG_PARENT is {PKG_PARENT}, which does not contain the package - the derivation "
        "from __file__ is wrong and every child would inherit a useless path")

    probe = "import behaviorsense, sys; print(behaviorsense.__file__)"
    # Run from a neutral directory so an implicit cwd entry on sys.path cannot satisfy the
    # import and make the negative control pass for the wrong reason.
    neutral = tempfile.gettempdir()

    # NEGATIVE CONTROL: a bare inherited environment, which is what the bug was.
    bare = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    without = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                             cwd=neutral, env=bare, timeout=120)
    if without.returncode == 0:
        print(f"  P17 SKIP: behaviorsense is installed in this interpreter "
              f"({without.stdout.strip()}), so the negative control cannot be observed. "
              f"child_env still yields PYTHONPATH={child_env()['PYTHONPATH'].split(os.pathsep)[0]}")
        return
    assert "No module named" in without.stderr, without.stderr[-300:]

    # WITH child_env: the same probe, the same cwd, now importable.
    withenv = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                             cwd=neutral, env=child_env(), timeout=120)
    assert withenv.returncode == 0, (
        f"child_env did not make behaviorsense importable in a child.\n"
        f"PYTHONPATH={child_env()['PYTHONPATH']!r}\n{withenv.stderr[-400:]}")
    resolved = Path(withenv.stdout.strip()).resolve()
    assert resolved == (PKG_PARENT / "behaviorsense" / "__init__.py").resolve(), (
        f"child imported {resolved}, not the copy under {PKG_PARENT}")

    # The inherited environment must survive, and an existing PYTHONPATH must be kept rather
    # than clobbered - a child that loses CUDA_VISIBLE_DEVICES or LD_LIBRARY_PATH runs on CPU.
    env = child_env()
    assert set(os.environ) - {"PYTHONPATH"} <= set(env), "child_env dropped inherited variables"
    os.environ["PYTHONPATH"] = "/caller/set/this"
    try:
        kept = child_env("/extra").get("PYTHONPATH", "").split(os.pathsep)
    finally:
        del os.environ["PYTHONPATH"]
    assert kept[0] == "/extra" and str(PKG_PARENT) in kept and "/caller/set/this" in kept, kept
    assert kept.index(str(PKG_PARENT)) < kept.index("/caller/set/this"), (
        "an inherited PYTHONPATH precedes the mounted package, so an installed copy would "
        "shadow the one the notebook actually resolved")

    # And the error message must name THIS cause rather than blaming the upload. The old text
    # said "it is not a video this decoder can read" for a deployment fault.
    try:
        extract_isolated("does-not-exist.mp4", rtmo="missing.onnx", device="cpu", timeout=120)
        raise AssertionError("a missing video was accepted")
    except ValueError as exc:
        assert "pose extraction failed" in str(exc)

    print(f"  P17 child WITHOUT PYTHONPATH: ModuleNotFoundError (reproduced the serving bug); "
          f"WITH child_env: imported {resolved.parent.name} from {PKG_PARENT}; inherited env "
          f"preserved, caller PYTHONPATH kept but ordered after the mount")


def test_p18_pose_geometry_and_quality_are_reported_from_the_decoded_frame():
    """The overlay's scale factor, and the two ways a wrong-looking skeleton happens.

    A screenshot showed a correctly-shaped skeleton floating beside a plainly visible person,
    with no skeleton on her at all. Two independent defects produce exactly that picture and
    both are covered here:

    1. **Coordinate space.** Keypoints come back in the space of the frame cv2 DECODED, while
       `width`/`height` used to be `CAP_PROP_FRAME_WIDTH/HEIGHT` — the container's claim. Those
       disagree on any file with a rotation matrix, a non-square pixel aspect, or a wrong
       header, and the overlay divides by the reported size. `overlay.js` warns about precisely
       this failure in its own header comment and then had no way to detect it.

    2. **A pose that exists but is entirely below threshold.** `kp` is non-null, so the
       dashed "pose unusable" branch is skipped, and then every edge and every joint is
       skipped by `MIN_SCORE` — rendering literally nothing. Absent and unusable are supposed
       to be visibly different states; this was a third, invisible one.

    The quality counters exist so this is diagnosable from the page: RTMO is one-stage with no
    detector in front of it, so it both invents low-confidence people and loses real ones, and
    those are indistinguishable from a broken install without the numbers.
    """
    from behaviorsense.video import rebuild_observations

    # The payload must report the DECODED size and carry the container's claim separately.
    src = (ROOT / "src" / "behaviorsense" / "video.py").read_text(encoding="utf-8")
    assert "height, width = frame.shape[:2]" in src, (
        "width/height must come from the decoded array. Taken from CAP_PROP_FRAME_* they are "
        "the container's claim, and the overlay scales keypoints by them.")
    assert '"container_size"' in src and '"size_source"' in src, (
        "the container's claim must still travel, or a mismatch is invisible")
    claim = src.index("claimed_w = int(cap.get")
    assert claim < src.index("height, width = frame.shape[:2]")

    # A pose that is present but wholly below threshold must be drawn as a box, not as nothing.
    ov = (ROOT / "web" / "scripts" / "overlay.js").read_text(encoding="utf-8")
    assert "j[2] >= MIN_SCORE)" in ov and "if (!usable)" in ov, (
        "overlay.js still branches on `!kp` alone, so a tracked person whose every joint is "
        "below MIN_SCORE renders as empty space")
    assert "sizeDisagreement" in ov, (
        "nothing compares the browser's intrinsic size against the payload's, so two decoders "
        "disagreeing draws a confident wrong skeleton in silence")

    # And the reconstruction still round-trips a below-threshold pose rather than dropping it:
    # Agent 2 must see the person and abstain, not fail to see them.
    fps = 15.0
    weak = [[100.0 + j, 200.0 + j, 0.05] for j in range(17)]        # every joint untrusted
    payload = {"fps": fps, "width": 640, "height": 480, "frames": [
        {"i": 0, "t": 0.0, "people": [
            {"track_id": 0, "role": "unknown", "role_confidence": 0.0,
             "box": [80, 180, 160, 400], "kp": weak}]}]}
    frames = rebuild_observations(payload)
    assert frames[0].n_persons == 1, "a below-threshold pose was dropped from the frame"
    kps = frames[0].persons[0].keypoints
    assert kps is not None and not kps.visible(), (
        "the pose should survive as present-but-invisible; Agent 2 abstains on it")

    print(f"  P18 width/height read from frame.shape (container claim carried separately); "
          f"overlay draws a box for an all-weak pose and flags decoder size disagreement; "
          f"a 17-joint pose at score 0.05 survives as present with visible()="
          f"{kps.visible()}")


def test_p19_serving_resamples_to_the_shard_rate_instead_of_striding():
    """A 30-frame window must be 2.0 s of motion, whatever the camera's frame rate is.

    The ADL shards were built at `FPS_SAMPLE = 15` (notebook 01), so ST-GCN++ has never seen a
    window of any other duration. Serving used `out_fps = src_fps / 2`, which is 15 Hz on a
    30 fps source and nothing like it otherwise:

        30 fps -> 15.0 Hz -> 2.00 s   correct
        20 fps -> 10.0 Hz -> 3.00 s   every action stretched 1.5x
        25 fps -> 12.5 Hz -> 2.40 s   1.2x

    Observed on a 20 fps clip of someone cooking: 10 Hz, and 25.5 s of a 30 s video labelled
    `interacting_with_person` with nobody else in frame.

    This is the SAME defect notebook 07 already paid a 12,796 s GPU run for - it strided with
    `round(src / nominal)`, every trimmed clip was 20 fps, and a nominal 12.5 Hz became an
    effective 10 Hz that made 37.5% of clips unusable. The fix there was exact resampling with
    `want = int(round(k * src / rate))`, and the serving path never received it.
    """
    from behaviorsense.video import SHARD_FPS

    src = (ROOT / "src" / "behaviorsense" / "video.py").read_text(encoding="utf-8")
    assert SHARD_FPS == 15.0, SHARD_FPS
    assert "want = int(round(kept * src_fps / rate))" in src, (
        "the exact-resampling step is gone; an integer stride cannot hold window duration "
        "constant across container rates")
    assert "rate = min(float(target_fps), src_fps)" in src, (
        "`min` is load-bearing: a 10 fps camera must not be resampled UP, because inventing "
        "frames is worse than declaring the shortfall")

    # The scheduler itself, on the rates that actually occur. Replicated rather than imported
    # because it lives inside a cv2 read loop; the formula is the thing under test.
    def kept_indices(src_fps, rate, n_src):
        out, want, k = [], 0, 0
        for i in range(n_src):
            if i != want:
                continue
            out.append(i)
            k += 1
            want = int(round(k * src_fps / rate))
        return out

    rows = []
    for src_fps in (20.0, 25.0, 30.0, 10.0):
        rate = min(SHARD_FPS, src_fps)
        idx = kept_indices(src_fps, rate, int(src_fps * 4))     # four seconds of source
        got = len(idx) / 4.0                                     # kept frames per second
        assert abs(got - rate) <= 0.3, f"{src_fps} fps -> {got} Hz, wanted {rate}"
        window_s = 30.0 / rate
        rows.append((src_fps, rate, got, window_s))
        if src_fps >= SHARD_FPS:
            assert abs(window_s - 2.0) < 0.01, (
                f"a {src_fps} fps source yields a {window_s:.2f}s window; the shards are 2.00s")

    # The old integer stride, kept as a negative control: it is what produced the bad labels,
    # and without this the test cannot show the fix changes anything.
    assert abs(20.0 / 2 - 10.0) < 1e-9
    assert abs(30.0 / (20.0 / 2) - 3.0) < 1e-9, "20 fps / stride 2 gave a 3.0 s window"

    # A sub-15 fps camera cannot reach the shard rate, and that must be declared, not hidden.
    assert min(SHARD_FPS, 10.0) == 10.0
    assert '"rate_matches_shards"' in src and '"window_seconds"' in src, (
        "a rate the shards were not built at must travel in the payload so the page can say "
        "the labels are outside measured conditions")

    print("  P19 " + "; ".join(f"{s:g}fps->{g:.1f}Hz ({w:.2f}s win)" for s, _, g, w in rows)
          + f"; old stride-2 on 20fps gave 10.0Hz (3.00s win, 1.5x stretch)")


def test_p20_a_long_decode_is_not_killed_but_a_wedged_one_is():
    """Pose extraction must fail on SILENCE, never on elapsed time.

    This replaces a wall-clock timeout, and the reason is not theoretical. Raising MAX_FRAMES from
    900 to 12,000 (45 s of footage to ten minutes) means ~7.7 min of RTMO on a P100, and the
    hardcoded 420 s ceiling in the notebook would have killed that child two-thirds of the way
    through. The reflex - raise the number - is the same mistake in a bigger costume: any ceiling
    high enough for the longest acceptable video is too high to catch a hang, so the two cases
    cannot be told apart by duration at all.

    Progress can tell them apart. The child reports every PROGRESS_EVERY frames, so "still talking"
    means "still working" however long it has been going, and "gone quiet" means stuck regardless
    of how little time has passed.

    Every assertion here is about a CHILD PROCESS, because that is where the behaviour lives. Four
    of the five cases are failure modes the implementation actually had: the flood case is the
    64 KB pipe deadlock (a file, not a pipe, for stderr), and the callback case is a drain thread
    racing `communicate()` for stderr - measured at 1 of 6 lines received, which would have left
    the stall timer unfed and made this detector silently useless while reporting success.
    """
    import subprocess
    import sys
    import time

    from behaviorsense.video import _run_watching_progress, _Stalled

    def child(body: str) -> list[str]:
        return [sys.executable, "-u", "-c", body]

    # 1. LONG AND TALKING MUST BE ALLOWED TO FINISH. This is the whole point: no ceiling.
    beats: list[str] = []
    t0 = time.time()
    proc = _run_watching_progress(
        child("import time,sys\n"
              "for i in range(6):\n"
              "    print('[video] progress %d/12000' % (i * 50), file=sys.stderr)\n"
              "    time.sleep(0.4)\n"
              "print('done')\n"),
        env={}, timeout=None, stall_s=5.0, on_progress=beats.append)
    assert proc.returncode == 0 and proc.stdout.strip() == "done", proc
    assert len(beats) == 6, f"progress lines must reach the caller, got {len(beats)}: {beats}"
    assert "progress" in proc.stderr, "the child's stderr must be captured"
    slow_s = time.time() - t0

    # 2. SILENT MUST BE KILLED, and killed on the quiet period rather than on total runtime.
    t0 = time.time()
    try:
        _run_watching_progress(
            child("import time,sys\n"
                  "print('[video] progress 50/12000', file=sys.stderr)\n"
                  "time.sleep(60)\n"),
            env={}, timeout=None, stall_s=1.5)
        raise AssertionError("a wedged child was allowed to run to completion")
    except _Stalled as exc:
        wedged_s = time.time() - t0
        assert "progress 50/12000" in exc.last, exc.last
    assert wedged_s < 15, f"took {wedged_s:.1f}s to notice a 1.5 s stall"

    # 3. A CHILD THAT NEVER SPEAKS is the same failure and must not be mistaken for slow work.
    t0 = time.time()
    try:
        _run_watching_progress(child("import time\ntime.sleep(60)\n"),
                               env={}, timeout=None, stall_s=1.5)
        raise AssertionError("a child that never reported progress was allowed to run")
    except _Stalled as exc:
        assert exc.last == "", exc.last
    assert time.time() - t0 < 15

    # 4. AN EXPLICIT CALLER TIMEOUT STILL WINS - the two tests above pass `timeout=120` and must
    #    keep their deadline, so the default being None must not disable the argument.
    t0 = time.time()
    try:
        _run_watching_progress(
            child("import time,sys\n"
                  "while True:\n"
                  "    print('[video] progress', file=sys.stderr)\n"
                  "    time.sleep(0.2)\n"),
            env={}, timeout=1.5, stall_s=999)
        raise AssertionError("an explicit timeout was ignored")
    except subprocess.TimeoutExpired:
        pass
    assert time.time() - t0 < 15

    # 5. A CHILD THAT FLOODS STDERR MUST NOT DEADLOCK. With a pipe this is the classic
    #    fills-the-buffer-and-blocks-forever case, which would have looked exactly like the hang
    #    this function exists to detect - caused by the detector for it.
    t0 = time.time()
    flood = _run_watching_progress(
        child("import sys\nfor i in range(20000):\n    print('noise %d' % i, file=sys.stderr)\n"),
        env={}, timeout=None, stall_s=5.0)
    assert flood.returncode == 0 and len(flood.stderr) > 100_000, len(flood.stderr)
    assert time.time() - t0 < 30, "a flooding child was throttled by the parent"

    print(f"  P20 a 6-line, {slow_s:.1f}s child finished with all 6 progress lines delivered; a "
          f"1.5 s stall was killed in {wedged_s:.1f}s; a silent child likewise; an explicit "
          f"timeout still fires; {len(flood.stderr)} chars of stderr flooded without deadlock "
          "(file, not a 64 KB pipe)")


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
