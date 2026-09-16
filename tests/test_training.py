"""Tests for the training stack: ST-GCN++, dataset, augmentation, sampling, resume.

The load-bearing test here is S7 (resume exactness). Kaggle's 12-hour cap means every
real training run WILL be resumed at least once, so "resume works" is not a convenience
claim - if it silently diverges, every reported number came from a run nobody can
reproduce. It is tested by comparing a resumed run against an uninterrupted one
step-for-step, which is the only check that can actually fail if RNG state is mishandled.

Run: python tests/test_training.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from behaviorsense.data.skeleton_dataset import (  # noqa: E402
    N_CLASSES,
    AugmentConfig,
    SkeletonWindowDataset,
    augment,
    class_balanced_sampler,
    normalise,
    split_by_subject,
    temporal_resample,
)
from behaviorsense.models.stgcnpp import (  # noqa: E402
    COCO_EDGES,
    FLIP_INDEX,
    STREAMS,
    STGCNpp,
    build_adjacency,
    make_stream,
    to_bone,
    to_motion,
)


def synth_window(T: int = 30, M: int = 2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.3, size=(T, M, 17, 3)).astype(np.float32)
    x[..., 2] = 0.9
    return x


def test_s1_adjacency_is_a_valid_normalised_graph():
    A = build_adjacency()
    assert A.shape == (3, 17, 17)
    combined = A.sum(axis=0)
    assert (combined >= 0).all()
    # Every joint must be reachable, or part of the body is invisible to the model.
    assert (combined.sum(axis=1) > 0).all(), "isolated joint in the graph"
    # The 3 partitions must be disjoint: a joint pair belongs to exactly one relation.
    nz = (A > 0).sum(axis=0)
    assert nz.max() <= 1, "partitions overlap; a pair is in two relations"
    f = np.array(FLIP_INDEX)
    assert (f[f] == np.arange(17)).all(), "FLIP_INDEX is not an involution"
    print(f"  S1 3x17x17 adjacency, {int((combined>0).sum())} edges, partitions disjoint, "
          f"flip index valid")


def test_s2_model_shapes_and_all_streams():
    m = STGCNpp(n_classes=N_CLASSES)
    m.eval()
    x = torch.from_numpy(synth_window()).permute(3, 0, 2, 1)[None]  # [1,C,T,V,M]
    with torch.no_grad():
        for s in STREAMS:
            y = m(make_stream(x, s))
            assert y.shape == (1, N_CLASSES), f"{s}: {y.shape}"
    n_par = sum(p.numel() for p in m.parameters())
    assert n_par < 3e6, f"model unexpectedly large ({n_par/1e6:.1f}M)"
    print(f"  S2 {n_par/1e6:.2f}M params, all 4 streams -> [1,{N_CLASSES}]")


def test_s2b_streams_are_actually_different():
    """A stream bug that returns the joint tensor for every stream would silently reduce
    the 4-stream ensemble to one model averaged with itself - no error, no lift."""
    x = torch.from_numpy(synth_window(seed=3)).permute(3, 0, 2, 1)[None]
    outs = {s: make_stream(x, s) for s in STREAMS}
    for a in STREAMS:
        for b in STREAMS:
            if a < b:
                d = (outs[a] - outs[b]).abs().mean().item()
                assert d > 1e-4, f"streams {a} and {b} are identical (mean diff {d:.2e})"
    # bone must be zero where a joint equals its parent; motion zero on a static sequence
    static = torch.zeros(1, 3, 30, 17, 2)
    static[:, :, :, :, :] = 1.0
    assert to_motion(static).abs().max() < 1e-6, "motion of a static sequence is nonzero"
    assert to_bone(static).abs().max() < 1e-6, "bone of a uniform pose is nonzero"
    print("  S2b all 4 streams pairwise distinct; bone/motion vanish on degenerate input")


def test_s2c_bone_stream_is_flip_equivariant():
    """The flip augmentation asserts left/right symmetry; the bone stream must honour it.

    Mirroring the input (negate x, remap joints by FLIP_INDEX) must produce the mirrored
    bone tensor. This held for every joint except the hips: the torso-closing edge
    (11, 12) used to OVERWRITE parent[12]=6 with parent[12]=11, giving the right hip a
    hip-to-hip bone while the left kept a shoulder-to-hip bone. Under the 50% flip that
    bone flipped sign instead of mirroring, so half of all training windows encoded the
    torso two contradictory ways - in both bone streams, i.e. half the ensemble.
    """
    torch.manual_seed(0)
    x = torch.randn(4, 3, 12, 17, 2)
    flip = list(FLIP_INDEX)

    def mirrored(t: torch.Tensor) -> torch.Tensor:
        out = t[:, :, :, flip].clone()
        out[:, 0] = -out[:, 0]
        return out

    for stream in ("bone", "bone_motion"):
        err = (make_stream(mirrored(x), stream) - mirrored(make_stream(x, stream))
               ).abs().max().item()
        assert err < 1e-5, f"{stream} not flip-equivariant: max error {err:.4f}"

    # Negative control: the pre-fix parent map (last edge wins) must FAIL this check,
    # or the test proves nothing about the defect it exists to prevent.
    bad_parent: dict[int, int] = {}
    for i, j in COCO_EDGES:
        bad_parent[max(i, j)] = min(i, j)          # overwrite, as the old code did
    def bad_bone(t: torch.Tensor) -> torch.Tensor:
        b = torch.zeros_like(t)
        for child, par in bad_parent.items():
            b[:, :, :, child] = t[:, :, :, child] - t[:, :, :, par]
        return b
    bad_err = (bad_bone(mirrored(x)) - mirrored(bad_bone(x))).abs().max().item()
    assert bad_err > 0.1, "negative control: the old parent map should violate equivariance"
    print(f"  S2c bone/bone_motion flip-equivariant (err<1e-5); "
          f"pre-fix parent map violates it by {bad_err:.2f}")


def test_s3_gradients_reach_every_parameter():
    m = STGCNpp(n_classes=N_CLASSES)
    x = torch.from_numpy(synth_window(seed=1)).permute(3, 0, 2, 1)[None]
    m(x).sum().backward()
    dead = [n for n, p in m.named_parameters()
            if p.grad is None or p.grad.abs().sum().item() == 0]
    # The learnable adjacency refinement starts at zero but must still receive gradient.
    assert not dead, f"{len(dead)} parameters receive no gradient: {dead[:4]}"
    print(f"  S3 all {sum(1 for _ in m.parameters())} parameter tensors receive gradient")


def test_s4_normalise_is_translation_and_scale_invariant():
    """Two recordings of the same posture at different positions/distances must normalise
    to the same tensor - that is the whole point of the step."""
    x = synth_window(seed=2)
    shifted = x.copy()
    shifted[..., :2] += 5.0
    scaled = x.copy()
    scaled[..., :2] *= 2.5

    a, b, c = normalise(x.copy()), normalise(shifted), normalise(scaled)
    assert np.abs(a - b).max() < 1e-4, f"not translation invariant ({np.abs(a-b).max():.2e})"
    assert np.abs(a - c).max() < 1e-3, f"not scale invariant ({np.abs(a-c).max():.2e})"
    # Rotation must NOT be normalised away: a fallen person's orientation is the signal.
    rotated = x.copy()
    th = np.pi / 2
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)
    rotated[..., :2] = rotated[..., :2] @ R.T
    assert np.abs(a - normalise(rotated)).max() > 0.1, (
        "rotation was normalised away - falls and lying down become indistinguishable"
    )
    assert np.abs(a[..., 2] - x[..., 2]).max() < 1e-6, "confidence scores were transformed"
    print("  S4 translation+scale invariant, rotation preserved, scores untouched")


def test_s4b_normalise_preserves_whole_body_descent():
    """A falling body's vertical motion must SURVIVE normalisation.

    Regression test for a real defect: `normalise()` originally root-centred on the
    mid-hip of every frame independently, which subtracts the body's trajectory and
    therefore erases whole-body descent - the primary fall cue. It was caught by the fall
    head's smoke test scoring AUROC 0.466 (below chance) on data whose only signal was a
    1.5-unit hip drop, because after centring the two classes were byte-identical.

    Position invariance and motion preservation are different requirements, and S4 only
    tested the first. Both are asserted here so they cannot be traded off silently again.
    """
    T = 30
    base = np.zeros((17, 2), dtype=np.float32)
    base[[5, 6]] = [[-0.2, 0.5], [0.2, 0.5]]
    base[[11, 12]] = [[-0.15, -0.5], [0.15, -0.5]]

    def make(drop: float, x_offset: float = 0.0) -> np.ndarray:
        w = np.zeros((T, 2, 17, 3), dtype=np.float32)
        person = np.tile(base, (T, 1, 1))
        person[..., 1] -= np.linspace(0, drop, T)[:, None]
        person[..., 0] += x_offset
        w[:, 0, :, :2] = person
        w[:, 0, :, 2] = 0.9
        return w

    falling = normalise(make(drop=1.5))
    standing = normalise(make(drop=0.0))

    def hip_travel(w: np.ndarray) -> float:
        mh = (w[:, 0, 11, 1] + w[:, 0, 12, 1]) / 2.0
        return float(mh[0] - mh[-1])

    fall_drop, stand_drop = hip_travel(falling), hip_travel(standing)
    assert fall_drop > 0.5, (
        f"whole-body descent was normalised away (hip travel {fall_drop:.4f}); "
        "falls become indistinguishable from standing"
    )
    assert abs(stand_drop) < 0.05, f"static body appears to move ({stand_drop:.4f})"
    assert np.abs(falling - standing).max() > 0.5, "falling and standing normalise alike"

    # ...while position in the room is still removed.
    here, there = normalise(make(1.5, 0.0)), normalise(make(1.5, 7.0))
    assert np.abs(here - there).max() < 1e-4, "translation invariance was lost"
    print(f"  S4b hip travel: falling {fall_drop:.3f} vs standing {stand_drop:.3f}; "
          f"still translation invariant (delta {np.abs(here-there).max():.2e})")


def test_s5_augmentation_preserves_structure():
    cfg = AugmentConfig(enabled=True)
    rng = np.random.default_rng(0)
    x = normalise(synth_window(seed=4))
    outs = [augment(x.copy(), cfg, np.random.default_rng(s)) for s in range(30)]

    assert all(o.shape == x.shape for o in outs)
    assert any(np.abs(o - x).max() > 1e-3 for o in outs), "augmentation is a no-op"
    for o in outs:
        assert np.isfinite(o).all(), "augmentation produced NaN/inf"
    # Disabled config must be an exact identity, or ablations measure the wrong thing.
    off = augment(x.copy(), AugmentConfig(enabled=False), rng)
    assert np.array_equal(off, x), "disabled augmentation still modified the input"
    # Scores are confidences. Two transforms legitimately touch them and a third must
    # not: a horizontal flip PERMUTES them along with their joints (after mirroring, the
    # left wrist's confidence belongs to the right wrist), joint dropout ZEROES them
    # (an occluded joint has no confidence), but no geometric transform may RESCALE them.
    # The first version of this assertion demanded elementwise equality and failed on
    # correct code; the second forbade dropout. The real invariant is that every surviving
    # score is a value that was present in the input.
    src = set(np.unique(x[..., 2]).tolist()) | {0.0}
    for o in outs:
        assert set(np.unique(o[..., 2]).tolist()) <= src, (
            "augmentation invented or rescaled confidence values"
        )
    no_drop = AugmentConfig(enabled=True, joint_dropout=0.0)
    for s in range(10):
        o = augment(x.copy(), no_drop, np.random.default_rng(s))
        assert np.allclose(np.sort(o[..., 2], axis=-1), np.sort(x[..., 2], axis=-1)), (
            "without dropout, the multiset of scores must be exactly preserved"
        )
    mean_delta = float(np.mean([np.abs(o - x).mean() for o in outs]))
    print(f"  S5 30 augmentations: finite, scores intact, mean|delta|={mean_delta:.3f}; "
          f"disabled == identity")


def test_s5b_temporal_resample_hits_exact_length():
    for T_in in (12, 30, 77):
        for T_out in (16, 30, 48):
            x = synth_window(T=T_in, seed=5)
            out = temporal_resample(x, T_out, None, None)
            assert out.shape[0] == T_out, f"{T_in}->{T_out} gave {out.shape[0]}"
            assert np.isfinite(out).all()
    jittered = temporal_resample(synth_window(T=30), 30,
                                 np.random.default_rng(1), AugmentConfig(enabled=True))
    assert jittered.shape[0] == 30
    print("  S5b resample exact for 9 (T_in,T_out) pairs incl. speed jitter")


def test_s6_subject_split_and_balanced_sampling():
    subjects = np.array([f"s{i%10:02d}" for i in range(500)])
    tr, va = split_by_subject(subjects, val_frac=0.2, seed=0)
    assert len(tr) + len(va) == 500
    assert not (set(subjects[tr]) & set(subjects[va])), "subject leakage"

    # A deliberately skewed label set: class 0 is 100x rarer than class 1.
    labels = np.array([0] * 5 + [1] * 500 + [2] * 100)
    sampler = class_balanced_sampler(labels)
    drawn = np.array([labels[i] for i in list(sampler)])
    frac = np.bincount(drawn, minlength=3) / len(drawn)
    raw = np.bincount(labels, minlength=3) / len(labels)
    assert frac[0] > raw[0] * 5, f"rare class barely upsampled: {raw[0]:.4f} -> {frac[0]:.4f}"
    assert frac[0] < 0.9, "rare class now dominates - that is memorisation, not balance"
    print(f"  S6 subjects disjoint ({len(set(subjects[tr]))}/{len(set(subjects[va]))}); "
          f"class-0 share {raw[0]:.3f} -> {frac[0]:.3f}")


def test_s6b_dataset_rejects_out_of_range_labels():
    """An out-of-range label must fail loudly at load, not 40 minutes into a session."""
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "bad.npz"
        np.savez_compressed(
            bad,
            skeletons=np.zeros((4, 30, 2, 17, 3), dtype=np.float16),
            labels=np.array([0, 1, 99, 3]),
            subjects=np.array(["a", "b", "c", "d"], dtype="<U32"),
            datasets=np.array(["x"] * 4, dtype="<U32"),
        )
        try:
            SkeletonWindowDataset([bad])
        except ValueError as exc:
            assert "99" in str(exc) or "outside" in str(exc), exc
            print(f"  S6b out-of-range label rejected at load: {str(exc)[:60]}...")
            return
    raise AssertionError("dataset accepted a label outside [0,20)")


def test_s7_resume_is_bit_exact():
    """An interrupted+resumed run must match an uninterrupted run exactly.

    Interruption is simulated with `--stop-after`, NOT by lowering `--epochs`. That
    distinction is the whole point: the cosine LR schedule is a function of TOTAL epochs,
    so a 3-epoch run and the first 3 epochs of a 6-epoch run legitimately follow
    different LR curves (0.0125 vs 0.0226 at step 12). The first version of this test
    conflated the two and reported a 3.34 weight divergence as a resume bug, when the
    real defect was that resuming with different hyperparameters is silently wrong -
    now refused outright by the guard tested in S7b.

    This is the test that justifies storing RNG state in the checkpoint: without it, a
    resumed run redraws augmentations and sampler order and drifts within one epoch.
    """
    script = ROOT / "scripts" / "train_adl.py"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        common = [sys.executable, str(script), "--device", "cpu", "--batch-size", "32",
                  "--workers", "0", "--warmup-epochs", "1", "--seed", "0",
                  "--smoke-classes", "6", "--epochs", "6"]

        full = subprocess.run(
            common + ["--smoke-shard", str(td / "a.npz"), "--out", str(td / "full")],
            capture_output=True, text=True, timeout=2400)
        assert full.returncode == 0, f"full run failed:\n{full.stdout[-1500:]}{full.stderr[-1500:]}"

        part = subprocess.run(
            common + ["--smoke-shard", str(td / "b.npz"), "--out", str(td / "part"),
                      "--stop-after", "3"],
            capture_output=True, text=True, timeout=2400)
        assert part.returncode == 0, f"partial run failed:\n{part.stderr[-1500:]}"
        assert "simulated timeout" in part.stdout

        cont = subprocess.run(
            common + ["--smoke-shard", str(td / "b.npz"), "--out", str(td / "part"),
                      "--resume", str(td / "part" / "last.pt")],
            capture_output=True, text=True, timeout=2400)
        assert cont.returncode == 0, f"resumed run failed:\n{cont.stderr[-1500:]}"
        assert "resumed from" in cont.stdout, "resume path was not taken"

        a = torch.load(td / "full" / "last.pt", map_location="cpu", weights_only=False)
        b = torch.load(td / "part" / "last.pt", map_location="cpu", weights_only=False)
        assert a["epoch"] == b["epoch"] == 5, f"{a['epoch']} vs {b['epoch']}"
        assert a["ema_step"] == b["ema_step"], f"EMA step {a['ema_step']} vs {b['ema_step']}"

        worst = max((a["ema"][k].float() - b["ema"][k].float()).abs().max().item()
                    for k in a["ema"])
        assert worst < 1e-4, f"resumed EMA weights diverge by {worst:.2e}"
        la, lb = a["metrics"]["train_loss"], b["metrics"]["train_loss"]
        assert abs(la - lb) < 1e-3, f"final loss {la:.6f} vs {lb:.6f}"
        print(f"  S7 6-epoch run == 3+resume run: max EMA delta {worst:.2e}, "
              f"final loss {la:.6f} vs {lb:.6f}")


def test_s7b_resume_refuses_changed_hyperparameters():
    """Resuming with a different --epochs must fail loudly, not silently re-curve the LR.

    This is the hazard S7's first version mistook for a bug: at step 12 of an identical
    run, --epochs 3 gives LR 0.0125 and --epochs 6 gives 0.0226. Continuing across that
    change produces a run whose schedule matches neither setting.
    """
    script = ROOT / "scripts" / "train_adl.py"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        base = [sys.executable, str(script), "--device", "cpu", "--batch-size", "32",
                "--workers", "0", "--warmup-epochs", "1", "--seed", "0",
                "--smoke-classes", "6", "--smoke-shard", str(td / "s.npz"),
                "--out", str(td / "run")]
        first = subprocess.run(base + ["--epochs", "2"], capture_output=True,
                               text=True, timeout=2400)
        assert first.returncode == 0, first.stderr[-1000:]

        bad = subprocess.run(base + ["--epochs", "9", "--resume", str(td / "run" / "last.pt")],
                             capture_output=True, text=True, timeout=2400)
        assert bad.returncode != 0, "resume accepted a changed --epochs"
        assert "resume mismatch" in (bad.stdout + bad.stderr), bad.stdout[-500:]

        ok = subprocess.run(base + ["--epochs", "2", "--resume", str(td / "run" / "last.pt")],
                            capture_output=True, text=True, timeout=2400)
        assert ok.returncode == 0, (
            f"control failed: matching hyperparameters were also refused\n{ok.stderr[-800:]}"
        )
        print("  S7b changed --epochs refused on resume; identical settings accepted")


def test_s5c_augmentation_keeps_missing_joints_at_exact_zero():
    """Train-time and eval-time "missing joint" must be the same representation.

    normalise() pins joints RTMO did not see at exact zero. Augmentation then translated
    and noised EVERY joint, so at train time a missing joint was a small random offset
    while at val/serve time (augmentation off) it stayed exact zero - the model learned
    an encoding of absence that evaluation never produces. augment() now re-zeroes
    score<=0 joints after the geometric transforms.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.5, size=(30, 2, 17, 3)).astype(np.float32)
    x[..., 2] = 0.9
    x[:, 0, 9, :] = 0.0          # left wrist invisible for the whole window
    x[:, 0, 15, :] = 0.0         # left ankle too
    x = normalise(x)
    assert np.abs(x[:, 0, [9, 15], :2]).max() == 0.0, "normalise no longer zeroes invisible"

    cfg = AugmentConfig(enabled=True, joint_dropout=0.0)
    n_invisible_in = int((x[..., 2] <= 0).sum())
    drift, moved = 0.0, 0.0
    for s in range(30):
        out = augment(x.copy(), cfg, np.random.default_rng(s))
        # The flip PERMUTES joints, so "the invisible joint" may sit in a different slot
        # now - but its score 0 travels with it. The invariant is positional-agnostic:
        # wherever the score says missing, the coords are exact zero.
        invis = out[..., 2] <= 0
        assert int(invis.sum()) == n_invisible_in, "augmentation created/destroyed absence"
        if invis.any():
            drift = max(drift, float(np.abs(out[..., :2][invis]).max()))
        moved = max(moved, float(np.abs(out - x).max()))
    assert drift == 0.0, (
        f"missing joints drifted to {drift:.4f} under augmentation - train-time "
        "'missing' no longer matches the exact zero that val and serving produce")
    # Discriminating control: the transforms must still MOVE visible joints, or a broken
    # identity augmentation would pass this test vacuously.
    assert moved > 1e-3, f"outputs differ from input by only {moved:.2e} - augmentation inert?"
    print(f"  S5c 30 augmentations: {n_invisible_in} missing-joint entries stay exact "
          f"zero (flip permutes their slots), visible joints moved up to {moved:.3f}")


def test_s4c_normalise_ignores_detection_dropout_frames():
    """Absent frames must not corrupt the scale or the origin of the visible ones.

    Extraction writes all-zero frames when detection fails. Including those in the torso
    mean deflated the scale (inflating every coordinate), and an absent frame 0 made the
    origin (0,0) - that window stayed in raw pixel coordinates while every other window
    sat in torso units.
    """
    clean = synth_window(T=20, M=1, seed=11)
    clean[..., :2] += 3.0                       # away from the origin, so centring matters

    # Same 20 visible frames, wrapped in dropout: 6 absent frames first, 4 mid-window.
    gappy = np.concatenate([np.zeros((6, 1, 17, 3), np.float32), clean[:3],
                            np.zeros((4, 1, 17, 3), np.float32), clean[3:]])
    vis = np.ones(30, dtype=bool); vis[:6] = False; vis[9:13] = False

    out_clean = normalise(clean)
    out_gappy = normalise(gappy)
    same = np.abs(out_gappy[vis] - out_clean).max()
    assert same < 1e-6, (
        f"visible frames differ by {same:.4f} once dropout frames are added - "
        "absent frames are still influencing scale or origin")
    assert np.abs(out_gappy[~vis]).max() == 0.0, "absent frames must stay exact zero"

    # Negative control: the old computation (torso mean over ALL frames, origin at
    # frame 0) must disagree, or this test cannot detect a regression to it.
    person = gappy[:, 0]
    mid_hip = (person[:, 11, :2] + person[:, 12, :2]) / 2.0
    mid_sh = (person[:, 5, :2] + person[:, 6, :2]) / 2.0
    diluted = np.linalg.norm(mid_sh - mid_hip, axis=-1).mean()
    proper = np.linalg.norm((mid_sh - mid_hip)[vis], axis=-1).mean()
    assert diluted < 0.75 * proper, "control: dilution should shrink the torso estimate"
    assert np.abs(mid_hip[0]).max() == 0.0, "control: old origin (frame 0) is (0,0) here"
    print(f"  S4c dropout frames excluded: visible output identical (delta {same:.1e}); "
          f"old scale would be {diluted/proper:.2f}x too small and origin (0,0)")


def test_s8_slot_assignment_survives_an_area_rank_flip():
    """Two people must keep their channels when their apparent sizes cross.

    Area-ranked assignment splices trajectories at every rank flip: person A bends,
    person B stands, and both channels teleport. assign_slots matches to the previous
    frame's centroids instead; area order applies only when there is no history.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from prepare_skeletons import assign_slots

    rng = np.random.default_rng(0)
    left = rng.normal(0, 0.1, (17, 2)).astype(np.float32) + [-2.0, 0.0]
    right = rng.normal(0, 0.1, (17, 2)).astype(np.float32) + [2.0, 0.0]
    big, small = 2.0, 1.0

    def detections(t):
        # Left person shrinks over time, right grows: ranks cross mid-sequence.
        sl = big + (small - big) * t / 9
        sr = small + (big - small) * t / 9
        drift = np.float32([0.02 * t, 0.0])
        return ([left * sl + drift, right * sr - drift],
                [np.full(17, 0.9, np.float32), np.full(17, 0.8, np.float32)])

    tracked, prev = [], None
    for t in range(10):
        kp, sc = detections(t)
        prev = assign_slots(kp, sc, prev)
        tracked.append(prev.copy())
    tracked = np.stack(tracked)
    jump_tracked = np.abs(np.diff(tracked[:, :, :, 0], axis=0)).max()
    assert jump_tracked < 1.0, (
        f"tracked slots jumped {jump_tracked:.2f} in x - identity swapped despite tracking")

    # Negative control: per-frame area order (the old notebook code) must swap.
    naive = []
    for t in range(10):
        kp, sc = detections(t)
        arr = np.zeros((2, 17, 3), np.float32)
        areas = [np.ptp(k[:, 0]) * np.ptp(k[:, 1]) for k in kp]
        for slot, pi in enumerate(np.argsort(areas)[::-1][:2]):
            arr[slot, :, :2] = kp[pi]
            arr[slot, :, 2] = sc[pi]
        naive.append(arr)
    jump_naive = np.abs(np.diff(np.stack(naive)[:, :, :, 0], axis=0)).max()
    assert jump_naive > 2.0, "control: area order should teleport at the rank flip"

    # A lone new entrant must not steal the resident's slot.
    solo_prev = assign_slots(*detections(0), None)
    entrant = assign_slots([right * 0.5], [np.full(17, 0.7, np.float32)],
                           np.concatenate([solo_prev[:1], np.zeros((1, 17, 3), np.float32)]))
    assert np.abs(entrant[0]).max() == 0.0 and np.abs(entrant[1]).max() > 0.0, (
        "a new entrant far from slot 0's resident should land in the empty slot 1")
    print(f"  S8 tracked slots: max x-jump {jump_tracked:.2f} vs naive {jump_naive:.2f} "
          f"at the rank flip; new entrant kept out of the resident's slot")



def test_s9_early_stopping_ends_a_run_without_losing_its_peak():
    """80 epochs cost ~2.4 of 2.7 GPU-hours and made the result WORSE.

    Measured on the real Charades shards, all four ADL streams:

        adl_bone          best 0.184 @ep11  ->  ep79 0.137   -25.5%
        adl_joint         best 0.179 @ep 9  ->  ep79 0.129   -27.9%
        adl_joint_motion  best 0.129 @ep10  ->  ep79 0.092   -28.7%
        adl_bone_motion   best 0.125 @ep 8  ->  ep79 0.091   -27.2%

    Four independent runs peaking in the same three-epoch window is not noise. The 80 came
    from ST-GCN++'s NTU-60 recipe, which has ~10x more labelled windows per class.

    The fall head measured the OPPOSITE: AUPRC peaked at ep56 of 60, still climbing. So
    the two heads get different budgets, and this test pins both - a single shared value
    would either overfit the ADL heads or truncate the fall head.
    """
    for script, want_epochs, want_patience in (("train_adl.py", 30, 8),
                                               ("train_fall.py", 60, 15)):
        src = (ROOT / "scripts" / script).read_text(encoding="utf-8")
        ep = re.search(r'"--epochs", type=int, default=(\d+)', src)
        pa = re.search(r'"--patience", type=int, default=(\d+)', src)
        assert ep and int(ep.group(1)) == want_epochs, (
            f"{script}: --epochs default is {ep and ep.group(1)}, want {want_epochs}")
        assert pa and int(pa.group(1)) == want_patience, (
            f"{script}: --patience default is {pa and pa.group(1)}, want {want_patience}")

    # Replay the real curves through the same stopping rule the scripts use.
    def stop_epoch(curve, patience):
        best = (-1.0, -1)
        for e, m in enumerate(curve):
            if m > best[0]:
                best = (m, e)
            if patience and e - best[1] >= patience:
                return e, best
        return len(curve) - 1, best

    adl = [0.05 + 0.134 * (e / 11) if e < 11 else 0.184 - 0.047 * min(1, (e - 11) / 68)
           for e in range(80)]
    stopped, best = stop_epoch(adl, 8)
    assert best[1] == 11 and abs(best[0] - 0.184) < 1e-6, (
        f"ADL peak lost: best {best}")
    assert stopped < 25, f"ADL ran to ep{stopped}; the point is to stop early"

    fall = [0.056 + 0.766 * (e / 56) ** 0.7 if e <= 56 else 0.822 - 0.009 * (e - 56)
            for e in range(60)]
    stopped, best = stop_epoch(fall, 15)
    assert best[1] >= 50, (
        f"fall head truncated at ep{stopped}, peak at ep{best[1]} - patience=15 must NOT "
        "cut a run that is still improving at ep56 of 60")

    # best.pt is written on improvement, so stopping cannot cost the peak.
    adl_src = (ROOT / "scripts" / "train_adl.py").read_text(encoding="utf-8")
    save_at = adl_src.index('torch.save(ck, out_dir / "best.pt")')
    stop_at = adl_src.index("early stop at epoch")
    assert save_at < stop_at, (
        "the early-stop branch runs before best.pt is written - a stopping run would "
        "discard the very epoch it stopped for")
    print("  S9 ADL 30ep/patience8 stops at ep19 keeping 0.184@ep11; fall 60ep/patience15 "
          "runs to ep59 keeping 0.822@ep56; best.pt saved before the stop check")


def test_s10_logit_adjusted_loss_reduces_to_plain_ce_at_tau_zero():
    """The default training path must be unchanged by adding this loss.

    `LogitAdjustedCE` sits in the default code path with `--tau-train 0`, so if tau=0 were
    not an exact no-op every existing result would silently shift. Asserted against
    `CrossEntropyLoss` on the same logits, including with label smoothing on, because that
    is how it is actually configured.
    """
    from train_adl import LogitAdjustedCE, label_prior

    torch.manual_seed(0)
    logits = torch.randn(64, N_CLASSES)
    target = torch.randint(0, N_CLASSES, (64,))
    labels = np.concatenate([np.zeros(400, int), np.arange(N_CLASSES)])
    prior = label_prior(labels, N_CLASSES)

    for smoothing in (0.0, 0.1):
        plain = torch.nn.CrossEntropyLoss(label_smoothing=smoothing)(logits, target)
        adjusted = LogitAdjustedCE(prior, tau=0.0, label_smoothing=smoothing)(logits, target)
        assert torch.allclose(plain, adjusted, atol=1e-6), (
            f"tau=0 changed the loss at label_smoothing={smoothing}: "
            f"{plain.item():.6f} vs {adjusted.item():.6f}"
        )

    # And tau>0 must actually move it, in the direction that up-weights the tail: the head
    # class gets a larger positive offset, so its logit needs to be higher to win.
    loss1 = LogitAdjustedCE(prior, tau=1.0)(logits, target)
    assert not torch.allclose(loss1, LogitAdjustedCE(prior, tau=0.0)(logits, target)), (
        "tau=1 produced the same loss as tau=0 - the offset is not being applied"
    )
    offset = LogitAdjustedCE(prior, tau=1.0).offset
    assert offset.argmax().item() == int(np.argmax(prior)), (
        "the largest offset is not on the most frequent class, so the adjustment has the "
        "wrong sign and would penalise the tail instead of the head"
    )

    # The buffer must be registered, or it stays on the CPU while logits are on the GPU.
    assert "offset" in dict(LogitAdjustedCE(prior).named_buffers())
    print(f"  S10 tau=0 == CrossEntropyLoss at smoothing 0.0/0.1; tau=1 offset peaks on "
          f"class {offset.argmax().item()} (prior {prior.max():.1%}); offset is a buffer")


def test_s11_balanced_sampling_and_logit_adjusted_loss_cannot_be_stacked():
    """Two corrections for one imbalance is a silent methodological error.

    Effective-number sampling already removes most of the head/tail ratio. Adding
    tau*log(prior) on top over-penalises the head, and the run produces a plausible number
    that means nothing. `train_adl.py` refuses the combination rather than warning - the
    same choice train_fall.py makes for focal alpha, and for the same reason: a warning in
    a 12-hour Kaggle log is not read until the result is already wrong.
    """
    with tempfile.TemporaryDirectory() as td:
        shard = Path(td) / "smoke.npz"
        base = [sys.executable, str(ROOT / "scripts/train_adl.py"), "--smoke",
                "--smoke-shard", str(shard), "--epochs", "1", "--stop-after", "1",
                "--out", str(Path(td) / "run")]

        bad = subprocess.run([*base, "--sampler", "balanced", "--tau-train", "1.0"],
                             capture_output=True, text=True, timeout=900)
        assert bad.returncode != 0, "stacking balanced sampling and tau-train was accepted"
        blob = bad.stdout + bad.stderr
        assert "corrects the class imbalance twice" in blob, blob[-400:]
        assert "--sampler natural" in blob, "the refusal does not say how to fix it"

        # Positive control: each alone must run. Without this the test would pass for a
        # script that refused everything.
        for extra in (["--sampler", "balanced"], ["--sampler", "natural", "--tau-train", "1.0"]):
            ok = subprocess.run([*base, *extra], capture_output=True, text=True, timeout=900)
            assert ok.returncode == 0, f"{extra} failed: {(ok.stdout + ok.stderr)[-500:]}"
    print("  S11 balanced+tau-train refused with a fix-it message; each alone still trains")


def test_s12_rng_state_reloads_when_the_checkpoint_came_back_off_a_device():
    """Resume must survive `map_location="cuda"` moving the RNG tensor.

    All five GPU runs died here: `torch.load(..., map_location=args.device)` moves EVERY
    tensor in the checkpoint to the GPU, the RNG state included, and
    `torch.set_rng_state` accepts only a CPU ByteTensor:

        TypeError: RNG state must be a torch.ByteTensor

    S7 never caught it because S7 resumes with `--device cpu`, where `map_location` is a
    no-op. There is no GPU here either, so the failure is reproduced by handing
    `load_rng_state` a state whose tensor did NOT come back as a CPU ByteTensor - the same
    precondition violation, without needing CUDA.
    """
    from train_adl import _byte_cpu, load_rng_state, rng_state

    torch.manual_seed(1234)
    saved = rng_state()
    expected = torch.rand(4)

    # Negative control: the raw torch API rejects a non-ByteTensor, so the fixture really
    # does reproduce the failure and a no-op _byte_cpu could not pass this test.
    moved = saved["torch"].to(torch.int64)
    try:
        torch.set_rng_state(moved)
        raise AssertionError("torch.set_rng_state accepted a non-ByteTensor; fixture is "
                             "no longer a faithful reproduction of the GPU failure")
    except (TypeError, RuntimeError):
        pass

    for mangled in (saved["torch"].to(torch.int64), saved["torch"].clone()):
        torch.manual_seed(999)
        load_rng_state({**saved, "torch": mangled})
        assert torch.allclose(torch.rand(4), expected), (
            "RNG state did not restore bit-exactly after coercion"
        )

    assert _byte_cpu(saved["torch"]).dtype == torch.uint8
    assert _byte_cpu(saved["torch"]).device.type == "cpu"
    assert _byte_cpu("not a tensor") == "not a tensor", "non-tensors must pass through"

    # train_fall.py imports this from train_adl, so one fix covers both trainers - assert
    # the import is still the shared one rather than a divergent copy.
    fall = (ROOT / "scripts/train_fall.py").read_text(encoding="utf-8")
    assert "load_rng_state" in fall and "from train_adl import" in fall, (
        "train_fall.py no longer shares load_rng_state with train_adl.py"
    )
    print("  S12 RNG state restores bit-exactly from an int64/off-device tensor; raw "
          "set_rng_state still rejects it (control); train_fall shares the fix")


def test_s13_actor_split_removes_the_leak_the_video_split_hides():
    """P1's split identity: video ids leak people; Charades actor ids must not.

    docs/07 recorded "Charades subject ids do not exist publicly" and the shards stored
    video ids as `subjects`. That was wrong - Charades_v1_train.csv has carried a `subject`
    column all along (267 actors, ~30 videos each). With that ratio a video-id split puts
    essentially every actor on both sides, so the "subject-disjoint" P1 was video-disjoint:
    same person, same home, same mannerisms in train and val. This is the exact failure
    `subject_from_path` prevents in the fall corpora, committed on the biggest corpus, and
    the disjointness assert could never catch it because the IDS really are disjoint.

    The test builds a Charades-shaped CSV (quoted fields with commas, as the real one has),
    shows the leak exists under video ids, and shows the remap eliminates it.
    """
    import csv as _csv

    from behaviorsense.data.skeleton_dataset import load_subject_map, remap_subjects

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "Charades_v1_train.csv"
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow(["id", "subject", "scene", "quality", "relevance", "verified",
                        "script", "descriptions", "actions", "length"])
            for i in range(30):
                w.writerow([f"VID{i:03d}", f"ACT{i % 5}", "Kitchen", 7, 7, "Yes",
                            "a person walks, then sits", "walks; sits",
                            "c093 0.0 2.0", "10.0"])
        mapping = load_subject_map(p)
        assert len(mapping) == 30 and mapping["VID007"] == "ACT2"

        subjects = np.array([f"VID{i % 30:03d}" for i in range(300)] + ["urfd-01"] * 10)
        remapped, cov = remap_subjects(subjects, mapping)
        assert abs(cov - 300 / 310) < 1e-9, cov
        assert remapped[-1] == "urfd-01", "non-Charades ids must pass through unchanged"
        assert len(set(remapped[:-10])) == 5, "30 videos should collapse to 5 actors"

        actors = np.array([mapping.get(s, s) for s in subjects])
        tr_v, va_v = split_by_subject(subjects, val_frac=0.2, seed=0)
        leak_video = set(actors[tr_v]) & set(actors[va_v]) - {"urfd-01"}
        assert leak_video, (
            "the video-id split leaked no actors on this fixture - the positive control "
            "is gone, so the actor-split assertion below proves nothing"
        )
        tr_a, va_a = split_by_subject(remapped, val_frac=0.2, seed=0)
        assert not (set(actors[tr_a]) & set(actors[va_a])), (
            "actor-id split still places one person on both sides"
        )

        bad = Path(td) / "bad.csv"
        bad.write_text("id,scene\nVID000,Kitchen\n", encoding="utf-8")
        try:
            load_subject_map(bad)
            raise AssertionError("CSV without a subject column was accepted")
        except ValueError:
            pass

    # And the trainer must expose the flag + refuse a resume across split identities.
    src = (ROOT / "scripts/train_adl.py").read_text(encoding="utf-8")
    assert "--subject-map" in src, "train_adl.py lost the --subject-map flag"
    assert '"subject_map"' in src, (
        "subject_map is not in the resume guard - a checkpoint from the other split "
        "identity would resume silently and smuggle its leakage into this protocol"
    )
    print(f"  S13 video-id split leaked {len(leak_video)} actor(s); actor-id split leaked "
          "0; pass-through + coverage + bad-CSV refusal + resume guard all hold")


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
