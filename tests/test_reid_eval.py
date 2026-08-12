"""Tests for the open-set ReID protocol and metrics.

The vacuity risk here is unusually high and specific
----------------------------------------------------
Almost every assertion one would naturally write about a ReID evaluation can be satisfied
by a broken implementation:

  - "AUROC is high" passes if the protocol leaks identities between enrolled and impostor.
  - "FAR is low" passes if the threshold rejects everything, including the resident.
  - "the split has N identities" passes while the builder silently dropped half of them.

So the metrics are driven by a SYNTHETIC embedder whose separability is a knob. The tests
assert the metrics MOVE with that knob - AUROC ~0.5 for a useless embedder, ~1.0 for a
perfect one, monotone in between - which no constant-output implementation can fake.
Protocol tests attack the three rigging modes directly by constructing rigged splits and
requiring `validate()` to reject them.

Run: python tests/test_reid_eval.py [path/to/MSMT17]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behaviorsense.data.reid_datasets import (  # noqa: E402
    Crop,
    OpenSetSplit,
    build_open_set_split,
    parse_crop,
    split_by_identity,
)
from behaviorsense.eval.reid_eval import (  # noqa: E402
    auroc,
    closed_set_rank1,
    eer,
    fit_threshold,
    format_report,
    l2_normalise,
    operating_point,
    score_split,
    sweep,
)

DIM = 64
REAL_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("MSMT17")


# ---------------------------------------------------------------------------
# Synthetic world
# ---------------------------------------------------------------------------


def synthetic_split(
    n_enrolled: int = 40,
    n_impostor: int = 200,
    enrol_per_id: int = 4,
    probes_per_id: int = 6,
) -> OpenSetSplit:
    """A protocol-shaped split with fabricated paths - no disk access."""
    s = OpenSetSplit(name="synthetic")
    for i in range(n_enrolled):
        pid = f"e{i:04d}"
        s.enrolment[pid] = [
            Crop(path=f"{pid}/enrol/{j}", pid=pid, camera="c1", frame=j)
            for j in range(enrol_per_id)
        ]
        s.enrol_cameras[pid] = {"c1"}
        s.genuine[pid] = [
            Crop(path=f"{pid}/probe/{j}", pid=pid, camera="c2", frame=100 + j)
            for j in range(probes_per_id)
        ]
    for i in range(n_impostor):
        pid = f"x{i:04d}"
        s.impostors.append(Crop(path=f"{pid}/imp/0", pid=pid, camera="c3", frame=0))
    s.validate()
    return s


def synthetic_embeddings(
    split: OpenSetSplit, separability: float, seed: int = 0
) -> dict[str, np.ndarray]:
    """Embeddings whose identity signal is controlled by `separability` in [0, 1].

    Each identity gets a random anchor direction. A crop's embedding is
    `separability * anchor + (1 - separability) * noise`. At 0 the embedding carries no
    identity at all (AUROC must be ~0.5); at 1 it is perfect (AUROC ~1.0). This knob is
    what makes the metric tests non-vacuous: a hard-coded metric cannot track it.
    """
    rng = np.random.default_rng(seed)
    anchors: dict[str, np.ndarray] = {}

    def anchor(pid: str) -> np.ndarray:
        if pid not in anchors:
            v = rng.normal(size=DIM)
            anchors[pid] = v / np.linalg.norm(v)
        return anchors[pid]

    out: dict[str, np.ndarray] = {}
    for crop in split.all_crops():
        noise = rng.normal(size=DIM)
        noise /= np.linalg.norm(noise)
        v = separability * anchor(crop.pid) + (1.0 - separability) * noise
        out[crop.path] = v / max(np.linalg.norm(v), 1e-12)
    return out


# ---------------------------------------------------------------------------
# R1-R4: protocol integrity
# ---------------------------------------------------------------------------


def test_r1_filename_parsing():
    ok = parse_crop("/d/0043_c12_0033.jpg", "0043_c12_0033.jpg", "msmt17")
    assert ok is not None and ok.pid == "0043" and ok.camera == "c12" and ok.frame == 33

    # Market-1501 reserves -1 (junk) and 0000 (distractor); MSMT17 does not.
    for name in ("-1_c1_0001.jpg", "0000_c1_0001.jpg"):
        assert parse_crop("/d/" + name, name, "market1501") is None, f"kept {name}"
        assert parse_crop("/d/" + name, name, "msmt17") is not None, f"dropped {name}"

    for junk in ("readme.txt", ".DS_Store", "weird.jpg", "0001_x1_0.jpg"):
        assert parse_crop("/d/" + junk, junk, "msmt17") is None, f"accepted {junk}"
    print("  R1 pid/camera/frame parsed; market1501 drops -1 and 0000, msmt17 keeps 0000")


def test_r2_validate_catches_identity_leakage():
    """An impostor that is also enrolled makes FAR meaningless. It must be rejected."""
    s = synthetic_split(n_enrolled=5, n_impostor=5)
    s.validate()  # positive control: the clean split passes

    pid = next(iter(s.enrolment))
    s.impostors.append(Crop(path="leak", pid=pid, camera="c9", frame=0))
    try:
        s.validate()
    except ValueError as exc:
        assert "enrolled and impostor" in str(exc)
        print(f"  R2 leaked id rejected: {str(exc)[:70]}...")
        return
    raise AssertionError("validate() accepted an identity that is both enrolled and impostor")


def test_r3_validate_catches_same_camera_probing():
    """Probing from the enrolment camera measures JPEG similarity, not re-identification."""
    s = synthetic_split(n_enrolled=5, n_impostor=5)
    pid = next(iter(s.enrolment))
    s.genuine[pid].append(Crop(path="samecam", pid=pid, camera="c1", frame=999))
    try:
        s.validate()
    except ValueError as exc:
        assert "enrolment camera" in str(exc)
        print(f"  R3 same-camera probe rejected: {str(exc)[:70]}...")
        return
    raise AssertionError("validate() accepted a probe from the enrolment camera")


def test_r3b_validate_catches_reused_crops_and_empty_impostors():
    s = synthetic_split(n_enrolled=5, n_impostor=5)
    pid = next(iter(s.enrolment))
    # Same FILE as an enrolment crop but claimed from camera c2: slips past the
    # same-camera check and must be caught by the shared-path check specifically.
    # (Reusing the crop verbatim tripped the camera check first and left the
    # shared-path branch untested - the R3b fixture bug from the last run.)
    reused = s.enrolment[pid][0]
    s.genuine[pid].append(Crop(path=reused.path, pid=pid, camera="c2", frame=999))
    try:
        s.validate()
        raise AssertionError("validate() accepted a crop used to both enrol and probe")
    except ValueError as exc:
        assert "enrol and to probe" in str(exc), exc

    s2 = synthetic_split(n_enrolled=5, n_impostor=5)
    s2.impostors.clear()
    try:
        s2.validate()
        raise AssertionError("validate() accepted a split with no impostors")
    except ValueError as exc:
        assert "no impostor" in str(exc), exc
    print("  R3b reused crops and impostor-free splits both rejected")


def test_r4_fit_test_identities_are_disjoint():
    s = synthetic_split(n_enrolled=40, n_impostor=200)
    fit, test = split_by_identity(s, seed=3)

    assert not set(fit.enrolment) & set(test.enrolment), "enrolled ids leak across halves"
    fi = {c.pid for c in fit.impostors}
    ti = {c.pid for c in test.impostors}
    assert not fi & ti, "impostor ids leak across halves"
    assert len(fit.enrolment) + len(test.enrolment) == len(s.enrolment)
    assert fit.n_impostor + test.n_impostor == s.n_impostor
    print(f"  R4 {len(s.enrolment)} ids -> fit {len(fit.enrolment)} / test "
          f"{len(test.enrolment)}, zero overlap; impostors {fit.n_impostor}/{test.n_impostor}")


# ---------------------------------------------------------------------------
# R5-R8: metrics respond to signal
# ---------------------------------------------------------------------------


def test_r5_auroc_tracks_embedding_quality():
    """AUROC must be ~0.5 for a useless embedder and rise monotonically with signal.

    This is the anti-vacuity test for the whole metrics module. A hard-coded or
    accidentally-inverted AUROC cannot produce a monotone response to this knob.
    """
    s = synthetic_split(n_enrolled=40, n_impostor=200)
    measured: list[tuple[float, float]] = []
    for sep in (0.0, 0.2, 0.4, 0.6, 0.8, 0.95):
        sc = score_split(s, synthetic_embeddings(s, sep, seed=1))
        measured.append((sep, auroc(sc)))

    assert abs(measured[0][1] - 0.5) < 0.06, (
        f"a signal-free embedder scored AUROC {measured[0][1]:.3f}, expected ~0.5 - "
        "the metric is not measuring separability"
    )
    assert measured[-1][1] > 0.99, f"a near-perfect embedder scored {measured[-1][1]:.3f}"
    for (s0, a0), (s1, a1) in zip(measured, measured[1:]):
        assert a1 >= a0 - 0.02, f"AUROC fell from {a0:.3f} to {a1:.3f} ({s0}->{s1})"
    print("  R5 AUROC vs separability: "
          + ", ".join(f"{s:.2f}->{a:.3f}" for s, a in measured))


def test_r5b_auroc_handles_ties():
    """A degenerate embedder returning one constant vector must score exactly 0.5."""
    s = synthetic_split(n_enrolled=10, n_impostor=40)
    const = np.ones(DIM) / np.sqrt(DIM)
    sc = score_split(s, {c.path: const.copy() for c in s.all_crops()})
    a = auroc(sc)
    assert abs(a - 0.5) < 1e-9, f"all-ties AUROC was {a:.6f}, expected exactly 0.5"
    print(f"  R5b constant embedder -> AUROC {a:.6f} (ties averaged, not broken by order)")


def test_r6_far_frr_trade_off_is_monotone():
    # Separability 0.40 sits in the measured informative regime (EER ~14%). At 0.6+ the
    # synthetic embedder is PERFECT: EER=0, and the "0 < rate" premise cannot hold -
    # which is how this test failed vacuously-in-reverse on its first run.
    s = synthetic_split(n_enrolled=40, n_impostor=200)
    sc = score_split(s, synthetic_embeddings(s, 0.40, seed=2))
    points = sweep(sc, n_points=200)

    fars = [p.far for p in points]
    frrs = [p.frr for p in points]
    assert all(b >= a - 1e-12 for a, b in zip(fars, fars[1:])), "FAR not non-decreasing in tau"
    assert all(b <= a + 1e-12 for a, b in zip(frrs, frrs[1:])), "FRR not non-increasing in tau"
    assert fars[0] == 0.0 and frrs[0] == 1.0, "tightest threshold should accept nothing"
    assert fars[-1] == 1.0 and frrs[-1] == 0.0, "loosest threshold should accept everything"

    rate, tau = eer(sc)
    assert 0.0 < rate < 0.5 and 0.0 < tau < 2.0
    print(f"  R6 FAR 0->1, FRR 1->0 monotone across {len(points)} thresholds; "
          f"EER {rate:.2%} at tau={tau:.3f}")


def test_r7_fit_threshold_respects_the_far_budget():
    # 0.40, not 0.65: at 0.65 separation is perfect, every FAR budget buys TAR=100%,
    # and "a looser budget buys more TAR" is unfalsifiable. See measured table in
    # docs/05_plan.md step 0.
    s = synthetic_split(n_enrolled=40, n_impostor=400)
    sc = score_split(s, synthetic_embeddings(s, 0.40, seed=4))

    prev_tar = -1.0
    rows = []
    for target in (0.001, 0.01, 0.05, 0.10):
        p = fit_threshold(sc, target)
        assert p.far <= target + 1e-12, f"FAR {p.far:.4f} exceeds budget {target}"
        assert p.tar >= prev_tar - 1e-9, "TAR fell as the FAR budget loosened"
        prev_tar = p.tar
        rows.append((target, p))
    assert rows[-1][1].threshold >= rows[0][1].threshold, "tau did not loosen with budget"
    assert rows[-1][1].tar > rows[0][1].tar, (
        "a 100x looser FAR budget bought no extra TAR - fit_threshold is not optimising"
    )
    print("  R7 " + "; ".join(f"FAR<={t:.1%}: tau={p.threshold:.3f} TAR={p.tar:.1%}"
                              for t, p in rows))


def test_r8_dir_is_stricter_than_tar():
    """DIR@1 must never exceed TAR, and must be strictly lower when misidentification occurs.

    TAR counts a probe accepted as the WRONG enrolled person as a success. DIR does not.
    That gap is the error that merges two residents' daily features, so the metric must
    expose it rather than hide it inside TAR.
    """
    s = synthetic_split(n_enrolled=60, n_impostor=200)
    sc = score_split(s, synthetic_embeddings(s, 0.45, seed=5))
    p = operating_point(sc, 0.5)

    assert p.dir_rank1 <= p.tar + 1e-12, f"DIR {p.dir_rank1:.3f} > TAR {p.tar:.3f}"
    assert p.dir_rank1 < p.tar, (
        "no misidentification at all in a deliberately weak-embedding regime - "
        "the test cannot distinguish DIR from TAR"
    )
    r1 = closed_set_rank1(sc)
    assert 0.0 < r1 < 1.0
    print(f"  R8 tau=0.5: TAR={p.tar:.1%} DIR@1={p.dir_rank1:.1%} "
          f"(gap {p.tar - p.dir_rank1:.1%} accepted-but-wrong), closed-set rank-1={r1:.1%}")


def test_r9_report_renders_and_flags_the_gap():
    s = synthetic_split(n_enrolled=40, n_impostor=200)
    sc = score_split(s, synthetic_embeddings(s, 0.55, seed=6))
    text = format_report(sc, current_threshold=0.30)
    for token in ("AUROC", "EER", "Operating points", "_current config_", "DIR@1"):
        assert token in text, f"report missing {token!r}"
    assert "| 1.0% |" in text, "target FAR row missing"
    print(f"  R9 report renders ({len(text.splitlines())} lines) with all sections")


# ---------------------------------------------------------------------------
# R10: the real dataset
# ---------------------------------------------------------------------------


def test_r10_real_dataset_split_is_exact_and_clean():
    """Build the real protocol from disk. Skips cleanly if the dataset is absent."""
    if not REAL_ROOT.is_dir():
        print(f"  R10 SKIP - {REAL_ROOT} not present")
        return

    n_enrolled, n_impostor_ids = 100, 400
    s = build_open_set_split(
        REAL_ROOT, n_enrolled=n_enrolled, n_impostor_ids=n_impostor_ids, seed=0
    )
    # The builder used to satisfy this silently-shrunken; an exact check is the guard.
    assert len(s.enrolment) == n_enrolled, (
        f"asked for {n_enrolled} enrolled ids, got {len(s.enrolment)} - the split shrank "
        "silently, which changes the experiment with no error"
    )
    assert len({c.pid for c in s.impostors}) == n_impostor_ids
    assert not set(s.enrolment) & {c.pid for c in s.impostors}
    for pid, probes in s.genuine.items():
        assert probes, f"enrolled id {pid} has no cross-camera probes"
        assert not {c.camera for c in probes} & s.enrol_cameras[pid]

    # Determinism: the same seed must rebuild the same experiment.
    again = build_open_set_split(
        REAL_ROOT, n_enrolled=n_enrolled, n_impostor_ids=n_impostor_ids, seed=0
    )
    assert sorted(again.enrolment) == sorted(s.enrolment), "seed does not reproduce the split"
    assert [c.path for c in again.impostors] == [c.path for c in s.impostors]

    other = build_open_set_split(
        REAL_ROOT, n_enrolled=n_enrolled, n_impostor_ids=n_impostor_ids, seed=1
    )
    assert sorted(other.enrolment) != sorted(s.enrolment), "different seeds gave one split"

    fit, test = split_by_identity(s, seed=0)
    assert not set(fit.enrolment) & set(test.enrolment)
    print(f"  R10 {s.summary()}")
    print(f"      seed-0 reproducible, seed-1 differs, fit/test disjoint "
          f"({len(fit.enrolment)}/{len(test.enrolment)} ids), "
          f"{len(s.all_crops())} crops need embeddings")


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
