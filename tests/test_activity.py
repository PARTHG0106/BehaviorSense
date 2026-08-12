"""Tests for Agent 2 (activity): calibration, fusion, smoothing, abstention, segments.

The load-bearing pair is A4/A4b. A smoother is judged by two opposing requirements:
remove single-window flicker (A4) and NOT remove a single-window fall (A4b). Any test
suite that checks only A4 would happily pass a smoother that erases falls - which for
this system is the worst possible defect. Every detection-flavoured test prints what it
measured, per the practice in results/tuning_log.md.

Run: python tests/test_activity.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behaviorsense.agents.activity import (  # noqa: E402
    CLASS_NAMES,
    FALLEN,
    FALLING,
    N_CLASSES,
    ActivityAgent,
    ActivityConfig,
    build_transition_matrix,
    fit_temperature,
    softmax,
    viterbi,
)
from behaviorsense.schemas import Role  # noqa: E402

T0 = datetime(2026, 3, 1, 9, 0, 0)
SITTING, WALKING, STANDING = 2, 0, 1
DRINKING, MEDICATION = 10, 12


def times(n: int) -> tuple[list[datetime], list[datetime]]:
    starts = [T0 + timedelta(seconds=i) for i in range(n)]
    ends = [s + timedelta(seconds=2) for s in starts]
    return starts, ends


def logits_for(labels: list[int], strength: float = 4.0, seed: int = 0) -> np.ndarray:
    """Logits favouring `labels[i]` by `strength` over N(0,0.5) noise."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 0.5, size=(len(labels), N_CLASSES))
    for i, lab in enumerate(labels):
        z[i, lab] += strength
    return z


def test_a1_transition_matrix_is_sane():
    A = build_transition_matrix(ActivityConfig())
    assert A.shape == (N_CLASSES, N_CLASSES)
    assert np.allclose(A.sum(axis=1), 1.0), "rows must be stochastic"
    assert (A > 0).all(), "zero transitions make states unreachable forever"
    # The two structural asymmetries actually present:
    assert A[SITTING, FALLING] >= 0.05 - 1e-9, "falling not reachable in one step"
    assert A[FALLING, FALLEN] > A[FALLING, FALLING], "falling should drain into fallen"
    assert A[FALLING, FALLING] < 0.5, "falling must not be a persistent state"
    print(f"  A1 rows sum 1; P(sit->fall)={A[SITTING, FALLING]:.3f}, "
          f"P(fall->fallen)={A[FALLING, FALLEN]:.2f}, P(fall->fall)={A[FALLING, FALLING]:.2f}")


def test_a2_temperature_fitting_recovers_known_miscalibration():
    """Scale a CALIBRATED logit set by a known factor; the fitter must recover it.

    The subtlety that broke this test's first version: `rng.normal` logits with a bump on
    the true class are NOT calibrated - they are badly UNDERconfident (mean max-posterior
    0.30 vs accuracy 0.57), needing T=0.49. Scaling them by 3 therefore requires
    T = 3 x 0.49 = 1.47, which is exactly what the fitter returned while the test
    demanded 3.0. The fitter was right; the fixture's premise was false.

    So the reference point is now *measured* rather than assumed: calibrate the raw
    logits first, verify that fixture is actually calibrated, then test recovery of a
    known scaling on top of it.
    """
    rng = np.random.default_rng(7)
    n = 4000
    true = rng.integers(0, N_CLASSES, size=n)
    raw = rng.normal(0, 1.0, size=(n, N_CLASSES))
    raw[np.arange(n), true] += 2.0

    t_raw = fit_temperature(raw, true)
    calibrated = raw / t_raw
    t_check = fit_temperature(calibrated, true)
    assert abs(t_check - 1.0) < 0.05, (
        f"fixture is not calibrated after dividing by {t_raw:.3f}: refit gave {t_check:.3f}"
    )
    # ECE-style sanity: a calibrated set's mean confidence should track its accuracy.
    p_cal = softmax(calibrated)
    acc = float((p_cal.argmax(axis=1) == true).mean())
    conf = float(p_cal.max(axis=1).mean())
    assert abs(conf - acc) < 0.06, f"calibrated conf {conf:.3f} vs accuracy {acc:.3f}"

    for factor in (2.0, 3.0, 4.0):
        t = fit_temperature(calibrated * factor, true)
        assert abs(t - factor) < 0.15 * factor, f"x{factor}: fitted {t:.2f}"

    over = softmax(calibrated * 3.0).max(axis=1).mean()
    fixed = softmax(calibrated * 3.0 / fit_temperature(calibrated * 3.0, true)).max(axis=1).mean()
    print(f"  A2 raw logits needed T={t_raw:.2f} (underconfident, not calibrated); "
          f"after calibration conf={conf:.3f} vs acc={acc:.3f}; "
          f"x2/x3/x4 recovered; overconfident {over:.2f} -> {fixed:.2f}")


def test_a3_viterbi_agrees_with_argmax_when_transitions_are_uniform():
    """With a flat transition prior Viterbi must reduce to per-window argmax."""
    rng = np.random.default_rng(3)
    post = softmax(rng.normal(size=(50, N_CLASSES)) * 2)
    log_A = np.log(np.full((N_CLASSES, N_CLASSES), 1.0 / N_CLASSES))
    log_pi = np.log(np.full(N_CLASSES, 1.0 / N_CLASSES))
    path = viterbi(np.log(post), log_A, log_pi)
    assert (path == post.argmax(axis=1)).all(), "uniform-prior Viterbi != argmax"
    print("  A3 uniform-prior Viterbi == argmax on 50 random windows")


def test_a4_smoothing_removes_isolated_flicker():
    """sitting x8, one spurious 'standing' window, sitting x8 -> flicker removed."""
    labels = [SITTING] * 8 + [STANDING] + [SITTING] * 8
    z = logits_for(labels, strength=3.0, seed=1)
    agent = ActivityAgent()
    starts, ends = times(len(labels))
    results = agent.classify_stream(z, starts, ends)

    smoothed = [r.smoothed_label for r in results]
    raw = list(np.asarray(z).argmax(axis=1))
    assert raw[8] == STANDING, "fixture broken: flicker not present in raw argmax"
    assert smoothed[8] == SITTING, f"flicker survived smoothing: {smoothed}"
    assert set(smoothed) == {SITTING}
    segs = agent.to_segments(results, track_id=0, role=Role.RESIDENT)
    assert len(segs) == 1 and segs[0].activity_name == "sitting"
    assert segs[0].n_windows == 17
    print(f"  A4 raw had flicker at w8 ({CLASS_NAMES[raw[8]]}); smoothed -> "
          f"1 segment x {segs[0].n_windows} windows ({segs[0].duration_s:.0f}s sitting)")


def test_a4b_smoothing_must_not_erase_a_one_window_fall():
    """THE safety-critical property, tested with the SAME evidence strength as A4.

    The spurious 'standing' in A4 and the real 'falling' here have identical logit
    strength (3.0) and identical one-window duration. The smoother cannot tell them
    apart by evidence - only the asymmetric transition prior distinguishes them. If
    this test fails while A4 passes, the transition structure is wrong; if both fail,
    the smoother is broken; if A4 fails while this passes, smoothing does nothing.
    """
    labels = [WALKING] * 8 + [FALLING] + [FALLEN] * 6
    z = logits_for(labels, strength=3.0, seed=2)
    agent = ActivityAgent()
    starts, ends = times(len(labels))
    results = agent.classify_stream(z, starts, ends)
    smoothed = [r.smoothed_label for r in results]

    assert smoothed[8] == FALLING, (
        f"smoothing ERASED a one-window fall (got {CLASS_NAMES[smoothed[8]]}); "
        "the emergency transition floor is not doing its job"
    )
    assert all(s == FALLEN for s in smoothed[9:]), f"post-fall state lost: {smoothed[9:]}"
    segs = agent.to_segments(results, track_id=0, role=Role.RESIDENT)
    names = [s.activity_name for s in segs]
    assert names == ["walking", "falling", "fallen_on_ground"], names
    assert segs[1].n_windows == 1
    print(f"  A4b one-window fall SURVIVED smoothing (same strength as A4's flicker); "
          f"segments: {' -> '.join(names)}")


def test_a4c_low_confidence_isolated_fall_is_still_kept():
    """A fall window with weaker evidence (strength 2.0) must also survive - falls are
    exactly the windows where pose quality is worst (motion blur, unusual pose)."""
    labels = [WALKING] * 6 + [FALLING] + [FALLEN] * 5
    z = logits_for(labels, strength=2.0, seed=9)
    agent = ActivityAgent()
    starts, ends = times(len(labels))
    results = agent.classify_stream(z, starts, ends)
    smoothed = [r.smoothed_label for r in results]
    fall_conf = results[6].confidence
    assert smoothed[6] == FALLING, f"weak fall erased ({CLASS_NAMES[smoothed[6]]})"
    assert not results[6].abstained, "fall must be exempt from abstention"
    print(f"  A4c weak (conf={fall_conf:.2f}) one-window fall kept, not abstained")


def test_a5_object_fusion_flips_medication_only_with_evidence():
    """Ambiguous drink-vs-medication posture: the pill bottle is the deciding evidence.

    Both directions asserted: WITH pill_bottle -> taking_medication; WITHOUT any
    medication object -> drinking, even if the GCN slightly prefers medication. A fusion
    that always boosts medication would pass only the first half.
    """
    agent = ActivityAgent()
    z = np.zeros((1, N_CLASSES))
    z[0, DRINKING] = 2.0
    z[0, MEDICATION] = 2.2  # GCN slightly prefers medication (it cannot really tell)

    starts, ends = times(1)
    with_obj = agent.classify_stream(z, starts, ends, objects_per_window=[("pill_bottle",)])
    without = agent.classify_stream(z, starts, ends, objects_per_window=[()])

    assert with_obj[0].smoothed_label == MEDICATION
    assert without[0].smoothed_label == DRINKING, (
        f"without object evidence the medication penalty must win "
        f"(got {CLASS_NAMES[without[0].smoothed_label]})"
    )
    p_med_with = with_obj[0].posterior[MEDICATION]
    p_med_without = without[0].posterior[MEDICATION]
    assert p_med_with > 2 * p_med_without
    print(f"  A5 P(medication): with bottle {p_med_with:.2f}, without {p_med_without:.2f} "
          f"-> labels {CLASS_NAMES[with_obj[0].smoothed_label]} / "
          f"{CLASS_NAMES[without[0].smoothed_label]}")


def test_a5b_fusion_leaves_unambiguous_windows_alone():
    """Object priors must nudge ambiguity, not overturn strong pose evidence."""
    agent = ActivityAgent()
    z = np.zeros((1, N_CLASSES))
    z[0, WALKING] = 6.0  # unambiguous walking
    starts, ends = times(1)
    r = agent.classify_stream(z, starts, ends, objects_per_window=[("tv", "couch", "cup")])
    assert r[0].smoothed_label == WALKING, (
        f"objects overturned strong pose evidence -> {CLASS_NAMES[r[0].smoothed_label]}"
    )
    print(f"  A5b walking @6.0 logits kept despite tv/couch/cup context "
          f"(P={r[0].confidence:.2f})")


def test_a6_abstention_routes_low_confidence_to_other_idle():
    agent = ActivityAgent()
    rng = np.random.default_rng(4)
    z = rng.normal(0, 0.3, size=(10, N_CLASSES))  # near-uniform: nothing recognisable
    starts, ends = times(10)
    results = agent.classify_stream(z, starts, ends)

    assert all(r.abstained for r in results), "diffuse posteriors must abstain"
    assert all(r.label == 19 for r in results)
    segs = agent.to_segments(results, track_id=0, role=Role.RESIDENT)
    assert len(segs) == 1 and segs[0].activity_name == "other_idle"
    confident = logits_for([WALKING] * 5, strength=5.0)
    ok = agent.classify_stream(confident, *times(5))
    assert not any(r.abstained for r in ok), "control: confident windows must not abstain"
    print(f"  A6 diffuse -> 10/10 abstain to other_idle; confident control 0/5 abstain "
          f"(mean conf {np.mean([r.confidence for r in ok]):.2f})")


def test_a7_segments_carry_the_evidence_chain():
    labels = [WALKING] * 3 + [SITTING] * 4
    z = logits_for(labels, strength=4.0, seed=5)
    agent = ActivityAgent()
    starts, ends = times(len(labels))
    results = agent.classify_stream(
        z, starts, ends, objects_per_window=[("couch",)] * len(labels)
    )
    segs = agent.to_segments(results, track_id=3, role=Role.RESIDENT, room="lounge")

    assert [s.activity_name for s in segs] == ["walking", "sitting"]
    for s in segs:
        assert s.track_id == 3 and s.role is Role.RESIDENT and s.room == "lounge"
        assert s.supporting_objects == ["couch"]
        assert s.n_windows in (3, 4)
        assert 0.0 <= s.confidence <= 1.0 and s.mean_logit_margin is not None
        assert s.end_time > s.start_time
    total = sum(s.duration_s for s in segs)
    print(f"  A7 2 segments, {total:.0f}s total, objects+margin+room threaded through")


def test_a8_stream_with_no_windows_is_empty_not_crash():
    agent = ActivityAgent()
    assert agent.classify_stream(np.zeros((0, N_CLASSES)), [], []) == []
    assert agent.to_segments([], track_id=0, role=Role.RESIDENT) == []
    print("  A8 empty stream -> empty outputs")


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
