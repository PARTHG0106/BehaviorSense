"""Tests for Agent 3.

Focus is on the claims the dissertation will make, not on line coverage:

  T1  robust baseline is not moved by a single outlier day  (mean/std would be)
  T2  MAD == 0 does not emit inf                            (a real crash path)
  T3  today never enters its own baseline                   (would mask deviations)
  T4  unreliable days are excluded                          (camera outage != decline)
  T5  falls alert with no baseline at all                   (safety on day 1)
  T6  CUSUM catches gradual decline that daily z misses     (the flagship claim)
  T7  every alert carries resolvable evidence               (auditability contract)
"""

from __future__ import annotations

import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behaviorsense.agents.behaviour import (  # noqa: E402
    BehaviourAnalyzer,
    BehaviourConfig,
    BaselineModel,
    DriftDetector,
    aggregate_daily_features,
)
from behaviorsense.schemas import (  # noqa: E402
    ActivitySegment,
    BaselineStats,
    DailyFeatures,
    Role,
    Z_CLAMP,
)

D0 = date(2026, 1, 1)


def make_day(
    day: date,
    walking_s: float = 1800.0,
    s2s: int = 20,
    transitions: int = 30,
    meals: int = 3,
    drinks: int = 6,
    meds: int = 1,
    social_s: float = 1200.0,
    lying_s: float = 28800.0,
    observed_hours: float = 24.0,
    coverage: float = 1.0,
    falls: int = 0,
    post_fall_s: float = 0.0,
    inactive_s: float = 3600.0,
) -> DailyFeatures:
    return DailyFeatures(
        subject_role=Role.RESIDENT,
        day=day,
        walking_duration_s=walking_s,
        walking_bouts=max(1, int(walking_s // 120)),
        mean_bout_duration_s=120.0,
        room_transitions=transitions,
        sit_to_stand_count=s2s,
        mean_sit_to_stand_duration_s=2.5,
        sitting_duration_s=14400.0,
        lying_duration_s=lying_s,
        standing_duration_s=3600.0,
        longest_inactive_block_s=inactive_s,
        meal_events=meals,
        eating_duration_s=meals * 900.0,
        drinking_events=drinks,
        cooking_duration_s=1800.0,
        medication_events=meds,
        tv_duration_s=7200.0,
        reading_duration_s=1800.0,
        phone_events=2,
        social_interaction_duration_s=social_s,
        visitor_count=1 if social_s > 0 else 0,
        housework_duration_s=1200.0,
        fall_events=falls,
        max_post_fall_immobility_s=post_fall_s,
        observed_hours=observed_hours,
        tracking_coverage=coverage,
    )


def test_t1_baseline_robust_to_outlier() -> None:
    """A single extreme day must not move the baseline. mean/std would shift hard."""
    model = BaselineModel(BehaviourConfig())
    for i in range(13):
        model.update(make_day(D0 + timedelta(days=i), walking_s=1800.0))
    model.update(make_day(D0 + timedelta(days=13), walking_s=100000.0))  # outlier

    stats = model.compute(before=D0 + timedelta(days=14))
    walk = stats["walking_duration_s"]
    assert abs(walk.median - 1800.0) < 1e-6, f"median moved to {walk.median}"
    # The mean would be ~8800 here - 4.9x the true central value.
    print(f"  T1 median={walk.median:.1f} (mean would be "
          f"{(13 * 1800 + 100000) / 14:.1f})")


def test_t2_zero_mad_is_floored_not_infinite() -> None:
    """Constant feature -> MAD 0. Must be floored, finite, and not hair-triggered.

    Asserts properties rather than a magic constant: the floor is a config parameter
    (mad_floor_frac), so pinning an exact z value would make the test brittle against
    a legitimate config change.
    """
    model = BaselineModel(BehaviourConfig())
    for i in range(10):
        model.update(make_day(D0 + timedelta(days=i), walking_s=1800.0))
    stats = model.compute(before=D0 + timedelta(days=10))
    walk = stats["walking_duration_s"]

    assert walk.mad == 0.0, "expected a degenerate baseline for a constant feature"
    assert walk.effective_mad() > 0.0, "floor must make effective MAD positive"
    assert abs(walk.effective_mad() - 0.05 * 1800.0) < 1e-9

    # Identical value -> exactly zero deviation.
    assert walk.robust_z(1800.0) == 0.0

    # A trivial 1% change must NOT look significant. This is the alert-storm guard.
    z_small = walk.robust_z(1800.0 * 1.01)
    assert abs(z_small) < 3.0, f"1% change scored |z|={abs(z_small):.2f}, too sensitive"

    # A halving is genuinely severe and must be flagged, finite, and clamped.
    z_big = walk.robust_z(900.0)
    assert math.isfinite(z_big)
    assert z_big < -3.0
    assert abs(z_big) <= Z_CLAMP

    # Pathological baseline (median 0, MAD 0) must still be finite and clamped.
    degenerate = BaselineStats(
        feature_name="x", median=0.0, mad=0.0, n_days=10,
        window_start=D0, window_end=D0 + timedelta(days=9),
    )
    z_path = degenerate.robust_z(500.0)
    assert math.isfinite(z_path) and abs(z_path) <= Z_CLAMP, f"got {z_path}"

    print(f"  T2 mad=0 -> eff_mad={walk.effective_mad():.1f}, "
          f"z(+1%)={z_small:+.2f}, z(halved)={z_big:+.2f}, "
          f"z(degenerate)={z_path:+.2f} (clamp={Z_CLAMP})")


def test_t3_today_excluded_from_own_baseline() -> None:
    model = BaselineModel(BehaviourConfig())
    for i in range(10):
        model.update(make_day(D0 + timedelta(days=i), walking_s=1800.0))
    today = D0 + timedelta(days=10)
    model.update(make_day(today, walking_s=0.0))
    stats = model.compute(before=today)
    assert abs(stats["walking_duration_s"].median - 1800.0) < 1e-6
    print("  T3 today excluded from its own baseline")


def test_t4_unreliable_days_excluded() -> None:
    """Camera outage days must not enter the baseline.

    Asserts the *property* (outage values absent from the median), not an exact n_days:
    with weekday/weekend stratification the count depends on which weekday `before`
    falls on, and pinning it would make this test fail for a reason unrelated to what
    it checks.
    """
    model = BaselineModel(BehaviourConfig())
    for i in range(20):
        model.update(make_day(D0 + timedelta(days=i), walking_s=1800.0))
    for i in range(20, 24):  # outage: low coverage, near-zero activity
        model.update(
            make_day(D0 + timedelta(days=i), walking_s=10.0,
                     observed_hours=2.0, coverage=0.1)
        )
    stats = model.compute(before=D0 + timedelta(days=24))
    s = stats["walking_duration_s"]
    assert abs(s.median - 1800.0) < 1e-6, f"outage days leaked into median: {s.median}"
    assert s.mad == 0.0, "outage days would have inflated MAD"
    assert s.n_days >= BehaviourConfig().min_baseline_days
    print(f"  T4 outage days excluded (median unchanged at 1800s, n_days={s.n_days})")


def test_t4b_weekend_stratified_baseline() -> None:
    """Weekends must be compared against weekend history, not pooled history.

    Measured on the simulator: pooled baselines produced 0.49 routine_deviation
    alerts/day on weekends vs 0.22 on weekdays (2.2x) for residents with NO injected
    anomaly, because weekend activity genuinely differs. That is a periodic confound,
    so a normal Saturday was being reported as a deviation.
    """
    cfg = BehaviourConfig(weekday_weekend_split=True)
    model = BaselineModel(cfg)
    # 8 weeks: weekdays 1800s, weekends 3600s. Uniform within each stratum.
    start = date(2026, 1, 5)  # a Monday
    for i in range(56):
        d = start + timedelta(days=i)
        model.update(make_day(d, walking_s=3600.0 if d.weekday() >= 5 else 1800.0))

    # Both queries must fall inside the generated range (ends 2026-03-01), otherwise
    # the stratum is empty for a reason unrelated to stratification.
    sat = date(2026, 2, 28)
    assert sat.weekday() == 5
    wknd = model.compute(before=sat)["walking_duration_s"]
    assert abs(wknd.median - 3600.0) < 1e-6, f"weekend baseline polluted: {wknd.median}"
    assert abs(wknd.robust_z(3600.0)) < 1e-9, "a typical Saturday must not deviate"

    wed = date(2026, 2, 25)
    assert wed.weekday() == 2
    week = model.compute(before=wed)["walking_duration_s"]
    assert abs(week.median - 1800.0) < 1e-6, f"weekday baseline polluted: {week.median}"
    assert abs(week.robust_z(1800.0)) < 1e-9, "a typical Wednesday must not deviate"

    # Pooled, the same Saturday looks like a large deviation - the bug being fixed.
    pooled = BaselineModel(BehaviourConfig(weekday_weekend_split=False))
    for i in range(56):
        d = start + timedelta(days=i)
        pooled.update(make_day(d, walking_s=3600.0 if d.weekday() >= 5 else 1800.0))
    pz = pooled.compute(before=sat)["walking_duration_s"].robust_z(3600.0)
    assert abs(pz) > 3.0, f"expected pooled baseline to misfire, got z={pz:.2f}"
    print(
        f"  T4b stratified: typical Saturday z=0.00 (weekend median 3600s); "
        f"pooled would give z={pz:.2f} -> false alert"
    )


def test_t4c_no_absence_alerts_from_camera_outage() -> None:
    """A partially-observed day must not produce absence-based alerts.

    Regression guard for the highest-volume false-positive source found by evaluation:
    outage days fired 14.0 alerts/day vs 0.33/day on observed days, and 15 such days
    generated 52% of all alerts across four no-anomaly residents. Zero observed meals
    during a 2.5-hour window is missing evidence, not a skipped meal.
    """
    analyzer = BehaviourAnalyzer()
    for i in range(20):
        analyzer.analyze_day(
            make_day(D0 + timedelta(days=i), walking_s=1800.0, meals=3,
                     meds=1, drinks=6, social_s=1200.0)
        )
    outage = make_day(
        D0 + timedelta(days=20), walking_s=0.0, meals=0, meds=0,
        drinks=0, social_s=0.0, observed_hours=2.5, coverage=0.2,
    )
    state = analyzer.analyze_day(outage)
    kinds = {a.kind.value for a in state.alerts}
    forbidden = {
        "meal_skipped", "medication_missed", "social_isolation",
        "mobility_decline", "sleep_disruption", "hydration_low", "routine_deviation",
    }
    assert not (kinds & forbidden), f"absence alerts fired on an outage day: {kinds}"

    # And the outage must not have poisoned CUSUM for the next observed day.
    nxt = analyzer.analyze_day(
        make_day(D0 + timedelta(days=21), walking_s=1800.0, meals=3,
                 meds=1, drinks=6, social_s=1200.0)
    )
    assert "mobility_decline" not in {a.kind.value for a in nxt.alerts}, (
        "outage-day z-score leaked into CUSUM and fired on the next observed day"
    )
    print("  T4c outage day -> 0 absence alerts, CUSUM state uncorrupted")


def test_t5_fall_alerts_without_baseline() -> None:
    """Safety rules must fire on day 1, before any baseline exists."""
    analyzer = BehaviourAnalyzer()
    state = analyzer.analyze_day(make_day(D0, falls=1, post_fall_s=300.0))
    assert state.baselines == {}, "expected no baseline on day 1"
    kinds = {a.kind.value for a in state.alerts}
    assert "fall_detected" in kinds
    assert "fall_no_recovery" in kinds
    sev = {a.severity.value for a in state.alerts}
    assert "emergency" in sev
    print(f"  T5 day-1 alerts fired: {sorted(kinds)}")


def test_t6_cusum_catches_gradual_decline() -> None:
    """The flagship claim, tested honestly.

    An earlier version of this test passed vacuously: the synthetic baseline had zero
    variance, so MAD was 0 and every deviation scored |z|>=6. The "CUSUM win" was
    really just a hair-trigger daily threshold, which is the alert-storm bug the
    mad_floor_frac guard now prevents.

    This version uses a NOISY baseline (sigma ~ 12% of median) and a slow decline
    (~1.2%/day). The decline is deliberately small relative to natural day-to-day
    variation, so daily thresholding should NOT fire early. We assert both halves of
    the claim:
        (a) CUSUM detects the decline, and
        (b) at detection time, no single day had breached |z| > 3.
    Without (b) the test proves nothing about CUSUM's added value.
    """
    rng = np.random.default_rng(20260803)
    analyzer = BehaviourAnalyzer()
    base_walk = 1800.0
    noise = 0.12

    def noisy(day: date, scale: float) -> DailyFeatures:
        jitter = float(rng.normal(1.0, noise))
        walk = max(60.0, base_walk * scale * jitter)
        return make_day(
            day,
            walking_s=walk,
            s2s=max(1, int(round(20 * scale * float(rng.normal(1.0, noise))))),
            transitions=max(1, int(round(30 * scale * float(rng.normal(1.0, noise))))),
        )

    for i in range(21):  # stable but noisy baseline
        analyzer.analyze_day(noisy(D0 + timedelta(days=i), scale=1.0))

    cusum_day: int | None = None
    z_breach_day: int | None = None
    scale = 1.0

    for i in range(21, 81):
        scale *= 0.988  # ~1.2% per day
        state = analyzer.analyze_day(noisy(D0 + timedelta(days=i), scale=scale))
        z = state.deviations.get("mobility_index", 0.0)

        if z_breach_day is None and z < -3.0:
            z_breach_day = i
        if cusum_day is None:
            for a in state.alerts:
                if a.kind.value == "mobility_decline":
                    cusum_day = i
                    drop = 100.0 * (1.0 - scale)
                    print(f"  T6 detected day {i} ({i - 21} days into decline, "
                          f"{drop:.1f}% cumulative), z={z:+.2f}, "
                          f"trigger='{a.rule_expression}'")
                    break
        if cusum_day is not None and z_breach_day is not None:
            break

    assert cusum_day is not None, "CUSUM failed to detect gradual decline at all"

    # (b) the discriminating assertion: daily thresholding must not have fired first.
    if z_breach_day is not None:
        assert cusum_day <= z_breach_day, (
            f"CUSUM ({cusum_day}) fired later than plain daily z ({z_breach_day}); "
            "CUSUM adds no value on this trajectory"
        )
        lead = z_breach_day - cusum_day
        print(f"  T6 daily |z|>3 first at day {z_breach_day} -> CUSUM lead = {lead} days")
    else:
        print("  T6 daily |z|>3 never fired in 60 days -> CUSUM is the ONLY detector")

    print(f"  T6 detection latency = {cusum_day - 21} days after decline onset")


def test_t6b_stable_resident_no_alert_storm() -> None:
    """Regression test for the alert-storm bug.

    A resident with a rigidly regular routine (medication 1/day, meals 3/day) produced
    MAD=0 features. The old fallback scored trivial deviations at |z|=6, firing alerts
    on noise for exactly the most stable users. Assert a quiet baseline stays quiet.
    """
    rng = np.random.default_rng(7)
    analyzer = BehaviourAnalyzer()

    for i in range(21):
        analyzer.analyze_day(make_day(D0 + timedelta(days=i)))  # perfectly constant

    total = 0
    for i in range(21, 35):
        # Normal day, tiny realistic jitter only.
        state = analyzer.analyze_day(
            make_day(
                D0 + timedelta(days=i),
                walking_s=1800.0 * float(rng.normal(1.0, 0.02)),
                meals=3,
                drinks=6,
                meds=1,
            )
        )
        total += len(state.alerts)

    assert total == 0, (
        f"alert storm: {total} alerts over 14 normal days for a stable resident"
    )
    print("  T6b 14 normal days for a rigidly-regular resident -> 0 alerts")


def test_t9_routine_deviation_requires_persistence() -> None:
    """A single atypical day must not raise a *routine* deviation.

    R10 tests ~11 features per day at |z|>3. On no-anomaly residents it produced 171 of
    192 residual alerts (29.9 per 100 resident-days), and 158 of those were isolated
    single days against only 9 that persisted - a multiple-comparisons artefact, made
    worse by heavy-tailed (lognormal) feature distributions. Requiring 2 consecutive
    days cut the suite's control false-alert rate from 33.57 to 5.59 per 100 days with
    no loss of recall.
    """
    cfg = BehaviourConfig()
    analyzer = BehaviourAnalyzer(cfg)
    for i in range(20):
        analyzer.analyze_day(make_day(D0 + timedelta(days=i), walking_s=1800.0))

    def routine(state) -> set[str]:
        return {
            a.rule_name.split(":", 1)[1]
            for a in state.alerts
            if a.kind.value == "routine_deviation"
        }

    # One atypical day: housework far above baseline. Must stay silent.
    spike = analyzer.analyze_day(
        make_day(D0 + timedelta(days=20), walking_s=1800.0, transitions=300)
    )
    assert "room_transitions" not in routine(spike), "single-day spike must not alert"

    # Back to normal - still silent, and the run is broken.
    normal = analyzer.analyze_day(make_day(D0 + timedelta(days=21), walking_s=1800.0))
    assert not routine(normal)

    # Two consecutive atypical days: now it is a routine change, so it must alert.
    analyzer.analyze_day(make_day(D0 + timedelta(days=22), walking_s=1800.0, transitions=300))
    second = analyzer.analyze_day(
        make_day(D0 + timedelta(days=23), walking_s=1800.0, transitions=300)
    )
    assert "room_transitions" in routine(second), (
        f"2 consecutive deviating days must alert, got {routine(second)}"
    )
    print("  T9 1 atypical day -> silent; 2 consecutive -> routine_deviation fires")


def test_t9b_routine_deviation_requires_consistent_direction() -> None:
    """A feature swinging +z then -z is erratic sensing, not a shifted routine."""
    cfg = BehaviourConfig()
    analyzer = BehaviourAnalyzer(cfg)
    for i in range(20):
        analyzer.analyze_day(make_day(D0 + timedelta(days=i), walking_s=1800.0))
    analyzer.analyze_day(make_day(D0 + timedelta(days=20), transitions=300))  # +z
    flip = analyzer.analyze_day(make_day(D0 + timedelta(days=21), transitions=0))  # -z
    names = {
        a.rule_name.split(":", 1)[1]
        for a in flip.alerts
        if a.kind.value == "routine_deviation"
    }
    assert "room_transitions" not in names, "sign-flipping deviation must not alert"
    print("  T9b +z then -z -> no routine_deviation (direction must be consistent)")


def test_t10_social_withdrawal_caught_by_drift() -> None:
    """Gradual social withdrawal must be detected even while visits continue.

    Same "boiled frog" structure as mobility decline: the rolling baseline follows the
    slow decay, so R8's daily-z test never fires. Measured recall for injected
    social_withdrawal was 50% with the daily rule alone and 100% once CUSUM was applied
    to social interaction. The CUSUM path must NOT require visitor_count == 0 - the
    clinically relevant signal is shrinking interaction duration during visits that
    still happen.

    Structured to avoid the trap this test fell into on first writing: a longer clean
    warm-up runs FIRST and must produce no alert, so a detection during the decline
    cannot be attributed to CUSUM having already drifted up on baseline noise. Without
    that control the test passed while "detecting" at 0 days into the decline, which
    proved nothing (the same failure mode as T6 before it was rewritten).
    """
    cfg = BehaviourConfig(social_drift_cusum=True)
    analyzer = BehaviourAnalyzer(cfg)
    rng = np.random.default_rng(7)
    base = 1200.0

    warmup = 40
    for i in range(warmup):
        state = analyzer.analyze_day(
            make_day(D0 + timedelta(days=i), social_s=base * float(rng.normal(1.0, 0.10)))
        )
        assert not any(a.kind.value == "social_isolation" for a in state.alerts), (
            f"false social_isolation on stable day {i}: CUSUM drifted on noise alone, "
            "so any later 'detection' would be meaningless"
        )

    social = base
    fired_day = None
    max_abs_z = 0.0
    for i in range(warmup, warmup + 80):
        social *= 0.985  # 1.5%/day decay - no single day is anomalous
        state = analyzer.analyze_day(
            make_day(D0 + timedelta(days=i), social_s=social * float(rng.normal(1.0, 0.10)))
        )
        z = state.deviations.get("social_interaction_duration_s", 0.0)
        max_abs_z = max(max_abs_z, abs(z))
        if fired_day is None and any(
            a.kind.value == "social_isolation" for a in state.alerts
        ):
            fired_day = i
            break

    assert fired_day is not None, "gradual social withdrawal was never detected"
    latency = fired_day - warmup
    assert latency >= 1, f"detected at {latency}d into decline - suspiciously immediate"
    cumulative = 100.0 * (1.0 - 0.985 ** latency)
    # visitor_count is 1 throughout (make_day sets it whenever social_s > 0), so this
    # detection cannot have come from the visitor_count == 0 conjunction.
    print(
        f"  T10 {warmup} stable days -> 0 alerts; withdrawal detected {latency} days "
        f"into decline ({cumulative:.1f}% cumulative), visits ongoing, "
        f"max daily |z| = {max_abs_z:.2f}"
    )


def test_t11_cusum_does_not_charge_on_noise() -> None:
    """CUSUM must not accumulate to threshold on a stable resident.

    This is the regression guard for the defect that made the first version of T10
    vacuous. A textbook one-sided CUSUM with k=0.5, h=5.0 false-alarms on 18.8% of
    200-day runs of pure N(0,1) noise - per tracked feature - so across ~5 drift-tracked
    features a spurious alert on a healthy resident was near-certain. Measured directly:
    the statistic reached 4.21 during a 20-day clean warm-up, then crossed on the first
    day of decline, producing a "detection" at 0 days' latency that proved nothing.

    Three changes fixed it: k 0.5 -> 0.75 (chosen on a measured signal/noise trade-off,
    see BehaviourConfig.cusum_k), accumulation gated on a settled baseline, and a ~2%/day
    decay so old charge does not persist indefinitely.
    """
    cfg = BehaviourConfig()
    analyzer = BehaviourAnalyzer(cfg)
    rng = np.random.default_rng(11)

    peaks: dict[str, float] = {}
    for i in range(200):
        state = analyzer.analyze_day(
            make_day(
                D0 + timedelta(days=i),
                walking_s=1800.0 * float(rng.normal(1.0, 0.12)),
                transitions=int(round(30 * float(rng.normal(1.0, 0.12)))),
                s2s=int(round(20 * float(rng.normal(1.0, 0.12)))),
                social_s=1200.0 * float(rng.normal(1.0, 0.12)),
            )
        )
        for name, value in state.drift_signals.items():
            peaks[name] = max(peaks.get(name, 0.0), value)
        drifty = [a for a in state.alerts if "CUSUM" in a.rule_expression]
        assert not drifty, (
            f"CUSUM false-alarmed on day {i} for a stable resident: "
            f"{[a.rule_name for a in drifty]}"
        )

    # The features that gate an alert (R7 mobility, R8 social) must stay under h.
    for name in ("mobility_index", "social_interaction_duration_s"):
        assert peaks.get(name, 0.0) < cfg.cusum_h, (
            f"{name} reached {peaks[name]:.2f} on noise alone (h={cfg.cusum_h}); "
            "its rule would false-alarm"
        )

    # Features tracked without a rule must also stay bounded. They are one code change
    # away from gating an alert, and a statistic that is already charged when the rule
    # is added would false-alarm on day one.
    worst = max(peaks.items(), key=lambda kv: kv[1])
    assert worst[1] < 2.0 * cfg.cusum_h, (
        f"{worst[0]} reached {worst[1]:.2f}, far above h={cfg.cusum_h}: decay is not "
        "bounding the statistic"
    )
    print(
        f"  T11 200 stable days, {len(peaks)} drift-tracked features -> 0 CUSUM alerts"
    )
    for name, value in sorted(peaks.items(), key=lambda kv: -kv[1]):
        gated = name in ("mobility_index", "social_interaction_duration_s")
        print(f"     peak {value:5.2f} / h={cfg.cusum_h}  {name}{'  [gates a rule]' if gated else ''}")


def test_t11b_cusum_warmup_and_decay() -> None:
    """The two structural CUSUM guards, tested directly rather than via the analyzer."""
    cfg = BehaviourConfig()

    # Warm-up: a large deviation while the baseline is unsettled must not accumulate.
    d = DriftDetector(cfg)
    for _ in range(5):
        out = d.update({"m": -4.0}, ("m",), baseline_days=3)
    assert out["m"] == 0.0, f"accumulated during warm-up: {out['m']}"

    # Past warm-up, the same deviation does accumulate.
    out = d.update({"m": -4.0}, ("m",), baseline_days=cfg.cusum_warmup_days)
    assert out["m"] > 0.0, "failed to accumulate with a settled baseline"

    # Decay: charge must fade once the deviations stop.
    charged = out["m"]
    for _ in range(60):
        out = d.update({"m": 0.0}, ("m",), baseline_days=30)
    assert out["m"] < charged, "statistic did not decay"
    print(
        f"  T11b warm-up held at 0.00; charged to {charged:.2f}; "
        f"decayed to {out['m']:.2f} after 60 quiet days"
    )


def test_t7_all_alerts_have_resolvable_evidence() -> None:
    """Auditability contract: every alert cites evidence resolvable in the state."""
    analyzer = BehaviourAnalyzer()
    for i in range(14):
        analyzer.analyze_day(make_day(D0 + timedelta(days=i)))

    state = analyzer.analyze_day(
        make_day(D0 + timedelta(days=14), walking_s=100.0, meals=0,
                 drinks=1, meds=0, social_s=0.0, falls=1, post_fall_s=200.0)
    )
    assert state.alerts, "expected alerts on a severely abnormal day"
    index = state.evidence_index()
    for alert in state.alerts:
        assert alert.evidence, f"{alert.kind} has no evidence"
        for ev in alert.evidence:
            assert ev.ref in index
            assert ev.ref.startswith("feat:")
            assert ev.ref.endswith(state.report_day.isoformat())
    print(f"  T7 {len(state.alerts)} alerts, "
          f"{len(index)} unique evidence refs, all resolvable")
    print(f"     kinds: {sorted({a.kind.value for a in state.alerts})}")


def test_t8_aggregation_from_segments() -> None:
    """Segment -> feature reduction: meal grouping and inactivity bridging."""
    base = datetime(2026, 1, 1, 8, 0, 0)

    def seg(sid: str, name: str, aid: int, start_min: float, dur_min: float,
            room: str = "kitchen", role: Role = Role.RESIDENT) -> ActivitySegment:
        return ActivitySegment(
            segment_id=sid, track_id=1, role=role, activity_id=aid,
            activity_name=name,
            start_time=base + timedelta(minutes=start_min),
            end_time=base + timedelta(minutes=start_min + dur_min),
            confidence=0.9, room=room,
        )

    segments = [
        seg("s1", "eating", 9, 0, 30, "kitchen"),      # breakfast
        seg("s2", "walking", 0, 30, 10, "hall"),
        seg("s3", "eating", 9, 240, 30, "kitchen"),    # lunch, >1h later
        seg("s4", "sitting", 2, 270, 60, "lounge"),
        seg("s5", "watching_tv", 13, 330, 90, "lounge"),  # contiguous sedentary
        seg("s6", "interacting_with_person", 18, 430, 20, "lounge"),
        seg("s7", "walking", 0, 450, 5, "kitchen", Role.VISITOR),
    ]

    f = aggregate_daily_features(segments, day=D0, subject_role=Role.RESIDENT)
    assert f.meal_events == 2, f"expected 2 meals, got {f.meal_events}"
    assert abs(f.eating_duration_s - 3600.0) < 1e-6
    assert abs(f.walking_duration_s - 600.0) < 1e-6, "visitor walking must not count"
    assert f.visitor_count == 1
    # sitting 60m + tv 90m contiguous -> 150m = 9000s
    assert abs(f.longest_inactive_block_s - 9000.0) < 1e-6, f.longest_inactive_block_s
    assert f.room_transitions >= 3
    print(f"  T8 meals={f.meal_events} walk={f.walking_duration_s:.0f}s "
          f"inactive={f.longest_inactive_block_s:.0f}s "
          f"transitions={f.room_transitions} mobility={f.mobility_index:.1f}")


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
