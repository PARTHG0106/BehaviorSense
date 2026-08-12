"""Agent 3 - Behaviour analysis.

Design decision, stated plainly: this layer is statistical, not learned.

No public dataset contains multi-week identity-resolved elderly home video, so a neural
baseline model would have to be trained on simulated data and validated on simulated
data. That is circular and would not survive review. Robust statistics give us:

  - falsifiable behaviour  (every number is hand-checkable)
  - auditable alerts       (rule_expression is printed in the alert)
  - no training data need  (works from day 1 of deployment)
  - clinical legibility    (a nurse can read median/MAD; they cannot read a GCN)

References for the specific estimators used here:
  - Median/MAD robust scaling: Leys et al. (2013), "Detecting outliers: do not use
    standard deviation around the mean". 50% breakdown point vs 0% for mean/std.
  - CUSUM change detection: Page (1954). Standard tool for detecting small persistent
    shifts, which is exactly what gradual mobility decline is.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from dataclasses import dataclass

import numpy as np

from behaviorsense.schemas import (
    ActivitySegment,
    Alert,
    AlertKind,
    AlertSeverity,
    BaselineStats,
    BehaviourState,
    DailyFeatures,
    Evidence,
    Role,
)

# Features the baseline model tracks. Restricted to those a rule consumes - tracking
# unused features wastes memory and inflates multiple-comparison risk.
TRACKED_FEATURES: tuple[str, ...] = (
    "walking_duration_s",
    "walking_bouts",
    "mean_bout_duration_s",
    "room_transitions",
    "sit_to_stand_count",
    "mean_sit_to_stand_duration_s",
    "sitting_duration_s",
    "lying_duration_s",
    "longest_inactive_block_s",
    "meal_events",
    "eating_duration_s",
    "drinking_events",
    "medication_events",
    "tv_duration_s",
    "social_interaction_duration_s",
    "housework_duration_s",
    "mobility_index",
    "sedentary_ratio",
)

# Activity id -> name, mirroring configs/taxonomy.yaml. Loaded from YAML at runtime in
# the real pipeline; duplicated here so this module is unit-testable standalone.
ACTIVITY_NAMES: dict[int, str] = {
    0: "walking", 1: "standing", 2: "sitting", 3: "lying_down",
    4: "standing_up", 5: "sitting_down", 6: "bending_reaching", 7: "falling",
    8: "fallen_on_ground", 9: "eating", 10: "drinking", 11: "cooking_food_prep",
    12: "taking_medication", 13: "watching_tv", 14: "reading", 15: "using_phone",
    16: "cleaning_housework", 17: "personal_hygiene", 18: "interacting_with_person",
    19: "other_idle",
}


@dataclass(frozen=True)
class BehaviourConfig:
    """Thresholds for the behaviour layer.

    Defaults are justified inline. They are config, not constants, because the right
    operating point is deployment-specific and must be tunable without code edits.
    """

    baseline_window_days: int = 14
    """14 days balances two failure modes: too short (a single odd day dominates the
    median) vs too long (genuine gradual decline gets absorbed into the baseline and
    becomes invisible). Ablated in the evaluation suite over 7/14/30."""

    min_baseline_days: int = 7
    """Below 7 days a median is not meaningful; we emit no deviation alerts and report
    'baseline establishing' rather than guessing."""

    deviation_z_threshold: float = 3.0
    """|robust z| > 3 flags a day as anomalous. With ~18 features this yields a
    non-trivial family-wise error rate, which is why single-day deviations are only
    ADVISORY and sustained multi-day deviations are required for URGENT."""

    sustained_days: int = 3
    """Consecutive days of deviation required to escalate. Cuts false positives from
    isolated atypical days (visitors, appointments) by roughly an order of magnitude."""

    cusum_k: float = 0.75
    """CUSUM slack in robust-z units. Shifts smaller than this are ignored as noise.

    Raised from 0.5 after simulation. A one-sided CUSUM with k=0.5, h=5 false-alarms on
    18.8% of 200-day runs of pure N(0,1) noise - per feature. Across ~5 drift-tracked
    features that is a near-certain spurious alert on every resident, which is what a
    control test caught (CUSUM reached 4.21 on baseline noise before any decline began,
    then "detected" on the first decline day - a meaningless detection).

    k was chosen against measured signal/noise trade-off, not by taste:

        k     h    FA(noise)   detect d=0.3   d=0.5    d=0.8
        0.50  5.0    18.8%      58d / 86%    27d/100%  12d/100%
        0.75  5.0     1.9%      95d / 26%    70d/ 76%  23d/100%
        1.00  5.0     0.2%      99d /  3%    95d/ 20%  63d/ 87%

    k=1.0 is rejected despite the best false-alarm rate: it detects a moderate (d=0.5)
    decline only 20% of the time. For a system whose purpose is catching gradual decline,
    missing 4 in 5 real declines is a worse failure than an occasional advisory."""

    cusum_h: float = 5.0
    """CUSUM decision interval. Higher = fewer false alarms, longer detection delay."""

    cusum_warmup_days: int = 7
    """Days of established baseline required before CUSUM starts accumulating.

    During warm-up the baseline median/MAD are themselves unstable, so early z-scores
    are dominated by estimation error rather than behaviour. Banking that error as drift
    is how the statistic reached 4.21 before any real change occurred."""

    cusum_decay: float = 0.98
    """Per-day multiplicative decay applied to the CUSUM statistic.

    A textbook CUSUM never forgets: charge accumulated from a fluctuation weeks ago sits
    at the same level forever, so the detector's effective threshold silently drops over
    time and the first modest dip months later trips it. A ~2%/day decay gives the
    statistic a ~35-day memory, which matches the clinical window of interest (weeks),
    while leaving a genuine sustained decline - which adds charge every day - unaffected."""

    inactivity_alert_s: float = 4 * 3600.0
    """4h of continuous immobility during waking hours. Below this, normal napping and
    extended TV viewing dominate and the alert becomes noise."""

    suppress_absence_rules_on_unreliable_days: bool = True
    """Do not infer *absence* of an activity from a day with degraded observation.

    Found by evaluation, not by inspection. Camera-outage days were excluded from the
    baseline but still analysed, so a day with 2.5 observed hours reported zero meals,
    zero medication and zero visitors - and fired meal_skipped + medication_missed +
    social_isolation + mobility_decline + sleep_disruption simultaneously. Measured
    alert load was 14.0 alerts/day on outage days vs 0.33/day on observed days: 15 days
    produced 52% of all alerts on residents with no injected anomaly.

    "I did not see it" is not "it did not happen". Rules that trigger on a LOW or ZERO
    count are unsound on partial observation and are suppressed; safety rules that
    trigger on a POSITIVE observation (a fall that was actually seen) stay active,
    because a fall observed during a 2-hour window is still a real fall."""

    min_observed_hours_for_absence: float = 8.0
    """Observation hours below which absence-based rules are suppressed. Matches the
    default in DailyFeatures.is_reliable so baseline admission and rule gating agree."""

    weekday_weekend_split: bool = True
    """Compare weekends against weekend history and weekdays against weekday history.

    Also found by evaluation: weekend routine_deviation load was 0.49 alerts/day vs
    0.22 on weekdays (2.2x) for residents with no anomaly. Weekend behaviour genuinely
    differs - visitors, outings, different meal times - so a pooled baseline treats a
    normal Saturday as a deviation. This is a periodic confound, not noise, and
    stratifying removes it at the cost of needing ~2x the calendar time to fill each
    stratum (mitigated by falling back to the pooled window when a stratum is short)."""

    post_fall_immobility_s: float = 60.0
    """1 minute immobile after a detected fall => cannot self-recover => EMERGENCY.
    Long-lie is the primary predictor of poor outcome after a fall."""

    min_meals_per_day: int = 2
    hydration_min_events: int = 4
    social_isolation_days: int = 3

    routine_deviation_sustained_days: int = 2
    """Consecutive days a generic feature must deviate before R10 emits.

    Set from measurement. R10 tests ~11 features per day at |z|>3; with a heavy-tailed
    (lognormal) feature distribution the nominal 3-sigma rate badly understates the true
    family-wise rate, and R10 produced 171 of 192 residual alerts on no-anomaly
    residents - 29.9 per 100 resident-days. The distribution of those alerts is the
    giveaway: 158 were isolated single days and only 9 persisted into a second day.
    A single atypical day is an appointment or a visitor; a *routine* deviation, which is
    what this rule claims to detect, lasts longer than one day. Requiring 2 consecutive
    days matches the rule's own semantics and suppresses the isolated-blip mass. Rules
    R1-R3 are unaffected: a fall must alert the same day."""

    social_drift_cusum: bool = True
    """Apply CUSUM drift detection to social interaction, not just mobility.

    Gradual social withdrawal has the same "boiled frog" structure as gradual mobility
    decline: the rolling baseline follows the slow decay, so no single day is ever
    anomalous and R8's daily-z test never fires. Measured recall for injected
    social_withdrawal was 50% while gradual mobility decline - which has CUSUM - was
    100%. The asymmetry was in the detector, not the phenomenon."""


class BaselineModel:
    """Rolling robust baseline per feature.

    Only reliable days (sufficient observed hours and tracking coverage) enter the
    baseline. This matters more than it looks: a camera outage produces a day of
    near-zero activity, and admitting it would drag the median down and mask the real
    decline the system exists to catch.
    """

    def __init__(self, config: BehaviourConfig | None = None) -> None:
        self.config = config or BehaviourConfig()
        self._history: list[DailyFeatures] = []

    def update(self, day_features: DailyFeatures) -> None:
        self._history.append(day_features)
        self._history.sort(key=lambda d: d.day)
        # Retain enough for the widest lookback any stratum uses (weekends look back
        # 3.5x the window), plus runway for drift detection.
        keep = int(self.config.baseline_window_days * 3.5) + self.config.baseline_window_days
        if len(self._history) > keep:
            self._history = self._history[-keep:]

    def _window(self, before: date) -> list[DailyFeatures]:
        """Reliable days strictly before `before`, within the baseline window.

        Strictly-before is deliberate: including today's value in today's baseline
        shrinks the very deviation we are trying to measure.

        When `weekday_weekend_split` is set, the window is restricted to days of the
        same type (weekend vs weekday) as `before`. The lookback is widened to keep the
        stratum populated - a 14-day window contains only 4 weekend days, which is below
        min_baseline_days and would otherwise suppress every weekend alert entirely.
        """
        cfg = self.config
        if not cfg.weekday_weekend_split:
            start = before - timedelta(days=cfg.baseline_window_days)
            return [d for d in self._history if start <= d.day < before and d.is_reliable()]

        is_weekend = before.weekday() >= 5
        # 7/2 and 7/5 are the calendar-days-per-stratum-day ratios.
        span = int(cfg.baseline_window_days * (7 / 2 if is_weekend else 7 / 5))
        start = before - timedelta(days=span)
        stratum = [
            d for d in self._history
            if start <= d.day < before
            and d.is_reliable()
            and (d.day.weekday() >= 5) == is_weekend
        ]
        if len(stratum) >= cfg.min_baseline_days:
            return stratum

        # Stratum too thin (early deployment). Fall back to the pooled window rather
        # than emitting no baseline at all: a slightly confounded baseline still
        # catches gross deviation, whereas no baseline catches nothing.
        start = before - timedelta(days=cfg.baseline_window_days)
        return [d for d in self._history if start <= d.day < before and d.is_reliable()]

    def compute(self, before: date) -> dict[str, BaselineStats]:
        window = self._window(before)
        if len(window) < self.config.min_baseline_days:
            return {}

        matrices = defaultdict(list)
        for day in window:
            items = day.numeric_items()
            for feature in TRACKED_FEATURES:
                if feature in items:
                    matrices[feature].append(items[feature])

        stats: dict[str, BaselineStats] = {}
        for feature, values in matrices.items():
            arr = np.asarray(values, dtype=np.float64)
            median = float(np.median(arr))
            mad = float(np.median(np.abs(arr - median)))
            stats[feature] = BaselineStats(
                feature_name=feature,
                median=median,
                mad=mad,
                n_days=len(arr),
                window_start=window[0].day,
                window_end=window[-1].day,
            )
        return stats

    @property
    def n_days(self) -> int:
        return len(self._history)

    def history(self) -> list[DailyFeatures]:
        return list(self._history)


class DriftDetector:
    """One-sided CUSUM per feature, for gradual decline.

    Deviation z-scores catch *sudden* changes. They systematically miss the clinically
    important case: mobility falling 2% per day for a month. Each day is within normal
    range, yet the cumulative change is severe. CUSUM accumulates small signed
    deviations so persistent drift crosses the threshold even when no single day does.

    We track only the decline direction for mobility-type features - a resident walking
    *more* is not a concern worth alerting on.

    Two departures from the textbook one-sided CUSUM, both added after measurement:

    1. Accumulation is gated on a settled baseline (`cusum_warmup_days`). Early z-scores
       reflect an unstable median/MAD more than real behaviour.
    2. The statistic decays (`cusum_decay`). A never-forgetting statistic lets charge from
       an old fluctuation persist, so the effective threshold erodes over months.
    """

    def __init__(self, config: BehaviourConfig | None = None) -> None:
        self.config = config or BehaviourConfig()
        self._neg: dict[str, float] = defaultdict(float)
        self._observations: int = 0

    def update(
        self,
        deviations: dict[str, float],
        features: Sequence[str] | None = None,
        baseline_days: int | None = None,
    ) -> dict[str, float]:
        """Accumulate one day of deviations.

        `baseline_days` is the number of days backing today's baseline. When it is below
        `cusum_warmup_days` the statistic is held at zero rather than accumulating
        estimation noise.
        """
        cfg = self.config
        targets = features or tuple(deviations)
        self._observations += 1

        warm = baseline_days is None or baseline_days >= cfg.cusum_warmup_days
        out: dict[str, float] = {}
        for name in targets:
            z = deviations.get(name)
            if z is None:
                continue
            if not warm:
                self._neg[name] = 0.0
                out[name] = 0.0
                continue
            # Decay first, then accumulate downward excursions beyond the slack k.
            decayed = self._neg[name] * cfg.cusum_decay
            self._neg[name] = max(0.0, decayed - z - cfg.cusum_k)
            out[name] = self._neg[name]
        return out

    def triggered(self, name: str) -> bool:
        return self._neg[name] >= self.config.cusum_h

    def value(self, name: str) -> float:
        """Current statistic without advancing it. Used to report drift on days whose
        observations are too partial to update state from."""
        return self._neg[name]

    def reset(self, name: str) -> None:
        self._neg[name] = 0.0


def aggregate_daily_features(
    segments: Sequence[ActivitySegment],
    day: date,
    subject_role: Role = Role.RESIDENT,
    observed_hours: float = 24.0,
    tracking_coverage: float = 1.0,
) -> DailyFeatures:
    """Reduce one day of activity segments to a DailyFeatures vector.

    Only the subject's segments contribute to personal features; other people's
    segments contribute solely to social-interaction and visitor counts.
    """
    subject = [s for s in segments if s.role is subject_role]
    others = [s for s in segments if s.role is not subject_role]

    def total_duration(activity: str) -> float:
        return sum(s.duration_s for s in subject if s.activity_name == activity)

    def count(activity: str) -> int:
        return sum(1 for s in subject if s.activity_name == activity)

    walking = [s for s in subject if s.activity_name == "walking"]
    walking_total = sum(s.duration_s for s in walking)

    sit_to_stand = [s for s in subject if s.activity_name == "standing_up"]
    s2s_mean = (
        sum(s.duration_s for s in sit_to_stand) / len(sit_to_stand)
        if sit_to_stand else 0.0
    )

    # Room transitions: count changes in the room label along the time-ordered timeline.
    ordered = sorted(subject, key=lambda s: s.start_time)
    transitions = sum(
        1
        for prev, curr in zip(ordered, ordered[1:])
        if prev.room and curr.room and prev.room != curr.room
    )

    # Longest inactive block: contiguous run of sedentary states, bridging small gaps.
    sedentary = {"sitting", "lying_down", "watching_tv", "reading", "other_idle"}
    longest = current = 0.0
    prev_end: datetime | None = None
    for seg in ordered:
        if seg.activity_name in sedentary:
            gap = (seg.start_time - prev_end).total_seconds() if prev_end else 0.0
            # Bridge sub-minute gaps: brief detection dropouts should not reset the run.
            current = seg.duration_s if gap > 60.0 else current + seg.duration_s
            longest = max(longest, current)
        else:
            current = 0.0
        prev_end = seg.end_time

    # Meals: eating segments separated by >1h are distinct meals, not one long meal.
    eating = sorted(
        (s for s in subject if s.activity_name == "eating"), key=lambda s: s.start_time
    )
    meals = 0
    last_end: datetime | None = None
    for seg in eating:
        if last_end is None or (seg.start_time - last_end).total_seconds() > 3600.0:
            meals += 1
        last_end = seg.end_time

    falls = [s for s in subject if s.activity_name == "falling"]
    fallen = [s for s in subject if s.activity_name == "fallen_on_ground"]
    max_post_fall = max((s.duration_s for s in fallen), default=0.0)

    visitor_ids = {s.track_id for s in others if s.role in (Role.VISITOR, Role.CARER)}

    return DailyFeatures(
        subject_role=subject_role,
        day=day,
        walking_duration_s=walking_total,
        walking_bouts=len(walking),
        mean_bout_duration_s=walking_total / len(walking) if walking else 0.0,
        room_transitions=transitions,
        sit_to_stand_count=len(sit_to_stand),
        mean_sit_to_stand_duration_s=s2s_mean,
        sitting_duration_s=total_duration("sitting"),
        lying_duration_s=total_duration("lying_down"),
        standing_duration_s=total_duration("standing"),
        longest_inactive_block_s=longest,
        meal_events=meals,
        eating_duration_s=total_duration("eating"),
        drinking_events=count("drinking"),
        cooking_duration_s=total_duration("cooking_food_prep"),
        medication_events=count("taking_medication"),
        tv_duration_s=total_duration("watching_tv"),
        reading_duration_s=total_duration("reading"),
        phone_events=count("using_phone"),
        social_interaction_duration_s=total_duration("interacting_with_person"),
        visitor_count=len(visitor_ids),
        housework_duration_s=total_duration("cleaning_housework"),
        fall_events=len(falls),
        max_post_fall_immobility_s=max_post_fall,
        observed_hours=observed_hours,
        tracking_coverage=tracking_coverage,
    )


class BehaviourAnalyzer:
    """Agent 3 top level: features -> baseline -> deviation -> drift -> alerts."""

    def __init__(self, config: BehaviourConfig | None = None) -> None:
        self.config = config or BehaviourConfig()
        self.baseline = BaselineModel(self.config)
        self.drift = DriftDetector(self.config)
        self._recent_deviations: list[dict[str, float]] = []

    def analyze_day(
        self, day_features: DailyFeatures, subject_role: Role = Role.RESIDENT
    ) -> BehaviourState:
        cfg = self.config
        baselines = self.baseline.compute(before=day_features.day)
        items = day_features.numeric_items()

        deviations = {
            name: stats.robust_z(items[name])
            for name, stats in baselines.items()
            if name in items
        }

        mobility_features = (
            "walking_duration_s", "room_transitions",
            "sit_to_stand_count", "mobility_index",
        )
        drift_features = mobility_features + (
            ("social_interaction_duration_s",) if cfg.social_drift_cusum else ()
        )

        # Detector state must not accumulate from unobserved time. An outage day drives
        # mobility_index to ~0, giving z at the clamp (-10), which alone pushes CUSUM
        # from 0 past h=5 in a single step. Suppressing the *alert* on that day is not
        # enough: the corrupted statistic persists and fires on the next observed day,
        # producing a mobility_decline alert caused entirely by a camera fault. Skipping
        # the update is the honest treatment - a missing day is missing evidence, not
        # evidence of decline.
        observable = (
            not cfg.suppress_absence_rules_on_unreliable_days
            or day_features.observed_hours >= cfg.min_observed_hours_for_absence
        )
        if observable:
            n_baseline = max(
                (s.n_days for s in baselines.values()), default=0
            )
            drift_signals = self.drift.update(
                deviations, drift_features, baseline_days=n_baseline
            )
            self._recent_deviations.append(deviations)
            keep = max(cfg.sustained_days, cfg.routine_deviation_sustained_days)
            if len(self._recent_deviations) > keep:
                self._recent_deviations.pop(0)
        else:
            drift_signals = {
                name: self.drift.value(name) for name in drift_features
            }

        alerts = self._evaluate_rules(day_features, baselines, deviations, drift_signals)

        # Record today only after analysis, so it never contaminates its own baseline.
        self.baseline.update(day_features)

        return BehaviourState(
            subject_role=subject_role,
            report_day=day_features.day,
            today=day_features,
            baselines=baselines,
            deviations=deviations,
            drift_signals=drift_signals,
            alerts=alerts,
            history_days=self.baseline.n_days,
        )

    # -- rules ------------------------------------------------------------------

    def _evaluate_rules(
        self,
        day: DailyFeatures,
        baselines: dict[str, BaselineStats],
        deviations: dict[str, float],
        drift: dict[str, float],
    ) -> list[Alert]:
        cfg = self.config
        alerts: list[Alert] = []
        now = datetime.combine(day.day, datetime.min.time())
        items = day.numeric_items()

        def evidence_for(feature: str) -> list[Evidence]:
            """Build evidence, synthesising a degenerate baseline if none exists yet.

            Critical rules (falls) must fire on day 1, before any baseline exists.
            A safety alert that waits a week for statistics is useless.
            """
            if feature in baselines:
                return [Evidence.build(feature, items[feature], baselines[feature], day.day)]
            placeholder = BaselineStats(
                feature_name=feature,
                median=0.0,
                mad=0.0,
                n_days=1,
                window_start=day.day,
                window_end=day.day,
            )
            return [Evidence.build(feature, items.get(feature, 0.0), placeholder, day.day)]

        def emit(
            kind: AlertKind,
            severity: AlertSeverity,
            rule_name: str,
            expression: str,
            evidence: list[Evidence],
        ) -> None:
            alerts.append(
                Alert(
                    alert_id=str(uuid.uuid4()),
                    kind=kind,
                    severity=severity,
                    subject_role=day.subject_role,
                    day=day.day,
                    triggered_at=now,
                    rule_name=rule_name,
                    rule_expression=expression,
                    evidence=evidence,
                )
            )

        # R1 - Fall. Baseline-independent, fires immediately.
        if day.fall_events > 0:
            emit(
                AlertKind.FALL_DETECTED,
                AlertSeverity.URGENT,
                "fall_detected",
                "fall_events > 0",
                evidence_for("fall_events"),
            )

        # R2 - Fall without recovery. Highest severity in the system.
        if day.max_post_fall_immobility_s >= cfg.post_fall_immobility_s:
            emit(
                AlertKind.FALL_NO_RECOVERY,
                AlertSeverity.EMERGENCY,
                "fall_no_recovery",
                f"max_post_fall_immobility_s >= {cfg.post_fall_immobility_s:.0f}",
                evidence_for("max_post_fall_immobility_s"),
            )

        # R3 - Prolonged inactivity.
        if day.longest_inactive_block_s >= cfg.inactivity_alert_s:
            emit(
                AlertKind.PROLONGED_INACTIVITY,
                AlertSeverity.URGENT,
                "prolonged_inactivity",
                f"longest_inactive_block_s >= {cfg.inactivity_alert_s:.0f}",
                evidence_for("longest_inactive_block_s"),
            )

        # Everything below infers a problem from a LOW or ABSENT count. That inference
        # is invalid when the day was only partially observed - see
        # BehaviourConfig.suppress_absence_rules_on_unreliable_days. R1-R3 above are
        # exempt because they trigger on something positively observed.
        if (
            cfg.suppress_absence_rules_on_unreliable_days
            and day.observed_hours < cfg.min_observed_hours_for_absence
        ):
            return alerts

        # Rules below need a baseline. Suppress rather than guess.
        if not baselines:
            return alerts

        # R4 - Meal skipping. Absolute floor OR a sustained drop vs personal norm.
        meal_z = deviations.get("meal_events", 0.0)
        if day.meal_events < cfg.min_meals_per_day or meal_z < -cfg.deviation_z_threshold:
            emit(
                AlertKind.MEAL_SKIPPED,
                AlertSeverity.ADVISORY,
                "meal_skipped",
                f"meal_events < {cfg.min_meals_per_day} or "
                f"robust_z(meal_events) < -{cfg.deviation_z_threshold}",
                evidence_for("meal_events"),
            )

        # R5 - Low hydration.
        if day.drinking_events < cfg.hydration_min_events:
            emit(
                AlertKind.HYDRATION_LOW,
                AlertSeverity.ADVISORY,
                "hydration_low",
                f"drinking_events < {cfg.hydration_min_events}",
                evidence_for("drinking_events"),
            )

        # R6 - Missed medication, only if a baseline habit exists to deviate from.
        med_baseline = baselines.get("medication_events")
        if med_baseline and med_baseline.median >= 1.0 and day.medication_events == 0:
            emit(
                AlertKind.MEDICATION_MISSED,
                AlertSeverity.URGENT,
                "medication_missed",
                "medication_events == 0 while baseline median >= 1",
                evidence_for("medication_events"),
            )

        # R7 - Mobility decline. Sustained deviation OR CUSUM drift.
        # This is the flagship rule: it is what "detects decline before it is obvious".
        sustained = self._sustained_decline("mobility_index")
        cusum_hit = drift.get("mobility_index", 0.0) >= cfg.cusum_h
        if sustained or cusum_hit:
            reason = (
                f"robust_z(mobility_index) < -{cfg.deviation_z_threshold} "
                f"for {cfg.sustained_days} consecutive days"
                if sustained
                else f"CUSUM(mobility_index) >= {cfg.cusum_h}"
            )
            ev = evidence_for("mobility_index")
            for extra in ("walking_duration_s", "sit_to_stand_count", "room_transitions"):
                if extra in baselines:
                    ev.extend(evidence_for(extra))
            emit(
                AlertKind.MOBILITY_DECLINE,
                AlertSeverity.URGENT,
                "mobility_decline",
                reason,
                ev,
            )
            if cusum_hit:
                # Reset so one sustained decline does not re-alert every day forever.
                self.drift.reset("mobility_index")

        # R8 - Social isolation. Daily conjunction OR gradual-withdrawal drift.
        social_z = deviations.get("social_interaction_duration_s", 0.0)
        social_cusum = drift.get("social_interaction_duration_s", 0.0) >= cfg.cusum_h
        if (social_z < -cfg.deviation_z_threshold and day.visitor_count == 0) or social_cusum:
            # The CUSUM path deliberately omits the visitor_count == 0 conjunction.
            # Gradual withdrawal shows up as steadily shrinking interaction *duration*
            # while visits still occur; requiring zero visitors as well is what held
            # measured recall for injected social_withdrawal at 50%.
            reason = (
                f"CUSUM(social_interaction_duration_s) >= {cfg.cusum_h}"
                if social_cusum
                else f"robust_z(social_interaction_duration_s) < "
                f"-{cfg.deviation_z_threshold} and visitor_count == 0"
            )
            emit(
                AlertKind.SOCIAL_ISOLATION,
                AlertSeverity.ADVISORY,
                "social_isolation",
                reason,
                evidence_for("social_interaction_duration_s"),
            )
            if social_cusum:
                self.drift.reset("social_interaction_duration_s")

        # R9 - Sleep disruption, in either direction (insomnia or hypersomnia).
        lying_z = deviations.get("lying_duration_s", 0.0)
        if abs(lying_z) > cfg.deviation_z_threshold:
            emit(
                AlertKind.SLEEP_DISRUPTION,
                AlertSeverity.ADVISORY,
                "sleep_disruption",
                f"|robust_z(lying_duration_s)| > {cfg.deviation_z_threshold}",
                evidence_for("lying_duration_s"),
            )

        # R10 - Generic routine deviation, for features without a dedicated rule.
        # Requires persistence: see BehaviourConfig.routine_deviation_sustained_days.
        handled = {
            "mobility_index", "meal_events", "drinking_events", "medication_events",
            "social_interaction_duration_s", "lying_duration_s", "fall_events",
            "longest_inactive_block_s", "max_post_fall_immobility_s",
        }
        for name, z in deviations.items():
            if name in handled or abs(z) <= cfg.deviation_z_threshold:
                continue
            if not self._sustained_deviation(name, cfg.routine_deviation_sustained_days):
                continue
            emit(
                AlertKind.ROUTINE_DEVIATION,
                AlertSeverity.INFO,
                f"routine_deviation:{name}",
                f"|robust_z({name})| > {cfg.deviation_z_threshold} "
                f"for {cfg.routine_deviation_sustained_days} consecutive days",
                evidence_for(name),
            )

        return alerts

    def _sustained_deviation(self, feature: str, n_days: int) -> bool:
        """True if `feature` exceeded the threshold, in the SAME direction, for `n_days`.

        Direction consistency matters: a feature that swings +4 then -4 is erratic
        sensing or an unusual pair of days, not a shifted routine, and reporting it as
        one would be wrong. `_recent_deviations` already contains today (it is appended
        before rules run), so the last `n_days` entries are the window.
        """
        cfg = self.config
        if n_days <= 1:
            return True
        if len(self._recent_deviations) < n_days:
            return False
        window = self._recent_deviations[-n_days:]
        zs = [d.get(feature) for d in window]
        if any(z is None for z in zs):
            return False
        thr = cfg.deviation_z_threshold
        return all(z > thr for z in zs) or all(z < -thr for z in zs)

    def _sustained_decline(self, feature: str) -> bool:
        """True if `feature` has been below -threshold for `sustained_days` in a row."""
        cfg = self.config
        if len(self._recent_deviations) < cfg.sustained_days:
            return False
        recent = self._recent_deviations[-cfg.sustained_days:]
        return all(d.get(feature, 0.0) < -cfg.deviation_z_threshold for d in recent)


__all__ = [
    "TRACKED_FEATURES",
    "ACTIVITY_NAMES",
    "BehaviourConfig",
    "BaselineModel",
    "DriftDetector",
    "BehaviourAnalyzer",
    "aggregate_daily_features",
]
