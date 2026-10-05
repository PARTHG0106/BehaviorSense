"""Longitudinal behaviour simulator with injected ground truth.

Why this exists
---------------
No public dataset contains multi-week, identity-resolved video of the same elderly
individual. The longest untrimmed ADL recording available is ~21 minutes (Toyota
Smarthome Untrimmed). Objectives like "detect mobility decline over weeks" therefore
cannot be supervised, validated, or even sanity-checked against real data.

Two dishonest options were rejected:
  1. Claim the behaviour layer works without evaluating it.
  2. Train a neural baseline model on simulation and validate it on the same simulation.

This module takes the third path: generate synthetic daily-feature sequences with
*known, injected* anomaly onsets, then measure whether Agent 3 recovers them. The
detector under test is statistical and fitted per-resident at runtime, so it never
"learns" the simulator's generative parameters - there is no train/test leakage to
worry about. What we measure is genuine: detection latency, precision, recall.

The simulator is a declared, versioned, released artifact. Its value is that it makes
routine-drift detection *reproducibly evaluable* by anyone, which the field currently
lacks.

Honest limits
-------------
Synthetic features are not real behaviour. Correlations between features are modelled
crudely (a mobility multiplier plus per-feature noise), day-of-week effects are simple
sinusoids, and no real pathology trajectory was fitted. Results here bound the
*algorithm's* sensitivity given clean features - they do not estimate clinical accuracy.
Reported numbers must be labelled as simulator results, never as clinical performance.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum

import numpy as np

from behaviorsense.schemas import DailyFeatures, Role

SIMULATOR_VERSION = "1.0.0"


class AnomalyKind(str, Enum):
    """Injectable anomaly patterns, each mapped to a plausible clinical scenario."""

    NONE = "none"
    GRADUAL_MOBILITY_DECLINE = "gradual_mobility_decline"
    """Slow multiplicative decay. Frailty progression, early dementia, arthritis."""

    ACUTE_MOBILITY_DROP = "acute_mobility_drop"
    """Step change. Post-fall fear of falling, acute illness, new medication."""

    MEAL_SKIPPING = "meal_skipping"
    """Declining meal frequency. Depression, dysphagia, cognitive decline."""

    SLEEP_DISRUPTION = "sleep_disruption"
    """Rising night-time activity / erratic lying duration. Sundowning, pain, delirium."""

    SOCIAL_WITHDRAWAL = "social_withdrawal"
    """Falling interaction and visitor counts. Depression, hearing loss."""

    MEDICATION_LAPSE = "medication_lapse"
    """Intermittent then absent medication events. Cognitive decline."""

    FALL_EVENT = "fall_event"
    """A single acute fall on one day, optionally with a long lie."""


@dataclass(frozen=True)
class InjectedAnomaly:
    """Ground truth for one injected anomaly.

    `onset_day` is the first day the underlying process changes - NOT the first day it
    is externally visible. Detection latency is measured from onset, which is the
    clinically meaningful reference point and a deliberately hard target.
    """

    kind: AnomalyKind
    onset_day: date
    end_day: date | None = None
    severity: float = 1.0
    ramp_days: int = 0
    """0 = step change; >0 = linear ramp to full severity over this many days."""

    notes: str = ""

    def is_active(self, day: date) -> bool:
        if day < self.onset_day:
            return False
        return self.end_day is None or day <= self.end_day

    def progress(self, day: date) -> float:
        """Fraction of full severity in effect on `day`, in [0, 1]."""
        if not self.is_active(day):
            return 0.0
        if self.ramp_days <= 0:
            return 1.0
        elapsed = (day - self.onset_day).days
        return min(1.0, (elapsed + 1) / self.ramp_days)


@dataclass
class ResidentProfile:
    """Baseline behavioural characteristics of one simulated resident.

    Four personas are provided (see `PERSONAS`) spanning the range that matters for
    detector evaluation: a highly regular resident (where MAD collapses toward zero and
    naive z-scores hair-trigger) through to a highly variable one (where real declines
    are easily masked by noise).
    """

    name: str
    walking_duration_s: float = 1800.0
    walking_bouts: int = 15
    room_transitions: int = 30
    sit_to_stand_count: int = 20
    sitting_duration_s: float = 14400.0
    lying_duration_s: float = 28800.0
    standing_duration_s: float = 3600.0
    meal_events: int = 3
    drinking_events: int = 6
    cooking_duration_s: float = 1800.0
    medication_events: int = 1
    tv_duration_s: float = 7200.0
    reading_duration_s: float = 1800.0
    phone_events: int = 2
    social_interaction_duration_s: float = 1200.0
    visitor_count: int = 1
    housework_duration_s: float = 1200.0

    noise_cv: float = 0.12
    """Coefficient of variation for day-to-day noise. 0.05 = rigid routine,
    0.25 = highly variable. The single most important knob for detector difficulty."""

    weekend_mobility_factor: float = 1.15
    """Weekends differ systematically. Included so the detector must tolerate a real
    periodic confound rather than i.i.d. noise."""

    dropout_prob: float = 0.02
    """Probability a day has degraded tracking (camera fault, occlusion). Such days are
    excluded from the baseline by DailyFeatures.is_reliable, which the detector relies
    on - so the simulator must actually produce them."""


PERSONAS: dict[str, ResidentProfile] = {
    "regular_margaret": ResidentProfile(
        name="regular_margaret",
        noise_cv=0.05,
        walking_duration_s=1500.0,
        meal_events=3,
        medication_events=2,
        social_interaction_duration_s=900.0,
    ),
    "active_george": ResidentProfile(
        name="active_george",
        noise_cv=0.15,
        walking_duration_s=3000.0,
        walking_bouts=25,
        room_transitions=45,
        sit_to_stand_count=32,
        housework_duration_s=2400.0,
        visitor_count=2,
    ),
    "frail_dorothy": ResidentProfile(
        name="frail_dorothy",
        noise_cv=0.10,
        walking_duration_s=700.0,
        walking_bouts=8,
        room_transitions=14,
        sit_to_stand_count=9,
        sitting_duration_s=21600.0,
        lying_duration_s=32400.0,
        tv_duration_s=10800.0,
        social_interaction_duration_s=600.0,
        visitor_count=1,
    ),
    "variable_arthur": ResidentProfile(
        name="variable_arthur",
        noise_cv=0.28,
        walking_duration_s=2100.0,
        meal_events=3,
        social_interaction_duration_s=1800.0,
        visitor_count=2,
        weekend_mobility_factor=1.35,
    ),
}


@dataclass(frozen=True)
class Scenario:
    """A complete, self-contained specification of one evaluation run.

    This is a dataclass rather than a tuple because an evaluation spec that can be
    partially applied is a trap: an earlier version of `standard_scenarios` accepted
    `n_days` and silently ignored it, so the caller's horizon and the scenario's
    intended horizon could disagree without any error. Bundling the whole spec means
    `scenario.run()` cannot be called with a mismatched horizon.
    """

    name: str
    profile: ResidentProfile
    anomalies: tuple[InjectedAnomaly, ...]
    seed: int
    start: date
    n_days: int

    @property
    def is_control(self) -> bool:
        return not self.anomalies

    @property
    def kind(self) -> AnomalyKind:
        """Primary injected anomaly kind, or NONE for a control."""
        return self.anomalies[0].kind if self.anomalies else AnomalyKind.NONE

    def run(self) -> SimulationResult:
        sim = BehaviourSimulator(self.profile, seed=self.seed)
        return sim.generate(
            start=self.start, n_days=self.n_days, anomalies=self.anomalies
        )


@dataclass
class SimulationResult:
    """Generated sequence plus the ground truth needed to score a detector."""

    profile_name: str
    days: list[DailyFeatures] = field(default_factory=list)
    anomalies: list[InjectedAnomaly] = field(default_factory=list)
    simulator_version: str = SIMULATOR_VERSION
    seed: int = 0

    @property
    def start_day(self) -> date:
        return self.days[0].day

    @property
    def end_day(self) -> date:
        return self.days[-1].day

    def onset_of(self, kind: AnomalyKind) -> date | None:
        for a in self.anomalies:
            if a.kind is kind:
                return a.onset_day
        return None

    def anomalous_days(self) -> set[date]:
        return {d.day for d in self.days if any(a.is_active(d.day) for a in self.anomalies)}

    def normal_days(self) -> set[date]:
        return {d.day for d in self.days} - self.anomalous_days()


class BehaviourSimulator:
    """Generates synthetic longitudinal daily features with injected anomalies.

    Usage::

        sim = BehaviourSimulator(PERSONAS["frail_dorothy"], seed=0)
        result = sim.generate(
            start=date(2026, 1, 1),
            n_days=120,
            anomalies=[InjectedAnomaly(
                kind=AnomalyKind.GRADUAL_MOBILITY_DECLINE,
                onset_day=date(2026, 2, 15),
                severity=0.5, ramp_days=45,
            )],
        )
    """

    def __init__(self, profile: ResidentProfile, seed: int = 0) -> None:
        self.profile = profile
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    # -- noise helpers ----------------------------------------------------------

    def _jitter(self, value: float, cv: float | None = None) -> float:
        """Multiplicative lognormal noise - keeps values positive and right-skewed,
        which matches real duration distributions better than additive Gaussian.
        """
        cv = self.profile.noise_cv if cv is None else cv
        if value <= 0:
            return 0.0
        sigma = np.sqrt(np.log(1.0 + cv**2))
        mu = -0.5 * sigma**2  # so E[multiplier] == 1
        return float(value * self._rng.lognormal(mu, sigma))

    def _jitter_count(self, value: float, cv: float | None = None) -> int:
        return max(0, int(round(self._jitter(value, cv))))

    def _weekly_factor(self, day: date) -> float:
        """Weekend uplift plus a mild weekly sinusoid."""
        base = self.profile.weekend_mobility_factor if day.weekday() >= 5 else 1.0
        return base * (1.0 + 0.04 * np.sin(2 * np.pi * day.weekday() / 7.0))

    # -- anomaly effects --------------------------------------------------------

    def _effects(
        self, day: date, anomalies: Sequence[InjectedAnomaly]
    ) -> dict[str, float]:
        """Multiplicative modifiers per feature group for `day`.

        Multiplicative rather than additive so effects compose sensibly and never drive
        a duration negative.
        """
        eff: dict[str, float] = {
            "mobility": 1.0,
            "meals": 1.0,
            "sleep": 1.0,
            "social": 1.0,
            "medication": 1.0,
            "sedentary": 1.0,
        }
        for a in anomalies:
            p = a.progress(day)
            if p <= 0.0:
                continue
            s = a.severity * p

            if a.kind is AnomalyKind.GRADUAL_MOBILITY_DECLINE:
                eff["mobility"] *= max(0.05, 1.0 - s)
                eff["sedentary"] *= 1.0 + 0.5 * s
            elif a.kind is AnomalyKind.ACUTE_MOBILITY_DROP:
                eff["mobility"] *= max(0.05, 1.0 - s)
                eff["sedentary"] *= 1.0 + 0.7 * s
            elif a.kind is AnomalyKind.MEAL_SKIPPING:
                eff["meals"] *= max(0.0, 1.0 - s)
            elif a.kind is AnomalyKind.SLEEP_DISRUPTION:
                # Erratic in both directions - insomnia and hypersomnia both occur.
                eff["sleep"] *= 1.0 + s * float(self._rng.choice([-0.45, 0.45]))
            elif a.kind is AnomalyKind.SOCIAL_WITHDRAWAL:
                eff["social"] *= max(0.0, 1.0 - s)
            elif a.kind is AnomalyKind.MEDICATION_LAPSE:
                # Intermittent at first, then absent - matches cognitive decline.
                eff["medication"] *= 0.0 if self._rng.random() < s else 1.0
        return eff

    # -- generation -------------------------------------------------------------

    def generate(
        self,
        start: date,
        n_days: int,
        anomalies: Sequence[InjectedAnomaly] = (),
        subject_role: Role = Role.RESIDENT,
    ) -> SimulationResult:
        p = self.profile
        out: list[DailyFeatures] = []

        for i in range(n_days):
            day = start + timedelta(days=i)
            eff = self._effects(day, anomalies)
            wk = self._weekly_factor(day)
            mob = eff["mobility"] * wk

            # Tracking dropout: a degraded day. Must be produced so the detector's
            # reliability gating is genuinely exercised.
            degraded = self._rng.random() < p.dropout_prob
            if degraded:
                observed_hours = float(self._rng.uniform(1.0, 7.0))
                coverage = float(self._rng.uniform(0.05, 0.5))
                visible = observed_hours / 24.0
            else:
                observed_hours = float(self._rng.uniform(22.5, 24.0))
                coverage = float(min(1.0, self._rng.normal(0.95, 0.03)))
                visible = 1.0

            walking = self._jitter(p.walking_duration_s * mob) * visible
            bouts = self._jitter_count(p.walking_bouts * mob) * (1 if not degraded else 0)

            fall_events = 0
            post_fall = 0.0
            for a in anomalies:
                if a.kind is AnomalyKind.FALL_EVENT and a.onset_day == day:
                    fall_events = 1
                    # severity >= 1 implies failure to self-recover (long lie).
                    post_fall = float(self._rng.uniform(120.0, 900.0)) if a.severity >= 1.0 else 0.0

            features = DailyFeatures(
                subject_role=subject_role,
                day=day,
                walking_duration_s=walking,
                walking_bouts=bouts,
                mean_bout_duration_s=(walking / bouts) if bouts else 0.0,
                room_transitions=self._jitter_count(p.room_transitions * mob) * (1 if not degraded else 0),
                sit_to_stand_count=self._jitter_count(p.sit_to_stand_count * mob) * (1 if not degraded else 0),
                mean_sit_to_stand_duration_s=self._jitter(2.5 / max(0.2, mob), 0.15),
                sitting_duration_s=self._jitter(p.sitting_duration_s * eff["sedentary"]) * visible,
                lying_duration_s=self._jitter(p.lying_duration_s * eff["sleep"]) * visible,
                standing_duration_s=self._jitter(p.standing_duration_s * mob) * visible,
                longest_inactive_block_s=self._jitter(
                    3600.0 * eff["sedentary"] / max(0.2, mob), 0.25
                ) * visible,
                meal_events=self._jitter_count(p.meal_events * eff["meals"], 0.08) if not degraded else 0,
                eating_duration_s=self._jitter(p.meal_events * 900.0 * eff["meals"]) * visible,
                drinking_events=self._jitter_count(p.drinking_events * eff["meals"], 0.15) if not degraded else 0,
                cooking_duration_s=self._jitter(p.cooking_duration_s * eff["meals"] * mob) * visible,
                medication_events=self._jitter_count(p.medication_events * eff["medication"], 0.02) if not degraded else 0,
                tv_duration_s=self._jitter(p.tv_duration_s * eff["sedentary"]) * visible,
                reading_duration_s=self._jitter(p.reading_duration_s) * visible,
                phone_events=self._jitter_count(p.phone_events * eff["social"], 0.2) if not degraded else 0,
                social_interaction_duration_s=self._jitter(
                    p.social_interaction_duration_s * eff["social"]
                ) * visible,
                visitor_count=self._jitter_count(p.visitor_count * eff["social"], 0.3) if not degraded else 0,
                housework_duration_s=self._jitter(p.housework_duration_s * mob) * visible,
                fall_events=fall_events,
                max_post_fall_immobility_s=post_fall,
                observed_hours=observed_hours,
                tracking_coverage=max(0.0, min(1.0, coverage)),
            )
            out.append(features)

        return SimulationResult(
            profile_name=p.name,
            days=out,
            anomalies=list(anomalies),
            seed=self.seed,
        )


# ---------------------------------------------------------------------------
# Scenario library - the standard evaluation suite
# ---------------------------------------------------------------------------


def standard_scenarios(
    start: date = date(2026, 1, 1),
    n_days: int = 150,
    baseline_days: int = 30,
) -> Iterator[Scenario]:
    """Yield the standard evaluation suite: 7 scenarios x 4 personas = 28 runs.

    Every scenario allows `baseline_days` of clean history before any onset, since the
    detector needs an established baseline. A scenario with an onset inside the warm-up
    period would measure warm-up behaviour, not detection.

    Coverage is deliberate: each anomaly kind appears against every persona, spanning
    noise_cv 0.05-0.28, because detector sensitivity depends far more on the resident's
    day-to-day variability than on the anomaly's label. Reporting a single pooled number
    across personas would hide exactly that effect, so the harness breaks results down
    by persona as well.

    `ramp_days` are expressed relative to the post-onset horizon so the suite stays
    coherent if `n_days` changes: a 60-day ramp inside a 40-day observation window would
    mean the anomaly never reaches full severity before the run ends.
    """
    if baseline_days >= n_days:
        raise ValueError(
            f"baseline_days ({baseline_days}) must be < n_days ({n_days}); "
            "otherwise no anomaly onset falls inside the observation window"
        )

    onset = start + timedelta(days=baseline_days)
    post_onset = n_days - baseline_days

    # Ramp lengths as a fraction of the post-onset horizon, capped so every anomaly
    # reaches full severity with time left to detect it.
    slow_ramp = max(7, int(post_onset * 0.50))
    med_ramp = max(5, int(post_onset * 0.25))
    fast_ramp = max(3, int(post_onset * 0.12))

    seed = 0
    for persona_name in (
        "regular_margaret", "frail_dorothy", "variable_arthur", "active_george",
    ):
        profile = PERSONAS[persona_name]

        specs: list[tuple[str, tuple[InjectedAnomaly, ...]]] = [
            # Slow decline - the flagship case. Hard: no single day is abnormal.
            (
                "gradual_decline",
                (InjectedAnomaly(
                    kind=AnomalyKind.GRADUAL_MOBILITY_DECLINE,
                    onset_day=onset, severity=0.55, ramp_days=slow_ramp,
                    notes="frailty progression",
                ),),
            ),
            # Acute step change - should be caught quickly by daily z-scores.
            (
                "acute_drop",
                (InjectedAnomaly(
                    kind=AnomalyKind.ACUTE_MOBILITY_DROP,
                    onset_day=onset, severity=0.45, ramp_days=0,
                    notes="post-fall fear of falling",
                ),),
            ),
            (
                "meal_skipping",
                (InjectedAnomaly(
                    kind=AnomalyKind.MEAL_SKIPPING,
                    onset_day=onset, severity=0.5, ramp_days=med_ramp,
                    notes="depression / appetite loss",
                ),),
            ),
            (
                "social_withdrawal",
                (InjectedAnomaly(
                    kind=AnomalyKind.SOCIAL_WITHDRAWAL,
                    onset_day=onset, severity=0.7, ramp_days=med_ramp,
                ),),
            ),
            (
                "medication_lapse",
                (InjectedAnomaly(
                    kind=AnomalyKind.MEDICATION_LAPSE,
                    onset_day=onset, severity=0.8, ramp_days=fast_ramp,
                ),),
            ),
            (
                "fall_with_long_lie",
                (InjectedAnomaly(
                    kind=AnomalyKind.FALL_EVENT,
                    onset_day=onset, end_day=onset, severity=1.0,
                    notes="fall with failure to self-recover",
                ),),
            ),
            # Control: no anomaly at all. Any alert here is a false positive. Without
            # controls, a detector that always fires would score perfect recall.
            ("control_no_anomaly", ()),
        ]

        for suffix, anomalies in specs:
            yield Scenario(
                name=f"{persona_name}__{suffix}",
                profile=profile,
                anomalies=anomalies,
                seed=seed,
                start=start,
                n_days=n_days,
            )
            seed += 1


__all__ = [
    "SIMULATOR_VERSION",
    "AnomalyKind",
    "InjectedAnomaly",
    "ResidentProfile",
    "PERSONAS",
    "Scenario",
    "SimulationResult",
    "BehaviourSimulator",
    "standard_scenarios",
]
