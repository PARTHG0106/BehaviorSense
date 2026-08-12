"""Evaluation harness for Agent 3 against simulator ground truth.

What this measures, precisely
-----------------------------
For each scenario the simulator injects an anomaly with a known `onset_day`. We replay
the generated days through `BehaviourAnalyzer` one at a time - exactly as the deployed
system would see them, with no lookahead - and ask:

  1. Did the *targeted* alert kind fire at all on or after onset?   -> recall
  2. How many days after onset did it first fire?                   -> detection latency
  3. Did that kind fire *before* onset, or on a control resident?   -> false positives

Naming correction, stated deliberately
--------------------------------------
`docs/04_roadmap.md` calls the headline metric "detection lead time". That is not what
this measures and the distinction matters. Lead time would require a reference point for
when a *human* would have noticed the decline, and no such reference exists in either
the simulator or any public dataset. What we can measure is **latency from injected
onset**, which is strictly harder to score well on: onset is the first day the
underlying process changes, long before any observer could see it. Reported numbers are
labelled `detection_latency_days` throughout, and the roadmap term should be read as
shorthand for it.

Scoring strictness
------------------
A hit requires the *targeted* alert kind (`EXPECTED_ALERTS`), not merely any alert. A
generic ROUTINE_DEVIATION on `walking_duration_s` does indicate something changed, but it
is INFO severity and does not name the clinical concern, so counting it as a mobility
decline detection would flatter the system. It is reported separately as
`any_alert_latency_days` so the gap between the two is visible rather than hidden.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from behaviorsense.agents.behaviour import BehaviourAnalyzer, BehaviourConfig
from behaviorsense.data.simulator import (
    SIMULATOR_VERSION,
    AnomalyKind,
    Scenario,
    standard_scenarios,
)
from behaviorsense.schemas import Alert, AlertKind, AlertSeverity

EVAL_VERSION = "1.0.0"

EXPECTED_ALERTS: dict[AnomalyKind, frozenset[AlertKind]] = {
    AnomalyKind.GRADUAL_MOBILITY_DECLINE: frozenset({AlertKind.MOBILITY_DECLINE}),
    AnomalyKind.ACUTE_MOBILITY_DROP: frozenset({AlertKind.MOBILITY_DECLINE}),
    AnomalyKind.MEAL_SKIPPING: frozenset({AlertKind.MEAL_SKIPPED}),
    AnomalyKind.SLEEP_DISRUPTION: frozenset({AlertKind.SLEEP_DISRUPTION}),
    AnomalyKind.SOCIAL_WITHDRAWAL: frozenset({AlertKind.SOCIAL_ISOLATION}),
    AnomalyKind.MEDICATION_LAPSE: frozenset({AlertKind.MEDICATION_MISSED}),
    AnomalyKind.FALL_EVENT: frozenset(
        {AlertKind.FALL_DETECTED, AlertKind.FALL_NO_RECOVERY}
    ),
    AnomalyKind.NONE: frozenset(),
}

# Rules driven by absolute thresholds rather than the resident's own baseline. These
# fire independently of any injected anomaly, so pooling them into the false-positive
# rate would blame the deviation detector for a separate design choice. Tracked and
# reported on their own line instead.
ABSOLUTE_THRESHOLD_KINDS: frozenset[AlertKind] = frozenset(
    {AlertKind.HYDRATION_LOW, AlertKind.PROLONGED_INACTIVITY}
)


@dataclass
class ScenarioResult:
    """Scored outcome of replaying one scenario through the detector."""

    scenario_name: str
    persona: str
    kind: AnomalyKind
    n_days: int
    warmup_days: int
    """Days with no usable baseline. Excluded from false-positive denominators, since
    the deviation rules are suppressed there by construction and counting them as
    'correctly silent' would inflate specificity."""

    detected: bool = False
    detection_latency_days: int | None = None
    any_alert_latency_days: int | None = None

    n_target_alerts: int = 0
    pre_onset_false_alerts: int = 0
    """Targeted-kind alerts fired BEFORE onset - unambiguous false positives: the
    process had not yet changed."""

    scoreable_pre_onset_days: int = 0
    alert_counts: Counter[AlertKind] = field(default_factory=Counter)
    absolute_threshold_alerts: int = 0
    max_severity: AlertSeverity | None = None

    @property
    def is_control(self) -> bool:
        return self.kind is AnomalyKind.NONE

    @property
    def false_alerts_per_100_days(self) -> float:
        if self.scoreable_pre_onset_days <= 0:
            return 0.0
        return 100.0 * self.pre_onset_false_alerts / self.scoreable_pre_onset_days


@dataclass
class EvaluationReport:
    """Aggregate over the whole suite. This is the object the dissertation tables cite."""

    results: list[ScenarioResult] = field(default_factory=list)
    eval_version: str = EVAL_VERSION
    simulator_version: str = SIMULATOR_VERSION
    config: BehaviourConfig = field(default_factory=BehaviourConfig)

    # -- top-level metrics ------------------------------------------------------

    @property
    def anomaly_results(self) -> list[ScenarioResult]:
        return [r for r in self.results if not r.is_control]

    @property
    def control_results(self) -> list[ScenarioResult]:
        return [r for r in self.results if r.is_control]

    @property
    def recall(self) -> float:
        """Fraction of injected anomalies the targeted rule caught at any latency."""
        anomalies = self.anomaly_results
        if not anomalies:
            return 0.0
        return sum(1 for r in anomalies if r.detected) / len(anomalies)

    @property
    def median_latency_days(self) -> float | None:
        lat = [
            r.detection_latency_days
            for r in self.anomaly_results
            if r.detection_latency_days is not None
        ]
        return statistics.median(lat) if lat else None

    @property
    def control_false_alert_rate(self) -> float:
        """Baseline-driven alerts per 100 days on residents with no injected anomaly.

        The single most important safety number in the system. Alert fatigue - not
        missed detection - is what gets clinical monitoring switched off in practice.
        """
        controls = self.control_results
        total_days = sum(r.scoreable_pre_onset_days for r in controls)
        if total_days <= 0:
            return 0.0
        n = sum(
            count
            for r in controls
            for kind, count in r.alert_counts.items()
            if kind not in ABSOLUTE_THRESHOLD_KINDS
        )
        return 100.0 * n / total_days

    def by_kind(self) -> dict[AnomalyKind, tuple[float, float | None, int]]:
        """kind -> (recall, median latency, n scenarios)."""
        groups: dict[AnomalyKind, list[ScenarioResult]] = defaultdict(list)
        for r in self.anomaly_results:
            groups[r.kind].append(r)
        out = {}
        for kind, rs in groups.items():
            lat = [r.detection_latency_days for r in rs if r.detection_latency_days is not None]
            out[kind] = (
                sum(1 for r in rs if r.detected) / len(rs),
                statistics.median(lat) if lat else None,
                len(rs),
            )
        return out

    def by_persona(self) -> dict[str, tuple[float, float | None, float]]:
        """persona -> (recall, median latency, control false alerts / 100 days).

        Broken out because the suite's central finding is that detector sensitivity
        tracks the resident's day-to-day variability far more than the anomaly's label.
        A pooled number would conceal that.
        """
        groups: dict[str, list[ScenarioResult]] = defaultdict(list)
        for r in self.results:
            groups[r.persona].append(r)

        out = {}
        for persona, rs in groups.items():
            anomalies = [r for r in rs if not r.is_control]
            controls = [r for r in rs if r.is_control]
            lat = [
                r.detection_latency_days
                for r in anomalies
                if r.detection_latency_days is not None
            ]
            ctrl_days = sum(r.scoreable_pre_onset_days for r in controls)
            ctrl_alerts = sum(
                count
                for r in controls
                for kind, count in r.alert_counts.items()
                if kind not in ABSOLUTE_THRESHOLD_KINDS
            )
            out[persona] = (
                sum(1 for r in anomalies if r.detected) / len(anomalies) if anomalies else 0.0,
                statistics.median(lat) if lat else None,
                100.0 * ctrl_alerts / ctrl_days if ctrl_days else 0.0,
            )
        return out

    def alert_kind_totals(self) -> Counter[AlertKind]:
        total: Counter[AlertKind] = Counter()
        for r in self.results:
            total.update(r.alert_counts)
        return total


def evaluate_scenario(
    scenario: Scenario, config: BehaviourConfig | None = None
) -> ScenarioResult:
    """Replay one scenario day-by-day through a fresh detector and score it.

    A fresh `BehaviourAnalyzer` per scenario is essential: the baseline and CUSUM state
    are per-resident, and carrying them across residents would leak one person's routine
    into another's baseline.
    """
    cfg = config or BehaviourConfig()
    analyzer = BehaviourAnalyzer(cfg)
    sim = scenario.run()

    expected = EXPECTED_ALERTS[scenario.kind]
    onset = scenario.anomalies[0].onset_day if scenario.anomalies else None

    result = ScenarioResult(
        scenario_name=scenario.name,
        persona=scenario.profile.name,
        kind=scenario.kind,
        n_days=len(sim.days),
        warmup_days=cfg.min_baseline_days,
    )

    severity_order = {
        AlertSeverity.INFO: 0,
        AlertSeverity.ADVISORY: 1,
        AlertSeverity.URGENT: 2,
        AlertSeverity.EMERGENCY: 3,
    }

    for day_features in sim.days:
        day = day_features.day
        # A day only counts toward the false-positive denominator once a baseline
        # could exist; before that the deviation rules are suppressed by construction.
        has_baseline = analyzer.baseline.n_days >= cfg.min_baseline_days
        pre_onset = onset is None or day < onset
        if pre_onset and has_baseline:
            result.scoreable_pre_onset_days += 1

        state = analyzer.analyze_day(day_features)
        alerts: Sequence[Alert] = state.alerts
        if not alerts:
            continue

        for a in alerts:
            result.alert_counts[a.kind] += 1
            if a.kind in ABSOLUTE_THRESHOLD_KINDS:
                result.absolute_threshold_alerts += 1
            if (
                result.max_severity is None
                or severity_order[a.severity] > severity_order[result.max_severity]
            ):
                result.max_severity = a.severity

        if onset is not None and day >= onset and result.any_alert_latency_days is None:
            result.any_alert_latency_days = (day - onset).days

        hits = [a for a in alerts if a.kind in expected]
        if not hits:
            continue

        if onset is None:
            # Control scenario: any targeted-kind alert is by definition spurious.
            continue
        if day < onset:
            result.pre_onset_false_alerts += len(hits)
            continue

        result.n_target_alerts += len(hits)
        if not result.detected:
            result.detected = True
            result.detection_latency_days = (day - onset).days

    return result


def evaluate_suite(
    scenarios: Iterable[Scenario] | None = None,
    config: BehaviourConfig | None = None,
) -> EvaluationReport:
    cfg = config or BehaviourConfig()
    scens = list(scenarios) if scenarios is not None else list(standard_scenarios())
    return EvaluationReport(
        results=[evaluate_scenario(s, cfg) for s in scens], config=cfg
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report(report: EvaluationReport) -> str:
    """Markdown tables, ready to paste into the dissertation results chapter."""
    lines: list[str] = []
    add = lines.append

    add("# Agent 3 behaviour-layer evaluation")
    add("")
    add(
        f"Simulator v{report.simulator_version}, harness v{report.eval_version}. "
        f"{len(report.results)} scenarios "
        f"({len(report.anomaly_results)} anomalies, {len(report.control_results)} controls)."
    )
    add("")
    add(
        "> These are **simulator** results. They bound the detector's sensitivity given "
        "clean features; they are not estimates of clinical accuracy."
    )
    add("")

    lat = report.median_latency_days
    add("## Headline")
    add("")
    add("| metric | value |")
    add("|---|---|")
    add(f"| recall (targeted alert kind) | {report.recall:.0%} |")
    add(f"| median detection latency | {'n/a' if lat is None else f'{lat:.0f} days'} |")
    add(
        f"| control false-alert rate | {report.control_false_alert_rate:.2f} "
        f"per 100 resident-days |"
    )
    add("")

    add("## By anomaly kind")
    add("")
    add("| anomaly | recall | median latency (days) | n |")
    add("|---|---|---|---|")
    for kind, (rec, med, n) in sorted(report.by_kind().items(), key=lambda kv: kv[0].value):
        add(f"| {kind.value} | {rec:.0%} | {'-' if med is None else f'{med:.0f}'} | {n} |")
    add("")

    add("## By persona (ordered by day-to-day variability)")
    add("")
    add("| persona | noise cv | recall | median latency (days) | control FP / 100 days |")
    add("|---|---|---|---|---|")
    from behaviorsense.data.simulator import PERSONAS

    per = report.by_persona()
    for name in sorted(per, key=lambda n: PERSONAS[n].noise_cv):
        rec, med, fp = per[name]
        cv = PERSONAS[name].noise_cv
        add(
            f"| {name} | {cv:.2f} | {rec:.0%} | "
            f"{'-' if med is None else f'{med:.0f}'} | {fp:.2f} |"
        )
    add("")

    add("## Per-scenario detail")
    add("")
    add("| scenario | detected | latency | any-alert latency | pre-onset FP | max severity |")
    add("|---|---|---|---|---|---|")
    for r in report.results:
        lat_s = "-" if r.detection_latency_days is None else str(r.detection_latency_days)
        any_s = "-" if r.any_alert_latency_days is None else str(r.any_alert_latency_days)
        sev = r.max_severity.value if r.max_severity else "-"
        mark = "yes" if r.detected else ("n/a" if r.is_control else "**NO**")
        add(
            f"| {r.scenario_name} | {mark} | {lat_s} | {any_s} | "
            f"{r.pre_onset_false_alerts} | {sev} |"
        )
    add("")

    add("## Alert volume by kind (all scenarios)")
    add("")
    add("| alert kind | count | absolute-threshold rule |")
    add("|---|---|---|")
    for kind, count in report.alert_kind_totals().most_common():
        flag = "yes" if kind in ABSOLUTE_THRESHOLD_KINDS else ""
        add(f"| {kind.value} | {count} | {flag} |")
    add("")

    return "\n".join(lines)


__all__ = [
    "EVAL_VERSION",
    "EXPECTED_ALERTS",
    "ABSOLUTE_THRESHOLD_KINDS",
    "ScenarioResult",
    "EvaluationReport",
    "evaluate_scenario",
    "evaluate_suite",
    "format_report",
]
