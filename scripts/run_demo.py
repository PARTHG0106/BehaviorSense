"""End-to-end replay demo: simulator -> Agent 3 -> Agent 4 -> verified report.

This is the GPU-independent demonstration path. It exercises the real Agent 3 statistics,
the real Agent 4 verifier, and the real report-rendering logic on simulated longitudinal
data with injected ground truth - no checkpoints, no video, no internet. If the trained
models are unavailable for any reason, this still runs and still demonstrates the
project's actual contribution: that a generated caregiver report is mechanically checked
against the data it claims to describe.

What it deliberately does NOT demonstrate: perception or activity accuracy. Those need
real video and are evaluated separately (P1/P2/P3 protocols). Conflating the two would
let a clean demo imply a level of validation the system does not have.

Usage:
    PYTHONPATH=src python scripts/run_demo.py
    PYTHONPATH=src python scripts/run_demo.py --scenario gradual_mobility_decline
    PYTHONPATH=src python scripts/run_demo.py --hallucinate 0.5   # show the verifier bite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behaviorsense.agents.behaviour import BehaviourAnalyzer  # noqa: E402
from behaviorsense.agents.reasoning.reporter import (  # noqa: E402
    CaregiverReporter,
    FaithfulStubLLM,
    HallucinatingStubLLM,
    ReporterConfig,
)
from behaviorsense.data.simulator import standard_scenarios  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default=None,
                    help="substring of a scenario name; default = first with an anomaly")
    ap.add_argument("--days", type=int, default=None, help="truncate the horizon")
    ap.add_argument("--hallucinate", type=float, default=0.0,
                    help="corrupt this fraction of claims, to show the verifier working")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scenarios = list(standard_scenarios())
    if args.scenario:
        matches = [s for s in scenarios if args.scenario in s.name]
        if not matches:
            raise SystemExit(
                f"no scenario matching {args.scenario!r}. Available:\n  "
                + "\n  ".join(sorted({s.name for s in scenarios}))
            )
        scenario = matches[0]
    else:
        scenario = next(s for s in scenarios if not s.is_control)

    print("=" * 78)
    print(f"BehaviorSense end-to-end replay: {scenario.name}")
    print("=" * 78)
    result = scenario.run()
    days = result.days[: args.days] if args.days else result.days
    print(f"simulated {len(days)} days, persona CV={scenario.profile.noise_cv:.2f}, "
          f"anomaly={scenario.kind.value}")

    # ---- Agent 3: longitudinal statistics -> alerts -------------------------
    analyzer = BehaviourAnalyzer()
    states = [analyzer.analyze_day(d) for d in days]
    alerting = [s for s in states if s.alerts]
    print(f"\nAgent 3: {sum(len(s.alerts) for s in states)} alerts across "
          f"{len(alerting)} day(s)")
    for s in alerting[:5]:
        kinds = ", ".join(f"{a.kind.value}({a.severity.value})" for a in s.alerts)
        print(f"  {s.report_day}: {kinds}")
    if len(alerting) > 5:
        print(f"  ... and {len(alerting) - 5} more alerting day(s)")

    # Report on the first alerting day if there is one; otherwise the last day, so the
    # demo always produces a report rather than silently doing nothing.
    target = alerting[0] if alerting else states[-1]

    # ---- Agent 4: generate, then verify -------------------------------------
    cfg = ReporterConfig()
    llm = (HallucinatingStubLLM(rate=args.hallucinate, seed=args.seed, config=cfg)
           if args.hallucinate > 0 else FaithfulStubLLM(config=cfg))
    reporter = CaregiverReporter(llm, config=cfg)
    outcome = reporter.report(target)

    print(f"\nAgent 4 ({llm.name}) on {target.report_day}:")
    print(f"  claims generated  : {len(outcome.report.claims)}")
    print(f"  verified faithful : {outcome.verification.n_faithful}")
    print(f"  hallucination rate: {outcome.hallucination_rate:.1%}")
    if outcome.dropped_claims:
        print(f"  schema-rejected   : {outcome.dropped_claims}")

    failures = [v for v in outcome.report.verifications if not v.is_faithful]
    if failures:
        print("\n  Claims caught by the deterministic verifier:")
        for v in failures[:4]:
            claim = next(c for c in outcome.report.claims if c.claim_id == v.claim_id)
            print(f"    [{v.claim_id}] {claim.text[:70]}")
            for note in v.notes[:2]:
                print(f"        -> {note}")

    print("\n" + "-" * 78)
    print("WHAT THE CAREGIVER SEES (verified claims only):")
    print("-" * 78)
    print(reporter.verified_summary(outcome))
    print("-" * 78)

    if args.hallucinate > 0:
        withheld = len(outcome.report.claims) - outcome.verification.n_faithful
        print(f"\n{withheld} unfaithful claim(s) withheld. Without the verifier they "
              f"would have\nreached the caregiver as fluent, confident prose.")


if __name__ == "__main__":
    main()
