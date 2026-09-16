"""Hallucination-rate benchmark: constrained vs unconstrained decoding.

Produces the headline LLM table. The metric is the fraction of generated claims failing
deterministic verification (C1-C4), measured over many simulated days so it is not a
single-report anecdote.

Two backends:

  --backend stub   (default) deterministic stubs, CPU, no weights. Establishes that the
                   MEASUREMENT works: a faithful generator must score 0.0 and a corrupted
                   one must score at its injection rate. Without those two anchors, a
                   number from a real model is uninterpretable.
  --backend qwen   the real Qwen2.5-7B-Instruct, run on Kaggle with staged weights.
                   Three arms - constrained, free, and free re-scored after format-only
                   repair.
  --backend openrouter   the HOSTED arm the system actually serves (GLM -> nemotron
                   chain, keys from OPENROUTER_API_KEY_1..N in the local environment).
                   Runs on the LOCAL machine - no GPU, no Kaggle, keys never leave it.
                   Same three arms as the Qwen benchmark, so the two tables sit side by
                   side: `constrained` here is OpenRouter's server-side `response_format`
                   enforcement rather than outlines' local grammar - the same guarantee
                   from the other end of the wire.

Two rates per row, and the distinction is the point
---------------------------------------------------
`hallucination rate` divides by SCHEMA-VALID claims; `unusable rate` divides by claims
EMITTED. The first real run needed both: the free arm emitted 245 claims, landed zero, and
reported `nan%`, which reads as missing data when the finding is total failure. A claim the
schema rejects is no more usable to a caregiver than one that is fluent and false, so the
emitted denominator is the one a deployment cares about - and the schema-valid denominator
is the one that isolates the verifier's own view.

The third arm re-scores the free arm's CACHED responses (see `RecordingLLM`) after
`repair_claim` fixes formatting only. Two arms could not distinguish "cannot emit our JSON"
from "misstates the data"; three can, and only the second is a faithfulness result.

Usage:
    PYTHONPATH=src python scripts/eval_hallucination.py --days 60
    PYTHONPATH=src python scripts/eval_hallucination.py --backend qwen \\
        --model-path /kaggle/input/qwen25-7b-instruct --report results/hallucination.md
    PYTHONPATH=src python scripts/eval_hallucination.py --backend openrouter \\
        --report results/hallucination_openrouter.md
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behaviorsense.agents.behaviour import BehaviourAnalyzer  # noqa: E402
from behaviorsense.agents.reasoning.reporter import (  # noqa: E402
    CaregiverReporter,
    FaithfulStubLLM,
    HallucinatingStubLLM,
    Qwen2_5Reporter,
    ReporterConfig,
)
from behaviorsense.data.simulator import standard_scenarios  # noqa: E402


def collect_states(n_days: int, n_scenarios: int):
    """Analysed states from several personas, biased toward days that alert.

    Days with alerts are where a report has something consequential to say, and therefore
    where a hallucination does real damage. Sampling uniformly would dilute the metric
    with quiet days on which almost any claim is trivially true.
    """
    states = []
    for scenario in list(standard_scenarios())[:n_scenarios]:
        analyzer = BehaviourAnalyzer()
        result = scenario.run()
        for day in result.days[:n_days]:
            st = analyzer.analyze_day(day)
            if st.baselines:
                states.append(st)
    alerting = [s for s in states if s.alerts]
    quiet = [s for s in states if not s.alerts]
    return alerting + quiet[: max(1, len(alerting))]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k/n.

    Reported because the first usable comparison of the two decoding arms came out 8.2%
    against 6.5% over ~500 claims each, and a bare pair of point estimates invites the
    reader to conclude that one arm is better. It is not: the intervals overlap heavily.
    Wilson rather than normal-approximation because these rates sit near a boundary where
    the Wald interval misbehaves.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """(z, two-sided p) for two independent rates. No SciPy dependency."""
    if not n1 or not n2:
        return (float("nan"), float("nan"))
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (k1 / n1 - k2 / n2) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p


class RecordingLLM:
    """Wraps an LLM, caching each prompt's response so a second arm can re-score it.

    Two reasons this is not just an optimisation. Cost: the first real run spent most of
    its 2.5 GPU-hours inside `generate`, and a third arm that regenerated would add ~50%
    for nothing. Correctness: comparing "unconstrained" against "unconstrained + repair"
    only isolates the repair if BOTH arms see the same bytes. Regenerating would sample a
    second time, so a difference could be sampling noise rather than the repair - and at
    temperature 0 it would be an unverifiable assumption that they matched.
    """

    def __init__(self, inner) -> None:
        self.inner = inner
        self.cache: dict[str, str] = {}
        self.replaying = False

    @property
    def name(self) -> str:
        return self.inner.name

    def generate(self, prompt: str, *, constrained: bool, schema: dict) -> str:
        key = f"{constrained}|{prompt}"
        if key in self.cache:
            self.replaying = True
            return self.cache[key]
        out = self.inner.generate(prompt, constrained=constrained, schema=schema)
        self.cache[key] = out
        return out


def run_condition(name: str, llm, states, cfg: ReporterConfig,
                  dump: list | None = None) -> dict:
    reporter = CaregiverReporter(llm, config=cfg)
    rates, claims, faithful, parse_fail, dropped = [], 0, 0, 0, 0
    emitted = repaired = 0
    causes: Counter[str] = Counter()
    checks: Counter[str] = Counter()
    t0 = time.time()
    print(f"[{name}] {len(states)} reports...", flush=True)
    for i, st in enumerate(states, 1):
        out = reporter.report(st)
        if out.report.claims:
            rates.append(out.hallucination_rate)
        claims += len(out.report.claims)
        emitted += out.n_emitted_claims
        repaired += out.repaired_claims
        faithful += out.verification.n_faithful
        parse_fail += int(out.parse_failed)
        dropped += out.dropped_claims
        for line in out.rejections:
            causes[line.split(" (got ")[0]] += 1
        # Which of C1-C4 bit. "8.2% hallucinated" is a headline; "39 of 41 failures are
        # C2 value mismatches and none are fabricated citations" is the error analysis a
        # reader needs, and it points at a different fix than the headline suggests.
        checks.update({k: v for k, v in out.verification.failure_breakdown().items() if v})
        if dump is not None:
            # Every claim, with the verifier's verdict and notes. Written so that any
            # further analysis - which values were misquoted and by how much, examples to
            # quote in the write-up, sensitivity to the 2% tolerance - is a CPU-second
            # rescore of this file rather than another 2-hour GPU session.
            by_id = {v.claim_id: v for v in out.verification.results}
            for c in out.report.claims:
                v = by_id.get(c.claim_id)
                dump.append({
                    "condition": name, "model": llm.name,
                    "report_id": out.report.report_id,
                    "day": out.report.report_day.isoformat(),
                    "claim": c.model_dump(mode="json"),
                    "faithful": bool(v and v.is_faithful),
                    "checks": {} if v is None else {
                        "ref_exists": v.ref_exists, "value_matches": v.value_matches,
                        "pct_matches": v.pct_matches,
                        "direction_consistent": v.direction_consistent},
                    "notes": [] if v is None else list(v.notes),
                })
        # Progress, because the first real run showed no output between minute 6 and
        # hour 2.6 and there was no way to tell generation from a hang.
        if i % 20 == 0 or i == len(states):
            rate = (time.time() - t0) / i
            print(f"[{name}] {i}/{len(states)} reports, {emitted} claims emitted, "
                  f"{faithful} faithful, {rate:.1f}s/report, "
                  f"eta {rate * (len(states) - i) / 60:.1f} min", flush=True)
    return {
        "condition": name,
        "model": llm.name,
        "n_reports": len(states),
        "n_emitted": emitted,
        "n_claims": claims,
        "n_repaired": repaired,
        "n_faithful": faithful,
        # Over SCORABLE claims: the verifier's own view, and undefined when the model
        # emitted nothing the schema accepted.
        "hallucination_rate": (claims - faithful) / claims if claims else float("nan"),
        # Over EMITTED claims: what a caregiver would experience. A claim thrown out by
        # the schema is no more usable than one that is fluent and false, so an arm that
        # emits 245 claims and lands 0 must not score `nan` and read as "no data" - it
        # scores 100% unusable, which is the actual finding.
        "unusable_rate": (emitted - faithful) / emitted if emitted else float("nan"),
        "per_report_median": statistics.median(rates) if rates else float("nan"),
        "parse_failures": parse_fail,
        "schema_rejected": dropped,
        "top_causes": causes.most_common(4),
        "failed_checks": dict(checks),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=("stub", "qwen", "openrouter"), default="stub")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--scenarios", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report", default=None)
    ap.add_argument("--dump", default=None,
                    help="JSONL of every claim + its verdict, so later analysis needs no "
                         "GPU. Writes next to --report by default when --report is given.")
    args = ap.parse_args()

    states = collect_states(args.days, args.scenarios)
    n_alert = sum(1 for s in states if s.alerts)
    print(f"{len(states)} analysed days ({n_alert} with alerts) from "
          f"{args.scenarios} scenarios", flush=True)

    rows = []
    records: list[dict] = []
    if args.backend == "stub":
        cfg = ReporterConfig()
        rows.append(run_condition("faithful stub (ceiling)", FaithfulStubLLM(cfg), states,
                                  cfg, records))
        for rate in (0.25, 0.50, 1.00):
            rows.append(run_condition(
                f"corrupted stub @ {rate:.0%}",
                HallucinatingStubLLM(rate=rate, seed=0, config=cfg), states, cfg, records))
    elif args.backend == "openrouter":
        # THE ARM WE ACTUALLY SERVE, and the reason this backend exists: notebook 04's Qwen
        # table measures a pinned checkpoint the deployment no longer uses, and a
        # dissertation claim about "Agent 4's model" should include the model in the
        # system. Runs locally - the keys are here and the models are hosted - so there is
        # no GPU session, no attached dataset, and no credential anywhere near Kaggle.
        from behaviorsense.agents.reasoning.openrouter import OpenRouterLLM, discover_keys
        keys = discover_keys()
        if not keys:
            raise SystemExit(
                "--backend openrouter requires OPENROUTER_API_KEY_1..N (or "
                "OPENROUTER_API_KEY_LIST) in the environment.")
        # ONE instance for all three arms, deliberately. Provider cooldowns are shared, and
        # `RecordingLLM` keys its cache on (constrained, prompt), so the repair arm replays
        # the unconstrained arm's bytes rather than resampling them - the property that
        # makes "unconstrained" and "unconstrained + repair" a controlled comparison.
        # NOT cfg.max_new_tokens: that is the local Qwen arm's 1,600-token budget, and a
        # hosted reasoning model spends part of any smaller budget thinking (measured:
        # 1,600 truncated every report, 4,096 truncated dots-3). HOSTED_MAX_TOKENS is the
        # right ceiling here and is OpenRouterLLM's own default.
        llm = RecordingLLM(OpenRouterLLM(keys=keys))
        arms = [("grammar-constrained", True, False),
                ("unconstrained", False, False),
                ("unconstrained + format repair", False, True)]
        for label, constrained, salvage in arms:
            cfg = ReporterConfig(constrained=constrained, salvage=salvage)
            row = run_condition(label, llm, states, cfg, records)
            row["replayed"] = llm.replaying
            rows.append(row)
    else:
        if not args.model_path:
            raise SystemExit("--backend qwen requires --model-path")
        # Three arms, because two were not enough to interpret the first real run. The
        # free arm emitted 245 claims and landed 0, which measured JSON compliance and
        # said nothing about truthfulness; the salvaged arm scores those same responses
        # after format-only repair, isolating faithfulness from formatting.
        arms = [("grammar-constrained", True, False),
                ("unconstrained", False, False),
                ("unconstrained + format repair", False, True)]
        loaded: dict[bool, RecordingLLM] = {}
        for label, constrained, salvage in arms:
            cfg = ReporterConfig(constrained=constrained, salvage=salvage)
            # One model instance per decoding mode: reloading 16 GB of weights per arm
            # cost ~3 minutes each in the first run for no benefit. `Qwen2_5Reporter`
            # reads `constrained` from the call, and `name` from its own cfg - so the
            # instance is keyed on the flag that changes its reported name. The recorder
            # means the repair arm re-scores the SAME responses instead of resampling.
            if constrained not in loaded:
                loaded[constrained] = RecordingLLM(Qwen2_5Reporter(
                    args.model_path, device=args.device,
                    config=ReporterConfig(constrained=constrained)))
            llm = loaded[constrained]
            row = run_condition(label, llm, states, cfg, records)
            row["replayed"] = llm.replaying
            rows.append(row)

    def pct(x: float) -> str:
        # An arm that emitted claims but landed none has a DEFINED rate of 100%. Only a
        # genuinely empty denominator is n/a, and saying so beats printing "nan%".
        return "n/a" if x != x else f"{x:.1%}"

    lines = ["# Hallucination rate", "",
             f"- {len(states)} analysed days ({n_alert} alerting) from "
             f"{args.scenarios} simulator scenarios",
             "- a claim is a hallucination if it fails any of C1-C5 (deterministic, no LLM)",
             "- `hallucination rate` is over SCHEMA-VALID claims (the verifier's view);",
             "  `unusable rate` is over ALL emitted claims, counting schema rejects as",
             "  unusable - which is what a caregiver actually experiences",
             ]
    if args.backend == "openrouter":
        # The hosted arm cannot promise what the Qwen table promises. Free providers
        # rotate without notice - two vanished from the tier in the week this backend was
        # written - and the chain routes around saturation, so the model that answered is
        # a property of the day, not of the config. Say so in the header rather than
        # letting the table claim the reproducibility it does not have.
        lines += [
            "- **HOSTED ARM**: served over OpenRouter's free tier; the provider that",
            "  answered each report is recorded in the JSONL dump. Free models rotate",
            "  without notice, so this is a DATED measurement of the deployed",
            "  configuration, not a pinned checkpoint - the Qwen table remains the",
            "  reproducible anchor.",
        ]
    lines += ["",
             "| condition | model | reports | emitted | scorable | repaired | faithful | "
             "hallucination rate | 95% CI | unusable rate | parse fail | schema rejected |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lo, hi = wilson(r["n_claims"] - r["n_faithful"], r["n_claims"])
        ci = "n/a" if lo != lo else f"{lo:.1%}-{hi:.1%}"
        lines.append(
            f"| {r['condition']} | `{r['model']}` | {r['n_reports']} | {r['n_emitted']} | "
            f"{r['n_claims']} | {r['n_repaired']} | {r['n_faithful']} | "
            f"**{pct(r['hallucination_rate'])}** | {ci} | "
            f"**{pct(r['unusable_rate'])}** | "
            f"{r['parse_failures']} | {r['schema_rejected']} |")

    # Whether the two decoding modes actually differ. Two point estimates side by side
    # invite the reader to rank them; over ~500 claims each, a 1.7-point gap does not
    # survive a significance test, and saying so is the honest reading.
    named = {r["condition"]: r for r in rows}
    a, b = named.get("grammar-constrained"), named.get("unconstrained")
    if a and b and a["n_claims"] and b["n_claims"]:
        z, p = two_proportion_z(a["n_claims"] - a["n_faithful"], a["n_claims"],
                                b["n_claims"] - b["n_faithful"], b["n_claims"])
        verdict = ("indistinguishable" if p >= 0.05
                   else "a real difference (p < 0.05)")
        lines += ["", "## Does the grammar help?", "",
                  f"- constrained {pct(a['hallucination_rate'])} vs free "
                  f"{pct(b['hallucination_rate'])}: difference "
                  f"{a['hallucination_rate'] - b['hallucination_rate']:+.1%}, "
                  f"z = {z:.2f}, two-sided p = {p:.2f} -> **{verdict}**",
                  "- the grammar's value is a *worst-case guarantee* of parseability, not",
                  "  a faithfulness gain: it cannot emit an invalid claim, whereas the free",
                  "  arm merely happened to emit none once the prompt stated the schema",
                  "- and it is not free: see the per-report timings in the run log"]

    # Why an arm failed, not merely that it did. A row of 245 rejections with no cause is
    # not a finding anyone can act on or write up.
    if any(r["top_causes"] for r in rows):
        lines += ["", "## Why claims were rejected", ""]
        for r in rows:
            if r["top_causes"]:
                causes = "; ".join(f"`{c}` x{n}" for c, n in r["top_causes"])
                lines.append(f"- **{r['condition']}**: {causes}")

    if any(r["failed_checks"] for r in rows):
        lines += ["", "## Which check caught it (C1-C5)", "",
                  "| condition | bad ref (C1) | value (C2) | pct (C3) | direction (C4) "
                  "| prose (C5) |",
                  "|---|---|---|---|---|---|"]
        for r in rows:
            c = r["failed_checks"]
            if not c:
                continue
            lines.append(
                f"| {r['condition']} | {c.get('missing_or_bad_ref', 0)} | "
                f"{c.get('value_mismatch', 0)} | {c.get('pct_mismatch', 0)} | "
                f"{c.get('direction_error', 0)} | "
                f"{c.get('prose_quoted_value_mismatch', 0)} |")
        lines += ["",
                  "Counts are per CHECK, not per claim - one claim can fail several - so "
                  "rows do not sum to the unfaithful total. C4 is the clinically "
                  "dangerous mode: a correct number narrated backwards. C3 alone is a "
                  "copying error.",
                  "",
                  "C5 asks whether the claim's PROSE echoes the number in `claimed_value`. "
                  "It exists because C2 compares the field against the evidence row and is "
                  "blind to a claim whose row is right and whose sentence is not - and the "
                  "sentence is what a caregiver reads. A run where C2 is 0 and C5 is not is "
                  "a run where the model is copying into the field correctly and "
                  "paraphrasing in the text."]

    repair_arm = next((r for r in rows if "repair" in r["condition"]), None)
    if repair_arm is not None:
        lines += ["", "## Note on the repair arm", "",
                  "`repair_claim` coerces FORMAT only - whitespace, `\"1800\"` -> `1800.0`,",
                  "`\"Decreased\"` -> `\"decrease\"`, `YYYY/MM/DD` -> `YYYY-MM-DD`. It never",
                  "invents an evidence_ref, changes a number, or infers an unstated",
                  "direction, so a repaired claim can move from unscorable to scored but",
                  "never from false to faithful. The row separates *cannot emit our JSON*",
                  "from *misstates the data*; only the latter is a faithfulness result."]
        if repair_arm["n_repaired"] == 0:
            # An identical row is the RESULT, not a redundancy: it says the free arm needed
            # no repair at all. Left in the table rather than suppressed, because a reader
            # who sees only two rows cannot tell that the third was run.
            lines += ["",
                      f"**0 of {repair_arm['n_emitted']} claims needed repair**, so this row "
                      "is identical to the unconstrained one by construction. That is the "
                      "finding: once the prompt states the output contract, free decoding "
                      "produced no malformed claims, and the 245 schema rejections in the "
                      "first run were caused by our prompt withholding the field names - "
                      "not by the model."]
        if any(r.get("replayed") for r in rows):
            lines.append("")
            lines.append("The repair arm re-scored the unconstrained arm's cached "
                         "responses, not fresh samples, so any delta is the repair alone.")

    if args.backend == "stub":
        ceiling = rows[0]["hallucination_rate"]
        lines += ["", "## Sanity anchors", "",
                  f"- faithful generator scored **{pct(ceiling)}** (must be 0.0%: any "
                  "higher means the verifier has false positives and every rate below "
                  "is inflated)"]
        for r in rows[1:]:
            target = float(r["condition"].split("@")[1].strip().rstrip("%")) / 100
            lines.append(f"- corrupted @ {target:.0%} scored {pct(r['hallucination_rate'])} "
                         f"(tracks the injection rate, so the metric is not saturating)")

    text = "\n".join(lines)
    print("\n" + text)
    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"\nwrote {out}")

    dump_path = args.dump or (str(Path(args.report).with_name(
        Path(args.report).stem + "_claims.jsonl")) if args.report else None)
    if dump_path and records:
        # Written unconditionally alongside the report. The C2 error analysis that this
        # table now demands - which values were misquoted and by how much - needed the
        # per-claim records, and the run that produced the table had not kept them.
        dp = Path(dump_path)
        dp.parent.mkdir(parents=True, exist_ok=True)
        with dp.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        print(f"wrote {dp} ({len(records)} claims) - rescore this without a GPU")


if __name__ == "__main__":
    main()
