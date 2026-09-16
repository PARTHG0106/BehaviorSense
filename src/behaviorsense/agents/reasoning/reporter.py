"""Agent 4 - Reporter: BehaviourState -> grammar-constrained claims -> verified report.

The boundary this file defends
------------------------------
The LLM never sees pixels, skeletons, or video. It receives a `BehaviourState` - a
structured JSON object of daily features, baselines, robust-z deviations and alerts - and
must return claims that each carry a machine-resolvable `evidence_ref`. Every claim then
passes through the deterministic verifier (C1-C4), which uses no LLM at all.

That ordering is the contribution. A system where the LLM both reads the data and
certifies its own summary can only be trusted as far as the LLM; here the certification
is arithmetic, so a fabricated number, a wrong percentage or an inverted narrative is
caught mechanically regardless of how fluent the prose is.

Why the LLM is behind a Protocol with stub implementations
---------------------------------------------------------
`Qwen2_5Reporter` needs ~16 GB of weights that only exist in the Kaggle environment.
Every claim this module makes about *faithfulness measurement* must be testable without
them, so the pipeline is exercised by two deterministic stubs:

  - `FaithfulStubLLM`   - reads the real evidence and reports it honestly (the ceiling)
  - `HallucinatingStubLLM` - injects the six failure modes we claim to catch, at a
    controlled rate, with a seed (the floor, and the thing that proves the verifier bites)

If the verifier cannot separate those two, it does not work - and that is measurable on
this CPU today. The real model swaps in behind the same `.generate()` call.

Why grammar-constrained decoding, and why it is not the whole answer
--------------------------------------------------------------------
Unconstrained LLMs emit prose with numbers inline, which cannot be checked without
brittle regex extraction. Constraining generation to the `Claim` schema forces the model
to *commit* to `claimed_value`, `claimed_pct_change` and `direction` as separate fields,
which is what makes C2/C3/C4 possible. But constraint only guarantees well-formed JSON -
a schema-valid claim can still be false. That is precisely why the verifier exists
downstream, and why we report hallucination rate for both conditions rather than treating
constrained decoding as a solution.

Measuring the unconstrained arm without measuring the wrong thing
----------------------------------------------------------------
The first real Qwen2.5-7B run made that comparison unusable, and the reasons are all here
rather than in the model. The free arm emitted 245 claims and landed zero: every one was
rejected by Pydantic, the denominator was empty, and the reported rate was `nan`. That
number says the model cannot produce our JSON. It says nothing about whether the model
tells the truth, which is the question. Three mechanisms address it:

  - `claim_schema()` binds the grammar's `maxItems` to `max_claims`. It allowed 12 while
    the reporter kept 6, so generation budget went to claims that were then discarded -
    which is how 9 of 116 constrained reports hit `max_new_tokens` and became parse
    failures rather than results.
  - `ReportOutcome.unusable_rate` uses claims EMITTED as its denominator, so an arm that
    lands nothing scores 100% instead of `nan`. A claim the schema rejects is no more
    usable to a caregiver than one that is fluent and false.
  - `repair_claim` coerces FORMAT only, and `ReporterConfig.salvage` re-scores the same
    responses through it. This separates "cannot comply with the schema" from "misstates
    the data"; only the second is a faithfulness result, and only that separation licenses
    the claim that constrained decoding improves faithfulness rather than parseability.

Repair is deliberately incapable of laundering a lie - it may not invent a reference,
change a number, or infer an unstated direction - and test R12 asserts that by replaying
all five content corruptions through it and requiring every one to still fail.
"""

from __future__ import annotations

import json
import pathlib
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from behaviorsense.agents.reasoning.verifier import FaithfulnessVerifier, VerificationReport
from behaviorsense.schemas import (
    AlertSeverity,
    BehaviourState,
    CaregiverReport,
    Claim,
    Evidence,
)

REPORTER_VERSION = "1.0.0"

# Features a caregiver report should mention when they deviate, in priority order. Not
# every tracked feature: a report listing 18 numbers is not read by anyone. Ordering is
# clinical, not alphabetical - mobility and medication before leisure.
REPORTABLE_FEATURES: tuple[str, ...] = (
    "walking_duration_s",
    "mobility_index",
    "sit_to_stand_count",
    "meal_events",
    "medication_events",
    "social_interaction_duration_s",
    # `visitor_count`, not buried: this is the only signal Agent 2's second-person context
    # produces for `interacting_with_person` (class 18, F1 0.022), and the social-isolation
    # alert is gated on `visitor_count == 0`. Showing the model the count, not just the
    # duration, is what lets it name "no visitors today" as the cause when it is.
    "visitor_count",
    "longest_inactive_block_s",
    # `lying_duration_s`, not `sleep_proxy_duration_s`: the latter never existed on
    # DailyFeatures, so state_to_payload skipped it silently via its `not in values`
    # guard and the model was never offered a sleep-related feature at all. A dead
    # entry in a list of "features a caregiver report should mention" is worse than
    # an absent one - it reads as covered.
    "lying_duration_s",
    "room_transitions",
    "housework_duration_s",
    # ADDED 2026-09-12, after the 9m56s upload: its report named walking, mobility index and the
    # 5.2 s longest-inactive block while never mentioning that the resident COOKED FOR 66.95 s and
    # DRANK FOUR TIMES - because neither feature was in this list, so the model was never offered
    # them. A caregiver report that omits cooking and drinking to lead with "longest inactive
    # block 5.2 s" is a report about the list, not the person.
    #
    # COST, stated rather than hidden: this list is the prompt's feature set, so the MEASURED
    # (Qwen) arm's hallucination table in results/evaluation.md was produced over the previous
    # 10-feature list and is now one revision behind the demo arm. Re-run notebook 04's
    # measurement to bring the table level; the comparison's structure is unchanged.
    "cooking_duration_s",
    "drinking_events",
)


@dataclass(frozen=True)
class ReporterConfig:
    max_claims: int = 6
    """A caregiver report with 20 claims is not read. Six is enough to cover the alerting
    features plus context, and keeps the hallucination denominator interpretable."""

    deviation_threshold: float = 1.5
    """Robust-z magnitude above which a feature is worth mentioning unprompted. Lower
    than the alert threshold (3.0) on purpose: the report should provide context around
    an alert, not merely restate it."""

    temperature: float = 0.0
    """Greedy decoding. Sampling adds variance to a metric we are trying to measure, and
    a caregiver report has no need for creative diversity."""

    max_new_tokens: int = 1600
    """Raised from 900 after the first real Qwen run lost 9 of 116 constrained reports to
    truncation: a grammar guarantees well-formed JSON only if generation is allowed to
    FINISH, and a cut-off object is unparseable no matter how valid its prefix was. Costs
    nothing on reports that already completed - decoding stops at EOS (free arm) or when
    the schema is satisfied (constrained arm), not at the cap."""

    constrained: bool = True
    """When False, the backend is asked for free-form JSON with no grammar. This is the
    comparison arm for the hallucination-rate table, not a supported deployment mode."""

    salvage: bool = False
    """Repair FORMATTING in claims Pydantic rejected, then re-score them.

    Off by default and never used in a deployment path. It exists because the first real
    run scored the unconstrained arm as 245 emitted claims, 0 scorable, rate `nan` - a
    number that says only "the model cannot produce our JSON" and nothing about whether it
    tells the truth. With salvage on, the same responses are scored a second time after
    deterministic format coercion, which separates two very different failures: *cannot
    comply with the schema* and *lies about the data*. Only that second row licenses the
    claim that constrained decoding improves faithfulness rather than merely parseability.

    The coercions may never change a claim's factual content - see `repair_claim`."""


@runtime_checkable
class ReportLLM(Protocol):
    """Text-in, JSON-out. The only surface the reporter needs from a language model."""

    @property
    def name(self) -> str: ...

    def generate(self, prompt: str, *, constrained: bool, schema: dict[str, Any]) -> str:
        """Return a JSON string conforming (ideally) to `schema`."""
        ...


# ---------------------------------------------------------------------------
# Prompt construction - the LLM's entire view of the world
# ---------------------------------------------------------------------------


def state_to_payload(state: BehaviourState, cfg: ReporterConfig) -> dict[str, Any]:
    """Serialise the parts of a BehaviourState an LLM may reason over.

    Deliberately includes the exact `ref` strings. The model is not asked to *construct*
    evidence references from a format description - it is given the resolvable refs and
    told to cite them. Asking a model to synthesise `feat:<name>:<date>` strings invites
    plausible-but-wrong refs, which then score as hallucinations that are really our
    prompt's fault. Measuring the model's honesty requires removing that trap.
    """
    day = state.report_day
    values = state.today.numeric_items()
    rows: list[dict[str, Any]] = []
    # A DAY'S TOTAL AND TEN MINUTES' TOTAL ARE NOT THE SAME QUANTITY, and comparing them is an
    # arithmetic error rather than a judgement call. Measured on a 9m56s upload: the reference day
    # observed 22.9 h against the clip's 0.166 h - a 138x difference - so walking read "-99.4%,
    # robust_z -10.00" while the actual walking RATE was 62.7 s per observed hour against the
    # baseline's 66.9 s/h. Essentially identical. Agent 4 read the -99.4% and recommended
    # contacting a healthcare professional about a person who was cooking and cleaning normally,
    # and all six claims passed C1-C5 because the arithmetic was internally consistent.
    #
    # 8 h is not a new threshold: `min_observed_hours_for_absence` already gates the ALERT rules at
    # exactly this figure, for exactly this reason. This extends the same gate to what may be
    # CLAIMED, which is the same decision applied to the other consumer of the same numbers.
    partial = not state.today.is_reliable()

    for name in REPORTABLE_FEATURES:
        if name not in values:
            continue
        baseline = state.baselines.get(name)
        if baseline is None:
            continue
        ev = Evidence.build(name, values[name], baseline, day)
        # A percentage change against a zero baseline median is undefined, not 0%. The
        # verifier rejects a quoted pct in that case (correctly - "0% change from 0" is
        # not a fact), so the pct is withheld from the model rather than offered as
        # something citable. Offering it caused the faithful reporter to score a 67%
        # hallucination rate on features the synthetic classifier never produces, which
        # is a reporter bug masquerading as a verifier failure.
        pct_defined = abs(ev.baseline_median) > 1e-9
        observed = abs(ev.observed_value) > 1e-9
        row = {
            "feature": name,
            "evidence_ref": ev.ref,
            "observed_value": round(ev.observed_value, 2),
            # PRESENCE IS EVIDENCE IN A SHORT WINDOW; ABSENCE IS NOT. Ten minutes can prove a
            # fall happened. It cannot prove no meal was eaten today. That asymmetry is the
            # whole rule, and it is why this flag is about the value rather than the window.
            "observed_in_window": observed,
            "baseline_comparable": not partial,
        }
        if partial:
            # WITHHELD, not merely flagged. The previous fix for a related failure added a prompt
            # instruction and left the number in place; the model used the number. A figure that
            # cannot support a claim should not be in front of the model at all - the same lesson
            # as the 2,000-character `summary` that filled with the model's own scratchpad.
            row["baseline_median"] = None
            row["pct_change"] = None
            row["robust_z"] = None
            # DIRECTION GOES TOO, and it was the last piece of this bug to be found. It is
            # `sign(observed - baseline_median)`, so on a partial window it is the same invalid
            # comparison wearing a word instead of a number - and words travel further. Observed:
            # every claim rendered "· decrease", including `drinking_events 4.00 · decrease`, so a
            # caregiver reading five decrease stamps would conclude decline from ten minutes of
            # someone cooking. C4 is already vacuous when `claim.direction is None` (the same
            # precedent as C3 on a null pct), so nothing downstream needs to change to accept it.
            row["direction"] = None
            row["why_withheld"] = (
                f"the window covers {state.today.observed_hours * 3600:.0f}s, so this total is "
                "not comparable to a full day's baseline")
        else:
            row["baseline_median"] = round(ev.baseline_median, 2)
            row["delta"] = round(ev.delta, 2)
            row["pct_change"] = round(ev.pct_change, 1) if pct_defined else None
            row["robust_z"] = round(ev.robust_z, 2)
            row["direction"] = ev.direction
        rows.append(row)

    return {
        "report_day": day.isoformat(),
        "subject": state.subject_role.value,
        "history_days": state.history_days,
        # THREE DECIMALS, AND SECONDS. At one decimal a 45-second clip is `0.0`, so Agent 4 was
        # told the observation window was ZERO HOURS - and it reasoned correctly from that: it
        # wrote "zero observed hours ... consistent with a system or sensor issue" and recommended
        # checking the equipment was powered. The model was not hallucinating, it was reading a
        # number this function had rounded away. Seconds are given as well because "0.012 hours"
        # is not a quantity anyone reasons about, and the whole point of a short window is that
        # absence claims are unwarranted rather than alarming.
        "observed_hours": round(state.today.observed_hours, 3),
        "observed_seconds": round(state.today.observed_hours * 3600.0, 1),
        "observation_is_reliable": bool(state.today.is_reliable()),
        # THE SINGLE FIELD THE PROMPT DISPATCHES ON. True whenever the window is shorter than the
        # 8 h `is_reliable` demands, in which case every feature row carries
        # `baseline_comparable: false` with its comparison numbers withheld. Measured on why this
        # has to exist: a 9m56s clip against a 22.9 h reference day is a 138x window mismatch, so
        # every total read as a collapse (-99% to -100%) while the actual rate was normal, and the
        # report recommended a healthcare contact for a person cooking and cleaning normally.
        "window_is_partial": partial,
        # `second_person_active` carries the only signal that fired for class 18
        # (`interacting_with_person`, F1 0.022): a second tracked person was in shot.
        # Exposing only the duration sum hides the cause; the model's best causal claim
        # without this is "social time fell", which C5 will match the row and which still
        # says nothing about WHY. With the flag it can name "no second person present".
        # None means we have no tracker log at all, not the absence of a second person -
        # the distinction matters at deployment time and the model is told so.
        "second_person_active": state.today.visitor_count > 0,
        "features": rows,
        "alerts": [
            {
                "kind": a.kind.value,
                "severity": a.severity.value,
                "rule": a.rule_name,
                "evidence_refs": [e.ref for e in a.evidence],
            }
            for a in state.alerts
        ],
    }


CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        # 400 AND 300, NOT 2000 AND 1000. A hosted model handed 4096 completion tokens filled the
        # summary with its own scratchpad - "This sentence is filler to meet length expectations",
        # "Now generating claims. Selecting six features...", "Claim one: walking duration" - and
        # C1-C5 did not catch a word of it, because the summary is PROSE and the checks verify
        # CLAIMS. Nothing verifies this field, so the only defence is leaving no room to ramble.
        # The claim sub-schema below is untouched, so every hallucination figure still stands.
        "summary": {"type": "string", "maxLength": 400},
        "recommendation": {"type": "string", "maxLength": 300},
        "escalate": {"type": "boolean"},
        "claims": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "text": {"type": "string", "maxLength": 500},
                    "evidence_ref": {
                        "type": "string",
                        "pattern": r"^feat:[a-z0-9_]+:\d{4}-\d{2}-\d{2}$",
                    },
                    "claimed_value": {"type": ["number", "null"]},
                    "claimed_pct_change": {"type": ["number", "null"]},
                    "direction": {
                        "type": ["string", "null"],
                        "enum": ["increase", "decrease", "unchanged", None],
                    },
                },
                "required": ["claim_id", "text", "evidence_ref"],
            },
        },
    },
    "required": ["summary", "recommendation", "escalate", "claims"],
}


def claim_schema(cfg: ReporterConfig) -> dict[str, Any]:
    """`CLAIM_SCHEMA` with `maxItems` bound to the config's claim budget.

    The grammar previously allowed 12 claims while `report()` kept only the first
    `max_claims`. Two costs, both silent: the model spent its token budget generating
    claims that were then discarded - which is how 9 of 116 constrained reports hit
    `max_new_tokens` and became parse failures - and the discarded claims were never
    counted anywhere, so a model emitting 12 mediocre claims scored the same as one
    emitting 6. Binding the grammar to the budget removes both.
    """
    schema = json.loads(json.dumps(CLAIM_SCHEMA))
    schema["properties"]["claims"]["maxItems"] = cfg.max_claims
    return schema

SYSTEM_PROMPT = """You write daily summaries for the caregiver of an older adult living \
alone. You are given ONLY structured behavioural statistics - never video or images.

Rules:
1. Every factual claim must cite one `evidence_ref` copied EXACTLY from the data given.
2. Copy `claimed_value` and `claimed_pct_change` verbatim from the cited row - same \
digits, same sign, same units in your prose. Do not recompute, round differently, or \
estimate. If a row's value is 820.0, your `text` MUST contain the literal "820.0" (or \
"820" - same number, same digits).
2a. If a row's `pct_change` is null, the percentage is UNDEFINED (its baseline is zero). \
Omit `claimed_pct_change` entirely and do not describe the change as a percentage.
3. `direction` must match the cited row's `direction` field, and your prose must agree \
with it. Never describe a decrease as an improvement or vice versa.
3a. If a row's `direction` is null, there is NO valid comparison for it. Omit `direction` \
entirely and describe only what was observed - do not call it a rise, a fall, or unchanged.
4. Report at most {max_claims} claims. When `window_is_partial` is false, prefer features \
with large |robust_z| and any feature named in an alert. When it is TRUE, rule 4a replaces \
this entirely.
4a. When `window_is_partial` is true, the camera watched a FRACTION of the day, so rows with \
`baseline_comparable: false` carry no pct_change and no robust_z and MUST NOT be claimed as \
decreases, collapses or absences - a ten-minute window cannot show that something did not \
happen today. Claim only rows with `observed_in_window: true`, describing what was observed \
(`claimed_value` = the observed value, `claimed_pct_change` omitted, prose describes \
occurrence, never decline). If an alert fired, report it - alerts are presence-based and \
remain valid at any window length.
5. Set `escalate` true only if an urgent or emergency alert is present.
6. If a value looks unremarkable, say so plainly. Do not invent concern.
7. `summary` is at most 400 characters and `recommendation` at most 300. Write only the \
finished prose a caregiver reads: no working out, no restating these instructions, no \
padding, and never the words "claim one" or "generating". Anything you would think through \
belongs nowhere in this object.
8. `observed_seconds` is how long the camera actually watched. When it is short, absence is \
UNOBSERVED, not measured: say a feature was not seen in the window rather than that it fell, \
and do not recommend clinical action on the strength of it. Rule 4a is what enforces this; \
this rule is how to talk about it in prose.
8a. When `window_is_partial` is true, `summary` must contain NO comparative word - not \
"limited", "low", "reduced", "little", "only", "minimal", "below" or any synonym. There is \
nothing to compare against, so such a word asserts something the data cannot support. State \
the window length and what was observed. "Housework was observed for 173.65 s during a 596 s \
window" is allowed; "showed limited housework activity" is not.

Beneath the DATA block is a FILLED TEMPLATE block. For each claim, copy the \
`claimed_value` digit-for-digit from the template's `value` cell into the JSON field of \
the same name. The template and the JSON row must agree on every number; C5 in the \
verifier catches any prose that does not.

Reply with ONE JSON object and nothing else, using EXACTLY these keys:

{{
  "summary": "<two sentences, at most 400 characters>",
  "recommendation": "<one sentence, at most 300 characters>",
  "escalate": <true or false>,
  "claims": [
    {{
      "claim_id": "c1",
      "text": "<one sentence about this feature, with the value digit-for-digit>",
      "evidence_ref": "<the evidence_ref string, copied character for character>",
      "claimed_value": <the template's value, copied digit-for-digit>,
      "claimed_pct_change": <the template's pct, or null if it is null>,
      "direction": "<the row's direction: increase, decrease or unchanged>"
    }}
  ]
}}

Output JSON only."""


def _template_rows(state: BehaviourState, cfg: ReporterConfig) -> list[dict[str, Any]]:
    """One pre-filled template row per feature, formatted as the model should COPY.

    This is the digit-for-digit cell the prompt asks the model to copy from. Writing it
    out so the digits are unmistakable ("claimed_value: 820.0", not just "820") reduces
    paraphrasing - "about 800", "1.8k", "over 700" - which is the bulk of C2 failures on
    the unconstrained arm and a non-trivial fraction of the constrained arm's residual.

    The C5 check added to the verifier reads this back against the model's prose, so a
    template that disagrees with the data is a bug - same field names, same JSON path.
    """
    payload = state_to_payload(state, cfg)
    return [
        {
            "evidence_ref": r["evidence_ref"],
            "feature": r["feature"],
            "value": r["observed_value"],
            "pct": r["pct_change"],
            "direction": r["direction"],
            # The flags rule 4a dispatches on. `pct: null` alone reads as "percentage undefined
            # (zero baseline)" under rule 2a, which is a DIFFERENT condition - a partial window
            # withholds the comparison entirely. Without the flags here the template could not
            # tell those two apart, and neither could anyone auditing the prompt.
            "observed_in_window": r["observed_in_window"],
            "baseline_comparable": r["baseline_comparable"],
        }
        for r in payload["features"]
    ]


def build_prompt(state: BehaviourState, cfg: ReporterConfig) -> str:
    payload = state_to_payload(state, cfg)
    return (
        SYSTEM_PROMPT.format(max_claims=cfg.max_claims)
        + "\n\nDATA:\n"
        + json.dumps(payload, indent=2)
        + "\n\nFILLED TEMPLATE - copy `value` and `pct` digit-for-digit into each claim:\n"
        + json.dumps(_template_rows(state, cfg), indent=2)
        + "\n\nJSON:"
    )


# ---------------------------------------------------------------------------
# Deterministic stubs - how faithfulness is measured without a GPU
# ---------------------------------------------------------------------------


def _rows(state: BehaviourState, cfg: ReporterConfig) -> list[dict[str, Any]]:
    payload = state_to_payload(state, cfg)
    rows = payload["features"]
    alerted = {ref for a in payload["alerts"] for ref in a["evidence_refs"]}

    def rank(r: dict[str, Any]) -> tuple:
        # An alerted feature outranks everything: an alert is presence-based and claimable at any
        # window length. After that, `robust_z` ranks only where it EXISTS - on a partial window
        # it is None by construction, and `abs(None)` is a TypeError that would take the reporter
        # down on exactly the days the gate was built for. Rule 4a's order applies instead:
        # observed before unobserved, because presence is evidence and absence is not.
        if r["evidence_ref"] in alerted:
            return (0, 0.0)
        if r["baseline_comparable"]:
            return (1, -abs(r["robust_z"] or 0.0))
        return (2, 0.0 if r["observed_in_window"] else 1.0)

    rows.sort(key=rank)
    return rows[: cfg.max_claims]


class FaithfulStubLLM:
    """Reports the data exactly. The measurement ceiling.

    Its hallucination rate MUST be 0.0. If it is not, the verifier has false positives
    and every rate it reports for a real model is inflated - so this stub is a control on
    the verifier, not merely a fixture.
    """

    def __init__(self, config: ReporterConfig | None = None) -> None:
        self.cfg = config or ReporterConfig()
        self._state: BehaviourState | None = None

    @property
    def name(self) -> str:
        return "stub-faithful"

    def bind(self, state: BehaviourState) -> None:
        self._state = state

    def generate(self, prompt: str, *, constrained: bool, schema: dict[str, Any]) -> str:
        if self._state is None:
            raise RuntimeError("call bind(state) before generate()")
        rows = _rows(self._state, self.cfg)

        def claim_text(r: dict[str, Any]) -> str:
            # On a partial window the stub follows the same rule the prompt gives a real model:
            # describe OCCURRENCE, never decline. The old text here - "showed a decrease at X
            # versus a baseline of Y" - is precisely the prose the 138x window mismatch produced,
            # and a control that commits the banned failure is not a control.
            if not r["baseline_comparable"]:
                if r["observed_in_window"]:
                    return (f"{r['feature'].replace('_', ' ')} was observed at "
                            f"{r['observed_value']} in the window.")
                return (f"{r['feature'].replace('_', ' ')} was not observed in the "
                        f"window (value 0.0).")
            verb = {"decrease": "showed a decrease", "increase": "showed an increase"}.get(
                r["direction"], "was stable at")
            return (f"{r['feature'].replace('_', ' ')} {verb} "
                    f"at {r['observed_value']} versus a baseline of {r['baseline_median']}.")

        # Rule 4a, applied to the control: on a partial window, absence is not claimable - but an
        # ALERT is, because alerts are presence-based and a fall observed in ten minutes is real.
        alerted_refs = {ref for a in self._state.alerts for e in a.evidence
                        for ref in (e.ref,)}
        claimable = [r for r in rows
                     if r["baseline_comparable"] or r["observed_in_window"]
                     or r["evidence_ref"] in alerted_refs]
        claims = [
            {
                "claim_id": f"c{i+1}",
                "text": claim_text(r),
                "evidence_ref": r["evidence_ref"],
                "claimed_value": r["observed_value"],
                "claimed_pct_change": r["pct_change"],
                "direction": r["direction"],
            }
            for i, r in enumerate(claimable)
        ]
        severe = any(
            a.severity in (AlertSeverity.EMERGENCY, AlertSeverity.URGENT)
            for a in self._state.alerts
        )
        return json.dumps(
            {
                "summary": f"{len(rows)} features reviewed for "
                           f"{self._state.report_day.isoformat()}.",
                "recommendation": "Continue routine monitoring."
                if not severe
                else "Contact the resident and review recent alerts.",
                "escalate": severe,
                "claims": claims,
            }
        )


# The six failure modes the verifier claims to catch. Named so test output says which
# corruption was applied rather than just "a claim failed".
CORRUPTIONS = (
    "fabricated_ref",       # C1: cites a feature that does not exist
    "invented_value",       # C2: right feature, wrong number
    "wrong_pct",            # C3: value right, percentage wrong
    "inverted_direction",   # C4: correct number, backwards story (the dangerous one)
    "prose_contradiction",  # C4: direction field right, prose says the opposite
    "sign_flipped_pct",     # C3/C4: quoted +50% when the delta is negative
)


class HallucinatingStubLLM:
    """Injects known corruptions at a controlled rate. The measurement floor.

    Every corruption is recorded in `self.injected`, so tests can assert the verifier
    caught *the specific fault that was inserted* rather than merely counting failures.
    A verifier that rejected everything would pass a count-based test; it cannot pass a
    per-corruption one.
    """

    def __init__(
        self,
        rate: float = 0.5,
        seed: int = 0,
        config: ReporterConfig | None = None,
        modes: tuple[str, ...] = CORRUPTIONS,
    ) -> None:
        self.cfg = config or ReporterConfig()
        self.rate = rate
        self.modes = modes
        self._rng = random.Random(seed)
        self._state: BehaviourState | None = None
        self.injected: dict[str, str] = {}

    @property
    def name(self) -> str:
        return f"stub-hallucinating@{self.rate:.2f}"

    def bind(self, state: BehaviourState) -> None:
        self._state = state
        self.injected = {}

    def generate(self, prompt: str, *, constrained: bool, schema: dict[str, Any]) -> str:
        if self._state is None:
            raise RuntimeError("call bind(state) before generate()")
        rows = _rows(self._state, self.cfg)
        claims: list[dict[str, Any]] = []

        for i, r in enumerate(rows):
            cid = f"c{i+1}"
            claim = {
                "claim_id": cid,
                "text": f"{r['feature'].replace('_', ' ')} was {r['observed_value']} "
                        f"against a baseline of {r['baseline_median']}, a "
                        f"{r['direction']}.",
                "evidence_ref": r["evidence_ref"],
                "claimed_value": r["observed_value"],
                "claimed_pct_change": r["pct_change"],
                "direction": r["direction"],
            }

            if self._rng.random() < self.rate:
                mode = self._rng.choice(self.modes)
                self.injected[cid] = mode
                day = self._state.report_day.isoformat()
                if mode == "fabricated_ref":
                    claim["evidence_ref"] = f"feat:stair_climbing_duration_s:{day}"
                elif mode == "invented_value":
                    claim["claimed_value"] = float(r["observed_value"]) * 1.9 + 13.0
                elif mode == "wrong_pct":
                    # A None pct means undefined (zero baseline); fabricating a number
                    # there is still a hallucination, and a worse one.
                    base = r["pct_change"]
                    claim["claimed_pct_change"] = (float(base) + 47.0
                                                   if base is not None else 47.0)
                elif mode == "inverted_direction":
                    flip = {"decrease": "increase", "increase": "decrease"}
                    claim["direction"] = flip.get(r["direction"], "increase")
                    claim["text"] = (
                        f"{r['feature'].replace('_', ' ')} improved to "
                        f"{r['observed_value']}, a clear increase."
                    )
                elif mode == "prose_contradiction":
                    opposite = "increased" if r["direction"] == "decrease" else "decreased"
                    claim["text"] = (
                        f"{r['feature'].replace('_', ' ')} {opposite} markedly "
                        f"to {r['observed_value']}."
                    )
                elif mode == "sign_flipped_pct":
                    # Only meaningful when the true change is a real decrease: flipping
                    # the sign of a ~0% change produces no arithmetic error at all, so
                    # the corruption would be a no-op mislabelled as a hallucination.
                    # Fall back to an outright wrong percentage in that case.
                    pct = float(r["pct_change"])
                    if r["direction"] == "decrease" and abs(pct) > 1.0:
                        claim["claimed_pct_change"] = abs(pct)
                        claim["text"] = (
                            f"{r['feature'].replace('_', ' ')} rose by {abs(pct):.1f}% today."
                        )
                    else:
                        mode = "wrong_pct"
                        self.injected[cid] = mode
                        claim["claimed_pct_change"] = pct + 47.0
            claims.append(claim)

        return json.dumps(
            {
                "summary": "Overall the resident appears to be doing well today.",
                "recommendation": "No action needed.",
                "escalate": False,
                "claims": claims,
            }
        )


SAMPLING_FLAGS = ("temperature", "top_p", "top_k", "typical_p", "epsilon_cutoff",
                  "eta_cutoff", "penalty_alpha", "repetition_penalty",
                  "no_repeat_ngram_size")


def force_greedy(model: Any) -> Any:
    """Strip sampling out of a model's `generation_config`, in place.

    This is not tidiness, it is the fix for a defect that inverted a headline result.

    `ReporterConfig.temperature = 0.0` documents greedy decoding, and the free-decoding
    path honoured it by passing `do_sample=False` to `generate()`. The grammar-constrained
    path passed only `max_new_tokens` to `outlines`, so it silently inherited
    Qwen2.5-Instruct's shipped `generation_config`: `do_sample=True, temperature=0.7,
    top_p=0.8, top_k=20`. The two arms of a constrained-vs-free comparison were therefore
    not decoding-matched - one was greedy, the other sampled at 0.7 - and the difference
    between them measured temperature, not grammar.

    It presented as irreproducibility. Across two runs of identical code on identical data
    the free arm was bit-identical (509 claims, 476 faithful, both times) while the
    constrained arm moved 498/457 to 504/452, and the significance verdict flipped from
    p = 0.29 ("indistinguishable") to p = 0.03 ("a real difference"). Only the sampled arm
    varied. Sampling also hurts precisely the thing being measured: at temperature 0.7 a
    digit of a copied value can be resampled, and C2 value mismatches are ~95% of all
    failures.

    `repetition_penalty` goes too, for a related reason. Qwen2.5-Instruct ships 1.05, which
    down-weights tokens that have already appeared - and this task is *deliberately*
    repetitive: six claims each re-quoting an evidence ref and a number verbatim from the
    payload. A penalty against repetition is a penalty against copying correctly. It applied
    equally to both arms, so it was never a confound between them, but it inflates the
    absolute rate the write-up headlines.
    """
    cfg = getattr(model, "generation_config", None)
    if cfg is None:
        return model
    cfg.do_sample = False
    for flag in SAMPLING_FLAGS:
        if hasattr(cfg, flag):
            setattr(cfg, flag, None)
    return model


class Qwen2_5Reporter:
    """Real backend: Qwen2.5-7B-Instruct with optional `outlines` grammar constraint.

    Imports torch/transformers lazily so this module stays importable (and testable) on a
    machine with neither. Nothing here runs during CPU testing; it is exercised on Kaggle
    with the staged offline weights.

    Weights are cached per (path, device, dtype) across instances. The hallucination
    benchmark needs one instance per decoding mode - `name` has to say which arm produced
    a row - and without the cache each instance re-materialised all 339 tensors from the
    Kaggle model mount. In the first real run the two loads dominated the cell: ~9,500 s
    total against ~620 s of actual generation.
    """

    _CACHE: dict[tuple[str, str, str], tuple[Any, Any]] = {}

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        config: ReporterConfig | None = None,
        max_memory: dict[int | str, str] | None = None,
    ) -> None:
        """`device` is passed straight to `device_map`, so `"auto"` shards across GPUs.

        That matters because of where this runs. Evaluation is offline on a single 96 GB
        card, where `"cuda"` is right and sharding would be pointless. Serving is online on
        T4 x2 - 16 GB each, 32 GB together - where a 9B model in 16-bit is ~18 GB of weights
        and only fits split, while `device_map="cuda"` pins it to one device and OOMs.

        `max_memory` bounds what `"auto"` may take per device, e.g.
        `{0: "11GiB", 1: "13GiB"}`. It is not tuning: the perception child process opens its
        own CUDA context for RTMO and OSNet on the same cards, so something has to leave it
        room. Sharding is a deployment decision and all three knobs are passed through
        rather than decided here.
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_memory = max_memory
        self.cfg = config or ReporterConfig()
        self._model = None
        self._tok = None
        self._outlines = None
        self._output_types: dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Derived from the weights on disk, never hardcoded.

        This used to return the literal `qwen2.5-7b-instruct` whatever `model_path` pointed
        at. That is fine until the moment somebody swaps the checkpoint - which is exactly
        when it matters, because `name` is what labels each row of the hallucination
        benchmark and what `report.model_name` shows in the UI. A comparison between two
        models where both arms are labelled with the first model's name is unreadable, and
        this project has already spent three two-hour reruns on a confound that a correct
        label would have made obvious.
        """
        stem = pathlib.Path(str(self.model_path)).name.lower() or "unknown-model"
        return f"{stem}({'constrained' if self.cfg.constrained else 'free'})"

    @staticmethod
    def canonical_id(name: str, *, space: str = "skeleton") -> int:
        """Typed wrapper around tsm_id — forces the space choice at the call site.

        The off-by-one between the two official trimmed id spaces is the single biggest
        landmine in the corpus. Reading an RGB label file with the skeleton map shifts every
        class by one and leaves a silent catch-all at 0. `canonical_id` forces the choice,
        and `canonical_name` refuses the RGB sentinel 0 so it can't be silently misinterpreted.
        """
        if space not in ("skeleton", "rgb"):
            raise ValueError(f"space must be 'skeleton' or 'rgb', got {space!r}")
        return tsm_id(name, space=space)

    @staticmethod
    def canonical_name(idx: int, *, space: str = "skeleton") -> str:
        """Inverse of canonical_id, refusing the RGB sentinel 0."""
        if space == "rgb" and idx == TSM_RGB_UNMATCHED:
            raise KeyError(
                "id 0 in the RGB space is `_name_to_int`'s unmatched-name fallback, not a "
                "class. A 0 in a released RGB label file means the name was not recognised."
            )
        return tsm_name(idx, space=space)

    def _load(self) -> None:
        if self._model is not None:
            return
        import time  # noqa: PLC0415

        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        key = (str(self.model_path), self.device, self.dtype)
        cached = type(self)._CACHE.get(key)
        if cached is not None:
            self._model, self._tok = cached
            print(f"[reporter] reusing already-loaded weights for {key[0]}", flush=True)
            return

        t0 = time.time()
        # bf16 needs compute capability 8.0. The training card (sm_120) has it; the SERVING
        # cards do not - T4 is sm_75 and P100 is sm_60 - so a bfloat16 load there is emulated
        # rather than native and is slower for no accuracy gain. Warned, not overridden:
        # silently switching to float16 would change the numerics under a measurement, and
        # Qwen weights are bf16-trained so fp16 has its own overflow risk. The operator
        # decides, with the fact in front of them.
        if self.dtype == "bfloat16" and torch.cuda.is_available() and not (
                getattr(torch.cuda, "is_bf16_supported", lambda: True)()):
            print(f"[reporter] WARNING: {torch.cuda.get_device_name(0)} has no native "
                  "bfloat16 (needs sm_80+). The load will be emulated and slow. Pass "
                  "dtype='float16' if you accept the numerics change.", flush=True)
        self._tok = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        kwargs: dict[str, Any] = {}
        if self.max_memory:
            # Only meaningful with device_map="auto". Needed because the perception child
            # process allocates its OWN CUDA context on the same cards: RTMO's onnxruntime
            # session plus that context is ~2 GB, and `auto` will otherwise fill both GPUs
            # to the brim and leave the child nothing. PyTorch's caching allocator does not
            # hand memory back, so the reservation has to be bounded up front rather than
            # hoped away.
            kwargs["max_memory"] = self.max_memory
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=getattr(torch, self.dtype),
            device_map=self.device,
            local_files_only=True,
            **kwargs,
        )
        self._model.eval()
        force_greedy(self._model)
        # Printed, because "the LLM cell took 2.6 hours" was not decomposable after the
        # fact - load and generation were indistinguishable in the log.
        placement = getattr(self._model, "hf_device_map", None)
        print(f"[reporter] loaded {self.model_path} in {time.time() - t0:.0f}s"
              f" (do_sample={self._model.generation_config.do_sample}"
              f"{f', devices={sorted(set(placement.values()))}' if placement else ''})",
              flush=True)
        self._assert_finite_logits()
        type(self)._CACHE[key] = (self._model, self._tok)

    def _assert_finite_logits(self) -> None:
        """One forward pass, checked for NaN/inf, before the model is trusted.

        Cheap tripwire for a dtype problem. Qwen weights are bf16-trained, the serving cards
        have no native bf16, and float16 has a narrower exponent range - so a cast that
        overflows is a real possibility. Without this check the symptom would appear ~50 s
        later as a garbage report, and with grammar-constrained decoding it would appear as
        the FSM picking whatever token a NaN-poisoned logit row happened to rank first. That
        failure would look like a bad model rather than a bad cast.

        A short prompt cannot prove the whole context length is safe, so this is a tripwire,
        not a guarantee. It catches the gross case for the price of one forward pass.
        """
        import torch  # noqa: PLC0415

        try:
            ids = self._tok("Report: walking duration fell to 820 s.",
                            return_tensors="pt").input_ids
            dev = next(self._model.parameters()).device
            with torch.no_grad():
                logits = self._model(ids.to(dev)).logits
            bad = int((~torch.isfinite(logits)).sum())
            if bad:
                raise RuntimeError(
                    f"{bad} non-finite values in the logits of a {self.dtype} forward pass. "
                    "The cast is overflowing; do not measure anything with this. Try "
                    "dtype='bfloat16' on an sm_80+ card, or float32 if memory allows."
                )
            print(f"[reporter] logit sanity ok: finite over {tuple(logits.shape)} "
                  f"in {self.dtype}", flush=True)
        except RuntimeError:
            raise
        except Exception as exc:                                   # noqa: BLE001
            # A tripwire that breaks the load is worse than one that reports it failed.
            print(f"[reporter] logit sanity check skipped: {type(exc).__name__}: {exc}",
                  flush=True)

    def _chat_text(self, prompt: str) -> str:
        """Apply the instruct chat template. Used by BOTH arms, which is the point.

        The free path applied it; the constrained path handed the raw prompt straight to
        `outlines`, which does not template a bare string. So the two arms of the comparison
        were sending *different text* to the model - one a properly delimited Qwen chat
        turn, the other a naked instruction block. An instruct model without its template
        follows instructions worse, and under a grammar it cannot express that by emitting
        malformed JSON: it emits well-formed JSON with worse content, which lands as C2
        value mismatches. That is exactly the direction the constrained arm was off.
        """
        return self._tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )

    def _output_type(self, schema: dict[str, Any]):
        """Compile the JSON-schema guide once per schema, not once per report.

        `outlines.types.json_schema(schema)` was constructed fresh inside every call, so any
        caching keyed on that object was defeated and the guide was rebuilt 116 times per
        arm. The constrained arm ran at 49.0 s/report against the free arm's 6.5 - a 7.5x
        gap on a 7B model that has no business being that slow - and turned this notebook
        into a two-hour job.
        """
        import outlines  # noqa: PLC0415

        key = json.dumps(schema, sort_keys=True)
        if key not in self._output_types:
            self._output_types[key] = outlines.types.json_schema(schema)
        return self._output_types[key]

    def generate(self, prompt: str, *, constrained: bool, schema: dict[str, Any]) -> str:
        self._load()
        text = self._chat_text(prompt)
        if constrained:
            try:
                import outlines  # noqa: PLC0415

                if self._outlines is None:
                    self._outlines = outlines.from_transformers(self._model, self._tok)
                # do_sample=False belongs here as well as in generation_config: which one
                # `outlines` honours is its business, and this measurement cannot depend on
                # guessing. Passing both means the arm is greedy under either behaviour.
                return str(
                    self._outlines(text, self._output_type(schema),
                                   max_new_tokens=self.cfg.max_new_tokens,
                                   do_sample=self.cfg.temperature > 0)
                )
            except ImportError:
                # Falling back silently would make the constrained-vs-free comparison a
                # lie: both arms would be unconstrained while the table claimed otherwise.
                raise RuntimeError(
                    "constrained=True requires `outlines`, which is not installed. "
                    "Install it or pass ReporterConfig(constrained=False) and label the "
                    "results as the unconstrained arm."
                ) from None

        import torch  # noqa: PLC0415

        inputs = self._tok(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=self.cfg.temperature > 0,
                temperature=self.cfg.temperature or None,
                pad_token_id=self._tok.eos_token_id,
            )
        return self._tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# The reporter itself
# ---------------------------------------------------------------------------


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Inflectional variants only. "improved" is NOT mapped to "increase": deciding that an
# increase in inactivity is an improvement is interpretation, and the verifier's C4 check
# exists precisely to catch that judgement being made wrongly. Repair may normalise how a
# claim is written; it may never decide what the claim means.
_DIRECTION_FORMS = {
    "increase": "increase", "increased": "increase", "increases": "increase",
    "increasing": "increase", "decrease": "decrease", "decreased": "decrease",
    "decreases": "decrease", "decreasing": "decrease", "unchanged": "unchanged",
    "no change": "unchanged", "none": None, "null": None, "": None,
}


def _coerce_number(value: Any) -> Any:
    """`"1,800.0"` / `"-12.5 %"` -> float. Anything ambiguous is returned unchanged.

    Returning the original on failure is deliberate: silently dropping an unparseable
    quoted value would make C2 vacuous and turn a fabricated number into a pass, which is
    the opposite of what this measurement is for.
    """
    if value is None or isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip().replace(",", "").replace("%", "").strip()
    if text.lower() in ("", "null", "none", "n/a", "nan"):
        return None
    match = _NUM_RE.fullmatch(text)
    return float(match.group()) if match else value


def repair_claim(item: dict[str, Any], position: int) -> dict[str, Any]:
    """Coerce FORMAT so a schema-invalid claim can be scored. Never alters content.

    Permitted: whitespace/backtick/quote stripping, `"1800"` -> `1800.0`, `"Decreased"` ->
    `"decrease"`, a non-string `claim_id` -> its string form, a missing `claim_id` -> its
    position, `YYYY/MM/DD` -> `YYYY-MM-DD` inside an otherwise well-formed ref.

    Forbidden, and this is the line that keeps the salvaged row honest: inventing an
    `evidence_ref`, changing any numeric value, rewriting `text`, or inferring a direction
    the model did not state. A claim this function cannot fix stays broken and is reported
    as unrepairable rather than quietly made to pass.
    """
    out = dict(item)

    cid = out.get("claim_id")
    out["claim_id"] = f"c{position + 1}" if cid in (None, "") else str(cid).strip()

    if isinstance(out.get("text"), str):
        out["text"] = out["text"].strip()[:500]

    ref = out.get("evidence_ref")
    if isinstance(ref, str):
        ref = ref.strip().strip("`").strip("'\"").strip()
        for prefix in ("evidence_ref:", "evidence_ref =", "ref:"):
            if ref.lower().startswith(prefix):
                ref = ref[len(prefix):].strip()
        # Date separator only - the feature name and the date itself are left alone, so a
        # ref naming a feature that does not exist still fails C1 as a fabrication.
        head, sep, tail = ref.rpartition(":")
        if sep and re.fullmatch(r"\d{4}[/.]\d{2}[/.]\d{2}", tail):
            ref = f"{head}:{tail.replace('/', '-').replace('.', '-')}"
        out["evidence_ref"] = ref

    for key in ("claimed_value", "claimed_pct_change"):
        if key in out:
            out[key] = _coerce_number(out[key])

    direction = out.get("direction")
    if isinstance(direction, str):
        key = direction.strip().lower()
        if key in _DIRECTION_FORMS:
            out["direction"] = _DIRECTION_FORMS[key]

    return out


def _rejection_reasons(exc: Exception, item: dict[str, Any]) -> list[str]:
    """`field: reason` lines for one schema-rejected claim.

    Pydantic's `errors()` names the offending field and rule; the exception type alone does
    not. The free-decoding arm rejected 245 of 245 claims reporting only
    "ValidationError", which is a measurement with no diagnosis attached - the whole point
    of scoring the unconstrained arm is to say WHAT it got wrong.
    """
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return [f"{type(exc).__name__}: {exc}"[:200]]
    out: list[str] = []
    try:
        for err in errors():
            field_name = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
            reason = err.get("type", "invalid")
            if field_name in item:
                got = repr(item[field_name])[:60]
            else:
                got = "<missing>"
            out.append(f"{field_name}: {reason} (got {got})")
    except Exception:  # noqa: BLE001 - never let diagnostics break the run
        return [f"{type(exc).__name__}: {exc}"[:200]]
    return out


def _extract_json(raw: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Needed only for the unconstrained arm - a grammar-constrained model returns bare
    JSON. Unconstrained models wrap it in prose or fences, and how well that recovery
    works is itself part of the comparison, so failures here are counted rather than
    swallowed.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
    start = raw.find("{")
    if start < 0:
        raise ValueError("no JSON object in response")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(raw[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(raw[start:i + 1])
    # Distinguishable on purpose: an unterminated object means the response was CUT OFF
    # (max_new_tokens), which is a budget problem, whereas "no JSON object" means the model
    # answered in prose, which is a compliance problem. Reporting them as one number hides
    # a fix that costs nothing behind a finding about the model.
    raise ValueError("unterminated JSON object in response (truncated generation)")


@dataclass
class ReportOutcome:
    """A generated report plus its verification, and what went wrong producing it."""

    report: CaregiverReport
    verification: VerificationReport
    parse_failed: bool = False
    dropped_claims: int = 0
    """Claims the LLM emitted that could not even be parsed into the Claim schema.

    Counted, never silently discarded: dropping malformed claims would flatter the
    unconstrained arm by removing its worst output before scoring."""
    n_emitted_claims: int = 0
    """Claims the model actually emitted (within the max_claims budget), parseable or not.

    The honest denominator. `verification.n_claims` counts only claims that survived
    Pydantic, so a rate computed from it silently excludes a model's worst output - which
    is exactly the bias `dropped_claims` was introduced to expose, and the free-decoding
    arm is where it bites: 245 emitted, 0 scorable, and a rate of `nan`."""
    repaired_claims: int = 0
    """Claims that only became scorable after `repair_claim` fixed their FORMAT.

    Reported separately so the salvaged row can never be mistaken for native compliance:
    "39.9% unfaithful after repairing 245 claims" and "39.9% unfaithful as emitted" are
    different results and the table must be able to tell them apart."""
    rejections: list[str] = field(default_factory=list)
    """`field: reason` per schema-rejected claim, so a total rejection is diagnosable.

    Recording only the exception type produced 245/245 rejections with no way to tell a
    wrong ref format from a non-string claim_id - a dead end that needed a second 2.6-hour
    GPU run to investigate."""
    notes: list[str] = field(default_factory=list)

    @property
    def hallucination_rate(self) -> float:
        """Fraction of SCORABLE claims failing verification (the verifier's view)."""
        return self.verification.hallucination_rate

    @property
    def unusable_rate(self) -> float:
        """Fraction of EMITTED claims that are not both well-formed and faithful.

        The end-to-end number a deployment cares about: a claim rejected by the schema is
        no more usable to a caregiver than one that is fluent and false."""
        if self.n_emitted_claims == 0:
            return float("nan")
        return 1.0 - (self.verification.n_faithful / self.n_emitted_claims)


class CaregiverReporter:
    """BehaviourState -> LLM -> Claim objects -> deterministic verification."""

    def __init__(
        self,
        llm: ReportLLM,
        config: ReporterConfig | None = None,
        verifier: FaithfulnessVerifier | None = None,
    ) -> None:
        self.llm = llm
        self.cfg = config or ReporterConfig()
        self.verifier = verifier or FaithfulnessVerifier()

    def report(self, state: BehaviourState) -> ReportOutcome:
        if hasattr(self.llm, "bind"):
            self.llm.bind(state)  # stubs need the state; a real LLM gets it in the prompt

        prompt = build_prompt(state, self.cfg)
        notes: list[str] = []
        rejections: list[str] = []
        parse_failed = False
        payload: dict[str, Any] = {}

        try:
            raw = self.llm.generate(
                prompt, constrained=self.cfg.constrained, schema=claim_schema(self.cfg)
            )
            payload = _extract_json(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            # An unparseable response is a total failure for that day, and it must appear
            # in the metric as such rather than as "no claims, therefore no lies".
            parse_failed = True
            notes.append(f"response did not parse: {exc}")

        claims: list[Claim] = []
        dropped = 0
        repaired = 0
        emitted = payload.get("claims") or []
        if not isinstance(emitted, list):
            # A model that returns claims as a dict or a string is not "0 claims" - it
            # failed to follow the shape, and that must not read as a clean report.
            notes.append(f"'claims' was {type(emitted).__name__}, not a list")
            emitted = []
        for item in emitted[: self.cfg.max_claims]:
            if not isinstance(item, dict):
                dropped += 1
                rejections.append(f"claims[]: not an object ({type(item).__name__})")
                continue
            try:
                claims.append(Claim(**item))
                continue
            except Exception as exc:  # noqa: BLE001 - pydantic ValidationError
                first_error = exc

            if self.cfg.salvage:
                # Second chance at FORMAT only. A repaired claim is still verified against
                # the data by the same C1-C4 checks, so this can move a claim from
                # "unscorable" to "scored" but never from "false" to "faithful".
                try:
                    claims.append(Claim(**repair_claim(item, len(claims) + dropped)))
                    repaired += 1
                    continue
                except Exception:  # noqa: BLE001 - unrepairable; fall through to reject
                    pass

            dropped += 1
            # Field-level detail, not just the exception type. The free-decoding arm
            # rejected 245/245 claims and the log said only "ValidationError", which
            # made the cause unknowable without another 2.6-hour GPU run.
            for line in _rejection_reasons(first_error, item):
                rejections.append(line)
            notes.append(f"claim {item.get('claim_id', '?')} rejected by schema: "
                         f"{type(first_error).__name__}")

        verification = self.verifier.verify_all(claims, state)

        def _prose(field: str, limit: int) -> str:
            """Enforce the schema's own length, and SAY when it had to be enforced.

            Cropping silently is the thing this class refuses to do everywhere else: a hosted model
            wrote 2,000 characters of scratchpad into `summary` and a quiet `[:400]` would have left
            a plausible-looking two sentences with no sign that the rest existed. Nothing verifies
            this field - C1-C5 check claims, not prose - so the violation has to be reported.
            """
            raw = str(payload.get(field, ""))
            if len(raw) > limit:
                notes.append(f"{field} was {len(raw)} chars, over the {limit} the schema allows; "
                             f"truncated. Nothing verifies this field, so over-length prose here "
                             f"is usually the model's own working-out leaking into it.")
            return raw[:limit]

        report = CaregiverReport(
            report_id=f"rep-{state.subject_role.value}-{state.report_day.isoformat()}",
            subject_role=state.subject_role,
            report_day=state.report_day,
            generated_at=datetime.now(),
            summary=_prose("summary", 400),
            claims=claims,
            recommendation=_prose("recommendation", 300),
            escalate=bool(payload.get("escalate", False)),
            verifications=list(verification.results),
            model_name=self.llm.name,
            constrained_decoding=self.cfg.constrained,
        )
        return ReportOutcome(
            report=report,
            verification=verification,
            parse_failed=parse_failed,
            dropped_claims=dropped,
            repaired_claims=repaired,
            n_emitted_claims=len(emitted[: self.cfg.max_claims]),
            rejections=rejections,
            notes=notes,
        )

    def verified_summary(self, outcome: ReportOutcome) -> str:
        """Render only the claims that passed verification, flagging what was withheld.

        This is what a caregiver would actually see. Unfaithful claims are removed rather
        than shown with a warning: a caveat next to a confident false statement is not
        protection, because the statement is what gets remembered.
        """
        rep = outcome.report
        ok = {v.claim_id for v in rep.verifications if v.is_faithful}
        lines = [f"Daily summary for {rep.report_day.isoformat()} ({rep.subject_role.value})"]
        if rep.escalate:
            lines.append("** ESCALATION RECOMMENDED **")
        for c in rep.claims:
            if c.claim_id in ok:
                lines.append(f"  - {c.text}")
        withheld = len(rep.claims) - len(ok)
        if withheld:
            lines.append(
                f"  [{withheld} claim(s) withheld: failed automated verification]"
            )
        if rep.recommendation:
            lines.append(f"Recommendation: {rep.recommendation}")
        return "\n".join(lines)


__all__ = [
    "REPORTER_VERSION",
    "REPORTABLE_FEATURES",
    "CORRUPTIONS",
    "CLAIM_SCHEMA",
    "claim_schema",
    "ReporterConfig",
    "ReportLLM",
    "ReportOutcome",
    "CaregiverReporter",
    "FaithfulStubLLM",
    "HallucinatingStubLLM",
    "Qwen2_5Reporter",
    "SAMPLING_FLAGS",
    "build_prompt",
    "force_greedy",
    "repair_claim",
    "state_to_payload",
]
