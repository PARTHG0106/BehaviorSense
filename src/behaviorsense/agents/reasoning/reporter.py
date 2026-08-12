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
    "longest_inactive_block_s",
    "sleep_proxy_duration_s",
    "room_transitions",
    "housework_duration_s",
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
        rows.append(
            {
                "feature": name,
                "evidence_ref": ev.ref,
                "observed_value": round(ev.observed_value, 2),
                "baseline_median": round(ev.baseline_median, 2),
                "delta": round(ev.delta, 2),
                "pct_change": round(ev.pct_change, 1) if pct_defined else None,
                "robust_z": round(ev.robust_z, 2),
                "direction": ev.direction,
            }
        )

    return {
        "report_day": day.isoformat(),
        "subject": state.subject_role.value,
        "history_days": state.history_days,
        "observed_hours": round(state.today.observed_hours, 1),
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
        "summary": {"type": "string", "maxLength": 2000},
        "recommendation": {"type": "string", "maxLength": 1000},
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
2. Copy `claimed_value` and `claimed_pct_change` verbatim from the cited row. Do not \
recompute, round differently, or estimate.
2a. If a row's `pct_change` is null, the percentage is UNDEFINED (its baseline is zero). \
Omit `claimed_pct_change` entirely and do not describe the change as a percentage.
3. `direction` must match the cited row's `direction` field, and your prose must agree \
with it. Never describe a decrease as an improvement or vice versa.
4. Report at most {max_claims} claims. Prefer features with large |robust_z| and any \
feature named in an alert.
5. Set `escalate` true only if an urgent or emergency alert is present.
6. If a value looks unremarkable, say so plainly. Do not invent concern.

Reply with ONE JSON object and nothing else, using EXACTLY these keys:

{{
  "summary": "<two sentences>",
  "recommendation": "<one sentence>",
  "escalate": <true or false>,
  "claims": [
    {{
      "claim_id": "c1",
      "text": "<one sentence about this feature>",
      "evidence_ref": "<the evidence_ref string, copied character for character>",
      "claimed_value": <the row's observed_value>,
      "claimed_pct_change": <the row's pct_change, or null if it is null>,
      "direction": "<the row's direction: increase, decrease or unchanged>"
    }}
  ]
}}

Output JSON only."""


def build_prompt(state: BehaviourState, cfg: ReporterConfig) -> str:
    payload = state_to_payload(state, cfg)
    return (
        SYSTEM_PROMPT.format(max_claims=cfg.max_claims)
        + "\n\nDATA:\n"
        + json.dumps(payload, indent=2)
        + "\n\nJSON:"
    )


# ---------------------------------------------------------------------------
# Deterministic stubs - how faithfulness is measured without a GPU
# ---------------------------------------------------------------------------


def _rows(state: BehaviourState, cfg: ReporterConfig) -> list[dict[str, Any]]:
    payload = state_to_payload(state, cfg)
    rows = payload["features"]
    alerted = {ref for a in payload["alerts"] for ref in a["evidence_refs"]}
    rows.sort(key=lambda r: (r["evidence_ref"] not in alerted, -abs(r["robust_z"])))
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
        claims = [
            {
                "claim_id": f"c{i+1}",
                "text": f"{r['feature'].replace('_', ' ')} showed a "
                        f"{'decrease' if r['direction'] == 'decrease' else 'increase' if r['direction'] == 'increase' else 'stable reading'} "
                        f"at {r['observed_value']} versus a baseline of {r['baseline_median']}.",
                "evidence_ref": r["evidence_ref"],
                "claimed_value": r["observed_value"],
                "claimed_pct_change": r["pct_change"],
                "direction": r["direction"],
            }
            for i, r in enumerate(rows)
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
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.cfg = config or ReporterConfig()
        self._model = None
        self._tok = None
        self._outlines = None

    @property
    def name(self) -> str:
        return f"qwen2.5-7b-instruct({'constrained' if self.cfg.constrained else 'free'})"

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
        self._tok = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=getattr(torch, self.dtype),
            device_map=self.device,
            local_files_only=True,
        )
        self._model.eval()
        # Printed, because "the LLM cell took 2.6 hours" was not decomposable after the
        # fact - load and generation were indistinguishable in the log.
        print(f"[reporter] loaded {self.model_path} in {time.time() - t0:.0f}s",
              flush=True)
        type(self)._CACHE[key] = (self._model, self._tok)

    def generate(self, prompt: str, *, constrained: bool, schema: dict[str, Any]) -> str:
        self._load()
        if constrained:
            try:
                import outlines  # noqa: PLC0415

                if self._outlines is None:
                    self._outlines = outlines.from_transformers(self._model, self._tok)
                return str(
                    self._outlines(prompt, outlines.types.json_schema(schema),
                                   max_new_tokens=self.cfg.max_new_tokens)
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

        messages = [{"role": "user", "content": prompt}]
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
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

        report = CaregiverReport(
            report_id=f"rep-{state.subject_role.value}-{state.report_day.isoformat()}",
            subject_role=state.subject_role,
            report_day=state.report_day,
            generated_at=datetime.now(),
            summary=str(payload.get("summary", ""))[:2000],
            claims=claims,
            recommendation=str(payload.get("recommendation", ""))[:1000],
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
    "build_prompt",
    "repair_claim",
    "state_to_payload",
]
