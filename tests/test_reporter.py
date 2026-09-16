"""Tests for Agent 4's reporter and the end-to-end faithfulness measurement.

What makes these tests non-vacuous
----------------------------------
A verifier that rejected every claim would score a perfect "catches hallucinations".
A verifier that accepted every claim would score a perfect "does not flag honest reports".
Only measuring BOTH ends, on the same pipeline and thresholds, distinguishes a working
verifier from a broken one - so R1 (faithful stub -> 0% hallucination) and R2/R3
(corrupted stub -> caught, per-corruption) are a matched pair and neither means anything
alone.

R3 goes further and asserts the verifier caught the SPECIFIC fault injected into each
claim. Counting failures cannot distinguish "detected the inverted direction" from
"rejected this claim for an unrelated reason", and the inverted-direction case is the one
that matters clinically: a fluent report describing a decline as an improvement.

Run: python tests/test_reporter.py
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behaviorsense.agents.behaviour import BehaviourAnalyzer  # noqa: E402
from behaviorsense.agents.reasoning.reporter import (  # noqa: E402
    CLAIM_SCHEMA,
    CORRUPTIONS,
    SAMPLING_FLAGS,
    CaregiverReporter,
    FaithfulStubLLM,
    HallucinatingStubLLM,
    ReporterConfig,
    _extract_json,
    build_prompt,
    claim_schema,
    force_greedy,
    repair_claim,
    state_to_payload,
)
from behaviorsense.agents.reasoning.verifier import FaithfulnessVerifier  # noqa: E402
from behaviorsense.schemas import Claim, DailyFeatures, Role  # noqa: E402

D0 = date(2026, 2, 1)


def make_day(day: date, walking_s: float = 1800.0, meals: int = 3,
             social_s: float = 1200.0, meds: int = 1) -> DailyFeatures:
    return DailyFeatures(
        subject_role=Role.RESIDENT,
        day=day,
        walking_duration_s=walking_s,
        walking_bouts=12,
        mean_bout_duration_s=walking_s / 12,
        room_transitions=24,
        sit_to_stand_count=14,
        mean_sit_to_stand_duration_s=2.2,
        sitting_duration_s=18000.0,
        lying_duration_s=28000.0,
        standing_duration_s=3000.0,
        longest_inactive_block_s=7200.0,
        meal_events=meals,
        eating_duration_s=meals * 1200.0,
        drinking_events=6,
        cooking_duration_s=1500.0,
        medication_events=meds,
        tv_duration_s=7000.0,
        reading_duration_s=1800.0,
        phone_events=3,
        social_interaction_duration_s=social_s,
        visitor_count=1,
        housework_duration_s=1200.0,
        observed_hours=23.5,
        tracking_coverage=0.97,
    )


def build_state(decline: bool = True):
    """21 stable days, then either a large genuine drop or another normal day."""
    analyzer = BehaviourAnalyzer()
    pattern = [1800.0, 1750.0, 1850.0, 1900.0, 1700.0, 1820.0, 1780.0]
    for i in range(21):
        analyzer.analyze_day(make_day(D0 + timedelta(days=i), walking_s=pattern[i % 7]))
    final = make_day(
        D0 + timedelta(days=21),
        walking_s=700.0 if decline else 1810.0,
        meals=1 if decline else 3,
        social_s=200.0 if decline else 1200.0,
    )
    return analyzer.analyze_day(final)


STATE = build_state(decline=True)
CALM = build_state(decline=False)


def test_r0_prompt_contains_resolvable_refs_and_no_pixels():
    """The LLM's entire view: structured stats with exact refs, never imagery."""
    payload = state_to_payload(STATE, ReporterConfig())
    prompt = build_prompt(STATE, ReporterConfig())

    assert payload["features"], "no features exposed to the model"
    from behaviorsense.agents.reasoning.verifier import FaithfulnessVerifier

    index = FaithfulnessVerifier().build_index(STATE)
    for row in payload["features"]:
        assert row["evidence_ref"] in index, f"prompt offers unresolvable {row['evidence_ref']}"

    banned = ("frame", "pixel", "image", "video", "keypoint", "skeleton", "bounding",
              "track_id", "confidence")
    # Scan the DATA PAYLOAD, not the whole prompt. The instructions legitimately contain
    # "never video or images" - that is the boundary being stated, not crossed, and a
    # blanket keyword scan cannot tell a prohibition from a leak. What must be free of
    # perception concepts is the evidence the model reasons over.
    serialised = json.dumps(payload).lower()
    for word in banned:
        assert word not in serialised, f"data payload leaks perception concept: {word!r}"
    assert "never video or images" in prompt.lower(), (
        "the prompt no longer states the no-perception boundary"
    )
    assert "walking_duration_s" in prompt
    print(f"  R0 prompt = {len(prompt)} chars, {len(payload['features'])} features, "
          f"{len(payload['alerts'])} alerts, all refs resolvable, no pixel concepts")


def test_r1_faithful_report_has_zero_hallucinations():
    """The measurement ceiling. If this fails, the verifier has false positives."""
    reporter = CaregiverReporter(FaithfulStubLLM())
    out = reporter.report(STATE)

    assert not out.parse_failed and out.dropped_claims == 0, out.notes
    assert out.report.claims, "no claims generated"
    assert out.hallucination_rate == 0.0, (
        f"honest report flagged at {out.hallucination_rate:.1%}; "
        f"notes={[n for v in out.verification.results for n in v.notes][:4]}"
    )
    assert all(v.is_faithful for v in out.report.verifications)
    rendered = reporter.verified_summary(out)
    assert "withheld" not in rendered
    print(f"  R1 {len(out.report.claims)} honest claims -> hallucination rate "
          f"{out.hallucination_rate:.1%}, {len(rendered.splitlines())} lines shown")


def test_r2_corrupted_report_is_caught():
    """The measurement floor, at full corruption: every claim is a lie and none survive."""
    reporter = CaregiverReporter(HallucinatingStubLLM(rate=1.0, seed=1))
    out = reporter.report(STATE)

    assert out.hallucination_rate == 1.0, (
        f"only {out.hallucination_rate:.0%} of fully-corrupted claims were caught"
    )
    rendered = reporter.verified_summary(out)
    for c in out.report.claims:
        assert c.text not in rendered, "an unfaithful claim reached the caregiver output"
    assert "withheld" in rendered
    print(f"  R2 {len(out.report.claims)}/{len(out.report.claims)} corrupted claims caught; "
          f"caregiver view shows 0 claims + withheld notice")


def test_r3_each_corruption_type_is_detected_specifically():
    """Every injected corruption must be caught, and by the RIGHT check.

    This is the test a blanket-reject verifier cannot pass: it maps each corruption to the
    verification field that must be false, so catching a fabricated ref via a value
    mismatch counts as a failure, not a success.
    """
    # Which check must fire. A set, because one corruption can be legitimately caught by
    # either of two checks: quoting +50% when the true change is -50% is simultaneously a
    # wrong percentage AND a contradictory story. Demanding one specific field there would
    # fail a verifier that caught the fault by the other equally-valid route.
    expected_fields = {
        "fabricated_ref": {"ref_exists"},
        "invented_value": {"value_matches"},
        "wrong_pct": {"pct_matches"},
        "inverted_direction": {"direction_consistent"},
        "prose_contradiction": {"direction_consistent"},
        "sign_flipped_pct": {"pct_matches", "direction_consistent"},
    }
    seen: dict[str, int] = {}
    misses: list[str] = []

    for seed in range(24):
        llm = HallucinatingStubLLM(rate=1.0, seed=seed)
        out = CaregiverReporter(llm).report(STATE)
        by_id = {v.claim_id: v for v in out.report.verifications}
        for cid, mode in llm.injected.items():
            v = by_id.get(cid)
            if v is None:
                continue  # claim rejected at schema level; counted in dropped_claims
            seen[mode] = seen.get(mode, 0) + 1
            if v.is_faithful:
                misses.append(f"{mode} (claim {cid}, seed {seed}) passed verification")
                continue
            failed = {f for f in ("ref_exists", "value_matches", "pct_matches",
                                  "direction_consistent") if getattr(v, f) is False}
            if not (failed & expected_fields[mode]):
                misses.append(
                    f"{mode} caught by the wrong check: expected one of "
                    f"{sorted(expected_fields[mode])}, failed {sorted(failed)} "
                    f"(seed {seed}, notes={v.notes[:1]})"
                )

    assert not misses, "\n".join(misses[:5])
    uncovered = set(CORRUPTIONS) - set(seen)
    assert not uncovered, f"corruptions never exercised: {uncovered}"
    print("  R3 " + ", ".join(f"{m}x{n}" for m, n in sorted(seen.items())) + " - all "
          "caught by the expected check")


def test_r4_partial_corruption_rate_is_tracked_proportionally():
    """A 50%-corrupt model must score near 50%, not 0% or 100%.

    Rate fidelity is what makes the constrained-vs-unconstrained table meaningful: a
    verifier that saturates at either end cannot rank two models.
    """
    rows = []
    for rate in (0.0, 0.5, 1.0):
        caught = injected = 0
        for seed in range(30):
            llm = HallucinatingStubLLM(rate=rate, seed=seed)
            out = CaregiverReporter(llm).report(STATE)
            injected += len(llm.injected)
            caught += sum(1 for v in out.report.verifications if not v.is_faithful)
        rows.append((rate, injected, caught))

    r0, r50, r100 = rows
    assert r0[2] == 0, f"{r0[2]} false positives at corruption rate 0"
    assert r100[1] == r100[2], f"missed {r100[1] - r100[2]} of {r100[1]} at rate 1.0"
    assert r50[1] == r50[2], f"missed {r50[1] - r50[2]} of {r50[1]} at rate 0.5"
    assert 0 < r50[1] < r100[1], "rate 0.5 did not inject an intermediate amount"
    print("  R4 " + "; ".join(f"rate {r:.1f}: {i} injected / {c} caught"
                              for r, i, c in rows))


def test_r5_calm_day_produces_no_escalation():
    """A normal day must not be dramatised - the false-alarm direction for the LLM."""
    out = CaregiverReporter(FaithfulStubLLM()).report(CALM)
    assert out.hallucination_rate == 0.0
    assert not out.report.escalate, "escalated on a day with no severe alert"
    severe = [a.kind.value for a in CALM.alerts]
    print(f"  R5 calm day: escalate={out.report.escalate}, alerts={severe or 'none'}, "
          f"{len(out.report.claims)} claims all faithful")


def test_r6_unparseable_and_malformed_responses_are_counted_not_hidden():
    """Garbage in must be recorded as failure, never as a clean empty report."""

    class BrokenLLM:
        name = "broken"

        def generate(self, prompt, *, constrained, schema):
            return "I'm sorry, I cannot help with that."

    out = CaregiverReporter(BrokenLLM()).report(STATE)
    assert out.parse_failed and out.notes, "unparseable response was silently accepted"
    assert out.report.claims == []

    class MalformedClaimLLM:
        name = "malformed"

        def generate(self, prompt, *, constrained, schema):
            return json.dumps({
                "summary": "s", "recommendation": "r", "escalate": False,
                "claims": [
                    {"claim_id": "c1", "text": "bad ref", "evidence_ref": "walking went down"},
                    {"claim_id": "c2", "text": "no ref at all"},
                ],
            })

    out2 = CaregiverReporter(MalformedClaimLLM()).report(STATE)
    assert out2.dropped_claims == 2, f"expected 2 dropped, got {out2.dropped_claims}"
    assert out2.report.claims == []
    print(f"  R6 prose response -> parse_failed=True; 2 schema-invalid claims -> "
          f"dropped_claims={out2.dropped_claims} (not silently discarded)")


def test_r7_json_extraction_survives_real_model_output_shapes():
    """The unconstrained arm needs this; nested braces and strings must not break it."""
    obj = {"summary": "a}b{c", "claims": [{"claim_id": "c1", "nested": {"x": [1, 2]}}]}
    body = json.dumps(obj)
    for wrapped in (body, f"```json\n{body}\n```", f"Here you go:\n{body}\nHope that helps!",
                    f"```\n{body}\n```"):
        assert _extract_json(wrapped) == obj, f"failed on: {wrapped[:40]}"
    for bad in ("no json here", '{"unterminated": '):
        try:
            _extract_json(bad)
            raise AssertionError(f"accepted invalid input: {bad!r}")
        except ValueError:
            pass
    print("  R7 JSON extracted through fences/prose/nested braces; invalid input raises")


def test_r8_claim_schema_matches_the_pydantic_contract():
    """The grammar handed to `outlines` must not permit claims Pydantic will reject.

    A drift between the two would show up as a high dropped_claims count on the real
    model and look like model failure rather than our bug.
    """
    props = CLAIM_SCHEMA["properties"]["claims"]["items"]["properties"]
    from behaviorsense.schemas import Claim

    fields = set(Claim.model_fields)
    assert set(props) <= fields, f"grammar has fields Claim lacks: {set(props) - fields}"
    assert props["evidence_ref"]["pattern"] == Claim.model_fields["evidence_ref"].metadata[0].pattern
    assert set(props["direction"]["enum"]) == {"increase", "decrease", "unchanged", None}
    for req in CLAIM_SCHEMA["properties"]["claims"]["items"]["required"]:
        assert req in fields
    print(f"  R8 grammar and Claim agree on {len(props)} fields, ref pattern and enum")


def test_r9_grammar_maxitems_matches_the_claim_budget():
    """The grammar must not invite claims `report()` will throw away.

    It allowed 12 while the reporter kept 6. Two silent costs on the real run: the model
    spent generation budget on claims that were then discarded (9 of 116 constrained
    reports hit max_new_tokens and became parse failures), and the discards were counted
    nowhere, so emitting 12 mediocre claims scored the same as emitting 6.
    """
    for budget in (3, 6, 10):
        schema = claim_schema(ReporterConfig(max_claims=budget))
        assert schema["properties"]["claims"]["maxItems"] == budget, (
            f"grammar allows {schema['properties']['claims']['maxItems']} claims but the "
            f"reporter keeps {budget}"
        )
    # And the module-level constant must not have been mutated by building one.
    assert CLAIM_SCHEMA["properties"]["claims"]["maxItems"] == 6, (
        "claim_schema() mutated the shared CLAIM_SCHEMA instead of copying it"
    )

    seen: dict[str, int] = {}

    class SchemaSpy:
        name = "spy"

        def generate(self, prompt, *, constrained, schema):
            seen["maxItems"] = schema["properties"]["claims"]["maxItems"]
            return json.dumps({"summary": "", "recommendation": "", "escalate": False,
                               "claims": []})

    CaregiverReporter(SchemaSpy(), config=ReporterConfig(max_claims=4)).report(STATE)
    assert seen["maxItems"] == 4, (
        f"reporter handed the backend a {seen['maxItems']}-claim grammar under a 4-claim "
        "budget - the fix is not wired to the call site"
    )
    print(f"  R9 grammar maxItems tracks max_claims (3/6/10) and reaches the backend "
          f"as {seen['maxItems']} for a 4-claim budget")


def test_r10_unusable_rate_counts_schema_rejects_as_failures():
    """245 emitted / 0 scorable must read 100%, not `nan`.

    The real unconstrained arm emitted 245 claims, landed none, and reported `nan%` -
    which looks like missing data rather than total failure. A claim the schema rejects is
    no more usable to a caregiver than one that is fluent and false, so the denominator
    that matters is claims EMITTED.
    """

    class AllRejectedLLM:
        name = "all-rejected"

        def generate(self, prompt, *, constrained, schema):
            return json.dumps({
                "summary": "s", "recommendation": "r", "escalate": False,
                # Well-formed JSON, invalid claims: the exact shape of the failure.
                "claims": [{"claim_id": f"c{i}", "text": "t", "evidence_ref": "walking down"}
                           for i in range(1, 5)],
            })

    out = CaregiverReporter(AllRejectedLLM()).report(STATE)
    assert out.n_emitted_claims == 4, out.n_emitted_claims
    assert out.dropped_claims == 4
    assert out.report.claims == []
    assert out.hallucination_rate != out.hallucination_rate or out.hallucination_rate == 0.0
    assert out.unusable_rate == 1.0, (
        f"an arm that landed 0 of 4 claims scored unusable_rate={out.unusable_rate}"
    )

    # And the honest arm must not be penalised by the same denominator.
    honest = CaregiverReporter(FaithfulStubLLM()).report(STATE)
    assert honest.unusable_rate == 0.0, honest.unusable_rate
    assert honest.n_emitted_claims == len(honest.report.claims)

    # Non-list `claims` is a shape failure, not "no claims, therefore no lies".
    class ShapeBreakerLLM:
        name = "shape-breaker"

        def generate(self, prompt, *, constrained, schema):
            return json.dumps({"summary": "s", "recommendation": "r", "escalate": False,
                               "claims": {"c1": "walking fell"}})

    shaped = CaregiverReporter(ShapeBreakerLLM()).report(STATE)
    assert shaped.notes and "not a list" in shaped.notes[0], shaped.notes
    print(f"  R10 4 emitted / 0 scorable -> unusable 100% (hallucination rate n/a); "
          f"honest stub 0%; dict-valued 'claims' recorded as a shape failure")


def test_r11_rejection_reasons_name_the_field_not_the_exception():
    """245 rejections logged as "ValidationError" is a measurement with no diagnosis."""

    class BadFieldsLLM:
        name = "bad-fields"

        def generate(self, prompt, *, constrained, schema):
            return json.dumps({
                "summary": "s", "recommendation": "r", "escalate": False,
                "claims": [
                    {"claim_id": "c1", "text": "t", "evidence_ref": "not-a-ref"},
                    {"claim_id": "c2", "text": "t",
                     "evidence_ref": "feat:walking_duration_s:2026-02-22",
                     "claimed_value": "eighteen hundred"},
                    {"claim_id": "c3", "text": "", "evidence_ref": "feat:x:2026-02-22"},
                ],
            })

    out = CaregiverReporter(BadFieldsLLM()).report(STATE)
    assert out.dropped_claims == 3, out.dropped_claims
    blob = " | ".join(out.rejections)
    for field_name in ("evidence_ref", "claimed_value", "text"):
        assert field_name in blob, f"rejections never name {field_name!r}: {blob}"
    assert "ValidationError" not in blob, (
        f"rejections still report only the exception type: {blob}"
    )
    print(f"  R11 3 rejections named their fields: {'; '.join(out.rejections[:3])[:120]}")


def test_r12_repair_fixes_format_only_and_never_launders_a_lie():
    """The salvage arm's integrity check, and the reason it can be reported at all.

    Repair exists to separate "cannot emit our JSON" from "misstates the data". That
    separation is only valid if repair cannot turn a false claim into a faithful one, so
    this test asserts both directions: format defects become scorable, and every content
    defect still fails verification after repair.
    """
    day = STATE.report_day.isoformat()
    ref = f"feat:walking_duration_s:{day}"
    truth = FaithfulnessVerifier().build_index(STATE)[ref]

    # 1. Format-only defects: same facts, unusable spelling.
    formatted = {
        "claim_id": 7,
        "text": "  walking duration s fell to 700.0 from a baseline of 1800.0.  ",
        "evidence_ref": f"`feat:walking_duration_s:{day.replace('-', '/')}` ",
        "claimed_value": f"{truth.observed_value:,.1f}",
        "claimed_pct_change": f"{truth.pct_change:.1f} %",
        "direction": "Decreased",
    }
    fixed = Claim(**repair_claim(formatted, 6))
    assert fixed.evidence_ref == ref, fixed.evidence_ref
    assert fixed.claimed_value == truth.observed_value
    assert abs(fixed.claimed_pct_change - round(truth.pct_change, 1)) < 1e-9
    assert fixed.direction == "decrease"
    assert fixed.claim_id == "7"
    v = FaithfulnessVerifier().verify_claim(fixed, {ref: truth})
    assert v.is_faithful, f"repaired honest claim still fails: {v.notes}"

    # 2. Content defects must survive repair as failures. This is the negative control:
    # if any of these passes, the salvaged row is laundering hallucinations into
    # faithfulness and its number is worthless.
    lies = {
        "invented_value": {"claimed_value": truth.observed_value * 2 + 11},
        "wrong_pct": {"claimed_pct_change": truth.pct_change + 47.0},
        "inverted_direction": {"direction": "increase"},
        "fabricated_ref": {"evidence_ref": f"feat:stair_climbing_duration_s:{day}"},
        "prose_contradiction": {"text": "walking duration s rose sharply today."},
    }
    survived = []
    for mode, patch in lies.items():
        item = {"claim_id": "c1",
                "text": "walking duration s fell to 700.0.",
                "evidence_ref": ref,
                "claimed_value": truth.observed_value,
                "claimed_pct_change": truth.pct_change,
                "direction": "decrease", **patch}
        repaired = Claim(**repair_claim(item, 0))
        if FaithfulnessVerifier().verify_claim(repaired, {ref: truth}).is_faithful:
            survived.append(mode)
    assert not survived, f"repair laundered {survived} into faithful claims"

    # 3. Repair must not invent what the model omitted.
    bare = repair_claim({"text": "walking was unremarkable.", "evidence_ref": ""}, 0)
    assert bare["evidence_ref"] == "", "repair invented an evidence_ref"
    assert "claimed_value" not in bare, "repair invented a claimed_value"
    assert bare.get("direction") is None, "repair inferred an unstated direction"

    # 4. End to end through the reporter: salvage off -> unscorable, on -> scored.
    class SloppyFormatLLM:
        name = "sloppy-format"

        def generate(self, prompt, *, constrained, schema):
            return json.dumps({"summary": "s", "recommendation": "r", "escalate": False,
                               "claims": [formatted]})

    off = CaregiverReporter(SloppyFormatLLM()).report(STATE)
    on = CaregiverReporter(SloppyFormatLLM(),
                           config=ReporterConfig(salvage=True)).report(STATE)
    assert off.dropped_claims == 1 and off.report.claims == []
    assert on.repaired_claims == 1 and len(on.report.claims) == 1, (
        f"salvage did not recover a format-only claim: {on.rejections}"
    )
    assert on.hallucination_rate == 0.0, on.verification.results[0].notes
    print(f"  R12 format repair recovered 1 unscorable claim (0% hallucination after "
          f"repair) while all {len(lies)} content defects still failed; "
          f"repair invented no refs, values or directions")


def test_r13_both_decoding_arms_are_greedy():
    """The two arms of the comparison must differ ONLY in the grammar.

    They did not. The free path passed `do_sample=False`; the constrained path passed only
    `max_new_tokens` to `outlines` and inherited Qwen2.5-Instruct's shipped
    `generation_config` (`do_sample=True, temperature=0.7, top_p=0.8, top_k=20`). So the
    constrained arm was sampled and the free arm greedy, and the gap between them measured
    temperature rather than grammar.

    It surfaced as irreproducibility: over two runs of identical code the free arm was
    bit-identical (509 claims / 476 faithful) while the constrained arm moved 498/457 ->
    504/452, flipping the verdict from p = 0.29 to p = 0.03.
    """

    class FakeGenerationConfig:
        def __init__(self):
            # Exactly what Qwen2.5-7B-Instruct ships.
            self.do_sample = True
            self.temperature = 0.7
            self.top_p = 0.8
            self.top_k = 20

    class FakeModel:
        def __init__(self):
            self.generation_config = FakeGenerationConfig()

    model = FakeModel()
    # Negative control: assert the fixture really is the broken state, so a no-op
    # force_greedy could not pass this test.
    assert model.generation_config.do_sample is True
    assert model.generation_config.temperature == 0.7

    force_greedy(model)
    assert model.generation_config.do_sample is False, "sampling still enabled"
    for flag in SAMPLING_FLAGS:
        if hasattr(model.generation_config, flag):
            assert getattr(model.generation_config, flag) is None, (
                f"{flag} survived force_greedy; transformers warns it is ignored, and "
                "whether it truly is depends on the backend"
            )

    # A model without a generation_config must not crash the loader.
    class Bare:
        pass

    force_greedy(Bare())

    # And the config that documents greedy decoding must actually say 0.0, since both
    # paths derive `do_sample` from it.
    assert ReporterConfig().temperature == 0.0
    src = Path(__file__).resolve().parents[1] / "src/behaviorsense/agents/reasoning/reporter.py"
    body = src.read_text(encoding="utf-8")
    # A window, not a split on the first ")" - the call contains the cached output type,
    # whose paren closes long before the kwargs.
    outlines_call = body.split("self._outlines(text", 1)[1][:300]
    assert "do_sample" in outlines_call, (
        "the outlines call no longer pins do_sample; it will inherit the model's "
        "generation_config and silently start sampling again"
    )
    print("  R13 force_greedy cleared do_sample + "
          f"{len(SAMPLING_FLAGS)} sampling flags from a Qwen-shaped config; the outlines "
          "call pins do_sample too, so neither arm can drift into sampling")


def test_r14_the_two_arms_differ_only_by_the_grammar():
    """Audit EVERY axis on which the arms could differ. One test, not one per run.

    Three Kaggle sessions at ~2 h each were spent finding these one at a time, each from a
    log rather than from the code: the prompt withheld the output field names (run 1), the
    constrained path inherited `do_sample=True, temperature=0.7` (run 3), and it also
    skipped the chat template that the free path applied. All three were visible on CPU by
    reading the two code paths side by side. This test is that reading, made permanent.

    An A/B comparison is only about its independent variable if literally everything else
    matches. What must match: the text sent to the model, the decoding parameters, the token
    budget, the claim budget, and the verifier. What may differ: the grammar. Nothing else.
    """
    src = (Path(__file__).resolve().parents[1]
           / "src/behaviorsense/agents/reasoning/reporter.py").read_text(encoding="utf-8")
    body = src.split("class Qwen2_5Reporter", 1)[1]
    gen = body.split("def generate(", 1)[1]
    constrained_path, free_path = gen.split("import torch", 1)

    # 1. Same text. The chat template must be applied once, before the branch, so neither
    # path can be templated differently from the other.
    assert "text = self._chat_text(prompt)" in gen.split("if constrained", 1)[0], (
        "the chat template is no longer applied before the constrained/free branch - the "
        "two arms can now send different text to the model, which is how the constrained "
        "arm ended up 4 points worse for reasons unrelated to the grammar"
    )
    assert "apply_chat_template" not in gen.split("_chat_text", 1)[1], (
        "a second apply_chat_template appeared inside a branch; template once or the arms "
        "diverge again"
    )
    # Each branch must feed the model the TEMPLATED text, by name. Grepping for "prompt"
    # would match the function signature and the docstrings, so pin the actual call sites.
    assert "self._outlines(text," in constrained_path, (
        "the constrained path no longer passes the templated `text` to outlines"
    )
    assert "self._tok(text," in free_path, (
        "the free path no longer tokenises the templated `text`"
    )

    # 2. Same decoding. Both paths must pin do_sample from the same config field.
    for path, label in ((constrained_path, "constrained"), (free_path, "free")):
        assert "do_sample=self.cfg.temperature > 0" in path, (
            f"the {label} path no longer derives do_sample from cfg.temperature; it will "
            "inherit the model's generation_config and silently sample"
        )
        assert "max_new_tokens=self.cfg.max_new_tokens" in path, (
            f"the {label} path no longer uses the shared token budget"
        )

    # 3. Same guard at load time, and it must actually strip Qwen's shipped sampling.
    assert "force_greedy(self._model)" in body, "load path no longer forces greedy"
    for flag in ("temperature", "top_p", "top_k", "repetition_penalty"):
        assert flag in SAMPLING_FLAGS, f"{flag} is no longer stripped"

    # 4. Same claim budget reaching the grammar and the reporter.
    for budget in (4, 6):
        assert claim_schema(ReporterConfig(max_claims=budget))[
            "properties"]["claims"]["maxItems"] == budget

    # 5. Same prompt contract: every REQUIRED Claim field must be named in the prompt, or
    # the model is being scored on guessing our schema. This is run 1's defect - 245 of 245
    # free-arm claims rejected - and it cost a full session to see in a log.
    prompt = build_prompt(STATE, ReporterConfig())
    required = CLAIM_SCHEMA["properties"]["claims"]["items"]["required"]
    for field_name in (*required, "claims", "summary", "recommendation", "escalate"):
        assert field_name in prompt, (
            f"the prompt never names {field_name!r}, so an unconstrained model has to guess "
            "it; that is what produced 245 emitted / 0 scorable"
        )
    for field_name in ("claimed_value", "claimed_pct_change", "direction"):
        assert field_name in prompt, f"prompt does not name optional field {field_name!r}"

    # 6. Same verifier for both arms - no per-arm tolerance.
    from behaviorsense.agents.reasoning.verifier import VerifierConfig
    assert VerifierConfig().value_rel_tol == 0.02
    assert VerifierConfig().pct_abs_tol == 2.0

    # 7. The guide must be compiled per schema, not per call. 116 rebuilds per arm is why
    # the constrained arm ran at 49 s/report and the notebook took two hours.
    assert "self._output_type(schema)" in constrained_path, (
        "the constrained path builds its output type inline again; that defeats caching"
    )
    assert "if key not in self._output_types" in body, "output-type cache is gone"

    print(f"  R14 arms parity: same templated text, same do_sample/max_new_tokens, "
          f"{len(SAMPLING_FLAGS)} sampling flags stripped, prompt names all "
          f"{len(required)} required + 3 optional claim fields, one verifier config, "
          "guide cached per schema")


def test_r15_openrouter_backend_rotates_keys_and_changes_nothing_about_verification():
    """Agent 4 over a hosted endpoint. The point is that C1-C5 do not care.

    Qwen2.5-7B on two T4s was 286 s of a ~340 s request while holding 17.8 GiB of VRAM and
    emulating bf16 on sm_75. Moving Agent 4 off-box is worth doing - and safe for one structural
    reason: Agent 4 receives NUMBERS ONLY and every claim it writes is checked arithmetically
    before anyone sees it. Substituting a model we know less about is precisely the case the
    verifier exists to cover, so this test asserts the verification path is untouched rather
    than asserting anything about the model.

    Three properties beyond "it runs":

    - **`name` declares the hosted model.** The page renders it, and a report written by a free
      endpoint must not be mistaken for one from the pinned checkpoint every measured number in
      `results/evaluation.md` came from.
    - **Keys never appear in an error or in `stats()`.** Twenty live credentials pass through
      here; a traceback that includes one is a leak that outlives the session.
    - **A 429 is routed to the right AXIS.** This is the part measurement changed. `stats()` was
      built to answer "does rotation help", and it did: thirteen keys returned thirteen 429s, one
      each, with `rate-limited upstream` in every body and dashboards showing several of those keys
      idle that day. The limit was the free provider's capacity, so the key was innocent and the
      only axis that could help was a different provider. A quota 429 must still walk the keys; an
      upstream 429 must walk the MODELS and leave the keys alone. Conflating them is what parked
      thirteen good keys for a minute each while the fix was never tried.
    """
    import json as _json

    from behaviorsense.agents.reasoning.openrouter import (FALLBACK_MODELS, HOSTED_MAX_TOKENS,
                                                           MAX_TOKENS_ESCALATION, OpenRouterLLM,
                                                           classify_429, discover_keys)

    # Discovery: numbered vars, a bare var, and a comma-separated list in one secret - because
    # twenty Kaggle secrets is twenty clicks and one secret holding twenty is not.
    env = {f"OPENROUTER_API_KEY_{i}": f"sk-or-v1-k{i:02d}" for i in range(1, 21)}
    env["OPENROUTER_API_KEY_LIST"] = "sk-or-v1-k01, sk-or-v1-zzz"     # k01 duplicates
    keys = discover_keys(env)
    assert len(keys) == 21 and len(set(keys)) == 21, len(keys)
    assert keys[:2] == ["sk-or-v1-k01", "sk-or-v1-k02"], keys[:2]

    # The two 429 bodies, both taken from live responses rather than invented. The upstream one is
    # the exact shape that produced the thirteen-key failure.
    UPSTREAM = ('{"error":{"message":"Provider returned error","code":429,"metadata":{"raw":'
                '"z-ai/glm-5.2:free is temporarily rate-limited upstream. Please retry shortly, '
                'or add your own key to accumulate your rate limit"}}}')
    QUOTA = '{"error":{"message":"Rate limit exceeded: free-models-per-day.","code":429}}'
    assert classify_429(UPSTREAM) == "upstream", classify_429(UPSTREAM)
    assert classify_429(QUOTA) == "quota", classify_429(QUOTA)
    assert classify_429("") == "unknown", "an unrecognised 429 must not be guessed at"

    # KEY AXIS: three keys are genuinely out of daily quota, the fourth answers. One model.
    tried: list[tuple[str, str]] = []
    canned = None

    def transport(url, body, key, timeout):
        tried.append((body["model"], key))
        assert "Bearer" not in _json.dumps(body), "the key must travel in the header, not the body"
        if len(tried) <= 3:
            return 429, QUOTA, None
        return 200, _json.dumps({"choices": [{"message": {"content": canned}}]}), None

    stub = FaithfulStubLLM()
    stub.bind(STATE)
    canned = stub.generate("", constrained=True, schema=claim_schema(ReporterConfig()))

    # `sleep` is injected so the fast suite never actually waits. It is also the reason cooldowns
    # are compared against `max(wall, start + waited)` inside the module: with a no-op sleep the
    # wall clock never moves, so a cooldown measured against `time.time()` alone would still be in
    # force after the code had waited past it, and every post-backoff assertion here would be
    # vacuous - which is the exact shape of two bugs this project has already paid for.
    waits: list[float] = []
    llm = OpenRouterLLM(keys=keys[:20], post=transport, cooldown_s=30, sleep=waits.append)
    assert "openrouter" in llm.name and "20 key" in llm.name, llm.name

    out = CaregiverReporter(llm, config=ReporterConfig(constrained=True)).report(STATE)
    assert len(tried) == 4 and llm.rotations == 3, (len(tried), llm.rotations)
    assert len({k for _, k in tried}) == 4, "a spent key must not be retried inside one sweep"
    assert len({m for m, _ in tried}) == 1, "a quota 429 is the KEY's fault, so the model must hold"
    assert waits == [], f"a fresh key costs nothing, so nothing may sleep to reach one: {waits}"

    # THE POINT: the verifier ran on every claim, exactly as with a local model.
    assert out.report.claims and len(out.report.verifications) == len(out.report.claims)
    for c in out.report.claims:
        v = [x for x in out.report.verifications if x.claim_id == c.claim_id]
        assert len(v) == 1, f"{c.claim_id} has {len(v)} verdicts"
        assert v[0].ref_exists is not None and v[0].prose_quoted_value is not None
    assert "openrouter" in out.report.model_name, out.report.model_name

    # The schema is enforced SERVER-SIDE, and `strict` is what makes it a guarantee.
    sent: dict = {}

    def capture(url, body, key, timeout):
        sent.update(body)
        return 200, _json.dumps({"choices": [{"message": {"content": "{}"}}]}), None

    OpenRouterLLM(keys=["sk-or-v1-one"], post=capture).generate(
        "p", constrained=True, schema={"type": "object"})
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["strict"] is True, sent["response_format"]
    assert sent["temperature"] == 0.0, "Agent 4 restates given numbers; sampling only adds drift"

    # Unconstrained must NOT send a schema, or the two arms stop differing by the grammar alone.
    sent.clear()
    OpenRouterLLM(keys=["sk-or-v1-one"], post=capture).generate(
        "p", constrained=False, schema={"type": "object"})
    assert "response_format" not in sent, sent.keys()

    # MODEL AXIS - the failure this replaced. The primary's provider is saturated; the key is fine.
    seen: list[tuple[str, str]] = []

    def saturated_primary(url, body, key, timeout):
        seen.append((body["model"], key))
        if body["model"] == "z-ai/glm-5.2:free":
            return 429, UPSTREAM, None
        return 200, _json.dumps({"model": body["model"],
                                 "choices": [{"message": {"content": "{}"}}]}), None

    swept: list[float] = []
    llm2 = OpenRouterLLM(keys=keys[:20], post=saturated_primary, sleep=swept.append)
    llm2.generate("p", constrained=True, schema={"type": "object"})
    assert len(seen) == 2, f"an upstream 429 must change MODEL, not spend twenty keys: {seen}"
    assert len({k for _, k in seen}) == 1, "the key was innocent, so it must be reused"
    assert not any(s["cooling_s"] for s in llm2.stats()["keys"]), (
        "an upstream 429 parked a key - that is the thirteen-key failure, reproduced")
    assert swept == [], "changing provider costs nothing, so nothing may sleep to do it"
    # The page names whoever actually wrote the report, read back from the response rather than
    # assumed - with a `models` array the router may have chosen for us.
    assert llm2.served_by == FALLBACK_MODELS[0], llm2.served_by
    assert FALLBACK_MODELS[0] in llm2.name and "z-ai/glm-5.2:free" in llm2.name, llm2.name
    mstats = {m["model"]: m for m in llm2.stats()["models"]}
    assert mstats["z-ai/glm-5.2:free"]["429_upstream"] == 1, mstats
    assert mstats[FALLBACK_MODELS[0]]["ok"] == 1, mstats

    # The chain is PROVIDER diversity, and the 550B NVIDIA model is excluded for cause: its own
    # page states no `response_format` support, so `constrained=True` cannot be honoured; it serves
    # 2 tok/s, making a 1,600-token report ~13 minutes against a 90 s watchdog; and 24-hour
    # availability is 75.5%. The Super variant is the same vendor, on a different provider from the
    # primary, and does support strict schemas - which is what was actually wanted.
    assert "nvidia/nemotron-3-ultra-550b-a55b:free" not in FALLBACK_MODELS
    assert any("nemotron-3-super" in m for m in FALLBACK_MODELS), FALLBACK_MODELS
    assert len(set(FALLBACK_MODELS)) == len(FALLBACK_MODELS) >= 3, FALLBACK_MODELS

    # RECOVERY AFTER A WAIT IS OBSERVABLE - this is what the shared clock buys. Every provider is
    # saturated for one sweep and the primary is healthy on the next, and the assertion can tell.
    n = [0]

    def recovers(url, body, key, timeout):
        n[0] += 1
        if n[0] <= len(FALLBACK_MODELS) + 1:
            return 429, UPSTREAM, None
        return 200, _json.dumps({"choices": [{"message": {"content": "{}"}}]}), None

    back: list[float] = []
    llm3 = OpenRouterLLM(keys=keys[:20], post=recovers, sleep=back.append)
    assert llm3.generate("p", constrained=False, schema={}) == "{}"
    assert n[0] == len(FALLBACK_MODELS) + 2, f"the sweep after the wait never ran: {n[0]}"
    assert back and back[0] > 0, "repeating something that already failed must wait"

    # Retry-After overrides the formula, and the total wait is bounded so a saturated provider
    # cannot hang a demo. Six attempts once burned in 4.19 s against a provider that had itself
    # said "retry shortly"; that is a burst, not a retry policy.
    hinted: list[float] = []
    llm4 = OpenRouterLLM(keys=keys[:3], post=lambda *a: (429, UPSTREAM, 7.0), backoff_s=1.0,
                         sleep=hinted.append)
    try:
        llm4.generate("p", constrained=False, schema={})
        raise AssertionError("every provider was saturated but generate() returned")
    except RuntimeError as exc:
        msg = str(exc)
        assert "rate-limited upstream" in msg, msg
        assert "openrouter.ai/settings/integrations" in msg, "the error must name the remaining fix"
        for k in keys[:3]:
            assert k not in msg, "a live key appeared in an error message"
    assert hinted and min(hinted) >= 7.0, f"Retry-After: 7 must beat a 1 s formula, got {hinted}"
    assert llm4.waited_s <= llm4.deadline_s, (llm4.waited_s, llm4.deadline_s)
    assert all(k not in _json.dumps(llm4.stats()) for k in keys[:3]), "key leaked via stats()"

    # Exhausted KEYS read differently from saturated PROVIDERS, because the actions differ: wait
    # out a daily quota, or change provider. One error message for both hides which one to take.
    dead = OpenRouterLLM(keys=keys[:2], post=lambda *a: (429, QUOTA, None), cooldown_s=999,
                         sleep=lambda _s: None)
    try:
        dead.generate("p", constrained=False, schema={})
        raise AssertionError("all keys exhausted but generate() returned")
    except RuntimeError as exc:
        msg = str(exc)
        assert "rate limited or out of credit" in msg and "requests/minute" in msg, msg
        for k in keys[:2]:
            assert k not in msg, "a live key appeared in an error message"
    assert all(k not in _json.dumps(dead.stats()) for k in keys[:2]), "key leaked via stats()"

    # `attempts` is a ceiling on REQUESTS per report: five models over twenty keys must not be
    # allowed to reach a hundred calls just because both loops exist.
    capped = OpenRouterLLM(keys=keys[:20], post=lambda *a: (429, QUOTA, None), attempts=7,
                           sleep=lambda _s: None)
    try:
        capped.generate("p", constrained=False, schema={})
    except RuntimeError:
        pass
    assert capped.calls == 7, capped.calls

    # A model that REFUSES the schema is retried without it rather than dropped - and it says so,
    # because "constrained" is a claim about enforcement the page would otherwise make falsely.

    # TRUNCATION HAS TWO SIGNALS, and the second one is the one that bit. `finish_reason: length`
    # is authoritative when sent, but dots-3-note-preview returned a 200 with an unterminated
    # object and a DIFFERENT finish reason - so nothing escalated and the cut surfaced two layers
    # down as `unterminated JSON object in response (truncated generation)`, losing the report.
    # Under `constrained=True` the endpoint is enforcing a JSON schema, so an unbalanced object
    # cannot be a legitimate completion: it is evidence the generation was cut.
    cut_log: list[tuple[str, int]] = []

    def cut_without_saying_so(url, body, key, timeout):
        cut_log.append((body["model"], body["max_tokens"]))
        if body["max_tokens"] < MAX_TOKENS_ESCALATION:
            return 200, _json.dumps({"model": body["model"], "choices": [{
                "finish_reason": "stop",                    # not "length" - the gap
                "message": {"content": '{"summary":"the resident was'}}]}), None
        return 200, _json.dumps({"model": body["model"], "choices": [{
            "finish_reason": "stop",
            "message": {"content": '{"summary":"ok","claims":[]}'}}]}), None

    llm8 = OpenRouterLLM(keys=keys[:20], post=cut_without_saying_so, sleep=lambda _s: None)
    assert llm8.generate("p", constrained=True, schema={"type": "object"}) == (
        '{"summary":"ok","claims":[]}')
    assert llm8.stats()["token_escalations"] == 1, llm8.stats()["token_escalations"]
    assert [n for _m, n in cut_log] == [HOSTED_MAX_TOKENS, MAX_TOKENS_ESCALATION], cut_log
    assert len({m for m, _n in cut_log}) == 1, (
        "a budget cut is not the provider's fault, so the model must not change")

    # ...and the two NON-truncation cases must not escalate, or the transport starts second-
    # guessing content it does not own. Prose with no JSON is a COMPLIANCE failure and the
    # reporter has to see it, or the hallucination measurement loses a real category.
    llm9 = OpenRouterLLM(keys=keys[:2], sleep=lambda _s: None,
                         post=lambda *a: (200, _json.dumps({"choices": [{
                             "finish_reason": "stop",
                             "message": {"content": "I cannot answer that."}}]}), None))
    assert llm9.generate("p", constrained=True, schema={}) == "I cannot answer that."
    assert llm9.stats()["token_escalations"] == 0, "prose is not a budget problem"
    # Unconstrained: no schema promise, so an unbalanced body is not evidence of anything.
    llm10 = OpenRouterLLM(keys=keys[:2], sleep=lambda _s: None,
                          post=lambda *a: (200, _json.dumps({"choices": [{
                              "finish_reason": "stop",
                              "message": {"content": '{"a":1'}}]}), None))
    assert llm10.generate("p", constrained=False, schema={}) == '{"a":1'
    assert llm10.stats()["token_escalations"] == 0

    asked: list[bool] = []

    def refuse_schema(url, body, key, timeout):
        asked.append("response_format" in body)
        if "response_format" in body:
            return 400, '{"error":{"message":"response_format is not supported"}}', None
        return 200, _json.dumps({"choices": [{"message": {"content": "{}"}}]}), None

    llm5 = OpenRouterLLM(keys=keys[:1], post=refuse_schema, sleep=lambda _s: None)
    llm5.generate("p", constrained=True, schema={"type": "object"})
    assert asked == [True, False], asked
    assert llm5.served_constrained is False and "schema not enforced" in llm5.name, llm5.name

    # ...but a 400 that is NOT about the schema must NOT be treated as one. Measured: the endpoint
    # answered `'models' array must have 3 items or fewer` - a request-shape bug in this file - and
    # the first version read it as a schema refusal, dropped server-side enforcement and sent the
    # same malformed body again. Silently trading away the guarantee `constrained=True` exists for,
    # to work around something unrelated to it, is worse than failing.
    shapes: list[bool] = []
    ARRAY_400 = ('{"error":{"message":"\'models\' array must have 3 items or fewer.","code":400},'
                 '"user_id":"user_3IMghGdpYJ3SNcwNSyvugXZL4jM"}')

    def bad_shape(url, body, key, timeout):
        shapes.append("response_format" in body)
        return 400, ARRAY_400, None

    llm6 = OpenRouterLLM(keys=keys[:3], post=bad_shape, sleep=lambda _s: None)
    try:
        llm6.generate("p", constrained=True, schema={"type": "object"})
        raise AssertionError("a malformed request must not be retried into every key and model")
    except RuntimeError as exc:
        msg = str(exc)
        assert shapes == [True], f"enforcement was dropped for an unrelated 400: {shapes}"
        assert "3 items or fewer" in msg, msg
        assert "no other key or model will fix it" in msg, msg
        # The account id is not diagnostic and this body is quoted verbatim on the demo page.
        assert "user_3IM" not in msg and "user_id" not in msg, msg
    assert llm6.served_constrained is None, "a failed call must not claim an enforcement outcome"

    # THE `models` ARRAY IS CAPPED AT 3 by the endpoint - a limit that is in no document, only in
    # a 400. The Python sweep still walks all five, so this truncates the routing HINT and not the
    # coverage; and it walks the chain FORWARD from the current model so a 3-slot array never
    # re-offers something already tried.
    arrays: list[list[str]] = []

    def note_models(url, body, key, timeout):
        arrays.append(list(body.get("models") or []))
        return 429, UPSTREAM, None

    llm7 = OpenRouterLLM(keys=keys[:2], post=note_models, sleep=lambda _s: None, sweeps=1)
    try:
        llm7.generate("p", constrained=False, schema={})
    except RuntimeError:
        pass
    assert arrays and all(len(a) <= 3 for a in arrays), f"the endpoint rejects 4+: {arrays}"
    assert arrays[0] == list(llm7.plan[:3]), arrays[0]
    assert arrays[0][0] == "z-ai/glm-5.2:free" and arrays[1][0] == FALLBACK_MODELS[0], arrays[:2]
    # A provider that just said `rate-limited upstream` is a wasted slot out of three.
    assert "z-ai/glm-5.2:free" not in arrays[-1], f"a cooling model kept a routing slot: {arrays}"
    assert len({m for a in arrays for m in a}) == len(llm7.plan), (
        f"every model in the chain must still be reachable across the sweep: {arrays}")

    # The `models` array only appears when fallbacks are configured, because OpenRouter routing to
    # a model nobody asked for would silently change which model wrote the report.
    sent.clear()
    OpenRouterLLM(keys=["sk-or-v1-one"], post=capture, fallback_models=()).generate(
        "p", constrained=False, schema={})
    assert "models" not in sent, sent.keys()
    sent.clear()
    OpenRouterLLM(keys=["sk-or-v1-one"], post=capture,
                  fallback_models=["a/b:free", "c/d:free"]).generate(
        "p", constrained=False, schema={})
    assert sent["models"] == ["z-ai/glm-5.2:free", "a/b:free", "c/d:free"], sent["models"]

    print(f"  R15 21 keys discovered (numbered + list, deduped); a QUOTA 429 walks the keys with "
          f"no delay (4 keys, 1 model, {len(waits)} sleeps) while an UPSTREAM 429 walks the models "
          f"and leaves the keys uncooled ({len(seen)} requests, not 20); {len(out.report.claims)} "
          f"claims each got exactly one C1-C5 verdict; model_name={out.report.model_name!r}; "
          f"strict schema only when constrained, dropped-and-declared when a model refuses it but "
          f"NOT for an unrelated 400; `models` capped at 3 and walked forward {arrays[:2]}; "
          f"Retry-After beats the formula; waits bounded by deadline_s, requests by attempts; the "
          f"sweep after a backoff is observable; no key material or account id in any error")


def test_r16_a_partial_window_cannot_claim_decline():
    """A short clip is not a short day, and comparing them is an arithmetic error.

    MEASURED, on the 9m56s upload that motivated this: the reference days carry
    `observed_hours = 22.9` while the clip carries `0.166` - a 138x window mismatch. Every
    feature total therefore read as a collapse (walking "-99.4%, robust_z -10.00" and five
    features at "-100.0%"), Agent 4 recommended contacting a healthcare professional, and all
    six claims passed C1-C5 - because the arithmetic was internally consistent. The comparison
    itself was invalid: the clip's actual walking RATE was 62.7 s per observed hour against the
    baseline's 66.9 s/h, i.e. an ordinary day of walking observed for ten minutes.

    This is the second confirmed instance of the same shape: faithful claims that pass every
    check while the thing they assert was never measured. C1-C5 verify a claim against the
    EVIDENCE; nothing verifies that the evidence itself supports the comparison. So the fix is
    upstream of the model - the numbers that cannot support a claim are WITHHELD from the prompt,
    not flagged next to it. (The first attempt at a related fix added a prompt instruction and
    left `-99.4%` in the data block; the model used the number.)

    The gate reuses the existing threshold: `min_observed_hours_for_absence = 8.0` already stops
    the ALERT rules on exactly this condition, for exactly this reason. This applies the same
    decision to what may be CLAIMED.
    """
    # The clip's own numbers, as the serving path measured them. `make_day` fills every field
    # with a full day's values, so the fields a 10-minute window did not observe are zeroed
    # explicitly: the real path builds features from this clip's segments alone, and a fixture
    # carrying 28,000 s of lying inside a 596 s window is not the scenario under test.
    clip = make_day(
        D0 + timedelta(days=21),
        walking_s=10.40, meals=0, social_s=0.0, meds=0,
    ).model_copy(update={
        "walking_bouts": 4, "mean_bout_duration_s": 2.6, "room_transitions": 0,
        "sit_to_stand_count": 0, "mean_sit_to_stand_duration_s": 0.0,
        "sitting_duration_s": 0.0, "lying_duration_s": 0.0, "standing_duration_s": 0.0,
        "longest_inactive_block_s": 5.2, "eating_duration_s": 0.0, "drinking_events": 4.0,
        "cooking_duration_s": 66.95, "tv_duration_s": 0.0, "reading_duration_s": 5.20,
        "phone_events": 5.0, "visitor_count": 0, "housework_duration_s": 173.65,
        "observed_hours": 0.166,          # 9m56s - 138x shorter than the reference days
    })
    analyzer = BehaviourAnalyzer()
    for i in range(21):
        analyzer.analyze_day(make_day(D0 + timedelta(days=i)))
    partial_state = analyzer.analyze_day(clip)

    assert not partial_state.today.is_reliable()
    payload = state_to_payload(partial_state, ReporterConfig())
    assert payload["window_is_partial"] is True, "the prompt dispatches rule 4a on this field"

    rows = {r["feature"]: r for r in payload["features"]}
    # WITHHELD, not flagged: a number that cannot support a claim must not be in front of the
    # model at all. These are the exact cells the six passing claims were copied from.
    for name, r in rows.items():
        assert r["baseline_comparable"] is False, name
        assert r["pct_change"] is None, f"{name}: a -99% against a 138x-shorter window leaked"
        assert r["robust_z"] is None, name
        assert r["baseline_median"] is None, name
        # DIRECTION TOO. It is sign(observed - baseline_median), so on a partial window it is the
        # same invalid comparison wearing a word instead of a number - and words travel further:
        # the page stamps it beside the value, and five "· decrease" stamps read as decline even
        # when every number beside them is an honest observation. Observed on a clip of someone
        # cooking: `drinking_events 4.00 · decrease`.
        assert r["direction"] is None, f"{name}: direction={r['direction']!r} on a partial window"
        assert r["why_withheld"], name
    # PRESENCE IS EVIDENCE; ABSENCE IS NOT. Same flag, opposite values, one rule.
    # (`housework_duration_s`, not `cooking_duration_s`: cooking is NOT in
    # REPORTABLE_FEATURES, so the headline activity of the motivating clip was never offered
    # to the model at all - noted separately, because adding to that list changes the measured
    # arm's prompt and is not this test's decision to make.)
    assert rows["walking_duration_s"]["observed_in_window"] is True
    assert rows["housework_duration_s"]["observed_in_window"] is True
    assert rows["meal_events"]["observed_in_window"] is False

    prompt = build_prompt(partial_state, ReporterConfig())
    assert '"pct": null' in prompt or '"pct":null' in prompt, (
        "the template must offer no percentage to copy")
    assert "baseline_comparable" in prompt, "rule 4a's flags must reach the template"
    assert "window_is_partial" in prompt

    # THE CONTROL MUST REMAIN THE CEILING UNDER THE NEW RULES. The faithful stub now writes
    # occurrence prose for partial rows, so the verified output demonstrates what the new report
    # looks like: observed values, no decline claims, every claim still passing C1-C5.
    out = CaregiverReporter(FaithfulStubLLM()).report(partial_state)
    assert out.hallucination_rate == 0.0, (
        f"the control scored {out.hallucination_rate} - the verifier is rejecting faithful prose")
    assert out.report.claims, "nothing claimable on a partial day, so rule 4a is too strict"
    for c in out.report.claims:
        assert c.claimed_pct_change is None, (
            f"{c.claim_id} quotes a pct on a partial window: {c.claimed_pct_change}")
        # No direction on the claim either, so the page renders no "· decrease" stamp. C4 is
        # vacuous when direction is None - the same precedent as C3 on a null pct - so the
        # verifier still runs on every claim and still reports a verdict.
        assert c.direction is None, f"{c.claim_id} carries direction={c.direction!r}"
        low = c.text.lower()
        assert "decrease" not in low and "versus a baseline" not in low, c.text
    assert all(v.direction_consistent for v in out.report.verifications), (
        "C4 must be vacuously satisfied by a withheld direction, not failed by it")
    texts = " ".join(c.text for c in out.report.claims)
    assert "housework" in texts and "173.65" in texts, (
        "the clip's real activity must be what the report is about")
    assert "meal" in texts or "walking" in texts

    # THE MEASURED ARM IS UNCHANGED. A reliable day still carries the full comparison - every
    # number in results/evaluation.md was produced from these cells and must keep existing.
    full = state_to_payload(STATE, ReporterConfig())
    assert full["window_is_partial"] is False
    full_rows = {r["feature"]: r for r in full["features"]}
    assert full_rows["walking_duration_s"]["baseline_comparable"] is True
    assert full_rows["walking_duration_s"]["pct_change"] is not None
    assert full_rows["walking_duration_s"]["robust_z"] is not None
    assert full_rows["walking_duration_s"]["direction"] == "decrease", (
        "a reliable day must keep its direction, or C4 stops catching inverted narration - "
        "the check this whole layer was built for")
    out_full = CaregiverReporter(FaithfulStubLLM()).report(STATE)
    assert out_full.hallucination_rate == 0.0

    print(f"  R16 a 0.166 h window against 22.9 h baselines ({22.9/0.166:.0f}x): all "
          f"{len(rows)} rows have pct/z/baseline withheld; {len(out.report.claims)} observed-value "
          f"claims, 0 decline claims, hallucination 0.0; walking rate {10.40/0.166:.1f} s/h vs "
          f"baseline {full_rows['walking_duration_s']['baseline_median']/22.9:.1f} s/h (an "
          f"ordinary day, previously reported as -99.4%); the reliable-day arm still carries pct "
          f"and z (walking z={full_rows['walking_duration_s']['robust_z']})")


def test_r17_a_budget_truncated_generation_is_escalated_not_lost():
    """A 200 with `finish_reason: length` is this file's own budget cutting the model off.

    MEASURED, on the 9m56s clip: the chain routed to dots-studio/dots-3-note-preview (GLM and
    nemotron both saturated upstream), and the model returned a 200 that was all thinking and no
    object - `unterminated JSON object in response (truncated generation)` at the then-4,096-token
    budget, and the whole report was lost. (HOSTED_MAX_TOKENS is 8,192 now; this test derives
    its budgets from the constant so raising it again cannot make the test lie.) The same failure had already happened to nemotron at
    1,600 before HOSTED_MAX_TOKENS was raised; both models are REASONERS that spend the
    completion budget thinking.

    Two fixes, both in the transport where the field that justifies them is read:

    * `reasoning: {"effort": "low"}` on every request - OpenRouter's normalised control. The
      report restates twenty-seven numbers; there is nothing here that needs deep reasoning.
    * On `finish_reason: length`, DOUBLE the budget and retry - honest, because the retry yields
      a genuinely complete generation where a parse repair yields a laundered one. At the
      escalation ceiling the model is abandoned with the reason in the message, not retried into
      the remaining keys (the same reasoning as the upstream-429 branch).
    """
    import json as _json

    from behaviorsense.agents.reasoning.openrouter import (HOSTED_MAX_TOKENS,
                                                           OpenRouterLLM)

    keys = [f"sk-or-v1-k{i:02d}" for i in range(1, 21)]
    budgets_seen: list[int] = []
    # DERIVED, never hardcoded. This asserted `[4096, 8192]` and broke the moment
    # HOSTED_MAX_TOKENS was legitimately raised to 8,192 - a test failing on a correct
    # change, which teaches the reader to edit the number rather than read the reason.
    first, doubled = HOSTED_MAX_TOKENS, HOSTED_MAX_TOKENS * 2

    def reasoner(url, body, key, timeout):
        budgets_seen.append(body["max_tokens"])
        # Effort control present, as the root-cause fix demands.
        assert body.get("reasoning") == {"effort": "low"}, body.get("reasoning")
        if body["max_tokens"] < doubled:
            # All thinking, no object: cut by the budget, exactly as observed.
            return 200, _json.dumps({
                "model": body["model"],
                "choices": [{"message": {"content": '{"summary": "thinking...'},
                                         "finish_reason": "length"}]}), None
        return 200, _json.dumps({
            "model": body["model"],
            "choices": [{"message": {"content": '{"summary": "ok", "claims": []}'},
                                 "finish_reason": "stop"}]}), None

    llm8 = OpenRouterLLM(keys=keys, post=reasoner, sleep=lambda _s: None, fallback_models=())
    out = llm8.generate("p", constrained=True, schema={"type": "object"})
    assert _json.loads(out)["summary"] == "ok"
    assert budgets_seen == [first, doubled], budgets_seen
    assert llm8.token_escalations == 1, llm8.token_escalations
    assert llm8.calls == 2, llm8.calls
    assert llm8.stats()["budgets"] == {"z-ai/glm-5.2:free": doubled}, llm8.stats()["budgets"]

    # The remembered budget is used by the NEXT report on the same instance.
    budgets_after_restart: list[dict] = []

    def capture_body(url, body, key, timeout):
        budgets_after_restart.append(body)
        return 200, _json.dumps({
            "model": body["model"],
            "choices": [{"message": {"content": '{"summary": "ok2", "claims": []}'},
                                 "finish_reason": "stop"}]}), None

    llm8._post = capture_body
    assert _json.loads(llm8.generate("p", constrained=False, schema={}))["summary"] == "ok2"
    assert budgets_after_restart[0]["max_tokens"] == doubled, (
        f"a model that needed {doubled} re-paid a failed {first} on its second report")

    # THE CEILING. A model still truncating at MAX_TOKENS_ESCALATION must be ABANDONED, not
    # retried into every key - the same reasoning as the upstream-429 branch. With the ceiling
    # pinned low, one key is enough to prove both the bounded escalation and the clear message.
    import behaviorsense.agents.reasoning.openrouter as _or
    real_cap = _or.MAX_TOKENS_ESCALATION
    # Pinned AT the start budget, not at a literal: with a hardcoded 4,096 this section became
    # unreachable the moment HOSTED_MAX_TOKENS passed it (8,192 < 4,096 is false, so the escalation
    # branch never ran and the model was abandoned on its first reply for the wrong reason). Equal
    # to `first` means "no headroom", which is exactly the condition under test.
    _or.MAX_TOKENS_ESCALATION = first
    try:
        seen_len: list[int] = []

        def always_length(url, body, key, timeout):
            seen_len.append(body["max_tokens"])
            return 200, _json.dumps({
                "model": body["model"], "choices": [{"message": {"content": "{\"claims\": ["},
                                                     "finish_reason": "length"}]}), None

        dead = OpenRouterLLM(keys=keys[:1], post=always_length, sleep=lambda _s: None,
                             fallback_models=())
        try:
            dead.generate("p", constrained=False, schema={})
            raise AssertionError("a model that cannot fit was not abandoned")
        except RuntimeError as exc:
            msg = str(exc)
            assert f"truncated at max_tokens={first}" in msg, msg
            assert "the model never reached the end of its own output" in msg, msg
        # Pinned at `first` there is no headroom to double into, so exactly one request is made.
        assert seen_len == [first], (
            f"budgets escalated past the ceiling: {seen_len}")
        assert dead.stats()["token_escalations"] == 0, "no escalation below the ceiling"
    finally:
        _or.MAX_TOKENS_ESCALATION = real_cap

    print(f"  R17 a length-cut 200 escalates {budgets_seen} and completes (1 escalation, "
          f"{llm8.calls} requests); at the ceiling the model is abandoned with the reason in the "
          f"message ({seen_len}); effort=low is sent; the budget is remembered for the next report")


def test_r18_every_published_number_verifies_at_face_value():
    """C2 must never reject a digit-for-digit copy of a value the system itself published.

    MEASURED, on a deployed clip: the reporter's template offers `observed_value` rounded to
    2 decimals; the model copied it exactly (which is what the prompt demands); and the
    verifier struck the claim anyway - `claimed 0.07, actual 0.074` on mobility_index, so a
    report where every figure was quoted correctly showed 20% unfaithful and one withheld
    claim. The prompt and the verifier disagreed about precision, and the model paid for it.

    Why nothing caught it before: R1's control runs on features with magnitudes in the
    hundreds, where 2% dwarfs 0.005 of rounding. The boundary only exists below 0.25 in
    magnitude, and no fixture ever claimed a feature that small until a real clip's
    mobility_index.

    Two guards here. The boundary sweep pins the tolerance arithmetic. The COUPLING sweep
    asserts the actual contract - for every row `state_to_payload` publishes, a claim of that
    row's `observed_value` passes C2 - so if the reporter's rounding ever changes without the
    verifier's floor following it, this fails rather than a live demo.
    """
    from behaviorsense.agents.reasoning.verifier import FaithfulnessVerifier

    v = FaithfulnessVerifier()
    agree = v._values_agree

    # 1. THE MEASURED CASE, and the boundary sweep around it: round(x, 2) must verify for
    #    any x, because that is exactly what the template publishes.
    cases = [0.074, 0.004, 0.006, 0.249, 0.251, 26.114, 1800.009, 0.096, 0.145]
    for x in cases:
        published = round(x, 2)
        assert agree(published, x), f"a published {published} (true {x}) failed C2"

    # 2. Invention at small magnitude still fails: 0.08 is NOT a rounding of 0.074, and
    #    admitting it would make the floor a general slack instead of a half-step.
    assert not agree(0.08, 0.074), "0.08 is not a rounding of 0.074 - invention passed"
    assert not agree(0.06, 0.074), "0.06 is not a rounding of 0.074 - invention passed"
    # 3. Large magnitudes keep the old behaviour: 2% dominates the half-step there.
    assert not agree(900.0, 820.0), "a large-magnitude misquote passed C2"
    assert agree(820.0, 820.0)
    # 4. C5's prose check gets the same floor: quoting the published value or the full
    #    precision in prose are both faithful renderings of the same number.
    assert v._prose_quotes_value("mobility index was 0.07.", 0.07)
    assert v._prose_quotes_value("mobility index was 0.074.", 0.07)

    # 5. THE COUPLING CONTRACT, on both a reliable day and a partial-window clip: every
    #    number the prompt offers must verify against the evidence the verifier holds.
    clip = make_day(
        D0 + timedelta(days=21), walking_s=10.40, meals=0, social_s=0.0, meds=0,
    ).model_copy(update={
        "walking_bouts": 4, "mean_bout_duration_s": 2.6, "observed_hours": 0.166,
        "housework_duration_s": 153.0, "cooking_duration_s": 70.2,
        "drinking_events": 4.0, "longest_inactive_block_s": 7.35, "reading_duration_s": 5.2,
        "phone_events": 5.0})
    analyzer = BehaviourAnalyzer()
    for i in range(21):
        analyzer.analyze_day(make_day(D0 + timedelta(days=i)))
    partial_state = analyzer.analyze_day(clip)

    for label, st in (("reliable", STATE), ("partial", partial_state)):
        payload = state_to_payload(st, ReporterConfig())
        index = FaithfulnessVerifier().build_index(st)
        published = [(r["feature"], r["observed_value"]) for r in payload["features"]
                     if r["evidence_ref"] in index]
        bad = [(f, val, index[ref].observed_value)
               for f, val, ref in ((f, val, next(r["evidence_ref"] for r in payload["features"]
                                                 if r["feature"] == f))
                                   for f, val in published)
               if not agree(val, index[ref].observed_value)]
        assert not bad, (
            f"{label} day: the prompt publishes values the verifier rejects at face value: "
            f"{bad} - a model copying them digit-for-digit would be struck as unfaithful")
        print(f"  {label}: all {len(published)} published values verify at face value")
    # The measured regime is pinned by the boundary sweep above (0.074 et al.), not by this
    # fixture: the simulator's computed mobility_index is large-magnitude, which is exactly
    # why every earlier control missed the boundary a real clip found.
    print(f"  R18 boundary sweep {len(cases)} published roundings pass (incl. 0.074 -> 0.07); "
          f"0.08-vs-0.074 and 900-vs-820 still fail; prose accepts both the published and "
          f"full-precision renderings; all published values verify on both days")


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
