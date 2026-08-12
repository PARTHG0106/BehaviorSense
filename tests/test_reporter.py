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
    CaregiverReporter,
    FaithfulStubLLM,
    HallucinatingStubLLM,
    ReporterConfig,
    _extract_json,
    build_prompt,
    claim_schema,
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
