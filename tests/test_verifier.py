"""Tests for the faithfulness verifier.

Each test injects a specific, documented hallucination mode observed in LLM-generated
clinical summaries, and asserts the verifier catches it. A verifier that passes
everything is worse than none at all - it manufactures false confidence - so the
negative cases matter more than the positive one.

Modes covered:
  V1  faithful claim                          -> PASS
  V2  fabricated evidence_ref                 -> caught by C1
  V3  malformed ref                           -> caught by C1
  V4  invented numeric value                  -> caught by C2
  V5  wrong percentage                        -> caught by C3
  V6  INVERTED direction, correct numbers     -> caught by C4  [most dangerous]
  V7  prose contradicting structured field    -> caught by C4 lexical cross-check
  V8  pct sign contradicting the data         -> caught by C4 sign check
  V9  rounding within tolerance               -> PASS (must not over-reject)
  V10 corpus-level hallucination-rate metric  -> arithmetic sanity
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@contextmanager
def pytest_raises(exc_type):
    """Minimal pytest.raises stand-in so this file runs without pytest installed."""
    try:
        yield
    except exc_type:
        return
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"expected {exc_type.__name__}, got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"expected {exc_type.__name__}, but no exception was raised")

from behaviorsense.agents.behaviour import BehaviourAnalyzer  # noqa: E402
from behaviorsense.agents.reasoning.verifier import (  # noqa: E402
    FaithfulnessVerifier,
    VerifierConfig,
    aggregate_reports,
)
from behaviorsense.schemas import Claim, DailyFeatures, Role  # noqa: E402

D0 = date(2026, 1, 1)


def make_day(day: date, walking_s: float = 1800.0, meals: int = 3) -> DailyFeatures:
    return DailyFeatures(
        subject_role=Role.RESIDENT,
        day=day,
        walking_duration_s=walking_s,
        walking_bouts=15,
        mean_bout_duration_s=120.0,
        room_transitions=30,
        sit_to_stand_count=20,
        mean_sit_to_stand_duration_s=2.5,
        sitting_duration_s=14400.0,
        lying_duration_s=28800.0,
        standing_duration_s=3600.0,
        longest_inactive_block_s=3600.0,
        meal_events=meals,
        eating_duration_s=meals * 900.0,
        drinking_events=6,
        cooking_duration_s=1800.0,
        medication_events=1,
        tv_duration_s=7200.0,
        reading_duration_s=1800.0,
        phone_events=2,
        social_interaction_duration_s=1200.0,
        visitor_count=1,
        housework_duration_s=1200.0,
        observed_hours=24.0,
        tracking_coverage=1.0,
    )


def build_state():
    """21 stable days, then a day with a large genuine drop in walking."""
    analyzer = BehaviourAnalyzer()
    rng_walk = [1800.0, 1750.0, 1850.0, 1900.0, 1700.0, 1820.0, 1780.0]
    for i in range(21):
        analyzer.analyze_day(make_day(D0 + timedelta(days=i), walking_s=rng_walk[i % 7]))
    return analyzer.analyze_day(make_day(D0 + timedelta(days=21), walking_s=900.0))


STATE = build_state()
DAY_STR = STATE.report_day.isoformat()
WALK_REF = f"feat:walking_duration_s:{DAY_STR}"
VERIFIER = FaithfulnessVerifier()
INDEX = VERIFIER.build_index(STATE)
TRUE_EV = INDEX[WALK_REF]


def test_v0_index_and_ground_truth() -> None:
    assert WALK_REF in INDEX, f"{WALK_REF} not resolvable; refs={sorted(INDEX)[:5]}"
    assert abs(TRUE_EV.observed_value - 900.0) < 1e-6
    assert TRUE_EV.direction == "decrease"
    assert TRUE_EV.delta < 0
    print(f"  V0 walking: observed={TRUE_EV.observed_value:.0f}s "
          f"baseline={TRUE_EV.baseline_median:.0f}s "
          f"pct={TRUE_EV.pct_change:+.1f}% z={TRUE_EV.robust_z:+.2f} "
          f"dir={TRUE_EV.direction} | {len(INDEX)} refs resolvable")


def test_v1_faithful_claim_passes() -> None:
    claim = Claim(
        claim_id="c1",
        text="Walking time decreased compared with the two-week baseline.",
        evidence_ref=WALK_REF,
        claimed_value=TRUE_EV.observed_value,
        claimed_pct_change=TRUE_EV.pct_change,
        direction="decrease",
    )
    r = VERIFIER.verify_claim(claim, INDEX)
    assert r.is_faithful, f"faithful claim rejected: {r.notes}"
    print("  V1 faithful claim accepted")


def test_v2_fabricated_ref_caught() -> None:
    """The model cites a plausible-looking but nonexistent feature."""
    claim = Claim(
        claim_id="c2",
        text="Stair climbing decreased significantly.",
        evidence_ref=f"feat:stair_climbing_duration_s:{DAY_STR}",
        claimed_value=42.0,
        direction="decrease",
    )
    r = VERIFIER.verify_claim(claim, INDEX)
    assert not r.is_faithful and not r.ref_exists
    print(f"  V2 fabricated ref caught: {r.notes[0]}")


def test_v3_malformed_ref_rejected_at_schema_level() -> None:
    """Malformed refs are blocked by the schema before the verifier runs.

    Defence in depth: the Claim pattern rejects them at construction, and
    verify_claim() also handles them for refs arriving from a non-Pydantic path
    (raw LLM JSON, a DB row written by an older schema version). Both layers are
    asserted here.
    """
    with pytest_raises(ValidationError):
        Claim(
            claim_id="c3",
            text="Walking decreased.",
            evidence_ref="feat:walking_duration_s:yesterday",
            direction="decrease",
        )
    print("  V3 layer 1: schema rejected malformed ref at construction")

    # Layer 2: bypass validation the way untrusted external input would.
    bypassed = Claim.model_construct(
        claim_id="c3b",
        text="Walking decreased.",
        evidence_ref="feat:walking_duration_s:yesterday",
        claimed_value=None,
        claimed_pct_change=None,
        direction="decrease",
    )
    r = VERIFIER.verify_claim(bypassed, INDEX)
    assert not r.is_faithful and not r.ref_exists
    assert "malformed" in r.notes[0]
    print(f"  V3 layer 2: verifier caught bypassed ref -> {r.notes[0]}")


def test_v4_invented_value_caught() -> None:
    """Correct ref and direction, but the number is invented."""
    claim = Claim(
        claim_id="c4",
        text="Walking time decreased to 1500 seconds.",
        evidence_ref=WALK_REF,
        claimed_value=1500.0,  # actual 900
        direction="decrease",
    )
    r = VERIFIER.verify_claim(claim, INDEX)
    assert not r.is_faithful and not r.value_matches
    assert r.ref_exists and r.direction_consistent
    print(f"  V4 invented value caught: {r.notes[0]}")


def test_v5_wrong_percentage_caught() -> None:
    claim = Claim(
        claim_id="c5",
        text="Walking time decreased by roughly 12%.",
        evidence_ref=WALK_REF,
        claimed_value=TRUE_EV.observed_value,
        claimed_pct_change=-12.0,  # actual ~ -50%
        direction="decrease",
    )
    r = VERIFIER.verify_claim(claim, INDEX)
    assert not r.is_faithful and not r.pct_matches
    print(f"  V5 wrong percentage caught: {r.notes[0]}")


def test_v6_inverted_direction_caught() -> None:
    """THE dangerous mode: every number correct, narrated backwards.

    C1-C3 all pass. Only the direction check saves the caregiver from being told the
    resident is improving while they decline.
    """
    claim = Claim(
        claim_id="c6",
        text="Walking time improved, rising well above the usual level.",
        evidence_ref=WALK_REF,
        claimed_value=TRUE_EV.observed_value,       # correct
        claimed_pct_change=abs(TRUE_EV.pct_change), # correct magnitude, wrong sign
        direction="increase",                        # WRONG
    )
    r = VERIFIER.verify_claim(claim, INDEX)
    assert r.ref_exists and r.value_matches, "C1/C2 should pass here"
    assert not r.direction_consistent
    assert not r.is_faithful
    print("  V6 inverted narration caught despite correct value:")
    for n in r.notes:
        print(f"     - {n}")


def test_v7_prose_contradicts_field() -> None:
    """Structured field right, prose wrong - internally inconsistent output."""
    claim = Claim(
        claim_id="c7",
        text="Mobility has clearly increased and the resident is more active.",
        evidence_ref=WALK_REF,
        claimed_value=TRUE_EV.observed_value,
        direction="decrease",  # field correct, prose contradicts it
    )
    r = VERIFIER.verify_claim(claim, INDEX)
    assert not r.is_faithful and not r.direction_consistent
    print(f"  V7 self-inconsistent claim caught: {len(r.notes)} note(s)")
    for n in r.notes:
        print(f"     - {n}")


def test_v8_pct_sign_contradiction() -> None:
    """Neutral prose, no direction field, but the percentage sign is wrong."""
    claim = Claim(
        claim_id="c8",
        text="Walking duration was recorded at the level shown.",
        evidence_ref=WALK_REF,
        claimed_pct_change=abs(TRUE_EV.pct_change),  # positive; truth is negative
    )
    r = VERIFIER.verify_claim(claim, INDEX)
    assert not r.is_faithful and not r.direction_consistent
    print(f"  V8 pct-sign contradiction caught: {r.notes[-1]}")


def test_v9_rounding_tolerated() -> None:
    """Sensible rounding must NOT be scored as hallucination."""
    claim = Claim(
        claim_id="c9",
        text="Walking fell to about 900 seconds.",
        evidence_ref=WALK_REF,
        claimed_value=900.0,
        claimed_pct_change=round(TRUE_EV.pct_change),  # e.g. -50 vs -50.14
        direction="decrease",
    )
    r = VERIFIER.verify_claim(claim, INDEX)
    assert r.is_faithful, f"over-rejected rounding: {r.notes}"
    print(f"  V9 rounding tolerated (claimed {round(TRUE_EV.pct_change)}% "
          f"vs actual {TRUE_EV.pct_change:.2f}%)")


def test_v10_corpus_metric() -> None:
    """Hallucination rate over a mixed batch, plus the failure breakdown."""
    claims = [
        Claim(claim_id="k1", text="Walking decreased.", evidence_ref=WALK_REF,
              claimed_value=900.0, direction="decrease"),
        Claim(claim_id="k2", text="Walking decreased by half.", evidence_ref=WALK_REF,
              claimed_pct_change=TRUE_EV.pct_change, direction="decrease"),
        Claim(claim_id="k3", text="Bathing decreased.",
              evidence_ref=f"feat:bathing_duration_s:{DAY_STR}", direction="decrease"),
        Claim(claim_id="k4", text="Walking rose sharply.", evidence_ref=WALK_REF,
              claimed_value=900.0, direction="increase"),
        Claim(claim_id="k5", text="Walking decreased to 1600s.", evidence_ref=WALK_REF,
              claimed_value=1600.0, direction="decrease"),
    ]
    report = VERIFIER.verify_all(claims, STATE)
    assert report.n_claims == 5
    assert report.n_faithful == 2, f"expected 2 faithful, got {report.n_faithful}"
    assert abs(report.hallucination_rate - 0.6) < 1e-9

    breakdown = report.failure_breakdown()
    assert breakdown["missing_or_bad_ref"] == 1
    assert breakdown["direction_error"] == 1
    assert breakdown["value_mismatch"] == 1

    print(f"  V10 {report.summary()}")

    filtered, _ = VERIFIER.filter_faithful(claims, STATE)
    assert len(filtered) == 2
    assert {c.claim_id for c in filtered} == {"k1", "k2"}
    print(f"  V10 deployment filter withheld 3/5 claims, surfaced "
          f"{sorted(c.claim_id for c in filtered)}")

    pooled = aggregate_reports([report, report])
    assert pooled.n_claims == 10
    assert abs(pooled.hallucination_rate - 0.6) < 1e-9
    print(f"  V10 pooled over 2 reports: {pooled.summary()}")


def test_v11_unconstrained_baseline_scoring() -> None:
    """Ablation support: unconstrained output has no refs at all.

    With require_evidence_ref=True (deployment) such claims score as unfaithful, which
    is exactly the constrained-vs-unconstrained contrast the paper reports.
    """
    claims = [
        Claim(claim_id="u1", text="The resident seems less active than usual.",
              evidence_ref=WALK_REF),  # schema forces a ref; content is vague
    ]
    strict = FaithfulnessVerifier(VerifierConfig(check_lexical_direction=True))
    report = strict.verify_all(claims, STATE)
    print(f"  V11 vague-but-referenced claim: faithful={report.n_faithful}/1 "
          f"(prose 'less active' matches actual decrease)")


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
