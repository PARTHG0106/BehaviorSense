"""Agent 4 - claim-level faithfulness verification.

This module is the project's primary research contribution.

The problem: "an LLM explains the alerts" is a demo, not research, because the obvious
question - *how do you know the explanation is true?* - has no answer. LLMs asked to
narrate numeric state routinely invent figures, invert directions ("increased" for a
decrease), and cite evidence that does not exist. In a caregiver-facing health context
that is not a cosmetic flaw.

The approach: constrain generation to a schema in which every claim must carry a
machine-resolvable `evidence_ref`, then verify each claim ARITHMETICALLY against the
behavioural state store. Four independent checks per claim:

    C1  ref_exists          does the cited evidence row exist at all?
    C2  value_matches       does the quoted number match the stored value?
    C3  pct_matches         does the quoted percentage match a recomputation?
    C4  direction_consistent  does the prose direction match the sign of the delta?

C4 is the subtle one and catches the most dangerous failure: a report that quotes the
right number while describing it backwards ("mobility improved by 42%" when it fell)
is worse than no report, and C1-C3 alone would pass it.

Design constraint - THE VERIFIER USES NO LLM. Using a model to check a model shares
failure modes and inherits the same blind spots; arithmetic does not. Every check here
is deterministic, cheap, and independently auditable.

Output metric: `hallucination_rate` = fraction of claims failing any check. Reported
constrained vs unconstrained, this is the headline experimental result.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from behaviorsense.schemas import (
    BehaviourState,
    Claim,
    Evidence,
    VerificationResult,
)

REF_PATTERN = re.compile(r"^feat:([a-z0-9_]+):(\d{4}-\d{2}-\d{2})$")

# Direction words that may appear in claim prose. Used by the lexical cross-check,
# which catches text/field disagreement inside a single claim.
INCREASE_WORDS = frozenset({
    "increase", "increased", "increases", "rose", "rise", "risen", "higher",
    "up", "improved", "improvement", "gained", "more", "longer", "greater",
    "above", "exceeded", "grew",
})
DECREASE_WORDS = frozenset({
    "decrease", "decreased", "decreases", "fell", "fall", "fallen", "lower",
    "down", "declined", "decline", "reduced", "reduction", "less", "fewer",
    "shorter", "below", "dropped", "drop", "worsened", "deteriorated",
})

_WORD_RE = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class VerifierConfig:
    """Tolerances for numeric agreement.

    Tolerances are deliberately tight. A generous tolerance would let a model paraphrase
    numbers ("about 40%") into a pass and inflate the faithfulness score, which would
    quietly destroy the metric's meaning.
    """

    value_rel_tol: float = 0.02
    """2% relative tolerance on quoted absolute values. Accommodates sensible rounding
    (1800.0 -> "1800" or "30 minutes") without admitting invention."""

    value_abs_tol: float = 1e-6
    """Absolute floor for near-zero comparisons where relative tolerance is meaningless."""

    pct_abs_tol: float = 2.0
    """Percentage points. A claim of "-42%" against a true -40.5% passes; "-25%" fails."""

    require_evidence_ref: bool = True
    """If False, claims without refs are counted unverifiable rather than unfaithful.
    Used to score unconstrained-baseline output fairly in the ablation."""

    check_lexical_direction: bool = True
    """Cross-check direction words in `text` against the structured `direction` field."""


@dataclass
class VerificationReport:
    """Aggregate verification outcome over a set of claims."""

    results: list[VerificationResult] = field(default_factory=list)
    n_claims: int = 0

    @property
    def n_faithful(self) -> int:
        return sum(1 for r in self.results if r.is_faithful)

    @property
    def hallucination_rate(self) -> float:
        """Fraction of claims failing at least one check. The headline metric."""
        if self.n_claims == 0:
            return 0.0
        return 1.0 - (self.n_faithful / self.n_claims)

    @property
    def faithfulness_rate(self) -> float:
        return 1.0 - self.hallucination_rate

    def failure_breakdown(self) -> dict[str, int]:
        """Per-check failure counts, for the error analysis table."""
        return {
            "missing_or_bad_ref": sum(1 for r in self.results if not r.ref_exists),
            "value_mismatch": sum(
                1 for r in self.results if r.ref_exists and not r.value_matches
            ),
            "pct_mismatch": sum(
                1 for r in self.results if r.ref_exists and not r.pct_matches
            ),
            "direction_error": sum(
                1 for r in self.results if r.ref_exists and not r.direction_consistent
            ),
        }

    def summary(self) -> str:
        b = self.failure_breakdown()
        return (
            f"claims={self.n_claims} faithful={self.n_faithful} "
            f"hallucination_rate={self.hallucination_rate:.3f} | "
            f"bad_ref={b['missing_or_bad_ref']} value={b['value_mismatch']} "
            f"pct={b['pct_mismatch']} direction={b['direction_error']}"
        )


class FaithfulnessVerifier:
    """Deterministic claim verifier.

    Usage::

        verifier = FaithfulnessVerifier()
        report = verifier.verify_all(claims, state)
        print(report.hallucination_rate)
    """

    def __init__(self, config: VerifierConfig | None = None) -> None:
        self.config = config or VerifierConfig()

    # -- evidence resolution ----------------------------------------------------

    def build_index(self, state: BehaviourState) -> dict[str, Evidence]:
        """Index every resolvable evidence ref for this state.

        Alert evidence is the primary source, but a claim may legitimately cite any
        tracked feature for the report day - the LLM is shown the full daily feature
        vector, not only alerting features. Synthesising evidence for non-alerting
        features keeps honest contextual claims ("meals were normal at 3") verifiable
        instead of being wrongly scored as hallucinations.
        """
        index: dict[str, Evidence] = dict(state.evidence_index())

        day = state.report_day
        items = state.today.numeric_items()
        for name, value in items.items():
            ref = f"feat:{name}:{day.isoformat()}"
            if ref in index:
                continue
            baseline = state.baselines.get(name)
            if baseline is None:
                continue
            index[ref] = Evidence.build(name, value, baseline, day)
        return index

    # -- individual checks ------------------------------------------------------

    def _values_agree(self, claimed: float, actual: float) -> bool:
        cfg = self.config
        if abs(actual) < cfg.value_abs_tol:
            return abs(claimed) < max(cfg.value_abs_tol, 1e-3)
        return abs(claimed - actual) <= cfg.value_rel_tol * abs(actual)

    def _lexical_direction(self, text: str) -> str | None:
        """Infer direction from prose. None if absent or contradictory.

        Contradictory text (both increase and decrease words) yields None rather than a
        guess - an ambiguous sentence should not be scored as a confident pass.
        """
        words = set(_WORD_RE.findall(text.lower()))
        up = bool(words & INCREASE_WORDS)
        down = bool(words & DECREASE_WORDS)
        if up == down:  # neither, or both
            return None
        return "increase" if up else "decrease"

    def verify_claim(
        self, claim: Claim, index: dict[str, Evidence]
    ) -> VerificationResult:
        cfg = self.config
        notes: list[str] = []

        # C1 - reference resolution.
        if not claim.evidence_ref:
            return VerificationResult(
                claim_id=claim.claim_id,
                ref_exists=not cfg.require_evidence_ref,
                value_matches=False,
                pct_matches=False,
                direction_consistent=False,
                notes=["no evidence_ref supplied"],
            )

        if not REF_PATTERN.match(claim.evidence_ref):
            return VerificationResult(
                claim_id=claim.claim_id,
                ref_exists=False,
                value_matches=False,
                pct_matches=False,
                direction_consistent=False,
                notes=[f"malformed evidence_ref: {claim.evidence_ref!r}"],
            )

        evidence = index.get(claim.evidence_ref)
        if evidence is None:
            return VerificationResult(
                claim_id=claim.claim_id,
                ref_exists=False,
                value_matches=False,
                pct_matches=False,
                direction_consistent=False,
                notes=[
                    f"evidence_ref {claim.evidence_ref!r} does not resolve "
                    "(fabricated citation)"
                ],
            )

        # C2 - quoted absolute value.
        if claim.claimed_value is None:
            value_matches = True
            notes.append("no absolute value quoted; C2 vacuous")
        else:
            value_matches = self._values_agree(
                claim.claimed_value, evidence.observed_value
            )
            if not value_matches:
                notes.append(
                    f"value mismatch: claimed {claim.claimed_value:g}, "
                    f"actual {evidence.observed_value:g}"
                )

        # C3 - quoted percentage change.
        if claim.claimed_pct_change is None:
            pct_matches = True
            notes.append("no percentage quoted; C3 vacuous")
        elif abs(evidence.baseline_median) < 1e-9:
            # Percentage change is undefined against a zero baseline; quoting one is
            # itself a defect, since no correct value exists.
            pct_matches = False
            notes.append(
                "percentage quoted against a zero baseline median (undefined)"
            )
        else:
            pct_matches = (
                abs(claim.claimed_pct_change - evidence.pct_change)
                <= cfg.pct_abs_tol
            )
            if not pct_matches:
                notes.append(
                    f"pct mismatch: claimed {claim.claimed_pct_change:+.1f}%, "
                    f"actual {evidence.pct_change:+.1f}%"
                )

        # C4 - direction. The check that catches inverted narration.
        actual_direction = evidence.direction
        direction_consistent = True

        if claim.direction is not None and claim.direction != actual_direction:
            direction_consistent = False
            notes.append(
                f"direction field wrong: claimed {claim.direction}, "
                f"actual {actual_direction} (delta {evidence.delta:+g})"
            )

        if cfg.check_lexical_direction:
            lexical = self._lexical_direction(claim.text)
            if lexical is not None and actual_direction != "unchanged":
                if lexical != actual_direction:
                    direction_consistent = False
                    notes.append(
                        f"prose says {lexical!r} but data shows {actual_direction!r} "
                        f"(delta {evidence.delta:+g})"
                    )
            # Internal contradiction between the structured field and the prose.
            if (
                lexical is not None
                and claim.direction is not None
                and lexical != claim.direction
            ):
                direction_consistent = False
                notes.append(
                    f"claim is self-inconsistent: text implies {lexical!r}, "
                    f"direction field says {claim.direction!r}"
                )

        # A quoted percentage whose sign contradicts the data is a direction error too,
        # even when the model omitted the direction field and used neutral prose.
        if (
            claim.claimed_pct_change is not None
            and abs(claim.claimed_pct_change) > 1e-9
            and abs(evidence.delta) > 1e-9
        ):
            if (claim.claimed_pct_change > 0) != (evidence.delta > 0):
                direction_consistent = False
                notes.append(
                    f"quoted pct sign ({claim.claimed_pct_change:+.1f}%) contradicts "
                    f"actual delta ({evidence.delta:+g})"
                )

        return VerificationResult(
            claim_id=claim.claim_id,
            ref_exists=True,
            value_matches=value_matches,
            pct_matches=pct_matches,
            direction_consistent=direction_consistent,
            notes=notes,
        )

    # -- batch ------------------------------------------------------------------

    def verify_all(
        self, claims: Sequence[Claim], state: BehaviourState
    ) -> VerificationReport:
        index = self.build_index(state)
        results = [self.verify_claim(c, index) for c in claims]
        return VerificationReport(results=results, n_claims=len(claims))

    def filter_faithful(
        self, claims: Sequence[Claim], state: BehaviourState
    ) -> tuple[list[Claim], VerificationReport]:
        """Return only claims that pass every check, plus the full report.

        This is the deployment path: unfaithful claims are withheld from the caregiver
        rather than shown with a warning. A flagged-but-visible false clinical claim is
        still a false clinical claim.
        """
        report = self.verify_all(claims, state)
        ok = {r.claim_id for r in report.results if r.is_faithful}
        return [c for c in claims if c.claim_id in ok], report


def aggregate_reports(reports: Iterable[VerificationReport]) -> VerificationReport:
    """Pool many per-report verifications into one corpus-level result."""
    pooled = VerificationReport()
    for r in reports:
        pooled.results.extend(r.results)
        pooled.n_claims += r.n_claims
    return pooled


__all__ = [
    "REF_PATTERN",
    "VerifierConfig",
    "VerificationReport",
    "FaithfulnessVerifier",
    "aggregate_reports",
]
