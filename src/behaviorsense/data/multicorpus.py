"""Training over corpora that label different halves of the taxonomy.

The problem this solves is specific and it is not solved by concatenation.

`behaviorsense-toyota-shards` covers 15 of the 22 coarse classes and cannot cover the other
seven: Toyota annotates `Sit_down` and `Get_up` as transitions and never labels the sustained
`sitting`/`standing` between them, has no `bending_reaching` in its vocabulary, contains no
falls at all, records one resident so `interacting_with_person` never occurs, and a reject
class has no positives to donate. The Charades shards cover the postures. The fall corpora
cover `falling`/`fallen_on_ground` and nothing else.

Concatenate them under ordinary cross-entropy and every Toyota window becomes a statement
that the resident is *not* sitting - across 365,492 windows, a third of which are the
unannotated gaps where sitting is most likely. That is not a small mislabelling; it is
training the model to suppress the exact classes another corpus is trying to teach it.

So the loss ABSTAINS. For a window from corpus C, only the classes C can label enter the
softmax; the rest are masked to -inf and receive no gradient, neither positive nor negative.
The model is asked "which of the things this corpus knows about is happening", which is the
only question the label can answer.

Fine labels ride along as AUXILIARY supervision rather than being collapsed away. Toyota's
31 trimmed and 51 untrimmed classes distinguish `Cook.Cut` from `Cook.Stir` and
`Drink.From_cup` from `Drink.From_glass`; the 22-class taxonomy does not. Collapsing before
the loss discards that, and a deliberately multi-modal target is the condition a softmax head
handles worst. The two fine spaces are kept SEPARATE - `Readbook` and `Read` are the same
activity under two spellings, but `Cook.Cleanup` has no exact untrimmed counterpart, and
inventing one would be another hand-written label map of precisely the kind that produced
macro-F1 0.128 on Charades.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from behaviorsense.data.toyota import (
    COARSE_V11,
    TSM_CLASSES,
    TSM_TO_COARSE,
    TSU_CLASSES,
    TSU_TO_COARSE,
    collapse_matrix,
    supervised_coarse,
)

N_COARSE = len(COARSE_V11)

# The `datasets` string each shard writes, and what that corpus can actually label.
#
# `supervised` is the set of coarse ids allowed into the softmax for a window from this
# corpus. For the Toyota entries it is DERIVED from the map rather than listed, so adding a
# class to `TSU_TO_COARSE` cannot silently leave the mask behind.
#
# Charades and the fall corpora are listed explicitly because their shards predate this
# module and carry no fine labels: the posture classes are what Charades is here for, and
# `falling`/`fallen_on_ground` are the only thing the fall corpora are trusted on. Both lists
# are narrower than what those shards nominally contain, and that is deliberate - see
# CHARADES_TRUSTED below.


def _ids(*names: str) -> tuple[int, ...]:
    return tuple(sorted(COARSE_V11.index(n) for n in names))


# Charades labels all 20 of the v1.0 classes, but its measured per-class F1 says which of
# them it can actually teach. `bending_reaching` sat at 26 val windows and F1 0.000 because
# the keyword map has no verb for it; `standing` at 123 windows and F1 0.005. Feeding those
# in as supervision adds noise to classes Toyota cannot correct. The postures Charades DOES
# carry - `sitting` 0.341, `lying_down` 0.435, both above everything else it produced - are
# the reason it stays in the mix at all.
CHARADES_TRUSTED = _ids(
    "walking", "standing", "sitting", "lying_down", "standing_up", "sitting_down",
    "bending_reaching", "eating", "drinking", "cooking_food_prep", "taking_medication",
    "watching_tv", "reading", "using_phone", "cleaning_housework", "personal_hygiene",
    "interacting_with_person", "other_idle",
)
FALL_TRUSTED = _ids("falling", "fallen_on_ground", "standing", "walking", "sitting",
                    "lying_down", "bending_reaching", "other_idle")


@dataclass(frozen=True)
class CorpusSpec:
    """One shard family: which fine space it uses and which coarse classes it may supervise."""

    tag: str
    """The value in the shard's `datasets` column."""
    fine_names: tuple[str, ...] = ()
    """Its own fine class list, or empty when the shards carry no `fine_labels`."""
    supervised: tuple[int, ...] = ()
    """Coarse ids allowed into the masked softmax."""
    fine_space: str = ""
    """Name of the auxiliary head this corpus's fine labels train. Corpora sharing a name
    share a head; an empty string means no fine supervision."""

    @property
    def n_fine(self) -> int:
        return len(self.fine_names)


CORPORA: dict[str, CorpusSpec] = {
    "toyota_trimmed": CorpusSpec(
        tag="toyota_trimmed",
        fine_names=TSM_CLASSES,
        supervised=supervised_coarse(TSM_TO_COARSE),
        fine_space="tsm31",
    ),
    "toyota_untrimmed": CorpusSpec(
        tag="toyota_untrimmed",
        fine_names=TSU_CLASSES,
        supervised=supervised_coarse(TSU_TO_COARSE),
        fine_space="tsu51",
    ),
    "charades": CorpusSpec(tag="charades", supervised=CHARADES_TRUSTED),
}
for _fall in ("gmdcsa", "urfd", "caucafall", "le2i", "omnifall"):
    CORPORA[_fall] = CorpusSpec(tag=_fall, supervised=FALL_TRUSTED)


def fine_spaces() -> dict[str, int]:
    """Auxiliary head name -> width. Two heads for Toyota's two vocabularies."""
    out: dict[str, int] = {}
    for spec in CORPORA.values():
        if spec.fine_space:
            out[spec.fine_space] = spec.n_fine
    return out


def spec_for(tag: str) -> CorpusSpec:
    """Look up a corpus by its `datasets` value, refusing an unknown one.

    Refusing matters: an unrecognised tag defaulting to "supervise everything" would
    reintroduce exactly the all-classes-are-negative problem this module exists to prevent,
    and it would do it silently for whichever corpus was added last.
    """
    if tag not in CORPORA:
        raise KeyError(
            f"unknown corpus tag {tag!r}. Add a CorpusSpec naming which coarse classes it "
            f"can supervise. Known: {sorted(CORPORA)}"
        )
    return CORPORA[tag]


def supervision_mask(tags: Sequence[str], n_coarse: int = N_COARSE) -> np.ndarray:
    """[N, n_coarse] bool: may this class enter the softmax for this window?"""
    mask = np.zeros((len(tags), n_coarse), dtype=bool)
    for i, tag in enumerate(tags):
        mask[i, list(spec_for(str(tag)).supervised)] = True
    return mask


def collapse_for(tag: str) -> np.ndarray | None:
    """[n_fine, n_coarse] membership matrix for a corpus, or None if it has no fine space."""
    spec = spec_for(tag)
    if not spec.fine_space:
        return None
    mapping = TSM_TO_COARSE if spec.fine_space == "tsm31" else TSU_TO_COARSE
    return collapse_matrix(spec.fine_names, mapping)


def masked_cross_entropy(logits, targets, mask, reduction: str = "mean"):
    """Cross-entropy over the classes a sample's corpus can label, and no others.

    Masked classes are set to -inf BEFORE the softmax, which is what makes the abstention
    real: they contribute nothing to the normaliser, so they receive no gradient in either
    direction. Zeroing their loss afterwards would not do this - the softmax would still
    normalise over them, so predicting a masked class would still be penalised, and the model
    would still learn that Toyota windows are "not sitting".

    Raises if a target is itself masked. That combination cannot be satisfied by any
    prediction, so the loss would be +inf and the run would produce NaN weights several
    minutes later with nothing in the log pointing at the label that caused it.
    """
    import torch                                                    # noqa: PLC0415
    import torch.nn.functional as F                                 # noqa: PLC0415

    if logits.shape != mask.shape:
        raise ValueError(f"logits {tuple(logits.shape)} vs mask {tuple(mask.shape)}")
    hit = mask.gather(1, targets.view(-1, 1)).squeeze(1)
    if not bool(hit.all()):
        bad = (~hit).nonzero().flatten()[:5].tolist()
        raise ValueError(
            f"{int((~hit).sum())} target(s) fall outside their own corpus's supervised set "
            f"(rows {bad}). A masked target is unsatisfiable - the loss is +inf and the "
            "weights go NaN minutes later. Check the CorpusSpec against the shard's labels."
        )
    return F.cross_entropy(logits.masked_fill(~mask, float("-inf")), targets,
                           reduction=reduction)


def pooled_coarse_logits(fine_logits, collapse):
    """Fine logits -> coarse logits by log-sum-exp over each coarse class's members.

    The probabilistic reading of "this coarse class occurred": the sum of its members'
    probabilities. `logsumexp` is that sum in log space, so the result composes with a
    softmax exactly as a logit should.

    Used to read a coarse posterior off an auxiliary fine head - not to train the coarse head
    itself, which has its own masked loss. Averaging instead of log-sum-exp would make five
    `Cook.*` members each dilute the group rather than reinforce it, so a class split into
    many members would be systematically under-predicted relative to a single-member one.
    """
    import torch                                                    # noqa: PLC0415

    members = torch.as_tensor(collapse, dtype=torch.bool, device=fine_logits.device)
    out = fine_logits.new_full((fine_logits.shape[0], members.shape[1]), float("-inf"))
    for c in range(members.shape[1]):
        sel = members[:, c]
        if bool(sel.any()):
            out[:, c] = torch.logsumexp(fine_logits[:, sel], dim=1)
    return out


__all__ = [
    "CHARADES_TRUSTED",
    "CORPORA",
    "FALL_TRUSTED",
    "N_COARSE",
    "CorpusSpec",
    "collapse_for",
    "fine_spaces",
    "masked_cross_entropy",
    "pooled_coarse_logits",
    "spec_for",
    "supervision_mask",
]
