"""Post-hoc accuracy levers for the ADL classifier, measured on SAVED validation logits.

Why this module exists
----------------------
P1 reported mean-class accuracy 0.189 for the best single stream and 0.151 for the
four-stream logit ensemble. Several standard remedies for exactly that shape of result -
long-tailed labels, a uniform average over streams of very unequal quality - need no
retraining and no GPU at all. They need the validation logits, which notebook 04 computed
and threw away, so testing any of them cost a fresh two-hour Kaggle session.

Notebook 04 now writes `results/val_logits.npz`. Everything here operates on that file, on
CPU, in seconds, as many times as you like. The expensive part (a forward pass over 24,908
windows with four checkpoints) happens once.

The three levers, and why each is plausible rather than hopeful
--------------------------------------------------------------
`logit_adjust` - the classifier was trained with cross-entropy on a distribution where
`other_idle` is 39.4% of windows, then scored with mean-class accuracy, which weights every
class equally. Those are different objectives. Subtracting tau * log(prior) from each logit
is the standard correction (Menon et al., "Long-tail learning via logit adjustment") and it
is a decision-rule change, not a model change.

`subset_scores` - the ensemble AVERAGES four streams whose individual mean-class accuracies
are 0.180, 0.189, 0.127, 0.123. Averaging a strong estimator with a much weaker one is not
guaranteed to help, and P1 already showed the four-stream ensemble losing to `bone` alone on
mean-class. The obvious next question - which SUBSET is best - was never asked because
asking it required the per-stream logits.

Neither is tuning on test data: this is the same subject-disjoint validation split used for
temperature calibration, and any number from here must be reported as validation-selected.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from behaviorsense.agents.activity import N_CLASSES

MIN_SUPPORT = 50
"""Below ~50 validation windows a class's F1 is one prediction wide. Mirrors train_adl.py
and notebook 04 so the three cannot drift."""


def scores(logits: np.ndarray, y: np.ndarray,
           min_support: int = MIN_SUPPORT) -> tuple[float, float, float, float]:
    """(top-1, mean-class, macro-F1, macro-F1 over well-supported classes).

    Single definition, imported by notebook 04 and by every script here. It used to live
    inline in the notebook, which meant any test of it validated a copy - the failure mode
    that let three notebook-04 defects reach Kaggle.
    """
    pred = logits.argmax(1)
    per_class = [np.mean(pred[y == c] == c) for c in range(N_CLASSES) if (y == c).any()]
    f1, f1_sup = [], []
    for c in range(N_CLASSES):
        tp = ((pred == c) & (y == c)).sum()
        fp = ((pred == c) & (y != c)).sum()
        fn = ((pred != c) & (y == c)).sum()
        if tp + fp + fn:
            v = 2 * tp / max(2 * tp + fp + fn, 1)
            f1.append(v)
            if (y == c).sum() >= min_support:
                f1_sup.append(v)
    return (float(np.mean(pred == y)), float(np.mean(per_class)), float(np.mean(f1)),
            float(np.mean(f1_sup)) if f1_sup else float("nan"))


def class_prior(y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """Empirical label distribution, floored so an absent class cannot produce -inf."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    return counts / counts.sum()


def logit_adjust(logits: np.ndarray, prior: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """`logits - tau * log(prior)`. tau=0 is a no-op; tau=1 is the balanced-error rule.

    Applied at DECISION time, so it changes nothing about the trained model and can be
    swapped or reverted in deployment by editing one number.
    """
    return logits - tau * np.log(prior)[None, :]


def combine(per_stream: dict[str, np.ndarray], streams: tuple[str, ...],
            mode: str = "logit") -> np.ndarray:
    """Average a subset of streams in logit space (product of experts) or probability space.

    Mirrors `EnsembleClassifier.logits` exactly - `log_softmax` first for `prob`, raw logits
    for `logit` - so a subset scored here means the same thing as the same subset served by
    the ensemble class.
    """
    stack = np.stack([per_stream[s].astype(np.float64) for s in streams])
    if mode == "logit":
        return stack.mean(axis=0)
    shifted = stack - stack.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)
    return np.log(np.clip(probs.mean(axis=0), 1e-12, None))


def subset_scores(per_stream: dict[str, np.ndarray], y: np.ndarray,
                  mode: str = "logit", min_size: int = 1) -> list[tuple]:
    """Every stream subset, scored. Returns rows sorted by mean-class accuracy, descending.

    Exhaustive rather than greedy: with four streams there are 15 non-empty subsets, so a
    search heuristic would add a failure mode for no saving.
    """
    names = sorted(per_stream)
    rows = []
    for size in range(min_size, len(names) + 1):
        for combo in combinations(names, size):
            rows.append((combo, *scores(combine(per_stream, combo, mode), y)))
    return sorted(rows, key=lambda r: -r[2])


def tau_sweep(logits: np.ndarray, y: np.ndarray,
              taus: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)) -> list[tuple]:
    """Score `logit_adjust` across tau. tau=0 must reproduce the unadjusted numbers."""
    prior = class_prior(y)
    return [(tau, *scores(logit_adjust(logits, prior, tau), y)) for tau in taus]


def sequences_by_subject(subjects: np.ndarray, values: np.ndarray,
                         min_len: int = 3) -> list[np.ndarray]:
    """Group per-window values into per-subject temporal sequences, in index order.

    Windows for one Charades video are consecutive in the shard and `split_by_subject`
    returns sorted indices, so index order within a subject IS time order. Sequences
    shorter than `min_len` are dropped: a transition prior cannot smooth what has no
    neighbours, and keeping them dilutes any measured effect with cases where smoothed and
    unsmoothed decoding agree by construction.
    """
    out = []
    for s in np.unique(subjects):
        where = np.flatnonzero(subjects == s)
        if len(where) >= min_len:
            out.append(values[where])
    return out


def fit_transition_matrix(label_sequences: list[np.ndarray], n_classes: int = N_CLASSES,
                          alpha: float = 1.0) -> np.ndarray:
    """Row-stochastic transition matrix counted from real label sequences.

    `build_transition_matrix` encodes structure rather than statistics - a flat 0.90
    self-transition with uniform leakage - because when it was written there was no
    labelled sequence data to estimate 400 rates from. There is now: 165,109 windows over
    7,985 Charades videos, in temporal order.

    That default cost real accuracy. Measured on the validation sequences, Viterbi decoding
    under the hand-set prior moved top-1 +0.011 and mean-class **-0.040**: a 0.90
    self-transition plus a 39% `other_idle` class swallows short rare-class runs into the
    surrounding majority, which flatters top-1 and destroys tail recall. Agent 3 consumes
    the smoothed path, so that was a live regression in the deployment metric, invisible
    until segment-level decoding was finally evaluated.

    Fit on TRAINING sequences only. Laplace `alpha` keeps unobserved transitions reachable
    rather than -inf, which matters most for the rare classes this is meant to protect.
    """
    counts = np.full((n_classes, n_classes), alpha, dtype=np.float64)
    for seq in label_sequences:
        seq = np.asarray(seq)
        if len(seq) < 2:
            continue
        np.add.at(counts, (seq[:-1], seq[1:]), 1.0)
    return counts / counts.sum(axis=1, keepdims=True)


def apply_emergency_floor(A: np.ndarray, falling: int, floor: float) -> np.ndarray:
    """Guarantee every state can reach `falling` in one step, exactly.

    A fitted matrix will floor `into-falling` at almost zero, because falls are 0.5% of
    windows - and then Viterbi can never enter the state a 0.90 self-transition is arguing
    against, so a one-window fall is smoothed away. The same renormalisation trick as
    `build_transition_matrix`: raise the column, then rescale the row's OTHER entries so
    the floor survives instead of being shaved back below itself.
    """
    A = np.array(A, dtype=np.float64, copy=True)
    for i in range(A.shape[0]):
        if i == falling or A[i, falling] >= floor:
            continue
        A[i, falling] = floor
        others = [j for j in range(A.shape[1]) if j != falling]
        A[i, others] *= (1.0 - floor) / A[i, others].sum()
    return A


def second_person_present(skeletons: np.ndarray, min_joints: int = 3,
                          min_frames: int = 5) -> np.ndarray:
    """[N, T, M, 17, 3] shard windows -> [N] bool: does person slot 1 hold a real track?

    The cheapest object-context signal in the system, and the only one that needs no
    detector at all: the shards already carry TWO person slots per window, and slot 1 is
    non-zero exactly when the tracker held a second person. `interacting_with_person` is
    the second-worst class in the measured per-class table (F1 0.042) *because* pose alone
    cannot express "someone else is here" - but the extraction pipeline recorded exactly
    that, and it has been sitting in every shard unused. `OBJECT_PRIORS["person"]` was
    written for this signal and has never once received it.

    Thresholds are deliberately conservative: >= `min_joints` visible joints in >=
    `min_frames` frames. A single-joint, single-frame flicker is tracker noise, and a false
    "second person" up-weights class 18 on a window where the resident is alone.
    """
    x = np.asarray(skeletons)
    if x.ndim != 5 or x.shape[-1] != 3 or x.shape[2] < 2:
        raise ValueError(f"expected [N,T,M>=2,17,3], got {x.shape}")
    visible = (x[:, :, 1, :, 2] > 0.05).sum(axis=2)          # [N, T] joints visible
    return (visible >= min_joints).sum(axis=1) >= min_frames


def count_runs(seq: np.ndarray) -> int:
    """Number of contiguous same-label runs in a label sequence."""
    seq = np.asarray(seq)
    if len(seq) == 0:
        return 0
    return 1 + int((seq[1:] != seq[:-1]).sum())


def fragmentation(pred_sequences: list[np.ndarray],
                  true_sequences: list[np.ndarray]) -> dict[str, float]:
    """Segment counts, predicted against ground truth. The half of the story accuracy misses.

    Per-window accuracy after decoding measures LABEL quality. It says nothing about segment
    STRUCTURE, and Agent 3 depends on both: `walking_duration_s` only needs the labels to be
    right, but `walking_bouts` and `mean_bout_duration_s` are counts of runs, so a decoding
    that shatters one true walking stretch into nine flickering ones reports nine bouts and a
    ninth of the mean duration while scoring identically on accuracy.

    That is why "argmax beats Viterbi on mean-class" is not by itself a reason to drop
    smoothing - smoothing exists to produce coherent segments, and this is the number that
    shows what it buys. Reported as a ratio: 1.0 is perfect, >1 fragments, <1 over-merges.
    """
    pred = sum(count_runs(s) for s in pred_sequences)
    true = sum(count_runs(s) for s in true_sequences)
    n = max(1, len(true_sequences))
    return {
        "segments_pred": float(pred),
        "segments_true": float(true),
        "per_sequence": pred / n,
        "ratio": pred / max(1, true),
    }


__all__ = [
    "MIN_SUPPORT",
    "apply_emergency_floor",
    "class_prior",
    "combine",
    "count_runs",
    "fit_transition_matrix",
    "fragmentation",
    "logit_adjust",
    "scores",
    "second_person_present",
    "sequences_by_subject",
    "subset_scores",
    "tau_sweep",
]
