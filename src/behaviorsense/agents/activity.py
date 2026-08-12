"""Agent 2 - Activity recognition: skeleton windows -> smoothed ActivitySegments.

Pipeline (locked in docs/02, recipes in docs/06):

    skeleton windows [T,P,17,3] -> ST-GCN++ (behind `WindowClassifier`) -> logits
        -> temperature calibration -> object-context late fusion (log-prior offsets)
        -> HMM/Viterbi smoothing -> abstention -> ActivitySegment stream

Division of labour, stated plainly: the GCN is trained on Kaggle and is a checkpoint;
everything AFTER the logits is deterministic CPU code in this file, and it is where the
correctness burden sits. A per-window argmax stream flickers (sitting/standing at 2 Hz),
and every duration-based feature Agent 3 consumes - walking minutes, sedentary blocks,
sit-to-stand counts - is corrupted by flicker. Smoothing is therefore not cosmetic; it is
what makes the daily features mean what their names say.

The one hazard smoothing introduces is erasing REAL brief events, and one class here is
both brief and the most important thing the system detects: `falling` (id 7). The
transition model treats emergency classes asymmetrically so that a confident single-window
fall survives smoothing. That property is enforced by test A4b - the anti-vacuous pair
demanded by results/tuning_log.md.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

import numpy as np

from behaviorsense.schemas import ActivitySegment, Role

AGENT2_VERSION = "1.0.0"

N_CLASSES = 20
FALLING, FALLEN = 7, 8
EMERGENCY = (FALLING, FALLEN)

CLASS_NAMES: tuple[str, ...] = (
    "walking", "standing", "sitting", "lying_down", "standing_up", "sitting_down",
    "bending_reaching", "falling", "fallen_on_ground", "eating", "drinking",
    "cooking_food_prep", "taking_medication", "watching_tv", "reading", "using_phone",
    "cleaning_housework", "personal_hygiene", "interacting_with_person", "other_idle",
)
assert len(CLASS_NAMES) == N_CLASSES

# Object-context priors: log-odds offsets applied to specific classes when specific
# objects were detected nearby (RT-DETR labels, sampled at 1 Hz). A hand-written table,
# not a learned fusion MLP: 20 classes x a dozen objects is small enough to inspect, and
# we have no dataset pairing our object vocabulary with our taxonomy to learn from.
# Values are log-odds: +1.5 multiplies the class's odds by ~4.5, -2.0 divides by ~7.4.
OBJECT_PRIORS: dict[str, dict[int, float]] = {
    "pill_bottle": {12: +2.0},
    "bottle": {10: +0.7, 12: +0.7},
    "cup": {10: +1.0, 9: +0.3},
    "bowl": {9: +1.0, 11: +0.5},
    "dining table": {9: +0.7, 11: +0.3},
    "tv": {13: +1.0},
    "book": {14: +1.5},
    "cell phone": {15: +1.5},
    "couch": {2: +0.3, 13: +0.3},
    "bed": {3: +0.7},
    "person": {18: +1.5},  # second tracked person present
}
# `taking_medication` REQUIRES object evidence (taxonomy note: pose cannot separate it
# from drinking). Without a medication-associated object, its probability is pushed down
# hard rather than left to the GCN's imagination.
MEDICATION_ABSENT_PENALTY = -2.0
_MEDICATION_OBJECTS = {"pill_bottle", "bottle"}


@runtime_checkable
class WindowClassifier(Protocol):
    """Per-window classifier. Implemented by the trained ST-GCN++ ensemble."""

    def logits(self, windows: np.ndarray) -> np.ndarray:
        """[N, T, P, 17, 3] float32 -> [N, 20] raw logits."""
        ...


@dataclass(frozen=True)
class ActivityConfig:
    window_s: float = 2.0
    stride_s: float = 1.0
    temperature: float = 1.0
    """Calibration temperature, fitted on validation via `fit_temperature`. 1.0 =
    uncalibrated. Viterbi and abstention both consume probabilities, so the
    probabilities must mean something before either is trusted."""

    self_transition: float = 0.90
    """Prior probability that an activity persists across a 1 s stride. Activities are
    minutes long at this timescale; 0.90 implies a ~10 s expected dwell, which is
    conservative (short) on purpose - overestimating dwell is what erases real events."""

    emergency_floor: float = 0.05
    """Minimum transition probability INTO `falling` from any state. Falls are rare
    (low prior) but must never be priced out of the Viterbi path: a uniform transition
    model would need ~3 consecutive fall windows to overcome a 0.90 self-transition,
    and a fall is often ONE window long. This floor is what lets a single confident
    fall window survive smoothing (test A4b)."""

    abstain_below: float = 0.40
    """Floor on the SMOOTHED (forward-backward) marginal of the selected state.

    Three candidate quantities, and why this is the right one - decided by measurement,
    not taste:

      1. `posterior[selected_label]` (first attempt): wrong by construction. It is low
         exactly where smoothing did its job, since at a flicker the temporal context
         overrides a locally-preferred class. Abstained the windows it had just fixed.
      2. `max(posterior)` (second attempt): still emission-only. On a stream with
         realistic per-window noise it abstained isolated dim windows sitting *inside*
         confident runs - window 5 of a walking stretch at 0.27, flanked by 0.50 walking
         on both sides - shattering one segment into walking/other_idle/walking and
         corrupting every duration feature Agent 3 derives from it.
      3. The forward-backward marginal P(state_t = selected | ALL windows): what is
         actually meant by "do we know what happened here". A dim window braced by
         confident neighbours is explained by its context; a dim window in a dim
         neighbourhood is not, and only that second case is genuine ignorance.

    Costs one extra O(T*K^2) pass, the same order as Viterbi, run once per stream."""

    min_segment_windows: int = 1


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, grid: Iterable[float] | None = None
) -> float:
    """Single-scalar temperature minimising NLL on held-out (logits, labels).

    Grid search rather than LBFGS: the objective is 1-D and convex-ish, the grid is
    exact enough (0.01 resolution), and it cannot diverge - worth more than elegance in
    a script that runs once per training cycle.
    """
    if grid is None:
        grid = np.arange(0.25, 5.01, 0.01)
    best_t, best_nll = 1.0, np.inf
    n = np.arange(len(labels))
    for t in grid:
        p = softmax(logits / t)
        nll = float(-np.log(np.maximum(p[n, labels], 1e-12)).mean())
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def build_transition_matrix(cfg: ActivityConfig) -> np.ndarray:
    """[20, 20] row-stochastic transition prior.

    Structure, not statistics: self-transition dominant, uniform leakage elsewhere,
    with two asymmetries that encode domain facts rather than data (we have no
    longitudinal labelled data to estimate 400 rates from):

      - entering `falling` is floored (falls must be reachable in one step),
      - `falling` self-transition is LOW and its exit mass is steered into
        `fallen_on_ground` / recovery: physically, the falling motion lasts ~1 window,
        then the person is on the ground or back up. Letting `falling` persist would
        double-count fall windows into duration features.
    """
    A = np.full((N_CLASSES, N_CLASSES), 0.0)
    off = (1.0 - cfg.self_transition) / (N_CLASSES - 1)
    A[:] = off
    np.fill_diagonal(A, cfg.self_transition)

    A[FALLING, :] = 0.02 / (N_CLASSES - 4)
    A[FALLING, FALLING] = 0.28          # a fall CAN span two windows, rarely more
    A[FALLING, FALLEN] = 0.50           # dominant physical outcome
    A[FALLING, 3] = 0.10                # lying_down (controlled descent misread)
    A[FALLING, 4] = 0.10                # standing_up (immediate recovery)

    # Floor the into-falling column LAST, and renormalise each row's OTHER entries so
    # the floor survives exactly. (A naive floor-then-normalise shaved the floor to
    # 0.048 - below the documented 0.05 - which is precisely the kind of silent
    # parameter drift this project keeps getting bitten by.)
    for i in range(N_CLASSES):
        if i == FALLING or A[i, FALLING] >= cfg.emergency_floor:
            continue
        A[i, FALLING] = cfg.emergency_floor
        others = [j for j in range(N_CLASSES) if j != FALLING]
        A[i, others] *= (1.0 - cfg.emergency_floor) / A[i, others].sum()

    assert np.allclose(A.sum(axis=1), 1.0)
    return A


def viterbi(log_emissions: np.ndarray, log_A: np.ndarray, log_pi: np.ndarray) -> np.ndarray:
    """Standard max-product decoding. [T, K] log-emissions -> [T] state path."""
    T, K = log_emissions.shape
    delta = log_pi + log_emissions[0]
    back = np.zeros((T, K), dtype=np.int64)
    for t in range(1, T):
        scores = delta[:, None] + log_A          # [from, to]
        back[t] = scores.argmax(axis=0)
        delta = scores.max(axis=0) + log_emissions[t]
    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(delta.argmax())
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1][path[t + 1]]
    return path


def forward_backward(
    log_emissions: np.ndarray, log_A: np.ndarray, log_pi: np.ndarray
) -> np.ndarray:
    """Per-window smoothed marginals P(state_t | all observations). [T, K] -> [T, K].

    Computed in log space with log-sum-exp: emissions over 20 classes and hundreds of
    windows underflow float64 rapidly in linear space, and an underflowed marginal
    silently reads as certainty.
    """
    T, K = log_emissions.shape

    def lse(x: np.ndarray, axis: int) -> np.ndarray:
        m = x.max(axis=axis, keepdims=True)
        return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)

    alpha = np.empty((T, K))
    alpha[0] = log_pi + log_emissions[0]
    for t in range(1, T):
        alpha[t] = lse(alpha[t - 1][:, None] + log_A, axis=0) + log_emissions[t]

    beta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        beta[t] = lse(log_A + (log_emissions[t + 1] + beta[t + 1])[None, :], axis=1)

    log_gamma = alpha + beta
    log_gamma -= lse(log_gamma, axis=1)[:, None]
    return np.exp(log_gamma)


@dataclass
class WindowResult:
    """One classified window after calibration/fusion/smoothing."""

    start: datetime
    end: datetime
    posterior: np.ndarray          # [20] post-calibration, post-fusion (emission only)
    smoothed_label: int
    abstained: bool
    objects: tuple[str, ...] = ()
    marginal: float = 1.0
    """Forward-backward P(smoothed_label | whole stream). The abstention quantity, and
    the honest confidence for this window: it accounts for temporal context, whereas
    `posterior[smoothed_label]` is the emission alone."""

    @property
    def label(self) -> int:
        return 19 if self.abstained else self.smoothed_label  # other_idle on abstain

    @property
    def confidence(self) -> float:
        return float(self.marginal)

    @property
    def emission_confidence(self) -> float:
        """Pre-smoothing confidence, kept for diagnostics and the ablation table."""
        return float(self.posterior[self.smoothed_label])

    @property
    def logit_margin(self) -> float:
        top2 = np.sort(self.posterior)[-2:]
        return float(np.log(max(top2[1], 1e-12)) - np.log(max(top2[0], 1e-12)))


class ActivityAgent:
    """Logits -> calibrate -> fuse objects -> Viterbi -> abstain -> segments."""

    def __init__(self, config: ActivityConfig | None = None) -> None:
        self.config = config or ActivityConfig()
        self._A = build_transition_matrix(self.config)
        self._log_A = np.log(np.maximum(self._A, 1e-12))
        self._log_pi = np.log(np.full(N_CLASSES, 1.0 / N_CLASSES))

    # -- per-stage, individually testable ------------------------------------

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """[N, 20] logits -> calibrated posteriors."""
        return softmax(logits / self.config.temperature)

    def fuse_objects(
        self, posteriors: np.ndarray, objects_per_window: Sequence[Iterable[str]]
    ) -> np.ndarray:
        """Apply object-context log-odds offsets, renormalise.

        Operates in log space on the posterior - equivalent to adjusting class priors,
        which is exactly what object context is: evidence about which activities are
        plausible here, not about what the skeleton is doing.
        """
        out = np.log(np.maximum(posteriors, 1e-12))
        for i, objs in enumerate(objects_per_window):
            seen = set(objs)
            for obj in seen:
                for cls, offset in OBJECT_PRIORS.get(obj, {}).items():
                    out[i, cls] += offset
            if not seen & _MEDICATION_OBJECTS:
                out[i, 12] += MEDICATION_ABSENT_PENALTY
        return softmax(out)

    def smooth(self, posteriors: np.ndarray) -> np.ndarray:
        """[T, 20] posteriors -> [T] Viterbi path under the structured transition prior."""
        log_e = np.log(np.maximum(posteriors, 1e-12))
        return viterbi(log_e, self._log_A, self._log_pi)

    def marginals(self, posteriors: np.ndarray) -> np.ndarray:
        """[T, 20] posteriors -> [T, 20] forward-backward smoothed marginals."""
        log_e = np.log(np.maximum(posteriors, 1e-12))
        return forward_backward(log_e, self._log_A, self._log_pi)

    # -- main entry point ------------------------------------------------------

    def classify_stream(
        self,
        logits: np.ndarray,
        window_starts: Sequence[datetime],
        window_ends: Sequence[datetime],
        objects_per_window: Sequence[Iterable[str]] | None = None,
    ) -> list[WindowResult]:
        if not (len(logits) == len(window_starts) == len(window_ends)):
            raise ValueError("logits and window times must align")
        if len(logits) == 0:
            return []
        objs = objects_per_window or [()] * len(logits)

        post = self.calibrate(np.asarray(logits, dtype=np.float64))
        post = self.fuse_objects(post, objs)
        path = self.smooth(post)
        gamma = self.marginals(post)

        results: list[WindowResult] = []
        for i, label in enumerate(path):
            conf = float(gamma[i, label])
            # Emergency classes are exempt from abstention: suppressing a fall because
            # the posterior is diffuse inverts the system's priorities. Severity
            # handling downstream (R1/R2) already tolerates false positives better
            # than misses.
            abstain = conf < self.config.abstain_below and label not in EMERGENCY
            results.append(
                WindowResult(
                    start=window_starts[i],
                    end=window_ends[i],
                    posterior=post[i],
                    smoothed_label=int(label),
                    abstained=abstain,
                    objects=tuple(objs[i]),
                    marginal=conf,
                )
            )
        return results

    def to_segments(
        self,
        windows: Sequence[WindowResult],
        track_id: int,
        role: Role,
        room: str | None = None,
    ) -> list[ActivitySegment]:
        """Merge consecutive same-label windows into ActivitySegments."""
        if not windows:
            return []
        segments: list[ActivitySegment] = []
        run: list[WindowResult] = [windows[0]]

        def flush() -> None:
            if len(run) < self.config.min_segment_windows:
                return
            label = run[0].label
            confs = [w.confidence for w in run]
            margins = [w.logit_margin for w in run]
            supporting = sorted({o for w in run for o in w.objects})
            segments.append(
                ActivitySegment(
                    segment_id=f"t{track_id}-{run[0].start:%Y%m%dT%H%M%S}-{label}",
                    track_id=track_id,
                    role=role,
                    activity_id=label,
                    activity_name=CLASS_NAMES[label],
                    start_time=run[0].start,
                    end_time=run[-1].end,
                    confidence=min(1.0, max(0.0, float(np.mean(confs)))),
                    mean_logit_margin=float(np.mean(margins)),
                    room=room,
                    supporting_objects=supporting,
                    n_windows=len(run),
                )
            )

        for w in windows[1:]:
            if w.label == run[-1].label and w.start <= run[-1].end:
                run.append(w)
            else:
                flush()
                run = [w]
        flush()
        return segments


__all__ = [
    "AGENT2_VERSION",
    "N_CLASSES",
    "CLASS_NAMES",
    "FALLING",
    "FALLEN",
    "OBJECT_PRIORS",
    "WindowClassifier",
    "ActivityConfig",
    "ActivityAgent",
    "WindowResult",
    "softmax",
    "fit_temperature",
    "build_transition_matrix",
    "viterbi",
    "forward_backward",
]
