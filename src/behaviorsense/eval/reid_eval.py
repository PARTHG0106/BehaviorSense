"""Open-set ReID evaluation: fit and report `reid_match_threshold`.

The metric that matters here is not rank-1 accuracy
---------------------------------------------------
Published ReID papers lead with rank-1 and mAP, both closed-set: given that this person
IS in the gallery, is the top hit correct? Our system's failure mode is the question those
metrics do not ask - a stranger walks in, and the nearest enrolled centroid is 0.28 away.
Closed-set rank-1 is blind to it because the stranger is never a query.

So the headline numbers are open-set:

    FAR   false accept rate - impostor probes matched to SOMEONE. Every one of these
          silently attributes a stranger's activity to the resident, corrupting the
          daily features Agent 3 baselines against. This is the expensive error.
    FRR   false reject rate - genuine probes rejected. Costly but recoverable: the
          resident is UNKNOWN for a few frames and the role vote absorbs it.
    DIR   detection & identification rate at rank 1 - genuine probes both accepted AND
          matched to the correct identity. Stricter than 1-FRR: accepting a probe as the
          *wrong* enrolled person counts as a failure here, and it is the failure that
          merges two residents' features.
    AUROC threshold-free separability of the genuine/impostor distance distributions.

`fit_threshold` picks the operating point at a target FAR rather than at best accuracy.
Accuracy is dominated by whichever class is larger and would move the threshold with the
impostor pool size; a FAR budget is a stated risk tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from behaviorsense.agents.perception import PerceptionConfig, cosine_distance
from behaviorsense.data.reid_datasets import Crop, OpenSetSplit

REID_EVAL_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


@dataclass
class OpenSetScores:
    """Nearest-centroid outcomes for every probe in a split.

    `genuine_d` / `impostor_d` are nearest-centroid distances; `genuine_correct` says
    whether that nearest centroid was the probe's own identity. Keeping correctness
    separate from distance is what lets DIR be computed at any threshold without
    recomputing embeddings.
    """

    genuine_d: np.ndarray
    genuine_correct: np.ndarray
    genuine_margin: np.ndarray
    impostor_d: np.ndarray
    n_enrolled: int
    name: str = ""

    def __post_init__(self) -> None:
        if len(self.genuine_d) != len(self.genuine_correct):
            raise ValueError("genuine distance/correctness arrays disagree in length")

    @property
    def n_genuine(self) -> int:
        return int(len(self.genuine_d))

    @property
    def n_impostor(self) -> int:
        return int(len(self.impostor_d))


def l2_normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, np.maximum(n, 1e-12))


def score_split(
    split: OpenSetSplit,
    embeddings: dict[str, np.ndarray],
    *,
    name: str | None = None,
) -> OpenSetScores:
    """Compute nearest-centroid distances for a split, given per-crop embeddings.

    `embeddings` maps crop path -> vector. Decoupling embedding extraction from scoring is
    deliberate: extraction needs a GPU and a checkpoint, scoring does not, so the protocol
    and metrics are testable offline and the expensive step runs once on Kaggle.
    """
    centroids: dict[str, np.ndarray] = {}
    for pid, crops in split.enrolment.items():
        vecs = [embeddings[c.path] for c in crops if c.path in embeddings]
        if not vecs:
            continue
        centroids[pid] = l2_normalise(l2_normalise(np.stack(vecs)).mean(axis=0))
    if not centroids:
        raise ValueError(f"{split.name}: no enrolment embeddings found")

    ids = sorted(centroids)
    matrix = np.stack([centroids[p] for p in ids])

    def nearest(vec: np.ndarray) -> tuple[float, float, str]:
        """(best distance, gap to second best, best id)."""
        d = 1.0 - matrix @ l2_normalise(vec)
        if len(d) == 1:
            return float(d[0]), float("inf"), ids[0]
        order = np.argpartition(d, 1)[:2]
        order = order[np.argsort(d[order])]
        return float(d[order[0]]), float(d[order[1]] - d[order[0]]), ids[order[0]]

    g_d, g_ok, g_margin = [], [], []
    for pid, crops in split.genuine.items():
        for c in crops:
            vec = embeddings.get(c.path)
            if vec is None:
                continue
            best, gap, who = nearest(vec)
            g_d.append(best)
            g_ok.append(who == pid)
            g_margin.append(gap)

    i_d = []
    for c in split.impostors:
        vec = embeddings.get(c.path)
        if vec is None:
            continue
        best, _, _ = nearest(vec)
        i_d.append(best)

    if not g_d or not i_d:
        raise ValueError(
            f"{split.name}: need both genuine ({len(g_d)}) and impostor ({len(i_d)}) "
            "probes with embeddings"
        )

    return OpenSetScores(
        genuine_d=np.asarray(g_d),
        genuine_correct=np.asarray(g_ok, dtype=bool),
        genuine_margin=np.asarray(g_margin),
        impostor_d=np.asarray(i_d),
        n_enrolled=len(centroids),
        name=name or split.name,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatingPoint:
    """Metrics at one threshold. Distances BELOW `threshold` are accepted as a match."""

    threshold: float
    far: float
    frr: float
    dir_rank1: float
    n_genuine: int
    n_impostor: int

    @property
    def tar(self) -> float:
        return 1.0 - self.frr

    def line(self) -> str:
        return (
            f"tau={self.threshold:.3f}  FAR={self.far:6.2%}  TAR={self.tar:6.2%}  "
            f"DIR@1={self.dir_rank1:6.2%}"
        )


def operating_point(scores: OpenSetScores, threshold: float) -> OperatingPoint:
    accept_g = scores.genuine_d < threshold
    accept_i = scores.impostor_d < threshold
    return OperatingPoint(
        threshold=float(threshold),
        far=float(accept_i.mean()),
        frr=float(1.0 - accept_g.mean()),
        dir_rank1=float((accept_g & scores.genuine_correct).mean()),
        n_genuine=scores.n_genuine,
        n_impostor=scores.n_impostor,
    )


def auroc(scores: OpenSetScores) -> float:
    """AUROC via the rank-sum identity, with correct handling of ties.

    Ties matter: an untrained or degenerate embedder returns identical distances for
    everything, and a naive `<` comparison would score that 0.0 or 1.0 instead of the
    correct 0.5. Getting 0.5 for a useless embedder is the sanity check that tells us the
    metric is measuring something.
    """
    g, i = scores.genuine_d, scores.impostor_d
    n_g, n_i = len(g), len(i)
    if n_g == 0 or n_i == 0:
        return float("nan")
    # Genuine should have SMALLER distances, so negate to make "higher = more genuine".
    values = np.concatenate([-g, -i])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    # Average ranks within tied groups.
    sorted_vals = values[order]
    start = 0
    for k in range(1, len(sorted_vals) + 1):
        if k == len(sorted_vals) or sorted_vals[k] != sorted_vals[start]:
            if k - start > 1:
                ranks[order[start:k]] = ranks[order[start:k]].mean()
            start = k
    rank_sum = ranks[:n_g].sum()
    return float((rank_sum - n_g * (n_g + 1) / 2) / (n_g * n_i))


def sweep(scores: OpenSetScores, n_points: int = 512) -> list[OperatingPoint]:
    """Operating points across the observed distance range."""
    all_d = np.concatenate([scores.genuine_d, scores.impostor_d])
    lo, hi = float(all_d.min()), float(all_d.max())
    span = max(hi - lo, 1e-6)
    taus = np.linspace(lo - 0.02 * span, hi + 0.02 * span, n_points)
    return [operating_point(scores, t) for t in taus]


def fit_threshold(scores: OpenSetScores, target_far: float = 0.01) -> OperatingPoint:
    """Largest threshold whose FAR stays within budget - i.e. best TAR at that FAR.

    Chosen over "threshold maximising accuracy" because accuracy depends on the
    genuine:impostor ratio, which is a property of how the split was built, not of the
    deployment. A FAR budget is a stated risk tolerance and transfers.

    If even the tightest threshold exceeds the budget, the strictest point is returned;
    the caller must report that the budget was not achievable rather than pretend it was.
    """
    points = sweep(scores)
    feasible = [p for p in points if p.far <= target_far]
    if not feasible:
        return min(points, key=lambda p: (p.far, p.threshold))
    return max(feasible, key=lambda p: (p.tar, -p.threshold))


def eer(scores: OpenSetScores) -> tuple[float, float]:
    """(equal error rate, threshold) where FAR crosses FRR."""
    points = sweep(scores)
    best = min(points, key=lambda p: abs(p.far - p.frr))
    return (best.far + best.frr) / 2.0, best.threshold


# ---------------------------------------------------------------------------
# Closed-set metrics, for comparability with published numbers
# ---------------------------------------------------------------------------


def closed_set_rank1(scores: OpenSetScores) -> float:
    """Rank-1 ignoring rejection - what ReID papers report. Included for calibration.

    If this is far below published OSNet numbers, the embedding extraction is broken and
    no open-set conclusion drawn from it is trustworthy. It is a smoke test, not a result.
    """
    return float(scores.genuine_correct.mean())


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report(
    scores: OpenSetScores,
    *,
    target_fars: Sequence[float] = (0.001, 0.01, 0.05),
    current_threshold: float | None = None,
) -> str:
    if current_threshold is None:
        current_threshold = PerceptionConfig().reid_match_threshold

    lines: list[str] = []
    a = auroc(scores)
    e_rate, e_tau = eer(scores)
    lines.append(f"# Open-set ReID: {scores.name}")
    lines.append("")
    lines.append(f"- enrolled identities: {scores.n_enrolled}")
    lines.append(f"- genuine probes: {scores.n_genuine}")
    lines.append(f"- impostor probes: {scores.n_impostor}")
    lines.append(f"- AUROC: **{a:.4f}**")
    lines.append(f"- EER: **{e_rate:.2%}** at tau={e_tau:.3f}")
    lines.append(f"- closed-set rank-1 (calibration only): {closed_set_rank1(scores):.2%}")
    lines.append("")

    lines.append("## Distance distributions")
    lines.append("")
    lines.append("| population | n | mean | p05 | median | p95 |")
    lines.append("|---|---|---|---|---|---|")
    for label, d in (("genuine", scores.genuine_d), ("impostor", scores.impostor_d)):
        lines.append(
            f"| {label} | {len(d)} | {d.mean():.3f} | {np.percentile(d, 5):.3f} | "
            f"{np.median(d):.3f} | {np.percentile(d, 95):.3f} |"
        )
    lines.append("")

    lines.append("## Operating points")
    lines.append("")
    lines.append("| target FAR | tau | actual FAR | TAR | DIR@1 |")
    lines.append("|---|---|---|---|---|")
    for far in target_fars:
        p = fit_threshold(scores, far)
        achieved = "" if p.far <= far else " ⚠ not achievable"
        lines.append(
            f"| {far:.1%} | **{p.threshold:.3f}** | {p.far:.2%}{achieved} | "
            f"{p.tar:.2%} | {p.dir_rank1:.2%} |"
        )
    cur = operating_point(scores, current_threshold)
    lines.append(
        f"| _current config_ | {cur.threshold:.3f} | {cur.far:.2%} | {cur.tar:.2%} | "
        f"{cur.dir_rank1:.2%} |"
    )
    lines.append("")

    gap = cur.dir_rank1 - (1.0 - cur.frr)
    if abs(gap) > 1e-9:
        lines.append(
            f"At the current threshold, {(1.0 - cur.frr - cur.dir_rank1):.2%} of genuine "
            "probes are accepted but matched to the WRONG enrolled identity. That is the "
            "error that merges two residents' daily features, so DIR@1 - not TAR - is the "
            "number to hold ourselves to."
        )
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "REID_EVAL_VERSION",
    "OpenSetScores",
    "OperatingPoint",
    "score_split",
    "operating_point",
    "auroc",
    "sweep",
    "fit_threshold",
    "eer",
    "closed_set_rank1",
    "format_report",
    "l2_normalise",
]
