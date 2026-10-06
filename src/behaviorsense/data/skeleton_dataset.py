"""Skeleton shard dataset, augmentation and sampling for Agent 2 training.

Shard format (produced by Stage A, `scripts/prepare_skeletons.py`):

    <name>.npz
      skeletons : float16 [N, T, M, 17, 3]   x, y, score
      labels    : int64   [N]                unified taxonomy id
      subjects  : <U32    [N]                subject id, for cross-subject splits
      datasets  : <U32    [N]                source dataset, for cross-dataset splits

Two invariants are enforced by assertion rather than assumed, because both failures are
silent and both invalidate every number downstream:

  1. **Subject disjointness.** A cross-subject protocol whose train and val share a
     subject reports memorisation as generalisation. `split_by_subject` asserts the
     partition is disjoint, exactly as `OpenSetSplit.validate()` does for ReID.
  2. **Label range.** An out-of-range label silently becomes a wrong class under
     `nn.CrossEntropyLoss` indexing rules, or crashes 40 minutes into a session.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from behaviorsense.models.stgcnpp import FLIP_INDEX, N_JOINTS

N_CLASSES = 20


@dataclass(frozen=True)
class AugmentConfig:
    """Skeleton augmentation. Every entry is justified by a real corruption we expect.

    Deliberately excluded: mixup and CutMix. Both blend labels, and for a taxonomy where
    one class (`falling`) is the safety-critical event, training the model to predict
    "30% falling" as a target degrades exactly the decision we care most about.
    """

    enabled: bool = True
    rotate_deg: float = 15.0
    """Camera roll varies between installations; ±15° covers realistic mounting."""
    scale_range: tuple[float, float] = (0.9, 1.1)
    """Distance from camera. Larger ranges start emulating a different camera geometry."""
    translate: float = 0.05
    """Fraction of frame. Position within the room carries no action information."""
    joint_noise: float = 0.01
    """RTMO keypoint jitter. Matches observed CCTV-scale localisation error."""
    joint_dropout: float = 0.05
    """Probability of zeroing a joint for a whole window - emulates persistent occlusion
    by furniture, which is the dominant real failure and is NOT i.i.d. per frame."""
    flip_prob: float = 0.5
    """Horizontal flip WITH left/right joint remap. Without the remap this teaches the
    model that left and right limbs are interchangeable."""
    temporal_scale: tuple[float, float] = (0.8, 1.2)
    """Activity speed varies with frailty - the very signal we monitor - so the model
    must not key on tempo."""


def augment(
    x: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator
) -> np.ndarray:
    """Augment one window [T, M, 17, 3] (x, y, score). Scores are never geometrically
    transformed - they are confidences, not coordinates."""
    if not cfg.enabled:
        return x
    x = x.copy()
    coords = x[..., :2]

    if rng.random() < cfg.flip_prob:
        coords[..., 0] = -coords[..., 0]
        x[:] = x[:, :, list(FLIP_INDEX), :]
        coords = x[..., :2]

    theta = np.deg2rad(rng.uniform(-cfg.rotate_deg, cfg.rotate_deg))
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    coords[:] = coords @ rot.T

    coords *= rng.uniform(*cfg.scale_range)
    coords += rng.uniform(-cfg.translate, cfg.translate, size=(1, 1, 1, 2)).astype(np.float32)
    coords += rng.normal(0, cfg.joint_noise, size=coords.shape).astype(np.float32)

    if cfg.joint_dropout > 0:
        drop = rng.random((x.shape[1], N_JOINTS)) < cfg.joint_dropout
        for m in range(x.shape[1]):
            x[:, m, drop[m], :] = 0.0

    # Restore normalise()'s invariant: joints RTMO did not see stay at EXACT zero.
    # Translation and noise above shift every joint, invisible ones included, which gave
    # train-time "missing" a small random offset that val-time "missing" (augmentation
    # off) never has - the model then learns an encoding of absence that evaluation and
    # serving never produce. Scores are untouched by the transforms, so score <= 0 still
    # identifies exactly the joints normalise() zeroed (plus this function's dropout,
    # which zeroes scores too - making occlusion-by-furniture and RTMO-missed look
    # identical, as they should).
    x[x[..., 2] <= 0] = 0.0
    return x


def temporal_resample(x: np.ndarray, out_T: int, rng: np.random.Generator | None,
                      cfg: AugmentConfig | None = None) -> np.ndarray:
    """Resample a window to `out_T` frames, optionally with random speed jitter."""
    T = x.shape[0]
    if rng is not None and cfg is not None and cfg.enabled:
        speed = rng.uniform(*cfg.temporal_scale)
        src_len = min(T, max(2, int(round(out_T * speed))))
        start = rng.integers(0, T - src_len + 1) if T > src_len else 0
        idx = np.linspace(start, start + src_len - 1, out_T)
    else:
        idx = np.linspace(0, T - 1, out_T)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (idx - lo).astype(np.float32)[:, None, None, None]
    return (1 - w) * x[lo] + w * x[hi]


def normalise(x: np.ndarray, keep_root_motion: bool = True) -> np.ndarray:
    """Root-centre and scale by torso length, per person.

    Removes absolute position and apparent size (camera distance) while preserving
    posture. Rotation is deliberately NOT normalised: a fallen person's orientation
    relative to gravity is the signal, and normalising it away would make falls and
    lying down indistinguishable.

    `keep_root_motion` is the subtle and safety-critical part. Centring on the mid-hip
    of EVERY frame independently subtracts the body's trajectory, which mathematically
    erases whole-body vertical descent - i.e. exactly what a fall is. The fall head's
    smoke test caught this as AUROC 0.466 (below chance) on data whose only signal was a
    1.5-unit hip drop: after per-frame centring the two classes were byte-identical.

    So the root is centred on the window's FIRST frame and each frame's root offset from
    it is retained. Position within the room still carries no information (the first
    frame is the origin), but motion through the room does.

    "First frame" and "torso length" both mean *of the frames where the person is
    actually detected*. Detection dropouts write all-zero frames, and including those
    in the torso mean deflated the scale (a person absent for a third of the window got
    coordinates inflated by ~1.5x), while an absent frame 0 made the origin (0, 0) - no
    centring at all, leaving that window in raw pixel coordinates while every other
    window sat in torso units.
    """
    x = x.copy()
    L_HIP, R_HIP, L_SH, R_SH = 11, 12, 5, 6
    for m in range(x.shape[1]):
        person = x[:, m]
        vis_frame = (person[..., 2] > 0).any(axis=-1)
        if not vis_frame.any():
            continue
        mid_hip = (person[:, L_HIP, :2] + person[:, R_HIP, :2]) / 2.0
        mid_sh = (person[:, L_SH, :2] + person[:, R_SH, :2]) / 2.0
        torso = np.linalg.norm(mid_sh - mid_hip, axis=-1)[vis_frame].mean()
        scale = torso if torso > 1e-3 else 1.0
        first = int(np.flatnonzero(vis_frame)[0])
        origin = mid_hip[first] if keep_root_motion else mid_hip
        valid = person[..., 2] > 0
        person[..., :2] = (person[..., :2] - (origin if keep_root_motion
                                              else origin[:, None, :])) / scale
        person[~valid] = 0.0   # keep invisible joints at exact zero after centring
        x[:, m] = person
    return x


class SkeletonWindowDataset(Dataset):
    """Windows from one or more shards, with optional augmentation."""

    def __init__(
        self,
        shards: Sequence[str | Path],
        n_frames: int = 30,
        augment_cfg: AugmentConfig | None = None,
        indices: np.ndarray | None = None,
        seed: int = 0,
        n_classes: int = N_CLASSES,
    ) -> None:
        skels, labels, subjects, datasets = [], [], [], []
        for shard in shards:
            z = np.load(shard, allow_pickle=False)
            skels.append(z["skeletons"])
            labels.append(z["labels"])
            subjects.append(z["subjects"])
            datasets.append(z["datasets"])
        self.skeletons = np.concatenate(skels)
        self.labels = np.concatenate(labels).astype(np.int64)
        self.subjects = np.concatenate(subjects)
        self.datasets = np.concatenate(datasets)

        # The head size belongs to the SHARDS. Charades carries the 20-class
        # `activity.CLASS_NAMES`; the Toyota RTMO shards carry the 22-class `COARSE_V11`, whose
        # ids 20 (`using_device`) and 21 (`object_interaction`) are legitimate there and out of
        # range here. Keeping the check but parameterising the bound is what lets one loader
        # serve both without either silently accepting a label the head cannot emit.
        bad = (self.labels < 0) | (self.labels >= n_classes)
        self.n_classes = n_classes
        if bad.any():
            raise ValueError(
                f"{bad.sum()} labels outside [0,{n_classes}): "
                f"{sorted(set(self.labels[bad].tolist()))[:5]}"
            )

        self.indices = np.arange(len(self.labels)) if indices is None else np.asarray(indices)
        self.n_frames = n_frames
        self.cfg = augment_cfg or AugmentConfig(enabled=False)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def class_counts(self) -> np.ndarray:
        return np.bincount(self.labels[self.indices], minlength=N_CLASSES)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        idx = int(self.indices[i])
        # Per-item RNG keyed by (seed, idx): augmentation is reproducible and independent
        # of DataLoader worker count, which otherwise silently changes the augmentation
        # stream and makes runs incomparable.
        rng = np.random.default_rng((self.seed, idx))
        x = self.skeletons[idx].astype(np.float32)
        x = temporal_resample(x, self.n_frames, rng, self.cfg)
        x = normalise(x)
        x = augment(x, self.cfg, rng)
        # [T, M, V, C] -> [C, T, V, M]
        x = np.transpose(x, (3, 0, 2, 1))
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.labels[idx])


def load_subject_map(csv_path: str | Path, id_col: str = "id",
                     subject_col: str = "subject") -> dict[str, str]:
    """`video id -> actor id` from an annotation CSV (Charades: 267 actors, 7,986 videos).

    Exists because the ADL shards store the VIDEO id in their `subjects` array, under the
    belief - recorded in docs/07 and wrong - that "Charades subject ids do not exist
    publicly". Charades_v1_train.csv has carried a `subject` column all along. With ~30
    videos per actor, a video-id split puts essentially every actor on both sides, so the
    "subject-disjoint" P1 was actually video-disjoint: same person, same home, same
    mannerisms in train and val. This is the exact failure `subject_from_path` was written
    to prevent in the fall corpora ("the ids are disjoint - the PEOPLE behind them are
    not"), committed on the biggest corpus.

    The map is applied at SPLIT time (train_adl --subject-map, notebook 04), never to the
    stored arrays: per-video ids are still what temporal sequence reconstruction needs, so
    the shards keep them and the two uses stop sharing one array.
    """
    import csv

    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if id_col not in (reader.fieldnames or []) or subject_col not in (reader.fieldnames or []):
            raise ValueError(
                f"{csv_path} lacks columns {id_col!r}/{subject_col!r}; "
                f"has {reader.fieldnames}"
            )
        out = {row[id_col]: row[subject_col] for row in reader if row.get(id_col)}
    if not out:
        raise ValueError(f"{csv_path} produced an empty subject map")
    return out


def remap_subjects(subjects: np.ndarray, mapping: dict[str, str]) -> tuple[np.ndarray, float]:
    """Apply a video->actor map; ids absent from the map pass through unchanged.

    Returns (remapped, coverage). Pass-through keeps mixed corpora working - fall-corpus
    ids are already people - and coverage is returned rather than asserted because the
    caller knows whether 0.0 means "no Charades here" (fine) or "wrong CSV" (not fine).
    """
    remapped = np.array([mapping.get(s, s) for s in subjects], dtype="<U32")
    coverage = float(np.mean([s in mapping for s in subjects])) if len(subjects) else 0.0
    return remapped, coverage


def split_by_subject(
    subjects: np.ndarray, val_frac: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-subject (P1) split, with disjointness asserted rather than trusted."""
    uniq = np.unique(subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_frac))
    val_subj = set(uniq[:n_val].tolist())

    is_val = np.array([s in val_subj for s in subjects])
    train_idx = np.flatnonzero(~is_val)
    val_idx = np.flatnonzero(is_val)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(f"degenerate split: {len(train_idx)} train / {len(val_idx)} val")
    overlap = set(subjects[train_idx].tolist()) & set(subjects[val_idx].tolist())
    if overlap:
        raise AssertionError(f"subject leakage: {sorted(overlap)[:5]}")
    return train_idx, val_idx


def class_balanced_sampler(
    labels: np.ndarray, beta: float = 0.9999, n_samples: int | None = None
) -> WeightedRandomSampler:
    """Effective-number class weighting (Cui et al., CVPR 2019).

    w_c = (1 - beta) / (1 - beta^n_c). Chosen over inverse frequency (1/n_c) because at
    our imbalance (~100:1 after Charades mapping) inverse frequency oversamples the rarest
    class so aggressively that the model sees the same few hundred windows thousands of
    times and memorises them. Effective number saturates, which is the intent.
    """
    counts = np.bincount(labels, minlength=N_CLASSES).astype(np.float64)
    eff = np.where(counts > 0, (1.0 - np.power(beta, counts)) / (1.0 - beta), 1.0)
    w_class = np.where(counts > 0, 1.0 / eff, 0.0)
    w_class = w_class / w_class.sum() * (counts > 0).sum()
    weights = w_class[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=n_samples or len(labels),
        replacement=True,
    )


__all__ = [
    "AugmentConfig",
    "SkeletonWindowDataset",
    "augment",
    "load_subject_map",
    "normalise",
    "remap_subjects",
    "temporal_resample",
    "split_by_subject",
    "class_balanced_sampler",
    "N_CLASSES",
]
