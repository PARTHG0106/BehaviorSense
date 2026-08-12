"""Stage A: raw video/annotations -> RTMO pose -> unified-taxonomy skeleton shards.

This is the only script that touches video. Everything downstream consumes shards, which
is what makes Agent 2 training reproducible offline and keeps RGB out of the artefacts we
publish (privacy: skeletons cannot be reversed into a recognisable face).

Design decisions worth stating:

  - **Quality gates are printed per dataset, never assumed.** A source that loses >15% of
    frames to failed pose extraction is reported, not silently averaged into the pool. The
    Charades/OmniFall camera qualities differ enough that a pooled mean hides it.
  - **Subject ids are mandatory.** Cross-subject (P1) evaluation is only honest if every
    window knows whose body it came from; a dataset that cannot supply one is rejected
    rather than given a synthetic id, which would silently defeat the split.
  - **The detector is injected**, so this script is testable without a GPU or an RTMO
    checkpoint (`--dry-run` uses a synthetic pose source and exercises every other stage).

Usage (Kaggle GPU session):
    python scripts/prepare_skeletons.py --dataset omnifall --root /kaggle/input/omnifall \\
        --out data/shards/omnifall.npz --checkpoint weights/rtmo-l.pth
    python scripts/prepare_skeletons.py --dry-run --out /tmp/x.npz   # CPU, no weights
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

N_CLASSES = 20
WINDOW_FRAMES = 30      # 2 s at 15 fps
STRIDE_FRAMES = 15      # 1 s
TARGET_FPS = 15.0
MAX_PERSONS = 2
MIN_VISIBLE_KEYPOINTS = 8


@dataclass
class ClipRecord:
    """One annotated clip, before windowing."""

    clip_id: str
    subject: str
    dataset: str
    label: int
    poses: np.ndarray          # [T, M, 17, 3]

    def __post_init__(self) -> None:
        if self.poses.ndim != 4 or self.poses.shape[2:] != (17, 3):
            raise ValueError(f"{self.clip_id}: expected [T,M,17,3], got {self.poses.shape}")


@dataclass
class QualityReport:
    """Per-dataset extraction quality. Printed and saved - never silently discarded."""

    dataset: str
    clips_seen: int = 0
    clips_kept: int = 0
    frames_seen: int = 0
    frames_with_pose: int = 0
    windows_emitted: int = 0
    dropped_reasons: Counter = field(default_factory=Counter)
    label_counts: Counter = field(default_factory=Counter)

    @property
    def pose_rate(self) -> float:
        return self.frames_with_pose / max(1, self.frames_seen)

    def summary(self) -> str:
        flag = "" if self.pose_rate >= 0.85 else "   <-- BELOW 85%, INVESTIGATE"
        drops = ", ".join(f"{k}={v}" for k, v in self.dropped_reasons.most_common(4)) or "none"
        return (
            f"{self.dataset}: {self.clips_kept}/{self.clips_seen} clips, "
            f"{self.windows_emitted} windows, pose rate {self.pose_rate:.1%}{flag}\n"
            f"    dropped: {drops}\n"
            f"    classes present: {len(self.label_counts)}/{N_CLASSES}, "
            f"rarest={min(self.label_counts.values()) if self.label_counts else 0}, "
            f"commonest={max(self.label_counts.values()) if self.label_counts else 0}"
        )


def load_taxonomy(path: Path) -> tuple[dict[str, int], dict[str, dict[str, str]]]:
    """Read configs/taxonomy.yaml -> (name->id, dataset->{source_label: our_name})."""
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    name_to_id = {c["name"]: int(c["id"]) for c in spec["classes"]}
    if len(name_to_id) != int(spec["n_classes"]):
        raise ValueError(f"taxonomy declares {spec['n_classes']} classes, lists {len(name_to_id)}")
    return name_to_id, spec.get("mappings", {})


def map_label(
    source_label: str, dataset: str, mappings: dict, name_to_id: dict[str, int]
) -> int | None:
    """Map a source label to a taxonomy id. None = drop this clip.

    Unmapped labels return None rather than falling back to `other_idle`: silently
    dumping unknown classes into the reject class poisons it with real activities and
    makes the reject class meaningless.
    """
    table = mappings.get(dataset, {})
    target = table.get(source_label)
    if target is None:
        return None
    if isinstance(target, dict):
        if target.get("drop"):
            return None
        target = target.get("to")
    return name_to_id.get(target)


def assign_slots(
    kpts, scores, prev: np.ndarray | None, max_persons: int = MAX_PERSONS,
) -> np.ndarray:
    """RTMO detections for ONE frame -> [max_persons, 17, 3] with temporally stable slots.

    Ranking by box area alone reassigns slots whenever the ranking flips - one person
    bends while the other stands and the two trajectories swap channels mid-window. The
    model then sees person A's pose continue into person B's, and the motion streams see
    a teleport at the swap frame. Detection is per-frame; identity is not, so identity
    has to be reconstructed here, at the only point that still has both frames in hand.

    Greedy nearest-centroid matching to the previous frame's occupied slots; candidates
    with no plausible predecessor (new entrants) fill EMPTY slots, largest first - a
    briefly detected second person must not steal slot 0 from the resident. Area order
    applies only on the first frame or after total detection loss.
    """
    arr = np.zeros((max_persons, 17, 3), np.float32)
    if len(kpts) == 0:
        return arr
    kpts = np.asarray(kpts, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    areas = np.ptp(kpts[..., 0], axis=-1) * np.ptp(kpts[..., 1], axis=-1)
    cand = list(np.argsort(areas)[::-1][:max_persons])

    prev_centroids: dict[int, np.ndarray] = {}
    if prev is not None:
        for slot in range(max_persons):
            vis = prev[slot, :, 2] > 0
            if vis.any():
                prev_centroids[slot] = prev[slot, vis, :2].mean(axis=0)

    def place(slot: int, pi: int) -> None:
        arr[slot, :, :2] = kpts[pi]
        arr[slot, :, 2] = scores[pi]

    if not prev_centroids:
        for slot, pi in enumerate(cand):
            place(slot, pi)
        return arr

    pairs = []
    for pi in cand:
        # Scale-free association gate: a real person cannot move 3 of their own bbox
        # diagonals in one frame step (66 ms at 15 fps would be >10 m/s). Beyond that,
        # the candidate is a new entrant (or a scene cut), not the same person - gluing
        # it onto the resident's trajectory is exactly the splice this function removes.
        # After a cut everything is "far", the frame lands empty, and area order
        # re-seeds on the next frame; one blank frame is cheaper than interpolating
        # across a cut.
        diag = float(np.hypot(np.ptp(kpts[pi, :, 0]), np.ptp(kpts[pi, :, 1])))
        for slot, pc in prev_centroids.items():
            d = float(np.linalg.norm(kpts[pi].mean(axis=0) - pc))
            if d <= 3.0 * max(diag, 1e-6):
                pairs.append((d, pi, slot))
    pairs.sort()
    placed: set[int] = set()
    free = set(range(max_persons))
    for _, pi, slot in pairs:
        if pi in placed or slot not in free:
            continue
        place(slot, pi)
        placed.add(pi)
        free.discard(slot)
    # New entrants -> slots with no recent occupant first. min(free) alone would drop
    # an entrant into the resident's channel the moment the resident blinks out for a
    # frame, splicing exactly like the area bug one level up.
    empty_first = sorted(free, key=lambda s: (s in prev_centroids, s))
    for pi in cand:
        if pi in placed or not empty_first:
            continue
        slot = empty_first.pop(0)
        place(slot, pi)
        placed.add(pi)
    return arr


FALL_DIR_MARKERS = ("fall", "falls", "chute")
ADL_DIR_MARKERS = ("adl", "notfall", "not_fall", "nofall", "normal", "daily")


def subject_from_path(root: Path, video: Path, source: str) -> str:
    """Subject id for one clip, derived from the DIRECTORY LAYOUT, not the filename.

    Fall corpora identify the person by folder, not by file. GMDCSA-24 is
    `Subject 1/Fall/01.mp4` - every subject numbers its clips from 01, so the obvious
    `stem.split("_")[0]` rule maps `Subject 1/ADL/01.mp4` and `Subject 2/ADL/01.mp4` to
    the SAME id. That is not a cosmetic bug: `split_by_subject` holds ids out, so one id
    spanning four people puts each person on both sides of the split, and the
    cross-subject (P1) number for the safety-critical head becomes memorisation reported
    as generalisation. The disjointness assert cannot catch it, because the ids really are
    disjoint - it is the PEOPLE behind them that are not.

    So: take the clip's path relative to the corpus root, drop the components that only
    encode fall/ADL or camera index, and use the first surviving directory. Flat corpora
    (URFD: `fall-01-cam0.mp4`) have no such directory, and fall back to the filename stem
    with the camera suffix stripped, which is that corpus's real sequence id.
    """
    try:
        rel = video.relative_to(root)
    except ValueError:
        rel = Path(video.name)
    parts = list(rel.parts[:-1])
    keep = [p for p in parts
            if p.lower() not in FALL_DIR_MARKERS + ADL_DIR_MARKERS
            and not re.fullmatch(r"cam\w*|camera\s*\d*", p, re.I)]
    if keep:
        return f"{source}_{keep[0].replace(' ', '')}"
    stem = re.sub(r"[-_]?cam\w*$", "", rel.stem, flags=re.I)
    return f"{source}_{stem}"


def is_fall_clip(root: Path, video: Path) -> bool:
    """Whether a clip depicts a fall, by directory marker first, then filename.

    Directory wins because it is how these corpora are actually organised (GMDCSA-24
    separates `Fall/` from `ADL/` while naming every clip `01.mp4`), and because an ADL
    clip sitting under a path that merely mentions falls must not be swept in - checking
    only `parent.name` and the stem missed both cases.
    """
    try:
        rel = video.relative_to(root)
    except ValueError:
        rel = Path(video.name)
    def tokens(name: str) -> set[str]:
        return set(re.split(r"[\s_.\-]+", name.lower())) - {""}

    for part in [q.lower() for q in rel.parts[:-1]]:
        toks = tokens(part)
        # TOKEN match, not exact: CAUCAFall names its activity folders "Fall forward",
        # "Fall backward", "Fall left"... An exact `part in FALL_DIR_MARKERS` test scored
        # all 50 of its fall clips as ADL, which would have fed real falls to the fall
        # head as negatives. ADL is checked first so "no fall" cannot read as "fall".
        if toks & set(ADL_DIR_MARKERS) or ("no" in toks and "fall" in toks):
            return False
        if toks & set(FALL_DIR_MARKERS):
            return True
    stem = tokens(rel.stem)
    if "no" in stem and "fall" in stem:
        return False
    return bool(stem & set(FALL_DIR_MARKERS))


def le2i_fall_frames(video: Path) -> tuple[int, int] | None:
    """Le2i's per-clip fall interval from its Annotation_files sibling, or None.

    Le2i is the one corpus here with no fall/ADL marker in any path component - every
    clip is `video (N).avi` under a room folder, and the label lives in a text file whose
    first two lines are the fall's start and end frame (0/0 meaning "no fall in this
    clip"). Treating a missing annotation as "not a fall" would write ~192 clips that DO
    contain falls into the negatives, so callers must handle None by EXCLUDING the clip.

    The upside: where the annotation exists it is a true frame interval, so Le2i gets
    exact temporal labels rather than the "descent is mid-clip" positional guess the
    other corpora need.
    """
    stem = video.stem
    for base in (video.parent, video.parent.parent):
        for folder in (base / "Annotation_files", base):
            cand = folder / f"{stem}.txt"
            if cand.is_file():
                try:
                    head = [ln.strip() for ln in
                            cand.read_text(encoding="utf-8", errors="ignore").splitlines()
                            if ln.strip()][:2]
                    start, end = int(float(head[0])), int(float(head[1]))
                except (ValueError, IndexError):
                    return None
                if start <= 0 and end <= 0:
                    return (0, 0)          # annotated, and there is no fall
                if end < start:
                    return None
                return (start, end)
    return None


def label_fall_windows(
    starts: list[int], window: int = WINDOW_FRAMES,
    fall_range: tuple[int, int] | None = None,
) -> list[int]:
    """Taxonomy ids for a fall clip's windows. `starts` are SAMPLED-space frame indices.

    Two labelling regimes, because the corpora differ in what they actually tell us:

      * `fall_range` given - Le2i publishes the fall's start/end frame, so windows are
        labelled by real overlap: intersecting the interval is `falling`, entirely after
        it is `fallen_on_ground`, before it is `standing`. `(0, 0)` means the annotation
        exists and says there is no fall, so the clip is ADL.
      * `fall_range` None - everything else only promises "the descent is somewhere in
        the middle". The window nearest mid-clip is ALWAYS `falling`, because a pure band
        test on [0.4, 0.6) leaves some clip lengths with none at all (a 5 s clip gives
        fractions {0, .33, .67, 1}), and short clips are what fall corpora contain.

    Positions come from `starts`, never from enumerate(): window_clip drops
    low-visibility windows, so list index is not temporal position.
    """
    if not starts:
        return []
    if fall_range is not None:
        lo, hi = fall_range
        if lo <= 0 and hi <= 0:
            return [19] * len(starts)          # annotated: no fall in this clip
        out = []
        for st in starts:
            w_lo, w_hi = st, st + window - 1
            if w_hi >= lo and w_lo <= hi:
                out.append(7)                  # overlaps the descent
            elif w_lo > hi:
                out.append(8)                  # after it
            else:
                out.append(1)                  # before it
        return out

    mids = [st + window / 2 for st in starts]
    span = max(mids[-1] - mids[0], 1e-9)
    fracs = [(m - mids[0]) / span for m in mids]
    nearest = min(range(len(fracs)), key=lambda i: abs(fracs[i] - 0.5))
    return [7 if (i == nearest or 0.4 <= f < 0.6) else (8 if f >= 0.6 else 1)
            for i, f in enumerate(fracs)]


def window_clip(
    poses: np.ndarray, window: int = WINDOW_FRAMES, stride: int = STRIDE_FRAMES,
    min_visible: int = MIN_VISIBLE_KEYPOINTS, with_starts: bool = False,
) -> list[np.ndarray] | list[tuple[int, np.ndarray]]:
    """Slice [T,M,17,3] into fixed windows, dropping ones with too little visible pose.

    `with_starts` returns (start_frame, window) pairs instead of bare windows, and exists
    because this function DROPS windows: a caller that infers temporal position from the
    returned list's index is wrong whenever anything was dropped. Notebook 02 labels fall
    clips by position ("the descent is mid-clip"), and with a 3 s occlusion at the head of
    a 10 s clip, index-derived position put `falling` two windows late - labelling the
    actual descent as `standing` and a standing window as `falling`, in the one head where
    that error is safety-critical. Positions must come from the frame index, never from
    enumerate().
    """
    out: list = []
    T = poses.shape[0]
    if T < window:
        return out
    for start in range(0, T - window + 1, stride):
        w = poses[start:start + window]
        # Primary person must be visible in most frames, or the window teaches the model
        # to classify from an absent skeleton.
        visible = (w[:, 0, :, 2] > 0.3).sum(axis=1)
        if (visible >= min_visible).mean() < 0.5:
            continue
        out.append((start, w) if with_starts else w)
    return out


def synthetic_clips(n: int = 40, seed: int = 0) -> list[ClipRecord]:
    """Dry-run source: structured fake poses, so every stage runs without a GPU."""
    rng = np.random.default_rng(seed)
    base = np.zeros((17, 2), dtype=np.float32)
    base[[5, 6]] = [[-0.2, 0.5], [0.2, 0.5]]
    base[[11, 12]] = [[-0.15, -0.5], [0.15, -0.5]]
    base[[15, 16]] = [[-0.15, -1.9], [0.15, -1.9]]
    clips = []
    for i in range(n):
        T = int(rng.integers(45, 120))
        label = i % 6
        poses = np.zeros((T, MAX_PERSONS, 17, 3), dtype=np.float32)
        person = np.tile(base, (T, 1, 1))
        person[:, 9 + label % 4, 0] += np.sin(np.linspace(0, 6, T)) * 0.5
        poses[:, 0, :, :2] = person + rng.normal(0, 0.02, person.shape).astype(np.float32)
        poses[:, 0, :, 2] = 0.9
        clips.append(ClipRecord(clip_id=f"syn{i:03d}", subject=f"s{i % 8:02d}",
                                dataset="dryrun", label=label, poses=poses))
    return clips


def build_shard(clips: list[ClipRecord], report: QualityReport) -> dict[str, np.ndarray]:
    """Window every clip and stack into shard arrays."""
    skels, labels, subjects, datasets = [], [], [], []
    for clip in clips:
        report.clips_seen += 1
        report.frames_seen += clip.poses.shape[0]
        report.frames_with_pose += int(((clip.poses[:, 0, :, 2] > 0.3).sum(axis=1) >= 1).sum())
        windows = window_clip(clip.poses)
        if not windows:
            report.dropped_reasons["too_short_or_occluded"] += 1
            continue
        report.clips_kept += 1
        for w in windows:
            skels.append(w.astype(np.float16))
            labels.append(clip.label)
            subjects.append(clip.subject)
            datasets.append(clip.dataset)
            report.label_counts[clip.label] += 1
    report.windows_emitted = len(skels)
    if not skels:
        raise SystemExit("no windows produced - check the source data and mappings")
    return {
        "skeletons": np.stack(skels),
        "labels": np.asarray(labels, dtype=np.int64),
        "subjects": np.asarray(subjects, dtype="<U32"),
        "datasets": np.asarray(datasets, dtype="<U32"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="dryrun")
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None, help="RTMO weights (omit for --dry-run)")
    ap.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="synthetic poses; exercises windowing/packing without a GPU")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tax_path = Path(args.taxonomy)
    if tax_path.is_file():
        name_to_id, mappings = load_taxonomy(tax_path)
        print(f"taxonomy: {len(name_to_id)} classes, "
              f"mappings for {sorted(mappings)}")
    else:
        print(f"WARNING: {tax_path} not found; label mapping unavailable")

    if args.dry_run:
        clips = synthetic_clips(seed=args.seed)
        report = QualityReport(dataset="dryrun")
    else:
        raise SystemExit(
            "real extraction requires RTMO and dataset-specific readers, which run in the "
            "Kaggle GPU session (see docs/06 section 2). Use --dry-run to validate the "
            "pipeline locally; the reader for each dataset is added as its data lands."
        )

    shard = build_shard(clips, report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **shard)

    print()
    print(report.summary())
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB), "
          f"{shard['skeletons'].shape} float16")
    meta = out.with_suffix(".quality.json")
    meta.write_text(json.dumps({
        "dataset": report.dataset, "clips_seen": report.clips_seen,
        "clips_kept": report.clips_kept, "windows": report.windows_emitted,
        "pose_rate": report.pose_rate,
        "label_counts": {str(k): v for k, v in report.label_counts.items()},
        "dropped": dict(report.dropped_reasons),
    }, indent=2), encoding="utf-8")
    print(f"wrote {meta}")


if __name__ == "__main__":
    main()
