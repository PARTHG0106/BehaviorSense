"""Toyota's released 3D skeletons -> training shards, on INRIA's cross-subject protocol.

Reads the trimmed archive (V1.2 by default), gates on detection coverage, resamples each clip to
a fixed length and writes `.npz` shards a graph network can train on directly. CPU only, minutes.

Four decisions, each one a place where an earlier version of this project went wrong:

**V1.2, not V1.1.** Same clips, same schema, but measured over 600 files each: 3.14% of frames
carry no detection against 5.81%, and the coverage gate drops 28 files against 42. SSTA-PRS's
refinement is a real improvement. V1.1's only advantage is a per-detection `cumscore`, which is
not needed when the gate works on coverage.

**The label comes from `tsm_id`, which refuses an unknown name.** Charades was mapped through 157
keyword rules whose fallback swallowed a large part of the class list and nobody noticed for
months. Here an unrecognised activity is collected and reported, never defaulted.

**The split comes from the SUBJECT id, not from any field in the file.** `protocol_side` derives
it from `CS_TRAIN_SUBJECTS`, which is asserted against two independent official repositories.

**Only the joint stream is stored.** Bone and motion are exact functions of it
(`make_stream(..., parents=dict(BONE_EDGES))`), so storing them would double the bytes and create
a second place for the bone tree to be wrong.

    python scripts/build_toyota_skeleton_shards.py --root /kaggle/input \\
        --out /kaggle/working/shards --archive v12
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behaviorsense.data.toyota import (  # noqa: E402
    ALL_SUBJECTS,
    MIN_POSE_COVERAGE,
    assert_corpus_layout,
    TSM_CLASSES,
    parse_tsm_name,
    protocol_side,
    read_pose3d,
    tsm_id,
)
from behaviorsense.data.toyota_skeleton import (  # noqa: E402
    BONE_EDGES,
    CLIP_FRAMES,
    build_clip,
)

ARCHIVES = {
    "v12": "*_p[0-9][0-9]_r*_c[0-9][0-9]_pose3d.json",
    "v11": "*_p[0-9][0-9]_r*_c[0-9][0-9].json",
}
FLUSH_BYTES = 300_000_000


def find_files(root: Path, archive: str) -> list[Path]:
    """Bounded-depth search. `**` over /kaggle/input cost 78 minutes once."""
    pattern = ARCHIVES[archive]
    slugs = sorted((root / "datasets").glob("*/*")) if (root / "datasets").is_dir() else [root]
    out: list[Path] = []
    for slug in slugs:
        out.extend(itertools.chain(slug.glob(pattern), slug.glob(f"*/{pattern}")))
    if archive == "v11":
        # V1.1's pattern also matches V1.2 names minus the tag, so a V1.2 archive answers it.
        out = [p for p in out if not p.stem.endswith("_pose3d")]
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working/shards")
    ap.add_argument("--archive", choices=sorted(ARCHIVES), default="v12")
    ap.add_argument("--frames", type=int, default=CLIP_FRAMES)
    ap.add_argument("--min-coverage", type=float, default=MIN_POSE_COVERAGE)
    ap.add_argument("--limit", type=int, default=0, help="0 = the whole archive")
    ap.add_argument("--layout-sample", type=int, default=96,
                    help="files sampled for the one corpus-level layout verdict")
    args = ap.parse_args()

    files = find_files(Path(args.root), args.archive)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"no {args.archive} files under {args.root} - is the skeleton dataset attached?")
        return 1
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"tsm_{args.archive}_{args.frames}f"

    joints: list[np.ndarray] = []
    labels: list[int] = []
    subjects: list[int] = []
    sides: list[str] = []
    clips: list[str] = []
    covs: list[float] = []
    bridged: list[float] = []
    shard, n_written = 0, 0
    # Every rejection is COUNTED BY REASON. A run that reports only a total cannot answer "did
    # the gate do this, or did the label mapping?" - and that question decided a re-extraction.
    drops: Counter = Counter()
    unknown_names: Counter = Counter()
    failures: list[tuple[str, str]] = []
    inconclusive: list[str] = []

    def flush() -> None:
        nonlocal shard, n_written
        if not joints:
            return
        path = out_dir / f"{prefix}_{shard:04d}.npz"
        np.savez_compressed(
            path,
            # fp16: these are metres to three decimals at most, and the joint stream is the
            # only array stored, so halving it is free accuracy-wise and halves the dataset.
            joint=np.stack(joints).astype(np.float16),
            labels=np.asarray(labels, dtype=np.int16),
            subjects=np.asarray(subjects, dtype=np.int16),
            split=np.asarray(sides),
            clips=np.asarray(clips),
            coverage=np.asarray(covs, dtype=np.float32),
            bridged_share=np.asarray(bridged, dtype=np.float32),
        )
        print(f"  wrote {path.name}: {len(joints):,} clips, "
              f"{path.stat().st_size / 1e6:.1f} MB")
        n_written += len(joints)
        shard += 1
        for lst in (joints, labels, subjects, sides, clips, covs, bridged):
            lst.clear()

    t0 = time.time()
    print(f"{len(files):,} {args.archive} clips -> {out_dir} "
          f"(T={args.frames}, gate >= {args.min_coverage:.0%})")

    # THE FLAT LAYOUT IS DECIDED ONCE, HERE, before anything is written. It is a property of how
    # the archive was authored, so one corpus-level answer protects every file - and a per-file
    # test inherits every posture the estimator saw. Rejecting per file cost 295 good clips, then
    # 73 after a confidence gate; this raises on an archive-wide transposition and on too little
    # evidence to judge, and never on an individual seated subject.
    layout = assert_corpus_layout(files, sample=args.layout_sample)
    print(f"  layout: {layout['n_agree']}/{layout['n_decisive']} decisive files agree "
          f"({layout['agreement']:.1%}) -> {layout['assumed']}"
          + (f"; dissenters {[d[0] for d in layout['dissenters'][:3]]}"
             if layout["dissenters"] else ""))
    for n, p in enumerate(files, 1):
        try:
            vid = parse_tsm_name(p.name)
        except Exception as exc:                                   # noqa: BLE001
            drops["unparseable_name"] += 1
            failures.append((p.name, f"name: {exc}"))
            continue
        if vid.subject not in ALL_SUBJECTS:
            # Not a silent skip: a subject id outside the official 18 means the archive holds
            # something the protocol does not describe, and the split would be undefined.
            drops["unknown_subject"] += 1
            failures.append((p.name, f"subject p{vid.subject:02d} is not one of the 18"))
            continue
        try:
            label = tsm_id(vid.activity)
        except Exception:                                          # noqa: BLE001
            drops["unknown_activity"] += 1
            unknown_names[vid.activity] += 1
            continue
        try:
            rec = read_pose3d(p)
        except Exception as exc:                                   # noqa: BLE001
            drops["unreadable_pose"] += 1
            failures.append((p.name, f"{type(exc).__name__}: {exc}"))
            continue
        if rec.get("layout_decisive") is False:
            # Kept, but counted. Anatomy could not confirm the layout because the subject is not
            # upright - seated, lying, or legs behind a counter. The constant is trusted there
            # (unanimous on every clip where anatomy DOES speak), and the tally is printed so a
            # sudden jump in it would be visible rather than silent.
            inconclusive.append(p.name)
        clip = build_clip(rec, n_frames=args.frames, min_coverage=args.min_coverage)
        if clip is None:
            drops["below_coverage_gate"] += 1
            continue

        joints.append(clip["joint"])
        labels.append(label)
        subjects.append(vid.subject)
        sides.append(protocol_side(vid, "CS"))
        clips.append(p.name)
        covs.append(clip["coverage"])
        bridged.append(clip["bridged_share"])
        if sum(a.nbytes for a in joints) > FLUSH_BYTES:
            flush()
        if n % 500 == 0:
            print(f"  {n:,}/{len(files):,} seen, {n_written + len(joints):,} kept "
                  f"[{time.time() - t0:.0f}s]")
    flush()
    return _report(out_dir, prefix, files, n_written, drops, unknown_names, failures,
                   inconclusive, layout, args, t0)


def _report(out_dir, prefix, files, n_written, drops, unknown_names, failures,
            inconclusive, layout, args, t0) -> int:
    """Re-read the shards and report what is actually on disk, not what was intended.

    Reading back is the point. A printed running total is a claim about memory; the per-class and
    per-split counts below come from the files a training run will actually open, which is the
    only version of the number worth quoting.
    """
    shards = sorted(out_dir.glob(f"{prefix}_*.npz"))
    per_class: Counter = Counter()
    per_split: Counter = Counter()
    subj_by_split: dict[str, set] = {}
    cov_all, bridged_all = [], []
    total = 0
    for s in shards:
        with np.load(s, allow_pickle=False) as z:
            assert z["joint"].shape[1:] == (3, args.frames, 15), z["joint"].shape
            total += len(z["labels"])
            per_class.update(int(v) for v in z["labels"])
            for side, subj in zip(z["split"], z["subjects"]):
                per_split[str(side)] += 1
                subj_by_split.setdefault(str(side), set()).add(int(subj))
            cov_all.append(z["coverage"])
            bridged_all.append(z["bridged_share"])

    print(f"\n{'=' * 72}\nSHARDS ON DISK\n{'=' * 72}")
    print(f"  {len(shards)} file(s), {total:,} clips in {time.time() - t0:.0f}s")
    print(f"  seen {len(files):,}  kept {total:,}  dropped {sum(drops.values()):,}")
    for reason, n in drops.most_common():
        print(f"    {reason:<22} {n:,}")
    if unknown_names:
        print("  UNRECOGNISED ACTIVITY NAMES (not defaulted, dropped):")
        for name, n in unknown_names.most_common(10):
            print(f"    {name!r}: {n}")
    for name, err in failures[:8]:
        print(f"    FAILED {name}: {err[:110]}")
    if inconclusive:
        print(f"  layout inconclusive but KEPT: {len(inconclusive):,} clips "
              f"({len(inconclusive) / max(1, len(files)):.1%}) - the subject is not upright, so "
              f"anatomy cannot confirm the flat layout and the verified constant is used")
        print(f"    e.g. {', '.join(inconclusive[:3])}")

    cov = np.concatenate(cov_all) if cov_all else np.zeros(0)
    br = np.concatenate(bridged_all) if bridged_all else np.zeros(0)
    if cov.size:
        print(f"  coverage      median {np.median(cov):.3f}  min {cov.min():.3f}")
        print(f"  bridged share median {np.median(br):.3f}  max {br.max():.3f}  "
              f"({int((br > 0.2).sum()):,} clips over 20% interpolated)")

    print(f"\n  split           clips   subjects")
    for side in ("train", "test"):
        subs = sorted(subj_by_split.get(side, ()))
        print(f"    {side:<12} {per_split.get(side, 0):>6}   {len(subs)} {subs}")
    leak = set(subj_by_split.get("train", ())) & set(subj_by_split.get("test", ()))
    assert not leak, f"SUBJECT LEAK across the CS boundary: {sorted(leak)}"
    print("    no subject appears on both sides of the CS split")

    missing = [TSM_CLASSES[i] for i in range(len(TSM_CLASSES)) if per_class.get(i, 0) == 0]
    present = [(TSM_CLASSES[i], n) for i, n in per_class.most_common()]
    print(f"\n  {len(present)}/{len(TSM_CLASSES)} classes have clips; imbalance "
          f"{present[0][1] / max(1, present[-1][1]):.0f}x "
          f"({present[0][0]} {present[0][1]} .. {present[-1][0]} {present[-1][1]})")
    if missing:
        # A class with zero clips cannot be learned and must not be quoted in a per-class table
        # as though it scored zero. Named here so the eval can exclude it explicitly.
        print(f"  CLASSES WITH NO CLIPS ({len(missing)}): {missing}")

    manifest = out_dir / f"{prefix}_manifest.json"
    manifest.write_text(json.dumps({
        "archive": args.archive, "clip_frames": args.frames, "joints": 15,
        "min_coverage": args.min_coverage, "streams_stored": ["joint"],
        "layout": layout,
        "bone_parents": {str(c): p for c, p in BONE_EDGES},
        "n_files_seen": len(files), "n_clips": total,
        "drops": dict(drops), "unknown_activities": dict(unknown_names),
        "n_layout_inconclusive": len(inconclusive),
        "layout_inconclusive_sample": inconclusive[:20],
        "per_split": dict(per_split),
        "subjects_by_split": {k: sorted(v) for k, v in subj_by_split.items()},
        "per_class": {name: per_class.get(i, 0) for i, name in enumerate(TSM_CLASSES)},
        "classes_with_no_clips": missing,
        "coverage_median": float(np.median(cov)) if cov.size else None,
        "bridged_median": float(np.median(br)) if br.size else None,
        "seconds": round(time.time() - t0, 1),
    }, indent=2), encoding="utf-8")
    # Written because a printed table is not a recorded one - this project lost 9,515 s of GPU
    # output to exactly that once.
    print(f"\nwrote {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
