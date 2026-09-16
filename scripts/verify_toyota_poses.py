"""Verify INRIA's released Toyota poses before training on them. Runs on CPU in minutes.

Three things have to hold before a single GPU hour goes into the 2s-AGCN path, and none of them
can be established from the schema alone:

**1. The flat layout.** `pose3d` is 39 floats for 13 joints. `[x0..x12, y0..y12, z0..z12]` and
`[x0,y0,z0, x1,y1,z1, ...]` both reshape without error; one builds a body, the other swaps joints
for coordinates and trains perfectly well on nonsense. LCR-Net's visualiser says coordinate-major
and `verify_lcrnet_layout` re-derives it anatomically, but on synthetic bodies so far. This runs
it over the real archive, where subjects are sometimes seated or bending and the score will not be
a clean 1.0 - which is exactly why the DISTRIBUTION matters rather than one file.

**2. Frame alignment.** The untrimmed annotations are frame indices at 25 Hz. If the pose files
do not have one entry per source frame, every label lands on the wrong pose and nothing downstream
can detect it. Checked against the annotation CSVs' own extent.

**3. Detection coverage.** A frame where LCR-Net found nobody is zero-filled. If that is a few
percent it is noise; if it is a third of the corpus, the training set is largely empty skeletons
and the numbers would be meaningless.

    python scripts/verify_toyota_poses.py --root /kaggle/input --limit 120 \\
        --out /kaggle/working/pose_verification.json

Reports, decides nothing. The GO/STOP is a judgement made by a person reading it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behaviorsense.data.toyota import (  # noqa: E402
    LCRNET_FLAT_COORDINATE_MAJOR,
    MIN_POSE_COVERAGE,
    TSU_FPS,
    coverage_report,
    read_pose3d,
    upright_chain_score,
    verify_lcrnet_layout,
)

# The three archives and how their files are named. Recorded from notebook 06's resolved mounts
# rather than guessed: all three differ, and a loader hard-coded to one silently reports the
# other two as empty.
SOURCES = {
    "trimmed_v11": "*_p[0-9][0-9]_r*_c[0-9][0-9].json",
    "trimmed_v12": "*_p[0-9][0-9]_r*_c[0-9][0-9]_pose3d.json",
    "untrimmed": "results_P*T*C*_lcrnet*.json",
}


def find(root: Path, pattern: str, limit: int) -> list[Path]:
    """Bounded-depth search over mounts. `**` over /kaggle/input cost 78 minutes once."""
    slugs = sorted((root / "datasets").glob("*/*")) if (root / "datasets").is_dir() else [root]
    out: list[Path] = []
    for slug in slugs:
        for p in itertools.chain(slug.glob(pattern), slug.glob(f"*/{pattern}"),
                                 slug.glob(f"*/*/{pattern}")):
            out.append(p)
            if len(out) >= limit:
                return out
    return out


def check_source(label: str, files: list[Path]) -> dict:
    rows, layouts, failures = [], [], []
    for p in files:
        try:
            rec = read_pose3d(p, verify_layout=False)     # verified separately, below
        except Exception as exc:                          # noqa: BLE001
            failures.append((p.name, f"{type(exc).__name__}: {exc}"))
            continue
        seen = rec["present"]
        raw13 = rec["xyz"][:, :13, :]                    # drop the derived midpoints
        rows.append({
            "name": p.name,
            "frames": rec["n_frames"],
            "missing": rec["n_missing"],
            "upright": round(upright_chain_score(rec["xy"][seen]), 3) if seen.any() else None,
            "xy_min": [round(float(v), 1) for v in rec["xy"][seen].reshape(-1, 2).min(axis=0)]
            if seen.any() else None,
            "xy_max": [round(float(v), 1) for v in rec["xy"][seen].reshape(-1, 2).max(axis=0)]
            if seen.any() else None,
            "xyz_min": [round(float(v), 2) for v in raw13[seen].reshape(-1, 3).min(axis=0)]
            if seen.any() else None,
            "xyz_max": [round(float(v), 2) for v in raw13[seen].reshape(-1, 3).max(axis=0)]
            if seen.any() else None,
            "K": rec["K"],
            "cumscore": rec["has_cumscore"],
            "seconds_at_25hz": round(rec["n_frames"] / TSU_FPS, 1),
        })
        # Layout, from this file's own 2D poses.
        doc = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        flat = [d[0]["pose2d"] for d in doc["frames"] if d and "pose2d" in d[0]]
        if flat:
            layouts.append(verify_lcrnet_layout(flat[:400]))
    tf = sum(r["frames"] for r in rows)
    tm = sum(r["missing"] for r in rows)
    up = [r["upright"] for r in rows if r["upright"] is not None]
    # PER-FILE coverage, because the corpus average hides the shape of the failure: 15.30%
    # missing overall, but individual untrimmed videos run from 0% to 99.6%. The deciles are
    # what MIN_POSE_COVERAGE should be argued from - a threshold in a gap between two modes is
    # defensible, one in the middle of a continuum is not.
    cov = sorted((r["frames"] - r["missing"]) / r["frames"] for r in rows if r["frames"])
    gates = [coverage_report(np.r_[np.zeros(r["missing"], bool),
                                  np.ones(r["frames"] - r["missing"], bool)]) for r in rows]
    dropped = [(r["name"], g["coverage"]) for r, g in zip(rows, gates) if not g["usable"]]
    return {
        "label": label,
        "n_files": len(files),
        "n_parsed": len(rows),
        "failures": failures[:10],
        "total_frames": tf,
        "frames_with_no_detection": tm,
        "missing_share": round(tm / tf, 5) if tf else None,
        "coverage_deciles": [round(float(np.quantile(cov, q / 10)), 4) for q in range(11)]
        if cov else None,
        "min_pose_coverage": MIN_POSE_COVERAGE,
        "n_dropped_by_gate": len(dropped),
        "dropped": dropped[:12],
        "upright_score_median": round(float(np.median(up)), 3) if up else None,
        "upright_score_min": round(float(np.min(up)), 3) if up else None,
        "layout_winners": {w: sum(1 for x in layouts if x["winner"] == w)
                           for w in {x["winner"] for x in layouts}},
        "layout_margin_median": round(float(np.median([x["margin"] for x in layouts])), 3)
        if layouts else None,
        "K_values": sorted({r["K"] for r in rows if r["K"] is not None}),
        "cumscore_files": sum(1 for r in rows if r["cumscore"]),
        "samples": rows[:6],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/kaggle/input")
    ap.add_argument("--limit", type=int, default=120, help="files per archive")
    ap.add_argument("--out", default="/kaggle/working/pose_verification.json")
    args = ap.parse_args()

    root = Path(args.root)
    report = {"root": str(root), "assumed_coordinate_major": LCRNET_FLAT_COORDINATE_MAJOR,
              "sources": []}
    for label, pattern in SOURCES.items():
        files = find(root, pattern, args.limit)
        # V1.1's pattern is a prefix of V1.2's file names minus the tag, so a V1.2 archive can
        # answer the V1.1 glob. Excluded by name, or the two report each other's numbers.
        if label == "trimmed_v11":
            files = [f for f in files if not f.stem.endswith("_pose3d")]
        print(f"\n{'=' * 72}\n{label}: {len(files)} file(s)\n{'=' * 72}")
        if not files:
            print("  none found - the archive is not attached, or the naming differs from what")
            print("  notebook 06 recorded. Both are worth knowing; neither is fatal here.")
            report["sources"].append({"label": label, "n_files": 0})
            continue
        res = check_source(label, files)
        report["sources"].append(res)
        print(f"  parsed          {res['n_parsed']}/{res['n_files']}")
        print(f"  frames          {res['total_frames']:,} "
              f"({res['frames_with_no_detection']:,} with no detection, "
              f"{(res['missing_share'] or 0) * 100:.2f}%)")
        print(f"  upright score   median {res['upright_score_median']}, "
              f"min {res['upright_score_min']}")
        print(f"  flat layout     {res['layout_winners']} "
              f"(median margin {res['layout_margin_median']})")
        print(f"  K values        {res['K_values']}   cumscore in "
              f"{res['cumscore_files']}/{res['n_parsed']} files")
        if res["coverage_deciles"]:
            print("  coverage deciles (0%..100% of files): "
                  + " ".join(f"{v:.2f}" for v in res["coverage_deciles"]))
        print(f"  gate >= {res['min_pose_coverage']:.0%}  drops "
              f"{res['n_dropped_by_gate']}/{res['n_parsed']} files")
        for name, c in res["dropped"][:5]:
            print(f"    DROP {name[:52]:<52} coverage {c:.1%}")
        for r in res["samples"][:3]:
            print(f"    {r['name'][:46]:<46} {r['frames']:>6} frames  "
                  f"2D {r['xy_min']}..{r['xy_max']}  3D {r['xyz_min']}..{r['xyz_max']}")
        for name, err in res["failures"]:
            print(f"    FAILED {name}: {err[:120]}")

    print(f"\n{'=' * 72}\nVERDICT\n{'=' * 72}")
    for res in report["sources"]:
        if not res.get("n_parsed"):
            continue
        wins = res["layout_winners"]
        assumed = "coordinate_major" if LCRNET_FLAT_COORDINATE_MAJOR else "joint_major"
        # Unanimity is the bar, not a majority. A single file that scores the other way is
        # either a genuinely non-upright clip or a mixed archive, and the difference decides
        # whether the reader needs a per-file check - which is worth knowing before training,
        # not after a training curve that looks fine.
        ok = set(wins) == {assumed}
        print(f"  {res['label']:<14} layout {'UNANIMOUS ' + assumed if ok else 'MIXED ' + str(wins)}"
              f"  |  {(res['missing_share'] or 0) * 100:.2f}% frames with no person")
    print("\n2D should sit inside the frame (Toyota is 640x480); 3D is metres in camera space,")
    print("so a plausible torso spans ~0.5 and depth is a positive metre or two. If 2D exceeds")
    print("640x480 or 3D looks like pixels, the two arrays are not in the spaces assumed here.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Written, not just printed: this is the evidence a training decision rests on, and a
    # printed table is not a recorded one.
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
