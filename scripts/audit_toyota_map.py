"""Audit the Toyota Smarthome -> taxonomy map, and emit it as a reviewable artefact.

Deliberately NOT a generator, and that is a change of plan worth stating. The Charades
equivalent (`scripts/build_charades_map.py`) exists because 157 class names had to be mapped
and hand-typing them invites transcription errors, so it applies ordered keyword rules and
prints what fired. That machinery cost us: its fallback swallowed a large part of the class
list into `other_idle`, `walking` came out of the real extraction at F1 0.093 on 1.5% of
windows, and nothing in the output pointed at why until a themed audit was bolted on.

Toyota has 51 + 31 names, all of them mapped EXPLICITLY in `data/toyota.py`, and
`collapse_matrix()` raises on any class that is not. Adding a keyword engine here would
reintroduce exactly the failure mode that produced the numbers we are trying to escape. So
this script audits and reports; it never assigns a label.

    python scripts/audit_toyota_map.py                          # selftest, no data needed
    python scripts/audit_toyota_map.py --annotations smarthome_CS_51.json

With `--annotations` it writes `configs/toyota_map.yaml` and
`configs/toyota_map_review.tsv` (one row per fine class, with its measured frame count and
share) and prints the coarse distribution, the unannotated-gap fraction, and the protocol
video counts. Review the TSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behaviorsense.data.toyota import (  # noqa: E402
    COARSE_V11,
    CS_TEST_SUBJECTS,
    CS_TRAIN_SUBJECTS,
    PROTOCOLS,
    TSM_CLASSES,
    TSM_TO_COARSE,
    TSU_CLASSES,
    TSU_FPS,
    TSU_TO_COARSE,
    assert_coarse_parity,
    collapse_matrix,
    frame_counts,
    iter_videos,
    load_tsu_annotations,
    slug,
    supervised_coarse,
)

# Known-correct targets, chosen to pin the decisions a reader is most likely to disagree
# with rather than the obvious ones. Every entry here is a judgement call recorded so that
# changing it is a visible edit and not a drift.
SELFTEST: tuple[tuple[str, str, str], ...] = (
    ("Take_pills", "taking_medication", "the class Charades could only reach by keyword"),
    ("Use_laptop", "using_device", "NOT other_idle - 6.7% of annotated frames"),
    ("Use_tablet", "using_device", "NOT other_idle - 5.2% of annotated frames"),
    ("Use_Drawer", "object_interaction", "NOT other_idle - purposeful, iadl_index"),
    ("Use_fridge", "object_interaction", "storage access, not food preparation itself"),
    ("Write", "reading", "same behavioural signal: cognitive_engagement"),
    ("Enter", "walking", "doorway traversal; walking already owns room_transition"),
    ("Leave", "walking", "as Enter"),
    ("Sit_down", "sitting_down", "TRANSITION, not the sustained posture"),
    ("Get_up", "standing_up", "transition"),
    ("Drink.From_glass", "drinking", "vessel is an appearance distinction, not a pose one"),
    ("Pour.From_bottle", "cooking_food_prep", "pouring is preparation, drinking is intake"),
    ("Stir_coffee/tea", "cooking_food_prep", "slash in the official name; slug() handles it"),
    ("Wipe_table", "cleaning_housework", "housework"),
    ("Use_glasses", "personal_hygiene", "dressing, read coarsely as the taxonomy says"),
    ("Breakfast.Eat_at_table", "eating", "intake"),
    ("Breakfast.Cut_bread", "cooking_food_prep", "preparation, despite the Breakfast prefix"),
)


def selftest() -> int:
    """Check parity, total coverage, and the judgement calls. No dataset required."""
    rc = 0
    assert_coarse_parity()
    print(f"coarse parity ok: {len(COARSE_V11)} classes, 0..19 match activity.CLASS_NAMES")

    for label, fine, mapping in (("untrimmed", TSU_CLASSES, TSU_TO_COARSE),
                                 ("trimmed", TSM_CLASSES, TSM_TO_COARSE)):
        try:
            M = collapse_matrix(fine, mapping)
        except KeyError as exc:
            print(f"  [FAIL] {label}: {exc}")
            rc = 1
            continue
        rows = M.sum(axis=1)
        if not np.all(rows == 1):
            print(f"  [FAIL] {label}: {int((rows != 1).sum())} classes map to != 1 target")
            rc = 1
        print(f"  [ok] {label}: {len(fine)} fine classes -> "
              f"{int((M.sum(axis=0) > 0).sum())} coarse classes, all mapped exactly once")

    for name, expected, why in SELFTEST:
        got = TSU_TO_COARSE.get(name)
        ok = got == expected
        rc |= 0 if ok else 1
        print(f"  [{'ok' if ok else 'WRONG'}] {name!r} -> {got} ({why})")

    sup = supervised_coarse(TSU_TO_COARSE)
    absent = [COARSE_V11[i] for i in range(len(COARSE_V11)) if i not in sup]
    print(f"\nToyota supervises {len(sup)}/{len(COARSE_V11)} coarse classes.")
    print(f"It CANNOT supervise: {', '.join(absent)}")
    print("Those must come from another corpus, and until they do the cross-corpus loss")
    print("must abstain on them - scoring them as negative teaches the model they are")
    print("absent whenever Toyota is the source. frame_mask() is the per-frame version.")
    return rc


def report(ann: Path, out_yaml: Path, out_tsv: Path) -> int:
    """Measure the map against the real annotations and write the review artefacts."""
    videos = load_tsu_annotations(ann)
    M = collapse_matrix(TSU_CLASSES, TSU_TO_COARSE)
    fine = frame_counts(videos.values())
    coarse = fine @ M
    tot = float(fine.sum())

    print(f"\n{len(videos)} videos, {tot:,.0f} annotated frames "
          f"= {tot / TSU_FPS / 3600:.1f} h at {TSU_FPS:g} Hz")
    print(f"official CS partition verified: {len(CS_TRAIN_SUBJECTS)} train subjects / "
          f"{len(CS_TEST_SUBJECTS)} test")
    for p in PROTOCOLS:
        counts = {s: len(list(iter_videos(videos, p, s)))
                  for s in ("train", "val", "test", "unused")}
        print(f"  {p:<4} " + "  ".join(f"{k} {v}" for k, v in counts.items() if v))

    gaps = np.array([1.0 - v.annotated_frames() / max(1, v.duration)
                     for v in videos.values()])
    print(f"\nunannotated fraction: mean {gaps.mean():.3f}  median "
          f"{np.median(gaps):.3f}  max {gaps.max():.3f}")
    print("A third of the corpus is gap. The 51-class benchmark head treats gaps as true")
    print("negatives, which is correct there; the unified head must MASK them, because it")
    print("also carries sitting and standing - the two things a gap most likely contains.")

    print(f"\n{'coarse class':<24}{'frames':>12}{'share%':>8}")
    for i in np.argsort(-coarse):
        if coarse[i]:
            print(f"{COARSE_V11[i]:<24}{int(coarse[i]):>12,}{100 * coarse[i] / tot:>7.2f}")
    nz = coarse[coarse > 0]
    print(f"{'-> imbalance':<24}{'':>12}{nz.max() / nz.min():>6.0f}x   "
          f"(fine: {fine[fine > 0].max() / fine[fine > 0].min():.0f}x)")
    idle = int(coarse[COARSE_V11.index("other_idle")])
    print(f"\nother_idle receives {idle} frames from Toyota. Zero is the intended answer: "
          "every\nclass here is a real activity, so nothing needs the reject class - which "
          "is the\nopposite of the Charades map, where the fallback WAS the largest class.")

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Toyota Smarthome -> BehaviorSense taxonomy v1.1.",
        "# GENERATED by scripts/audit_toyota_map.py from data/toyota.py's explicit tables.",
        f"# Measured against {ann.name}: {len(videos)} videos, {tot:,.0f} annotated frames.",
        "# Do not edit by hand - edit TSU_TO_COARSE / TSM_TO_COARSE and re-run.",
        "untrimmed:",
    ]
    for name in TSU_CLASSES:
        lines.append(f"  {slug(name)}: {TSU_TO_COARSE[name]}   # {name}")
    lines.append("trimmed:")
    for name in TSM_CLASSES:
        lines.append(f"  {slug(name)}: {TSM_TO_COARSE[name]}   # {name}")
    out_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = ["fine_id\tfine_name\tcoarse\tframes\tshare_pct\tinstances"]
    inst = {i: sum(1 for v in videos.values() for a in v.actions if a[0] == i)
            for i in range(len(TSU_CLASSES))}
    for i, name in enumerate(TSU_CLASSES):
        rows.append(f"{i}\t{name}\t{TSU_TO_COARSE[name]}\t{int(fine[i])}\t"
                    f"{100 * fine[i] / tot:.3f}\t{inst[i]}")
    out_tsv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\nwrote {out_yaml} and {out_tsv}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", default=None,
                    help="smarthome_CS_51.json from dairui01/Toyota_Smarthome/pipline/data")
    ap.add_argument("--out", default="configs/toyota_map.yaml")
    ap.add_argument("--review-tsv", default="configs/toyota_map_review.tsv")
    args = ap.parse_args()

    rc = selftest()
    if rc:
        print("\nselftest FAILED - fix the map before measuring anything with it.")
        return rc
    if args.annotations:
        rc |= report(Path(args.annotations), Path(args.out), Path(args.review_tsv))
    else:
        print("\n(no --annotations: selftest only. Pass smarthome_CS_51.json for the "
              "measured distribution and the review TSV.)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
