"""Inventory the local MSMT17 copy without decoding every image.

Filenames carry all the metadata we need (pid, camera, frame), so a directory scan is
enough to verify the split structure and build the open-set protocol. Only a handful of
images are opened, to confirm the crop geometry.
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "MSMT17")
SPLITS = ("bounding_box_train", "bounding_box_test", "query")


def scan(split: Path) -> tuple[Counter[str], Counter[str], int]:
    pids: Counter[str] = Counter()
    cams: Counter[str] = Counter()
    total = 0
    with os.scandir(split) as it:
        for entry in it:
            name = entry.name
            if not name.endswith(".jpg") or name.startswith("."):
                continue
            parts = name.split("_")
            if len(parts) < 3:
                continue
            pids[parts[0]] += 1
            cams[parts[1]] += 1
            total += 1
    return pids, cams, total


def main() -> None:
    if not ROOT.exists():
        raise SystemExit(f"not found: {ROOT.resolve()}")

    per_split: dict[str, tuple[Counter[str], Counter[str], int]] = {}
    for s in SPLITS:
        d = ROOT / s
        if not d.exists():
            print(f"MISSING {s}")
            continue
        per_split[s] = scan(d)

    print(f"{'split':<20} {'images':>8} {'ids':>6} {'cams':>5} {'img/id':>7}")
    for s, (pids, cams, total) in per_split.items():
        print(f"{s:<20} {total:>8} {len(pids):>6} {len(cams):>5} {total/max(1,len(pids)):>7.1f}")

    ids = {s: set(v[0]) for s, v in per_split.items()}
    if "bounding_box_train" in ids and "bounding_box_test" in ids:
        tr, te = ids["bounding_box_train"], ids["bounding_box_test"]
        print(f"\ntrain n_ids={len(tr)}  test n_ids={len(te)}  overlap={len(tr & te)}")
    if "query" in ids and "bounding_box_test" in ids:
        q, g = ids["query"], ids["bounding_box_test"]
        print(f"query ids not in gallery: {len(q - g)}  gallery-only ids: {len(g - q)}")

    all_cams: Counter[str] = Counter()
    for _, cams, _ in per_split.values():
        all_cams.update(cams)
    print(f"\ncameras ({len(all_cams)}): "
          + ", ".join(f"{c}:{n}" for c, n in sorted(all_cams.items(), key=lambda kv: kv[0])))

    # Images per identity distribution - decides how many can be held out for enrolment.
    if "bounding_box_test" in per_split:
        counts = sorted(per_split["bounding_box_test"][0].values())
        n = len(counts)
        q = lambda p: counts[min(n - 1, int(p * n))]  # noqa: E731
        print(f"\ntest images/id: min={counts[0]} p10={q(0.10)} median={q(0.50)} "
              f"p90={q(0.90)} max={counts[-1]}")
        cams_per_id: defaultdict[str, set[str]] = defaultdict(set)
        with os.scandir(ROOT / "bounding_box_test") as it:
            for e in it:
                if e.name.endswith(".jpg") and not e.name.startswith("."):
                    p = e.name.split("_")
                    cams_per_id[p[0]].add(p[1])
        multi = sum(1 for v in cams_per_id.values() if len(v) >= 2)
        print(f"test ids seen by >=2 cameras: {multi}/{len(cams_per_id)} "
              f"(cross-camera evaluation is possible for these)")

    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        print("\n(PIL unavailable - skipping crop geometry check)")
        return
    d = ROOT / "bounding_box_train"
    sizes = []
    with os.scandir(d) as it:
        for e in it:
            if e.name.endswith(".jpg") and not e.name.startswith("."):
                with Image.open(e.path) as im:
                    sizes.append(im.size)
            if len(sizes) >= 20:
                break
    if sizes:
        uniq = Counter(sizes)
        print(f"\ncrop sizes (20 sampled): {dict(uniq)}")


if __name__ == "__main__":
    main()
