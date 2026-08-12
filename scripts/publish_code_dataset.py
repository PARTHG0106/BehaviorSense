"""Publish `behaviorsense-code` from a CLEAN staging copy, not from the working tree.

Why this exists
---------------
`kaggle datasets version -p .` uploads the working directory as it stands. `.kaggleignore`
sits next to it listing what to leave out - and the Kaggle CLI does not read that file.
It never has; it was documentation wearing the costume of a config.

The consequence was not a large upload, it was a corrupted training run. Two artefacts
rode along and neither announced itself:

  * `data/shards/_smoke_fall.npz` - 400 synthetic windows from `train_fall.py --smoke`.
    The filename contains "fall", so notebook 03's shard resolver matched it into the
    REAL fall corpus and the fall head was partly fitted on a fixture.
  * `runs/adl/last.pt` - a 25-epoch CPU smoke checkpoint (`smoke=True`). Notebook 03's
    carry-forward copied it in and resumed from it, putting an 80-epoch run on a
    25-epoch cosine schedule. Invisible in the loss curve.

Both notebooks now defend themselves (tests N10b/N10c), but a guard downstream of a bad
upload is the second line of defence, not the first. This script is the first: it copies
only what the notebooks actually read into a temp directory and publishes that.

Usage:
    python scripts/publish_code_dataset.py --message "why this version exists"
    python scripts/publish_code_dataset.py --dry-run      # list, do not upload
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# What the notebooks read at runtime, and nothing else. Derived from the repo contract in
# notebooks/_generate.py plus the imports each notebook makes - see docs/07_kaggle_plan.md
# "When to re-upload behaviorsense-code".
INCLUDE = ("src", "scripts", "configs", "weights")

# Belt and braces inside the included trees. `weights/` legitimately holds .pth files, so
# a blanket extension rule would drop the OSNet checkpoints the preflight requires.
EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", ".pytest_cache", "runs", "shards"}
EXCLUDE_SUFFIX = {".pyc", ".pyo", ".npz", ".db", ".log"}


def is_fixture(path: Path) -> bool:
    """Synthetic test DATA, by this repo's naming convention.

    Deliberately scoped to data files: an earlier version matched on the name alone and
    excluded `scripts/kaggle_smoke_test.py` - a real script the preflight depends on and
    a repo-contract token. The staging check caught it, which is the point of running the
    contract against the staged copy rather than the working tree.
    """
    if path.suffix.lower() not in {".npz", ".npy", ".pt", ".pth", ".db"}:
        return False
    return path.name.startswith("_") or "smoke" in path.name.lower()


def stage(dest: Path) -> list[tuple[str, int]]:
    """Copy the publishable subset into `dest`; return (relpath, bytes) for each file."""
    manifest: list[tuple[str, int]] = []
    for top in INCLUDE:
        src_root = ROOT / top
        if not src_root.is_dir():
            continue
        for src in sorted(src_root.rglob("*")):
            if not src.is_file():
                continue
            rel = src.relative_to(ROOT)
            if set(rel.parts) & EXCLUDE_DIRS:
                continue
            if src.suffix.lower() in EXCLUDE_SUFFIX or is_fixture(src):
                continue
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            manifest.append((rel.as_posix(), src.stat().st_size))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--message", "-m", default="sync", help="version message")
    ap.add_argument("--dry-run", action="store_true", help="stage and list, do not upload")
    args = ap.parse_args()

    meta_src = ROOT / "dataset-metadata.json"
    if not meta_src.is_file():
        print("dataset-metadata.json missing - create the dataset first "
              "(see docs/07_kaggle_plan.md Step 0)")
        return 1

    with tempfile.TemporaryDirectory() as td:
        dest = Path(td)
        manifest = stage(dest)
        shutil.copy2(meta_src, dest / "dataset-metadata.json")

        total = sum(n for _, n in manifest)
        print(f"staging {len(manifest)} files, {total / 1e6:.1f} MB")
        for top in INCLUDE:
            n = sum(1 for r, _ in manifest if r.startswith(f"{top}/"))
            b = sum(sz for r, sz in manifest if r.startswith(f"{top}/"))
            print(f"  {top:<10} {n:>4} files  {b / 1e6:>8.1f} MB")

        # The contract tokens must survive staging, or the upload fails every notebook's
        # resolver cell - a check that costs nothing and catches an INCLUDE typo.
        gen = (ROOT / "notebooks" / "_generate.py").read_text(encoding="utf-8")
        body = gen.split("CONTRACT = [", 1)[1].split("]", 1)[0]
        import re
        missing = []
        for rel, token in re.findall(r'\("([\w/.]+)",\s*(?:"([^"]*)"|None)', body):
            target = dest / rel
            if not target.is_file():
                missing.append(f"{rel} (absent)")
            elif token and token not in target.read_text(encoding="utf-8", errors="ignore"):
                missing.append(f"{rel} lacks {token!r}")
        if missing:
            print("\nSTAGING IS INCOMPLETE - the notebooks would reject this upload:")
            for m in missing:
                print(f"  {m}")
            return 1
        print("  contract: all tokens present in the staged copy")

        # And prove the leak that motivated this cannot recur.
        leaked = [r for r, _ in manifest
                  if r.endswith(".npz") or "/runs/" in r or is_fixture(Path(r))]
        if leaked:
            print(f"\nREFUSING: {len(leaked)} artefact(s) still staged: {leaked[:5]}")
            return 1
        print("  no shards, run directories, or smoke fixtures staged")

        if args.dry_run:
            print("\n--dry-run: nothing uploaded")
            return 0

        cmd = ["kaggle", "datasets", "version", "-p", str(dest),
               "--dir-mode", "zip", "-m", args.message]
        print(f"\n$ {' '.join(cmd[:6])} -m {args.message!r}")
        return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
