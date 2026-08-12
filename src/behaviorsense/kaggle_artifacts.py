"""Which files under /kaggle/input are REAL data, and which are development leftovers.

Why this module exists
----------------------
`behaviorsense-code` is an upload of the working tree, so it carries `runs/` and
`data/shards/` - CPU smoke checkpoints and synthetic fixtures sitting beside the real
thing. Four separate notebook call sites had to learn this independently:

  * `shard_paths()`   matched `_smoke_fall.npz` into the real fall corpus (notebook 02/03)
  * carry-forward     resumed an 80-epoch run from a 25-epoch `smoke=True` checkpoint
  * the summary cell  reported that smoke run as if it were a result
  * `run_dir()`       resolved to `EmotionSense-Extended/runs`, which holds `adl/` rather
                      than `adl_joint/`, and notebook 04 died on "no checkpoints matching
                      adl_<stream>/best.pt"

Each was fixed on its own, which is the actual defect: one rule, four copies, and the
fourth was missed. The rule lives here now, is tested once, and the notebooks import it.

`scripts/publish_code_dataset.py` stops these artefacts being uploaded at all. This is the
second line of defence, for datasets published before that script existed - including the
ones already on Kaggle.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

# A mounted checkout of the repo. Anything under one of these is development state, not
# session output: the code dataset is published from the working tree.
CODE_MOUNT_MARKERS = ("behaviorsense-code", "behavioursense-code", "emotionsense")

# Synthetic data written by `--smoke` runs. Scoped to DATA suffixes on purpose: matching
# the name alone would exclude `scripts/kaggle_smoke_test.py`, a real script the preflight
# depends on and a repo-contract token.
FIXTURE_SUFFIXES = frozenset({".npz", ".npy", ".pt", ".pth", ".db"})


def is_under_code_mount(path: Path) -> bool:
    """True if `path` sits inside a mounted copy of the repo."""
    lowered = [part.lower() for part in path.parts]
    return any(marker in part for part in lowered for marker in CODE_MOUNT_MARKERS)


def is_fixture(path: Path) -> bool:
    """True if `path` is synthetic test DATA by this repo's naming convention."""
    if path.suffix.lower() not in FIXTURE_SUFFIXES:
        return False
    name = path.name.lower()
    return name.startswith("_") or "smoke" in name


def is_real_artifact(path: Path) -> bool:
    """True if `path` is genuine session output rather than a development leftover."""
    if is_fixture(path):
        return False
    if is_under_code_mount(path):
        # A checkpoint or shard inside the code mount is leftover local state. Other
        # files there (src/, scripts/, configs/) are exactly what we mount it for, so
        # only DATA suffixes are excluded.
        return path.suffix.lower() not in FIXTURE_SUFFIXES | {".npz", ".pt"}
    return True


def real_artifacts(paths: Iterable[Path]) -> Iterator[Path]:
    """Filter an iterable of paths down to genuine session output."""
    return (p for p in paths if is_real_artifact(p))


def find_run_dir(root: Path, prefix: str = "adl_") -> Path:
    """The directory holding trained per-stream checkpoints, by CONTENT not by name.

    Prefers a directory that actually contains `<prefix><stream>/` subdirectories, which
    is what `EnsembleClassifier.from_run_dir` needs. The previous rule - first `last.pt`
    in sorted order - picked `behaviorsense-code/.../runs` because "code" sorts before
    "runs", and that directory holds `adl/` and `fall/` rather than `adl_joint/`. The
    error named a missing checkpoint when the real problem was the wrong directory.
    """
    candidates: dict[Path, int] = {}
    for ckpt in root.glob("**/*.pt"):
        if not is_real_artifact(ckpt):
            continue
        if ckpt.name not in ("best.pt", "last.pt"):
            continue
        run_root = ckpt.parent.parent
        if ckpt.parent.name.startswith(prefix):
            candidates[run_root] = candidates.get(run_root, 0) + 1
    if not candidates:
        raise FileNotFoundError(
            f"no {prefix}<stream>/best.pt or last.pt under {root}. Checked "
            f"{sum(1 for _ in root.glob('**/*.pt'))} .pt file(s); none sat in a "
            f"'{prefix}<stream>' directory outside a mounted code checkout. Attach the "
            "dataset produced by notebook 03 (behaviorsense-runs)."
        )
    # Most streams wins - a real run has four, a stray directory has one.
    return max(candidates.items(), key=lambda kv: (kv[1], str(kv[0])))[0]


__all__ = [
    "CODE_MOUNT_MARKERS",
    "FIXTURE_SUFFIXES",
    "find_run_dir",
    "is_fixture",
    "is_real_artifact",
    "is_under_code_mount",
    "real_artifacts",
]
