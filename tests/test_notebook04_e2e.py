"""Execute notebook 04 end-to-end on synthetic data, locally, on CPU.

Why this exists
---------------
Notebook 04 failed on Kaggle four times in a row, each time after 8-12 minutes of GPU
time, each time on a different cell:

  1. `run_dir()` resolved to the code dataset's leftover `runs/` (wrong directory)
  2. P2 built its tensor with `fds[i][0]` -> [C,T,V,M], which `logits()` rejects
  3. the ablation cell unpacked 3 values from a `scores()` that returns 4

Every one is a defect that a local execution would have caught in seconds. The existing
tests re-implemented each block in Python and validated the copy - which is precisely why
they passed while the notebook did not: N5 tested its own version of P1, so P2's bug and
the ablation's arity mismatch were invisible.

Then the notebook ran to completion for 9,515 seconds and produced almost nothing durable:
the bundling cell globbed `results/*.md` and found a single file written by a subprocess,
so P1, the per-class breakdown, calibration, P2 and the ablation survived only as session
scrollback. That is the failure this harness now also covers - a cell that *runs* while
writing nothing is not a passing cell, so the assertions below check file CONTENT, not
just that the cells executed.

This harness runs the REAL cell sources, in order, sharing one namespace, against tiny
synthetic shards and checkpoints. It cannot verify the numbers are good - only a GPU
session on real data does that - but it verifies every cell RUNS and that the tables reach
disk, which are the failure modes that have actually cost time.

Run: python tests/test_notebook04_e2e.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from behaviorsense.data.skeleton_dataset import N_CLASSES  # noqa: E402
from behaviorsense.models.stgcnpp import STGCNpp  # noqa: E402

NL = chr(10)
NOTEBOOK = ROOT / "notebooks" / "04_evaluate_blackwell_offline.ipynb"
FALL_CLASSES = (7, 8)


def make_shard(path: Path, n: int, seed: int, datasets: list[str],
               fall_frac: float = 0.0) -> None:
    """A shard in the layout notebooks 01/02 write."""
    rng = np.random.default_rng(seed)
    sk = rng.normal(0, 0.4, (n, 30, 2, 17, 3)).astype(np.float16)
    sk[..., 2] = 0.9
    if fall_frac:
        labels = np.where(rng.random(n) < fall_frac,
                          rng.choice(FALL_CLASSES, n),
                          rng.integers(0, 7, n)).astype(np.int64)
    else:
        labels = rng.integers(0, N_CLASSES, n).astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        skeletons=sk,
        labels=labels,
        # Enough distinct subjects that a 15% split is non-degenerate.
        subjects=np.array([f"v{i % 40:03d}" for i in range(n)], dtype="<U32"),
        datasets=np.array([datasets[i % len(datasets)] for i in range(n)], dtype="<U32"),
    )


def save_ckpt(path: Path, stream: str, seed: int) -> None:
    torch.manual_seed(seed)
    model = STGCNpp(n_classes=N_CLASSES)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "ema": model.state_dict(), "ema_step": 1,
         "epoch": 29, "global_step": 100, "best": 0.19,
         "args": {"stream": stream, "epochs": 30, "smoke": False}, "metrics": {}},
        path,
    )


def build_input_tree(root: Path) -> None:
    """The Kaggle mount layout: /kaggle/input/datasets/<owner>/<name>/..."""
    ds = root / "datasets" / "someowner"

    adl = ds / "behaviorsense-adl-shards" / "shards"
    for i in range(2):
        make_shard(adl / f"charades_{i:04d}.npz", n=200, seed=i, datasets=["charades"])

    fall = ds / "behaviorsense-fall-shards" / "shards"
    for i, corpus in enumerate(("gmdcsa", "urfd", "caucafall", "le2i")):
        make_shard(fall / f"falls_{i:04d}.npz", n=120, seed=10 + i,
                   datasets=[corpus], fall_frac=0.3)

    runs = ds / "behaviorsense-runs" / "runs"
    for i, stream in enumerate(("joint", "bone", "joint_motion", "bone_motion")):
        save_ckpt(runs / f"adl_{stream}" / "best.pt", stream, seed=i)
        save_ckpt(runs / f"adl_{stream}" / "last.pt", stream, seed=i)

    # The code dataset must be a REAL copy of the checkout, not a stub: the resolver cell
    # enforces the repo contract, so a fixture holding only __init__.py is correctly
    # rejected as stale. Copying the real tree also means this harness exercises the
    # contract itself - if a contract token goes missing locally, this fails too.
    code = ds / "behaviorsense-code" / "EmotionSense-Extended"
    code.mkdir(parents=True)
    for sub in ("src", "scripts", "configs"):
        shutil.copytree(ROOT / sub, code / sub,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # The contaminated leftovers that the guards must skip, exactly as they ship.
    save_ckpt(code / "runs" / "adl" / "last.pt", "joint", seed=99)
    make_shard(code / "data" / "shards" / "_smoke_fall.npz", n=20, seed=99,
               datasets=["smoke"], fall_frac=0.5)

    ww = ds / "behavioursense-ww"
    (ww / "weights").mkdir(parents=True)
    (ww / "weights" / "rtmo-l.onnx").write_bytes(b"x")
    (ww / "wheels").mkdir(parents=True)
    (ww / "wheels" / "torch-2.7.1-cp311-cp311-linux_x86_64.whl").write_bytes(b"x")


def cells_to_run() -> list[tuple[int, str]]:
    """Code cells, minus the ones that need network, wheels, or a 16 GB LLM.

    The skip rule matches how a cell INVOKES the thing, never a mention of its name. Both
    weaker rules have already failed here: a substring check for "pip install" hit a
    comment in the resolver cell, and a check for "eval_hallucination" hit the repo
    contract, which now lists that script as a required file. Either one skips the cell
    that defines INPUT, and every later cell then dies with NameError - a green-looking
    skip that silently stops testing the notebook.
    """
    doc = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    out = []
    for i, cell in enumerate(doc["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        code_only = NL.join(L for L in src.splitlines()
                            if not L.lstrip().startswith("#"))
        if "pip" in code_only and "install" in code_only:
            continue                      # offline wheel install
        if '"--backend", "qwen"' in code_only:
            continue                      # needs the 16 GB model; N7 covers it
        out.append((i, src))
    return out


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        inp = root / "input"
        inp.mkdir()
        build_input_tree(inp)
        results = root / "results"

        # The bundling cell reads BS_RESULTS_DIR so it can be exercised off-Kaggle. It
        # writes evaluation.md - the P1/P2/calibration/ablation tables - which the first
        # real run did NOT produce: nine hours of GPU output existed only in the session
        # log. A cell that writes the dissertation numbers has to be executed by this
        # harness, not skipped as housekeeping.
        os.environ["BS_RESULTS_DIR"] = str(results)

        ns: dict = {"__name__": "nb04"}
        ran = 0
        for idx, src in cells_to_run():
            # Point the cell at the fixture tree, and keep everything on CPU.
            patched = (src.replace('pathlib.Path("/kaggle/input")',
                                   f'pathlib.Path(r"{inp}")')
                          .replace('device="cuda"', 'device="cpu"'))
            try:
                exec(compile(patched, f"<cell {idx}>", "exec"), ns)
                ran += 1
            except Exception as exc:                      # noqa: BLE001
                print(f"\nFAIL cell {idx}: {type(exc).__name__}: {exc}")
                import traceback
                traceback.print_exc()
                print("\n0/1 passed")
                return 1

        try:
            # The cells printed their own tables above; assert the load-bearing names
            # exist so a cell that silently no-ops cannot pass.
            for name in ("ADL", "clf", "X", "y", "ens_logits", "T",
                         "P1_ROWS", "P2_ROWS", "ABLATION_ROWS"):
                assert name in ns, f"cell ran but never bound {name!r}"
            assert ns["X"].shape[1:] == (30, 2, 17, 3), ns["X"].shape
            assert ns["ens_logits"].shape[1] == N_CLASSES

            # A written file, with the numbers in it. Globbing for *.md would pass on an
            # empty directory, which is exactly how the old bundler reported success
            # while writing nothing.
            report = results / "evaluation.md"
            assert report.is_file(), f"bundling cell wrote no evaluation.md into {results}"
            text = report.read_text(encoding="utf-8")
            for needle in ("# P1", "## Calibration", "## P2", "## Ablation",
                           "ENSEMBLE", "logit-average"):
                assert needle in text, f"evaluation.md is missing {needle!r}"
            # Every P2 fold must appear by name: a table that silently drops a corpus
            # looks complete and is not.
            for held, *_ in ns["P2_ROWS"]:
                assert held in text, f"P2 fold {held!r} missing from evaluation.md"
        except AssertionError as exc:
            print(f"\n  FAIL: {exc}")
            print("\n0/1 passed")
            return 1

        print(f"\n  E2E notebook 04: {ran} cells executed end to end on CPU; "
              f"P1 tensor {ns['X'].shape}, temperature {ns['T']:.2f}, "
              f"evaluation.md {len(text.splitlines())} lines covering "
              f"{len(ns['P1_ROWS'])} P1 rows / {len(ns['P2_ROWS'])} P2 folds")
        # run_tests.py aggregates the last line containing "passed"; without it this
        # suite reports "?" in the summary table and its result is invisible.
        print("\n1/1 passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
