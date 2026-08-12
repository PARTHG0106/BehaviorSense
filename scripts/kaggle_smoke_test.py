"""Kaggle preflight: verify the GPU and every offline asset in ~1 minute.

Why this script exists
----------------------
Training sessions are capped at 12 hours and have NO INTERNET. The two ways a session is
wasted are (a) the torch build does not support the GPU's compute capability, and (b) an
asset that was assumed present is missing or is a Git-LFS pointer rather than real
weights. Both fail *late* - often 40 minutes in, after preprocessing - and both are
detectable in under a minute.

Run this FIRST in every session, before any training command:

    PYTHONPATH=src python scripts/kaggle_smoke_test.py \
        --assets /kaggle/input/behaviorsense-weights

Exit code is nonzero if anything would break a training run, so it can gate a chained
command: `python scripts/kaggle_smoke_test.py && python scripts/train_adl.py ...`
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Assets the project depends on, and which session profiles actually BLOCK on each.
#
# The earlier version had a single `optional` flag, which conflated "the project needs
# this" with "this session needs this" and made the preflight fail a training run over an
# asset training never loads. Measured: notebooks 03/04 import nothing from osnet.py —
# they train ST-GCN++ from pre-extracted .npz shards and evaluate with Qwen — while OSNet
# is Agent 1 serving-time, and its operating point (tau=0.3546) is already fitted and
# committed in results/reid_eval.md.
#
# So scope the requirement to what the session does:
#   extraction  notebooks 01/02 - RTMO pose extraction from video
#   training    notebook 03     - ST-GCN++ from shards; loads no staged weights
#   serving     Agent 1 live    - identity resolution, needs the ReID checkpoints
# An asset listed for no profile is reported as a warning and never blocks.
PROFILES = ("extraction", "training", "serving", "all")

EXPECTED_ASSETS: tuple[tuple[str, str, frozenset[str]], ...] = (
    ("osnet_ain_x1_0_msmt17.pth", "OSNet-AIN ReID weights (Agent 1 identity)",
     frozenset({"serving"})),
    ("osnet_ain_x1_0_imagenet.pth", "ImageNet control for the ReID negative control",
     frozenset()),
    ("rtmo-l.onnx", "RTMO-l pose extraction (Stage A)",
     frozenset({"extraction", "serving"})),
    ("rtdetr-l.onnx", "RT-DETR object context (Agent 1, 1 Hz)", frozenset()),
    ("stgcnpp_ntu60_joint.pth", "ST-GCN++ NTU pretrain (Agent 2 init)", frozenset()),
)

MIN_FREE_VRAM_GB = 20.0
"""Below this, the planned 4-stream parallel training will not fit and the session should
be reconfigured to sequential rather than discovering OOM at epoch 3."""


def check_gpu() -> list[str]:
    problems: list[str] = []
    try:
        import torch
    except ImportError:
        return ["torch is not installed"]

    print(f"torch {torch.__version__}")
    if not torch.cuda.is_available():
        return ["CUDA is not available - training would silently run on CPU"]

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {name}  sm_{cap[0]}{cap[1]}  {total:.1f} GB")

    arch_list = torch.cuda.get_arch_list()
    sm = f"sm_{cap[0]}{cap[1]}"
    if sm not in arch_list:
        problems.append(
            f"this torch build does not support {sm} (supports {arch_list}). "
            "Kernels will fail or fall back catastrophically - install a cu128 build."
        )
    else:
        print(f"  {sm} is in the supported arch list")

    if total < MIN_FREE_VRAM_GB:
        problems.append(f"only {total:.1f} GB VRAM; expected >= {MIN_FREE_VRAM_GB}")

    # A real matmul + autocast + backward step: catches driver/kernel mismatches that
    # `cuda.is_available()` reports as fine.
    try:
        t0 = time.time()
        a = torch.randn(4096, 4096, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            b = (a @ a).sum()
        b.backward() if b.requires_grad else None
        torch.cuda.synchronize()
        print(f"  bf16 matmul 4096^2 OK ({time.time() - t0:.2f}s)")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"bf16 matmul failed: {type(exc).__name__}: {exc}")

    try:
        x = torch.randn(8, 3, 30, 17, 2, device="cuda", requires_grad=True)
        from behaviorsense.models.stgcnpp import STGCNpp

        model = STGCNpp(n_classes=20).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x).float().sum()
        loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  ST-GCN++ fwd+bwd under bf16 autocast OK (peak {peak:.2f} GB)")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"ST-GCN++ step failed: {type(exc).__name__}: {exc}")

    return problems


def check_assets(root: Path | None, profile: str = "all") -> list[str]:
    problems: list[str] = []
    if root is None:
        print("\n(--assets not given; skipping offline asset check)")
        return problems

    print(f"\nassets in {root}  (profile: {profile})")
    if not root.is_dir():
        return [f"asset directory {root} does not exist"]

    def blocking(required_for: frozenset[str]) -> bool:
        return bool(required_for) if profile == "all" else profile in required_for

    n_blocking = sum(1 for _, _, req in EXPECTED_ASSETS if blocking(req))
    if n_blocking == 0:
        # Say so out loud. A check that blocks on nothing must not read like a pass.
        print(f"  (no staged asset blocks the '{profile}' profile; all findings are "
              f"warnings)")

    present = {p.name: p for p in root.rglob("*") if p.is_file()}
    for name, purpose, required_for in EXPECTED_ASSETS:
        must_have = blocking(required_for)
        path = present.get(name)
        if path is None:
            msg = f"MISSING {name} - {purpose}"
            print(f"  [{'FAIL' if must_have else 'warn'}] {msg}")
            if must_have:
                problems.append(msg)
            continue

        size_mb = path.stat().st_size / 1e6
        # A Git-LFS pointer is a ~130-byte text file that loads as garbage. Catching it
        # here is the difference between a 1-minute fix and a lost session. This is
        # profile-independent: a corrupt file that IS present is always worth reporting.
        if size_mb < 0.05:
            problems.append(f"{name} is {size_mb*1000:.0f} KB - likely a Git-LFS pointer")
            print(f"  [FAIL] {name}: {size_mb*1000:.0f} KB (too small to be weights)")
            continue
        print(f"  [ok] {name}: {size_mb:.1f} MB")

        if name.endswith(".pth"):
            try:
                import torch

                torch.load(str(path), map_location="cpu", weights_only=False)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{name} failed to load: {type(exc).__name__}")
                print(f"  [FAIL] {name} does not load: {exc}")
    return problems


def check_imports() -> list[str]:
    """Confirm the packages training needs are importable offline."""
    problems: list[str] = []
    print("\nimports:")
    for mod, needed_for, optional in (
        ("numpy", "everything", False),
        ("pydantic", "schemas", False),
        ("torch", "training", False),
        ("PIL", "ReID crops", False),
        ("yaml", "taxonomy config", False),
        ("transformers", "Qwen reporter", True),
        ("outlines", "constrained decoding", True),
        ("onnxruntime", "RTMO/RT-DETR inference", True),
    ):
        try:
            __import__(mod)
            print(f"  [ok] {mod}")
        except ImportError:
            print(f"  [{'warn' if optional else 'FAIL'}] {mod} missing ({needed_for})")
            if not optional:
                problems.append(f"{mod} is not importable but is required for {needed_for}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", default=None, help="directory holding staged weights")
    ap.add_argument("--profile", default="all", choices=PROFILES,
                    help="which session this is; scopes which assets BLOCK "
                         "(default: all, the strictest)")
    args = ap.parse_args()

    print("=" * 70)
    print("BehaviorSense Kaggle preflight")
    print("=" * 70)

    problems = check_gpu() + check_imports() + check_assets(
        Path(args.assets) if args.assets else None, args.profile
    )

    print("\n" + "=" * 70)
    if problems:
        print(f"PREFLIGHT FAILED - {len(problems)} blocking problem(s):")
        for p in problems:
            print(f"  - {p}")
        print("\nFix these before starting a training run; each one would otherwise")
        print("surface late in the session, after preprocessing time is already spent.")
        sys.exit(1)
    print("PREFLIGHT PASSED - safe to start training")


if __name__ == "__main__":
    main()
