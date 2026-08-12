"""Train ST-GCN++ on unified-taxonomy skeleton windows (Agent 2 ADL head).

Kaggle sessions are capped at 12 hours and can die without warning, so resume is a
correctness requirement rather than a convenience. The checkpoint therefore carries
model + EMA + optimizer + scheduler + epoch + **RNG state** (python/numpy/torch/CUDA).
Omitting RNG state makes a resumed run silently re-draw the same augmentations and
sampler order it already used, which is neither a fresh epoch nor a continuation - and
it is invisible in the loss curve.

Recipe (docs/06): SGD+Nesterov, cosine schedule with warmup, label smoothing 0.1,
class-balanced sampling, EMA weights for evaluation, bf16 autocast on GPU.

Usage:
    python scripts/train_adl.py --shards data/shards/*.npz --stream joint --epochs 80
    python scripts/train_adl.py ... --resume runs/adl_joint/last.pt
    python scripts/train_adl.py --smoke        # tiny synthetic run, CPU, ~1 min
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behaviorsense.data.skeleton_dataset import (  # noqa: E402
    N_CLASSES,
    AugmentConfig,
    SkeletonWindowDataset,
    class_balanced_sampler,
    split_by_subject,
)
from behaviorsense.models.stgcnpp import STGCNpp, make_stream  # noqa: E402

# A class with a handful of validation windows cannot have its F1 estimated: one
# prediction moves it by tens of points. 50 is the floor at which the number means
# something at our 20%% val split; below it the class is reported, not averaged.
MIN_SUPPORT = 50


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def load_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class EMA:
    """Exponential moving average of weights; the EMA model is what gets evaluated.

    Cheap (+0.3-0.8% typically) and it stabilises small-data fine-tuning, where the last
    epoch's weights are a noisier estimate than the average of the last few hundred steps.

    The decay is WARMED UP: `min(decay, (1 + step) / (10 + step))`. Without this, a fixed
    0.999 decay leaves the average ~95% random-initialisation for the first ~50 steps, so
    early-epoch validation measures the initialisation rather than the model. The CPU
    smoke test caught exactly this - training loss fell while validation sat at chance -
    and the same flaw would have quietly invalidated early checkpoint selection (and thus
    `best.pt`) in the real 80-epoch run.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.step = 0
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                self.shadow[k].copy_(v.detach())

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict({k: v.to(dtype=p.dtype) for (k, v), p
                               in zip(self.shadow.items(), model.state_dict().values())})


def cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, stream: str, device: str) -> dict:
    """Mean-class accuracy and macro-F1 - never top-1 alone, which imbalance flatters."""
    model.eval()
    n_cls = N_CLASSES
    conf = np.zeros((n_cls, n_cls), dtype=np.int64)
    for x, y in loader:
        x = make_stream(x.to(device), stream)
        pred = model(x).argmax(dim=1).cpu().numpy()
        for t, p in zip(y.numpy(), pred):
            conf[t, p] += 1

    per_class = np.divide(np.diag(conf), conf.sum(axis=1),
                          out=np.zeros(n_cls), where=conf.sum(axis=1) > 0)
    present = conf.sum(axis=1) > 0
    prec = np.divide(np.diag(conf), conf.sum(axis=0),
                     out=np.zeros(n_cls), where=conf.sum(axis=0) > 0)
    f1 = np.divide(2 * prec * per_class, prec + per_class,
                   out=np.zeros(n_cls), where=(prec + per_class) > 0)
    # Per-class F1 AND its support, because macro-F1 alone is unreadable at our imbalance.
    # Measured on the real Charades extraction (7,985 videos, 165,109 windows): two classes
    # arrive starved - standing 342 windows (0.21%) and bending_reaching 69 (0.04%) - because
    # Charades names those actions "Putting a box somewhere" / "Taking a bag from somewhere",
    # which no keyword rule in build_charades_map.py matches, so they fall through to
    # other_idle (39.4%). Their F1 is ~0, and averaged over 18 present classes that alone
    # costs ~6 macro-F1 points. Reporting only the mean hides which classes failed and why,
    # so the caller gets both the vector and a support-thresholded mean.
    support = conf.sum(axis=1)
    supported = support >= MIN_SUPPORT
    return {
        "top1": float(np.diag(conf).sum() / max(1, conf.sum())),
        "mean_class_acc": float(per_class[present].mean()),
        "macro_f1": float(f1[present].mean()),
        # Same metric over classes that have enough support to estimate it at all. Reported
        # ALONGSIDE macro_f1, never instead of it - dropping the starved classes silently
        # would be tuning the metric rather than measuring the model.
        "macro_f1_supported": (float(f1[supported].mean()) if supported.any()
                               else float("nan")),
        "n_supported": int(supported.sum()),
        "per_class_f1": [round(float(v), 4) for v in f1],
        "per_class_support": [int(v) for v in support],
        "n": int(conf.sum()),
    }


def make_smoke_shard(path: Path, n: int = 480, n_classes: int = 6, seed: int = 0) -> Path:
    """Synthetic shard with a SEPARABLE signal, for the CPU smoke test.

    Two properties this fixture must have, both learned by getting them wrong first:

      1. **Classes must be distinguishable.** The first version keyed class c on joint
         `c % 17` over 20 classes, so classes 0/17, 1/18 and 2/19 were pixel-identical.
         An unlearnable fixture cannot distinguish a broken training loop from an
         impossible task - the same trap as a vacuous test, inverted.
      2. **The signal must survive preprocessing.** `normalise()` divides by torso length,
         so a signal written into shoulder or hip joints is partly normalised away. It is
         written into wrist/ankle joints instead, and the shard uses a realistic body
         layout rather than pure noise so torso length means something.

    `n_classes` is 6, not 20: the point is to prove the loop learns, and 20 classes over
    a few hundred CPU windows cannot separate "not learning" from "not enough data".
    """
    rng = np.random.default_rng(seed)
    T, M, V, C = 30, 2, 17, 3
    # Rough upright body layout in normalised units, so torso length is ~1 and the
    # normalisation step behaves as it would on real skeletons.
    base = np.zeros((V, 2), dtype=np.float32)
    base[[5, 6]] = [[-0.2, 0.5], [0.2, 0.5]]        # shoulders
    base[[11, 12]] = [[-0.15, -0.5], [0.15, -0.5]]  # hips
    base[[7, 8]] = [[-0.4, 0.1], [0.4, 0.1]]        # elbows
    base[[9, 10]] = [[-0.5, -0.2], [0.5, -0.2]]     # wrists
    base[[13, 14]] = [[-0.15, -1.2], [0.15, -1.2]]  # knees
    base[[15, 16]] = [[-0.15, -1.9], [0.15, -1.9]]  # ankles
    base[0] = [0.0, 0.8]

    # One moving limb per class, on extremities that survive torso normalisation.
    movers = [9, 10, 15, 16, 7, 8, 13, 14]
    labels = rng.integers(0, n_classes, size=n)
    x = np.zeros((n, T, M, V, C), dtype=np.float32)
    for i, lab in enumerate(labels):
        joint = movers[lab % len(movers)]
        phase = np.linspace(0, 2 * np.pi, T) + rng.uniform(0, 0.5)
        person = np.tile(base, (T, 1, 1))
        person[:, joint, 0] += np.sin(phase) * 0.6
        person[:, joint, 1] += np.cos(phase) * 0.6
        person += rng.normal(0, 0.02, size=person.shape).astype(np.float32)
        x[i, :, 0, :, :2] = person
        x[i, :, 0, :, 2] = 0.9
    subjects = np.array([f"s{i % 12:02d}" for i in range(n)], dtype="<U32")
    datasets = np.array(["smoke"] * n, dtype="<U32")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, skeletons=x.astype(np.float16), labels=labels, subjects=subjects, datasets=datasets
    )
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", nargs="*", default=[])
    ap.add_argument("--stream", default="joint", choices=["joint", "bone", "joint_motion", "bone_motion"])
    # 80 came from ST-GCN++'s NTU-60 recipe and was wrong for this data. Measured on the
    # real Charades shards (165k windows, 18 classes present): all four streams peaked at
    # epoch 8-11 and then LOST ~27% mean-class-accuracy over the remaining 70 epochs -
    # adl_bone 0.184@ep11 -> 0.137@ep79, and the same story for the other three. NTU has
    # roughly 10x more labelled windows per class; copying its epoch budget across meant
    # ~2.4 of 2.7 GPU-hours spent memorising. 30 leaves headroom above the observed peak
    # while --patience ends the run once it stops improving.
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8,
                    help="stop if mean-class accuracy has not improved for N epochs "
                         "(0 disables). best.pt already holds the peak, so this only "
                         "saves time - it cannot lose a result.")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=None, help="default: 0.1 * batch/128")
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/adl")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic CPU run")
    ap.add_argument("--smoke-shard", default=None,
                    help="path for the synthetic shard; implies --smoke (used by tests)")
    ap.add_argument("--smoke-classes", type=int, default=6)
    ap.add_argument("--stop-after", type=int, default=None,
                    help="exit after this many epochs, keeping the schedule for the "
                         "full --epochs. Simulates a session timeout; used by tests.")
    args = ap.parse_args()

    if args.smoke_shard:
        # Explicit-path smoke mode: the test harness controls epochs/seed itself, so
        # nothing is overridden here. Overriding them would make the resume test
        # compare two runs that never used the requested epoch counts.
        args.smoke = True
        shards = [str(make_smoke_shard(Path(args.smoke_shard),
                                       n_classes=args.smoke_classes, seed=args.seed))]
    elif args.smoke:
        args.epochs, args.batch_size, args.workers = 25, 32, 0
        args.warmup_epochs, args.device = 1, "cpu"
        shards = [str(make_smoke_shard(Path("data/shards/_smoke.npz"),
                                       n_classes=args.smoke_classes, seed=args.seed))]
    else:
        shards = [p for pat in args.shards for p in glob.glob(pat)]
        if not shards:
            raise SystemExit("no shards matched; pass --shards 'data/shards/*.npz' or --smoke")

    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lr = args.lr if args.lr is not None else 0.1 * args.batch_size / 128

    probe = SkeletonWindowDataset(shards, n_frames=args.n_frames)
    train_idx, val_idx = split_by_subject(probe.subjects, val_frac=0.2, seed=args.seed)

    train_ds = SkeletonWindowDataset(
        shards, n_frames=args.n_frames, augment_cfg=AugmentConfig(enabled=True),
        indices=train_idx, seed=args.seed,
    )
    val_ds = SkeletonWindowDataset(
        shards, n_frames=args.n_frames, augment_cfg=AugmentConfig(enabled=False),
        indices=val_idx, seed=args.seed,
    )
    # The sampler deliberately gets NO explicit generator: WeightedRandomSampler then
    # draws from the global torch RNG, and load_rng_state() restores that on resume, so
    # the draw order is part of the checkpointed state. An explicit generator would need
    # its own state checkpointed separately or resume would silently replay epoch 0's
    # sampling order.
    sampler = class_balanced_sampler(train_ds.labels[train_idx])
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=args.workers, pin_memory=args.device == "cuda",
                          drop_last=len(train_ds) > args.batch_size)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)

    model = STGCNpp(n_classes=N_CLASSES).to(args.device)
    ema = EMA(model, decay=args.ema_decay)
    # No weight decay on norm/bias: decaying them shifts normalisation statistics rather
    # than regularising, and costs accuracy for free.
    decay_p = [p for n, p in model.named_parameters() if p.ndim > 1]
    nodecay_p = [p for n, p in model.named_parameters() if p.ndim <= 1]
    opt = torch.optim.SGD(
        [{"params": decay_p, "weight_decay": args.weight_decay},
         {"params": nodecay_p, "weight_decay": 0.0}],
        lr=lr, momentum=0.9, nesterov=True,
    )
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    use_amp = args.device == "cuda"

    steps_per_epoch = max(1, len(train_ld))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    start_epoch, gstep, best = 0, 0, -1.0
    if args.resume and Path(args.resume).is_file():
        ck = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.shadow = {k: v.to(args.device) for k, v in ck["ema"].items()}
        opt.load_state_dict(ck["optimizer"])
        # The LR schedule is a function of TOTAL epochs, so resuming a run with a
        # different --epochs silently puts the optimiser on a different cosine curve
        # than the one it was interrupted on. That is invisible in the loss trace and
        # makes the run unreproducible - refuse it rather than warn.
        prev = ck.get("args", {})
        for key in ("epochs", "batch_size", "lr", "warmup_epochs", "stream", "seed"):
            old, new = prev.get(key), getattr(args, key)
            if old is not None and old != new:
                raise SystemExit(
                    f"resume mismatch: --{key.replace('_','-')} was {old!r} in "
                    f"{args.resume} but is {new!r} now. Re-run with the original value, "
                    "or start a fresh run - continuing would change the LR schedule "
                    "mid-training and silently invalidate the result."
                )
        start_epoch, gstep, best = ck["epoch"] + 1, ck["global_step"], ck["best"]
        # EMA step must survive resume, or the decay warmup restarts and the
        # average is re-contaminated by the current weights mid-training.
        ema.step = ck.get("ema_step", gstep)
        load_rng_state(ck["rng"])
        print(f"resumed from {args.resume} at epoch {start_epoch} (best {best:.4f})")

    # (value, epoch) of the best validation score, so patience measures distance from the
    # PEAK rather than from the last improvement of a noisy metric.
    best_at_epoch = (best, start_epoch - 1)

    history = []
    if start_epoch > 0 and (out_dir / "history.json").is_file():
        # Resume continues the curve rather than overwriting it: the dissertation's
        # training figure comes from this file, and a run resumed at epoch 40 that
        # reports only epochs 40-79 looks like a 40-epoch run with a warm start.
        history = json.loads((out_dir / "history.json").read_text(encoding="utf-8"))
        history = [h for h in history if h["epoch"] < start_epoch]
        # Recover WHEN the peak happened. Without this, patience restarts its count at the
        # resume point and a run that already plateaued would train another full patience
        # window before stopping.
        if history:
            top = max(history, key=lambda h: h.get("mean_class_acc", -1))
            best_at_epoch = (top.get("mean_class_acc", best), top.get("epoch", start_epoch - 1))
    print(f"train {len(train_ds)} / val {len(val_ds)} windows | stream={args.stream} | "
          f"lr={lr:.4f} | device={args.device} | classes present={int((train_ds.class_counts>0).sum())}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for x, y in train_ld:
            for g in opt.param_groups:
                g["lr"] = cosine_lr(gstep, total_steps, warmup_steps, lr)
            x = make_stream(x.to(args.device, non_blocking=True), args.stream)
            y = y.to(args.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = crit(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ema.update(model)
            run_loss += loss.item() * len(y)
            seen += len(y)
            gstep += 1

        eval_model = STGCNpp(n_classes=N_CLASSES).to(args.device)
        eval_model.load_state_dict(model.state_dict())
        ema.copy_to(eval_model)
        metrics = evaluate(eval_model, val_ld, args.stream, args.device)
        metrics.update(epoch=epoch, train_loss=run_loss / max(1, seen),
                       lr=opt.param_groups[0]["lr"], secs=time.time() - t0)
        history.append(metrics)
        print(f"ep{epoch:03d} loss {metrics['train_loss']:.4f} "
              f"top1 {metrics['top1']:.3f} mca {metrics['mean_class_acc']:.3f} "
              f"f1 {metrics['macro_f1']:.3f} ({metrics['secs']:.0f}s)")

        ck = {
            "model": model.state_dict(), "ema": ema.shadow, "ema_step": ema.step,
            "optimizer": opt.state_dict(),
            "epoch": epoch, "global_step": gstep,
            "best": max(best, metrics["mean_class_acc"]),
            "rng": rng_state(), "args": vars(args), "metrics": metrics,
            "augment": asdict(AugmentConfig()),
        }
        torch.save(ck, out_dir / "last.pt")
        if metrics["mean_class_acc"] > best:
            best = metrics["mean_class_acc"]
            torch.save(ck, out_dir / "best.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if metrics["mean_class_acc"] > best_at_epoch[0]:
            best_at_epoch = (metrics["mean_class_acc"], epoch)
        stale = epoch - best_at_epoch[1]
        if args.patience and stale >= args.patience:
            # Everything after the peak is degradation, and best.pt already holds the
            # peak - so this ends the run without costing a result. Say WHY, with the
            # numbers, so the log reads as a decision rather than a truncation.
            print(f"early stop at epoch {epoch}: no improvement for {stale} epochs "
                  f"(best {best_at_epoch[0]:.4f} @ep{best_at_epoch[1]}, "
                  f"now {metrics['mean_class_acc']:.4f}). best.pt holds the peak.")
            break

        if args.stop_after is not None and (epoch - start_epoch + 1) >= args.stop_after:
            print(f"stopping after {args.stop_after} epoch(s) (simulated timeout)")
            return

    print(f"done. best mean-class acc {best:.4f} -> {out_dir/'best.pt'}")
    if history:
        last = history[-1]
        f1s, sup = last["per_class_f1"], last["per_class_support"]
        weak = [(i, f1s[i], sup[i]) for i in range(len(f1s))
                if 0 < sup[i] < MIN_SUPPORT]
        print(f"macro_f1 {last['macro_f1']:.4f} over {int(sum(s > 0 for s in sup))} "
              f"present classes | {last['macro_f1_supported']:.4f} over the "
              f"{last['n_supported']} with support >= {MIN_SUPPORT}")
        if weak:
            # Named here so a low macro-F1 is attributable rather than mysterious. These
            # are starved by the Charades label map, not by the model.
            print(f"  {len(weak)} class(es) below {MIN_SUPPORT} val windows - their F1 "
                  "is noise, and they drag macro_f1 down:")
            for cid, f, n in weak:
                print(f"    class {cid:>2}: support {n:>4}, f1 {f:.3f}")
    if args.smoke and not args.smoke_shard and history:
        first, last = history[0]["train_loss"], history[-1]["train_loss"]
        drop = (first - last) / first
        print(f"SMOKE: loss {first:.4f} -> {last:.4f} ({drop:.0%} drop), "
              f"mca {history[0]['mean_class_acc']:.3f} -> {history[-1]['mean_class_acc']:.3f}")
        if drop < 0.15:
            raise SystemExit(f"SMOKE FAILED: loss fell only {drop:.0%} on learnable data")
        print("SMOKE PASSED")


if __name__ == "__main__":
    main()
