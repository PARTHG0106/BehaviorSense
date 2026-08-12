"""Train the binary fall head (Agent 2), and fit its operating point honestly.

Two things separate this from `train_adl.py`, both forced by falls being rare and
safety-critical:

  1. **Focal loss** (Lin et al., ICCV 2017), gamma=2, alpha=0.75. Plain BCE on a ~2%
     positive rate optimises the majority class: predicting "no fall" always scores 98%
     accuracy and is useless. Focal down-weights the easy negatives that dominate the
     gradient.
  2. **The threshold is fitted at a false-alarms-per-hour budget, never left at 0.5.**
     0.5 is an arbitrary point on the ROC that corresponds to no stated risk tolerance.
     A carer can act on "at most 1 false alarm per 24 h"; nobody can act on "we used the
     default". Same philosophy as the CUSUM k table (results/tuning_log.md) and the ReID
     tau (results/reid_eval.md).

Usage:
    python scripts/train_fall.py --shards data/shards/fall_*.npz --epochs 60
    python scripts/train_fall.py --smoke
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from behaviorsense.data.skeleton_dataset import (  # noqa: E402
    AugmentConfig,
    SkeletonWindowDataset,
    split_by_subject,
)
from behaviorsense.models.stgcnpp import STGCNpp, make_stream  # noqa: E402
from train_adl import EMA, cosine_lr, load_rng_state, rng_state, set_seed  # noqa: E402

FALL_CLASSES = (7, 8)  # falling, fallen_on_ground
WINDOW_S = 2.0


class FocalLoss(nn.Module):
    """Binary focal loss with class balancing.

    alpha=0.75 weights the positive (fall) class, gamma=2 down-weights well-classified
    examples. Both matter here: without alpha the head is dominated by negatives,
    without gamma it spends capacity on the thousands of obviously-not-a-fall windows.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_t * (1 - p_t).pow(self.gamma) * ce).mean()


class FallHead(nn.Module):
    """ST-GCN++ backbone + a single-logit head."""

    def __init__(self) -> None:
        super().__init__()
        self.net = STGCNpp(n_classes=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@torch.no_grad()
def collect_scores(model: nn.Module, loader: DataLoader, stream: str,
                   device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, targets = [], []
    for x, y in loader:
        x = make_stream(x.to(device), stream)
        scores.append(torch.sigmoid(model(x)).cpu().numpy())
        targets.append((np.isin(y.numpy(), FALL_CLASSES)).astype(np.float32))
    return np.concatenate(scores), np.concatenate(targets)


def fit_operating_point(
    scores: np.ndarray, targets: np.ndarray, window_s: float = WINDOW_S,
    fa_per_hour_budget: float = 1.0,
) -> dict:
    """Choose the threshold with the highest sensitivity within a false-alarm budget.

    The budget is expressed per HOUR of monitored video, which is the unit a deployment
    actually cares about, rather than as a rate over an arbitrary test-set composition.
    Reported alongside AUROC and AUPRC - AUPRC being the honest summary under class
    imbalance, since AUROC flatters a detector on a 98%-negative set.
    """
    neg = scores[targets == 0]
    pos = scores[targets == 1]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError(f"need both classes; got {len(pos)} positive, {len(neg)} negative")

    hours = len(neg) * window_s / 3600.0
    best = None
    for thr in np.unique(np.round(scores, 4)):
        fa = int((neg >= thr).sum())
        fa_per_hour = fa / max(hours, 1e-9)
        sens = float((pos >= thr).mean())
        if fa_per_hour <= fa_per_hour_budget and (best is None or sens > best["sensitivity"]):
            best = {"threshold": float(thr), "sensitivity": sens,
                    "false_alarms_per_hour": fa_per_hour,
                    "specificity": float((neg < thr).mean())}
    if best is None:
        thr = float(scores.max())
        best = {"threshold": thr, "sensitivity": float((pos >= thr).mean()),
                "false_alarms_per_hour": float((neg >= thr).sum() / max(hours, 1e-9)),
                "specificity": float((neg < thr).mean()),
                "note": "budget unachievable at any threshold"}

    order = np.argsort(-scores)
    t_sorted = targets[order]
    tp = np.cumsum(t_sorted)
    fp = np.cumsum(1 - t_sorted)
    tpr = tp / max(tp[-1], 1)
    fpr = fp / max(fp[-1], 1)
    best["auroc"] = float(np.trapezoid(tpr, fpr))
    precision = tp / np.maximum(tp + fp, 1)
    best["auprc"] = float(np.trapezoid(precision, tpr))
    best["positive_rate"] = float(targets.mean())
    best["monitored_hours"] = float(hours)
    return best


def make_smoke_shard(path: Path, n: int = 400, seed: int = 0) -> Path:
    """Synthetic fall data: falls have a large downward displacement, others do not."""
    rng = np.random.default_rng(seed)
    T, M, V = 30, 2, 17
    is_fall = rng.random(n) < 0.15
    labels = np.where(is_fall, 7, rng.integers(0, 6, size=n))
    x = np.zeros((n, T, M, V, 3), dtype=np.float32)
    base = np.zeros((V, 2), dtype=np.float32)
    base[[5, 6]] = [[-0.2, 0.5], [0.2, 0.5]]
    base[[11, 12]] = [[-0.15, -0.5], [0.15, -0.5]]
    base[[15, 16]] = [[-0.15, -1.9], [0.15, -1.9]]
    base[[9, 10]] = [[-0.5, -0.2], [0.5, -0.2]]
    for i in range(n):
        person = np.tile(base, (T, 1, 1))
        if is_fall[i]:
            drop = np.linspace(0, 1.5, T)[:, None]
            person[..., 1] -= drop            # torso descends
            person[..., 0] += drop * 0.4      # and tips sideways
        person += rng.normal(0, 0.03, size=person.shape).astype(np.float32)
        x[i, :, 0, :, :2] = person
        x[i, :, 0, :, 2] = 0.9
    subjects = np.array([f"s{i % 10:02d}" for i in range(n)], dtype="<U32")
    datasets = np.array(["smoke"] * n, dtype="<U32")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, skeletons=x.astype(np.float16), labels=labels,
                        subjects=subjects, datasets=datasets)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", nargs="*", default=[])
    ap.add_argument("--stream", default="joint")
    # 60 STAYS, deliberately. The ADL heads overfit badly at 80 epochs (peak at 8-11,
    # then -27%), and the obvious move is to cut this budget too - but the measured fall
    # run says the opposite: AUPRC peaked at epoch 56 of 60, i.e. 93% of the way through,
    # still climbing. The fall head is one logit over ~1.5% positives, so it needs the
    # epochs the 18-class head does not. Patience below protects it either way.
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15,
                    help="stop if AUPRC has not improved for N epochs (0 disables). "
                         "Wider than the ADL default because the measured fall run was "
                         "still improving at epoch 56 of 60.")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--focal-alpha", type=float, default=0.75)
    ap.add_argument("--fa-budget", type=float, default=1.0,
                    help="false alarms per hour of monitored video")
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/fall")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-shard", default=None)
    args = ap.parse_args()

    if args.smoke_shard or args.smoke:
        args.smoke = True
        if not args.smoke_shard:
            args.epochs, args.batch_size, args.workers = 20, 32, 0
            args.warmup_epochs, args.device = 1, "cpu"
        shards = [str(make_smoke_shard(Path(args.smoke_shard or "data/shards/_smoke_fall.npz"),
                                       seed=args.seed))]
    else:
        shards = [p for pat in args.shards for p in glob.glob(pat)]
        if not shards:
            raise SystemExit("no shards matched; pass --shards or --smoke")

    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lr = args.lr if args.lr is not None else 0.1 * args.batch_size / 128

    probe = SkeletonWindowDataset(shards, n_frames=args.n_frames)
    tr_idx, va_idx = split_by_subject(probe.subjects, val_frac=0.25, seed=args.seed)
    train_ds = SkeletonWindowDataset(shards, n_frames=args.n_frames,
                                     augment_cfg=AugmentConfig(enabled=True),
                                     indices=tr_idx, seed=args.seed)
    val_ds = SkeletonWindowDataset(shards, n_frames=args.n_frames,
                                   augment_cfg=AugmentConfig(enabled=False),
                                   indices=va_idx, seed=args.seed)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=len(train_ds) > args.batch_size)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)

    model = FallHead().to(args.device)
    ema = EMA(model, decay=0.999)
    decay_p = [p for _, p in model.named_parameters() if p.ndim > 1]
    nodecay_p = [p for _, p in model.named_parameters() if p.ndim <= 1]
    opt = torch.optim.SGD([{"params": decay_p, "weight_decay": args.weight_decay},
                           {"params": nodecay_p, "weight_decay": 0.0}],
                          lr=lr, momentum=0.9, nesterov=True)
    crit = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    use_amp = args.device == "cuda"

    steps = max(1, len(train_ld))
    total, warm = steps * args.epochs, steps * args.warmup_epochs
    start_epoch, gstep, best = 0, 0, -1.0

    if args.resume and Path(args.resume).is_file():
        ck = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.shadow = {k: v.to(args.device) for k, v in ck["ema"].items()}
        ema.step = ck.get("ema_step", ck["global_step"])
        opt.load_state_dict(ck["optimizer"])
        # Same refusal as train_adl.py, for the same reason: the cosine schedule is a
        # function of TOTAL epochs, so resuming under different hyperparameters silently
        # puts the optimiser on a different curve than the one it was interrupted on.
        # The fall head also fits its operating point against these numbers - a changed
        # focal alpha/gamma mid-run makes the fitted threshold meaningless.
        prev = ck.get("args", {})
        for key in ("epochs", "batch_size", "lr", "warmup_epochs", "stream", "seed",
                    "focal_gamma", "focal_alpha"):
            old, new = prev.get(key), getattr(args, key)
            if old is not None and old != new:
                raise SystemExit(
                    f"resume mismatch: --{key.replace('_','-')} was {old!r} in "
                    f"{args.resume} but is {new!r} now. Re-run with the original value, "
                    "or start a fresh run."
                )
        start_epoch, gstep, best = ck["epoch"] + 1, ck["global_step"], ck["best"]
        load_rng_state(ck["rng"])
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    pos_rate = float(np.isin(train_ds.labels[tr_idx], FALL_CLASSES).mean())
    print(f"train {len(train_ds)} / val {len(val_ds)} windows | positive rate "
          f"{pos_rate:.3f} | lr={lr:.4f} | device={args.device}")

    # alpha weights the POSITIVE class; (1 - alpha) the negative. That is only balancing
    # if positives are the rare class. Passing fall shards alone makes the data 82%
    # positive, and the default alpha=0.75 then weights the MAJORITY 3x the minority -
    # the exact opposite of the intent, with nothing in the loss curve to show for it.
    # Refuse rather than warn: the run would complete, report a plausible AUPRC, and be
    # wrong in the one head where that matters most.
    if pos_rate > 0.5 and args.focal_alpha > 0.5:
        raise SystemExit(
            f"focal alpha={args.focal_alpha} up-weights the positive class, but this data "
            f"is {pos_rate:.1%} POSITIVE - the majority would be weighted "
            f"{args.focal_alpha / (1 - args.focal_alpha):.1f}x the minority.\n"
            f"  Fall shards alone are ~82% positive; pass the ADL shards too so the ADL "
            f"windows become negatives (notebook 03 does this), or set --focal-alpha "
            f"{1 - args.focal_alpha:.2f} if you really mean to train on this mix."
        )
    # The operating point is quoted per HOUR of monitored video, so a val set with few
    # negatives cannot express the budget: 141 negatives is 0.08 h, and "1 FA/hour" then
    # permits 0.08 of an alarm. The fitted threshold collapses to whatever admits zero,
    # which is a property of the test set, not a deployment decision.
    n_val_neg = int((~np.isin(val_ds.labels[va_idx], FALL_CLASSES)).sum())
    val_hours = n_val_neg * WINDOW_S / 3600.0
    if val_hours * args.fa_budget < 5:
        print(f"  WARNING: {n_val_neg} negative val windows = {val_hours:.2f} h; at "
              f"--fa-budget {args.fa_budget}/h that is {val_hours * args.fa_budget:.2f} "
              f"permitted false alarms. The fitted threshold will be dominated by test-set "
              f"size - report it with the negative count, or add more negatives.")

    history = []
    if start_epoch > 0 and (out_dir / "history.json").is_file():
        # Same as train_adl.py: resume continues the recorded curve, not restarts it.
        history = json.loads((out_dir / "history.json").read_text(encoding="utf-8"))
        history = [h for h in history if h["epoch"] < start_epoch]
    best_at_epoch = (best, start_epoch - 1)
    if history:
        top = max(history, key=lambda h: h.get("auprc", -1))
        best_at_epoch = (top.get("auprc", best), top.get("epoch", start_epoch - 1))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, run, seen = time.time(), 0.0, 0
        for x, y in train_ld:
            for g in opt.param_groups:
                g["lr"] = cosine_lr(gstep, total, warm, lr)
            x = make_stream(x.to(args.device), args.stream)
            target = torch.from_numpy(
                np.isin(y.numpy(), FALL_CLASSES).astype(np.float32)
            ).to(args.device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = crit(model(x), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ema.update(model)
            run += loss.item() * len(y)
            seen += len(y)
            gstep += 1

        eval_model = FallHead().to(args.device)
        eval_model.load_state_dict(model.state_dict())
        ema.copy_to(eval_model)
        scores, targets = collect_scores(eval_model, val_ld, args.stream, args.device)
        op = fit_operating_point(scores, targets, fa_per_hour_budget=args.fa_budget)
        op.update(epoch=epoch, train_loss=run / max(1, seen), secs=time.time() - t0)
        history.append(op)
        print(f"ep{epoch:03d} loss {op['train_loss']:.4f} auprc {op['auprc']:.3f} "
              f"auroc {op['auroc']:.3f} sens@{args.fa_budget}fa/h {op['sensitivity']:.3f} "
              f"thr {op['threshold']:.3f} ({op['secs']:.0f}s)")

        ck = {"model": model.state_dict(), "ema": ema.shadow, "ema_step": ema.step,
              "optimizer": opt.state_dict(), "epoch": epoch, "global_step": gstep,
              "best": max(best, op["auprc"]), "rng": rng_state(),
              "args": vars(args), "operating_point": op}
        torch.save(ck, out_dir / "last.pt")
        if op["auprc"] > best:
            best = op["auprc"]
            torch.save(ck, out_dir / "best.pt")
        if op["auprc"] > best_at_epoch[0]:
            best_at_epoch = (op["auprc"], epoch)
        stale = epoch - best_at_epoch[1]
        if args.patience and stale >= args.patience:
            print(f"early stop at epoch {epoch}: AUPRC has not improved for {stale} "
                  f"epochs (best {best_at_epoch[0]:.4f} @ep{best_at_epoch[1]}). "
                  "best.pt holds the peak.")
            break
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    final = history[-1]
    print(f"done. best AUPRC {best:.4f}")
    print(f"OPERATING POINT: threshold {final['threshold']:.4f} -> "
          f"sensitivity {final['sensitivity']:.3f} at "
          f"{final['false_alarms_per_hour']:.2f} false alarms/hour "
          f"({final['monitored_hours']:.1f} h of negatives)")
    if args.smoke and not args.smoke_shard:
        first = history[0]["train_loss"]
        drop = (first - final["train_loss"]) / first
        print(f"SMOKE: loss {first:.4f} -> {final['train_loss']:.4f} ({drop:.0%}), "
              f"AUPRC {history[0]['auprc']:.3f} -> {final['auprc']:.3f}")
        if final["auprc"] < 0.6:
            raise SystemExit(f"SMOKE FAILED: AUPRC {final['auprc']:.3f} on separable falls")
        print("SMOKE PASSED")


if __name__ == "__main__":
    main()
