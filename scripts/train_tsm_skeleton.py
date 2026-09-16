"""Train ST-GCN++ on Toyota Smarthome's own 3D skeletons, INRIA's cross-subject protocol.

Consumes the shards `build_toyota_skeleton_shards.py` writes and reports what the Toyota paper
reports: top-1 accuracy AND mean per-class accuracy on the CS test split, plus the full per-class
table. Both metrics, always, because they disagree here by construction - `Walk` has 3,942 clips
and `Cutbread` has 45, so a model that learns the head of the distribution and nothing else
scores well on one and badly on the other. Quoting only top-1 on an 88x imbalance is how a
useless model looks finished.

Three things this does that a default training loop would not:

**Class-balanced loss (Cui et al. 2019).** Weights by effective number, not by inverse frequency:
1/n over-corrects hard when the tail has tens of samples, because the 45 `Cutbread` clips are not
88 times as informative as one `Walk` clip - they overlap heavily with each other.

**Streams are derived, never stored.** Bone and motion are exact functions of the joint stream, so
the shards hold only joints and `make_stream(..., parents=...)` produces the rest on the GPU.
One source of truth for the bone tree.

**Support is reported beside every per-class number.** A class with 12 test clips has a per-class
accuracy quantised to steps of 8 percentage points, and that has to be visible next to the value
rather than inferred later - the same reason `MIN_SUPPORT` exists on the Charades side.

    python scripts/train_tsm_skeleton.py --shards /kaggle/input/.../shards \\
        --out /kaggle/working/runs/tsm_bone --stream bone --epochs 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behaviorsense.data.toyota import (  # noqa: E402
    AGCN_JOINTS,
    COARSE_V11,
    TSM_CLASSES,
    TSM_TO_COARSE,
    coarse_id,
)
from behaviorsense.data.toyota_skeleton import (  # noqa: E402
    BONE_EDGES,
    ROOT_JOINT,
    flip_lr,
)
from behaviorsense.models.stgcnpp import STGCNpp, build_adjacency, make_stream  # noqa: E402

PARENTS = dict(BONE_EDGES)
STREAMS = ("joint", "bone", "joint_motion", "bone_motion")


class ClipDataset(Dataset):
    """Clips from the shards, one split, with the 50% mirror on train only.

    The flip is applied to the JOINT stream and the bone/motion streams are derived after, which
    is the order that keeps them consistent - `flip_lr` and `bone_stream` are proven to commute
    at import time, so either order is correct here, but deriving after means one code path.
    """

    def __init__(self, joints: np.ndarray, labels: np.ndarray, *, train: bool,
                 rng_seed: int = 0) -> None:
        self.joints = joints                     # [N, C, T, V] float16 on disk
        self.labels = labels.astype(np.int64)
        self.train = train
        self.rng = np.random.default_rng(rng_seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        x = self.joints[i].astype(np.float32)             # [C, T, V]
        if self.train and self.rng.random() < 0.5:
            # flip_lr wants [T, V, C]; the shards are channels-first for the model's benefit.
            x = flip_lr(x.transpose(1, 2, 0)).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.labels[i])


def load_shards(shard_dir: Path) -> dict:
    """Every `*.npz` under `shard_dir`, concatenated, with the CS split as stored."""
    files = sorted(shard_dir.rglob("tsm_*_*.npz"))
    if not files:
        raise SystemExit(f"no tsm_*.npz under {shard_dir}")
    joints, labels, splits, subjects, clips = [], [], [], [], []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            joints.append(z["joint"])
            labels.append(z["labels"])
            splits.append(z["split"])
            subjects.append(z["subjects"])
            clips.append(z["clips"])
    out = {
        "joint": np.concatenate(joints),
        "labels": np.concatenate(labels),
        "split": np.concatenate(splits),
        "subjects": np.concatenate(subjects),
        "clips": np.concatenate(clips),
        "files": [f.name for f in files],
    }
    v = out["joint"].shape[-1]
    if v != len(AGCN_JOINTS):
        raise SystemExit(f"shards hold {v} joints, the graph has {len(AGCN_JOINTS)}")
    return out


def effective_number_weights(counts: np.ndarray, beta: float = 0.9999) -> np.ndarray:
    """Cui et al. 2019. `(1-beta) / (1-beta**n)`, normalised to mean 1.

    Inverse frequency would weight the 45 `Cutbread` clips 88x a `Walk` clip, which assumes each
    tail sample carries 88x the information. They do not - tail clips of one action by one subject
    overlap heavily. Effective number saturates as n grows and is the standard correction for
    exactly this shape of long tail.
    """
    n = np.maximum(counts.astype(np.float64), 1.0)
    w = (1.0 - beta) / (1.0 - np.power(beta, n))
    return (w / w.mean()).astype(np.float32)


@torch.no_grad()
def evaluate(model, loader, stream: str, device: str, n_classes: int,
             keep_logits: bool = False) -> dict:
    """Top-1 AND mean per-class accuracy, with per-class support.

    Both, because on a 110x imbalance they answer different questions: top-1 is dominated by
    `Walk`, mean per-class is what the Toyota paper reports precisely so the tail cannot be
    ignored. A single headline number here would be a choice about which classes matter.

    `keep_logits` returns the raw test logits so post-hoc levers - logit adjustment tau,
    temperature - can be swept without retraining, which is how the Charades side found +0.05
    mean-class for free. Sweeping them by re-running training would cost 23 minutes a point.
    """
    model.eval()
    correct = np.zeros(n_classes, dtype=np.int64)
    support = np.zeros(n_classes, dtype=np.int64)
    all_logits, all_targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).unsqueeze(-1)      # [N, C, T, V, 1]
        logits = model(make_stream(x, stream, parents=PARENTS))
        pred = logits.argmax(1).cpu().numpy()
        y = y.numpy()
        if keep_logits:
            all_logits.append(logits.float().cpu().numpy())
            all_targets.append(y)
        for t, p in zip(y, pred):
            support[t] += 1
            correct[t] += int(t == p)
    seen = support > 0
    per_class = np.zeros(n_classes, dtype=np.float64)
    per_class[seen] = correct[seen] / support[seen]
    out = {
        "top1": float(correct.sum() / max(1, support.sum())),
        # Mean over classes that HAVE test clips. Averaging a zero for an absent class would
        # silently penalise the model for data it was never given.
        "mean_class": float(per_class[seen].mean()),
        "n_classes_scored": int(seen.sum()),
        "per_class": {TSM_CLASSES[i]: {"acc": round(float(per_class[i]), 4),
                                       "support": int(support[i])}
                      for i in range(n_classes) if seen[i]},
    }
    if keep_logits:
        out["logits"] = np.concatenate(all_logits)
        out["targets"] = np.concatenate(all_targets)
    return out


def coarse_metrics(logits: np.ndarray, targets: np.ndarray) -> dict:
    """The same predictions scored on the 22-class taxonomy the PRODUCT uses.

    Toyota separates `Drink.Fromcup` from `Drink.Fromcan`; `COARSE_V11` does not, because
    Agent 3's feature is `drinking_events` and no downstream number depends on the vessel. The
    31-class figure is the right one to compare against published work and the wrong one to
    describe this system's usefulness - and the gap between them is exactly the fine-grained
    confusion the skeleton cannot resolve (11 of 31 fine classes collapse into
    `cooking_food_prep` alone).

    Two collapse rules, measured rather than assumed:

      `argmax`    take the fine decision, then map it. What a served pipeline does if it reads
                  the fine argmax.
      `logsumexp` pool the fine logits belonging to each coarse class FIRST, then decide. This
                  is the principled one - four sibling classes each scoring 0.3 should beat one
                  unrelated class at 0.5, and only pooling can express that.
    """
    fine_to_coarse = np.array([coarse_id(TSM_TO_COARSE[n]) for n in TSM_CLASSES])
    n_coarse = len(COARSE_V11)
    y = fine_to_coarse[targets]

    pooled = np.full((len(logits), n_coarse), -np.inf, dtype=np.float64)
    for c in range(n_coarse):
        members = np.flatnonzero(fine_to_coarse == c)
        if members.size:
            m = logits[:, members].astype(np.float64)
            top = m.max(axis=1, keepdims=True)
            pooled[:, c] = (top + np.log(np.exp(m - top).sum(axis=1, keepdims=True)))[:, 0]

    out = {}
    for rule, pred in (("argmax", fine_to_coarse[logits.argmax(1)]),
                       ("logsumexp", pooled.argmax(1))):
        sup = np.bincount(y, minlength=n_coarse)
        cor = np.bincount(y[pred == y], minlength=n_coarse)
        seen = sup > 0
        pc = np.zeros(n_coarse)
        pc[seen] = cor[seen] / sup[seen]
        out[rule] = {
            "top1": float(cor.sum() / max(1, sup.sum())),
            "mean_class": float(pc[seen].mean()),
            "n_classes_scored": int(seen.sum()),
            "per_class": {COARSE_V11[i]: {"acc": round(float(pc[i]), 4), "support": int(sup[i])}
                          for i in range(n_coarse) if seen[i]},
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", default="/kaggle/working/runs/tsm")
    ap.add_argument("--stream", choices=STREAMS, default="bone")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--beta", type=float, default=0.9999)
    ap.add_argument("--no-balance", action="store_true", help="plain cross-entropy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", default=None, metavar="CHECKPOINT",
                    help="load best.pt, report fine + coarse metrics, dump logits, exit")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_shards(Path(args.shards))
    n_classes = len(TSM_CLASSES)

    tr = data["split"] == "train"
    te = data["split"] == "test"
    # The guarantee that makes the number quotable. Charades leaked actors between train and val
    # for months and cost macro-F1 0.152 -> 0.128 when it was fixed; asserted here rather than
    # trusted, because the shards are built by a different program.
    leak = set(data["subjects"][tr]) & set(data["subjects"][te])
    assert not leak, f"subject leak across the CS split: {sorted(leak)}"

    train_ds = ClipDataset(data["joint"][tr], data["labels"][tr], train=True, rng_seed=args.seed)
    test_ds = ClipDataset(data["joint"][te], data["labels"][te], train=False)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True,
                          num_workers=2, pin_memory=(device == "cuda"))
    test_ld = DataLoader(test_ds, batch_size=args.batch * 2, shuffle=False, num_workers=2)

    counts = np.bincount(data["labels"][tr], minlength=n_classes)
    weights = (None if args.no_balance
               else torch.from_numpy(effective_number_weights(counts, args.beta)).to(device))
    A = build_adjacency(edges=BONE_EDGES, n_nodes=len(AGCN_JOINTS), centre=ROOT_JOINT)
    model = STGCNpp(n_classes=n_classes, n_joints=len(AGCN_JOINTS), adjacency=A,
                    n_person=1, in_channels=3).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True,
                          weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(1, len(train_ld)), pct_start=0.15)
    lossf = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    print(f"device {device} | stream {args.stream} | {int(tr.sum()):,} train / "
          f"{int(te.sum()):,} test clips | {len(AGCN_JOINTS)} joints | "
          f"{sum(p.numel() for p in model.parameters()):,} params")
    print(f"class balance: {'off' if args.no_balance else f'effective-number beta={args.beta}'}"
          f" | imbalance {counts.max() / max(1, counts[counts > 0].min()):.0f}x")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = {"mean_class": -1.0}
    history = []
    t0 = time.time()

    if args.eval_only:
        # Scoring a trained checkpoint is seconds, so the coarse taxonomy and the post-hoc
        # levers do not cost another 23-minute run. The checkpoint carries the stream it was
        # trained on; honouring that beats trusting a flag, because scoring a `bone` model on
        # `joint` input produces a plausible number from the wrong tensor.
        ck = torch.load(args.eval_only, map_location=device, weights_only=False)
        if ck.get("stream") and ck["stream"] != args.stream:
            print(f"  checkpoint was trained on {ck['stream']!r}, not {args.stream!r} - using "
                  f"the checkpoint's own stream")
            args.stream = ck["stream"]
        model.load_state_dict(ck["model"])
        best = {**evaluate(model, test_ld, args.stream, device, n_classes, keep_logits=True),
                "epoch": int(ck.get("epoch", -1))}
        history = [{"epoch": best["epoch"], "loss": None,
                    "top1": round(best["top1"], 4),
                    "mean_class": round(best["mean_class"], 4)}]
        return _finish(out_dir, args, best, history, counts, data, t0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        run, seen = 0.0, 0
        for x, y in train_ld:
            x = x.to(device, non_blocking=True).unsqueeze(-1)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(make_stream(x, args.stream, parents=PARENTS)), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            run += float(loss) * len(y)
            seen += len(y)
        m = evaluate(model, test_ld, args.stream, device, n_classes)
        history.append({"epoch": epoch, "loss": round(run / max(1, seen), 4),
                        "top1": round(m["top1"], 4), "mean_class": round(m["mean_class"], 4)})
        # SELECTED ON MEAN PER-CLASS, not top-1. Top-1 would pick the epoch that best fits
        # `Walk`, which is the opposite of what an imbalanced benchmark is measuring.
        if m["mean_class"] > best["mean_class"]:
            best = {**m, "epoch": epoch}
            torch.save({"model": model.state_dict(), "stream": args.stream,
                        "n_classes": n_classes, "n_joints": len(AGCN_JOINTS),
                        "epoch": epoch, "metrics": {k: v for k, v in m.items()
                                                    if k != "per_class"}},
                       out_dir / "best.pt")
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}  loss {run / max(1, seen):.4f}  "
                  f"top1 {m['top1']:.4f}  mean-class {m['mean_class']:.4f}  "
                  f"[{time.time() - t0:.0f}s]")

    # Re-score the SELECTED weights with logits kept. Reusing the last epoch's logits would
    # describe a different model from the one saved, which is the kind of mismatch that makes a
    # recorded number unquotable.
    ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    best = {**evaluate(model, test_ld, args.stream, device, n_classes, keep_logits=True),
            "epoch": best["epoch"]}
    return _finish(out_dir, args, best, history, counts, data, t0)


def _finish(out_dir, args, best, history, counts, data, t0) -> int:
    """Print the per-class table with support beside every number, and record it to disk."""
    print(f"\n{'=' * 72}\nCS TEST - best epoch {best['epoch']} of {args.epochs}\n{'=' * 72}")
    print(f"  top-1            {best['top1']:.4f}")
    print(f"  mean per-class   {best['mean_class']:.4f}   over "
          f"{best['n_classes_scored']}/{len(TSM_CLASSES)} classes with test clips")
    print(f"\n  {'class':<24} {'acc':>7} {'test':>6} {'train':>6}   note")
    thin = []
    for name, rec in sorted(best["per_class"].items(), key=lambda kv: -kv[1]["acc"]):
        idx = TSM_CLASSES.index(name)
        # A per-class accuracy over 12 clips moves in steps of 8 points. Saying so beside the
        # number is the difference between a measurement and a decoration.
        note = "" if rec["support"] >= 30 else f"+/-{100 / rec['support']:.0f}pt granularity"
        if rec["support"] < 30:
            thin.append(name)
        print(f"  {name:<24} {rec['acc']:>7.3f} {rec['support']:>6} "
              f"{int(counts[idx]):>6}   {note}")
    if thin:
        print(f"\n  {len(thin)} class(es) have fewer than 30 test clips; their accuracies are "
              f"indicative only: {thin}")

    # THE PRODUCT'S OWN TAXONOMY. 31 fine classes collapse to 13 coarse ones, and 11 of the fine
    # classes land in `cooking_food_prep` alone - so the 31-class figure measures a distinction no
    # downstream feature uses. Agent 3 asks for `drinking_events`, not which vessel.
    coarse = None
    if "logits" in best:
        coarse = coarse_metrics(best["logits"], best["targets"])
        print(f"\n{'=' * 72}\nSAME PREDICTIONS ON THE PRODUCT'S COARSE TAXONOMY\n{'=' * 72}")
        for rule in ("argmax", "logsumexp"):
            c = coarse[rule]
            print(f"  {rule:<10} top-1 {c['top1']:.4f}   mean per-class {c['mean_class']:.4f}"
                  f"   over {c['n_classes_scored']} coarse classes")
        gain = coarse["logsumexp"]["mean_class"] - coarse["argmax"]["mean_class"]
        print(f"  pooling siblings' logits before deciding: {gain:+.4f} mean-class")
        if gain < 0:
            # MEASURED, and it contradicts the argument for pooling. LogSumExp over k comparable
            # terms is about max + log(k), so an 11-member class gets an unearned +log(11) over a
            # single-member one. Summing probabilities IS correct marginalisation - but only over
            # a calibrated posterior, and label smoothing plus class weights spread diffuse mass
            # across all eleven cooking classes on nearly every clip. Fit a temperature first
            # (scripts/rescore_tsm.py) before concluding pooling cannot work.
            print(f"  pooling LOST. LogSumExp(k terms) ~ max + log(k), so the 11-member "
                  f"cooking_food_prep gains ~{np.log(11):.1f} over a 1-member class regardless of "
                  f"evidence. Marginalising an UNCALIBRATED posterior favours large families.")
        members: dict[str, int] = Counter(TSM_TO_COARSE[f] for f in TSM_CLASSES)
        # BOTH rules per class. An earlier version tabulated only `logsumexp` - which turned out
        # to be the worse rule - and therefore under-reported every class it printed.
        print(f"\n  {'coarse class':<24} {'argmax':>7} {'lse':>7} {'test':>6}  members")
        for name, rec in sorted(coarse["argmax"]["per_class"].items(),
                                key=lambda kv: -kv[1]["acc"]):
            lse = coarse["logsumexp"]["per_class"].get(name, {}).get("acc", float("nan"))
            print(f"  {name:<24} {rec['acc']:>7.3f} {lse:>7.3f} {rec['support']:>6}  "
                  f"{members[name]:>7}")
        # The logits themselves, so tau and temperature can be swept post-hoc instead of by
        # retraining - the lever worth +0.05 mean-class on the Charades side, for free.
        np.savez_compressed(out_dir / "test_logits.npz",
                            logits=best["logits"].astype(np.float32),
                            targets=best["targets"].astype(np.int16),
                            classes=np.asarray(TSM_CLASSES))
        print(f"\n  wrote {out_dir / 'test_logits.npz'} "
              f"({best['logits'].shape[0]:,} x {best['logits'].shape[1]}) for post-hoc sweeps")

    payload = {
        "protocol": "CS", "stream": args.stream, "epochs": args.epochs,
        "clip_frames": int(data["joint"].shape[2]), "joints": int(data["joint"].shape[3]),
        "class_balance": None if args.no_balance else f"effective_number_beta_{args.beta}",
        "n_train": int((data["split"] == "train").sum()),
        "n_test": int((data["split"] == "test").sum()),
        "train_subjects": sorted({int(s) for s in data["subjects"][data["split"] == "train"]}),
        "test_subjects": sorted({int(s) for s in data["subjects"][data["split"] == "test"]}),
        "best_epoch": best["epoch"],
        "top1": round(best["top1"], 4),
        "mean_class": round(best["mean_class"], 4),
        "n_classes_scored": best["n_classes_scored"],
        "per_class": best["per_class"],
        "coarse": coarse,
        "train_counts": {TSM_CLASSES[i]: int(c) for i, c in enumerate(counts)},
        "thin_test_classes": thin,
        "shard_files": data["files"],
        "seconds": round(time.time() - t0, 1),
        "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Written because a printed table is not a recorded one - 9,515 s of GPU output survived
    # only as scrollback once in this project.
    #
    # Only files that EXIST are named. The unconditional version claimed to have written
    # `best.pt` in `--eval-only` mode, where no checkpoint is saved at all - a log that
    # overstates what reached disk is how an irreplaceable artefact gets assumed safe.
    written = [p.name for p in (out_dir / "metrics.json", out_dir / "test_logits.npz",
                                out_dir / "best.pt") if p.is_file()]
    print(f"\nwrote {out_dir}: {', '.join(written)}")
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
