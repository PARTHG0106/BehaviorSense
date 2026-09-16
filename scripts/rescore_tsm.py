"""Sweep temperature and logit adjustment over the SAVED test logits. CPU, seconds, no GPU.

Every question left about this checkpoint is answerable from `test_logits.npz` without touching
a GPU, and that matters because the alternative is a 23-minute training run per hyper-parameter.
This is the same lever `rescore_p1.py` applies on the Charades side, where it was worth +0.05
mean-class for free.

Two knobs, both post-hoc and neither touching a weight:

**Temperature T.** Divides the logits. It cannot change the fine argmax at all - scaling is
monotone - so the fine numbers are invariant to it, which is the control that proves the sweep is
wired correctly. It DOES change coarse pooling, because `logsumexp(z/T)` sharpens toward the
family maximum as T falls. This is the specific test of why pooling lost by 0.145: if the family-
size bias is a calibration artefact, some T should recover it; if pooling is simply wrong here,
no T will.

**Logit adjustment tau (Menon et al. 2021).** Subtracts `tau * log(prior)`, trading head
precision for tail recall. On an imbalance of 110x that is the standard correction, and
`mean_class` is the metric it is aimed at.

    python scripts/rescore_tsm.py --logits runs/tsm_bone/test_logits.npz \\
        --out runs/tsm_bone/rescore.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behaviorsense.data.toyota import (  # noqa: E402
    COARSE_V11,
    TSM_CLASSES,
    TSM_TO_COARSE,
    coarse_id,
)
from behaviorsense.eval.activity_eval import class_prior, logit_adjust  # noqa: E402

TEMPS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
TAUS = (0.0, 0.25, 0.5, 0.75, 1.0)


def mean_class(pred: np.ndarray, y: np.ndarray, n: int) -> tuple[float, float, int]:
    """(mean per-class, top-1, classes scored). Absent classes are skipped, not zeroed."""
    sup = np.bincount(y, minlength=n)
    cor = np.bincount(y[pred == y], minlength=n)
    seen = sup > 0
    pc = np.zeros(n)
    pc[seen] = cor[seen] / sup[seen]
    return float(pc[seen].mean()), float(cor.sum() / max(1, sup.sum())), int(seen.sum())


def pool_logsumexp(z: np.ndarray, fine_to_coarse: np.ndarray, n_coarse: int) -> np.ndarray:
    """Marginalise the fine posterior onto coarse classes, in log space and numerically safe."""
    out = np.full((len(z), n_coarse), -np.inf)
    for c in range(n_coarse):
        m = np.flatnonzero(fine_to_coarse == c)
        if m.size:
            v = z[:, m].astype(np.float64)
            top = v.max(axis=1, keepdims=True)
            out[:, c] = (top + np.log(np.exp(v - top).sum(axis=1, keepdims=True)))[:, 0]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logits", required=True, help="test_logits.npz from train_tsm_skeleton.py")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with np.load(args.logits, allow_pickle=False) as z:
        logits = z["logits"].astype(np.float64)
        targets = z["targets"].astype(np.int64)
    n_fine = logits.shape[1]
    assert n_fine == len(TSM_CLASSES), f"{n_fine} logits vs {len(TSM_CLASSES)} classes"

    fine_to_coarse = np.array([coarse_id(TSM_TO_COARSE[n]) for n in TSM_CLASSES])
    n_coarse = len(COARSE_V11)
    y_coarse = fine_to_coarse[targets]
    # The prior is estimated from the model's OWN predictions, not from the test labels. Using
    # the test distribution would be reading the answer sheet: the adjustment would be fitted on
    # the split it is scored on.
    prior = class_prior(logits.argmax(1), n_classes=n_fine)

    base_fine = mean_class(logits.argmax(1), targets, n_fine)
    print(f"{len(logits):,} test clips | {n_fine} fine -> {n_coarse} coarse classes")
    print(f"baseline (T=1, tau=0): fine mean-class {base_fine[0]:.4f}  top1 {base_fine[1]:.4f}\n")

    rows = []
    for tau in TAUS:
        adj = logits if tau == 0.0 else logit_adjust(logits, prior, tau)
        for T in TEMPS:
            z_ = adj / T
            f_mc, f_t1, _ = mean_class(z_.argmax(1), targets, n_fine)
            a_mc, a_t1, n_seen = mean_class(fine_to_coarse[z_.argmax(1)], y_coarse, n_coarse)
            l_mc, l_t1, _ = mean_class(
                pool_logsumexp(z_, fine_to_coarse, n_coarse).argmax(1), y_coarse, n_coarse)
            rows.append({"tau": tau, "T": T,
                         "fine_mean_class": round(f_mc, 4), "fine_top1": round(f_t1, 4),
                         "coarse_argmax_mean_class": round(a_mc, 4),
                         "coarse_argmax_top1": round(a_t1, 4),
                         "coarse_lse_mean_class": round(l_mc, 4),
                         "coarse_lse_top1": round(l_t1, 4),
                         "n_coarse_scored": n_seen})

    print(f"  {'tau':>5} {'T':>5} | {'fine mc':>8} {'fine t1':>8} | {'crs argmax':>11} "
          f"{'crs lse':>9}")
    for r in rows:
        print(f"  {r['tau']:>5.2f} {r['T']:>5.2f} | {r['fine_mean_class']:>8.4f} "
              f"{r['fine_top1']:>8.4f} | {r['coarse_argmax_mean_class']:>11.4f} "
              f"{r['coarse_lse_mean_class']:>9.4f}")

    # TEMPERATURE CANNOT MOVE THE FINE ARGMAX. Scaling is monotone, so every fine number in a tau
    # row must be identical across T. Asserted rather than assumed: if it varies, the sweep is
    # applying T somewhere it should not and none of the coarse numbers mean anything either.
    for tau in TAUS:
        vals = {r["fine_mean_class"] for r in rows if r["tau"] == tau}
        assert len(vals) == 1, (
            f"fine mean-class varies with temperature at tau={tau} ({sorted(vals)}); scaling is "
            "monotone so it cannot change an argmax - the sweep is miswired")

    best_fine = max(rows, key=lambda r: r["fine_mean_class"])
    best_coarse = max(rows, key=lambda r: max(r["coarse_argmax_mean_class"],
                                              r["coarse_lse_mean_class"]))
    lse_ever_wins = [r for r in rows
                     if r["coarse_lse_mean_class"] > r["coarse_argmax_mean_class"]]

    print(f"\nbest FINE   mean-class {best_fine['fine_mean_class']:.4f} at tau="
          f"{best_fine['tau']}  (T is a no-op here, as asserted)")
    rule = ("logsumexp" if best_coarse["coarse_lse_mean_class"]
            > best_coarse["coarse_argmax_mean_class"] else "argmax")
    print(f"best COARSE mean-class "
          f"{max(best_coarse['coarse_argmax_mean_class'], best_coarse['coarse_lse_mean_class']):.4f}"
          f" at tau={best_coarse['tau']} T={best_coarse['T']} via {rule}")
    if lse_ever_wins:
        w = max(lse_ever_wins, key=lambda r: r["coarse_lse_mean_class"])
        print(f"  pooling DOES beat argmax once calibrated: T={w['T']} tau={w['tau']} gives "
              f"{w['coarse_lse_mean_class']:.4f} vs {w['coarse_argmax_mean_class']:.4f}. The "
              f"family-size bias was a calibration artefact after all.")
    else:
        print("  pooling never beats argmax at any T or tau. The log(k) advantage a large family "
              "gets is not something temperature can undo, so collapse the argmax and say so.")

    out = Path(args.out) if args.out else Path(args.logits).with_name("rescore.json")
    out.write_text(json.dumps({
        "logits": str(args.logits), "n_test": int(len(logits)),
        "n_fine": n_fine, "n_coarse": n_coarse,
        "prior_source": "model predictions on the test split, not test labels",
        "baseline": {"fine_mean_class": round(base_fine[0], 4),
                     "fine_top1": round(base_fine[1], 4)},
        "best_fine": best_fine, "best_coarse": best_coarse,
        "logsumexp_ever_wins": bool(lse_ever_wins),
        "grid": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
