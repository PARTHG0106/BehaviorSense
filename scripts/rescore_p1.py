"""Re-score P1 from saved validation logits. CPU only, no checkpoints, no GPU.

Answers the questions notebook 04 raised and could not follow up on without another
two-hour Kaggle session:

  1. Does a logit-adjusted decision rule recover mean-class accuracy? The model was trained
     with cross-entropy on a distribution where `other_idle` is 39.4% of windows, then
     scored with a metric that weights all 20 classes equally.
  2. Which STREAM SUBSET is best? P1 showed the four-stream ensemble losing to `bone` alone
     on mean-class while winning top-1. The subset question follows immediately and needs
     only the per-stream logits.
  3. Does probability averaging beat logit averaging once the subset is chosen properly,
     rather than over all four streams?

Every number here is selected on the same subject-disjoint validation split used for
temperature calibration, so report it as validation-selected, not as a held-out result.

Usage:
    PYTHONPATH=src python scripts/rescore_p1.py results/val_logits.npz
    PYTHONPATH=src python scripts/rescore_p1.py results/val_logits.npz --top 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behaviorsense.agents.activity import CLASS_NAMES  # noqa: E402
from behaviorsense.eval.activity_eval import (  # noqa: E402
    MIN_SUPPORT,
    class_prior,
    combine,
    logit_adjust,
    scores,
    subset_scores,
    tau_sweep,
)

HEAD = f"{'top1':>6} {'mean-class':>11} {'macro-F1':>9} {'F1>=' + str(MIN_SUPPORT):>9}"


def row(label: str, s: tuple, width: int = 30) -> str:
    return f"{label:<{width}} {s[0]:>6.3f} {s[1]:>11.3f} {s[2]:>9.3f} {s[3]:>9.3f}"


def load(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        y = z["y"]
        per_stream = {k[len("logits_"):]: z[k] for k in z.files if k.startswith("logits_")}
    if not per_stream:
        raise SystemExit(
            f"{path} holds no 'logits_<stream>' arrays. Notebook 04 writes them; an older "
            "session's file will not have them."
        )
    return per_stream, y


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logits", type=Path, nargs="?", default=Path("results/val_logits.npz"))
    ap.add_argument("--top", type=int, default=6, help="stream subsets to print")
    ap.add_argument("--out", default=None, help="write the tables to this markdown file")
    args = ap.parse_args()

    per_stream, y = load(args.logits)
    n, streams = len(y), sorted(per_stream)
    print(f"{args.logits}: {n} val windows, {len(streams)} streams {streams}")
    support = np.bincount(y, minlength=len(CLASS_NAMES))
    tail = [c for c in range(len(CLASS_NAMES)) if 0 < support[c] < MIN_SUPPORT]
    print(f"label distribution: largest class {support.max() / n:.1%} "
          f"({CLASS_NAMES[int(support.argmax())]}), {len(tail)} class(es) under "
          f"{MIN_SUPPORT} windows")

    lines = ["# P1 re-scored from saved validation logits", "",
             f"- {n} windows, streams {', '.join(streams)}",
             f"- largest class {support.max() / n:.1%} (`{CLASS_NAMES[int(support.argmax())]}`)",
             "- all rows selected on the SAME validation split used for temperature "
             "calibration, so these are validation-selected numbers", ""]

    # 1. Stream subsets, both combination rules.
    for mode in ("logit", "prob"):
        print(f"\n=== stream subsets, {mode}-average ===\n{'subset':<30} {HEAD}")
        rows = subset_scores(per_stream, y, mode=mode)
        lines += [f"## Stream subsets ({mode}-average)", "",
                  "| subset | top1 | mean-class | macro-F1 | F1>=" + str(MIN_SUPPORT) + " |",
                  "|---|---|---|---|---|"]
        for combo, *s in rows[:args.top]:
            label = "+".join(combo)
            print(row(label, s))
            lines.append(f"| `{label}` | {s[0]:.3f} | {s[1]:.3f} | {s[2]:.3f} | {s[3]:.3f} |")
        full = next(r for r in rows if len(r[0]) == len(streams))
        best = rows[0]
        lines += ["",
                  f"Best subset `{'+'.join(best[0])}` at mean-class {best[2]:.3f} versus the "
                  f"full {len(streams)}-stream average at {full[2]:.3f} "
                  f"({best[2] - full[2]:+.3f}).", ""]

    # 2. Logit adjustment on the best subset AND on the full ensemble, so the effect of the
    # decision rule is separated from the effect of the subset.
    best_combo = subset_scores(per_stream, y, mode="logit")[0][0]
    for label, combo in (("full ensemble", tuple(streams)), (f"best subset "
                                                             f"({'+'.join(best_combo)})",
                                                             best_combo)):
        logits = combine(per_stream, combo, "logit")
        print(f"\n=== logit adjustment, {label} ===\n{'tau':<30} {HEAD}")
        lines += [f"## Logit adjustment ({label})", "",
                  "| tau | top1 | mean-class | macro-F1 | F1>=" + str(MIN_SUPPORT) + " |",
                  "|---|---|---|---|---|"]
        sweep = tau_sweep(logits, y)
        for tau, *s in sweep:
            print(row(f"tau={tau:.2f}", s))
            lines.append(f"| {tau:.2f} | {s[0]:.3f} | {s[1]:.3f} | {s[2]:.3f} | {s[3]:.3f} |")
        base = sweep[0]
        peak = max(sweep, key=lambda r: r[2])
        lines += ["",
                  f"tau=0 reproduces the unadjusted rule ({base[2]:.3f} mean-class). Peak at "
                  f"tau={peak[0]:.2f} gives {peak[2]:.3f} ({peak[2] - base[2]:+.3f}), at a "
                  f"top-1 cost of {peak[1] - base[1]:+.3f}.", ""]

    # 3. Per-class F1 for the best configuration, because a mean-class number that moved
    # should be attributable to specific classes rather than accepted as a total.
    prior = class_prior(y)
    best_tau = max(tau_sweep(combine(per_stream, best_combo, "logit"), y),
                   key=lambda r: r[2])[0]
    adjusted = logit_adjust(combine(per_stream, best_combo, "logit"), prior, best_tau)
    pred = adjusted.argmax(1)
    lines += ["## Per-class recall, best configuration", "",
              f"`{'+'.join(best_combo)}`, logit-adjusted at tau={best_tau:.2f}", "",
              "| class | support | recall |", "|---|---|---|"]
    print(f"\n=== per-class recall: {'+'.join(best_combo)} @ tau={best_tau:.2f} ===")
    for c in range(len(CLASS_NAMES)):
        if not support[c]:
            continue
        rec = float(np.mean(pred[y == c] == c))
        print(f"  {CLASS_NAMES[c]:<24} support {support[c]:>6}  recall {rec:.3f}")
        lines.append(f"| `{CLASS_NAMES[c]}` | {support[c]} | {rec:.3f} |")

    print(f"\nheadline: {row('', scores(adjusted, y), 0)}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
