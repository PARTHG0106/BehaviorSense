"""Fit and report the open-set ReID operating point from extracted embeddings.

Produces the number that replaces the guessed `reid_match_threshold` in
`PerceptionConfig`. Method:

  1. Rebuild the identical protocol split (same root/params/seed as extraction).
  2. Halve it BY IDENTITY into fit/test (two probes of one person are not independent
     samples, so splitting by probe would leak).
  3. Fit tau on the fit half: largest threshold with FAR <= budget (default 1%).
  4. Report every metric on the test half only.

An ImageNet-only embedding file can be passed as --control: it must score far worse than
the ReID-trained one on the same protocol, or the pipeline is measuring something other
than embedding quality (the real-data analogue of test R5).

Usage:
    PYTHONPATH=src python scripts/eval_reid.py \
        --root MSMT17 --embeddings results/embeddings/msmt17__msmt17w.npz \
        --control results/embeddings/msmt17__imagenetw.npz \
        --report results/reid_eval.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behaviorsense.data.reid_datasets import build_open_set_split, split_by_identity  # noqa: E402
from behaviorsense.eval.reid_eval import (  # noqa: E402
    auroc,
    closed_set_rank1,
    eer,
    fit_threshold,
    operating_point,
    score_split,
)

FAR_ROWS = (0.001, 0.01, 0.05)


def load_embeddings(npz_path: str) -> tuple[dict[str, np.ndarray], dict]:
    z = np.load(npz_path, allow_pickle=False)
    paths = [str(p) for p in z["paths"]]
    embs = z["embeddings"]
    meta = {k: z[k].item() if z[k].shape == () else z[k] for k in
            ("weights", "root", "n_enrolled", "n_impostor_ids", "seed") if k in z}
    return dict(zip(paths, embs)), meta


def evaluate(root: str, emb: dict[str, np.ndarray], meta: dict, far_budget: float):
    split = build_open_set_split(
        root,
        n_enrolled=int(meta.get("n_enrolled", 100)),
        n_impostor_ids=int(meta.get("n_impostor_ids", 400)),
        seed=int(meta.get("seed", 0)),
    )
    fit_half, test_half = split_by_identity(split, seed=0)
    fit_scores = score_split(fit_half, emb)
    test_scores = score_split(test_half, emb)

    chosen = fit_threshold(fit_scores, far_budget)
    on_test = operating_point(test_scores, chosen.threshold)
    return {
        "split": split,
        "fit_scores": fit_scores,
        "test_scores": test_scores,
        "tau": chosen.threshold,
        "fit_point": chosen,
        "test_point": on_test,
        "auroc": auroc(test_scores),
        "eer": eer(test_scores),
        "rank1": closed_set_rank1(test_scores),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--embeddings", required=True, help="ReID-trained embedding npz")
    ap.add_argument("--control", default=None, help="ImageNet-only embedding npz (negative control)")
    ap.add_argument("--far-budget", type=float, default=0.01)
    ap.add_argument("--report", default=None, help="write markdown report here")
    args = ap.parse_args()

    emb, meta = load_embeddings(args.embeddings)
    r = evaluate(args.root, emb, meta, args.far_budget)

    lines: list[str] = []
    w = lines.append
    w("# Open-set ReID evaluation — fitted operating point")
    w("")
    w(f"- protocol: `{r['split'].summary()}`")
    w(f"- embeddings: `{meta.get('weights', args.embeddings)}`")
    w(f"- fit/test split: by identity, seed 0; tau fitted on fit half at FAR <= "
      f"{args.far_budget:.1%}, all reported numbers from the disjoint test half")
    w("")
    w(f"**Fitted threshold: tau = {r['tau']:.4f}** "
      f"(fit half: FAR {r['fit_point'].far:.2%}, TAR {r['fit_point'].tar:.2%})")
    w("")
    w("## Test-half results at the fitted tau")
    w("")
    tp = r["test_point"]
    w("| metric | value |")
    w("|---|---|")
    w(f"| FAR (stranger accepted as enrolled) | **{tp.far:.2%}** |")
    w(f"| TAR (enrolled accepted) | **{tp.tar:.2%}** |")
    w(f"| DIR@1 (accepted AND correctly named) | {tp.dir_rank1:.2%} |")
    w(f"| AUROC | {r['auroc']:.4f} |")
    w(f"| EER | {r['eer'][0]:.2%} (tau={r['eer'][1]:.3f}) |")
    w(f"| closed-set rank-1 (calibration smoke test) | {r['rank1']:.2%} |")
    w(f"| n genuine / impostor probes | {tp.n_genuine} / {tp.n_impostor} |")
    w("")
    w("## Operating-point table (fit half)")
    w("")
    w("| FAR budget | tau | TAR | DIR@1 |")
    w("|---|---|---|---|")
    for far in FAR_ROWS:
        p = fit_threshold(r["fit_scores"], far)
        w(f"| {far:.1%} | {p.threshold:.4f} | {p.tar:.2%} | {p.dir_rank1:.2%} |")
    w("")

    if args.control:
        cemb, cmeta = load_embeddings(args.control)
        c = evaluate(args.root, cemb, cmeta, args.far_budget)
        w("## Negative control (ImageNet-only weights, same protocol)")
        w("")
        w("Features never trained for re-identification must do much worse, or this")
        w("evaluation is not measuring embedding quality (real-data analogue of test R5).")
        w("")
        w("| embeddings | AUROC | EER | TAR@fitted-tau | rank-1 |")
        w("|---|---|---|---|---|")
        w(f"| ReID-trained | {r['auroc']:.4f} | {r['eer'][0]:.2%} | {r['test_point'].tar:.2%} "
          f"| {r['rank1']:.2%} |")
        w(f"| ImageNet-only | {c['auroc']:.4f} | {c['eer'][0]:.2%} | {c['test_point'].tar:.2%} "
          f"| {c['rank1']:.2%} |")
        w("")
        gap = r["auroc"] - c["auroc"]
        verdict = "OK" if gap > 0.05 else "SUSPICIOUS - investigate before trusting tau"
        w(f"AUROC gap: **{gap:+.4f}** -> {verdict}")
        w("")

    w("## Config consequence")
    w("")
    w(f"`PerceptionConfig.reid_match_threshold = {r['tau']:.3f}` (was 0.30, guessed). ")
    w("Re-fit whenever the embedder checkpoint changes; this file is the procedure.")

    text = "\n".join(lines)
    print(text)
    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
