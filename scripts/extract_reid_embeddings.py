"""Extract OSNet-AIN embeddings for every crop in the open-set ReID protocol.

Deterministic contract with eval_reid.py: both scripts build the protocol with the SAME
(root, n_enrolled, n_impostor_ids, seed), so the crop list is identical and the npz can
be keyed by path. Only protocol crops are embedded (~3k), not the whole dataset (~126k)
- a 40x saving that also keeps derived data minimal (licence: derived files stay private).

Usage:
    PYTHONPATH=src python scripts/extract_reid_embeddings.py \
        --root MSMT17 --weights weights/osnet_ain_x1_0_msmt17.pth \
        --out results/embeddings/msmt17__msmt17w.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behaviorsense.data.reid_datasets import build_open_set_split  # noqa: E402
from behaviorsense.models.osnet import embed_image_files, load_osnet_ain  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="dataset root (Market-1501/MSMT17 layout)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--n-enrolled", type=int, default=100)
    ap.add_argument("--n-impostor-ids", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    split = build_open_set_split(
        args.root,
        n_enrolled=args.n_enrolled,
        n_impostor_ids=args.n_impostor_ids,
        seed=args.seed,
    )
    print(split.summary())

    crops = split.all_crops()
    paths = sorted({c.path for c in crops})
    print(f"embedding {len(paths)} unique crops with {args.weights} on {args.device}")

    model = load_osnet_ain(args.weights, device=args.device)
    t0 = time.time()
    embs = embed_image_files(
        model, paths, batch_size=args.batch_size, device=args.device, log_every=5
    )
    dt = time.time() - t0
    print(f"done in {dt:.0f}s ({len(paths) / dt:.1f} img/s)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        paths=np.array(paths),
        embeddings=embs,
        weights=str(args.weights),
        root=str(args.root),
        n_enrolled=args.n_enrolled,
        n_impostor_ids=args.n_impostor_ids,
        seed=args.seed,
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
