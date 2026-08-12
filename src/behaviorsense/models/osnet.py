"""OSNet-AIN loading and embedding extraction.

The architecture itself is vendored verbatim from deep-person-reid (MIT licence,
Kaiyang Zhou) in `_osnet_ain_vendored.py` - vendoring rather than depending on torchreid
means the offline Kaggle environment needs no mmcv/torchreid install, and the checkpoint
keys are guaranteed to match (verified 552/552 on both published checkpoints).

Weights provenance (matters for what each eval is allowed to claim):
  - `osnet_ain_x1_0_msmt17.pth`   trained on MSMT17 bounding_box_train (1,041 ids).
    Those ids are disjoint from the 3,060 test ids our open-set protocol draws from, so
    MSMT17 evaluation is identity-disjoint but IN-domain; Market-1501 evaluation with
    this checkpoint is fully CROSS-domain. Chosen over the stronger multi-source DG
    checkpoints because those were trained with DukeMTMC-reID, which this project
    excluded on ethics grounds - using its weights through the back door would make that
    exclusion cosmetic.
  - `osnet_ain_x1_0_imagenet.pth` ImageNet-only init: the NEGATIVE CONTROL. Features
    never trained for re-identification must score visibly worse on the same protocol,
    or the evaluation pipeline is not measuring embedding quality at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from behaviorsense.models._osnet_ain_vendored import osnet_ain_x1_0

# Standard ReID preprocessing (torchreid FeatureExtractor defaults). Height x width is
# 256x128 - person crops are portrait; feeding square inputs silently squashes people.
INPUT_HW = (256, 128)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

EMBED_DIM = 512


def load_osnet_ain(weights: str | Path, device: str = "cpu") -> nn.Module:
    """Load OSNet-AIN x1.0 for feature extraction, refusing a silent partial load.

    The classifier head is dropped (identity count differs per training set and is
    irrelevant for embeddings). Every remaining key must load - `strict=False` with no
    accounting is how a randomly-initialised backbone masquerades as a pretrained one
    and quietly produces near-random embeddings.
    """
    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith("classifier.")}

    model = osnet_ain_x1_0(num_classes=1, pretrained=False)
    expected = {k for k in model.state_dict() if not k.startswith("classifier.")}
    missing = expected - set(sd)
    if missing:
        raise RuntimeError(
            f"{weights}: {len(missing)} backbone keys absent (e.g. {sorted(missing)[:3]}); "
            "refusing a partial load"
        )
    model.load_state_dict(sd, strict=False)  # only classifier.* is absent, by construction
    model.eval()
    return model.to(device)


def _preprocess(img: "np.ndarray") -> np.ndarray:
    """HWC uint8 RGB -> CHW float32 normalised."""
    x = img.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def embed_image_files(
    model: nn.Module,
    paths: Sequence[str | Path],
    batch_size: int = 64,
    device: str = "cpu",
    log_every: int = 0,
) -> np.ndarray:
    """Embed image files -> [N, 512] float32, L2-normalised.

    L2-normalising here (not at comparison time) keeps every consumer - gallery
    centroids, cosine distances, the npz on disk - working in the same space.
    """
    from PIL import Image  # noqa: PLC0415 - keep PIL optional for non-extraction use

    out = np.empty((len(paths), EMBED_DIM), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            arrs = []
            for p in chunk:
                with Image.open(p) as im:
                    im = im.convert("RGB").resize(
                        (INPUT_HW[1], INPUT_HW[0]), Image.BILINEAR
                    )
                    arrs.append(_preprocess(np.asarray(im)))
            batch = torch.from_numpy(np.stack(arrs)).to(device)
            feats = model(batch).cpu().numpy().astype(np.float32)
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            out[start : start + len(chunk)] = feats / np.maximum(norms, 1e-12)
            if log_every and (start // batch_size) % log_every == 0:
                print(f"  embedded {min(start + batch_size, len(paths))}/{len(paths)}")
    return out


class OSNetEmbedder:
    """Live `ReIDEmbedder` (see agents/perception.py) backed by OSNet-AIN.

    Crops the person box out of the frame with a small context margin - tight boxes
    amputate heads and feet, which carry gait/build signal.
    """

    def __init__(self, weights: str | Path, device: str = "cpu", margin: float = 0.05):
        self.model = load_osnet_ain(weights, device=device)
        self.device = device
        self.margin = margin

    @property
    def dim(self) -> int:
        return EMBED_DIM

    def embed(self, frame: np.ndarray, box) -> np.ndarray:  # box: schemas.BoundingBox
        from PIL import Image  # noqa: PLC0415

        h, w = frame.shape[:2]
        mx = (box.x2 - box.x1) * self.margin
        my = (box.y2 - box.y1) * self.margin
        x1 = max(0, int(box.x1 - mx))
        y1 = max(0, int(box.y1 - my))
        x2 = min(w, int(box.x2 + mx))
        y2 = min(h, int(box.y2 + my))
        if x2 - x1 < 2 or y2 - y1 < 2:
            raise ValueError("degenerate crop")
        crop = Image.fromarray(frame[y1:y2, x1:x2]).convert("RGB")
        crop = crop.resize((INPUT_HW[1], INPUT_HW[0]), Image.BILINEAR)
        with torch.no_grad():
            batch = torch.from_numpy(_preprocess(np.asarray(crop))[None]).to(self.device)
            feat = self.model(batch)[0].cpu().numpy().astype(np.float32)
        return feat / max(float(np.linalg.norm(feat)), 1e-12)


__all__ = ["load_osnet_ain", "embed_image_files", "OSNetEmbedder", "EMBED_DIM", "INPUT_HW"]
