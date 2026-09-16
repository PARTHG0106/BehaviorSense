"""The bridge from trained checkpoints to Agent 2's runtime.

Without this module the project has a hole in the middle: `train_adl.py` writes
checkpoints, `ActivityAgent` consumes logits, and nothing connects them. `WindowClassifier`
was a Protocol with no implementation, so the trained models could not actually be used.

Two implementations:

  `EnsembleClassifier`  - loads N per-stream checkpoints and averages their logits.
  `SyntheticClassifier` - a deterministic stand-in for CPU testing and the demo.

Why averaging happens in LOGIT space
------------------------------------
The 4-stream ensemble (joint / bone / joint-motion / bone-motion) is the single biggest
accuracy lever available to us, and *where* the streams are combined changes what the
combination means:

  - averaging probabilities is a mixture: one stream that is confidently wrong drags the
    result toward its answer,
  - averaging logits is a product-of-experts after the softmax: a stream that assigns a
    class near-zero mass can veto it.

Skeleton streams are complementary views of one skeleton rather than independent voters -
bone geometry and joint motion disagree about *different* things - so veto behaviour is
what we want, and logit averaging is what PYSKL and the GCN literature report. The
alternative is implemented behind a flag so the ablation is one argument, not a rewrite.

Calibration ordering matters and is easy to get backwards
---------------------------------------------------------
Temperature calibration belongs AFTER ensembling, not per-stream, because the ensemble's
confidence distribution is not any single member's. `ActivityAgent.calibrate()` applies it
downstream of `logits()`, so this class deliberately returns *raw* averaged logits and
does no scaling of its own. Two temperature applications would silently double-soften
every posterior and quietly break abstention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from behaviorsense.agents.activity import N_CLASSES

_RAW_PIXEL_SCALE = 25.0
"""Above this peak |x|,|y| a window is raw pixels, not torso units.

`normalise()` divides by torso length, so a joint sits within ~2 torso lengths of the hip and a
walking person's retained root motion adds a few more across a 30-frame window - call it 10 at the
extreme. A 640x480 frame peaks near 300-700 (LCR-Net and RTMO both extrapolate past the frame
edge). The gap between 10 and 300 is what makes 25 a safe line rather than a tuned one."""
from behaviorsense.models.stgcnpp import FLIP_INDEX, STREAMS, STGCNpp, make_stream

CombineMode = Literal["logit", "prob"]


def windows_to_tensor(windows: np.ndarray):
    """[N, T, M, 17, 3] (dataset layout) -> [N, 3, T, 17, M] (model layout).

    The permutation is where a silent bug would live: transposing M and V produces a
    tensor of exactly the right shape that is complete nonsense, and the model would train
    or infer on it without complaint. Asserted in test E1 against a hand-built case.
    """
    import torch

    x = np.asarray(windows, dtype=np.float32)
    if x.ndim != 5 or x.shape[-1] != 3 or x.shape[-2] != 17:
        raise ValueError(
            f"expected [N,T,M,17,3], got {x.shape}. Windows come from the shard layout; "
            "the model wants [N,C,T,V,M] and this function is the only place that "
            "conversion should happen."
        )
    # [N,T,M,V,C] -> [N,C,T,V,M]
    return torch.from_numpy(x).permute(0, 4, 1, 3, 2).contiguous()


class SyntheticClassifier:
    """Deterministic pseudo-classifier for CPU tests and the offline demo.

    It does NOT pretend to be accurate: it derives logits from two crude geometric cues
    (vertical hip travel and overall motion energy) so that a falling window scores toward
    `falling` and a still window toward `sitting`. That makes the demo and the service
    exercisable end-to-end without a checkpoint, while being obviously not a trained model
    to anyone reading it - which matters, because a stub that looked accurate would invite
    someone to quote its numbers.
    """

    def __init__(self, seed: int = 0, noise: float = 0.6) -> None:
        self.rng = np.random.default_rng(seed)
        self.noise = noise

    @property
    def name(self) -> str:
        return "synthetic-geometric"

    def logits(self, windows: np.ndarray) -> np.ndarray:
        x = np.asarray(windows, dtype=np.float32)
        n = len(x)
        out = self.rng.normal(0.0, self.noise, size=(n, N_CLASSES)).astype(np.float32)
        if n == 0:
            return out

        hips = (x[:, :, 0, 11, 1] + x[:, :, 0, 12, 1]) / 2.0     # [N, T]
        drop = hips[:, 0] - hips[:, -1]
        coords = x[:, :, 0, :, :2]
        energy = np.abs(np.diff(coords, axis=1)).mean(axis=(1, 2, 3))

        for i in range(n):
            if drop[i] > 0.5:
                out[i, 7] += 4.0 + drop[i]        # falling
                out[i, 8] += 2.0                  # fallen_on_ground
            elif energy[i] > 0.02:
                out[i, 0] += 3.0                  # walking
            else:
                out[i, 2] += 3.0                  # sitting
        return out


def flip_windows(windows: np.ndarray) -> np.ndarray:
    """Mirror [N, T, M, 17, 3] windows: negate x AND remap left/right joints.

    Both halves are required. Negating x alone turns a left-handed reach into a
    right-handed one performed by a body whose left wrist is on its right side - a pose no
    human can adopt - and the model was never trained on it. `FLIP_INDEX` is the same
    permutation `augment()` uses, so a test-time mirror is a transform the model has
    genuinely seen during training rather than an out-of-distribution input.

    Scores (channel 2) are confidences, not coordinates, so they are permuted with their
    joints and never negated.
    """
    x = np.asarray(windows, dtype=np.float32).copy()
    if x.ndim != 5 or x.shape[-1] != 3 or x.shape[-2] != 17:
        raise ValueError(f"expected [N,T,M,17,3], got {x.shape}")
    x[..., 0] = -x[..., 0]
    return x[:, :, :, list(FLIP_INDEX), :]


class EnsembleClassifier:
    """Multi-stream ST-GCN++ ensemble satisfying `WindowClassifier`.

    Usage::

        clf = EnsembleClassifier.from_run_dir("runs")        # finds adl_<stream>/best.pt
        clf = EnsembleClassifier({"joint": "runs/adl_joint/best.pt"})
        logits = clf.logits(windows)                          # [N, 20]
    """

    def __init__(
        self,
        checkpoints: dict[str, str | Path],
        device: str = "cpu",
        combine: CombineMode = "logit",
        batch_size: int = 64,
        use_ema: bool = True,
        n_classes: int | None = None,
        tta: bool = False,
    ) -> None:
        import torch

        if not checkpoints:
            raise ValueError("no checkpoints given; an empty ensemble cannot classify")
        unknown = set(checkpoints) - set(STREAMS)
        if unknown:
            raise ValueError(f"unknown stream(s) {sorted(unknown)}; expected {STREAMS}")

        self.device = device
        self.combine = combine
        self.batch_size = batch_size
        self.tta = tta
        """Average each window's logits with its mirror image. Doubles inference cost and
        nothing else - no retraining, no extra checkpoints - and the mirror is a transform
        the model was trained on (`AugmentConfig.flip_prob = 0.5`), so it is averaging two
        legitimate views rather than smearing in an out-of-distribution one."""
        self.streams = sorted(checkpoints, key=STREAMS.index)
        self.models: dict[str, STGCNpp] = {}
        self.loaded_from: dict[str, str] = {}
        self._head: int | None = None

        for stream in self.streams:
            path = Path(checkpoints[stream])
            ck = torch.load(str(path), map_location="cpu", weights_only=False)

            # A checkpoint trained on a DIFFERENT stream loads without error and produces
            # confident garbage, because the architecture is stream-agnostic. The trained
            # stream is recorded in the checkpoint args, so verify rather than trust the
            # filename.
            trained_stream = ck.get("args", {}).get("stream")
            if trained_stream is not None and trained_stream != stream:
                raise ValueError(
                    f"{path} was trained on stream {trained_stream!r} but is being loaded "
                    f"as {stream!r}. The architecture is stream-agnostic so this would "
                    "load silently and produce confident nonsense."
                )

            # THE HEAD SIZE IS A PROPERTY OF THE CHECKPOINT, read from it rather than assumed.
            # Charades checkpoints have 20 outputs; the Toyota RTMO run has 22, because
            # `COARSE_V11` appends `using_device` and `object_interaction`. A hard-coded 20 does
            # crash rather than mislead - `load_state_dict` raises on a shape mismatch even with
            # `strict=False`, verified - but crashing three minutes into a served demo for a fact
            # the file already records is a poor trade.
            trained_n = ck.get("args", {}).get("n_classes")
            head = n_classes if n_classes is not None else (trained_n or N_CLASSES)
            if trained_n is not None and trained_n != head:
                raise ValueError(
                    f"{path} was trained with a {trained_n}-class head but is being loaded as "
                    f"{head}. Pass n_classes={trained_n}, or omit it and let the checkpoint "
                    "decide - a head size mismatch is not something to override by accident."
                )
            if self._head is not None and self._head != head:
                raise ValueError(
                    f"ensemble mixes head sizes: {self._head} and {head}. Averaging logits "
                    "across different class spaces adds up unrelated activities."
                )
            self._head = head
            state = ck.get("ema") if (use_ema and ck.get("ema")) else ck.get("model", ck)
            model = STGCNpp(n_classes=head)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                raise RuntimeError(
                    f"{path}: {len(missing)} weights missing (e.g. {list(missing)[:3]}); "
                    "refusing a partial load - a partly-random backbone is worse than no "
                    "model because it still returns confident answers"
                )
            model.eval().to(device)
            self.models[stream] = model
            self.loaded_from[stream] = str(path)
        self.n_classes = self._head or N_CLASSES

    @classmethod
    def from_run_dir(
        cls, root: str | Path, prefix: str = "adl_", checkpoint: str = "best.pt", **kw
    ) -> EnsembleClassifier:
        """Discover `<root>/<prefix><stream>/<checkpoint>` for every available stream."""
        root = Path(root)
        found: dict[str, Path] = {}
        for stream in STREAMS:
            p = root / f"{prefix}{stream}" / checkpoint
            if p.is_file():
                found[stream] = p
        if not found:
            raise FileNotFoundError(
                f"no checkpoints under {root} matching {prefix}<stream>/{checkpoint}. "
                f"Train at least one stream first (scripts/train_adl.py)."
            )
        return cls(found, **kw)

    @property
    def name(self) -> str:
        tta = "+tta" if self.tta else ""
        mode = "logit" if self.combine == "logit" else "prob (mixture)"
        return f"stgcnpp-ensemble[{'+'.join(self.streams)}]/{mode}{tta}"

    def logits(self, windows: np.ndarray) -> np.ndarray:
        """[N, T, M, 17, 3] -> [N, n_classes] raw averaged logits (uncalibrated).

        Windows must already be `normalise()`d - root-centred and divided by torso length, so
        coordinates are dimensionless and order +/-2. That is the only representation the
        checkpoints have ever seen, because `SkeletonWindowDataset.__getitem__` normalises every
        training window.

        The scale is CHECKED here rather than trusted, because the convention was "the caller
        normalises" and one of two callers did not: notebook 04's evaluation cell called
        `normalise()` explicitly while `ActivityPipeline` handed over raw 640x480 pixels. A ~150x
        scale error saturates the network into one constant class, and the demo emitted a single
        29.3 s segment of the same label on every clip - a confident answer, so nothing raised.
        """
        arr = np.asarray(windows, dtype=np.float32)
        if arr.size:
            # Ignore the exact zeros normalise() leaves where RTMO saw no joint.
            mag = np.abs(arr[..., :2])
            peak = float(mag[mag > 0].max()) if (mag > 0).any() else 0.0
            if peak > _RAW_PIXEL_SCALE:
                raise ValueError(
                    f"windows peak at {peak:.0f} in x/y, which is pixel scale, not the torso "
                    f"units the checkpoints were trained on (normalised windows peak below "
                    f"~{_RAW_PIXEL_SCALE:g}). Call `normalise()` before classifying - the network "
                    "does not fail on out-of-range input, it saturates and returns one confident "
                    "class for everything."
                )
        raw = self._logits(windows)
        if not self.tta:
            return raw
        # Average in LOGIT space regardless of self.combine: the two views are the same
        # model on the same window, not two experts, so there is no mixture to form.
        return ((raw + self._logits(flip_windows(windows))) / 2.0).astype(np.float32)

    def _logits(self, windows: np.ndarray) -> np.ndarray:
        import torch

        x = np.asarray(windows)
        if len(x) == 0:
            return np.zeros((0, N_CLASSES), dtype=np.float32)

        tensor = windows_to_tensor(x)
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(tensor), self.batch_size):
                batch = tensor[start:start + self.batch_size].to(self.device)
                per_stream = []
                for stream in self.streams:
                    out = self.models[stream](make_stream(batch, stream)).float()
                    per_stream.append(
                        torch.log_softmax(out, dim=-1) if self.combine == "prob" else out
                    )
                stacked = torch.stack(per_stream)
                if self.combine == "prob":
                    # Mixture: mean of probabilities, returned as log-probs so the
                    # downstream softmax recovers them unchanged.
                    merged = torch.log(stacked.exp().mean(dim=0).clamp_min(1e-12))
                else:
                    merged = stacked.mean(dim=0)
                chunks.append(merged.cpu().numpy())
        return np.concatenate(chunks).astype(np.float32)

    def per_stream_logits(self, windows: np.ndarray) -> dict[str, np.ndarray]:
        """Individual stream outputs - needed for the per-stream ablation table."""
        import torch

        tensor = windows_to_tensor(windows)
        out: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for stream in self.streams:
                parts = []
                for start in range(0, len(tensor), self.batch_size):
                    batch = tensor[start:start + self.batch_size].to(self.device)
                    parts.append(
                        self.models[stream](make_stream(batch, stream)).float().cpu().numpy()
                    )
                out[stream] = np.concatenate(parts).astype(np.float32)
        return out


__all__ = [
    "torch_supports_device",
    "EnsembleClassifier",
    "SyntheticClassifier",
    "flip_windows",
    "windows_to_tensor",
    "CombineMode",
]


def torch_supports_device(index: int = 0) -> tuple[bool, str]:
    """Does the INSTALLED PyTorch have kernels for this GPU? `(ok, reason)`.

    `torch.cuda.is_available()` answers "is there a driver and a device", not "was this wheel
    compiled for it". On a P100 (sm_60) against a build targeting sm_70+, `is_available()` is True,
    the model moves to the device without complaint, and the first kernel launch fails with:

        CUDA error: no kernel image is available for execution on the device

    which arrives as an AcceleratorError from whichever op happened to run first - so it reads as
    a bug in that op rather than as an unsupported card. Observed exactly that on Kaggle's P100
    while RTMO was fine, because onnxruntime carries its own kernels and PTX.

    `get_arch_list()` is the wheel's own answer, so this is a fact check rather than a guess.
    """
    import torch

    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"
    major, minor = torch.cuda.get_device_capability(index)
    sm = f"sm_{major}{minor}"
    archs = list(torch.cuda.get_arch_list())
    if not archs:
        return False, "this torch build lists no CUDA architectures (CPU-only wheel)"
    if sm in archs:
        return True, f"{torch.cuda.get_device_name(index)} is {sm}, built for {archs}"
    # PTX lets a newer card run older code; it never lets an OLDER card run newer code, so a
    # compute_XX entry only helps when the device is at or above XX.
    ptx = [a for a in archs if a.startswith("compute_")]
    for p in ptx:
        try:
            if int(p.split("_")[1]) <= major * 10 + minor:
                return True, f"{sm} via PTX from {p}"
        except ValueError:                                          # noqa: PERF203
            continue
    return False, (
        f"{torch.cuda.get_device_name(index)} is {sm} but this torch was built for "
        f"{archs} - no kernel image exists for it"
    )
