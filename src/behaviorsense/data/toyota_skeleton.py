"""Toyota's 3D skeletons -> fixed-length clips a graph network can train on.

This is the shared half of INRIA's skeleton pipeline: everything between `read_pose3d` and a
model's `forward`. Both backbones consume it - 2s-AGCN for the reproduction arm, ST-GCN++ for
the deployable comparison - so it is written once and neither owns it.

Three decisions here are load-bearing and each one has already cost this project a run somewhere.

**Temporal resize, not striding.** Trimmed clips run from ~35 to ~1,200 frames and a graph net
needs a fixed T. Integer striding is what gave notebook 07 an effective 10 Hz and 37.5% unusable
clips, and what gave serving 3-second windows against a 2-second model. Frames are resampled to
exact positions and interpolated, so a 35-frame clip and a 1,170-frame clip both arrive as T
frames covering the same *proportion* of the action.

**Missing frames are interpolated across, not zero-filled into the input.** `read_pose3d`
zero-fills so indices stay aligned to source frames - that is right for annotation lookup and
wrong for a model, because a zero skeleton is a body collapsed at the origin and reads as
motion. Here the gaps are bridged from the neighbours that do exist and the fraction bridged
travels with the clip.

**Root-centring is offered, not applied by default.** Per-frame root-centring is what drove fall
AUROC to 0.466 - it removes exactly the global translation a fall consists of. Toyota's 3D is
already near-origin and its ADL classes are position-invariant, so INRIA's pipeline centres; the
flag exists so the fall corpora can be trained without it later rather than silently ruined.
"""

from __future__ import annotations

import numpy as np

from behaviorsense.data.toyota import AGCN_JOINTS, MIN_POSE_COVERAGE

# The skeleton as a TREE, written child -> parent with the pelvis as the root. Direction is
# load-bearing, not decoration: `bone_stream` computes `child - parent`, so a pair written the
# other way round produces an inverted bone AND leaves the real child with none at all.
#
# Written as an anatomical edge LIST first, this was wrong on eight of fourteen bones - `head`,
# both wrists and the root all came out zero while `neck` was overwritten three times. The
# named joints did not prevent it; only the structural assertion below does. Every non-root
# node must appear exactly once on the left.
SKELETON_BONES: tuple[tuple[str, str], ...] = (
    ("right_ankle", "right_knee"),
    ("left_ankle", "left_knee"),
    ("right_knee", "right_hip"),
    ("left_knee", "left_hip"),
    ("right_hip", "mid_hip"),
    ("left_hip", "mid_hip"),
    ("neck", "mid_hip"),
    ("head", "neck"),
    ("right_shoulder", "neck"),
    ("left_shoulder", "neck"),
    ("right_elbow", "right_shoulder"),
    ("left_elbow", "left_shoulder"),
    ("right_wrist", "right_elbow"),
    ("left_wrist", "left_elbow"),
)

BONE_EDGES: tuple[tuple[int, int], ...] = tuple(
    (AGCN_JOINTS.index(child), AGCN_JOINTS.index(parent)) for child, parent in SKELETON_BONES
)

ROOT_JOINT = AGCN_JOINTS.index("mid_hip")
"""The pelvis. INRIA centres on it; `normalise=False` keeps absolute position instead."""

# Left/right mirror as a permutation of joint indices. Built from NAMES by swapping the
# `right_`/`left_` prefix, so it cannot drift out of step with `AGCN_JOINTS` - a hand-written
# index list is the thing that has to be re-checked every time a joint moves.
FLIP_INDEX: tuple[int, ...] = tuple(
    AGCN_JOINTS.index(
        name.replace("right_", "left_", 1) if name.startswith("right_")
        else name.replace("left_", "right_", 1) if name.startswith("left_")
        else name
    )
    for name in AGCN_JOINTS
)
assert sorted(FLIP_INDEX) == list(range(len(AGCN_JOINTS))), FLIP_INDEX
assert tuple(FLIP_INDEX[i] for i in FLIP_INDEX) == tuple(range(len(AGCN_JOINTS))), (
    "the mirror is not an involution: flipping twice must be the identity"
)

# STRUCTURAL CHECK AT IMPORT. A malformed tree cannot raise later - it trains.
_children = [c for c, _ in SKELETON_BONES]
assert len(_children) == len(set(_children)) == len(AGCN_JOINTS) - 1, (
    f"{len(_children)} bones for {len(AGCN_JOINTS)} joints, "
    f"{len(set(_children))} distinct children - every non-root joint needs exactly one parent"
)
assert AGCN_JOINTS[ROOT_JOINT] not in _children, "the root must not have a parent"
assert set(_children) == set(AGCN_JOINTS) - {AGCN_JOINTS[ROOT_JOINT]}, (
    f"unparented joints: {sorted(set(AGCN_JOINTS) - {AGCN_JOINTS[ROOT_JOINT]} - set(_children))}"
)
del _children

CLIP_FRAMES = 64
"""Frames every clip is resampled to.

A power of two above the median trimmed clip length, so most clips are downsampled rather than
stretched - upsampling a 35-frame clip to 300 (2s-AGCN's NTU default) would repeat each frame
~9x and hand the network a stack of near-duplicates. Toyota's trimmed half has a median around
100 frames at 20 fps, and 64 keeps the whole action in view at roughly a third of the temporal
resolution, which for 20-second cooking segments is ample.
"""


def bone_adjacency(n_nodes: int = len(AGCN_JOINTS)) -> np.ndarray:
    """Symmetric adjacency with self-loops, for the graph convolution.

    Self-loops matter: without them a node's own features never reach its own output and the
    first layer can only ever mix neighbours.
    """
    A = np.eye(n_nodes, dtype=np.float32)
    for i, j in BONE_EDGES:
        A[i, j] = A[j, i] = 1.0
    return A


def bridge_missing(xyz: np.ndarray, present: np.ndarray) -> tuple[np.ndarray, float]:
    """Fill undetected frames by interpolating from the detected ones. Returns (filled, share).

    A zero-filled frame is a body collapsed at the origin, and a graph network reads the jump
    into and out of it as violent motion. Linear interpolation between the nearest real frames
    is the honest reconstruction: it asserts nothing except that the person did not teleport.

    A clip with NO detections at all is returned unchanged with share 1.0 - the caller's
    coverage gate is what rejects it, not this function silently inventing a skeleton.
    """
    arr = np.asarray(xyz, dtype=np.float32).copy()
    seen = np.asarray(present, dtype=bool)
    if seen.all():
        return arr, 0.0
    if not seen.any():
        return arr, 1.0
    idx = np.arange(len(arr))
    good = idx[seen]
    for j in range(arr.shape[1]):
        for c in range(arr.shape[2]):
            arr[:, j, c] = np.interp(idx, good, arr[good, j, c])
    return arr, float((~seen).mean())


def resample_time(xyz: np.ndarray, n_out: int = CLIP_FRAMES) -> np.ndarray:
    """[T, J, C] -> [n_out, J, C] by linear interpolation over exact frame positions.

    Not striding. A stride can only produce rates that divide T, so it silently changes how much
    action a clip covers - the defect that cost notebook 07 a 12,796 s run and serving its window
    duration. Interpolation holds the PROPORTION of the action constant for every clip length.
    """
    arr = np.asarray(xyz, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected [T, J, C], got {arr.shape}")
    t = arr.shape[0]
    if t == n_out:
        return arr
    if t == 1:
        return np.repeat(arr, n_out, axis=0)
    src = np.linspace(0.0, t - 1.0, num=n_out, dtype=np.float32)
    lo = np.floor(src).astype(np.int64)
    hi = np.minimum(lo + 1, t - 1)
    w = (src - lo).astype(np.float32)[:, None, None]
    return arr[lo] * (1.0 - w) + arr[hi] * w


def bone_stream(xyz: np.ndarray) -> np.ndarray:
    """Joint stream -> bone stream: each joint minus its parent along the skeleton tree.

    The second half of "2s"-AGCN. Bones are translation-invariant by construction, which is why
    the two streams disagree usefully rather than redundantly: the joint stream carries where the
    body is, the bone stream carries what shape it is in.

    The root has no parent, so its bone is zero. Left as zero rather than dropped, because the
    graph's adjacency is indexed by node and a 14-node bone tensor would not align with it.
    """
    arr = np.asarray(xyz, dtype=np.float32)
    out = np.zeros_like(arr)
    for child, parent in BONE_EDGES:
        out[:, child] = arr[:, child] - arr[:, parent]
    return out


def centre_on_root(xyz: np.ndarray) -> np.ndarray:
    """Subtract the pelvis, per frame. INRIA's normalisation, and a documented hazard.

    This removes global translation. For Toyota's ADL classes that is a feature - `Cook` is the
    same action wherever in the kitchen it happens - and for falls it is fatal: a fall IS global
    translation, and per-frame root-centring drove held-out AUROC to 0.466 in this project's own
    measurement. Never apply it to a corpus containing falls.
    """
    arr = np.asarray(xyz, dtype=np.float32)
    return arr - arr[:, ROOT_JOINT:ROOT_JOINT + 1, :]


def build_clip(rec: dict, *, n_frames: int = CLIP_FRAMES, normalise: bool = True,
               min_coverage: float = MIN_POSE_COVERAGE) -> dict | None:
    """One `read_pose3d` record -> `{joint, bone, ...}` ready for a graph network, or None.

    None means the clip failed the coverage gate. Returned rather than raised: a corpus pass
    over 1,736 files must report how many it dropped and why, not stop at the first one.

    `joint` and `bone` are [C, T, V] float32 - channels first, as the ST-GCN family expects.
    Every clip also carries `bridged_share`, so a model trained on heavily interpolated clips
    can be identified as such after the fact instead of being assumed clean.
    """
    present = np.asarray(rec["present"], dtype=bool)
    coverage = float(present.mean()) if present.size else 0.0
    if coverage < min_coverage:
        return None

    filled, bridged = bridge_missing(rec["xyz"], present)
    resized = resample_time(filled, n_frames)
    if normalise:
        resized = centre_on_root(resized)
    joint = np.ascontiguousarray(resized.transpose(2, 0, 1))     # [C, T, V]
    bone = np.ascontiguousarray(bone_stream(resized).transpose(2, 0, 1))
    return {
        "joint": joint.astype(np.float32),
        "bone": bone.astype(np.float32),
        "coverage": round(coverage, 4),
        "bridged_share": round(bridged, 4),
        "source_frames": int(len(present)),
        "normalised": bool(normalise),
    }


def flip_lr(xyz: np.ndarray) -> np.ndarray:
    """Mirror the skeleton: swap left/right joints and negate x. [T, V, C] in and out.

    The 50% flip is the one augmentation that reliably helps skeleton action recognition, and it
    is also where the COCO path had a real bug worth not repeating: the parent map derived from
    an edge list did not commute with the joint swap, so the right-hip bone flipped SIGN instead
    of mirroring and half of all training windows encoded the torso two contradictory ways.
    Measured flip-equivariance error 5.9 on unit tensors before the fix.

    Here the tree is left/right symmetric by construction - each ankle's parent is its own knee -
    so `flip(bone(x)) == bone(flip(x))`, and the import-time assertion below proves it rather
    than assuming it.
    """
    arr = np.asarray(xyz, dtype=np.float32)[:, FLIP_INDEX, :].copy()
    arr[..., 0] *= -1.0
    return arr


# FLIP-EQUIVARIANCE, CHECKED AT IMPORT. If the bone tree and the mirror disagree, the augmented
# half of the corpus contradicts the other half and nothing raises.
_probe = np.arange(len(AGCN_JOINTS) * 3 * 4, dtype=np.float32).reshape(4, len(AGCN_JOINTS), 3)
assert np.allclose(bone_stream(flip_lr(_probe)), flip_lr(bone_stream(_probe)), atol=1e-4), (
    "the bone tree does not commute with the left/right mirror: under the flip augmentation "
    "some bones would change sign instead of mirroring, and half the training set would encode "
    "the skeleton two contradictory ways"
)
del _probe


__all__ = [
    "BONE_EDGES",
    "CLIP_FRAMES",
    "FLIP_INDEX",
    "ROOT_JOINT",
    "SKELETON_BONES",
    "bone_adjacency",
    "bone_stream",
    "bridge_missing",
    "build_clip",
    "centre_on_root",
    "flip_lr",
    "resample_time",
]
