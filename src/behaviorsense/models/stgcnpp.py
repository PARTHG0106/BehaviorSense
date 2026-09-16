"""ST-GCN++ for 17-joint COCO skeletons - self-contained (no mmcv / mmaction2).

Why reimplement rather than import PYSKL
----------------------------------------
PYSKL depends on mmcv, whose CUDA ops are compiled against specific torch/CUDA versions.
Our training environment is Kaggle with NO INTERNET and an sm_120 Blackwell GPU needing
torch >= 2.7 / cu128 - the exact configuration where a prebuilt mmcv wheel is least likely
to exist and cannot be built on the fly. A ~300-line dependency-free model that we control
removes an entire class of "the session died in setup" failure.

The architecture follows Duan et al., "PYSKL: Towards Good Practices for Skeleton Action
Recognition" (ACM MM 2022) - ST-GCN++ = ST-GCN with (a) learnable, non-shared adjacency
per layer, (b) residual connections throughout, (c) a multi-branch temporal module with
different dilations instead of a single 9x1 conv. Those three changes are what lift plain
ST-GCN (~89% NTU X-Sub) to ST-GCN++ (~92%) without the cost of CTR-GCN.

Input convention: [N, C, T, V, M]
  N batch, C channels (3: x, y, score), T frames, V joints (17), M persons (max 2).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# COCO-17 skeleton. Symmetric pairs are declared once and mirrored, so a typo produces a
# visibly broken graph rather than a subtly asymmetric one.
COCO_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),            # head
    (0, 5), (0, 6),                            # neck -> shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),           # arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
)
N_JOINTS = 17

# Joint remap for horizontal flip augmentation: left <-> right.
FLIP_INDEX: tuple[int, ...] = (
    0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15,
)


def build_adjacency(strategy: str = "spatial", *,
                    edges: tuple[tuple[int, int], ...] = COCO_EDGES,
                    n_nodes: int = N_JOINTS, centre: int = 0) -> np.ndarray:
    """[3, V, V] normalised adjacency: self / centripetal / centrifugal.

    The 3-partition spatial strategy (Yan et al. 2018) lets one conv learn different
    weights for "this joint", "joint closer to the body centre" and "joint further out" -
    which is what distinguishes reaching from retracting. A single averaged adjacency
    cannot express direction and measurably underperforms.

    `edges`/`n_nodes`/`centre` are parameters so a second skeleton layout can reuse this.
    Toyota's released poses are a 15-node tree whose centre is the pelvis - anatomically the
    right root, and better than COCO-17's, where the nose is used because COCO has no spine
    joint at all. The defaults keep the COCO behaviour byte-identical.
    """
    V = n_nodes
    hop = np.full((V, V), np.inf)
    np.fill_diagonal(hop, 0)
    for i, j in edges:
        hop[i, j] = hop[j, i] = 1
    # Floyd-Warshall for distance-to-centre; V<=17 makes the cubic cost irrelevant.
    for k in range(V):
        hop = np.minimum(hop, hop[:, k, None] + hop[None, k, :])

    dist = hop[:, centre]
    if not np.isfinite(dist).all():
        # A disconnected node has infinite distance to the centre, lands in no partition, and
        # then receives no message from anywhere - it trains as a constant. Cheap to check and
        # impossible to see in a loss curve.
        orphans = [i for i in range(V) if not np.isfinite(dist[i])]
        raise ValueError(
            f"joints {orphans} are not connected to joint {centre}; they would sit in no "
            f"adjacency partition and receive no messages. {len(edges)} edges over {V} nodes."
        )

    A = np.zeros((3, V, V), dtype=np.float32)
    for i in range(V):
        for j in range(V):
            if hop[i, j] > 1:
                continue
            if dist[j] == dist[i]:
                A[0, i, j] = 1.0
            elif dist[j] > dist[i]:
                A[1, i, j] = 1.0     # centrifugal: j is further from centre
            else:
                A[2, i, j] = 1.0     # centripetal

    # Symmetric normalisation D^-1/2 A D^-1/2 keeps activations scale-stable across
    # joints of differing degree (wrists have 1 neighbour, shoulders have 3).
    for k in range(3):
        deg = A[k].sum(axis=0)
        inv = np.zeros_like(deg)
        nz = deg > 0
        inv[nz] = deg[nz] ** -0.5
        A[k] = np.diag(inv) @ A[k] @ np.diag(inv)
    return A


class SpatialGCN(nn.Module):
    """Graph convolution with a learnable per-layer adjacency refinement.

    `refine` is the ST-GCN++ change: the fixed anatomical graph is a prior, not a
    constraint. Actions couple joints that share no bone (both hands during eating), and
    a purely anatomical adjacency cannot represent that. Initialised at zero so training
    starts exactly at the anatomical prior and departs only if the data pays for it.
    """

    def __init__(self, in_ch: int, out_ch: int, A: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("A", torch.from_numpy(A))
        self.n_sub = A.shape[0]
        self.refine = nn.Parameter(torch.zeros(self.n_sub, A.shape[1], A.shape[2]))
        self.conv = nn.Conv2d(in_ch, out_ch * self.n_sub, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, T, V]
        N, _, T, V = x.shape
        y = self.conv(x).view(N, self.n_sub, -1, T, V)
        A = self.A + self.refine
        out = torch.einsum("nkctv,kvw->nctw", y, A)
        return self.bn(out)


class MultiScaleTCN(nn.Module):
    """Multi-branch temporal conv: parallel dilations + max-pool, concatenated.

    Replaces ST-GCN's single 9x1 conv. Activities in our taxonomy span very different
    timescales - a fall is ~0.5 s, eating is minutes - and one kernel size forces a
    compromise. Branch count is kept at 4 so channels divide evenly.
    """

    def __init__(self, ch: int, stride: int = 1, dilations: tuple[int, ...] = (1, 2)) -> None:
        super().__init__()
        n_branch = len(dilations) + 2
        branch_ch = ch // n_branch
        rem = ch - branch_ch * n_branch

        self.branches = nn.ModuleList()
        for i, d in enumerate(dilations):
            oc = branch_ch + (rem if i == 0 else 0)
            pad = d  # kernel 3 with dilation d needs padding d to preserve length
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(ch, oc, kernel_size=(3, 1), stride=(stride, 1),
                              padding=(pad, 0), dilation=(d, 1)),
                    nn.BatchNorm2d(oc),
                )
            )
        self.branches.append(
            nn.Sequential(
                nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
                nn.Conv2d(ch, branch_ch, kernel_size=1),
                nn.BatchNorm2d(branch_ch),
            )
        )
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(ch, branch_ch, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(branch_ch),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([b(x) for b in self.branches], dim=1)


class STGCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, A: np.ndarray, stride: int = 1,
                 residual: bool = True) -> None:
        super().__init__()
        self.gcn = SpatialGCN(in_ch, out_ch, A)
        self.tcn = MultiScaleTCN(out_ch, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = None
        elif in_ch == out_ch and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = 0 if self.residual is None else self.residual(x)
        y = self.tcn(self.relu(self.gcn(x)))
        return self.relu(y + res)


class STGCNpp(nn.Module):
    """ST-GCN++ backbone + classification head.

    `base_channels=64` with 6 blocks is the "small" configuration: ~1.4M parameters.
    Deliberately not scaled up despite 96 GB of VRAM - our labelled data is the binding
    constraint, not capacity, and a larger model would overfit faster while making the
    4-stream ensemble (the actual accuracy lever) more expensive.
    """

    def __init__(
        self,
        n_classes: int = 20,
        in_channels: int = 3,
        base_channels: int = 64,
        n_person: int = 2,
        dropout: float = 0.2,
        n_joints: int = N_JOINTS,
        adjacency: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        # `adjacency` is passed in for a non-COCO layout (Toyota's released poses are a 15-node
        # tree). It must agree with `n_joints`, or `data_bn` normalises over a different number
        # of features than the graph convolution consumes - shapes that broadcast and a model
        # that trains on misaligned channels.
        A = build_adjacency() if adjacency is None else np.asarray(adjacency, dtype=np.float32)
        if A.shape != (3, n_joints, n_joints):
            raise ValueError(
                f"adjacency is {A.shape}, expected (3, {n_joints}, {n_joints}). Build it with "
                "`build_adjacency(edges=..., n_nodes=..., centre=...)` for the same layout the "
                "input tensors use."
            )
        self.n_person = n_person
        self.n_joints = n_joints
        self.data_bn = nn.BatchNorm1d(n_person * in_channels * n_joints)

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels, c1, A, residual=False),
            STGCNBlock(c1, c1, A),
            STGCNBlock(c1, c2, A, stride=2),
            STGCNBlock(c2, c2, A),
            STGCNBlock(c2, c3, A, stride=2),
            STGCNBlock(c3, c3, A),
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c3, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, T, V, M]
        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for blk in self.blocks:
            x = blk(x)

        x = self.pool(x).view(N, M, -1)
        # Mean over persons: an activity involving two people is one activity. Max-pool
        # would let the more energetic person dominate the label for both.
        x = x.mean(dim=1)
        return self.fc(self.drop(x))


def to_bone(joints: torch.Tensor, *, parents: dict[int, int] | None = None) -> torch.Tensor:
    """Joint stream -> bone stream (vector from each joint to its parent).

    `setdefault`, not assignment: the edge list closes the torso with (11, 12), and
    letting that edge overwrite gave the right hip a DIFFERENT bone semantics from the
    left (hip-to-hip instead of shoulder-to-hip). The parent map then failed to commute
    with FLIP_INDEX, so under the 50% flip augmentation the right-hip bone flipped sign
    instead of mirroring - half of all training windows encoded the torso two
    contradictory ways. Measured before the fix: flip-equivariance error 5.9 on unit
    tensors; after: 0. First-listed edge wins, and edges are listed root-outward, so
    every child's parent is its anatomical neighbour toward the nose.

    `parents` overrides that derivation with an explicit child -> parent map, which is what a
    non-COCO layout needs: the `max(i, j)` heuristic only identifies the child when the edge
    list happens to be ordered root-outward, and Toyota's 15-node tree states the direction
    outright rather than encoding it in index order.
    """
    if parents is None:
        parents = {}
        for i, j in COCO_EDGES:
            parents.setdefault(max(i, j), min(i, j))
    bone = torch.zeros_like(joints)
    for child, par in parents.items():
        bone[:, :, :, child] = joints[:, :, :, child] - joints[:, :, :, par]
    return bone


def to_motion(x: torch.Tensor) -> torch.Tensor:
    """Temporal difference stream. Last frame repeats to preserve length."""
    motion = torch.zeros_like(x)
    motion[:, :, :-1] = x[:, :, 1:] - x[:, :, :-1]
    return motion


STREAMS = ("joint", "bone", "joint_motion", "bone_motion")


def make_stream(x: torch.Tensor, stream: str, *,
                parents: dict[int, int] | None = None) -> torch.Tensor:
    """Derive one of the 4 ensemble streams from the joint input [N,C,T,V,M].

    `parents` is threaded through to `to_bone` so a non-COCO skeleton derives its bones from
    its own tree rather than from COCO's index ordering.
    """
    if stream == "joint":
        return x
    if stream == "bone":
        return to_bone(x, parents=parents)
    if stream == "joint_motion":
        return to_motion(x)
    if stream == "bone_motion":
        return to_motion(to_bone(x, parents=parents))
    raise ValueError(f"unknown stream {stream!r}; expected one of {STREAMS}")


__all__ = [
    "STGCNpp",
    "STGCNBlock",
    "SpatialGCN",
    "MultiScaleTCN",
    "build_adjacency",
    "COCO_EDGES",
    "FLIP_INDEX",
    "N_JOINTS",
    "STREAMS",
    "make_stream",
    "to_bone",
    "to_motion",
]
