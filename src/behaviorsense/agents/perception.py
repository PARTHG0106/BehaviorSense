"""Agent 1 - Perception: detect -> track -> re-identify -> assign role.

Architecture (locked in docs/02_architecture.md):

    frame -> RTMO-l (detect + pose, one pass) -> BoT-SORT (track_id)
                                                    |
                                             OSNet-AIN embedding
                                                    |
                                          open-set gallery match
                                                    |
                             role: RESIDENT | VISITOR | CARER | UNKNOWN

Why this module is structured around interfaces rather than concrete models
--------------------------------------------------------------------------
The detector, embedder and object detector are `Protocol`s with deterministic stub
implementations. That is not architectural decoration - it is what makes the identity
layer testable at all. The interesting logic here is *not* the neural inference; it is:

  - open-set rejection      (when to say UNKNOWN rather than guess)
  - gallery centroid update (how enrolment drifts without being poisoned)
  - track-level voting      (a role belongs to a track, not to a frame)
  - hysteresis              (a role must not flicker frame-to-frame)

All of that is pure logic, all of it is where the bugs live, and none of it needs a GPU.
The 96GB Blackwell session is for training weights; correctness of the identity layer is
established here, offline, in milliseconds. Swapping the stub for real RTMO changes one
constructor argument.

The open-set decision is the part most student projects get wrong
----------------------------------------------------------------
Standard ReID is closed-set: it returns the nearest gallery identity, always. A home
monitor that always names the nearest match will confidently label a stranger as the
resident. We threshold cosine distance and emit UNKNOWN, and `PersonObservation` has a
validator that rejects UNKNOWN paired with high confidence so the contradiction cannot
even be represented.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

import numpy as np

from behaviorsense.schemas import (
    BoundingBox,
    DetectedObject,
    FrameObservation,
    Keypoints,
    PersonObservation,
    Role,
)

PERCEPTION_VERSION = "1.0.0"

# COCO-17 indices used by the geometric fall cue and the usability gate.
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_HIP, _R_HIP = 11, 12
_L_ANKLE, _R_ANKLE = 15, 16


# ---------------------------------------------------------------------------
# Model interfaces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawDetection:
    """One person detected in one frame, before tracking assigns an identity."""

    box: BoundingBox
    confidence: float
    keypoints: Keypoints | None = None


@runtime_checkable
class PoseDetector(Protocol):
    """Detection + pose in one pass. Implemented by RTMO in deployment."""

    def detect(self, frame: np.ndarray) -> list[RawDetection]: ...


@runtime_checkable
class ReIDEmbedder(Protocol):
    """Appearance embedding for a person crop. Implemented by OSNet-AIN."""

    @property
    def dim(self) -> int: ...

    def embed(self, frame: np.ndarray, box: BoundingBox) -> np.ndarray: ...


@runtime_checkable
class ObjectDetector(Protocol):
    """Object context at ~1 Hz. Implemented by RT-DETR."""

    def detect(self, frame: np.ndarray) -> list[DetectedObject]: ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerceptionConfig:
    """Thresholds for Agent 1. Justified inline; tunable without code edits."""

    det_confidence: float = 0.35
    """Score required to START a new track ("high" detections in ByteTrack terms)."""

    det_low_confidence: float = 0.10
    """Score required to CONTINUE an existing track.

    This two-threshold split is the core of ByteTrack/BoT-SORT and the reason a single
    `det_confidence` is wrong. A partially-occluded person scores ~0.2; discarding them
    kills the track, and the resident then re-enters as a new track and a new identity,
    fragmenting their day. Allowing weak detections to *extend* an established track
    while refusing to let them *create* one gets occlusion robustness without inviting
    every flicker to become a tracked "person"."""

    min_visible_keypoints: int = 8
    """Below this, pose is too occluded for reliable action recognition. Enforced by
    Keypoints.is_usable; recorded here so the gate is configurable in one place."""

    # -- tracking --------------------------------------------------------------
    max_age_frames: int = 30
    """Frames a track survives without a detection before deletion. At 15 fps this is
    2 seconds - long enough to cross behind furniture, short enough that an ID is not
    handed to a different person who walks through the same spot."""

    min_hits: int = 3
    """Detections before a track is confirmed. Suppresses single-frame false positives
    from becoming tracked "people"."""

    iou_threshold: float = 0.3
    """IoU floor for greedy association. Deliberately loose: indoor motion between frames
    is small, and a tight threshold breaks tracks during fast movement (i.e. falls)."""

    # -- open-set ReID ---------------------------------------------------------
    reid_match_threshold: float = 0.355
    """Cosine distance below which a track matches a gallery identity.

    FITTED, not guessed. Procedure and full table: results/reid_eval.md, reproducible via
    `scripts/eval_reid.py`. Protocol: 100 enrolled identities (4 crops each, single
    camera - deployment-realistic enrolment), 400 unseen impostor identities, probes from
    other cameras only; tau fitted on one identity-disjoint half at a FAR <= 1% budget and
    reported on the other.

        FAR budget    tau      TAR       (fit half, MSMT17, OSNet-AIN)
          0.1%      0.320    97.24%
          1.0%      0.355    99.08%   <- adopted
          5.0%      0.391    99.77%

    Held-out half at tau=0.355: FAR 2.25%, TAR 96.86%, DIR@1 96.65%, AUROC 0.9978.
    The FAR budget is the stated risk tolerance and transfers across deployments;
    "threshold maximising accuracy" would not, since accuracy depends on the
    genuine:impostor ratio, a property of how the split was built.

    Why 1% and not 0.1%: at 0.1% the threshold rejects ~3% of genuine observations,
    which fragments a resident's day across identities and quietly biases every
    duration feature downward. A 2% stranger-acceptance rate on a *per-observation*
    basis is tolerable because role assignment votes over a window of 15 frames
    (`role_vote_window`) - an isolated false accept cannot flip a track's role.

    Sanity check that this measures anything at all: ImageNet-only weights on the same
    protocol score AUROC 0.534 vs 0.998, TAR 3.35% vs 96.86%.

    MUST be re-fitted when the embedder checkpoint changes."""

    reid_margin: float = 0.05
    """Required gap between best and second-best gallery distance.

    A match that is nearly tied between two enrolled people is not a match. Without this,
    two residents with similar clothing swap identities intermittently - and because the
    behaviour layer aggregates per-role, a swap silently merges two people's daily
    features into one."""

    gallery_momentum: float = 0.9
    """EMA weight for updating an enrolled centroid with a new confident observation.

    Enrolment must adapt (clothing changes through the day) without being poisoned. High
    momentum means a single mis-assigned crop moves the centroid by 10%, which decays
    away; low momentum would let one error redefine the identity."""

    min_embed_box_area: float = 1200.0
    """Crops smaller than this yield embeddings dominated by upsampling artefacts. A
    distant figure produces a low-information vector that lands near the gallery mean and
    matches everything; skip it rather than trust it."""

    # -- role assignment -------------------------------------------------------
    role_vote_window: int = 15
    """Frames of per-track role history used for the majority vote. A role is a property
    of a track, not of a frame: one bad crop must not relabel a person."""

    role_min_votes: int = 5
    """Votes required before a track leaves UNKNOWN. Below this the evidence is too thin
    and reporting a confident identity would be unjustified."""

    role_switch_margin: int = 3
    """Extra votes a challenger role needs to displace the incumbent (hysteresis).

    Without this, a track sitting near the decision boundary alternates
    RESIDENT/VISITOR every few frames, and the activity segmenter downstream attributes
    alternating slices of one continuous action to two different people."""

    object_sample_hz: float = 1.0
    """Object detection rate. Home scenes are near-static, so 1 Hz loses almost nothing
    and cuts object-detection cost ~30x (see DetectedObject in schemas.py)."""


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection over union of two boxes."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / (a.area + b.area - inter)


@dataclass
class Track:
    """State for one tracked person."""

    track_id: int
    box: BoundingBox
    confidence: float
    keypoints: Keypoints | None = None
    hits: int = 1
    age: int = 0
    """Frames since the last successful detection match."""
    frames_seen: int = 1
    embedding: np.ndarray | None = None
    role: Role = Role.UNKNOWN
    role_confidence: float = 0.0
    reid_distance: float | None = None
    role_votes: deque[Role] = field(default_factory=deque)
    velocity: tuple[float, float] = (0.0, 0.0)
    min_hits: int = 3
    """Copied from config at creation so `confirmed` cannot silently ignore the setting."""

    @property
    def confirmed(self) -> bool:
        return self.hits >= self.min_hits

    def predict(self) -> BoundingBox:
        """Constant-velocity prediction of where this track is NOW, coasted by `age`.

        A linear model is sufficient here and deliberately chosen over a Kalman filter:
        at 15 fps indoor displacement per frame is small, and a mis-specified motion
        model hurts most during exactly the event we care about (a fall, which is not
        constant-velocity).

        Scaling by `age` is what lets a track survive an occlusion: after 8 missed
        frames the person is 8 steps further along, so associating against the last
        *observed* box would fail and the resident would be re-identified as a new
        person. The coasted box is used ONLY to widen association - it is never emitted,
        because `active_tracks()` filters to `age == 0`. A predicted position is not an
        observation and must not reach Agent 2 as one.
        """
        vx, vy = self.velocity
        steps = max(1, self.age)
        try:
            return BoundingBox(
                x1=self.box.x1 + vx * steps, y1=self.box.y1 + vy * steps,
                x2=self.box.x2 + vx * steps, y2=self.box.y2 + vy * steps,
            )
        except ValueError:
            return self.box


class SimpleTracker:
    """Motion-based association with two-threshold lifecycle management.

    A BoT-SORT-*shaped* stand-in, not a reimplementation. It implements the parts that
    downstream correctness depends on - two-threshold association (weak detections extend
    tracks but cannot create them), constant-velocity coasting through occlusion, and
    age-based deletion - and deliberately omits the two things that need a GPU and a
    checkpoint: appearance-gated association and camera-motion compensation.

    Association here is motion-only. In this pipeline embeddings are computed *after*
    association (one crop per confirmed track, not per detection), so there is no
    embedding available to gate association with; real BoT-SORT computes them inside the
    tracker and does gate on them. That is a real capability gap, stated rather than
    papered over: the consequence is that two people crossing at the same depth can swap
    IDs here where BoT-SORT would not. The role-vote window and hysteresis exist partly
    to blunt exactly that failure.

    It exists so that everything downstream - ReID gallery logic, role voting, hysteresis
    and all of Agent 2 - is testable and debuggable now, on CPU, with no checkpoint.
    Tracking metrics (HOTA/MOTA/IDF1) are reported for real BoT-SORT on MOT17; this class
    is never presented as a tracking contribution.
    """

    def __init__(self, config: PerceptionConfig | None = None) -> None:
        self.config = config or PerceptionConfig()
        self._tracks: dict[int, Track] = {}
        self._next_id = 0

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def active_tracks(self) -> list[Track]:
        """Tracks matched in the current frame and confirmed."""
        return [t for t in self._tracks.values() if t.age == 0 and t.confirmed]

    def update(self, detections: Sequence[RawDetection]) -> list[Track]:
        cfg = self.config
        for track in self._tracks.values():
            track.age += 1

        # Any detection above the LOW floor may extend an existing track.
        unmatched = [
            j for j, d in enumerate(detections) if d.confidence >= cfg.det_low_confidence
        ]
        # Match confirmed tracks first: an established identity should win a contested
        # detection over a tentative one-frame track.
        order = sorted(
            self._tracks.values(), key=lambda t: (not t.confirmed, t.age, -t.frames_seen)
        )

        for track in order:
            best_j, best_iou = None, cfg.iou_threshold
            predicted = track.predict()
            for j in unmatched:
                score = max(iou(track.box, detections[j].box), iou(predicted, detections[j].box))
                if score > best_iou:
                    best_iou, best_j = score, j
            if best_j is None:
                continue

            det = detections[best_j]
            prev_cx, prev_cy = track.box.centroid
            new_cx, new_cy = det.box.centroid
            # Divide by the frame gap: after coasting through 8 missed frames the total
            # displacement covers 8 frames, and storing it as per-frame velocity would
            # make the next prediction overshoot by 8x.
            gap = max(1, track.age)
            track.velocity = ((new_cx - prev_cx) / gap, (new_cy - prev_cy) / gap)
            track.box = det.box
            track.confidence = det.confidence
            track.keypoints = det.keypoints
            track.hits += 1
            track.frames_seen += 1
            track.age = 0
            unmatched.remove(best_j)

        # Only HIGH-score leftovers may start a track.
        for j in unmatched:
            det = detections[j]
            if det.confidence < cfg.det_confidence:
                continue
            self._tracks[self._next_id] = Track(
                track_id=self._next_id,
                box=det.box,
                confidence=det.confidence,
                keypoints=det.keypoints,
                min_hits=cfg.min_hits,
            )
            self._next_id += 1

        dead = [tid for tid, t in self._tracks.items() if t.age > cfg.max_age_frames]
        for tid in dead:
            del self._tracks[tid]

        return self.active_tracks()


# ---------------------------------------------------------------------------
# Open-set ReID gallery
# ---------------------------------------------------------------------------


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2]. Zero-norm vectors return max distance."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 2.0
    return 1.0 - float(np.dot(a, b) / (na * nb))


@dataclass
class GalleryEntry:
    """One enrolled person: a role plus a running centroid of their embeddings."""

    name: str
    role: Role
    centroid: np.ndarray
    n_updates: int = 1
    enrolled_at: datetime | None = None


class OpenSetGallery:
    """Enrolled-identity gallery with explicit rejection.

    Rejection is the point. Three conditions must hold for a match:

      1. best distance < `reid_match_threshold`   (absolute similarity)
      2. best-to-second gap > `reid_margin`       (unambiguous)
      3. crop area >= `min_embed_box_area`        (enough pixels to mean anything)

    Any failure returns (None, distance) and the caller emits UNKNOWN. Reporting
    "unknown person present" is a useful signal in elderly care; silently mislabelling a
    stranger as the resident corrupts every downstream behavioural feature and is the
    failure mode that makes the whole system untrustworthy.
    """

    def __init__(self, config: PerceptionConfig | None = None) -> None:
        self.config = config or PerceptionConfig()
        self._entries: dict[str, GalleryEntry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def names(self) -> list[str]:
        return list(self._entries)

    def enrol(
        self,
        name: str,
        role: Role,
        embeddings: Sequence[np.ndarray],
        enrolled_at: datetime | None = None,
    ) -> GalleryEntry:
        """Enrol a person from one or more reference embeddings.

        The centroid is the mean of L2-normalised vectors: normalising *before* averaging
        prevents a high-magnitude crop from dominating, since only direction carries
        identity under cosine distance.
        """
        if not embeddings:
            raise ValueError(f"cannot enrol {name!r} with no embeddings")
        if role is Role.UNKNOWN:
            raise ValueError(
                "cannot enrol a person as UNKNOWN; UNKNOWN is the rejection outcome, "
                "not an identity"
            )
        stacked = np.stack([_l2(e) for e in embeddings])
        entry = GalleryEntry(
            name=name,
            role=role,
            centroid=_l2(stacked.mean(axis=0)),
            n_updates=len(embeddings),
            enrolled_at=enrolled_at,
        )
        self._entries[name] = entry
        return entry

    def match(
        self, embedding: np.ndarray, box_area: float | None = None
    ) -> tuple[GalleryEntry | None, float]:
        """Return (entry, distance), or (None, distance) on rejection."""
        cfg = self.config
        if not self._entries:
            return None, 2.0
        if box_area is not None and box_area < cfg.min_embed_box_area:
            # Too few pixels: the embedding is upsampling artefact, and a low-information
            # vector sits near the gallery mean and matches everything.
            return None, 2.0

        query = _l2(embedding)
        scored = sorted(
            ((cosine_distance(query, e.centroid), e) for e in self._entries.values()),
            key=lambda pair: pair[0],
        )
        best_dist, best_entry = scored[0]
        if best_dist >= cfg.reid_match_threshold:
            return None, best_dist
        if len(scored) > 1 and (scored[1][0] - best_dist) < cfg.reid_margin:
            # Ambiguous between two enrolled people. Refusing is correct: a swap merges
            # two people's daily features under one role.
            return None, best_dist
        return best_entry, best_dist

    def update_centroid(self, name: str, embedding: np.ndarray) -> None:
        """Adapt an enrolled centroid toward a new confident observation."""
        entry = self._entries.get(name)
        if entry is None:
            return
        m = self.config.gallery_momentum
        entry.centroid = _l2(m * entry.centroid + (1.0 - m) * _l2(embedding))
        entry.n_updates += 1


def _l2(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v if n < 1e-9 else v / n


# ---------------------------------------------------------------------------
# Geometric cues
# ---------------------------------------------------------------------------


def torso_vertical_ratio(kp: Keypoints, threshold: float = 0.3) -> float | None:
    """|shoulder-hip vertical span| / torso length. Near 1 upright, near 0 horizontal.

    A cheap, model-free fall cue used as a sanity feature alongside the learned
    classifier - not as a detector. It is view-dependent (a person lying along the
    optical axis looks upright), which is precisely why it supports rather than replaces
    the skeleton model.
    """
    idx = (_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP)
    if any(kp.scores[i] < threshold for i in idx):
        return None
    sx = (kp.xy[_L_SHOULDER][0] + kp.xy[_R_SHOULDER][0]) / 2.0
    sy = (kp.xy[_L_SHOULDER][1] + kp.xy[_R_SHOULDER][1]) / 2.0
    hx = (kp.xy[_L_HIP][0] + kp.xy[_R_HIP][0]) / 2.0
    hy = (kp.xy[_L_HIP][1] + kp.xy[_R_HIP][1]) / 2.0
    length = float(np.hypot(sx - hx, sy - hy))
    if length < 1e-6:
        return None
    return abs(sy - hy) / length


# ---------------------------------------------------------------------------
# Agent 1
# ---------------------------------------------------------------------------


class PerceptionAgent:
    """Agent 1 top level: frame -> FrameObservation.

    Emits only schema objects. Agent 2 consumes `FrameObservation` and never touches
    pixels, which is what keeps the agent boundary auditable (see schemas.py).
    """

    def __init__(
        self,
        detector: PoseDetector,
        embedder: ReIDEmbedder | None = None,
        object_detector: ObjectDetector | None = None,
        gallery: OpenSetGallery | None = None,
        config: PerceptionConfig | None = None,
        fps: float = 15.0,
    ) -> None:
        self.config = config or PerceptionConfig()
        self.detector = detector
        self.embedder = embedder
        self.object_detector = object_detector
        self.gallery = gallery or OpenSetGallery(self.config)
        self.tracker = SimpleTracker(self.config)
        self.fps = fps
        self._frame_idx = 0
        self._last_objects: list[DetectedObject] = []
        self._object_interval = max(1, int(round(fps / self.config.object_sample_hz)))
        self._track_names: dict[int, str] = {}

    # -- role assignment --------------------------------------------------------

    def _vote_role(self, track: Track, observed: Role) -> None:
        """Update a track's role by windowed majority vote with hysteresis.

        A role is a property of a track, not of a frame. Voting absorbs single-frame
        embedding noise; the switch margin stops a track near the decision boundary from
        oscillating, which would otherwise split one continuous activity across two
        people downstream.
        """
        cfg = self.config
        track.role_votes.append(observed)
        while len(track.role_votes) > cfg.role_vote_window:
            track.role_votes.popleft()

        counts = Counter(track.role_votes)
        known = {r: n for r, n in counts.items() if r is not Role.UNKNOWN}
        if not known:
            track.role, track.role_confidence = Role.UNKNOWN, 0.0
            return

        leader, lead_votes = max(known.items(), key=lambda kv: kv[1])
        if lead_votes < cfg.role_min_votes:
            # Evidence too thin to claim an identity. UNKNOWN with zero confidence is
            # the only representable honest answer (PersonObservation enforces this).
            track.role, track.role_confidence = Role.UNKNOWN, 0.0
            return

        if track.role is not Role.UNKNOWN and leader is not track.role:
            incumbent = counts.get(track.role, 0)
            if lead_votes - incumbent < cfg.role_switch_margin:
                return  # challenger has not earned the switch

        track.role = leader
        track.role_confidence = min(1.0, lead_votes / max(1, len(track.role_votes)))

    def _identify(self, frame: np.ndarray, track: Track) -> Role:
        """Embed this track's crop and match it against the gallery."""
        if self.embedder is None:
            return Role.UNKNOWN
        try:
            embedding = self.embedder.embed(frame, track.box)
        except Exception:
            # A crop at the frame edge can fail to embed. That is an absence of
            # evidence, not evidence of a stranger - fall through to UNKNOWN and let
            # the vote window decide.
            return Role.UNKNOWN

        track.embedding = embedding
        entry, distance = self.gallery.match(embedding, box_area=track.box.area)
        track.reid_distance = distance
        if entry is None:
            self._track_names.pop(track.track_id, None)
            return Role.UNKNOWN

        self._track_names[track.track_id] = entry.name
        # Only adapt the centroid on a comfortable match. Updating on a borderline match
        # is how a gallery slowly drifts onto the wrong person.
        if distance < self.config.reid_match_threshold * 0.6:
            self.gallery.update_centroid(entry.name, embedding)
        return entry.role

    def track_name(self, track_id: int) -> str | None:
        """Enrolled name currently associated with a track, if any."""
        return self._track_names.get(track_id)

    # -- main entry point -------------------------------------------------------

    def process_frame(
        self, frame: np.ndarray, timestamp: datetime, room: str | None = None
    ) -> FrameObservation:
        cfg = self.config
        # Pass BOTH score bands to the tracker: it decides which may start a track and
        # which may only extend one. Filtering to `det_confidence` here would silently
        # disable low-score recovery and reintroduce the occlusion ID-switch.
        detections = [
            d for d in self.detector.detect(frame) if d.confidence >= cfg.det_low_confidence
        ]
        tracks = self.tracker.update(detections)

        # Objects at ~1 Hz; the cached result is reused between samples.
        if self.object_detector is not None and self._frame_idx % self._object_interval == 0:
            self._last_objects = self.object_detector.detect(frame)

        persons: list[PersonObservation] = []
        for track in tracks:
            observed = self._identify(frame, track)
            self._vote_role(track, observed)

            keypoints = track.keypoints
            if keypoints is not None and not keypoints.is_usable(cfg.min_visible_keypoints):
                # Keep the person (they are present and tracked) but drop unusable pose:
                # Agent 2 must abstain rather than classify from a mostly-occluded
                # skeleton, and passing it on would produce confident nonsense.
                keypoints = None

            persons.append(
                PersonObservation(
                    frame_idx=self._frame_idx,
                    timestamp=timestamp,
                    track_id=track.track_id,
                    box=track.box,
                    detection_confidence=min(1.0, max(0.0, track.confidence)),
                    keypoints=keypoints,
                    role=track.role,
                    role_confidence=track.role_confidence,
                    reid_distance=track.reid_distance,
                    room=room,
                )
            )

        observation = FrameObservation(
            frame_idx=self._frame_idx,
            timestamp=timestamp,
            persons=persons,
            objects=list(self._last_objects),
        )
        self._frame_idx += 1
        return observation

    def process_sequence(
        self,
        frames: Sequence[np.ndarray],
        timestamps: Sequence[datetime],
        room: str | None = None,
    ) -> list[FrameObservation]:
        if len(frames) != len(timestamps):
            raise ValueError(
                f"frames ({len(frames)}) and timestamps ({len(timestamps)}) must align"
            )
        return [
            self.process_frame(f, t, room=room) for f, t in zip(frames, timestamps)
        ]

    def reset(self) -> None:
        """Clear tracking state, keeping the gallery. Use between videos.

        The gallery is enrolment (persistent); tracks are per-video. Carrying track IDs
        across videos would silently merge two recordings into one identity.
        """
        self.tracker = SimpleTracker(self.config)
        self._frame_idx = 0
        self._last_objects = []
        self._track_names.clear()


def role_durations(
    observations: Sequence[FrameObservation], fps: float = 15.0
) -> dict[Role, float]:
    """Seconds of presence per role. A quick sanity check on a processed video."""
    counts: dict[Role, int] = defaultdict(int)
    for obs in observations:
        for person in obs.persons:
            counts[person.role] += 1
    return {role: n / fps for role, n in counts.items()}


__all__ = [
    "PERCEPTION_VERSION",
    "RawDetection",
    "PoseDetector",
    "ReIDEmbedder",
    "ObjectDetector",
    "PerceptionConfig",
    "Track",
    "SimpleTracker",
    "GalleryEntry",
    "OpenSetGallery",
    "PerceptionAgent",
    "iou",
    "cosine_distance",
    "torso_vertical_ratio",
    "role_durations",
]
