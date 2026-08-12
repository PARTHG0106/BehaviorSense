"""The runtime pipeline: FrameObservations -> windows -> ActivitySegments -> DailyFeatures.

The missing seam
----------------
Agent 1 emits per-frame `FrameObservation`s. Agent 2 consumes fixed-length skeleton
windows *per tracked person*. Nothing converted between them, so the two agents could not
actually be run back to back. This module is that conversion, and it is deliberately
separate from both agents because the tricky decisions belong to neither:

  - windows are built **per track**, never per frame. Pooling two people into one window
    would let a visitor's motion be classified as the resident's activity, and because
    Agent 3 aggregates per role, the contamination would be invisible downstream.
  - a track that vanishes mid-window does not get a padded window. Padding invents
    skeletons, and an invented skeleton at the end of a window is exactly the shape of a
    fall (body leaves frame -> zeros -> looks like collapse).
  - gaps are broken, not bridged. If a track is absent for longer than the window, the
    next window starts fresh rather than splicing across the gap, which would fabricate
    motion that never happened.

Room and role are carried through from perception, because Agent 3's features are
role-scoped and room transitions are one of its mobility signals.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from behaviorsense.agents.activity import ActivityAgent, ActivityConfig, WindowClassifier
from behaviorsense.schemas import (
    ActivitySegment,
    DailyFeatures,
    FrameObservation,
    Role,
)

WINDOW_FRAMES = 30
STRIDE_FRAMES = 15
MAX_PERSONS = 2
N_JOINTS = 17


@dataclass
class TrackWindows:
    """Windows extracted for one track, with the metadata Agent 2 needs to label them."""

    track_id: int
    role: Role
    windows: list[np.ndarray] = field(default_factory=list)
    starts: list[datetime] = field(default_factory=list)
    ends: list[datetime] = field(default_factory=list)
    rooms: list[str | None] = field(default_factory=list)
    objects: list[tuple[str, ...]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.windows)


@dataclass
class PipelineStats:
    """What the pipeline discarded and why. Printed, never silently accumulated."""

    frames_seen: int = 0
    tracks_seen: int = 0
    windows_built: int = 0
    dropped_no_pose: int = 0
    dropped_short_track: int = 0
    gaps_broken: int = 0

    def summary(self) -> str:
        return (
            f"{self.frames_seen} frames, {self.tracks_seen} tracks -> "
            f"{self.windows_built} windows "
            f"(dropped: {self.dropped_no_pose} unusable pose, "
            f"{self.dropped_short_track} too-short tracks; "
            f"{self.gaps_broken} gaps broken)"
        )


def frames_to_windows(
    frames: Sequence[FrameObservation],
    window: int = WINDOW_FRAMES,
    stride: int = STRIDE_FRAMES,
    max_gap_frames: int = 5,
    stats: PipelineStats | None = None,
) -> list[TrackWindows]:
    """Group per-frame observations into per-track skeleton windows.

    `max_gap_frames` is the tolerance for a brief detection miss inside a window. Beyond
    it the run is cut: bridging a long gap would interpolate a trajectory the person never
    took, and interpolated descent is indistinguishable from a fall.
    """
    stats = stats or PipelineStats()
    stats.frames_seen += len(frames)

    # Collect per-track observation runs, keyed by track, preserving frame order.
    runs: dict[int, list[tuple[int, datetime, np.ndarray | None, Role, str | None,
                              tuple[str, ...]]]] = defaultdict(list)
    for frame in frames:
        objs = tuple(sorted({o.label for o in frame.objects}))
        for person in frame.persons:
            kp = person.keypoints
            arr = None
            if kp is not None:
                arr = np.zeros((N_JOINTS, 3), dtype=np.float32)
                arr[:, :2] = np.asarray(kp.xy, dtype=np.float32)
                arr[:, 2] = np.asarray(kp.scores, dtype=np.float32)
            runs[person.track_id].append(
                (frame.frame_idx, person.timestamp, arr, person.role, person.room, objs)
            )

    out: list[TrackWindows] = []
    for track_id, obs in runs.items():
        stats.tracks_seen += 1
        obs.sort(key=lambda o: o[0])

        # Split into contiguous segments, breaking on long gaps.
        segments: list[list] = [[]]
        for i, item in enumerate(obs):
            if i and (item[0] - obs[i - 1][0]) > max_gap_frames:
                stats.gaps_broken += 1
                segments.append([])
            segments[-1].append(item)

        tw = TrackWindows(track_id=track_id, role=obs[0][3])
        for seg in segments:
            usable = [o for o in seg if o[2] is not None]
            stats.dropped_no_pose += len(seg) - len(usable)
            if len(usable) < window:
                if usable:
                    stats.dropped_short_track += 1
                continue
            for start in range(0, len(usable) - window + 1, stride):
                chunk = usable[start:start + window]
                w = np.zeros((window, MAX_PERSONS, N_JOINTS, 3), dtype=np.float32)
                for t, item in enumerate(chunk):
                    w[t, 0] = item[2]
                tw.windows.append(w)
                tw.starts.append(chunk[0][1])
                tw.ends.append(chunk[-1][1])
                # Majority room over the window; a window straddling a doorway belongs to
                # where most of it happened.
                rooms = [c[4] for c in chunk if c[4]]
                tw.rooms.append(max(set(rooms), key=rooms.count) if rooms else None)
                tw.objects.append(tuple(sorted({o for c in chunk for o in c[5]})))
                # Role can change mid-track (open-set ReID resolving late); the window
                # takes the role in force at its end, which is the best-informed guess.
                tw.role = chunk[-1][3]
        if tw.windows:
            stats.windows_built += len(tw.windows)
            out.append(tw)
    return out


class ActivityPipeline:
    """FrameObservations -> ActivitySegments, using a trained or synthetic classifier."""

    def __init__(
        self,
        classifier: WindowClassifier,
        config: ActivityConfig | None = None,
        agent: ActivityAgent | None = None,
    ) -> None:
        self.classifier = classifier
        self.config = config or ActivityConfig()
        self.agent = agent or ActivityAgent(self.config)

    def run(
        self, frames: Sequence[FrameObservation], stats: PipelineStats | None = None
    ) -> list[ActivitySegment]:
        stats = stats or PipelineStats()
        segments: list[ActivitySegment] = []

        for tw in frames_to_windows(frames, stats=stats):
            logits = self.classifier.logits(np.stack(tw.windows))
            results = self.agent.classify_stream(
                logits, tw.starts, tw.ends, objects_per_window=tw.objects
            )
            # Segment per room: a continuous 'walking' run that crosses two rooms is two
            # segments, because room transitions are one of Agent 3's mobility features
            # and merging them would erase the transition.
            run_start = 0
            for i in range(1, len(results) + 1):
                boundary = i == len(results) or tw.rooms[i] != tw.rooms[run_start]
                if boundary:
                    segments.extend(
                        self.agent.to_segments(
                            results[run_start:i],
                            track_id=tw.track_id,
                            role=tw.role,
                            room=tw.rooms[run_start],
                        )
                    )
                    run_start = i
        segments.sort(key=lambda s: (s.start_time, s.track_id))
        return segments

    def run_to_features(
        self,
        frames: Sequence[FrameObservation],
        day=None,
        subject_role: Role = Role.RESIDENT,
        observed_hours: float | None = None,
        stats: PipelineStats | None = None,
    ) -> tuple[DailyFeatures, list[ActivitySegment]]:
        """Full path to Agent 3's input: frames -> segments -> DailyFeatures."""
        from behaviorsense.agents.behaviour import aggregate_daily_features

        segments = self.run(frames, stats=stats)
        if day is None:
            if not frames:
                raise ValueError("cannot infer the day from an empty frame sequence")
            day = frames[0].timestamp.date()

        features = aggregate_daily_features(segments, day=day, subject_role=subject_role)
        if observed_hours is not None:
            # observed_hours drives Agent 3's reliability gate, and it is a property of the
            # CAMERA schedule, not of the segments - a day with the camera off for 12 h
            # must not look like a day the resident was motionless.
            features = features.model_copy(update={"observed_hours": observed_hours})
        return features, segments


def observed_hours_from_frames(frames: Iterable[FrameObservation], fps: float = 15.0) -> float:
    """Hours of wall-clock actually covered by frames, ignoring gaps.

    Counting first-to-last timestamp would report a full day for a camera that ran for ten
    minutes at midnight and ten at noon, and Agent 3 would then treat an unobserved day as
    a genuinely inactive one - the exact confusion that produced 14 alerts/day on outage
    days in the first honest evaluation run.
    """
    stamps = sorted(f.timestamp for f in frames)
    if len(stamps) < 2:
        return 0.0
    gap_limit = timedelta(seconds=5.0)
    total = timedelta()
    for a, b in zip(stamps, stamps[1:]):
        step = b - a
        if step <= gap_limit:
            total += step
    return total.total_seconds() / 3600.0


__all__ = [
    "ActivityPipeline",
    "TrackWindows",
    "PipelineStats",
    "frames_to_windows",
    "observed_hours_from_frames",
    "WINDOW_FRAMES",
    "STRIDE_FRAMES",
]
