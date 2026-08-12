"""Full four-agent replay: video-like input -> perception -> activity -> behaviour -> report.

`run_demo.py` starts from simulated *daily features*, which skips Agents 1 and 2 entirely.
That is the right safety-net demo (it cannot be broken by a missing checkpoint), but it
leaves the perception->activity seam undemonstrated. This script closes that: it runs
Agent 1 on synthetic frames, converts to windows, classifies, aggregates to daily features,
and reports - so every schema boundary in the system is crossed for real.

The frames are synthetic because no elderly-care video exists that we may redistribute.
That is stated rather than hidden: the point here is that the *plumbing* is correct
end-to-end, not that accuracy has been demonstrated. Accuracy needs the trained
checkpoints and the P1/P2/P3 protocols.

Usage:
    PYTHONPATH=src python scripts/run_full_pipeline.py
    PYTHONPATH=src python scripts/run_full_pipeline.py --checkpoints runs
    PYTHONPATH=src python scripts/run_full_pipeline.py --days 21 --inject-fall
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behaviorsense.agents.behaviour import BehaviourAnalyzer  # noqa: E402
from behaviorsense.agents.perception import (  # noqa: E402
    OpenSetGallery,
    PerceptionAgent,
    PerceptionConfig,
    RawDetection,
)
from behaviorsense.agents.reasoning.reporter import (  # noqa: E402
    CaregiverReporter,
    FaithfulStubLLM,
    ReporterConfig,
)
from behaviorsense.models.ensemble import EnsembleClassifier, SyntheticClassifier  # noqa: E402
from behaviorsense.pipeline import (  # noqa: E402
    ActivityPipeline,
    PipelineStats,
    observed_hours_from_frames,
)
from behaviorsense.schemas import BoundingBox, DetectedObject, Keypoints, Role  # noqa: E402

FPS = 15.0
DIM = 512


class ScriptedScene:
    """Generates detections for a scripted day: the resident, sometimes a visitor.

    Stands in for RTMO. Deterministic, so the demo is reproducible.
    """

    def __init__(self, n_frames: int, fall_at: int | None = None,
                 visitor_from: int | None = None, seed: int = 0) -> None:
        self.n = n_frames
        self.fall_at = fall_at
        self.visitor_from = visitor_from
        self.rng = np.random.default_rng(seed)
        self.i = 0

    def _kp(self, x: float, y: float) -> Keypoints:
        xy = []
        for j in range(17):
            jitter = self.rng.normal(0, 1.2, 2)
            xy.append((x + jitter[0] + (j % 3) * 4.0, y + jitter[1] + j * 3.0))
        xy[5], xy[6] = (x - 18, y + 20), (x + 18, y + 20)
        xy[11], xy[12] = (x - 14, y + 80), (x + 14, y + 80)
        return Keypoints(xy=xy, scores=[0.9] * 17)

    def detect(self, frame: np.ndarray) -> list[RawDetection]:
        i, out = self.i, []
        # Resident: paces left-right; after a fall, drops and stays low.
        phase = (i % 120) / 120.0
        x = 120.0 + 180.0 * abs(2 * phase - 1)
        y = 100.0
        if self.fall_at is not None and i >= self.fall_at:
            drop = min(1.0, (i - self.fall_at) / 20.0)
            y = 100.0 + 130.0 * drop
            x = 240.0
        h = 160.0 - 100.0 * ((y - 100.0) / 130.0)
        out.append(RawDetection(
            box=BoundingBox(x1=x, y1=y, x2=x + 60, y2=y + max(50.0, h)),
            confidence=0.92, keypoints=self._kp(x + 30, y),
        ))
        if self.visitor_from is not None and i >= self.visitor_from:
            vx = 500.0 + 40.0 * np.sin(i / 25.0)
            out.append(RawDetection(
                box=BoundingBox(x1=vx, y1=100.0, x2=vx + 58, y2=262.0),
                confidence=0.88, keypoints=self._kp(vx + 29, 100.0),
            ))
        self.i += 1
        return out


class ScriptedEmbedder:
    """Position-keyed embeddings: the resident on the left, a stranger on the right."""

    def __init__(self, resident: np.ndarray, stranger: np.ndarray) -> None:
        self.resident, self.stranger = resident, stranger

    @property
    def dim(self) -> int:
        return DIM

    def embed(self, frame: np.ndarray, box: BoundingBox) -> np.ndarray:
        cx, _ = box.centroid
        base = self.resident if cx < 420 else self.stranger
        return base + np.random.default_rng(int(cx)).normal(0, 0.01, DIM)


class RoomObjects:
    def __init__(self, labels: tuple[str, ...]) -> None:
        self.labels = labels

    def detect(self, frame: np.ndarray) -> list[DetectedObject]:
        return [DetectedObject(label=l, confidence=0.8,
                               box=BoundingBox(x1=10, y1=10, x2=40, y2=40))
                for l in self.labels]


def simulate_day(
    day: date, pipeline: ActivityPipeline, *, fall: bool = False, visitor: bool = True,
    frames_per_day: int = 450, seed: int = 0,
) -> tuple:
    """Run Agents 1 and 2 over one synthetic day."""
    rng = np.random.default_rng(1234)
    resident_emb = rng.normal(size=DIM)
    resident_emb /= np.linalg.norm(resident_emb)
    stranger_emb = np.random.default_rng(999).normal(size=DIM)
    stranger_emb /= np.linalg.norm(stranger_emb)

    cfg = PerceptionConfig()
    gallery = OpenSetGallery(cfg)
    gallery.enrol("resident", Role.RESIDENT, [resident_emb])

    scene = ScriptedScene(
        frames_per_day,
        fall_at=int(frames_per_day * 0.6) if fall else None,
        visitor_from=int(frames_per_day * 0.3) if visitor else None,
        seed=seed,
    )
    agent1 = PerceptionAgent(
        detector=scene,
        embedder=ScriptedEmbedder(resident_emb, stranger_emb),
        object_detector=RoomObjects(("couch", "tv", "cup")),
        gallery=gallery,
        config=cfg,
        fps=FPS,
    )

    start = datetime.combine(day, datetime.min.time()).replace(hour=9)
    blank = np.zeros((8, 8, 3), dtype=np.uint8)
    frames = [
        agent1.process_frame(blank, start + timedelta(seconds=i / FPS), room="lounge")
        for i in range(frames_per_day)
    ]

    stats = PipelineStats()
    hours = observed_hours_from_frames(frames)
    feats, segments = pipeline.run_to_features(
        frames, day=day, subject_role=Role.RESIDENT,
        observed_hours=max(hours, 8.5),  # pretend the camera ran all day
        stats=stats,
    )
    return frames, segments, feats, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=24)
    ap.add_argument("--checkpoints", default=None,
                    help="run dir with adl_<stream>/best.pt; omitted = synthetic classifier")
    ap.add_argument("--inject-fall", action="store_true",
                    help="inject a fall on the final day")
    ap.add_argument("--frames-per-day", type=int, default=450)
    args = ap.parse_args()

    if args.checkpoints:
        classifier = EnsembleClassifier.from_run_dir(args.checkpoints)
        print(f"classifier: {classifier.name}")
        for s, p in classifier.loaded_from.items():
            print(f"  {s}: {p}")
    else:
        classifier = SyntheticClassifier(seed=0)
        print(f"classifier: {classifier.name} "
              "(NOT trained - demonstrates plumbing, not accuracy)")

    pipeline = ActivityPipeline(classifier)
    analyzer = BehaviourAnalyzer()
    d0 = date(2026, 3, 1)

    print(f"\nrunning {args.days} days x {args.frames_per_day} frames "
          f"({args.frames_per_day / FPS:.0f}s of video per day)")
    print(f"{'day':<12} {'frames':>7} {'segs':>5} {'walk_s':>7} {'roles':>18} {'alerts':>7}")

    states = []
    for d in range(args.days):
        day = d0 + timedelta(days=d)
        is_last = d == args.days - 1
        frames, segments, feats, stats = simulate_day(
            day, pipeline,
            fall=args.inject_fall and is_last,
            visitor=(d % 3 == 0),
            frames_per_day=args.frames_per_day,
            seed=d,
        )
        state = analyzer.analyze_day(feats)
        states.append(state)

        roles = sorted({p.role.value for f in frames for p in f.persons})
        kinds = ",".join(a.kind.value for a in state.alerts) or "-"
        print(f"{day.isoformat():<12} {len(frames):>7} {len(segments):>5} "
              f"{feats.walking_duration_s:>7.0f} {'+'.join(roles):>18} {kinds:>7}")

    # ---- Agent 4 ----------------------------------------------------------
    alerting = [s for s in states if s.alerts]
    target = alerting[-1] if alerting else states[-1]
    reporter = CaregiverReporter(FaithfulStubLLM(), config=ReporterConfig())
    outcome = reporter.report(target)

    print(f"\n{'=' * 74}")
    print(f"Agent 4 report for {target.report_day} "
          f"({len(outcome.report.claims)} claims, "
          f"{outcome.hallucination_rate:.0%} hallucination rate)")
    print("=" * 74)
    print(reporter.verified_summary(outcome))
    print("=" * 74)
    print("\nAll four agents crossed their schema boundaries: "
          "RawDetection -> PersonObservation ->\nActivitySegment -> DailyFeatures -> "
          "BehaviourState -> Claim -> VerificationResult.")


if __name__ == "__main__":
    main()
