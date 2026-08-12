"""Inter-agent data contracts for BehaviorSense AI.

Every agent boundary is a versioned, validated schema persisted to SQLite rather than an
in-process function call. This buys three things that matter on a tight timeline:

1. Independent testability   - feed Agent 3 synthetic ActivitySegments, no video needed.
2. Replayability             - rerun reasoning in seconds without redoing perception.
3. Auditability              - every LLM claim points at a row that provably exists.

Point 3 is load-bearing for the research contribution: the faithfulness verifier
(agents/reasoning/verifier.py) resolves `evidence_ref` strings against these records.
If the schema drifts, verification silently degrades - hence SCHEMA_VERSION.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0.0"

Z_CLAMP = 10.0
"""Hard bound on reported robust z-scores.

A degenerate baseline can otherwise produce values in the thousands, which would be
quoted verbatim to a caregiver ("mobility is 4000 standard deviations below normal").
Clamping keeps every reported number physically plausible.
"""

# Reusable constrained scalars.
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegFloat = Annotated[float, Field(ge=0.0)]


class Role(str, Enum):
    """Person role.

    Deliberately NOT identity. Face recognition is rejected for this system: CCTV
    yields too few facial pixels, every available ADL dataset blurs faces (making it
    both untrainable and untestable), and it maximises ethics exposure for no gain.
    Roles are what the behavioural layer actually consumes.
    """

    RESIDENT = "resident"
    VISITOR = "visitor"
    CARER = "carer"
    UNKNOWN = "unknown"  # open-set reject. Never silently coerced to RESIDENT.


class AlertSeverity(str, Enum):
    INFO = "info"
    ADVISORY = "advisory"
    URGENT = "urgent"
    EMERGENCY = "emergency"


class AlertKind(str, Enum):
    FALL_DETECTED = "fall_detected"
    FALL_NO_RECOVERY = "fall_no_recovery"
    PROLONGED_INACTIVITY = "prolonged_inactivity"
    MEAL_SKIPPED = "meal_skipped"
    HYDRATION_LOW = "hydration_low"
    MEDICATION_MISSED = "medication_missed"
    MOBILITY_DECLINE = "mobility_decline"
    SLEEP_DISRUPTION = "sleep_disruption"
    SOCIAL_ISOLATION = "social_isolation"
    ROUTINE_DEVIATION = "routine_deviation"
    UNKNOWN_PERSON = "unknown_person"


# ---------------------------------------------------------------------------
# Agent 1 - Perception
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _check_ordering(self) -> BoundingBox:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"degenerate box: ({self.x1},{self.y1})-({self.x2},{self.y2})")
        return self

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def aspect_ratio(self) -> float:
        """Height / width. A cheap but effective fall cue: a standing person is ~2.5,
        a fallen person is <1.0. Used as a sanity feature alongside the learned model.
        """
        return (self.y2 - self.y1) / max(self.x2 - self.x1, 1e-6)


class Keypoints(BaseModel):
    """COCO-17 2D keypoints for one person in one frame."""

    model_config = ConfigDict(frozen=True)

    xy: list[tuple[float, float]] = Field(min_length=17, max_length=17)
    scores: list[Confidence] = Field(min_length=17, max_length=17)

    COCO_NAMES: ClassVar[tuple[str, ...]] = (
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    )

    def visible(self, threshold: float = 0.3) -> int:
        return sum(1 for s in self.scores if s >= threshold)

    def is_usable(self, min_visible: int = 8) -> bool:
        """Gate for downstream action recognition.

        Windows built from mostly-occluded skeletons produce confident nonsense, which
        is worse than an abstention in a clinical setting. Filter early.
        """
        return self.visible() >= min_visible


class DetectedObject(BaseModel):
    """Object context, sampled at ~1 Hz rather than per-frame.

    Rationale: home scenes are near-static, so 1 Hz loses almost nothing and cuts
    object-detection cost ~30x. This exists because skeletons alone cannot separate
    `taking_medication` from `drinking` - see taxonomy.yaml class 12.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    confidence: Confidence
    box: BoundingBox


class PersonObservation(BaseModel):
    """Agent 1 output: one person, one frame."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    frame_idx: int = Field(ge=0)
    timestamp: datetime
    track_id: int = Field(ge=0)
    box: BoundingBox
    detection_confidence: Confidence
    keypoints: Keypoints | None = None
    role: Role = Role.UNKNOWN
    role_confidence: Confidence = 0.0
    reid_distance: float | None = Field(
        default=None,
        description="Cosine distance to the matched gallery centroid. None if no gallery "
        "match was attempted. Above the open-set threshold => role stays UNKNOWN.",
    )
    room: str | None = None

    @model_validator(mode="after")
    def _unknown_role_has_no_confidence(self) -> PersonObservation:
        if self.role is Role.UNKNOWN and self.role_confidence > 0.5:
            raise ValueError(
                "UNKNOWN role with high confidence is contradictory; "
                "open-set rejection must not report confident identity"
            )
        return self


class FrameObservation(BaseModel):
    """All perception output for a single frame."""

    schema_version: str = SCHEMA_VERSION
    frame_idx: int = Field(ge=0)
    timestamp: datetime
    persons: list[PersonObservation] = Field(default_factory=list)
    objects: list[DetectedObject] = Field(default_factory=list)

    @property
    def n_persons(self) -> int:
        return len(self.persons)


# ---------------------------------------------------------------------------
# Agent 2 - Activity
# ---------------------------------------------------------------------------


class ActivitySegment(BaseModel):
    """A temporally-contiguous activity for one tracked person.

    Produced by windowed classification followed by HMM/Viterbi smoothing. Raw
    per-window argmax flickers badly (sitting/standing alternating at 2 Hz), which
    corrupts every duration-based behavioural feature downstream.
    """

    schema_version: str = SCHEMA_VERSION
    segment_id: str
    track_id: int = Field(ge=0)
    role: Role
    activity_id: int = Field(ge=0, le=19)
    activity_name: str
    start_time: datetime
    end_time: datetime
    confidence: Confidence
    mean_logit_margin: float | None = Field(
        default=None,
        description="Mean top1-top2 logit gap over the segment's windows. Low margin "
        "flags genuinely ambiguous classifications for review.",
    )
    room: str | None = None
    supporting_objects: list[str] = Field(default_factory=list)
    n_windows: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _check_interval(self) -> ActivitySegment:
        if self.end_time <= self.start_time:
            raise ValueError(f"segment {self.segment_id}: end_time must follow start_time")
        return self

    @property
    def duration_s(self) -> float:
        return (self.end_time - self.start_time).total_seconds()


# ---------------------------------------------------------------------------
# Agent 3 - Behaviour
# ---------------------------------------------------------------------------


class DailyFeatures(BaseModel):
    """One day of behaviour for one person, reduced to comparable scalars.

    This is the unit the baseline model aggregates over and the unit LLM claims cite.
    Field names are stable API: `evidence_ref` strings embed them as
    `feat:<field_name>:<date>`.
    """

    schema_version: str = SCHEMA_VERSION
    subject_role: Role
    day: date

    # Mobility
    walking_duration_s: NonNegFloat = 0.0
    walking_bouts: int = Field(default=0, ge=0)
    mean_bout_duration_s: NonNegFloat = 0.0
    room_transitions: int = Field(default=0, ge=0)
    sit_to_stand_count: int = Field(default=0, ge=0)
    mean_sit_to_stand_duration_s: NonNegFloat = 0.0

    # Posture budget
    sitting_duration_s: NonNegFloat = 0.0
    lying_duration_s: NonNegFloat = 0.0
    standing_duration_s: NonNegFloat = 0.0
    longest_inactive_block_s: NonNegFloat = 0.0

    # Nutrition / health
    meal_events: int = Field(default=0, ge=0)
    eating_duration_s: NonNegFloat = 0.0
    drinking_events: int = Field(default=0, ge=0)
    cooking_duration_s: NonNegFloat = 0.0
    medication_events: int = Field(default=0, ge=0)

    # Leisure / social / IADL
    tv_duration_s: NonNegFloat = 0.0
    reading_duration_s: NonNegFloat = 0.0
    phone_events: int = Field(default=0, ge=0)
    social_interaction_duration_s: NonNegFloat = 0.0
    visitor_count: int = Field(default=0, ge=0)
    housework_duration_s: NonNegFloat = 0.0

    # Critical
    fall_events: int = Field(default=0, ge=0)
    max_post_fall_immobility_s: NonNegFloat = 0.0

    # Data quality - without this, a camera outage looks like a behavioural collapse.
    observed_hours: NonNegFloat = 0.0
    tracking_coverage: Confidence = 1.0

    def is_reliable(self, min_hours: float = 8.0, min_coverage: float = 0.6) -> bool:
        return self.observed_hours >= min_hours and self.tracking_coverage >= min_coverage

    @property
    def mobility_index(self) -> float:
        """Composite mobility scalar; the primary frailty proxy.

        Weights reflect clinical emphasis on sit-to-stand and ambulation. Deliberately
        simple and inspectable - a caregiver-facing system should not hide its
        arithmetic inside a network.
        """
        return (
            0.4 * (self.walking_duration_s / 60.0)
            + 0.3 * self.sit_to_stand_count
            + 0.3 * self.room_transitions
        )

    @property
    def sedentary_ratio(self) -> float:
        total = self.sitting_duration_s + self.lying_duration_s + self.standing_duration_s
        if total <= 0:
            return 0.0
        return (self.sitting_duration_s + self.lying_duration_s) / total

    def numeric_items(self) -> dict[str, float]:
        """Flat name -> value map of every numeric feature, including derived ones.

        The faithfulness verifier uses exactly this to resolve `feat:<name>:<date>`
        references, so derived properties must appear here or claims about them are
        unverifiable.
        """
        out: dict[str, float] = {
            name: float(value)
            for name, value in self.__dict__.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        out["mobility_index"] = self.mobility_index
        out["sedentary_ratio"] = self.sedentary_ratio
        return out


class BaselineStats(BaseModel):
    """Robust per-feature baseline over a rolling window.

    Median/MAD rather than mean/std: with a 14-day window a single atypical day
    (a hospital visit, a family gathering) shifts the mean enough to mask real drift.
    MAD has a 50% breakdown point; std has 0%.
    """

    schema_version: str = SCHEMA_VERSION
    feature_name: str
    median: float
    mad: float
    n_days: int = Field(ge=1)
    window_start: date
    window_end: date
    mad_floor_frac: float = Field(
        default=0.05,
        ge=0.0,
        description="Minimum effective MAD as a fraction of |median|. Guards against "
        "hair-trigger z-scores on low-variance features (see robust_z).",
    )

    def effective_mad(self) -> float:
        """MAD with a floor proportional to |median|.

        Why this floor exists - it fixes a real alert-storm bug. Many behavioural
        features are near-constant for a person with a stable routine: `medication_events`
        is 1 every day, `meal_events` is 3 every day. Over a 14-day window MAD is then
        exactly 0, and an unfloored robust z divides by zero. A naive fallback (return
        +/-6 for any nonzero deviation) makes a 0.1% change look like a 6-sigma event,
        firing alerts on noise for precisely the most regular residents. That is the
        classic alert-fatigue failure mode that gets monitoring systems switched off.

        Flooring MAD at 5% of |median| treats "5% day-to-day spread" as the minimum
        plausible variability. It binds only when observed variance is implausibly low;
        for a genuinely variable feature the real MAD dominates and this is a no-op.
        """
        return max(self.mad, self.mad_floor_frac * abs(self.median), 1e-9)

    def robust_z(self, value: float) -> float:
        """Robust z-score, floored and clamped.

        0.6745 rescales MAD into sigma-equivalent units for normally-distributed data
        (MAD ~= 0.6745 * sigma), so thresholds are interpretable on the familiar scale.

        Clamped to +/-Z_CLAMP so a degenerate baseline (median 0, MAD 0) cannot emit an
        astronomical score that would later be quoted verbatim in a caregiver report.
        """
        if abs(value - self.median) < 1e-12:
            return 0.0
        z = 0.6745 * (value - self.median) / self.effective_mad()
        return max(-Z_CLAMP, min(Z_CLAMP, z))

    def pct_change(self, value: float) -> float:
        """Percentage change vs baseline median.

        Returns 0.0 when the median is 0, because percentage change is undefined there.
        Consumers MUST NOT infer direction from this value - use `Evidence.delta`, whose
        sign is always meaningful. The verifier relies on that distinction.
        """
        if abs(self.median) < 1e-9:
            return 0.0
        return 100.0 * (value - self.median) / self.median


class Evidence(BaseModel):
    """A single verifiable fact backing an alert or an LLM claim.

    `ref` format: `feat:<feature_name>:<ISO date>`. Machine-resolvable by design -
    a human-readable-only citation cannot be checked automatically.
    """

    model_config = ConfigDict(frozen=True)

    ref: str = Field(pattern=r"^feat:[a-z0-9_]+:\d{4}-\d{2}-\d{2}$")
    feature_name: str
    observed_value: float
    baseline_median: float
    robust_z: float
    pct_change: float
    delta: float = Field(
        default=0.0,
        description="observed_value - baseline_median. Unlike pct_change this is always "
        "sign-meaningful (pct_change collapses to 0.0 when the median is 0), so the "
        "verifier checks claimed direction against this field.",
    )
    baseline_n_days: int = Field(default=0, ge=0)
    day: date

    @classmethod
    def build(
        cls, feature_name: str, value: float, baseline: BaselineStats, day: date
    ) -> Evidence:
        return cls(
            ref=f"feat:{feature_name}:{day.isoformat()}",
            feature_name=feature_name,
            observed_value=value,
            baseline_median=baseline.median,
            robust_z=baseline.robust_z(value),
            pct_change=baseline.pct_change(value),
            delta=value - baseline.median,
            baseline_n_days=baseline.n_days,
            day=day,
        )

    @property
    def direction(self) -> Literal["increase", "decrease", "unchanged"]:
        """Ground-truth direction. The verifier compares LLM claims against this."""
        if abs(self.delta) < 1e-9:
            return "unchanged"
        return "increase" if self.delta > 0 else "decrease"


class Alert(BaseModel):
    """A triggered behavioural alert.

    Every alert carries its evidence. An alert a caregiver cannot interrogate is not
    actionable, and a medical-adjacent system that cannot explain its own trigger is
    not deployable.
    """

    schema_version: str = SCHEMA_VERSION
    alert_id: str
    kind: AlertKind
    severity: AlertSeverity
    subject_role: Role
    day: date
    triggered_at: datetime
    rule_name: str
    rule_expression: str = Field(
        description="Human-readable trigger condition, e.g. "
        "'robust_z(walking_duration_s) < -3.0 for 3 consecutive days'"
    )
    evidence: list[Evidence] = Field(min_length=1)
    detection_lead_days: int | None = Field(
        default=None,
        description="Days between this alert and the ground-truth onset. Only populated "
        "in simulator evaluation; the headline metric for the behaviour layer.",
    )

    @field_validator("evidence")
    @classmethod
    def _require_evidence(cls, v: list[Evidence]) -> list[Evidence]:
        if not v:
            raise ValueError("an alert without evidence is not auditable")
        return v


class BehaviourState(BaseModel):
    """Complete Agent 3 output and the sole input to Agent 4.

    Agent 4 sees this object and nothing else - never pixels, never raw video. That
    boundary is what makes the auditability claim meaningful.
    """

    schema_version: str = SCHEMA_VERSION
    subject_role: Role
    report_day: date
    today: DailyFeatures
    baselines: dict[str, BaselineStats]
    deviations: dict[str, float] = Field(
        default_factory=dict, description="feature_name -> robust z-score"
    )
    drift_signals: dict[str, float] = Field(
        default_factory=dict, description="feature_name -> CUSUM statistic"
    )
    alerts: list[Alert] = Field(default_factory=list)
    history_days: int = Field(default=0, ge=0)

    def evidence_index(self) -> dict[str, Evidence]:
        """All evidence in this state, keyed by ref, for O(1) verifier lookup."""
        return {e.ref: e for e in (ev for a in self.alerts for ev in a.evidence)}


# ---------------------------------------------------------------------------
# Agent 4 - LLM reasoning (schema-constrained)
# ---------------------------------------------------------------------------


class Claim(BaseModel):
    """One factual assertion in a generated report.

    The LLM is grammar-constrained to this shape. Numeric values are echoed back
    rather than left implicit in prose specifically so they can be checked: prose like
    "walking dropped noticeably" is unverifiable, whereas claimed_value=18.2 is not.
    """

    claim_id: str
    text: str = Field(min_length=1, max_length=500)
    evidence_ref: str = Field(pattern=r"^feat:[a-z0-9_]+:\d{4}-\d{2}-\d{2}$")
    claimed_value: float | None = None
    claimed_pct_change: float | None = None
    direction: Literal["increase", "decrease", "unchanged"] | None = None


class VerificationResult(BaseModel):
    """Deterministic verifier output for one claim. No LLM involved - by design.

    Using a model to check a model shares failure modes; arithmetic does not.
    """

    claim_id: str
    ref_exists: bool
    value_matches: bool
    pct_matches: bool
    direction_consistent: bool
    notes: list[str] = Field(default_factory=list)

    @property
    def is_faithful(self) -> bool:
        return (
            self.ref_exists
            and self.value_matches
            and self.pct_matches
            and self.direction_consistent
        )


class CaregiverReport(BaseModel):
    """Final Agent 4 output, post-verification."""

    schema_version: str = SCHEMA_VERSION
    report_id: str
    subject_role: Role
    report_day: date
    generated_at: datetime
    summary: str = Field(max_length=2000)
    claims: list[Claim] = Field(default_factory=list)
    recommendation: str = Field(max_length=1000)
    escalate: bool = False
    verifications: list[VerificationResult] = Field(default_factory=list)
    model_name: str = ""
    constrained_decoding: bool = True

    @property
    def hallucination_rate(self) -> float:
        """Fraction of claims failing verification. The headline LLM metric."""
        if not self.claims:
            return 0.0
        unfaithful = sum(1 for v in self.verifications if not v.is_faithful)
        return unfaithful / len(self.claims)

    @property
    def faithful_claims(self) -> list[Claim]:
        ok = {v.claim_id for v in self.verifications if v.is_faithful}
        return [c for c in self.claims if c.claim_id in ok]


__all__ = [
    "SCHEMA_VERSION",
    "Role",
    "AlertSeverity",
    "AlertKind",
    "BoundingBox",
    "Keypoints",
    "DetectedObject",
    "PersonObservation",
    "FrameObservation",
    "ActivitySegment",
    "DailyFeatures",
    "BaselineStats",
    "Evidence",
    "Alert",
    "BehaviourState",
    "Claim",
    "VerificationResult",
    "CaregiverReport",
]
