https://vsllm.com


# EmotionSense-Extended

**AI-Based Multimodal Well-Being Monitoring System for Elderly Care**

This repository extends the original emotion-recognition scope into a full multimodal
well-being monitoring platform for elderly care. Emotion is now one weak, optional signal
among many, not the product. The product is **explainable, longitudinal well-being
monitoring for caregivers**.

> **Positioning.** This is decision-support software. It surfaces observations and risk
> signals for a human caregiver. It does not diagnose, prescribe, or take autonomous
> health decisions.

---

## 1. What the system does

| Capability | Description |
| --- | --- |
| Continuous perception | Person detection, tracking, re-identification, pose estimation from CCTV |
| Identity persistence | Resident vs. visitor, robust to clothing change, occlusion, low resolution |
| Activity understanding | ADLs, posture states, transitions, room events, falls |
| Behavior understanding | Personal routine baselines, mobility trends, anomaly scoring |
| Reasoning and reporting | Grounded natural-language alert explanations, daily/weekly reports |
| Caregiver interface | Calm clinical dashboard, event timeline, alert triage, notes |

## 2. Layered architecture

```
Cameras (RTSP) / optional mic / optional IoT sensors
        |
  L1 PERCEPTION      detect -> track -> ReID -> pose -> identity resolve
        |  PerceptionFrame (per frame, per track)
  L2 ACTIVITY        posture states, transitions, ADLs, falls, room events
        |  ActivityEvent (timestamped, confidence, evidence refs)
  L3 BEHAVIOR        personal baselines, routine model, anomaly scoring
        |  BehaviorProfile + AnomalySignal
  L4 REASONING       alert assembly + LLM explanation over structured facts only
        |  Alert + Explanation + Report
  DASHBOARD / NOTIFICATIONS / REPORTS
```

**Hard interface rule:** every layer consumes only the typed contract of the layer below
it. Only L1 touches pixels. The LLM in L4 never sees a frame and never performs
detection. This is what makes the system testable, module-replaceable, and auditable.

## 3. Repository layout

```
configs/            YAML configuration; no thresholds hardcoded in code
docs/               architecture, datasets, training, API, schema, safety, evaluation
kaggle/prep/        internet-enabled dataset download / convert / package scripts
kaggle/train/       offline GPU training scripts (no network calls, fixed seeds)
sql/                PostgreSQL schema
src/wellbeing/
  contracts/        pydantic contracts shared by every layer (single source of truth)
  perception/       detector, tracker, reid, pose, identity resolver
  activity/         posture, transitions, ADL, fall detection, event assembly
  behavior/         features, baselines, trends, anomaly detection
  reasoning/        alert builder, explanation renderer, reports
  storage/          SQLAlchemy models and repositories
  api/              FastAPI application
  pipeline/         L1 -> L4 orchestration
tests/              unit and integration tests
dashboard/          React caregiver dashboard (calm clinical design system)
```

## 4. Documentation index

| Doc | Contents |
| --- | --- |
| [01-architecture.md](docs/01-architecture.md) | Layers, agents, data flow, failure modes |
| [02-data-contracts.md](docs/02-data-contracts.md) | The typed contracts between layers |
| [03-dataset-plan.md](docs/03-dataset-plan.md) | Datasets per task and why each helps elderly care |
| [04-benchmark-plan.md](docs/04-benchmark-plan.md) | Baselines, ablations, selection criteria |
| [05-kaggle-workflow.md](docs/05-kaggle-workflow.md) | Offline GPU training + internet-enabled prep notebook |
| [06-api-design.md](docs/06-api-design.md) | REST + WebSocket surface |
| [07-database-schema.md](docs/07-database-schema.md) | PostgreSQL design, identity/event separation |
| [08-dashboard-spec.md](docs/08-dashboard-spec.md) | Caregiver UI, design tokens, alert hierarchy |
| [09-evaluation.md](docs/09-evaluation.md) | Metrics per layer and acceptance gates |
| [10-privacy-safety.md](docs/10-privacy-safety.md) | Data minimization, access control, safety rules |
| [11-deployment.md](docs/11-deployment.md) | Edge + server topology, Docker, optimization |
| [12-research-gaps.md](docs/12-research-gaps.md) | Honest weaknesses and future scope |
| [13-roadmap.md](docs/13-roadmap.md) | Build order with numeric exit criteria |

## 5. Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                    # runs fully on CPU with stub model adapters
uvicorn wellbeing.api.main:app --reload
```

Every model adapter has a deterministic **stub** implementation, so the whole pipeline and
test suite run with no GPU and no downloaded weights. Real adapters are swapped in through
configuration, not code changes.

```bash
python -m wellbeing.pipeline.replay --trace tests/fixtures/trace_day.jsonl
```

## 6. Design principles held throughout

1. Temporal context over single-frame decisions. No alert is ever raised from one frame.
2. Personal baselines over global thresholds. "Unusual" is defined per resident.
3. Low false alarms over aggressive detection. Critical alerts require corroboration.
4. Explainability is a data requirement, not a prompt trick. Every alert carries the
   evidence rows that produced it.
5. Privacy first. Raw video is transient; identity data is stored apart from event data.
6. Reproducibility. Fixed seeds, versioned datasets, offline experiment logs.
7. Graceful degradation. Losing ReID must not stop fall detection.

## 7. Three positions this project takes deliberately

1. **Emotion recognition is demoted.** Facial affect from low-resolution CCTV is
   unreliable and clinically weak. Behavior over time is the signal that matters.
2. **Identity persistence is the hard problem**, not activity classification. Everything
   downstream is per-resident. A wrong identity permanently corrupts a baseline, so the
   system prefers an explicit `unknown` over a confident guess.
3. **Fall detection is rule-first.** Learned models refine a kinematic rule; they do not
   replace it. Published fall-detection F1 scores come from staged falls in clear view and
   do not survive real homes.

## 8. Status

Initial architecture and scaffold. See [docs/13-roadmap.md](docs/13-roadmap.md) for the
incremental build order and the exit criteria gating each module.

## 9. License

MIT. See [LICENSE](LICENSE).
