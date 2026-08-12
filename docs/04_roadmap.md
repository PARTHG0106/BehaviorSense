# BehaviorSense AI — 4-Week Roadmap

Ordering principle: **every week ends with something demonstrable.** If Week 4 vanishes, Week 3's
output is still a defensible project. No component depends on a later component to be evaluable.

---

## Week 1 — Foundation & Data (the week that decides the project)

| Day | Task | Deliverable | Risk |
|---|---|---|---|
| 1 | Preflight: verify Blackwell sm_120 + torch/CUDA versions. **Submit Toyota Smarthome + NTU requests.** | `preflight_report.json` | HIGH — version mismatch |
| 1 | Repo scaffold, configs, schemas, SQLite DDL | runnable package | — |
| 2 | Stage A1–A2: download + inventory all Tier-1 data | `raw_manifest.json` | MED — dead links |
| 2 | Build full Charades 157→20 map | `taxonomy.yaml` complete | — |
| 3 | Stage A3–A5: decode + RTMO pose + RT-DETR objects | pose cache | HIGH — throughput |
| 4 | Stage A6–A10: map, window, normalise, split, pack | `bsai-tensors` v1 | — |
| 5 | Publish datasets; verify offline mount + hashes | offline preflight passes | HIGH — the whole contract |
| 6 | Agent 1 end-to-end on one video | annotated MP4 | — |
| 7 | Buffer / catch-up | — | — |

**Week 1 exit gate:** an offline GPU notebook loads packed tensors, prints shapes, and trains one
epoch of a dummy model. If this fails, nothing downstream works. Do not proceed past it.

---

## Week 2 — Perception & Activity (the core ML)

| Day | Task | Deliverable | Metric |
|---|---|---|---|
| 8 | ST-GCN++ implementation + training loop | `train_adl.py` | trains |
| 9 | Train ADL 20-class, P1 cross-subject | checkpoint + metrics | macro-F1 |
| 10 | Train fall model, focal loss, class-balanced | checkpoint | recall@fall, PR-AUC |
| 11 | Evaluate P2 cross-dataset + P3 staged→wild | `results_adl.json` | the honest numbers |
| 12 | OSNet ReID fine-tune on Market-1501 | checkpoint | mAP, Rank-1 |
| 13 | BoT-SORT integration + open-set role assignment | `agent1_perception.py` | IDF1, open-set F1 |
| 14 | Tracker eval on MOT17; ablation: ByteTrack vs BoT-SORT | `results_tracking.json` | HOTA/MOTA/IDF1 |

**Week 2 exit gate:** video in → identity-resolved activity timeline out, with published metrics on
three protocols. **This alone is a complete final-year project.** Everything after is upside.

---

## Week 3 — Behaviour & Reasoning (the novelty)

| Day | Task | Deliverable |
|---|---|---|
| 15 | Longitudinal simulator: 90 days × 4 personas, injected anomalies | `simulator.py` + GT |
| 16 | Feature extraction: daily vectors from activity timelines | `agent3_behaviour.py` |
| 17 | Robust baseline (median/MAD) + robust-z deviation + CUSUM drift | anomaly scores |
| 18 | Rule layer: fall, prolonged inactivity, meal skip, mobility decline | `Alert` objects |
| 19 | Evaluate anomaly detection vs injected GT | PR-AUC, **detection lead time (days)** |
| 20 | Agent 4: Qwen2.5-7B + `outlines` schema-constrained generation | `agent4_llm.py` |
| 21 | **Faithfulness verifier + hallucination-rate measurement** | `results_llm.json` |

**Week 3 exit gate:** structured caregiver report with every claim verified against the state store,
and a measured hallucination rate. This is the publishable contribution.

*Detection lead time is the metric to lead with* — "detected mobility decline N days before the
injected event became clinically visible" is far more compelling to a health audience than F1.

---

## Week 4 — Integration, Evaluation, Writing

| Day | Task | Deliverable |
|---|---|---|
| 22 | FastAPI service + SQLite persistence + job queue | `api/` |
| 23 | Dashboard: timeline, behavioural profile, alerts, report | `ui/` |
| 24 | Docker + compose + offline model mount | `Dockerfile` |
| 25 | SkateFormer swap → backbone comparison table | ablation results |
| 26 | Full ablation suite (see below) | `results_ablations.json` |
| 27 | README, architecture diagrams, demo video, figures | portfolio-ready |
| 28 | Dissertation draft: lit review, method, results, limitations | document |

**Stretch (only if Days 22–24 finish early):** live webcam mode.

---

## Ablations to run (these are what reviewers actually look for)

| Ablation | Question it answers | Why it matters |
|---|---|---|
| Skeleton-only vs +object-context | Does object fusion earn its cost? | Should show `taking_medication` F1 jumping — validates §2.5 |
| ST-GCN++ vs SkateFormer | Does a SOTA backbone help *this* task? | May show it doesn't — that's a finding |
| RTMO vs ViTPose poses | Is pose quality the accuracy ceiling? | Tells you where to invest next |
| ByteTrack vs BoT-SORT | Does ReID-in-tracker reduce ID switches? | Justifies the choice empirically |
| Closed-set vs open-set ReID | Cost of forcing a match on strangers | Safety-relevant |
| Window 32/64/128 frames | Temporal context vs latency | Standard, cheap |
| CAUCAFall lighting conditions | Robustness to illumination | Uses metadata almost nobody uses |
| Free-text vs schema-constrained LLM | Does constraining reduce hallucination? | **The headline result** |
| Baseline window 7/14/30 days | Sensitivity vs stability of drift detection | Practically important |

---

## Risk register

| Risk | P | Impact | Mitigation | Trigger |
|---|---|---|---|---|
| Blackwell/torch version mismatch | M | CRITICAL | Pre-baked wheels in assets | Day 1 preflight |
| Pose extraction too slow | M | HIGH | GPU session w/ internet; ONNX-CPU fallback | Day 3 >2h for 10% |
| Charades map too noisy | M | HIGH | Restrict to high-confidence subset; report coverage | Day 9 macro-F1 <0.40 |
| Fall recall too low | M | HIGH | Raise class weight, lower threshold, accept FP | Day 10 recall <0.80 |
| LLM ignores schema | L | MED | `outlines` grammar-constrained decoding (hard guarantee) | Day 20 |
| Gated data never arrives | H | LOW | Plan already excludes it | — |
| 12h session kills a run | L | MED | Epoch checkpointing + auto-resume | — |
| Scope creep | **H** | **CRITICAL** | Weekly exit gates; cut list is final | Any "wouldn't it be cool if" |

Scope creep is the highest-probability project-killer. The cut list (audio, emotion, age, gait,
face ID) is closed. Reopening any item means dropping something from Weeks 3–4.

---

## Definition of done

- [ ] Offline training runs end-to-end with zero network calls
- [ ] ADL model: P1 / P2 / P3 all reported
- [ ] Fall model: recall ≥0.85 staged, cross-domain number reported honestly
- [ ] ReID: Market-1501 mAP/R1 + open-set metrics
- [ ] Tracking: HOTA/MOTA/IDF1 on MOT17
- [ ] Anomaly detection: PR-AUC + lead time vs simulator GT
- [ ] LLM: hallucination rate, constrained vs unconstrained
- [ ] FastAPI + dashboard + Docker
- [ ] 9 ablations
- [ ] README, diagrams, demo video
- [ ] Limitations section written honestly
