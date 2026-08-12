# BehaviorSense AI — Architecture & Model Selection

Every choice below is justified against the alternatives you listed. Hardware assumption:
**RTX PRO 6000 Blackwell 96GB on Kaggle, bf16, 12h session cap, no internet at train time.**

---

## 1. System architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│  ONLINE (internet)          │  OFFLINE (no internet)                      │
│  Kaggle CPU / TPU           │  Kaggle GPU 96GB                            │
│  ─ download                 │  ─ train ADL model                          │
│  ─ decode / extract poses   │  ─ train fall model                         │
│  ─ map to 20-class taxonomy │  ─ fine-tune ReID                           │
│  ─ pack .npz shards         │  ─ evaluate                                 │
│  └──► Kaggle Dataset ───────┼──►  (mounted read-only)                     │
└───────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
╔═══════════════════════════════════════════════════════════════════════════╗
║                          INFERENCE PIPELINE                               ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ video frame                                                               ║
║     │                                                                     ║
║     ▼                                                                     ║
║ ┌─────────────────────── AGENT 1: PERCEPTION ────────────────────────┐    ║
║ │  RTMO (detect + pose, one pass)                                    │    ║
║ │       │                                                            │    ║
║ │       ├──► BoT-SORT tracking ──► track_id                          │    ║
║ │       │         │                                                  │    ║
║ │       │         └──► OSNet ReID embed ──► gallery match            │    ║
║ │       │                     │                                      │    ║
║ │       │                     ▼                                      │    ║
║ │       │            role: RESIDENT|VISITOR|CARER|UNKNOWN            │    ║
║ │       └──► RT-DETR objects (cup, bottle, tv, ...) [1 Hz only]      │    ║
║ └────────────────────────────┬───────────────────────────────────────┘    ║
║                              ▼  PersonObservation                         ║
║ ┌─────────────────── AGENT 2: ACTIVITY ──────────────────────────────┐    ║
║ │  64-frame skeleton window ──► ST-GCN++ ──► 20-class ADL logits     │    ║
║ │                                   ▲                                │    ║
║ │           object-context features ┘  (late fusion)                 │    ║
║ │  ──► temporal smoothing (HMM/Viterbi) ──► ActivitySegment           │    ║
║ └────────────────────────────┬───────────────────────────────────────┘    ║
║                              ▼  activity timeline per role                ║
║ ┌─────────────────── AGENT 3: BEHAVIOUR ─────────────────────────────┐    ║
║ │  daily feature vector (durations, counts, transitions, mobility)   │    ║
║ │  ──► rolling baseline (robust median/MAD, 14d window)              │    ║
║ │  ──► deviation scoring (robust z) + CUSUM drift                    │    ║
║ │  ──► rule layer (fall, prolonged inactivity, meal skip)            │    ║
║ └────────────────────────────┬───────────────────────────────────────┘    ║
║                              ▼  BehaviourState + Alert (structured)       ║
║ ┌─────────────────── AGENT 4: LLM REASONING ─────────────────────────┐    ║
║ │  Qwen2.5-7B-Instruct, JSON-schema-constrained                       │    ║
║ │  input: STRUCTURED STATE ONLY — never pixels                       │    ║
║ │  output: {claims:[{text, evidence_ref, ...}], recommendation}      │    ║
║ │  ──► FAITHFULNESS VERIFIER ──► reject/flag ungrounded claims       │    ║
║ └────────────────────────────┬───────────────────────────────────────┘    ║
║                              ▼   caregiver report + audit trail           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

**Key structural decision:** agents communicate through **versioned Pydantic schemas persisted to
SQLite**, not function calls. Each agent is independently testable and replayable — you can rerun
Agent 3 and 4 in seconds without re-running perception. This is what makes a 4-week timeline
survivable, and it directly satisfies "each module independently testable."

---

## 2. Model selection — decisions and rejected alternatives

### 2.1 Detection + Pose → **RTMO-l** (single model for both)

| Candidate | Verdict |
|---|---|
| **RTMO-l** ✅ | **CHOSEN.** One-stage: detection + pose in a single pass. 74.8 AP COCO, 141 FPS on V100. Faster than top-down beyond ~4 people (CVPR 2024). |
| RTMPose + separate detector | Rejected: two models, two failure modes; top-down cost scales linearly with people. Multi-person is our normal case. |
| ViTPose++ | Rejected as primary: highest accuracy (79.1 AP) but needs a detector and is slow. **Keep as an offline "oracle" to bound how much pose quality limits ADL accuracy** — that's a good ablation. |
| YOLOv8/11-pose | Rejected: measured lowest accuracy in independent comparison (75.3% vs ViTPose 86.6%). RTMO dominates it at similar cost. |
| MediaPipe | Rejected: single-person-oriented, weak under occlusion. |
| OpenPose | Rejected: 2019-era, superseded on both axes. |

**Why one-stage matters here:** the whole ADL model is skeleton-driven, so pose is not a side
output — it is *the* input. A detector-dependent pipeline compounds detector failures into pose
failures into action failures. RTMO removes one link from that chain.

### 2.2 Tracking → **BoT-SORT** (ByteTrack fallback)

| Candidate | Verdict |
|---|---|
| **BoT-SORT** ✅ | **CHOSEN.** ByteTrack's low-score association + camera-motion compensation + **native ReID embedding integration**. That last point is decisive: our identity layer needs appearance embeddings anyway, so we get them for free inside the tracker. |
| ByteTrack | Strong and simpler, but motion-only — ID switches after occlusion, which is exactly our failure case (person leaves room, returns). Kept as an ablation baseline. |
| DeepSORT | Rejected: superseded; weaker association. Cite as historical baseline only. |
| Transformer trackers (MOTR etc.) | Rejected: heavy, needs training, no clear win at our scale. Wrong use of 4 weeks. |

### 2.3 ReID → **OSNet-AIN** (`osnet_ain_x1_0`)

- Omni-scale features suit indoor scale variation; ~2M params so the gallery match is nearly free.
- Fine-tune on Market-1501, report mAP/Rank-1 for literature comparability.
- **Critical addition — open-set:** standard ReID is closed-set (always returns nearest). Home
  monitoring must say *"I don't know."* We threshold cosine distance and emit `UNKNOWN`, then
  report open-set metrics. Most student projects miss this; it is a genuine correctness issue.

### 2.4 Identity → **role classification, not face recognition**

Rejected face recognition outright: CCTV gives too few facial pixels, ADL datasets blur faces
(so it is untrainable *and* untestable here), and it maximises ethics exposure.

Adopted: ReID embedding + enrolled gallery + spatio-temporal continuity → `RESIDENT | VISITOR |
CARER | UNKNOWN`. Defensible, deployable, satisfies objectives 3–5.

### 2.5 Activity recognition → **ST-GCN++ (skeleton) + object-context late fusion**

This is the highest-leverage decision in the project.

| Candidate | Verdict |
|---|---|
| **ST-GCN++ / PoseConv3D** ✅ | **CHOSEN.** Input is ~64×17×3 floats, not video. Trains in **minutes-to-hours**, not GPU-days. Privacy-preserving by construction. Strong NTU-120 performance. |
| SkateFormer (ECCV 2024) | **Chosen as the upgrade path.** SOTA-class skeletal-temporal transformer. Implement ST-GCN++ first (safe), swap in SkateFormer in Week 3 → gives a clean "our method vs. SOTA backbone" comparison table. |
| VideoMAE-v2 | Rejected as primary: 2–5 GPU-days + multi-TB video I/O. **Keep as an RGB baseline on a class subset** to justify the skeleton choice empirically rather than by assertion. |
| SlowFast | Rejected: superseded by VideoMAE; same I/O problem. |
| ActionFormer | Rejected as the recogniser, but **its temporal-localisation formulation is the right reference** for untrimmed segmentation. We use HMM smoothing instead (far cheaper); cite ActionFormer as the alternative. |
| ProtoGCN (CVPR 2025) | Noted as current SOTA-adjacent; mention in lit review, not in the 4-week build. |

**The tradeoff, stated honestly:** skeletons discard object context. "Taking medicine" vs.
"drinking water" is nearly identical in joint space. That is precisely why RT-DETR object detection
runs at 1 Hz and its outputs late-fuse into the classifier. Skeleton-only would fail the medication
class — a limitation worth measuring and reporting as an ablation.

### 2.6 Objects → **RT-DETR-L**, sampled at 1 Hz

| Candidate | Verdict |
|---|---|
| **RT-DETR-L** ✅ | **CHOSEN.** NMS-free (deterministic, no threshold tuning), beats YOLOv8 on COCO at comparable latency. Apache-2.0. |
| YOLOv8/v11 | Viable, but **AGPL-3.0** — a real problem for a public portfolio repo, and NMS adds a tuned hyperparameter. Explicitly noting this because you asked me not to assume YOLO. |
| Grounding DINO | Rejected for the loop: open-vocab is appealing but ~10× slower. **Use offline for auto-labelling novel objects** — good use of the tool. |
| SAM2 / FastSAM | Rejected: segmentation masks add cost without adding signal for our features. SAM2's video tracking overlaps BoT-SORT. |

Static home scenes → 1 Hz is ample and cuts object-detection cost ~30×.

### 2.7 Behaviour analysis → **statistical, not deep**

Deliberately rejecting a learned behaviour model. With no multi-week real data, a neural baseline
model would be trained on simulated data and validated on simulated data — circular and
unpublishable. Instead:

- **Baseline:** rolling 14-day median + MAD per feature (robust to outliers; mean/σ breaks with n≈14).
- **Deviation:** robust z-score `0.6745(x−median)/MAD`.
- **Drift:** CUSUM for slow decline (mobility reduction over weeks) — the classic tool for exactly this.
- **Rules:** falls, prolonged inactivity, meal skipping — deterministic, auditable, clinically legible.

Interpretable, auditable, and honest about what is learned. A caregiver-facing medical system that
cannot explain its own trigger is not deployable; this design makes every alert traceable to a
number.

### 2.8 LLM → **Qwen2.5-7B-Instruct**, schema-constrained

| Candidate | Verdict |
|---|---|
| **Qwen2.5-7B-Instruct** ✅ | **CHOSEN.** Apache-2.0, strong structured-output/JSON adherence, fits in 96GB bf16 unquantised, downloadable to a Kaggle dataset for offline use. |
| Llama-3.1-8B | Close second; Llama license is more restrictive. Keep as swap-in comparison. |
| Qwen2.5-14B/32B | Fits in 96GB but slower; use only if 7B faithfulness is inadequate. |
| Qwen-VL / InternVL | **Rejected — and this is a design principle, not a resource constraint.** Your spec says the LLM must not perform perception. A VLM would blur that boundary and destroy the auditability claim. Correct call in your original spec; I'm reinforcing it. |
| GPT/Claude API | Rejected: no internet at train time, cost, non-reproducible, data leaves the device — unacceptable for elderly-care video. |

**No fine-tuning.** Prompt engineering + constrained decoding is sufficient and leaves the time
budget intact. If Week 4 has slack, LoRA on synthetic report pairs is the stretch goal.

---

## 3. The novel component: faithfulness verification

This is what elevates the project from integration to research.

```
BehaviourState (SQLite)
        │
        ▼
  prompt: only structured facts, each with a stable id
        │
        ▼
  Qwen2.5-7B  +  JSON schema constraint (outlines / guided decoding)
        │
        ▼
  {"claims":[{"claim_id":"c1",
              "text":"Walking duration fell 42% vs baseline",
              "evidence_ref":"feat:walk_duration:2026-08-01",
              "claimed_value":18.2,
              "claimed_delta_pct":-42.0}], ...}
        │
        ▼
  ┌──────────── VERIFIER (deterministic, no LLM) ────────────┐
  │ for each claim:                                          │
  │   1. does evidence_ref exist in the store?               │
  │   2. does claimed_value match the stored value (±ε)?     │
  │   3. does claimed_delta_pct match recomputed delta?      │
  │   4. is the direction word consistent with the sign?     │
  └──────────────────────────────────────────────────────────┘
        │
        ▼
  metrics: claim-level precision, hallucination rate,
           unsupported-claim rate, numeric-error rate
```

**Reportable result:** *"Constrained generation + verification reduced ungrounded clinical claims
from X% to Y% across N generated reports."* That is a concrete, quantitative, novel contribution —
and it is the sort of number the LLM-in-healthcare literature is conspicuously missing.

---

## 4. Optimisation plan (96GB Blackwell, 12h cap)

| Technique | Applies to | Note |
|---|---|---|
| **bf16 autocast** | all training | Blackwell native. Prefer over fp16 — no loss-scaling instability. |
| **Large batch** | skeleton models | Skeleton tensors are tiny; batch 512–1024 fits easily. Scale LR by √k. |
| **`torch.compile`** | ST-GCN++/SkateFormer | 20–40% speedup. Compile once, reuse. |
| **channels_last** | CNN/ReID | Better tensor-core utilisation. |
| **Gradient checkpointing** | only if RGB baseline is run | Unnecessary for skeleton models — don't pay the 30% cost for nothing. |
| **Pre-extracted pose cache** | everything | **The single biggest win.** Poses extracted once online; GPU training never decodes video. Turns a 3-day job into a 3-hour one. |
| **Memory-mapped `.npz` shards** | data loading | Zero-copy, no decode, `num_workers=8`. |
| **Checkpoint every epoch + auto-resume** | mandatory | 12h session cap makes this non-optional. Resume from `/kaggle/working` or a versioned output dataset. |
| **DDP** | not needed | Single GPU. Code will be DDP-compatible but we won't use it. |

⚠️ **Blackwell is sm_120** — requires CUDA ≥12.8 and PyTorch ≥2.7. Verify `torch.cuda.get_device_capability()`
on Day 1. If Kaggle's default image is older, the correct wheels must be pre-baked into the offline
asset bundle. Discovering this in Week 3 costs a day.

---

## 5. Why this beats the original proposal

| Original | Revised | Reason |
|---|---|---|
| RGB video action recognition | Skeleton-first + object fusion | ~100× cheaper, privacy-preserving, feasible in 4 weeks |
| Face-based identity | Role-based ReID, open-set | Trainable and testable on available data; ethically defensible |
| Learned behaviour model | Statistical + simulator with injected GT | Falsifiable; avoids training and validating on the same simulation |
| Free-text LLM explanation | Schema-constrained + verified | Turns the weakest component into the publishable one |
| YOLO assumed | RT-DETR (Apache-2.0, NMS-free) | Avoids AGPL contamination of a public repo |
| 10 modalities | 2 (RGB + derived skeleton) | Integration surface is what kills 4-week projects |
