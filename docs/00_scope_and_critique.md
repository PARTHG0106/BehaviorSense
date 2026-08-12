# BehaviorSense AI — Phase 0: Critical Review, Scope Lock, and Contribution Statement

Status: DRAFT for approval
Date: 2026-08-03

---

## 1. Executive verdict

The proposed system is **architecturally sound but scoped ~4x beyond a 1-month budget**, and it
contains **one fatal, unstated assumption**: that a dataset exists which supports supervised
learning of *longitudinal behaviour* for *identified elderly individuals* in *home CCTV*.

No such public dataset exists. Not gated — *nonexistent*.

This document reframes the project so that it is (a) buildable in 4 weeks, (b) honest about what
is learned vs. simulated, and (c) **more** publishable than the original framing, not less.

---

## 2. The three fatal risks

### RISK-1 (CRITICAL) — The longitudinal data gap

Objectives 7 and 8 (learn daily/weekly routine, detect mobility decline, meal skipping) require
**weeks-to-months of continuous, identity-resolved video of the same individual**. Survey of
available data:

| Requirement | Best available | Verdict |
|---|---|---|
| Elderly subjects, real home | Toyota Smarthome (18 subjects, 60–80yo) | Gated, ~1–2wk approval |
| Untrimmed long video | Toyota Smarthome Untrimmed (536 videos, ~21 min avg) | Gated |
| Multi-day continuity per subject | — | **Does not exist publicly** |
| Identity labels across days | — | **Does not exist publicly** |
| Clinical outcome labels (decline, falls-in-situ) | — | **Does not exist publicly** |

Longest public untrimmed ADL video is ~21 minutes. "Weekly routine learning" cannot be
supervised, validated, or even sanity-checked against it.

**Consequence if ignored:** Agent 3 becomes an unfalsifiable component. A dissertation examiner
or reviewer will ask "how did you validate the baseline model?" and there is no answer. This is
the single most common failure mode of student behaviour-monitoring projects.

**Mitigation (adopted):** split the system at the *behavioural state* boundary.
- Everything **below** the boundary (detect → track → ReID → pose → activity) is trained and
  evaluated on **real video with real ground truth**.
- Everything **above** the boundary (routine baseline, drift, anomaly, report) is evaluated on a
  **programmatic longitudinal simulator** with *known injected ground-truth anomalies*, plus a
  real-video smoke test.
- The simulator is declared, versioned, and released. **It becomes a contribution, not a hack.**

### RISK-2 (CRITICAL) — Hardware claim does not match Kaggle

The spec states training on "NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM, running on Kaggle."

Kaggle does not offer RTX PRO 6000. Kaggle accelerators are: **P100 16GB**, **T4 ×2 (16GB each)**,
and **TPU v3-8 / v5e-8**. A 96GB Blackwell card is a local/institutional or cloud rental machine.

This matters because it changes almost every downstream decision:

| If hardware is... | Then |
|---|---|
| Kaggle T4 ×2 (16GB) | fp16 (no bf16 on T4), batch ≤ 32 for skeleton models, 7B LLM needs 4-bit quant, 12h session cap, must checkpoint/resume |
| Local RTX PRO 6000 96GB | bf16, batch 256+, 7B LLM in bf16 unquantised, no session cap, can fine-tune LLM |

Also: Blackwell (sm_120) requires **CUDA ≥12.8 and PyTorch ≥2.7**. Older wheels will not run. In
an offline environment this must be pre-baked into the asset bundle or training will fail on
import — a failure mode that costs a full day if discovered late.

**Action required:** confirm actual hardware before Phase 3.

### RISK-3 (HIGH) — The LLM agent as currently specified is not evaluable

"Agent 4 explains alerts" is a demo feature, not research. Every reviewer asks the same question:
**how do you know the explanation is true?** Free-text LLM output over structured input
hallucinates ~10–30% of the time in the numeric-grounding regime this system operates in.

**Mitigation (adopted):** constrain Agent 4 to emit **structured, schema-validated JSON** where
every claim carries an `evidence_ref` pointing at a row in the behavioural state store. Then
implement an automatic **faithfulness checker** that re-verifies each claim against the store and
computes a hallucination rate. Report that number.

This converts the weakest component into **the most publishable one** (see §5).

---

## 3. Additional design challenges

### 3.1 Face recognition for identity is the wrong tool here
Objective 5 ("identify monitored elderly resident") implicitly suggests face recognition. Reject:
- Ceiling-mounted CCTV rarely yields >40px inter-ocular distance → face recognition degrades hard.
- Toyota Smarthome **blurs all faces**; most ADL datasets do the same. You cannot train or test it.
- It is the highest-risk component ethically and the hardest to defend in an ethics review.

**Adopted instead:** appearance ReID (OSNet) + soft-biometric priors + spatio-temporal continuity,
resolved to **roles** not identities: `RESIDENT | VISITOR | CARER | UNKNOWN`. Enrollment is a small
labelled gallery per deployment. Open-set: unknown-person rejection by distance threshold.
This is more defensible, more deployable, and still fully satisfies objectives 3–5.

### 3.2 Do not train a video backbone from scratch
VideoMAE / SlowFast fine-tuning on raw RGB for 20 classes is 2–5 GPU-days *and* needs multi-TB
video I/O. In 4 weeks with a preprocessing pipeline to also build, this is a schedule bomb.

**Adopted instead:** **skeleton-first** action recognition. Extract 2D poses once, offline, cache
as small tensors (~1000× smaller than video), then train ST-GCN++/SkateFormer-class models in
**minutes-to-hours**. Skeleton models are also privacy-preserving by construction — a genuine
selling point for elderly-care deployment and a defensible research angle.
Keep a small RGB branch only for object-context classes (medicine, TV, cooking) where skeletons
are insufficient.

### 3.3 The label-space problem nobody plans for
Charades has 157 classes. Fall datasets have 2–16. Toyota has 31. NTU has 120. None of them is an
*ADL clinical taxonomy*. Naively training on any one gives a model that cannot serve the
behavioural layer.

**Adopted:** define a **unified 20-class ADL taxonomy** mapped from all sources, published as a
YAML mapping file with per-source provenance. The mapping itself is a legitimate contribution and
directly enables cross-dataset evaluation.

---

## 4. Locked scope (what we build in 4 weeks)

| # | Objective | Status | Evaluation |
|---|---|---|---|
| 1 | Person detection | BUILD (pretrained) | MOT17 / COCO-person |
| 2 | Multi-person tracking | BUILD (pretrained) | MOT17 HOTA/MOTA/IDF1 |
| 3 | ReID across re-entry | BUILD (fine-tune) | Market-1501 mAP/R1 |
| 4 | Role assignment (not face ID) | BUILD | Held-out gallery, open-set F1 |
| 5 | Pose estimation | BUILD (pretrained) | COCO-keypoints AP (sanity) |
| 6 | ADL recognition, 20-class | **TRAIN — core ML** | Cross-subject + cross-dataset |
| 7 | Fall detection | **TRAIN — core ML** | Cross-domain (train staged → test wild) |
| 8 | Routine baseline + drift | BUILD (statistical) | Simulator, injected GT |
| 9 | Anomaly detection | BUILD (statistical) | Simulator, PR-AUC, lead time |
| 10 | LLM report + faithfulness | **BUILD + EVAL — core novelty** | Hallucination rate, expert rubric |
| — | Speech / audio | **CUT** | out of scope |
| — | Emotion recognition | **CUT** | it is what we pivoted away from |
| — | Age estimation | **CUT** | no value to the pipeline |
| — | Gait ID | **DEFER** | stretch goal only |

Cut items are cut because they add integration surface without adding to the contribution.

---

## 5. Contribution statement (what makes this publishable)

Three defensible claims, in descending strength:

1. **Grounded, verifiable LLM reporting over structured behavioural state.**
   A schema-constrained generation protocol with automatic claim-level faithfulness verification,
   and a measured hallucination rate. Most "LLM + healthcare monitoring" papers report no such
   metric. This is a real gap.

2. **Cross-domain generalisation of fall/ADL recognition under identity-aware tracking.**
   Train on staged data, test on wild data (OmniFall's OOPS split). Staged-to-wild transfer is a
   known, unsolved, and quantifiable problem.

3. **An open longitudinal behavioural simulator + unified 20-class ADL taxonomy**, enabling
   reproducible evaluation of routine-drift detection in the absence of multi-week real data.

Target venues: a workshop at a CV or health-informatics venue is realistic for a 1-month project.
Full-conference submission would need the real-data component extended.

---

## 6. Open questions blocking Phase 3

1. **Actual training hardware** — Kaggle T4×2/P100, or a real 96GB Blackwell machine?
2. **Toyota Smarthome** — do you have an academic email? If yes, submit the request on Day 1; it
   arrives in time to matter. If no, the plan holds without it.
3. **Deployment demo target** — live webcam, or offline video file processing? Changes the
   FastAPI/streaming design substantially.
