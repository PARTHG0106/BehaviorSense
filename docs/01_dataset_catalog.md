# BehaviorSense AI — Dataset Catalog

Verified: 2026-08-03. Every "instant download" claim below was checked this session.

**Legend — Gate column**
- 🟢 **INSTANT** — direct download, no account, no form.
- 🟡 **ACCOUNT** — free account only (Kaggle / HuggingFace login). No human approval.
- 🔴 **APPROVAL** — form + human review. Days to weeks.

**Legend — Tier column**
- **T1** = core, plan depends on it. **T2** = valuable, use if time. **T3** = optional/stretch.

---

## 1. TIER 1 — The critical path (all ungated)

### 1.1 OmniFall — the anchor dataset for falls

| Field | Value |
|---|---|
| **Name** | OmniFall (`simplexsigil2/omnifall`) |
| **Source** | https://huggingface.co/datasets/simplexsigil2/omnifall |
| **Gate** | 🟡 ACCOUNT (HF). **Annotations only** — raw video from original sources. |
| **License** | Annotations CC BY-NC-SA 4.0. Videos retain per-source licenses. |
| **Size** | ~52k labelled segments. OF-Staged ~14h; OF-Wild ~2.7h / 818 videos; OF-Synthetic 12k videos ~17h |
| **Modalities** | RGB video + frame-level temporal labels |
| **Classes** | Unified **16-class** taxonomy (walk, sit, stand, lie, fall, etc.) |
| **Annotations** | Dense frame-level segments; official cross-subject splits |
| **Strengths** | Solves the fall-dataset fragmentation problem outright. Unifies 8 staged datasets under one taxonomy with one loader. Provides the **staged→wild** split, which is exactly the generalisation gap worth publishing on. Official splits = comparable numbers. |
| **Weaknesses** | You still must fetch most raw video yourself. Component licenses are heterogeneous — MCFD needs author permission, so **exclude MCFD** and document it. Subjects are mostly young. |
| **Use** | **Primary fall benchmark + primary source of the unified taxonomy design.** Train on OF-Staged (cross-subject), test on OF-Wild for the headline generalisation result. |
| **Difficulty** | Medium — loader is easy, video assembly is the work |
| **Preprocessing** | `load_dataset(...)` for labels → fetch component videos → pose extraction → window into clips |
| **Internet later?** | No. Fully cacheable. |

> **Why this is the single best decision in the dataset plan:** it converts "I collected some fall
> videos" into "I evaluated staged-to-wild transfer on a published benchmark with official splits."
> Reviewer-legible, and it costs no extra effort.

### 1.2 Charades — the ADL backbone

| Field | Value |
|---|---|
| **Name** | Charades (Allen AI) |
| **Source** | https://prior.allenai.org/projects/charades — 480p: `Charades_v1_480.zip`; annotations: `Charades.zip` |
| **Gate** | 🟢 INSTANT (direct zip, no form) |
| **License** | AI2 custom. **Academic/non-profit OK. No redistribution, no commercial, no public release of modified data.** |
| **Size** | 9,848 videos, ~30s avg. 480p zip ≈ 13GB |
| **Modalities** | RGB video, 27,847 text descriptions |
| **Classes** | 157 actions, 46 objects, 33 verbs — **home/indoor** |
| **Annotations** | 66,500 temporal segments (start/end), multi-label |
| **Strengths** | Genuinely domestic scenes in real homes. Untrimmed + temporally localised — trains the *segmentation* ability the behaviour layer needs. Object annotations directly support context classes (medicine/TV/cooking). Instant download. Well-established mAP protocol. |
| **Weaknesses** | Acted, not natural. Crowd-sourced actors are young — **the elderly gap is real and must be stated as a limitation.** 157 classes is long-tailed. Multi-label complicates the taxonomy map. |
| **Use** | **Primary ADL training source.** Map its 157 → our 20 classes. |
| **Difficulty** | Medium |
| **Preprocessing** | Download 480p → filter to mapped classes → pose extract → window |
| **Internet later?** | No |

⚠️ **License compliance:** "no right to distribute" means your Kaggle asset dataset holding Charades-derived tensors **must be PRIVATE**. Do not publish it. Note this in the dissertation.

### 1.3 Le2i — classic fall benchmark

| Field | Value |
|---|---|
| **Name** | Le2i Fall Detection (Charfi et al. 2013) |
| **Source** | search-data.ubfc.fr (IMVIA lab, Univ. Bourgogne) |
| **Gate** | 🟢 INSTANT |
| **License** | **CC-BY-NC-SA 3.0** — redistribution-friendly (non-commercial, share-alike) |
| **Size** | 143 fall + 79 ADL videos; 25fps @ 320×240 |
| **Modalities** | RGB |
| **Classes** | 3 fall types + 6 ADL types, 4 rooms (home, coffee room, office, lecture room) |
| **Annotations** | Fall start/end frames — **only for Home and Coffee-room subsets** |
| **Strengths** | Permissive license. Realistic rooms. Included in OmniFall so annotations are already unified. |
| **Weaknesses** | Low resolution hurts pose extraction. Small. Partial annotations. Young subjects. |
| **Use** | Core fall training data (via OmniFall) + the license-clean subset for public demos/figures |
| **Difficulty** | Low |
| **Preprocessing** | Use OmniFall's unified labels rather than parsing originals |

### 1.4 CAUCAFall — best-licensed fall data

| Field | Value |
|---|---|
| **Name** | CAUCAFall (Univ. del Cauca, 2022) |
| **Source** | https://data.mendeley.com/datasets/7w7fccy7ky/4 — DOI 10.17632/7w7fccy7ky.4 |
| **Gate** | 🟢 INSTANT |
| **License** | **CC BY 4.0 — fully open, redistributable** |
| **Size** | 10 subjects × 10 activities; 7,388 fall + 12,466 non-fall frames |
| **Modalities** | RGB video + extracted frames + per-frame labels |
| **Classes** | 5 fall types + 5 ADLs |
| **Annotations** | Per-frame fall/nofall + **lux, fall angle, camera distance** metadata |
| **Strengths** | **The only genuinely open-license fall dataset here** — safe for public repo, papers, demos. Deliberate lighting/occlusion/clothing variation. The lux + angle metadata enables a robustness ablation almost nobody runs. |
| **Weaknesses** | Small. Single camera. Young subjects. |
| **Use** | Fall training + **the robustness-vs-illumination ablation** + all public-facing figures |
| **Difficulty** | Low |

### 1.5 GMDCSA-24 — MIT-licensed, newest

| Field | Value |
|---|---|
| **Name** | GMDCSA-24 (2024) |
| **Source** | https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos |
| **Gate** | 🟢 INSTANT (git clone) |
| **License** | **MIT** |
| **Size** | 4 subjects, 3 home setups; small |
| **Modalities** | RGB (0.92MP webcam) |
| **Classes** | Fall + ADL |
| **Strengths** | MIT = zero license friction. **Low-cost webcam footage — closest proxy to real deployment hardware.** Day + low-night lighting. Published 2024 (Data in Brief, DOI 10.1016/j.dib.2024.110892). |
| **Weaknesses** | Very small — test/robustness only, not training. |
| **Use** | **Held-out realistic test set.** Low quality is the point: it tests deployment robustness. |
| **Difficulty** | Low |

### 1.6 Market-1501 — ReID

| Field | Value |
|---|---|
| **Name** | Market-1501 |
| **Source** | Kaggle mirror `rayiooo/reid_market-1501`; also standard academic mirrors |
| **Gate** | 🟡 ACCOUNT (Kaggle) |
| **License** | Research only. No redistribution/commercial. |
| **Size** | 32,668 images, 1,501 IDs, 6 cameras (~1GB) |
| **Modalities** | RGB crops |
| **Classes** | 1,501 identities. Train 12,936 img / 751 ID; test 19,732 img / 750 ID; query 3,368 |
| **Annotations** | ID + camera ID; DPM-detected gallery boxes (realistic misalignment) |
| **Strengths** | The standard ReID benchmark — mAP/Rank-1 are directly comparable to all literature. Already on Kaggle = trivially available offline. Detector-noise in gallery matches our real pipeline. |
| **Weaknesses** | Outdoor pedestrians, not indoor residents. Domain gap to home CCTV is significant and must be acknowledged. |
| **Use** | Pretrain/validate the OSNet ReID embedder; report standard metrics for credibility |
| **Difficulty** | Low |

> **Do NOT use DukeMTMC-reID.** Withdrawn by its authors over ethics concerns. Using it in an
> elderly-care ethics-sensitive project would be a straightforward reviewer rejection.

---

## 2. TIER 2 — Strong additions

### 2.1 UR Fall Detection (URFD)
- **Source:** Univ. of Rzeszów (fenix.ur.edu.pl/~mkepski/ds/uf.html) — 🟢 INSTANT
- **License:** publicly downloadable; explicit license not stated — **verify before redistributing**
- **Content:** 30 falls + 40 ADL, 2 Kinect cameras, RGB + depth + accelerometer
- **Why:** most-used fall dataset in the literature (~40 studies) → comparability. Depth enables a fair privacy-preserving-modality comparison.
- **Weakness:** tiny; license ambiguity → keep out of any redistributed bundle.

### 2.2 MOT17 — tracking evaluation
- **Source:** motchallenge.net — 🟡 ACCOUNT
- **License:** CC BY-NC-SA
- **Content:** 14 sequences, dense pedestrian boxes + track IDs
- **Why:** the only way to report credible HOTA/MOTA/IDF1 for the tracker. Without it, tracking quality is an unsupported assertion.
- **Weakness:** street scenes, not homes. Use for *tracker validation only*, not domain training.

### 2.3 COCO (person + keypoints)
- **Source:** cocodataset.org — 🟢 INSTANT
- **License:** **CC BY 4.0**
- **Content:** 200k+ labelled people; 17-keypoint annotations
- **Why:** sanity-check detector/pose AP; source of pretrained weights; the `person` + `bottle`/`cup`/`tv` classes support object-context features.
- **Note:** val2017 (~1GB) is sufficient — do not pull the 19GB train set.

### 2.4 Toyota Smarthome / TSU — 🔴 APPROVAL (submit Day 1, do not depend on)
- **Source:** https://project.inria.fr/toyotasmarthome/ — form + academic email
- **Content:** **18 subjects aged 60–80** — real elderly. Trimmed: 16,115 RGB+D clips / 31 activities. Untrimmed (TSU): 536 videos, ~21min avg, 51 densely-annotated activities.
- **Why it matters:** the only real elderly ADL video benchmark. If approval lands by Week 3, it becomes the headline external-validity result — *"trained on young-adult data, evaluated on genuine elderly subjects"* is a strong domain-shift finding.
- **Why we don't depend on it:** 1–2 week human review, may be refused, faces are blurred.
- **Bonus:** INRIA publishes **pre-extracted I3D features + a trained model** — if approval lands late, features alone still yield a fast result.

### 2.5 NTU RGB+D 120 — 🔴 APPROVAL
- **Source:** ROSE Lab, NTU Singapore — register + form + agreement
- **Content:** 120 action classes, 3D skeletons, 106 subjects
- **Why:** the standard skeleton-action benchmark; pretraining on it materially improves small-data ADL accuracy.
- ⚠️ **License warning:** deriving new datasets and redistribution are explicitly prohibited. Third-party mirrors (e.g. MMAction2 skeleton pickles) exist but **using them does not release you from the license** — I do not recommend routing around the form for a dissertation you will publish.
- **Data hygiene:** 535 samples have missing/incomplete skeletons — must be filtered.

---

## 3. TIER 3 — Considered and rejected (with reasons)

| Dataset | Why rejected |
|---|---|
| **Kinetics-400** | 4TB+, YouTube link-rot, no official copy, version drift breaks reproducibility. Our use for it (backbone pretraining) is already covered by downloadable pretrained weights. |
| **DukeMTMC-reID** | Withdrawn over ethics concerns. Disqualifying for an elderly-care project. |
| **MCFD** | Requires explicit author permission. OmniFall includes it — **we exclude it from our splits** and document that. |
| **CMDFall / UP-Fall / EDF / OCCU** | All need author permission per OmniFall's own documentation. Not worth the latency. |
| **YouHome** | Only labelled *images* + sensor data, no raw video. Cannot train temporal models. |
| **ASCC ADL** | Chest-mounted egocentric — wrong viewpoint for CCTV monitoring. |
| **ElderSim / KIST SynADL** | Synthetic elderly ADL. Genuinely interesting, but adds a sim-to-real gap we lack time to characterise. Note as future work. |
| **Emotion / speech / age datasets** | Cut with the emotion pivot. Adding them re-imports EmotionSense's failure mode. |

---

## 4. Final selection

```
FALLS       OmniFall (staged→wild) + CAUCAFall + Le2i + GMDCSA-24 + URFD
ADL         Charades (157→20 mapped)          [+ Toyota Smarthome if approved]
REID        Market-1501
TRACKING    MOT17 (eval only)
DET/POSE    COCO val2017 (sanity only; pretrained weights do the work)
SKELETON    [NTU-120 if approved — pretraining bonus]
BEHAVIOUR   Longitudinal simulator (ours — no dataset exists)
```

**Total download ≈ 20–25 GB.** Comfortable inside Kaggle limits. No component blocks the critical
path on human approval.

---

## 5. The honest limitation paragraph (write this into the dissertation now)

> All ungated fall and ADL datasets used in this work feature predominantly young-adult subjects
> performing scripted activities. Gait, posture, movement velocity, and fall kinematics differ
> materially in adults aged 65+. Consequently, reported accuracies should be read as an upper bound
> on real-world elderly-care performance. We mitigate this by (i) evaluating staged→wild transfer
> via OmniFall's in-the-wild split, (ii) reserving low-resolution webcam data (GMDCSA-24) as a
> deployment-realism test, and (iii) [if approval lands] validating on Toyota Smarthome's genuine
> 60–80yo cohort. Removing this limitation requires ethically-approved primary data collection,
> which we identify as the critical prerequisite for clinical deployment.

Stating this explicitly is a strength. Examiners punish unacknowledged limitations far harder than
acknowledged ones.
