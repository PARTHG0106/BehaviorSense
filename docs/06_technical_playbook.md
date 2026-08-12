# 06 — End-to-end technical playbook

Every algorithm, technique, optimizer and setting for the whole pipeline, with the
justification attached. Complements [05_plan.md](05_plan.md) (sequencing) — this is the
*what and how*. Decisions locked in [02_architecture.md](02_architecture.md) are not
reopened here; they are given their training/inference recipes.

---

## 0. What the GPU actually buys — an honest framing

RTX PRO 6000 Blackwell, 96 GB VRAM, 12 h Kaggle sessions, **no internet during
training**. The unfashionable truth: our trained models are *skeleton* models (a few
million params, ~2–3 GB peak each). 96 GB does not buy accuracy through model size here —
**data is the bottleneck, not capacity**. The VRAM is spent where it does convert to
accuracy:

1. **Ensembles trained in parallel** — all 4 skeleton streams simultaneously in one
   session (biggest single accuracy lever available, +2–4% mean-class acc, standard
   result in the GCN literature).
2. **Multi-seed replicates** (3 seeds per config) — error bars on every headline number;
   a dissertation claim with a ± beats a slightly higher point estimate.
3. **Large-batch throughput** — full sweep of the small hyperparameter space in hours.
4. **The LLM in full bf16** (no quantization loss) — Qwen2.5-7B needs ~16 GB; even 32B
   fits in bf16 (~64 GB) as a stretch.

Blackwell caveat: RTX PRO 6000 is sm_120; requires torch ≥ 2.7 cu128 wheels. **First
Kaggle action is a 10-minute smoke test** (matmul + autocast + one training step), not a
training run. `torch.compile` is enabled only if that smoke test passes with it on.

---

## 1. Environment, reproducibility, offline assets

- **Pins:** one `requirements-train.txt` (torch/cu128, pyskl or mmaction2, mmpose,
  numpy, pydantic v2) and one `requirements-serve.txt` (CPU: fastapi, uvicorn, pydantic,
  numpy, onnxruntime). Exact versions frozen after the smoke test; recorded in the repo.
- **Seeds:** every entry point takes `--seed`; seeds {0,1,2} for replicated runs. RNG
  state (python/numpy/torch/CUDA) is saved inside every checkpoint so resume ≠ silent
  re-randomisation.
- **Offline asset manifest** (each becomes a private Kaggle dataset, verified loadable in
  a throwaway offline notebook *before* training week):

  | Asset | Size | Used by |
  |---|---|---|
  | RTMO-l COCO checkpoint + mmpose config | ~0.3 GB | Stage A pose extraction, Agent 1 |
  | OSNet-AIN `osnet_ain_x1_0` (Market/MSMT pretrained) | ~10 MB | Agent 1 ReID |
  | RT-DETR-L COCO | ~0.2 GB | Agent 1 object context |
  | ST-GCN++ NTU-pretrained (PYSKL) | ~0.1 GB | Agent 2 init |
  | Qwen2.5-7B-Instruct weights + tokenizer | ~16 GB | Agent 4 |
  | `outlines` + deps wheels | ~0.1 GB | Agent 4 |
  | Preprocessed skeleton shards (Stage A output) | ~5–15 GB | Agent 2 training |

- **Licensing discipline (unchanged):** Charades / Market-1501 / MSMT17 derived tensors
  stay **private**; publish code + metrics, never data.

---

## 2. Stage A — data pipeline (online notebooks, CPU/TPU + one GPU pass)

Per dataset: download → integrity check (counts vs published) → decode → pose → windows.

- **Decode:** PyAV/ffmpeg, resample to **15 fps** (deployment rate; training/test at the
  same rate avoids a train-serve skew), shorter side 640 px, no re-encode of RGB to disk
  beyond what pose extraction consumes (skeletons are the product; RGB is discarded —
  privacy + 1000× smaller).
- **Pose extraction:** RTMO-l, score floor 0.35/0.10 two-band (matches
  `PerceptionConfig`), keep top-2 people per frame by box area (ADL clips are 1–2
  people). Output per clip: `[T, P, 17, 3]` (x, y, score) float16 + frame timestamps.
  Runs in GPU sessions with the staged checkpoint (fastest); nothing else needs GPU.
- **Quality gates (printed per dataset, not assumed):** % frames with a usable skeleton
  (≥8 visible kpts), % clips dropped, mean people/frame. A dataset losing >15% of frames
  gets investigated before use, not averaged over.
- **Taxonomy mapping:** `scripts/build_charades_map.py` → 20-class unified ADL taxonomy
  (`configs/taxonomy.yaml`); OmniFall/Le2i/CAUCAFall/GMDCSA/URFD → {fall, lying,
  no-fall-ADL} for the fall head. Mapping tables are versioned files, reviewed by eye —
  label noise from silent auto-mapping is unrecoverable later.
- **Windowing:** 2 s @ 15 fps (T=30), stride 1 s. Normalisation: root-center on
  mid-hip, scale by torso length, no rotation normalisation (falls are
  orientation-meaningful).
- **Packing:** memory-mapped `.npz` shards (~500 MB each) + a parquet index
  (clip id, dataset, subject id, class, split). Subject ids are what make P1
  cross-subject splits honest — enforced disjoint by assertion in the split builder,
  same pattern as the ReID protocol's `validate()`.
- **MSMT17/Market-1501:** no video; crop manifests via `data/reid_datasets.py` (done),
  embeddings via `scripts/extract_reid_embeddings.py` (CPU, minutes).

---

## 3. Agent 1 — perception (inference-only; recipes, no training)

All frozen pretrained; fine-tuning any of them is explicitly rejected (no in-domain
labels; 1-month budget). Accuracy comes from *operating points*, which we fit.

- **RTMO-l:** as staged. det thresholds 0.35 new-track / 0.10 continue (two-band,
  tested P4).
- **BoT-SORT (deployment tracker):** ReID-gated association ON (native OSNet hooks),
  camera-motion compensation OFF (static home cameras — CMC costs CPU and can only
  hallucinate motion), `track_high_thresh=0.35`, `track_low_thresh=0.10`,
  `new_track_thresh=0.4`, `track_buffer=30` (2 s @ 15 fps), `match_thresh=0.8`.
  `SimpleTracker` remains the CPU test double; its measured association ceiling
  (0.538 box-widths/frame, P3b) is documented as the gap BoT-SORT closes.
- **OSNet-AIN open-set ReID:** τ and margin **fitted, not guessed** — FAR ≤ 1% budget on
  the fit half, reported on the identity-disjoint test half; fitted on Market-1501,
  transferred to MSMT17 (and vice versa) as the cross-dataset table. Gallery: 4 crops /
  identity / 1 camera (deployment-realistic enrolment), EMA momentum 0.9 (tested P8).
  Metrics: AUROC, TAR@FAR{0.1,1,5}%, DIR@1, EER + closed-set rank-1 as calibration
  smoke test against published OSNet numbers.
- **RT-DETR-L:** COCO classes filtered to context set {person, cup, bottle, bowl, chair,
  couch, bed, dining table, tv, cell phone, book}, conf 0.5, at 1 Hz.

---

## 4. Agent 2 — activity recognition (the trained core)

### Backbone and the maximum-accuracy configuration

**ST-GCN++ (PYSKL), 4-stream ensemble: joint, bone, joint-motion, bone-motion; logit
averaging.** This is the decision from 02 plus the standard ensemble lift. Rationale over
alternatives, kept honest:

- *SkateFormer (ECCV 2024)* is +1–2% over GCN ensembles on NTU but a riskier third-party
  implementation and hungrier on data we don't have. **Stretch goal only**, trained as a
  5th ensemble member if week 3 has slack — ensembling a transformer with GCNs is where
  its diversity actually pays.
- *CTR-GCN* ≈ ST-GCN++ accuracy; PYSKL's ST-GCN++ implementation and NTU-pretrained
  weights are the better-maintained path.
- *RGB models (VideoMAE etc.)* stay rejected: ~1000× data footprint, privacy, and the
  age-gap problem worsens (appearance bias).

Init from NTU-pretrained, replace the head (20 classes), **full fine-tune** (not
linear-probe — domain gap between mocap-quality NTU skeletons and RTMO CCTV skeletons is
exactly what fine-tuning must absorb).

### Training recipe (per stream)

| Item | Setting | Why |
|---|---|---|
| Optimizer | **SGD, momentum 0.9, Nesterov** | AdamW gives nothing on GCNs and generalises slightly worse (PYSKL benchmarks); transformers (SkateFormer, if used) get AdamW β=(0.9,0.999), wd 0.05 |
| LR schedule | 0.1 × BS/128, **cosine anneal**, 5-epoch linear warmup, 80 epochs | warmup because large batch + pretrained init; cosine beats step on small data |
| Weight decay | 5e-4 (not on BN/bias) | PYSKL standard |
| Batch size | 512 (bf16) | 96 GB makes this free; LR scaled accordingly |
| Precision | **bf16 autocast** + TF32 matmul | Blackwell-native; no loss-scale fragility of fp16 |
| Label smoothing | 0.1 | taxonomy classes have genuinely soft boundaries (reading vs sitting) |
| Class imbalance | **class-balanced sampler** (effective-number weights, β=0.9999) | Charades-derived classes are heavily skewed; plain CE learns the prior |
| EMA of weights | decay 0.999, evaluated model = EMA | free +0.3–0.8%, stabilises small-data fine-tune |
| Augmentation | random rotation ±15°, scale 0.9–1.1, joint jitter σ=0.01, temporal crop/resample 0.8–1.2×, random joint dropout p=0.05, **left-right flip with joint remap** | skeleton-standard; dropout doubles as occlusion robustness matching RTMO reality |
| Regularisation extras | DropHead/mixup **rejected** — marginal on GCNs, more tuning surface than the month affords | |
| Checkpointing | every epoch: weights + optimizer + EMA + scheduler + RNG; auto-resume by glob | 12 h session limit makes this mandatory, not hygiene |

Gradient checkpointing: **not used** — models are small; it trades compute for memory we
have in excess.

### Fall detection head

Separate binary head (fall vs not) trained on OmniFall + Le2i + CAUCAFall + GMDCSA + URFD
mapped labels, sharing the backbone with a 0.1× LR multiplier:

- **Loss: focal (γ=2, α=0.75)** — falls are rare positives; plain BCE optimises the
  majority class.
- **Operating point fitted, not defaulted:** threshold chosen on validation at a fixed
  false-alarm budget (per hour of video), reported as sensitivity @ FA/h — same
  philosophy as the CUSUM k table and the ReID τ.
- Geometric cues (`aspect_ratio`, `torso_vertical_ratio`) enter as *features to the alert
  rule*, never as the detector.

### Temporal structure (post-hoc, CPU, testable now)

- **HMM/Viterbi smoothing** over per-window log-probs. Transition matrix from label
  statistics with a self-transition floor (activities persist at 1 Hz); emission =
  calibrated window posteriors.
- **Calibration: temperature scaling** on validation (single scalar) — Viterbi and the
  abstention threshold both consume probabilities, so they must mean something.
- **Object-context late fusion:** per-class log-prior offsets conditioned on RT-DETR
  detections in the last 3 s (e.g. `taking_medication` requires bottle/cup evidence;
  `watching_tv` boosted by tv). A learned fusion MLP is rejected — 20×11 prior table is
  inspectable and needs no training data we don't have.
- **Abstention:** below max-posterior 0.4 after smoothing → `unknown_activity`; Agent 3
  already treats missing windows via `is_reliable()`.
- Tests (synthetic logits, CPU): smoothing removes isolated flips **and provably does not
  erase a genuine 1-window fall** (the anti-vacuous pair); fusion flips
  medication/drinking only with object evidence; calibration monotone.

### Evaluation

- **P1** cross-subject (subject-disjoint by assertion), **P2** leave-one-dataset-out,
  **P3** staged→wild (OmniFall OOPS split) — expect and report the drop.
- Metrics: **mean-class accuracy** and macro-F1 (never top-1 alone — imbalance), falls:
  sensitivity @ fixed FA/h, AUROC; ECE after calibration.
- 3 seeds → mean ± std on every table.
- Ablations (9, from 02): −ensemble, −pretrain, −smoothing, −fusion, −calibration,
  −class-balancing, −augmentation, per-stream, fall-head-only.

---

## 5. Agent 3 — behaviour (done; frozen)

No training. Parameters frozen as measured ([results/tuning_log.md](../results/tuning_log.md)):
robust z (median/MAD, 0.6745, 5% floor), weekday/weekend stratified 14-day baseline,
CUSUM k=0.75 h=5.0 warmup=7 decay=0.98, R10 2-day same-direction persistence, absence
rules suppressed on unreliable days. 100% recall / 5.07 FP/100d on the 28-scenario suite.
Only permitted change: re-run the suite if Agent 2's real error profile (from P1) is
injected into the simulator as feature noise — a planned robustness check, not a re-tune.

## 6. Agent 4 — reporter + verifier

- **Model: Qwen2.5-7B-Instruct, bf16, frozen.** No fine-tuning — the contribution is the
  *verifier*, and fine-tuning an LLM to be "more faithful" would blur exactly the claim
  we can currently make arithmetically. Stretch: Qwen2.5-32B bf16 (~64 GB, fits) for a
  model-scale row in the hallucination table.
- **Decoding: `outlines` grammar-constrained JSON** against the `Claim`/`Report` schema,
  temperature 0, greedy. Unconstrained condition (same prompts, free JSON) is the
  comparison arm.
- Prompt: structured `BehaviourReport` JSON in, 2 few-shot exemplars, explicit
  "every claim must carry evidence_ref".
- **Headline: hallucination rate** (claims failing C1–C4) constrained vs unconstrained,
  plus verifier catch-rate on seeded corruptions (inverted narration etc., already
  tested V1–V12).

## 7. Session plan for training week (12 h units)

1. **S1 (smoke, 30 min):** wheel/sm_120 check, one batch each model, assets load offline.
2. **S2:** Stage A GPU pass — RTMO pose extraction over all video datasets → shards.
3. **S3:** 4 streams × ADL head, seeds 0–2 (parallel; ~all fit at once in 96 GB).
4. **S4:** fall head + calibration + operating points; SkateFormer stretch if S3 clean.
5. **S5:** ablations + P2/P3 evals + Qwen hallucination runs (inference only).
Each session ends by pushing checkpoints + metrics parquet as a Kaggle dataset version
(resume insurance across sessions).

## 8. Serving & demo (week 4, CPU)

FastAPI: `/ingest` (FrameObservation), `/day/{date}/features`, `/alerts`, `/report`
(reporter+verifier). SQLite persistence of all schema objects (already the agent
boundary). Replay demo: simulator → Agent 3 → Agent 4 → verified report (GPU-independent
safety net). Optional ONNX export of RTMO/OSNet for CPU demo. Docker: serve image only.

## 9. Rejected maximum-accuracy techniques — for the record

| Technique | Why rejected |
|---|---|
| Knowledge distillation / self-training on wild clips | weeks of iteration risk, pseudo-label noise on the exact distribution gap we must report honestly |
| RGB–skeleton two-stream fusion | reintroduces privacy + appearance-bias problems for ~1–2% |
| LLM fine-tuning (LoRA) for reporting | undermines the verifier claim; faithfulness must be *checked*, not *trained in* |
| Kalman/learned motion models in SimpleTracker | test double only; BoT-SORT owns deployment tracking |
| Face recognition for identity | unchanged from 02: no facial pixels, blurred datasets, max ethics exposure |
| fp16 loss-scaled training | bf16 exists; fp16 adds failure modes for zero gain on this GPU |

---

*Sequencing, risks and week map: [05_plan.md](05_plan.md). Nothing in this document
changes a locked decision; it operationalises them.*
