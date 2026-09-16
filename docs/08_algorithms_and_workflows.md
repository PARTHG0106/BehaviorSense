# BehaviorSense AI — Algorithms & Workflows, Step by Step

Every algorithm in the pipeline, where it runs, why it was chosen over its alternatives,
and — where we measured it — what the choice cost or bought. Companion to
[`02_architecture.md`](02_architecture.md) (system design) and
[`results/evaluation.md`](../results/evaluation.md) (the numbers). Where a choice was
later proven wrong by measurement, that is stated here rather than hidden, because half
of these decisions were only *validated* by the evaluation runs — and three were
overturned by them.

The one-sentence version of the whole system: **perception turns pixels into structured
records, statistics turn records into evidence, and the LLM is never trusted — every
claim it makes is re-derived arithmetically before a caregiver sees it.**

```
video ──► AGENT 1  RTMO pose ► BoT-SORT tracks ► OSNet ReID ► role
                   RT-DETR objects (1 Hz)
                        │ PersonObservation (Pydantic, SQLite)
                        ▼
         AGENT 2  windows ► ST-GCN++ ensemble ► calibrate ► fuse objects
                   ► Viterbi smooth ► abstain ► ActivitySegment
                        │ DailyFeatures
                        ▼
         AGENT 3  median/MAD baseline ► robust-z ► CUSUM drift ► Alerts
                        │ BehaviourState (numbers + resolvable evidence refs)
                        ▼
         AGENT 4  Qwen2.5-7B ► Claims ► DETERMINISTIC VERIFIER (C1–C4)
                   ► unfaithful claims withheld ► caregiver report
```

---

## Stage 0 — Data acquisition & staging (notebooks 00, 01a, 02-acquire)

### Content-based asset resolution
**Algorithm:** never resolve a Kaggle mount by dataset *name*; identify each asset by a
file only it contains (`**/*.whl` scored by marker packages, `**/rtmo-l.onnx`,
`**/src/behaviorsense/__init__.py`), then verify a 21-token **repo contract** — specific
files and API tokens the notebooks call — against the mounted code.

**Why best:** Kaggle mount paths are not predictable (datasets nest under
`/kaggle/input/datasets/<owner>/<name>/`, Save Version adds `kaggle/working/`, titles
are free text with two spellings in this project alone). Name-based resolution cost a
session ("charades-480p is not attached" while it was attached). The contract converts
the *stale-snapshot* failure mode — the most common failure in a two-dataset workflow —
from a misleading error deep in a run into a cell-1 failure that names the missing token
and prints the fixing command. It needs no version numbers: it fails exactly when the
snapshot is too old for the notebook running against it.

**Alternative rejected:** pinning dataset names/versions — brittle against Kaggle's own
mount-layout changes (which changed once mid-project) and against renames.

### Offline-first wheel staging
**Algorithm:** notebook 00 downloads every wheel (torch cu128 for sm_120, rtmlib
`--no-deps`, `onnxruntime-gpu==1.26.0`, outlines ≥ 1.0) and both OSNet checkpoints into
a dataset; training notebooks install with `pip --no-index --find-links`.

**Why best:** GPU notebooks run with **no internet** (project constraint), and the two
version traps here are silent: onnxruntime-gpu ≥ 1.27 is built against CUDA 13 while
Kaggle is CUDA 12 — the provider *lists* as available and then fails to load
(`libcublasLt.so.13`), pushing RTMO onto CPU at ~1/50 speed; and rtmlib's dependency on
CPU `onnxruntime` clobbers the GPU build because both own the same module directory.
Pinning 1.26.0 and installing rtmlib `--no-deps` removes both.

---

## Stage 1 — Skeleton extraction (notebooks 01, 02)

### RTMO for detection + pose (one pass)
**Why over the alternatives:** top-down pipelines (detector + per-person HRNet/RTMPose)
cost one backbone pass *per person* and need a separate detector; bottom-up OpenPose is
legacy-grade accuracy. RTMO is a one-stage, one-pass detector+pose model — constant cost
per frame regardless of person count, ONNX-exportable, and accurate enough that pose
quality is not the bottleneck (the label map is). On the RTX PRO 6000 it sustained
**1,333 videos/hour**, extracting all 7,985 Charades videos in ~6 h.

### Temporally stable person slots — `assign_slots()`
**Algorithm:** greedy nearest-neighbour association of the top-2 people between
consecutive frames, gated at 3× bbox diagonal; residents keep their slot, entrants take
the lowest *free* slot.

**Why:** the naive rule ("two largest boxes, area order") swaps the two people whenever
their apparent sizes cross — splicing both trajectories and teleporting every motion
stream at the swap frame. The first fix had its own second-order bug (an entrant taking
`min(free)` could steal a resident's slot), caught by writing the test before trusting
the code (S8). Slot stability is what makes person slot 1 a *signal* (see
second-person context, Stage 4) rather than noise.

### Windowing: 2 s / 30 frames @ 15 Hz, stride 1 s
**Why:** falls complete in 1–2 s, so anything longer smears the event across a window
boundary; ADLs are minutes long, so the cost of short windows is only that labels are
noisy per window — which temporal smoothing (Stage 4) and daily aggregation (Stage 5)
absorb. 15 Hz halves decode cost against 30 with no measured accuracy loss on skeleton
models, whose information is in joint *trajectories*, not per-frame detail.

### Label assignment: best-covering mapped action (≥ 60% overlap)
**Why:** Charades actions overlap heavily. The first version took the *first* action in
CSV order with ≥ 60% overlap — a barely-qualifying interval listed first beat a
fully-covering one, and an unmapped first qualifier forced `other_idle` even when a
mapped action covered the window. Coverage decides now; unmapped actions cannot claim a
window at all.

### Fall labelling: exact intervals where they exist, position otherwise
**Algorithm:** `label_fall_windows()` uses Le2i's frame-exact annotation files when
present; corpora with only fall/ADL folder structure get positional labelling — the
window whose *centre* falls in the descent region is `falling`, later windows
`fallen_on_ground`. Window positions come from true start frames (`with_starts=True`),
never from `enumerate()`, because dropped windows shift enumeration.

### Native-crash isolation for video decode
**Algorithm:** a subprocess **decode probe** journals `TRY <clip>` before each decode
and restarts past clips that kill the interpreter; PyAV (its own ffmpeg build) is the
second-chance backend; shards flush incrementally.

**Why:** cv2→ffmpeg faults are SIGSEGV/SIGABRT — invisible to try/except, fatal to the
whole kernel. Before this, one bad Le2i clip destroyed a session *including all
already-extracted corpora*. The probe also corrected a misattribution: two crashes
blamed on URFD were proven (70/70 clean) to be Le2i's first clip.

### Subject identity — the project's most instructive bug
**Algorithm now:** fall corpora derive subjects from the directory layout
(`subject_from_path()`); Charades videos are mapped **video id → actor id** through the
`subject` column of `Charades_v1_train.csv` (267 actors) at *split time*
(`load_subject_map` / `remap_subjects`).

**Why this matters:** the split's job is to keep *people*, not files, out of training.
GMDCSA numbers every subject's clips `01..`, so filename-stem ids merged four people
into one; and Charades was split by video id for weeks on the recorded-and-wrong belief
that actor ids "do not exist publicly." With ~30 videos per actor, essentially every
person sat on both sides — ids disjoint, people not. The disjointness assert can never
catch this class of bug, which is why the identity *derivation* is the algorithm, tested
with a positive control (S13: video-split leaks 4/5 actors, actor-split leaks 0).

---

## Stage 2 — Identity (Agent 1, serving-time)

### OSNet-AIN + open-set ReID with a fitted threshold
**Algorithm:** OSNet-AIN embeddings; a track matches an enrolled identity if cosine
similarity ≥ **τ = 0.355**, else UNKNOWN. τ fitted at FAR ≤ 1% on one identity-disjoint
half of MSMT17 (100 enrolled + 400 impostors), reported on the other half.

**Why open-set:** a home has visitors; closed-set ReID would force every stranger onto
the nearest resident. **Why fitted, not guessed:** the previous τ = 0.30 was a guess.
**Why trustworthy:** an ImageNet-only control run through the *identical* protocol
scores AUROC 0.534 vs 0.997 — a +0.46 gap proving the protocol can't be flattered by an
arbitrary embedding. **Why not DukeMTMC:** withdrawn dataset, excluded on ethics.

### BoT-SORT for tracking
Motion+appearance association, standard and strong; the in-repo `SimpleTracker` is
explicitly a test double (motion-only, measured ceiling 0.538 box-widths/frame) so the
identity logic is testable on CPU.

---

## Stage 3 — Activity model (notebook 03)

### ST-GCN++, four streams, from scratch
**Why ST-GCN++ over transformers (SkateFormer etc.):** graph convolutions encode the
skeleton's adjacency as an inductive bias, which wins at our data scale (165k windows is
small for a transformer); self-contained implementation with no mmcv (a dependency that
does not build on the Kaggle image).

**Streams:** joint / bone / joint-motion / bone-motion, each its own model.
`to_bone()` uses `parent.setdefault(max(i,j), min(i,j))` so the bone stream is
**flip-equivariant** — measured error 5.93 before the fix, meaning half the ensemble was
training on sign noise under the flip augmentation (S2c holds the negative control).

### Normalisation that cannot erase the label
**Algorithm:** `normalise()` centres on the **first visible frame's** torso and scales
by torso length computed over *visible frames only*, with `keep_root_motion` preserving
whole-body translation.

**Why:** per-frame root-centring subtracts exactly the motion that defines a fall — the
smoke test measured **AUROC 0.466, below chance**, on data whose only signal was a
1.5-unit hip drop. And dropout frames (all-zero) diluted the torso estimate ~1.5× while
dragging the origin toward (0,0); visibility-masking fixed both (S4b, S4c).

### Augmentation: flip (with joint remap) + rotate + scale + jitter + joint dropout
Two non-obvious rules, both measured: the horizontal flip must remap left/right joint
indices (else it teaches anatomically impossible bodies), and any transform must re-zero
score-0 joints afterwards (`x[x[...,2] <= 0] = 0`), else train-time "missing" joints
move while serve-time ones sit at zero — a distribution skew between training and
deployment (S5c).

### Class imbalance: effective-number sampling, *not* stacked with logit adjustment
**Algorithm:** `WeightedRandomSampler` with effective-number weights
`(1−β)/(1−β^n)`, β = 0.9999 (Cui et al., CVPR 2019).

**Why over inverse frequency:** at ~942:1 imbalance (65k `other_idle` vs 69
`bending_reaching`), 1/n oversamples the rarest class so hard the model memorises a few
hundred windows. Effective-number weighting **saturates by design** — it removes ~145×
of the ratio and deliberately leaves the rest. That residue is why the *post-hoc* logit
adjustment (Stage 4) still finds +0.050 mean-class at τ = 0.25 — the small fitted τ is
itself evidence the sampler did most of the work. A training-time variant
(`--tau-train`, Menon et al., ICLR 2021) exists but **refuses** to run alongside
balanced sampling: two corrections for one imbalance is a silent methodological error,
and the script exits with the fix rather than warning inside a 12-hour log (S10, S11).

### Optimisation recipe
SGD + Nesterov, cosine schedule with warmup, label smoothing 0.1, **no weight decay on
norm/bias** (decaying them shifts normalisation statistics rather than regularising),
bf16 autocast, EMA of weights with **warmed-up decay** `min(0.999, (1+t)/(10+t))` — a
fixed 0.999 leaves the average ~95% random init for the first ~50 steps, so early
validation measures the initialisation and corrupts `best.pt` selection.

### Epoch budget: 30 with patience 8 — measured, not guessed
The NTU-recipe 80 epochs was wrong for this data: all four streams peaked at epoch 8–11
and *lost ~27% mean-class* over the remaining 70 (memorisation; NTU has ~10× more
windows per class). ~2.4 of 2.7 GPU-hours were being spent making the model worse.

### Fall head: binary, ADL windows as negatives, focal loss with guarded α
The fall corpora alone are ~82% positive — backwards for deployment. Mixing in all ADL
windows as negatives gives a 0.51% positive rate and, crucially, **23.2 hours of
negatives**, which is what makes a false-alarms-per-hour operating point a decision
rather than an artefact (the earlier 141-negative pool was 0.08 h).
`train_fall.py` *refuses* `focal α > 0.5` when positives are the majority — α points at
the positive class, and with inverted data that silently up-weights the majority while
reporting a plausible AUPRC.

### Resume as a correctness requirement
Kaggle sessions die at 12 h, so every real run resumes. The checkpoint carries model +
EMA + optimizer + **RNG state** (python/numpy/torch/CUDA), restored bit-exactly (S7
compares a resumed run against an uninterrupted one step-for-step). Hyperparameter
changes across resume are *refused*, including the split identity (`subject_map`) —
resuming a video-disjoint checkpoint into an actor-disjoint run would smuggle the old
split's leakage into the new protocol. Two real traps fixed here: the LR schedule is a
function of total epochs (changing `--epochs` on resume silently bends the curve), and
`map_location="cuda"` moves the RNG tensor to GPU where `set_rng_state` rejects it —
found because it killed all five runs of one session, invisible to the CPU-only test.

---

## Stage 4 — Serving-time decision rule (Agent 2, measured in notebook 04)

Ordered pipeline: **logits → ensemble → calibrate → object fusion → Viterbi → abstain**.
Each step's placement is an argument:

### Ensembling in logit space — and the measured surprise
Logit averaging is a product of experts (a stream can veto); probability averaging is a
mixture. PYSKL reports logit averaging, so it is the headline. **Measured:** the
ensemble wins top-1 (0.385 vs 0.341) but *loses* mean-class to single streams (0.152 vs
0.190) — the veto behaviour deletes rare classes, the exact head/tail trade again. The
ablation row exists so this is a documented choice, not an assertion, and on macro-F1
the credible options are within noise of each other (0.169–0.172), the honest summary
being that the default ensemble is the *worst* credible configuration.

### Temperature calibration AFTER ensembling
T = 0.76 fitted on val logits (confidence 0.396 vs accuracy 0.385 — nearly calibrated).
After, because the ensemble's confidence distribution is no single member's; per-stream
calibration would double-soften. Viterbi and abstention both consume probabilities, so
those probabilities must mean something first.

### Post-hoc logit adjustment — the lever that worked
`logits − τ·log(prior)` at decision time (balanced-error decision rule, Menon et al.).
τ swept on saved val logits with τ = 0 asserted as an exact no-op; peak at **τ = 0.25**:
mean-class 0.152 → 0.202 (ensemble) and → 0.222 (`bone`). Free — no retraining, one
number in deployment. Caveat stated with it: mean-class is macro-*recall*; on macro-F1
(the right arbiter, since Agent 3's durations are hurt by over- and under-prediction
alike) the gain is +0.020.

### Object-context fusion as log-odds offsets
A hand-written table (`OBJECT_PRIORS`), not a learned MLP: 20 classes × a dozen objects
is small enough to inspect, and no dataset pairs our object vocabulary with our
taxonomy to learn from. `taking_medication` additionally *requires* object evidence
(−2.0 log-odds absent a medication object) because pose cannot separate it from
drinking. The measured per-class table validates the design: the F1 floor is exactly
the object-dependent classes (`taking_medication` 0.043, `interacting_with_person`
0.042, `watching_tv` 0.089, `using_phone` 0.091).

**Second-person context** is the one object signal needing no detector: shards carry
two person slots, slot 1 is non-zero exactly when the tracker held a second person, and
that flag (≥3 joints in ≥5 frames, so tracker flicker doesn't count) feeds
`OBJECT_PRIORS["person"]` through the same `fuse_objects()` path serving uses (P15).

### Viterbi smoothing — kept for structure, not accuracy
Max-product decoding under a transition prior with an **emergency floor into `falling`**
(≥ 0.05 from every state) so a one-window fall survives smoothing that a 0.90
self-transition would otherwise erase.

**Measured honestly:** smoothing the raw posterior *costs* 0.040 mean-class (top-1
+0.011) — the head class absorbs short rare-class runs. My predicted fix (fit the
transition matrix from 165k real sequences; fitted self-transition 0.764 vs hand-set
0.90) recovered only 0.004 of it — the regression is intrinsic to max-product decoding
over a 39%-head posterior, not a mis-set rate. What *did* work is ordering: smooth the
**adjusted** posterior (0.112 → 0.174). And smoothing is not judged on accuracy alone:
Agent 3 counts `walking_bouts` from segment structure, so a decoding that shatters one
true stretch into nine reports nine bouts while scoring identically per window — the
**fragmentation ratio** (predicted/true segment count, 1.00 ideal) is reported beside
accuracy, and the deployment choice is made on both.

### Abstention on the forward–backward marginal
Abstain when P(state_t = selected | **all** windows) < 0.40 — the third attempt,
decided by measurement. `posterior[selected]` abstained exactly where smoothing had
just fixed a flicker; `max(posterior)` shattered confident runs on isolated dim windows
*inside* them. The smoothed marginal is the only candidate that distinguishes "dim
window braced by confident neighbours" (context explains it) from "dim window in a dim
neighbourhood" (genuine ignorance).

---

## Stage 5 — Behaviour analysis (Agent 3): statistical by argument

**Why not a learned model:** no public dataset contains multi-week, identity-resolved
elderly home video. A neural baseline would be trained on simulation and validated on
simulation — circular. Robust statistics have provable properties and every parameter
is auditable (all of them justified in `results/tuning_log.md`).

- **Baseline:** rolling 14-day **median/MAD** — 50% breakdown point, so one hospital
  visit cannot corrupt it (mean/std has 0%).
- **Deviation:** robust z `0.6745·(x−median)/MAD`, with **MAD floored at 5% of
  |median|** — a rigid routine gives MAD = 0, an unfloored z divides by zero, and the
  naive ±6 fallback turns a 0.1% change into a 6σ event for exactly the most regular
  residents. The classic alert-fatigue failure, regression-tested.
- **Drift:** one-sided **CUSUM** (Page 1954) — accumulates small persistent deviations.
  Measured lead over daily thresholding: **8 days** on a 1.2%/day decline under σ≈12%
  noise. Detection lead time in days is the clinically meaningful metric.
- **Measured end-to-end:** 28 scenarios × 150 days: 100% recall (24/24 injected
  declines), 4-day median latency, false alerts **67.66 → 5.07** per 100 control-days
  across five distinct measured fixes (absence inferred from camera outages, weekday
  stratification, multiple-comparisons correction, a missing drift detector, CUSUM
  charging on noise).

---

## Stage 6 — Verified reporting (Agent 4): the research contribution

### The boundary
The LLM (Qwen2.5-7B-Instruct, offline) **never sees pixels** — only a structured
`BehaviourState`, with the exact resolvable `evidence_ref` strings included in the
prompt. Asking a model to *synthesise* refs invites plausible-but-wrong citations that
score as hallucinations but are the prompt's fault. Measured consequence: **C1
(fabricated citation) = 0 across every arm of every run.** One design choice deleted an
entire failure mode.

### Deterministic verification (C1–C4) — no LLM checks the LLM
Every claim must carry a machine-resolvable ref and is re-derived arithmetically:
C1 ref exists, C2 value matches (2% rel. tol.), C3 percentage matches (2 pp), C4
direction consistent — structured field, prose lexicon, and pct-sign against the true
delta. C4 is the clinically dangerous mode: "mobility improved by 50%" when it fell 50%
passes C1–C3. Tolerances stay tight because loosening them lets paraphrase inflate
faithfulness and quietly destroys the metric. Unfaithful claims are **withheld**, not
flagged — a caveat next to a confident false clinical claim is not protection.

**Why the measurement is trustworthy:** anchored both ends. A faithful stub must score
0.0% (any more = verifier false positives inflate every reported rate); corrupted stubs
at 25/50/100% injection score 24.7/52.0/100.0% (the metric tracks, doesn't saturate);
and every one of six corruption types must be caught **by the specific check
responsible** (R3) — a blanket-reject verifier passes count tests but not that one.

### Grammar-constrained decoding: measured as a guarantee, not a gain
Outlines-constrained JSON vs free decoding, decoding-matched (greedy both arms, same
chat-templated text — getting the arms truly identical took four runs and found four
confounds, all ours: prompt withheld the output envelope, grammar allowed 12 claims vs
the reporter's 6, the constrained arm inherited `temperature=0.7` from Qwen's
generation_config, and it skipped the chat template). Final, reproduced bit-identically
across three runs: **7.7% vs 8.2% hallucination, p = 0.79 — no faithfulness gain — at
7.6× the decoding cost.** The grammar's real value is a worst-case *guarantee* of
parseability. The dissertation claim is therefore not "constraints fix hallucination";
it is: **~8% of a 7B model's claims about a resident's day are wrong against the data
it was handed, the errors are almost entirely misquoted values, and a deterministic
arithmetic layer catches all of it before a caregiver reads a word.**

---

## Cross-cutting workflow algorithms

- **Two-phase Kaggle split:** online CPU notebooks acquire and stage; offline GPU
  notebooks train and evaluate from mounted datasets. GPU-hours are never spent
  downloading.
- **Single source of truth:** all six notebooks are generated from
  `notebooks/_generate.py`, which **compiles every cell** before emitting (a recurring
  `\n`-in-literal defect became impossible); shared constants (epoch budgets, split
  fraction, metric definitions) live in one place so copies cannot drift.
- **Tests execute the real artefact:** `test_notebook04_e2e.py` runs the actual cell
  sources against synthetic fixtures on CPU — built after three Kaggle failures whose
  root cause was tests validating *re-implementations* of notebook logic. Detection
  tests print the quantity they measured; rejection tests carry positive controls
  (S13's video-split leak, R12's unrepairable lies, the sabotage check on the results
  writer). 166 tests, one command, no pytest dependency.
- **Results reach disk, and artefacts enable re-analysis:** every table is written to
  `results/` from in-memory rows (9,515 s of output once survived only as scrollback);
  `val_logits.npz` and the per-claim JSONL turn every post-hoc question — τ sweeps,
  subset search, tolerance sensitivity, quotable failure examples — into CPU-seconds
  instead of a 2-hour GPU booking.
- **Refuse, don't warn:** wrong hyperparameter pairings (focal α on majority-positive
  data, balanced sampling + logit-adjusted loss, resume across split identities, stale
  code snapshots) all *exit with the fix* rather than warning — a warning inside a
  12-hour offline log is read only after the GPU-hours are spent.

## The three decisions measurement overturned

Kept visible because they are the method working as intended:

1. **The 4-stream ensemble** was predicted to add +2–4 mean-class; it *loses* to single
   streams on mean-class and macro-F1 (product-of-experts veto vs a long tail).
2. **Grammar-constrained decoding** was the presumed hallucination fix; decoding-matched
   measurement shows no faithfulness gain at 7.6× cost — the deterministic verifier is
   what actually catches errors.
3. **The fitted transition matrix** was my predicted fix for the smoothing regression;
   it recovered 0.004 of 0.040. The working fix (adjust-then-smooth) came from the
   measurement, not the prediction.
