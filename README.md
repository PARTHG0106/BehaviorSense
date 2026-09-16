# BehaviorSense AI

**Identity-aware multi-agent behaviour monitoring with verifiable LLM reporting for elderly care.**

Successor to EmotionSense. The pivot: away from emotion recognition (weak healthcare
signal, poor dataset support, text-dependent) toward *longitudinal behavioural
intelligence* — which is what actually predicts decline in elderly residents.

> ### Built with [AgentRouter](https://agentrouter.org/register?aff=89mv)
>
> Huge thanks to **AgentRouter** for making this project possible. Their API gave us
> sustained access to Claude Opus 5, and that is what turned a month of
> evenings into a system with 152 tests, six offline Kaggle notebooks, and results we can
> defend — including the ones that came out against our own hypothesis.
>
> You can sign up here:
> **https://agentrouter.org/register?aff=89mv** *(referral link)*

---

## Status

| Phase | State |
|---|---|
| Phase 0 — critique & scope lock | ✅ [`docs/00_scope_and_critique.md`](docs/00_scope_and_critique.md) |
| Phase 1 — dataset catalog | ✅ [`docs/01_dataset_catalog.md`](docs/01_dataset_catalog.md) |
| Phase 2 — architecture & model selection | ✅ [`docs/02_architecture.md`](docs/02_architecture.md) |
| Phase 3 — data pipeline design | ✅ [`docs/03_data_pipeline.md`](docs/03_data_pipeline.md) |
| Phase 4 — roadmap | ✅ [`docs/04_roadmap.md`](docs/04_roadmap.md) |
| Phase 5 — execution plan | ✅ [`docs/05_plan.md`](docs/05_plan.md) |
| Phase 6 — technical playbook (A–Z) | ✅ [`docs/06_technical_playbook.md`](docs/06_technical_playbook.md) |
| Algorithms & workflows, step by step | ✅ [`docs/08_algorithms_and_workflows.md`](docs/08_algorithms_and_workflows.md) |
| Agent 1 — perception | ✅ implemented + tested (19/19) |
| Agent 2 — activity recognition | ✅ implemented + tested (11/11) |
| Agent 3 — behaviour analysis | ✅ implemented + tested (16/16), evaluated |
| Agent 4 — faithfulness verifier | ✅ implemented + tested (12/12) |
| Agent 4 — LLM reporter | ✅ implemented + tested (13/13), benchmarked |
| ST-GCN++ + training stack | ✅ implemented + tested (21/21), smoke-verified |
| Open-set ReID protocol + fitted τ | ✅ implemented + tested (12/12), **τ measured on MSMT17** |
| Ensemble loader + frames→segments pipeline | ✅ implemented + tested (15/15) |
| Longitudinal simulator | ✅ implemented + 28-scenario eval suite |
| FastAPI service + SQLite | ✅ implemented + tested (6/6), DB isolation asserted |
| Docker + requirements + preflight + Charades map | ✅ |
| Front end + video upload + Kaggle backend | ✅ [`web/`](web/README.md) + [`notebooks/05`](notebooks/05_serve_inference_online.ipynb); verifier ported to JS, cross-checked against the server |
| ADL shard extraction (Charades → skeletons) | ✅ **165,109 windows / 7,985 videos**, [results/adl_extraction.md](results/adl_extraction.md) |
| Fall shard extraction (4 corpora → skeletons) | ✅ **3,064 windows / 520 clips**, Le2i via PyAV fallback |
| GPU training run (P1/P2 + ablations + Agent 4 table) | ✅ **all measured, person-disjoint**, [results/evaluation.md](results/evaluation.md); P3 staged→wild still pending |
| Accuracy levers (5, all measured) | ✅ logit-adjust **+0.053 mean-class**, subsets **+0.018 macro-F1**, second-person **class-18 recall 0.060→0.148**, TTA nil, Viterbi −**0.032** (a regression the frag ratio then resolved) |
| Person-disjoint P1 (Charades actor ids) | ✅ **retrained + re-evaluated**; split-identity guard refuses cross-identity resume |

**167 tests pass** across 11 suites (146 fast + 21 training). The full four-agent path is
implemented and verified on CPU (`scripts/run_full_pipeline.py`): RawDetection →
PersonObservation → ActivitySegment → DailyFeatures → BehaviourState → Claim →
VerificationResult. P1, calibration, P2, the combination ablation and the Agent 4
hallucination table are all measured on real data
([results/evaluation.md](results/evaluation.md)); P3 staged→wild transfer is the one
protocol still outstanding.

---

## Architecture

```
video ──► AGENT 1  RTMO detect+pose ──► BoT-SORT ──► OSNet ReID ──► role
                                                                     │
          AGENT 2  skeleton windows ──► ST-GCN++ ──► 20-class ADL ◄───┘
                   + RT-DETR object context (1 Hz, late fusion)
                                                    │
          AGENT 3  daily features ──► robust baseline (median/MAD)
                   ──► robust-z deviation ──► CUSUM drift ──► Alerts
                                                    │
          AGENT 4  Qwen2.5-7B, schema-constrained ──► Claims
                   ──► DETERMINISTIC VERIFIER ──► verified report
```

Agents communicate through **versioned Pydantic schemas persisted to SQLite**, not
function calls. Each stage is independently testable and replayable — Agent 3 and 4 can
be re-run in seconds without touching video.

**The LLM never sees pixels.** It receives structured state only. That boundary is what
makes the auditability claim meaningful.

---

## The research contribution

Most "LLM explains health monitoring" systems cannot answer the obvious question:
*how do you know the explanation is true?*

This system answers it arithmetically. Every generated claim must carry a
machine-resolvable `evidence_ref`, and a **deterministic verifier** (no LLM — using a
model to check a model shares failure modes) applies four independent checks:

| Check | Catches |
|---|---|
| C1 `ref_exists` | fabricated citations |
| C2 `value_matches` | invented numbers |
| C3 `pct_matches` | wrong percentages |
| C4 `direction_consistent` | **inverted narration** — right number, backwards story |

C4 is the one that matters most. A report stating *"mobility improved by 50%"* when
mobility **fell** 50% passes C1–C3 and would reach a caregiver. Verified caught:

```
V6 inverted narration caught despite correct value:
   - pct mismatch: claimed +50.0%, actual -50.0%
   - direction field wrong: claimed increase, actual decrease (delta -900)
   - prose says 'increase' but data shows 'decrease' (delta -900)
   - quoted pct sign (+50.0%) contradicts actual delta (-900)
```

Headline metric: **hallucination rate**, reported constrained vs unconstrained.

---

## Why the behaviour layer is statistical, not learned

No public dataset contains multi-week, identity-resolved elderly home video. A neural
baseline model would be trained on simulated data and validated on simulated data —
circular, and it would not survive review.

Robust statistics instead:

- **Baseline** — rolling 14-day median + MAD. MAD has a 50% breakdown point; mean/std
  has 0%, so one hospital visit would corrupt the baseline (Leys et al., 2013).
- **Deviation** — robust z, `0.6745·(x−median)/MAD`, MAD floored at 5% of |median|.
- **Drift** — one-sided CUSUM (Page, 1954) for gradual decline.

**Why the MAD floor exists.** For a resident with a rigid routine, `medication_events`
is 1 every day → MAD = 0 → an unfloored z-score divides by zero, and a naive ±6 fallback
makes a 0.1% change look like a 6-sigma event. That fires alerts on noise for precisely
the most regular residents — the classic alert-fatigue failure mode that gets monitoring
systems switched off. Regression-tested in `test_t6b`.

**Why CUSUM, demonstrated not asserted** (`test_t6`, noisy baseline σ≈12%, decline
1.2%/day):

```
T6 detected day 37 (16 days into decline, 18.6% cumulative), z=-2.17
T6 daily |z|>3 first at day 45 -> CUSUM lead = 8 days
```

Daily thresholding is 8 days slower and only fires once the decline is already 25%+.
This is the gap CUSUM closes, and the reason "detection lead time in days" is the metric
to lead with.

---

## Measured results

28 scenarios — 4 personas (day-to-day variability CV 0.05–0.28) × (6 injected anomalies
+ 1 no-anomaly control), 150 days each. Full report: [results/behaviour_eval.md](results/behaviour_eval.md).

| metric | value |
|---|---|
| recall (targeted alert kind) | **100%** (24/24) |
| median detection latency | **4 days** |
| control false-alert rate | **5.07 per 100 resident-days** |

| anomaly | recall | median latency |
|---|---|---|
| fall_event | 100% | 0 d |
| acute_mobility_drop | 100% | 1 d |
| meal_skipping | 100% | 4 d |
| medication_lapse | 100% | 4 d |
| social_withdrawal | 100% | 10 d |
| gradual_mobility_decline | 100% | 20 d |

The false-alert rate started at **67.66** per 100 control-days and reached 5.07 through
five fixes, each one a distinct defect found by evaluation rather than inspection —
absence inferred from camera outages, an unstratified baseline flagging every weekend,
R10 multiple comparisons, social withdrawal having no drift detector, and CUSUM charging
on noise. Every parameter change is justified against measurement in
[results/tuning_log.md](results/tuning_log.md), including the Monte Carlo that rejected
the *lowest* false-alarm setting because it missed 80% of real declines.

> Simulator results. They bound the algorithm's sensitivity given clean features; they
> are not estimates of clinical accuracy. See [docs/00_scope_and_critique.md](docs/00_scope_and_critique.md).

---

## Quick start

Everything below runs on **CPU, with no checkpoints and no internet** — the behaviour,
verification and reporting layers have no deep-learning dependency by design.

```bash
pip install -r requirements-serve.txt
```

**1. Run the tests** (~4 min for the fast suites; the two notebook suites dominate,
because they execute real notebook cells and real subprocess entry points):

```bash
python run_tests.py
```

**2. See the whole system work end to end** — simulator → Agent 3 → Agent 4 → verified report:

```bash
PYTHONPATH=src python scripts/run_demo.py
```

**3. Watch the verifier catch a lying model** — the demo that shows the contribution:

```bash
PYTHONPATH=src python scripts/run_demo.py --hallucinate 0.5 --seed 3
```

**3b. Run all four agents back to back** — detections → identity → activity → statistics →
verified report, every schema boundary crossed for real (plumbing demo, not accuracy):

```bash
PYTHONPATH=src python scripts/run_full_pipeline.py --days 8 --inject-fall
```

**4. Reproduce the behaviour-layer evaluation** (28 scenarios, 150 days each):

```bash
PYTHONPATH=src python -m behaviorsense.eval.behaviour_eval
```

**5. Reproduce the hallucination benchmark**:

```bash
PYTHONPATH=src python scripts/eval_hallucination.py --days 60 --report results/hallucination.md
```

**6. Start the API**:

```bash
PYTHONPATH=src uvicorn behaviorsense.service.api:app --reload
```

Then open http://localhost:8000/docs. Or with Docker:

```bash
docker build -t behaviorsense . && docker run -p 8000:8000 behaviorsense
```

**7. Open the front end** — no build step, no dependencies:

```bash
python -m http.server 5173 --directory web
```

The daily report at the top is verified *in the browser*: `web/scripts/verifier.js` is a port
of the Python verifier with the same tolerances and word lists, so the four cells beside each
claim are real check results rather than a picture of some.

The models do not run there. Qwen2.5-7B needs ~16 GB in bf16 and the ADL ensemble needs a GPU,
so `notebooks/05_serve_inference_online.ipynb` serves both from a Kaggle T4/P100 behind a
Cloudflare tunnel and prints an address you paste into the page. `Generate live` then runs the
real model, and the page **re-derives the verdicts from the evidence it was sent and compares
them with the server's** — two independent implementations agreeing on live output, with
disagreement reported as a reason to distrust the page rather than hidden. Details, states and
the security caveats: [web/README.md](web/README.md).

The **On your own footage** band takes a clip and runs Agents 1 and 2 on it: RTMO poses for
every person in shot, the tracker holding an identity across frames, OSNet deciding which one
is the resident, and the ensemble labelling each person's activity on their own windows. The
skeletons are drawn back over the playing video from the coordinates the model returned, with
one timeline per person and fall segments struck in pencil.

It does **not** produce a caregiver report, and says so in the payload. Agent 3's baseline is a
rolling 14-day median, so a single clip has no history to deviate from — and fabricating that
history would yield a report verified against invented numbers. Decode runs in a child process,
because ffmpeg raises SIGSEGV on malformed streams and a signal is not catchable: inline, one
bad upload would take down the kernel, the tunnel and the demo together.

To develop that path without booking a GPU:

```bash
python web/dev_backend.py    # same routes, no model, one deliberately false claim
```

### Fitting the ReID threshold on real data

Requires a Market-1501/MSMT17-layout dataset directory. CPU-only, ~9 min for 2,912 crops:

```bash
python scripts/extract_reid_embeddings.py --root MSMT17 --weights weights/osnet_ain_x1_0_msmt17.pth --out results/embeddings/msmt17.npz
```

```bash
python scripts/eval_reid.py --root MSMT17 --embeddings results/embeddings/msmt17.npz --report results/reid_eval.md
```

### On Kaggle (recommended: five notebooks, online/offline split)

Training needs internet OFF → the work splits into ONLINE notebooks (stage
wheels/weights/shards, CPU/T4) and OFFLINE notebooks (train/eval, no
internet). The five-notebook workflow, dataset DAG, and session sequence are in
[docs/07_kaggle_plan.md](docs/07_kaggle_plan.md); the notebooks themselves live in
[notebooks/](notebooks/) and are generated from
[notebooks/_generate.py](notebooks/_generate.py) — edit the generator, never the `.ipynb`.

Before the first Kaggle session, prove the offline wheel staging can actually resolve
(needs internet, ~1 min, downloads a few hundred KB rather than the 1 GB wheel):

```bash
python scripts/verify_wheel_resolution.py
```

Notebook 00 died in pip's resolver twice, and each attempt cost a session; this checks
every pinned `nvidia-*`/`triton` dependency of torch 2.7.1+cu128 against the `--platform`
ladder the notebook uses. `tests/test_notebooks.py` N9/N10 keep the notebook, the verifier
and the staging preflight from drifting apart.

### On a Kaggle GPU session (scripts, single-session alternative)

Always run preflight first — it catches an unsupported compute capability or a missing
asset in one minute, instead of 40 minutes into a run:

```bash
PYTHONPATH=src python scripts/kaggle_smoke_test.py --assets /kaggle/input/behaviorsense-weights
```

```bash
PYTHONPATH=src python scripts/train_adl.py --shards data/shards/*.npz --stream joint --epochs 80 --out runs/adl_joint
```

Training resumes exactly after a session timeout — RNG state is checkpointed, and
resuming with different hyperparameters is refused rather than silently re-curving the LR
schedule:

```bash
PYTHONPATH=src python scripts/train_adl.py --shards data/shards/*.npz --stream joint --epochs 80 --out runs/adl_joint --resume runs/adl_joint/last.pt
```

---

## Repo layout

```
configs/taxonomy.yaml                 unified 20-class ADL taxonomy + source mappings
docs/                                 design documents (00-08)
results/behaviour_eval.md             28-scenario evaluation report
results/tuning_log.md                 every threshold, and the measurement behind it
results/reid_eval.md                  fitted open-set ReID operating point
results/hallucination.md              hallucination rate + calibration anchors
results/evaluation.md                 P1 / calibration / P2 / ablation / levers, from the GPU run
results/val_logits.npz                saved val logits — replay accuracy levers on CPU
src/behaviorsense/
  schemas.py                          inter-agent contracts, robust statistics
  agents/perception.py                Agent 1: track, open-set ReID, role assignment
  agents/activity.py                  Agent 2: calibration, Viterbi, object fusion
  agents/behaviour.py                 Agent 3: baseline, drift, 10 alert rules
  agents/reasoning/verifier.py        Agent 4: deterministic faithfulness verification
  agents/reasoning/reporter.py        Agent 4: constrained generation + verified render
  models/stgcnpp.py                   ST-GCN++ (self-contained, no mmcv)
  models/ensemble.py                  checkpoint→WindowClassifier bridge (4-stream logit avg)
  models/osnet.py                     OSNet-AIN loader + embedder
  pipeline.py                         Agent 1→2 seam: frames → per-track windows → segments
  video.py                            uploaded clip → poses → observations, crash-isolated
  data/simulator.py                   longitudinal simulator with injected ground truth
  data/skeleton_dataset.py            shards, augmentation, subject splits, sampling
  data/reid_datasets.py               open-set ReID protocol construction
  eval/behaviour_eval.py              scores Agent 3 against injected onsets
  eval/reid_eval.py                   AUROC / TAR@FAR / DIR@1 / EER
  service/api.py                      FastAPI + SQLite persistence
scripts/                              train, evaluate, extract, demo, preflight
tests/                                167 tests, all runnable without pytest
run_tests.py                          one command, one verdict
web/                                  static front end + Kaggle-backed live mode
```

---

## Measured results — Agent 2 activity and falls (real data)

30-epoch budget with `--patience 8`, EMA weights. **35,698 val windows over 41 held-out
actors** — a person-disjoint split via the Charades `subject` column, not the video-id proxy
this project used until it was caught (see limitation 5). Full report:
[results/evaluation.md](results/evaluation.md).

| configuration | top-1 | mean-class | macro-F1 |
|---|---|---|---|
| `joint` alone | 0.327 | 0.181 | 0.141 |
| `bone` alone | 0.324 | **0.187** | **0.146** |
| `joint_motion` | **0.344** | 0.121 | 0.094 |
| `bone_motion` | 0.333 | 0.121 | 0.090 |
| **ensemble** (logit avg, P1 headline) | **0.375** | 0.156 | 0.128 |
| ensemble, prob avg | 0.358 | 0.169 | 0.131 |
| ensemble + logit-adjust τ=0.25 | 0.239 | 0.209 | 0.135 |
| `bone+joint` + logit-adjust τ=0.25 | 0.185 | **0.219** | 0.133 |
| ensemble + TTA flip | 0.376 | 0.156 | 0.129 |

Effective sample size is **41 people**, not 35,698 windows — windows from one actor are not
independent, so treat gaps under ~2 points as unresolved.

**The ensemble wins top-1 by 4.8 points and loses 2.5 of mean-class and 1.8 of macro-F1
versus `joint` alone.** Reported as measured, against the prediction. Logit averaging is a
product of experts, so a stream that is confidently wrong on a rare class can veto it; with
two streams at mean-class 0.121 that veto trades tail recall for head accuracy. Probability
averaging recovers part of it, and **neither combination beats `bone` alone on any
class-balanced metric** — four-stream averaging is simply the wrong default for this label
distribution.

**Logit adjustment moves mean-class and costs macro-F1**, and it needs no retraining:
subtracting `τ·log(prior)` at decision time takes mean-class from 0.156 to 0.209 (ensemble) or
0.219 (`bone+joint`). τ peaked at 0.25, well below 1.0 — consistent with effective-number
sampling having already removed most of the imbalance during training.

But mean-class is macro-*recall*, and under the honest split the two metrics now disagree at
the top. Every τ>0 configuration costs macro-F1 (0.146 → 0.133–0.135): the extra tail recall is
bought with precision. Agent 3 sums window predictions into daily *durations*, so
over-predicting a class inflates a duration exactly as missing one deflates it — which makes
**macro-F1 the right arbiter here, not mean-class**. On macro-F1 the winner is **`bone` alone
at 0.146**, with no adjustment at all, and the P1 headline ensemble is the worst credible
option at 0.128. Dropping the ensemble for one stream is the single largest free gain:
**+0.018 macro-F1**.

**TTA flip is a wash: +0.001 top-1, ±0.000 mean-class, +0.001 macro-F1.** Confirmed on both
splits. The mirror is a transform the model trained on, so it had already learned the
invariance. Measured, reported, not deployed.

**Second-person context works exactly where it was aimed.** Every shard already carries two
person slots, and slot 1 is non-zero when the tracker held a second person — the signal
`OBJECT_PRIORS["person"]` was written for and had never received. It fires on 7.5% of windows
and lifts `interacting_with_person` recall **0.060 → 0.148 at unchanged precision** (F1 0.047 →
0.058), while global mean-class dips 0.156 → 0.152 because the offset costs other classes where
it fires. A targeted fix with a small global cost — worth deploying because
`social_interaction_duration_s` feeds Agent 3's social-withdrawal alert, not worth it as a
general lever. It is the cheap half of object context; the three classes needing a real
detector are untouched.

**Segment-level decoding found a live regression, my fix for it mostly failed, and the
fragmentation column changed the answer.** Agent 3 consumes smoothed segments, never windows,
and that path had never been evaluated on real data.

| decoding | top-1 | mean-class | frag |
|---|---|---|---|
| per-window argmax | 0.375 | 0.156 | 1.86 |
| Viterbi, hand-set prior (0.90) | **0.403** | 0.124 | 0.48 |
| Viterbi, fitted prior (0.764) | **0.403** | 0.126 | 0.48 |
| argmax + logit-adjust τ=0.25 | 0.239 | **0.209** | 3.32 |
| **Viterbi fitted + logit-adjust τ=0.25** | 0.298 | 0.194 | **0.62** |

I predicted the cause was a mis-specified self-transition and fitted the matrix from 165,109
training windows instead. The fitted value is 0.764, so the hand-set prior *was* too sticky —
and correcting it recovered **0.002 of the 0.032**. The regression is intrinsic to max-product
decoding over a posterior whose head class holds ~40% of the mass: whatever the transition
rates, the cheapest path through a short rare-class run is to absorb it into the majority.

On label quality alone, argmax + adjustment wins at 0.209 — and it **fragments 3.32×**. Agent 3
counts `walking_bouts` and `mean_bout_duration_s` from segment structure, so deploying that
would report roughly three times the true bouts at a third of the true mean duration while
looking best in the accuracy column. `Viterbi fitted + logit-adjust` is the deployment choice:
mean-class 0.194 (0.015 behind) at frag 0.62, the only row defensible on both axes — and one
that neither column would have selected alone.

The fall head is strong in domain and does not transfer:

| metric | value |
|---|---|
| AUPRC (validation) | **0.822** |
| AUROC (validation) | 1.000 |
| sensitivity @ 0.99 false alarms/hour | **0.951** |
| negative duration behind that operating point | 23.2 h |
| positive rate | 0.49% |
| **AUROC, held-out corpora** | **0.471 – 0.603** |

AUPRC rather than AUROC, because AUROC flatters at a 0.49% positive rate. An operating point
quoted per hour needs enough negative *hours* to be a decision rather than an artefact — the
earlier 141-negative pool was 0.08 h.

**But a validation AUROC of 1.000 beside held-out AUROCs of 0.471–0.603 is the finding.** A
perfect in-domain ranking next to at-or-below-chance transfer means the validation negatives
are separable by something other than "is this a fall" — recording setup, camera geometry,
compression. 0.822 AUPRC is an in-domain ceiling, not a portable claim, and the operating point
is scoped to the training distribution.

Cross-corpus transfer (P2, leave-one-dataset-out) is weak, corpus-dependent, and got **worse**
under the honest split: AUROC 0.603 caucafall, 0.571 le2i, 0.560 urfd, 0.471 gmdcsa (was
0.734 / 0.586 / 0.722 / 0.545 with the leaky checkpoints). GMDCSA is below chance. Calibration
fits T = 0.63 (mean confidence 0.381 vs accuracy 0.375) — down from 0.76, i.e. the honest
ensemble is *more* under-confident, which is what you would expect once it can no longer
recognise the person.

### Five accuracy levers, measured — two work, one is targeted, one is a wash, one found a bug

0.187 mean-class over 20 classes is poor, and P1's own numbers said where the slack was. All
five are measured rather than assumed; four are pure post-processing on logits the notebook
already computes, and the fifth is one extra forward pass:

| lever | what it changes | outcome |
|---|---|---|
| **logit adjustment** | subtracts `τ·log(prior)` at decision time; τ peaked at 0.25 | free, **+0.053 mean-class**, −0.011 macro-F1 |
| **stream subset selection** | all 15 subsets, both rules; `bone` alone won | free, **+0.018 macro-F1** |
| **second-person context** | slot-1 occupancy → `OBJECT_PRIORS["person"]` | free, **class-18 recall 0.060 → 0.148** |
| **test-time flip** | averages each window's logits with its mirror | 1 extra pass, **no gain** |
| **Viterbi + fragmentation** | the metric Agent 3 actually consumes | free, **found a −0.032 regression** and picked the deployment config |

Logit adjustment is warranted because training already uses effective-number balanced
sampling, which **saturates by design**: at 65,000 vs 69 windows it removes ~145× of a ~942×
ratio, so residual imbalance survives it. τ is swept rather than derived, because how much the
sampler left is a property of the data — and it peaked at 0.25, well below 1.0, exactly as that
reasoning predicts. Subset selection is warranted because the four-stream average *already
lost* to `bone` alone, so it was never automatically right and there were 15 answers.

The segment metric mattered most, and not in the direction hoped for: it exposed a live
regression rather than an improvement, then the fragmentation column overturned the conclusion
the accuracy column alone would have produced. Every other accuracy number here is per-window,
and Agent 3 derives every daily feature from smoothed *segments* — a path `viterbi()` and
`smooth()` had never been evaluated on with real data since Agent 2 was written.

`train_adl.py` also gained `--sampler natural --tau-train τ` for logit-adjusted *training*,
which is stronger than the post-hoc version but costs a run. It **refuses** to combine with
balanced sampling — two corrections for one imbalance is a silent methodological error, so
the script exits with the fix rather than warning inside a 12-hour log (test S11).

And the artefact that makes all of this repeatable: notebook 04 writes
`results/val_logits.npz` (~8 MB). It had computed those exact arrays four times across four
sessions and discarded them four times. `scripts/rescore_p1.py` replays every post-hoc lever
off that file on a laptop in seconds — which is how τ was swept without booking a GPU.

---

## Measured results — Agent 1 identity (real data)

Open-set ReID on **MSMT17**, 100 enrolled identities and 400 unseen impostor identities,
τ fitted at FAR ≤ 1% on one identity-disjoint half and reported on the other. Full report:
[results/reid_eval.md](results/reid_eval.md).

| metric | value |
|---|---|
| fitted threshold τ | **0.355** (was 0.30, guessed) |
| TAR (enrolled accepted) | **96.86%** |
| FAR (stranger accepted as resident) | **2.25%** |
| DIR@1 (accepted *and* correctly named) | 96.65% |
| AUROC | 0.9978 |
| closed-set rank-1 (calibration check) | 99.58% |

The negative control is what makes those numbers trustworthy: ImageNet-only weights run
through the identical protocol score **AUROC 0.5343, TAR 3.35%** — a +0.4634 gap. Without
it, a protocol bug that flattered any embedding would be indistinguishable from a working
identity layer.

## Measured results — Agent 4 faithfulness

696 claims over 116 analysed days ([results/hallucination.md](results/hallucination.md)):

| generator | hallucination rate |
|---|---|
| faithful | **0.0%** |
| corrupted @ 25% | 24.7% |
| corrupted @ 50% | 52.0% |
| corrupted @ 100% | 100.0% |

Both anchors matter. 0.0% on honest output proves the verifier has no false positives, so
every rate it reports is real; tracking the injection rate proves it does not saturate, so
it can rank two models. All six corruption types — fabricated citation, invented value,
wrong percentage, inverted direction field, contradictory prose, sign-flipped percentage —
are caught **by the specific check responsible**, verified over 144 injections.

The table also reports which of C1–C4 fired. On the corrupted arms C4 (direction) catches
the most claims, which is the point of having it: C1–C3 alone would pass a report that
quotes the right number and narrates it backwards.

### Measured on the real model

Qwen2.5-7B-Instruct, 116 analysed days, 6 personas. Both arms send byte-identical templated
text with identical greedy decoding, so the only difference between them is the grammar
([results/evaluation.md](results/evaluation.md)):

| condition | emitted | faithful | hallucination rate | 95% CI | s/report |
|---|---|---|---|---|---|
| grammar-constrained | 518 | 478 | **7.7%** | 5.7–10.3% | 50.3 |
| unconstrained | 514 | 472 | **8.2%** | 6.1–10.9% | 6.6 |
| **pooled** | **1,032** | 950 | **7.9%** | 6.4–9.8% | — |

**Grammar-constrained decoding does not measurably improve faithfulness** — −0.4 points,
z = −0.27, **p = 0.79**. Reported as measured, against the expectation. The smallest
difference this design could detect is ~3 points, so the true effect would have to be seven
times larger than observed to register. What the grammar buys is a *worst-case guarantee*
that no invalid claim can be emitted, where the free arm merely happened to emit none; worth
having in a caregiver-facing system, but it costs **7.6× the decoding time** and is not a
faithfulness result.

Getting to a comparison that meant anything took four GPU sessions and four defects, none of
them in the model: the prompt withheld the output envelope, the grammar allowed twice the
claims the reporter kept, the constrained arm was sampling at temperature 0.7 while the free
arm was greedy, and the constrained arm skipped the chat template the free arm applied. All
four were visible on CPU by reading the two code paths side by side. `test_r14` now audits
every axis on which the arms can differ, in under a second.

What is robust across all four runs: **C1 = 0**, always. The model has never fabricated a
citation, because it is handed resolvable evidence refs rather than asked to synthesise them
— one design choice removing one failure mode entirely. **C2 carries essentially all of the
residual** (39 of 40, 42 of 42): quoting a number more than 2% off the stored value. **C4 is
rare and not reliably estimable here** — 3, 3, 3, 3, 2, 0 occurrences per ~500 claims across
runs. The check exists because inverted narration is catastrophic when it happens, not
because it is frequent.

So the contribution never rested on constrained-versus-free. It rests on this: **1 claim in
13 that a 7B model makes about a resident's day is wrong against the data it was given**, and
a deterministic arithmetic check catches all of it before anyone reads it.

### Two denominators, deliberately

The first real Qwen run scored the unconstrained arm as **245 claims emitted, 0 scorable,
rate `nan`** — a number that measures JSON compliance and says nothing about truthfulness.
Every row now carries both:

- **hallucination rate** — over schema-valid claims. The verifier's own view.
- **unusable rate** — over *emitted* claims, counting schema rejects as failures. A claim
  thrown out by the schema is no more usable to a caregiver than one that is fluent and
  false, so an arm that lands nothing reads 100%, not `nan`.

A third arm re-scores the unconstrained arm's *cached* responses after format-only repair
(`repair_claim`: whitespace, `"1800"` → `1800.0`, `"Decreased"` → `"decrease"`,
`YYYY/MM/DD` → `YYYY-MM-DD`). It never invents a reference, changes a number, or infers an
unstated direction — asserted by a negative control that replays all five content
corruptions through repair and requires every one to still fail verification.

On both later runs that arm repaired **0 of 509 claims**, which is itself the result: once
the prompt stated the output contract, free decoding produced nothing malformed. The 245
rejections in run 1 came from a prompt that named `evidence_ref` and `claimed_value` in
prose but never gave the JSON envelope — no `claim_id`, no `text`, no `claims` array. Fixing
that removed **more than 30 points** of measured "hallucination" that was never the model's
fault, the same class of error as the earlier zero-baseline defect.

Every claim and its verdict are now written to JSONL beside the report, so error analysis,
quotable examples and tolerance sensitivity are a CPU rescore rather than another two-hour
GPU session — a lesson learned by needing exactly that and not having it.

---

## Key engineering decisions

| Decision | Rejected alternative | Reason |
|---|---|---|
| Skeleton-first ADL | RGB VideoMAE/SlowFast | ~1000× less data, hours not GPU-days, privacy-preserving |
| RTMO (one-stage) | RTMPose + detector | No detector dependency; faster beyond ~4 people |
| RT-DETR | YOLOv8/v11 | Apache-2.0 vs AGPL-3.0; NMS-free |
| BoT-SORT | ByteTrack / DeepSORT | Native ReID integration — we need embeddings anyway |
| Role classification | Face recognition | CCTV lacks facial pixels; ADL datasets blur faces → untrainable *and* untestable |
| Qwen2.5-7B text-only | Qwen-VL / InternVL | A VLM would blur the perception boundary and void auditability |
| Statistical behaviour layer | Learned baseline model | No multi-week real data exists → would be unfalsifiable |

Full justifications with citations: [`docs/02_architecture.md`](docs/02_architecture.md).

---

## Datasets

All Tier-1 sources are downloadable without human approval. Full table with licenses,
sizes, and preprocessing in [`docs/01_dataset_catalog.md`](docs/01_dataset_catalog.md).

| Purpose | Dataset | Gate | License |
|---|---|---|---|
| Falls (anchor) | [OmniFall](https://huggingface.co/datasets/simplexsigil2/omnifall) | HF account | CC BY-NC-SA 4.0 (annotations) |
| ADL | [Charades](https://prior.allenai.org/projects/charades) | none | AI2 academic, no redistribution |
| Falls | [CAUCAFall](https://data.mendeley.com/datasets/7w7fccy7ky/4) | none | **CC BY 4.0** |
| Falls | [GMDCSA-24](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos) | none | **MIT** |
| Falls | [URFD](https://fenix.ur.edu.pl/~mkepski/ds/uf.html) | none | research use, per-file URLs |
| Falls | Le2i (Charfi et al. 2012) | none | CC BY-NC-SA 3.0 — **origin host dead**, see below |
| ReID (primary, reproducible) | Market-1501 | Kaggle | research only |
| ReID (cross-dataset benchmark) | MSMT17 | signed licence | research only |
| Tracking eval | MOT17 | account | CC BY-NC-SA |

**Le2i provenance, stated rather than glossed.** The canonical host (`le2i.cnrs.fr`)
refuses connections and the IMVIA successor page 404s; the Wayback index holds 44 captures
of that domain, none of them the video files. The dataset is reachable only through a
Kaggle mirror (`tuyenldvn/falldataset-imvia`, 10 GB, licence field "Unknown"). Notebook 02
therefore attaches that mirror rather than downloading, and the dissertation cites
Charfi et al. (2012) as the source with the mirror named as the access path. Derived
shards stay private either way.

Gated but requested (**not depended on**): Toyota Smarthome (18 subjects aged 60–80 —
the only real elderly ADL video benchmark), NTU RGB+D 120.

⚠️ Charades, Market-1501 and MSMT17 forbid redistribution → **derived Kaggle datasets must
be private**. Publish code and metrics, never the tensors.

**MSMT17 and the DukeMTMC exclusion.** DukeMTMC-reID is excluded from this project on
consent grounds. MSMT17 is also campus surveillance collected without individual consent,
so consistency requires stating the distinction rather than skipping it: DukeMTMC was
**withdrawn by its own authors**, MSMT17 has not been, and the copy used here was provided
through an institutional research channel. It is used strictly to *fit and report an
operating point* — never redistributed, and never on the reproducibility critical path,
which is why Market-1501 remains the primary. A reader who cannot obtain MSMT17 can still
reproduce the headline ReID number.

---

## Known limitations (stated deliberately)

1. **Age gap.** All ungated fall/ADL data features predominantly young adults. Gait,
   posture, and fall kinematics differ materially at 65+. Reported accuracies are an
   **upper bound** on real elderly performance.
2. **Behaviour layer validated on simulation.** No multi-week real longitudinal data
   exists publicly. Anomaly detection is evaluated against injected ground truth in a
   released simulator, not real clinical outcomes.
3. **Falls are staged.** Real falls differ from acted ones. Mitigated by evaluating
   staged→wild transfer on OmniFall's OOPS split, and reported honestly — expect P3 to
   be substantially below P1.
4. **No clinical validation.** No claim of diagnostic validity. This is a research
   prototype, not a medical device.
5. **P1 was video-disjoint, not person-disjoint — found late, fixed, retrained, reported.**
   The split used video ids on the recorded-and-wrong belief that Charades publishes no
   actor ids; `Charades_v1_train.csv` has a `subject` column (267 actors in the CSV, 209 in
   our windows, ~30 videos each), so nearly every person sat on both sides. This is the
   exact failure `subject_from_path` prevents in the fall corpora, committed on the biggest
   corpus, and the disjointness assertion could never catch it — the ids *were* disjoint.
   Every number in this README is now from the actor-disjoint retrain. The mean moved little
   (ensemble mean-class 0.152 → 0.156, best lever 0.222 → 0.219) but **macro-F1 fell 0.152 →
   0.128**: what leakage bought was *precision*, not recall. Peaks arrived far earlier (ep
   3–9 vs 10–17), habit-and-context classes collapsed (`cleaning_housework` 0.193 → 0.098,
   `personal_hygiene` 0.176 → 0.074, `eating` 0.164 → 0.082) while purely pose-defined
   classes *improved* (`lying_down` 0.408 → 0.435, `sitting` 0.318 → 0.341), and P2
   cross-corpus AUROC dropped across all four corpora. The pipeline remaps to actor ids
   whenever the CSV is attached and refuses to resume a checkpoint across split identities.
6. **Agent 2 accuracy is low, and measured.** Person-disjoint P1 mean-class is 0.187
   (`bone`, the best stream) over 20 classes, 0.219 with post-hoc logit adjustment, top-1
   0.375 for the ensemble — well above the 0.05 chance rate, well below anything deployable
   as a standalone activity classifier. Effective sample size is 41 actors, not 35,698
   windows. Four of the weakest classes have no skeleton-visible evidence at all, which is
   why object context is part of the architecture rather than an extension; one of the four
   (`interacting_with_person`) is now partly addressed by second-person slot occupancy, the
   other three need RT-DETR wired to `fuse_objects()`.
   See [results/evaluation.md](results/evaluation.md).
7. **The fall head is strong in domain and does not transfer.** Validation AUPRC 0.822 with
   AUROC **1.000**, sensitivity 0.951 at ~1 false alarm/hour over 23.2 h of negatives — but
   leave-one-dataset-out AUROC is **0.471–0.603** across CaucaFall, GMDCSA, Le2i and URFD,
   with GMDCSA below chance. A perfect in-domain ranking beside at-or-below-chance transfer
   means the validation negatives are separable by recording setup rather than by "is this a
   fall". 0.822 is an in-domain ceiling, not a portable claim, and the operating point is
   scoped to the training distribution. P3 staged→wild on OmniFall's OOPS split is the
   protocol that would settle how far it generalises and remains pending.
8. **`walking` is under-populated by the label map.** 545 windows (1.5% of val) at F1 0.093,
   for the most pose-separable activity in the taxonomy, in a corpus of people moving around
   their homes. The Charades keyword rules are almost certainly routing locomotion into
   `other_idle`. A class the map barely populates cannot be learned regardless of the
   objective, and this is one case where a data fix likely beats every decoding lever
   combined — `charades_map_review.tsv` needs an audit.
9. **SimpleTracker is a test double, not a tracker.** Association is motion-only, with a
   measured ceiling of 0.538 box-widths/frame; two people crossing at the same depth can
   swap IDs where real BoT-SORT would not. Deployment uses BoT-SORT with ReID-gated
   association; the stand-in exists so the identity logic is testable on CPU.

Unacknowledged limitations are punished far harder than acknowledged ones.

---

## Defects found by measurement, not inspection

Kept as a record because each one passed code review and was caught only by a test that
printed what it measured:

| defect | how it surfaced | consequence if shipped |
|---|---|---|
| `normalise()` erased whole-body descent | fall smoke test scored **AUROC 0.466** — below chance — on data whose only signal was a 1.5-unit hip drop | the fall head could never have learned falls; per-frame root-centring subtracts exactly the motion that defines one |
| abstention gated on the Viterbi-selected label | a spurious `other_idle` segment mid-walk in the fall test | abstention would discard precisely the windows smoothing had just repaired |
| 67.66 false alerts / 100 control-days | first honest Agent 3 evaluation run | alert fatigue; five distinct defects, now 5.07 |
| CUSUM charged 4.21 on pure noise | a *passing* test reporting 0-day detection latency | 18.8% per-feature false-alarm rate |
| open-set split silently shrank 100 → 60 ids | asserting the exact requested count | the experiment would differ from the one described, with no error |
| resume changed the LR curve | comparing a resumed run against an uninterrupted one | every number from a 12-hour-capped session unreproducible |
| reporter quoted "0%" against a zero baseline | the *faithful* stub scored 67% hallucination in the full-pipeline demo | a percentage change from a zero median is undefined, not 0% — the verifier was right, the prompt payload was offering an uncitable number |
| grammar allowed 12 claims, reporter kept 6 | the real Qwen run lost 9 of 116 constrained reports to `max_new_tokens` | budget spent generating claims that were then silently discarded; a report emitting 12 mediocre claims scored the same as one emitting 6 |
| unconstrained arm reported `nan%` | 245 claims emitted, 0 schema-valid, and the rate had a zero denominator | total failure would have read as missing data; the arm was measuring JSON compliance, not truthfulness |
| notebook 04 printed nine hours of tables and saved none | the bundling cell globbed `results/*.md` and found only a subprocess's file | P1, per-class, calibration, P2 and the ablation existed solely as session scrollback |
| the prompt never gave the LLM its output envelope | free decoding lost 245/245 claims to Pydantic; constrained decoding scored 39.9% "hallucination" | >30 points of the headline metric were our prompt, not the model — it fell to 6.5% once the field mapping was stated |
| P1 split by video id, not by actor | the Charades `subject` column exists after all — 267 actors, ~30 videos each, so nearly every person sat on both sides | "subject-disjoint" P1 was video-disjoint; the retrain cost 0.024 macro-F1 and collapsed the habit-and-context classes, which is what leakage had been buying |
| fall-head validation AUROC of 1.000 | leave-one-dataset-out AUROC came back 0.471–0.603, GMDCSA below chance | the in-domain negatives are separable by recording setup, not by "is this a fall" — 0.822 AUPRC is a ceiling, not a portable claim |
| accuracy alone chose the wrong decoder | argmax+adjustment won mean-class at 0.209 and fragmented segments 3.32x | Agent 3 counts bouts from segment structure, so it would have reported ~3x the true bouts while looking best in the accuracy column |
| one decoding arm sampled while the other was greedy | the free arm reproduced bit-identically across runs; the constrained arm moved 498/457 → 504/452 and flipped p=0.29 → p=0.03 | the constrained-vs-free comparison measured temperature, not grammar — `outlines` inherited Qwen's `do_sample=True, temperature=0.7` while the free path passed `do_sample=False` |
| resume moved the RNG state to the GPU | all five GPU runs died with `TypeError: RNG state must be a torch.ByteTensor`, while the CPU suite passed | `map_location="cuda"` moves every checkpoint tensor including the RNG state, and `set_rng_state` takes only a CPU ByteTensor — S7 resumes on CPU, where `map_location` is a no-op |
| P1 and training split at different fractions | notebook 04 evaluated at `val_frac=0.15` while `train_adl.py` stops at `0.20` | no leakage (the smaller set nests inside the larger), but P1 was not measured on the set that selected `best.pt` and threw away a fifth of the evidence |
| one arm skipped the chat template | found by reading the two code paths side by side after three runs, not from a log | `outlines` does not template a bare string, so one arm got a Qwen chat turn and the other a naked instruction block; under a grammar that shows up as worse *content*, not malformed JSON |

The pattern: **a green suite is not evidence something works — it is evidence the
assertion was satisfiable.** Every detection test in this repo prints the quantity it
measured, and every rejection test is paired with a positive control on the same
thresholds.

---

## References

- Lu et al. (2024) *RTMO: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation*, CVPR. [arXiv:2312.07526](https://arxiv.org/abs/2312.07526)
- Schneider et al. (2025) *OmniFall: A Unified Staged-to-Wild Benchmark for Human Fall Detection*. [arXiv:2505.19889](https://arxiv.org/abs/2505.19889)
- Sigurdsson et al. (2016) *Hollywood in Homes: Crowdsourcing Data Collection for Activity Understanding*, ECCV.
- Das et al. (2019) *Toyota Smarthome: Real-World Activities of Daily Living*, ICCV.
- Do & Kim (2024) *SkateFormer: Skeletal-Temporal Transformer for Human Action Recognition*, ECCV. [arXiv:2403.09508](https://arxiv.org/abs/2403.09508)
- Leys et al. (2013) *Detecting outliers: Do not use standard deviation around the mean*, JESP.
- Page (1954) *Continuous Inspection Schemes*, Biometrika.
