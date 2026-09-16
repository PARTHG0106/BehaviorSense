# Evaluation — P1 / levers / segments / P2 / ablation / Agent 4

Notebook-04 sessions on the RTX PRO 6000 Blackwell (sm_120). **All activity numbers in the
Charades sections are from the person-disjoint (actor-id) split** — the protocol of record.
Earlier sessions split by video id, which leaked the same actor into train and val; that
defect, and the measured cost of fixing it, is documented under P1. Notebook 04 now **refuses**
to run P1 without `Charades_v1_train.csv` attached rather than falling back to the leaky split,
because one session did exactly that and produced macro-F1 0.256 against the 0.128 recorded
here. The Toyota section at the end uses that corpus's own **CS (cross-subject)** protocol
instead, with its split stated there.

**The activity tables have now reproduced identically across three independent sessions** —
every figure in P1, the levers, the segment grid, calibration and the ablation. The Agent 4
table changed this session because the prompt and the verifier changed; the activity half did
not change and did not move.

Checkpoints: `adl_{joint,bone,joint_motion,bone_motion}/best.pt`, EMA weights, 30-epoch
budget with `--patience 8`. Under the honest split all four streams peaked at epochs 3–9 and
self-terminated at patience, so training is complete. The fall head carried forward untouched
(60/60 epochs, best AUPRC 0.822 @ ep56). Shards: 165,109 ADL windows (7,985 Charades videos,
209 mapped actors) + 3,064 fall windows (4 corpora).

## P1 — person-disjoint validation

35,698 val windows over **41 held-out actors** (209 mapped actors total), `val_frac=0.2`,
`seed=0`. These windows influenced training only through early stopping and `best.pt`
selection, which makes every number here validation-selected rather than held-out.

| model | top-1 | mean-class | macro-F1 | F1>=50 |
|---|---|---|---|---|
| `joint` | 0.327 | 0.181 | 0.141 | 0.149 |
| `bone` | 0.324 | **0.187** | **0.146** | **0.155** |
| `joint_motion` | **0.344** | 0.121 | 0.094 | 0.099 |
| `bone_motion` | 0.333 | 0.121 | 0.090 | 0.095 |
| **ENSEMBLE** (logit-avg) | **0.375** | 0.156 | 0.128 | 0.136 |

The ensemble again buys top-1 (+4.8 over `joint`) and loses mean-class (−2.5) and macro-F1
(−1.3): logit averaging is a product of experts, so the two motion streams at mean-class
0.121 can veto a rare class the strong streams got right.

**Effective sample size is 41 people, not 35,698 windows.** Windows from one actor are not
independent, so the confidence interval on any mean-class figure here is far wider than the
window count suggests. Treat differences under ~2 points between configurations as unresolved.

`bending_reaching` (26 val windows) remains starved by the Charades label map — Charades has
no verb for it, so "Putting a box somewhere" routes to `other_idle`. A label-map limitation
to state in the write-up, not a model result.

### The split-identity defect, and what fixing it cost

These are the **person-disjoint** numbers. The earlier ones were not.

The shards store video ids as `subjects`, on a belief recorded in `docs/07`: *"Charades
subject ids do not exist publicly."* That was wrong — `Charades_v1_train.csv` has carried a
`subject` column all along (267 actors in the CSV, 209 present in our windows, ~30 videos
each). At that ratio nearly every *person* appears on both sides of a video-id split. The ids
are disjoint; the people behind them are not. It is the exact failure `subject_from_path` was
written to prevent in the fall corpora, committed on the biggest corpus, and the disjointness
assertion could never catch it.

`train_adl.py --subject-map` now remaps the split to actor ids (notebook 03 passes it
automatically when the CSV is attached, notebook 04 auto-detects and prints the mode), and
the carry-forward guard **refuses** a checkpoint across split identities — verified live:
all four video-disjoint checkpoints were rejected with `split identity mismatch`, the fall
head carried forward untouched.

| | actor-disjoint | video-disjoint | delta |
|---|---|---|---|
| ensemble top-1 | 0.375 | 0.385 | −0.010 |
| ensemble mean-class | 0.156 | 0.152 | +0.004 |
| ensemble macro-F1 | **0.128** | **0.152** | **−0.024** |
| ensemble F1>=50 | 0.136 | 0.161 | −0.025 |
| best single stream (mean-class) | 0.187 `bone` | 0.190 `bone` | −0.003 |
| best lever (mean-class) | 0.219 | 0.222 | −0.003 |
| peak epochs | ep 3–9 | ep 10–17 | much earlier |

**The leak barely moved mean-class and clearly moved macro-F1.** −0.024 macro-F1 and −0.025
F1>=50 against +0.004 mean-class: mean-class is macro-*recall*, so what leakage was buying was
**precision** — the model was using person-specific cues to avoid false positives, not to find
true ones. That is the more honest reading than "the leak didn't matter", and it is only
visible because both metrics are reported.

**Peaks arrived far earlier (ep 3–9 vs 10–17).** The extra epochs in the video-disjoint runs
were spent learning things that transferred only to already-seen people. Consistent with that,
per-class composition shifted toward pose-readable classes even where the mean held:
`cleaning_housework` 0.193 → 0.098, `personal_hygiene` 0.176 → 0.074, `eating` 0.164 → 0.082 —
the habit-and-context classes where knowing the *person* helps most. (Read with care: the val
population changed too.)

## Accuracy levers — measured; two work, one is a wash, one is targeted

All post-processing on the same weights. No retraining.

| configuration | top-1 | mean-class | macro-F1 |
|---|---|---|---|
| ensemble, logit-avg (P1 headline) | 0.375 | 0.156 | 0.128 |
| ensemble + TTA flip | **0.376** | 0.156 | 0.129 |
| ensemble, prob-avg | 0.358 | 0.169 | 0.131 |
| `bone` alone (best logit-avg subset) | 0.324 | 0.187 | **0.146** |
| `bone+joint` (best prob-avg subset) | 0.333 | 0.187 | 0.144 |
| ensemble + logit-adjust τ=0.25 | 0.239 | 0.209 | 0.135 |
| `bone+joint` + logit-adjust τ=0.25 | 0.185 | **0.219** | 0.133 |
| ensemble + second-person context | 0.364 | 0.152 | 0.125 |
| ensemble + logit-adjust τ=0.25 + second-person | 0.245 | 0.205 | 0.134 |

**Logit adjustment remains the strongest lever: mean-class 0.156 → 0.209 on the ensemble,
→ 0.219 on `bone+joint`.** τ swept with τ=0 asserted as an exact no-op; peaked at 0.25, well
below 1.0, consistent with effective-number sampling having already removed most of the
imbalance and only a residue being left to correct.

**But under the honest split, macro-F1 now disagrees with mean-class, and it disagrees the
other way.** The best macro-F1 is `bone` alone at 0.146 — *no* adjustment. Every τ>0 row costs
macro-F1 (0.146 → 0.133–0.135) while raising mean-class. Under the leaky split these two
metrics agreed at the top (ensemble+τ won both); they no longer do. mean-class is
macro-*recall*, so what τ buys is tail recall paid for in precision, and Agent 3 sums window
predictions into daily *durations* where an over-predicted class inflates a duration exactly
as a missed one deflates it. **On the metric this pipeline actually needs, the answer is
`bone` alone at 0.146** — and the P1 headline ensemble is the worst credible option at 0.128.
Dropping the ensemble for one stream is still the single largest free gain: **+0.018 macro-F1**.

**TTA flip does nothing: +0.001 top-1, ±0.000 mean-class, +0.001 macro-F1.** Confirmed twice
now, on both splits. The mirror is a transform the model trained on (`flip_prob=0.5`), so it
had already learned the invariance. Code stays because the number is measured; not deployed.

**Second-person context works exactly where it was aimed, and only there.** Slot 1 of the
shard is non-zero when the tracker held a second person — the signal `OBJECT_PRIORS["person"]`
was written for and never received. It fires on 2,689 / 35,698 windows (7.5%) against a class-18
support of 614:

| class 18 (`interacting_with_person`) | predicted | correct | precision | recall | F1 |
|---|---|---|---|---|---|
| without context | 950 | 37 | 0.039 | 0.060 | 0.047 |
| with context | 2,549 | 91 | 0.036 | **0.148** | **0.058** |

Recall 2.5× at essentially unchanged precision, so class-18 F1 rises 0.047 → 0.058. Global
mean-class dips 0.156 → 0.152 because the +1.5 log-odds offset costs other classes on the
7.5% of windows where it fires. **A targeted fix with a small global cost** — worth deploying
if `social_interaction_duration_s` matters to Agent 3 (it feeds the social-withdrawal alert),
not worth it as a general accuracy lever. The honest read is that this is the *cheap half* of
object context; the classes needing a real detector (`taking_medication`, `watching_tv`,
`using_phone`) are untouched.

## Segment-level decoding — measured, and the fix I predicted mostly did not work

1,521 sequences, 32,992 windows. Agent 3 never sees a window: it consumes smoothed segments
and derives every daily feature from their durations.

| decoding | top-1 | mean-class | frag ratio |
|---|---|---|---|
| per-window argmax | 0.375 | 0.156 | 1.86 |
| Viterbi, hand-set prior (`self_transition=0.90`) | **0.403** | 0.124 | 0.48 |
| Viterbi, fitted prior (mean self-transition **0.764**) | **0.403** | 0.126 | 0.48 |
| argmax + logit-adjust τ=0.25 | 0.239 | **0.209** | 3.32 |
| Viterbi fitted + logit-adjust τ=0.25 | 0.298 | 0.194 | **0.62** |

1,629 sequences, 35,660 windows. `frag ratio` is predicted segments over true segments; 1.00
is ideal, above 1 fragments, below 1 over-merges.

**Smoothing under the hand-set prior costs 0.032 of mean-class** while gaining 0.028 top-1.
`viterbi()` and `smooth()` had existed since Agent 2 was written and had only ever been
exercised on synthetic fixtures.

**Fitting the transition matrix from 165,109 training windows recovered almost none of it:
0.124 → 0.126.** The hypothesis was that a hand-set 0.90 self-transition was too sticky for a
1 s stride; the fitted value is 0.764, so the prior *was* too sticky — and it barely mattered.
The regression is not a mis-specified self-transition. It is intrinsic to max-product decoding
over a posterior whose head class holds ~40% of the mass: whatever the transition rates, the
cheapest path through a short rare-class run is to absorb it into the surrounding majority.
Recorded because a prediction that fails is worth as much as one that lands, and this one was
mine.

**The fragmentation column changed the answer, which is exactly what it was added for.** On
label quality alone, argmax + logit adjustment wins at 0.209 mean-class — and it fragments
**3.32×**. Agent 3 derives `walking_bouts` and `mean_bout_duration_s` from segment counts, so
deploying that configuration would report roughly three times the true number of bouts and a
third of the true mean duration, while its per-window accuracy looked best in the table. Plain
Viterbi sits at the opposite failure: frag 0.48 over-merges by half and craters mean-class to
0.124.

**`Viterbi fitted + logit-adjust τ=0.25` is the deployment choice: mean-class 0.194 (0.015
below the label-quality winner) at frag 0.62 (closest to 1.0 of any smoothed variant).** It is
the only row that is defensible on both axes, and it would not have been picked from either
column alone.

## Per-class F1 — `bone`, its best epoch, video- vs person-disjoint

Both columns are notebook 03's own run summary at `val_frac=0.2`. The left column is the old
video-disjoint run (best @ep10, 33,044 val windows); the right is the person-disjoint retrain
(best @ep3, 35,698 val windows over 41 actors). **The val populations differ, so read the
ordering and the shifts, not the individual deltas.**

| class | n (person-disj.) | F1 video-disj. | F1 person-disj. | |
|---|---|---|---|---|
| `other_idle` | 14,655 | 0.509 | **0.507** | ~40% of val |
| `lying_down` | 749 | 0.408 | **0.435** | pose-distinctive |
| `sitting` | 2,952 | 0.318 | **0.341** | pose-distinctive |
| `drinking` | 1,137 | 0.195 | 0.195 | |
| `cooking_food_prep` | 419 | 0.206 | 0.187 | |
| `reading` | 1,746 | 0.151 | 0.176 | |
| `sitting_down` | 375 | 0.127 | 0.100 | transition, 1–2 windows |
| `cleaning_housework` | 3,620 | 0.193 | **0.098** | habit/context |
| `walking` | 545 | 0.192 | **0.093** | label-map suspect, see below |
| `standing_up` | 368 | 0.093 | 0.088 | transition |
| `eating` | 1,498 | 0.164 | **0.082** | habit/context |
| `using_phone` | 2,477 | 0.091 | 0.080 | needs an object |
| `watching_tv` | 774 | 0.089 | 0.074 | needs an object |
| `personal_hygiene` | 3,261 | 0.176 | **0.074** | habit/context |
| `taking_medication` | 359 | 0.043 | 0.072 | needs an object |
| `interacting_with_person` | 614 | 0.042 | 0.022 | needs a second track |
| `standing` | 123 | 0.037 | 0.005 | support-starved |
| `bending_reaching` | 26 | 0.000 | 0.000 | starved by the label map |

**The three classes that collapsed are the three most person-specific.**
`cleaning_housework` 0.193 → 0.098, `personal_hygiene` 0.176 → 0.074, `eating` 0.164 → 0.082.
These are habit-and-context activities — how *this* person wipes a counter, brushes their
teeth, holds a fork. Under the leaky split the model could recognise the individual and
recall their routine; it cannot any more. Meanwhile the two purely pose-defined classes went
*up* (`lying_down` 0.408 → 0.435, `sitting` 0.318 → 0.341), which is the signature of a model
that has stopped relying on identity and started relying on posture. That contrast is the
clearest evidence in this project that the split fix mattered, and it is invisible in the
mean.

**`walking` fell 0.192 → 0.093 on 545 windows (1.5% of val).** Walking is the most
pose-separable activity in the taxonomy and it has the support of a rare class. In a corpus of
people moving around their homes that is implausible, so the Charades keyword rules are almost
certainly routing locomotion into `other_idle`. Worth auditing `charades_map_review.tsv` before
any further training: a class the label map barely populates cannot be learned regardless of
the objective, and this is the one case where a data fix likely beats every decoding lever
combined.

**The object-dependent floor is unchanged and is the real ceiling.** `interacting_with_person`
0.022, `taking_medication` 0.072, `watching_tv` 0.074, `using_phone` 0.080. A hand holding a
pill bottle and a hand holding a phone are the same pose; watching TV and reading are the same
seated posture. `second_person_present()` now supplies the one signal of these four that needs
no detector (class 18 recall 0.060 → 0.148); the other three need RT-DETR wired to
`fuse_objects()`. `rtdetr-l.onnx` is staged and verified by preflight; nothing implements the
`ObjectDetector` protocol yet.

**Two transition classes are structurally hard.** `sitting_down` 0.100 and `standing_up` 0.088
last one or two windows at a 1 s stride, so a 2 s window straddles the transition and its
neighbours. Viterbi smoothing works against them by design — and the segment-level table above
now quantifies that trade instead of assuming it.

## Calibration

- fitted temperature **T = 0.63**
- mean confidence 0.381 vs accuracy 0.375 — nearly calibrated, slightly over-confident
- set `ActivityConfig(temperature=0.63)` in deployment

Fitted after ensembling, not per stream: Viterbi smoothing and abstention both consume
posteriors, so those posteriors have to mean something first. T < 1 sharpens, which is
consistent with logit averaging having already flattened the distribution — and T fell from
0.76 to 0.63 under the person-disjoint split, i.e. the honest ensemble is *more*
under-confident, which is what you would expect once it can no longer recognise the person.

## P2 — leave-one-dataset-out (fall sources), both models scored

Train-side generalisation is fixed (the ensemble saw Charades ADL plus the other fall
sources), so this measures transfer of the fall signal to an unseen recording setup.

**Two columns, because the single-column version told the opposite story for months.** P2 used
to score the ADL ensemble's fall-class posterior while `runs/fall/best.pt` was resolved and
never loaded — so every "the fall head does not transfer" sentence in earlier versions of this
file was about the wrong model.

| held-out | n | fall% | ADL-ensemble AUROC | **fall-head AUROC** |
|---|---|---|---|---|
| `caucafall` | 1,159 | 28.8% | 0.603 | **0.970** |
| `gmdcsa` | 914 | 24.5% | 0.471 | **0.928** |
| `le2i` | 752 | 34.7% | 0.571 | **0.989** |
| `urfd` | 239 | 16.3% | 0.560 | **0.991** |

**The fall head transfers. The ADL ensemble does not.** 0.928–0.991 across four unseen
recording setups, against a validation AUROC of 1.000 — so the earlier reading, that a perfect
in-domain ranking beside 0.47–0.60 held-out meant the negatives were separable by nuisance
cues, was wrong about which model produced the 0.47–0.60. It was the 20-class ensemble's
posterior, which is at or below chance on `gmdcsa` (0.471).

That contrast is the justification for the two-model design rather than a single 20-class head
with `falling` as class 7: a fall is a transient event at a 0.49% base rate, and a dedicated
binary head detects it where a multi-class posterior does not. The dual column is what made
the difference visible; neither number alone identifies which model it belongs to.

**0.822 AUPRC and the 0.951-sensitivity-at-0.99-FA/h operating point are therefore portable
claims**, not in-domain ceilings. Le2i's fold remains thinner than its window count suggests
(74 of 190 clips; 3 of 6 scenes), so read that row with the others. P3 staged→wild on
OmniFall's OOPS split would still add an unstaged distribution and remains pending.

## Ablation — logit vs probability combination

| combination | top-1 | mean-class | macro-F1 | F1>=50 |
|---|---|---|---|---|
| `logit-average` | **0.375** | 0.156 | 0.128 | 0.136 |
| `prob-average` | 0.358 | **0.169** | **0.131** | **0.139** |

The two rules disagree in opposite directions, which is the useful finding: logit averaging
(product of experts) wins top-1 by 1.7 points; probability averaging (mixture) wins mean-class
by 1.3, macro-F1 by 0.3 and F1>=50 by 0.3. The headline uses logits because that is what the
GCN literature reports, and this row is what justifies the choice rather than asserting it —
but on every class-balanced criterion the mixture is better, and **neither beats `bone` alone
(0.187 mean-class, 0.146 macro-F1)**. The consistent story across both splits is that
four-stream averaging is the wrong default for this label distribution.

## Fall head — strong in domain AND across recording setups

| metric | value |
|---|---|
| AUPRC (validation) | **0.822** (best @ epoch 56) |
| AUROC (validation) | 1.000 |
| sensitivity @ 0.99 false alarms/hour | **0.951** |
| threshold | 0.7826 |
| negative duration | 23.2 h |
| positive rate | 0.49% |
| **AUROC, held-out corpora (P2)** | **0.928 – 0.991** |

AUPRC, not AUROC: at a 0.49% positive rate AUROC flatters. The operating point is fitted at a
false-alarms-per-hour budget, which requires enough negative *hours* to mean anything — 23.2 h
is a decision, whereas the earlier 141-negative pool (0.08 h) was an artefact. This result is
directly attributable to feeding ADL windows in as negatives.

**Held-out AUROC 0.928–0.991 on four corpora the head never saw.** An earlier version of this
file said the opposite, on the strength of a P2 table that was scoring the ADL ensemble's
fall-class posterior while `runs/fall/best.pt` sat unloaded. With both models in the table the
0.471–0.603 range is identified as the ensemble's and the fall head's own transfer is strong.

The deployment-facing sentence, therefore: *the fall head reaches 0.951 sensitivity at ~1 false
alarm per hour, and holds 0.93–0.99 AUROC on four unseen recording setups.* P3 staged→wild on
OmniFall's OOPS split would add a genuinely unstaged distribution — staged falls in four labs
are still staged falls — and remains the protocol that would test the remaining gap.

## Agent 4 — hallucination rate (Qwen2.5-7B-Instruct)

116 analysed days (58 alerting) over 6 simulator personas. Stub anchors
([hallucination.md](hallucination.md)) are what make these interpretable: a faithful
generator scores 0.0% on the identical pipeline, and corrupted generators track their
injection rate at 24.7 / 52.0 / 100.0%. **This table has now reproduced bit-identically across
four consecutive GPU sessions** — the same 518/478 and 514/472 counts, the same C1–C4
breakdown — which is what confirms the greedy-decoding fix and makes p = 0.79 a stable
conclusion rather than one run's draw.

| condition | emitted | scorable | repaired | faithful | hallucination rate | 95% CI | s/report |
|---|---|---|---|---|---|---|---|
| grammar-constrained | 575 | 575 | 0 | 559 | **2.8%** | 1.7–4.5% | 50.3 |
| unconstrained | 545 | 545 | 0 | 525 | **3.7%** | 2.4–5.6% | 7.9 |
| unconstrained + format repair | 545 | 545 | 0 | 525 | 3.7% | 2.4–5.6% | cached |
| **pooled** | **1,120** | 1,120 | 0 | 1,084 | **3.2%** | 2.3–4.4% | — |

This is the first run with the **filled-template prompt** and the **C5** check. Both arms send
byte-identical templated text under identical greedy decoding, so the only difference between
them is still the grammar. Earlier tables (39.9%, 8.2%, 10.3%, 7.7%/8.2%) came from arms that
differed in other ways or lacked C5 and should not be quoted.

**Grammar-constrained decoding still does not measurably improve faithfulness.** 2.8% vs 3.7%,
difference −0.9 points, z = −0.84, **p = 0.40**. That verdict has now survived the prompt
change that moved the absolute rate by more than a factor of two, which is the strongest form
of the result: the grammar's value is a *worst-case parseability guarantee*, not a faithfulness
gain, and it costs **6.4×** the decoding time (50.3 vs 7.9 s/report).

### Where the failures are — and the residual has moved

| condition | bad ref (C1) | value (C2) | pct (C3) | direction (C4) | **prose (C5)** |
|---|---|---|---|---|---|
| grammar-constrained | 1 | **0** | 0 | 6 | **9** |
| unconstrained | 0 | **0** | 0 | 7 | **13** |

The counts now sum exactly to the unfaithful totals (1+6+9 = 16 of 575; 7+13 = 20 of 545), so
no claim fails two checks and every failure has one identified cause.

- **C2 went from 39/40 and 42/42 to ZERO.** The prompt now carries a `FILLED TEMPLATE` block
  and instructs the model to copy `value` digit-for-digit into the JSON field. Field-level
  numeric misquoting — which was essentially the entire residual across four previous runs —
  stopped completely. This is the largest single measured improvement in the Agent 4 pipeline.
- **C5 is now the largest failure category**: 9 of 16 and 13 of 20. It asks whether the claim's
  *prose* echoes the number in `claimed_value`. So the model copies the figure into the field
  correctly and then paraphrases it in the sentence — "walking fell noticeably" where the field
  says 820. C2 is blind to that by construction, because C2 compares the field against the
  evidence row and both are right. The sentence is what a caregiver reads.
- **C1 = 1, once, in the constrained arm.** The first fabricated citation in five runs, against
  0 in every previous arm. At this rate it is not estimable; it is recorded, not interpreted.
- **C4 is stable and rare**: 6 and 7, against 3/3/3/3/2/0 in previous runs. Inverted narration
  remains the clinically dangerous mode and remains uncommon.

**Without C5 the headline would have been 1.2% and 1.3%** — and it would have been wrong, in
the specific way this project keeps finding: a metric that improves because a check was
missing. Adding a check that finds 9 and 13 real defects while the total still falls by more
than half is the honest version of the improvement.

### Run history, and what each run actually measured

| | run 1 | run 2 | run 3 | run 4 |
|---|---|---|---|---|
| constrained | 39.9% ⚠ | 8.2% ⚠ | 10.3% ⚠ | **7.7%** |
| free | `nan` ⚠ | 6.5% ⚠ | 6.5% ⚠ | **8.2%** |
| constrained parse failures | 9 / 116 | 0 | 0 | 0 |
| arms decoding-matched | no | no | no | **yes** |

Four defects were found, none of them in the model, and all four were visible on CPU by
reading the two code paths side by side rather than reading Kaggle logs:

1. **The prompt never gave the LLM its output envelope** (run 1). It named `evidence_ref` and
   `claimed_value` in prose but not `claim_id`, `text` or the `claims` array. The free model
   had to guess the schema and lost 245 of 245 claims to Pydantic; the constrained model was
   forced into the right shape by the grammar but still mis-assigned which payload field
   belonged in which claim field. This alone accounted for >30 points of apparent
   "hallucination".
2. **The grammar allowed 12 claims while the reporter kept 6** (run 1). Budget spent on
   claims that were then discarded, which is how 9 reports hit `max_new_tokens`.
3. **The constrained arm was sampling** (found at run 3). It passed only `max_new_tokens` to
   `outlines` and inherited Qwen's `do_sample=True, temperature=0.7`; the free arm passed
   `do_sample=False`. The gap between the arms measured temperature. It presented as
   irreproducibility — the free arm was bit-identical across runs 2 and 3 while the
   constrained arm moved 498/457 → 504/452 and the verdict flipped from p=0.29 to p=0.03.
4. **The constrained arm skipped the chat template** (found at run 3). `outlines` does not
   template a bare string, so one arm received a Qwen chat turn and the other a naked
   instruction block. Under a grammar a model cannot express confusion by emitting malformed
   JSON — it emits valid JSON with worse content, landing as C2.

Test R14 now audits every axis on which the arms can differ — templated text, `do_sample`,
token budget, claim budget, prompt/schema field coverage, verifier config, guide caching —
in under a second on CPU. Written after four runs; it would have prevented three of them.

The free arm also moved between runs 3 and 4 (6.5% → 8.2%), because `force_greedy` cleared
Qwen's `repetition_penalty=1.05` from the free path too. That change is not significant
(z = −1.04, p = 0.30) and the composition shifted rather than the total: C2 30 → 42 while
C3 and C4 went to zero.

### One hypothesis that was wrong

The constrained arm's 49 s/report was attributed to `outlines.types.json_schema(schema)`
being reconstructed inside every call, defeating any guide caching. It is now compiled once
per schema — and the time went from 49.0 to **50.3 s/report**, i.e. no improvement.
Per-claim cost is identical (claims/report rose 4.34 → 4.47 over the same interval). The
7.6× gap is inherent to per-token mask computation, not to guide compilation. The cache is
harmless and stays, but it bought nothing.

**What this means for the contribution.** Roughly **1 claim in 31 that a 7B model makes about
a resident's day is wrong against the data it was given** (3.2% pooled, 95% CI 2.3–4.4%,
1,120 claims). Fabricated citations are almost designed out (1 in five runs). Field-level
numeric misquoting is designed out entirely by giving the model a filled template to copy from
— it went from ~40 per run to zero. The residual is dominated by **prose paraphrase**: the
model records the right figure and then describes it loosely, which only a check on the
sentence rather than the field can see. A deterministic arithmetic layer catches all of it
before a caregiver reads a word. None of that depends on constrained-versus-free, which is the
axis that took three runs to measure properly and turned out not to matter — and which now
holds across a factor-of-two change in the absolute rate.

### Hosted arm — the deployed configuration (2026-09-13)

Serving moved off Qwen to hosted models in September (a 7B model was 286 s of a ~340 s request
and cannot run on the P100 at all), so the deployed configuration now has its own measurement:
same 116 simulated days, same 6 personas, same verifier, same greedy decoding — the model is
the only thing that moved. Source: [hallucination_openrouter.md](hallucination_openrouter.md),
per-claim dump `hallucination_openrouter_claims.jsonl` (2,080 records).

| condition | emitted | scorable | repaired | faithful | hallucination rate | 95% CI | s/report |
|---|---|---|---|---|---|---|---|
| grammar-constrained | 692 | 692 | 0 | 692 | **0.0%** | 0.0–0.6% | 35.0 |
| unconstrained | 694 | 694 | 0 | 694 | **0.0%** | 0.0–0.6% | 24.6 |
| unconstrained + format repair | 694 | 694 | 0 | 694 | 0.0% | 0.0–0.6% | cached |

**0 of 1,386 claims failed any of C1–C5.** Zero parse failures, zero schema rejections,
nothing in need of repair, and constrained-vs-free indistinguishable at p = 1.00 — the same
verdict as the Qwen table, now at the floor. Every report was served by
`nvidia/nemotron-3-super-120b-a12b` (GLM's free provider was saturated throughout; the chain
routed around it, which is the behaviour it exists for).

Three things belong beside this number rather than after it:

- **It is not a model-vs-model comparison with the Qwen table.** The prompt carries two more
  reportable features (cooking, drinking) than the Qwen runs had, and a 120B MoE reasoner is a
  different class of model from a 7B dense one. The safe claim is about the configuration: the
  deployed system, measured on the same day distribution, produced no unfaithful claim in
  1,386 attempts, upper bound 0.6%.
- **This benchmark cannot exhibit the failure mode deployment actually found.** The simulated
  days are full-length and reliable; the partial-window failures (below) never arise here,
  because the benchmark never asks the model to summarise ten minutes of a 24-hour baseline.
  A 0.0% faithfulness rate and a live warrantless decline report were both true of the same
  system in the same week.
- **It is a dated measurement, not a reproducible one.** Free providers rotate without notice
  — two left the tier during the week this was written — so the dump, not the endpoint, is
  the record. The Qwen table above remains the pinned anchor, reproducible from the
  checkpoint; both are kept deliberately.

The grammar's cost on this arm is ~10 s/report (35.0 vs 24.6), far below Qwen's 6.4×, because
enforcement happens server-side in `response_format` rather than in a per-token local mask.

### Faithful is not warranted — the failure the verifier cannot see

C1–C5 verify a claim against the evidence, and nothing verifies that the *evidence supports
the comparison*. Three measured instances of that gap, all on deployed clips, all closed:

| what the system said | why it passed every check | fix |
|---|---|---|
| "no walking today, −100%" from a 9m56s clip whose walking *rate* was normal (62.7 s/h vs the baseline's ~67) | the arithmetic was internally consistent; a ten-minute total against a 22.9 h median is a 138× unit error, not a lie | comparison numbers (pct, z, baseline, direction) are **withheld** on unreliable windows, not flagged |
| every claim stamped `· decrease`, including `drinking_events 4.00 · decrease` | `direction` is `sign(observed − baseline)`, the same invalid comparison wearing a word | `direction` withheld with them |
| "the resident showed **limited**… activity" over 173.65 s of housework in 596 s | the summary is prose; C1–C5 check claims | prompt rule 8a bans comparative words on partial windows; the card states plainly that the checks cover claims, not prose |

The first two are structural — a number that cannot support a claim is now never in front of
the model. The third is instruction-following, verified once live and not guaranteed. A sixth
check on prose would close it structurally; that is a design decision about the verifier's
contract, recorded here rather than made silently.

## Toyota Smarthome

### Ingestion (preflight, 2026-08-23)

`06_toyota_preflight_online.ipynb` over the nine private mounts.

| check | result |
|---|---|
| untrimmed videos / annotation CSVs | 536 / 536 |
| trimmed clips | 16,115 |
| annotation rows | 41,343 across 53 distinct class names |
| merges required by the README | `Make_tea.Insert_tea_bag` (77 rows), `Make_coffee.Get_water` (39) |
| class-name resolution | **51/51, none unmapped** |
| CS protocol | 351 train / 185 test videos, partition verified from the ids |
| annotated span | 10,705,053 frames = 118.9 h at 25 Hz |
| untrimmed container | 25 fps, 640×480, frame count matches `duration` exactly (delta +0) |
| trimmed container | **20 fps**, not the 30 the RGB baseline's README documents |

**Two annotation sources, and they are 99.889% the same.** `Annotation_v1.0.tar.gz` ships one
CSV per video; every published baseline trains on the aggregated `smarthome_CS_51.json`, which
is a derived mirror. 454 of 536 videos are span-identical. The other 82 are mostly the JSON
merging adjacent same-class intervals — a pure split scores 1.00000 at frame level — but 59
videos have genuinely moved boundaries:

| | value |
|---|---|
| frame-level agreement, mean | **0.99889** |
| frame-level agreement, worst video | 0.97828 |
| videos below 0.999 | 59 of 536 |

**Decision: train on the JSON, and quote 0.99889 beside the numbers.** The JSON is what PDAN's
32.7% f-mAP and every other published TSU result is computed over, and comparability is worth
more than 1 frame in 900 when the task's state of the art is 33% f-mAP. The CSVs remain the
authoritative source and the disagreement is recorded rather than resolved.

**Class distribution under the 22-class taxonomy** (frames, share of annotated):
`reading` 40.71% · `using_device` 11.87% · `watching_tv` 9.55% · `cooking_food_prep` 8.80% ·
`cleaning_housework` 6.23% · `object_interaction` 5.55% · `walking` 5.35% · `eating` 4.24% ·
`using_phone` 2.91% · `drinking` 2.60% · `sitting_down` 0.72% · `standing_up` 0.67% ·
**`taking_medication` 0.56%** · `lying_down` 0.16% · `personal_hygiene` 0.09%.

`other_idle` receives **0 frames**, which is the intended answer and the opposite of the
Charades map, where the keyword fallback was the largest single class. `taking_medication` gets
60,229 frames of real supervision against the 359 windows Charades could reach by keyword.

**A third of the corpus is unannotated gap** (mean 0.328, median 0.230). For the 51-class
benchmark head those gaps are true negatives, which is what the official loader assumes. For
the unified 22-class head they must be *masked*, because a gap most likely contains the sitting
and standing that Toyota never labels — 7 of the 22 coarse classes get no positive frames from
this corpus at all.

### Activity recognition — trained, cross-subject (2026-08-26)

Source: `behaviorsense-tsm-eval/metrics.json` — the same weights as `behaviorsense-tsm-runs`
(epoch 48 of 60), and the record that supersedes the training run's own metrics because it
carries the coarse block. Nothing on this page is scrollback.

Protocol: **CS** — 11 train subjects, 7 test subjects (2, 10, 11, 14, 16, 18, 20) never seen in
training; 10,157 train / 5,203 test clips. `bone` stream, 15 joints (13 LCR-Net + derived neck
and mid-hip), 64-frame clips, effective-number class balance (β=0.9999).

| head | top-1 | mean-class | classes scored |
|---|---|---|---|
| fine (Toyota's vocabulary, 31 classes with CS-test support) | 0.678 | 0.594 | 31 |
| **coarse, argmax-collapse — the product head** | **0.769** | **0.736** | 13 of 22 |
| coarse, logsumexp-collapse | 0.523 | 0.591 | 13 of 22 |

**Argmax-collapse beats logsumexp by +14.5 mean-class points, and the reason is family size.**
11 of the 31 fine classes collapse into `cooking_food_prep` alone, and
`logsumexp(k) ≈ max + log k`, so summing fine-class probabilities inflates large families and
hands them the coarse decision — under logsumexp `cooking_food_prep` reaches 0.934 per-class
while `walking` collapses to 0.151. Taking the argmax over the fine head first carries no such
bias. Argmax-collapse is the rule in evaluation and in serving.

Not comparable to the published RGB baselines without saying so: PDAN's 32.7% is frame-level
f-mAP over 51 classes on trimmed RGB; the numbers above are clip-level accuracy of a skeleton
model. Both are honest, neither validates the other.

Per-class (coarse, argmax) — strongest and weakest over the 13 scoreable:

| | class | acc | test clips |
|---|---|---|---|
| strongest | `lying_down` | 0.954 | 65 |
| | `sitting_down` | 0.920 | 437 |
| | `standing_up` | 0.902 | 316 |
| | `cleaning_housework` | 0.836 | 219 |
| | `walking` | 0.835 | 1,333 |
| | `cooking_food_prep` | 0.820 | 529 |
| weakest | `taking_medication` | 0.434 | 136 |
| | `reading` | 0.484 | 333 |
| | `watching_tv` | 0.576 | 229 |
| | `using_device` | 0.627 | 193 |

`drinking` sits at **0.760 over 964 test clips** — the arm that later detected four drinking
events in an unscripted 10-minute deployment clip. `taking_medication` is the weakest
supervised class despite real supervision (60,229 frames against the 359 keyword windows
Charades could reach) — 0.56% of frames does not buy a 15-joint signature for pill-taking.

Two caveats that belong beside the numbers, not after them:

- **13 of the 22 product classes are scoreable under CS.** The other 9 receive no positive
  frames in the CS test split — the sustained `sitting`/`standing` states, `bending_reaching`,
  `interacting_with_person`, `other_idle`, the fall classes, and two more with train-only
  frames. Quoting 0.736 as "22-class accuracy" would be wrong; it is 13-of-22.
- **10 fine classes have < 30 test clips** (from `Makecoffee.Pourwater` at 24 down to
  `Drink.Fromglass` at 6); their per-class numbers are one to three clips wide and are
  reported, not averaged on.

**The served arm is a different input source, and the gap is measurable.** The table above
trains and evaluates on INRIA's own 3D poses — near-ground-truth input. Deployment runs RTMO
on video, so the served checkpoint (`adl_toyota_bone`, 22 classes, 20 Hz) was trained and
evaluated on RTMO-extracted poses instead: **mean-class 0.600 / macro-F1 0.585 over the 15
classes present** (session-log provenance — **checked 2026-09-13**: no `val_logits.npz` for
`adl_toyota_bone` exists in any attached mount, so these figures stand on the training
session's log, quoted exactly as printed; re-deriving them needs one ~40 s eval pass over the
Toyota RTMO shards, not a retrain). The 0.736 → 0.600 gap between the arms is the measurable
cost of live pose extraction, and it is the number a deployment claim should carry.

Since 2026-09, `ADL_PREFER = "toyota"` serves this corpus in preference to the Charades
checkpoints the P1 tables above describe.

## Deployment observations — one unscripted clip, four defects the benchmarks missed

A single 9m56s kitchen upload (one older adult, cooking, cleaning, drinking coffee, watching
television; 640×480, 11,936 frames at 20 Hz) was run through the full served system repeatedly
during development. It found four defects that every benchmark in this file had structurally
cannot exhibit, each fixed with a regression test, and it is the honest argument for running
unscripted footage at all.

| what the clip showed | why no benchmark could | fix |
|---|---|---|
| a clinically alarming decline report from ten minutes of normal activity (walking read −99.4%, five features −100%) | simulator days are full-length, so the 138× window-mismatch between a 0.166 h clip and 22.9 h baselines never arises | comparison numbers withheld on unreliable windows (R16) |
| every claim stamped `· decrease`, including `drinking_events 4.00` | direction is `sign(observed − baseline)`; benchmarks only produce meaningful directions | direction withheld with the other comparison fields |
| "the resident showed **limited**… activity" over 29% of the observed time | the summary is prose; C1–C5 check claims, and benchmark summaries are about full days | prompt rule 8a; the card states the checks' actual scope |
| a **faithful** claim struck as unfaithful (`claimed 0.07, actual 0.074`) | every fixture feature is large-magnitude, where 2% tolerance dwarfs display rounding; the boundary exists only below 0.25 | C2/C5 floored at the display half-step; every published value must verify at face value (R18) |

Final state on this clip, after the fixes: **5 claims, 0 withheld, 0.0% unfaithful**, summary in
pure observation form ("During the 596.8-second observation window…"), no comparative language,
no escalation. The third row is instruction-following rather than structural and is reported as
such above.

The same clip measured the tracker bridge: detection dropout (34% of frames) previously
fragmented one person into **39 track ids**; with confirmed-track bridging (15 s at 20 Hz) the
same clip produces **11 tracks, 9 reunited by the span merge**, and person-records *rose*
7,834 → 7,893 because detections that previously attached to never-confirmed tentative tracks —
invisible to Agent 2 — now attach to the bridged identity. Bridging preserves identity across a
gap; it does not create observations inside one, so the missed-frame share is unchanged and
remains the top accuracy lever.
