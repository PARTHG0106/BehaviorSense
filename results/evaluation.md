# Evaluation — P1 / calibration / P2 / ablation / Agent 4

Two notebook-04 sessions on the RTX PRO 6000 Blackwell (sm_120): 9,515 s and 7,569 s. The
P1 / calibration / P2 / ablation numbers are byte-identical across both runs, as they should
be — nothing about the activity model changed between them. The Agent 4 table changed a
great deal, and most of that change was a defect in our own prompt rather than in the model.

Checkpoints: `adl_{joint,bone,joint_motion,bone_motion}/best.pt`, EMA weights, 30-epoch
budget with `--patience 8`. Shards: 165,109 ADL windows (7,985 Charades videos) +
3,064 fall windows (4 corpora).

## P1 — subject-disjoint validation

24,908 val windows, 1,166 subjects, split seed 0 (identical to training, so these windows
influenced training only through early stopping).

| model | top-1 | mean-class | macro-F1 | F1>=50 |
|---|---|---|---|---|
| `joint` | 0.337 | 0.180 | 0.169 | 0.179 |
| `bone` | 0.331 | **0.189** | 0.168 | 0.177 |
| `joint_motion` | 0.331 | 0.127 | 0.105 | 0.112 |
| `bone_motion` | 0.325 | 0.123 | 0.117 | 0.123 |
| **ENSEMBLE** | **0.375** | 0.151 | 0.150 | 0.158 |

**The ensemble improves top-1 by 3.8 points and loses 3.8 points of mean-class accuracy
versus `bone` alone (0.151 vs 0.189).** This contradicts the usual expectation and is
reported as measured. Logit averaging is a product of experts: a stream that assigns a
class near-zero mass can veto it, and on a long-tailed label distribution that trades tail
recall for head accuracy. `bone` is therefore the better deployment checkpoint on the
mean-class criterion, and the ensemble the better one on top-1 — the choice depends on
which Agent 3 features matter, not on which number is larger.

macro-F1 averages the 18 present classes; the last column averages the 17 with >= 50 val
windows. One class is starved by the Charades label map: `bending_reaching` (support 11).
Charades has no verb for it — "Putting a box somewhere" and similar route to `other_idle` —
so this is a label-map limitation to state in the write-up, not a model result.

The four weakest classes (`taking_medication`, `interacting_with_person`, `standing`,
`watching_tv`) share a property: none has skeleton-visible evidence. A person holding a
pill bottle and a person holding a phone have the same pose. That is the argument for the
RT-DETR object-context branch, and it is a diagnosis rather than an excuse — the per-class
spread is broad and non-zero (`other_idle` 0.509 → `bending_reaching` 0.000), which is the
signature of a hard task, not of a collapsed model.

## Calibration

- fitted temperature **T = 0.77**
- mean confidence 0.391 vs accuracy 0.375 — nearly calibrated, slightly over-confident
- set `ActivityConfig(temperature=0.77)` in deployment

Fitted after ensembling, not per stream: Viterbi smoothing and abstention both consume
posteriors, so those posteriors have to mean something first. T < 1 sharpens, which is
consistent with logit averaging having already flattened the distribution.

## P2 — leave-one-dataset-out (fall sources)

Train-side generalisation is fixed (the ensemble saw Charades ADL plus the other fall
sources), so this measures transfer of the fall signal to an unseen recording setup.

| held-out | n | fall% | AUROC |
|---|---|---|---|
| `caucafall` | 1,159 | 28.8% | 0.734 |
| `gmdcsa` | 914 | 24.5% | 0.545 |
| `le2i` | 752 | 34.7% | 0.586 |
| `urfd` | 239 | 16.3% | 0.722 |

Spread 0.545–0.734 across four corpora is the honest headline: cross-setup transfer is
weak and corpus-dependent. `gmdcsa` and `le2i` are near chance. Le2i also contributes only
74 of its 190 clips (3 of 6 scenes; 116 skipped for missing annotation files), so its fold
is thinner than the window count suggests.

## Ablation — logit vs probability combination

| combination | top-1 | mean-class | macro-F1 | F1>=50 |
|---|---|---|---|---|
| `logit-average` | **0.375** | 0.151 | 0.150 | 0.158 |
| `prob-average` | 0.358 | **0.167** | 0.156 | 0.165 |

The two combination rules disagree in opposite directions, which is the useful finding:
logit averaging (product of experts) wins top-1 by 1.7 points, probability averaging
(mixture) wins mean-class by 1.6 points and macro-F1 by 0.6. The headline uses logits
because that is what the GCN literature reports, and this row is what justifies the choice
rather than asserting it — but on a mean-class criterion the mixture is better, and neither
beats `bone` alone.

## Fall head

| metric | value |
|---|---|
| AUPRC | **0.822** (best @ epoch 56) |
| sensitivity @ 0.99 false alarms/hour | **0.951** |
| threshold | 0.7826 |
| negative duration | 23.2 h |
| positive rate | 0.49% |

AUPRC, not AUROC: at a 0.49% positive rate AUROC flatters. ~168× better than chance. The
operating point is fitted at a false-alarms-per-hour budget, which requires enough negative
*hours* to mean anything — 23.2 h is a decision, whereas the earlier 141-negative pool
(0.08 h) was an artefact. This result is directly attributable to feeding ADL windows in as
negatives.

## Agent 4 — hallucination rate (Qwen2.5-7B-Instruct)

116 analysed days (58 alerting) over 6 simulator personas, 7,569 s wall. Stub anchors
([hallucination.md](hallucination.md)) are what make these interpretable: a faithful
generator scores 0.0% on the identical pipeline, and corrupted generators track their
injection rate at 24.7 / 52.0 / 100.0%.

| condition | emitted | scorable | repaired | faithful | hallucination rate | 95% CI | s/report |
|---|---|---|---|---|---|---|---|
| grammar-constrained | 498 | 498 | 0 | 457 | **8.2%** | 6.1–11.0% | 49.0 |
| unconstrained | 509 | 509 | 0 | 476 | **6.5%** | 4.7–9.0% | 6.5 |
| unconstrained + format repair | 509 | 509 | 0 | 476 | 6.5% | 4.7–9.0% | cached |

**The grammar does not improve faithfulness.** Difference +1.7 points, z = 1.06, two-sided
p = 0.29; the intervals overlap across most of their range. What it does buy is a
*worst-case guarantee* — it cannot emit an invalid claim — whereas the free arm merely
*happened* to emit none. For a caregiver-facing system that guarantee has value, but it
costs 7.5× the decoding time (49.0 vs 6.5 s/report) and must not be sold as a faithfulness
result.

**The repair arm was a no-op: 0 of 509 claims needed repair.** That is the finding, not a
redundancy. It means the free arm's output was schema-clean, and therefore that the 245
schema rejections in the first run were caused by our prompt withholding the field names —
not by the model.

### The first run's numbers were mostly our bug

| | run 1 | run 2 |
|---|---|---|
| constrained hallucination rate | 39.9% | **8.2%** |
| constrained parse failures | 9 / 116 | **0** |
| free arm | 245 emitted, 0 scorable, `nan%` | **509 emitted, 509 scorable, 6.5%** |

Run 1's prompt named `evidence_ref`, `claimed_value`, `claimed_pct_change` and `direction`
in prose but never gave the JSON envelope — not `claim_id`, not `text`, not the `claims`
array. The free model had to guess our schema and lost every claim to Pydantic; the
constrained model was forced into the right shape by the grammar but still mis-assigned
which payload field belonged in which claim field. Adding an explicit field mapping to the
prompt removed 31.7 points of measured "hallucination" that was never the model's fault.
It is the same class of error as the earlier zero-baseline defect: a reporter bug wearing a
verifier failure's clothes.

Attribution is clean despite two simultaneous changes. `maxItems` 12 → 6 and
`max_new_tokens` 900 → 1600 fixed the 9 truncations, but the reporter already discarded
claims beyond 6, so neither could alter the *content* of a scored claim. Only the prompt
template can.

### Where the residual failures are

| condition | bad ref (C1) | value (C2) | pct (C3) | direction (C4) |
|---|---|---|---|---|
| grammar-constrained | 0 | 39 | 0 | 3 |
| unconstrained | 0 | 30 | 2 | 3 |

Counts are per check, so they do not sum to the unfaithful totals (41 and 33).

- **C1 = 0 in both arms.** The model never fabricates a citation, because `state_to_payload`
  hands it resolvable refs rather than asking it to synthesise `feat:<name>:<date>`. That
  design choice is doing real work.
- **C2 dominates: 39 of 41, and 30 of 33.** The residual failure is quoting a number that
  differs from the stored value by more than 2%. What the misquotes look like is the next
  question, and `hallucination_qwen_claims.jsonl` now holds every claim with its verdict so
  that analysis needs no GPU.
- **C4 = 3 in each arm.** Inverted narration — the clinically dangerous mode, a right number
  told backwards — occurs in ~2.6% of reports. Small, non-zero, and invisible to C1–C3. This
  is the case the verifier exists for.

**What this means for the contribution.** The claim is not that constrained decoding fixes
hallucination; measured, it does not. The claim is that roughly **1 in 13 claims a 7B model
makes about a resident's day is wrong against the data it was given**, that the error is
concentrated in quoted values with a small but real tail of inverted narration, and that a
deterministic arithmetic check catches all of it before a caregiver sees it. That argument
no longer depends on the constrained-versus-free axis at all, which makes it stronger.
