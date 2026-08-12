# ADL shard extraction — measured result (2026-08-07)

Charades → RTMO pose → unified-taxonomy skeleton windows. One offline Kaggle session,
RTX PRO 6000 Blackwell, `CUDAExecutionProvider` confirmed in use.

| | |
|---|---|
| videos | **7,985 / 7,985** (whole `Charades_v1_train.csv`) |
| windows | **165,109** (2 s, stride 1 s, 15 fps, fp16) |
| shards | 6 × `.npz`, 221 MB total |
| wall clock | **6.26 h** (22,537 s) of a 12 h session |
| throughput | 1,339 videos/h, stable ±1% across the run |

The 9,848 figure often quoted for Charades is train + test; only the 7,985 train videos
carry action annotations, so that is what extraction consumes.

## Class distribution

| id | class | windows | share |
|---:|---|---:|---:|
| 19 | other_idle | 65,088 | 39.42% |
| 16 | cleaning_housework | 16,232 | 9.83% |
| 17 | personal_hygiene | 15,837 | 9.59% |
| 2 | sitting | 14,766 | 8.94% |
| 15 | using_phone | 11,405 | 6.91% |
| 14 | reading | 8,897 | 5.39% |
| 9 | eating | 6,294 | 3.81% |
| 10 | drinking | 5,457 | 3.31% |
| 3 | lying_down | 3,909 | 2.37% |
| 13 | watching_tv | 3,713 | 2.25% |
| 11 | cooking_food_prep | 2,823 | 1.71% |
| 18 | interacting_with_person | 2,705 | 1.64% |
| 0 | walking | 2,524 | 1.53% |
| 12 | taking_medication | 1,838 | 1.11% |
| 4 | standing_up | 1,768 | 1.07% |
| 5 | sitting_down | 1,442 | 0.87% |
| 1 | standing | **342** | 0.21% |
| 6 | bending_reaching | **69** | 0.04% |
| 7 | falling | **0** | — |
| 8 | fallen_on_ground | **0** | — |

## Three things to state rather than hide

**Classes 7/8 are empty by design.** Charades contains no falls. They come from the
OmniFall sources via notebook 02 and are trained by `train_fall.py` as a separate binary
head — the ADL head never needs them. `class_balanced_sampler` gives absent classes zero
weight and `evaluate()` masks them out of macro-F1, so nothing degrades.

**Classes 1 and 6 are starved as a consequence of a deliberate mapping rule.** Charades
has no verb for a bare posture — reaching and placing appear only as hand-object phrases
(*"Putting a box somewhere"*, *"Taking a bag from somewhere"*). `build_charades_map.py`
routes that whole family to `other_idle` on purpose, and says why at
[scripts/build_charades_map.py:14](../scripts/build_charades_map.py#L14): *"They carry no
signal any behavioural feature consumes, and the taxonomy's reject class exists precisely
so ambiguous windows have somewhere honest to go."* The keyword rules for these classes do
exist (`\bstanding\b(?! up)` at line 48, `bending_reaching` at line 52) — they simply
match few Charades names. This is most of why `other_idle` reaches 39.4%.

That rationale is verified against the code rather than assumed: no entry in
`TRACKED_FEATURES` reads class 1 or 6 (checked — the 18 features cover walking, sitting,
lying, sit-to-stand, meals, drinking, medication, TV, social and housework), so the
behaviour layer is genuinely unaffected. Re-mapping the family would need a re-review of
the 157-class map plus a 6.3 h re-extraction for no downstream gain. The cost is confined
to one metric, and it is now reported rather than absorbed:

- `macro_f1` — all present classes, the honest headline
- `macro_f1_supported` — classes with ≥ 50 val windows (`MIN_SUPPORT`)
- `per_class_f1` / `per_class_support` — the full vector, so any gap is attributable

Simulated against this exact distribution, the gap is **+0.043 macro-F1** from one class
at 13 val windows. `standing` (68 val windows) clears the floor and stays in both averages
with a near-zero F1, so the threshold does not flatter the result. Both numbers go in the
write-up.

**`other_idle` at 39.4% is a reject class doing real work.** `map_class` returns
`(FALLBACK, "fallback (REVIEW)")` for unmatched names and prints every such class for eye
review, while camera-performance classes return `None` and are **dropped** — so "reject"
and "not an activity" stay distinct rather than being pooled. Two classes were dropped in
this run, as expected.

The consequence is that a 39% plurality makes top-1 accuracy easy to inflate by predicting
`other_idle`. That is why `best.pt` is selected on **mean-class accuracy**, not top-1, and
why the P1 table reports top-1, mean-class and both macro-F1 columns side by side.
