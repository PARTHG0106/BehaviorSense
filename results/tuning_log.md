# Agent 3 tuning log — parameters chosen by measurement

Every threshold in `BehaviourConfig` that was changed from its first-guess value is
recorded here with the measurement that forced the change. This file exists because
"we tuned the thresholds" is not a defensible claim in a dissertation; "we measured
18.8% false-alarm rate at k=0.5 and 1.9% at k=0.75, and here is the signal cost" is.

Reproduce any row with:

```bash
PYTHONPATH=src python -m behaviorsense.eval.behaviour_eval
```

> All numbers are **simulator** results (simulator v1.0.0, harness v1.0.0). They bound
> the detector's sensitivity given clean features. They are not clinical accuracy.

---

## Headline progression

The suite is 28 scenarios: 4 personas × (6 injected anomalies + 1 no-anomaly control),
150 days each. Recall is measured against the *targeted* alert kind; the false-alert
rate is measured on the controls, where any alert is by construction a false positive.

| # | change | recall | median latency | control FP / 100 days |
|---|--------|--------|----------------|----------------------|
| 0 | first honest run | 96% | 4 d | **67.66** |
| 1 | + suppress absence rules on unobserved days | 96% | 4 d | 33.57 |
| 2 | + weekday/weekend stratified baseline | 96% | 4 d | 33.57 |
| 3 | + R10 requires 2 consecutive days | 100% | 4 d | 5.59 |
| 4 | + CUSUM on social interaction | 100% | 4 d | 5.59 |
| 5 | + CUSUM warm-up gate, decay, k=0.75 | **100%** | **4 d** | **5.07** |

Net: **13.4× fewer false alerts with recall up from 96% to 100%.** Each step is a
distinct defect, described below.

---

## 1. Absence rules on unobserved days

**Symptom.** Control residents with no injected anomaly were generating alerts at
14.0 per day on camera-outage days versus 0.33 on observed days. Fifteen outage days
produced **52% of all control alerts**.

**Cause.** `DailyFeatures.is_reliable()` correctly excluded degraded days from the
*baseline*, but those days were still *analysed*. A day with 2.5 observed hours reports
zero meals, zero medication events and zero visitors — so R4 (meal_skipped), R6
(medication_missed), R8 (social_isolation), R7 (mobility_decline) and R9
(sleep_disruption) all fired at once, telling the caregiver the resident had skipped
meals and missed medication when the truth was that the camera was down.

**Fix.** `suppress_absence_rules_on_unreliable_days`. Rules that trigger on a LOW or
ZERO count are unsound under partial observation and are suppressed. Rules that trigger
on a POSITIVE observation (R1 fall detected, R2 no recovery, R3 prolonged inactivity)
stay active — a fall seen during a 2-hour window is still a real fall.

**Principle.** *"I did not see it" is not "it did not happen."* This distinction is
invisible to any evaluation that only tests on clean data, which is why the simulator
injects outage days at all.

## 2. Weekday/weekend stratified baseline

**Symptom.** On no-anomaly residents, `routine_deviation` fired at 0.49 per weekend day
versus 0.22 per weekday — 2.2×.

**Cause.** Weekend behaviour genuinely differs (visitors, outings, later meals). A
pooled 14-day baseline mixes both populations, so its MAD is inflated by the weekday /
weekend gap and a perfectly normal Saturday scores as a deviation. This is a *periodic
confound*, not noise: it does not average out with more data.

**Fix.** `weekday_weekend_split`. Weekends are compared against weekend history,
weekdays against weekday history. The lookback widens per stratum (7/2 for weekends,
7/5 for weekdays) so each stratum stays populated, with a documented fallback to the
pooled window when a stratum is short — early in deployment a slightly confounded
baseline still catches gross deviation, whereas no baseline catches nothing.

**Cost.** Roughly 2× the calendar time to fill a stratum. Accepted: the alternative is
a detector that cries wolf every weekend, which is how monitoring systems get switched
off.

## 3. R10 multiple comparisons

**Symptom.** After fixes 1–2, `routine_deviation` was **171 of 192** remaining control
alerts (29.9 per 100 resident-days).

**Cause.** R10 tests ~11 features per day at |z| > 3. Under a heavy-tailed (lognormal)
feature distribution the nominal 3σ rate badly understates the true family-wise rate.
The distribution of the alerts is the tell: **158 were isolated single days, only 9
persisted into a second day.**

**Fix.** `routine_deviation_sustained_days = 2`, plus a same-direction requirement. A
single atypical day is an appointment or a visitor; a *routine* deviation — which is
what the rule's own name claims — lasts longer than one day. A feature that swings +4σ
then −4σ is erratic sensing, not a shifted routine, and reporting it as one would be
wrong.

**Result.** 33.57 → 5.59 per 100 days, with no loss of recall. Guarded by `T9`/`T9b`.

## 4. Social withdrawal was undetectable by design

**Symptom.** Recall for injected `social_withdrawal` was 50%, against 100% for gradual
mobility decline.

**Cause.** Gradual withdrawal has the same "boiled frog" structure as mobility decline:
the rolling baseline follows the slow decay, so no single day is ever anomalous and
R8's daily-z test never fires. Mobility had CUSUM; social interaction did not. **The
asymmetry was in the detector, not the phenomenon.**

A second bug compounded it: R8 required `visitor_count == 0`. The clinically relevant
signal is shrinking interaction *duration* during visits that still happen, so the
conjunction excluded the exact case of interest.

**Fix.** `social_drift_cusum`, with the CUSUM path deliberately omitting the
`visitor_count == 0` conjunction. Recall 50% → 100%. Guarded by `T10`.

## 5. CUSUM charged on noise

**Symptom.** `T10` was written to prove CUSUM catches gradual withdrawal. It passed —
reporting detection at **0 days into the decline** with max daily |z| = 2.37. A
detection at zero latency proves nothing.

**Cause.** Instrumenting the warm-up showed the statistic reaching **4.21 during 20
clean baseline days**, then crossing h=5.0 on the first day of decline. The "detection"
was noise that happened to cross at a convenient moment.

Monte Carlo over 4,000 runs of pure N(0,1) noise, per feature:

| k | h | false alarm (200 d) | detect δ=0.3 | δ=0.5 | δ=0.8 |
|---|---|---|---|---|---|
| 0.50 | 5.0 | **18.8%** | 58 d / 86% | 27 d / 100% | 12 d / 100% |
| 0.75 | 5.0 | **1.9%** | 95 d / 26% | 70 d / 76% | 23 d / 100% |
| 1.00 | 5.0 | **0.2%** | 99 d / 3% | 95 d / 20% | 63 d / 87% |
| 1.00 | 4.0 | 1.4% | 99 d / 12% | 85 d / 45% | 40 d / 97% |

At k=0.5 an 18.8% per-feature false-alarm rate across ~5 drift-tracked features makes a
spurious alert on a healthy resident near-certain.

**Fix.** Three changes: `cusum_k` 0.5 → 0.75, `cusum_warmup_days = 7` (do not
accumulate while the median/MAD are themselves unstable), and `cusum_decay = 0.98`
(a textbook CUSUM never forgets, so charge from an old fluctuation erodes the effective
threshold forever; ~2%/day gives a ~35-day memory matching the clinical window).

**Why not k=1.0**, despite the best false-alarm rate: it detects a moderate (δ=0.5)
decline only 20% of the time. For a system whose stated purpose is catching gradual
decline before it is obvious, missing 4 in 5 real declines is a worse failure than an
occasional advisory. **k=0.75 is the defensible operating point, not the safest-looking
one.**

Guarded by `T11` (200 stable days, 5 tracked features, 0 CUSUM alerts) and `T11b`
(warm-up and decay tested directly).

---

## Two vacuous tests, and what they cost

Twice in this project a test passed while proving nothing, and both times the cause was
the same: a synthetic fixture so clean that a degenerate path satisfied the assertion.

| test | passed because | real defect it was hiding |
|---|---|---|
| `T6` (CUSUM vs mobility decline) | zero-variance baseline → MAD=0 → every deviation scored \|z\|≥6, so *daily* thresholding "detected" instantly | naive MAD fallback made a 0.1% change look 6-sigma — alert storm for the most regular residents |
| `T10` (CUSUM vs social withdrawal) | CUSUM had already charged to 4.21 on baseline noise before the decline started | 18.8% per-feature false-alarm rate at k=0.5 |

Both were caught by reading the *diagnostic output* of a passing test, not by a failure.
Both are now guarded by a control assertion — a stretch of stable days that must produce
zero alerts — placed **before** the signal, so a detector that fires on noise fails the
test before it can take credit for a detection.

**Practice adopted:** every detection test prints its latency and the maximum daily |z|
it saw. A detection at 0 days, or one where daily |z| exceeded the threshold anyway, is
treated as a failed test regardless of the assertion's verdict.

---

## Pre-training audit of the ADL/fall recipe (2026-08-07)

Notebook 03 has not burned a GPU hour yet, which made this the last free moment to audit
the training path — any defect fixed after the first 12-hour session invalidates that
session. Two were found, both silent, both in the category "the loss curve would have
looked fine".

**The bone stream was not flip-equivariant.** `to_bone()` built its parent map by
overwriting: the torso-closing edge (11, 12) replaced the right hip's parent (shoulder 6)
with the left hip, so the left hip's bone was a vertical shoulder→hip vector while the
right hip's was a horizontal hip→hip vector. Mirroring a pose (which `flip_prob=0.5` does
to half of all training windows) then flipped that bone's *sign* instead of mirroring it —
measured flip-equivariance error 5.93 where every other joint gave 0. Half the ensemble
(`bone`, `bone_motion`) trained against self-contradictory torso encodings. Fixed with
`parent.setdefault` (first-listed, root-outward edge wins); guarded by `S2c`, whose
negative control re-builds the old map and demands it fail.

**Augmentation broke the missing-joint representation.** `normalise()` pins joints RTMO
did not detect at exact zero; translation and noise then shifted *every* joint, so at
train time "missing" was a small random offset while at val/serve time (augmentation off)
it stayed exact zero. The model was being taught an encoding of absence that evaluation
never produces. `augment()` now re-zeroes score≤0 joints after the geometric transforms —
which also makes RTMO-missed and dropout-occluded joints byte-identical, as they should
be. Guarded by `S5c` (invariant is positional-agnostic: the flip legitimately permutes
which *slot* is missing).

Also closed while reading: `train_fall.py` accepted changed hyperparameters on resume
(train_adl refuses them; the fall head's 60-epoch cosine schedule is just as corruptible),
and both trainers overwrote `history.json` on resume, which would have made a resumed
80-epoch run look like a warm-started 40-epoch one in the dissertation's training figure.

Deliberately NOT changed, with reasons: lr = 0.1·batch/128 with 5-epoch warmup (standard
linear scaling, warmup covers the 0.4 peak); model size (data-bound, not capacity-bound —
docstring already argues this); mixup/CutMix (still excluded: blended labels degrade the
safety-critical `falling` decision); focal α=0.75/γ=2 (documented, and the operating point
is re-fitted downstream anyway); no test-time augmentation (would change the eval protocol
mid-project for a fraction of a point).

---

## Second audit round: the extraction path (2026-08-07, same day)

Asked whether the recipe was now clean, the honest answer was no — the remaining defects
were upstream of training, in extraction, where they would have been baked into every
shard. Extraction has not run yet, so these were still free; after the first 12-hour
session each would have meant re-extracting.

**Window labels went to the first-listed action, not the best-covering one.** Charades
actions overlap heavily. The labelling loop broke on the first interval with >= 60%
window coverage *in CSV order*, so a barely-qualifying action listed first beat a
fully-covering one — and an UNMAPPED first qualifier forced `other_idle` even when a
mapped action also covered the window. Coverage decides now, and unmapped actions cannot
claim a window. Pure label-noise removal at zero cost.

**Person slots were assigned by box area per frame, with no memory.** When two people's
apparent sizes cross (one bends, one stands), the slot ranking flips and both channels
splice mid-window — person A's trajectory continues into person B's, and the motion
streams see a teleport at the swap frame. `assign_slots()` (scripts/prepare_skeletons.py)
now matches detections to the previous frame's slot centroids, greedy nearest-first, with
a scale-free association gate (3x the candidate's own bbox diagonal — no real person
moves 3 body-widths in 66 ms) so a new entrant is not glued onto a briefly-undetected
resident's channel. Guarded by `S8`, whose negative control replays the old area-order
rule and demands it teleport.

**normalise() let detection dropouts corrupt scale and origin.** All-zero frames entered
the torso mean (deflating the scale — a person absent for a third of the window got
coordinates inflated ~1.5x), and an absent frame 0 made the origin (0,0), leaving that
window in raw pixel coordinates while every other window sat in torso units. Both moments
now use only frames where the person is detected. Guarded by `S4c`: identical visible
content wrapped in dropout frames must normalise identically (delta observed: 0.0).

The audit stops here deliberately. Remaining candidates and why they wait:
- **2-stream fall head** (joint + joint_motion averaged): plausible sensitivity gain on
  THE safety metric, but it changes serving (pipeline loads one FallHead) and the
  operating-point fit; a design change, not a defect. Decide after the first P2 numbers.
- **Flip TTA at eval**: ~+0.3-0.5% typical, but it changes the eval protocol; if adopted
  it must be reported as such in every table. Decide at notebook 04 time.
- **NTU-60 pretrain init**: listed as an optional asset, but a stock 25-joint 3D NTU
  checkpoint cannot initialise this 17-joint 2D COCO architecture without a conversion
  story; out of scope for the timeline. The smoke-test entry stays as documentation.

---

## Pre-run audit of notebook 02, the fall labels (2026-08-08)

Notebook 02 had not been run yet, so the same free-fix window applied as with the ADL
recipe. Two defects, both in the *only* label source the fall head has.

**Window labels were derived from list index, not from time.** `window_clip()` drops
low-visibility windows, so `enumerate(window_clip(...))` is not a temporal position. With
3 s of occlusion at the head of a 10 s clip the descent label landed 2 windows (2 s) late:
the real descent became `standing` and a standing window became `falling`. Measured, not
theorised. `window_clip(..., with_starts=True)` now returns `(start_frame, window)` pairs
and the notebook labels by each window's midpoint as a fraction of clip duration.

**Some clip lengths produced no `falling` window at all.** The index band `0.4 <= frac
< 0.6` misses entirely when the fraction grid steps over it: a 3 s clip gives {0, 1} and a
5 s clip gives {0, .33, .67, 1}. Short clips are exactly what fall corpora contain, so the
safety-critical class was being starved by arithmetic. The window nearest mid-clip is now
always `falling`; verified every length from 3 s to 15 s yields at least one.

Also added, because a silent zero-yield run inviting "Save Version" is how an unusable
dataset gets published over a good one (Kaggle versions REPLACE): the shard gate asserts
windows exist, that BOTH fall and non-fall windows are present (a binary head trained on
one class reports a meaningless AUROC near 0.5 and nothing says the data was degenerate),
and that at least one `falling` window survived. `N15` locks all of it, including a guard
that fails if the notebook reverts to `enumerate(wins)`.

**P2 is not viable as configured.** Only GMDCSA-24 clones automatically; Le2i, CAUCAFall
and URFD are commented out pending private uploads. Leave-one-dataset-out needs >= 2
corpora, so with one the protocol cannot run. This is a printed WARNING rather than an
assert — a single-corpus shard is still valid for P1/P3 and for a first end-to-end pass —
but it must be stated in the write-up rather than silently reported as a P2 number.

---

## Notebook 02 fall corpora: how each is actually read (2026-08-08)

Asked how the four fall datasets get extracted, and the honest answer was that only one
of them is wired up — and the code that reads it was wrong in a way the tests could not
see. Verified against the real GMDCSA-24 tree via the GitHub API (160 videos, 1.11 GB, no
LFS): `Subject N/{ADL,Fall}/01..25.mp4`, 79 fall / 81 ADL.

**The fall/ADL label lives in the DIRECTORY, not the filename.** Every clip is named
`01.mp4`..`25.mp4`. The old rule (`"fall" in stem or "fall" in parent.name`) happened to
work here via the `Fall/` parent, but it is the wrong thing to key on and would misread a
corpus nesting ADL clips under a fall-ish path. `is_fall_clip()` now walks the path
components, preferring an explicit `ADL`/`Fall` marker and only falling back to the stem.

**Subject ids collapsed four people into one.** `f"{source}_{stem.split('_')[0]}"` maps
`Subject 1/ADL/01.mp4` and `Subject 2/ADL/01.mp4` to the same `gmdcsa_01`. This is not
cosmetic: `split_by_subject` holds ids out, so one id spanning four people puts every
person on both sides of the split, and the cross-subject (P1) number for the safety-
critical head becomes memorisation reported as generalisation. The disjointness assert
cannot catch it — the ids genuinely are disjoint; the *people* behind them are not.
`subject_from_path()` derives the id from the directory layout instead (verified across
GMDCSA / URFD / Le2i / CAUCAFall shapes), and the shard gate now prints every id with its
window count and refuses fewer than two.

Le2i, CAUCAFall and URFD remain commented out pending private uploads — they have no
reliable public mirror. With one corpus, **P2 leave-one-dataset-out cannot run**; the
notebook says so rather than letting notebook 04 report a single fold as a result.

## The service tests were not isolated, and had not been for 40 runs

`connect(path: Path = DB_PATH)` bound the default at function-definition time, so
`api.DB_PATH = tmp / "test.db"` in the test fixture did nothing and every run wrote to the
committed `data/behaviorsense.db`. It surfaced only when `test_x5` compared a total
against `/alerts?limit=50`: the row count grew by 2 per run (58 → 60 → 62 → 64 → 66) until
it crossed the page size, four dozen runs after the leak began.

Two fixes, because either alone leaves the trap open:

- `connect()` resolves `DB_PATH` at call time.
- `fresh_client()` **asserts** the redirect landed, via `PRAGMA database_list`. A fixture
  that silently fails to isolate makes every test above it vacuous — the same lesson as
  the two vacuous detection tests, in a different disguise.

`test_x5` now compares against an unpaginated read and bounds the plausible total, so a
leaked database fails loudly instead of waiting to overflow a page. The true count is
**2 alerts** from one injected fall (`fall_detected` + `fall_no_recovery`); the other 64
rows were accumulated debris. The stale DB was deleted.

---

## All four fall corpora wired into notebook 02 (2026-08-08)

Asked whether the notebook would handle Le2i / CAUCAFall / GMDCSA-24 / URFD, the answer
was no: only GMDCSA was acquired, and testing the label rules against the other three
layouts found two silent failures. Sources verified live before writing any code.

**Acquisition — three fetch themselves, one attaches:**

| corpus | route | verified |
|---|---|---|
| GMDCSA-24 | `git clone --depth 1` | GitHub API: 160 videos, 1.11 GB, no LFS, 79 fall / 81 ADL |
| URFD | 70 direct MP4s, `fenix.ur.edu.pl` | HTTP 206 on fall-01, fall-30, adl-01, adl-40 |
| CAUCAFall | Mendeley file API walk | root listing is figures only; S3 bulk zip is 403 |
| Le2i | attached Kaggle mount | canonical host refuses connections; IMVIA page 404s |

The URFD host every paper cites (`fenix.univ.rzeszow.pl`) is dead; the university's new
domain serves everything. Le2i has no surviving origin at all — 44 Wayback captures of
`le2i.cnrs.fr`, none of them video — so it is consumed as `tuyenldvn/falldataset-imvia`
(10 GB, licence "Unknown") and cited as Charfi et al. (2012) with the mirror named as the
access path.

**Two labelling defects, both silent, both measured:**

*CAUCAFall.* Its activity folders are `Fall forward`, `Fall backward`, `Fall left`... and
`is_fall_clip` tested `part.lower() in FALL_DIR_MARKERS` — an EXACT match. All 50 fall
clips scored ADL, which would have fed real falls to the binary head as negatives. Now a
token match, with ADL checked first so "no fall" cannot read as "fall".

*Le2i.* No path component says fall or ADL: every clip is `video (N).avi` under a room
folder, and the truth lives in `Annotation_files/*.txt` whose first two lines are the
fall's start and end frame. Defaulting to "not a fall" would have buried its ~192 fall
clips in the negatives. `le2i_fall_frames()` parses the annotation and returns None when
it cannot; the notebook SKIPS those clips and counts them rather than guessing. Where the
annotation exists it beats the heuristic outright — real frame boundaries instead of "the
descent is somewhere mid-clip" — so `label_fall_windows()` takes an optional interval and
falls back to the positional rule only for corpora that publish nothing better.

One conversion matters: annotations are in ORIGINAL frame numbers, windows are in
subsampled ones, so the interval is divided by the decode step. Skipping that scales the
fall interval ~2x at 15 fps. Asserted by `N16`.

The mount is also identified by having BOTH video and annotations — a video-only Le2i
mirror is refused loudly instead of being labelled entirely ADL.

---

## Notebook 02's first real run died and lost 27 minutes of finished work (2026-08-08)

The session acquired 3/4 corpora, extracted gmdcsa (160 clips) and urfd (70) cleanly, then
died on a Le2i clip at 1660 s — three `[mp3float] Header missing` lines, then
`DeadKernelError`. Everything was lost. Four defects, none of them in the extraction
itself.

**No incremental flush.** Notebook 01 flushes every 400 MB; notebook 02 accumulated all
four corpora and wrote one npz after the loop. The run had no way to keep what it had
already earned. Now flushes on the same 400 MB trigger *and* at every corpus boundary, so
a crash in corpus N+1 cannot cost corpus N.

**No frame cap.** The mp3float errors are the tell: a malformed container hands back
frames from a broken index indefinitely, and `poses` grows until RAM is gone. Capped at
3000 sampled frames (200 s at 15 fps — no clip in these corpora is close), and clips that
hit it are reported.

**Le2i's root resolved to the whole mount.** Kaggle nests datasets at
`/kaggle/input/datasets/<owner>/<name>/`, and iterating `INPUT.glob("*")` accepted
`/kaggle/input/datasets` itself. Every Le2i subject id then became the owner name —
190 videos collapsed onto one id, the exact P1-defeating collapse fixed for GMDCSA the day
before, reintroduced by a different route. The root is now anchored on the annotations
(`<corpus>/<scene>/Annotation_files` → grandparent), and a candidate is accepted only if
`le2i_fall_frames` actually resolves for at least half a sampled 20 clips — a functional
check, not a shape check.

**CAUCAFall 403.** Mendeley rejects urllib's default User-Agent from a Kaggle IP while
serving the same URL to a laptop. Browser UA on both the API walk and the file fetches.

**And one the fix itself introduced, caught by executing rather than reading.** Rewriting
the labelling left the superseded positional block reachable, so every window was appended
TWICE — the second copy labelled by the old rule, which would have given *ADL clips fall
labels*. A synthetic run over all four corpus layouts caught it: 126 windows where 63 were
expected. `N17` now asserts exactly one `skels.append(` site.

Two tests had to change with it, and both were the good kind of failure. `N15` keyed on
`mids = [` and re-implemented the mid-point maths inline — so it would have kept passing
against its own arithmetic after the notebook moved to `label_fall_windows()`. It now
calls the shared function. `N15`'s gate check also demanded `assert skels`, which is the
pre-flush behaviour; asserting it would have required the very bug being removed.

---

## The kernel death was native, not Python (2026-08-08, second and third runs)

Run 2 died at 1660 s, run 3 at 2367 s. Both inside **URFD**, both ~0.4 s after three
`[mp3float] Header missing` lines. My earlier note said the first death was on a Le2i
clip; that was wrong — checking the timestamps against the corpus order puts both squarely
in URFD.

The flush fix worked: run 3 printed `wrote falls_0000.npz (914 windows)` and
`gmdcsa done` before dying, so gmdcsa survived a crash that would previously have erased
it. But the frame cap and the per-video `try/except` **both failed to fire**, which is the
diagnostic: the fault is not in Python. It is `cv2 -> ffmpeg -> mp3float`, and a
SIGSEGV/abort in native code kills the interpreter outright — no Python-level guard can
see it, and no amount of `except Exception` will.

Checked before designing around it: all 70 URFD mp4s are well-formed `mp42`, video-only,
none truncated (HTTP range probe of every file). So this is not a bad download and cannot
be screened by inspection. URFD's own page calls the mp4s playback previews — the primary
distribution is zipped PNG frame sequences — which is consistent with a container ffmpeg
mishandles.

**Fix: decode every clip once in a child process, before extraction.** The child journals
each attempt *before* trying it, unbuffered. If it dies, its last unmatched `TRY` line
names the poison clip; the parent restarts and the child skips everything already
journalled. Converges in 1 + n_poison runs, each crash costing exactly one clip instead of
a corpus. Extraction then consumes only the cleared list.

Cost is one extra decode pass (~5-8 min). It buys a session that finishes, which matters
because **Save & Run All is headless** — there is no stdin, no prompt, no way to intervene.
A notebook that needs a human mid-run cannot be used that way at all.

`N18` runs the notebook's own child script — extracted via `exec`, not retyped — against a
stub that `_exit(139)`s on marked files, and asserts both poison clips are named and the
loop converges in 3 runs. It also asserts `buffering=1` on the journal: buffered, the TRY
line is still in the child's buffer when it dies and the crash goes back to being
anonymous. `N19` asserts extraction actually filters on the cleared set, since a probe
whose result is ignored is worse than no probe.

Also this round: **CAUCAFall's Mendeley DOI publishes no video at all** — nine
documentation files, 1.1 MB, versions 1-3 return 451, the S3 bulk zip is 403 under any
User-Agent. `Mendeley walk found no video` was accurate; my "it fetches itself" claim was
not. It now resolves from the Kaggle mirror `tuyenldvn/caucafall` (8.33 GB). And Le2i's
root took the *first* annotation folder's grandparent, so it captured `Coffee_room_01`
alone — 48 of ~190 videos, all collapsing onto one subject id. Now the common ancestor of
every annotation folder, with the resolve-check sampling across the tree rather than the
first 20 (a head sample sits entirely inside one scene).

---

## The probe named the culprit, and it was not URFD (2026-08-08, run 4)

Run 4 was the first to survive long enough to be diagnostic, and it overturns the
attribution in the entry above. I wrote that both earlier deaths were inside URFD. They
were not.

```
230/420 clips decode cleanly          = gmdcsa 160 + urfd 70, exactly
11 clip(s) crash the decoder          all Coffee_room_01/video (N).avi
179 clips                             never probed - budget exhausted
```

URFD probes 70/70 clean. The excluded clips are Le2i's, in alphabetical order —
`video (1)`, `(10)`, `(11)` … `(19)` — which is not per-file corruption but the whole
mirror's AVI codec being unreadable by the bundled cv2/ffmpeg. Re-reading run 3's
timestamps with that in hand: it printed `urfd: 70 clips` at 2107.7 s and died at
2366.6 s, ~259 s later — where URFD *ends* and Le2i's first clip begins, before Le2i's own
print. The evidence was there; I had matched the crash to the last thing printed rather
than to what ran next.

`exit -6` (SIGABRT) and `exit -11` (SIGSEGV) confirm the native diagnosis, and the probe
did its job: it converted an anonymous kernel death into a named list.

**Three fixes.**

*The restart budget was a guess and it was wrong.* 11 restarts was sized for "a couple of
bad files"; Le2i crashes on every clip, so it ran out after 11 and the remaining **179
were never probed** — silently dropped from extraction while the log said only "continuing
with what passed". The budget is now one restart per clip (progress is guaranteed: the
child journals TRY before decoding, so every run advances at least one). Never-probed
clips are now counted and reported *separately* from poison, because "we tried and it
crashed" and "we never tried" are different facts and merging them hides the failure.

*Poison is reported by directory, not as a flat list.* 11 lines of near-identical paths
obscured the pattern that made the diagnosis obvious.

*PyAV as a second chance, before writing anything off.* cv2 aborting proves one decoder
failed, not that a clip is unreadable — and Le2i is the only corpus here with **exact
frame-level fall annotations**, the ones that beat the positional heuristic outright.
PyAV links its own ffmpeg and `av>=12.0` was already declared in requirements-train.txt.
Clips that fail cv2 are retried with it; survivors are tagged per-clip in `BACKEND` and
decoded that way during extraction, converting RGB->BGR to match every other corpus (a
colour-order mismatch would be a silent distribution shift in the fall head's training
data). Self-verifying: if PyAV also dies, those clips stay excluded and the cost is a few
minutes of probing.

`N18` runs the full two-backend protocol against a stub — 4/5 kept, 2 rescued, 1 true
poison excluded, cv2 sweep converging in 4 runs. Its stub is built from scratch rather
than string-patched out of the real child: the earlier version patched it, the child grew
a backend branch, the patch stopped matching, and the test *errored instead of testing*.
`N18b` keeps the structural checks against the real child.

---

## Fall shards extracted, and the class balance they imply (2026-08-08)

Notebook 02 completed: **3,064 windows, 4 corpora, 12 shards**, after PyAV rescued all 190
Le2i clips that cv2 aborted on. Composition — caucafall 1159, gmdcsa 914, le2i 794,
urfd 197.

**Le2i contributes 74 of its 190 clips, spanning 3 of 6 scenes.** 116 clips are skipped
because they carry no readable `Annotation_files` entry, and Le2i is the one corpus with no
fall/ADL marker in its paths — so skipping is correct (guessing would put real falls in the
negatives), but the loss was invisible: the skip counter was incremented in two places and
printed in none. It is now reported per corpus and per run. `Lecture_room` and `Office`
ship no annotations at all; `Coffee_room_02` is absent for reasons the mirror does not
explain.

**The shards are 81.6% POSITIVE** (2501 fall / 563 non-fall), which is inevitable — fall
corpora are made of fall clips — and it broke two things downstream that the shard files
themselves cannot show:

*Focal alpha was inverted.* `alpha=0.75` up-weights the POSITIVE class. Against 82%
positives that weights the majority 3x the minority, the exact inverse of what focal loss
is for, with nothing in the loss curve to reveal it. `train_fall.py` now refuses
`pos_rate > 0.5 and alpha > 0.5` outright rather than warning — the run would otherwise
complete and report a plausible AUPRC.

*The operating point was unfittable.* `fit_operating_point` expresses the budget as false
alarms per HOUR of monitored video. 563 negatives is 0.31 h, and 0.08 h after the val
split, so a "1 FA/hour" budget permits 0.08 of one alarm and the fitted threshold collapses
to whatever admits zero — an artefact of test-set size, not a deployment choice. Measured
on a synthetic detector of *fixed* quality:

| negatives in val | fitted sens | FA/h | AUPRC |
|---|---|---|---|
| 141 (today's shards) | 0.65 | 0.00 | 0.993 |
| 1,400 | 0.54 | 0.00 | 0.957 |
| 14,000 (realistic mix) | 0.37 | 0.90 | 0.790 |

Same model, three different stories. The fix is upstream of the loss: `train_fall.py`
derives its target from labels 7/8, so **ADL windows are valid negatives** and notebook 03
now passes `*FALL, *ADL`. That takes the positive rate from 81.6% to ~1.5%, still
optimistic against a real home but the right side of realistic. A warning fires whenever
the val negatives cannot express the FA budget. `N18` locks all of it, executing both
branches of the alpha guard rather than grepping for it.

**Subject-id granularity, stated because it decides what P1 measures.** URFD is flat
(`fall-01-cam0.mp4`) and publishes no clip→volunteer mapping — only per-frame posture
labels — so every clip becomes its own "subject". Its ~70 sequences come from a handful of
volunteers, so cross-subject splitting cannot separate them and P1 is optimistic for the
URFD portion, exactly as the Charades video-id proxy already is. The measurable consequence
is variance: holding out 20% of *subjects* holds out 3–45% of *windows*, and across 12
seeds val folds ranged 160–1406 windows (1 of 12 degenerate). Notebook 02 now prints the
val-fold size across 8 seeds and warns under 200.

---

## Notebook 03: shard selection failed on a nested mount (2026-08-09)

Preflight passed, the env fix from the previous round held, and then:

```
attached  ['competitions', 'datasets']
ADL mounts:
fall mounts:
ADL shards: 0, fall shards: 0
AssertionError: no ADL shards found
```

All four datasets were correctly attached. `shard_paths()` iterated `INPUT.iterdir()` and
matched the mount NAME against "adl"/"fall" — but attaching a dataset by URL nests it as
`/kaggle/input/datasets/<owner>/<name>/`, so the top level holds only `competitions` and
`datasets`. Neither contains "adl" or "fall", so both lists came back empty.

The resolver cell three cells earlier was immune to this and resolved wheels, weights and
code correctly, because it globs with `**/`. This cell was the last one still matching on
the top level — the third bug from the same root cause, after notebook 01's carry-forward
and notebook 02's Le2i root ([[kaggle-mount-nesting]] records the pattern).

Both notebooks 03 and 04 now group `INPUT.glob("**/*.npz")` by whichever path component
names the corpus, so mount depth is irrelevant. `N10b` builds flat and nested fixtures and
asserts both yield the same shards, with the `behaviour`/`behavior` spelling deliberately
mismatched in the fixture to prove tolerance survived the rewrite.

**A note on how the test caught its own author.** N10b's first version asserted
`"INPUT.iterdir()" not in cell` and failed — on the fix's own explanatory comment, which
names the old call. Checking a string against source that includes prose is not checking
the code; it now strips comment lines first. Cheap here, but the same shape as a test that
passes for the wrong reason.

---

## Shard selection ignored nested mounts (2026-08-09)

Notebook 03 aborted at cell 4 with `no ADL shards found. Attached: ['competitions',
'datasets']` while all four datasets were correctly attached. `shard_paths()` iterated
`INPUT.iterdir()` and matched the mount NAME, so with a URL-attached dataset — nested as
`/kaggle/input/datasets/<owner>/<name>/` — it compared `"adl"`/`"fall"` against the two
container directories and returned nothing.

The resolver cell in the same notebook was immune because it globs `**/`. This one cell
was still name-on-top-level, which is the third time this project has paid for that
assumption (see [[kaggle-mount-nesting]]): notebook 01's carry-forward reported "fresh
start" against a real prior version, notebook 02's Le2i root resolved to the whole mount
and collapsed 190 videos onto one subject id, and now this.

Both notebooks 03 and 04 carried the same defect; 04's variant was silent (no per-mount
print), so it would have surfaced as an empty evaluation rather than an assertion.

Fixed by grouping every `**/*.npz` by whichever *path component* names the corpus, which
works at any depth and keeps the `behaviour`/`behavior` spelling tolerance. `N10b` now
runs the real cell against both flat and nested fixtures and asserts the code — not the
prose — never iterates the top level.

Two things this run got right that earlier ones did not: the preflight passed
(`PREFLIGHT PASSED - safe to start training`, GPU sm_120, bf16 matmul, ST-GCN++ fwd+bwd
at 0.19 GB peak), confirming the `env={**os.environ, ...}` fix; and the contract reported
12/12, confirming the uploaded snapshot carries the `train_fall.py` alpha guard.

---

## The ADL heads overfit for 70 of 80 epochs (2026-08-09)

First real Blackwell training run, 2.7 h. Four ADL streams plus the fall head, on the
165,109-window Charades extraction. The headline is that **80 epochs was ~8x too many and
actively made the result worse.**

| stream | best mca | at epoch | ep79 | lost |
|---|---:|---:|---:|---:|
| adl_bone | 0.184 | 11 | 0.137 | −25.5% |
| adl_joint | 0.179 | 9 | 0.129 | −27.9% |
| adl_joint_motion | 0.129 | 10 | 0.092 | −28.7% |
| adl_bone_motion | 0.125 | 8 | 0.091 | −27.2% |

Four independent runs — different input streams, different data views — peaking inside the
same three-epoch window is not noise. The 80 was copied from ST-GCN++'s NTU-60 recipe,
which has roughly 10x more labelled windows per class; the budget was chosen by analogy
rather than by measurement. Roughly 2.4 of the 2.7 GPU-hours went into memorising.

Nothing was lost — `best.pt` is written on improvement and notebook 04 loads `best.pt` —
but the ADL numbers to report are the **best** column, not the final epoch.

**The fall head measured the opposite**, which is why the fix is asymmetric rather than a
single global change: AUPRC peaked at **epoch 56 of 60**, still climbing. One logit over a
~1.5% positive rate needs the epochs an 18-class head does not. So `train_adl.py` drops to
30 epochs with `--patience 8`, while `train_fall.py` keeps 60 with `--patience 15`. Both
defaults are replayed against the real measured curves in `S9`: the ADL rule stops at
ep19 while keeping 0.184@ep11, and the fall rule runs to ep59 while keeping 0.822@ep56.
`S9` also asserts `best.pt` is written *before* the stop check, so stopping can never
discard the epoch it stopped for.

## What the per-class breakdown settled

macro-F1 of 0.15 could mean a hard task, a broken pipeline, or a starved label map — three
diagnoses needing opposite responses. The per-class vector was already in `history.json`
and simply was never printed. Printed, it reads:

```
other_idle 0.518 | lying_down 0.424 | sitting 0.304 | cooking 0.207 | walking 0.191
...  17/18 classes above F1 0.02; 1 at ~0: bending_reaching (13 val windows)
```

A broad non-zero spread across 17 of 18 classes rules out collapse and rules out a broken
pipeline. The model learns real structure from 2 s skeleton windows of untrimmed,
multi-label Charades — a genuinely hard task where a window often contains no evidence of
its labelled action. That is the honest framing for the write-up, and it is now
attributable rather than asserted.

## Three reporting bugs found in the same cell

The summary cell that displays all of this was wrong three ways, each silent:

1. It read a key called `mca`; the trainer writes `mean_class_acc`. `max()` fell through
   to `default=0`, so four 80-epoch runs all reported `best=0.000` — a value real training
   noise never produces.
2. Re-run in a fresh session it printed **nothing at all**, then invited "Save Version".
   `/kaggle/working` is wiped between sessions, the glob yielded zero iterations, and the
   loop body never executed. The same silent-empty-iteration failure already fixed in
   notebooks 00, 01 and 02 — it survived here.
3. It globbed `/kaggle/input/**/history.json` and so read the smoke runs that ship inside
   `behaviorsense-code`, reporting a 25-epoch CPU fixture (`mca 0.647`) as the best result
   on screen. The carry-forward guard blocks those from being *resumed*; the same
   exclusion was never applied to *reporting*.

Run type is now inferred from the metrics a history actually recorded rather than from its
directory name — the name heuristic died on `smoke_fall`, a fall run whose name does not
start with "fall". `N14` covers all three.

**Also added: `_generate.py` now compiles every code cell before writing the `.ipynb`.**
I introduced the same escaping defect three times in one day — a `\n` written inside a
triple-quoted cell literal becomes a real newline and splits a string — each time found
only by compiling afterwards. Verified with a negative control: injecting the defect now
aborts generation and names the likely cause.
