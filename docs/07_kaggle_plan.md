# 07 — Running the project on Kaggle (notebooks, online/offline split)

The training GPU (RTX PRO 6000 Blackwell, 96 GB, sm_120) runs with **no internet**, so the
work splits into ONLINE notebooks that stage everything and OFFLINE notebooks that consume
it. All five live in [`notebooks/`](../notebooks/) and are generated from
[`notebooks/_generate.py`](../notebooks/_generate.py) — edit that file, not the JSON.

## The notebook DAG

```
(local CLI) upload repo ──► behaviorsense-code ─┬──────────────┬───────────────┐
                                                │              │               │
00  stage assets   (ONLINE, CPU)  ──► behaviorsense-wheels ────┼──► 03 train (OFFLINE) ──► behaviorsense-runs
                                  ──► behaviorsense-weights ───┤        │  ▲ (resume: attach own output)
01a stage Charades (ONLINE, CPU)  ──► charades-480p ──┐        │        ▼
01  ADL shards     (OFFLINE, GPU) ◄───────────────────┘        └──► 04 evaluate (OFFLINE) ──► results/*.md
                                  ──► behaviorsense-adl-shards ─┤       ▲
    ▲ (continuation: attach own output)                         │       │
02  fall shards    (ONLINE, GPU)  ──► behaviorsense-fall-shards ┘       │
                                       Kaggle Model: Qwen2.5-7B-Instruct ┘
```

**Why 01 is offline.** Extraction is ~20-30 h of RTMO inference, so it never fitted in one
session regardless. Running it online forced it onto a T4 *and* re-downloaded the 13 GB
Charades archive at the start of each of the 3-4 sessions — ~30 min of idle GPU per run,
purely because the notebook needed the network. Splitting the download into 01a (CPU, no
accelerator, run once) lets extraction run offline on the Blackwell instead.

Expect roughly 2-3x, not the raw hardware ratio: RTMO is called at `batch=1` on 480p
frames, which is latency-bound rather than throughput-bound, and video decode (~20% of the
work, every frame decoded while every 2nd is inferred at `FPS_SAMPLE=15`) does not speed up
at all. The reliable win is the eliminated download and the faster card, not a 10x.

## Who needs what

| notebook | environment | inputs (attach) | outputs (create/update dataset) |
|---|---|---|---|
| `00_stage_assets_online` | ONLINE, CPU | `behaviorsense-code` | `behaviorsense-wheels` (~7 GB), `behaviorsense-weights` (~0.3 GB) |
| `01a_stage_charades_online` | ONLINE, **CPU (no accelerator)** | nothing | `charades-480p` (~13 GB, private) |
| `01_prepare_adl_shards_offline` | **OFFLINE**, Blackwell | `behaviorsense-code`, `-wheels`, `charades-480p`; **from run 2:** `behaviorsense-adl-shards` (its own previous version) | `behaviorsense-adl-shards` |
| `02_prepare_fall_shards_online` | ONLINE, GPU | `behaviorsense-code` + Kaggle dataset `tuyenldvn/falldataset-imvia` (Le2i) | `behaviorsense-fall-shards` |
| `03_train_blackwell_offline` | **OFFLINE**, Blackwell | `behaviorsense-code`, `-wheels`, `-weights`, `-adl-shards`, `-fall-shards`; **resume:** `behaviorsense-runs` | `behaviorsense-runs` |
| `04_evaluate_blackwell_offline` | **OFFLINE**, Blackwell | everything 03 uses + `behaviorsense-runs` + **Kaggle Model** `Qwen2.5-7B-Instruct` | `results/*.md` (optionally `behaviorsense-results`) |

## Step 0 — upload the repo from this machine (local CLI, once)

```bash
pip install kaggle
```

```bash
cd C:/Users/Admin/Desktop/EmotionSense-Extended && printf '{"title": "behaviorsense-code", "id": "YOUR_USERNAME/behaviorsense-code", "licenses": [{"name": "other"}]}' > dataset-metadata.json && kaggle datasets create -p . --dir-mode zip
```

Later updates go through the publisher, **not** `kaggle datasets version -p .` directly:

```bash
python scripts/publish_code_dataset.py -m "why this version exists"
```

Why a script rather than the raw command: `-p .` uploads the working tree verbatim, and
`.kaggleignore` is **not read by the Kaggle CLI**. That cost a session. `runs/` and
`data/shards/` went up, and notebook 03 then resumed `runs/adl/last.pt` (a 25-epoch CPU
smoke checkpoint, `smoke=True`) onto an 80-epoch cosine schedule while
`data/shards/_smoke_fall.npz` — 400 synthetic windows whose name contains "fall" — joined
the real fall corpus. Neither failed; both produced plausible numbers from contaminated
inputs.

The publisher stages only `src/ scripts/ configs/ weights/` into a temp directory,
re-checks the repo contract **against the staged copy**, and refuses to upload if an
`.npz`, a `runs/` path, or a `_smoke*` fixture survived. Notebooks 03/04 also reject both
independently (tests N10b/N10c) — a guard downstream of a bad upload is the second line of
defence, not the first.
Keep it **private** (Charades/Market-1501/MSMT17 licence discipline). The repo already
contains the OSNet weights (27 MB), so no separate weight upload is needed for those.

## Step 0b — prove the wheel staging resolves before spending a session on it

Notebook 00 died in pip's resolver twice, and each attempt cost a session. Run this from
your own machine (needs internet, ~1 min, downloads a few hundred KB) before touching
Kaggle:

```bash
python scripts/verify_wheel_resolution.py
```

It reads torch 2.7.1+cu128's own `METADATA` out of the remote 1 GB wheel over HTTP range
requests, then checks every pinned `nvidia-*`/`triton` dependency publishes a wheel whose
platform tag notebook 00's `--platform` ladder actually names. Exit 0 means notebook 00
will resolve; nonzero names the pin to fix. Measured on 2026-08-06: 15/15 pins stageable
for cp311 and cp312.

Why it is not just `pip download --dry-run`: `--platform` governs wheel *tag* matching and
does **not** set `platform_system`, so a dry run on Windows or macOS marker-excludes every
`nvidia-*` dependency (`platform_system == "Linux"`), resolves 10 packages happily, and
proves nothing about the path that broke.

That failure path, for the record: `torch 2.7.1+cu128` pins
`nvidia-cudnn-cu12==9.7.1.26`, which publishes **only** a `manylinux_2_27_x86_64` wheel.
The original `--platform {manylinux2014, manylinux_2_28}` list excluded that tag, the
candidate set went empty, and pip backtracked across every cu128 torch (2.7 → 2.11) before
reporting `ResolutionImpossible`. Six of the fifteen pins were unreachable under that list,
across three different tags (`manylinux2010`, `manylinux_2_12`, `manylinux_2_27`) — so
notebook 00 enumerates the whole glibc ladder oldest-first rather than hand-picking tags.
Oldest-first matters: pip prefers earlier `--platform` entries, so a package shipping
several wheels stages the most portable one, which is the right default when the offline
image's glibc is not knowable from here. `tests/test_notebooks.py::N9` fails if the
notebook's ladder and the verifier's ever drift.

## When to re-upload `behaviorsense-code`

Not every session. The dataset is a snapshot of `src/`, `scripts/`, `configs/` and
`weights/` — the notebooks live in Kaggle's editor, not in the dataset, so editing a
notebook needs no upload.

Re-upload when any of these change:

| changed | affects |
|---|---|
| `src/behaviorsense/**` | 01, 02, 04 (imported directly) |
| `scripts/*.py` | 01 (`build_charades_map`, `prepare_skeletons`), 03 (`train_adl`, `train_fall`, `kaggle_smoke_test`), 04 (`eval_hallucination`) |
| `configs/taxonomy.yaml` | 01 (class map) |
| `weights/*.pth` | serving-time ReID only |

Editing `tests/`, `docs/`, `README.md` or `notebooks/` changes nothing on Kaggle.

```bash
python scripts/publish_code_dataset.py -m "sync"
```

**You do not have to track this by hand.** The resolver cell in every notebook enforces a
repo contract — the specific files and APIs that notebook calls must exist in the mounted
copy. A stale snapshot fails on the resolver cell with the file, the missing token, and the
command to fix it, instead of surfacing later as something misleading.

That guard exists because of a real failure mode: notebook 03 passes `--profile training`
to `kaggle_smoke_test.py`, and a snapshot predating that flag exits 2 from argparse, which
notebook 03 reports as `PREFLIGHT FAILED`. That reads as a missing weight file, and the
weights are fine — it is the *code* that is old. Feature-detecting each token means the
contract needs no manual version bumping: it fails exactly when the snapshot is too old for
the notebook running against it. `tests/test_notebooks.py::N13` also asserts every contract
token exists in the local checkout, so the guard can never reject a fresh upload.

The contract is now 15 tokens. Two were added after notebook 04's first complete run,
because a stale snapshot would have silently produced the *old* measurement rather than
failing: `repair_claim` and `unusable_rate` gate the three-arm hallucination table, and
without them the free-decoding arm reports `nan%` from an empty denominator — a number that
looks like missing data when the finding is total failure.

## Two Kaggle behaviours that cost a session each

**onnxruntime-gpu must be pinned to 1.26.x.** From 1.27 the PyPI GPU wheels are built
against **CUDA 13**; Kaggle's image is CUDA 12. The mismatch does not fail loudly — the
provider is listed by `get_available_providers()`, then fails to load with
`libcublasLt.so.13: cannot open shared object file`, and RTMO extracts on **CPU** at
roughly 1/50th speed. 1.26.x is the newest CUDA 12.8 build, the same toolkit as
torch 2.7.1+cu128. Two guards follow from this:

- the setup cell installs `rtmlib` with `--no-deps` and uninstalls CPU `onnxruntime` first.
  rtmlib depends on the CPU package, and both own the same `onnxruntime` module directory,
  so whichever pip writes last wins — observed: GPU installed, then CPU overwrote it.
- the pose probe asserts on `body.session.get_providers()`, not on
  `ort.get_available_providers()`. The first reports what is **in effect**; the second only
  reports what the build was compiled with, and passes while CUDA is unusable.

**Kaggle auto-extracts archives in dataset output.** Notebook 01a saves
`Charades_v1_480.zip` unopened, but the published `charades-480p` contains
`Charades_v1_480/Charades_v1_480/*.mp4` (9848 files) with no `.zip` anywhere — so the
zip-only reader asserted "charades-480p is not attached" against a dataset that was
attached *and* extracted, seven minutes into the session.

Notebook 01 now detects both layouts and prints which one it found. Pre-extracted is the
better case and needs no unpack step, so nothing lands in the 20 GB working quota; the zip
path still slice-extracts. Both expose the same `rows` / `video_path()` interface, so the
extraction cell has no branch of its own. `N2d` runs the cell against both fixture layouts
and asserts identical resolution; `N2c` locks the pose-probe ordering.

## Dataset naming does not matter

The offline notebooks resolve every asset by **content**, not by dataset name: the wheel
cache is "the directory containing `*.whl`", the weights are "the directory containing
`rtmo-l.onnx`", the repo is "the directory containing `src/behaviorsense/__init__.py`".

This is deliberate. Three things vary and none is knowable when the notebook is written:

- notebook 00 emits `wheels/` and `weights/`, which you may publish as **one** dataset or
  two (observed in practice: a single `behavioursense-WW` holding both)
- the dataset title is free text, and this project has been uploaded as both
  `behaviorsense-*` and `behavioursense-*`
- Save Version nests the working directory, so files land at `<mount>/kaggle/working/...`

A name-matching notebook is wrong under at least one of those, and it fails on cell 1 of a
12-hour session. Shard datasets still match on name (ADL and fall shards are the same file
type, so only the mount distinguishes them) but the match is spelling-tolerant and asserts
non-empty. `tests/test_notebooks.py` N12/N13 cover all these layouts.

So: name the datasets whatever you like. Attach them all; the notebooks sort it out.

## Session sequence

1. **00** once (CPU, ~30 min, mostly downloads), after Step 0b passes locally. Save Version
   → create the two datasets.
2. **01a** once (**CPU, accelerator None**, ~30–40 min, pure download). Save Version →
   create a PRIVATE dataset `charades-480p`. Never needs re-running.
3. **01 — ✅ DONE 2026-08-07** in a single session: 7,985/7,985 videos, 165,109 windows,
   **6.26 h** wall clock, 221 MB across 6 shards. Full result and the class-distribution
   caveats: [results/adl_extraction.md](../results/adl_extraction.md). The instructions
   below apply only to a re-extraction with different settings.

   (**OFFLINE Blackwell**, ~6 h of a 12 h session.) Leave
   `VID_START, VID_END = 0, 99999` and attach `charades-480p` + the wheels + the code.
   **Quick Save** at the end — "Save & Run All" re-executes from scratch and spends
   another 6 h re-extracting.

   Measured 2026-08-07: **1333 videos/h** with `CUDAExecutionProvider` confirmed in use.
   The notebook reads `Charades_v1_train.csv` = **7,985** videos (the 9,848 figure is
   train + test; only train carries labels), so the full pass is ~6 h and yields ~232k
   windows / ~0.8 GB compressed — far under the 20 GB working cap. The earlier
   "2-4 sessions, 20-30 h" estimate predates this measurement and was 3-5x pessimistic.

   The `VID_START/VID_END` slicing stays for *timeout recovery*, not for fitting the job:
   shards flush every 400 MB (~1,100 videos), so an interrupted session keeps everything
   already written, and a continuation run sets `VID_START` to the number the gate cell
   prints and attaches the previous `behaviorsense-adl-shards` version. The carry-forward
   cell is what lets that accumulate (Kaggle versions REPLACE content). The gate reports
   `COMPLETE n/7985` or `PARTIAL n/7985` so you never have to work this out by hand.

   The setup cell asserts CUDA is **in use** (`body.session.get_providers()`), not merely
   compiled in — `get_available_providers()` lists what the build supports even when the
   provider failed to load, which is how a CUDA-13 wheel once extracted on CPU at ~1/50th
   speed while reporting success.
4. **02** once (**GPU T4 x2, internet ON**, ~15-20 min). Attach `behaviorsense-code` and
   the Kaggle dataset **`tuyenldvn/falldataset-imvia`** ("Le2i Fall Dataset", ~10 GB).
   Everything else fetches itself.

   Four corpora, four different ways of saying "this clip is a fall" — verified against
   the live sources on 2026-08-08:

   | corpus | acquired by | fall label lives in |
   |---|---|---|
   | GMDCSA-24 | `git clone --depth 1` (160 clips, 1.11 GB, no LFS) | `Subject N/Fall/` vs `Subject N/ADL/` |
   | URFD | 70 direct MP4s from `fenix.ur.edu.pl` | `fall-NN` / `adl-NN` filename prefix |
   | CAUCAFall | Mendeley file API walk, DOI `10.17632/7w7fccy7ky` | activity folder, **multi-word** (`Fall forward`) |
   | Le2i | **attached mount**, found by content | `Annotation_files/*.txt`, first two lines = fall start/end frame |

   Why the Blackwell is not used: it has no internet, and three of the four corpora are
   downloaded. The job is ~15 min either way — the fixed costs (clone, pip, model load)
   dominate the ~2-5 min of inference over ~600 short clips.

   Two traps this configuration exists to avoid, both measured rather than assumed:
   CAUCAFall's `Fall forward` folders failed an exact-match rule and scored all 50 of its
   fall clips as ADL; Le2i has **no** marker in any path component, so a default of "not a
   fall" would have written its ~192 fall clips into the negatives. Le2i clips whose
   annotation cannot be read are SKIPPED and counted, never guessed — and where the
   annotation exists it beats the heuristic outright, giving real frame boundaries instead
   of "the descent is somewhere mid-clip". `N16` locks every case.

   The first real run of this notebook died at 28 minutes on a malformed Le2i AVI and lost
   both corpora it had already finished. Three things changed as a result, and they are the
   reason a rerun is cheap: shards are **banked incrementally** (every 400 MB and at each
   corpus boundary, as notebook 01 already did), each clip is isolated so one bad file is
   reported instead of fatal, and frame reads are **capped at 3000 sampled frames** —
   a broken container otherwise hands back frames from a corrupt index until RAM is gone,
   which is what the `[mp3float] Header missing` lines preceding the kernel death were.

   Four corpora is also what makes **P2 leave-one-dataset-out** meaningful: each fold
   holds out a genuinely different capture setup (phone video, Kinect at 1 m, IR camera in
   a wall corner at 1080×960, 320×240 across four rooms). With one corpus it is a
   formality, and the shard gate says so rather than letting notebook 04 report a single
   fold as a result.
   **Measured 2026-08-08:** 3,064 windows from 4 corpora in ~40 min on T4 x2.
   caucafall 1159, gmdcsa 914, le2i 794, urfd 197. PyAV rescued all 190 Le2i clips (cv2
   SIGABRTs on that mirror's codec); 520/520 clips decoded across the two backends.

   Le2i contributes **74 of 190 clips, 3 of 6 scenes** — 116 carry no readable
   `Annotation_files` entry and are skipped rather than guessed, since Le2i has no
   fall/ADL marker in its paths. `Lecture_room` and `Office` ship no annotations at all.
   The skip counts are printed per corpus; quote them in the write-up next to the shard
   totals.

   The shards are **81.6% positive**, which is what a corpus of fall clips looks like and
   is why notebook 03 trains the fall head on `*FALL, *ADL` rather than the fall shards
   alone — see step 5.

5. **03** one to three times (OFFLINE Blackwell). Order inside the notebook is fixed:
   preflight (1 min) → carry forward checkpoints → 4 ADL streams in parallel → fall head.

   The fall head consumes **both** shard sets. `train_fall.py` derives its binary target
   from labels 7/8, so every ADL window is a valid negative — and the fall shards alone are
   81.6% positive, which inverts focal `alpha` (0.75 up-weights the *majority* there) and
   makes the operating point unfittable: 563 negatives is 0.31 h of "monitored video", so a
   1 FA/hour budget permits 0.08 of an alarm and the threshold collapses to whatever admits
   zero. Adding the ADL shards puts the positive rate near 1.5%. `train_fall.py` refuses
   the inverted-alpha mix outright rather than completing with a plausible-looking AUPRC.
   On timeout: Save Version anyway; next session attaches `behaviorsense-runs` and resume
   continues **bit-exactly** (RNG state is checkpointed; changed hyperparameters are
   refused rather than silently re-curving the LR schedule).
6. **04** once per finished training state. Produces P1/P2/ablation tables, the fitted
   serving temperature, and the Qwen constrained-vs-unconstrained hallucination table.

## Decisions worth stating

- **No combine/merge notebook.** `train_adl.py --shards` takes any number of paths, and
  notebook 03 globs across *all* attached shard datasets. A merge notebook would copy
  bytes into a third dataset, double the storage bill, and change nothing downstream.
  Accumulation across sessions is handled inside notebook 01 by carrying its own previous
  output forward.
- **Qwen2.5-7B-Instruct is attached as a Kaggle Model, not uploaded.** Model inputs are
  mounted read-only regardless of the internet toggle, which sidesteps a ~16 GB upload.
- **Wheels are staged for python 3.11 AND 3.12** because the offline image's interpreter
  is unknown ahead of time; notebook 03 checks `torch.cuda.get_arch_list()` at runtime
  and installs the staged cu128 build only if the preinstalled torch lacks sm_120.
- **MSMT17 stays local.** τ = 0.355 is already fitted and committed
  ([results/reid_eval.md](../results/reid_eval.md)); re-uploading a licence-restricted
  dataset to cloud storage buys nothing and risks the redistribution line. ReID therefore
  needs **no Kaggle notebook**.
- **NTU-pretrained ST-GCN++ init is a stretch goal, not a dependency.** Our
  self-contained `STGCNpp` trains from scratch; consuming PYSKL checkpoints would need a
  key-mapping shim that does not exist yet. The plan says so instead of staging weights
  nothing consumes.
- **Charades subject ids do not exist publicly**, so the P1 split uses video id as the
  subject proxy — stated in the eval tables, not hidden.
- `/kaggle/working` is capped at 20 GB → large downloads go to `/tmp` (ephemeral);
  working holds only shards/checkpoints/results.

## What comes back from the GPU week

- `behaviorsense-runs`: 4 ADL stream checkpoints + fall head (EMA + optimizer + RNG —
  resumable and reproducible).
- `results/`: P1 per-stream + ensemble table, fitted temperature for
  `ActivityConfig`, P2 leave-one-dataset-out fall AUROC, logit-vs-prob ablation,
  `hallucination_qwen.md` (constrained vs unconstrained, interpretable against the stub
  anchors in [results/hallucination.md](../results/hallucination.md)).
- Those numbers land in README §Measured results and replace every claim currently
  marked "pending".
