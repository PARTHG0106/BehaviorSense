"""Generate the five Kaggle notebooks from one reviewable source.

Notebooks are JSON, which is miserable to hand-edit and to diff. This file is the single
source of truth: edit the cell lists here, re-run, and the .ipynb files are rebuilt
deterministically. Run: python notebooks/_generate.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def nb(cells: list[tuple[str, str]]) -> dict:
    out = []
    for kind, src in cells:
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        out.append(cell)
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": out,
    }


def write(name: str, cells: list[tuple[str, str]]) -> None:
    path = HERE / name
    for i, (kind, src) in enumerate(cells):
        if kind != "code":
            continue
        try:
            compile(src, f"{name}:cell{i}", "exec")
        except SyntaxError as exc:
            raise SystemExit(
                f"{name} cell {i} is not valid Python: {exc.msg} (line {exc.lineno})\\n"
                f"  A common cause: a backslash-n written inside a triple-quoted cell "
                f"literal becomes a real newline and splits a string.\\n"
                f"  {(exc.text or '').strip()[:100]}") from None
    path.write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
    print(f"wrote {path.name} ({len(cells)} cells)")


# Appended to both resolvers. The repo dataset is a SNAPSHOT: it does not track the local
# checkout, so a notebook can call a flag or a module that the mounted copy predates. That
# failure is nasty because it does not name its cause - passing --profile to a
# kaggle_smoke_test.py that predates it exits 2, and notebook 03 reports "PREFLIGHT FAILED",
# which reads as a missing asset rather than a stale dataset.
#
# So feature-detect. Each entry is (path, token, why): the file must exist, and if a token
# is given it must appear in the file. Tokens are the specific APIs the notebooks call, so
# this needs no manual version bumping - it fails exactly when the snapshot is too old for
# the notebook running against it, and says which command fixes it.
REPO_CONTRACT = '''

CONTRACT = [
    ("scripts/kaggle_smoke_test.py", "--profile",      "notebook 03 preflight"),
    ("scripts/train_adl.py",         "--stop-after",   "resume guard (notebook 03)"),
    ("scripts/train_adl.py", "--tau-train",
     "notebook 03 passes --sampler/--tau-train; a snapshot predating them exits 2 from "
     "argparse, which notebook 03 reports as a failed stream rather than stale code"),
    ("src/behaviorsense/data/skeleton_dataset.py", "def load_subject_map",
     "video-id -> Charades actor-id remap for a person-disjoint P1 split (notebooks "
     "03/04). A stale snapshot silently reverts P1 to video-disjoint - same person in "
     "train and val - while printing numbers that look identical"),
    ("scripts/train_fall.py", "pos_rate > 0.5 and args.focal_alpha > 0.5",
     "refuses focal alpha that up-weights the majority; a snapshot without it trains "
     "the fall head on 82% positives and reports a plausible but meaningless AUPRC"),
    ("scripts/prepare_skeletons.py", "def assign_slots",
     "slot tracking + windowing (notebooks 01/02)"),
    ("scripts/prepare_skeletons.py", "with_starts",
     "fall labelling by true frame position (notebook 02); index-derived position "
     "mislabels the descent whenever a window is dropped"),
    ("scripts/prepare_skeletons.py", "def le2i_fall_frames",
     "Le2i has no fall/ADL marker in any path component; without this its ~192 fall "
     "clips land in the negatives (notebook 02)"),
    ("scripts/prepare_skeletons.py", "def label_fall_windows",
     "shared fall labelling: exact interval for Le2i, positional fallback elsewhere"),
    ("src/behaviorsense/data/skeleton_dataset.py", "keep_root_motion",
     "falls are unlearnable without it"),
    ("src/behaviorsense/models/ensemble.py", "def per_stream_logits", "notebook 04 ablation"),
    ("src/behaviorsense/models/ensemble.py", "def flip_windows",
     "test-time flip augmentation (notebook 04 levers cell). A stale snapshot would accept "
     "`clf.tta = True` as a new attribute and silently do no TTA, reporting the "
     "unaugmented number as if it were augmented"),
    ("src/behaviorsense/models/stgcnpp.py", "parents.setdefault",
     "flip-equivariant bone stream; without it half the ensemble trains on sign noise. The "
     "token tracked the local name `parent` and broke when the function grew an explicit "
     "`parents` argument for Toyota's 15-node tree - a rename silently disarming a staleness "
     "guard is exactly what this list exists to catch, so it caught itself"),
    ("src/behaviorsense/kaggle_artifacts.py", "def find_run_dir",
     "one rule for 'is this real session output or a dev leftover'; four call sites "
     "learned it separately and the fourth was missed"),
    ("src/behaviorsense/agents/reasoning/reporter.py", "def force_greedy",
     "pins BOTH decoding arms to greedy (notebook 04). Without it the constrained arm "
     "inherits Qwen's generation_config (do_sample=True, temperature=0.7) while the free "
     "arm is greedy, so the comparison measures temperature instead of grammar - the "
     "constrained rate moved 8.2% -> 10.3% between two runs of identical code"),
    ("src/behaviorsense/agents/reasoning/reporter.py", "def _chat_text",
     "both arms send the SAME templated text (notebook 04). The constrained path used to "
     "hand outlines the raw prompt, so one arm got a Qwen chat turn and the other a naked "
     "instruction block - a second confound on top of the sampling one"),
    ("src/behaviorsense/agents/reasoning/reporter.py", "def repair_claim",
     "format-only claim repair + maxItems bound to max_claims (notebook 04). Without "
     "it the unconstrained arm scores 245 emitted / 0 scorable / nan%, which measures "
     "JSON compliance rather than faithfulness"),
    ("scripts/eval_hallucination.py", "unusable_rate",
     "three-arm hallucination table with a denominator over EMITTED claims; a stale "
     "snapshot silently reports the two-arm nan% version"),
    ("src/behaviorsense/eval/activity_eval.py", "def logit_adjust",
     "notebook 04's P1 cell imports MIN_SUPPORT and scores() from here, so a stale "
     "snapshot fails with ImportError at cell 3; also carries the post-hoc accuracy "
     "levers scripts/rescore_p1.py replays off the saved val logits"),
    ("src/behaviorsense/video.py", "def child_env",
     "notebook 05's /video endpoint decodes uploads in a CHILD process (ffmpeg raises SIGSEGV "
     "on malformed streams and a signal is not catchable, so without the boundary one bad "
     "upload kills the kernel, the tunnel and the demo together). `child_env` is what puts "
     "behaviorsense on that child's PYTHONPATH - sys.path does not cross a process boundary, "
     "and a snapshot without it 422s EVERY upload with \\"No module named 'behaviorsense'\\""),
    ("src/behaviorsense/pipeline.py", "def frames_to_windows", "Agent 1 -> Agent 2 seam"),
    ("configs/taxonomy.yaml",        None,             "class map (notebook 01)"),
]
_stale = []
for _rel, _token, _why in CONTRACT:
    _p = CODE / _rel
    if not _p.is_file():
        _stale.append(f"{_rel} is MISSING ({_why})")
        continue
    if _token and _token not in _p.read_text(encoding="utf-8", errors="ignore"):
        _stale.append(f"{_rel} lacks {_token!r} ({_why})")
if _stale:
    raise AssertionError(
        "The attached code dataset is OLDER than these notebooks:\\n  - "
        + "\\n  - ".join(_stale)
        + f"\\n\\nMounted: {CODE}\\nRe-upload from your checkout, then restart this notebook:"
          "\\n  kaggle datasets version -p . --dir-mode zip -m \\"sync\\""
    )
print(f"  contract {len(CONTRACT)}/{len(CONTRACT)} - mounted code is current")'''


# ONE mount index, shared by every notebook's resolver cell.
#
# This used to be two copies. `RESOLVE` (notebooks 03/04) got the mount-indexed rewrite and
# `RESOLVE_CODE` (notebooks 01/02/05/06/07) kept the `INPUT.glob("**/...")` version, so the
# speed fix landed in two notebooks and missed five - and notebook 07, which calls
# `find_asset` for rtmo-l.onnx, got a cell that had never defined it. That run cost 370 s
# and died with `NameError: name 'find_asset' is not defined`.
#
# A shared prelude is the structural fix: there is now nowhere for the two to diverge.
MOUNT_PRELUDE = """import pathlib
from itertools import islice

INPUT = pathlib.Path("/kaggle/input")
def attached_mounts():
    # Datasets do NOT sit directly under /kaggle/input. They mount at
    # /kaggle/input/datasets/<owner>/<name>/, and competitions at
    # /kaggle/input/competitions/<name>/. Listing INPUT.iterdir() therefore always
    # reports ['competitions', 'datasets'] whatever is attached - which is what the
    # "code: nothing matches ..." failure printed, telling us nothing about whether
    # the code dataset was attached. Descend to the level that names real mounts, and
    # report what each one CONTAINS, since a dataset can be attached and still be
    # missing the directory the notebook needs.
    out = []
    for container in ("datasets", "competitions"):
        base = INPUT / container
        if not base.is_dir():
            continue
        for owner in sorted(base.iterdir()):
            kids = sorted(owner.iterdir()) if owner.is_dir() else []
            if kids and all(k.is_dir() for k in kids[:1]) and container == "datasets":
                for ds in kids:
                    # islice, NOT sorted(...)[:6]. `sorted()` materialises the whole listing
                    # first, and on Kaggle's FUSE mount a slug whose files sit at its root -
                    # `toyota-smarthome-skeleton-v1-2` holds 16,115 - makes that a full
                    # network directory read per mount. Eleven mounts of that shape is most
                    # of the 14 minutes this cell took on the first Toyota run. Six names
                    # are all this diagnostic needs, so stop after six.
                    top = sorted(islice((q.name for q in ds.iterdir()), 6)) \\
                        if ds.is_dir() else []
                    out.append(f"{ds.name} (top level: {top})")
            else:
                out.append(owner.name)
    # Fall back to the flat layout so this keeps working if Kaggle changes the mount
    # shape back, rather than reporting nothing at all.
    return out or sorted(p.name for p in INPUT.iterdir())

ATTACHED = attached_mounts() if INPUT.is_dir() else []

# MOUNT-INDEXED SEARCH, and why two earlier fixes were not enough.
#
# `INPUT.glob("**/x")` walks every directory under /kaggle/input - 93 minutes with the
# Toyota corpus mounted, notebook 06 measured, and 349 s for the single
# `**/src/behaviorsense/__init__.py` probe in notebook 07's second run. The first "fix",
# FIXED-DEPTH globs like `datasets/*/*/*/rtmo-l.onnx`, was depth-bounded but not
# COST-bounded: to match at depth 3 pathlib scandirs EVERY slug child directory, including
# `toyota-smarthome-skeleton-v1-2`'s 16,115-file root and MSMT17's 65,242 crops.
#
# The mounts are KNOWN at depth 2 (`datasets/<owner>/<slug>/`), so enumerate them once and
# resolve everything else with is_file() stats - one metadata call per mount per candidate,
# never a sibling-directory listing.
_SLUGS = (sorted((INPUT / "datasets").glob("*/*"))
          + sorted((INPUT / "competitions").glob("*")))

def find_fast(tail, what, required=True):
    # Known staging prefixes, each costing one stat per mount:
    #   ""                      files at the slug root
    #   EmotionSense-Extended/  the code dataset was created by zipping the repo FOLDER,
    #                           so everything sits one level below the slug
    #   kaggle/working/         Save Version nests the working directory
    # The prefixes apply to MULTI-COMPONENT tails too. They used to be tried only for bare
    # filenames, which quietly sent `src/behaviorsense/__init__.py` - the one probe every
    # notebook makes - down the deep-search path it was written to avoid.
    # `weights/` is additionally tried for a bare filename, the staged weights layout.
    prefixes = ("", "EmotionSense-Extended/", "kaggle/working/")
    mids = ("",) if "/" in tail else ("", "weights/")
    for t in [p + m + tail for p in prefixes for m in mids]:
        hits = [s / t for s in _SLUGS if (s / t).is_file()]
        if hits:
            return hits[0]
    hits = sorted(INPUT.glob(f"**/{tail}"))   # last resort: unusual layout, slow, once
    if hits:
        print(f"  {what:<9} found only by deep search ({tail}) - layout is unusual")
        return hits[0]
    if required:
        raise AssertionError(
            f"{what}: nothing matches {tail!r} in any mount. Attached: {ATTACHED}")
    print(f"  {what:<9} ABSENT (optional)")
    return None

def find_asset(pattern, what, required=True):
    # Callers pass a `**/...` pattern. The leading `**/` is stripped and the mount-indexed
    # search runs first, so every existing call site gets the speed-up unchanged.
    tail = pattern[3:] if pattern.startswith("**/") else pattern
    return find_fast(tail, what, required=required)

def find_dir(subdir, pattern, roots=None):
    # "Which mount holds the most files matching this pattern in this subdirectory?" - one
    # scandir of ONE named directory per mount, never a recursive walk. Used for corpora
    # (Toyota's mp4/ and Videos_mp4/) where the answer is a directory, not a file.
    from fnmatch import fnmatch
    import os
    best, best_n = None, 0
    for root in (roots if roots is not None else _SLUGS):
        base = root / subdir if subdir else root
        if not base.is_dir():
            continue
        n = sum(1 for e in os.scandir(base) if e.is_file() and fnmatch(e.name, pattern))
        if n > best_n:
            best, best_n = base, n
    return best, best_n
"""


RESOLVE_CODE = """# Resolve the repo mount by CONTENT, not by dataset name.
#
# The dataset title is free text and this project has already been uploaded under more
# than one spelling ("behaviorsense-*" and "behavioursense-*"). Hard-coding the name makes
# cell 1 of a 12-hour session fail on a typo, so find the repo by a file only it contains.
""" + MOUNT_PRELUDE + """
SRC     = find_asset("**/src/behaviorsense/__init__.py", "code").parent.parent
CODE    = SRC.parent
SCRIPTS = CODE / "scripts"
CONFIGS = CODE / "configs"
import sys
sys.path.insert(0, str(SRC)); sys.path.insert(0, str(SCRIPTS))
print(f"  code {CODE}")
print(f"  attached {ATTACHED}")""" + REPO_CONTRACT


# Shared by both OFFLINE notebooks. Injected as their first code cell so every later cell
# refers to WHEELS / WEIGHTS / SRC / SCRIPTS instead of a guessed dataset name.
RESOLVE = """# Resolve every attached asset by CONTENT, not by dataset name.
#
# Kaggle mount paths are not predictable from here, and three separate things vary:
#   - notebook 00 emits two folders, which can be published as ONE dataset or two
#     (observed: a single "behavioursense-WW" holding both wheels/ and weights/)
#   - the dataset title is free text, and "behaviour" vs "behavior" both occur
#   - Save Version nests the working directory inside the dataset, so files end up at
#     <mount>/kaggle/working/... rather than <mount>/...
#
# Guessing the name has already cost one session, and notebook 01's carry-forward bug
# showed how the failure presents: a wrong path reads as "nothing attached", the run
# continues, and work is skipped or destroyed rather than failing loudly.
#
# So identify each asset by a file only it has. A directory holding *.whl is the wheel
# cache no matter what the dataset is called.
""" + MOUNT_PRELUDE + """
def find_charades_csv():
    # The charades-480p dataset nests the CSV one level down (its root holds
    # Charades_annotations/ and Charades_v1_480/), and a `**` glob for it walks the code
    # dataset's MSMT17 copy - minutes for one file. Probe the known shapes per mount;
    # only a genuinely unknown layout falls through to the deep search.
    for slug in _SLUGS:
        if "charades" not in slug.name.lower():
            continue
        for base in (slug, slug / "Charades_annotations", slug / "Charades_v1_480",
                     slug / "kaggle" / "working"):
            p = base / "Charades_v1_train.csv"
            if p.is_file():
                return [p]
    return sorted(INPUT.glob("**/Charades_v1_train.csv"))

def find_wheel_dir():
    # "The directory containing *.whl" is not specific enough: /kaggle/input also holds
    # attached COMPETITIONS, and at least one (arc-prize-2026) ships its own wheels. The
    # first sorted hit was that competition's, and the offline install then failed on a
    # cache that simply does not contain torch. Score candidate directories by how many
    # of OUR packages they hold and take the best.
    MARKERS = {"torch", "rtmlib", "onnxruntime-gpu", "nvidia-cudnn-cu12", "triton"}
    dirs = {}
    for slug in _SLUGS:
        for wdir in (slug / "wheels", slug / "kaggle" / "working" / "wheels"):
            if not wdir.is_dir():
                continue
            for w in wdir.glob("*.whl"):
                dirs.setdefault(wdir, set()).add(
                    w.name.split("-")[0].lower().replace("_", "-"))
        if dirs:
            break
    if not dirs:
        for w in INPUT.glob("**/*.whl"):        # last resort
            dirs.setdefault(w.parent, set()).add(
                w.name.split("-")[0].lower().replace("_", "-"))
    if not dirs:
        raise AssertionError(f"no *.whl anywhere under /kaggle/input. Attached: {ATTACHED}")
    best, hits = max(dirs.items(), key=lambda kv: len(kv[1] & MARKERS))
    if not (hits & MARKERS):
        raise AssertionError(
            f"found {len(dirs)} wheel director(ies) but none holds any of {sorted(MARKERS)} "
            f"- the staged cache from notebook 00 is not attached. Candidates: "
            f"{[str(d) for d in dirs]}")
    return best

WHEELS  = find_wheel_dir()
WEIGHTS = find_asset("**/rtmo-l.onnx", "weights").parent
SRC     = find_asset("**/src/behaviorsense/__init__.py", "code").parent.parent
CODE    = SRC.parent
SCRIPTS = CODE / "scripts"

CONFIGS = CODE / "configs"

# sys.path belongs HERE, in the cell that resolves SRC, and unconditionally.
#
# It used to live at the end of the wheel-install cell. Notebook 04 put it outside that
# cell's `if not sm120_ok()` branch and worked; notebook 03 never had it at all and worked
# anyway, because every heavy step there is a subprocess launched with PYTHONPATH set. Then
# `is_real_artifact` was added to notebook 03's shard resolver - the first in-process import
# of `behaviorsense` in that notebook - and the next run died at cell 4 with
# `ModuleNotFoundError: No module named 'behaviorsense'`, six minutes in, one cell after
# PREFLIGHT PASSED. A path set up as a side effect of an unrelated, conditional cell is a
# dependency nobody can see.
import sys
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SRC))

for _label, _path in (("wheels", WHEELS), ("weights", WEIGHTS), ("code", CODE)):
    print(f"  {_label:<8} {_path}")
print(f"  attached  {ATTACHED}")
print("NOTE: if you restart the kernel below, re-run from THIS cell - these names "
      "are what every later cell uses.")""" + REPO_CONTRACT


# ===========================================================================
# 00 - stage assets (ONLINE, CPU)
# ===========================================================================

N00: list[tuple[str, str]] = [
("markdown", """# 00 — Stage offline assets (ONLINE, CPU, internet ON)

Builds the datasets the **offline** Blackwell notebooks depend on. Run once; re-run only
when a dependency version changes.

| attach as input | produces (create dataset from output) |
|---|---|
| `behaviorsense-code` (the repo, uploaded via Kaggle CLI — see docs/07_kaggle_plan.md) | `behaviorsense-wheels`, `behaviorsense-weights` |

Why this notebook exists: the training GPU (RTX PRO 6000 Blackwell, sm_120) runs with
**no internet**, and needs torch >= 2.7 cu128. If the offline image lacks it, there is no
way to fetch it from inside the session — so every wheel and weight is staged here, and
notebook 03 decides at runtime whether to use the staged torch or the preinstalled one."""),
("code", """# torch cu128 — the build carrying sm_120 (Blackwell) kernels. Staged even though the
# offline image may already have it: notebook 03 checks torch.cuda.get_arch_list() first
# and only installs from here if sm_120 is missing.
#
# Everything below is about making pip's resolver deterministic, because it has failed
# here twice in ways that each cost a session:
#   1. Unpinned, pip backtracks across every cu128 torch (2.7 → 2.11) hunting for a
#      satisfiable nvidia-* set, then reports ResolutionImpossible. So: pin it.
#   2. Pinning alone is NOT enough. torch 2.7.1+cu128 requires nvidia-cudnn-cu12==9.7.1.26,
#      which publishes ONLY a manylinux_2_27_x86_64 wheel. A --platform list of
#      {manylinux2014, manylinux_2_28} excludes that tag, the candidate set goes empty,
#      and pip backtracks into the same error with a different cause. Guessing individual
#      tags does not scale — the cu128 index alone publishes manylinux_2_5 / 2_12 / 2_18 /
#      2_25 / 2_27 / 2_28 across torch's dependency set — so PLATFORMS enumerates the whole
#      glibc ladder, OLDEST FIRST. pip prefers earlier entries, so when a package ships
#      several wheels it stages the most portable one, which is the right default for an
#      offline image whose glibc is not knowable from here.
#
# Wheels for BOTH 3.11 and 3.12: the offline image's interpreter is not knowable from
# here either, and missing it by one minor version costs a full 12-hour session. The
# nvidia-* dependencies are py3-none wheels, so they are shared, not duplicated.
import subprocess, pathlib
WHEELS = pathlib.Path("/kaggle/working/wheels"); WHEELS.mkdir(parents=True, exist_ok=True)

TORCH_PIN = "torch==2.7.1"          # satisfies the >=2.7 floor in requirements-train.txt
PLATFORMS = (["manylinux1_x86_64", "manylinux2010_x86_64", "manylinux2014_x86_64"]
             + [f"manylinux_2_{m}_x86_64" for m in range(5, 40)]
             + ["linux_x86_64"])
PLAT_ARGS = [a for p in PLATFORMS for a in ("--platform", p)]

for pyver in ("3.11", "3.12"):
    subprocess.run(
        ["pip", "download", TORCH_PIN, "-d", str(WHEELS),
         "--index-url", "https://download.pytorch.org/whl/cu128",
         "--python-version", pyver, "--only-binary=:all:", *PLAT_ARGS],
        check=True)
print(len(list(WHEELS.glob("*.whl"))), "wheels so far")"""),
("code", """# CPU-side packages. `accelerate` (and some outlines builds) declare a torch dependency,
# so an unconstrained download here quietly fetches the *PyPI* torch as well — verified:
# an unconstrained resolve of this list picks torch 2.13.0, another ~1 GB per python
# version, built without sm_120 and therefore useless on Blackwell, plus a second set of
# nvidia-* pins for the resolver to fight with.
#
# The constraints file pins torch to the version staged in the previous cell, and
# --find-links lets pip satisfy that from the local cu128 wheel instead of the network.
# (PEP 440: a specifier carrying no local label ignores local labels when matching, so
# `torch==2.7.1` does match `2.7.1+cu128`, and the local build sorts higher than PyPI's.
# Confirmed by dry-run: with the constraint the resolve picks 2.7.1+cu128, without it 2.13.0.)
#
# The floors below are the ones the code actually needs, not decoration: reporter.py calls
# outlines.from_transformers and outlines.types.json_schema, which exist only in outlines
# 1.x, so a 0.x resolve would import-error inside the offline session.
import subprocess, pathlib
WHEELS = pathlib.Path("/kaggle/working/wheels")
TORCH_PIN = "torch==2.7.1"
PLATFORMS = (["manylinux1_x86_64", "manylinux2010_x86_64", "manylinux2014_x86_64"]
             + [f"manylinux_2_{m}_x86_64" for m in range(5, 40)]
             + ["linux_x86_64"])
PLAT_ARGS = [a for p in PLATFORMS for a in ("--platform", p)]

CONSTRAINTS = pathlib.Path("/tmp/constraints.txt")
CONSTRAINTS.write_text("\\n".join([TORCH_PIN, "outlines>=1.0", "transformers>=4.44"]) + "\\n")
CPU_PKGS = ["numpy", "pydantic", "PyYAML", "Pillow",
            "transformers", "accelerate", "outlines", "safetensors"]
for pyver in ("3.11", "3.12"):
    subprocess.run(
        ["pip", "download", *CPU_PKGS, "-d", str(WHEELS),
         "--constraint", str(CONSTRAINTS), "--find-links", str(WHEELS),
         "--python-version", pyver, "--only-binary=:all:", *PLAT_ARGS],
        check=True)
total = sum(f.stat().st_size for f in WHEELS.glob("*"))
print(f"{len(list(WHEELS.glob('*.whl')))} wheels, {total/1e9:.1f} GB")"""),
("code", """# Pose-extraction packages, so shard extraction can also run OFFLINE on the Blackwell.
#
# Why this cell exists: extraction was originally online-only, purely because it downloads
# Charades. That forced ~20-30 h of RTMO inference onto a T4 while the fast card sat idle,
# and re-downloaded 13 GB at the start of each of the 3-4 sessions with the GPU doing
# nothing. Notebook 01a stages the archive once; these wheels are what let notebook 01
# then run with the internet off.
#
# onnxruntime-gpu, not onnxruntime: the CPU build silently ignores device="cuda" and runs
# RTMO on four vCPUs, which turns a 12 h session into something like a 200 h one. The
# offline notebook asserts CUDAExecutionProvider is actually present rather than trusting
# the install.
#
# The CUDA 12 build links cuDNN 9, which is already staged as a torch dependency, so no
# extra nvidia-* pins are needed here.
EXTRACT_PKGS = ["rtmlib", "onnxruntime-gpu==1.26.0", "opencv-python-headless"]
for pyver in ("3.11", "3.12"):
    subprocess.run(
        ["pip", "download", *EXTRACT_PKGS, "-d", str(WHEELS),
         "--constraint", str(CONSTRAINTS), "--find-links", str(WHEELS),
         "--python-version", pyver, "--only-binary=:all:", *PLAT_ARGS],
        check=True)
have = {f.name.split("-")[0].lower().replace("_", "-") for f in WHEELS.glob("*.whl")}
for pkg in ("onnxruntime-gpu", "opencv-python-headless"):
    assert pkg in have, (
        f"{pkg} did not stage. Offline extraction (notebook 01) cannot run without it; "
        f"staged prefixes: {sorted(have)}")
total = sum(f.stat().st_size for f in WHEELS.glob("*"))
print(f"{len(list(WHEELS.glob('*.whl')))} wheels, {total/1e9:.1f} GB "
      f"(extraction packages included)")"""),
("code", """# Fail HERE, online, if the staged set cannot satisfy an offline install — not at hour
# three of a GPU session. Checks the things that silently go missing: a torch wheel tagged
# for each interpreter, cudnn (without it torch imports fine and then dies on the first
# conv layer), and an outlines new enough for the API reporter.py actually calls.
#
# Also writes WHEELS_LOCK.txt. The staged wheels ARE the offline environment, and the CPU
# packages are floor-pinned rather than exact-pinned (numpy resolves differently for 3.11
# and 3.12, so one exact list cannot serve both). That means re-running this notebook in
# six weeks stages a different set. The lock file is what makes a training run reproducible
# after the fact, and what the dissertation cites as the environment.
import pathlib, re
WHEELS = pathlib.Path("/kaggle/working/wheels")
names = sorted(p.name for p in WHEELS.glob("*.whl"))
for pyver, cp in (("3.11", "cp311"), ("3.12", "cp312")):
    hits = [n for n in names if n.startswith("torch-") and cp in n]
    assert hits, f"no torch wheel tagged {cp} — offline python {pyver} would have nothing"
    print(f"  py{pyver}: {hits[0]}")
cudnn = [n for n in names if n.startswith("nvidia_cudnn_cu12-")]
assert cudnn, "nvidia-cudnn-cu12 missing — torch would import, then fail on conv layers"
print("  cudnn:", *cudnn)

out = [n for n in names if n.startswith("outlines-")]
assert out, "outlines missing — Agent 4 cannot do grammar-constrained decoding"
major = int(out[0].split("-")[1].split(".")[0])
assert major >= 1, (f"staged {out[0]}: reporter.py calls outlines.from_transformers, "
                    "a 1.x-only API, so this would ImportError offline")
print(f"  outlines: {out[0]} (major {major} >= 1, has from_transformers)")

lock = sorted({f"{m.group(1)}=={m.group(2)}" for n in names
               if (m := re.match(r"(.+?)-([0-9][^-]*)-", n))})
(WHEELS / "WHEELS_LOCK.txt").write_text("\\n".join(lock) + "\\n")
print(f"  wrote WHEELS_LOCK.txt ({len(lock)} distinct package versions)")

pypi_torch = [n for n in names if n.startswith("torch-") and "+cu128" not in n
              and "%2Bcu128" not in n]
if pypi_torch:
    print("  NOTE: non-cu128 torch also present (wastes space; check the constraint):",
          *pypi_torch)"""),
("code", """# Weights: copy the two OSNet checkpoints out of the repo dataset and PROVE they arrived.
# A Git-LFS pointer is a 130-byte text file that fails at hour three of an offline
# session; this cell is where that mistake gets caught instead.
#
# The loop is deliberately NOT the whole check. On the first real run this cell printed
# NOTHING and the notebook carried on: the uploaded dataset had no weights/ directory, so
# glob() matched nothing, the body never executed, and a missing-asset bug rendered as
# silence. An empty iteration is not a pass. So: name the files expected, diff against
# what actually landed, and report the shortfall.
#
# Not fatal by itself — notebooks 03/04 never load these (ReID tau=0.3546 is already
# fitted and committed in results/reid_eval.md), so this raises only if the directory is
# there but unusable, and otherwise prints a WARNING loud enough to act on.
import shutil, pathlib
_init = sorted(pathlib.Path("/kaggle/input").glob("**/src/behaviorsense/__init__.py"))
CODE = _init[0].parent.parent.parent if _init else pathlib.Path("/kaggle/input/__missing__")
W = pathlib.Path("/kaggle/working/weights"); W.mkdir(parents=True, exist_ok=True)
EXPECTED = {"osnet_ain_x1_0_msmt17.pth", "osnet_ain_x1_0_imagenet.pth"}

src_dir = CODE / "weights"
found = sorted(src_dir.glob("*.pth")) if src_dir.is_dir() else []
for f in found:
    shutil.copy(f, W / f.name)
    mb = (W / f.name).stat().st_size / 1e6
    assert mb > 1, f"{f.name} is {mb:.2f} MB - a pointer file, not weights"
    print(f"  {f.name}: {mb:.1f} MB")

missing = EXPECTED - {f.name for f in found}
if missing:
    print(f"  WARNING: {len(missing)} OSNet checkpoint(s) absent from the dataset: "
          f"{sorted(missing)}")
    print(f"           looked in {src_dir} (exists: {src_dir.is_dir()})")
    print("           Not blocking: notebooks 03/04 do not load these, and the ReID")
    print("           numbers are already measured (results/reid_eval.md). Re-upload")
    print("           behaviorsense-code with weights/ included if you want them staged.")
else:
    print(f"  all {len(EXPECTED)} OSNet checkpoints staged")"""),
("code", """# RTMO-L ONNX: used ONLINE by notebooks 01/02 for pose extraction, and checked by the
# offline preflight. Primary source is the openmmlab CDN; fallback is rtmlib's cache.
import urllib.request, zipfile, pathlib, shutil
W = pathlib.Path("/kaggle/working/weights")
URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
       "rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip")
try:
    dst = pathlib.Path("/tmp/rtmo.zip")
    urllib.request.urlretrieve(URL, dst)
    with zipfile.ZipFile(dst) as z:
        onnx = [n for n in z.namelist() if n.endswith(".onnx")]
        z.extract(onnx[0], "/tmp/rtmo")
    shutil.copy(pathlib.Path("/tmp/rtmo") / onnx[0], W / "rtmo-l.onnx")
    print("rtmo-l.onnx:", (W / "rtmo-l.onnx").stat().st_size / 1e6, "MB")
except Exception as exc:
    print("CDN failed:", exc)
    import subprocess
    subprocess.run(["pip", "install", "-q", "rtmlib", "onnxruntime"], check=True)
    from rtmlib import RTMO   # first construction downloads the onnx to ~/.cache
    RTMO(mode="performance", backend="onnxruntime", device="cpu")
    cached = list(pathlib.Path.home().glob(".cache/**/rtmo*[!.zip]"))
    onnx = [p for p in cached if p.suffix == ".onnx"]
    assert onnx, f"no cached onnx found in {cached}"
    shutil.copy(onnx[0], W / "rtmo-l.onnx")
    print("rtmo-l.onnx staged via rtmlib cache")"""),
("code", """# OPTIONAL extras — failures here do not block training (preflight marks both optional):
#   rtdetr-l.onnx        Agent 1 object context (serving-time, not training-time)
#   stgcnpp_ntu60_*.pth  init stretch goal only: our self-contained STGCNpp trains from
#                        scratch, and loading PYSKL checkpoints would need a key-mapping
#                        shim that does not exist yet. Stated honestly rather than staged
#                        as if it were consumed.
import shutil
try:
    import subprocess
    subprocess.run(["pip", "install", "-q", "ultralytics"], check=True)
    from ultralytics import RTDETR
    RTDETR("rtdetr-l.pt").export(format="onnx", imgsz=640)
    shutil.copy("rtdetr-l.onnx", "/kaggle/working/weights/rtdetr-l.onnx")
    print("rtdetr-l.onnx staged")
except Exception as exc:
    print(f"RT-DETR skipped (optional): {exc}")"""),
("code", """# Manifest + inventory. Notebook 03's preflight re-verifies this offline.
import pathlib, json
W = pathlib.Path("/kaggle/working/weights")
manifest = {f.name: f.stat().st_size for f in sorted(W.glob("*")) if f.name != "MANIFEST.json"}
(W / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
for name, size in manifest.items():
    print(f"  {name:<40} {size/1e6:>9.1f} MB")
print()
# rtmo-l.onnx is the one asset in this notebook that notebooks 01/02 cannot proceed
# without. An empty or pose-less manifest must not be followed by a cheerful
# "Save Version" — that is how an unusable dataset gets published as if it were fine.
assert manifest, "no weights staged at all - do not Save Version, nothing would be in it"
assert "rtmo-l.onnx" in manifest, (
    f"rtmo-l.onnx missing - notebooks 01/02 cannot extract pose without it. "
    f"Staged: {sorted(manifest)}")
# The staged onnxruntime-gpu decides whether extraction runs on the GPU at all. 1.27+ is
# built against CUDA 13 and silently falls back to CPU on Kaggle's CUDA 12 image, so the
# version is asserted HERE, where it can still be fixed, rather than discovered offline.
ort_whl = sorted(pathlib.Path("/kaggle/working/wheels").glob("onnxruntime_gpu-*.whl"))
assert ort_whl, "onnxruntime-gpu did not stage - notebook 01 cannot use the GPU"
for w in ort_whl:
    ver = w.name.split("-")[1]
    major, minor = (int(x) for x in ver.split(".")[:2])
    assert (major, minor) <= (1, 26), (
        f"staged {w.name}: onnxruntime-gpu >= 1.27 is a CUDA 13 build. Kaggle runs CUDA 12, "
        "so the provider fails to load (libcublasLt.so.13) and RTMO extracts on CPU at "
        "~1/50th speed. Pin onnxruntime-gpu==1.26.0 in the extraction cell.")
    print(f"  {w.name}  (CUDA 12 build, correct)")

print()
print("Save Version -> UPDATE your existing wheels+weights dataset (e.g. behavioursense-WW).")
print("Both folders can live in ONE dataset; the notebooks resolve them by content,")
print("so the dataset name does not matter. Do not create a second copy - it costs quota")
print("and the resolver would then have two candidates to choose between.")"""),
]

# ===========================================================================
# 01 - Charades ADL shards (ONLINE, GPU)
# ===========================================================================

N01: list[tuple[str, str]] = [
("markdown", """# 01 — Charades → ADL skeleton shards (**OFFLINE**, RTX PRO 6000 Blackwell, NO INTERNET)

Runs RTMO over the staged Charades archive at 15 fps (the deployment rate: extracting at
a different rate than serving is a train/serve skew), maps the 157 classes through the
reviewed taxonomy map, windows into 2 s / stride 1 s, and writes fp16 `.npz` shards in
the layout `SkeletonWindowDataset` loads. The download happened once in notebook 01a;
nothing here touches the network, so the whole 12 h goes to inference on the Blackwell.

| attach as input | produces |
|---|---|
| `behaviorsense-code` | `behaviorsense-adl-shards` |
| the wheels dataset (e.g. `behavioursense-WW`) | (updated version of the same) |
| `charades-480p` (from notebook 01a) | |
| `behaviorsense-adl-shards` (**continuation runs only**) | |

**One session DOES finish this — measured, on the Blackwell with CUDA confirmed.** The
extraction loop reported **1333 videos/h**, and this notebook reads `Charades_v1_train.csv`
(**7,985** rows — the 9,848 figure is train + test, and only train is labelled), so a full
pass is **~6 h** against the 12 h limit. Output is ~232k windows, ~0.8 GB compressed,
nowhere near the 20 GB `/kaggle/working` cap.

So set `VID_END` past the end of the split (`99999` — Python slicing clamps) and do it in
one run. The earlier `3000` default predates the measurement and cost nothing but three
extra sessions' worth of setup.

The slicing machinery stays, because it is what makes a *timeout* survivable rather than
what makes the job fit: shards flush every 400 MB (~1,100 videos), so an interrupted
session keeps everything already written, and the next run resumes at
`VID_START=<where you stopped>` with the carry-forward cell bringing prior shards forward.
Kaggle dataset versions REPLACE content — that cell is what makes accumulation work.

⚠️ **Quick Save, not "Save & Run All".** Save & Run All re-executes from scratch and would
spend another 6 h re-extracting. Quick Save snapshots `/kaggle/working` as it stands."""),
("code", """# Whole train split in one session: 7,985 rows / 1333 videos-per-h ~= 6 h < 12 h.
# Slicing clamps, so 99999 means "to the end" and needs no exact row count here.
# Set VID_START to where a timed-out session stopped; shards already flushed are kept.
VID_START, VID_END = 0, 99999
FPS_SAMPLE = 15
import subprocess, sys, pathlib

# OFFLINE install from the staged wheels. --no-index is what makes "offline" true rather
# than aspirational: without it pip reaches for PyPI, hangs on a dead socket, and the
# failure reads as a package problem instead of a network one.
INPUT = pathlib.Path("/kaggle/input")

# Pick the wheel cache by CONTENT. /kaggle/input also holds attached competitions, and at
# least one (arc-prize-2026) ships .whl files - "first *.whl found" selected that one.
_MARKERS = {"torch", "rtmlib", "onnxruntime-gpu", "nvidia-cudnn-cu12", "triton"}
_dirs = {}
for _w in INPUT.glob("**/*.whl"):
    _dirs.setdefault(_w.parent, set()).add(_w.name.split("-")[0].lower().replace("_", "-"))
assert _dirs, f"no wheels attached. Mounted: {sorted(p.name for p in INPUT.iterdir())}"
WHEEL_DIR, _hits = max(_dirs.items(), key=lambda kv: len(kv[1] & _MARKERS))
assert _hits & _MARKERS, (
    f"no staged wheel cache found - candidates hold none of {sorted(_MARKERS)}: "
    f"{[str(d) for d in _dirs]}")

# rtmlib DEPENDS ON onnxruntime (the CPU build). Installing it normally therefore drags
# the CPU package in, and pip installs it AFTER onnxruntime-gpu - both own the same
# `onnxruntime` module directory, so the CPU build overwrites the GPU one's provider
# registration. Observed exactly this: "Successfully installed onnxruntime-1.28.0
# onnxruntime-gpu-1.28.0 rtmlib-0.0.16", then providers = [Azure, CPU] and no CUDA.
#
# So: remove any CPU build, install rtmlib WITHOUT its deps (numpy/opencv/tqdm are all in
# the Kaggle image already), and install onnxruntime-gpu LAST so nothing can clobber it.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                "--find-links", str(WHEEL_DIR), "rtmlib"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                "--find-links", str(WHEEL_DIR),
                "onnxruntime-gpu", "opencv-python-headless", "PyYAML"], check=True)
print(f"installed from {WHEEL_DIR}")"""),
("code", RESOLVE_CODE),
("code", """# Prove the GPU is actually in use BEFORE spending the session.
#
# onnxruntime falls back to CPU silently: device="cuda" against a CPU-only build, or a CUDA
# build whose provider .so fails to load, both yield correct poses at a fraction of the
# speed. That is a 12-hour session extracting a fraction of one slice, and it presents as
# "the GPU is slow" rather than "the GPU is not being used".
import onnxruntime as ort
import importlib.metadata as _md

providers = ort.get_available_providers()
print("onnxruntime providers:", providers)
_installed = {d.metadata["Name"].lower() for d in _md.distributions()
              if d.metadata.get("Name")}
_both = {"onnxruntime", "onnxruntime-gpu"} & _installed
_hint = ("BOTH onnxruntime builds are installed: they share the `onnxruntime` module "
         "directory, so whichever pip wrote last wins. Re-run the setup cell - it "
         "uninstalls the CPU build and installs rtmlib with --no-deps."
         if len(_both) == 2 else
         "Check that notebook 00 staged onnxruntime-gpu (not onnxruntime).")
assert "CUDAExecutionProvider" in providers, (
    f"onnxruntime has no CUDA provider - extraction would run on CPU. "
    f"available={providers}, packages={sorted(_both) or 'none'}. {_hint}")
print(f"CUDA provider compiled in (packages: {sorted(_both)})")"""),
("code", """# Build the pose model from the STAGED onnx, and prove it works before the loop.
#
# This notebook has no internet, so rtmlib must not be left to fetch anything: the
# constructor is given the explicit path to the rtmo-l.onnx that notebook 00 staged.
# Passing mode= instead would have rtmlib resolve a URL, which offline either hangs or
# raises - and notebook 00 asserts this file exists precisely because 01/02 need it, so
# leaving it unused was a staged asset nothing consumed.
#
# The probe below runs ONE synthetic frame. A five-second failure here is the whole point:
# a wrong kwarg or a bad onnx otherwise surfaces at the first real video, after the unzip.
import numpy as np
from rtmlib import RTMO

_onnx = sorted(pathlib.Path("/kaggle/input").glob("**/rtmo-l.onnx"))
assert _onnx, (
    "rtmo-l.onnx not found under /kaggle/input. Attach the dataset notebook 00 produced "
    f"(wheels+weights, e.g. behavioursense-WW). Mounted: {ATTACHED}")
RTMO_ONNX = _onnx[0]
print(f"  rtmo    {RTMO_ONNX}  ({RTMO_ONNX.stat().st_size/1e6:.0f} MB)")

body = RTMO(onnx_model=str(RTMO_ONNX), model_input_size=(640, 640),
            backend="onnxruntime", device="cuda")
_k, _s = body(np.zeros((480, 640, 3), np.uint8))
print(f"  probe OK - keypoints {np.asarray(_k).shape}, scores {np.asarray(_s).shape}")

# get_available_providers() lists what the build was COMPILED with, not what LOADED.
# Observed: onnxruntime-gpu 1.28 (a CUDA 13 build) on Kaggle's CUDA 12 image lists
# CUDAExecutionProvider, then logs "Failed to load library ... libcublasLt.so.13" and
# runs the model on CPU anyway. The cell above passes; the session is ~50x too slow.
# The session object is the only honest source: it reports the providers in EFFECT.
_sess = getattr(body, "session", None)
_active = list(_sess.get_providers()) if _sess is not None else []
assert "CUDAExecutionProvider" in _active, (
    f"RTMO is running on {_active or 'an unknown provider'} - the CUDA provider was "
    "listed but did NOT load. Scroll up for a 'Failed to load library' line naming the "
    "missing .so: libcublasLt.so.13 means an onnxruntime-gpu built for CUDA 13 on this "
    "CUDA 12 image. Notebook 00 pins onnxruntime-gpu==1.26.0 (the newest CUDA 12.8 "
    "build) - if this cache predates that pin, re-run notebook 00.")
print(f"  provider IN USE: {_active[0]}")"""),
("code", """# Locate this session's slice of Charades. TWO layouts are possible and which one you
# get is Kaggle's choice, not yours:
#
#   PRE-EXTRACTED  Kaggle auto-unpacks archives when building a dataset from notebook
#                  output, so 01a's Charades_v1_480.zip arrives as 9,848 loose mp4s under
#                  <mount>/Charades_v1_480/Charades_v1_480/. This is the better case:
#                  read the files in place, no unpack, nothing in the 20 GB working quota.
#   ZIPS           the archives survived as .zip (direct upload, or auto-extract off).
#                  Unpacking all of them costs ~13 GB and minutes of every session, so the
#                  annotation CSV fixes the video order and only this slice is extracted.
#
# Assuming zips cost a session: the assert read "charades-480p is not attached" while it
# was plainly attached and extracted. So detect, and say which layout was found.
import zipfile, pathlib, csv, time

INPUT = pathlib.Path("/kaggle/input")
ATTACHED_NOW = sorted(p.name for p in INPUT.iterdir()) if INPUT.is_dir() else []
_ann_csv = sorted(INPUT.glob("**/Charades_v1_train.csv"))
_mp4     = sorted(INPUT.glob("**/Charades_v1_480/**/*.mp4"))[:1]
_zips    = sorted(INPUT.glob("**/Charades*.zip"))
assert _ann_csv or _zips, (
    "Charades is not attached in either layout: no **/Charades_v1_train.csv (extracted) "
    f"and no **/Charades*.zip (archived) under /kaggle/input. Attached: {ATTACHED_NOW}. "
    "Run notebook 01a and attach its output dataset.")

TMP = pathlib.Path("/tmp/charades"); TMP.mkdir(parents=True, exist_ok=True)
if _ann_csv:
    ann_csv = _ann_csv[0]
    print(f"layout PRE-EXTRACTED (Kaggle unpacked the archives)")
else:
    with zipfile.ZipFile(next(p for p in _zips if "annotation" in p.name.lower())) as z:
        z.extractall(TMP / "annotations")
    ann_csv = next((TMP / "annotations").rglob("Charades_v1_train.csv"))
    print(f"layout ZIPS ({len(_zips)} archives)")

rows = list(csv.DictReader(open(ann_csv, encoding="utf-8")))[VID_START:VID_END]
wanted = {r["id"] for r in rows}
assert wanted, (f"slice {VID_START}..{VID_END} selected 0 videos from {ann_csv.name} "
                f"({len(list(csv.DictReader(open(ann_csv, encoding='utf-8'))))} rows total)")
print(f"  slice {VID_START}..{VID_START + len(rows)} -> {len(wanted)} video ids")

t0 = time.time()
if _mp4:
    # Read in place. VIDEO_DIRS is what the extraction cell resolves paths against.
    VIDEO_DIRS = sorted({p.parent for p in INPUT.glob("**/Charades_v1_480/**/*.mp4")})
    n_vid = sum(len(list(d.glob("*.mp4"))) for d in VIDEO_DIRS)
    print(f"  {n_vid} mp4s readable in place across {len(VIDEO_DIRS)} dir(s) - no unpack")
else:
    VIDEOS = TMP / "videos"; VIDEOS.mkdir(exist_ok=True)
    with zipfile.ZipFile(next(p for p in _zips if "480" in p.name)) as z:
        members = [n for n in z.namelist()
                   if n.endswith(".mp4") and pathlib.Path(n).stem in wanted]
        for i, m in enumerate(members):
            target = VIDEOS / pathlib.Path(m).name
            if not target.exists():
                with z.open(m) as src, open(target, "wb") as dst:
                    dst.write(src.read())
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(members)} extracted")
    VIDEO_DIRS = [VIDEOS]
    n_vid = len(list(VIDEOS.glob("*.mp4")))
    print(f"  {n_vid} videos unpacked in {(time.time()-t0)/60:.1f} min")
assert n_vid, f"0 videos found for this slice - layout unexpected. Attached: {ATTACHED_NOW}"

# One resolver for both layouts, so the extraction cell needs no branch of its own.
def video_path(vid):
    for d in VIDEO_DIRS:
        p = d / f"{vid}.mp4"
        if p.is_file():
            return p
    return None
_hit = sum(video_path(v) is not None for v in wanted)
assert _hit, (f"none of the {len(wanted)} slice ids resolved to a file under "
              f"{[str(d) for d in VIDEO_DIRS]} - id/filename mismatch")
print(f"  {_hit}/{len(wanted)} slice ids resolve to a readable mp4")"""),
("code", """# Build + REVIEW the class map. The fallback list printed here is the eyeball check the
# taxonomy demands — label noise from silent auto-mapping is unrecoverable after training.
import subprocess, glob
# Both layouts: the extracted dataset has it under the mount, the zip layout under /tmp.
_cls = (sorted(pathlib.Path("/kaggle/input").glob("**/Charades_v1_classes.txt"))
        or sorted(pathlib.Path("/tmp/charades").rglob("Charades_v1_classes.txt")))
assert _cls, "Charades_v1_classes.txt not found in either layout"
classes_txt = str(_cls[0])
subprocess.run(
    ["python", str(SCRIPTS / "build_charades_map.py"),
     "--classes", classes_txt,
     "--out", "/kaggle/working/charades_map.yaml",
     "--review-tsv", "/kaggle/working/charades_map_review.tsv"],
    check=True)"""),
("code", """# Carry forward shards from the previous dataset version (continuation runs).
#
# This cell is where a 12-hour session gets destroyed if it is sloppy. Kaggle dataset
# versions REPLACE content — they do not merge. So if a previous version exists and this
# cell fails to copy it forward, Save Version publishes only the current slice and the
# earlier extraction is gone. "No shards found" therefore has two very different meanings:
#   - dataset not attached at all      -> genuinely the first run, proceed
#   - dataset attached but nothing in it -> wrong path or wrong version. REFUSE. Printing
#     "fresh start" here and carrying on is how prior work gets overwritten.
import shutil, pathlib
OUT = pathlib.Path("/kaggle/working/shards"); OUT.mkdir(exist_ok=True)

# Find the shards mount by name, tolerating the behaviour/behavior spelling this project
# has already been uploaded under. A hard-coded name that misses is indistinguishable from
# "not attached", which is precisely the branch that destroys prior work.
MOUNTS = [m for m in sorted(pathlib.Path("/kaggle/input").iterdir())
          if "adl" in m.name.lower().replace("behaviour", "behavior")] \
         if pathlib.Path("/kaggle/input").is_dir() else []

if not MOUNTS:
    print("fresh start - no ADL shards dataset attached")
else:
    # Attached. Find the shards wherever they sit, rather than assuming one layout.
    found = sorted(p for m in MOUNTS for p in m.rglob("*.npz"))
    assert found, (
        f"an ADL shards dataset IS attached ({[m.name for m in MOUNTS]}) but contains no "
        f".npz. Saving now would REPLACE the dataset with only this session's slice and "
        f"destroy the previous extraction.")
    for f in found:
        shutil.copy(f, OUT / f.name)
    print(f"carried forward {len(found)} shard(s) from {found[0].parent}")"""),
("code", """# Pose extraction. Top-2 people by box area per frame -> [T, 2, 17, 3] float16.
# Window labels: the Charades action interval covering >= 60% of the window, mapped
# through the reviewed YAML; unmapped -> other_idle; drop:true classes excluded.
# Subject ids: Charades publishes no worker ids, so the VIDEO id is the subject proxy
# and the P1 split is by video — stated in the eval tables, not hidden.
import csv, glob, time, yaml, pathlib
import numpy as np
import cv2
from prepare_skeletons import assign_slots, window_clip   # tested repo code, not copies

mapping = yaml.safe_load(open("/kaggle/working/charades_map.yaml"))["charades"]
tax = yaml.safe_load(open(CONFIGS / "taxonomy.yaml"))
name_to_id = {c["name"]: c["id"] for c in tax["classes"]}

# rows and video_path() were resolved by the layout cell above - do not re-derive them
# here, or this cell silently disagrees with the one that verified the slice exists.
print(f"videos {VID_START}..{VID_START + len(rows)} ({len(VIDEO_DIRS)} source dir(s))")

skels, labels, subjects, sources = [], [], [], []
shard_idx = len(list(OUT.glob("*.npz")))
t0 = time.time()

def flush():
    global skels, labels, subjects, sources, shard_idx
    if not skels:
        return
    np.savez_compressed(
        OUT / f"charades_{shard_idx:04d}.npz",
        skeletons=np.stack(skels).astype(np.float16),
        labels=np.asarray(labels, dtype=np.int64),
        subjects=np.asarray(subjects, dtype="<U32"),
        datasets=np.asarray(sources, dtype="<U32"))
    print(f"  shard {shard_idx}: {len(skels)} windows")
    shard_idx += 1
    skels, labels, subjects, sources = [], [], [], []

for n, row in enumerate(rows):
    vid = row["id"]
    _p = video_path(vid)
    if _p is None:
        continue
    path = str(_p)
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24
    step = max(1, round(src_fps / FPS_SAMPLE))
    poses, f = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f % step == 0:
            kpts, scores = body(frame)
            # Slot assignment tracks the previous frame (assign_slots): area order alone
            # swaps the two people whenever their apparent sizes cross, splicing both
            # trajectories and teleporting the motion streams at the swap frame.
            poses.append(assign_slots(kpts, scores, poses[-1] if poses else None))
        f += 1
    cap.release()
    if len(poses) < 30:
        continue
    poses = np.stack(poses)

    acts = []
    for a in (row["actions"] or "").split(";"):
        if a.strip():
            cid, s, e = a.split()
            acts.append((cid, float(s), float(e)))
    for w_i, w in enumerate(window_clip(poses)):
        ws, we = w_i * 1.0, w_i * 1.0 + 2.0
        # Best-covering mapped action wins the window. The first version broke on the
        # FIRST action with >= 60% overlap in CSV order - Charades actions overlap
        # heavily, so a barely-qualifying interval listed first beat a fully-covering
        # one, and an UNMAPPED first qualifier forced other_idle even when a mapped
        # action also covered the window. Coverage decides now; unmapped actions cannot
        # claim a window at all.
        lab, best_ov = "other_idle", 0.0
        for cid, s, e in acts:
            ov = min(we, e) - max(ws, s)
            if ov < 1.2 or ov <= best_ov:                 # >= 60% of the 2 s window
                continue
            slugs = [k for k in mapping if k.startswith(cid + "_")]
            m = mapping.get(slugs[0]) if slugs else None
            if m is None:
                continue
            best_ov, lab = ov, (None if isinstance(m, dict) else m)  # dict == drop:true
        if lab is None:
            continue
        skels.append(w)
        labels.append(name_to_id[lab])
        subjects.append(vid)
        sources.append("charades")
    if sum(s.nbytes for s in skels) > 400e6:
        flush()
    if n % 100 == 0 and n:
        rate = n / max(time.time() - t0, 1)
        eta_h = (len(rows) - n) / max(rate, 1e-9) / 3600
        print(f"{n}/{len(rows)} videos ({rate * 3600:.0f}/h, ETA {eta_h:.1f} h)")
flush()"""),
("code", """# Quality gates — printed, not assumed. A class distribution that surprises you here is
# cheaper than one that surprises you after 80 epochs.
import numpy as np, pathlib, collections, csv
OUT = pathlib.Path("/kaggle/working/shards")
counts, n_windows = collections.Counter(), 0
for f in OUT.glob("*.npz"):
    z = np.load(f)
    n_windows += len(z["labels"])
    counts.update(z["labels"].tolist())
print(f"{n_windows} windows across {len(list(OUT.glob('*.npz')))} shards")
for lab, n in counts.most_common():
    print(f"  class {lab:>2}: {n:>7}  ({n / max(n_windows, 1):.1%})")
print()
# Never invite a Save Version that would publish an empty dataset over a good one.
assert n_windows > 0, ("0 windows extracted - do NOT Save Version, it would replace "
                       "behaviorsense-adl-shards with nothing. Check VID_START/VID_END "
                       "and that the Charades videos actually downloaded.")
print("Save Version (QUICK SAVE - not 'Save & Run All', which re-extracts from scratch)")
print("  -> update dataset behaviorsense-adl-shards from /kaggle/working.")
# Report completion against the SPLIT, not against VID_END. With VID_END=99999 (meaning
# "to the end") the old hint said "next session: VID_START=99999", which is both wrong
# and alarming. What matters is whether every row of the split was actually consumed.
_done = VID_START + len(rows)
_total = len(list(csv.DictReader(open(ann_csv, encoding="utf-8"))))
if _done >= _total:
    print(f"COMPLETE: {_done}/{_total} videos of the train split extracted. "
          "No continuation session needed - go to notebook 02.")
else:
    print(f"PARTIAL: {_done}/{_total} videos. Next session set VID_START={_done} "
          "and attach this output dataset as an input so its shards carry forward.")"""),
]

# ===========================================================================
# 02 - fall shards (ONLINE, GPU)
# ===========================================================================

N02: list[tuple[str, str]] = [
("markdown", """# 02 — Fall datasets → skeleton shards (ONLINE, GPU, internet ON)

Same extraction recipe as notebook 01, over the fall corpora: **Le2i, CAUCAFall,
GMDCSA-24, URFD**. A few GB total — one session.

| attach as input | produces |
|---|---|
| `behaviorsense-code` | `behaviorsense-fall-shards` |
| your uploaded copies of Le2i / CAUCAFall / URFD (mirror fallback) | |

Academic fall-dataset mirrors are famously flaky, so each source is fetched in a
try/except and the honest fallback is attaching a copy you uploaded as a private dataset.
The `datasets` column records the source per window — that column is what makes the P2
leave-one-dataset-out protocol possible later, so it is not optional metadata."""),
("code", """import subprocess, sys
# onnxruntime-gpu 1.27+ is built against CUDA 13; Kaggle ships CUDA 12, so the CUDA
# provider fails to load (libcublasLt.so.13 missing) and RTMO silently runs on CPU.
# 1.26.x is the newest CUDA-12.8 build - the same toolkit torch 2.7.1+cu128 uses.
# --no-deps on rtmlib: it depends on the CPU `onnxruntime`, which would overwrite the
# GPU package's provider registration.
subprocess.run(["pip", "uninstall", "-y", "onnxruntime"], check=False)
subprocess.run(["pip", "install", "-q", "--no-deps", "rtmlib"], check=True)
# av: PyAV links its OWN ffmpeg, which is why it can read clips that abort cv2's.
subprocess.run(["pip", "install", "-q", "onnxruntime-gpu==1.26.0", "pyyaml", "tqdm",
                "av>=12.0"], check=True)"""),
("code", RESOLVE_CODE),
("code", """from rtmlib import RTMO
# Same staged rtmo-l.onnx as notebook 01. This notebook has internet and rtmlib could
# fetch its own copy, but then the ADL and fall shards would come from two separately
# resolved checkpoints - the fall head would be trained on a different pose distribution
# than the ADL heads, and nothing would say so.
_onnx = sorted(pathlib.Path("/kaggle/input").glob("**/rtmo-l.onnx"))
assert _onnx, ("rtmo-l.onnx not found - attach the weights dataset from notebook 00 "
               f"(e.g. behavioursense-WW). Mounted: {sorted(p.name for p in pathlib.Path('/kaggle/input').iterdir())}")
body = RTMO(onnx_model=str(_onnx[0]), model_input_size=(640, 640),
            backend="onnxruntime", device="cuda")
# Same live-provider check as notebook 01: get_available_providers() reports the build,
# the session reports reality. A CUDA-13 onnxruntime-gpu on a CUDA-12 image lists the
# provider, fails to load it, and extracts on CPU without saying so.
_active = list(getattr(body, "session").get_providers())
assert "CUDAExecutionProvider" in _active, (
    f"RTMO is running on {_active} - CUDA did not load. This notebook has internet, so "
    "fix it here: pip install onnxruntime-gpu==1.26.0 (newest CUDA 12.8 build).")
print(f"pose model: {_onnx[0]}  provider: {_active[0]}")"""),
("code", """# Acquire all four fall corpora. Three fetch themselves; Le2i attaches as a mount.
#
# Verified reachable 2026-08-08. Le2i is the exception: its canonical host
# (le2i.cnrs.fr) refuses connections, the IMVIA successor page 404s, and the Wayback
# captures under that domain are staff pages with no archived video - so it is consumed
# as an attached Kaggle dataset instead of a download. Attach `tuyenldvn/falldataset-imvia`
# ("Le2i Fall Dataset", ~10 GB); the mount is found by CONTENT below, so any mirror or
# any dataset title works.
#
# Each corpus is independent: one unreachable source must not cost the others a session,
# so failures are collected and reported, never raised.
import subprocess, pathlib, urllib.request, urllib.error, json, zipfile, shutil
import sys as _sys
_sys.path.insert(0, str(SCRIPTS))
from prepare_skeletons import le2i_fall_frames

TMP = pathlib.Path("/tmp/falls"); TMP.mkdir(parents=True, exist_ok=True)
INPUT = pathlib.Path("/kaggle/input")
SOURCES, FAILED = {}, {}

# Mendeley answered 403 to urllib's default User-Agent from a Kaggle IP (it works from
# a laptop), so every request here carries a browser UA. Same opener is used for the
# API walk below, or the walk 403s while the file fetches succeed.
_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
_opener = urllib.request.build_opener()
_opener.addheaders = list(_UA.items())
urllib.request.install_opener(_opener)

def _get(url, dst, timeout=120):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 1000:
        return True
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f)
        return dst.stat().st_size > 1000
    except (urllib.error.URLError, OSError, TimeoutError):
        dst.unlink(missing_ok=True)
        return False

# --- GMDCSA-24: git clone -------------------------------------------------------------
try:
    d = TMP / "gmdcsa"
    if not d.exists():
        subprocess.run(["git", "clone", "--depth", "1",
            "https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos",
            str(d)], check=True, capture_output=True)
    SOURCES["gmdcsa"] = d
except Exception as exc:
    FAILED["gmdcsa"] = f"clone failed: {exc}"

# --- URFD: 70 direct MP4s. The host every paper cites (fenix.univ.rzeszow.pl) is dead;
# fenix.ur.edu.pl is the live one. Its primary distribution is zipped PNG frames, but we
# decode video anyway, so the MP4s are the cheaper path.
try:
    d = TMP / "urfd"; base = "https://fenix.ur.edu.pl/~mkepski/ds/data"
    want = ([(f"fall-{i:02d}-cam0.mp4") for i in range(1, 31)]
            + [f"adl-{i:02d}-cam0.mp4" for i in range(1, 41)])
    got = sum(_get(f"{base}/{n}", d / n) for n in want)
    if got:
        SOURCES["urfd"] = d
    if got < len(want):
        print(f"  urfd      {got}/{len(want)} clips (partial - continuing)")
    if not got:
        FAILED["urfd"] = "no clips downloaded"
except Exception as exc:
    FAILED["urfd"] = str(exc)

# --- CAUCAFall: ATTACHED MOUNT, not a download. Its Mendeley record (DOI
# 10.17632/7w7fccy7ky) holds nine documentation files - one xlsx and eight jpegs, 1.1 MB
# total - and no video at all; versions 1-3 answer 451 and the S3 bulk zip is 403 under any
# User-Agent. So a walk of that record can only ever find figures, which is exactly what
# "Mendeley walk found no video" meant. Attach `tuyenldvn/caucafall` (8.33 GB) instead -
# same publisher as the Le2i mirror already in use.
try:
    cauca = None
    for cand in INPUT.glob("**/*"):
        # Subject.1 .. Subject.10, each holding one folder per activity. The corpus root is
        # their parent, found by content so the mirror's nesting depth does not matter.
        if not cand.is_dir():
            continue
        if cand.name.lower().replace(" ", "").replace("_", "") not in ("subject.1", "subject1"):
            continue
        if any(cand.parent.rglob("*.avi")) or any(cand.parent.rglob("*.mp4")):
            cauca = cand.parent
            break
    if cauca is not None:
        SOURCES["caucafall"] = cauca
    else:
        FAILED["caucafall"] = ("not attached - add the Kaggle dataset tuyenldvn/caucafall "
                               "(8.33 GB). Mendeley publishes only figures for this DOI.")
except Exception as exc:
    FAILED["caucafall"] = str(exc)

# --- Le2i: attached mount, located by CONTENT. Its annotation .txt files are the only
# thing that distinguishes its fall clips from its ADL clips, so the mount is identified
# by having BOTH video and Annotation_files - a video-only mirror is refused loudly in
# the labelling cell rather than silently labelled ADL.
# The root must be the CORPUS root, not the mount. Kaggle nests datasets as
# /kaggle/input/datasets/<owner>/<name>/..., and iterating INPUT.glob("*") accepted
# `/kaggle/input/datasets` itself - so every Le2i clip's subject id became the OWNER
# name, collapsing all 190 videos onto one id and defeating the P1 split. Anchor on the
# annotations instead: Annotation_files sits at <corpus>/<scene>/Annotation_files, so
# its grandparent is the corpus root and the scene survives as the subject.
le2i = None
_anno_dirs = sorted(INPUT.glob("**/Annotation_files"))
if _anno_dirs:
    # The corpus root is the COMMON ancestor of every annotation folder, not the
    # grandparent of the first one. Taking the grandparent and breaking picked up
    # Coffee_room_01 alone - 48 of ~190 videos, and since subject ids are derived
    # relative to the root, all 48 collapsed onto the single id "le2i_Coffee_room_01".
    # commonpath spans the scenes whatever depth the mirror nests them at, and keeps the
    # scene as the first relative component, which is what becomes the subject.
    import os
    if len(_anno_dirs) == 1:
        _root = _anno_dirs[0].parent.parent
    else:
        _root = pathlib.Path(os.path.commonpath([str(a) for a in _anno_dirs]))
    _cands = [_root]
else:   # some mirrors drop the folder and keep the .txt beside the video
    _cands = sorted({t.parent.parent for t in INPUT.glob("**/*.txt")
                     if (t.parent / f"{t.stem}.avi").exists()
                     or (t.parent.parent / "Videos" / f"{t.stem}.avi").exists()})
for cand in _cands:
    vids = sorted(cand.rglob("*.avi"))
    if not vids:
        continue
    # Functional check, not a shape check: annotations must actually RESOLVE. Sample
    # ACROSS the tree rather than the first 20 - a contiguous head sample lives in one
    # scene and says nothing about the rest.
    step = max(1, len(vids) // 20)
    sample = vids[::step][:20]
    hits = sum(le2i_fall_frames(v) is not None for v in sample)
    if hits >= max(1, len(sample) // 2):
        le2i = cand
        scenes = sorted({v.relative_to(cand).parts[0] for v in vids})
        print(f"  le2i      annotations resolve for {hits}/{len(sample)} sampled clips "
              f"across {len(scenes)} scene(s): {scenes[:6]}")
        break
if le2i:
    SOURCES["le2i"] = le2i
else:
    FAILED["le2i"] = (f"no attached mount has resolvable Le2i annotations "
                      f"(checked {len(_cands)} candidate root(s)) - attach "
                      "tuyenldvn/falldataset-imvia")

for k in ("gmdcsa", "urfd", "caucafall", "le2i"):
    if k in SOURCES:
        n = sum(1 for _ in SOURCES[k].rglob("*") if _.suffix.lower() in (".avi", ".mp4"))
        print(f"  {k:<10} OK    {n:>4} videos  {SOURCES[k]}")
    else:
        print(f"  {k:<10} ABSENT      {FAILED.get(k, '?')}")
assert SOURCES, f"no corpus acquired: {FAILED}"
print(f"\\n{len(SOURCES)}/4 corpora - P2 leave-one-dataset-out needs >=2 to mean anything")"""),
("code", """# Decode probe: find the clips that KILL the interpreter, before spending a session.
#
# Two runs died here, both inside URFD, both ~0.4 s after three "[mp3float] Header
# missing" lines. Neither the per-video try/except nor the 3000-frame cap fired, because
# the fault is not in Python: it is in cv2 -> ffmpeg -> the mp3float decoder, and a
# SIGSEGV/abort there takes the whole interpreter with it. No in-process guard can catch
# that. (Checked: all 70 URFD mp4s are well-formed video-only mp42 containers, none
# truncated - so this is not a bad download and cannot be screened by inspection.)
#
# So decode every clip ONCE in a child process first, journalling each attempt BEFORE it
# starts. If the child dies, its last unfinished entry names the poison file; the parent
# restarts and the child skips everything already journalled. Converges in 1 + n_poison
# runs, each crash costing exactly one clip instead of the corpus.
#
# This costs a decode pass (~5-8 min) and buys a session that finishes.
import subprocess, sys, pathlib

WORK = pathlib.Path("/kaggle/working")
JOURNAL = WORK / "decode_probe.tsv"
LISTFILE = WORK / "decode_list.txt"

all_vids = []
for _src, _root in SOURCES.items():
    all_vids += sorted(str(v) for v in _root.rglob("*")
                       if v.suffix.lower() in (".mp4", ".avi"))
LISTFILE.write_text(chr(10).join(all_vids) + chr(10), encoding="utf-8")
print(f"probing {len(all_vids)} clips for decoder crashes")

TAB, NL = chr(9), chr(10)
CHILD = NL.join([
    "import sys, os",
    "backend = sys.argv[3] if len(sys.argv) > 3 else 'cv2'",
    "lst, out = sys.argv[1], sys.argv[2]",
    "seen = set()",
    "if os.path.exists(out):",
    "    for line in open(out):",
    "        parts = line.rstrip(chr(10)).split(chr(9))",
    "        if len(parts) >= 2:",
    "            seen.add(parts[1])",
    "f = open(out, 'a', buffering=1)",
    "for line in open(lst):",
    "    p = line.strip()",
    "    if not p or p in seen:",
    "        continue",
    "    f.write('TRY' + chr(9) + p + chr(10))",   # journalled BEFORE the risky decode
    "    n = 0",
    "    if backend == 'av':",
    "        import av",
    "        with av.open(p) as container:",
    "            for _fr in container.decode(video=0):",
    "                n += 1",
    "                if n > 20000:",
    "                    break",
    "    else:",
    "        import cv2",
    "        cap = cv2.VideoCapture(p)",
    "        while True:",
    "            ok, _fr = cap.read()",
    "            if not ok:",
    "                break",
    "            n += 1",
    "            if n > 20000:",
    "                break",
    "        cap.release()",
    "    f.write('OK' + chr(9) + p + chr(9) + str(n) + chr(10))",
])
(WORK / "_decode_probe.py").write_text(CHILD, encoding="utf-8")

# One restart per poison clip. A cap of 11 was set for "a couple of bad files" and was
# immediately wrong: Le2i's mirror crashes on EVERY clip, so the budget ran out after 11
# and the remaining 179 were never probed - silently excluded from extraction while the
# log said only "continuing with what passed". Each restart costs ~0.25 s, so budgeting
# for the worst case (every clip poison) costs ~2 min and removes the failure mode.
# Progress is guaranteed: the child journals TRY before decoding, so each run advances at
# least one clip.
budget = len(all_vids) + 5
for attempt in range(1, budget + 1):
    r = subprocess.run([sys.executable, str(WORK / "_decode_probe.py"),
                        str(LISTFILE), str(JOURNAL), "cv2"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        break
    if attempt <= 3 or attempt % 25 == 0:
        print(f"  probe crashed (exit {r.returncode}) - restart {attempt}/{budget}")
else:
    print(f"  WARNING: probe never converged in {budget} restarts")

# Second chance with PyAV before writing anything off. cv2 SIGABRTs/SIGSEGVs on every
# clip in the Le2i mirror - 190 videos, including the only corpus here with exact
# frame-level fall annotations - and that is a codec the bundled ffmpeg mishandles, not
# corrupt data. PyAV links its own ffmpeg, so it is a genuinely different decoder, and
# `av>=12.0` is already declared in requirements-train.txt for exactly this job.
#
# This is self-verifying: if PyAV also dies, those clips stay excluded and the only cost
# is a few minutes of probing. Nothing is assumed to work.
_rows = [l.split(TAB) for l in JOURNAL.read_text(encoding="utf-8").splitlines() if l.strip()]
_ok_cv2 = {r[1] for r in _rows if r[0] == "OK"}
_failed = sorted(set(all_vids) - _ok_cv2)
AV_JOURNAL = WORK / "decode_probe_av.tsv"
AV_OK = set()
if _failed:
    print(f"  retrying {len(_failed)} cv2-failed clip(s) with PyAV")
    AVLIST = WORK / "decode_list_av.txt"
    AVLIST.write_text(chr(10).join(_failed) + chr(10), encoding="utf-8")
    for attempt in range(1, len(_failed) + 5):
        r = subprocess.run([sys.executable, str(WORK / "_decode_probe.py"),
                            str(AVLIST), str(AV_JOURNAL), "av"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            break
    if AV_JOURNAL.exists():
        AV_OK = {q.split(TAB)[1] for q in
                 AV_JOURNAL.read_text(encoding="utf-8").splitlines()
                 if q.startswith("OK")}
    print(f"  PyAV rescued {len(AV_OK)}/{len(_failed)}")

rows = [l.split(TAB) for l in JOURNAL.read_text(encoding="utf-8").splitlines() if l.strip()]
USABLE = {r[1] for r in rows if r[0] == "OK"} | AV_OK
# Per-clip backend, so extraction decodes each file with whatever actually read it.
BACKEND = {q: ("av" if q in AV_OK else "cv2") for q in USABLE}
POISON = sorted(set(all_vids) - USABLE)
UNPROBED = sorted(set(all_vids) - USABLE - set(POISON))
print(f"  {len(USABLE)}/{len(all_vids)} clips decode cleanly")
if POISON:
    # Named, not silently dropped: real data is being excluded and the count belongs in
    # the write-up next to the shard totals.
    by_dir = {}
    for q in POISON:
        by_dir[str(pathlib.Path(q).parent)] = by_dir.get(str(pathlib.Path(q).parent), 0) + 1
    print(f"  {len(POISON)} clip(s) crash the decoder, by directory:")
    for d, n in sorted(by_dir.items(), key=lambda kv: -kv[1]):
        print(f"    {n:>4}  {d}")
if UNPROBED:
    # Distinct from POISON: these were never even attempted, so nothing is known about
    # them. Reporting them as one number with the crashes would hide a budget failure.
    print(f"  {len(UNPROBED)} clip(s) NEVER PROBED (budget exhausted) - also excluded")
assert USABLE, "no clip decoded cleanly - do NOT Save Version"
"""),
("code", """# Extract + label. Fall clips are short and pre-trimmed; windows are labelled by clip
# type and position: the descent lands mid-clip, so windows around the drop are `falling`
# (7), later windows `fallen_on_ground` (8), earlier ones `standing` (1); ADL clips ->
# other_idle (19). Coarse by construction — the OmniFall staged->wild protocol is where
# finer temporal labels come from; this labelling is stated in the eval, not hidden.
import cv2, collections, pathlib
import numpy as np
from prepare_skeletons import (   # tested repo code, not copies
    WINDOW_FRAMES, assign_slots, is_fall_clip, label_fall_windows, le2i_fall_frames,
    subject_from_path, window_clip)
OUT = pathlib.Path("/kaggle/working/shards"); OUT.mkdir(exist_ok=True)

def extract_av(video, fps_sample=15):
    # PyAV path, for the clips whose codec aborts cv2. Frames come out RGB; RTMO was fed
    # BGR for every other corpus, so convert - mixing colour orders across corpora would
    # be a silent distribution shift in the fall head's training data.
    import av
    poses, i, step = [], 0, 1
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        src = float(stream.average_rate or 25)
        step = max(1, round(src / fps_sample))
        for frame in container.decode(video=0):
            if i % step == 0:
                kp, sc = body(frame.to_ndarray(format="bgr24"))
                poses.append(assign_slots(kp, sc, poses[-1] if poses else None))
                if len(poses) >= 3000:
                    break
            i += 1
    return (np.stack(poses) if len(poses) >= 30 else None), step


def extract(video, fps_sample=15):
    # -> (poses, step). step is returned because Le2i's annotations are in ORIGINAL
    # frame numbers while the windows are in subsampled ones; the caller needs it to
    # convert. A docstring cannot go here - this cell is itself a '''...''' literal.
    if BACKEND.get(str(video)) == "av":
        return extract_av(video, fps_sample)
    cap = cv2.VideoCapture(str(video))
    src = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, round(src / fps_sample))
    # No clip in these corpora runs past a few minutes, so an unbounded read loop buys
    # nothing and risks everything: a malformed container hands back frame after frame
    # from a broken index until RAM is gone. The run that died logged three
    # "[mp3float] Header missing" lines and lost the kernel on the next one.
    MAX_SAMPLED = 3000                      # 200 s at 15 fps
    poses, f, capped = [], 0, False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f % step == 0:
            kp, sc = body(frame)
            # Tracked slots, not per-frame area order - see notebook 01's extraction cell.
            poses.append(assign_slots(kp, sc, poses[-1] if poses else None))
            if len(poses) >= MAX_SAMPLED:
                capped = True
                break
        f += 1
    cap.release()
    if capped:
        print(f"    CAPPED {video.name} at {MAX_SAMPLED} sampled frames")
    return (np.stack(poses) if len(poses) >= 30 else None), step

skels, labels, subjects, sources = [], [], [], []
unlabelled = collections.Counter()
shard_i, banked = 0, 0

# Flush every ~400 MB and at every corpus boundary, exactly like notebook 01. The first
# version accumulated all four corpora and wrote ONE npz after the loop; the kernel died
# 27 minutes in, on a Le2i clip, and took the already-finished gmdcsa (160 clips) and
# urfd (70 clips) with it. Nothing about that extraction was wrong - the run simply had
# no way to keep what it had already earned.
def flush():
    global skels, labels, subjects, sources, shard_i, banked
    if not skels:
        return
    np.savez_compressed(
        OUT / f"falls_{shard_i:04d}.npz",
        skeletons=np.stack(skels).astype(np.float16),
        labels=np.asarray(labels, dtype=np.int64),
        subjects=np.asarray(subjects, dtype="<U32"),
        datasets=np.asarray(sources, dtype="<U32"))
    banked += len(skels)
    print(f"    wrote falls_{shard_i:04d}.npz ({len(skels)} windows, {banked} banked)")
    shard_i += 1
    skels, labels, subjects, sources = [], [], [], []
for source, root in SOURCES.items():
    if not root.exists():
        continue
    vids = sorted(v for v in list(root.rglob("*.mp4")) + list(root.rglob("*.avi"))
                  if str(v) in USABLE)
    print(f"{source}: {len(vids)} clips (probe-cleared)")
    for v in vids:
        # One unreadable clip must not end the corpus, let alone the session.
        try:
            poses, step = extract(v)
        except Exception as exc:                  # noqa: BLE001
            unlabelled[f"{source}:read-error"] += 1
            print(f"    SKIP {v.name}: {type(exc).__name__} {exc}")
            continue
        if poses is None:
            continue
        # How a clip is labelled depends on what its corpus actually tells us.
        #
        # Le2i is the awkward one: no path component says fall or ADL - every clip is
        # `video (N).avi` under a room folder, and the truth lives in an Annotation_files
        # .txt whose first two lines are the fall's start/end frame. Treating a missing
        # annotation as "not a fall" would write the ~192 clips that DO contain falls
        # into the negatives, so an unlabellable clip is SKIPPED and counted, never
        # guessed. Where the annotation exists it beats the heuristic outright: real
        # frame boundaries instead of "the descent is somewhere in the middle".
        fall_range = None
        if source == "le2i":
            ann = le2i_fall_frames(v)
            if ann is None:
                unlabelled[source] += 1
                continue
            fall_range = ann
            is_fall = ann != (0, 0)
        else:
            # Directory/filename markers, token-matched: CAUCAFall names its folders
            # "Fall forward"/"Fall backward", which an exact-match rule scored as ADL -
            # all 50 of its fall clips would have become negatives.
            is_fall = is_fall_clip(root, v)

        wins = window_clip(poses, with_starts=True)
        if not wins:
            continue
        starts = [st for st, _ in wins]
        if is_fall or fall_range is not None:
            # Sampled space: annotations are in ORIGINAL frames, windows are in
            # subsampled frames, so the interval is converted with the same step the
            # decoder used. Skipping this scales the fall interval by ~2x at 15 fps.
            rng = None if fall_range is None else (fall_range[0] // step,
                                                   fall_range[1] // step)
            labs = label_fall_windows(starts, WINDOW_FRAMES, rng)
        else:
            labs = [19] * len(starts)

        subj = subject_from_path(root, v, source)
        for (_st, w), lab in zip(wins, labs):
            skels.append(w)
            labels.append(lab)
            subjects.append(subj)
            sources.append(source)

        if sum(x.nbytes for x in skels) > 400e6:
            flush()
    # Corpus boundary: bank what is finished before starting the next one.
    flush()
    # Report what this corpus actually CONTRIBUTED, not just the running total. The run
    # that motivated this acquired 190 Le2i clips and shipped 3 of its 6 scenes; the
    # difference was counted in `unlabelled` and never printed, so the loss was invisible
    # in a log that otherwise looked healthy.
    _used = sum(1 for q in vids if str(q) in USABLE)
    _skipped = sum(n for k, n in unlabelled.items() if k.startswith(source))
    _scenes = sorted({subject_from_path(root, q, source) for q in vids
                      if str(q) in USABLE}) if _used else []
    print(f"  {source} done - {banked} windows on disk | {_used}/{len(vids)} clips "
          f"decodable, {_skipped} skipped, {len(_scenes)} subject id(s) contributed")
    if _skipped:
        for k, n in sorted(unlabelled.items()):
            if k.startswith(source):
                print(f"      skipped: {k.split(':', 1)[-1] if ':' in k else 'no-annotation'}"
                      f" x{n}")

# Gate before writing. Notebook 01 taught this the hard way: a silent zero-yield run that
# still invites "Save Version" publishes an unusable dataset as if it were fine, and Kaggle
# versions REPLACE content. Every assert below is a thing that has a plausible silent path.
# Read back from DISK, not from the in-memory lists: extraction now flushes every 400 MB
# and at every corpus boundary, so by the time this cell runs those lists are empty by
# design. Checking them would assert "NO windows extracted" on a perfectly good run - and
# a gate that cries wolf is a gate that gets commented out. Only the label/subject/dataset
# vectors are loaded; the skeletons stay on disk.
if unlabelled:
    print("\\nclips skipped during extraction (counted, and now reported):")
    for k, n in sorted(unlabelled.items()):
        print(f"  {k:<34} {n:>5}")
    print("  A clip is skipped when its label cannot be established - for Le2i that means")
    print("  no readable Annotation_files entry. Skipping is correct (guessing would put")
    print("  real falls in the negatives), but the COUNT belongs in the write-up.")

shards = sorted(OUT.glob("falls_*.npz"))
assert shards, (
    f"NO shard written from {list(SOURCES)}. Nothing to save - do NOT Save Version. "
    f"Check the clone succeeded and that clips are .mp4/.avi under those roots.")
labels, subjects, sources = [], [], []
for sp in shards:
    with np.load(sp, allow_pickle=False) as z:
        labels.extend(z["labels"].tolist())
        subjects.extend(z["subjects"].tolist())
        sources.extend(z["datasets"].tolist())
print(f"{len(shards)} shard(s) on disk, {len(labels)} windows total")

counts = collections.Counter(labels)
names = {1: "standing", 7: "falling", 8: "fallen_on_ground", 19: "other_idle"}
print(f"{len(labels)} windows from {len(set(sources))} source(s):")
for lab in sorted(counts):
    print(f"  {lab:>2} {names.get(lab, '?'):<18} {counts[lab]:>6}")

# The fall head is a BINARY discriminator. Without both fall and non-fall windows it
# trains on one class, reports a meaningless AUROC near 0.5, and nothing in the metric
# says the data was degenerate rather than the model bad.
have_fall = counts[7] + counts[8]
have_neg  = counts[1] + counts[19]
assert have_fall > 0, (
    f"ZERO fall windows ({dict(counts)}). The is_fall rule matched no clip - it keys on "
    f"'fall' in the filename or parent directory, so a corpus using other names (e.g. "
    f"'Coup'/'Chute') needs the rule extended. Do NOT Save Version: notebook 03's fall "
    f"head would train on a single class.")
assert have_neg > 0, (
    f"ZERO non-fall windows ({dict(counts)}) - every clip matched the is_fall rule, so "
    f"the binary head has no negatives. Check the ADL clips are actually present.")
assert counts[7] > 0, (
    f"fall clips found but ZERO 'falling' windows ({dict(counts)}) - the descent labelling "
    f"failed. Expected at least one per fall clip.")
print(f"  balance: {have_fall} fall / {have_neg} non-fall")

# Subject ids decide the P1 split, and a collapsed id is INVISIBLE downstream:
# split_by_subject asserts the ids are disjoint, which they are - it is the PEOPLE behind
# them that would not be. GMDCSA-24 numbers every subject's clips 01..25, so the old
# stem-based rule mapped four people onto one id and put each of them on both sides of
# the split. Print the ids and refuse a degenerate count.
by_subject = collections.Counter(subjects)
print(f"  {len(by_subject)} subject id(s):")
for sub, n in sorted(by_subject.items()):
    print(f"    {sub:<28} {n:>6} windows")
assert len(by_subject) >= 2, (
    f"only {len(by_subject)} subject id ({list(by_subject)}) - cross-subject (P1) "
    "evaluation needs at least 2, and split_by_subject would raise on a degenerate "
    "split. Check subject_from_path against this corpus's directory layout.")
# Every subject should carry both kinds of clip, or the split can hand the fall head a
# validation fold containing no falls at all.
sub_has_fall = collections.defaultdict(set)
for sub, lab in zip(subjects, labels):
    sub_has_fall[sub].add(lab in (7, 8))
one_sided = [k for k, v in sub_has_fall.items() if len(v) == 1]
if one_sided:
    print(f"  NOTE: {len(one_sided)} subject(s) have only one clip type "
          f"({one_sided[:4]}) - a split isolating them yields a fold with no positives.")

# Subject-id GRANULARITY, stated because it decides what P1 actually measures.
#
# URFD is flat (fall-01-cam0.mp4) and publishes no clip->volunteer mapping - only
# per-frame posture labels - so subject_from_path falls back to the filename stem and
# every clip becomes its own "subject". Its ~70 sequences come from a handful of
# volunteers, so cross-subject splitting cannot separate them and P1 is OPTIMISTIC for
# the URFD portion. That is a property of the corpus, not a bug to code around, and it
# belongs in the eval tables next to the number - exactly as the Charades video-id proxy
# already is.
#
# The measurable consequence is variance: holding out 20% of SUBJECTS holds out anywhere
# from 3% to 45% of WINDOWS when most ids are single clips. Measured on this shard set,
# val folds ranged 160-1406 windows across 12 seeds. So report the spread rather than
# trusting one draw.
tiny = [k for k, n in by_subject.items() if n < 20]
print(f"\\n  subject granularity: {len(by_subject)} ids, {len(tiny)} with <20 windows")
if tiny:
    per_ds = collections.Counter(k.split("_")[0] for k in tiny)
    print(f"    small ids by corpus: {dict(per_ds)}")
    print("    URFD publishes no clip->volunteer mapping, so each CLIP is its own subject.")
    print("    P1 is therefore optimistic for URFD - state this in the eval table.")
from behaviorsense.data.skeleton_dataset import split_by_subject as _sbs
_subs = np.asarray(subjects)
_sizes = []
for _seed in range(8):
    try:
        _tr, _va = _sbs(_subs, val_frac=0.2, seed=_seed)
        _sizes.append(len(_va))
    except Exception:
        _sizes.append(0)
print(f"    val-fold size across 8 seeds: min {min(_sizes)} / max {max(_sizes)} windows")
if min(_sizes) < 200:
    print("    WARNING: at least one seed yields a val fold under 200 windows. Fit the")
    print("    fall operating point on a fixed seed and report the fold size with it.")

# P2 is leave-one-DATASET-out (docs/06 section 'protocols'). One corpus cannot support it:
# there is no second dataset to hold out. This is a loud warning, not an assert, because a
# single-corpus shard is still fine for P1/P3 and for a first end-to-end pass.
n_src = len(set(sources))
if n_src < 2:
    print(f"\\n  WARNING: only {n_src} dataset ({sorted(set(sources))}). P2 "
          f"leave-one-dataset-out is NOT possible - it needs at least 2 corpora, "
          f"ideally 3-4.")
    print("           Notebook 04 will skip the P2 table and say so. To enable it, upload "
          "Le2i / CAUCAFall / URFD as private datasets and uncomment them in SOURCES above.")
else:
    print(f"\\n  P2 viable: {n_src} datasets -> {n_src} leave-one-out folds "
          f"({sorted(set(sources))})")

# Nothing to write here: extraction already banked every window. A savez at this point
# would rebuild falls_0000.npz from the now-empty in-memory lists and OVERWRITE the
# real first shard with an empty one - the flush refactor's sharpest edge.
mb = sum(sp.stat().st_size for sp in shards) / 1e6
print(f"\\n{len(shards)} shard(s), {mb:.1f} MB total:")
for sp in shards:
    print(f"  {sp.name}  {sp.stat().st_size/1e6:>7.1f} MB")
print("\\nSave Version -> create dataset behaviorsense-fall-shards.")"""),
]

# ===========================================================================
# 03 - training (OFFLINE, Blackwell)
# ===========================================================================

N03: list[tuple[str, str]] = [
("markdown", """# 03 — Training (OFFLINE, RTX PRO 6000 Blackwell, NO INTERNET)

| attach as input | holds | produces |
|---|---|---|
| the repo dataset | `src/`, `scripts/`, `configs/` | `behaviorsense-runs` |
| the wheels+weights dataset(s) | `*.whl` and `rtmo-l.onnx` | (create/update from `/kaggle/working/runs`) |
| the ADL shards | `*.npz` from notebook 01 | |
| the fall shards | `*.npz` from notebook 02 | |
| `behaviorsense-runs` | previous `last.pt` — **resume runs only** | |

**Dataset names and groupings do not matter.** Notebook 00 emits `wheels/` and `weights/`
as two folders, and publishing them as ONE dataset (e.g. `behavioursense-WW`) or two is
your choice — the resolver below finds each by CONTENT, so both layouts work and the
spelling `behaviour`/`behavior` is tolerated. Attach whatever you have; the cell prints
what it resolved.

Session discipline, in order: **preflight (1 min) → carry forward checkpoints → train**.
Every failure mode this ordering prevents costs hours: wrong-arch torch fails at minute
one instead of minute forty; a resumed run continues bit-exactly instead of restarting.

Resume is bit-exact (RNG state is checkpointed) and refuses changed hyperparameters —
if a cell errors with `resume mismatch`, re-run with the ORIGINAL values rather than
deleting the guard."""),
("code", RESOLVE),
("code", """# Torch: use the preinstalled build if it supports sm_120, else install from the staged
# wheels. Deciding at runtime is the whole point of staging both.
import subprocess, sys
def sm120_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and "sm_120" in torch.cuda.get_arch_list()
    except Exception:
        return False
if not sm120_ok():
    print("preinstalled torch unusable on sm_120 - installing staged cu128 build")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-index",
         "--find-links", str(WHEELS),
         "torch", "numpy", "pydantic", "PyYAML", "Pillow"],
        check=True)
    print("RESTART the kernel now (Run -> Restart) so the new torch is imported, "
          "then run from the next cell.")
else:
    import torch
    print(f"torch {torch.__version__} already supports sm_120")"""),
("code", """# Preflight: GPU arch, bf16 matmul, one ST-GCN++ fwd/bwd, every staged asset loads.
# Nonzero exit here means DO NOT start training - each listed problem surfaces late
# otherwise, after preprocessing time is already spent.
#
# --profile training scopes WHICH assets block. This notebook trains ST-GCN++ from
# pre-extracted .npz shards and loads no staged checkpoint, so the OSNet ReID weights are
# reported but do not fail it: they are Agent 1 serving-time, and their operating point
# (tau=0.3546) is already fitted in results/reid_eval.md. Everything that would actually
# stop this session - wrong arch, no bf16, a corrupt file - still blocks.
import subprocess, sys, os
# INHERIT the environment and only add PYTHONPATH. A hand-built env looks tidy and is
# wrong: it drops the CUDA variables Kaggle sets (LD_LIBRARY_PATH, NVIDIA_VISIBLE_DEVICES,
# CUDA_MODULE_LOADING...), so torch.cuda.is_available() is False in the child while the
# parent kernel sees the GPU perfectly well. The result was a preflight that reported
# "CUDA is not available - training would silently run on CPU" one line after the cell
# above printed "torch 2.10.0+cu128 already supports sm_120" - a self-contradicting
# session abort, caused entirely by this dict. tests/test_notebooks.py hit the same trap
# on Windows (a minimal env strips SystemRoot and breaks Winsock in torch.distributed)
# and fixed it there; the notebook kept the bug.
r = subprocess.run(
    [sys.executable, str(SCRIPTS / "kaggle_smoke_test.py"),
     "--assets", str(WEIGHTS), "--profile", "training"],
    env={**os.environ, "PYTHONPATH": str(SRC)})
assert r.returncode == 0, "PREFLIGHT FAILED - fix before burning GPU hours"
"""),
("code", """# Collect shards from every attached shard dataset. Training accepts many --shards
# paths, which is why there is NO merge/combine notebook: merging would only copy bytes
# into a third dataset and double the storage without changing a single trained weight.
#
# ADL and fall shards are the same file type, so only the mount distinguishes them; match
# on the dataset name, tolerating the behaviour/behavior spelling, then assert non-empty.
# Training on an empty set for eleven hours is the failure this prevents.
from behaviorsense.kaggle_artifacts import is_real_artifact

def shard_paths(*keywords):
    # Group EVERY .npz by which ancestor directory names the corpus - do not iterate the
    # top level of /kaggle/input. Attaching by URL nests the mount as
    # /kaggle/input/datasets/<owner>/<name>/..., so `INPUT.iterdir()` yields just
    # ['competitions', 'datasets'], neither of which contains "adl" or "fall": both lists
    # came back EMPTY with the shard datasets correctly attached. The resolver cell above
    # never had this bug because it globs with **/; this cell was still name-on-top-level.
    out = []
    groups = {}
    for q in INPUT.glob("**/*.npz"):
        # Skip synthetic fixtures. `train_fall.py --smoke` writes data/shards/_smoke_fall.npz,
        # the code dataset is an upload of the working tree, and "_smoke_fall" contains
        # "fall" - so the fixture was matched into the real fall corpus and the head was
        # partly fitted on 400 windows of generated data. The leading underscore is the
        # convention every smoke fixture in this repo uses.
        if not is_real_artifact(q):
            print(f"  skipping {q.name} (fixture or code-checkout leftover)")
            continue
        rel = [part.lower().replace("behaviour", "behavior") for part in q.parts]
        owner = next((part for part in rel if any(k in part for k in keywords)), None)
        if owner:
            groups.setdefault(owner, []).append(str(q))
    for owner in sorted(groups):
        hits = sorted(groups[owner])
        print(f"  {owner}: {len(hits)} .npz")
        out += hits
    return sorted(out)

print("ADL mounts:")
ADL = shard_paths("adl")
print("fall mounts:")
FALL = shard_paths("fall")
print(f"ADL shards: {len(ADL)}, fall shards: {len(FALL)}")
assert ADL, f"no ADL shards found. Attached: {ATTACHED}"
assert FALL, f"no fall shards found. Attached: {ATTACHED}"

# The two sets must be DISJOINT. They are matched by mount name, so the only way a file
# appears in both is a dataset that actually holds the other corpus - e.g. Save Version
# publishing a working directory that still had the carried-forward ADL shards in it.
# That is not cosmetic: the fall head would count the same windows as positives and
# negatives, and P2's leave-one-dataset-out split would leak across folds. Equal counts
# in both lists is the symptom that prompted this check.
_dup = sorted(set(pathlib.Path(a).name for a in ADL)
              & set(pathlib.Path(f).name for f in FALL))
assert not _dup, (
    f"{len(_dup)} filename(s) appear in BOTH the ADL and fall shard sets: {_dup[:6]}"
    f"\\n  One of those datasets contains the other corpus. Check the per-mount counts"
    f"\\n  above, then re-publish the offending dataset from a clean /kaggle/working.")

# Report what the fall head will actually see. The fall corpora alone are ~82% positive;
# with the ADL windows as negatives it should land near 1-2%. A number outside that range
# means the mix is wrong, and every AUPRC/threshold downstream would be an artefact of it.
import numpy as np
_pos = _tot = 0
for _p in FALL + ADL:
    with np.load(_p) as _z:
        _lab = _z["labels"]
    _pos += int(np.isin(_lab, (7, 8)).sum()); _tot += int(_lab.size)
print(f"fall-head training mix: {_tot:,} windows, {_pos:,} positive "
      f"({_pos / max(_tot, 1):.2%})")
assert _tot > 0, "shards contain no windows at all"
"""),
("code", """# Carry forward checkpoints for resume. Inputs are read-only; checkpoints get
# overwritten during training, so they must live in /kaggle/working.
import pathlib, shutil
RUNS = pathlib.Path("/kaggle/working/runs")
RUNS.mkdir(parents=True, exist_ok=True)
# Locate a previous run by the checkpoint itself. Same reasoning as the shard
# carry-forward in notebook 01: an unfound checkpoint must not read as "fresh",
# because resuming from nothing silently restarts an 11-hour run from epoch 0.
#
# But "any last.pt under /kaggle/input" is too generous, and it cost a session. The code
# dataset is an upload of the working tree, so it carried the LOCAL runs/ directory -
# CPU smoke checkpoints from `train_adl.py --smoke` (epochs=25, smoke=True). Those were
# copied in and resumed as though they were prior GPU work, which restarts an 80-epoch
# run on a 25-epoch cosine schedule. Nothing in the loss curve would say so.
#
# So every candidate is OPENED and checked before it is trusted. A checkpoint qualifies
# only if its own recorded args match the run this notebook is about to launch.
import torch
# Epoch budgets differ per head, so one WANT_EPOCHS is wrong: the ADL streams run 80 and
# the fall head runs 60. A single value rejected every legitimate fall checkpoint with
# "epochs=60, this run wants 80" - a false alarm on exactly the resume path this cell
# exists to serve. Keyed by the run directory name that the training cells write to.
# Budgets live HERE and the training cells read them, so the two cannot drift. They did:
# the ADL budget was cut 80 -> 30 after measuring that every stream peaked at epoch 8-11,
# and this guard was left at 80. It would have accepted the old 80-epoch checkpoints,
# then train_adl.py's own resume guard would have refused them ("--epochs was 80 but is
# 30 now") and killed all four streams at startup.
EPOCH_BUDGET = {"adl": 30, "fall": 60}
PATIENCE = {"adl": 8, "fall": 15}
WANT_EPOCHS = EPOCH_BUDGET

# Split identity for the ADL streams. The shards store VIDEO ids; Charades_v1_train.csv
# maps them to its 267 real actors, making the split person-disjoint. A checkpoint trained
# under one identity CANNOT resume under the other: the optimiser saw different windows, so
# carrying it forward would smuggle the old split's leakage into the new protocol. The
# acceptance check below therefore compares subject_map presence, not just epochs.
SUBJECT_MAP = find_charades_csv()
SUBJECT_MAP = str(SUBJECT_MAP[0]) if SUBJECT_MAP else None
print("ADL split identity:", "actor-id (Charades subjects)" if SUBJECT_MAP
      else "video-id proxy - attach charades-480p annotations for a person-disjoint P1")

def expected_epochs(run_name):
    # runs/adl_joint, runs/adl_bone_motion, ... -> "adl";  runs/fall -> "fall"
    return WANT_EPOCHS.get(run_name.split("_", 1)[0])

rejected = {}

def acceptable(ck_path):
    try:
        meta = torch.load(ck_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        rejected[str(ck_path)] = f"unreadable ({type(exc).__name__})"
        return False
    args = meta.get("args") or {}
    if args.get("smoke") or args.get("smoke_shard"):
        rejected[str(ck_path)] = "smoke run (synthetic fixture, not real training)"
        return False
    want = expected_epochs(ck_path.parent.name)
    if want is None:
        rejected[str(ck_path)] = (f"unknown run directory {ck_path.parent.name!r} - this "
                                  "notebook only writes runs/adl_* and runs/fall")
        return False
    # A checkpoint trained for a different total changes the LR schedule on resume, and
    # train_adl.py refuses it anyway - better to skip it here than fail four streams in.
    if args.get("epochs") not in (None, want):
        rejected[str(ck_path)] = f"epochs={args.get('epochs')}, this run wants {want}"
        return False
    # Split-identity match, ADL runs only (the fall corpora already use real people).
    # bool-compare rather than path-compare: the same CSV mounts at a different absolute
    # path per session, and what changes the split is WHETHER it was used, not where from.
    if ck_path.parent.name.startswith("adl"):
        had_map = bool(args.get("subject_map"))
        if had_map != bool(SUBJECT_MAP):
            rejected[str(ck_path)] = (
                f"split identity mismatch: checkpoint was trained "
                f"{'actor-disjoint' if had_map else 'video-disjoint'}, this run is "
                f"{'actor-disjoint' if SUBJECT_MAP else 'video-disjoint'} - resuming would "
                "carry the other split's train/val boundary into this protocol")
            return False
    return True

ckpts = [c for c in sorted(INPUT.glob("**/last.pt")) if acceptable(c)]
if rejected:
    print(f"ignoring {len(rejected)} checkpoint(s) that are not resumable here:")
    for q, why in sorted(rejected.items()):
        print(f"  {pathlib.Path(q).parent.name:<14} {why}")
if ckpts:
    # Copy the ACCEPTED run directories one at a time. Copying their common parent would
    # drag rejected siblings along - the smoke checkpoints sat beside real ones, so a
    # parent-level copytree reinstates exactly what acceptable() just refused.
    for ck_path in ckpts:
        shutil.copytree(ck_path.parent, RUNS / ck_path.parent.name, dirs_exist_ok=True)
    found = sorted(q.parent.name for q in RUNS.glob("*/last.pt"))
    assert found, f"copied {len(ckpts)} checkpoint(s) but none landed in {RUNS}"
    print(f"resuming: {found}")
else:
    print("fresh training - no resumable last.pt under any attached dataset")"""),
("code", """# Train the 4 ADL streams. 96 GB VRAM fits all four CONCURRENTLY (each peaks at a few
# GB at batch 512 bf16) - wall-clock becomes the slowest stream instead of the sum.
# Set PARALLEL=False to serialise if anything OOMs.
import os, subprocess, sys, pathlib
PARALLEL = True
# 80 -> 30. Measured, not guessed: on the real Charades shards all four streams peaked
# at epoch 8-11 and then LOST ~27% mean-class-accuracy over the remaining 70 epochs
# (adl_bone 0.184@ep11 -> 0.137@ep79). 80 came from ST-GCN++'s NTU-60 recipe, which has
# ~10x more labelled windows per class. ~2.4 of the 2.7 GPU-hours went into memorising.
# --patience 8 ends each stream once it stops improving; best.pt already holds the peak,
# so this cannot cost a result - simulated against the real curve it stops at ep19.
EPOCHS, BATCH, SEED = str(EPOCH_BUDGET["adl"]), "512", "0"
STREAMS = ["joint", "bone", "joint_motion", "bone_motion"]
env = {**os.environ, "PYTHONPATH": str(SRC)}

# Class-imbalance strategy. mean-class accuracy is the metric of record and the label map
# leaves other_idle at 39.4%, so the objective matters. Two options, and they are mutually
# exclusive - train_adl.py refuses the combination because stacking them corrects the
# imbalance twice:
#   ("balanced", 0.0)  effective-number sampling, plain CE. What P1's 0.189 came from.
#   ("natural", 1.0)   natural batches, logit-adjusted loss (Menon et al. 2021).
# Effective-number weighting saturates by design: at 65,000 vs 69 windows it removes ~145x
# of a ~942x ratio, so residual imbalance survives it. Try the post-hoc adjustment on the
# EXISTING checkpoints first - scripts/rescore_p1.py does it on CPU in seconds off the saved
# val logits - and only spend a training run here if that is not enough.
SAMPLER, TAU_TRAIN = "balanced", 0.0

def launch(stream):
    out = f"/kaggle/working/runs/adl_{stream}"
    cmd = [sys.executable, str(SCRIPTS / "train_adl.py"),
           "--shards", *ADL, "--stream", stream, "--epochs", EPOCHS,
           "--batch-size", BATCH, "--device", "cuda", "--seed", SEED,
           "--patience", str(PATIENCE["adl"]), "--sampler", SAMPLER,
           "--tau-train", str(TAU_TRAIN), "--out", out]
    if SUBJECT_MAP:
        cmd += ["--subject-map", SUBJECT_MAP]   # person-disjoint split (see resolver cell)
    if pathlib.Path(out, "last.pt").exists():
        cmd += ["--resume", f"{out}/last.pt"]
    log = open(f"/kaggle/working/train_{stream}.log", "a")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)

if PARALLEL:
    procs = {s: launch(s) for s in STREAMS}
    RC = {s: p.wait() for s, p in procs.items()}
else:
    RC = {s: launch(s).wait() for s in STREAMS}
for s, rc in RC.items():
    print(s, "->", "OK" if rc == 0 else f"FAILED (see train_{s}.log)")
TRAINED_OK = sum(rc == 0 for rc in RC.values())"""),
("code", """# Fall head (focal loss, operating point fitted at a false-alarm budget - never 0.5).
#
# BOTH shard sets, not just the fall corpora. train_fall.py derives its binary target
# from labels 7/8, so every ADL window is simply a negative - and the fall shards alone
# are 81.6% POSITIVE (2501 fall / 563 non-fall as extracted). Two things break if the
# ADL shards are withheld:
#
#   1. The operating point stops meaning anything. fit_operating_point expresses the
#      budget as false alarms per HOUR of monitored video, and 563 negative windows is
#      0.31 h - 0.08 h after the val split. A budget of "1 FA/hour" then permits 0.08 of
#      one alarm, so the fitted threshold is whatever admits zero, which is an artefact
#      of test-set size rather than a deployment choice. Measured on a synthetic detector
#      of fixed quality: sensitivity 0.65 at 141 negatives vs 0.37 at a realistic mix,
#      AUPRC 0.993 vs 0.790. The same model, three different stories.
#   2. AUPRC is inflated for the same reason - it is the honest summary only when the
#      positive rate resembles deployment, and 81.6% positive resembles nothing.
#
# The ADL shards contribute ~165k negatives, which puts the positive rate near 1.5% -
# still optimistic against a real home, but the right side of realistic.
import subprocess, sys, os, pathlib
out = "/kaggle/working/runs/fall"
FALL_TRAIN = [*FALL, *ADL]
assert ADL, "ADL shards absent - the fall head would train on an 82% positive rate"
cmd = [sys.executable, str(SCRIPTS / "train_fall.py"),
       "--shards", *FALL_TRAIN, "--epochs", str(EPOCH_BUDGET["fall"]),
       "--batch-size", "256",
       # 60 epochs KEPT: unlike the ADL heads, the measured fall run peaked at ep56 of
       # 60 - still climbing near the end. Patience is wider for the same reason.
       "--device", "cuda", "--seed", "0", "--patience", str(PATIENCE["fall"]), "--out", out]
if pathlib.Path(out, "last.pt").exists():
    cmd += ["--resume", f"{out}/last.pt"]
r = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": str(SRC)})
FALL_OK = r.returncode == 0
print("fall head:", "OK" if FALL_OK else "FAILED")"""),
("code", """# Session summary from each run's history.json - what to paste into the log book.
#
# Look in BOTH places, and REFUSE to print nothing. /kaggle/working is wiped between
# sessions, so re-running this cell in a fresh session finds an empty directory - the
# glob then yields zero iterations, the loop body never executes, and the cell ends by
# printing "Save Version" as though all were well. That is the same silent-empty-iteration
# failure this project has already fixed in notebooks 00, 01 and 02; it survived here.
#
# A completed batch run keeps its /kaggle/working as the version output, so attach that
# output (behaviorsense-runs) and the files are found under /kaggle/input instead.
import json, pathlib
def _real_runs(paths):
    # The code dataset is an upload of the working tree, so it carries the LOCAL runs/
    # directory - CPU smoke checkpoints (`--smoke`, 25 epochs) sitting beside the real
    # ones. The carry-forward cell already refuses to RESUME those; the summary was still
    # reading them and reporting a 25-epoch toy run as a result. Same exclusion, applied
    # to reporting: anything under a mounted code checkout, or named for a fixture.
    out = []
    for q in paths:
        parts = {part.lower() for part in q.parts}
        if "smoke" in q.parent.name.lower():
            continue
        if any("behaviorsense-code" in part or "emotionsense" in part for part in parts):
            continue
        out.append(q)
    return out

HISTS = sorted(pathlib.Path("/kaggle/working/runs").glob("*/history.json"))
WHERE = "/kaggle/working/runs"
if not HISTS:
    HISTS = _real_runs(sorted(pathlib.Path("/kaggle/input").glob("**/runs/*/history.json")))
    WHERE = "attached dataset"
if not HISTS:
    HISTS = _real_runs(sorted(pathlib.Path("/kaggle/input").glob("**/history.json")))
assert HISTS, (
    "no history.json anywhere. /kaggle/working is empty in a fresh session - the training "
    "outputs live in the completed run's version output. Attach that dataset "
    "(behaviorsense-runs) as an input, or re-run this notebook from the top. "
    "Searched: /kaggle/working/runs, then /kaggle/input/**")
print(f"reading {len(HISTS)} run(s) from {WHERE}")
for hist in HISTS:
    h = json.loads(hist.read_text())
    if not h:
        continue
    last = h[-1]
    # The key is `mean_class_acc`, not `mca`. Looking for the wrong name made every ADL
    # stream report `best=0.000` via max()'s default - the checkpoints were fine, the
    # summary was lying. A "best" that is exactly 0.000 for four independent runs is the
    # tell: real training noise never lands on precisely zero.
    # Identify the run by the metrics it RECORDED, not by its directory name. The name
    # heuristic broke on `smoke_fall` - a fall run whose name does not start with "fall" -
    # and asserted for `mean_class_acc` against a history that only has AUPRC. What a run
    # is, is what it measured.
    metric = ("auprc" if "auprc" in last else
              "mean_class_acc" if "mean_class_acc" in last else None)
    if metric is None:
        print(f"  {hist.parent.name:<18} SKIPPED - no recognised metric "
              f"(keys: {sorted(last)[:6]})")
        continue
    shown = [k for k in ("mean_class_acc", "macro_f1", "macro_f1_supported", "top1",
                         "auprc", "auroc", "sensitivity") if k in last]
    best = max(e[metric] for e in h if metric in e)
    best_ep = next(e.get("epoch", i) for i, e in enumerate(h)
                   if e.get(metric) == best)
    print(f"{hist.parent.name:<18} epoch {last.get('epoch', len(h)-1):>3}  "
          + "  ".join(f"{k}={last[k]:.3f}" for k in shown)
          + f"  | best {metric}={best:.3f} @ep{best_ep}")

# Per-class breakdown for the ADL streams. macro-F1 alone cannot distinguish "the task is
# hard" from "the pipeline is broken", and those need opposite responses. The vector is
# already in history.json (per_class_f1 / per_class_support) - it just was never printed.
#
# Read it like this:
#   a broad low-but-nonzero spread   -> genuinely hard (Charades is untrimmed and
#                                       multi-label; a 2 s window often contains no
#                                       evidence of the labelled action at all)
#   a few classes at ~0, rest fine   -> label-map starvation, already documented
#   everything ~0 except other_idle  -> collapse; the signal is being destroyed upstream
CLASS_NAMES = ["walking", "standing", "sitting", "lying_down", "standing_up",
               "sitting_down", "bending_reaching", "falling", "fallen_on_ground",
               "eating", "drinking", "cooking_food_prep", "taking_medication",
               "watching_tv", "reading", "using_phone", "cleaning_housework",
               "personal_hygiene", "interacting_with_person", "other_idle"]
for hist in [q for q in HISTS if q.parent.name.startswith("adl")]:
    h = json.loads(hist.read_text())
    best = max(h, key=lambda e: e.get("mean_class_acc", 0))
    f1s, sup = best.get("per_class_f1"), best.get("per_class_support")
    if not f1s:
        continue
    print()
    print(f"{hist.parent.name} - per-class F1 at best epoch {best.get('epoch')}:")
    rows = sorted(((f1s[i], sup[i], CLASS_NAMES[i]) for i in range(len(f1s)) if sup[i]),
                  reverse=True)
    for f1, n, name in rows:
        bar = "#" * int(f1 * 40)
        print(f"  {name:<24} n={n:>6}  f1={f1:.3f}  {bar}")
    dead = [name for f1, n, name in rows if f1 < 0.02]
    print(f"  -> {len(rows) - len(dead)}/{len(rows)} classes above F1 0.02; "
          f"{len(dead)} at ~0: {dead[:6]}")
    break            # one stream is enough to diagnose; they share the same data

print()
# Do NOT tell the user to publish a session that trained nothing. This cell used to print
# "Save Version -> create/update dataset behaviorsense-runs" unconditionally, and it printed
# it after a session where all five trainers had crashed on startup: /kaggle/working/runs
# held byte-identical carry-forward copies, and the only new files were four crash
# tracebacks in train_*.log - which a new dataset version would have published OVER the real
# training logs. Advice that is wrong in exactly the situation where it is most likely to be
# followed is worse than no advice.
_ok = globals().get("TRAINED_OK", 0) + int(globals().get("FALL_OK", False))
if _ok:
    print(f"{_ok} run(s) advanced this session.")
    print("Save Version -> create/update dataset behaviorsense-runs from /kaggle/working/runs")
    print("Timed out mid-training? Save Version anyway - resume continues bit-exactly next "
          "session.")
else:
    print("NOTHING TRAINED this session - every run failed to start or had already"
          " finished.")
    print("Do NOT publish a new behaviorsense-runs version: /kaggle/working/runs holds"
          " carry-forward copies")
    print("of the checkpoints you already have, and train_*.log here contains only this"
          " session's errors,")
    print("which would replace the real training logs in the existing dataset.")
    print("Fix the cause above, or go straight to notebook 04 - best.pt already holds every"
          " peak.")"""),
]

# ===========================================================================
# 04 - evaluation (OFFLINE, Blackwell)
# ===========================================================================

N04: list[tuple[str, str]] = [
("markdown", """# 04 — Evaluation: P1 / P2 / ablations / Qwen hallucination (OFFLINE, Blackwell)

| attach as input | produces |
|---|---|
| `behaviorsense-code`, `behavioursense-WW`, `behaviorsense-adl-shards`, `behaviorsense-fall-shards`, `behaviorsense-runs` | `results/evaluation.md` + `results/hallucination_qwen.md` in the output (publish as `behaviorsense-results`) |
| Kaggle Model: **Qwen2.5-7B-Instruct** (attach via Add Input -> Models; no upload needed) | |

Protocols: **P1** subject-disjoint validation (same split seed as training, so these
windows influenced training only via early stopping); **P2** leave-one-dataset-out on the
fall sources; per-stream and combination **ablations**; the **hallucination table** with
the stub anchors from `results/hallucination.md` giving the 0%/injection-rate calibration
that makes the Qwen numbers interpretable.

Two things changed after the first complete run of this notebook:

1. **Every table is written to `results/evaluation.md`.** The previous bundling cell
   globbed `results/*.md` and found only the file a subprocess had written, so nine hours
   of P1 / per-class / calibration / P2 / ablation output existed solely as session
   scrollback. The session log is not a record.
2. **The hallucination table has three arms, two denominators and a confidence interval.**
   Run 1 scored the free arm as 245 claims emitted / 0 scorable / `nan%`, and the
   constrained arm at 39.9%. Both were artefacts of a prompt that never named the output
   fields: the free model could not guess the envelope, and the constrained model
   mis-assigned values. With the mapping stated, run 2 gave 8.2% / 6.5% and zero schema
   rejections in either arm — so most of that 39.9% was our prompt.
3. **Both arms are now pinned to greedy decoding.** Runs 2 and 3 disagreed: the free arm was
   bit-identical (509 claims, 476 faithful, twice) while the constrained arm moved 498/457 →
   504/452 and the verdict flipped from p=0.29 to p=0.03. The free path passed
   `do_sample=False`; the constrained path passed only `max_new_tokens` to `outlines` and
   inherited Qwen's `generation_config` (`do_sample=True, temperature=0.7`). One arm was
   greedy, the other sampled, so the gap measured temperature rather than grammar.
   `force_greedy()` pins both — **this is the first decoding-matched run, and the earlier
   constrained numbers should not be quoted.**"""),
("code", RESOLVE),
("code", """import subprocess, sys
def sm120_ok():
    try:
        import torch
        return torch.cuda.is_available() and "sm_120" in torch.cuda.get_arch_list()
    except Exception:
        return False
if not sm120_ok():
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                    "--find-links", str(WHEELS),
                    "torch", "numpy", "pydantic", "PyYAML", "Pillow"], check=True)
    print("RESTART the kernel, then continue from the next cell.")
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                "--find-links", str(WHEELS),
                "transformers", "accelerate", "outlines", "safetensors"], check=True)
# sys.path is set by the resolver cell now, unconditionally, for every offline notebook.
# It used to be set here, which made an import path depend on a wheel-install cell."""),
("code", """# P1 - subject-disjoint mean-class accuracy / macro-F1, per stream and ensemble.
#
# val_frac=0.2 and seed=0 are train_adl.py's OWN defaults, so this is exactly the set the
# streams were early-stopped against - not a re-split. It previously used val_frac=0.15,
# which `split_by_subject` makes a strict SUBSET (it shuffles subjects by seed and takes a
# prefix), so there was never leakage - but the comment claimed to "reproduce the
# training-time val split" and did not, and the subset threw away a fifth of the evidence
# for nothing. These windows influenced training only through early stopping and best.pt
# selection, which is what makes every number here validation-selected rather than held-out.
#
# Eval windows are built from the RAW shard arrays with normalise() only. Two traps this
# avoids: ds[i] returns [C,T,V,M] (already permuted for the model) whereas
# EnsembleClassifier.logits() wants the dataset layout [N,T,M,17,3]; and ds[i] would also
# apply temporal_resample jitter. Augmentation is already off (augment_cfg=None ->
# AugmentConfig(enabled=False)), but building eval inputs explicitly is what makes that
# guarantee visible instead of assumed.
import numpy as np
from behaviorsense.kaggle_artifacts import is_real_artifact
from behaviorsense.data.skeleton_dataset import (SkeletonWindowDataset, split_by_subject,
                                                 normalise, N_CLASSES)
from behaviorsense.models.ensemble import EnsembleClassifier
from behaviorsense.agents.activity import CLASS_NAMES
# MIN_SUPPORT and scores() live in the library so this cell, scripts/rescore_p1.py and the
# tests share ONE definition. They used to be inline here, which meant any test of the
# metric validated its own copy - the pattern behind three notebook-04 defects.
from behaviorsense.eval.activity_eval import MIN_SUPPORT, scores

def shard_paths(*keywords):
    # Match on ANY path component, not the top-level mount name: attaching by URL nests
    # the dataset under /kaggle/input/datasets/<owner>/<name>/, so iterating the top level
    # finds only 'datasets' and returns nothing. Cost notebook 03 a session.
    out = []
    for q in INPUT.glob("**/*.npz"):
        if not is_real_artifact(q):
            continue          # fixture, or a leftover inside a mounted code checkout
        rel = [part.lower().replace("behaviour", "behavior") for part in q.parts]
        if any(k in part for part in rel for k in keywords):
            out.append(str(q))
    return sorted(out)

# find_run_dir picks the directory that actually holds adl_<stream>/ subdirectories,
# and skips anything inside a mounted code checkout. The previous rule took the first
# last.pt in sorted order, which selected behaviorsense-code/.../runs ("code" sorts
# before "runs") - a directory holding adl/ and fall/, not adl_joint/. Notebook 04 then
# died reporting a missing checkpoint when the real fault was the wrong directory.
from behaviorsense.kaggle_artifacts import find_run_dir

def run_dir():
    return str(find_run_dir(INPUT))

ADL = shard_paths("adl")
assert ADL, f"no ADL shards found. Attached: {ATTACHED}"
ds = SkeletonWindowDataset([*map(str, ADL)])

# The shards store VIDEO ids as `subjects` - docs/07 said "Charades subject ids do not
# exist publicly", and that was wrong: Charades_v1_train.csv has a `subject` column (267
# actors, ~30 videos each). A video-id split puts nearly every actor on both sides, so
# "subject-disjoint" P1 was video-disjoint: same person, same home, train and val. When
# the CSV is attached (charades-480p carries it), the split is remapped to ACTOR ids and
# P1 becomes person-disjoint. MUST match how the checkpoints were trained - the split
# mode is printed and recorded next to the numbers, because a person-disjoint eval of
# video-disjoint checkpoints reports leakage-free numbers for a leaky model.
from behaviorsense.data.skeleton_dataset import load_subject_map, remap_subjects
_csv = find_charades_csv()
split_subjects, SPLIT_MODE = ds.subjects, "video-id (proxy)"
if _csv:
    split_subjects, _cov = remap_subjects(ds.subjects, load_subject_map(_csv[0]))
    if _cov >= 0.5:
        SPLIT_MODE = f"actor-id ({len(set(split_subjects))} actors, {_cov:.0%} mapped)"
    else:
        split_subjects = ds.subjects
        print(f"subject CSV found but covers only {_cov:.1%} of windows - keeping video ids")

# REFUSE the leaky protocol unless it is asked for explicitly.
#
# This used to fall back to video ids with one printed line, and that is exactly what
# happened: a session ran without `charades-480p` attached, printed "P1 split: video-id
# (proxy)", and produced macro-F1 0.256 / mean-class 0.281 / T=0.58 - numbers that look
# like a large improvement over the actor-disjoint 0.128 / 0.156 / 0.63 and are in fact
# the leak this project spent a session removing. Two hours of Blackwell time, and the
# output is unquotable.
#
# The split is the protocol of record (results/evaluation.md), so producing the other one
# has to be a decision, not an accident. Set BS_ALLOW_VIDEO_SPLIT=1 to override.
import os
if SPLIT_MODE.startswith("video-id") and not os.environ.get("BS_ALLOW_VIDEO_SPLIT"):
    raise AssertionError(
        "P1 would run on the VIDEO-ID split, which leaks actors between train and val "
        "(~30 videos per actor, so nearly every person appears on both sides). "
        "`Charades_v1_train.csv` was not found under /kaggle/input.\\n\\n"
        "  FIX: attach the `charades-480p` dataset - it carries the CSV.\\n"
        "  Every activity number produced without it is inflated and must not be quoted "
        "next to the actor-disjoint tables in results/evaluation.md.\\n"
        "  To measure the leaky protocol deliberately, set BS_ALLOW_VIDEO_SPLIT=1."
    )
print(f"P1 split: {SPLIT_MODE}")
_, val_idx = split_by_subject(split_subjects, val_frac=0.2, seed=0)
print(f"P1 val: {len(val_idx)} windows, {len(set(split_subjects[val_idx]))} subjects")
assert not (set(split_subjects[val_idx]) & set(np.delete(split_subjects, val_idx))), \\
    "subject leakage between train and val"

clf = EnsembleClassifier.from_run_dir(run_dir(), device="cuda")
X = np.stack([normalise(ds.skeletons[i].astype(np.float32)) for i in val_idx])
y = ds.labels[val_idx]
assert X.shape[1:] == (30, 2, 17, 3), f"unexpected eval window layout {X.shape}"

# MIN_SUPPORT mirrors train_adl.py: below ~50 val windows a class's F1 is one prediction
# wide, so it is reported but not averaged. This matters concretely here - the real
# extraction (7,985 videos) yields standing 342 windows (0.21%) and bending_reaching 69
# (0.04%), because Charades calls those "Putting a box somewhere" / "Taking a bag from
# somewhere" and no keyword rule matches, so they land in other_idle (39.4%). Their F1 is
# ~0 and over 18 present classes that alone costs ~6 macro-F1 points. A single macro-F1
# number would make the ensemble look worse than it is with no way to see why, so the
# table carries BOTH and the starved classes are named underneath.

support = np.bincount(y, minlength=N_CLASSES)
starved = [(c, int(support[c])) for c in range(N_CLASSES) if 0 < support[c] < MIN_SUPPORT]
n_sup = int((support >= MIN_SUPPORT).sum())

per_stream = clf.per_stream_logits(X)
P1_ROWS = []
print(f"{'model':<16} {'top1':>6} {'mean-class':>10} {'macro-F1':>9} "
      f"{'F1>=' + str(MIN_SUPPORT):>9}")
for s, lg in per_stream.items():
    t1, mca_s, f1_s, f1_sup_s = scores(lg, y)
    P1_ROWS.append((s, float(t1), float(mca_s), float(f1_s), float(f1_sup_s)))
    print(f"{s:<16} {t1:>6.3f} {mca_s:>10.3f} {f1_s:>9.3f} {f1_sup_s:>9.3f}")
ens_logits = clf.logits(X)
t1, mca, f1, f1_sup = scores(ens_logits, y)
P1_ROWS.append(("ENSEMBLE", float(t1), float(mca), float(f1), float(f1_sup)))
print(f"{'ENSEMBLE':<16} {t1:>6.3f} {mca:>10.3f} {f1:>9.3f} {f1_sup:>9.3f}"
      "   <- headline P1")
print(f"\\nmacro-F1 averages {int((support > 0).sum())} present classes; the last column "
      f"averages the {n_sup} with >= {MIN_SUPPORT} val windows.")
if starved:
    # Named, not silently dropped: the gap between the two columns is entirely these
    # classes, and it is a LABEL-MAP limitation to state in the write-up, not a model
    # result. Neither feeds a downstream Agent 3 feature, so it does not affect the
    # behaviour layer - which is the reason for reporting rather than re-extracting.
    print(f"{len(starved)} class(es) starved by the Charades label map "
          f"(< {MIN_SUPPORT} windows):")
    for c, n in starved:
        print(f"  class {c:>2} {CLASS_NAMES[c]:<22} support {n:>5}")
    print("  Cause: Charades has no verb for these (e.g. 'Putting a box somewhere'),")
    print("  so build_charades_map.py routes them to other_idle. Report both columns.")

# Persist the val logits. ~8 MB, and it converts a whole class of accuracy experiments -
# logit adjustment for the long-tailed label distribution, stream-subset selection,
# probability-vs-logit combination, per-class thresholds - from "book another 2-hour Kaggle
# session" into "run scripts/rescore_p1.py on a laptop". P1 computed exactly these arrays
# four times across four sessions and discarded them four times.
import pathlib, os
_RES = pathlib.Path(os.environ.get("BS_RESULTS_DIR", "/kaggle/working/results"))
_RES.mkdir(parents=True, exist_ok=True)
np.savez_compressed(_RES / "val_logits.npz", y=y.astype(np.int64),
                    subjects=ds.subjects[val_idx].astype("<U32"),
                    logits_ensemble=ens_logits.astype(np.float32),
                    **{f"logits_{s}": lg.astype(np.float32)
                       for s, lg in per_stream.items()})
print(f"wrote {_RES / 'val_logits.npz'} - re-score without a GPU: "
      f"PYTHONPATH=src python scripts/rescore_p1.py results/val_logits.npz")"""),
("code", """# Accuracy levers, measured. Three of the four are pure post-processing on the logits
# already computed above, so they cost seconds and no GPU; TTA costs one extra forward pass.
#
# Why each is here rather than assumed:
#  - LOGIT ADJUSTMENT: training uses effective-number balanced sampling, which removes only
#    ~145x of the ~942x head/tail ratio, so residual imbalance remains and mean-class is the
#    metric of record. tau is SWEPT rather than derived, with tau=0 as the control - the
#    right correction depends on how much imbalance the sampler left, which is not something
#    to guess.
#  - STREAM SUBSET: P1 shows the 4-stream average LOSING to bone alone on mean-class. So
#    the full average is not automatically right, and the subset question has 15 answers.
#  - TTA: mirror each window and average logits. AugmentConfig(flip_prob=0.5) means the
#    model trained on mirrored windows, so this averages two views it has seen.
from behaviorsense.eval.activity_eval import subset_scores, tau_sweep, combine, logit_adjust, class_prior
LEVERS = []
print(f"{'configuration':<34} {'top1':>6} {'mean-class':>11} {'macro-F1':>9}")
def lever(label, lg):
    s = scores(lg, y)
    LEVERS.append((label, *s))
    print(f"{label:<34} {s[0]:>6.3f} {s[1]:>11.3f} {s[2]:>9.3f}")
    return s

base = lever("ensemble, logit-avg (P1 headline)", ens_logits)

# 1. Stream subsets, both combination rules.
best_sub = None
for mode in ("logit", "prob"):
    rows = subset_scores(per_stream, y, mode=mode)
    top = rows[0]
    if best_sub is None or top[2] > best_sub[1][2]:
        best_sub = (mode, top)
    lever(f"best {mode}-avg subset: {'+'.join(top[0])}", combine(per_stream, top[0], mode))

# 2. Logit adjustment on the headline ensemble and on the best subset.
prior = class_prior(y)
sub_logits = combine(per_stream, best_sub[1][0], best_sub[0])
for label, lg in (("ensemble", ens_logits), (f"{'+'.join(best_sub[1][0])}", sub_logits)):
    sweep = tau_sweep(lg, y)
    assert abs(sweep[0][2] - scores(lg, y)[1]) < 1e-9, "tau=0 is not a no-op"
    peak = max(sweep, key=lambda r: r[2])
    lever(f"{label} + logit-adjust tau={peak[0]:.2f}", logit_adjust(lg, prior, peak[0]))

# 3. Test-time flip augmentation on the ensemble.
clf.tta = True
tta_logits = clf.logits(X)
clf.tta = False
lever("ensemble + TTA flip", tta_logits)

# 4. Second-person context - the one object signal that needs NO detector. The shards
# carry two person slots; slot 1 is non-zero exactly when the tracker held a second
# person, and interacting_with_person (F1 0.042, second-worst class) fails precisely
# because pose alone cannot say "someone else is here". OBJECT_PRIORS['person'] (+1.5
# log-odds on class 18) was written for this signal and has never received it. Applied
# via fuse_objects so serving and evaluation share one code path. NOTE: uncalibrated
# posteriors on purpose - T is fitted in the NEXT cell, and log-odds offsets commute with
# argmax under any temperature, so ordering does not change the decision rule.
from behaviorsense.eval.activity_eval import second_person_present
from behaviorsense.agents.activity import ActivityAgent, ActivityConfig, softmax
_second = second_person_present(ds.skeletons[val_idx])
print(f"second person visible in {int(_second.sum()):,}/{len(_second):,} val windows "
      f"({_second.mean():.1%}); class 18 support {int((y == 18).sum()):,}")
_agent = ActivityAgent(ActivityConfig())
_objs = [("person",) if s else () for s in _second]
fused = np.log(np.clip(_agent.fuse_objects(softmax(ens_logits), _objs), 1e-12, None))
lever("ensemble + second-person context", fused)
adj_fused = np.log(np.clip(_agent.fuse_objects(
    softmax(logit_adjust(ens_logits, prior, 0.25)), _objs), 1e-12, None))
lever("ens + logit-adjust 0.25 + 2nd-person", adj_fused)
# Recall/precision on class 18 specifically: a +1.5 log-odds prior helps only if the
# second-person windows are actually where class 18 lives. Print the confusion so the
# lever is diagnosable, not just a delta in an average.
for label, lg in (("without", logit_adjust(ens_logits, prior, 0.25)), ("with", adj_fused)):
    p18 = lg.argmax(1) == 18
    tp = int((p18 & (y == 18)).sum())
    print(f"  class 18 {label} context: predicted {int(p18.sum()):>5}, correct {tp:>4}, "
          f"recall {tp / max(1, int((y == 18).sum())):.3f}")

best = max(LEVERS, key=lambda r: r[2])
print()
print(f"best mean-class: {best[0]} at {best[2]:.3f} "
      f"({best[2] - base[1]:+.3f} vs the P1 headline)")
print("Selected on the validation split, so report as validation-selected, not held-out.")"""),
("code", """# Calibration: fit the temperature on these val logits. The fitted T goes into
# ActivityConfig(temperature=...) at serving time - Viterbi and abstention both consume
# probabilities, so they must mean something first.
from behaviorsense.agents.activity import fit_temperature, softmax
T = fit_temperature(ens_logits, y)
conf = softmax(ens_logits / T).max(1).mean()
acc = (ens_logits.argmax(1) == y).mean()
print(f"fitted temperature T={T:.2f}; mean confidence {conf:.3f} vs accuracy {acc:.3f}")
print(f"-> set ActivityConfig(temperature={T:.2f}) in deployment")"""),
("code", """# SEGMENT-level accuracy under Viterbi smoothing - the metric Agent 3 actually consumes.
#
# Every number above is per-WINDOW. Agent 3 never sees a window: it sees segments, built by
# smoothing the posterior sequence with a transition prior, and derives every daily feature
# from segment durations. So per-window accuracy is not the deployment metric.
#
# The first time this was measured it found a REGRESSION: under the hand-set prior
# (self_transition=0.90, uniform leakage) smoothing moved top-1 +0.011 and mean-class
# -0.040. A dominant 39% other_idle plus a sticky self-transition swallows short rare-class
# runs into the surrounding majority - it flatters top-1 and destroys tail recall, the same
# head/tail trade logit averaging makes. Two fixes are measured here against that baseline:
#   1. FIT the transition matrix from TRAINING label sequences instead of asserting it. The
#      structural prior was written when no labelled sequence data existed; there are now
#      165k windows in temporal order. Fitted on train only, evaluated on val.
#   2. Smooth the LOGIT-ADJUSTED posterior rather than the raw one. Smoothing the
#      configuration that already collapses onto the head class is the worst case for it.
# The emergency floor into `falling` is re-applied to any fitted matrix: falls are 0.5% of
# windows, so a counted matrix makes the state nearly unreachable and one-window falls are
# smoothed out of existence.
import numpy as np
from behaviorsense.agents.activity import (ActivityConfig, FALLING, build_transition_matrix,
                                           viterbi)
from behaviorsense.eval.activity_eval import (apply_emergency_floor, fit_transition_matrix,
                                              fragmentation, sequences_by_subject)
cfg_sm = ActivityConfig(temperature=float(T))
train_idx, _ = split_by_subject(ds.subjects, val_frac=0.2, seed=0)
A_fit = apply_emergency_floor(
    fit_transition_matrix(sequences_by_subject(ds.subjects[train_idx], ds.labels[train_idx])),
    FALLING, cfg_sm.emergency_floor)
log_pi = np.log(np.full(N_CLASSES, 1.0 / N_CLASSES))
subj = ds.subjects[val_idx]
y_seqs = sequences_by_subject(subj, y)
n_seq = len(y_seqs); n_win = sum(len(s) for s in y_seqs)
print(f"{n_seq} sequences, {n_win} windows (sequences of >=3 windows only)")
print(f"fitted prior: mean self-transition {np.mean(np.diag(A_fit)):.3f} "
      f"vs hand-set {cfg_sm.self_transition:.2f}")

_prior = class_prior(y)
_tau = max(tau_sweep(ens_logits, y), key=lambda r: r[2])[0]
SEGMENT_ROWS = []
def seg_eval(label, logits, A):
    post = softmax(logits / float(T))
    seqs = sequences_by_subject(subj, post)
    hit = 0; pc = np.zeros((N_CLASSES, 2), dtype=np.int64); preds = []
    for sy, sp in zip(y_seqs, seqs):
        pred = sp.argmax(1) if A is None else viterbi(
            np.log(np.clip(sp, 1e-12, None)), np.log(A), log_pi)
        preds.append(pred)
        hit += int((pred == sy).sum())
        for c, p in zip(sy, pred):
            pc[c, 0] += int(p == c); pc[c, 1] += 1
    present = pc[:, 1] > 0
    mca_ = float(np.mean(pc[present, 0] / pc[present, 1]))
    frag = fragmentation(preds, y_seqs)
    SEGMENT_ROWS.append((label, hit / n_win, mca_, frag["ratio"]))
    print(f"{label:<40} {hit / n_win:>6.3f} {mca_:>11.3f} {frag['ratio']:>9.2f}")

# Accuracy alone cannot decide argmax vs Viterbi. Agent 3 derives walking_bouts and
# mean_bout_duration_s from segment COUNTS, so a decoding that shatters one true stretch into
# nine flickering ones reports nine bouts while scoring identically per window. The
# fragmentation ratio (predicted segments / true segments, 1.0 ideal) is what smoothing is
# actually for, and without it "argmax wins on mean-class" is half an argument.
print(f"{'decoding':<40} {'top1':>6} {'mean-class':>11} {'frag':>9}")
A_hand = build_transition_matrix(cfg_sm)
adj = logit_adjust(ens_logits, _prior, _tau)
seg_eval("per-window argmax", ens_logits, None)
seg_eval("Viterbi, hand-set prior", ens_logits, A_hand)
seg_eval("Viterbi, fitted prior", ens_logits, A_fit)
seg_eval(f"argmax, logit-adjust tau={_tau:.2f}", adj, None)
seg_eval(f"Viterbi fitted + logit-adjust tau={_tau:.2f}", adj, A_fit)
_best = max(SEGMENT_ROWS, key=lambda r: r[2])
print()
print(f"best mean-class: {_best[0]} ({_best[2]:.3f}, "
      f"{_best[2] - SEGMENT_ROWS[0][2]:+.3f} vs per-window argmax) at fragmentation "
      f"{_best[3]:.2f}x")
print("Pick on BOTH columns: mean-class is label quality, frag is whether the segment")
print("structure Agent 3 counts bouts from survives. Neither alone decides it.")"""),
("code", """# P2 - leave-one-dataset-out on the fall sources: train-side generalisation is fixed
# (the ensemble saw only Charades ADL + the other fall sources via train_fall), so this
# measures how the FALL signal transfers to an unseen recording setup.
#
# TWO MODELS ARE SCORED, because this cell used to score only the first and the write-up
# then attributed its numbers to the second. The ADL ensemble's classes 7+8 are a fall
# signal, but `runs/fall/best.pt` is the DEDICATED binary head - the one that reports
# AUPRC 0.822 and owns the 0.951-sensitivity operating point - and it was never evaluated
# leave-one-dataset-out at all. Its 0.822 comes from train_fall.py's own internal val
# split, so "the fall head does not transfer" was a claim about a different model.
import glob, numpy as np, torch
from behaviorsense.data.skeleton_dataset import SkeletonWindowDataset, normalise
from behaviorsense.models.ensemble import windows_to_tensor
from behaviorsense.models.stgcnpp import STGCNpp, make_stream
FALL = shard_paths("fall")
assert FALL, f"no fall shards found. Attached: {ATTACHED}"
fds = SkeletonWindowDataset([*map(str, FALL)])
sources = np.asarray(fds.datasets)
FALL_CLASSES = (7, 8)
is_fall = np.isin(fds.labels, FALL_CLASSES)

def auroc(score, y):
    if not (y.any() and (~y).any()):
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    return (ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())

# Load the dedicated fall head if it is present. Same stream the trainer used, read from
# the checkpoint rather than assumed - a wrong stream loads silently and scores nonsense.
P2_DEVICE = getattr(clf, "device", "cuda")   # follows the ensemble; harness patches it
_fh = sorted(pathlib.Path(run_dir()).glob("fall/best.pt"))
fall_head, fall_stream = None, None
if _fh:
    _ck = torch.load(str(_fh[0]), map_location="cpu", weights_only=False)
    fall_stream = (_ck.get("args") or {}).get("stream", "joint")
    _m = STGCNpp(n_classes=1)
    _state = _ck.get("ema") or _ck["model"]
    _m.load_state_dict({k[len("net."):] if k.startswith("net.") else k: v
                        for k, v in _state.items()}, strict=True)
    fall_head = _m.eval().to(P2_DEVICE)
    print(f"fall head loaded from {_fh[0].parent.name}/best.pt, stream={fall_stream}, "
          f"recorded best AUPRC {_ck.get('best', float('nan')):.3f}")
else:
    print("no runs/fall/best.pt under the mounted runs - ADL-ensemble column only")

P2_ROWS = []
print(f"{'held-out':<12} {'n':>6} {'fall%':>6} {'ADL-ens AUROC':>14} {'FALL-HEAD AUROC':>16}")
for held in sorted(set(sources)):
    idx = np.where(sources == held)[0]
    # SHARD layout [N,T,M,17,3]. fds[i][0] returns [C,T,V,M] - already permuted for the
    # model - and logits() rejects it. P1 was fixed for exactly this and P2 was left
    # behind, so it died after 11 minutes with "expected [N,T,M,17,3], got (1159,3,30,17,2)".
    Xh = np.stack([normalise(fds.skeletons[i].astype(np.float32)) for i in idx])
    yh = is_fall[idx]
    a_ens = auroc(softmax(clf.logits(Xh))[:, list(FALL_CLASSES)].sum(1), yh)
    a_head = float("nan")
    if fall_head is not None:
        with torch.no_grad():
            t = windows_to_tensor(Xh)
            parts = [torch.sigmoid(fall_head(make_stream(t[s:s + 64].to(P2_DEVICE),
                                                         fall_stream))).cpu().numpy()
                     for s in range(0, len(t), 64)]
        # FallHead.forward squeezes the single logit; we load the bare STGCNpp, which
        # returns [N,1]. ravel() rather than squeeze() so a 1-window fold cannot
        # collapse to a scalar and silently break the ranking.
        a_head = auroc(np.concatenate(parts).ravel(), yh)
    P2_ROWS.append((str(held), int(len(idx)), float(yh.mean()), float(a_ens), float(a_head)))
    print(f"{held:<12} {len(idx):>6} {yh.mean():>6.1%} {a_ens:>14.3f} {a_head:>16.3f}")
print()
print("The FALL-HEAD column is the one the 0.822 AUPRC / 0.951-sensitivity claims belong to.")
print("If it is far below its own validation AUROC, that operating point is in-domain only.")"""),
("code", """# Ablation: logit vs probability combination (product-of-experts vs mixture).
alt = EnsembleClassifier.from_run_dir(run_dir(),
                                      device="cuda", combine="prob")
# scores() returns FOUR values since the F1>=MIN_SUPPORT column was added; this call
# site still unpacked three and died with "too many values to unpack" after P1 and P2
# had already run. Unpack all four here too.
t1a, mcaa, f1a, f1_supa = scores(alt.logits(X), y)
ABLATION_ROWS = [("logit-average", float(t1), float(mca), float(f1), float(f1_sup)),
                 ("prob-average", float(t1a), float(mcaa), float(f1a), float(f1_supa))]
print(f"{'combination':<16} {'top1':>6} {'mean-class':>10} {'macro-F1':>9} "
      f"{'F1>=' + str(MIN_SUPPORT):>9}")
print(f"{'logit-average':<16} {t1:>6.3f} {mca:>10.3f} {f1:>9.3f} {f1_sup:>9.3f}"
      "   <- headline")
print(f"{'prob-average':<16} {t1a:>6.3f} {mcaa:>10.3f} {f1a:>9.3f} {f1_supa:>9.3f}")
print()
print("Logit averaging is a product of experts, probability averaging a mixture. The")
print("headline uses logits; this row is the ablation that justifies that choice rather")
print("than asserting it.")"""),
("code", """# Hallucination benchmark with the REAL model. The Kaggle Model mount path varies -
# find it, then run all three arms. Stub anchors (already in results/hallucination.md) are
# what make these numbers interpretable.
#
# Run 1 produced 'unconstrained: 245 claims emitted, 0 scorable, nan%' and 39.9% for the
# constrained arm. Both were artefacts of the prompt never naming the output fields, so the
# free model could not guess the envelope and the constrained model mis-assigned values.
# With the field mapping stated, run 2 gave 8.2% / 6.5% with zero schema rejections.
#
# Runs 2 and 3 then disagreed. Free was bit-identical (509 claims, 476 faithful, twice) but
# constrained moved 498/457 -> 504/452, flipping the verdict from p=0.29 to p=0.03. A code
# audit of the two paths side by side - which is what should have happened after run 1 -
# found THREE ways they differed other than the grammar:
#   1. free passed do_sample=False; constrained passed only max_new_tokens to outlines and
#      inherited Qwen's generation_config (do_sample=True, temperature=0.7)
#   2. free applied the chat template; constrained handed outlines the raw prompt, so one
#      arm got a Qwen chat turn and the other a naked instruction block
#   3. repetition_penalty=1.05 from Qwen's config penalised exactly the verbatim copying
#      this task is scored on (both arms equally, so not a confound - but it inflates the
#      absolute rate)
# force_greedy() + _chat_text() + a cached output type fix all three. Test R14 now audits
# every axis on which the arms could differ, on CPU, so this class of defect costs seconds
# instead of a 2-hour session. The cached guide should also cut the constrained arm's
# 49.0 s/report substantially - watch the progress lines.
#
# This is the first decoding-matched run. Earlier constrained numbers should not be quoted.
#
# --dump writes every claim plus its verdict to JSONL. Run 2 could not answer "which values
# were misquoted, and by how much" because it kept only aggregates, and answering it cost a
# second 2-hour session. Now any re-analysis is a CPU rescore of that file.
import glob, subprocess, sys, os
qwen = sorted(glob.glob("/kaggle/input/**/config.json", recursive=True))
qwen = [p for p in qwen if "qwen" in p.lower()]
assert qwen, "attach the Qwen2.5-7B-Instruct Kaggle Model as an input"
QWEN_PATH = os.path.dirname(qwen[0])
print("Qwen at:", QWEN_PATH)
subprocess.run(
    [sys.executable, str(SCRIPTS / "eval_hallucination.py"),
     "--backend", "qwen", "--model-path", QWEN_PATH, "--days", "60",
     "--report", "/kaggle/working/results/hallucination_qwen.md",
     "--dump", "/kaggle/working/results/hallucination_qwen_claims.jsonl"],
    env={**os.environ, "PYTHONPATH": str(SRC)},
    check=True)"""),
("code", """# Write every table this session produced, then list what is in results/.
#
# The first full run printed P1, the per-class breakdown, calibration, P2 and the ablation
# to the log and wrote NOTHING - the bundler only globbed for *.md, and the single file
# present was hallucination_qwen.md from the subprocess. Nine hours of GPU output existed
# only as scrollback. These tables are the dissertation numbers, so they are written from
# the in-memory rows collected above.
import pathlib, os
RES = pathlib.Path(os.environ.get("BS_RESULTS_DIR", "/kaggle/working/results"))
RES.mkdir(parents=True, exist_ok=True)

def table(header, rows, fmt):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(fmt(r)) + " |" for r in rows]
    return out

L = ["# P1 - subject-disjoint validation", "",
     f"- {len(y)} val windows, {len(set(ds.subjects[val_idx]))} subjects, "
     f"split seed 0 (same as training)",
     f"- macro-F1 averages {int((support > 0).sum())} present classes; the last column "
     f"averages the {n_sup} with >= {MIN_SUPPORT} windows", ""]
L += table(["model", "top1", "mean-class", "macro-F1", f"F1>={MIN_SUPPORT}"], P1_ROWS,
           lambda r: [f"`{r[0]}`" if r[0] != "ENSEMBLE" else "**ENSEMBLE**",
                      f"{r[1]:.3f}", f"{r[2]:.3f}", f"{r[3]:.3f}", f"{r[4]:.3f}"])
# State the finding that contradicts the usual expectation instead of leaving a reader to
# infer it from the table: the ensemble wins top-1 and LOSES mean-class.
best_mca = max(P1_ROWS[:-1], key=lambda r: r[2])
if P1_ROWS[-1][2] < best_mca[2]:
    L += ["", f"**The ensemble improves top-1 but loses mean-class accuracy versus "
              f"`{best_mca[0]}` alone ({P1_ROWS[-1][2]:.3f} vs {best_mca[2]:.3f}).** "
              f"Logit averaging is a product of experts, so a stream that is confidently "
              f"wrong on a rare class can veto it; on a long-tailed label distribution "
              f"that trades tail recall for head accuracy. `{best_mca[0]}` is therefore "
              f"the better deployment checkpoint on the mean-class criterion."]
if starved:
    L += ["", f"Starved by the Charades label map (< {MIN_SUPPORT} windows): "
              + ", ".join(f"`{CLASS_NAMES[c]}` ({n})" for c, n in starved)
              + ". A label-map limitation, not a model result."]
L += ["", "## Calibration", "",
      f"- fitted temperature **T={T:.2f}**; mean confidence {conf:.3f} vs accuracy {acc:.3f}",
      f"- set `ActivityConfig(temperature={T:.2f})` in deployment", ""]
L += ["## P2 - leave-one-dataset-out (fall sources)", ""]
L += table(["held-out", "n", "fall%", "ADL-ens AUROC", "fall-head AUROC"], P2_ROWS,
           lambda r: [f"`{r[0]}`", str(r[1]), f"{r[2]:.1%}",
                      "n/a" if r[3] != r[3] else f"{r[3]:.3f}",
                      "n/a" if r[4] != r[4] else f"{r[4]:.3f}"])
L += ["",
      "Two models, deliberately. The ADL ensemble's classes 7+8 are a fall signal; "
      "`runs/fall/best.pt` is the dedicated binary head that owns the AUPRC 0.822 and the "
      "0.951-sensitivity operating point. Only the second column speaks to whether those "
      "claims transfer - the cell used to score the first and the write-up attributed its "
      "numbers to the second.", ""]
L += ["", "## Accuracy levers (validation-selected)", ""]
L += table(["configuration", "top1", "mean-class", "macro-F1"], LEVERS,
           lambda r: [f"`{r[0]}`", f"{r[1]:.3f}", f"{r[2]:.3f}", f"{r[3]:.3f}"])
_best = max(LEVERS, key=lambda r: r[2])
L += ["",
      f"Best mean-class: `{_best[0]}` at {_best[2]:.3f}, {_best[2] - LEVERS[0][2]:+.3f} "
      f"against the P1 headline. Logit adjustment and subset selection are decision-rule "
      f"changes on the same weights; TTA is one extra forward pass. None needs retraining. "
      f"All were selected on this validation split, so they are validation-selected numbers "
      f"and must be reported as such.", ""]
L += ["## Segment-level accuracy (what Agent 3 consumes)", ""]
L += table(["decoding", "top1", "mean-class", "frag ratio"], SEGMENT_ROWS,
           lambda r: [f"`{r[0]}`", f"{r[1]:.3f}", f"{r[2]:.3f}", f"{r[3]:.2f}"])
L += ["",
      f"Fitted transition prior: mean self-transition {np.mean(np.diag(A_fit)):.3f} against "
      f"the hand-set {cfg_sm.self_transition:.2f}. `frag ratio` is predicted segments over "
      f"true segments (1.00 ideal): mean-class is label quality, frag is whether the segment "
      f"structure Agent 3 counts bouts from survives. Neither column alone decides the "
      f"deployment choice.", ""]
L += ["## Ablation - logit vs probability combination", ""]
L += table(["combination", "top1", "mean-class", "macro-F1", f"F1>={MIN_SUPPORT}"],
           ABLATION_ROWS,
           lambda r: [f"`{r[0]}`", f"{r[1]:.3f}", f"{r[2]:.3f}", f"{r[3]:.3f}",
                      f"{r[4]:.3f}"])
L += ["", "Logit averaging is a product of experts, probability averaging a mixture. The",
      "headline uses logits; this row is the ablation that justifies that choice."]
(RES / "evaluation.md").write_text("\\n".join(L) + "\\n", encoding="utf-8")

for f in sorted(RES.glob("*.md")):
    print(f"- {f.name} ({f.stat().st_size} bytes)")
print()
print("Save Version, then create/update dataset behaviorsense-results from the output -")
print("these tables are the dissertation numbers and the session log is not a record.")"""),
]

N01A: list[tuple[str, str]] = [
("markdown", """# 01a — Stage Charades once (ONLINE, **CPU — do not attach a GPU**)

| attach as input | produces |
|---|---|
| nothing | `charades-480p` (~13 GB, **PRIVATE**) |

Pure I/O: downloads the Charades 480p archive and its annotations and republishes them as
a Kaggle dataset. **Set the accelerator to None.** Nothing here touches a GPU, and a GPU
session spent downloading is the exact waste this notebook exists to remove — extraction
previously re-fetched 13 GB at the start of each of 3-4 GPU sessions, ~30 min of idle card
every time.

Once this dataset exists, notebook 01 can run with the internet OFF, which means shard
extraction moves onto the Blackwell instead of a T4.

**Licence:** Charades forbids redistribution. Keep this dataset **private** — it is your
own working copy, exactly like the derived shard datasets.

The archives are stored as-is rather than unpacked. A single 13 GB file republishes far
faster than 9,848 individual mp4s, and notebook 01 unzips only the slice it needs."""),
("code", """# Download to /kaggle/working so Save Version picks it up. The 480p archive is ~13 GB
# and working is capped at 20 GB, so the zips are stored WITHOUT unpacking - unpacking
# here would need ~26 GB and blow the cap.
import urllib.request, pathlib, time

OUT = pathlib.Path("/kaggle/working"); OUT.mkdir(parents=True, exist_ok=True)
SOURCES = [
    ("https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip",
     "Charades_v1_480.zip"),
    ("https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades.zip",
     "Charades_annotations.zip"),
]
for url, name in SOURCES:
    dst = OUT / name
    if dst.exists():
        print(f"{name}: already present ({dst.stat().st_size/1e9:.1f} GB)")
        continue
    t0 = time.time()
    print(f"downloading {name} ...")
    urllib.request.urlretrieve(url, dst)
    print(f"  {dst.stat().st_size/1e9:.1f} GB in {(time.time()-t0)/60:.1f} min")"""),
("code", """# Verify before publishing. A truncated download produces a zip that opens fine at the
# header and fails halfway through extraction - four hours into an offline session, with
# no way to re-fetch. testzip() reads every member's CRC, so it catches that here.
import zipfile, pathlib

OUT = pathlib.Path("/kaggle/working")
for name, expect_min_gb in (("Charades_v1_480.zip", 10.0),
                            ("Charades_annotations.zip", 0.001)):
    p = OUT / name
    gb = p.stat().st_size / 1e9
    assert gb >= expect_min_gb, f"{name} is {gb:.2f} GB - truncated download"
    with zipfile.ZipFile(p) as z:
        bad = z.testzip()
        assert bad is None, f"{name}: corrupt member {bad}"
        members = z.namelist()
    print(f"  {name:<28} {gb:>5.1f} GB, {len(members)} members, CRC ok")

with zipfile.ZipFile(OUT / "Charades_v1_480.zip") as z:
    vids = [n for n in z.namelist() if n.endswith(".mp4")]
assert len(vids) > 9000, f"only {len(vids)} mp4s - expected ~9,848"
print(f"\\n{len(vids)} videos ready")
print("Save Version -> create a PRIVATE dataset named charades-480p")
print("Then notebook 01 runs OFFLINE: attach this + behaviorsense-code + the wheels.")"""),
]



N05: list[tuple[str, str]] = [
("markdown", """# 05 — Inference backend for the web front end (ONLINE, **GPU P100**)

| attach as input | produces |
|---|---|
| `behaviorsense-code`, `behaviorsense-runs` | a public HTTPS URL you paste into the front end |
| Kaggle Model: **Qwen2.5-7B-Instruct** | only if Agent 4 runs locally - see below |

## Pick P100, not T4 x2, and not a TPU

**P100.** With Agent 4 hosted, nothing left on this GPU uses tensor cores: `rtmo-l.onnx` is an
fp32 export, and so are OSNet and ST-GCN++. On fp32 the cards invert - T4 is 8.1 TFLOPS at
320 GB/s, P100 is 9.3 TFLOPS at **732 GB/s** - and pose extraction is bandwidth-bound. T4's 65
TFLOPS of fp16 tensor throughput only ever mattered for Qwen. One card also removes the
pipeline-parallel PCIe hop that had both T4s reading 0% utilisation while the CPU spun.

VRAM stops being the constraint: RTMO's session is ~1-2 GB, OSNet ~0.5 GB, ST-GCN++ ~50 MB.
Under 4 GB of 16. The two T4s were only ever needed because a 7B model would not fit on one.

**Not a TPU**, whatever its RAM. `onnxruntime` has no TPU execution provider, so RTMO - the one
genuinely GPU-bound stage - cannot run on it at all and would fall back to CPU. TPUs are also
throughput devices, and this is batch-of-one interactive inference: the shape they are worst at.

## Agent 4 runs off-box by default

Set **`OPENROUTER_API_KEY_LIST`** in Add-ons -> Secrets (comma-separated; numbered
`OPENROUTER_API_KEY_1..N` also work) and the local 7B is never loaded - no 70 s load, no 17.8 GiB,
no 286 s report. Without keys it falls back to Qwen and says so.

Qwen stays the MEASURED arm: every figure in `results/evaluation.md` came from it under
grammar-constrained decoding on a pinned checkpoint. A `:free` endpoint can be deprecated without
notice, so it is the demo arm, and `/health` reports `agent4` so the page states which model
wrote a report.

**Internet ON, GPU on.** This notebook is the only one that serves rather than computes: it
loads the trained ADL streams and Qwen behind FastAPI, opens a Cloudflare quick tunnel, and
prints the address. `web/` then talks to that address from anywhere.

Why a tunnel rather than hosting the model near the front end: Qwen2.5-7B needs ~16 GB in
bf16, and the ADL ensemble needs a GPU to be worth calling at all. Neither fits a laptop, and
quantising to fit would trade the exact numbers `results/evaluation.md` reports for ones
nobody has measured. Kaggle's T4 x2 / P100 are free and already hold every weight.

**The address changes every session.** That is inherent to a quick tunnel with no account,
and it is why the front end asks you to paste it rather than hard-coding one. Run this
notebook, copy the line it prints, paste it into the field in the header.

**This API is unauthenticated.** Anyone with the URL can post to it for as long as the
session lives. That is acceptable for a demo you start and stop deliberately; it is not a
deployment. `BS_TOKEN` below turns on a shared-secret header if you want one — the front end
has a field for it.

Deployment configuration is not a guess: `logit_adjust_tau=0.25` and the fitted transition
prior are the settings the notebook-04 lever table selected, and `do_sample=False` is the
greedy decoding the reporter requires. Serving anything else would mean the demo and the
measurements describe different systems."""),
("code", RESOLVE_CODE),
("code", """# Deps. `cloudflared` is a single static binary - no account, no config file. Pinned to a
# release rather than `latest` so a breaking change upstream cannot silently take the demo
# down: the URL format this notebook greps for is part of that contract.
#
# onnxruntime-gpu is PINNED to 1.26.0 and the pin is load-bearing. From 1.27 the PyPI GPU
# wheels are built against CUDA 13 while Kaggle's image is CUDA 12; the provider is still
# LISTED by get_available_providers() and then fails to load, so RTMO runs on CPU at roughly
# 1/50th speed and the only symptom is a video that takes forever. video.py asserts on the
# session's providers, which is the only honest source.
import subprocess, sys, os, pathlib

# ORDER IS LOAD-BEARING, and this cell had it backwards. `rtmlib` depends on the CPU
# `onnxruntime`, and the CPU and GPU packages own the SAME `onnxruntime/` directory. The
# previous sequence installed everything - pip pulling the CPU build in as rtmlib's
# dependency - and then uninstalled `onnxruntime` afterwards, which deletes files SHARED
# with onnxruntime-gpu. The result imports and is hollow:
#
#   AttributeError: module 'onnxruntime' has no attribute '__version__'
#
# The comment that used to be here claimed removing the CPU build after the install avoided
# the GPU one being "shadowed". That was wrong, and notebooks 01 and 07 already do it the
# right way round - three successful extraction runs prove the sequence below:
#
#   1. remove any CPU onnxruntime FIRST
#   2. install rtmlib with --no-deps so it cannot drag the CPU build back in
#   3. install onnxruntime-gpu LAST, so nothing writes over its provider registration
#
# onnxruntime-gpu is PINNED to 1.26.0 and the pin is load-bearing. From 1.27 the PyPI GPU
# wheels are built against CUDA 13 while Kaggle's image is CUDA 12; the provider is still
# LISTED by get_available_providers() and then fails to load, so RTMO runs on CPU at roughly
# 1/50th speed and the only symptom is a video that takes forever. video.py asserts on the
# session's providers, which is the only honest source.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "onnxruntime"],
               check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "rtmlib"],
               check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "fastapi", "uvicorn", "nest_asyncio", "python-multipart",
                "outlines>=1.0", "opencv-python-headless"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "onnxruntime-gpu==1.26.0"], check=True)

CF = pathlib.Path("/usr/local/bin/cloudflared")
if not CF.exists():
    url = ("https://github.com/cloudflare/cloudflared/releases/download/2024.12.2/"
           "cloudflared-linux-amd64")
    subprocess.run(["curl", "-fsSL", "-o", str(CF), url], check=True)
    CF.chmod(0o755)
print(subprocess.run([str(CF), "--version"], capture_output=True, text=True).stdout.strip())

# Check the install is WHOLE, not just importable. A missing `__version__` is the signature
# of the shared-directory breakage above, so it gets a message that names the cause instead
# of an AttributeError six lines into a demo.
import onnxruntime
_ver = getattr(onnxruntime, "__version__", None)
assert _ver is not None, (
    "onnxruntime imported but has no __version__ - the package directory is incomplete, "
    "which happens when the CPU build is uninstalled AFTER onnxruntime-gpu (they share "
    "the directory). Restart the kernel and re-run this cell: it now removes the CPU build "
    "first and installs the GPU build last.")
_prov = onnxruntime.get_available_providers()
print("onnxruntime", _ver, _prov)
assert any("CUDA" in p for p in _prov), (
    f"no CUDAExecutionProvider in {_prov}. RTMO would run on CPU at ~1/50th speed and the "
    "only symptom would be a very slow upload.")"""),
("code", """# Resolve the weights. Same content-based rule as every other notebook: never trust a
# dataset name. find_run_dir picks the directory holding adl_<stream>/ and skips leftovers
# inside a mounted code checkout - the trap that cost notebook 04 a session.
import glob, os
from behaviorsense.kaggle_artifacts import find_run_dir

RUNS = str(find_run_dir(INPUT))
# Per-mount, not a `**` walk. `**/config.json` visits every file under /kaggle/input -
# including MSMT17's 65,242 crops inside behaviorsense-code - to find one file in a
# directory whose name already says "qwen".
_qwen = [c for s in _SLUGS if "qwen" in s.name.lower()
         for c in ((s / "config.json"), *(s.glob("*/config.json"))) if c.is_file()]
assert _qwen, f"attach the Qwen2.5-7B-Instruct Kaggle Model. Attached: {ATTACHED}"
QWEN = str(_qwen[0].parent)

# Pose + re-id weights for the video path. RTMO is required for it; OSNet is optional and
# its absence degrades roles to UNKNOWN rather than to a guess.
_rtmo_p = find_asset("**/rtmo-l.onnx", "rtmo", required=False)
RTMO = str(_rtmo_p) if _rtmo_p else None
_osnet_p = find_asset("**/osnet_ain_x1_0_msmt17.pth", "osnet", required=False)
# OSNet weights. Enablement is PER REQUEST (see `use_osnet` in /video), because matching
# against an enrolled gallery. So every track returns `unidentified` whether it runs or not - the
# `subject of Agent 3's features` badge comes from the most-present-track assertion, not from
# re-identification. On CPU (which a P100 forces, since torch has no sm_60 kernels) it pushed
# 1,202 person crops through a ReID CNN and took pose extraction from 29.8 s to 134.2 s to produce
# the string "unknown". Measured, so it stays off until something actually enrols someone.
# PER-REQUEST, not per-notebook. OSNet used to be off unconditionally: with no enrolled
# resident it could only return `unidentified`, and on CPU (which a P100 forces - torch has
# no sm_60 kernels) it pushed 1,202 crops through a ReID CNN for 104 s to produce that
# string. The gallery changes the economics: when a request CARRIES one (the local backend
# attaches the operator's gallery.json, see web/local_backend.py) or asks to enrol, matching
# can actually answer; sampling (reid_embed_interval) brings the CPU cost from 104 s to a
# few seconds. No gallery and no enrol request -> OSNet stays off, exactly as before.
OSNET = str(_osnet_p) if _osnet_p else None
if _osnet_p:
    print(f"  osnet found at {OSNET}: enabled per request when a gallery or an enrol "
          "name is present (embedding sampled every "
          "reid_embed_interval frames, not every frame)")
else:
    print("  osnet ABSENT: re-identification stays off and identity is asserted, "
          "not recognised")

# PREFLIGHT THE DECODE CHILD, here, rather than discovering it on the first upload.
#
# Decode runs in a separate interpreter, and `sys.path` does not cross a process boundary.
# The resolver cell above made `behaviorsense` importable in THIS process; a fresh one knows
# nothing about it. That is a real failure this notebook shipped: an uploaded clip came back
# `pose extraction failed: No module named 'behaviorsense'` while the parent was importing
# the package perfectly well. `behaviorsense.video.child_env` now puts the package on the
# child's PYTHONPATH, derived from the module's own location.
#
# It is preflighted because of WHEN it would otherwise fail. Every other prerequisite here is
# checked at startup; this one used to surface mid-demo, on someone else's video, as a 422
# that read like a bad file. A one-second probe now converts that into a failure at the point
# where it can still be fixed.
if RTMO:
    import subprocess as _sp
    from behaviorsense.video import PKG_PARENT as _PKGP, child_env as _cenv
    _probe = _sp.run([sys.executable, "-c", "import behaviorsense.video as v; print(v.__file__)"],
                     capture_output=True, text=True, env=_cenv(), timeout=120)
    assert _probe.returncode == 0, (
        "the decode CHILD cannot import behaviorsense, so every /video upload would 422.\\n"
        f"  PYTHONPATH given to the child: {_cenv()['PYTHONPATH']}\\n"
        f"  package expected under:        {_PKGP}\\n"
        f"  child stderr: {_probe.stderr.strip()[-400:]}\\n"
        "  FIX: re-upload `behaviorsense-code` from your checkout - this needs the version of "
        "src/behaviorsense/video.py that sets the child's PYTHONPATH (it defines `child_env`).")
    print(f"  decode child OK -> {_probe.stdout.strip()}")
else:
    print("  rtmo-l.onnx not attached: /video is off, decode child not probed")

# Serve the configuration notebook 04 SELECTED, not the defaults. Every value here is a
# measured choice from results/evaluation.md, and each one was wrong at some point:
#
#   STREAMS   the 4-stream ensemble was the headline and is the WORST credible option on
#             macro-F1 (0.128 vs 0.146 for `bone` alone). macro-F1 is the arbiter because
#             Agent 3 turns window predictions into daily durations, where over-predicting
#             a class inflates a duration exactly as missing one deflates it.
#   TEMPERATURE  must be fitted FOR THE SERVED STREAMS. The 0.63 in the run log was fitted
#             on the 4-stream ensemble's logits; serving `bone` alone with it would be
#             miscalibrated, and Viterbi plus abstention both consume those posteriors.
#             So it is refitted here from the saved val logits - free, on CPU.
#   TAU       0.25, swept in notebook 04 with tau=0 asserted as an exact no-op.
STREAMS = ("bone",)                 # best macro-F1; ("bone", "joint") for best mean-class
TAU = 0.25
# SAMPLE_FPS belongs in this block for the same reason STREAMS and TAU do: it is a property of
# the CHECKPOINT, not of the camera, and getting it wrong is silent.
#
# A 30-frame window is `30 / SAMPLE_FPS` seconds of motion, and ST-GCN++ has only ever seen the
# duration its training shards were built at. The corpora do not agree:
#
#   Charades (notebook 01)  FPS_SAMPLE = 15  ->  2.00 s   <- what runs/adl_bone/best.pt saw
#   fall     (notebook 02)  fps_sample = 15  ->  2.00 s
#   Toyota   (notebook 07)  FPS_SAMPLE = 20  ->  1.50 s
#
# So this must be flipped to 20.0 AT THE SAME TIME as the Toyota-trained checkpoint is served,
# not before and not after. Serving 20 Hz against the Charades checkpoint would stretch every
# action by 1.33x - the same defect as the old `src_fps / 2` stride, just moved to a constant.
# The rate travels in the /video payload and the page draws a banner when it is not 15, so a
# mismatch is visible rather than absorbed as bad accuracy.
# The rate to use when the checkpoint does not record one. It MUST agree with ADL_PREFER
# below: they are two constants in this cell describing one choice, and they were
# contradicting each other - ADL_PREFER said "toyota" while this said None, so the loader
# fail-closed on every start and the notebook could not serve the checkpoint it was
# configured for.
#
# `None` remains available and means DERIVE-OR-REFUSE. It is the right value once every
# checkpoint records its own rate (`train_adl.py` now does), and the wrong one while a
# pre-fix checkpoint is still being served - refusing to start is only useful if there is
# something the operator can do about it, and here the answer was already known.
#
#   20.0  Toyota RTMO shards (1.50 s windows) - matches ADL_PREFER = "toyota"
#   15.0  Charades shards    (2.00 s windows) - set this if you serve `adl_bone/`
SAMPLE_FPS = 20.0
# Fallback only. The loader below DERIVES the rate from the served checkpoint's own shard paths
# (`..._20hz_...`) and writes it to STATE["sample_fps"]; this value is used only when the shard
# names carry no rate at all.
#
# ADL_PREFER disambiguates when several runs are attached for the same stream - `adl_bone/`
# (Charades, 20 classes, 15 Hz) and `adl_toyota_bone/` (Toyota RTMO, 22 classes, 20 Hz) both
# match stream `bone`. Set to "toyota" to serve the corpus filmed in a real home on mounted
# cameras; set to "" to take whatever sorts first and accept the coin toss.
ADL_PREFER = "toyota"
MAX_UPLOAD_MB = 60
TOKEN = os.environ.get("BS_TOKEN", "")      # optional shared secret; "" disables the check

# Refit T for the streams actually being served. val_logits.npz is written by notebook 04;
# without it a hardcoded temperature silently belongs to a different model.
TEMPERATURE = None
_vl_p = find_asset("**/val_logits.npz", "val_logits", required=False)
_vl = [_vl_p] if _vl_p else []
if _vl:
    import numpy as _np
    from behaviorsense.agents.activity import fit_temperature as _fitT
    from behaviorsense.eval.activity_eval import combine as _combine
    with _np.load(_vl[0], allow_pickle=False) as _z:
        _per = {k[len("logits_"):]: _z[k] for k in _z.files
                if k.startswith("logits_") and k != "logits_ensemble"}
        _y = _z["y"]
    _missing = set(STREAMS) - set(_per)
    assert not _missing, f"val_logits.npz lacks {sorted(_missing)}; has {sorted(_per)}"
    TEMPERATURE = float(_fitT(_combine(_per, tuple(STREAMS), "logit"), _y))
    # The head size these logits describe. `val_logits.npz` carries no link to the checkpoint
    # that produced it, and the first served run fitted T=0.79 from CHARADES logits (20 classes,
    # 35,698 windows) onto the 22-class Toyota model without a murmur - the exact failure the
    # comment above warns about. `_load` compares this against the served head and discards the
    # fit if they disagree, because a temperature from another model is worse than none.
    TEMPERATURE_N_CLASSES = int(next(iter(_per.values())).shape[1])
    TEMPERATURE_SOURCE = f"{len(_y):,} val windows, {TEMPERATURE_N_CLASSES}-class"
    print(f"T refitted for {'+'.join(STREAMS)} on {TEMPERATURE_SOURCE}: {TEMPERATURE:.2f}")
else:
    TEMPERATURE = 1.0
    TEMPERATURE_N_CLASSES = None
    TEMPERATURE_SOURCE = "unfitted"
    print("WARNING: no val_logits.npz attached - serving UNCALIBRATED (T=1.0). Attach "
          "behaviorsense-results, or Viterbi and abstention consume meaningless posteriors.")
print(f"runs  {RUNS}")
print(f"qwen  {QWEN}")
print(f"rtmo  {RTMO or 'ABSENT - /video will refuse'}")
print(f"osnet {OSNET or 'ABSENT - roles will be UNKNOWN'}")
print(f"tau   {TAU}   T {TEMPERATURE}   auth {'on' if TOKEN else 'OFF (demo only)'}")
print()
print(f"  serving {len(STREAMS)} stream(s) from {RUNS}: {'+'.join(STREAMS)}")
print("  The 4-stream ensemble is the published headline; the served configuration is")
print("  the subset the lever table picked for macro-F1 (0.146 vs 0.128 for 4-stream).")
print("  See results/evaluation.md -> 'Accuracy levers' for the ablation. T is refit")
print("  for the served set above; tau swept with tau=0 asserted as an exact no-op.")"""),
("code", """# The API. Models load on a BACKGROUND thread so the tunnel address prints in seconds
# instead of after a three-minute weight load - /health reports `loading` until it is ready,
# which is what lets the front end say "models loading" rather than "offline".
import re, threading, time, numpy as np
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from behaviorsense.agents.activity import ActivityConfig, FALLING, softmax, viterbi
from behaviorsense.eval.activity_eval import (apply_emergency_floor, class_prior,
                                              fit_transition_matrix, logit_adjust,
                                              sequences_by_subject)
from behaviorsense.models.ensemble import EnsembleClassifier, torch_supports_device
from behaviorsense.agents.reasoning.openrouter import FALLBACK_MODELS, OpenRouterLLM, discover_keys
# ONE subject decision for both halves of the split. It used to be inlined in this cell AND
# implemented in `staging.py`, which is two places for a rule about who the clip is about.
from behaviorsense.service.staging import assert_subject as _assert_subject

# Agent 4's hosted model. Pinned here rather than left to a default so the page's `model` field
# and this constant cannot disagree. GLM's OpenRouter page states it supports structured outputs
# via a JSON schema in `response_format`, which is what replaces `outlines`' local FSM.
OPENROUTER_MODEL = "z-ai/glm-5.2:free"
# THE FALLBACK CHAIN IS THE FIX, and it is a chain of PROVIDERS rather than of models.
# Measured 2026-08-28: thirteen keys returned thirteen 429s, one each, and OpenRouter's own body
# said `z-ai/glm-5.2:free is temporarily rate-limited UPSTREAM`; the dashboards showed several of
# those keys with no requests that day at all. The ceiling was Decart's capacity - one free
# endpoint shared by everyone - so no number of keys could help, because all of them arrive there.
# `FALLBACK_MODELS` is checked against `/api/v1/models/<id>/endpoints`: every entry supports strict
# `response_format` and every entry sits on a different provider. See that module for why
# `nemotron-3-ultra-550b-a55b:free` is not among them despite being the larger NVIDIA model.
OPENROUTER_FALLBACKS: tuple[str, ...] = FALLBACK_MODELS

app = FastAPI(title="BehaviorSense inference")
# The front end is served from a different origin by design (static host + Kaggle tunnel),
# so CORS is not optional. Methods are restricted; the origin cannot be, because the tunnel
# address is unknown until it is created.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"],
                   allow_headers=["*"])

STATE = {"ready": False, "error": None, "loaded_at": None}

def _load():
    try:
        import torch as _torch
        # Fragmentation, not capacity, is what the OOM message itself suggested trying
        # ("If reserved but unallocated memory is large try expandable_segments"). Set before
        # the first allocation or it has no effect.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # Serve STREAMS, not everything on disk. from_run_dir() loads every adl_<stream>/
        # it finds, which is how the 4-stream ensemble became the default by accident.
        #
        # RESOLVED BY CONTENT, not by directory name. The Charades run wrote `adl_bone/`; the
        # Toyota RTMO run wrote `adl_toyota_bone/`, and a name-built path finds neither when
        # both are attached. Every checkpoint records its own `args.stream`, `args.n_classes`
        # and the shard paths it trained on, so the directory name is the least reliable thing
        # about it. ADL_PREFER picks between candidates for the same stream and is printed with
        # the alternatives, because with both datasets mounted the choice is real and silent
        # selection of the wrong corpus is the failure that matters.
        import torch as _t
        _cands = {}
        for _p in sorted(pathlib.Path(RUNS).glob("adl_*/best.pt")):
            try:
                _a = _t.load(str(_p), map_location="cpu", weights_only=False)
            except Exception as _e:                       # noqa: BLE001
                print(f"  skipped {_p.parent.name}: {type(_e).__name__}")
                continue
            _args, _m = _a.get("args", {}), _a.get("metrics", {})
            _cands.setdefault(_args.get("stream"), []).append({
                "path": _p, "dir": _p.parent.name,
                "n_classes": _args.get("n_classes"),
                "shards": " ".join(map(str, _args.get("shards", []))),
                "sample_fps": _args.get("sample_fps"),
                "mca": _m.get("mean_class_acc"), "f1": _m.get("macro_f1"),
            })
        print(f"  {len(_cands)} stream(s) with checkpoints under {RUNS}:")
        for _s, _lst in sorted(_cands.items(), key=lambda kv: str(kv[0])):
            for _c in _lst:
                print(f"    {_c['dir']:<26} stream={_s} classes={_c['n_classes']} "
                      f"mca={_c['mca']} f1={_c['f1']}")
        _ck, _absent = {}, []
        for _s in STREAMS:
            _lst = _cands.get(_s, [])
            _hit = [c for c in _lst if ADL_PREFER in c["dir"]] or _lst
            if not _hit:
                _absent.append(_s)
                continue
            if len(_hit) > 1:
                print(f"  WARNING {_s}: {len(_hit)} candidates match ADL_PREFER={ADL_PREFER!r} "
                      f"({[c['dir'] for c in _hit]}); taking the first. Narrow ADL_PREFER.")
            _ck[_s] = _hit[0]["path"]
            print(f"  serving {_s} from {_hit[0]['dir']} ({_hit[0]['n_classes']} classes)")
        assert not _absent, (
            f"no checkpoint for stream(s) {_absent} under {RUNS}. Found streams "
            f"{sorted(str(k) for k in _cands)}. Attach the runs dataset, or set STREAMS to what "
            "is actually trained.")
        # DOES THIS TORCH HAVE KERNELS FOR THIS CARD? `is_available()` says "there is a driver
        # and a device", not "this wheel was compiled for it". On Kaggle's P100 (sm_60) against a
        # torch built for sm_70+, the model moved to CUDA without complaint and Agent 2 then died
        # with `no kernel image is available for execution on the device` - which surfaced as an
        # AcceleratorError from whatever op ran first, reading like a bug in that op. RTMO was
        # unaffected because onnxruntime carries its own kernels.
        #
        # ST-GCN++ is 787k parameters over ~28 windows, so CPU costs a fraction of a second and is
        # the right answer rather than a degradation. P100 stays worthwhile: pose extraction is the
        # expensive stage, it is bandwidth-bound, and it ran 29.8 s there against 40 s on a T4.
        _torch_ok, _why = torch_supports_device()
        _clf_dev = "cuda" if _torch_ok else "cpu"
        print(f"  torch on GPU: {_torch_ok} - {_why}")
        if not _torch_ok:
            print(f"  ST-GCN++ and OSNet run on CPU (787k params over ~28 windows is "
                  f"sub-second); RTMO keeps the GPU through onnxruntime")
        clf = EnsembleClassifier(_ck, device=_clf_dev)
        STATE["torch_device"] = _clf_dev
        STATE["clf"] = clf
        STATE["streams"] = clf.streams
        STATE["n_classes"] = clf.n_classes

        # DOES THE TEMPERATURE BELONG TO THIS MODEL? `val_logits.npz` carries no link to the
        # checkpoint that produced it, so the config cell above can fit a temperature from any
        # file it finds. On the first served run it fitted T=0.79 from Charades logits (20
        # classes, 35,698 windows) onto this 22-class Toyota model - a calibration constant from
        # a different network, silently applied to the posteriors Viterbi and abstention consume.
        #
        # The head size is a decisive, free check, so it is made here where `clf` finally knows
        # its own. Discarding the fit costs calibration; keeping a foreign one corrupts every
        # smoothed label, and only one of those is recoverable.
        global TEMPERATURE
        if TEMPERATURE_N_CLASSES is not None and TEMPERATURE_N_CLASSES != clf.n_classes:
            print(f"  WARNING discarding T={TEMPERATURE:.2f}: it was fitted on "
                  f"{TEMPERATURE_SOURCE} logits but this checkpoint has {clf.n_classes} "
                  f"classes. Serving UNCALIBRATED (T=1.0). To calibrate, save val logits from "
                  f"the {clf.n_classes}-class run and attach those instead.")
            TEMPERATURE = 1.0
            STATE["temperature_source"] = "discarded (class-count mismatch)"
        else:
            STATE["temperature_source"] = TEMPERATURE_SOURCE
        STATE["temperature"] = TEMPERATURE
        # THE SAMPLING RATE COMES FROM THE CHECKPOINT'S OWN SHARDS, not from a constant here.
        # A 30-frame window is `30 / rate` seconds of motion and the model has seen exactly one
        # duration; Charades shards are 15 Hz (2.00 s) and the Toyota RTMO shards are 20 Hz
        # (1.50 s). Serving the Toyota weights at 15 Hz stretches every action by 1.33x, which
        # is the same defect as the old `src_fps / 2` stride wearing a different hat. Deriving it
        # from `args.shards` makes the pair impossible to separate.
        _sh = next((c["shards"] for l in _cands.values() for c in l
                    if c["path"] in _ck.values()), "")
        # `args.shards` is whatever the operator typed. Passed as a GLOB ("shards/*.npz") it
        # carries no rate at all, which is exactly what happened on the first served run - and
        # the fallback then quietly served 15 Hz weights trained at 20 Hz, a 1.33x stretch on
        # every action. So: expand the glob if the shards happen to be mounted, then try the
        # directory name, and REFUSE if neither answers.
        import glob as _glob
        _parts = [_sh]
        for _t in _sh.split():
            _parts.append(str(pathlib.Path(_t).parent))      # the shard DIRECTORY may name it
            if "*" in _t:
                _parts.extend(_glob.glob(_t))                 # expand, if they are mounted
        _hay = " ".join(_parts)
        # FIRST CHOICE: the rate the training run recorded. `train_adl.py` derives it from the
        # RESOLVED shard filenames, where the information actually exists - `args.shards` is
        # whatever was typed and a glob carries no rate at all, which is what made this refuse to
        # start once. Checkpoints trained before that fix fall through to the paths below.
        _rec = next((c.get("sample_fps") for l in _cands.values() for c in l
                     if c["path"] in _ck.values() and c.get("sample_fps")), None)
        _m = None if _rec else re.search(r"_(\d+(?:\.\d+)?)hz", _hay)
        if _rec:
            STATE["sample_fps"] = float(_rec)
            _why = "recorded by the training run"
        elif _m:
            STATE["sample_fps"] = float(_m.group(1))
            _why = "derived from the checkpoint's own shard paths"
        elif SAMPLE_FPS is not None:
            STATE["sample_fps"] = float(SAMPLE_FPS)
            _why = f"ASSERTED by SAMPLE_FPS={SAMPLE_FPS:g} (not derivable from args.shards)"
        else:
            # FAIL CLOSED. A wrong rate is invisible: the model returns confident labels for
            # windows of a duration it has never seen. Refusing costs a restart; guessing costs
            # every number the demo produces.
            raise AssertionError(
                "cannot determine the sampling rate this checkpoint was trained at. "
                f"args.shards = {_sh!r}. "
                "A 30-frame window is 30/rate seconds of motion and ST-GCN++ has seen exactly "
                "one duration; serving 15 Hz weights at 20 Hz - or the reverse - stretches every "
                "action by 1.33x and shows up only as bad accuracy. "
                "FIX: set SAMPLE_FPS in the config cell to the rate the shards were built at "
                "(15.0 for Charades, 20.0 for the Toyota RTMO shards), or re-run training with "
                "an expanded --shards list so the rate is recorded in args.shards.")
        print(f"  sample rate {STATE['sample_fps']:g} Hz "
              f"({30 / STATE['sample_fps']:.2f}s windows) {_why}")
        from behaviorsense.agents.reasoning.reporter import CaregiverReporter, ReporterConfig
        cfg = ReporterConfig(constrained=True)
        # SHARD ACROSS BOTH T4s. `device="cuda"` pins every layer to device 0, and Qwen2.5-7B
        # in 16-bit is ~15.2 GB of weights against a 14.56 GiB usable card - so it LOADED with
        # ~200 MB spare and then died on the first KV-cache allocation of generation:
        #
        #   OutOfMemoryError: Tried to allocate 34.00 MiB. GPU 0 has ... 10.81 MiB is free
        #
        # while `nvidia-smi` showed GPU 1 holding 3 MiB. The second card was never used. The
        # reporter has taken `max_memory` for exactly this since the T4 serving path was added;
        # this cell simply never passed it.
        #
        # The bounds are not tuning, they are reservations. Three other things want VRAM on
        # these same two cards:
        #   - the ST-GCN++ ensemble above, on device 0 (small, but it is already resident)
        #   - the RTMO + OSNet CHILD process, which opens its own CUDA context per upload
        #     (~2 GB with the onnxruntime session) and cannot share PyTorch's allocator
        #   - the KV cache, which grows during generation on whichever device holds the
        #     later layers - the allocation that actually failed
        # PyTorch's caching allocator does not return memory to the driver, so `auto` filling
        # both cards to the brim would starve the child even though the weights fit.
        _mm = {i: "11GiB" for i in range(_torch.cuda.device_count())} or None

        # DTYPE: float16 on a card WITHOUT native bfloat16, which is every serving card here.
        #
        # bf16 needs compute capability 8.0. T4 is sm_75 and P100 is sm_60, so a bf16 matmul does
        # not reach the tensor cores - it runs on a fallback path that is several times slower for
        # numerics nobody is measuring at serving time. That is most of why a 6-claim report took
        # 286 s with both GPUs reading 0% utilisation: emulated matmuls on top of pipeline-parallel
        # sharding across PCIe, at batch size 1.
        #
        # fp16 has a narrower exponent range and Qwen's weights are bf16-trained, so this is not
        # free - but `_assert_finite_logits()` runs one forward pass at load and refuses a
        # non-finite result, which is the failure mode fp16 actually has. The evaluation numbers
        # in results/evaluation.md were measured on the Blackwell in bf16 and are untouched by a
        # serving-side dtype.
        # From the CAPABILITY, not from `is_bf16_supported()`. That helper returned True on a
        # P100 whose torch build has no sm_60 kernels at all, and the loader duly printed
        # "reporter dtype bfloat16" for a card that cannot run bf16 or anything else. bf16 needs
        # sm_80; ask the device.
        _cap = _torch.cuda.get_device_capability(0) if _torch.cuda.is_available() else (0, 0)
        _bf16_native = _torch_ok and _cap >= (8, 0)
        _DTYPE = "bfloat16" if _bf16_native else "float16"
        print(f"  reporter dtype {_DTYPE}"
              + ("" if _bf16_native else
                 f" ({_torch.cuda.get_device_name(0)} is pre-sm_80, so bfloat16 would be "
                 "emulated off the tensor cores)"))
        # AGENT 4: HOSTED FIRST, LOCAL WEIGHTS AS THE FALLBACK.
        #
        # Qwen on two T4s was 286 s of a ~340 s request - 85% of the wall clock - for 17.8 GiB of
        # VRAM, emulated bf16 on sm_75, and ~13 GiB of host RAM retained per analysis. A hosted
        # endpoint at 164 tok/s makes that stage single-digit seconds and gives the pose model the
        # whole GPU. Skipping the local load also skips its 70 s and its footprint entirely.
        #
        # Safe for one structural reason, not because the hosted model is better: Agent 4 receives
        # NUMBERS ONLY - never frames, never skeletons - and every claim it writes goes through
        # C1-C5 arithmetically before anyone sees it. Substituting a model we know less about is
        # exactly the case the verifier exists to cover.
        #
        # Qwen remains the MEASURED arm. Every figure in results/evaluation.md came from it under
        # grammar-constrained decoding on a pinned checkpoint, and a `:free` endpoint can be
        # deprecated without notice. `model_name` travels in the payload and the page renders it,
        # so which model wrote a report is stated rather than assumed.
        _or_keys = discover_keys()
        if not _or_keys:
            try:
                from kaggle_secrets import UserSecretsClient   # noqa: PLC0415
                _sec = UserSecretsClient()
                # One secret holding a comma-separated list, because Kaggle secrets are one per
                # name and twenty of them is twenty clicks. Numbered names still work.
                for _n in ("OPENROUTER_API_KEY_LIST", "OPENROUTER_API_KEY"):
                    try:
                        _v = (_sec.get_secret(_n) or "").strip()
                    except Exception:                          # noqa: BLE001, S112
                        continue
                    if _v:
                        os.environ[_n] = _v
                _or_keys = discover_keys()
            except Exception as _e:                            # noqa: BLE001
                print(f"  kaggle_secrets unavailable ({type(_e).__name__}); "
                      "set OPENROUTER_API_KEY_LIST in Add-ons -> Secrets to use the hosted model")

        # NO LOCAL WEIGHTS ON THIS PATH. Qwen is gone from serving entirely: it was 286 s of a
        # ~340 s request on two T4s, and on the P100 it cannot run at all - torch has no sm_60
        # kernels, so it loaded for 106 s and then died with `no kernel image is available`
        # exactly as Agent 2 had. Agent 4 is a hosted call or it is not served.
        #
        # Qwen remains the MEASURED arm in notebook 04 on the Blackwell, which is where every
        # figure in results/evaluation.md came from and where a pinned checkpoint belongs.
        if _or_keys:
            # NOT cfg.max_new_tokens: that is the measured local arm's 1600, and a reasoning
            # model spends part of its budget thinking, so every hosted report came back as
            # `unterminated JSON object (truncated generation)`. See HOSTED_MAX_TOKENS.
            _llm = OpenRouterLLM(OPENROUTER_MODEL, keys=_or_keys,
                                 fallback_models=OPENROUTER_FALLBACKS)
            print(f"  agent 4: {_llm.name} - no local weights, no VRAM")
            print(f"    provider chain ({len(_llm.plan)}): {' -> '.join(_llm.plan)}")
            print("    a 429 saying 'rate-limited upstream' is the free PROVIDER saturated, not "
                  "your keys, so the chain changes provider rather than spending more keys on the "
                  "same one. If every provider is saturated at once, add a BYOK provider key at "
                  "openrouter.ai/settings/integrations for dedicated limits.")
        else:
            # Serving Agents 1-2 only is a WORKING configuration, not a failure. `/video?stages=12`
            # stops at the Agent 2 -> Agent 3 seam and the caller runs Agents 3-4 where its keys
            # already are - which is the point, since the keys need never reach Kaggle at all.
            _llm = None
            print("  agent 4: NOT SERVED HERE - no OpenRouter keys in this session.")
            print("    This is the intended split when your keys are local: run")
            print("      python web/local_backend.py --kaggle <this tunnel address>")
            print("    and point the front end at http://127.0.0.1:8899. It calls "
                  "/video?stages=12 here")
            print("    and runs Agents 3, 4 and C1-C5 on your machine, keys included.")
            print("    To serve Agent 4 from Kaggle instead, put a comma-separated list in")
            print("    Add-ons -> Secrets as OPENROUTER_API_KEY_LIST.")
        STATE["reporter"] = None if _llm is None else CaregiverReporter(_llm, config=cfg)
        STATE["agent4"] = None if _llm is None else _llm.name
        STATE["ready"] = True
        STATE["loaded_at"] = time.time()
    except Exception as exc:                      # noqa: BLE001 - surfaced via /health
        STATE["error"] = f"{type(exc).__name__}: {exc}"

threading.Thread(target=_load, daemon=True).start()

def _auth(tok):
    if TOKEN and tok != TOKEN:
        raise HTTPException(401, "bad or missing X-BS-Token")

@app.get("/health")
def health():
    import torch
    return {"service": "behaviorsense", "ready": STATE["ready"], "error": STATE["error"],
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "streams": STATE.get("streams", []), "tau": TAU, "auth": bool(TOKEN),
            # The front end hides the upload control when pose is unavailable rather than
            # offering one that 503s: a disabled button nobody can explain is worse than
            # a control that is honestly not there.
            # Which model writes the reports. The page shows it, because a hosted `:free`
            # endpoint must never be mistaken for the pinned checkpoint the measured numbers
            # came from.
            "agent4": STATE.get("agent4"),
            "video": bool(RTMO), "reid": bool(OSNET), "max_upload_mb": MAX_UPLOAD_MB,
            "max_frames": 900}"""),
("code", """# Three endpoints, mirroring the two halves of the system plus the one the front end needs.
#
# /demo exists because the alternative is worse: to exercise Agent 4 the browser would have
# to POST a fully-formed BehaviourState - DailyFeatures, thirty days of baselines, alerts -
# and hand-building that in JavaScript would mean the demo tests a payload nobody else
# constructs. Instead the backend runs the SAME simulator the evaluation uses, picks a day
# that actually alerts, and returns the claims, the verdicts, AND the evidence index they
# were checked against. The front end then re-runs its own port of the verifier over that
# evidence and compares - so the page shows two independent implementations agreeing on real
# model output, rather than asking anyone to trust one.
import json
from fastapi.responses import StreamingResponse
from behaviorsense.data.simulator import standard_scenarios

class Windows(BaseModel):
    windows: list          # [N, 30, 2, 17, 3] skeleton windows
    smooth: bool = True

class StatePayload(BaseModel):
    state: dict            # a serialised BehaviourState

@app.post("/activity")
def activity(body: Windows, x_bs_token: str = Header(default="")):
    _auth(x_bs_token)
    if not STATE["ready"]:
        raise HTTPException(503, STATE["error"] or "models still loading")
    X = np.asarray(body.windows, dtype=np.float32)
    if X.ndim != 5 or X.shape[-2:] != (17, 3):
        raise HTTPException(422, f"expected [N,T,M,17,3], got {list(X.shape)}")

    logits = STATE["clf"].logits(X)
    # Adjust BEFORE smoothing. Measured: smoothing the raw posterior cost 0.040 mean-class,
    # smoothing the adjusted one cost 0.028 and kept most of the top-1 gain.
    adjusted = logit_adjust(logits, class_prior(logits.argmax(1)), TAU)
    post = softmax(adjusted / 0.76)              # T fitted in notebook 04
    labels = post.argmax(1)
    if body.smooth:
        cfg = ActivityConfig()
        A = apply_emergency_floor(
            fit_transition_matrix([labels]), FALLING, cfg.emergency_floor)
        labels = viterbi(np.log(np.clip(post, 1e-12, None)), np.log(A),
                         np.log(np.full(post.shape[1], 1.0 / post.shape[1])))
    return {"labels": labels.tolist(), "confidence": post.max(1).round(4).tolist(),
            "tau": TAU, "smoothed": body.smooth}

def _blocking(fn, label, every=10.0):
    # Run a blocking call in a worker thread, emitting a heartbeat while it works.
    #
    # (Comments, not a docstring: this whole cell is one triple-quoted literal in
    # _generate.py and an inner triple quote terminates it early - the same trap the
    # _AdjustedClassifier note in the /video cell already describes.)
    #
    # A generator cannot yield from inside a blocking call, so streaming one line per AGENT
    # left the longest silence in the stream equal to the slowest stage: Agent 3's line went
    # out, then nothing at all for the whole of generation. The browser's silence watchdog
    # fired at 90 s and reported a dead session while the GPU was working perfectly normally.
    # Observed on a 5-second clip.
    #
    # Per-stage lines were never sufficient by themselves. The silence that matters is the one
    # INSIDE a stage, and only a second thread can interrupt it. `every` bounds that silence
    # regardless of how long the work takes, which is the property the watchdog needs: the
    # client can then treat silence as genuine failure rather than as a slow model.
    #
    # Yields ("beat", {...}) zero or more times, then exactly one ("done", value). Exceptions
    # propagate out of `fut.result()` on the caller's thread, so existing try/except still
    # catches them where it did before.
    import concurrent.futures as _cf
    with _cf.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        t0 = time.time()
        while True:
            try:
                value = fut.result(timeout=every)
                break
            except _cf.TimeoutError:
                yield "beat", {"step": label, "elapsed_s": round(time.time() - t0, 1)}
    yield "done", value


def _serialise(out, state):
    from behaviorsense.agents.reasoning.verifier import FaithfulnessVerifier
    index = FaithfulnessVerifier().build_index(state)
    return {
        "day": state.report_day.isoformat(),
        "subject": state.subject_role.value,
        "summary": out.report.summary,
        "recommendation": out.report.recommendation,
        "escalate": out.report.escalate,
        "claims": [c.model_dump(mode="json") for c in out.report.claims],
        "verifications": [v.model_dump(mode="json") for v in out.report.verifications],
        "evidence": {k: v.model_dump(mode="json") for k, v in index.items()},
        "alerts": [{"kind": a.kind.value, "severity": a.severity.value} for a in state.alerts],
        "hallucination_rate": out.hallucination_rate,
        "emitted": out.n_emitted_claims,
        "schema_rejected": out.dropped_claims,
        "model": out.report.model_name,
    }

@app.get("/demo")
def demo(scenario: int = 0, x_bs_token: str = Header(default="")):
    # STREAMED for the same reason /video is. Measured 2026-08-25 against a live T4: this
    # endpoint was cut at 125.8 s with Cloudflare's 524 while the model was still generating,
    # twice, on two different sessions. The cap is on time-to-first-byte, so `accepted` goes
    # out immediately and `progress` lines keep the connection busy while Qwen runs.
    #
    # NDJSON, same envelope as /video:
    #   {"event":"accepted"} -> {"event":"progress",...}* -> {"event":"result","payload":{...}}
    #   or {"event":"error","detail":...}
    # `result.payload` is exactly what this endpoint returned before.
    _auth(x_bs_token)
    if not STATE["ready"]:
        raise HTTPException(503, STATE["error"] or "models still loading")
    # 503 with the reason, not a KeyError 500. This backend can legitimately serve Agents 1-2
    # only - on a GPU whose torch build has no kernels, there is nothing to write reports with.
    if STATE.get("reporter") is None:
        raise HTTPException(503, "this backend serves Agents 1-2 only: no reporter is configured. "
                                 "Set OPENROUTER_API_KEY_LIST, attach the Qwen model on a "
                                 "supported GPU, or use /video?stages=12 and run Agent 4 yourself.")
    scenarios = list(standard_scenarios())
    if not 0 <= scenario < len(scenarios):
        raise HTTPException(404, f"scenario {scenario} of {len(scenarios)}")

    def stream():
        def emit(obj):
            return json.dumps(obj, default=str) + "\\n"

        key = f"demo:{scenario}"
        try:
            yield emit({"event": "accepted", "scenario": scenario,
                        "cached": key in STATE})
            if key in STATE:                  # a generation takes >2 min; cache per scenario
                yield emit({"event": "result", "payload": STATE[key]})
                return
            from behaviorsense.agents.behaviour import BehaviourAnalyzer
            analyzer = BehaviourAnalyzer()
            picked = None
            for day in scenarios[scenario].run().days:
                st = analyzer.analyze_day(day)
                if st.baselines and st.alerts:
                    picked = st               # last alerting day: the decline is developed
            if picked is None:
                yield emit({"event": "error", "kind": "simulator",
                            "detail": "simulator produced no alerting day"})
                return
            yield emit({"event": "progress", "step": "state_ready",
                        "day": picked.report_day.isoformat(),
                        "alerts": len(picked.alerts),
                        "note": "writing the report - this is the slow stage"})
            # HEARTBEAT THROUGH GENERATION. Without this the stream goes quiet for the whole
            # of `report()`, which is minutes on a T4, and the client cannot tell that from a
            # dead session.
            out = None
            for _kind, _obj in _blocking(lambda: STATE["reporter"].report(picked),
                                         "qwen_generating"):
                if _kind == "beat":
                    yield emit({"event": "heartbeat", **_obj})
                else:
                    out = _obj
            STATE[key] = _serialise(out, picked)
            yield emit({"event": "result", "payload": STATE[key]})
        except Exception as exc:                                   # noqa: BLE001
            import traceback
            yield emit({"event": "error", "kind": "server",
                        "detail": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-1200:]})

    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store",
                                      "X-Accel-Buffering": "no"})

@app.post("/report")
def report(body: StatePayload, x_bs_token: str = Header(default="")):
    _auth(x_bs_token)
    if not STATE["ready"]:
        raise HTTPException(503, STATE["error"] or "models still loading")
    if STATE.get("reporter") is None:
        raise HTTPException(503, "no reporter is configured on this backend - it serves Agents "
                                 "1-2 only. Run Agent 4 where your API keys are.")
    from behaviorsense.schemas import BehaviourState
    state = BehaviourState(**body.state)
    return _serialise(STATE["reporter"].report(state), state)"""),
("code", """# /video - the only endpoint that touches pixels, and the only one that can be handed a
# file nobody vetted.
#
# Decode runs in a CHILD process (behaviorsense.video.extract_isolated). That is not
# defensive habit: ffmpeg raises SIGSEGV/SIGABRT on malformed streams, a signal is not an
# exception, and notebook 02 lost finished corpora to exactly this before it grew a
# journalling subprocess probe. Inline, one bad upload would kill this kernel and take the
# tunnel, the models and the demo with it. Behind the boundary it is a 422.
#
# What comes back is STAGED: one record per agent, in order, with its own timing, payload and
# status, plus the C1-C5 verdict on every individual claim. That shape exists so the front end
# can show what each agent produced and what the next one received, for one person or several.
#
# Agents 3 and 4 DO run on a single clip, which earlier versions refused. The refusal was
# right about the science and wrong about the remedy: Agent 3's baseline is a 14-day rolling
# median, so one clip cannot supply this person's history. Rather than omit the stages or
# fabricate a history, the baseline is a DECLARED simulated reference and every affected
# number says so. The feature values are measured from the video and are real; the robust-z
# and alerts are reference-relative. C1-C4 still check each claim against the state computed
# from this video, so the verifier's guarantee is unchanged - what the reference cannot
# support is the clinical reading, and the payload says that too.
import json, shutil, tempfile, threading, time, numpy as np
_VIDEO_LOCK = threading.Lock()


def _rss_mb():
    # Resident set size, from /proc - no dependency, exact, and the only number that answers
    # "which stage grows the process". RAM went 16.5 -> 29.6 GiB across two uploads on a 30 GiB
    # box, so a third would be killed. Guessing has already cost two wrong diagnoses; the
    # simulator rebuild, for one, is 1.3 MB measured and is not the cause.
    try:
        with open("/proc/self/status") as _f:
            for _l in _f:
                if _l.startswith("VmRSS:"):
                    return round(int(_l.split()[1]) / 1024.0, 1)
    except Exception:                                              # noqa: BLE001
        pass
    return None
from fastapi import File, Form, UploadFile
from fastapi.responses import StreamingResponse
from behaviorsense.agents.activity import (CLASS_NAMES, EXTENDED_CLASS_NAMES,
                                            FALLING, FALLEN)
def class_names_served():
    # The name list matching the head that is ACTUALLY loaded. Resolved per request.
    #
    # (Comments, not a docstring: this whole cell is one triple-quoted literal in _generate.py and
    # an inner triple quote terminates it early - the same trap _AdjustedClassifier documents.)
    #
    # Charades checkpoints emit 20 classes; the Toyota RTMO run emits 22, where ids 20-21 are
    # `using_device` and `object_interaction`. Indexing the 20-entry tuple at 20 raises
    # `IndexError: tuple index out of range`.
    #
    # This was a module-level constant, and that could not work: `_load()` runs on a BACKGROUND
    # thread so the tunnel address prints in seconds rather than after the weights load, so
    # STATE has no `n_classes` when this cell executes. The `.get()` default won every time and the
    # 20-name tuple was captured permanently - for a 22-class model. It surfaced in the top-3
    # diagnostic, but the same tuple names every segment, so any clip Agent 2 labelled
    # `using_device` or `object_interaction` would have 500'd the whole upload.
    #
    # Read from STATE at call time, falling back to the classifier itself rather than to a length:
    # a wrong name is worse than a missing one, because it is reported as fact.
    n = STATE.get("n_classes")
    if n is None:
        clf = STATE.get("clf")
        n = getattr(clf, "n_classes", None) or len(CLASS_NAMES)
    return EXTENDED_CLASS_NAMES if int(n) > len(CLASS_NAMES) else CLASS_NAMES
from behaviorsense.agents.activity import ActivityConfig
from behaviorsense.pipeline import (ActivityPipeline, frames_to_windows,
                                   observed_hours_from_frames)
from behaviorsense.data.skeleton_dataset import normalise as _normalise
from behaviorsense.schemas import Role
from behaviorsense.video import MAX_FRAMES, extract_isolated, rebuild_observations

@app.post("/video")
async def video(file: UploadFile = File(...), x_bs_token: str = Header(default=""),
                stages: str = "1234",
                # The operator's gallery, as JSON, and an optional enrol name. Both arrive
                # from the LOCAL backend (web/local_backend.py), which owns gallery.json -
                # the file never lives on Kaggle, so biometric templates never touch a
                # third-party service. Direct-to-tunnel uploads simply omit them.
                gallery: str = Form(""),
                enrol: str = Form("")):
    _auth(x_bs_token)
    if not STATE["ready"]:
        raise HTTPException(503, STATE["error"] or "models still loading")
    if not RTMO:
        raise HTTPException(503, "rtmo-l.onnx is not attached, so pose extraction is off")

    # The upload is read HERE, while an HTTP status is still available to reject it. Once the
    # streaming response has begun, headers are sent and 413/422 are no longer options.
    td = tempfile.mkdtemp(prefix="bs-upload-")
    # The operator's gallery, materialised as a file for the decode child. Written under the
    # upload's own temp dir so the existing finally-rmtree removes it - biometric templates
    # must not outlive the request on this machine, which is the whole reason the gallery
    # lives on the operator's side and travels per request.
    use_osnet = OSNET and bool((gallery or "").strip() or enrol.strip())
    gallery_path = None
    if use_osnet and gallery.strip():
        gallery_path = pathlib.Path(td) / "gallery.json"
        gallery_path.write_text(gallery.strip(), encoding="utf-8")
    dst = pathlib.Path(td) / (pathlib.Path(file.filename or "clip").name or "clip")
    size = 0
    try:
        with dst.open("wb") as fh:
            # Streamed in chunks and capped as it arrives. Reading the whole body first to
            # measure it would let a large upload exhaust memory before the check runs.
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_MB << 20:
                    raise HTTPException(413, f"upload exceeds {MAX_UPLOAD_MB} MB")
                fh.write(chunk)
        if not size:
            raise HTTPException(422, "empty upload")
    except BaseException:
        shutil.rmtree(td, ignore_errors=True)
        raise

    # ONE CLIP AT A TIME. Two concurrent uploads each spawn a decode child holding its own CUDA
    # context for RTMO and OSNet, and both then contend for a Qwen generation that already fills
    # both T4s - the kernel died rather than either finishing. A 429 with a plain reason is a
    # worse demo than a queue and a far better one than a dead tunnel, and the front end can act
    # on it. `_VIDEO_LOCK` is non-blocking on purpose: queueing behind a 5-minute request would
    # be indistinguishable from a hang.
    if not _VIDEO_LOCK.acquire(blocking=False):
        raise HTTPException(429, "a clip is already being analysed - this session runs one at a "
                                 "time because pose extraction and the 7B model each need the "
                                 "whole GPU. Wait for the current run to finish and retry.")
    try:
            # `?stages=12` stops after Agent 2 and returns poses + segments only. That is the
        # AGENT 2 -> AGENT 3 seam, which is where this project already claims pixels stop, so a
        # caller can run Agents 3 and 4 wherever their LLM keys live - off this box, out of Kaggle
        # Secrets, and off the GPU. The full "1234" default keeps standalone Kaggle working.
        return StreamingResponse(_video_stream(td, dst, time.time(), stages=stages,
                                              use_osnet=use_osnet,
                                              gallery_path=gallery_path),
                                 media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-store",
                                          "X-Accel-Buffering": "no"})
    except BaseException:
        _VIDEO_LOCK.release()
        raise


def _video_stream(td, dst, t_start, stages="1234", use_osnet=False, gallery_path=None):
    # STREAMED, one line per agent, and this is a correctness fix rather than a nicety.
    #
    # The quick tunnel drops a request whose origin has not answered in about two minutes:
    # measured 2026-08-25, /demo was cut at 125.8 s with Cloudflare's 524 while the T4 was
    # still generating. /video is strictly slower - RTMO over up to 900 frames, then
    # ST-GCN++, then that same Qwen pass - so one synchronous JSON response cannot carry it.
    #
    # Cloudflare's limit is on time-to-first-byte, so the fix is to send a byte at once and
    # keep sending. Shortening the report or lowering max_frames would trade measured
    # behaviour for a timeout, which is the trade this project keeps refusing.
    #
    # Newline-delimited JSON. `stage` arrives as each agent finishes, which is also exactly
    # what the front end wants to draw, so the timeout fix and the feature are the same code:
    #   {"event":"accepted"}              first, immediately - stops the 524 clock
    #   {"event":"stage","stage":{...}}   one per agent, in order, as it completes
    #   {"event":"heartbeat",...}         inside a long stage, so silence stays bounded
    #   {"event":"result","payload":{...}} the SAME dict this endpoint used to return
    #   {"event":"error","detail":...}    IN-BAND: headers are already sent by now, so an
    #                                    HTTP status can no longer carry a failure
    # `result` is byte-for-byte the old payload, so stages.js and the contract tests keep
    # describing one shape.
    def emit(obj):
        return json.dumps(obj, default=str) + "\\n"

    try:
        yield emit({"event": "accepted", "max_frames": MAX_FRAMES,
                    "note": "decoding in a child process; stages follow as they finish"})
        try:
            # Pose runs in a child process over up to MAX_FRAMES frames. It is minutes of GPU on a
            # long upload, and the child itself only speaks every PROGRESS_EVERY frames - so
            # `_blocking` emits a heartbeat on its own 10 s timer regardless of what the child
            # reports. That timer, not the child, is what keeps the browser's silence watchdog fed.
            poses = None
            for _kind, _obj in _blocking(
                    lambda: extract_isolated(str(dst), rtmo=RTMO, device="cuda",
                                             # RTMO on CUDA via onnxruntime; OSNet follows torch's
                                             # own arch list, which is a different question - on a
                                             # P100 the ONNX model runs and every torch kernel
                                             # fails with "no kernel image is available".
                                             osnet_device=STATE.get("torch_device", "cuda"),
                                             # OSNet only when this request can use it: a
                                             # gallery to match against, or an enrolment to
                                             # collect crops for. Otherwise the 104 s CPU
                                             # cost buys the string "unidentified".
                                             osnet=OSNET if use_osnet else None,
                                             # The operator's gallery, as a file under this
                                             # request's temp dir. `extract_isolated` passes
                                             # it by PATH so biometric data never appears in
                                             # a process listing, and the existing
                                             # finally-rmtree removes it with the upload.
                                             gallery=gallery_path,
                                             target_fps=STATE.get("sample_fps", SAMPLE_FPS),
                                             # NO TIMEOUT. `extract_isolated` watches the child's
                                             # own progress lines instead and gives up only on
                                             # silence, because any wall-clock ceiling has to be
                                             # set high enough for the longest acceptable video and
                                             # so cannot tell a slow clip from a wedged GPU.
                                             max_frames=MAX_FRAMES),
                    "rtmo_decoding"):
                if _kind == "beat":
                    yield emit({"event": "heartbeat", **_obj})
                else:
                    poses = _obj
        except ValueError as exc:
            yield emit({"event": "error", "kind": "decode", "detail": str(exc)})
            return
        finally:
            shutil.rmtree(td, ignore_errors=True)

        t_pose = time.time()
        for kind, obj in _video_stages(poses, t_start, t_pose, stages=stages,
                                       reid_note={"requested": bool(use_osnet),
                                                  "gallery_attached": gallery_path is not None,
                                                  "weights_present": bool(OSNET)}):
            if kind == "stage":
                yield emit({"event": "stage", "stage": obj})
            elif kind == "heartbeat":
                yield emit({"event": "heartbeat", **obj})
            else:
                yield emit({"event": "result", "payload": obj})
    except Exception as exc:                                       # noqa: BLE001
        # In-band, because the response has already begun. A truncated NDJSON stream with no
        # error line is indistinguishable to the client from a dropped connection, and this
        # project has spent enough sessions on failures that looked like something else.
        import traceback
        yield emit({"event": "error", "kind": "server",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-1200:]})
    finally:
        shutil.rmtree(td, ignore_errors=True)
        # RECLAIM BEFORE THE NEXT UPLOAD. Two runs took RSS from 16.5 to 29.6 GiB of 30, so a
        # third is killed by the OOM reaper - and a killed kernel takes the tunnel with it. This
        # does not fix a leak, it buys the demo a third run: `gc.collect()` frees the reference
        # cycles a generator's frames leave behind, and `empty_cache()` returns the allocator's
        # unused blocks so the decode child can open its CUDA context next time.
        try:
            import gc as _gc, torch as _tt
            _freed = _gc.collect()
            if _tt.cuda.is_available():
                _tt.cuda.empty_cache()
            print(f"  reclaimed {_freed} objects; RSS now {_rss_mb()} MB")
        except Exception as _e:                                    # noqa: BLE001
            print(f"  reclaim skipped: {type(_e).__name__}: {_e}")
        # Released HERE, not where the response was constructed: the work happens while this
        # generator is consumed, so releasing earlier would let a second upload in mid-run.
        _VIDEO_LOCK.release()


def _video_stages(poses, t_start, t_pose, stages="1234", reid_note=None):
    _rss = {"start": _rss_mb()}
    # The four agents over one clip, yielding each stage record as it is finished rather than
    # after all of them are. Transport lives in `_video_stream`; this function knows nothing
    # about HTTP, which is what lets the contract suite execute it on a laptop.
    frames = rebuild_observations(poses)

    def _stage(n, name, status, elapsed, payload):
        return {"agent": n, "name": name, "status": status,
                "elapsed_s": round(elapsed, 2), "payload": payload}

    # AGENT 1, emitted NOW - before Agent 2 has run, which is the point of streaming.
    #
    # `n_people` counts distinct track ids in AGENT 1's own output, not in Agent 2's segment
    # list. Those differ, and the difference is the interesting case: a person who is tracked
    # but too occluded to classify produces no segments, so counting Agent 2's tracks reported
    # fewer people than the overlay draws. The page documents that state as a dashed
    # "pose unusable" box - "present, unreadable" and "not there" are different facts, and
    # Agent 1's own card must report the one Agent 1 established.
    tracked = {}
    _seen_roles = {}
    for _f in frames:
        for _p in _f.persons:
            # THE SETTLED ROLE, not the first frame's. `setdefault` kept the FIRST
            # observation, and the first observation of every track is `unknown` by
            # construction - the role vote has no evidence yet (see P3d's "honest ramp-up").
            # So Agent 1's card said `unidentified · track 0` for a track Agent 2 reported on
            # the same page as `resident ... Mary`. Two cards disagreeing about identity is
            # worse than either answer alone. Majority over the non-unknown observations: the
            # ramp-up frames do not outvote what the track actually converged to.
            _seen_roles.setdefault(_p.track_id, []).append(_p.role.value)
    # Resolve BEFORE stage1 reads `tracked` - it reports `n_people` as len(tracked), so a
    # resolution placed later left Agent 1 announcing zero people on a clip with two.
    for _tid, _roles in _seen_roles.items():
        _known = [r for r in _roles if r != "unknown"]
        tracked[_tid] = max(set(_known), key=_known.count) if _known else "unknown"

    stage1 = _stage(1, "perception", "done", t_pose - t_start, {
        "fps": poses["fps"], "frames_kept": poses["frames_kept"],
        "truncated": poses["truncated"], "providers": poses["providers"],
        "n_people": len(tracked), "reid": poses["reid"],
        # WHY re-identification did or did not run. Without this, "re-id off" on a request
        # that asked to enrol is indistinguishable from a request that did not ask - which
        # is exactly the ambiguity a failed first enrolment produced.
        "reid_note": reid_note or {},
        # Pose QUALITY, so a bad overlay can be diagnosed from the page instead of guessed at.
        # RTMO has no detector in front of it: on a cluttered scene it invents low-confidence
        # people, and it also loses real ones whose joints all fall under the threshold. Those
        # two look the same on screen, and both look like "the model is broken".
        "mean_kp_score": poses.get("mean_kp_score"),
        "weak_person_records": poses.get("weak_person_records"),
        "person_records": poses.get("person_records"),
        "container_size": poses.get("container_size"),
        "decoded_size": [poses.get("width"), poses.get("height")],
        # Temporal geometry. A 30-frame window is 2.0 s at the 15 Hz the shards were built at,
        # and ST-GCN++ has seen no other duration. Reported so a rate mismatch is visible
        # rather than absorbed as bad accuracy.
        "source_fps": poses.get("source_fps"),
        "window_seconds": poses.get("window_seconds"),
        # Against the SERVED rate, not `video.SHARD_FPS`. That constant is 15.0 (Charades), so a
        # correctly-configured 20 Hz Toyota run was told its labels were outside measured
        # conditions - the banner accusing the one configuration that is right.
        "rate_matches_shards": abs(poses["fps"] - STATE.get("sample_fps", SAMPLE_FPS)) < 0.01,
        "trained_fps": STATE.get("sample_fps", SAMPLE_FPS),
        "tracks": [{"track_id": k, "role": v} for k, v in sorted(tracked.items())],
    })
    _rss["after_pose"] = _rss_mb()
    yield "stage", stage1

    # AGENT 2 THROUGH THE LIBRARY PATH, not a re-implementation. This endpoint used to do
    # its own logit_adjust -> softmax -> argmax -> run-length pass, which SKIPPED Viterbi
    # smoothing and abstention entirely - so the served labels were not the configuration
    # notebook 04 evaluated, and the demo would have shown numbers no table backs. The
    # adapter applies the selected tau inside `logits()` so `ActivityPipeline` sees exactly
    # the posteriors the measured configuration produces.
    class _AdjustedClassifier:
        # The served ensemble with the selected logit adjustment folded in. A `\"\"\"`
        # docstring cannot go here: this whole cell is one triple-quoted literal in
        # _generate.py, and an inner triple quote terminates it early - which is exactly
        # how this cell broke the generator once already.
        def __init__(self, inner, tau):
            self.inner, self.tau = inner, tau
            # FORWARD the head size. `ActivityPipeline` reads `n_classes` off the classifier to
            # size the transition matrix and the Viterbi prior; an adapter that swallows the
            # attribute silently reverts the decoder to 20 classes, and the mismatch surfaces as
            # `operands could not be broadcast together with shapes (20,) (22,)` inside Viterbi -
            # three stages after the wrapper that caused it.
            _n = getattr(inner, "n_classes", None)
            if _n:
                self.n_classes = int(_n)

        def logits(self, windows):
            lg = self.inner.logits(windows)
            # n_classes FROM THE LOGITS. `class_prior` defaults to 20, so a 22-class
            # head produced a (20,) prior against (N,22) logits and raised
            # "operands could not be broadcast together with shapes (28,22) (1,20)" -
            # in Agent 2, after Agent 1 had already spent 35 s on pose.
            return logit_adjust(
                lg, class_prior(lg.argmax(1), n_classes=lg.shape[1]), self.tau)

    pipe = ActivityPipeline(_AdjustedClassifier(STATE["clf"], TAU),
                            ActivityConfig(temperature=TEMPERATURE))
    obs_hours = observed_hours_from_frames(frames, fps=poses["fps"] or 15.0)
    segments = pipe.run(frames)

    # WHO IS THE SUBJECT? On an uploaded clip, nobody - and that silently zeroed everything.
    #
    # `aggregate_daily_features` attributes personal features only to segments whose role IS
    # the subject role: `subject = [s for s in segments if s.role is subject_role]`. OSNet
    # decides that role by matching against an ENROLLED gallery, and a stranger's clip has no
    # gallery, so every track comes back `unknown`, `subject` is empty, and all 27 features are
    # 0.0 - including `social_interaction_duration_s` while Agent 2 was reporting 4.4 s of
    # `interacting_with_person`. Deviations then clamp at -10 on every feature and Agent 4
    # writes a confident report about a person who did nothing. Observed, on a 5-second clip of
    # someone washing dishes.
    #
    # Refusing to run is not better - it would hide the verifier, which is the part worth
    # showing. So the clip's most-present track is treated as the subject and that assumption
    # is DECLARED, exactly as the simulated baseline already is. Identity here is ASSERTED,
    # not re-identified, and `subject_provenance` says so on the payload and in the banner.
    #
    # A real ReID match is never overridden: if any segment already carries RESIDENT, the
    # gallery spoke and this stays out of the way.
    _seg_roles = {s.role for s in segments}      # noqa: F841 - kept for the assertion below
    # Captured BEFORE any relabel. The cards must report what Agent 1's re-identifier actually
    # decided; showing "resident" where OSNet said `unknown` would be the system lying about
    # its own identity layer to make the demo tidier.
    reid_roles = {s.track_id: s.role.value for s in segments}
    subject_track, subject_provenance = None, "reid_matched"
    subject_tracks: list[int] = []
    # HOISTED out of the branch below: a caller running Agents 3-4 itself needs these counts to
    # assert a subject, and they were only built when re-identification had already failed - so
    # the seam return NameError'd on exactly the clips where the gallery did match.
    _frames_per_track = {}
    _spans: dict[int, tuple[int, int]] = {}
    _boxes: dict[int, tuple[list[float], list[float]]] = {}
    for _f in frames:
        for _p in _f.persons:
            _frames_per_track[_p.track_id] = _frames_per_track.get(_p.track_id, 0) + 1
            _i = int(_f.frame_idx)
            _cur = _spans.get(_p.track_id)
            _spans[_p.track_id] = (_i, _i) if _cur is None else (min(_cur[0], _i),
                                                                max(_cur[1], _i))
            _box = [float(_p.box.x1), float(_p.box.y1),
                    float(_p.box.x2), float(_p.box.y2)]
            _seen = _boxes.get(_p.track_id)
            # First and last box only: the merge asks whether a person could have WALKED from
            # where one fragment ended to where the next began, so the ends are the whole question.
            _boxes[_p.track_id] = ((_box, _box) if _seen is None else (_seen[0], _box))
    # ONE implementation of the subject decision, shared with `web/local_backend.py`. This used to
    # be inlined here, which is how the two halves could disagree about who the clip is about; the
    # merge rule is subtle enough that two copies of it is not a risk worth taking.
    segments, subject_track, subject_provenance, subject_tracks = _assert_subject(
        segments, _frames_per_track, spans=_spans, boxes=_boxes,
        fps=float(poses["fps"] or SAMPLE_FPS))

    # Resolved ONCE per request, from the classifier that is actually loaded. Both the segment
    # names and the diagnostic below index it, and a stale 20-entry tuple raises on exactly the two
    # classes Toyota added.
    _names = class_names_served()
    if len(_names) < (STATE.get("n_classes") or len(_names)):
        raise AssertionError(
            f"the served head has {STATE.get('n_classes')} classes but only {len(_names)} names "
            "are available - naming a class by the wrong string would report a wrong activity as "
            "fact. Extend EXTENDED_CLASS_NAMES to match the taxonomy the checkpoint was trained "
            "on.")

    by_track = {}
    for s in segments:
        # `s.role`, not `s.subject_role`. ActivitySegment carries the role of the ONE person
        # it belongs to; `subject_role` is a BehaviourState field (the day's subject). Getting
        # this wrong 500'd every upload, and it 500'd OUTSIDE the try/except below, so it took
        # the whole request rather than degrading to a failed stage.
        rec = by_track.setdefault(s.track_id, {
            "track_id": s.track_id,
            # The RE-IDENTIFIED role, not the possibly-asserted one. `is_subject` carries the
            # assertion separately so the page can say "this track was treated as the subject"
            # without claiming OSNet recognised anybody.
            "role": reid_roles.get(s.track_id, s.role.value),
            "is_subject": s.track_id == subject_track,
            # The enrolled NAME the gallery matched this track to, when it matched. The role
            # already says resident/stranger; the name is what a caregiver reads.
            "matched_name": (poses.get("track_names") or {}).get(str(s.track_id)),
            "segments": []})
        rec["segments"].append({
            "label": int(s.activity_id), "name": _names[int(s.activity_id)],
            "t0": round((s.start_time - frames[0].timestamp).total_seconds(), 2),
            "t1": round((s.end_time - frames[0].timestamp).total_seconds(), 2),
            "confidence": round(float(s.confidence), 3),
            "room": s.room,
        })
    tracks = list(by_track.values())
    for t in tracks:
        t["n_segments"] = len(t["segments"])
        t["fall_segments"] = sum(1 for q in t["segments"] if q["label"] in (FALLING, FALLEN))
    t_cls = time.time()

    # AGENT 2, emitted as soon as it has finished - the front end draws the timelines while
    # Agent 3 is still priming and Qwen has not been called.
    # WHAT WAS AGENT 2 CHOOSING BETWEEN? A single segment spanning the whole clip is either a
    # confident correct call or Viterbi's self-transition prior (0.9) collapsing 28 uncalibrated
    # windows into one state, and the segment list alone cannot tell those apart. The top-3
    # per-window posteriors make the difference visible instead of arguable.
    _top = []
    try:
        # NORMALISE HERE TOO. This diagnostic bypassed `ActivityPipeline` to reach the raw
        # windows and therefore reproduced the exact bug it exists to diagnose - the guard
        # refused it with "windows peak at 356 in x/y". Written to explain a scale error and
        # containing one.
        _w = np.stack([_normalise(w.astype(np.float32))
                       for w in next(iter(frames_to_windows(frames))).windows])
        _lg = STATE["clf"].logits(_w)
        _pp = softmax(logit_adjust(_lg, class_prior(_lg.argmax(1), n_classes=_lg.shape[1]), TAU)
                      / max(TEMPERATURE, 1e-6))
        for _row in _pp[:40]:
            _o = np.argsort(-_row)[:3]
            _top.append([[_names[int(i)], round(float(_row[int(i)]), 3)] for i in _o])
    except Exception as _e:                                        # noqa: BLE001
        _top = [[["diagnostic unavailable", 0.0]]]
        print(f"  top-3 diagnostic skipped: {type(_e).__name__}: {_e}")

    stage2 = _stage(2, "activity", "done" if tracks else "skipped", t_cls - t_pose, {
        "window_top3": _top,
        "n_windows": len(_top),
        "streams_served": STATE.get("streams"), "tau": TAU, "temperature": TEMPERATURE,
        "decoding": "calibrate -> object fusion -> Viterbi -> abstain -> segments",
        "n_segments": len(segments),
        "per_track": tracks,
        "subject_provenance": subject_provenance,
        "subject_track": subject_track,
        # ALL the tracks attributed to the subject, not just the primary. One person fragmented
        # into three ids on a 45 s clip, and naming only the most-present one made the page claim
        # a single track while Agent 3 counted a merged set.
        "subject_tracks": list(subject_tracks),
    })
    _rss["after_classify"] = _rss_mb()
    yield "stage", stage2

    # AGENTS 3 AND 4 ON ONE CLIP - and why this needs a DECLARED provenance, not a quiet
    # default. Agent 3's deviation is robust-z against a 14-day rolling median. One upload is
    # a single moment, so there is no history for this person. Two options existed and only
    # one is honest: fabricate a baseline and let the report read as though it were checked
    # against the resident's own past, or run the layer against a declared reference and say
    # so on every affected number.
    #
    # The verifier is unaffected either way, and that is the point. C1-C4 check each claim
    # against the state actually computed from THIS video, so an invented figure, a wrong
    # percentage or an inverted direction is still caught arithmetically. What a reference
    # baseline cannot support is the clinical reading ("mobility declined"), because the
    # comparison point is not this person. Both facts ship in the response.
    stage3, stage4, checks = None, None, []
    t3 = t_cls
    st = None
    if "3" not in stages:
        # STOP AT THE SEAM. The caller runs Agents 3 and 4 themselves - typically a local backend
        # holding the LLM keys, so no credential ever reaches Kaggle Secrets or a saved notebook.
        # `stages` is echoed in the payload so a consumer cannot mistake a truncated run for a
        # complete one that produced no claims.
        yield "result", {
            "fps": poses["fps"], "width": poses["width"], "height": poses["height"],
            "frames_kept": poses["frames_kept"], "truncated": poses["truncated"],
            "reid": poses["reid"], "providers": poses["providers"],
            "frames": poses["frames"], "tracks": tracks, "n_people": len(tracked),
            "stages": [stage1, stage2], "checks": [], "stages_run": "12",
            "subject_provenance": subject_provenance, "subject_track": subject_track,
            "subject_tracks": list(subject_tracks),
            "frames_per_track": {str(k): v for k, v in _frames_per_track.items()},
            # Per-track first/last frame. Only this half has the per-frame observations, and the
            # local half needs them to reunite one person's fragmented ids - a track that leaves
            # detection for longer than the tracker's ~15 s bridge comes back with a new id.
            "track_spans": {str(k): [v[0], v[1]] for k, v in _spans.items()},
            # ENROLMENT MATERIAL for the local half: the subject track's best crops as
            # embeddings (never pixels), plus whether a gallery actually matched this clip -
            # the local backend persists these only when the operator asked to enrol.
            "gallery_updates": {
                "subject_track": subject_track,
                "matched": subject_provenance == "reid_matched",
                "embeddings": (poses.get("enrolment") or {}).get(str(subject_track), []),
            },
            # First and last box per track. The local half needs these to reject a fragment the
            # person could not have walked to - see `reachable` in service/staging.py. Without them
            # a television in shot merges into the resident, because a screen never shares frames
            # with the room and so looks exactly like a person who stepped out of detection.
            "track_boxes": {str(k): [v[0], v[1]] for k, v in _boxes.items()},
            "fps": poses["fps"],
            "day": frames[0].timestamp.date().isoformat(),
            "observed_hours": round(obs_hours, 6),
            "timing": {"pose_s": round(t_pose - t_start, 1),
                       "classify_s": round(t_cls - t_pose, 1),
                       "behaviour_s": 0.0, "report_s": 0.0},
            "rss_mb": _rss, "tau": TAU, "temperature": TEMPERATURE,
            "report": None, "report_provenance": None,
        }
        return
    if tracks:
        # SEPARATE try blocks per agent, deliberately. One block around both meant a Qwen
        # exception left `stage4` as None, which the stage record then reported as
        # "no reporter loaded (Qwen weights absent) or no track found" - blaming missing
        # weights for a crash in weights that had loaded. A failure must be attributed to the
        # agent that produced it, or the log sends the next person to the wrong place.
        try:
            from behaviorsense.agents.behaviour import BehaviourAnalyzer, aggregate_daily_features
            from behaviorsense.data.simulator import standard_scenarios
            from datetime import timedelta

            analyzer = BehaviourAnalyzer()
            ref = next(iter(standard_scenarios())).run()
            primed = 0
            for day in ref.days[:21]:
                analyzer.analyze_day(day)
                primed += 1
            # AGGREGATE THE SEGMENTS WE ALREADY HAVE. `pipe.run_to_features(frames)` calls
            # `self.run(frames)` internally, so it classified every window a SECOND time -
            # doubling the ST-GCN++ pass for nothing, and worse, discarding the subject relabel
            # above because its segments were a fresh set. Two independent runs also means
            # Agent 2's card and Agent 3's figures could disagree, which is the one thing a
            # stage-by-stage view must never do.
            today = aggregate_daily_features(
                segments, day=frames[0].timestamp.date(), observed_hours=obs_hours)
            # DATE THE CLIP AFTER THE REFERENCE WINDOW, or the whole layer comes back empty.
            #
            # Baselines are computed with `before=day_features.day`, and deviations exist only
            # for features that HAVE a baseline. An uploaded clip's timestamps start at the
            # decoder's own origin, which is 2026-01-01 - the same date the simulator's first
            # reference day carries. `before=2026-01-01` matches nothing, so baselines were
            # empty, so deviations were empty, so Agent 4 wrote "0 features reviewed" and
            # emitted no claims, so the C1-C5 table rendered blank on every upload. Every
            # stage still reported `done`, which is why this looked like it worked.
            today = today.model_copy(update={
                "day": ref.days[primed - 1].day + timedelta(days=1)})
            st = analyzer.analyze_day(today)
            stage3 = {
                "baseline_provenance": "reference_cohort_simulated",
                "subject_provenance": subject_provenance,
                "subject_track": subject_track,
                "subject_tracks": list(subject_tracks),
                "baseline_days": primed,
                "real_days_from_this_video": 1,
                "observed_hours": round(obs_hours, 3),
                # CALL IT. `is_reliable` is a METHOD, so `bool(getattr(...))` evaluates a bound
                # method - always truthy - and a 29-second clip (observed_hours 0.01) was reported
                # as a reliable day. `stages.js` suppresses its own "far below the 8 h a day needs
                # to be called reliable" warning on this field, so the bug deleted the one sentence
                # telling a caregiver not to act on the numbers.
                "is_reliable": bool(today.is_reliable()),
                # THE SAME GATE THE PROMPT USES, for the card. A ten-minute total against a
                # 22.9 h baseline median is a unit error (the clip's walking RATE was normal
                # while the raw z read -10.00), so the deviation table is withheld on a partial
                # window exactly as the numbers are withheld from Agent 4, and the page says why.
                "deviations_withheld": not bool(today.is_reliable()),
                "features_from_video": {k: round(float(v), 2)
                                        for k, v in today.numeric_items().items()},
                "deviations": ([] if not today.is_reliable() else
                               [{"feature": k, "robust_z": round(float(v), 2)}
                                for k, v in sorted(st.deviations.items(),
                                                   key=lambda kv: -abs(kv[1]))[:10]]),
                "alerts": [{"kind": a.kind.value, "severity": a.severity.value,
                            "rule": a.rule_name} for a in st.alerts],
                "caveats": [
                    "The BASELINE is a simulated reference persona, not this person's "
                    "history - one clip cannot supply 14 days.",
                    "Feature VALUES are measured from the uploaded video and are real.",
                    "Robust-z and alerts are therefore reference-relative: read them as "
                    "'unlike the reference', never as 'this person has declined'.",
                ] + ([
                    "IDENTITY IS ASSERTED, not re-identified: no resident is enrolled, so "
                    f"the most-present track ({subject_track}) was treated as the subject. "
                    "OSNet reports every track as unidentified, which is correct - it has no "
                    "gallery to match against. Without this the subject filter matches nothing "
                    "and all 27 features read 0.",
                ] if subject_provenance == "asserted_most_present_track" else []) + ([
                    f"TRACKS {', '.join(str(t) for t in subject_tracks)} WERE MERGED into one "
                    "subject because their frame spans never overlap, so they cannot be two "
                    "people present at the same time. A person who leaves detection for more "
                    "than the tracker's ~15 s bridge returns with a new track id, and "
                    "counting only one fragment "
                    "discarded the rest of their activity - measured once as cooking_duration_s "
                    "reading 0.00 beside a cooking segment of 8.6 s. Two tracks seen in the SAME "
                    "frame are never merged.",
                ] if subject_provenance == "asserted_disjoint_track_chain" else []),
            }
            t3 = time.time()
        except Exception as exc:                                   # noqa: BLE001
            # A demo that dies at stage 3 is worse than one that reports stage 3 failed.
            stage3 = {"error": f"{type(exc).__name__}: {exc}"}
            t3 = time.time()

    # AGENT 3 GOES OUT BEFORE QWEN IS CALLED. Generation is the slow stage - >125 s on a T4 -
    # so emitting Agent 3 first is the difference between a page that fills in progressively
    # and one that shows nothing for two minutes.
    yield "stage", _stage(3, "behaviour",
                          "done" if stage3 and "error" not in stage3
                          else ("failed" if stage3 else "skipped"),
                          t3 - t_cls, stage3 or {})

    t4 = t3
    if st is not None and stage3 and "error" not in stage3:
        if STATE.get("reporter") is None:
            stage4 = {"why_skipped": "no reporter is configured on this backend, so Agents 1-3 "
                                     "ran and the report was not written. Either the Qwen weights "
                                     "are unattached, or torch has no kernels for this GPU. Run "
                                     "Agents 3-4 yourself with /video?stages=12 - "
                                     "web/local_backend.py does that with local API keys."}
        else:
            try:
                out = None
                # HEARTBEAT THROUGH GENERATION - the longest silence in the whole request.
                # Agent 3's line used to go out and then nothing until Agent 4 finished, so a
                # normal Qwen pass tripped the client's 90 s silence watchdog and reported a
                # wedged GPU. Observed on a 5-second clip.
                for _kind, _obj in _blocking(lambda: STATE["reporter"].report(st),
                                             "qwen_generating"):
                    if _kind == "beat":
                        yield "heartbeat", _obj
                    else:
                        out = _obj
                rate = out.hallucination_rate
                stage4 = {
                    "model": out.report.model_name,
                    "constrained_decoding": out.report.constrained_decoding,
                    "summary": out.report.summary,
                    "recommendation": out.report.recommendation,
                    "escalate": out.report.escalate,
                    "claims_emitted": out.n_emitted_claims,
                    "claims_scorable": len(out.report.claims),
                    "hallucination_rate": None if rate != rate else round(rate, 3),
                    "parse_failed": out.parse_failed,
                    # WHY it did not parse. `_extract_json` distinguishes "no JSON object in
                    # response" (the model answered in prose - a compliance failure) from
                    # "unterminated JSON object (truncated generation)" (max_tokens too low for a
                    # model that reasons before answering - a budget failure). One boolean on the
                    # card cannot tell those apart, and they need different fixes.
                    "notes": list(out.notes),
                    "caveats": ["Claims cite features measured from this video; the baseline "
                                "they are compared against is the declared reference above."],
                }
                by_id = {c.claim_id: c for c in out.report.claims}
                for v in out.report.verifications:
                    c = by_id.get(v.claim_id)
                    checks.append({
                        "claim_id": v.claim_id,
                        "text": None if c is None else c.text,
                        "evidence_ref": None if c is None else c.evidence_ref,
                        "claimed_value": None if c is None else c.claimed_value,
                        "claimed_pct_change": None if c is None else c.claimed_pct_change,
                        "direction": None if c is None else c.direction,
                        # C1-C5 separately: "verified" as one boolean hides which guarantee
                        # is doing the work. C4 is the one that matters clinically; C5 is
                        # the one the latest measured run says catches the most - prose
                        # that never quotes the figure the field records.
                        "C1_ref_exists": v.ref_exists,
                        "C2_value_matches": v.value_matches,
                        "C3_pct_matches": v.pct_matches,
                        "C4_direction_consistent": v.direction_consistent,
                        "C5_prose_quoted_value": v.prose_quoted_value,
                        "faithful": v.is_faithful,
                        "notes": list(v.notes),
                        "shown_to_caregiver": v.is_faithful,
                    })
            except Exception as exc:                               # noqa: BLE001
                stage4 = {"error": f"{type(exc).__name__}: {exc}"}
    elif not tracks:
        stage4 = {"why_skipped": "no person was tracked long enough to fill a window, so "
                                 "there is nothing for the report to describe."}
    else:
        stage4 = {"why_skipped": "Agent 3 did not complete, so there is no verified state to "
                                 "write a report from."}
    t4 = time.time()
    _rss["after_report"] = _rss_mb()

    # Printed as well as returned, so it survives in the notebook log even if the client
    # disconnects mid-stream - which is exactly when a leak matters most.
    print("  RSS MB " + " -> ".join(f"{k} {v}" for k, v in _rss.items()))

    stage4_status = ("done" if stage4 and "error" not in stage4 and "why_skipped" not in stage4
                     else ("failed" if stage4 and "error" in stage4 else "skipped"))
    yield "stage", _stage(4, "report", stage4_status, t4 - t3, stage4 or {})

    stages = [stage1, stage2,
              _stage(3, "behaviour",
                     "done" if stage3 and "error" not in stage3
                     else ("failed" if stage3 else "skipped"), t3 - t_cls, stage3 or {}),
              _stage(4, "report", stage4_status, t4 - t3, stage4 or {})]

    yield "result", {
        "fps": poses["fps"], "width": poses["width"], "height": poses["height"],
        "frames_kept": poses["frames_kept"], "truncated": poses["truncated"],
        "reid": poses["reid"], "providers": poses["providers"],
        "frames": poses["frames"],          # per-frame skeletons, for the overlay
        "tracks": tracks,
        # People TRACKED, matching Agent 1's own card. `len(tracks)` would be people
        # CLASSIFIED, which is smaller whenever someone is present but too occluded to
        # label - and the overlay draws that person, so the count must not omit them.
        "n_people": len(tracked),
        "stages": stages,                   # Agent 1 -> 2 -> 3 -> 4, in order, with timings
        "checks": checks,                   # C1-C5 per claim, individually
        "timing": {"pose_s": round(t_pose - t_start, 1),
                   "classify_s": round(t_cls - t_pose, 1),
                   "behaviour_s": round(t3 - t_cls, 1),
                   "report_s": round(t4 - t3, 1)},
        # RSS AT EACH SEAM. The delta between consecutive entries is what a stage RETAINED, which
        # is the difference between a leak and a transient peak.
        "rss_mb": _rss,
        "tau": TAU, "temperature": TEMPERATURE,
        "report": (stage4 or {}).get("summary"),
        "report_provenance": (
            None if not (stage4 or {}).get("summary") else
            "Claims are verified against features measured from THIS video. The baseline "
            "they are compared to is a declared simulated reference, because a 14-day "
            "rolling median cannot come from one clip - see stages[2].caveats."),
    }"""),
("code", """# Serve, then tunnel. KEEP THIS CELL RUNNING - the address dies with the process.
import nest_asyncio, re, subprocess, threading, time, uvicorn
nest_asyncio.apply()

threading.Thread(
    target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning"),
    daemon=True).start()
time.sleep(3)

tun = subprocess.Popen([str(CF), "tunnel", "--url", "http://localhost:8000",
                        "--no-autoupdate"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       bufsize=1)
URL = None
for line in tun.stdout:
    m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", line)
    if m:
        URL = m.group(0)
        break
assert URL, "cloudflared exited without printing a URL - check the output above"

print()
print("=" * 68)
print("  PASTE THIS INTO THE FRONT END HEADER:")
print(f"    {URL}")
print("=" * 68)
print()
print("  /health   readiness + which streams loaded")
print("  /activity [N,30,2,17,3] windows -> smoothed labels")
print("  /report   a BehaviourState -> claims + C1-C4 verdicts")
print("  /video    an uploaded clip -> per-frame skeletons, tracks, roles, segments")
print()
print("Leave this cell running. Weights finish loading in the background - /health")
print("reports ready=false until then, and the front end says so rather than failing.")

# Block, so Kaggle does not treat the session as idle and reap it mid-demo.
tun.wait()"""),
]


N06: list[tuple[str, str]] = [
("markdown", """# 06 — Toyota Smarthome preflight (ONLINE, CPU, internet ON)

| attach as input | produces |
|---|---|
| `behaviorsense-code`, `behaviorsense-weights` | `behaviorsense-toyota-meta` (a few MB) |
| all nine Toyota / MSMT mounts | a go/no-go verdict, printed |

**Run this before the extraction session, not instead of it.** Extraction is a ~6 h offline
job whose output is only as good as its labels, and there are four things about this corpus
that can be wrong in ways that do not crash:

1. **The archive ships per-video CSVs, not the aggregated JSON.** `Annotation_v1.0.tar.gz`
   gives `Annotation/P18/P18T13C07.csv`. Every published baseline trains against
   `smarthome_CS_51.json`, which is a DERIVED artefact mirrored on GitHub. Both are read
   here and compared span-for-span.
2. **The CSVs carry no duration.** `frame_multilabel` maps feature step to annotation
   position as `step * duration / n_steps`, so a wrong duration rescales the whole label
   tensor and leaves its shape correct. Durations come from the JSON and are checked.
3. **Frames, not seconds.** Untrimmed is 25 fps at x1.25 speed, trimmed is 30 fps. One
   shared constant silently stretches one half by 20%.
4. **Two official id spaces for the trimmed classes**, differing by one, with 0 meaning
   "unmatched name" in the RGB one.

This notebook is ONLINE for one reason: the annotation JSON has to come off GitHub. It then
writes it to `/kaggle/working` so the OFFLINE extraction notebook can read it from a private
dataset instead of the network.

**The JSON is not committed to the repo.** `github.com/PARTHG0106/BehaviorSense` is public
and this dataset is licensed for academic research only, granted per-request. Staging it into
a private Kaggle dataset is the same pattern the other restricted assets already use."""),
("code", RESOLVE_CODE),
("code", """# Resolve the nine mounts BY CONTENT, at BOUNDED DEPTH.
#
# Never by dataset name - that rule has paid for itself twice. But the first run of this
# notebook proved the rule is not free: nine `**` globs over ~130,000 mounted files took
# **78 minutes**, because `**` descends into `mp4/` (16,115 entries) and
# `bounding_box_train/` (65,242) once per pattern.
#
# Datasets mount at `datasets/<owner>/<slug>/...`, so each asset is described by the
# SUBDIRECTORY it lives in and the pattern inside it. Testing `is_dir()` on a candidate is
# O(1), and counting is then one directory read instead of a tree walk. The observed layouts
# are all covered, including `toyota-smarthome-skeleton-v1-2` whose files sit at the slug
# root with no subfolder at all.
import collections, json, os, re, sys, time

ROOTS = sorted((INPUT / "datasets").glob("*/*")) if (INPUT / "datasets").is_dir() else []
print(f"{len(ROOTS)} dataset mount(s):")
for r in ROOTS:
    print(f"  {r.name}")

def count_dir(d, pattern):
    # scandir, not glob: for a 16k-entry directory this is one syscall loop and no Path
    # object per miss. The count is what tells a truncated upload from a complete one.
    n = 0
    try:
        with os.scandir(d) as it:
            for e in it:
                if e.is_file() and _fnmatch(e.name, pattern):
                    n += 1
    except OSError:
        return 0
    return n

from fnmatch import fnmatch as _fnmatch

# (subdir inside the slug, filename glob). "" means the slug root.
ASSETS = {
    "annotation_csv":   ("Annotation",  "P*T*C*.csv",  True),
    "rgb_untrimmed":    ("Videos_mp4",  "P*T*C*.mp4",  True),
    "pose_untrimmed":   ("Skeleton",    "results_P*T*C*_lcrnet*.json", False),
    "depth_untrimmed":  ("Depth",       "P*T*C*.mp4",  False),
    "rgb_trimmed":      ("mp4",         "*_p[0-9][0-9]_r*_c[0-9][0-9].mp4", True),
    "skel_trimmed_v11": ("json",        "*_p[0-9][0-9]_r*_c[0-9][0-9].json", False),
    "skel_trimmed_v12": ("",            "*_p[0-9][0-9]_r*_c[0-9][0-9]_pose3d.json", False),
    "depth_trimmed":    ("depth",       "*_p[0-9][0-9]_r*_c[0-9][0-9].mp4", False),
}

RESOLVED, COUNTS = {}, {}
t0 = time.time()
for key, (subdir, pattern, required) in ASSETS.items():
    best, best_n = None, 0
    for root in ROOTS:
        # The annotation CSVs are one level deeper (Annotation/P02/*.csv), so try both the
        # subdir itself and its immediate children before giving up on this root.
        base = root / subdir if subdir else root
        if not base.is_dir():
            continue
        n = count_dir(base, pattern)
        if n == 0:
            n = sum(count_dir(c, pattern) for c in sorted(base.iterdir()) if c.is_dir())
        if n > best_n:
            best, best_n = base, n
    RESOLVED[key], COUNTS[key] = best, best_n
    if best is None or best_n == 0:
        if required:
            raise AssertionError(
                f"{key}: no mount holds {subdir or '<root>'}/{pattern}. Mounts: "
                f"{[r.name for r in ROOTS]}")
        print(f"  {key:<20} ABSENT (optional)")
    else:
        print(f"  {key:<20} {best_n:>7,} files   {best}")
print(f"resolved in {time.time() - t0:.1f}s (the `**` version took 78 minutes)")

# Expected magnitudes, from the dataset's own README. A truncated upload trains fine and
# scores slightly worse, which is the most expensive way to be wrong.
EXPECT = {"annotation_csv": 536, "rgb_untrimmed": 536, "rgb_trimmed": 16115}
print()
for key, want in EXPECT.items():
    got = COUNTS.get(key, 0)
    print(f"  {key:<20} {got:>7,} / {want:,}  "
          f"{'ok' if got == want else 'SHORT' if got < want else 'MORE THAN EXPECTED'}")"""),
("code", """# The annotation JSON, off GitHub. This is the ONE thing this notebook needs internet for.
#
# Two files: CS is the protocol every baseline reports (PDAN 32.7% f-mAP), CV is the
# cross-view one. They are read for `duration` and as an independent copy of the same spans.
import urllib.request, pathlib

BASE = ("https://raw.githubusercontent.com/dairui01/Toyota_Smarthome/main/pipline/data/")
OUT = pathlib.Path("/kaggle/working/toyota_meta")
OUT.mkdir(parents=True, exist_ok=True)

JSONS = {}
for name in ("smarthome_CS_51.json", "smarthome_CV_51.json", "Action_list"):
    dst = OUT / name
    if not dst.exists():
        urllib.request.urlretrieve(BASE + name, dst)
    print(f"  {name:<24} {dst.stat().st_size:>9,} bytes")
    if name.endswith(".json"):
        JSONS[name] = json.loads(dst.read_text(encoding="utf-8"))

# The class list is the id space. Verify the download against the tuple we train on rather
# than trusting either - a reordered Action_list would relabel the entire corpus.
from behaviorsense.data.toyota import TSU_CLASSES
_lines = [l.split("\\t") for l in (OUT / "Action_list").read_text().splitlines() if l.strip()]
_names = tuple(p[1] for p in _lines)
assert _names == TSU_CLASSES, (
    "the downloaded Action_list disagrees with TSU_CLASSES. First difference at "
    f"{next(i for i,(a,b) in enumerate(zip(_names, TSU_CLASSES)) if a != b)}")
print(f"  Action_list agrees with TSU_CLASSES on all {len(TSU_CLASSES)} names")"""),
("code", """# SURVEY the label vocabulary before parsing anything for real.
#
# This cell exists because of how the first run failed. `read_annotation_csv` raised on the
# first unrecognised class name - 93 minutes in, having examined ONE file of 536 - and the
# name was `Make_coffee.Get_water`. The dataset README had the answer all along:
#
#   "please merge `Make_coffee.Get_water` & `Get Water` and `Insert_tea_bag` &
#    `Make_tea.Insert_tea_bag` to have 51 action classes"
#
# The CSVs ship the PRE-MERGE 53-name vocabulary; `Action_list` is the post-merge 51 that
# every published number is computed over. `TSU_ALIASES` now applies the merge, and the
# survey below reports EVERY unrecognised name in one pass, so a second surprise costs one
# run rather than one run per name.
from behaviorsense.data.toyota import survey_annotation_csvs, TSU_ALIASES, TSU_CLASSES

ANN_ROOT = RESOLVED["annotation_csv"]
if ANN_ROOT.name.startswith("P") and ANN_ROOT.parent.name == "Annotation":
    ANN_ROOT = ANN_ROOT.parent          # resolver may land on a per-subject subfolder

SURVEY = survey_annotation_csvs(ANN_ROOT)
print(f"{SURVEY['n_files']} CSVs, {SURVEY['n_rows']:,} annotation rows, "
      f"{SURVEY['n_distinct_names']} distinct class names")
print(f"\\nmerges applied ({len(TSU_ALIASES)} aliases known):")
for raw, n in sorted(SURVEY["aliased"].items(), key=lambda kv: -kv[1]):
    print(f"  {raw:<28} -> {TSU_ALIASES[raw]:<20} {n:>6,} rows")

if SURVEY["unknown"]:
    print(f"\\nSTOP: {len(SURVEY['unknown'])} class name(s) resolve to no class:")
    for name, where in SURVEY["unknown"].items():
        print(f"  {name!r}  first seen in {where}")
    print("\\nAdd them to TSU_ALIASES (if they are merges) or to TSU_CLASSES (if the")
    print("Action_list is incomplete), re-upload behaviorsense-code, and re-run.")
    raise SystemExit("unknown class names - see above")
print(f"\\nGO: every name resolves into the {len(TSU_CLASSES)}-class space.")"""),
("code", """# Parse the CSVs, then check them against the JSON span-for-span. This is the gate.
from behaviorsense.data.toyota import (TSU_FPS, cross_check_annotations, frame_counts,
                                       load_tsu_annotations, load_tsu_annotations_from_csv,
                                       read_annotation_csv, collapse_matrix, COARSE_V11,
                                       TSU_TO_COARSE, iter_videos, PROTOCOLS)

JSON_SIDE = load_tsu_annotations(OUT / "smarthome_CS_51.json")
print(f"JSON: {len(JSON_SIDE)} videos, official CS partition verified on read")

# Show one raw CSV before parsing anything, so the confirmed schema is visible and not
# taken on faith. The first run printed exactly this and it is how the merge was found.
_one = sorted(ANN_ROOT.glob("**/P*T*C*.csv"))[0]
print(f"\\nraw {_one.name}:")
for _l in _one.read_text(encoding="utf-8-sig").splitlines()[:5]:
    print(f"    {_l}")
print(f"parsed -> {read_annotation_csv(_one)[:3]}")

DURATIONS = {v: JSON_SIDE[v].duration for v in JSON_SIDE}
CSV_SIDE = load_tsu_annotations_from_csv(ANN_ROOT, DURATIONS)
print(f"\\nCSV:  {len(CSV_SIDE)} videos parsed")

REPORT = cross_check_annotations(CSV_SIDE, JSON_SIDE)
print(f"\\ncross-check: {REPORT['compared']} videos compared, "
      f"{REPORT['agree']} span-identical, {len(REPORT['disagree'])} span-different")
print(f"FRAME-level agreement: mean {REPORT['frame_agreement_mean']:.5f}  "
      f"worst {REPORT['frame_agreement_min']:.5f}  "
      f"videos below 0.999: {REPORT['videos_below_999']}")
print()
print("Span equality is the wrong yardstick on its own and the first run showed why: the")
print("CSVs carry FINER spans and the JSON merges adjacent same-class intervals, so")
print("(2,1340,2960)+(2,2960,2963) against (2,1480,2963) counts as a disagreement while")
print("describing almost the same frames. Frames are what the model sees, so that is what")
print("is scored. A pure split scores 1.00000; only a moved boundary costs anything.")
print()
for d in REPORT["disagree"][:5]:
    print(f"  {d['vid']}: csv {d['csv_spans']} spans vs json {d['json_spans']}, "
          f"frame agreement {d['frame_agreement']:.5f} "
          f"({d['cells_differing']:,} cells differ)")
if REPORT["csv_only_videos"]:
    print(f"  only in CSVs ({len(REPORT['csv_only_videos'])}): "
          f"{REPORT['csv_only_videos'][:5]}")
if REPORT["json_only_videos"]:
    print(f"  only in JSON ({len(REPORT['json_only_videos'])}): "
          f"{REPORT['json_only_videos'][:5]}")
print()
if REPORT["frame_agreement_mean"] >= 0.999 and REPORT["compared"] == len(JSON_SIDE):
    print("GO. The two sources describe the same frames to better than 1 part in 1,000.")
    print("Train on the JSON, because that is what every published baseline reports on")
    print("(PDAN 32.7% f-mAP, CS) - the CSVs are authoritative but not comparable.")
elif REPORT["frame_agreement_mean"] >= 0.99:
    print("GO WITH A CAVEAT. Frame agreement is 0.99-0.999, so a minority of videos have")
    print("genuinely moved boundaries rather than merged spans. Train on the JSON for")
    print("comparability and record the agreement figure next to the numbers.")
else:
    print("STOP. Frame agreement below 0.99 means the sources disagree about content, not")
    print("just about how spans are grouped. Do not spend an extraction session on either")
    print("until the videos listed above have been looked at.")"""),
("code", """# The distribution, printed so the extraction pass can be checked against it later.
import numpy as np

FINE = frame_counts(JSON_SIDE.values())
M = collapse_matrix(TSU_CLASSES, TSU_TO_COARSE)
COARSE = FINE @ M
TOT = float(FINE.sum())
print(f"{TOT:,.0f} annotated frames = {TOT / TSU_FPS / 3600:.1f} h at {TSU_FPS:g} Hz")
for p in PROTOCOLS:
    sides = {s: len(list(iter_videos(JSON_SIDE, p, s)))
             for s in ("train", "val", "test", "unused")}
    print(f"  {p:<4} " + "  ".join(f"{k} {v}" for k, v in sides.items() if v))

print(f"\\n{'coarse class':<24}{'frames':>12}{'share%':>8}")
for i in np.argsort(-COARSE):
    if COARSE[i]:
        print(f"{COARSE_V11[i]:<24}{int(COARSE[i]):>12,}{100 * COARSE[i] / TOT:>7.2f}")
_idle = int(COARSE[COARSE_V11.index("other_idle")])
print(f"\\nother_idle: {_idle} frames. Zero is the intended answer - every Toyota class is a "
      "real activity, so nothing needs the reject class.")

_gaps = np.array([1 - v.annotated_frames() / max(1, v.duration) for v in JSON_SIDE.values()])
print(f"unannotated: mean {_gaps.mean():.3f} median {np.median(_gaps):.3f} max {_gaps.max():.3f}")
print("A third of the corpus is gap. The 51-class benchmark head treats gaps as true")
print("negatives; the unified head must MASK them, because a gap most likely contains the")
print("sitting and standing that Toyota never labels.")"""),
("code", """# Confirm the RGB actually decodes, and that the filename parsers survive the real names.
# One video per protocol side, because a corrupt archive member is cheaper to find now than
# 4 hours into extraction.
import cv2
from behaviorsense.data.toyota import (TSM_FPS, TSM_FPS_DOCUMENTED, parse_tsm_name,
                                       parse_tsu_filename,
                                       protocol_side)

def probe(path):
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    got, _frame = cap.read()
    cap.release()
    return ok and got, n, fps, w, h

print("untrimmed RGB (expect 25 fps, 640x480):")
for side in ("train", "test"):
    # islice, not a full sorted glob: 536 Path objects is cheap but 16,115 is not, and the
    # trimmed loop below would pay it for two probes.
    import itertools
    for p in itertools.islice(RESOLVED["rgb_untrimmed"].glob("P*T*C*.mp4"), 200):
        v = parse_tsu_filename(p)
        if protocol_side(v, "CS") != side:
            continue
        ok, n, fps, w, h = probe(p)
        d = JSON_SIDE.get(v.tsu)
        drift = "" if d is None else f"  json duration {d.duration} (delta {n - d.duration:+d})"
        print(f"  [{side}] {p.name}: decode={ok} frames={n:,} {fps:g}fps {w}x{h}{drift}")
        break

print(f"\\ntrimmed RGB (expect {TSM_FPS:g} fps from the container, {TSM_FPS_DOCUMENTED:g} in the README):")
for p in itertools.islice(RESOLVED["rgb_trimmed"].glob("*.mp4"), 2):
    v = parse_tsm_name(p.name)
    ok, n, fps, w, h = probe(p)
    verdict = ("matches the container constant" if abs(fps - TSM_FPS) < 0.6 else
               "matches the README" if abs(fps - TSM_FPS_DOCUMENTED) < 0.6 else
               f"matches NEITHER - {fps:g} is a third value")
    print(f"  {p.name}: {v.activity} p{v.subject:02d} c{v.camera:02d} "
          f"decode={ok} frames={n} {fps:g}fps {w}x{h}  -> {verdict}")
    print(f"      {n} frames = {n / max(fps, 1):.1f}s at the container rate, "
          f"{n / TSM_FPS_DOCUMENTED:.1f}s at the README's {TSM_FPS_DOCUMENTED:g}")
print("  The README and the files disagree, so the extractor reads fps PER FILE and this")
print("  cell is what says which value to expect. Assuming either one silently rescales")
print("  every trimmed duration by 1.5x.")

if RESOLVED["skel_trimmed_v12"] is not None:
    _s = next(iter(RESOLVED["skel_trimmed_v12"].glob("*_pose3d.json")))
    _v = parse_tsm_name(_s.name)
    print(f"\\nV1.2 skeleton: {_s.name} -> {_v.activity} p{_v.subject:02d} tag={_v.tag!r}")
    print("  (the tag group is why this archive parses at all - an anchored pattern without")
    print("   it rejects every file here and reports it as an empty mount)")"""),
("code", """# Stage the metadata for the OFFLINE extraction notebook, plus the resolved manifest.
#
# The manifest records which mount answered which pattern and how many files it held. When
# extraction later disagrees with these numbers, the question 'did the mounts change?' has a
# recorded answer instead of a reconstruction.
MANIFEST = {
    "resolved": {k: (str(v) if v else None) for k, v in RESOLVED.items()},
    "counts": COUNTS,
    "expected": EXPECT,
    "cross_check": {k: v for k, v in REPORT.items() if k != "disagree"},
    "n_disagree": len(REPORT["disagree"]),
    "fine_frame_counts": FINE.tolist(),
    "tsu_fps": TSU_FPS, "tsm_fps": TSM_FPS,
    "durations": DURATIONS,
}
(OUT / "manifest.json").write_text(json.dumps(MANIFEST, indent=1), encoding="utf-8")
print(f"wrote {OUT}/ :")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name:<26} {f.stat().st_size:>10,} bytes")

print()
print("=" * 70)
print("  Save Version -> create a PRIVATE dataset named behaviorsense-toyota-meta")
print("=" * 70)
print("The offline extraction notebook attaches that instead of reaching for GitHub.")
print()
print("Do not commit these files to the repo: it is PUBLIC and this dataset is licensed")
print("for academic research only. .gitignore already refuses them.")"""),
]



N07: list[tuple[str, str]] = [
("markdown", """# 07 — Toyota Smarthome extraction (OFFLINE, Blackwell)

| attach as input | produces |
|---|---|
| `behaviorsense-code`, `behavioursense-WW`, `behaviorsense-toyota-meta` | `behaviorsense-toyota-shards` |
| `toyota-smarthome-rgb` (MODE="trimmed") **or** `rgb-untrimmed` (MODE="untrimmed") | |
| **resume:** `behaviorsense-toyota-shards` (its own previous version) | |

**Internet OFF, Blackwell.** Runs RTMO over Toyota Smarthome and writes skeleton windows in
the same shard layout as `behaviorsense-adl-shards`, so the trainer reads both without a new
loader.

## Why RTMO and not the skeletons that ship with the dataset

Toyota provides 3D skeletons (V1.1 from LCR-Net, V1.2 refined by SSTA-PRS). They are not used
as the model's input, and that is deliberate: they are **15-joint 3D**, and the deployment
estimator is **RTMO producing 17-joint COCO 2D**. Training on one and serving the other is a
guaranteed train/serve skew, and this project has already paid twice for that class of bug
(`normalise()` erasing falls; the fall head's val AUROC 1.000 against held-out 0.47). The
V1.2 skeletons are worth having as an occlusion-robust QA reference, which is a different job.

## The two modes, and why trimmed comes first

| | clips | frames to pose | est. GPU | what it buys |
|---|---|---|---|---|
| `trimmed` | 16,115 | ~0.8 M at stride 2 | **~2 h** | clip-level labels for the window classifier |
| `untrimmed` | 536 videos, 149 h | ~6.7 M | **~9 h** | dense per-frame labels; f-mAP against PDAN's 32.7% |

Trimmed is what one session buys and it is what Agent 2 actually is: a window classifier.
Untrimmed is four times the cost and its value is mostly *evaluation*, so
`UNTRIMMED_SIDE = "test"` extracts only the 185 CS test videos (~3.5 h) and leaves the 351
training videos for a later session. Both modes write to the same dataset and resume from it.

**Set `MODE` in the first code cell.** Everything else is resolved by content."""),
("code", """# Configuration. The only cell to edit.
MODE = "trimmed"           # "trimmed" (~7 h) or "untrimmed" (~3.5 h for the CS test side)
UNTRIMMED_SIDE = "test"    # CS side when MODE="untrimmed": test | train | both
FPS_SAMPLE = 20.0          # see below - measured, not chosen
MAX_CLIPS = 99999          # cap for a short session; shards already flushed are kept
WINDOWS_PER_SHARD = 8000

# FPS_SAMPLE = 20, and the first full run is why. It ran at 12.5, and because every trimmed
# container reports exactly 20 fps (`source frame rates seen: {20: 16115}`), the integer
# stride came out `round(20/12.5) = 2` - an effective 10 Hz. At 10 Hz a 30-frame window
# spans 3.0 s, so any clip shorter than 3 s produced NO window at all:
#
#   6,040 of 16,115 clips unusable (37.5%), and the loss was not uniform. `Sitdown` kept
#   233 windows and `Getup` 149, against annotated corpus shares of 0.72% and 0.67% -
#   three to four times under-represented, because a sit-to-stand IS a short clip. Those
#   two classes are what Agent 3 counts `sit_to_stand_count` from, and that is a validated
#   clinical frailty marker, so starving them undermines a headline feature rather than a
#   tail statistic.
#
# At 20 Hz the stride is 1, the window spans 1.50 s, and a clip needs only 1.5 s to yield
# one. Estimated ~322k windows against 97,787 - and 1.50 s is CLOSER to the Charades
# shards' 2.0 s than 3.00 s was, so window-extent consistency improves at the same time.
#
# It costs about twice the GPU: the first run took 3.55 h, so budget ~7 h of a 12 h session.
# `MAX_CLIPS` plus the resume path exist for the case where that runs out.
#
# The rate is part of the shard name (see PREFIX), so windows sampled at different rates can
# never end up in one training set by accident.
import subprocess, sys, pathlib
INPUT = pathlib.Path("/kaggle/input")

_MARKERS = {"torch", "rtmlib", "onnxruntime-gpu", "nvidia-cudnn-cu12", "triton"}
_dirs = {}
for _shape in ("datasets/*/*/wheels/*.whl", "datasets/*/*/*/wheels/*.whl",
               "datasets/*/*/*.whl"):
    for _w in INPUT.glob(_shape):
        _dirs.setdefault(_w.parent, set()).add(
            _w.name.split("-")[0].lower().replace("_", "-"))
    if _dirs:
        break
assert _dirs, "no wheels attached - behavioursense-WW carries them"
WHEEL_DIR, _hits = max(_dirs.items(), key=lambda kv: len(kv[1] & _MARKERS))
assert _hits & _MARKERS, f"wheel dir holds none of {sorted(_MARKERS)}: {WHEEL_DIR}"

# rtmlib pulls the CPU onnxruntime; both own the same module directory, so the CPU build
# silently clobbers the GPU one's provider registration. Remove it, install rtmlib without
# deps, put onnxruntime-gpu last. Same order as notebook 01, for the same measured reason.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                "--find-links", str(WHEEL_DIR), "rtmlib"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                "--find-links", str(WHEEL_DIR),
                "onnxruntime-gpu", "opencv-python-headless", "PyYAML"], check=True)
print(f"MODE={MODE}  FPS_SAMPLE={FPS_SAMPLE}  installed from {WHEEL_DIR}")"""),
("code", RESOLVE),
("code", """# Prove the GPU is in use BEFORE spending the session. onnxruntime falls back to CPU
# silently - correct poses at 1/50th the speed - and that presents as "the GPU is slow"
# rather than "the GPU is not being used". Notebook 01 lost a session to exactly this.
import onnxruntime as ort
providers = ort.get_available_providers()
print("onnxruntime providers:", providers)
assert any("CUDA" in p for p in providers), (
    f"no CUDAExecutionProvider: {providers}. The staged onnxruntime-gpu did not load - "
    "check the CPU build was removed before it was installed.")

import collections, cv2, json, os, shutil, time
import numpy as np

# `find_dir` and `find_asset` come from the resolver cell above (MOUNT_PRELUDE), which is
# the same code notebooks 01-06 use. This cell used to define a second, local `find_dir`
# over its own `ROOTS` while calling `find_asset` from a cell that did not export it -
# NameError after 370 s of session. One definition, one place.
ROOTS = _SLUGS

# The meta files sit at <slug>/toyota_meta/, NOT the slug root: notebook 06 wrote them to
# /kaggle/working/toyota_meta/ and Save Version nests the working directory inside the
# dataset. The first run of this notebook failed here for exactly that reason - the mount
# was attached, the assert just looked one level too shallow. Try the known layout first,
# then the root, and name what was actually seen before giving up.
META, _ = find_dir("toyota_meta", "smarthome_CS_51.json")
if META is None:
    META, _ = find_dir("", "smarthome_CS_51.json")
assert META is not None, (
    "attach behaviorsense-toyota-meta (notebook 06's output). This notebook is OFFLINE "
    "and cannot fetch the annotations. Looked for <mount>/toyota_meta/smarthome_CS_51.json "
    f"and <mount>/smarthome_CS_51.json. Mounts: {[r.name for r in ROOTS]}")
if MODE == "trimmed":
    VIDEO_DIR, N_VID = find_dir("mp4", "*_p[0-9][0-9]_r*_c[0-9][0-9].mp4")
else:
    VIDEO_DIR, N_VID = find_dir("Videos_mp4", "P*T*C*.mp4")
assert VIDEO_DIR is not None, f"no {MODE} RGB mount. Mounts: {[r.name for r in ROOTS]}"
RTMO_PATH = str(WEIGHTS / "rtmo-l.onnx")
# WEIGHTS comes from the RESOLVE cell above, which already located the weights mount by
# content. Re-resolving here with `find_asset` was how version 2 of this notebook died: it
# used the lighter RESOLVE_CODE cell, which resolves only the repo, so `find_asset` was
# never exported - NameError 370 s into the session, on a line unrelated to the real cause.
assert pathlib.Path(RTMO_PATH).is_file(), f"no rtmo-l.onnx under {WEIGHTS}"
print(f"meta   {META}")
print(f"video  {VIDEO_DIR}  ({N_VID:,} files)")
print(f"rtmo   {RTMO_PATH}")"""),
("code", """# The work list, and resume. Shards already written are kept; a session that times out is
# resumed by attaching this notebook's own previous output version, exactly as notebook 01
# carries `behaviorsense-adl-shards` forward.
from behaviorsense.data.toyota import (CS_TEST_SUBJECTS, CS_TRAIN_SUBJECTS, COARSE_V11,
                                       TSM_CLASSES, TSM_TO_COARSE, TSU_CLASSES,
                                       TSU_TO_COARSE, collapse_matrix, frame_mask,
                                       frame_multilabel, load_tsu_annotations,
                                       parse_tsm_name, parse_tsu_filename, protocol_side)

OUT = pathlib.Path("/kaggle/working/shards"); OUT.mkdir(parents=True, exist_ok=True)
# The sampling rate is in the shard name. Without it, re-running at a different FPS_SAMPLE
# would resume against shards whose windows span a different number of seconds, and the two
# rates would mix silently inside one training set - each clip contributing twice, at two
# different speeds, with no field recording which. A rate change now forces a clean
# extraction instead.
PREFIX = f"toyota_{MODE}_{FPS_SAMPLE:g}hz"

DONE = set()
# Carry forward EVERY toyota shard, not just this run's prefix, and this is a data-loss
# guard rather than tidiness. Save Version publishes whatever sits in /kaggle/working: an
# untrimmed run that copied only `toyota_untrimmed_*` forward would produce a new version of
# `behaviorsense-toyota-shards` containing ONLY the untrimmed half, silently dropping the
# 214,913 trimmed windows from the latest version. Copy everything in, add to it, publish the
# union.
#
# Mount-indexed, like every other lookup: stat <slug>/shards/ per mount instead of globbing.
# Even a FIXED-DEPTH glob (`datasets/*/*/*/shards/...`) makes pathlib scandir every slug
# child directory - the 16k skeleton root and MSMT17's crops included - which is the ~8.5
# minutes the first run of this notebook spent in its resolver cell. is_dir() guards each
# candidate, so nothing large is ever listed.
_prev_hits: list = []
for _slug in ROOTS:
    for _cand in (_slug / "shards", _slug / "kaggle" / "working" / "shards"):
        if _cand.is_dir():
            _prev_hits += list(_cand.glob("toyota_*.npz"))
for _prev in _prev_hits:
    if not (OUT / _prev.name).exists():
        shutil.copy2(_prev, OUT / _prev.name)
# DONE is built from THIS prefix only. Clip names do not collide between the halves
# (`Walk_p03_r01_v15_c07.mp4` against `P15T17C03.mp4`), but keying resume on the rate and
# mode that produced a shard is what makes a future rate change safe.
for _sh in sorted(OUT.glob(f"{PREFIX}_*.npz")):
    with np.load(_sh, allow_pickle=False) as _z:
        if "clips" in _z.files:
            DONE.update(str(c) for c in _z["clips"])
_carried = sorted(p.name.rsplit("_", 1)[0] for p in OUT.glob("toyota_*.npz"))
print(f"carried forward {len(list(OUT.glob('toyota_*.npz')))} shard(s) across "
      f"{len(set(_carried))} shard set(s): {sorted(set(_carried))}")
print(f"resuming {PREFIX}: {len(list(OUT.glob(PREFIX + '_*.npz')))} shard(s), "
      f"{len(DONE):,} clip(s) already extracted")

ANN, M_FINE = None, None
if MODE == "trimmed":
    WORK, skipped = [], 0
    for p in sorted(VIDEO_DIR.glob("*.mp4")):
        try:
            v = parse_tsm_name(p.name)
        except ValueError:
            skipped += 1                   # not a trimmed clip name; skip, never guess
            continue
        if v.activity not in TSM_TO_COARSE:
            skipped += 1                   # outside the 31-class space, counted not hidden
            continue
        WORK.append((p, v))
    print(f"{len(WORK):,} clips in the 31-class space, {skipped:,} outside it or unparseable")
else:
    ANN = load_tsu_annotations(META / "smarthome_CS_51.json")
    M_FINE = collapse_matrix(TSU_CLASSES, TSU_TO_COARSE)
    sides = ("train", "test") if UNTRIMMED_SIDE == "both" else (UNTRIMMED_SIDE,)
    WORK = []
    for p in sorted(VIDEO_DIR.glob("P*T*C*.mp4")):
        v = parse_tsu_filename(p)
        if protocol_side(v, "CS") in sides and v.tsu in ANN:
            WORK.append((p, v))
    print(f"{len(WORK)} untrimmed videos on the CS {'+'.join(sides)} side")

WORK = [w for w in WORK if w[0].name not in DONE][:MAX_CLIPS]
print(f"{len(WORK):,} to do this session")"""),
("code", """# Extract. RTMO per frame, slot assignment, fixed windows - all through the repo's own
# helpers so this notebook cannot drift from what the trained model expects.
from rtmlib import RTMO
from prepare_skeletons import WINDOW_FRAMES, assign_slots, window_clip

body = RTMO(onnx_model=RTMO_PATH, model_input_size=(640, 640),
            backend="onnxruntime", device="cuda")
_sess = getattr(body, "session", None)
assert _sess is None or any("CUDA" in p for p in _sess.get_providers()), (
    f"RTMO loaded on {_sess.get_providers()}, not CUDA")

def poses_for(path, fps_sample=FPS_SAMPLE, cap_frames=40_000):
    # -> ([T,M,17,3], source fps, effective rate, reason-if-empty). The rate is READ rather
    # than assumed: notebook 06 measured the trimmed containers at 20 fps while their own
    # README documents 30, and the first full run confirmed it on all 16,115 clips.
    #
    # EXACT resampling, not integer striding. Integer striding cost the first run 6,040 clips
    # (37.5%): at 20 fps a nominal 12.5 Hz rounds to stride 2, an effective 10 Hz, a 3.0 s
    # window, and every clip under 3 s produced nothing - concentrated in exactly the short
    # transition classes Agent 3 counts sit_to_stand_count from.
    #
    # It also cannot hold the window duration constant across the two halves: 25 fps
    # untrimmed strides to 25 Hz (1.20 s) while 20 fps trimmed strides to 20 Hz (1.50 s), so
    # the same 30-frame window would span different real time in the two shard sets. Picking
    # frame k as round(k * src / fps_sample) hits the requested rate on both, so a window is
    # 1.50 s everywhere and the two halves are directly comparable.
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, 0.0, 0.0, "decoder refused the file"
    src = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not (1.0 <= src <= 240.0):
        src = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    rate = min(fps_sample, src)          # never invent frames the container does not have
    if total and int(total * rate / src) < WINDOW_FRAMES:
        cap.release()
        return None, src, rate, f"only {total} frames at {src:g} fps, needs {WINDOW_FRAMES}"
    out, i, k = [], 0, 0
    want = 0
    while len(out) < cap_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i == want:
            kp, sc = body(frame)
            out.append(assign_slots(np.asarray(kp), np.asarray(sc), out[-1] if out else None))
            k += 1
            want = int(round(k * src / rate))
        i += 1
    cap.release()
    if len(out) < WINDOW_FRAMES:
        return None, src, rate, f"decoded {len(out)} frames, needs {WINDOW_FRAMES}"
    return np.stack(out), src, rate, ""

skels, fine, coarse, subs, srcs, clips, starts = [], [], [], [], [], [], []
shard_idx = len(list(OUT.glob(PREFIX + "_*.npz")))
n_ok = n_empty = n_gap = 0
src_rates = collections.Counter()
# Named drop reasons and the strides actually used - not one opaque "unusable" count. The
# first run reported 6,040 unusable and nothing about WHY, so finding the striding cause
# needed the class histogram plus arithmetic afterwards rather than the log itself.
drop_reasons = collections.Counter()
rates = collections.Counter()
t0 = time.time()

def flush():
    global skels, fine, coarse, subs, srcs, clips, starts, shard_idx
    if not skels:
        return
    np.savez_compressed(
        OUT / f"{PREFIX}_{shard_idx:04d}.npz",
        skeletons=np.stack(skels).astype(np.float16),
        # `labels` is the COARSE id so the existing loader's shape is unchanged;
        # `fine_labels` carries the 31/51-class id the benchmark head trains on. Both are
        # written because collapsing before the loss destroys the supervision that separates
        # the members - all five Cook.* classes become one target - and collapsing after
        # keeps every discriminative gradient.
        labels=np.asarray(coarse, dtype=np.int64),
        fine_labels=np.asarray(fine, dtype=np.int64),
        subjects=np.asarray(subs, dtype="<U32"),
        datasets=np.asarray(srcs, dtype="<U32"),
        clips=np.asarray(clips, dtype="<U64"),
        # The window's start index WITHIN its clip, at the sampled rate. Without it a
        # shard is an unordered bag: the kept windows are not an arithmetic sequence
        # (the visibility gate and the gap mask both drop some), so neither the temporal
        # order for segment metrics nor a per-frame timeline for mAP can be rebuilt.
        # Eight bytes a window against a seven-hour re-extraction.
        starts=np.asarray(starts, dtype=np.int64))
    print(f"  shard {shard_idx}: {len(skels):,} windows")
    shard_idx += 1
    skels, fine, coarse, subs, srcs, clips, starts = [], [], [], [], [], [], []

for seen, (path, vid) in enumerate(WORK, start=1):
    poses, src_fps, eff_rate, why = poses_for(path)
    src_rates[round(src_fps)] += 1
    if eff_rate:
        rates[round(eff_rate, 1)] += 1
    if poses is None:
        n_empty += 1
        drop_reasons[why or "unknown"] += 1
    else:
        windows = window_clip(poses, with_starts=True)
        if not windows:
            n_empty += 1
            drop_reasons["no window passed the visibility gate"] += 1
        elif MODE == "trimmed":
            f_id = TSM_CLASSES.index(vid.activity)
            c_id = COARSE_V11.index(TSM_TO_COARSE[vid.activity])
            for _start, w in windows:
                skels.append(w); fine.append(f_id); coarse.append(c_id)
                subs.append(f"p{vid.subject:02d}"); srcs.append("toyota_trimmed")
                clips.append(path.name); starts.append(int(_start))
            n_ok += 1
        else:
            # Dense labels. The multilabel tensor is built at the SAMPLED length so a
            # window's frame range maps back through the same step the decoder used.
            v = ANN[vid.tsu]
            n_steps = len(poses)
            lab = frame_multilabel(v, n_steps, official=False)
            msk = frame_mask(v, n_steps)
            added = 0
            for start, w in windows:
                sl = slice(start, start + WINDOW_FRAMES)
                if msk[sl].mean() < 0.5:
                    n_gap += 1
                    continue               # mostly unannotated: no supervised class here
                counts = lab[sl].sum(axis=0)
                f_id = int(counts.argmax())
                if counts[f_id] < WINDOW_FRAMES * 0.5:
                    n_gap += 1
                    continue               # no single class covers half the window
                skels.append(w); fine.append(f_id)
                coarse.append(int(np.argmax(M_FINE[f_id])))
                subs.append(f"p{vid.subject:02d}"); srcs.append("toyota_untrimmed")
                clips.append(path.name); starts.append(int(start))
                added += 1
            n_ok += 1 if added else 0
            n_empty += 0 if added else 1
    if len(skels) >= WINDOWS_PER_SHARD:
        flush()
    if seen % 500 == 0 or seen == len(WORK):
        el = time.time() - t0
        eta = el / seen * (len(WORK) - seen) / 60
        print(f"  {seen:,}/{len(WORK):,} clips, {n_ok:,} ok, {n_empty:,} unusable, "
              f"{len(skels):,} pending, {el / 60:.1f} min, eta {eta:.0f} min")
flush()
print(f"\\n{n_ok:,} clips extracted, {n_empty:,} unusable, {n_gap:,} windows dropped as gap")
print(f"source frame rates seen: {dict(src_rates)}")
print(f"effective sample rates: {dict(rates)} Hz   (exact resampling, so a 30-frame "
      f"window is {30 / FPS_SAMPLE:.2f}s on BOTH halves regardless of container rate)")
if drop_reasons:
    print("why clips were unusable:")
    for _r, _n in drop_reasons.most_common():
        print(f"  {_n:>6,}  {_r}")
print(f"{time.time() - t0:.0f}s total")"""),
("code", """# Verify what was written BEFORE Save Version. A shard set that trains but is mislabelled is
# the expensive failure, so every check here is on the CONTENT rather than the file count.
shards = sorted(OUT.glob(PREFIX + "_*.npz"))
assert shards, "no shards written - do NOT Save Version"

N, fine_hist, coarse_hist, subj = 0, collections.Counter(), collections.Counter(), set()
for sh in shards:
    with np.load(sh, allow_pickle=False) as z:
        N += len(z["labels"])
        fine_hist.update(z["fine_labels"].tolist())
        coarse_hist.update(z["labels"].tolist())
        subj.update(z["subjects"].tolist())
        assert z["skeletons"].shape[1:] == (WINDOW_FRAMES, 2, 17, 3), z["skeletons"].shape
        assert z["skeletons"].dtype == np.float16, z["skeletons"].dtype
        assert len(z["labels"]) == len(z["fine_labels"]) == len(z["subjects"])
        # `starts` is what makes a shard an ORDERED record rather than a bag of windows.
        # Older shards predate it; say so instead of failing, because the trimmed half's
        # clip-level labels need no timeline and are still usable without it.
        if "starts" not in z.files:
            print(f"  NOTE {sh.name} has no `starts` - segment metrics and per-frame mAP "
                  "cannot be rebuilt from it")
        else:
            assert len(z["starts"]) == len(z["labels"])

FINE_NAMES = TSM_CLASSES if MODE == "trimmed" else TSU_CLASSES
print(f"{N:,} windows in {len(shards)} shard(s), {len(subj)} subjects: {sorted(subj)}")
assert max(fine_hist) < len(FINE_NAMES), f"fine label {max(fine_hist)} outside the id space"
assert max(coarse_hist) < len(COARSE_V11), f"coarse label {max(coarse_hist)} out of range"

# The CS partition must be intact IN THE SHARDS, not just in the plan.
_tr = {s for s in subj if int(s[1:]) in CS_TRAIN_SUBJECTS}
_te = {s for s in subj if int(s[1:]) in CS_TEST_SUBJECTS}
print(f"CS subjects present - train {len(_tr)}, test {len(_te)}")
assert not (_tr & _te), "a subject appears on both CS sides"

print(f"\\n{'fine class':<34}{'windows':>9}{'share%':>8}")
for i, n in fine_hist.most_common(15):
    print(f"{FINE_NAMES[i]:<34}{n:>9,}{100 * n / N:>7.2f}")
if len(fine_hist) > 15:
    print(f"  ... and {len(fine_hist) - 15} more of {len(FINE_NAMES)} fine classes")
print(f"\\n{'coarse class':<34}{'windows':>9}{'share%':>8}")
for i, n in coarse_hist.most_common():
    print(f"{COARSE_V11[i]:<34}{n:>9,}{100 * n / N:>7.2f}")
_idle = coarse_hist.get(COARSE_V11.index("other_idle"), 0)
print(f"\\nother_idle: {_idle} windows. Zero is intended - every Toyota class is a real "
      "activity, which is the opposite of the Charades map where the fallback was largest.")
print(f"imbalance {max(fine_hist.values()) / max(1, min(fine_hist.values())):.0f}x over "
      f"{len(fine_hist)}/{len(FINE_NAMES)} fine classes present")

# MIN_SUPPORT mirrors the eval path: below ~50 windows a class's F1 is one prediction wide,
# so it is reported but not averaged. Naming those classes here is what stops a per-class
# mean being quoted over classes that cannot support one - the exact trap the Charades run
# fell into with `bending_reaching` at 26 windows.
MIN_SUPPORT = 50
_thin = sorted(((n, FINE_NAMES[i]) for i, n in fine_hist.items() if n < MIN_SUPPORT))
if _thin:
    print(f"\\n{len(_thin)} fine class(es) below MIN_SUPPORT={MIN_SUPPORT} - report them, "
          "do NOT average them:")
    for _n, _name in _thin:
        print(f"  {_n:>5}  {_name}")
else:
    print(f"every present fine class has >= {MIN_SUPPORT} windows")

# What the OUTPUT DATASET will contain, not just what this run produced. A Save Version
# publishes every shard in /kaggle/working, so this is the line that says whether both halves
# survived the carry-forward.
_all = sorted(OUT.glob("toyota_*.npz"))
_sets: dict = {}
for _sh in _all:
    _key = _sh.name.rsplit("_", 1)[0]
    with np.load(_sh, allow_pickle=False) as _z:
        _sets[_key] = _sets.get(_key, 0) + len(_z["labels"])
print(f"\\nthe dataset this Save Version will publish:")
for _key, _n in sorted(_sets.items()):
    print(f"  {_key:<28}{_n:>10,} windows")
print(f"  {'TOTAL':<28}{sum(_sets.values()):>10,} windows in {len(_all)} shard(s)")

pathlib.Path("/kaggle/working/toyota_shard_manifest.json").write_text(json.dumps({
    "mode": MODE, "fps_sample": FPS_SAMPLE, "window_frames": WINDOW_FRAMES,
    "untrimmed_side": UNTRIMMED_SIDE if MODE == "untrimmed" else None,
    "n_windows": N, "n_shards": len(shards), "subjects": sorted(subj),
    "source_frame_rates": {str(k): v for k, v in src_rates.items()},
    "effective_rates_hz": {str(k): v for k, v in rates.items()},
    "window_seconds": WINDOW_FRAMES / FPS_SAMPLE,
    "drop_reasons": dict(drop_reasons),
    "fine_histogram": {FINE_NAMES[i]: n for i, n in fine_hist.items()},
    "coarse_histogram": {COARSE_V11[i]: n for i, n in coarse_hist.items()},
    "clips_extracted": n_ok, "clips_unusable": n_empty, "windows_dropped_as_gap": n_gap,
}, indent=1), encoding="utf-8")
print("wrote /kaggle/working/toyota_shard_manifest.json")

print()
print("=" * 70)
print("  Save Version -> create/update PRIVATE dataset behaviorsense-toyota-shards")
print("=" * 70)
if MODE == "untrimmed" and UNTRIMMED_SIDE != "both":
    print(f"This run covered the CS {UNTRIMMED_SIDE} side only. Attach this dataset back as")
    print("an input, switch UNTRIMMED_SIDE, and re-run: carried-forward shards are kept and")
    print("their clips are skipped.")
elif len(WORK) >= MAX_CLIPS:
    print("MAX_CLIPS was hit, so this run did not cover everything. Attach this dataset")
    print("back as an input and re-run to continue where it stopped.")"""),
]


write("00_stage_assets_online.ipynb", N00)
write("01a_stage_charades_online.ipynb", N01A)
write("01_prepare_adl_shards_offline.ipynb", N01)
write("02_prepare_fall_shards_online.ipynb", N02)
write("03_train_blackwell_offline.ipynb", N03)
write("04_evaluate_blackwell_offline.ipynb", N04)
write("05_serve_inference_online.ipynb", N05)
write("06_toyota_preflight_online.ipynb", N06)
write("07_toyota_extract_offline.ipynb", N07)

for f in sorted(HERE.glob("*.ipynb")):
    json.loads(f.read_text(encoding="utf-8"))
print("all notebooks are valid JSON")
