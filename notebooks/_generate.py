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
    ("src/behaviorsense/models/stgcnpp.py", "parent.setdefault",
     "flip-equivariant bone stream; without it half the ensemble trains on sign noise"),
    ("src/behaviorsense/kaggle_artifacts.py", "def find_run_dir",
     "one rule for 'is this real session output or a dev leftover'; four call sites "
     "learned it separately and the fourth was missed"),
    ("src/behaviorsense/agents/reasoning/reporter.py", "def repair_claim",
     "format-only claim repair + maxItems bound to max_claims (notebook 04). Without "
     "it the unconstrained arm scores 245 emitted / 0 scorable / nan%, which measures "
     "JSON compliance rather than faithfulness"),
    ("scripts/eval_hallucination.py", "unusable_rate",
     "three-arm hallucination table with a denominator over EMITTED claims; a stale "
     "snapshot silently reports the two-arm nan% version"),
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


RESOLVE_CODE = """# Resolve the repo mount by CONTENT, not by dataset name.
#
# The dataset title is free text and this project has already been uploaded under more
# than one spelling ("behaviorsense-*" and "behavioursense-*"). Hard-coding the name makes
# cell 1 of a 12-hour session fail on a typo, so find the repo by a file only it contains.
import pathlib

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
                    top = sorted(q.name for q in ds.iterdir())[:6] if ds.is_dir() else []
                    out.append(f"{ds.name} (top level: {top})")
            else:
                out.append(owner.name)
    # Fall back to the flat layout so this keeps working if Kaggle changes the mount
    # shape back, rather than reporting nothing at all.
    return out or sorted(p.name for p in INPUT.iterdir())

ATTACHED = attached_mounts() if INPUT.is_dir() else []
_init = sorted(INPUT.glob("**/src/behaviorsense/__init__.py"))
assert _init, (f"repo dataset not found: nothing matches **/src/behaviorsense/__init__.py "
               f"under /kaggle/input. Attached datasets: {ATTACHED}")
SRC     = _init[0].parent.parent
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
import pathlib

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
                    top = sorted(q.name for q in ds.iterdir())[:6] if ds.is_dir() else []
                    out.append(f"{ds.name} (top level: {top})")
            else:
                out.append(owner.name)
    # Fall back to the flat layout so this keeps working if Kaggle changes the mount
    # shape back, rather than reporting nothing at all.
    return out or sorted(p.name for p in INPUT.iterdir())

ATTACHED = attached_mounts() if INPUT.is_dir() else []

def find_asset(pattern, what, required=True):
    hits = sorted(INPUT.glob(pattern))
    if not hits:
        if required:
            raise AssertionError(
                f"{what}: nothing matches {pattern!r} under /kaggle/input. "
                f"Attached datasets: {ATTACHED}")
        print(f"  {what:<9} ABSENT (optional)")
        return None
    return hits[0]

def find_wheel_dir():
    # "The directory containing *.whl" is not specific enough: /kaggle/input also holds
    # attached COMPETITIONS, and at least one (arc-prize-2026) ships its own wheels. The
    # first sorted hit was that competition's, and the offline install then failed on a
    # cache that simply does not contain torch. Score candidate directories by how many
    # of OUR packages they hold and take the best.
    MARKERS = {"torch", "rtmlib", "onnxruntime-gpu", "nvidia-cudnn-cu12", "triton"}
    dirs = {}
    for w in INPUT.glob("**/*.whl"):
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

def launch(stream):
    out = f"/kaggle/working/runs/adl_{stream}"
    cmd = [sys.executable, str(SCRIPTS / "train_adl.py"),
           "--shards", *ADL, "--stream", stream, "--epochs", EPOCHS,
           "--batch-size", BATCH, "--device", "cuda", "--seed", SEED,
           "--patience", str(PATIENCE["adl"]), "--out", out]
    if pathlib.Path(out, "last.pt").exists():
        cmd += ["--resume", f"{out}/last.pt"]
    log = open(f"/kaggle/working/train_{stream}.log", "a")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)

if PARALLEL:
    procs = {s: launch(s) for s in STREAMS}
    for s, p in procs.items():
        print(s, "->", "OK" if p.wait() == 0 else "FAILED (see train_%s.log)" % s)
else:
    for s in STREAMS:
        print(s, "->", "OK" if launch(s).wait() == 0 else f"FAILED (train_{s}.log)")"""),
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
print("fall head:", "OK" if r.returncode == 0 else "FAILED")"""),
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
print("Save Version -> create/update dataset behaviorsense-runs from /kaggle/working/runs")
print("Timed out mid-training? Save Version anyway - resume continues bit-exactly next "
      "session.")"""),
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
   mis-assigned values. With the mapping stated, run 2 gave **8.2% constrained vs 6.5%
   free, zero schema rejections in either arm** — so most of that 39.9% was our prompt.
   The two arms are statistically indistinguishable (z = 1.06, p = 0.29), which is the
   honest result: the grammar buys a *guarantee* of parseability at ~7.5x the decoding
   time, not better faithfulness. The verifier is what catches the residual 6-8%."""),
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
sys.path.insert(0, str(SRC))"""),
("code", """# P1 - subject-disjoint mean-class accuracy / macro-F1, per stream and ensemble.
# split_by_subject with the SAME seed as training reproduces the training-time val split.
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
_, val_idx = split_by_subject(ds.subjects, val_frac=0.15, seed=0)
print(f"P1 val: {len(val_idx)} windows, {len(set(ds.subjects[val_idx]))} subjects")
assert not (set(ds.subjects[val_idx]) & set(np.delete(ds.subjects, val_idx))), \\
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
MIN_SUPPORT = 50

def scores(logits, y):
    pred = logits.argmax(1)
    per_class = [np.mean(pred[y == c] == c) for c in range(N_CLASSES) if (y == c).any()]
    f1, f1_sup = [], []
    for c in range(N_CLASSES):
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum()
        fn = ((pred != c) & (y == c)).sum()
        if tp + fp + fn:
            v = 2 * tp / max(2 * tp + fp + fn, 1)
            f1.append(v)
            if (y == c).sum() >= MIN_SUPPORT:
                f1_sup.append(v)
    return (np.mean(pred == y), np.mean(per_class), np.mean(f1),
            np.mean(f1_sup) if f1_sup else float("nan"))

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
    print("  so build_charades_map.py routes them to other_idle. Report both columns.")"""),
("code", """# Calibration: fit the temperature on these val logits. The fitted T goes into
# ActivityConfig(temperature=...) at serving time - Viterbi and abstention both consume
# probabilities, so they must mean something first.
from behaviorsense.agents.activity import fit_temperature, softmax
T = fit_temperature(ens_logits, y)
conf = softmax(ens_logits / T).max(1).mean()
acc = (ens_logits.argmax(1) == y).mean()
print(f"fitted temperature T={T:.2f}; mean confidence {conf:.3f} vs accuracy {acc:.3f}")
print(f"-> set ActivityConfig(temperature={T:.2f}) in deployment")"""),
("code", """# P2 - leave-one-dataset-out on the fall sources: train-side generalisation is fixed
# (the ensemble saw only Charades ADL + the other fall sources via train_fall), so this
# measures how the FALL signal transfers to an unseen recording setup.
import glob, numpy as np
from behaviorsense.data.skeleton_dataset import SkeletonWindowDataset, normalise
FALL = shard_paths("fall")
assert FALL, f"no fall shards found. Attached: {ATTACHED}"
fds = SkeletonWindowDataset([*map(str, FALL)])
sources = np.asarray(fds.datasets)
FALL_CLASSES = (7, 8)
is_fall = np.isin(fds.labels, FALL_CLASSES)
# Collected, not just printed. The first full run left every P1/P2/ablation table only in
# the Kaggle log - the bundling cell wrote one file, hallucination_qwen.md - so the
# dissertation numbers would have died with the session output. The final cell writes
# these to results/ from P2_ROWS.
P2_ROWS = []
print(f"{'held-out':<12} {'n':>6} {'fall%':>6} {'AUROC(fall posterior)':>22}")
for held in sorted(set(sources)):
    idx = np.where(sources == held)[0]
    # SHARD layout [N,T,M,17,3], same as the P1 block above. fds[i][0] returns
    # [C,T,V,M] - already permuted for the model - and logits() rejects it. The P1 cell
    # was fixed for exactly this and P2 was left behind, so it died after 11 minutes with
    # "expected [N,T,M,17,3], got (1159, 3, 30, 17, 2)".
    Xh = np.stack([normalise(fds.skeletons[i].astype(np.float32)) for i in idx])
    post = softmax(clf.logits(Xh))
    score = post[:, list(FALL_CLASSES)].sum(1)
    yh = is_fall[idx]
    if yh.any() and (~yh).any():
        order = np.argsort(score)
        ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
        auroc = (ranks[yh].sum() - yh.sum() * (yh.sum() + 1) / 2) / (yh.sum() * (~yh).sum())
    else:
        auroc = float("nan")
    P2_ROWS.append((str(held), int(len(idx)), float(yh.mean()), float(auroc)))
    print(f"{held:<12} {len(idx):>6} {yh.mean():>6.1%} {auroc:>22.3f}")"""),
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
# With the field mapping stated, run 2 gave 8.2% / 6.5% with zero schema rejections in
# either arm - i.e. most of that 39.9% was our prompt, not the model.
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
L += table(["held-out", "n", "fall%", "AUROC"], P2_ROWS,
           lambda r: [f"`{r[0]}`", str(r[1]), f"{r[2]:.1%}",
                      "n/a" if r[3] != r[3] else f"{r[3]:.3f}"])
L += ["", "## Ablation - logit vs probability combination", ""]
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



write("00_stage_assets_online.ipynb", N00)
write("01a_stage_charades_online.ipynb", N01A)
write("01_prepare_adl_shards_offline.ipynb", N01)
write("02_prepare_fall_shards_online.ipynb", N02)
write("03_train_blackwell_offline.ipynb", N03)
write("04_evaluate_blackwell_offline.ipynb", N04)

for f in sorted(HERE.glob("*.ipynb")):
    json.loads(f.read_text(encoding="utf-8"))
print("all notebooks are valid JSON")
