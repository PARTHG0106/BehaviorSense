"""Verify the Kaggle notebooks against the real codebase, locally, before a GPU session.

The problem this solves
-----------------------
A notebook that calls a function with the wrong shape, a stale keyword, or a renamed
attribute fails *in the session* - after the wheels install, after the shards mount,
often 40 minutes in. The offline notebooks cannot be debugged interactively either
(no internet, and a 12-hour clock). So the load-bearing logic of notebooks 03 and 04 is
extracted here and run against synthetic shards on CPU.

This is not a test of Kaggle. It is a test that the CODE THE NOTEBOOKS CALL still has the
signatures the notebooks assume. It caught one real defect already: notebook 04 built its
eval tensor from `ds[i][0]`, which returns [C,T,V,M] (already permuted for the model),
while EnsembleClassifier.logits() expects the dataset layout [N,T,M,17,3].

Run: python tests/test_notebooks.py
"""

from __future__ import annotations

import ast

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import os

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from behaviorsense.data.skeleton_dataset import (  # noqa: E402
    N_CLASSES,
    SkeletonWindowDataset,
    normalise,
    split_by_subject,
)
from behaviorsense.models.ensemble import EnsembleClassifier  # noqa: E402
from behaviorsense.models.stgcnpp import STGCNpp  # noqa: E402

NOTEBOOKS = ROOT / "notebooks"
# Inherit the real environment and only ADD PYTHONPATH. A minimal env looks tidy but
# strips SystemRoot on Windows, which breaks Winsock init inside torch.distributed
# (WinError 10106) - a failure that looks like a torch bug and is actually the test's.
SUBPROC_ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
FALL_CLASSES = (7, 8)



def _cell_containing(notebook: str, needle: str) -> str:
    """The one code cell of `notebook` containing `needle`, asserting it is unique.

    Uniqueness matters: a test that silently picks the first of several matches stops
    testing what its name says as soon as the notebook grows a similar cell.
    """
    doc = json.loads((NOTEBOOKS / notebook).read_text(encoding="utf-8"))
    hits = [c["source"] for c in doc["cells"]
            if c["cell_type"] == "code" and needle in c["source"]]
    assert len(hits) == 1, f"{notebook}: {len(hits)} cells contain {needle!r}, want 1"
    return hits[0]


def contract_entries() -> list[tuple[str, str | None]]:
    """(path, token) pairs the notebooks' repo-staleness guard enforces.

    Parsed out of the generated notebook rather than duplicated here, so the test and the
    notebook cannot disagree about what "current enough" means.
    """
    doc = json.loads(
        (NOTEBOOKS / "03_train_blackwell_offline.ipynb").read_text(encoding="utf-8"))
    cell = [c["source"] for c in doc["cells"] if "CONTRACT = [" in c["source"]][0]
    body = cell.split("CONTRACT = [", 1)[1].split("]", 1)[0]
    out: list[tuple[str, str | None]] = []
    for path, token in re.findall(r'\("([\w/.]+)",\s*(?:"([^"]*)"|None)', body):
        out.append((path, token or None))
    assert out, "could not parse the repo contract out of notebook 03"
    return out



def _nb_epoch_budget() -> dict[str, int]:
    """EPOCH_BUDGET as notebook 03 defines it, so fixtures cannot drift from the code.

    The ADL budget moved 80 -> 30 after measuring that every stream peaked at epoch 8-11.
    A test that hard-codes the old value silently starts testing the resume guard instead
    of what its name says.
    """
    doc = json.loads(
        (NOTEBOOKS / "03_train_blackwell_offline.ipynb").read_text(encoding="utf-8"))
    src = chr(10).join(c["source"] for c in doc["cells"] if c["cell_type"] == "code")
    body = re.search(r"EPOCH_BUDGET\s*=\s*\{([^}]*)\}", src)
    assert body, "notebook 03 no longer defines EPOCH_BUDGET"
    return {k: int(v) for k, v in re.findall(r'"(\w+)":\s*(\d+)', body.group(1))}


def make_shard(path: Path, n: int = 120, seed: int = 0) -> Path:
    """A shard in the exact layout notebooks 01/02 write."""
    rng = np.random.default_rng(seed)
    sk = rng.normal(0, 0.4, (n, 30, 2, 17, 3)).astype(np.float16)
    sk[..., 2] = 0.9
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        skeletons=sk,
        labels=rng.integers(0, N_CLASSES, n).astype(np.int64),
        subjects=np.array([f"v{i % 12:03d}" for i in range(n)], dtype="<U32"),
        datasets=np.array(["charades"] * n, dtype="<U32"),
    )
    return path


def save_ckpt(path: Path, stream: str, seed: int = 0) -> Path:
    torch.manual_seed(seed)
    m = STGCNpp(n_classes=N_CLASSES)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": m.state_dict(), "ema": m.state_dict(), "ema_step": 1, "epoch": 0,
         "global_step": 1, "best": 0.0, "args": {"stream": stream, "epochs": 1},
         "metrics": {}},
        path,
    )
    return path


# ---------------------------------------------------------------------------
# N: notebook structure
# ---------------------------------------------------------------------------


def test_n1_all_notebooks_are_valid_and_declare_their_inputs():
    expected = {
        "00_stage_assets_online.ipynb": ("behaviorsense-code",),
        "01a_stage_charades_online.ipynb": (),
        "01_prepare_adl_shards_offline.ipynb": ("behaviorsense-code",),
        "02_prepare_fall_shards_online.ipynb": ("behaviorsense-code",),
        # Notebook 03 deliberately names NO dataset: notebook 00's wheels/ and weights/
        # may be published as one dataset or two, and the resolver finds every asset by
        # content. Requiring the literal names here would demand a header that lies about
        # what you must attach. What it must document is the ASSETS.
        "03_train_blackwell_offline.ipynb": (
            "*.whl", "rtmo-l.onnx", "ADL shards", "fall shards", "behaviorsense-runs"),
        "04_evaluate_blackwell_offline.ipynb": (
            "behaviorsense-code", "behaviorsense-runs"),
    }
    for name, needs in expected.items():
        path = NOTEBOOKS / name
        assert path.is_file(), f"missing {name}"
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc["cells"], f"{name} has no cells"
        assert doc["cells"][0]["cell_type"] == "markdown", f"{name} lacks a header cell"
        text = json.dumps(doc)
        header = doc["cells"][0]["source"]
        for dataset in needs:
            assert dataset in text, f"{name} never references input {dataset}"
            # Each declared input must appear in the HEADER, not merely somewhere in the
            # file - the header is what you read while configuring the session.
            assert dataset in header, (
                f"{name} mentions {dataset!r} only in code; the header is the only thing "
                "read while attaching inputs")
        for cell in doc["cells"]:
            if cell["cell_type"] == "code":
                compile(cell["source"], f"{name}:cell", "exec")
    print(f"  N1 {len(expected)} notebooks: valid JSON, documented inputs, "
          "every code cell compiles")



def test_n1b_subprocesses_inherit_the_environment():
    """A hand-built env= dict silently removes CUDA from the child process.

    Notebook 03's preflight passed `env={"PYTHONPATH": ..., "PATH": ...}`, which drops
    every CUDA variable Kaggle sets. The child then reported "CUDA is not available -
    training would silently run on CPU" ONE LINE after the parent cell printed
    "torch 2.10.0+cu128 already supports sm_120", and aborted the session on a GPU that
    was working perfectly. Same trap this test file hit on Windows, where a minimal env
    strips SystemRoot and breaks Winsock inside torch.distributed.

    The rule is: inherit, then ADD. Never construct.
    """
    sites = 0
    for nb_path in sorted(NOTEBOOKS.glob("*.ipynb")):
        doc = json.loads(nb_path.read_text(encoding="utf-8"))
        for i, cell in enumerate(doc["cells"]):
            if cell["cell_type"] != "code":
                continue
            for m in re.finditer(r"env=\{[^}]*\}", cell["source"]):
                sites += 1
                assert "os.environ" in m.group(0), (
                    f"{nb_path.name} cell {i} builds a subprocess env from scratch: "
                    f"{m.group(0)[:80]} - this strips CUDA (and PATH, and HOME) from the "
                    "child. Use {**os.environ, ...}")
    assert sites >= 3, f"only {sites} env= sites found - did the pattern change?"
    print(f"  N1b {sites} subprocess env= sites across the notebooks all inherit "
          "os.environ (a constructed env hides the GPU from the child)")



def test_n1c_failure_message_names_the_actual_datasets():
    """`Attached datasets: ['competitions', 'datasets']` told us nothing.

    Datasets mount at /kaggle/input/datasets/<owner>/<name>/, so listing the top level
    of /kaggle/input always yields those two container directories no matter what is
    attached. That string was the ONLY diagnostic printed when the resolver failed, and
    it could not answer the one question that mattered: is behaviorsense-code attached,
    and does it contain src/?

    The listing must descend to real mounts and report what each one holds - a dataset
    can be attached and still be missing the directory the notebook needs, which is
    exactly the case that produced this test.
    """
    import contextlib
    import io
    import shutil
    import tempfile

    cell = _cell_containing("03_train_blackwell_offline.ipynb", "attached_mounts")
    root = Path(tempfile.mkdtemp())
    ds = root / "datasets" / "someowner"
    (root / "competitions" / "arc-prize-2026").mkdir(parents=True)
    # The real failure: code dataset attached, but with no src/ inside it.
    code = ds / "behaviorsense-code" / "EmotionSense-Extended"
    for sub in ("docs", "notebooks", "tests"):
        (code / sub).mkdir(parents=True)
    ww = ds / "behavioursense-ww"
    (ww / "wheels").mkdir(parents=True)
    (ww / "weights").mkdir(parents=True)
    (ww / "wheels" / "torch-2.7.1-cp311-cp311-linux_x86_64.whl").write_bytes(b"x")
    (ww / "weights" / "rtmo-l.onnx").write_bytes(b"x")

    src = cell.replace('pathlib.Path("/kaggle/input")', f'pathlib.Path(r"{root}")')
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src, "<resolver>", "exec"), {"__name__": "nb"})
        raise AssertionError("resolver accepted a code dataset with no src/")
    except AssertionError as exc:
        msg = str(exc)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    assert "behaviorsense-code" in msg, (
        f"failure message does not name the attached datasets: {msg}")
    assert "behavioursense-ww" in msg, "wheels/weights mount not listed"
    assert msg.count("top level") >= 2, (
        "message lists dataset names but not their contents - a dataset can be attached "
        "and still lack the directory the notebook needs")
    assert "['competitions', 'datasets']" not in msg, (
        "message still reports the container directories instead of real mounts")
    print("  N1c resolver failure names each attached dataset and its top-level contents")


def test_n2_offline_notebooks_never_reach_the_network():
    """An offline notebook that calls pip/urllib without --no-index hangs, then dies."""
    banned = ("urllib.request.urlretrieve", "git clone", "kagglehub.")
    for name in ("01_prepare_adl_shards_offline.ipynb",
                 "03_train_blackwell_offline.ipynb",
                 "04_evaluate_blackwell_offline.ipynb"):
        doc = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
        for cell in doc["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = cell["source"]
            for token in banned:
                assert token not in src, f"{name} performs network access: {token}"
            if '"pip", "install"' in src or '"-m", "pip", "install"' in src:
                assert "--no-index" in src, (
                    f"{name} pip-installs without --no-index; it would hang offline"
                )
    # Control: the ONLINE notebooks must genuinely download, or they are staging nothing.
    for name in ("00_stage_assets_online.ipynb", "01a_stage_charades_online.ipynb"):
        online = json.dumps(json.loads((NOTEBOOKS / name).read_text(encoding="utf-8")))
        assert "urlretrieve" in online, f"{name} downloads nothing - it stages nothing"
    print("  N2 3 offline notebooks: no downloads, all pip installs use --no-index; "
          "both online notebooks do download")


def test_n2b_header_prose_matches_what_the_notebook_actually_does():
    """The header is the only instruction you read while configuring the session.

    Notebook 01 was converted from online-on-a-T4 to offline-on-the-Blackwell, and its
    header kept saying "ONLINE, GPU T4/P100, internet ON" plus "Downloads Charades" -
    a stale header is worse than none, because it is followed: you would attach the
    internet, skip the wheels dataset, and the run dies on cell 1. Code cells are
    already covered by N2; this covers the prose.
    """
    checked = 0
    for nb_path in sorted(NOTEBOOKS.glob("*.ipynb")):
        doc = json.loads(nb_path.read_text(encoding="utf-8"))
        header = doc["cells"][0]["source"]
        assert doc["cells"][0]["cell_type"] == "markdown", f"{nb_path.name}: no header"
        offline_name = "offline" in nb_path.stem
        # The filename declares the contract; the header must not contradict it.
        if offline_name:
            assert re.search(r"\bOFFLINE\b", header), (
                f"{nb_path.name} is an offline notebook but its header never says OFFLINE")
            assert "internet ON" not in header, (
                f"{nb_path.name} is offline but its header claims 'internet ON'")
            for claim in ("T4/P100", "Downloads "):
                assert claim not in header, (
                    f"{nb_path.name} header still claims {claim!r} - stale from when it "
                    "ran online")
        else:
            assert "ONLINE" in header, (
                f"{nb_path.name} is an online notebook but its header never says ONLINE")
        # A header that names input datasets must name the ones the code resolves.
        body = " ".join(c["source"] for c in doc["cells"] if c["cell_type"] == "code")
        if "**/*.whl" in body:
            assert re.search(r"wheel|WW", header, re.I), (
                f"{nb_path.name} needs the wheels dataset but its header never says so")
        checked += 1
    assert checked == 9, f"expected 9 notebooks, checked {checked}"
    print(f"  N2b {checked} headers agree with their filenames, internet mode, and the "
          "datasets their code resolves")


def test_n2c_pose_model_is_built_from_the_staged_onnx():
    """RTMO must be given an explicit local path, never left to resolve one itself.

    Notebook 00 goes to real trouble to stage `rtmo-l.onnx` and asserts it exists
    "because notebooks 01/02 cannot extract pose without it" - but both notebooks used to
    construct `RTMO(mode="performance", ...)`, which never referenced that file. Offline
    that either hangs on a download or raises; either way the staged asset was consumed
    by nothing, and the failure lands after the 13 GB unzip rather than on cell 1.
    """
    checked = 0
    for name in ("01_prepare_adl_shards_offline.ipynb",
                 "02_prepare_fall_shards_online.ipynb"):
        doc = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
        body = "\n".join(c["source"] for c in doc["cells"] if c["cell_type"] == "code")
        ctor = re.search(r"RTMO\(([^)]*)\)", body, re.S)
        assert ctor, f"{name} never constructs RTMO"
        args = ctor.group(1)
        assert "onnx_model=" in args, (
            f"{name} builds RTMO without onnx_model= - it would resolve its own weights "
            f"instead of the staged rtmo-l.onnx. Got: RTMO({args.strip()[:80]}...)")
        assert "mode=" not in args, (
            f"{name} still passes mode= to RTMO, which selects a downloadable checkpoint")
        assert 'glob("**/rtmo-l.onnx")' in body, (
            f"{name} must locate rtmo-l.onnx by content, like every other asset")
        checked += 1
    # The offline one must also PROVE the model runs before the extraction loop.
    off = json.loads(
        (NOTEBOOKS / "01_prepare_adl_shards_offline.ipynb").read_text(encoding="utf-8"))
    cells = [c["source"] for c in off["cells"] if c["cell_type"] == "code"]
    probe = next((i for i, s in enumerate(cells) if "probe OK" in s), None)
    loop = next((i for i, s in enumerate(cells) if "cap.read()" in s), None)
    assert probe is not None, "notebook 01 has no pose-model probe"
    assert loop is not None and probe < loop, (
        "the probe must run BEFORE the extraction loop, or it proves nothing in time")
    print(f"  N2c {checked} extraction notebooks build RTMO from the staged onnx; "
          "notebook 01 probes it before the loop")


def test_n2d_charades_layout_detection_handles_both_shapes():
    """Kaggle auto-extracts archives, so the zip layout is not the one you usually get.

    01a saves Charades_v1_480.zip unopened; the published dataset arrived as 9,848 loose
    mp4s under Charades_v1_480/Charades_v1_480/. The zip-only reader asserted
    "charades-480p is not attached" against a dataset that was attached AND extracted -
    and it did so after the wheels install, ~7 minutes into the session.
    """
    import contextlib
    import io
    import tempfile

    doc = json.loads(
        (NOTEBOOKS / "01_prepare_adl_shards_offline.ipynb").read_text(encoding="utf-8"))
    codes = [c["source"] for c in doc["cells"] if c["cell_type"] == "code"]
    cell = next(c for c in codes if "layout PRE-EXTRACTED" in c)

    def build(kind: str) -> Path:
        root = Path(tempfile.mkdtemp()) / "charades-480p"
        ids = [f"VID{i:03d}" for i in range(5)]
        csv_text = "id,actions\n" + "\n".join(f"{v},c001 0.0 5.0" for v in ids) + "\n"
        if kind == "extracted":
            vd = root / "Charades_v1_480" / "Charades_v1_480"
            vd.mkdir(parents=True)
            for v in ids:
                (vd / f"{v}.mp4").write_bytes(b"\x00" * 32)
            ann = root / "Charades_annotations" / "Charades"
            ann.mkdir(parents=True)
            (ann / "Charades_v1_train.csv").write_text(csv_text, encoding="utf-8")
        else:
            root.mkdir(parents=True)
            import zipfile as _zf
            with _zf.ZipFile(root / "Charades_v1_480.zip", "w") as z:
                for v in ids:
                    z.writestr(f"Charades_v1_480/{v}.mp4", b"\x00" * 32)
            with _zf.ZipFile(root / "Charades_annotations.zip", "w") as z:
                z.writestr("Charades/Charades_v1_train.csv", csv_text)
        return root.parent

    for kind, expect in (("extracted", "PRE-EXTRACTED"), ("zips", "ZIPS")):
        root = build(kind)
        src = cell.replace('pathlib.Path("/kaggle/input")',
                           f"pathlib.Path({root.as_posix()!r})")
        assert root.as_posix() in src, "redirect failed"
        ns = {"__name__": "nb01", "VID_START": 0, "VID_END": 3}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(src, f"<{kind}>", "exec"), ns)
        out = buf.getvalue()
        assert expect in out, f"{kind}: expected layout {expect}, got:\n{out}"
        # Both layouts must expose the same interface to the extraction cell.
        assert len(ns["rows"]) == 3, f"{kind}: slice gave {len(ns['rows'])} rows, want 3"
        resolved = [ns["video_path"](r["id"]) for r in ns["rows"]]
        assert all(p is not None and p.is_file() for p in resolved), (
            f"{kind}: video_path() failed to resolve {resolved}")

    # Neither layout present must refuse, naming both patterns it looked for.
    empty = Path(tempfile.mkdtemp())
    (empty / "unrelated").mkdir()
    src = cell.replace('pathlib.Path("/kaggle/input")',
                       f"pathlib.Path({empty.as_posix()!r})")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src, "<empty>", "exec"),
                 {"__name__": "nb01", "VID_START": 0, "VID_END": 3})
        raise AssertionError("missing Charades resolved anyway - the check proves nothing")
    except AssertionError as exc:
        msg = str(exc)
        assert "Charades_v1_train.csv" in msg and "Charades*.zip" in msg, (
            f"refusal names only one layout, so it misdiagnoses the other: {msg}")
    print("  N2d both Charades layouts resolve to the same rows/video_path interface; "
          "absence refuses naming both")



def test_n16_every_fall_corpus_is_labelled_by_its_own_convention():
    """Four corpora, four different ways of saying "this clip is a fall".

    A single rule cannot serve all of them, and getting it wrong is silent:
      - GMDCSA-24  `Subject N/Fall/01.mp4`      - directory, exact token
      - URFD       `fall-01-cam0.mp4`           - filename prefix
      - CAUCAFall  `Subject.1/Fall forward/...` - directory, MULTI-WORD. An exact-match
        rule scored all 50 fall clips as ADL, feeding real falls to the fall head as
        negatives.
      - Le2i       `Home_01/video (24).avi`     - NO marker anywhere; the truth is in an
        Annotation_files .txt. Defaulting to "not a fall" would bury ~192 fall clips in
        the negatives, so an unlabellable clip must be SKIPPED, never guessed.
    """
    import tempfile
    from prepare_skeletons import (is_fall_clip, label_fall_windows, le2i_fall_frames,
                                   subject_from_path)

    marker_cases = [
        ("Subject 1/Fall/01.mp4", True), ("Subject 1/ADL/01.mp4", False),
        ("fall-01-cam0.mp4", True), ("adl-40-cam0.mp4", False),
        ("Subject.1/Fall forward/v.avi", True), ("Subject.1/Fall backward/v.avi", True),
        ("Subject.1/Walking/v.avi", False), ("Subject.1/Picking up object/v.avi", False),
        ("no_fall/clip.avi", False),
    ]
    root = Path("/tmp/_corpus")
    for rel, want in marker_cases:
        got = is_fall_clip(root, root / rel)
        assert got == want, f"is_fall_clip({rel!r}) = {got}, want {want}"

    # Subject ids must stay distinct per PERSON; every corpus numbers clips from 01.
    ids = {subject_from_path(root, root / f"Subject {i}/Fall/01.mp4", "gmdcsa")
           for i in range(1, 5)}
    assert len(ids) == 4, f"GMDCSA subjects collapsed to {ids}"

    # Le2i annotations, including the two cases that must NOT become negatives.
    tmp = Path(tempfile.mkdtemp()) / "le2i" / "Home_01"
    (tmp / "Videos").mkdir(parents=True)
    (tmp / "Annotation_files").mkdir(parents=True)
    NL = chr(10)
    for stem, txt, want in [("video (1)", "421" + NL + "480" + NL, (421, 480)),
                            ("video (2)", "0" + NL + "0" + NL, (0, 0)),
                            ("video (3)", "", None)]:
        (tmp / "Videos" / f"{stem}.avi").write_bytes(b"\x00" * 64)
        (tmp / "Annotation_files" / f"{stem}.txt").write_text(txt, encoding="utf-8")
        assert le2i_fall_frames(tmp / "Videos" / f"{stem}.avi") == want, stem
    (tmp / "Videos" / "orphan.avi").write_bytes(b"\x00" * 64)
    assert le2i_fall_frames(tmp / "Videos" / "orphan.avi") is None, (
        "an unannotated Le2i clip must return None so the caller skips it")

    # Exact-interval labelling beats the positional guess where annotations exist.
    starts = list(range(0, 151, 15))
    exact = label_fall_windows(starts, 30, (60, 90))
    assert exact[:3] == [1, 1, 1] and 7 in exact and exact[-1] == 8, exact
    assert label_fall_windows(starts, 30, (0, 0)) == [19] * len(starts), (
        "an annotated no-fall clip is ADL, not a fall with no descent")
    # And no clip length may lose its falling window under the heuristic.
    for n in range(2, 16):
        labs = label_fall_windows(list(range(0, n * 15, 15)), 30, None)
        assert 7 in labs, f"{n}-window clip produced no `falling` label"

    src = _cell_containing("02_prepare_fall_shards_online.ipynb", "unlabelled[source]")
    assert "unlabelled[source] += 1" in src and "continue" in src, (
        "notebook 02 must SKIP unlabellable Le2i clips, not default them to ADL")
    assert "fall_range[0] // step" in src, (
        "Le2i annotations are in ORIGINAL frames; without dividing by the decode step "
        "the interval is ~2x wrong at 15 fps")
    print(f"  N16 {len(marker_cases)} marker cases + Le2i annotations (fall/no-fall/"
          "unparseable/orphan); exact intervals and the positional fallback both hold")



def test_n17_notebook_02_banks_work_and_appends_each_window_once():
    """Two defects that cost a 27-minute session, and one that nearly shipped after it.

    1. No incremental flush. gmdcsa (160 clips) and urfd (70) had extracted cleanly when
       the kernel died on a Le2i clip - and all of it was lost, because the cell held
       everything in memory and wrote one npz at the end.
    2. No frame cap. The run logged three "[mp3float] Header missing" lines and died on
       the next: cv2 handing back frames from a broken index until RAM was gone.
    3. Fixing (1) left the OLD positional-labelling block reachable, so every window was
       appended TWICE - the second copy labelled by the superseded rule, which would have
       given ADL clips fall labels. Caught by executing the cell logic, not reading it.
    """
    src = _cell_containing("02_prepare_fall_shards_online.ipynb", "def flush()")

    assert src.count("np.savez_compressed") == 1, (
        "exactly one savez, inside flush(); a second one at the end rebuilds shard 0 "
        "from the now-empty lists and overwrites real data with an empty file")
    assert "> 400e6" in src, "no size-triggered flush - a crash loses everything since"
    assert src.count("flush()") >= 3, (
        "flush must be called on the size trigger AND at every corpus boundary, so a "
        "crash in corpus N+1 cannot cost corpus N")
    assert "MAX_SAMPLED" in src, "no frame cap - a malformed container can exhaust RAM"
    assert "read-error" in src, "no per-video guard - one bad clip ends the corpus"

    # The gate must read shards from DISK: flushing empties the in-memory lists, so a
    # gate checking `skels` would assert "NO windows" on a perfectly good run. It shares
    # the extraction cell, so `src` is the right place to look.
    assert 'OUT.glob("falls_*.npz")' in src, "gate still reads the emptied in-memory list"

    # No window may be appended twice: exactly one append site per field.
    assert src.count("skels.append(") == 1, (
        f"{src.count('skels.append(')} append sites - the superseded positional block is "
        "still reachable and every window lands twice")
    assert "nearest = min(range(len(fracs))" not in src, (
        "the old index-bucket labelling block survived the rewrite")

    # Le2i's root must come from the annotations, not the mount. Kaggle nests datasets at
    # /kaggle/input/datasets/<owner>/<name>, and accepting the mount made every Le2i
    # subject id the OWNER name - 190 videos collapsed onto one id.
    acq = _cell_containing("02_prepare_fall_shards_online.ipynb", "FAILED[")
    assert 'glob("**/Annotation_files")' in acq, "Le2i root is not anchored on annotations"
    # The root must span EVERY scene. Taking the first annotation folder's grandparent
    # and breaking picked up Coffee_room_01 alone - 48 of ~190 videos - and because
    # subject ids are derived relative to the root, all 48 collapsed onto one id.
    assert "commonpath" in acq, (
        "Le2i root is the first scene, not the common ancestor of all scenes")
    assert "vids[::step]" in acq, (
        "annotation sampling is contiguous; a head sample sits inside one scene and "
        "says nothing about the rest of the corpus")
    # CAUCAFall's Mendeley DOI publishes figures only - no video in any reachable
    # version - so it must come from a mount, not a download.
    assert "data.mendeley.com" not in acq, (
        "CAUCAFall is back on the Mendeley API, which serves 9 documentation files and "
        "no video; versions 1-3 are 451 and the S3 zip is 403")
    assert "tuyenldvn/caucafall" in acq, "CAUCAFall mount is not named in the failure hint"
    assert "le2i_fall_frames(v) is not None for v in sample" in acq, (
        "Le2i mount accepted on shape alone; annotations must actually resolve")
    assert "_UA" in acq and "User-Agent" in acq, (
        "Mendeley 403s urllib's default UA from Kaggle - CAUCAFall needs a browser UA")
    print("  N17 nb02 flushes on size and at corpus boundaries, caps runaway decodes, "
          "isolates bad clips, gates from disk, and appends each window exactly once")



def test_n18_fall_head_trains_against_realistic_class_balance():
    """The fall shards alone are 82% POSITIVE, which breaks two things silently.

    `train_fall.py` derives its target from labels 7/8, so ADL windows are valid
    negatives - but notebook 03 passed only `*FALL`, leaving 563 negatives against 2501
    positives. Consequences, both measured rather than argued:

      * focal `alpha=0.75` up-weights the POSITIVE class, so the majority got 3x the
        minority - the inverse of what focal loss is for, and invisible in the loss curve.
      * the operating point is quoted per HOUR of monitored video. 563 negatives is 0.31 h
        (0.08 h after the val split), so a "1 FA/hour" budget permits 0.08 of one alarm
        and the fitted threshold collapses to whatever admits zero. On a synthetic detector
        of FIXED quality, sensitivity read 0.65 at 141 negatives and 0.37 at a realistic
        mix; AUPRC 0.993 vs 0.790. Same model, three different stories.
    """
    # Needle must be the INVOCATION, not the filename: the resolver cell now names
    # train_fall.py in the repo contract too, so a bare filename matches two cells.
    cell = _cell_containing("03_train_blackwell_offline.ipynb", "FALL_TRAIN")
    assert "*FALL, *ADL" in cell or "FALL_TRAIN" in cell, (
        "notebook 03 still trains the fall head on fall shards only; at 82% positive the "
        "operating point and AUPRC are artefacts of shard composition")
    assert 'assert ADL' in cell, (
        "missing ADL shards must fail loudly - silently training on 82% positives "
        "produces a plausible-looking number that means nothing")

    # The guard in train_fall.py must refuse the inverted mix outright.
    src = (ROOT / "scripts" / "train_fall.py").read_text(encoding="utf-8")
    assert "pos_rate > 0.5 and args.focal_alpha > 0.5" in src, (
        "train_fall.py accepts an alpha that up-weights the majority class")
    assert "val_hours * args.fa_budget < 5" in src, (
        "no warning when the val set holds too few negatives to express the FA budget")

    # And it must actually fire: 80% positive refuses, 3% positive proceeds.
    import subprocess
    import tempfile
    tmp = Path(tempfile.mkdtemp())

    def shard(path, n, pos_frac):
        rng = np.random.default_rng(0)
        np.savez_compressed(
            path,
            skeletons=rng.normal(0, 0.4, (n, 30, 2, 17, 3)).astype(np.float16),
            labels=np.where(rng.random(n) < pos_frac, 7, 19).astype(np.int64),
            subjects=np.array([f"s{i % 8:02d}" for i in range(n)], dtype="<U32"),
            datasets=np.array(["t"] * n, dtype="<U32"))
        return str(path)

    def run(path):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train_fall.py"), "--shards", path,
             "--epochs", "1", "--device", "cpu", "--workers", "0"],
            capture_output=True, text=True, env=SUBPROC_ENV)

    bad = run(shard(tmp / "pos.npz", 400, 0.82))
    assert bad.returncode != 0, "82% positive data was accepted with alpha=0.75"
    assert "POSITIVE" in bad.stdout + bad.stderr, "refusal does not explain the imbalance"
    good = run(shard(tmp / "neg.npz", 1200, 0.03))
    assert good.returncode == 0, f"realistic mix was refused: {good.stderr[-400:]}"
    print("  N18 fall head gets ADL negatives; train_fall refuses alpha that up-weights "
          "the majority (verified both branches) and warns on a too-small FA budget")


def test_n3_notebooks_only_pass_flags_the_scripts_accept():
    """A stale --flag is a 1-second failure that wastes a session's setup."""
    import re

    scripts = {
        "train_adl.py": ROOT / "scripts/train_adl.py",
        "train_fall.py": ROOT / "scripts/train_fall.py",
        "eval_hallucination.py": ROOT / "scripts/eval_hallucination.py",
        "kaggle_smoke_test.py": ROOT / "scripts/kaggle_smoke_test.py",
        "build_charades_map.py": ROOT / "scripts/build_charades_map.py",
    }
    accepted = {
        name: set(re.findall(r'add_argument\("(--[a-z-]+)"', p.read_text(encoding="utf-8")))
        for name, p in scripts.items()
    }
    checked = 0
    for nb_path in sorted(NOTEBOOKS.glob("*.ipynb")):
        doc = json.loads(nb_path.read_text(encoding="utf-8"))
        for cell in doc["cells"]:
            if cell["cell_type"] != "code":
                continue
            # Scan the actual argv lists, not the whole cell. The resolver cell names
            # scripts and flags as CONTRACT data, and a whole-cell scan misreads those as
            # invocations - it reported notebook 01 passing --profile to train_adl.py,
            # which appears nowhere in that notebook.
            for m in re.finditer(r'\[\s*(?:sys\.executable|"python")[^\]]*\]',
                                 cell["source"], re.S):
                argv = m.group(0)
                for script, flags in accepted.items():
                    if script not in argv:
                        continue
                    used = set(re.findall(r'"(--[a-z-]+)"', argv))
                    unknown = used - flags
                    assert not unknown, (
                        f"{nb_path.name} passes {sorted(unknown)} to {script}, which "
                        f"accepts {sorted(flags)}"
                    )
                    checked += 1
    assert checked >= 4, f"only {checked} script invocations checked - patterns drifted?"
    print(f"  N3 {checked} script invocations across the notebooks use only real flags")



def test_n15_notebook_02_fall_labels_track_true_time_not_list_index():
    """The fall head's ONLY label source. Two defects, both found by measurement.

    window_clip DROPS low-visibility windows, so a caller deriving temporal position from
    enumerate() is wrong by however many were dropped before it. Notebook 02 labels the
    descent as "mid-clip", and with 3 s of occlusion at the head of a 10 s clip the old
    index-based version put `falling` 2 windows (2 s) late - labelling the real descent
    `standing` and a standing window `falling`.

    Second defect: an index-bucket band [0.4, 0.6) leaves some clip lengths with NO
    `falling` window at all (5 s -> fracs {0, .33, .67, 1}), and short clips are exactly
    what fall corpora contain.
    """
    import re
    src = _cell_containing("02_prepare_fall_shards_online.ipynb", "label_fall_windows(")
    assert "with_starts=True" in src, (
        "notebook 02 no longer requests true frame positions; index-derived position "
        "mislabels the descent whenever window_clip drops a window")
    assert not re.search(r"for i, w in enumerate\(wins\)", src), (
        "notebook 02 is labelling by list index again - that is the bug N15 exists for")

    # Exercise the SHARED function, not a copy of it. This test used to re-implement the
    # mid-point maths inline, which meant it kept passing after the notebook switched to
    # label_fall_windows() - it was validating the test's own arithmetic.
    from prepare_skeletons import WINDOW_FRAMES, label_fall_windows, window_clip

    def label(poses):
        wins = window_clip(poses, with_starts=True)
        if not wins:
            return []
        starts = [st for st, _ in wins]
        return list(zip(starts, label_fall_windows(starts, WINDOW_FRAMES, None)))

    def clip(T, occlude=None):
        p = np.zeros((T, 2, 17, 3), np.float32)
        p[..., 2] = 0.9
        if occlude:
            p[occlude[0]:occlude[1], 0, :, 2] = 0.0
        return p

    # A: an occluded head must not shift the descent label.
    T, mid = 150, 75
    occ = label(clip(T, occlude=(0, 45)))
    falling = [st for st, lab in occ if lab == 7]
    assert falling, "occluded fall clip produced no `falling` window"
    err = min(abs(st + WINDOW_FRAMES / 2 - mid) for st in falling)
    assert err <= WINDOW_FRAMES, (
        f"descent labelled {err} frames ({err/15:.1f} s) from the true midpoint - "
        f"position is being derived from the surviving-window index again")

    # B: no clip length may yield zero `falling` windows.
    empty = [secs for secs in range(3, 16)
             if (rows := label(clip(secs * 15))) and not any(l == 7 for _, l in rows)]
    assert not empty, f"clip lengths yielding NO `falling` window: {empty} s"

    # C: the shard gate must refuse to publish a degenerate single-class corpus.
    gate = _cell_containing("02_prepare_fall_shards_online.ipynb", "np.savez_compressed")
    # "assert skels" was the pre-flush check. Extraction now banks shards incrementally
    # and clears those lists by design, so the gate reads the written shards instead -
    # asserting on `skels` here would demand the very bug the flush refactor removed.
    for needle in ("assert shards", "ZERO fall windows", "ZERO non-fall windows"):
        assert needle in gate, f"notebook 02 lost its shard gate: {needle!r} missing"
    assert "assert skels," not in gate, (
        "gate is back to checking the in-memory list, which flush() empties - it would "
        "fire on a good run")
    assert "P2" in gate and "leave-one-dataset-out is NOT possible" in gate, (
        "notebook 02 must say so when only one corpus is present - P2 needs >= 2")
    print(f"  N15 fall labels track frame position (occluded descent within "
          f"{err:.0f} frames), every 3-15 s clip yields a `falling` window, shard gate "
          f"refuses degenerate corpora")


def test_n18_decode_probe_isolates_a_native_crash():
    """A clip that segfaults ffmpeg must cost one clip, not the corpus.

    Notebook-02 sessions died mid-extraction with no Python traceback: the fault is in
    cv2 -> ffmpeg native code, and SIGABRT/SIGSEGV there takes the interpreter with it,
    so neither the per-video try/except nor the frame cap can see it. The decode
    therefore runs in a child that journals each attempt BEFORE trying it - a crash
    leaves its last TRY unmatched, naming the poison clip, and the parent restarts past
    it.

    The stub below is built from scratch rather than string-patched out of the real
    child. An earlier version patched it, the child grew a backend branch, the patch
    stopped matching, and the test errored instead of testing - which is exactly the
    failure mode it is supposed to catch elsewhere. Structural checks on the REAL child
    live in N18b.
    """
    stub = chr(10).join([
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
        "    f.write('TRY' + chr(9) + p + chr(10))",
        "    if backend == 'av':",
        "        if 'RESCUE' not in p:",
        "            os._exit(139)",
        "    else:",
        "        if 'POISON' in p or 'RESCUE' in p:",
        "            os._exit(134)",
        "    f.write('OK' + chr(9) + p + chr(9) + '7' + chr(10))",
    ])

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "child.py").write_text(stub, encoding="utf-8")
        vids = ["ok_0.mp4", "c_POISON_1.avi", "ok_2.mp4",
                "c_RESCUE_3.avi", "c_RESCUE_4.avi"]

        def sweep(names, listname, journalname, backend):
            (tmp / listname).write_text(chr(10).join(names) + chr(10), encoding="utf-8")
            for attempt in range(1, len(names) + 6):
                r = subprocess.run(
                    [sys.executable, str(tmp / "child.py"), str(tmp / listname),
                     str(tmp / journalname), backend], capture_output=True, text=True)
                if r.returncode == 0:
                    break
            text = (tmp / journalname).read_text(encoding="utf-8")
            ok = {ln.split(chr(9))[1] for ln in text.splitlines() if ln.startswith("OK")}
            return ok, attempt

        ok_cv2, runs_cv2 = sweep(vids, "l1.txt", "j1.tsv", "cv2")
        failed = sorted(set(vids) - ok_cv2)
        av_ok, _ = sweep(failed, "l2.txt", "j2.tsv", "av")

    usable = ok_cv2 | av_ok
    backend = {q: ("av" if q in av_ok else "cv2") for q in usable}
    poison = sorted(set(vids) - usable)

    assert sorted(ok_cv2) == ["ok_0.mp4", "ok_2.mp4"], sorted(ok_cv2)
    assert runs_cv2 == 4, f"cv2 sweep took {runs_cv2} runs, want 1 + 3 crashes"
    assert sorted(av_ok) == ["c_RESCUE_3.avi", "c_RESCUE_4.avi"], sorted(av_ok)
    assert poison == ["c_POISON_1.avi"], poison
    assert backend["ok_0.mp4"] == "cv2" and backend["c_RESCUE_3.avi"] == "av"
    print(f"  N18 decode probe: {len(usable)}/5 clips kept ({len(av_ok)} rescued by "
          f"PyAV), 1 true poison excluded, cv2 sweep converged in {runs_cv2} runs")


def test_n18b_probe_retries_cv2_failures_with_pyav():
    """cv2 aborting is not proof a clip is unreadable - only that ONE decoder failed.

    The Le2i mirror SIGABRTs/SIGSEGVs cv2 on all 190 clips, including the only corpus
    here with exact frame-level fall annotations. PyAV links its own ffmpeg, so it is a
    genuinely different decoder; writing those clips off without trying it would discard
    the best-annotated corpus on one library's say-so.

    Also guards the budget: a cap of 11 restarts was set for 'a couple of bad files' and
    ran out after 11, leaving 179 clips NEVER PROBED and silently excluded while the log
    said only 'continuing with what passed'.
    """
    cell = _cell_containing("02_prepare_fall_shards_online.ipynb", "decode_probe")
    head = cell.index("TAB, NL = ")
    ns: dict = {}
    exec(compile(cell[head:cell.index("(WORK /", head)], "<child>", "exec"), ns)
    child = ns["CHILD"]
    compile(child, "<real-child>", "exec")
    for need in ("backend = sys.argv[3]", "if backend == 'av':", "av.open(p)",
                 "import cv2", "buffering=1"):
        assert need in child, f"child probe lacks {need!r}"

    assert "budget = len(all_vids)" in cell, (
        "probe restart budget is fixed; one wholly-unreadable corpus exhausts it and the "
        "rest are never probed")
    assert "NEVER PROBED" in cell, (
        "never-probed clips are not reported separately from poison - a budget failure "
        "would read as if every clip had been tested")
    assert "PyAV rescued" in cell, "no PyAV second chance for cv2 failures"

    # The rescued clips must be decoded WITH PyAV downstream, in BGR - every other
    # corpus feeds RTMO BGR, and mixing colour order across corpora is a silent
    # distribution shift in the fall head's training data.
    ext = _cell_containing("02_prepare_fall_shards_online.ipynb",
                           "for source, root in SOURCES")
    assert "BACKEND" in ext, "extraction ignores which decoder cleared each clip"
    assert "bgr24" in ext, "PyAV path does not convert RGB->BGR"
    print("  N18b probe retries cv2 failures with PyAV, budgets one restart per clip, "
          "reports never-probed separately, and decodes rescued clips in BGR")


def test_n19_extraction_only_consumes_probe_cleared_clips():
    """The probe is pointless if extraction still globs the raw directory."""
    ext = _cell_containing("02_prepare_fall_shards_online.ipynb",
                           "for source, root in SOURCES")
    assert "USABLE" in ext, (
        "extraction does not filter on USABLE - the probe would name the poison clip "
        "and extraction would decode it anyway, killing the kernel")
    assert "probe-cleared" in ext, "extraction does not say its list is filtered"
    print("  N19 extraction consumes only probe-cleared clips")



def test_n14_session_summary_reads_real_metric_keys_and_refuses_to_print_nothing():
    """The summary cell reported `best=0.000` for four runs, and later printed NOTHING.

    Two defects, one lesson each:

      1. It looked for a key called `mca`; train_adl.py writes `mean_class_acc`. `max()`
         fell through to `default=0`, so four independent 80-epoch runs all reported
         exactly 0.000 - a value real training noise never produces. A metric read by the
         wrong name is worse than a missing metric: it looks like a result.
      2. Re-run in a FRESH session, /kaggle/working is empty, the glob yielded zero
         iterations, and the cell ended by inviting "Save Version" having printed no rows.
         Same silent-empty-iteration failure already fixed in notebooks 00/01/02.

    So: read the key the trainer actually writes, and assert something was found.
    """
    cell = _cell_containing("03_train_blackwell_offline.ipynb", "history.json")
    written = set(re.findall(r'"(\w+)":',
                  (ROOT / "scripts/train_adl.py").read_text(encoding="utf-8")
                  .split("def evaluate")[1].split("def make_smoke_shard")[0]))
    for key in re.findall(r'SELECT = \{[^}]*"adl": "(\w+)"', cell):
        assert key in written, (
            f"summary selects {key!r} for ADL runs but train_adl.py writes {sorted(written)}")

    sup = [int(n * .25) for n in
           [2524, 342, 14766, 3909, 1768, 1442, 69, 0, 0, 6294, 5457, 2823, 1838, 3713,
            8897, 11405, 16232, 15837, 2705, 65088]]
    f1 = [.11, 0, .28, .19, .05, .04, 0, 0, 0, .16, .14, .09, .07, .21, .24, .31, .26,
          .29, .06, .55]
    adl = [{"epoch": 79, "mean_class_acc": 0.19, "macro_f1": 0.138, "top1": 0.42,
            "per_class_f1": f1, "per_class_support": sup}]
    fall = [{"epoch": 56, "auprc": 0.822}, {"epoch": 59, "auprc": 0.796}]

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        work, inp = root / "working" / "runs", root / "input"
        patched = (cell.replace('pathlib.Path("/kaggle/working/runs")',
                                f'pathlib.Path(r"{work}")')
                       .replace('pathlib.Path("/kaggle/input")', f'pathlib.Path(r"{inp}")'))

        def run():
            import contextlib, io
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(patched, "<summary>", "exec"),
                         {"json": json, "pathlib": Path.__mro__[0].__module__ and
                          __import__("pathlib")})
                return True, buf.getvalue()
            except AssertionError as exc:
                return False, str(exc)

        def place(base):
            for st in ("joint", "bone", "joint_motion", "bone_motion"):
                q = base / f"adl_{st}" / "history.json"
                q.parent.mkdir(parents=True, exist_ok=True)
                q.write_text(json.dumps(adl), encoding="utf-8")
            q = base / "fall" / "history.json"
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_text(json.dumps(fall), encoding="utf-8")

        # 1. Nothing anywhere: must REFUSE rather than print a cheerful Save Version.
        work.mkdir(parents=True); inp.mkdir(parents=True)
        ok, msg = run()
        assert not ok and "no history.json" in msg, (
            f"empty session printed instead of refusing: {msg[:200]}")
        assert "Save Version" not in msg

        # 2. Same session as training.
        place(work)
        ok, txt = run()
        assert ok, txt
        assert "best=0.000" not in txt, "the wrong-metric-key bug is back"
        assert "best mean_class_acc=0.190" in txt and "best auprc=0.822 @ep56" in txt
        assert "per-class F1" in txt and "16/18 classes above F1 0.02" in txt

        # 3. Fresh session with the runs dataset attached, Save-Version-nested, AND the
        #    contaminated runs/ that ships inside behaviorsense-code. Observed live: the
        #    cell found 9 runs, reported a 25-epoch CPU smoke run as a result, then died
        #    on `smoke_fall` - a FALL run whose name does not start with "fall", so the
        #    name-prefix heuristic demanded mean_class_acc from an AUPRC-only history.
        shutil.rmtree(work); work.mkdir(parents=True)
        place(inp / "datasets" / "owner" / "behaviorsense-runs" / "kaggle" / "working" / "runs")
        code = inp / "datasets" / "owner" / "behaviorsense-code" / "EmotionSense-Extended" / "runs"
        for nm, h in (("adl", adl), ("fall", fall), ("smoke", adl), ("smoke_fall", fall)):
            q = code / nm / "history.json"
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_text(json.dumps(h), encoding="utf-8")
        ok, txt = run()
        assert ok, txt
        assert "reading 5 run(s)" in txt, (
            f"expected only the 5 real runs, got: {txt.splitlines()[0]}")
        head = txt.split("per-class")[0]
        assert "smoke" not in head, "a smoke run was reported as a result"
        assert "from attached dataset" in txt and "best auprc=0.822 @ep56" in txt

    print("  N14 summary cell: real metric keys, both locations, refuses empty output")

# ---------------------------------------------------------------------------
# E: the offline notebooks' actual logic
# ---------------------------------------------------------------------------


def test_n4_training_invocation_from_notebook_03_runs():
    """Run train_adl.py exactly as notebook 03 builds the command."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shard = make_shard(td / "shards/adl_0000.npz", n=96)
        out = td / "runs/adl_joint"
        cmd = [sys.executable, str(ROOT / "scripts/train_adl.py"),
               "--shards", str(shard), "--stream", "joint", "--epochs", "2",
               "--batch-size", "32", "--device", "cpu", "--seed", "0",
               "--workers", "0", "--warmup-epochs", "1", "--out", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                           env=SUBPROC_ENV)
        assert r.returncode == 0, f"notebook 03's command failed:\n{r.stdout[-1200:]}{r.stderr[-1200:]}"
        assert (out / "last.pt").exists() and (out / "history.json").exists()

        # The resume branch notebook 03 takes on a continuation session.
        r2 = subprocess.run(cmd + ["--resume", str(out / "last.pt")], capture_output=True,
                            text=True, timeout=1800,
                            env=SUBPROC_ENV)
        assert r2.returncode == 0, f"resume branch failed:\n{r2.stderr[-1000:]}"
        assert "resumed from" in r2.stdout
        hist = json.loads((out / "history.json").read_text())
        print(f"  N4 train_adl.py ran as notebook 03 invokes it ({len(hist)} epochs) "
              "and the resume branch works")


def test_n21_p1_evaluates_on_the_split_the_streams_were_stopped_against():
    """Notebook 04's val split must match `train_adl.py`'s, fraction and seed.

    They had drifted: training splits at `val_frac=0.2`, P1 evaluated at `0.15`. Because
    `split_by_subject` shuffles subjects by seed and takes a PREFIX, the 15% set is a strict
    subset of the 20% one, so no training subject ever leaked - but the notebook's comment
    claimed to "reproduce the training-time val split" and did not, and the mismatch quietly
    discarded a fifth of the evaluation evidence. Two numbers meant to describe the same set
    must be pinned to each other, not maintained in parallel.
    """
    train = (ROOT / "scripts/train_adl.py").read_text(encoding="utf-8")
    m = re.search(r"split_by_subject\(\s*split_subjects,\s*val_frac=([\d.]+),\s*seed=([\w.]+)",
                  train)
    assert m, "could not find train_adl.py's split call"
    train_frac = float(m.group(1))

    doc = json.loads((NOTEBOOKS / "04_evaluate_blackwell_offline.ipynb")
                     .read_text(encoding="utf-8"))
    cells = [c["source"] for c in doc["cells"] if c["cell_type"] == "code"]
    p1 = [c for c in cells if "split_by_subject(" in c and "P1 val:" in c]
    assert len(p1) == 1, f"expected one P1 cell using split_by_subject, found {len(p1)}"
    n = re.search(r"split_by_subject\(split_subjects,\s*val_frac=([\d.]+),\s*seed=(\d+)\)", p1[0])
    assert n, "could not parse notebook 04's split call"
    nb_frac, nb_seed = float(n.group(1)), int(n.group(2))

    # Both sides must derive the split identity the same way: remap to actor ids when the
    # Charades CSV is available, video ids otherwise. One side remapping without the other
    # evaluates checkpoints on a boundary they were not stopped against.
    assert "remap_subjects" in p1[0], "notebook 04 no longer remaps to actor ids"
    assert "remap_subjects" in train, "train_adl.py no longer supports the actor remap"

    assert nb_frac == train_frac, (
        f"notebook 04 evaluates on a {nb_frac:.0%} split while train_adl.py stops on "
        f"{train_frac:.0%}; P1 is then not measured on the set that selected best.pt"
    )
    assert nb_seed == 0, f"notebook 04 uses seed={nb_seed}; train_adl.py defaults to 0"

    # And the nesting property the old code silently relied on must still hold, so the
    # historical 15% numbers remain interpretable as a subset rather than a re-split.
    subjects = np.array([f"v{i:04d}" for i in range(500)]).repeat(3)
    _, small = split_by_subject(subjects, val_frac=0.15, seed=0)
    _, large = split_by_subject(subjects, val_frac=0.20, seed=0)
    assert set(subjects[small]) <= set(subjects[large]), (
        "a smaller val_frac is no longer a subset of a larger one at the same seed"
    )
    print(f"  N21 P1 and train_adl.py both split at val_frac={train_frac} seed=0; "
          f"smaller fractions still nest ({len(set(subjects[small]))} inside "
          f"{len(set(subjects[large]))} subjects)")


def test_n5_notebook_04_p1_evaluation_logic_runs():
    """The exact P1 block from notebook 04: split, build windows, score.

    This is the test that caught the [C,T,V,M] vs [N,T,M,17,3] layout bug.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shards = [str(make_shard(td / f"shards/adl_{i:04d}.npz", n=80, seed=i))
                  for i in range(2)]
        for i, s in enumerate(("joint", "bone")):
            save_ckpt(td / f"runs/adl_{s}/best.pt", s, seed=i)

        ds = SkeletonWindowDataset(shards)
        _, val_idx = split_by_subject(ds.subjects, val_frac=0.15, seed=0)
        assert len(val_idx) > 0
        assert not (set(ds.subjects[val_idx]) & set(np.delete(ds.subjects, val_idx))), \
            "subject leakage - the notebook's own assertion would fire"

        clf = EnsembleClassifier.from_run_dir(td / "runs", device="cpu")
        X = np.stack([normalise(ds.skeletons[i].astype(np.float32)) for i in val_idx])
        y = ds.labels[val_idx]
        assert X.shape[1:] == (30, 2, 17, 3), X.shape

        def scores(logits, y):
            pred = logits.argmax(1)
            per_class = [np.mean(pred[y == c] == c)
                         for c in range(N_CLASSES) if (y == c).any()]
            f1 = []
            for c in range(N_CLASSES):
                tp = ((pred == c) & (y == c)).sum()
                fp = ((pred == c) & (y != c)).sum()
                fn = ((pred != c) & (y == c)).sum()
                if tp + fp + fn:
                    f1.append(2 * tp / max(2 * tp + fp + fn, 1))
            return np.mean(pred == y), np.mean(per_class), np.mean(f1)

        per_stream = clf.per_stream_logits(X)
        assert set(per_stream) == {"joint", "bone"}
        ens = clf.logits(X)
        t1, mca, f1 = scores(ens, y)
        assert 0.0 <= t1 <= 1.0 and 0.0 <= mca <= 1.0 and 0.0 <= f1 <= 1.0

        # Calibration block.
        from behaviorsense.agents.activity import fit_temperature, softmax
        T = fit_temperature(ens, y)
        assert 0.2 < T < 6.0, T
        conf = softmax(ens / T).max(1).mean()
        print(f"  N5 P1 block ran on {len(val_idx)} windows: top1={t1:.3f} mca={mca:.3f} "
              f"f1={f1:.3f}, fitted T={T:.2f} (conf {conf:.3f})")


def test_n6_notebook_04_p2_leave_one_dataset_out_logic_runs():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rng = np.random.default_rng(0)
        n = 60
        sk = rng.normal(0, 0.4, (n, 30, 2, 17, 3)).astype(np.float16)
        sk[..., 2] = 0.9
        labels = np.where(np.arange(n) % 3 == 0, 7, 19).astype(np.int64)
        sources = np.array(["le2i"] * 30 + ["urfd"] * 30, dtype="<U32")
        p = td / "shards/falls_0000.npz"
        p.parent.mkdir(parents=True)
        np.savez_compressed(p, skeletons=sk, labels=labels,
                            subjects=np.array([f"s{i%5}" for i in range(n)], dtype="<U32"),
                            datasets=sources)
        save_ckpt(td / "runs/adl_joint/best.pt", "joint")

        from behaviorsense.agents.activity import softmax
        fds = SkeletonWindowDataset([str(p)])
        clf = EnsembleClassifier.from_run_dir(td / "runs", device="cpu")
        srcs = np.asarray(fds.datasets)
        is_fall = np.isin(fds.labels, FALL_CLASSES)
        assert set(srcs) == {"le2i", "urfd"}

        rows = []
        for held in sorted(set(srcs)):
            idx = np.where(srcs == held)[0]
            Xh = np.stack([normalise(fds.skeletons[i].astype(np.float32)) for i in idx])
            post = softmax(clf.logits(Xh))
            score = post[:, list(FALL_CLASSES)].sum(1)
            yh = is_fall[idx]
            order = np.argsort(score)
            ranks = np.empty(len(score))
            ranks[order] = np.arange(1, len(score) + 1)
            auroc = ((ranks[yh].sum() - yh.sum() * (yh.sum() + 1) / 2)
                     / (yh.sum() * (~yh).sum()))
            assert 0.0 <= auroc <= 1.0, auroc
            rows.append((held, len(idx), auroc))
        print("  N6 P2 block ran: " + ", ".join(f"{h} n={n} auroc={a:.3f}"
                                                for h, n, a in rows))


def test_n6b_every_logits_call_site_builds_the_shard_layout():
    """P1 was fixed for the [C,T,V,M] layout bug; P2 kept it and died 11 minutes in.

    `ds[i]` returns [C,T,V,M] - already permuted for the model - while
    `EnsembleClassifier.logits()` wants the SHARD layout [N,T,M,17,3]. N5 caught this in
    P1 by re-implementing that block in Python, which is exactly why it missed P2: a test
    that copies the logic tests the copy, not the notebook.

    So check the notebook SOURCE - every tensor handed to logits()/per_stream_logits()
    must be built from `.skeletons[...]`, never from `ds[i][0]` - and then execute the
    real shapes once to prove the layout that survives is the one logits() accepts.
    """
    doc = json.loads(
        (NOTEBOOKS / "04_evaluate_blackwell_offline.ipynb").read_text(encoding="utf-8"))
    code = [c["source"] for c in doc["cells"] if c["cell_type"] == "code"]

    call_sites = 0
    for cell in code:
        for var in re.findall(r"(?:per_stream_)?logits\((\w+)\)", cell):
            call_sites += 1
            build = re.search(rf"{var}\s*=\s*np\.stack\(\[([^\]]+)", cell)
            if build is None:
                continue          # e.g. logits(X) reusing a tensor built in an earlier cell
            expr = build.group(1)
            assert ".skeletons[" in expr, (
                f"{var} is built from {expr.strip()[:60]!r}; logits() needs the shard "
                "layout [N,T,M,17,3], and ds[i][0] gives [C,T,V,M]")
    assert call_sites >= 3, f"only {call_sites} logits call sites found - pattern drifted?"

    # And prove the two layouts really are different, so the check above is not cosmetic.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shard = make_shard(td / "shards/fall_0000.npz", n=40, seed=7)
        for i, stream in enumerate(("joint", "bone")):
            save_ckpt(td / f"runs/adl_{stream}/best.pt", stream, seed=i)
        ds = SkeletonWindowDataset([str(shard)])

        good = np.stack([normalise(ds.skeletons[i].astype(np.float32)) for i in range(8)])
        bad = np.stack([ds[i][0] for i in range(8)])
        assert good.shape[1:] == (30, 2, 17, 3), good.shape
        assert bad.shape[1:] == (3, 30, 17, 2), bad.shape

        clf = EnsembleClassifier.from_run_dir(td / "runs", device="cpu")
        out = clf.logits(good)
        assert out.shape == (8, N_CLASSES), out.shape
        try:
            clf.logits(bad)
            raise AssertionError("logits() accepted [C,T,V,M] - the guard is gone")
        except ValueError as exc:
            assert "N,T,M,17,3" in str(exc)

    print(f"  N6b {call_sites} logits call sites all build from .skeletons; "
          "the [C,T,V,M] layout still raises")


def test_n7_hallucination_eval_runs_as_notebook_04_invokes_it():
    """The stub arm of the same script notebook 04 runs with --backend qwen."""
    with tempfile.TemporaryDirectory() as td:
        report = Path(td) / "hallucination.md"
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts/eval_hallucination.py"),
             "--days", "30", "--scenarios", "2", "--report", str(report)],
            capture_output=True, text=True, timeout=1800,
            env=SUBPROC_ENV)
        assert r.returncode == 0, r.stderr[-1000:]
        text = report.read_text(encoding="utf-8")
        assert "hallucination rate" in text and "faithful stub" in text
        assert "**0.0%**" in text, "faithful anchor is not 0% - verifier false positives"
        print(f"  N7 eval_hallucination.py ran and wrote a {len(text.splitlines())}-line "
              "report with the 0% faithful anchor intact")


def test_n20_offline_notebooks_put_src_on_sys_path_unconditionally():
    """Every in-process `import behaviorsense` must be reachable, in every code path.

    The failure this prevents, verbatim from a Kaggle log: `ModuleNotFoundError: No module
    named 'behaviorsense'` at cell 4 of notebook 03, six minutes in, on the line immediately
    after PREFLIGHT PASSED. `sys.path` had been set at the end of the wheel-install cell.
    Notebook 04 placed that line outside the cell's `if not sm120_ok()` branch and worked;
    notebook 03 never had it at all and still worked, because every heavy step there is a
    subprocess launched with PYTHONPATH set. Adding `is_real_artifact` to notebook 03's
    shard resolver introduced its first in-process import and the next run died.

    Two assertions, because either alone is satisfiable by a broken notebook: the resolver
    cell must extend sys.path at TOP LEVEL (not inside a conditional), and no cell before it
    may import the package.
    """
    checked = 0
    for name in ("01_prepare_adl_shards_offline", "03_train_blackwell_offline",
                 "04_evaluate_blackwell_offline"):
        doc = json.loads((NOTEBOOKS / f"{name}.ipynb").read_text(encoding="utf-8"))
        code = [c["source"] for c in doc["cells"] if c["cell_type"] == "code"]
        resolver = next((i for i, s in enumerate(code) if "sys.path.insert" in s), None)
        assert resolver is not None, f"{name}: no cell extends sys.path at all"

        # Top level, not nested. A conditional insert is how notebook 03 broke.
        tree = ast.parse(code[resolver])
        top_level = {getattr(n, "lineno", 0) for n in tree.body}
        inserts = [n for n in ast.walk(tree)
                   if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and n.func.attr == "insert"
                   and isinstance(n.func.value, ast.Attribute)
                   and n.func.value.attr == "path"]
        assert inserts, f"{name}: sys.path.insert present as text but not as a call"
        guarded = [n for n in ast.walk(tree)
                   if isinstance(n, (ast.If, ast.Try, ast.While))
                   for c in ast.walk(n) if c in inserts]
        assert not guarded, (
            f"{name}: sys.path.insert is inside an if/try - it will be skipped on whichever "
            "branch the container happens to take, which is exactly how notebook 03 lost it"
        )
        # A `for` loop over the paths is fine; what matters is that it is unconditional.
        assert any(ln in top_level for ln in
                   {getattr(n, "lineno", -1) for n in ast.walk(tree)} & top_level), name

        earlier = "\n".join(code[:resolver])
        for pat in ("import behaviorsense", "from behaviorsense"):
            assert pat not in earlier, (
                f"{name}: cell before the resolver already does `{pat}`, so the import "
                "runs before sys.path is extended"
            )
        checked += 1

    print(f"  N20 {checked} offline notebooks extend sys.path unconditionally in their "
          "resolver cell, with no earlier behaviorsense import")


def test_n8_charades_map_builder_and_its_fallback_audit():
    """The builder as notebook 01 calls it, plus the audit that answers `walking`.

    N8's body was silently absorbed into N20 by an earlier edit that consumed the `def`
    line - the assertions still ran, under the wrong name, and the suite count was off by
    one. Restored as its own test, and extended.

    `walking` came out of the real extraction with 545 of 35,698 val windows (1.5%) at
    F1 0.093 - implausible for the most pose-separable activity in a corpus of people moving
    around their homes. The suspicion was that locomotion classes fall through to
    `other_idle`, and nothing in the builder's output could confirm it. The audit groups the
    fallback by candidate theme so a misrouted FAMILY is visible, and it is diagnostic only:
    it must never assign a label, because a wrong rule teaches a wrong label and then costs
    a ~6-hour re-extraction to undo.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        classes = td / "Charades_v1_classes.txt"
        classes.write_text(
            "c093 Walking through a doorway\nc108 Sitting in a chair\n"
            "c063 Eating a sandwich\nc141 Taking a picture\nc096 Holding a pillow\n"
            # The locomotion family the rules currently miss - the point of the audit.
            "c010 Going to a room\nc011 Leaving a room\n"
            "c012 Entering a room through a doorway\nc013 Going upstairs\n"
            "c014 Running somewhere\n",
            encoding="utf-8")
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts/build_charades_map.py"),
             "--classes", str(classes), "--out", str(td / "map.yaml"),
             "--review-tsv", str(td / "review.tsv")],
            capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-800:]
        import yaml
        mapping = yaml.safe_load((td / "map.yaml").read_text())["charades"]
        assert any(v == "walking" for v in mapping.values())
        assert any(isinstance(v, dict) and v.get("drop") for v in mapping.values())
        assert (td / "review.tsv").exists()

        out = r.stdout
        assert "fallback grouped by candidate theme" in out, "the audit did not run"
        assert "locomotion?" in out, "the locomotion probe is gone"
        # A probe that silently matches nothing is the vacuous-pass pattern again.
        loco = int(out.split("locomotion?", 1)[1].split("class(es)")[0].strip())
        assert loco >= 3, f"locomotion probe found {loco} classes, expected >= 3"
        for name in ("Going to a room", "Leaving a room", "Going upstairs"):
            assert name in out, f"audit did not surface {name!r}"
        # Diagnostic ONLY - the probed classes must still be unlabelled, not auto-mapped.
        for slug, target in mapping.items():
            if slug.startswith(("c010", "c011", "c012", "c013")):
                assert target == "other_idle", (
                    f"{slug} was auto-assigned {target!r}; the audit must diagnose, never "
                    "assign - a wrong rule costs a 6-hour re-extraction to undo"
                )
        assert "RE-EXTRACTING" in out, "the audit no longer states the cost of a rule change"
        print(f"  N8 builder produced {len(mapping)} mappings + review TSV; fallback audit "
              f"flagged {loco} locomotion class(es) starving `walking`, assigned none")


def test_n9_notebook_00_platform_ladder_matches_the_verifier():
    """The staging cells and scripts/verify_wheel_resolution.py must agree.

    The verifier proves, against the live cu128 index, that every pinned nvidia-*/triton
    wheel has a tag notebook 00 will accept. That proof is about *its* PLATFORMS list. If
    the notebook's list drifts from it, the verifier keeps printing PASS while the notebook
    stages a set that cannot install -- a green check for a list nobody runs. So compare
    them directly, and pin the one tag whose absence caused the failure.
    """
    import ast
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "vwr", ROOT / "scripts" / "verify_wheel_resolution.py")
    vwr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vwr)

    nb = json.loads((NOTEBOOKS / "00_stage_assets_online.ipynb").read_text(encoding="utf-8"))
    found: list[list[str]] = []
    pins: list[str] = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse(cell["source"])
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if target.id == "PLATFORMS":
                # Execute only this assignment, never the cell (the cells shell out to pip).
                ns: dict = {}
                exec(compile(ast.Module(body=[node], type_ignores=[]), "<nb00>", "exec"), ns)
                found.append(ns["PLATFORMS"])
            elif target.id == "TORCH_PIN":
                pins.append(ast.literal_eval(node.value))

    assert found, "notebook 00 defines no PLATFORMS -- staging is back to pip's defaults"
    assert all(p == vwr.PLATFORMS for p in found), (
        f"notebook 00's PLATFORMS differs from the verifier's; "
        f"nb has {len(found[0])} entries, verifier has {len(vwr.PLATFORMS)}")

    # The exact regression: cudnn 9.7.1.26 is manylinux_2_27-only, and its absence from the
    # list is what emptied the candidate set and produced ResolutionImpossible.
    assert "manylinux_2_27_x86_64" in vwr.PLATFORMS, (
        "manylinux_2_27_x86_64 dropped from the ladder -- nvidia-cudnn-cu12==9.7.1.26 "
        "publishes no other tag, so this is the exact failure that cost a Kaggle session")

    assert pins, "notebook 00 no longer pins torch -- unpinned, pip backtracks 2.7 -> 2.11"
    assert all(p == pins[0] for p in pins), f"notebook 00 pins torch inconsistently: {pins}"
    assert pins[0] == f"torch=={vwr.TORCH_VERSION}", (
        f"notebook pins {pins[0]} but the verifier checked torch=={vwr.TORCH_VERSION}")

    print(f"  N9 {len(found)} staging cells agree on a {len(vwr.PLATFORMS)}-tag ladder "
          f"(manylinux_2_27 present) and pin {pins[0]}, matching the verifier")


def test_n10_notebook_00_staging_verifier_catches_a_broken_wheel_set():
    """Run notebook 00's own preflight cell against fabricated wheel directories.

    That cell is the last thing standing between a bad staging run and a wasted offline
    session, and its assertions had never executed. Each rejection case below is paired
    with the passing set, so a cell that simply always raised would fail this test too.
    """
    import tempfile

    nb = json.loads((NOTEBOOKS / "00_stage_assets_online.ipynb").read_text(encoding="utf-8"))
    cells = [c["source"] for c in nb["cells"]
             if c["cell_type"] == "code" and "WHEELS_LOCK.txt" in c["source"]]
    assert len(cells) == 1, f"expected exactly one staging-verifier cell, found {len(cells)}"
    source = cells[0]

    good = [
        "torch-2.7.1+cu128-cp311-cp311-manylinux_2_28_x86_64.whl",
        "torch-2.7.1+cu128-cp312-cp312-manylinux_2_28_x86_64.whl",
        "nvidia_cudnn_cu12-9.7.1.26-py3-none-manylinux_2_27_x86_64.whl",
        "nvidia_cuda_nvrtc_cu12-12.8.61-py3-none-manylinux2010_x86_64.whl",
        "outlines-1.3.2-py3-none-any.whl",
        "outlines_core-0.2.14-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "numpy-2.5.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
    ]

    def run(wheel_names: list[str]) -> tuple[bool, str, Path]:
        td = Path(tempfile.mkdtemp())
        for name in wheel_names:
            (td / name).write_bytes(b"stub")
        # A plain replacement string would treat the Windows path's backslashes as regex
        # template escapes ("bad escape \\U"), so replace via a callable.
        patched = re.sub(r'WHEELS = pathlib\.Path\("[^"]+"\)',
                         lambda _m: f"WHEELS = pathlib.Path({td.as_posix()!r})", source)
        assert "WHEELS = pathlib.Path(" in patched and str(td.name) in patched, (
            "could not redirect the wheels path")
        try:
            exec(compile(patched, "<nb00-verify>", "exec"), {"__name__": "nb00"})
            return True, "", td
        except AssertionError as exc:
            return False, str(exc), td

    ok, err, td = run(good)
    assert ok, f"the passing set was rejected: {err}"
    lock = (td / "WHEELS_LOCK.txt").read_text(encoding="utf-8").splitlines()
    assert "torch==2.7.1+cu128" in lock, f"lock file lost torch's local version: {lock}"
    assert "outlines==1.3.2" in lock, f"lock file lost outlines: {lock}"
    assert "nvidia_cuda_nvrtc_cu12==12.8.61" in lock, (
        f"lock regex mishandles digit-leading versions after underscores: {lock}")
    assert len(lock) == 6, f"expected 6 distinct packages from 7 wheels, got {len(lock)}: {lock}"

    rejects = {
        "no cp311 torch": [w for w in good if "cp311" not in w],
        "no cudnn": [w for w in good if "cudnn" not in w],
        "no outlines": [w for w in good if not w.startswith("outlines-")],
        "outlines 0.x": [("outlines-0.1.14-py3-none-any.whl" if w.startswith("outlines-")
                          else w) for w in good],
    }
    for label, names in rejects.items():
        ok, err, _ = run(names)
        assert not ok, f"{label}: the verifier accepted a set it should have rejected"
        print(f"  N10 rejected {label:<16} -> {err[:64]}")

    print(f"  N10 accepted the good set, wrote a {len(lock)}-package lock, "
          f"rejected {len(rejects)} broken sets")



def test_n10b_shard_selection_survives_a_nested_mount():
    """Attaching a dataset by URL nests it; the top level then holds only 'datasets'.

    Observed: `attached ['competitions', 'datasets']` with all four datasets correctly
    attached, and shard_paths() - which iterated INPUT.iterdir() and matched the mount
    NAME - compared "adl"/"fall" against those two container directories, returned
    nothing, and aborted the session on `no ADL shards found`. The resolver cell in the
    same notebook was immune because it globs with **/; this cell was not.

    Both layouts must work, and the behaviour/behavior spelling must stay tolerated.
    """
    import contextlib
    import io
    import shutil
    import tempfile

    NL = chr(10)
    for nb_name in ("03_train_blackwell_offline.ipynb",
                    "04_evaluate_blackwell_offline.ipynb"):
        doc = json.loads((NOTEBOOKS / nb_name).read_text(encoding="utf-8"))
        cell = next(c["source"] for c in doc["cells"]
                    if c["cell_type"] == "code" and "def shard_paths" in c["source"])
        # Check the CODE, not the prose - the fix's own comment names the old call.
        code_only = NL.join(L for L in cell.splitlines()
                            if not L.lstrip().startswith("#"))
        assert "INPUT.iterdir()" not in code_only, (
            f"{nb_name} selects shards by iterating the top level of /kaggle/input, "
            "which holds only 'datasets' when a dataset is attached by URL")
        assert 'INPUT.glob("**/*.npz")' in cell, (
            f"{nb_name} must find .npz at ANY depth")

    def make_shard(path, labels):
        path.parent.mkdir(parents=True, exist_ok=True)
        n = len(labels)
        np.savez_compressed(
            path,
            skeletons=np.zeros((n, 30, 2, 17, 3), np.float16),
            labels=np.array(labels, np.int64),
            subjects=np.array([f"s{i}" for i in range(n)], dtype="<U32"),
            datasets=np.array(["x"] * n, dtype="<U32"))

    cell = next(c["source"] for c in json.loads(
        (NOTEBOOKS / "03_train_blackwell_offline.ipynb").read_text(encoding="utf-8")
    )["cells"] if c["cell_type"] == "code" and "def shard_paths" in c["source"])

    for layout in ("flat", "nested"):
        root = Path(tempfile.mkdtemp())
        # `nested` is what Kaggle actually produces for a URL-attached dataset.
        base = root if layout == "flat" else root / "datasets" / "someowner"
        make_shard(base / "behaviorsense-adl-shards" / "c1.npz", [0, 1, 2] * 40)
        make_shard(base / "behaviorsense-adl-shards" / "c2.npz", [3, 4, 5] * 40)
        # Deliberately the OTHER spelling, to prove tolerance survived the rewrite.
        make_shard(base / "behavioursense-FALL-shards" / "l1.npz", [7, 8, 0] * 10)
        ns = {"pathlib": Path.__module__ and __import__("pathlib"), "np": np,
              "INPUT": root, "ATTACHED": sorted(q.name for q in root.iterdir())}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(cell, f"<{layout}>", "exec"), ns)
        assert len(ns["ADL"]) == 2, f"{layout}: {len(ns['ADL'])} ADL shards, want 2"
        assert len(ns["FALL"]) == 1, f"{layout}: {len(ns['FALL'])} fall shards, want 1"
        shutil.rmtree(root, ignore_errors=True)

    print("  N10b shard selection works at any mount depth in notebooks 03 and 04; "
          "behaviour/behavior spelling still tolerated")



def test_n10c_smoke_artefacts_never_reach_a_real_training_run():
    """The code dataset is an upload of the working tree, fixtures and all.

    Two leaked into a real session and neither announced itself:

      - `data/shards/_smoke_fall.npz` (400 synthetic windows from `train_fall.py
        --smoke`) matched the keyword "fall" and joined the real fall corpus.
      - `runs/adl/last.pt` (a 25-epoch CPU smoke run, `smoke=True`) was carried forward
        and resumed as though it were prior GPU work, putting an 80-epoch run on a
        25-epoch cosine schedule.

    `.kaggleignore` lists both directories, but the Kaggle CLI does not read that file -
    it is documentation, not enforcement. So the notebook has to defend itself.
    """
    import contextlib
    import io
    import shutil
    import tempfile

    import torch

    from behaviorsense.models.stgcnpp import STGCNpp

    root = Path(tempfile.mkdtemp())
    base = root / "datasets" / "someowner"

    def make_shard(path, labels):
        path.parent.mkdir(parents=True, exist_ok=True)
        n = len(labels)
        np.savez_compressed(
            path, skeletons=np.zeros((n, 30, 2, 17, 3), np.float16),
            labels=np.array(labels, np.int64),
            subjects=np.array([f"s{i % 4}" for i in range(n)], dtype="<U32"),
            datasets=np.array(["x"] * n, dtype="<U32"))

    def make_ckpt(path, **args):
        path.parent.mkdir(parents=True, exist_ok=True)
        m = STGCNpp(n_classes=N_CLASSES)
        torch.save({"model": m.state_dict(), "ema": m.state_dict(), "ema_step": 1,
                    "epoch": 24, "global_step": 100, "best": 0.1, "args": args,
                    "metrics": {}}, path)

    make_shard(base / "behaviorsense-adl-shards" / "charades_0000.npz", list(range(20)) * 50)
    make_shard(base / "behaviorsense-fall-shards" / "falls_0000.npz", [7, 8, 1, 19] * 50)
    # The fixture, exactly where the upload put it.
    code = base / "behaviorsense-code" / "EmotionSense-Extended"
    make_shard(code / "data" / "shards" / "_smoke_fall.npz", [7, 8, 0] * 10)
    make_ckpt(code / "runs" / "adl" / "last.pt", epochs=25, smoke=True, stream="joint")
    make_ckpt(code / "runs" / "fall" / "last.pt", epochs=1, smoke=False, stream="joint")
    # ...and one legitimate prior GPU run, which MUST still be picked up.
    make_ckpt(base / "behaviorsense-runs" / "adl_joint" / "last.pt",
              epochs=80, smoke=False, stream="joint")

    doc = json.loads(
        (NOTEBOOKS / "03_train_blackwell_offline.ipynb").read_text(encoding="utf-8"))
    cells = [c["source"] for c in doc["cells"] if c["cell_type"] == "code"]
    shard_cell = next(c for c in cells if "def shard_paths" in c)
    carry_cell = next(c for c in cells if "WANT_EPOCHS" in c)

    ns = {"pathlib": __import__("pathlib"), "np": np, "INPUT": root,
          "ATTACHED": sorted(q.name for q in root.iterdir())}

    # The shard cell calls `find_charades_csv()` for split-identity detection, but the
    # definition lives in the resolver cell (cell 0) - moving the lookup out of the shard
    # cell was the fix for a `**` glob that walked the code dataset's MSMT17 copy. Exec'ing
    # the whole resolver cell here would fail on its asset assertions, so lift ONLY the
    # statements the shard cell needs - the function def and the `_SLUGS` mount index it
    # reads - and run the real lookup against this fixture tree, where no charades mount
    # exists and it must return [] without touching the network.
    import ast as _ast
    resolver = cells[0]
    keep = []
    for node in _ast.parse(resolver).body:
        if isinstance(node, _ast.FunctionDef) and node.name == "find_charades_csv":
            keep.append(node)
        targets = {t.id for t in node.targets} if isinstance(node, _ast.Assign) else set()
        if "_SLUGS" in targets:
            keep.append(node)
    assert keep, "resolver cell no longer defines find_charades_csv; update this harness"
    mod = _ast.Module(body=keep, type_ignores=[])
    # The lifted `_SLUGS` statement reads INPUT at exec time, so seed the namespace with
    # the fixture's mount root; the update() below then carries both into ns and ns2.
    resolver_prelude: dict = {"INPUT": root, "pathlib": __import__("pathlib")}
    exec(compile(_ast.fix_missing_locations(mod), "<resolver-def>", "exec"), resolver_prelude)
    ns.update(resolver_prelude)

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(shard_cell, "<shards>", "exec"), ns)
    text = out.getvalue()

    assert len(ns["FALL"]) == 1, (
        f"the smoke fixture joined the fall corpus: {ns['FALL']}")
    assert not any("_smoke" in q for q in ns["ADL"] + ns["FALL"]), (
        "a synthetic fixture reached a real corpus")
    # The wording moved when the rule was centralised in kaggle_artifacts; what must
    # hold is that the exclusion is REPORTED, not that it uses a particular phrase.
    assert "skipping" in text.lower() and "_smoke_fall.npz" in text, (
        f"fixture exclusion is silent; it must name what it dropped. Got: {text[:200]}")

    # Carry-forward is EXECUTED, not string-matched: it must drop the fixtures while
    # keeping the real runs from the same dataset. Rejecting everything would be a
    # different bug wearing the same green tick, and the two live side by side here.
    #
    # The epoch budget is per head - ADL streams run 80, the fall head runs 60. A single
    # WANT_EPOCHS value refused every legitimate fall checkpoint with "epochs=60, this
    # run wants 80", breaking exactly the resume path the cell exists for.
    runs = base / "behaviorsense-runs"
    for stream in ("joint", "bone", "joint_motion", "bone_motion"):
        # Epochs must match what notebook 03 requests, parsed from EPOCH_BUDGET rather
        # than hard-coded here. With a stale 80 the resume guard correctly refuses these
        # as hyperparameter mismatches, and the test then measures the guard instead of
        # the fixture exclusion it is named for - which is exactly what happened when the
        # ADL budget moved 80 -> 30.
        make_ckpt(runs / f"adl_{stream}" / "last.pt", epochs=_nb_epoch_budget()["adl"],
                  smoke=False, stream=stream)
    make_ckpt(runs / "fall" / "last.pt", epochs=_nb_epoch_budget()["fall"],
              smoke=False, stream="joint")

    working = Path("/kaggle/working/runs")
    shutil.rmtree(working, ignore_errors=True)
    # The carry cell also consults find_charades_csv for the split-identity resume guard,
    # so it needs the same lifted resolver prelude as the shard cell.
    ns2 = {"pathlib": __import__("pathlib"), "INPUT": root, "shutil": shutil,
           **resolver_prelude}
    out2 = io.StringIO()
    with contextlib.redirect_stdout(out2):
        exec(compile(carry_cell, "<carry>", "exec"), ns2)
    carried = sorted(q.parent.name for q in working.glob("*/last.pt"))
    shutil.rmtree("/kaggle", ignore_errors=True)

    expect = ["adl_bone", "adl_bone_motion", "adl_joint", "adl_joint_motion", "fall"]
    assert carried == expect, f"expected {expect}, carried {carried}"
    assert "smoke run" in out2.getvalue(), "smoke checkpoints were accepted silently"
    assert not any(n in carried for n in ("smoke", "smoke_fall")), (
        f"a smoke checkpoint reached the working runs dir: {carried}")
    print(f"  N10c carried {len(carried)} real runs (ADL@80 + fall@60), refused 3 fixtures")

    shutil.rmtree(root, ignore_errors=True)
    print("  N10c smoke shards excluded from both corpora (and the skip is printed)")



def test_n10d_run_dir_finds_the_real_checkpoints_not_the_code_leftovers():
    """`no checkpoints matching adl_<stream>/best.pt` - with the runs dataset attached.

    run_dir() took the first `last.pt` in sorted order. behaviorsense-code carries the
    working tree's `runs/`, and "code" sorts before "runs", so it resolved to
    `EmotionSense-Extended/runs` - a directory holding `adl/` and `fall/` rather than
    `adl_joint/`. The error blamed a missing checkpoint; the fault was the wrong
    directory, and the real one was attached the whole time.

    This was the FOURTH call site to need the same exclusion (after shard_paths,
    carry-forward and the summary cell), which is why the rule now lives in one module.
    """
    import shutil
    import tempfile

    from behaviorsense.kaggle_artifacts import find_run_dir, is_real_artifact

    root = Path(tempfile.mkdtemp())
    code = root / "datasets" / "me" / "behaviorsense-code" / "EmotionSense-Extended"
    runs = root / "datasets" / "me" / "behaviorsense-runs" / "runs"
    for name in ("adl", "fall", "smoke", "smoke_fall"):
        (code / "runs" / name).mkdir(parents=True)
        (code / "runs" / name / "last.pt").write_bytes(b"x")
    (code / "src" / "behaviorsense").mkdir(parents=True)
    (code / "src" / "behaviorsense" / "__init__.py").write_text("", encoding="utf-8")
    for stream in ("joint", "bone", "joint_motion", "bone_motion"):
        (runs / f"adl_{stream}").mkdir(parents=True)
        (runs / f"adl_{stream}" / "best.pt").write_bytes(b"x")

    # The old rule, kept as a negative control: it must pick the WRONG directory, or
    # this test is not exercising the bug it exists for.
    old_pick = sorted(root.glob("**/last.pt"))[0].parent.parent
    assert sorted(q.name for q in old_pick.iterdir()) == ["adl", "fall", "smoke",
                                                          "smoke_fall"], (
        "control: first-sorted last.pt no longer selects the code leftovers")

    got = find_run_dir(root)
    assert got == runs, f"find_run_dir picked {got}, want {runs}"
    assert sorted(q.name for q in got.iterdir())[0] == "adl_bone"

    # The exclusion must not swallow the code we mount the dataset FOR.
    assert is_real_artifact(code / "src" / "behaviorsense" / "__init__.py")
    assert not is_real_artifact(code / "runs" / "smoke" / "last.pt")
    shutil.rmtree(root, ignore_errors=True)

    # And notebook 04 must call it rather than re-deriving the rule.
    cell = _cell_containing("04_evaluate_blackwell_offline.ipynb", "def run_dir")
    assert "find_run_dir" in cell, "notebook 04 still has its own checkpoint search"
    assert 'sorted(INPUT.glob("**/last.pt"))' not in cell, (
        "the first-sorted-last.pt rule is back in notebook 04")
    print("  N10d run_dir resolves the real run directory past the code-mount leftovers "
          "(old rule verified to pick the wrong one)")


def test_n11_notebook_00_weights_cell_never_passes_silently():
    """A missing weights/ directory must be reported, not rendered as no output.

    On the first real Kaggle run this cell printed nothing at all and the notebook moved
    on: the uploaded dataset had no weights/ directory, glob() matched nothing, and the
    loop body never ran. An empty iteration read exactly like a pass. This pins the three
    cases apart -- all present, none present, LFS pointer -- and asserts the middle one
    actually says something.
    """
    import contextlib
    import io
    import tempfile

    nb = json.loads((NOTEBOOKS / "00_stage_assets_online.ipynb").read_text(encoding="utf-8"))
    cells = [c["source"] for c in nb["cells"]
             if c["cell_type"] == "code" and "osnet_ain_x1_0" in c["source"]]
    assert len(cells) == 1, f"expected exactly one weights cell, found {len(cells)}"
    source = cells[0]

    def run(files: dict[str, int]) -> tuple[str, str | None]:
        """files: name -> size in bytes. Empty dict means no weights/ dir at all."""
        root = Path(tempfile.mkdtemp())
        out = Path(tempfile.mkdtemp())
        if files:
            (root / "weights").mkdir()
            for name, size in files.items():
                (root / "weights" / name).write_bytes(b"\0" * size)
        patched = (source
                   # notebook 00 now resolves the repo mount by content; short-circuit
                   # that lookup to the fixture instead of matching a literal path.
                   .replace('_init = sorted(pathlib.Path("/kaggle/input")'
                            '.glob("**/src/behaviorsense/__init__.py"))', "_init = []")
                   .replace('CODE = _init[0].parent.parent.parent if _init '
                            'else pathlib.Path("/kaggle/input/__missing__")',
                            f"CODE = pathlib.Path({root.as_posix()!r})")
                   .replace('pathlib.Path("/kaggle/working/weights")',
                            f"pathlib.Path({out.as_posix()!r})"))
        assert root.as_posix() in patched and out.as_posix() in patched, "redirect failed"
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(patched, "<nb00-weights>", "exec"), {"__name__": "nb00"})
            return buf.getvalue(), None
        except AssertionError as exc:
            return buf.getvalue(), str(exc)

    real = {"osnet_ain_x1_0_msmt17.pth": 17_293_009,
            "osnet_ain_x1_0_imagenet.pth": 10_929_757}

    text, err = run(real)
    assert err is None, f"the good set was rejected: {err}"
    assert "WARNING" not in text, f"good set warned anyway:\n{text}"
    assert "all 2 OSNet checkpoints staged" in text, f"no success line:\n{text}"

    # The regression: nothing on disk must NOT mean nothing on screen.
    text, err = run({})
    assert err is None, "absent weights should warn, not raise (03/04 do not need them)"
    assert text.strip(), "MISSING WEIGHTS PRODUCED NO OUTPUT -- the original silent skip"
    assert "WARNING" in text, f"missing weights did not warn:\n{text}"
    for name in real:
        assert name in text, f"warning does not name {name}:\n{text}"

    # Partial upload is the sneakier version of the same bug.
    text, err = run({"osnet_ain_x1_0_msmt17.pth": 17_293_009})
    assert err is None and "WARNING" in text, f"partial upload not flagged:\n{text}"
    assert "osnet_ain_x1_0_imagenet.pth" in text, f"missing file unnamed:\n{text}"

    # An LFS pointer still has to be fatal: it looks staged and is not.
    text, err = run({"osnet_ain_x1_0_msmt17.pth": 130,
                     "osnet_ain_x1_0_imagenet.pth": 10_929_757})
    assert err is not None and "pointer file" in err, (
        f"a 130-byte LFS pointer was accepted as weights (out={text!r}, err={err!r})")

    print("  N11 weights cell: staged 2/2 quietly, warned on none and on partial "
          "(both named), still fatal on an LFS pointer")


def test_n12_notebook_01_carry_forward_cannot_silently_drop_prior_shards():
    """Kaggle dataset versions REPLACE content, so a failed carry-forward destroys work.

    Notebook 01 runs 3-4 times, each session extracting one slice of Charades and saving
    a new version of behaviorsense-adl-shards. If the carry-forward copies nothing and the
    notebook proceeds, Save Version publishes only the current slice and the earlier
    sessions are gone. The dangerous case is not "no dataset" -- it is "dataset attached,
    shards not where we looked", which the previous version reported as "fresh start".

    The `nested` layout is the one that matters: saving /kaggle/working produces a dataset
    carrying that directory structure, so the .npz files sit under kaggle/working/shards,
    not directly under shards/.
    """
    import contextlib
    import io
    import tempfile

    nb = json.loads(
        (NOTEBOOKS / "01_prepare_adl_shards_offline.ipynb").read_text(encoding="utf-8"))
    cells = [c["source"] for c in nb["cells"]
             if c["cell_type"] == "code" and "carried forward" in c["source"]]
    assert len(cells) == 1, f"expected one carry-forward cell, found {len(cells)}"
    source = cells[0]

    def run(layout: str, mount_name: str = "behaviorsense-adl-shards"):
        root = Path(tempfile.mkdtemp())
        mount = root / mount_name
        out = Path(tempfile.mkdtemp())
        if layout == "proper":
            (mount / "shards").mkdir(parents=True)
            (mount / "shards" / "a.npz").write_bytes(b"x")
        elif layout == "nested":
            deep = mount / "kaggle" / "working" / "shards"
            deep.mkdir(parents=True)
            (deep / "a.npz").write_bytes(b"x")
            (deep / "b.npz").write_bytes(b"x")
        elif layout == "empty":
            (mount / "shards").mkdir(parents=True)
        elif layout == "absent":
            root.mkdir(exist_ok=True)
        patched = (source
                   .replace('pathlib.Path("/kaggle/input")',
                            f"pathlib.Path({root.as_posix()!r})")
                   .replace('pathlib.Path("/kaggle/working/shards")',
                            f"pathlib.Path({out.as_posix()!r})"))
        assert root.as_posix() in patched and out.as_posix() in patched, "redirect failed"
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(patched, "<nb01-carry>", "exec"), {"__name__": "nb01"})
            return buf.getvalue(), None, len(list(out.glob("*.npz")))
        except AssertionError as exc:
            return buf.getvalue(), str(exc), len(list(out.glob("*.npz")))

    text, err, n = run("absent")
    assert err is None and n == 0, "an unattached dataset is a legitimate first run"
    assert "fresh start" in text, f"first run should say so:\n{text}"

    text, err, n = run("proper")
    assert err is None and n == 1, f"standard layout carried {n} shards, expected 1"

    # The regression: previously reported "fresh start", then overwrote the dataset.
    text, err, n = run("nested")
    assert err is None, f"nested layout raised: {err}"
    assert n == 2, f"nested layout carried {n} shards, expected 2 -- prior work would be lost"
    assert "fresh start" not in text, (
        "an attached dataset was reported as a fresh start; Save Version would then "
        f"REPLACE it with only this session's slice:\n{text}")

    # The spelling this project was actually uploaded under. A name-exact match would
    # miss it, print "fresh start", and destroy the previous version on save.
    text, err, n = run("nested", mount_name="behavioursense-ADL-shards")
    assert err is None and n == 2, (
        f"British spelling carried {n} shards, expected 2 -- a rename would silently "
        f"discard prior sessions:\n{text}")

    text, err, n = run("empty")
    assert err is not None, (
        "an attached-but-empty dataset must refuse, not proceed to Save Version")
    assert "destroy" in err or "REPLACE" in err, f"refusal does not explain the stakes: {err}"

    print("  N12 carry-forward: fresh start when unattached, 1/2 shards for flat/nested, "
          "2 for the behaviour spelling, refuses when attached but empty")


def test_n13_offline_notebooks_resolve_assets_by_content_not_dataset_name():
    """The offline notebooks must find wheels/weights however Kaggle happens to mount them.

    Three things vary independently and none is knowable from this machine: notebook 00's
    two output folders can be published as ONE dataset or two, the dataset title is free
    text (this project exists as both `behaviorsense-*` and `behavioursense-WW`), and Save
    Version nests the working directory inside the dataset. A name-matching resolver is
    wrong under at least one of those, and the failure mode is a 12-hour session dying on
    its first cell.
    """
    import contextlib
    import io
    import shutil
    import tempfile

    for name in ("03_train_blackwell_offline.ipynb", "04_evaluate_blackwell_offline.ipynb"):
        nb = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
        cells = [c["source"] for c in nb["cells"]
                 if c["cell_type"] == "code" and "Resolve every attached asset" in c["source"]]
        assert len(cells) == 1, f"{name}: expected 1 resolver cell, found {len(cells)}"
        assert nb["cells"][1]["source"] is cells[0] or nb["cells"][1]["source"] == cells[0], (
            f"{name}: the resolver must be the FIRST code cell; later cells depend on it")

    resolver = [c["source"] for c in json.loads(
        (NOTEBOOKS / "03_train_blackwell_offline.ipynb").read_text(encoding="utf-8"))["cells"]
        if "Resolve every attached asset" in c["source"]][0]

    def build(layout: str) -> Path:
        root = Path(tempfile.mkdtemp())
        code = root / "behaviorsense-code"
        (code / "src" / "behaviorsense").mkdir(parents=True)
        (code / "src" / "behaviorsense" / "__init__.py").write_text("")
        (code / "scripts").mkdir()
        # The resolver now also enforces the repo contract, so the fixture has to look
        # like a current checkout. Copying the real files keeps the fixture honest: if a
        # contract token disappears upstream, this test notices.
        for rel, _tok in contract_entries():
            dst = code / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, dst)
        if layout == "single":            # one dataset holding both folders
            ww = root / "behavioursense-WW"
            (ww / "wheels").mkdir(parents=True)
            (ww / "wheels" / "torch-2.7.1+cu128-cp312-cp312-manylinux_2_28_x86_64.whl").write_bytes(b"x")
            (ww / "weights").mkdir()
            (ww / "weights" / "rtmo-l.onnx").write_bytes(b"x")
        elif layout == "split":           # two separate datasets
            a = root / "behaviorsense-wheels" / "wheels"
            a.mkdir(parents=True)
            (a / "torch-2.7.1+cu128-cp312-cp312-manylinux_2_28_x86_64.whl").write_bytes(b"x")
            b = root / "behaviorsense-weights" / "weights"
            b.mkdir(parents=True)
            (b / "rtmo-l.onnx").write_bytes(b"x")
        elif layout == "nested":          # Save Version nested /kaggle/working
            w = root / "ww" / "kaggle" / "working"
            (w / "wheels").mkdir(parents=True)
            (w / "wheels" / "torch-2.7.1+cu128-cp312-cp312-manylinux_2_28_x86_64.whl").write_bytes(b"x")
            (w / "weights").mkdir()
            (w / "weights" / "rtmo-l.onnx").write_bytes(b"x")
        return root

    def resolve(root: Path):
        src = resolver.replace('pathlib.Path("/kaggle/input")',
                               f"pathlib.Path({root.as_posix()!r})")
        assert root.as_posix() in src, "redirect failed"
        ns: dict = {"__name__": "nb03"}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src, "<resolve>", "exec"), ns)
        return ns

    for layout in ("single", "split", "nested"):
        ns = resolve(build(layout))
        assert Path(ns["WHEELS"]).name == "wheels", f"{layout}: wheels -> {ns['WHEELS']}"
        assert Path(ns["WEIGHTS"]).name == "weights", f"{layout}: weights -> {ns['WEIGHTS']}"
        assert Path(ns["SRC"]).name == "src", f"{layout}: SRC -> {ns['SRC']}"
        assert Path(ns["SCRIPTS"]).name == "scripts", f"{layout}: SCRIPTS -> {ns['SCRIPTS']}"

    # Positive control above; now prove it refuses rather than resolving to something wrong.
    try:
        resolve(build("code_only"))
        raise AssertionError("missing wheels resolved anyway - the check proves nothing")
    except AssertionError as exc:
        # The refusal must name what is missing AND what is mounted, so the reader can see
        # the mismatch without exploring the filesystem. find_wheel_dir() words it
        # differently from find_asset() because it also rejects a wheel directory that
        # belongs to some other attached dataset.
        msg = str(exc)
        assert ("nothing matches" in msg or "no *.whl anywhere" in msg
                or "none holds any of" in msg), f"unhelpful refusal: {exc}"
        assert "behaviorsense-code" in msg, (
            f"refusal does not list what IS attached, so it cannot be diagnosed: {exc}")

    # The repo dataset is a snapshot and drifts behind the notebooks. Every token in the
    # contract must exist in THIS checkout, or the guard fires on a current upload and
    # trains the user to ignore it - worse than no guard.
    # Parse via the shared helper, which slices the CONTRACT list specifically. A loose
    # regex over the whole resolver also matches the ("wheels", WHEELS) print loop and
    # then reports a missing file named "wheels".
    contract = contract_entries()
    assert len(contract) >= 6, f"contract shrank to {len(contract)} entries"
    for rel, token in contract:
        target = ROOT / rel
        assert target.is_file(), f"contract names {rel}, absent from this checkout"
        if token:
            text = target.read_text(encoding="utf-8", errors="ignore")
            assert token in text, (
                f"contract requires {token!r} in {rel}, but this checkout lacks it - the "
                "guard would reject a freshly uploaded dataset")
        if token:
            assert token in target.read_text(encoding="utf-8", errors="ignore"), (
                f"contract requires {token!r} in {rel}, but this checkout lacks it - "
                f"the guard would reject a freshly uploaded dataset")

    print(f"  N13 resolver: single-dataset, split and nested layouts all resolve "
          f"wheels/weights/src; refuses when wheels absent; {len(contract)} contract "
          f"tokens all present in this checkout")


def test_n14_preflight_blocks_on_what_the_session_actually_needs():
    """The preflight must fail a training run only for assets training actually loads.

    Regression: `osnet_ain_x1_0_msmt17.pth` was marked required outright, so notebook 03's
    preflight failed a session over a checkpoint it never opens -- ST-GCN++ trains from
    pre-extracted .npz shards, and OSNet is Agent 1 serving-time with its operating point
    already fitted (results/reid_eval.md). A preflight that cries wolf gets deleted, which
    costs far more than the check was worth.

    Both directions are asserted: training must not block on OSNet, and serving must.
    Otherwise "nothing blocks" would pass this test by doing nothing at all.
    """
    import tempfile

    smoke = ROOT / "scripts" / "kaggle_smoke_test.py"
    staged = Path(tempfile.mkdtemp())
    # Exactly what notebook 00 produced: the two ONNX models, no .pth files.
    for name, mb in (("rtmo-l.onnx", 176), ("rtdetr-l.onnx", 132)):
        (staged / name).write_bytes(b"\0" * (mb * 1000))

    def blocking_problems(profile: str) -> list[str]:
        r = subprocess.run(
            [sys.executable, str(smoke), "--assets", str(staged), "--profile", profile],
            capture_output=True, text=True, timeout=300, env=SUBPROC_ENV)
        # Only the asset findings matter here; this machine has no GPU, and that line is
        # a genuine blocker on Kaggle, so it is filtered rather than treated as noise.
        return [ln.strip(" -") for ln in r.stdout.splitlines()
                if ln.strip().startswith("- ") and "CUDA" not in ln]

    training = blocking_problems("training")
    assert not any("osnet" in p.lower() for p in training), (
        f"training blocks on OSNet, which notebook 03 never loads: {training}")

    serving = blocking_problems("serving")
    assert any("osnet_ain_x1_0_msmt17" in p for p in serving), (
        f"serving does NOT block on the ReID weights it cannot run without: {serving}")

    # A corrupt file that IS present must fail regardless of profile - the Git-LFS pointer
    # case, which is about the file being wrong rather than about who needs it.
    pointer = Path(tempfile.mkdtemp())
    (pointer / "rtmo-l.onnx").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    r = subprocess.run(
        [sys.executable, str(smoke), "--assets", str(pointer), "--profile", "training"],
        capture_output=True, text=True, timeout=300, env=SUBPROC_ENV)
    assert "Git-LFS pointer" in r.stdout, (
        f"a 41-byte rtmo-l.onnx was accepted under --profile training:\n{r.stdout[-600:]}")

    # And notebook 03 must actually pass the scoped profile, not the strict default.
    nb = json.loads(
        (NOTEBOOKS / "03_train_blackwell_offline.ipynb").read_text(encoding="utf-8"))
    calls = [m.group(0) for c in nb["cells"] if c["cell_type"] == "code"
             for m in re.finditer(r'\[\s*sys\.executable[^\]]*\]', c["source"], re.S)
             if "kaggle_smoke_test.py" in m.group(0)]
    assert len(calls) == 1, f"expected one preflight invocation, found {len(calls)}"
    assert '"--profile", "training"' in calls[0], (
        "notebook 03 invokes the preflight without --profile training, so it inherits the "
        "strict default and fails on serving-only assets again")

    print(f"  N14 preflight: training blocks on {len(training)} asset(s) (not OSNet), "
          f"serving blocks on OSNet, LFS pointers fail in every profile")


def test_n22_notebook06_imports_and_asset_patterns_resolve():
    """Notebook 06 is the gate before a 6 h extraction session, so it must not fail late.

    Two failure modes are checked, both of which have precedent in this project:

      - an import that no longer exists. Notebook 06 pulls eleven names out of
        `data.toyota`; a rename would surface as a `ImportError` on Kaggle six minutes in,
        after the wheels install and the mounts resolve.
      - an asset glob that matches nothing. The nine Toyota mounts are resolved by CONTENT
        and the patterns encode real filenames the user reported from their own mounts, so
        each one is exercised here against a synthetic tree with those exact names. A
        pattern that silently matches nothing reads as "dataset not attached", which is the
        failure mode `find_asset` was written to end.
    """
    import importlib

    nb = json.loads(
        (NOTEBOOKS / "06_toyota_preflight_online.ipynb").read_text(encoding="utf-8"))
    code = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
    joined = "\n".join(code)

    # -- every behaviorsense import in the notebook must exist ---------------------------
    wanted: dict[str, set[str]] = {}
    for m in re.finditer(r"from (behaviorsense[\w.]*) import \(([^)]*)\)|"
                         r"from (behaviorsense[\w.]*) import ([^\n(]+)", joined):
        mod = m.group(1) or m.group(3)
        names = m.group(2) or m.group(4)
        wanted.setdefault(mod, set()).update(
            n.strip() for n in names.replace("\n", " ").split(",") if n.strip())
    assert wanted, "no behaviorsense imports found in notebook 06 - did the cells change?"

    missing: list[str] = []
    for mod, names in sorted(wanted.items()):
        obj = importlib.import_module(mod)
        for name in sorted(names):
            if not hasattr(obj, name):
                missing.append(f"{mod}.{name}")
    assert not missing, (
        f"notebook 06 imports {len(missing)} name(s) that do not exist: {missing}")

    # -- the asset table must describe the layouts actually in the mounts ----------------
    # Notebook 06 resolves at BOUNDED DEPTH: `(subdir, filename_glob, required)` per asset,
    # applied inside each `datasets/<owner>/<slug>/` root. The first Kaggle run proved why -
    # nine `**` globs over ~130,000 mounted files took 78 minutes and the run then died on a
    # parse error it could have reached in seconds.
    real = {
        "annotation-v1-0": ("Annotation/P18", "P18T13C07.csv"),
        "rgb-untrimmed": ("Videos_mp4", "P15T17C03.mp4"),
        "pose-untrimmed": ("Skeleton", "results_P17T07C02_lcrnet+v3d.json"),
        "depth-untrimmed": ("Depth", "P15T17C03.mp4"),
        "toyota-smarthome-rgb": ("mp4", "Walk_p03_r01_v15_c07.mp4"),
        "toyota-smarthome-skeleton": ("json", "Drink.Fromcup_p20_r02_v02_c05.json"),
        "toyota-smarthome-skeleton-v1-2": ("", "Walk_p25_r12_v15_c06_pose3d.json"),
        "toyota-smarthome-depth": ("depth", "Walk_p03_r01_v15_c07.mp4"),
    }
    table = re.findall(r'"(\w+)":\s*\("([^"]*)",\s*"([^"]+)",\s*(True|False)\)', joined)
    assert len(table) == 8, f"expected 8 asset entries in notebook 06, found {len(table)}"

    from fnmatch import fnmatch
    with tempfile.TemporaryDirectory() as td:
        roots = Path(td) / "datasets" / "owner"
        for slug, (sub, fname) in real.items():
            d = roots / slug / sub if sub else roots / slug
            d.mkdir(parents=True, exist_ok=True)
            (d / fname).write_bytes(b"x")

        unmatched = []
        for key, subdir, pattern, _req in table:
            hit = False
            for slug in sorted(p.name for p in roots.iterdir()):
                base = roots / slug / subdir if subdir else roots / slug
                if not base.is_dir():
                    continue
                cands = [base, *[c for c in base.iterdir() if c.is_dir()]]
                if any(fnmatch(f.name, pattern)
                       for c in cands for f in c.iterdir() if f.is_file()):
                    hit = True
                    break
            if not hit:
                unmatched.append(f"{key} ({subdir or '<root>'}/{pattern})")
        assert not unmatched, (
            f"{len(unmatched)} asset entr(ies) match none of the real mount layouts: "
            f"{unmatched}. An entry that matches nothing reads as an unattached dataset.")

        # Negative control: the table must not be matching everything indiscriminately.
        decoy = roots / "unrelated-slug"
        decoy.mkdir(parents=True, exist_ok=True)
        (decoy / "notes.txt").write_bytes(b"x")
        for key, subdir, pattern, _req in table:
            base = decoy / subdir if subdir else decoy
            if base.is_dir():
                assert not any(fnmatch(f.name, pattern)
                               for f in base.iterdir() if f.is_file()), (
                    f"{key} matched an unrelated file; the pattern is too loose")

    print(f"  N22 notebook 06: {sum(len(v) for v in wanted.values())} imports across "
          f"{len(wanted)} module(s) all resolve; {len(table)}/8 bounded-depth asset entries "
          f"match the real mount layouts, decoy unmatched")


def test_n23_notebook07_extraction_contract():
    """Notebook 07 spends ~2 h of Blackwell per run, so its label plumbing is checked here.

    Three failure modes, each of which would produce a shard set that trains cleanly and is
    wrong:

      - a fine label written into `labels` (which the loader validates against the COARSE
        class count) or vice versa. The shard carries both and they index different spaces.
      - the frame rate assumed rather than read. The trimmed containers report 20 fps while
        their own README says 30, so a hardcoded divisor rescales every window.
      - resume that does not actually skip. The clip list has to come out of the shards.
    """
    import importlib

    nb = json.loads(
        (NOTEBOOKS / "07_toyota_extract_offline.ipynb").read_text(encoding="utf-8"))
    code = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
    joined = "\n".join(code)

    # -- imports resolve -----------------------------------------------------------------
    sys.path.insert(0, str(ROOT / "scripts"))
    missing = []
    for m in re.finditer(r"from ([\w.]+) import \(([^)]*)\)|"
                         r"from ([\w.]+) import ([^\n(]+)", joined):
        mod = m.group(1) or m.group(3)
        names = m.group(2) or m.group(4)
        if not (mod.startswith("behaviorsense") or mod == "prepare_skeletons"):
            continue
        obj = importlib.import_module(mod)
        for name in (x.strip() for x in names.replace("\n", " ").split(",")):
            if name and not hasattr(obj, name):
                missing.append(f"{mod}.{name}")
    assert not missing, f"notebook 07 imports names that do not exist: {missing}"

    # -- both label spaces are written, and to the right keys -----------------------------
    assert "fine_labels=np.asarray(fine" in joined, (
        "the shard must carry `fine_labels` - collapsing to coarse before the loss destroys "
        "the supervision that separates Cook.Cut from Cook.Stir")
    assert "labels=np.asarray(coarse" in joined, (
        "`labels` must hold the COARSE id: that is the key the existing loader validates "
        "against the coarse class count")
    assert "clips=np.asarray(clips" in joined, "resume needs the clip list in the shard"
    assert "starts=np.asarray(starts" in joined, (
        "the shard must record each window's START index within its clip. The kept windows "
        "are not an arithmetic sequence - the visibility gate and the gap mask both drop "
        "some - so without it a shard is an unordered bag: neither the temporal order that "
        "segment metrics need nor a per-frame timeline for mAP can be rebuilt, and the only "
        "remedy is a seven-hour re-extraction")

    # -- the frame rate is read from the file, not assumed --------------------------------
    assert "cv2.CAP_PROP_FPS" in joined, "the source rate must be read per file"
    assert "TSM_FPS" not in joined, (
        "notebook 07 must not use the TSM_FPS constant for decoding - the containers "
        "disagree with the README, so the rate is read per file and the constant is only "
        "the expected value notebook 06 reports against")
    assert "FPS_SAMPLE = 20.0" in joined, (
        "the sample rate must be 20 Hz for the trimmed half. Every container reports 20 fps, "
        "so 12.5 gave an integer stride of 2 = an effective 10 Hz, a 3.0 s window, and 6,040 "
        "of 16,115 clips (37.5%) too short to yield one - concentrated in the short "
        "transition classes Agent 3 counts sit_to_stand_count from")

    # -- sampling must be EXACT resampling, never integer striding ------------------------
    assert "want = int(round(k * src / rate))" in joined, (
        "frames must be picked by exact resampling. Integer striding cost the first run "
        "6,040 clips (37.5%) - at 20 fps a nominal 12.5 Hz rounds to stride 2, giving a "
        "3.0 s window that every short transition clip fails - and it cannot hold the "
        "window duration equal across a 20 fps and a 25 fps corpus")
    assert "rate = min(fps_sample, src)" in joined, (
        "the effective rate must be capped at the container rate: resampling UP would "
        "duplicate frames and invent motion that is not in the video")
    assert "drop_reasons" in joined and "rates = collections.Counter()" in joined, (
        "unusable clips must be counted BY REASON and the effective rates recorded - the "
        "first run reported 6,040 unusable with no breakdown, so the cause needed "
        "arithmetic on the class histogram to find")

    # -- the sampling rate must be part of the shard name --------------------------------
    assert 'PREFIX = f"toyota_{MODE}_{FPS_SAMPLE:g}hz"' in joined, (
        "the rate belongs in the shard name: without it, re-running at a different "
        "FPS_SAMPLE resumes against shards whose windows span a different number of "
        "seconds, and two rates mix silently inside one training set")

    # -- resume reads the clip list back --------------------------------------------------
    assert re.search(r'DONE\.update\(str\(c\) for c in _z\["clips"\]\)', joined), (
        "resume must repopulate DONE from the carried-forward shards")
    assert "if w[0].name not in DONE" in joined, "the work list must exclude done clips"
    # ...and the resume lookup itself must be mount-indexed stats, not a glob: even a
    # fixed-depth glob (`datasets/*/*/*/shards/...`) makes pathlib scandir every slug
    # child, which is the ~8.5 minutes the first run of this notebook spent resolving.
    assert '_cand in (_slug / "shards"' in joined, (
        "resume must stat <slug>/shards/ per mount instead of globbing across mounts")

    # -- carry-forward must copy EVERY shard set, or Save Version destroys the other half --
    assert '_cand.glob("toyota_*.npz")' in joined, (
        "carry-forward must copy every toyota shard set, not just this run's prefix. Save "
        "Version publishes whatever is in /kaggle/working, so an untrimmed run that copied "
        "only its own prefix would publish a dataset version WITHOUT the 214,913 trimmed "
        "windows - silent data loss, discoverable only when training came up short")
    assert 'OUT.glob(f"{PREFIX}_*.npz")' in joined, (
        "resume must still key on the CURRENT prefix, so a rate or mode change re-extracts "
        "rather than believing another shard set's clips are done")
    assert "the dataset this Save Version will publish" in joined, (
        "the verify cell must report the whole output dataset, not only this run's half - "
        "that line is what says the carry-forward worked")

    # -- the CS partition is asserted in the SHARDS, not just planned ---------------------
    assert "a subject appears on both CS sides" in joined, (
        "the verify cell must assert CS disjointness on the subjects actually written")

    # -- both modes exist and the expensive one defaults to the cheap side ----------------
    assert 'MODE = "trimmed"' in joined, "trimmed is the affordable default"
    assert 'UNTRIMMED_SIDE = "test"' in joined, (
        "untrimmed must default to the CS test side - the full corpus is ~9 h and does not "
        "fit one session")
    print("  N23 notebook 07: imports resolve, both label spaces written, fps read per "
          "file, 20 Hz exact resampling (1.50s window on both halves), resume rebuilds from shard clip lists, CS "
          "disjointness asserted on written subjects")


def test_n24_every_notebook_cell_defines_what_it_calls():
    """Static name resolution across each notebook's cells, in execution order.

    The failure this exists for: notebook 07 used the lighter `RESOLVE_CODE` cell (which
    resolves only the repo) while cell 3 called `find_asset`, which at that time was only
    exported by the fuller `RESOLVE` cell. NameError 370 s into a Blackwell session, on a
    line that had nothing to do with the real cause - and the existing import test could not
    see it, because it only checked `behaviorsense.*` module attributes.

    Checked names are the notebook-level helpers the resolver cells export. A notebook that
    calls one must have a cell before it that defines it.
    """
    HELPERS = ("find_asset", "find_fast", "find_dir", "find_wheel_dir",
               "find_charades_csv", "attached_mounts", "sm120_ok", "shard_paths",
               "run_dir", "video_path")
    RESOLVED_VARS = ("WHEELS", "WEIGHTS", "SRC", "CODE", "SCRIPTS", "CONFIGS", "ATTACHED",
                     "_SLUGS", "INPUT")

    problems: list[str] = []
    for nb_path in sorted(NOTEBOOKS.glob("*.ipynb")):
        doc = json.loads(nb_path.read_text(encoding="utf-8"))
        cells = [c["source"] for c in doc["cells"] if c["cell_type"] == "code"]
        defined: set[str] = set()
        for n, src in enumerate(cells, start=1):
            # Comments and string literals are stripped before scanning. Without this the
            # check fires on prose: RESOLVE_CODE's own comment explains a past bug with the
            # words "if not sm120_ok()", which is documentation, not a call.
            code_only = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"\n]*"|'
                               r"'[^'\n]*'|#[^\n]*", " ", src)
            for name in HELPERS:
                if re.search(rf"\b{name}\s*\(", code_only) and name not in defined:
                    if not re.search(rf"^\s*def {name}\b", code_only, re.M):
                        problems.append(f"{nb_path.name} cell {n}: calls {name}() before "
                                        "any cell defines it")
            for name in RESOLVED_VARS:
                reads = re.search(rf"\b{name}\b(?!\s*=)", code_only)
                assigns_here = re.search(rf"^\s*{name}\s*=", code_only, re.M)
                if reads and not assigns_here and name not in defined:
                    problems.append(f"{nb_path.name} cell {n}: reads {name} before any "
                                    "cell assigns it")
            # Everything this cell defines becomes available to later cells.
            defined.update(re.findall(r"^\s*def (\w+)", code_only, re.M))
            defined.update(re.findall(r"^(\w+)\s*=", code_only, re.M))
            for m in re.finditer(r"^(\w+)\s*,\s*(\w+)\s*=", code_only, re.M):
                defined.update(m.groups())
            for m in re.finditer(r"^from [\w.]+ import \(?([^)]*)\)?", code_only, re.M):
                defined.update(x.strip() for x in m.group(1).replace("\n", " ").split(","))

    assert not problems, "notebook cell dependencies unresolved:\n  " + "\n  ".join(problems)
    print(f"  N24 {len(list(NOTEBOOKS.glob('*.ipynb')))} notebooks: every tracked helper "
          f"and resolved variable is defined by an earlier cell ({len(HELPERS)} helpers, "
          f"{len(RESOLVED_VARS)} variables tracked)")


def test_n25_onnxruntime_install_order_in_every_notebook_that_needs_it():
    """The CPU onnxruntime must be removed BEFORE onnxruntime-gpu is installed, everywhere.

    Both packages own the same `onnxruntime/` directory. Uninstalling the CPU build
    afterwards deletes files SHARED with the GPU build, leaving a module that imports and is
    hollow - notebook 05 died on `AttributeError: module 'onnxruntime' has no attribute
    '__version__'` for exactly this reason, and its comment argued for the wrong order while
    notebooks 01 and 07 already did it right.

    Required order per notebook: `pip uninstall onnxruntime` first, `rtmlib` installed with
    `--no-deps` so it cannot drag the CPU build back in, and `onnxruntime-gpu` last.
    """
    checked = 0
    for nb_path in sorted(NOTEBOOKS.glob("*.ipynb")):
        doc = json.loads(nb_path.read_text(encoding="utf-8"))
        joined = "\n".join(c["source"] for c in doc["cells"] if c["cell_type"] == "code")

        # Parse pip invocations as (verb, packages, position). `pip download` must NOT count:
        # notebook 00 STAGES the GPU wheel for the offline notebooks and separately installs
        # the CPU build on purpose, because it runs without an accelerator and only needs
        # rtmlib to fetch RTMO's onnx into the cache. Treating that as an install made this
        # test fail on a correct notebook.
        calls = []
        for m in re.finditer(r'"pip",\s*"(install|uninstall|download)"(.*?)\]', joined, re.S):
            pkgs = [p for p in re.findall(r'"([^"]+)"', m.group(2))
                    if not p.startswith("-")]
            calls.append((m.group(1), pkgs, m.start()))

        gpu_installs = [c for c in calls
                        if c[0] == "install" and any("onnxruntime-gpu" in p for p in c[1])]
        if not gpu_installs:
            continue
        checked += 1

        cpu_removals = [c for c in calls
                        if c[0] == "uninstall" and "onnxruntime" in c[1]]
        rtmlib_installs = [c for c in calls
                           if c[0] == "install" and "rtmlib" in c[1]]

        assert cpu_removals, (
            f"{nb_path.name} installs onnxruntime-gpu but never uninstalls the CPU build")
        assert min(c[2] for c in cpu_removals) < min(c[2] for c in gpu_installs), (
            f"{nb_path.name} uninstalls onnxruntime AFTER installing onnxruntime-gpu. They "
            "share a directory, so that deletes the GPU build's files and leaves a module "
            "with no __version__ - which is exactly how notebook 05 died")
        for verb, pkgs, pos in rtmlib_installs:
            seg = joined[max(0, pos - 200):pos + 200]
            assert "--no-deps" in seg, (
                f"{nb_path.name} installs rtmlib without --no-deps, so pip pulls the CPU "
                "onnxruntime back in as its dependency")
            assert pos < min(c[2] for c in gpu_installs), (
                f"{nb_path.name} installs rtmlib after onnxruntime-gpu; the GPU build must "
                "be written last so nothing overwrites its provider registration")

    assert checked >= 3, f"expected at least 3 notebooks installing onnxruntime-gpu, saw {checked}"
    print(f"  N25 {checked} notebooks install onnxruntime-gpu: CPU build removed first, "
          "rtmlib --no-deps, GPU build last in every one (pip download excluded)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            print(f"{fn.__name__}:")
            fn()
            print("  PASS")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
