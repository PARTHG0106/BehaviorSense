"""Video in, `FrameObservation` out — behind a subprocess boundary.

Two boundaries coincide here, and that is the whole design.

**The pixel boundary.** Agent 2 consumes `FrameObservation` and never touches pixels; that
is what makes the agent split auditable. So everything that needs frames — decode, RTMO pose,
OSNet re-identification, tracking, role voting — happens on one side of a line, and what
crosses it is schema objects.

**The crash boundary.** Arbitrary video is the input class that has already killed this
project's kernels. `cv2` delegates to ffmpeg, ffmpeg raises SIGSEGV/SIGABRT on malformed
streams, and a signal is not an exception: `try/except` cannot see it, a frame cap cannot
prevent it, and the interpreter simply stops. Notebook 02 lost finished corpora to exactly
this and needed a journalling subprocess probe plus a PyAV fallback to finish at all. Serving
user uploads from the same process as the models would mean one bad file takes down the
session, the tunnel and the demo together.

Putting both on the same line means `extract_isolated()` can promise something useful: a
corrupt upload returns a 422, not a dead kernel. The child is the only thing at risk, and the
child holds nothing that cannot be recreated.

Run as a module, which is how the parent invokes it:

    python -m behaviorsense.video --video clip.mp4 --out poses.json \\
        --rtmo weights/rtmo-l.onnx [--osnet weights/osnet_ain_x1_0_msmt17.pth]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from behaviorsense.agents.perception import PerceptionAgent, RawDetection
from behaviorsense.schemas import (
    BoundingBox,
    FrameObservation,
    Keypoints,
    PersonObservation,
    Role,
)

# Frames analysed per upload, at the serving rate - 12,000 is ten minutes at 20 Hz.
#
# It was 900 (45 s), on the stated grounds that "a 30-second clip at 15 Hz is 450 frames and about
# 8 MB of JSON". That size estimate was wrong by about 35x: a frame with one person serialises to
# ~509 bytes, so 900 frames is ~0.3 MB and even 12,000 is ~4 MB at the 0.69 persons/frame actually
# observed. Bandwidth was never the constraint, so the cap was refusing analysis for no reason -
# a ten-minute video of someone cooking, cleaning, watching television and drinking from a mug was
# being judged on its first 45 seconds, and the three activities after that read as 0.00.
#
# What it DOES cost is pose time: RTMO runs one frame at a time through rtmlib at ~26 frames/s on a
# P100, so 12,000 frames is roughly 7.7 minutes of GPU. Nothing puts a wall-clock ceiling on that,
# deliberately - see `extract_isolated`, which gives up on the child going SILENT instead, because
# any ceiling high enough for the longest acceptable video is too high to catch a hang. Measured:
# against a talking-but-slow child and a wedged one, no value of `timeout` both let the first finish
# and killed the second, while a progress watcher separated them on the first try.
#
# Long uploads are still REFUSED rather than truncated: a report over the first minutes of a longer
# video, presented as a report over the video, is the quiet misrepresentation this project keeps
# finding and removing. `truncated` in the payload says which happened.
MAX_FRAMES = 12_000
# The rate the ADL shards were built at (notebook 01, `FPS_SAMPLE = 15`). Serving MUST resample
# to it, because a 30-frame window is only 2.0 s of motion at 15 Hz and ST-GCN++ has never seen
# any other duration.
SHARD_FPS = 15.0
KP_DECIMALS = 1         # pixel coordinates; a tenth of a pixel is already beyond the model
# Frames between the child's progress lines. At ~26 fps that is a report every ~2 s, which is
# frequent enough for `stall_s` to be a tight liveness check and rare enough to be free.
PROGRESS_EVERY = 50

# The directory that CONTAINS the `behaviorsense` package - `src/` in a checkout, the dataset
# mount on Kaggle. Derived from this file rather than passed in, because the module already
# knows where it lives and a caller that has to be told would eventually be told wrongly.
PKG_PARENT = Path(__file__).resolve().parent.parent


def child_env(extra_path: str | Path | None = None) -> dict[str, str]:
    """The environment for the decode child: inherited, plus `behaviorsense` on the path.

    `sys.path` does not cross a process boundary. Notebook 05 makes the package importable
    with `sys.path.insert(0, str(SRC))`, which configures THIS interpreter and says nothing
    to a fresh one - so the child died on `No module named 'behaviorsense'` while the parent
    was importing it perfectly well. The environment is what crosses, so the path goes there.

    INHERIT and add, never build from scratch: the child needs CUDA_VISIBLE_DEVICES,
    LD_LIBRARY_PATH and Kaggle's proxy variables, and a hand-built env that looks tidy is how
    onnxruntime ends up with no GPU. This mirrors what every training subprocess in
    `_generate.py` already does with `{**os.environ, "PYTHONPATH": str(SRC)}`; the only
    difference is that here the caller cannot reach the `subprocess.run` to do it.

    PREPENDED to any existing PYTHONPATH rather than replacing it, so a caller who set one
    keeps it, and an already-installed copy of the package cannot shadow the mounted one.
    """
    parts = [str(PKG_PARENT)]
    if extra_path:
        parts.insert(0, str(extra_path))
    inherited = os.environ.get("PYTHONPATH", "")
    if inherited:
        parts.append(inherited)
    return {**os.environ, "PYTHONPATH": os.pathsep.join(parts)}


def box_from_keypoints(xy: np.ndarray, scores: np.ndarray, *, pad: float = 0.12,
                       thresh: float = 0.3) -> BoundingBox | None:
    """Bounding box from the visible joints, padded.

    RTMO is a one-stage pose model: it returns keypoints, not boxes, and the tracker and the
    re-identifier both need one. Padding matters for re-id specifically — a box drawn tight
    to the joints crops the clothing that OSNet actually keys on, and the operating point
    (tau=0.355) was fitted on person crops, not skeleton crops.
    """
    keep = scores >= thresh
    if keep.sum() < 2:
        return None
    pts = xy[keep]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    x1, x2 = x1 - w * pad, x2 + w * pad
    y1, y2 = y1 - h * pad, y2 + h * pad
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))


class RTMODetector:
    """`PoseDetector` over rtmlib's RTMO. Constructed only in the child process."""

    def __init__(self, onnx: str, device: str = "cuda", input_size: int = 640) -> None:
        from rtmlib import RTMO                                    # noqa: PLC0415

        backend = "onnxruntime"
        self.model = RTMO(onnx_model=onnx, model_input_size=(input_size, input_size),
                          backend=backend, device=device)
        # get_available_providers() reports what the WHEEL was compiled with; the session
        # reports what actually loaded. onnxruntime-gpu >= 1.27 is built against CUDA 13 and
        # silently falls back to CPU on Kaggle's CUDA 12 image - a ~50x slowdown that looks
        # like a slow video rather than a broken install. Assert on the session.
        sess = getattr(self.model, "session", None)
        self.providers = list(sess.get_providers()) if sess is not None else []
        if device == "cuda" and not any("CUDA" in p for p in self.providers):
            raise RuntimeError(
                f"RTMO loaded on {self.providers or 'an unknown provider'}, not CUDA. "
                "Pin onnxruntime-gpu==1.26.0 (1.27+ is built against CUDA 13)."
            )

    def detect(self, frame: np.ndarray) -> list[RawDetection]:
        kpts, scores = self.model(frame)
        out: list[RawDetection] = []
        for person_xy, person_scores in zip(np.asarray(kpts), np.asarray(scores)):
            person_xy = np.asarray(person_xy, dtype=np.float32).reshape(-1, 2)[:17]
            person_scores = np.asarray(person_scores, dtype=np.float32).reshape(-1)[:17]
            if len(person_xy) < 17:
                continue
            box = box_from_keypoints(person_xy, person_scores)
            if box is None:
                continue
            out.append(RawDetection(
                box=box,
                confidence=float(np.clip(person_scores.max(), 0.0, 1.0)),
                keypoints=Keypoints(
                    xy=[(float(x), float(y)) for x, y in person_xy],
                    scores=[float(np.clip(s, 0.0, 1.0)) for s in person_scores],
                ),
            ))
        return out


def _osnet_embedder(weights: str, device: str):
    """Optional. Without it every track is UNKNOWN, which is honest, not broken.

    Role assignment is what makes the analysis identity-aware — resident against visitor —
    so it is worth having. But a missing checkpoint must degrade to "we do not know who this
    is" rather than to a guess, because `PersonObservation` refuses a role with no confidence
    behind it and a fabricated identity is worse than an absent one.
    """
    from behaviorsense.models.osnet import OSNetEmbedder                # noqa: PLC0415

    return OSNetEmbedder(weights, device=device)


def _gallery_from_json(raw: str):
    """Operator-owned gallery JSON -> `OpenSetGallery`, or None when unparsable.

    The file is the OPERATOR'S, written by the local backend (see web/local_backend.py) -
    never a Kaggle dataset, deliberately: these are biometric templates of a real resident,
    and the write path must stay a plain deletable file on the machine that owns the keys.
    Roles round-trip through their enum values; a name with an unknown role is skipped
    loudly rather than guessed at.
    """
    from behaviorsense.agents.perception import OpenSetGallery

    try:
        doc = json.loads(raw)
    except (TypeError, ValueError) as exc:
        print(f"[video] gallery ignored: not valid JSON ({exc})", file=sys.stderr, flush=True)
        return None
    gallery = OpenSetGallery()
    residents = doc.get("residents") if isinstance(doc, dict) else None
    if not isinstance(residents, dict):
        return None
    for name, rec in residents.items():
        try:
            centroid = np.asarray(rec["centroid"], dtype=np.float32).ravel()
            role = Role(rec.get("role", "resident"))
        except Exception as exc:                                      # noqa: BLE001
            print(f"[video] gallery entry {name!r} skipped: {exc}", file=sys.stderr,
                  flush=True)
            continue
        if centroid.size == 0:
            continue
        # A single vector enrolment is exact (see extract_observations); n_updates is
        # carried so the record does not claim to have seen one crop when it has seen many.
        entry = gallery.enrol(str(name), role, [centroid])
        entry.n_updates = int(rec.get("n_updates", 1))
    return gallery


def extract_observations(
    video: str | Path,
    *,
    rtmo: str,
    osnet: str | None = None,
    device: str = "cuda",
    stride: int | None = None,
    target_fps: float = SHARD_FPS,
    max_frames: int = MAX_FRAMES,
    osnet_device: str | None = None,
    gallery_json: str | None = None,
) -> dict[str, Any]:
    """Decode, pose, track, assign roles. Returns a JSON-safe dict. CHILD SIDE ONLY.

    `target_fps` is EXACT-RESAMPLED to, not approximated by an integer stride. `stride` is
    accepted and ignored; it is the parameter this function used to take.
    """
    import cv2                                                          # noqa: PLC0415

    path = str(video)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open {path} — unsupported container or corrupt file")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not (1.0 <= src_fps <= 240.0):
        src_fps = 30.0                    # some containers lie; 30 is the safe assumption
    # The CONTAINER's claim. Kept only to be compared against reality below - it is not what
    # the keypoints are measured in.
    claimed_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    claimed_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    width = height = 0
    # EXACT RESAMPLING, not an integer stride. This is the same bug notebook 07 already paid
    # for: it sampled with `round(src / nominal)`, every trimmed clip was 20 fps, a nominal
    # 12.5 Hz rounded to stride 2, and the corpus came out at an effective 10 Hz - a 3.0 s
    # window where training used 2.0 s. 37.5% of clips were unusable and the transition classes
    # collapsed.
    #
    # Serving had the identical defect and it was worse, because it is silent. `out_fps =
    # src_fps / 2` is 15 Hz ONLY on a 30 fps source. A 20 fps camera gave 10 Hz, so every
    # 30-frame window carried 3.0 s of motion to a model that has never seen a window longer
    # than 2.0 s - every action stretched by 1.5x. Observed: a 30 fps assumption applied to a
    # 20 fps clip of someone cooking, labelled `interacting_with_person` for 25.5 s of 30.
    #
    # `min` because a 10 fps camera cannot be resampled UP: inventing frames would be worse
    # than reporting the rate honestly, so the shortfall is declared instead (see below).
    rate = min(float(target_fps), src_fps)
    out_fps = rate

    detector = RTMODetector(rtmo, device=device)
    embedder = None
    if osnet:
        try:
            # OSNet is PyTorch; RTMO is onnxruntime. They do NOT share a device question - a
            # P100 (sm_60) against a torch built for sm_70+ runs the ONNX model fine and fails
            # every torch kernel with `no kernel image is available for execution on the device`.
            # `osnet_device` therefore defaults to the torch-safe choice rather than to `device`.
            embedder = _osnet_embedder(osnet, osnet_device or device)
        except Exception as exc:                                # noqa: BLE001
            print(f"[video] re-id disabled: {type(exc).__name__}: {exc}", file=sys.stderr)

    gallery = None
    if gallery_json and embedder is not None:
        # The gallery arrives as JSON (name -> centroid + role) because the operator's
        # machine owns it and only the vectors need to cross. Rebuilding the entry from its
        # stored centroid is exact: enrol() normalises then averages, and one normalised
        # vector is its own mean. A gallery WITHOUT an embedder is a no-op, not an error -
        # weights absent means matching cannot run, and the run degrades to asserted
        # identity exactly as it does today.
        gallery = _gallery_from_json(gallery_json)
        if gallery is not None and len(gallery):
            print(f"[video] re-id gallery: {len(gallery)} enrolled ({gallery.names})",
                  file=sys.stderr, flush=True)
    agent = PerceptionAgent(detector, embedder=embedder, gallery=gallery, fps=out_fps)
    t0 = datetime(2026, 1, 1, 9, 0, 0)

    frames: list[dict[str, Any]] = []
    read = kept = 0
    want = 0                    # index of the next SOURCE frame this rate calls for
    while kept < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        i, read = read, read + 1
        if i != want:
            continue
        # The size RTMO ACTUALLY SAW, taken from the array rather than from the container.
        # Keypoints come back in this space, so this is the only defensible thing to report as
        # `width`/`height`: the overlay divides by it. CAP_PROP_FRAME_WIDTH is the container's
        # claim and disagrees with the decoded array whenever there is a rotation matrix, a
        # non-square pixel aspect, or a plainly wrong header - and the symptom is a
        # correctly-shaped skeleton drawn beside the person instead of on them.
        height, width = frame.shape[:2]
        obs = agent.process_frame(frame, t0 + timedelta(seconds=kept / out_fps))
        frames.append({
            "i": kept,
            "t": round(kept / out_fps, 3),
            "people": [
                {
                    "track_id": p.track_id,
                    "role": p.role.value,
                    "role_confidence": round(p.role_confidence, 3),
                    "box": [round(p.box.x1, 1), round(p.box.y1, 1),
                            round(p.box.x2, 1), round(p.box.y2, 1)],
                    # None when the pose was present but too occluded to use. Reported as
                    # null rather than dropped: "tracked, pose unusable" and "not there"
                    # are different facts, and the overlay should say which.
                    "kp": None if p.keypoints is None else [
                        [round(x, KP_DECIMALS), round(y, KP_DECIMALS), round(s, 2)]
                        for (x, y), s in zip(p.keypoints.xy, p.keypoints.scores)
                    ],
                }
                for p in obs.persons
            ],
        })
        kept += 1
        want = int(round(kept * src_fps / rate))
        if kept % PROGRESS_EVERY == 0:
            # THE PARENT'S LIVENESS SIGNAL, and the reason there is no wall-clock timeout. Printed
            # to stderr because stdout carries nothing here and the parent reads the two streams
            # separately. Flushed explicitly: stderr is block-buffered when it is a pipe rather
            # than a terminal, so without this the whole point - regular output - is defeated by
            # the buffer, which is the same shape of bug as the 64 KB read that hid the heartbeats.
            print(f"[video] progress {kept}/{max_frames}", file=sys.stderr, flush=True)
    cap.release()

    if not frames:
        raise ValueError(f"decoded 0 frames from {path} — the stream is empty or unreadable")

    # POSE QUALITY, measured rather than assumed. RTMO is a one-stage model with no detector in
    # front of it, so on a cluttered scene it emits low-confidence people that are not people -
    # and the reverse, a plainly visible person whose joints all land under the threshold. Both
    # look identical on screen to "the model is broken", so the counts travel with the payload:
    # `weak_people` is how many tracked people had NO joint the pipeline would trust, and
    # `mean_kp_score` is the average confidence over the ones it would.
    n_people = n_weak = 0
    score_sum = score_n = 0.0
    for f in frames:
        for p in f["people"]:
            n_people += 1
            good = [j[2] for j in (p["kp"] or []) if j[2] >= 0.3]
            if not good:
                n_weak += 1
            else:
                score_sum += sum(good)
                score_n += len(good)

    return {
        "fps": round(out_fps, 3),
        "width": width,
        "height": height,
        # The container's own claim, carried so a mismatch is visible instead of silent.
        "container_size": [claimed_w, claimed_h],
        "size_source": "decoded_frame",
        "frames_read": read,
        "frames_kept": kept,
        "truncated": kept >= max_frames,
        "source_fps": round(src_fps, 3),
        "target_fps": float(target_fps),
        # 30 frames at this rate. The shards are 2.0 s; anything else is a temporal mismatch
        # the classifier was never trained on, so it is reported rather than absorbed.
        "window_seconds": round(30.0 / rate, 3),
        "rate_matches_shards": abs(rate - SHARD_FPS) < 0.01,
        "providers": detector.providers,
        "reid": embedder is not None,
        "person_records": n_people,
        "weak_person_records": n_weak,
        "mean_kp_score": round(score_sum / score_n, 3) if score_n else 0.0,
        "frames": frames,
        # Per-track enrolled NAME for the matched identity (the role itself already travels
        # on every person record); None for unmatched tracks is the honest answer.
        "track_names": {str(k): v for k, v in agent._track_names.items()},
        # Enrolment candidates per track, best crop first. Emitted only when an embedder
        # ran; the local half decides whether to keep them, so a clip is enrolment-ready
        # without the operator having committed to anything.
        "enrolment": ({str(tid): [[round(float(x), 5) for x in e] for e in emb]
                       for tid, emb in agent.enrolment_samples().items()}
                      if embedder is not None else {}),
    }


def rebuild_observations(payload: dict[str, Any],
                         base: datetime | None = None) -> list[FrameObservation]:
    """Compact wire format -> `FrameObservation` objects. PARENT SIDE.

    The child sends a shape sized for the overlay (rounded pixels, short keys) because the
    browser receives the same bytes and 450 frames of full Pydantic dumps is several times
    larger for no gain. This is the one function that knows how to read it back, so the
    wire format and the reconstruction cannot drift into two different beliefs about it.

    Timestamps are regenerated from `t` rather than transmitted: they exist so
    `frames_to_windows` can measure gaps, and a float offset from a fixed base carries
    exactly that information at a fraction of the size.
    """
    base = base or datetime(2026, 1, 1, 9, 0, 0)
    out: list[FrameObservation] = []
    for f in payload.get("frames", []):
        ts = base + timedelta(seconds=float(f["t"]))
        persons = []
        for p in f.get("people", []):
            x1, y1, x2, y2 = p["box"]
            kp = p.get("kp")
            persons.append(PersonObservation(
                frame_idx=int(f["i"]),
                timestamp=ts,
                track_id=int(p["track_id"]),
                box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                detection_confidence=float(p.get("role_confidence") or 0.0) or 0.9,
                keypoints=None if kp is None else Keypoints(
                    xy=[(float(j[0]), float(j[1])) for j in kp],
                    scores=[float(j[2]) for j in kp],
                ),
                role=Role(p.get("role", "unknown")),
                role_confidence=float(p.get("role_confidence") or 0.0),
            ))
        out.append(FrameObservation(frame_idx=int(f["i"]), timestamp=ts, persons=persons))
    return out


def crash_reason(returncode: int | None) -> str | None:
    """Describe an abnormal child exit, on either platform, or None for a clean error.

    The distinction matters because it is the whole justification for the subprocess: a
    child that RAISED produced a message worth showing the user, while a child that was
    KILLED produced nothing and would have taken this process with it if the work had been
    done inline. Reporting them identically hides the failure mode being defended against.

    POSIX signals a crash with a negative code (`-11` for SIGSEGV). Windows reports an NTSTATUS
    exception as a large unsigned status (`0xC0000005` for an access violation), so a check for
    `returncode < 0` alone is silently correct on Kaggle and silently wrong on the machine the
    tests run on.
    """
    if returncode is None or returncode == 0:
        return None
    if returncode < 0:
        return f"the decoder was killed by signal {-returncode}"
    if returncode >= 0xC0000000:
        return f"the decoder crashed with status 0x{returncode:08X}"
    return None


class _Stalled(Exception):
    """The child stopped reporting progress. Distinct from "the child is taking a while"."""

    def __init__(self, quiet_s: float, last: str) -> None:
        super().__init__(f"no progress for {quiet_s:.0f}s")
        self.quiet_s = quiet_s
        self.last = last


def _run_watching_progress(cmd: list[str], *, env: dict[str, str],
                           timeout: float | None, stall_s: float,
                           on_progress: Any = None,
                           poll_s: float = 0.5) -> subprocess.CompletedProcess:
    """`subprocess.run`, but giving up on SILENCE rather than on elapsed time.

    Why this replaces a timeout entirely: any wall-clock ceiling has to be set high enough for the
    longest video anyone might upload, which makes it useless for catching a hang - a wedged GPU
    then sits there for the full ceiling. Progress is the signal that separates the two, and a
    child that has reported in the last `stall_s` seconds is working no matter how long it takes.

    STDERR GOES TO A FILE, NOT A PIPE, and that is load-bearing twice over:

    * A pipe holds ~64 KB; a child that fills it while the parent is not reading blocks forever in
      `write()`. A file cannot block, so that whole deadlock class disappears along with the drain
      thread it would have needed.
    * `communicate()` reads the child's stderr ITSELF, so a drain thread and `communicate()` cannot
      both own that pipe. Measured: the drain thread received 1 of 6 progress lines because
      `communicate()` swallowed the rest - which would have left the stall timer unfed and made
      this detector silently useless. Its own bug, in the thing written to catch bugs.

    `on_progress` receives each new progress line, so a caller can turn the child's liveness into
    heartbeats of its own rather than inferring liveness from a timer.
    """
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as errf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf,
                                text=True, env=env, bufsize=1)
        out: list[str] = []
        watch = threading.Thread(target=lambda: out.extend(proc.stdout or []), daemon=True)
        watch.start()

        last_seen = 0
        last_at = time.monotonic()
        last_line = ""
        started = last_at

        def _read_new() -> str:
            nonlocal last_seen, last_at, last_line
            errf.flush()
            errf.seek(last_seen)
            chunk = errf.read()
            if not chunk:
                return ""
            last_seen += len(chunk)
            for line in chunk.splitlines():
                if "progress" in line:
                    last_at = time.monotonic()
                    last_line = line.strip()
                    if on_progress is not None:
                        on_progress(last_line)
            return chunk

        text = ""
        while True:
            try:
                proc.wait(timeout=poll_s)
                break
            except subprocess.TimeoutExpired:
                text += _read_new()
                now = time.monotonic()
                if timeout is not None and now - started > timeout:
                    proc.kill()
                    proc.wait()
                    raise
                if now - last_at > stall_s:
                    proc.kill()
                    proc.wait()
                    watch.join(timeout=2.0)
                    raise _Stalled(now - last_at, last_line) from None
        text += _read_new()
        watch.join(timeout=5.0)
        return subprocess.CompletedProcess(cmd, proc.returncode, "".join(out), text)


def extract_isolated(
    video: str | Path,
    *,
    rtmo: str,
    osnet: str | None = None,
    device: str = "cuda",
    target_fps: float = SHARD_FPS,
    max_frames: int = MAX_FRAMES,
    osnet_device: str | None = None,
    # NO WALL-CLOCK CEILING BY DEFAULT. This was 420 s, then 1800 s, and both are the same mistake
    # in different clothes: a number that has to be guessed high enough for the longest acceptable
    # video and therefore cannot distinguish a slow clip from a wedged GPU. A ten-minute upload is
    # ~7.7 min of RTMO on a P100 and perhaps three times that on a busier card, so any ceiling
    # tight enough to catch a hang also kills legitimate work.
    #
    # What replaces it is PROGRESS, not patience: the child prints a progress line as it decodes
    # and the parent gives up only when those stop (see `stall_s`). That distinguishes "long" from
    # "stuck", which a timeout never can. Pass `timeout=` explicitly if a caller genuinely needs a
    # deadline - the tests do.
    timeout: float | None = None,
    # Give up when the CHILD GOES QUIET for this long. Progress lines are emitted every
    # `PROGRESS_EVERY` frames, so 180 s is many missed reports rather than one slow moment, and it
    # holds regardless of how long the whole clip takes.
    stall_s: float = 180.0,
    python: str | None = None,
    pythonpath: str | Path | None = None,
    on_progress: Any = None,
    gallery: str | Path | None = None,
) -> dict[str, Any]:
    """Run `extract_observations` in a CHILD process. PARENT SIDE.

    A segfault in the child is a non-zero return code here, which is the entire point: the
    caller stays alive and can answer 422. Anything else - threads, signal handlers, a
    `try/except` around the decode - cannot survive a native crash, and this project has
    already proved that the expensive way.

    The child is given `PYTHONPATH` explicitly (see `child_env`). It is a fresh interpreter,
    so whatever the parent did to `sys.path` is invisible to it.
    """

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "poses.json"
        cmd = [python or sys.executable, "-m", "behaviorsense.video",
               "--video", str(video), "--out", str(out), "--rtmo", rtmo,
               "--device", device, "--target-fps", str(target_fps),
               "--max-frames", str(max_frames)]
        if osnet:
            cmd += ["--osnet", osnet]
        if osnet_device:
            cmd += ["--osnet-device", osnet_device]
        if gallery:
            # The operator's gallery, as a file the child reads. Passed by path rather than
            # inline so the command line never carries biometric data in a process listing.
            cmd += ["--gallery", str(gallery)]
        try:
            proc = _run_watching_progress(cmd, env=child_env(pythonpath),
                                          timeout=timeout, stall_s=stall_s,
                                          on_progress=on_progress)
        except _Stalled as exc:
            raise ValueError(
                f"pose extraction produced no progress for {exc.quiet_s:.0f}s "
                f"(last report: {exc.last or 'none at all'}). The clip's length is not the "
                f"problem - there is no time limit on decoding - so this is a wedged GPU, a "
                f"stream the decoder cannot advance past, or a killed child."
            ) from None
        except subprocess.TimeoutExpired:
            raise ValueError(
                f"pose extraction exceeded an explicit {timeout:.0f}s limit. Nothing sets one by "
                f"default; a caller passed `timeout=`."
            ) from None

        if proc.returncode != 0 or not out.is_file():
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            detail = tail[-1] if tail else f"exit {proc.returncode} with no output"
            crash = crash_reason(proc.returncode)
            if crash:
                detail = f"{crash} — it is not a video this decoder can read. {detail}"
            elif "No module named" in (proc.stderr or ""):
                # Distinct failure from a bad video, and the message above would have blamed
                # the upload for a deployment fault. Name the real cause.
                detail = (f"the child interpreter could not import behaviorsense. "
                          f"PYTHONPATH was {child_env(pythonpath)['PYTHONPATH']!r} and the "
                          f"package should sit in {PKG_PARENT}. {detail}")
            raise ValueError(f"pose extraction failed: {detail}")
        return json.loads(out.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rtmo", required=True)
    ap.add_argument("--osnet", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target-fps", type=float, default=SHARD_FPS)
    ap.add_argument("--osnet-device", default=None)
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    ap.add_argument("--gallery", default=None,
                    help="path to the operator's gallery JSON; enables ReID matching")
    args = ap.parse_args()

    payload = extract_observations(
        args.video, rtmo=args.rtmo, osnet=args.osnet, device=args.device,
        target_fps=args.target_fps, max_frames=args.max_frames,
        osnet_device=args.osnet_device,
        gallery_json=Path(args.gallery).read_text(encoding="utf-8") if args.gallery else None)
    Path(args.out).write_text(json.dumps(payload), encoding="utf-8")
    print(f"[video] {payload['frames_kept']} frames at {payload['fps']} Hz, "
          f"providers={payload['providers']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "MAX_FRAMES",
    "PKG_PARENT",
    "SHARD_FPS",
    "RTMODetector",
    "box_from_keypoints",
    "child_env",
    "crash_reason",
    "extract_isolated",
    "extract_observations",
    "rebuild_observations",
]
