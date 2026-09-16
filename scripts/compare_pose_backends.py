"""Compare pose backends on YOUR footage, before paying to re-extract anything.

Agent 1's keypoints are the input to everything downstream: if a joint is wrong, ST-GCN++ is
being asked to classify a body that was never there, and no amount of training fixes it. The
serving path uses **RTMO-l**, which is one-stage — it detects and poses in a single 640x640
pass. That is fast and good in crowds, and it is the weakest choice for the case this project
actually has: one person, far from the camera, lower body behind a counter.

The standard alternative is **top-down**: a person detector finds the box, the crop is resized
to fill the pose model's input, and the pose model sees a person occupying 256x192 instead of
~90x190 pixels in the corner of a letterboxed frame. Roughly four times the pixels per joint.

This script does not assume that wins. It measures it, because switching is expensive in a way
that is easy to overlook: **the pose model is part of the representation.** The shards were
extracted with RTMO, so serving with a different pose model is exactly the train/serve skew
that the 10 Hz sampling bug already was. Winning here means re-extracting Toyota (~13 h) and
Charades as well, and that is a decision to make on numbers rather than on plausibility.

    python scripts/compare_pose_backends.py --video clip.mp4 --rtmo weights/rtmo-l.onnx \\
        --out results/pose_backends.json

Needs a GPU, `rtmlib` and `onnxruntime-gpu` — so it runs in a Kaggle session (internet ON for
the top-down weights, which are downloaded on first construction), not on a laptop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behaviorsense.video import SHARD_FPS, box_from_keypoints  # noqa: E402

MIN_SCORE = 0.3          # the threshold the pipeline actually trusts a joint at
TORSO = (5, 6, 11, 12)   # shoulders and hips: present in almost any usable view


def decode(video: str, target_fps: float, max_frames: int) -> tuple[list, float, float]:
    """Exact-resampled frames, the same schedule serving uses.

    Sampling at a different rate than serving would compare two pose models on two different
    videos, which is worse than not comparing them.
    """
    import cv2

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    src = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not (1.0 <= src <= 240.0):
        src = 30.0
    rate = min(float(target_fps), src)
    out, read, kept, want = [], 0, 0, 0
    while kept < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        i, read = read, read + 1
        if i != want:
            continue
        out.append(frame)
        kept += 1
        want = int(round(kept * src / rate))
    cap.release()
    if not out:
        raise SystemExit(f"decoded 0 frames from {video}")
    return out, src, rate


def summarise(name: str, per_frame: list[list[tuple]], elapsed: float,
              frame_h: int) -> dict:
    """Per-frame (kp, scores) lists -> the numbers that decide the switch.

    `usable` is the one that matters. A pose record with no joint above the threshold renders
    as a dashed box and Agent 2 abstains on it, so a backend that emits many of them is not
    detecting people, whatever its raw count says.
    """
    records = sum(len(f) for f in per_frame)
    usable = weak = 0
    score_sum = score_n = 0
    torso_ok = 0
    heights = []
    for frame in per_frame:
        for _kp, sc in frame:
            good = sc >= MIN_SCORE
            if good.any():
                usable += 1
                score_sum += float(sc[good].sum())
                score_n += int(good.sum())
            else:
                weak += 1
            if good[list(TORSO)].all():
                torso_ok += 1
    for frame in per_frame:
        for kp, sc in frame:
            box = box_from_keypoints(kp, sc)
            if box is not None:
                heights.append(box.y2 - box.y1)
    n = len(per_frame)
    return {
        "backend": name,
        "frames": n,
        "seconds": round(elapsed, 1),
        "fps_throughput": round(n / elapsed, 1) if elapsed else None,
        "person_records": records,
        "records_per_frame": round(records / n, 2) if n else 0.0,
        "usable_records": usable,
        "weak_records": weak,
        # The headline: a frame with no usable pose is a frame Agent 2 cannot see at all.
        "frames_with_usable_pose": sum(
            1 for f in per_frame if any((sc >= MIN_SCORE).any() for _k, sc in f)),
        "mean_kp_score": round(score_sum / score_n, 4) if score_n else 0.0,
        "torso_complete_records": torso_ok,
        "median_person_height_px": round(float(np.median(heights)), 1) if heights else None,
        "median_person_height_frac": (
            round(float(np.median(heights)) / frame_h, 3) if heights and frame_h else None),
    }


def run_rtmo(frames: list, onnx: str) -> list[list[tuple]]:
    from rtmlib import RTMO

    model = RTMO(onnx_model=onnx, model_input_size=(640, 640),
                 backend="onnxruntime", device="cuda")
    sess = getattr(model, "session", None)
    provs = list(sess.get_providers()) if sess is not None else []
    if not any("CUDA" in p for p in provs):
        raise SystemExit(f"RTMO loaded on {provs}, not CUDA. Pin onnxruntime-gpu==1.26.0.")
    out = []
    for frame in frames:
        kpts, scores = model(frame)
        out.append([(np.asarray(k, np.float32).reshape(-1, 2)[:17],
                     np.asarray(s, np.float32).reshape(-1)[:17])
                    for k, s in zip(np.asarray(kpts), np.asarray(scores))])
    return out


def run_topdown(frames: list) -> list[list[tuple]]:
    """rtmlib's `Body`: RTMDet person detector + RTMPose-l on each crop.

    Downloads its own ONNX on first construction, which is why this needs internet. Same
    COCO-17 joint order as RTMO, so the two are directly comparable — that is the only reason
    this comparison means anything.
    """
    from rtmlib import Body

    model = Body(mode="performance", backend="onnxruntime", device="cuda")
    out = []
    for frame in frames:
        kpts, scores = model(frame)
        out.append([(np.asarray(k, np.float32).reshape(-1, 2)[:17],
                     np.asarray(s, np.float32).reshape(-1)[:17])
                    for k, s in zip(np.asarray(kpts), np.asarray(scores))])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--rtmo", required=True, help="path to rtmo-l.onnx")
    ap.add_argument("--out", default="results/pose_backends.json")
    ap.add_argument("--target-fps", type=float, default=SHARD_FPS)
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--skip-topdown", action="store_true")
    args = ap.parse_args()

    frames, src_fps, rate = decode(args.video, args.target_fps, args.max_frames)
    h, w = frames[0].shape[:2]
    print(f"{len(frames)} frames from {args.video}: {w}x{h}, source {src_fps:g} fps, "
          f"resampled to {rate:g} Hz (30-frame window = {30 / rate:.2f}s)\n")

    rows = []
    t0 = time.time()
    rows.append(summarise("rtmo-l (one-stage, 640x640)",
                          run_rtmo(frames, args.rtmo), time.time() - t0, h))
    if not args.skip_topdown:
        t0 = time.time()
        rows.append(summarise("rtmdet+rtmpose-l (top-down)",
                              run_topdown(frames), time.time() - t0, h))

    cols = [("backend", 30), ("frames_with_usable_pose", 6), ("records_per_frame", 6),
            ("weak_records", 6), ("mean_kp_score", 7), ("torso_complete_records", 6),
            ("median_person_height_frac", 7), ("fps_throughput", 7)]
    print(f"{'backend':<30} {'usable':>6} {'ppl/f':>6} {'weak':>6} {'kpconf':>7} "
          f"{'torso':>6} {'ht%':>7} {'fps':>7}")
    for r in rows:
        print(" ".join(f"{str(r.get(k, '—')):<{n}}" if k == "backend"
                       else f"{str(r.get(k, '—')):>{n}}" for k, n in cols))

    if len(rows) == 2:
        a, b = rows
        d_usable = b["frames_with_usable_pose"] - a["frames_with_usable_pose"]
        d_conf = b["mean_kp_score"] - a["mean_kp_score"]
        d_torso = b["torso_complete_records"] - a["torso_complete_records"]
        print(f"\ntop-down vs one-stage: {d_usable:+d} frames with a usable pose "
              f"({d_usable / max(1, len(frames)):+.1%} of the clip), "
              f"{d_conf:+.4f} mean keypoint confidence, {d_torso:+d} torso-complete records.")
        # The threshold is a judgement, stated so it can be argued with rather than buried.
        # Re-extracting Toyota and Charades is ~13 h of GPU plus a retrain; a few percent is
        # not worth that, and a large gain on the frames Agent 2 currently cannot see is.
        big = d_usable >= 0.10 * len(frames) or d_conf >= 0.05
        print("VERDICT: " + (
            "top-down is materially better. Switching means re-extracting BOTH corpora with "
            "it - the pose model is part of the representation, and changing it on the serving "
            "side alone is the same class of bug as sampling at the wrong frame rate."
            if big else
            "not a material difference on this clip. Keep RTMO on both sides; the accuracy "
            "problem is in the activity model's training corpus, not in the keypoints."))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Written, not just printed: a printed table is not a recorded one, and this is the number
    # a 13-hour re-extraction decision rests on.
    out.write_text(json.dumps({
        "video": args.video, "frame_size": [w, h], "source_fps": src_fps,
        "sample_rate_hz": rate, "window_seconds": 30 / rate,
        "min_score": MIN_SCORE, "backends": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
