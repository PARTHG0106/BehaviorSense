# BehaviorSense AI — Online Preprocessing → Offline Training Pipeline

The hard constraint: **training notebooks have no internet.** Everything — data, weights,
tokenizers, configs, and Python wheels — must pre-exist inside mounted Kaggle datasets.

---

## 1. The two-environment contract

```
╔════════════════════ STAGE A: ONLINE (Kaggle CPU/TPU, internet ON) ═════════════════╗
║                                                                                    ║
║  A1  Acquire raw data          curl/git/HF-hub → /kaggle/working/raw               ║
║  A2  Validate + inventory      checksum, duration, fps, resolution → manifest      ║
║  A3  Decode & sample frames    ffmpeg, target 15fps, short side 480                ║
║  A4  Pose extraction           RTMO → per-frame 17×3 keypoints        [BOTTLENECK] ║
║  A5  Object context            RT-DETR @1Hz → class histogram per window           ║
║  A6  Taxonomy mapping          source labels → 20-class via taxonomy.yaml          ║
║  A7  Windowing                 64-frame windows, stride 32, majority label         ║
║  A8  Normalisation             centre on pelvis, scale by torso, per-window        ║
║  A9  Split                     GroupKFold by SUBJECT (never by clip)               ║
║  A10 Pack                      float16 .npz shards ≤512MB + index.parquet          ║
║  A11 Publish                   kaggle datasets create/version                      ║
║                                                                                    ║
╠══════════════════ STAGE B: OFFLINE (Kaggle GPU 96GB, internet OFF) ════════════════╣
║                                                                                    ║
║  B0  Preflight assert          no-network check, sm_120 check, asset hash check     ║
║  B1  Mount datasets            /kaggle/input/bsai-{tensors,assets,wheels}           ║
║  B2  Train ADL model           ST-GCN++ → SkateFormer                              ║
║  B3  Train fall model          same backbone, binary head, focal loss               ║
║  B4  Fine-tune ReID            OSNet on Market-1501                                ║
║  B5  Evaluate                  cross-subject + cross-dataset + staged→wild          ║
║  B6  Export                    checkpoints + metrics.json → output dataset          ║
╚════════════════════════════════════════════════════════════════════════════════════╝
```

**Contract enforcement:** Stage B begins with an assertion that network calls fail and that every
expected asset hash matches the manifest Stage A wrote. Fail loudly in second 5, not hour 3.

---

## 2. Stage A bottleneck analysis — read this before choosing the runtime

Naive estimate: ~35h of video × 15fps ≈ **1.9M frames** needing pose extraction.

| Runtime | Throughput | Wall time | Verdict |
|---|---|---|---|
| Kaggle CPU (4 vCPU) | ~2 fps | ~11 days | **Infeasible** |
| Kaggle TPU v5e-8 (224 vCPU) | ~90 fps (CPU-parallel ONNX) | ~6 h | Feasible |
| Kaggle GPU T4 | ~120 fps | ~4.5 h | Feasible but wastes GPU quota |
| **Kaggle GPU (Blackwell), batched** | ~600 fps | **~1 h** | **Best** |

**Correction to the original plan, and it matters:** you specified pose extraction on the CPU/TPU
box. RTMO is a **CUDA-bound convnet**; TPU cannot run it without a full JAX reimplementation, and
224 vCPUs give you ~90 fps of ONNX CPU inference at best. The right split is:

- **CPU/TPU box:** download, ffmpeg decode, label mapping, packing, publishing — all genuinely
  I/O- and CPU-bound. This is where 224 vCPU / 384 GB RAM actually pays off (parallel ffmpeg is
  near-linear in cores).
- **GPU box (internet ON, separate session):** pose extraction only. ~1 h.

There is no rule that the *preprocessing* GPU session must be offline — only the *training* one.
Use a GPU session with internet for A4, publish the tensors, then run training offline. This turns
a 6-hour step into a 1-hour step.

**If you would rather keep A4 off the GPU:** export RTMO to ONNX and run `onnxruntime` with
`multiprocessing.Pool(200)` on the TPU box's 224 vCPUs. ~6 h, still acceptable, uses zero GPU quota.
Both paths are implemented; pick per quota.

---

## 3. Packed tensor format

```
bsai-tensors/                        # Kaggle dataset, PRIVATE (license compliance)
├── manifest.json                    # version, hashes, provenance, taxonomy version
├── index.parquet                    # one row per window — the queryable index
├── adl/
│   ├── train_shard_000.npz          # ≤512MB each
│   ├── ...
│   └── val_shard_000.npz
├── falls/
│   ├── staged_train_shard_000.npz
│   └── wild_test_shard_000.npz      # never trained on. Held out by construction.
└── reid/
    └── market1501_train.npz
```

Each shard:

| Array | Shape | dtype | Bytes/window |
|---|---|---|---|
| `keypoints` | (N, 64, 17, 2) | float16 | 4,352 |
| `scores` | (N, 64, 17) | float16 | 2,176 |
| `obj_ctx` | (N, 16) | float16 | 32 |
| `label` | (N,) | int8 | 1 |
| `subject_id` | (N,) | int16 | 2 |
| `source_id` | (N,) | int8 | 1 |
| `window_meta` | (N, 4) | int32 | 16 |

**≈6.6 KB per window.** 500k windows ≈ **3.3 GB total.** Compare with the source video (~25 GB) and
raw frames (multi-TB). This 1000× reduction is what makes the 4-week timeline work: training reads
3 GB from memory-mapped npz instead of decoding video.

`window_meta = [video_idx, start_frame, track_id, n_persons]` — preserves full traceability back to
source video for qualitative figures and error analysis.

---

## 4. Split policy (the mistake that invalidates results)

```python
# WRONG — same subject appears in train and val. Inflates accuracy 10-20%.
train, val = train_test_split(windows, test_size=0.2, random_state=42)

# WRONG — adjacent windows from the same clip share frames. Leakage.
train, val = train_test_split(windows, stratify=labels)

# RIGHT — group by subject, so no subject is in both splits.
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(gkf.split(X, y, groups=subject_ids))
```

Three evaluation protocols, reported separately:

| Protocol | Train | Test | Question answered |
|---|---|---|---|
| **P1 cross-subject** | 80% subjects | 20% subjects | Does it generalise to new people? |
| **P2 cross-dataset** | Charades + OmniFall-staged | GMDCSA-24 | Does it survive a new environment + camera? |
| **P3 staged→wild** | OmniFall-staged | OmniFall-wild (OOPS) | Does it survive *real* accidents? |

P3 is the headline number. It will be substantially worse than P1 — **report it anyway**. A paper
showing P1=0.92 / P3=0.61 with analysis is far more credible than one reporting only P1=0.92.

---

## 5. Offline asset bundle (`bsai-assets`)

Every file training touches. Missing one = a failed run discovered hours in.

```
bsai-assets/
├── models/
│   ├── rtmo/rtmo-l_16xb16-600e_body7-640x640.pth        # + .onnx
│   ├── rtdetr/rtdetr-l.pt
│   ├── osnet/osnet_ain_x1_0_imagenet.pth
│   └── stgcnpp/stgcnpp_ntu120_pretrained.pth            # if NTU approved
├── llm/qwen2.5-7b-instruct/                             # ~15GB bf16
│   ├── config.json  generation_config.json
│   ├── model-0000{1..4}-of-00004.safetensors
│   ├── tokenizer.json  tokenizer_config.json  vocab.json  merges.txt
│   └── special_tokens_map.json
├── wheels/                                              # ⚠️ CRITICAL FOR BLACKWELL
│   ├── torch-2.7.0+cu128-cp311-cp311-linux_x86_64.whl
│   ├── torchvision-*.whl
│   ├── outlines-*.whl        # constrained JSON decoding
│   ├── mmpose-*.whl  mmcv-*.whl  mmengine-*.whl
│   └── ...  (pip download --no-deps -r requirements-offline.txt)
├── configs/  taxonomy.yaml  model_configs/
└── ASSET_MANIFEST.json                                  # sha256 of every file
```

**The Blackwell trap, concretely:** sm_120 needs CUDA ≥12.8 / PyTorch ≥2.7. If Kaggle's image ships
an older torch, `pip install` is impossible offline. Verify on Day 1:

```python
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())
# need (12, 0) capability support → torch>=2.7, cuda>=12.8
```

If it mismatches, `pip download` the correct wheels during Stage A and `pip install --no-index
--find-links=/kaggle/input/bsai-assets/wheels` offline. Finding this in Week 3 costs a day;
finding it on Day 1 costs ten minutes.

---

## 6. Kaggle-specific operational notes

| Constraint | Handling |
|---|---|
| **12 h session cap** | Checkpoint every epoch to `/kaggle/working`; `--resume auto` finds the latest. Skeleton training is ~1–3 h so this is insurance, not routine. |
| **20 GB `/kaggle/working`** | Keep only last + best checkpoint. Delete intermediate frames immediately after pose extraction. |
| **Dataset size limits** | Shard across `bsai-tensors`, `bsai-assets`, `bsai-llm` rather than one giant dataset. Also makes versioning independent. |
| **Dataset must be PRIVATE** | Charades and Market-1501 forbid redistribution. Public would be a license violation. Publish only *code* + *derived metrics*. |
| **Non-determinism** | Seed everything; `torch.use_deterministic_algorithms(True)` where it doesn't kill throughput. Log the seed in `metrics.json`. |
| **Version pinning** | Every dataset mount is versioned (`bsai-tensors/version/3`). Never mount `latest` in a run you intend to cite. |

---

## 7. Stage A execution order (with realistic timings)

| Step | Runtime | Time | Parallelism |
|---|---|---|---|
| A1 download | CPU | 1.5 h | 8 concurrent |
| A2 validate | CPU | 10 m | 200 proc |
| A3 ffmpeg decode | CPU (224 vCPU) | 45 m | 200 proc ← where the big box pays off |
| **A4 pose extraction** | **GPU (internet on)** | **1 h** | batch 256 |
| A5 object @1Hz | GPU | 15 m | batch 128 |
| A6 taxonomy map | CPU | 5 m | single |
| A7 windowing | CPU | 20 m | 200 proc |
| A8 normalise | CPU | 10 m | vectorised |
| A9 split | CPU | 2 m | single |
| A10 pack | CPU | 25 m | 32 proc (I/O bound) |
| A11 publish | CPU | 30 m | upload |
| **Total** | | **≈5.5 h** | across 2 sessions |

Comfortably one day of wall-clock, including debugging.
