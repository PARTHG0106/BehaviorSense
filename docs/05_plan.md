# 05 — Execution plan (written 2026-08-04, ~3 weeks remaining)

Status snapshot and the plan to completion. Follows from decisions locked in docs 00–04;
nothing here reopens them.

---

## Where we are

| Component | State | Evidence |
|---|---|---|
| Agent 3 behaviour | ✅ done | 16/16 tests; 100% recall, 4d median latency, 5.07 FP/100d ([results/tuning_log.md](../results/tuning_log.md)) |
| Agent 4 verifier | ✅ done | 12/12 tests; C1–C4 checks incl. inverted narration |
| Simulator + eval harness | ✅ done | 28-scenario suite ([results/behaviour_eval.md](../results/behaviour_eval.md)) |
| Agent 1 perception | ✅ done | 19/19 tests: two-threshold tracking, occlusion coasting (+ measured association ceiling), open-set gallery, role voting/hysteresis |
| Open-set ReID protocol | ✅ done | `data/reid_datasets.py`; MSMT17 split verified: 100 enrolled / 400 impostor ids, 2,912 crops; leakage/rigging rejected by `validate()` |
| ReID metrics | 🟡 9/12 tests | `eval/reid_eval.py`; 3 failures diagnosed, all test-fixture bugs (below) |
| `reid_match_threshold` | ❌ still a guess | 0.30 placeholder; the whole point of the ReID eval work |
| Agent 2 activity | ⬜ | the only remaining *trained* component — now the critical path |
| Stage A preprocessing | ⬜ | scripts not written |
| Training scripts | ⬜ | |
| Agent 4 reporter (LLM) | ⬜ | verifier done, generator not |
| FastAPI / dashboard / Docker | ⬜ | week 4 |

**Dataset decision (this session):** MSMT17 (university-provided) is adopted as the
**cross-dataset open-set benchmark** — 15 cameras, indoor+outdoor, 4 days × 3 time slots
(the only source that tests appearance drift across days, which is the deployment failure
mode). Market-1501 stays the **reproducible primary** because a third party can fetch it
from Kaggle immediately. Derived embeddings for both stay **private**. An ethics note
distinguishing MSMT17 from the excluded DukeMTMC-reID goes in the limitations section.

---

## Step 0 — close the ReID loop (½ day, do first)

The three failing tests are fixture bugs, not code bugs; the informative-regime sweep is
already measured:

| separability | AUROC | EER | TAR@FAR=0.1% | TAR@FAR=10% |
|---|---|---|---|---|
| 0.35 | 0.806 | 26.5% | 22.5% | 50.8% |
| 0.40 | 0.942 | 13.6% | 54.2% | 84.2% |
| 0.45 | 0.996 | 2.0% | 87.5% | 99.2% |
| 0.60+ | 1.000 | ~0% | saturated | saturated |

1. **R6 / R7:** both used separability 0.6–0.65 where the synthetic embedder is *perfect*
   — EER=0 and every FAR budget buys the same TAR, so the tests' premises can't hold.
   Move both to separability **0.40**, where the trade-off actually exists.
2. **R3b:** the reused-crop fixture reuses the enrolment crop verbatim (camera c1), so the
   same-camera check fires before the shared-path check it was meant to exercise. Fix:
   same *path*, camera c2.
3. **Embedding extraction can run locally — no Kaggle wait.** `osnet_ain_x1_0` is ~2M
   params; 2,912 crops on this CPU is minutes. Write
   `scripts/extract_reid_embeddings.py` (crop manifest → `.npz` keyed by path) and
   `scripts/eval_reid.py` (fit τ on the fit half at FAR ≤ 1%, report on the test half,
   both datasets, cross-dataset transfer table).
4. **Deliverables:** `results/reid_eval.md` (tuning-log style), measured
   `reid_match_threshold` + `reid_margin` in `PerceptionConfig` with the ROC citation
   replacing the "must be re-fitted" apology, README row.

Definition of done: 12/12 tests; τ fitted on one dataset, reported on the other; the
config value traceable to a table.

## Step 1 — Agent 2 activity recognition (week 2, critical path)

The only component left that needs training, therefore the only one exposed to Kaggle
risk. Same interface discipline as Agent 1: the research-bearing logic must run on CPU.

- `agents/activity.py`: ST-GCN++ behind a `Protocol`; windowing (2s @ 15fps, stride 1s);
  **HMM/Viterbi smoothing** over per-window logits (transition matrix from taxonomy
  priors); **object-context late fusion** (RT-DETR labels reweight ambiguous classes —
  the `taking_medication` vs `drinking` split needs a pill bottle, not a better GCN).
- Emits `ActivitySegment` only; `aggregate_daily_features` already consumes it (T8).
- Tests with synthetic logits: smoothing must fix isolated misclassifications
  (control: and must *not* erase a real 1-window fall — the anti-vacuous pair);
  fusion must flip medication/drinking only when the object is present; segment
  boundaries exact; abstention below confidence floor.

## Step 2 — Kaggle asset staging (parallel, start now)

**No-internet training means an asset missed today costs a 12h session next week.**
Build `scripts/kaggle_manifest.py` + checklist dataset now: RTMO-l, OSNet-AIN,
RT-DETR-L, ST-GCN++ (NTU-pretrained), Qwen2.5-7B-Instruct weights, tokenizers, configs.
Verify each loads offline in a fresh Kaggle notebook *before* training week.

## Step 3 — Stage A preprocessing + training (week 3)

- Stage A scripts: download/decode → RTMO pose extraction → taxonomy mapping
  (`scripts/build_charades_map.py` is still owed) → windowed skeleton shards (npz).
- `train_adl.py`, `train_fall.py`: bf16, large batch, **checkpoint/resume mandatory**
  (12h limit), cross-subject P1 / cross-dataset P2 / staged→wild P3 protocols.
- Optional ablation only if time: OSNet fine-tune on MSMT17 train ids.

## Step 4 — Agent 4 reporter + integration (week 3→4)

- `agents/reasoning/reporter.py`: Qwen2.5-7B + `outlines` grammar-constrained JSON →
  existing verifier. Headline: hallucination rate, constrained vs unconstrained.
- End-to-end replay demo (simulator → Agent 3 → reporter → verifier) is the safety net
  demo that cannot be blocked by GPU trouble.
- Week 4: FastAPI service, minimal dashboard, Docker, SQLite schema, docs pass.

---

## Risks

| Risk | Mitigation |
|---|---|
| Blackwell/torch wheel mismatch on Kaggle | smoke-test a 10-min training step in week 2, not week 3 |
| Missing offline asset discovered mid-session | Step 2 checklist verified in a throwaway notebook first |
| Charades scale (~76 GB) vs preprocessing quota | pose-extract on TPU/CPU in shards; only skeletons reach the GPU dataset |
| Qwen2.5-7B weights ~15 GB | stage early; fallback Qwen2.5-3B acceptable (reporter quality is not the headline — verifier catches faults either way) |
| Agent 2 accuracy below usable | HMM smoothing + abstention already designed in; behaviour layer tolerates missing windows (`is_reliable` gate) |

**Ordering rationale:** finish the ReID loop first because it is half a day from done and
retires the config's weakest number; Agent 2 next because everything downstream is
already built and tested against its output schema; assets staged in parallel because
that risk has a week of latency; UI last because it demonstrates rather than proves.
