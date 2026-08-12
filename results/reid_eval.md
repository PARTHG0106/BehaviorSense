# Open-set ReID evaluation — fitted operating point

- protocol: `msmt17:bounding_box_test: 100 enrolled ids (4.0 crops/id), 912 genuine probes (9.1/id), 1600 impostor probes from 400 unseen ids`
- embeddings: `weights/osnet_ain_x1_0_msmt17.pth`
- fit/test split: by identity, seed 0; tau fitted on fit half at FAR <= 1.0%, all reported numbers from the disjoint test half

**Fitted threshold: tau = 0.3546** (fit half: FAR 1.00%, TAR 99.08%)

## Test-half results at the fitted tau

| metric | value |
|---|---|
| FAR (stranger accepted as enrolled) | **2.25%** |
| TAR (enrolled accepted) | **96.86%** |
| DIR@1 (accepted AND correctly named) | 96.65% |
| AUROC | 0.9978 |
| EER | 2.90% (tau=0.362) |
| closed-set rank-1 (calibration smoke test) | 99.58% |
| n genuine / impostor probes | 478 / 800 |

## Operating-point table (fit half)

| FAR budget | tau | TAR | DIR@1 |
|---|---|---|---|
| 0.1% | 0.3201 | 97.24% | 97.24% |
| 1.0% | 0.3546 | 99.08% | 99.08% |
| 5.0% | 0.3914 | 99.77% | 99.77% |

## Negative control (ImageNet-only weights, same protocol)

Features never trained for re-identification must do much worse, or this
evaluation is not measuring embedding quality (real-data analogue of test R5).

| embeddings | AUROC | EER | TAR@fitted-tau | rank-1 |
|---|---|---|---|---|
| ReID-trained | 0.9978 | 2.90% | 96.86% | 99.58% |
| ImageNet-only | 0.5343 | 47.35% | 3.35% | 18.62% |

AUROC gap: **+0.4634** -> OK

## Config consequence

`PerceptionConfig.reid_match_threshold = 0.355` (was 0.30, guessed). 
Re-fit whenever the embedder checkpoint changes; this file is the procedure.