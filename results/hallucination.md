# Hallucination rate

- 116 analysed days (58 alerting) from 6 simulator scenarios
- a claim is a hallucination if it fails any of C1-C4 (deterministic, no LLM)
- `hallucination rate` is over SCHEMA-VALID claims (the verifier's view);
  `unusable rate` is over ALL emitted claims, counting schema rejects as
  unusable - which is what a caregiver actually experiences

| condition | model | reports | emitted | scorable | repaired | faithful | hallucination rate | 95% CI | unusable rate | parse fail | schema rejected |
|---|---|---|---|---|---|---|---|---|---|---|---|
| faithful stub (ceiling) | `stub-faithful` | 116 | 696 | 696 | 0 | 696 | **0.0%** | 0.0%-0.5% | **0.0%** | 0 | 0 |
| corrupted stub @ 25% | `stub-hallucinating@0.25` | 116 | 696 | 696 | 0 | 524 | **24.7%** | 21.7%-28.1% | **24.7%** | 0 | 0 |
| corrupted stub @ 50% | `stub-hallucinating@0.50` | 116 | 696 | 696 | 0 | 334 | **52.0%** | 48.3%-55.7% | **52.0%** | 0 | 0 |
| corrupted stub @ 100% | `stub-hallucinating@1.00` | 116 | 696 | 696 | 0 | 0 | **100.0%** | 99.5%-100.0% | **100.0%** | 0 | 0 |

## Which check caught it (C1-C4)

| condition | bad ref (C1) | value (C2) | pct (C3) | direction (C4) |
|---|---|---|---|---|
| corrupted stub @ 25% | 30 | 33 | 52 | 88 |
| corrupted stub @ 50% | 58 | 69 | 121 | 186 |
| corrupted stub @ 100% | 126 | 130 | 237 | 354 |

Counts are per CHECK, not per claim - one claim can fail several - so rows do not sum to the unfaithful total. C4 is the clinically dangerous mode: a correct number narrated backwards. C3 alone is a copying error.

## Sanity anchors

- faithful generator scored **0.0%** (must be 0.0%: any higher means the verifier has false positives and every rate below is inflated)
- corrupted @ 25% scored 24.7% (tracks the injection rate, so the metric is not saturating)
- corrupted @ 50% scored 52.0% (tracks the injection rate, so the metric is not saturating)
- corrupted @ 100% scored 100.0% (tracks the injection rate, so the metric is not saturating)