# Hallucination rate

- 1 analysed days (0 alerting) from 1 simulator scenarios
- a claim is a hallucination if it fails any of C1-C4 (deterministic, no LLM)
- `hallucination rate` is over SCHEMA-VALID claims (the verifier's view);
  `unusable rate` is over ALL emitted claims, counting schema rejects as
  unusable - which is what a caregiver actually experiences
- **HOSTED ARM**: served over OpenRouter's free tier; the provider that
  answered each report is recorded in the JSONL dump. Free models rotate
  without notice, so this is a DATED measurement of the deployed
  configuration, not a pinned checkpoint - the Qwen table remains the
  reproducible anchor.

| condition | model | reports | emitted | scorable | repaired | faithful | hallucination rate | 95% CI | unusable rate | parse fail | schema rejected |
|---|---|---|---|---|---|---|---|---|---|---|---|
| grammar-constrained | `z-ai/glm-5.2:free via nvidia/nemotron-3-super-120b-a12b:free (openrouter, 20 key(s))` | 1 | 6 | 6 | 0 | 6 | **0.0%** | 0.0%-39.0% | **0.0%** | 0 | 0 |
| unconstrained | `z-ai/glm-5.2:free via nvidia/nemotron-3-super-120b-a12b:free (openrouter, 20 key(s), schema not enforced)` | 1 | 6 | 6 | 0 | 6 | **0.0%** | 0.0%-39.0% | **0.0%** | 0 | 0 |
| unconstrained + format repair | `z-ai/glm-5.2:free via nvidia/nemotron-3-super-120b-a12b:free (openrouter, 20 key(s), schema not enforced)` | 1 | 6 | 6 | 0 | 6 | **0.0%** | 0.0%-39.0% | **0.0%** | 0 | 0 |

## Does the grammar help?

- constrained 0.0% vs free 0.0%: difference +0.0%, z = 0.00, two-sided p = 1.00 -> **indistinguishable**
- the grammar's value is a *worst-case guarantee* of parseability, not
  a faithfulness gain: it cannot emit an invalid claim, whereas the free
  arm merely happened to emit none once the prompt stated the schema
- and it is not free: see the per-report timings in the run log

## Note on the repair arm

`repair_claim` coerces FORMAT only - whitespace, `"1800"` -> `1800.0`,
`"Decreased"` -> `"decrease"`, `YYYY/MM/DD` -> `YYYY-MM-DD`. It never
invents an evidence_ref, changes a number, or infers an unstated
direction, so a repaired claim can move from unscorable to scored but
never from false to faithful. The row separates *cannot emit our JSON*
from *misstates the data*; only the latter is a faithfulness result.

**0 of 6 claims needed repair**, so this row is identical to the unconstrained one by construction. That is the finding: once the prompt states the output contract, free decoding produced no malformed claims, and the 245 schema rejections in the first run were caused by our prompt withholding the field names - not by the model.

The repair arm re-scored the unconstrained arm's cached responses, not fresh samples, so any delta is the repair alone.