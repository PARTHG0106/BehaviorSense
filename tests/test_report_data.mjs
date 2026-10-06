import test from "node:test";
import assert from "node:assert/strict";
import { demoReport, videoReport } from "../web/scripts/report-data.js";
import { createTranslationQueue } from "../web/scripts/translation-queue.js";

const checked = { claim_id: "c1", text: "Walking duration was 30 s.",
  evidence_ref: "feat:walking_duration_s:2026-10-06", claimed_value: 30,
  claimed_pct_change: null, direction: null, faithful: true,
  C1_ref_exists: true, C2_value_matches: true, C3_pct_matches: true,
  C4_direction_consistent: true, C5_prose_quoted_value: true };
const payload = (checks) => ({ checks, stages: [
  { agent: 3, status: "done", payload: { observed_hours: 0.1, is_reliable: false,
    baseline_provenance: "reference_persona" } },
  { agent: 4, status: "done", payload: { summary: "A short observation.",
    recommendation: "Review it.", claims_emitted: 5 } },
] });

test("caregiver video excludes failed, missing-check, hidden and ambiguous claims", () => {
  const missing = { ...checked, claim_id: "missing" }; delete missing.C5_prose_quoted_value;
  const result = videoReport(payload([checked, missing,
    { ...checked, claim_id: "failed", C4_direction_consistent: false },
    { ...checked, claim_id: "hidden", shown_to_caregiver: false }]));
  assert.deepEqual(result.claims.map((c) => c.claim_id), ["c1"]);
  assert.equal(result.totalClaims, 5); assert.equal(result.withheldClaims, 4);
  assert.equal(result.isReliable, false); assert.equal(result.provenance, "simulated");
  assert.equal(result.escalate, undefined);
  assert.equal(result.day, "2026-10-06");
  const fixture = payload([checked]); fixture.stages[1].payload.model = "dev-stub (not a model)";
  assert.equal(videoReport(fixture).provenance, "demo");
  assert.deepEqual(videoReport(payload([checked, { ...checked }])).claims, []);
});

test("failed report stages cannot publish a leftover summary or checked claim", () => {
  const input = payload([checked]); input.stages[1].status = "failed";
  input.stages[1].payload.escalate = true;
  const output = videoReport(input);
  assert.equal(output.summary, ""); assert.equal(output.recommendation, "");
  assert.deepEqual(output.claims, []); assert.equal(output.escalate, undefined);
});

test("demo independently verified claims respect server rejection and schema counts", () => {
  const claim = { ...checked };
  const input = { summary: "Sample", emitted: 3, evidence: {}, verifications: [
    { claim_id: "c1", ref_exists: true, value_matches: true, pct_matches: true,
      direction_consistent: false, prose_quoted_value: true },
  ] };
  const before = JSON.stringify(input);
  const report = demoReport(input, [{ claim, faithful: true }]);
  assert.deepEqual(report.claims, []); assert.equal(report.withheldClaims, 3);
  assert.equal(report.escalate, undefined); assert.equal(JSON.stringify(input), before);
  assert.equal(demoReport({ emitted: 1 }, [{ claim, faithful: true }]).claims.length, 1);
});

test("translations serialize shared backend access and skip superseded languages", async () => {
  const started = [], complete = [];
  let language = "hi";
  const queue = createTranslationQueue((value) => {
    started.push(value);
    return new Promise((resolve, reject) => complete.push({ resolve, reject }));
  }, (value) => value.language === language);
  const first = queue({ language: "hi" });
  await new Promise((resolve) => setImmediate(resolve));
  language = "mr";
  const next = queue({ language: "mr" });
  assert.equal(started.length, 1);
  complete[0].reject(new Error("first provider failed"));
  await assert.rejects(first, /provider failed/);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(started.length, 2);
  const obsolete = queue({ language: "hi" });
  const obsoleteCheck = assert.rejects(obsolete, /superseded/);
  complete[1].resolve({ summary: "मराठी अहवाल" });
  assert.deepEqual(await next, { summary: "मराठी अहवाल" });
  await obsoleteCheck;
  assert.equal(started.length, 2);
});
