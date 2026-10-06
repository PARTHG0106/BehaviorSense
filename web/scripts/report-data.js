/* Adapters for the caregiver view. Only accepted, unambiguous claims cross this seam.
 * Keep the original report and evidence untouched for the English audit. */

const VIDEO_CHECKS = ["C1_ref_exists", "C2_value_matches", "C3_pct_matches",
  "C4_direction_consistent", "C5_prose_quoted_value"];
const SOURCE_CHECKS = ["ref_exists", "value_matches", "pct_matches",
  "direction_consistent", "prose_quoted_value"];

function uniqueClaims(claims) {
  const counts = new Map();
  for (const claim of claims) counts.set(claim.claim_id, (counts.get(claim.claim_id) || 0) + 1);
  return claims.filter((claim) => claim.claim_id && counts.get(claim.claim_id) === 1);
}

export function demoReport(payload, results) {
  const server = new Map((payload.verifications || []).map((v) => [v.claim_id, v]));
  const accepted = uniqueClaims(results.map((result) => result.claim)).filter((claim) => {
    const result = results.find((r) => r.claim === claim);
    const verdict = server.get(claim.claim_id);
    return result.faithful && (!verdict || SOURCE_CHECKS.every((key) => verdict[key] === true));
  });
  const total = Math.max(results.length, Number(payload.emitted) || 0);
  return {
    summary: payload.summary || "", recommendation: payload.recommendation || "",
    claims: accepted, evidence: payload.evidence || {}, totalClaims: total,
    withheldClaims: total - accepted.length,
    escalate: typeof payload.escalate === "boolean" ? payload.escalate : undefined,
    day: payload.day, provenance: "demo",
  };
}

export function videoReport(payload) {
  const stages = payload.stages || [];
  const behaviour = stages.find((s) => s.agent === 3)?.payload || {};
  const reportStage = stages.find((s) => s.agent === 4);
  const report = reportStage?.payload || {};
  const checks = payload.checks || [];
  const claims = reportStage?.status === "done"
    ? uniqueClaims(checks).filter((c) => c.faithful === true
      && c.shown_to_caregiver !== false && VIDEO_CHECKS.every((key) => c[key] === true)) : [];
  const total = Math.max(checks.length, Number(report.claims_emitted) || 0);
  const days = [...new Set(claims.map((claim) =>
    /:(\d{4}-\d{2}-\d{2})$/.exec(claim.evidence_ref || "")?.[1]).filter(Boolean))];
  return {
    summary: reportStage?.status === "done" ? report.summary || "" : "",
    recommendation: reportStage?.status === "done" ? report.recommendation || "" : "",
    claims, evidence: payload.evidence || {}, totalClaims: total,
    withheldClaims: total - claims.length,
    escalate: reportStage?.status === "done" && typeof report.escalate === "boolean" ? report.escalate : undefined,
    day: days.length === 1 ? days[0] : behaviour.day || payload.day,
    provenance: report.model === "dev-stub (not a model)" ? "demo"
      : behaviour.baseline_provenance === "resident_history" ? "resident_history" : "simulated",
    observedHours: behaviour.observed_hours,
    isReliable: behaviour.is_reliable === true,
  };
}
