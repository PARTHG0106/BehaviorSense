/* The verifier: C1-C5, deterministic, no language model.
 *
 * A faithful port of `src/behaviorsense/agents/reasoning/verifier.py` — same tolerances,
 * same word lists, same order of checks, same treatment of a percentage against a zero
 * baseline. It runs here so the report at the top of the page is really being verified
 * rather than showing a pre-baked answer.
 *
 * C5 is the newest check and the one the measured run made the case for. C2 compares the
 * structured `claimed_value` against the evidence row, so it is blind to a claim whose row
 * is right and whose SENTENCE is not — and the sentence is what a caregiver reads. In the
 * latest run C2 was zero and C5 was the largest failure category (9 of 16 constrained, 13
 * of 20 free): the model copies the figure into the field correctly, then paraphrases it
 * in the prose. Without this port the page's agreement banner would report drift against
 * the server on exactly those claims, so C5 ships here the moment it shipped in Python.
 *
 * One deliberate difference, and it is presentational only. When a citation does not
 * resolve, Python returns C2-C5 as `false` because there is nothing to compare against.
 * Reporting five failures for one fault would overstate what went wrong, so unreachable
 * checks are marked `skip` here and named as such in the margin. The verdict — the claim
 * is unfaithful and is withheld — is identical.
 */

export const REF_PATTERN = /^feat:[a-z0-9_]+:\d{4}-\d{2}-\d{2}$/;

/* Notes are HTML by design — `<b>` marks the numbers being compared — so anything
 * interpolated into them has to be escaped, and the fields that need it are exactly the
 * ones the model chose: `evidence_ref` and `direction`. This mattered the moment the page
 * started rendering LIVE claims: they arrive from an unauthenticated tunnel, having been
 * written by a hosted language model, and a claim body reading `<img onerror=…>` is not a
 * hypothetical for text nobody wrote by hand. Exported so the ledger uses the same one.
 */
export const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* displayHalfStep mirrors verifier.py: the reporter publishes values rounded to 2
 * decimals, and a digit-for-digit copy of one must verify - 2% of a small value
 * (mobility_index 0.074) is tighter than the rounding itself. */
const TOL = { valueRel: 0.02, valueAbs: 1e-6, pctAbs: 2.0, displayHalfStep: 0.005 };

const INCREASE = new Set(["increase", "increased", "increases", "rose", "rise", "risen",
  "higher", "up", "improved", "improvement", "gained", "more", "longer", "greater",
  "above", "exceeded", "grew"]);
const DECREASE = new Set(["decrease", "decreased", "decreases", "fell", "fall", "fallen",
  "lower", "down", "declined", "decline", "reduced", "reduction", "less", "fewer",
  "shorter", "below", "dropped", "drop", "worsened", "deteriorated"]);

const directionOf = (delta) =>
  Math.abs(delta) < 1e-9 ? "unchanged" : delta > 0 ? "increase" : "decrease";

/* Every numeric literal in prose, for C5. The comma form is matched FIRST in the
 * alternation so "1,800" reads as one number — a scan that matched "1" then "800" was the
 * bug the Python port hit first, and the fix is the same shape here: match the grouped
 * form, then strip the commas before parsing. The k/M scale suffix needs a negative
 * lookahead so "5 minutes" does not read as 5,000,000. */
const NUM_RE = /-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:[kKmM](?![a-zA-Z0-9]))?|-?\d+(?:\.\d+)?(?:[kKmM](?![a-zA-Z0-9]))?/g;

function numbersInText(text) {
  return [...String(text).matchAll(NUM_RE)].map((m) => {
    let s = m[0].replace(/,/g, "");
    let scale = 1;
    const suffix = s.slice(-1);
    if (suffix === "k" || suffix === "K") { scale = 1e3; s = s.slice(0, -1); }
    else if (suffix === "m" || suffix === "M") { scale = 1e6; s = s.slice(0, -1); }
    const v = parseFloat(s);
    return Number.isFinite(v) ? v * scale : null;
  }).filter((v) => v !== null);
}

/* C5 — does the prose echo the quoted value? Same 2% relative tolerance as C2, applied
 * symmetrically: a caregiver says "1,800 s" or "1.8k", and neither is a misquote. A claim
 * with no `claimed_value` has nothing to echo, so C5 is vacuous rather than failing. */
function proseQuotesValue(text, value) {
  for (const n of numbersInText(text)) {
    if (Math.abs(value) < TOL.valueAbs) {
      if (Math.abs(n) < Math.max(TOL.valueAbs, 1e-3)) return true;
      continue;
    }
    if (Math.abs(n - value) <= Math.max(TOL.valueRel * Math.abs(value),
                                        TOL.displayHalfStep)) return true;
  }
  return false;
}

function valuesAgree(claimed, actual) {
  if (Math.abs(actual) < TOL.valueAbs) return Math.abs(claimed) < Math.max(TOL.valueAbs, 1e-3);
  return Math.abs(claimed - actual) <= Math.max(TOL.valueRel * Math.abs(actual),
                                                TOL.displayHalfStep);
}

/** Direction implied by the prose. Null when absent or self-contradictory: an ambiguous
 * sentence must not be scored as a confident pass. */
function lexicalDirection(text) {
  const words = new Set((text.toLowerCase().match(/[a-z]+/g) ?? []));
  let up = false, down = false;
  for (const w of words) {
    if (INCREASE.has(w)) up = true;
    if (DECREASE.has(w)) down = true;
  }
  if (up === down) return null;
  return up ? "increase" : "decrease";
}

const fmt = (n) => {
  const abs = Math.abs(n);
  const digits = abs >= 100 || Number.isInteger(n) ? 0 : 1;
  return n.toLocaleString("en-GB", { minimumFractionDigits: digits, maximumFractionDigits: digits });
};
const pct = (n) => `${n >= 0 ? "+" : "−"}${Math.abs(n).toFixed(1)}%`;

/**
 * Verify one claim against the evidence index.
 * @returns {{checks: Array<{id: string, state: "pass"|"fail"|"skip"}>, faithful: boolean,
 *            notes: string[]}}
 */
export function verifyClaim(claim, index) {
  const notes = [];
  const mk = (c1, c2, c3, c4, c5) => [
    { id: "C1", state: c1 }, { id: "C2", state: c2 },
    { id: "C3", state: c3 }, { id: "C4", state: c4 }, { id: "C5", state: c5 },
  ];

  // C1 — reference resolution.
  if (!claim.evidence_ref) {
    return { checks: mk("fail", "skip", "skip", "skip", "skip"), faithful: false,
      notes: ["No evidence reference supplied, so nothing can be checked."] };
  }
  if (!REF_PATTERN.test(claim.evidence_ref)) {
    return { checks: mk("fail", "skip", "skip", "skip", "skip"), faithful: false,
      notes: [`Malformed reference <b>${escapeHtml(claim.evidence_ref)}</b>.`] };
  }
  const e = index[claim.evidence_ref];
  if (!e) {
    return { checks: mk("fail", "skip", "skip", "skip", "skip"), faithful: false,
      notes: [`Cites <b>${escapeHtml(claim.evidence_ref.split(":")[1])}</b>, which was not `
        + "measured today — a fabricated citation. C2 to C5 cannot be evaluated."] };
  }

  // C2 — quoted absolute value.
  let c2 = "pass";
  if (claim.claimed_value == null) {
    notes.push("No value quoted, so C2 is vacuous.");
  } else if (!valuesAgree(claim.claimed_value, e.observed_value)) {
    c2 = "fail";
    notes.push(`Quotes <b>${fmt(claim.claimed_value)}</b>, recorded value is `
      + `<b>${fmt(e.observed_value)}</b>.`);
  }

  // C3 — quoted percentage change.
  let c3 = "pass";
  if (claim.claimed_pct_change == null) {
    notes.push("No percentage quoted, so C3 is vacuous.");
  } else if (Math.abs(e.baseline_median) < 1e-9) {
    c3 = "fail";
    notes.push("Quotes a percentage against a zero baseline, where no correct value exists.");
  } else if (Math.abs(claim.claimed_pct_change - e.pct_change) > TOL.pctAbs) {
    c3 = "fail";
    notes.push(`Quotes <b>${pct(claim.claimed_pct_change)}</b>, recomputes to `
      + `<b>${pct(e.pct_change)}</b>.`);
  }
  return finishDirection(claim, e, mk, c2, c3, notes);
}

/* C4 — the check that catches inverted narration, and the reason C1-C3 are not enough.
 * Four independent ways a claim can tell the wrong story with the right number: the
 * structured field disagrees with the data, the prose disagrees with the data, the prose
 * disagrees with the field, or a quoted percentage carries the wrong sign. C5 then asks
 * the last question: does the sentence the caregiver actually reads contain the figure
 * the field records? */
function finishDirection(claim, e, mk, c2, c3, notes) {
  const actual = directionOf(e.delta);
  const lexical = lexicalDirection(claim.text);
  let c4 = "pass";

  if (claim.direction != null && claim.direction !== actual) {
    c4 = "fail";
    notes.push(`States <b>${escapeHtml(claim.direction)}</b>; the change is `
      + `<b>${actual}</b> (Δ ${e.delta > 0 ? "+" : "−"}${fmt(Math.abs(e.delta))}).`);
  }
  if (lexical != null && actual !== "unchanged" && lexical !== actual) {
    c4 = "fail";
    notes.push(`The sentence reads as <b>${lexical}</b>; the data show <b>${actual}</b>.`);
  }
  if (lexical != null && claim.direction != null && lexical !== claim.direction) {
    c4 = "fail";
    notes.push(`Self-inconsistent: prose implies <b>${lexical}</b>, field says `
      + `<b>${escapeHtml(claim.direction)}</b>.`);
  }
  if (claim.claimed_pct_change != null && Math.abs(claim.claimed_pct_change) > 1e-9
      && Math.abs(e.delta) > 1e-9
      && (claim.claimed_pct_change > 0) !== (e.delta > 0)) {
    c4 = "fail";
    notes.push(`Quoted percentage sign contradicts the measured change.`);
  }

  let c5 = "pass";
  if (claim.claimed_value == null) {
    notes.push("No value quoted, so C5 is vacuous.");
  } else if (!proseQuotesValue(claim.text, claim.claimed_value)) {
    c5 = "fail";
    notes.push(`The sentence never quotes the <b>${fmt(claim.claimed_value)}</b> the field `
      + "records — a caregiver reading the prose would take away a different figure.");
  }

  const checks = mk("pass", c2, c3, c4, c5);
  const faithful = checks.every((c) => c.state !== "fail");
  return { checks, faithful, notes: faithful ? [] : notes.filter((n) => !n.includes("vacuous")) };
}

/** Verify a whole report. */
export function verifyAll(claims, index) {
  return claims.map((claim) => ({ claim, ...verifyClaim(claim, index) }));
}


