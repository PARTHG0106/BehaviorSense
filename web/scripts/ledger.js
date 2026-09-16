/* The ledger: renders one report as slips and runs the checks in view.
 *
 * The resolve sequence is the point of the page. Each claim's four cells settle left to
 * right, rows staggered, and a claim that fails is struck in pencil with the arithmetic
 * exposed underneath. Showing the finished verdict immediately would hide the only thing
 * worth watching: that the checking is mechanical.
 */

import { EVIDENCE, SUMMARY, RECOMMENDATION, claimsFor, SCENARIOS } from "bs/fixtures";
import { verifyAll, escapeHtml } from "bs/verifier";

const CELL_STEP = 85;   // ms between checks within a claim
const ROW_STEP  = 130;  // ms between claims

const reduced = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const num = (n) => n.toLocaleString("en-GB", { maximumFractionDigits: 1 });
const signed = (n) => `${n > 0 ? "+" : "−"}${num(Math.abs(n))}`;

/* Which evidence table the slips are being measured against. The stored day by default; the
 * backend's own index once a live report replaces it. Held here rather than threaded through
 * every function because the record line and the verifier must never disagree about it — a
 * claim checked against one table and printed against another would look like a verifier bug. */
let activeEvidence = EVIDENCE;

/* Restored verbatim when the ledger goes back to the stored day. Kept next to the live copy
 * so the two cannot drift into describing the same thing differently. */
const STORED_NOTE = "The checks below each claim are the real arithmetic, run in your browser "
  + "against the same evidence table the model was given. Faults are injected here so the "
  + "checks are visible &mdash; in the measured run, Qwen2.5&#8209;7B failed "
  + "<b>2.8&ndash;3.7%</b> of its claims unprompted, and most of those were prose that "
  + "never quoted the recorded figure (C5).";

/* The record the claim is measured against, printed under every claim whether it
 * passed or not. A verdict a reader cannot check for themselves is just another
 * assertion, which is the failure mode this whole system exists to avoid. */
function recordLine(claim) {
  const e = activeEvidence[claim.evidence_ref];
  if (!e) {
    return `<p class="slip__cite"><span>record</span>${escapeHtml(claim.evidence_ref)} — no
      such row for this day</p>`;
  }
  return `<p class="slip__cite"><span>record</span>observed ${num(e.observed_value)} ·
    baseline ${num(e.baseline_median)} · Δ ${signed(e.delta)} (${signed(e.pct_change)}%) ·
    z ${signed(e.robust_z)}</p>`;
}

function slipMarkup(result, n) {
  const { claim, checks, faithful, notes } = result;
  const cells = checks.map((c) =>
    `<span class="cell" data-check="${c.id}" data-want="${c.state}">${c.id}</span>`).join("");
  const margin = faithful ? "" : `
    <p class="slip__margin"><span class="stamp">withheld</span>${notes.join(" ")}
      ${claim.fault ? `<span class="slip__fault">fault injected: ${claim.fault}</span>` : ""}</p>`;

  // claim.text is escaped, not interpolated raw. In replay mode it is our own fixture copy;
  // in live mode it is whatever the configured model wrote, over the operator's tunnel.
  return `<li class="slip" data-verdict="${faithful ? "shown" : "withheld"}" data-resolved="0">
    <p class="slip__n">${String(n).padStart(2, "0")}</p>
    <p class="slip__text">${escapeHtml(claim.text)}</p>
    <div class="gutter" role="img" aria-label="${faithful
      ? "All five checks passed" : "Failed verification, withheld"}">${cells}</div>
    ${recordLine(claim)}
    ${margin}
  </li>`;
}

function paintTallies(results) {
  const shown = results.filter((r) => r.faithful).length;
  document.getElementById("t-total").textContent = results.length;
  document.getElementById("t-shown").textContent = shown;
  document.getElementById("t-withheld").textContent = results.length - shown;
}

let runToken = 0;

/** Settle each cell into its verified state, left to right, row by row. */
function resolve(list) {
  const token = ++runToken;
  const slips = [...list.querySelectorAll(".slip")];

  if (reduced()) {
    for (const slip of slips) {
      slip.querySelectorAll(".cell").forEach((c) => { c.dataset.state = c.dataset.want; });
      slip.dataset.resolved = "1";
    }
    return;
  }

  slips.forEach((slip, row) => {
    const cells = [...slip.querySelectorAll(".cell")];
    cells.forEach((cell, i) => {
      window.setTimeout(() => {
        if (token !== runToken) return;
        cell.dataset.state = cell.dataset.want;
      }, row * ROW_STEP + i * CELL_STEP);
    });
    window.setTimeout(() => {
      if (token !== runToken) return;
      slip.dataset.resolved = "1";
    }, row * ROW_STEP + cells.length * CELL_STEP);
  });
}

function render(scenario) {
  activeEvidence = EVIDENCE;
  const claims = claimsFor(scenario);
  const results = verifyAll(claims, EVIDENCE);
  const list = document.getElementById("slips");

  list.innerHTML = results.map((r, i) => slipMarkup(r, i + 1)).join("");
  paintTallies(results);
  document.getElementById("summary").textContent = `${SUMMARY} ${RECOMMENDATION}`;
  return list;
}

/* Render a report the backend just generated.
 *
 * The claims, the evidence table and the server's own verdicts all arrive together, and this
 * page re-derives the verdicts from the evidence rather than displaying the ones it was sent.
 * Then it compares. Agreement between two independent implementations — Python on a GPU,
 * JavaScript in a browser — is worth far more than either one asserting a number, and
 * disagreement is worth knowing immediately, so it is reported rather than smoothed over.
 */
export function renderLive(payload) {
  activeEvidence = payload.evidence || {};
  const claims = payload.claims || [];
  const results = verifyAll(claims, activeEvidence);
  const list = document.getElementById("slips");

  list.innerHTML = results.length
    ? results.map((r, i) => slipMarkup(r, i + 1)).join("")
    : `<li class="slip" data-verdict="withheld" data-resolved="1"><p class="slip__text">The
       model emitted no schema-valid claims for this day.</p></li>`;
  paintTallies(results);
  document.getElementById("summary").textContent =
    `${payload.summary || ""} ${payload.recommendation || ""}`.trim();

  const day = document.getElementById("ledger-day");
  if (day && payload.day) {
    day.dateTime = payload.day;
    day.textContent = new Date(payload.day + "T00:00:00").toLocaleDateString("en-GB",
      { day: "numeric", month: "long", year: "numeric" });
  }

  // Compare our verdicts with the server's, claim by claim. `prose_quoted_value !== false`
  // rather than a truthiness check so a payload from a build older than C5 (field absent,
  // undefined) still counts as agreement instead of manufacturing drift.
  const theirs = new Map((payload.verifications || []).map((v) => [v.claim_id, v]));
  let compared = 0, agreed = 0;
  for (const r of results) {
    const v = theirs.get(r.claim.claim_id);
    if (!v) continue;
    compared += 1;
    const serverFaithful = v.ref_exists && v.value_matches && v.pct_matches
      && v.direction_consistent && v.prose_quoted_value !== false;
    if (serverFaithful === r.faithful) agreed += 1;
  }

  const banner = document.getElementById("ledger-live");
  if (banner) {
    const withheld = results.filter((r) => !r.faithful).length;
    const rate = results.length
      ? `${((withheld / results.length) * 100).toFixed(0)}%` : "—";
    const parts = [
      `<b>Live</b> · ${payload.model || "model"} generated ${payload.emitted ?? results.length}`,
      `claim${(payload.emitted ?? results.length) === 1 ? "" : "s"} for ${payload.day || "this day"}`,
      `· ${withheld} withheld (${rate})`,
    ];
    if (payload.schema_rejected) parts.push(`· ${payload.schema_rejected} rejected by schema`);
    banner.innerHTML = compared && agreed === compared
      ? `${parts.join(" ")} · the ${compared} checks below were re-run in this browser and
         <b>agree with the server on every claim</b>.`
      : compared
        ? `${parts.join(" ")} · <b>${compared - agreed} of ${compared} verdicts disagree with
           the server</b> — the two verifier implementations have drifted apart and the
           numbers on this page should not be trusted until that is resolved.`
        : `${parts.join(" ")} · the server returned no verdicts to compare against.`;
    banner.dataset.tone = compared && agreed !== compared ? "disagree" : "";
    banner.hidden = false;
  }

  const note = document.getElementById("ledger-note");
  if (note) {
    // Names the model the payload reported rather than hard-coding "Qwen2.5-7B". The dev
    // stub in web/dev_backend.py returns "dev-stub (not a model)", and a page that claimed
    // a 7B had spoken when it had not would be exactly the unverified assertion this whole
    // system exists to refuse.
    const model = payload.model || "the backend";
    note.innerHTML = `These claims were produced by <b>${escapeHtml(model)}</b> moments ago, `
      + "and the checks below each one were run here, in your browser, against the evidence "
      + "table the model was given. Nothing on this line is pre-computed.";
  }
  return list;
}

/** Wire the fault controls and run the first pass. */
export function mountLedger() {
  const chips = [...document.querySelectorAll(".chip[data-rate]")];
  let started = false;

  const show = (rate, animate) => {
    const list = render(SCENARIOS[rate] ?? "some");
    if (animate) resolve(list); else {
      list.querySelectorAll(".cell").forEach((c) => { c.dataset.state = c.dataset.want; });
      list.querySelectorAll(".slip").forEach((s) => { s.dataset.resolved = "1"; });
    }
  };

  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      chips.forEach((c) => c.setAttribute("aria-pressed", String(c === chip)));
      // Returning to the stored day has to clear the live banner and restore the note, or
      // the page keeps claiming a provenance it no longer has.
      const banner = document.getElementById("ledger-live");
      if (banner) banner.hidden = true;
      const note = document.getElementById("ledger-note");
      if (note) note.innerHTML = STORED_NOTE;
      document.getElementById("chip-live")?.setAttribute("aria-pressed", "false");
      show(Number(chip.dataset.rate), true);
    });
  });

  // Render immediately so there is never an empty panel, but hold the resolve until the
  // ledger is actually on screen — an animation that plays above the fold is a wasted one.
  const initial = Number(
    document.querySelector('.chip[aria-pressed="true"]')?.dataset.rate ?? 0.34);
  show(initial, false);
  const list = document.getElementById("slips");
  list.querySelectorAll(".cell").forEach((c) => { c.dataset.state = ""; });
  list.querySelectorAll(".slip").forEach((s) => { s.dataset.resolved = "0"; });

  // Observe the claim LIST at a near-zero threshold, not the whole ledger at 25%. The
  // ledger is taller than a phone viewport, so a fractional threshold on it can never be
  // met and the checks silently never resolved on mobile.
  const io = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting && !started) {
        started = true;
        resolve(document.getElementById("slips"));
        io.disconnect();
      }
    }
  }, { threshold: 0.01 });
  io.observe(list);

  return {
    /** Swap the stored day for one the backend just generated. */
    showLive(payload) {
      chips.forEach((c) => c.setAttribute("aria-pressed", "false"));
      document.getElementById("chip-live")?.setAttribute("aria-pressed", "true");
      started = true;                   // the observer must not re-resolve over this
      resolve(renderLive(payload));
    },
  };
}

