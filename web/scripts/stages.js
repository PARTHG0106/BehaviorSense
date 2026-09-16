/* Four agents over one uploaded clip, each one's output shown where it was produced.
 *
 * The backend returns `stages`: one record per agent, in order, with its own status, timing
 * and payload, plus `checks` — C1 to C5 on every individual claim. This module renders that
 * and nothing else. It does not compute, infer or fill in; if a stage says `skipped`, the
 * card says skipped and gives the reason the backend gave.
 *
 * Two things are load-bearing and neither is decoration.
 *
 * **Provenance travels with the number.** Agent 3's deviation is a robust-z against a
 * 14-day rolling median, and one upload is a single moment. The backend therefore runs the
 * stage against a DECLARED simulated reference and says so in `baseline_provenance`. That
 * field is rendered as a banner above the figures rather than a footnote below them,
 * because a reference-relative z-score read as "this person has declined" is precisely the
 * confusion the whole verification layer exists to prevent. The feature VALUES are measured
 * from the video and are real; the comparison point is not this person.
 *
 * **The verification block reuses the ledger's own markup.** Same `.slip`, same `.gutter`,
 * same `.cell[data-state]`, same withheld stamp. An examiner looking at their own footage
 * sees the identical visual grammar as the stored day, so "verified" means the same thing in
 * both places — which it could not if this view invented a second way to draw a passing
 * check.
 */

import { escapeHtml } from "bs/verifier";

const num = (v, d = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d);
const secs = (n) => `${Number(n).toFixed(1)}s`;
const int = (v) => (v === null || v === undefined ? "—" : Number(v).toLocaleString());

/* Seam labels are the actual schema types, matching the pipeline band above. The point of
 * naming them here is that the reader can see the SAME type name on the static diagram and
 * on their own clip's result — the diagram is describing this run, not an idealised one. */
const SEAMS = {
  1: { emits: "PersonObservation", to: "Agent 2 receives one per person per frame" },
  2: { emits: "ActivitySegment", to: "Agent 3 receives the segments, per identity" },
  3: { emits: "BehaviourState", to: "Agent 4 receives numbers only — never pixels" },
  4: { emits: "CaregiverReport", to: "every claim goes to C1–C5 before anyone sees it" },
};

const STATUS_TONE = { done: "ok", skipped: "skip", failed: "bad" };

/* One row per track, bars positioned from the CLIP's length rather than each track's, so
 * two people's timelines line up and can be compared — the whole point of an
 * identity-aware system. Falls are struck in pencil, the page's only saturated hue. */
function timelineMarkup(track, span) {
  const bars = (track.segments || []).map((s) => {
    const left = (s.t0 / span) * 100;
    const width = Math.max(0.6, ((s.t1 - s.t0) / span) * 100);
    const fall = s.name === "falling" || s.name === "fallen_on_ground";
    return `<span class="seg${fall ? " seg--fall" : ""}"
      style="left:${left.toFixed(2)}%;width:${width.toFixed(2)}%"
      title="${escapeHtml(s.name)} · ${secs(s.t0)}–${secs(s.t1)} · confidence ${
        s.confidence}${s.room ? ` · ${escapeHtml(s.room)}` : ""}"></span>`;
  }).join("");

  const named = (track.segments || [])
    .map((s) => `${escapeHtml(s.name)} ${secs(s.t1 - s.t0)}`).slice(0, 6).join(" · ");
  const role = track.role === "unknown" ? "unidentified" : track.role;
  // The enrolled NAME, when the gallery matched. "resident" alone says a match happened;
  // the name is what a caregiver reads, and "Mary" beats "resident" on a demo stage.
  const matched = track.matched_name
    ? ` <span class="track__subject">${escapeHtml(track.matched_name)}</span>` : "";
  const falls = track.fall_segments
    ? ` <span class="track__fall">${track.fall_segments} fall segment${
        track.fall_segments === 1 ? "" : "s"}</span>`
    : "";
  // The subject marker is separate from the role on purpose. On an uploaded clip the role is
  // honestly `unidentified` and the subject is an assertion; collapsing the two into
  // "resident" would have the system claim its re-identifier recognised somebody it did not.
  const subject = track.is_subject
    ? ` <span class="track__subject">subject of Agent 3's features</span>` : "";

  return `<div class="track" data-role="${escapeHtml(track.role || "unknown")}"
      ${track.is_subject ? 'data-subject="1"' : ""}>
    <p class="track__id"><b>${escapeHtml(role)}</b> <span>track ${track.track_id} ·
      ${track.n_segments ?? (track.segments || []).length} segment${
        (track.n_segments ?? 0) === 1 ? "" : "s"}</span>${matched}${subject}</p>
    <div class="track__bar">${bars}</div>
    <p class="track__list">${named || "no segments"}${falls}</p>
  </div>`;
}

/* A definition row: label on the left, measured figure on the right, mono and tabular. */
const rows = (pairs) => `<ul class="readout__rows">${pairs
  .map(([k, v]) => `<li><span>${escapeHtml(k)}</span><b>${v}</b></li>`).join("")}</ul>`;

/* WHY re-identification did or did not run. "re-id off" on a request that asked to enrol is
 * indistinguishable from a request that never asked, and that ambiguity cost a debugging
 * round-trip: a first enrolment produced no card at all, and the page could not say whether
 * the name had failed to arrive, the weights were missing, or OSNet had crashed. A function
 * rather than more nested ternaries inside the template - the nested version put two string
 * literals across line breaks, which `node --check` accepts for a module and the browser
 * does not. */
function reidNote(payload) {
  const note = payload.reid_note;
  if (payload.reid || !note) return "";
  if (note.requested) {
    const why = note.weights_present
      ? "The weights are attached, so this is a loading failure — the notebook's own output "
        + "has the traceback."
      : "The OSNet weights are not attached to this Kaggle session, so matching and "
        + "enrolment cannot run at all.";
    return `<p class="stage__flag"><b>Re-identification was requested but did not run.</b> `
      + `${escapeHtml(why)}</p>`;
  }
  if (note.weights_present === false) {
    return `<p class="stage__flag">No OSNet weights in this session: identity can only ever `
      + `be asserted here, never recognised.</p>`;
  }
  return "";
}

function perceptionBody(p, payload) {
  const people = payload.n_people ?? 0;
  const cuda = (payload.providers || []).some((s) => /CUDA/i.test(s));
  const who = (payload.tracks || []).map((t) =>
    `${escapeHtml(t.role === "unknown" ? "unidentified" : t.role)} · track ${t.track_id}`);
  // Pose quality is stated, not implied. RTMO is one-stage with no detector in front of it, so
  // a cluttered kitchen produces low-confidence people that are not people, and loses real ones
  // whose joints all fall below the trust threshold. Both render as a wrong-looking overlay, and
  // a viewer cannot tell them apart — or tell either from a broken install — unless the numbers
  // are on the card.
  const weak = payload.weak_person_records, recs = payload.person_records;
  const weakShare = recs ? weak / recs : 0;
  const cs = payload.container_size, ds = payload.decoded_size;
  const sizeMismatch = cs && ds && cs[0] && (cs[0] !== ds[0] || cs[1] !== ds[1]);
  return `
    <p class="stage__body">Decoded ${int(payload.frames_kept)} frames at
      ${num(payload.fps, 1)} Hz and ran RTMO over every one of them.
      ${people} ${people === 1 ? "track was" : "tracks were"} produced.
      ${people > 1
        ? "That is track IDs, not people: a confirmed identity keeps its ID through detection "
          + "gaps of up to about 15 s, and a longer gap — or a person never confirmed — still "
          + "mints a new one. Agent 2 says which of them were merged into one subject."
        : ""}
      ${payload.reid
        ? "OSNet re-identification is on, so a role is a fitted decision rather than a guess."
        : "Re-identification is off — every track is reported as unidentified, which is "
          + "honest rather than broken."}</p>
    ${reidNote(payload)}
    ${sizeMismatch ? `<p class="stage__flag"><b>The container and the decoder disagree on the
      frame size.</b> The header says ${cs[0]}&times;${cs[1]}, the decoded frame is
      ${ds[0]}&times;${ds[1]}. Keypoints are measured in the decoded size, which is what the
      overlay scales from.</p>` : ""}
    ${rows([
      ["Frames kept", int(payload.frames_kept)],
      ["Source rate", payload.source_fps ? `${num(payload.source_fps, 1)} fps` : "—"],
      ["Sample rate", `${num(payload.fps, 1)} Hz`],
      ["Window length", payload.window_seconds
        ? `${num(payload.window_seconds, 2)} s (30 frames)` : "—"],
      ["Track IDs produced", int(people)],
      ["Decoded size", ds && ds[0] ? `${ds[0]}&times;${ds[1]}` : "—"],
      ["Mean keypoint confidence", payload.mean_kp_score === undefined
        ? "—" : num(payload.mean_kp_score, 3)],
      ["Unusable pose records", recs === undefined || recs === null
        ? "—" : `${int(weak)} of ${int(recs)} (${(weakShare * 100).toFixed(0)}%)`],
      ["Execution provider", cuda ? "CUDA" : escapeHtml((payload.providers || ["—"])[0])],
      ["Truncated", payload.truncated ? "yes — clip hit the frame cap" : "no"],
    ])}
    ${payload.rate_matches_shards === false ? `<p class="stage__flag"><b>This clip is sampled at
      ${num(payload.fps, 1)} Hz, not the 15 Hz the classifier was trained at.</b> Its
      ${num(payload.window_seconds, 2)} s windows carry
      ${num(payload.window_seconds / 2, 2)}&times; the motion of the 2.0 s windows ST-GCN++ has
      seen, so Agent 2's labels below are outside the conditions any measured number covers. The
      source is only ${num(payload.source_fps, 1)} fps — resampling up would mean inventing
      frames, which would be worse than saying this.</p>` : ""}
    ${weakShare > 0.25 ? `<p class="stage__flag"><b>${(weakShare * 100).toFixed(0)}% of pose
      records carry no joint above the 0.3 confidence threshold.</b> Those people are drawn as
      dashed boxes rather than skeletons, and Agent 2 abstains on them rather than guessing.
      RTMO has no person detector in front of it, so a cluttered or distant scene degrades this
      way — the pose model is the limit here, not the classifier.</p>` : ""}
    ${who.length ? `<p class="stage__who">${who.join(" · ")}</p>` : ""}`;
}

function activityBody(p, payload, span) {
  const streams = (payload.streams_served || []).join(" + ") || "—";
  const tracks = payload.per_track || [];
  const asserted = payload.subject_provenance
    && payload.subject_provenance !== "reid_matched";
  const merged = (payload.subject_tracks || []).length > 1
    ? (payload.subject_tracks || []).join(", ") : "";
  return `
    <p class="stage__body">Each person's windows are classified on their own skeletons,
      independently — nobody's activity is inferred from anybody else's. The served
      configuration is the one notebook 04 selected, not the defaults:
      <code>${escapeHtml(streams)}</code> at &tau;=${num(payload.tau, 2)} and
      T=${num(payload.temperature, 2)}.</p>
    ${asserted ? `<p class="stage__flag"><b>Identity is asserted, not re-identified.</b>
      ${merged
        ? `No resident is enrolled, and one person leaving detection for longer than the
           tracker's ~15 s bridge comes back with a new track id — so tracks <b>${escapeHtml(merged)}</b>, whose frame spans
           never overlap and each of which begins within walking distance of where the previous
           one ended, are treated as one subject. Tracks seen in the same frame, or separated by
           more than a person could walk in the gap, are never merged — which is what keeps a
           person on a television out of the resident's numbers. Counting only the most-present
           fragment discarded the rest of the same person's activity.`
        : `No resident is enrolled, so track ${payload.subject_track} — the most-present one — is
           treated as the subject for Agent 3's features.`}
      Every role below still reads as
      <b>unidentified</b> because that is what OSNet decided, and it is right to: it has no
      gallery to match a stranger against. Without the assertion the subject filter matches
      nobody and all 27 features read zero.</p>` : ""}
    ${rows([
      ["Streams served", escapeHtml(streams)],
      ["Logit adjustment &tau;", num(payload.tau, 2)],
      ["Temperature", num(payload.temperature, 2)],
      ["Segments emitted", int(payload.n_segments)],
    ])}
    <p class="seam"><span>decoding</span><code>${escapeHtml(payload.decoding || "—")}</code></p>
    <div class="clip__tracks">${tracks.length
      ? tracks.map((t) => timelineMarkup(t, span)).join("")
      : `<p class="clip__empty">No person was tracked long enough to fill a window.
         Nothing is being withheld — there is nothing to classify.</p>`}</div>`;
}

/* PROVENANCE FIRST. The banner is above the figures on purpose: every robust-z below it is
 * measured against a reference persona, not against this resident's own history, and a
 * caveat printed under a number is a caveat nobody reads. */
function behaviourBody(p, payload) {
  if (payload.error) {
    return `<p class="stage__body">This stage failed and the run continued rather than
      dying: <code>${escapeHtml(payload.error)}</code></p>`;
  }
  const simulated = payload.baseline_provenance !== "resident_history";
  const feats = Object.entries(payload.features_from_video || {});
  const devs = payload.deviations || [];
  const alerts = payload.alerts || [];

  return `
    ${simulated ? `<p class="stage__flag"><b>Baseline is a declared reference, not this
      person.</b> ${payload.baseline_days ?? 0} reference days primed against
      ${payload.real_days_from_this_video ?? 1} real day from this clip. The feature values
      below are measured from your video and are real; every robust-z is
      &ldquo;unlike the reference&rdquo;, never &ldquo;this person has declined&rdquo;.</p>`
      : ""}
    <p class="stage__body">Agent 3 is statistics, not a model: a robust median and MAD per
      feature, a z-score against it, and CUSUM for slow drift. It is written that way so a
      flagged decline can be explained to the person acting on it.
      ${payload.is_reliable
        ? ""
        : `Observed span is ${num(payload.observed_hours, 3)} h, far below the
           ${"8 h"} a day needs to be called reliable — so the day is marked unreliable
           rather than quietly averaged in.`}</p>
    ${feats.length ? `<p class="stage__sub">Measured from this clip</p>${rows(
      feats.map(([k, v]) => [k.replace(/_/g, " "), num(v, 2)]))}` : ""}
    ${devs.length ? `<p class="stage__sub">Deviation against the reference (robust&#8209;z)</p>
      ${rows(devs.slice(0, 6).map((d) => [d.feature.replace(/_/g, " "),
        `${d.robust_z > 0 ? "+" : ""}${num(d.robust_z, 2)}`]))}`
      : payload.deviations_withheld
        ? `<p class="stage__sub">Deviation against the reference</p>
           <p class="clip__empty">Withheld — a ${num(payload.observed_hours, 3)} h window cannot be
           compared to full-day baselines. A ten-minute total against a ~23-hour median reads as a
           collapse no matter what the person did; the values above are what was actually
           measured.</p>`
        : ""}
    ${alerts.length
      ? `<ul class="stage__alerts">${alerts.map((a) =>
          `<li data-severity="${escapeHtml(a.severity)}"><b>${escapeHtml(a.kind)}</b>
           <span>${escapeHtml(a.severity)}</span>
           ${a.rule ? `<code>${escapeHtml(a.rule)}</code>` : ""}</li>`).join("")}</ul>`
      : `<p class="stage__body">No alert rule fired.</p>`}`;
}

function reportBody(p, payload) {
  if (payload.why_skipped) {
    return `<p class="stage__body">${escapeHtml(payload.why_skipped)}</p>`;
  }
  const rate = payload.hallucination_rate;
  return `
    <p class="stage__body">The model receives the numbers above and nothing else — no
      frames, no skeletons, no pixels. It writes two things: the prose below, and a set of
      separate citable claims. <b>C1–C5 check the claims, not the prose.</b> The claims are
      listed under this card with their verdicts; the paragraph is the model's own wording and
      carries no arithmetic guarantee, which is why the figures a caregiver should act on are
      the verified rows rather than the sentence.</p>
    <blockquote class="stage__quote" data-unverified="1">${escapeHtml(payload.summary || "")}
    </blockquote>
    ${payload.recommendation
      ? `<p class="stage__rec"><span>Recommendation</span>
         ${escapeHtml(payload.recommendation)}</p>` : ""}
    ${rows([
      ["Model", escapeHtml(payload.model || "—")],
      ["Constrained decoding", payload.constrained_decoding ? "on (JSON grammar)" : "off"],
      ["Claims emitted", int(payload.claims_emitted)],
      ["Schema-valid claims", int(payload.claims_scorable)],
      ["Unfaithful share", rate === null || rate === undefined
        ? "not scorable" : `${(rate * 100).toFixed(1)}%`],
      ["Escalate", payload.escalate ? "yes" : "no"],
    ])}
    ${payload.parse_failed
      ? `<p class="stage__flag"><b>The generation did not parse.</b> Reported rather than
         retried silently — a repaired-until-valid report is not the one the model wrote.
         ${(payload.notes || []).filter((n) => /did not parse/.test(n))
           .map((n) => escapeHtml(n.replace(/^response did not parse: /, "")))
           .join("; ")}</p>`
      : ""}`;
}

const BODIES = { 1: perceptionBody, 2: activityBody, 3: behaviourBody, 4: reportBody };

/* The enrolment record is not one of the four agents - it is a side effect of Agent 1 that
 * the LOCAL half emits when the operator asked to enrol. Rendering it through
 * perceptionBody would draw a card of undefined readouts, so it gets its own shape. */
function enrolmentCard(stage) {
  const p = stage.payload || {};
  return `<li class="stage stage--live" data-status="${escapeHtml(stage.status)}">
    <p class="stage__n">Re-ID gallery
      <span class="stage__pill" data-tone="${STATUS_TONE[stage.status] || "skip"}">$
        {escapeHtml(stage.status)}</span></p>
    <h4 class="stage__name">enrolment</h4>
    ${stage.status === "done"
      ? `<p class="stage__body"><b>${escapeHtml(p.enrolled || "?")}</b> enrolled from the
         best ${int(p.crops)} crops of this clip's subject track.</p>
         ${p.matched_this_clip
           ? `<p class="stage__flag">The gallery already matched this clip's subject before
              enrolment - the match and the enrolment agree.</p>`
           : `<p class="stage__flag">This clip was NOT matched before enrolment ${
              p.caveats ? "" : ""}- either the gallery was empty or the match was below
              threshold. The next upload is the test.</p>`}
         ${(p.caveats || []).map((c) => `<p class="stage__body">${escapeHtml(c)}</p>`).join("")}`
      : `<p class="stage__body">${escapeHtml(p.error || "Enrolment failed.")}</p>`}
  </li>`;
}

function stageCard(stage, span) {
  if (stage.name === "enrolment") return enrolmentCard(stage);
  const seam = SEAMS[stage.agent] || {};
  const body = BODIES[stage.agent];
  const tone = STATUS_TONE[stage.status] || "skip";
  const inner = stage.status === "done" && body
    ? body(stage, stage.payload || {}, span)
    : `<p class="stage__body">${stage.status === "failed"
        ? escapeHtml((stage.payload || {}).error || "This stage failed.")
        : escapeHtml((stage.payload || {}).why_skipped
            || "This stage did not run for this clip.")}</p>`;

  return `<li class="stage stage--live${stage.agent === 4 ? " stage--gate" : ""}"
      data-status="${escapeHtml(stage.status)}">
    <p class="stage__n">Agent ${stage.agent}
      <span class="stage__pill" data-tone="${tone}">${escapeHtml(stage.status)}</span>
      <span class="stage__time">${secs(stage.elapsed_s || 0)}</span></p>
    <h4 class="stage__name">${escapeHtml(stage.name)}</h4>
    ${inner}
    <p class="seam"><span>hands on</span><code>${escapeHtml(seam.emits || "—")}</code>
      ${seam.to ? `<em>${escapeHtml(seam.to)}</em>` : ""}</p>
  </li>`;
}

/* C1–C5 per claim, in the ledger's own markup so a pass looks identical in both places.
 *
 * One deliberate difference from the Python, carried over from the replay ledger: when C1
 * fails there is nothing left to compare against, so the verifier returns C2–C5 as false.
 * Drawing five failures for one fault overstates what went wrong, so they are drawn as
 * `skip` — not reached. The verdict is unchanged.
 *
 * C5 (does the prose echo the quoted figure?) is drawn from `C5_prose_quoted_value`. A
 * backend build that predates C5 does not send the key at all, and `undefined` is drawn as
 * `skip` — dashed, not reached — rather than as a pass. Treating an absent check as a
 * passed check is how a page ends up vouching for a verifier it never ran. */
function checkCells(c) {
  const state = (v) => (c.C1_ref_exists === false ? "skip" : v ? "pass" : "fail");
  const c5 = c.C5_prose_quoted_value === undefined ? "skip" : state(c.C5_prose_quoted_value);
  return [
    ["C1", c.C1_ref_exists ? "pass" : "fail"],
    ["C2", state(c.C2_value_matches)],
    ["C3", state(c.C3_pct_matches)],
    ["C4", state(c.C4_direction_consistent)],
    ["C5", c5],
  ].map(([id, s]) =>
    `<span class="cell" data-check="${id}" data-state="${s}" data-want="${s}"
      title="${id} ${s}">${id}</span>`).join("");
}

function checkRow(c, n) {
  const why = (c.notes || []).map((s) => escapeHtml(s)).join(" ");
  return `<li class="slip" data-verdict="${c.faithful ? "shown" : "withheld"}"
      data-resolved="1">
    <p class="slip__n">${String(n).padStart(2, "0")}</p>
    <p class="slip__text">${escapeHtml(c.text || "(claim carried no text)")}</p>
    <div class="gutter" role="img" aria-label="${c.faithful
      ? "All five checks passed" : "Failed verification, withheld"}">${checkCells(c)}</div>
    ${c.faithful ? "" : `<p class="slip__margin"><span class="stamp">withheld</span>
      ${why || "A check failed, so this sentence was not shown to a caregiver."}</p>`}
    <p class="slip__cite"><span>record</span>${escapeHtml(c.evidence_ref || "no citation")}
      ${c.claimed_value === null || c.claimed_value === undefined
        ? "" : ` · claimed ${num(c.claimed_value, 2)}`}
      ${c.claimed_pct_change === null || c.claimed_pct_change === undefined
        ? "" : ` · ${num(c.claimed_pct_change, 1)}%`}
      ${c.direction ? ` · ${escapeHtml(c.direction)}` : ""}</p>
  </li>`;
}

/** Render the four stage cards and the per-claim check table. Pure DOM writes. */
export function renderStages(payload, { pipelineEl, checksEl }) {
  const span = Math.max(0.1, (payload.frames_kept || 1) / (payload.fps || 15));
  const stages = payload.stages || [];

  if (pipelineEl) {
    pipelineEl.innerHTML = stages.length
      ? `<ol class="stages stages--live">${
          stages.map((s) => stageCard(s, span)).join("")}</ol>`
      : `<p class="clip__empty">This backend returned no stage records. It is an older
         build of notebook 05 — re-run the current one to see per-agent output.</p>`;
  }

  if (!checksEl) return;
  const checks = payload.checks || [];
  const withheld = checks.filter((c) => !c.faithful).length;
  // A backend built before C5 simply omits the key, and an omitted key would render as a
  // dashed "not reached" cell on every claim - quietly looking like agreement. Say
  // outright that the backend predates the check rather than letting absence read as a pass.
  const preC5 = checks.length > 0
    && checks.every((c) => c.C5_prose_quoted_value === undefined);
  checksEl.innerHTML = checks.length
    ? `<p class="stage__sub">C1–C5 on every claim from your clip</p>
       ${preC5 ? `<p class="stage__flag"><b>This backend predates check C5.</b> Its claims
         are verified against C1–C4 only; re-run the current
         <code>05_serve_inference_online.ipynb</code> to see the prose-echo check.</p>` : ""}
       <ol class="slips">${checks.map((c, i) => checkRow(c, i + 1)).join("")}</ol>
       <p class="clip__caveat">${checks.length} claim${checks.length === 1 ? "" : "s"},
         ${withheld} withheld. A withheld claim is not deleted — it is struck, stamped, and
         the arithmetic that failed is printed beside it, because a verification layer whose
         rejections are invisible cannot be audited. ${escapeHtml(
           payload.report_provenance || "")}</p>`
    : `<p class="clip__empty">No claims were produced for this clip, so there is nothing to
       verify. That is reported rather than shown as a clean bill of health.</p>`;
}

export { timelineMarkup };
