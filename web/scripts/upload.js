/* The clip panel: a video in, all four agents' output on screen, then C1–C5 per claim.
 *
 * The panel used to stop after Agent 2 and print a paragraph explaining that a caregiver
 * report needs Agent 3's 14-day baseline, so one upload could not have one. The science in
 * that paragraph was right and the conclusion was wrong: Agent 3 does two separable things,
 * and only one of them needs history.
 *
 *   Feature extraction  — durations, transition counts, fall count, observed span. Computed
 *                         from the segments of THIS clip. No history required, fully real.
 *   Deviation detection — robust-z and CUSUM against a 14-day rolling median. Needs history
 *                         this person has not supplied.
 *
 * So the stages all run, the measured features are real, and the comparison point is a
 * DECLARED simulated reference that says so in the payload and again in the banner
 * `stages.js` draws above the figures. C1–C5 still check every claim against the state
 * computed from this video, so an invented figure or an inverted direction is caught
 * arithmetically exactly as it is on the stored day. What the reference cannot support is
 * the clinical reading, and that is the one thing the banner refuses to let it imply.
 *
 * The alternative — omitting the stages — was worse for the honest reason and not the
 * convenient one: it left an examiner unable to see the verifier work on their own footage,
 * which is the part of this system that is actually novel.
 */

import { state, uploadVideo } from "bs/api";
import { SkeletonOverlay } from "bs/overlay";
import { renderStages } from "bs/stages";

const node = (id) => document.getElementById(id);

/* What is happening AFTER the agent that just reported. Keyed by the agent that finished, so
 * the line always names the stage the user is currently waiting on rather than the one
 * already on screen. Agent 4 has none: when it lands, the run is over. */
const STAGE_WAIT = {
  1: "Poses extracted. Agent 2 is classifying each person's windows independently…",
  2: "Activities labelled. Agent 3 is measuring durations, transitions and falls…",
  3: "Behaviour measured. Agent 4 is writing the report — this is the slow stage, "
     + "and every claim is verified before you see it…",
};

export function mountUpload() {
  const panel = node("upload");
  if (!panel) return null;

  const input = node("clip-input");
  const drop = node("clip-drop");
  const stage = node("clip-stage");
  const video = node("clip-video");
  const canvas = node("clip-canvas");
  const status = node("clip-status");
  const meta = node("clip-meta");
  const pipeline = node("clip-pipeline");
  const checks = node("clip-checks");
  const overlay = new SkeletonOverlay(canvas, video);

  let blobUrl = null;

  const say = (text, tone = "") => {
    status.dataset.tone = tone;
    status.textContent = text;
  };

  async function analyse(file) {
    if (!file) return;
    if (state.mode !== "live") {
      say("Connect a backend first — pose extraction needs the GPU session.", "bad");
      return;
    }
    // Show the clip immediately. The full chain takes tens of seconds to a couple of
    // minutes — Qwen alone runs ~50 s per report — and watching a still frame is a better
    // wait than watching nothing.
    if (blobUrl) URL.revokeObjectURL(blobUrl);
    blobUrl = URL.createObjectURL(file);
    video.src = blobUrl;
    stage.hidden = false;
    if (pipeline) pipeline.innerHTML = "";
    if (checks) checks.innerHTML = "";
    meta.textContent = "";
    // HONEST DURATION, not a marketing one: pose is ~2 s per second of footage on the
    // served GPU, so a ten-minute clip is minutes of analysis. Saying "a minute or two"
    // made a legitimate run look wedged.
    say(`Running all four agents over ${file.name}. RTMO first, then the ST-GCN++ `
      + "ensemble, then the behaviour layer, then a hosted model writes and the verifier "
      + "checks it — roughly two seconds of analysis per second of footage.", "wait");

    try {
      // PROGRESSIVE. The backend streams one line per agent, so each card is drawn the moment
      // that agent finishes rather than all four at the end. This is not only nicer: the
      // tunnel drops a request whose origin has not answered in about two minutes, and Qwen
      // alone outlasts that, so streaming is what makes the chain deliverable at all.
      //
      // `checksEl` is deliberately null while streaming. C1-C5 only exist once Agent 4 has
      // written and the verifier has run; rendering the table early would show "no claims
      // were produced", which reads as a clean bill of health rather than as not-yet.
      const streamed = [];
      const span = { fps: 15, frames_kept: 1 };
      // The enrol name, if the operator typed one. Passed to uploadVideo so the request
      // carries it twice: a form field for the GPU half, a header for the local half.
      const enrolName = (node("clip-enrol")?.value || "").trim();
      const data = await uploadVideo(file, {
        enrol: enrolName,
        onStage: (s) => {
          streamed.push(s);
          if (s.agent === 1 && s.payload) {
            span.fps = s.payload.fps || 15;
            span.frames_kept = s.payload.frames_kept || 1;
          }
          renderStages({ ...span, stages: streamed.slice() },
            { pipelineEl: pipeline, checksEl: null });
          if (STAGE_WAIT[s.agent]) say(STAGE_WAIT[s.agent], "wait");
        },
      });
      overlay.load(data);
      overlay.start();
      video.play().catch(() => {});   // autoplay may be refused; the overlay still draws

      // THE VERIFICATION TABLE GOES UP FIRST, before anything optional. The size check below
      // is a diagnostic; a stale cached copy of overlay.js made it a TypeError, that threw
      // inside this try, and the C1-C5 table — the entire point of the page — never rendered
      // because of a warning. Nothing advisory may preempt the result.
      renderStages(data, { pipelineEl: pipeline, checksEl: checks });

      const people = data.n_people ?? (data.tracks || []).length;
      const t = data.timing || {};
      // "TRACKS", NOT "PEOPLE". `n_people` counts track IDs, and one person behind furniture for
      // ten minutes produced 39 of them - the headline row of the result card read "39 people
      // tracked" over a video of one person, which is the one line a reader believes first.
      // Agent 2's card says which tracks were merged into the subject.
      const subjectTracks = (data.stages || []).find((s) => s.agent === 2)
        ?.payload?.subject_tracks || [];
      const merged = subjectTracks.length > 1 ? ` (${subjectTracks.length} merged into one subject)` : "";
      meta.innerHTML = [
        `${data.frames_kept} frames at ${data.fps} Hz`,
        `${data.width}&times;${data.height}`,
        `${people} ${people === 1 ? "track" : "tracks"}${merged}`,
        `pose ${t.pose_s ?? "—"}s · classify ${t.classify_s ?? "—"}s · behaviour ${
          t.behaviour_s ?? "—"}s · report ${t.report_s ?? "—"}s`,
        data.reid ? "re-id on" : "re-id off, roles unidentified",
        `&tau;=${data.tau}`,
      ].join(" · ") + (data.truncated
        ? ` · <b>truncated at ${data.frames_kept} frames</b>` : "");

      const failed = (data.stages || []).filter((s) => s.status === "failed");
      // Both decoders must agree on the frame size or the skeletons are drawn in the wrong
      // space. `typeof`-guarded: this page is served with no cache headers and browsers hold
      // ES modules hard, so a viewer can genuinely be running last week's overlay.js against
      // this week's payload. A missing diagnostic is acceptable; a page that shows no results
      // because a diagnostic is missing is not.
      const mismatch = typeof overlay.sizeDisagreement === "function"
        ? overlay.sizeDisagreement() : null;
      if (mismatch) {
        // Outranks the rest: if the overlay is drawing in the wrong coordinate space, nothing
        // else on screen should be read with confidence.
        say(mismatch, "bad");
      } else if (failed.length) {
        say(`Agent ${failed[0].agent} failed and the run continued — its card says why.`,
          "bad");
      } else {
        const checks = data.checks || [];
        const withheld = checks.filter((c) => !c.faithful).length;
        // ZERO CLAIMS IS NOT A PASS. "Every claim passed all five checks" over an empty set
        // is vacuously true and reads as a clean bill of health - observed on a 5.7 s clip
        // where the model emitted no claims at all and its prose said "no activity was
        // detected", while Agent 3 had measured 5.2 s of cooking. The verifier had nothing
        // to catch because nothing was claimed, and that is a finding about the report, not
        // a verdict about its truthfulness.
        say(checks.length === 0
          ? "Done — but the model produced NO verifiable claims for this clip, so C1–C5 had "
            + "nothing to check. Read the prose with that in mind: an empty claim set is not "
            + "a clean bill of health."
          : withheld
            ? `Done. ${withheld} claim${withheld === 1 ? " was" : "s were"} withheld by the `
              + "verifier and struck below."
            : `Done. All ${checks.length} claims passed all five checks.`, "");
      }
    } catch (err) {
      overlay.stop();
      say(err.message, "bad");
    }
  }

  input.addEventListener("change", () => analyse(input.files?.[0]));

  // Drag and drop, because handing someone a file picker for a video they already have on
  // screen is a needless step.
  for (const evt of ["dragenter", "dragover"]) {
    drop.addEventListener(evt, (e) => { e.preventDefault(); drop.dataset.over = "1"; });
  }
  for (const evt of ["dragleave", "drop"]) {
    drop.addEventListener(evt, (e) => { e.preventDefault(); delete drop.dataset.over; });
  }
  drop.addEventListener("drop", (e) => analyse(e.dataTransfer?.files?.[0]));

  video.addEventListener("loadedmetadata", () => overlay.resize());
  video.addEventListener("seeked", () => overlay.draw());
  window.addEventListener("resize", () => overlay.resize());

  // The upload control only appears when the backend says pose is available. /health
  // reports `video: false` when rtmo-l.onnx is not attached, and offering a button that
  // would 503 is worse than not offering one.
  return {
    sync(s) {
      const ok = s.mode === "live" && s.health?.video !== false;
      panel.dataset.ready = ok ? "1" : "0";
      if (!ok && s.mode === "live") {
        say("This backend has no rtmo-l.onnx attached, so pose extraction is off.", "bad");
      } else if (!ok) {
        say("Connect a backend to analyse a clip.", "");
      }
    },
  };
}
