/* Wiring. Keep this thin: each module owns its own behaviour. */

import { mountLedger } from "bs/ledger";
import { drawAnchors, drawLodo } from "bs/charts";
import { boot as bootApi, fetchDemo, onChange } from "bs/api";
import { mountDock } from "bs/dock";
import { mountUpload } from "bs/upload";

function revealOnEntry() {
  const targets = document.querySelectorAll(".band__head, .checks, .stages, .boundary, .calib, .fig--wide, .todo, .cta, .clip");
  for (const node of targets) node.classList.add("reveal");

  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    for (const node of targets) node.classList.add("is-in");
    return;
  }
  const io = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      entry.target.classList.add("is-in");
      io.unobserve(entry.target);
    }
  }, { rootMargin: "0px 0px -12% 0px", threshold: 0.08 });
  for (const node of targets) io.observe(node);
}

/* One live generation. The hosted model takes anywhere from ~15 s to ~90 s per report
 * depending on which free provider answers, so the button has to say so — an unlabelled
 * minute of nothing is indistinguishable from a hang, and the honest fix is to name the
 * wait rather than fake a progress bar. */
async function generateLive(ledger, chip) {
  const label = chip.textContent;
  chip.setAttribute("aria-busy", "true");
  chip.disabled = true;
  chip.textContent = "Generating…";
  const status = document.getElementById("dock-status");
  try {
    // The response is streamed, so the button can report which half of the wait it is in.
    // Measured: >2 minutes on a T4, most of it in generation — a single unchanging label for
    // that long is the "indistinguishable from a hang" problem this comment already names.
    ledger.showLive(await fetchDemo(0, {
      onEvent: (ev) => {
        if (ev.event === "accepted") {
          chip.textContent = ev.cached ? "Fetching…" : "Simulating days…";
        } else if (ev.event === "progress" && ev.step === "state_ready") {
          chip.textContent = "Model writing…";
        }
      },
    }));
  } catch (err) {
    if (status) {
      status.dataset.tone = "bad";
      status.textContent = `Live generation failed: ${err.message}`;
    }
    document.getElementById("dock")?.removeAttribute("hidden");
  } finally {
    chip.removeAttribute("aria-busy");
    chip.disabled = false;
    chip.textContent = label;
  }
}

function boot() {
  const ledger = mountLedger();
  drawAnchors(document.getElementById("anchors-plot"));
  drawLodo(document.getElementById("lodo-plot"));
  revealOnEntry();
  mountDock({ onLive: (chip) => generateLive(ledger, chip) });
  const upload = mountUpload();
  if (upload) onChange((s) => upload.sync(s));
  // Last, and allowed to fail: the page must not depend on a backend being up.
  bootApi();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot, { once: true });
} else {
  boot();
}
