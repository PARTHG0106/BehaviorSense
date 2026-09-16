/* The dock: paste a backend address, see exactly what answered.
 *
 * Every message here names what happened rather than reporting a colour. "No answer within
 * 6 s" and "answers, but it is not this service" are different problems with different
 * fixes, and collapsing them into a red dot is how a demo becomes unfixable in front of an
 * audience.
 */

import { state, connect, disconnect, onChange, validate } from "bs/api";

const node = (id) => document.getElementById(id);

/* Per state: the rail label, the tone, and a sentence that says what to do next. */
function describe(s) {
  const h = s.health || {};
  switch (s.mode) {
    case "live": {
      const gpu = h.gpu || "no GPU reported";
      const streams = (h.streams || []).length;
      return {
        label: "Live",
        tone: "ok",
        // Names what /health reported and nothing more. It does not say "the real model",
        // because /health does not identify one — the ledger's own banner names the model
        // that actually produced the claims, which is the only place that can be checked.
        text: `Connected. ${gpu}, ${streams} ADL stream${streams === 1 ? "" : "s"} loaded, `
            + `logit-adjust tau=${h.tau ?? "?"}${h.auth ? ", token required" : ""}. `
            + `"Generate live" in the ledger now generates a report on the backend.`,
      };
    }
    case "loading":
      return {
        label: "Loading",
        tone: "wait",
        text: "The tunnel answers and the models are still loading — RTMO and the activity "
            + "head are a minute or two. This will flip to Live on its own.",
      };
    case "checking":
      return { label: "Checking", tone: "wait", text: "Probing /health…" };
    case "failed":
      return {
        label: "Backend error",
        tone: "bad",
        text: `The backend reached /health but failed to load: ${h.error}. `
            + "The notebook's own output has the traceback.",
      };
    case "wrong":
      return {
        label: "Wrong service",
        tone: "bad",
        text: "Something answered at that address, but it did not identify itself as "
            + "BehaviorSense. Check you copied the whole line the notebook printed.",
      };
    case "invalid":
      return { label: "Bad address", tone: "bad", text: h.error };
    case "down":
      // The browser collapses DNS failure, connection refused, CORS rejection and mixed
      // content into one opaque "Failed to fetch", and nothing in JS can tell them apart.
      // Naming the plausible causes beats asserting the wrong one confidently.
      return {
        label: "No answer",
        tone: "bad",
        text: `${h.error} Either the Kaggle session has ended — the last cell must stay `
            + "running, and the address changes on every restart — or the address is not "
            + "the one the notebook printed.",
      };
    default:
      return {
        label: "Replay",
        tone: "",
        text: "Not connected. The ledger below is replaying a stored day, verified in your "
            + "browser. That path needs no backend at all.",
      };
  }
}

export function mountDock({ onLive } = {}) {
  const conn = node("conn");
  const dock = node("dock");
  const form = node("dock-form");
  const url = node("api-url");
  const token = node("api-token");
  const status = node("dock-status");
  const submit = node("dock-connect");
  const forget = node("dock-forget");
  const chip = node("chip-live");

  url.value = state.base;
  token.value = state.token;

  const setOpen = (open) => {
    dock.hidden = !open;
    conn.setAttribute("aria-expanded", String(open));
    if (open) url.focus({ preventScroll: true });
  };

  conn.addEventListener("click", () => setOpen(dock.hidden));

  // Escape closes it. A panel that traps you is worse than one that is hard to find.
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !dock.hidden) { setOpen(false); conn.focus(); }
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const check = validate(url.value);
    if (check.error) {
      status.dataset.tone = "bad";
      status.textContent = check.error;
      return;
    }
    submit.disabled = true;
    submit.textContent = "Connecting…";
    try {
      await connect(url.value, token.value);
    } finally {
      submit.disabled = false;
      submit.textContent = "Connect";
    }
  });

  forget.addEventListener("click", () => {
    disconnect();
    url.value = "";
    token.value = "";
  });

  if (chip && onLive) {
    chip.addEventListener("click", () => onLive(chip));
  }

  onChange((s) => {
    const { label, tone, text } = describe(s);
    conn.dataset.mode = s.mode;
    conn.querySelector(".conn__text").textContent = label;
    conn.title = s.base ? `${s.base} — ${label.toLowerCase()}` : "No backend connected";
    status.dataset.tone = tone;
    status.textContent = text;
    // The live control appears only when it would work. A disabled button people cannot
    // explain is worse than a control that is simply not there yet.
    if (chip) chip.hidden = s.mode !== "live";
  });

  return { open: () => setOpen(true) };
}
