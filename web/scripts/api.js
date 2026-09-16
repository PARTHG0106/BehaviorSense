/* The connection to the inference backend.
 *
 * The backend does not run next to this page. RTMO and the activity model need a GPU, so
 * they live in a Kaggle notebook (notebooks/05_serve_inference_online.ipynb) behind a
 * Cloudflare quick tunnel, while the report is written by a hosted model whose keys stay
 * on the operator's machine (web/local_backend.py). That tunnel
 * hands out a fresh https://<random>.trycloudflare.com on every session, which is why the
 * address is PASTED rather than configured: hard-coding one would be wrong within the hour,
 * and pretending otherwise would be the kind of decorative "live" badge this page exists to
 * argue against.
 *
 * Three states, and the page says which one it is in rather than degrading silently:
 *
 *   replay   no address, or nothing answers. The ledger runs on a stored day, verified in
 *            the browser. This is the default and it is fully useful.
 *   loading  the tunnel answers but the models are still loading (~1-2 min). The
 *            backend loads them on a background thread precisely so it can say this
 *            instead of timing out.
 *   live     ready. /demo returns claims the configured model just produced, together with
 *            the evidence they were checked against, and this page re-verifies them here.
 */

const HEALTH_TIMEOUT = 6000;   // a cold tunnel takes a second or two to route
const POLL_MS = 15000;
const STORE = "bs.api";
const STORE_TOKEN = "bs.token";

export const state = {
  base: localStorage.getItem(STORE) || "",
  token: localStorage.getItem(STORE_TOKEN) || "",
  mode: "replay",
  health: null,
};

const listeners = new Set();
export const onChange = (fn) => { listeners.add(fn); return () => listeners.delete(fn); };
const emit = () => { for (const fn of listeners) fn(state); };

/* ngrok's free tier answers anything with a browser User-Agent with its own interstitial
 * (ERR_NGROK_6024) instead of forwarding, and that page carries no Access-Control-Allow-Origin.
 * So a backend behind `ngrok http` fails here as a bare "Failed to fetch" that looks for all
 * the world like a dead tunnel: the request never reaches the backend, and the CORS headers the
 * backend would have sent never exist. Any value for this header skips the page. Cloudflare
 * quick tunnels and localhost ignore the header, so it is safe to send unconditionally rather
 * than sniffing the hostname. Note it is not CORS-safelisted, so it forces a preflight on every
 * call — local_backend.py answers OPTIONS with `Allow-Headers: *`, which is why that is fine. */
function reqHeaders() {
  const h = { "ngrok-skip-browser-warning": "1" };
  if (state.token) h["X-BS-Token"] = state.token;
  return h;
}

function setMode(mode, health = null) {
  state.mode = mode;
  state.health = health;
  emit();
}

async function ask(path, { timeout = HEALTH_TIMEOUT, method = "GET" } = {}) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeout);
  try {
    const res = await fetch(`${state.base}${path}`, {
      method,
      signal: ctl.signal,
      headers: reqHeaders(),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
    return body;
  } finally {
    clearTimeout(timer);
  }
}

/* Rejecting a bad address here, with the reason, beats a failed fetch and a console trace.
 * Mixed content is the one people hit and never diagnose: a page on https cannot call
 * http, and the browser reports it as a generic network error. */
export function validate(raw) {
  const url = raw.trim().replace(/\/+$/, "");
  if (!url) return { error: "Paste the address the notebook printed." };
  if (!/^https?:\/\//i.test(url)) return { error: "Needs to start with https://" };
  if (url.startsWith("http://") && location.protocol === "https:") {
    return { error: "This page is on https, so it cannot call an http address." };
  }
  return { url };
}

let poll = null;

/** Probe /health and set the mode. Returns the health body, or null. */
export async function refresh() {
  if (!state.base) { setMode("replay"); return null; }
  try {
    const health = await ask("/health");
    if (health.service !== "behaviorsense") {
      setMode("wrong", { error: "That address answers, but it is not this service." });
      return null;
    }
    if (health.error) { setMode("failed", health); return null; }
    setMode(health.ready ? "live" : "loading", health);
    return health;
  } catch (err) {
    // AbortError means the timeout fired: the tunnel is unreachable, not slow-and-fine.
    const reason = err.name === "AbortError" ? "No answer within 6 s." : err.message;
    setMode("down", { error: reason });
    return null;
  }
}

/** Point at a backend, persist it, and start polling. */
export async function connect(rawUrl, token = "") {
  const { url, error } = validate(rawUrl);
  if (error) { setMode("invalid", { error }); return null; }

  state.base = url;
  state.token = token.trim();
  localStorage.setItem(STORE, state.base);
  localStorage.setItem(STORE_TOKEN, state.token);

  setMode("checking");
  const health = await refresh();
  clearInterval(poll);
  // Poll while connected: a Kaggle session ends without warning, and a page still claiming
  // "live" twenty minutes after the kernel died is exactly the lie to avoid.
  poll = setInterval(refresh, POLL_MS);
  return health;
}

export function disconnect() {
  clearInterval(poll);
  poll = null;
  state.base = "";
  state.token = "";
  localStorage.removeItem(STORE);
  localStorage.removeItem(STORE_TOKEN);
  setMode("replay");
}

/* NDJSON, because the tunnel drops a slow request.
 *
 * Measured 2026-08-25 against a live T4: /demo was cut at 125.8 s with Cloudflare's 524 while
 * the model was still generating. /video is slower still. The cap is on time-to-first-byte, so
 * the backend answers immediately and streams one line per agent; this reads them as they
 * land. That is also exactly what the stage cards want to draw, so the timeout fix and the
 * progressive reveal are the same mechanism.
 *
 * Wire format, one JSON object per line:
 *   {"event":"accepted"}                first, immediately
 *   {"event":"stage","stage":{...}}     per agent, as it finishes  (video only)
 *   {"event":"progress",...}            keeps the connection busy  (demo only)
 *   {"event":"result","payload":{...}}  the whole payload
 *   {"event":"error","detail":"..."}    IN-BAND — see below
 *
 * Errors arrive in-band because by the time one happens the response headers are long gone,
 * so there is no HTTP status left to carry it. A stream that just stops is indistinguishable
 * from a dropped tunnel, which is why the backend always sends the error line.
 *
 * The timeout is a WATCHDOG ON SILENCE, not on total duration. A 480 s ceiling on the whole
 * request would abort a legitimately slow report; what actually indicates a dead backend is
 * no bytes at all for a while. Every line received resets it.
 */
async function askStream(path, {
  method = "GET",
  body = null,
  silenceTimeout = 90000,
  onEvent = null,
  headers = {},
} = {}) {
  if (!state.base) throw new Error("No backend connected.");
  const ctl = new AbortController();
  let timer = null;
  const arm = () => {
    clearTimeout(timer);
    timer = setTimeout(() => ctl.abort(), silenceTimeout);
  };
  arm();

  try {
    const res = await fetch(`${state.base}${path}`, {
      method,
      body,
      signal: ctl.signal,
      headers: { ...reqHeaders(), ...headers },
    });

    // A pre-stream rejection (401/413/422/503) is still a normal JSON body with a status.
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    if (!res.body) throw new Error("This browser gave no readable stream for the response.");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let result = null;
    let failure = null;

    // Split on newlines and keep the remainder: a chunk boundary lands mid-line often enough
    // that parsing per-chunk would drop roughly one event in five.
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      arm();
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        let ev;
        try {
          ev = JSON.parse(line);
        } catch {
          continue;                    // a partial line survives in `buf`; never guess at one
        }
        if (onEvent) { try { onEvent(ev); } catch { /* rendering must not kill the read */ } }
        if (ev.event === "result") result = ev.payload;
        else if (ev.event === "error") failure = ev;
      }
    }

    if (failure) {
      const err = new Error(failure.detail || "The backend reported an error.");
      err.kind = failure.kind;
      err.serverTrace = failure.traceback;
      throw err;
    }
    if (!result) {
      throw new Error("The backend closed the stream without sending a result. The Kaggle "
        + "session may have ended mid-run — check the notebook is still running.");
    }
    return result;
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error(`No data from the backend for ${Math.round(silenceTimeout / 1000)} s. `
        + "The session may have ended, or the GPU is wedged.");
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

/** One live report: real claims, real verdicts, and the evidence they were checked against. */
export function fetchDemo(scenario = 0, { onEvent = null } = {}) {
  return askStream(`/demo?scenario=${scenario}`, { onEvent });
}

/** Upload a clip. Streams one stage per agent, then the full payload.
 *
 * `onStage` fires as each agent finishes, so the cards fill in while Qwen is still writing —
 * which matters because generation alone outlasts the tunnel's old synchronous budget.
 *
 * The backend's own 413/422 text still surfaces: "upload exceeds 60 MB" and "the decoder was
 * killed by signal 11" are the two answers a user can act on, and a generic "request failed"
 * hides both. Those arrive before the stream starts, so they are still real HTTP statuses.
 */
export async function uploadVideo(file, { onStage = null, onEvent = null, enrol = "" } = {}) {
  const body = new FormData();
  body.append("file", file, file.name || "clip.mp4");
  // The enrol name travels twice, deliberately. The form field reaches the GPU half (it is
  // what turns OSNet on and collects the subject's crops); the header tells the local
  // backend which name to persist the enrolment under - a local decision about a local file
  // that the Kaggle side has no business making.
  const name = (enrol || "").trim();
  if (name) body.append("enrol", name);
  return askStream("/video", {
    method: "POST",
    body,
    headers: name ? { "X-BS-Enrol": name } : {},
    onEvent: (ev) => {
      if (ev.event === "stage" && onStage) onStage(ev.stage);
      if (onEvent) onEvent(ev);
    },
  });
}

/** Restore a saved address on load, without blocking the page on it. */
export function boot() {
  if (!state.base) { setMode("replay"); return; }
  setMode("checking");
  refresh().then(() => {
    clearInterval(poll);
    poll = setInterval(refresh, POLL_MS);
  }).catch(() => {});
}
