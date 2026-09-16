# web/

The front end for BehaviorSense. No build step, no dependencies, no framework — open it with
any static server and it runs.

```bash
python -m http.server 5173 --directory web
# then http://localhost:5173
```

## What it is

A single page whose hero is the product rather than a picture of it. The daily report at the
top is verified in the browser: `scripts/verifier.js` is a faithful port of
`src/behaviorsense/agents/reasoning/verifier.py` — same tolerances (2% relative on values,
2 percentage points on changes), same direction word lists, same treatment of a percentage
quoted against a zero baseline. The four cells beside each claim are real check results, and
the arithmetic under a withheld claim is the actual comparison that failed.

The fault controls mirror `HallucinatingStubLLM`: the same six corruptions the Python test
suite injects, applied as fixed scripts rather than at random so the page is reproducible and
each slip can name the fault it was given.

One deliberate difference from the Python. When a citation does not resolve, the verifier
returns C2–C4 as `false` because there is nothing left to compare against; here they are drawn
as *not reached* instead, because showing four failures for one fault overstates what went
wrong. The verdict — unfaithful, withheld — is identical.

## The backend does not run here

Qwen2.5-7B needs ~16 GB in bf16 and the ADL ensemble needs a GPU to be worth calling. Neither
fits a laptop, and quantising to fit would trade the numbers in `results/evaluation.md` for
ones nobody has measured. So the models live in a Kaggle notebook and this page talks to them
over a tunnel:

```
web/ (static, anywhere)  ──fetch──▶  https://…trycloudflare.com  ──▶  Kaggle T4/P100
                                                                       Qwen2.5-7B
                                                                       4 ADL streams
```

1. Run `notebooks/05_serve_inference_online.ipynb` with **internet ON** and a GPU. Attach
   `behaviorsense-code`, `behaviorsense-runs`, and the Qwen2.5-7B-Instruct Kaggle Model.
2. The last cell prints an address. **Leave that cell running** — the tunnel dies with it.
3. Click the state indicator in the page header, paste the address, Connect.

The address is different every session. That is inherent to a quick tunnel with no account,
which is why it is pasted rather than configured — a hard-coded one would be wrong within the
hour, and a page that pretended otherwise would be the decorative "live" badge this project
argues against.

### Why both slow endpoints stream

The quick tunnel drops a request whose origin has not answered in about two minutes. Measured
2026-08-25 against a live T4: `/demo` was cut at **125.8 s with Cloudflare's 524** while the
model was still generating, on two separate sessions. `/video` is strictly slower — RTMO over
up to 900 frames, then ST-GCN++, then that same Qwen pass.

So neither can be a single synchronous JSON reply. The obvious alternatives were to shorten the
report or lower `max_frames`, and both trade measured behaviour for a timeout. The cap is on
*time-to-first-byte*, so the fix is to answer immediately and keep sending: both endpoints
return newline-delimited JSON.

```
{"event":"accepted"}                first, immediately — stops the 524 clock
{"event":"stage","stage":{...}}     one per agent, as it finishes        (/video)
{"event":"progress",...}            named waypoints                      (/demo)
{"event":"result","payload":{...}}  the whole payload, unchanged
{"event":"error","detail":"..."}    in-band
```

Errors travel in-band because by the time one happens the response headers are long gone and
there is no HTTP status left to carry it. Rejections that happen *before* the stream — 413 for
an oversized upload, 422 for an empty one — are still real statuses, which is why the upload is
read to completion before the first chunk goes out.

The client timeout is a **watchdog on silence**, not on total duration: a ceiling on the whole
request would abort a legitimately slow report, whereas what actually indicates a dead session
is no bytes at all. Every line resets it.

This is the rare case where the workaround and the feature are the same code. The page already
wanted to draw each agent as it finished; the tunnel made that mandatory rather than merely
nice.

### What connecting buys

`Generate live` in the ledger runs `/demo` on the backend: the same simulator the evaluation
uses picks a day that actually alerts, Qwen writes the report, and the response carries the
claims, the server's own C1–C4 verdicts, **and the evidence table they were checked against**.

The page then re-derives the verdicts from that evidence with its own verifier and compares.
Two independent implementations — Python on a GPU, JavaScript in a browser — agreeing on live
model output is worth more than either asserting a number, so agreement is stated explicitly
and **disagreement is reported as a warning not to trust the page**, rather than smoothed over.

### States, and why each is named

| indicator | what it means |
|---|---|
| `Replay` | no backend. The ledger runs a stored day, verified locally. Fully useful. |
| `Checking` | probing `/health`. |
| `Loading` | the tunnel answers, weights are still materialising (~2–3 min). |
| `Live` | ready. `/health` reported the GPU, the streams that loaded, and τ. |
| `No answer` | nothing responded. Session ended, or the wrong address. |
| `Wrong service` | something answered but did not identify as BehaviorSense. |
| `Backend error` | `/health` returned a load failure; the traceback is in the notebook. |

The backend loads models on a background thread specifically so it can report `loading`
instead of timing out — a three-minute silence is indistinguishable from a crash, and that
distinction is the difference between waiting and debugging.

## Upload a clip

The **On your own footage** band takes a video and runs **all four agents** over it, showing
each stage's own output where it was produced:

- **Agent 1** — **RTMO** extracts COCO-17 poses for every person in shot, the tracker holds
  an identity across frames, **OSNet** decides which one is the resident
- **Agent 2** — **ST-GCN++** labels each person's activity independently, on their own
  windows, at the τ and temperature notebook 04 selected; one timeline per person, falls
  struck in pencil
- **Agent 3** — durations, transition counts, fall count and observed span measured from the
  clip, then robust-z and the alert rules
- **Agent 4** — Qwen2.5-7B writes the report from those numbers alone, and **C1–C5 are shown
  per claim**, in the same `.slip`/`.gutter` markup the replay ledger uses. C5 asks whether
  the sentence actually quotes the figure the field records — the check the latest measured
  run justified, where C2 was zero and prose paraphrase was the largest failure category. A
  backend built before C5 renders a warning banner rather than letting the missing key read
  as a pass.

Each card names the schema type it hands on (`PersonObservation` → `ActivitySegment` →
`BehaviourState` → `CaregiverReport`), so the static pipeline diagram above is describing
*this run* rather than an idealised one.

Identity is shown by **weight and fill, not colour**. The palette has three hues and each
already means something — ink is verified, pencil is a correction, ochre is "watch this" — so
using pencil for "second person" would say that person is a mistake. The resident is drawn
heavy with filled joints, everyone else light with hollow ones, and the role is stated in mono
type beside the head, which is where this system puts facts.

A person who is tracked but too occluded to classify is drawn as a **dashed box labelled
"pose unusable"** rather than omitted. "Present, unreadable" and "not there" are different
facts, and Agent 2 abstains on the first rather than guessing.

### The baseline is declared, because one clip cannot supply fourteen days

Agent 3 does two separable things and only one of them needs history:

| | needs history? | on one clip |
|---|---|---|
| feature extraction — durations, transition counts, falls, observed span | no | **real, measured from your video** |
| deviation detection — robust-z and CUSUM against a 14-day rolling median | yes | against a **declared simulated reference** |

Earlier versions refused Agents 3 and 4 outright. The science was right and the remedy was
wrong: it left a visitor unable to see the verifier work on their own footage, which is the
part of this system that is actually novel. So the stages run, `baseline_provenance` says in
the payload what the comparison point is, and `stages.js` draws that as an **ochre banner
above the figures** — not a footnote under them, because a caveat printed below a number is a
caveat nobody reads.

C1–C5 are unaffected, and that is the point. Every claim is still checked against the state
computed from *this* video, so an invented figure, a wrong percentage, an inverted
direction or a sentence that never quotes its number is caught arithmetically exactly as it
is on the stored day. What a reference baseline cannot support is the clinical reading —
"mobility declined" — because the comparison point is not this person. Both facts ship in
the response and both are on screen.

### Limits, and why they are refusals rather than truncations

60 MB and 900 frames (about a minute at 15 Hz). A longer upload is **rejected**, not silently
cut short: a report over the first twenty seconds of a ten-minute video, presented as a report
over the video, is the kind of quiet misrepresentation this project keeps finding and removing.

Decode runs in a **child process** (`behaviorsense.video.extract_isolated`). cv2 delegates to
ffmpeg, ffmpeg raises SIGSEGV/SIGABRT on malformed streams, and a signal is not an exception —
`try/except` cannot see it and the interpreter simply stops. Notebook 02 lost finished corpora
to exactly this. Inline, one bad upload would kill the kernel, the tunnel and the demo
together; behind the boundary it is a 422 that names the signal.

## Developing the live path without a GPU

Booking a GPU session to check that a banner renders is absurd, so there is a stand-in:

```bash
python web/dev_backend.py          # http://127.0.0.1:8899
```

Same routes, same shapes, no model. Three things about its fixtures are deliberate: the report
fixture contains one claim with a right number told backwards, so the page can be seen
*rejecting* something; its withheld note carries the **real failing arithmetic** in
`verifier.py`'s own wording, because a stub that returned no note would render the generic
fallback and quietly demonstrate the weaker version of the feature; and `/video` returns the
**full four-stage payload** over two synthetic figures — one of whom falls — with a
twelve-frame stretch where one pose is unusable, so the overlay's scaling, the two-person
distinction, the dashed-box state, every stage card and the five-cell C1–C5 check table are
all exercised rather than assumed.

It also **streams**, chunked and flushed per line, exactly as the notebook does. A stub that
replied in one shot would let the page be built against a shape the GPU never sends — the
specific failure this file exists to prevent. `BS_STREAM_DELAY` (default `0.6`, seconds per
line) fakes the per-agent wait so the progressive reveal is visible locally; the test suite
sets it to `0`.

`tests/test_web_contract.py` asserts the stub and notebook 05 emit the same `/video` keys
(W1–W4), the same NDJSON event names (W7), and — since the payload is built by real code
rather than described by it — **executes** the notebook's own staging function over real
schema objects (W6). A stub that drifts is worse than no stub.

## Security, stated plainly

The Kaggle API is **unauthenticated by default**. Anyone with the URL can post to it while the
session lives. That is acceptable for a demo you start and stop deliberately; it is not a
deployment. Set `BS_TOKEN` in the notebook to require a shared secret — the page has a field
for it, sent as `X-BS-Token`.

Live claims are arbitrary text written by a language model, arriving over that unauthenticated
tunnel, and they are rendered into the page. Every field that reaches `innerHTML` is escaped
(`escapeHtml` in `scripts/verifier.js`): the claim body, the evidence reference, the direction
and the model name. This is not hypothetical hygiene — the moment the page stopped rendering
its own fixtures and started rendering model output, escaping stopped being optional.

## Files

```
index.html            structure and all copy
styles/tokens.css     colour, type and spacing tokens, with the reasoning behind them
styles/base.css       reset, type roles, the sticky rail, connection states
styles/components.css ledger, checks, stages, readouts, figures, dock, footage, close
scripts/verifier.js   C1-C4, ported from the Python, plus escapeHtml
scripts/fixtures.js   one analysed day + the six fault transforms
scripts/ledger.js     renders the report, resolves the checks, swaps in live payloads
scripts/charts.js     the two figures, hand-drawn SVG
scripts/api.js        the backend connection: validation, health polling, /demo, /video
scripts/dock.js       the paste-an-address panel and every state message
scripts/overlay.js    COCO-17 skeletons drawn over the playing video
scripts/stages.js     the four agent cards and the per-claim C1-C4 table
scripts/upload.js     the clip panel: upload, overlay, wiring
scripts/main.js       wiring
dev_backend.py        a stand-in backend for developing the live path
```

A note for anyone iterating on the scripts: browsers cache ES modules hard, and a plain
`http.server` sends no cache headers to argue with. If an edit appears not to take, hard-reload
(`Ctrl`/`Cmd`+`Shift`+`R`) rather than assuming the change was wrong.

## Design notes

The direction is a diazo specification sheet on a desk: cool blue-grey print stock, content on
lifted sheets, hairline rules. The load-bearing decision is that **verification is the
difference between ink and pencil** — a claim that passes is set in ink and is
indistinguishable from the page's own voice, while a claim that fails is struck in red pencil
and stamped. So there is no green "verified" badge anywhere: the verified state is simply the
document. Pencil is the only saturated hue in the system, which is why the eye lands on the one
thing that matters.

Type carries three roles and each earns its place. **Instrument Sans** is the instrument's own
voice. **Newsreader** appears only inside the generated report, because the report is a
document and the chrome is not. **IBM Plex Mono** is reserved for measurement.

The connection states borrow the same logic rather than inventing a palette: ink for live,
ochre — the hue reserved for "watch this" — for loading, pencil for every failure. Nothing in
the system is green, so nothing about being connected is congratulatory.

There is no dark variant, declared with `color-scheme: light` rather than left to default. The
scheme rests on ink being permanent and pencil being a correction, and neither gesture survives
inversion — pencil on black is not the same mark. The ground is a mid grey-green rather than
white, which is what makes a light-only page tolerable at night.
