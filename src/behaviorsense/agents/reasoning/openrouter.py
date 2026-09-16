"""Agent 4 over OpenRouter: a `ReportLLM` that runs no weights locally.

Why this exists
---------------
Qwen2.5-7B on two T4s was **286 s of a ~340 s** request - about 85% of the wall clock - while
holding 17.8 GiB of VRAM sharded across PCIe, emulating bf16 on sm_75 hardware, and (per the RSS
instrumentation) retaining ~13 GiB of host RAM per analysis. On the P100 it cannot run at all:
torch ships no sm_60 kernels, so it loaded for 106 s and then died exactly as Agent 2 had. A
hosted endpoint turns that stage into single-digit seconds and gives the pose model the whole GPU.

What does NOT change, and is the reason this swap is safe
--------------------------------------------------------
Agent 4 receives **numbers only** - durations, counts, z-scores - never frames, never skeletons.
And every claim it writes is checked arithmetically by C1-C5 before anyone sees it. The entire
architecture is built on not trusting the model, so substituting a model we know less about is
precisely the case the verifier exists to cover. A hallucinated figure is caught by C2 whether it
came from Qwen on a local GPU or from a hosted endpoint.

Qwen stays the MEASURED arm. Every number in `results/evaluation.md` was produced by it under
grammar-constrained decoding on a pinned local checkpoint, and a dissertation table has to be
reproducible from something that cannot be deprecated. This is the demo arm, and `name` says so,
so the page always states which model wrote a report.

What the rotation counters actually answered
--------------------------------------------
Rotation was implemented and INSTRUMENTED rather than argued about, and one run settled it:
**thirteen keys, thirteen 429s, one each**, and the body said `z-ai/glm-5.2:free is temporarily
rate-limited UPSTREAM`. The OpenRouter dashboards agreed - several of those keys had made no
requests at all that day, one none in three days, and none more than ten. The ceiling is the free
PROVIDER's capacity (a single endpoint, shared by everyone), not the account's quota.

So a 429 has two causes that need OPPOSITE responses:

* **provider saturated** - the key is innocent. Every other key reaches the same provider, so the
  axis that helps is a different MODEL on a different provider.
* **the key's own quota** - 50 requests/day below 10 purchased credits. Cool the key, keep going.

Conflating them is what produced the failure: thirteen innocent keys parked for 60 s each while
the one thing that would have worked - another provider - was never tried at all.

Why a sweep contains no waiting
-------------------------------
The first retry policy put exponential backoff between *every* attempt, so reaching key 14 cost
24 s of accumulated waiting and a 150 s deadline expired with seven keys never tried. But a fresh
key costs nothing and a different provider costs nothing; only REPEATING a request that already
failed has to wait. Hence the shape here: sweep every model across every eligible key with no
delay at all, and back off only between whole sweeps.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "z-ai/glm-5.2:free"

# Ordered fallback chain. Every entry was checked against `/api/v1/models` and
# `/api/v1/models/<id>/endpoints` on 2026-08-28: each advertises `response_format` AND
# `structured_outputs` (so `strict` is a guarantee, not a hint), and - the entire point - each sits
# on a DIFFERENT provider, so a saturated one is routed around instead of retried into:
#
#     z-ai/glm-5.2:free                       Decart      256k ctx   100.0% uptime/30m
#     nvidia/nemotron-3-super-120b-a12b:free  Nvidia      262k ctx    99.6%
#     dots-studio/dots-3-note-preview:free    AtlasCloud  512k ctx   100.0%
#     minimax/minimax-m3:free                 GMICloud      1M ctx    99.6%
#     liquid/lfm-2.5-2.6b:free                Liquid       64k ctx    96.3%
#
# `nvidia/nemotron-3-ultra-550b-a55b:free` is deliberately ABSENT despite being the larger model,
# on three independent counts: its own page states it does not support `response_format`, so
# `constrained=True` cannot be honoured; it serves **2 tok/s**, which makes a 1,600-token report
# roughly thirteen minutes against a ~110-125 s tunnel cap and a 90 s front-end watchdog; and its
# 24-hour availability is 75.5%. The Super variant is the same vendor, strict-schema capable, and
# on a different provider from Decart, which is what was actually wanted.
#
# A 2.6B model is last on purpose rather than excluded: Agent 4 only restates numbers it was
# handed, and C1-C5 checks every one of them, so a small model degrades prose - not correctness.
FALLBACK_MODELS: tuple[str, ...] = (
    "nvidia/nemotron-3-super-120b-a12b:free",
    "dots-studio/dots-3-note-preview:free",
    "minimax/minimax-m3:free",
    "liquid/lfm-2.5-2.6b:free",
)

# Statuses worth trying a different KEY for. 429 is handled separately, because its cause decides
# whether the key or the model is the thing to change. 599 is not a real status; it is this file's
# local marker for a transport failure (timeout, DNS), which is indistinguishable from a 503 here.
ROTATE_ON = (402, 408, 500, 502, 503, 504, 599)

# Substrings identifying a 429 as the PROVIDER being saturated rather than the key being spent.
# Taken from the body actually observed, not invented:
#   {"error":{"message":"Provider returned error","code":429,
#             "metadata":{"raw":"z-ai/glm-5.2:free is temporarily rate-limited upstream.
#                                Please retry shortly, or add your own key to accumulate ..."}}}
UPSTREAM_MARKERS = ("rate-limited upstream", "rate limited upstream", "temporarily rate-limited",
                    "provider returned error", "no instances available", "no allowed providers",
                    "overloaded", "capacity", "upstream error", "retry shortly")

# Substrings identifying a 429 as THIS KEY's quota - OpenRouter's own limiter names the bucket it
# refused from, which is how the two cases can be told apart at all.
QUOTA_MARKERS = ("free-models-per-day", "rate limit exceeded", "requests per day", "per-day",
                 "daily limit", "add 10 credits", "x-ratelimit")

# Hard cap on the server-side routing array, from the endpoint's own 400:
#   {"error":{"message":"'models' array must have 3 items or fewer.","code":400}}
# It is not in the rate-limit docs. The Python sweep below still walks the WHOLE chain, so this
# truncates the routing HINT, not the coverage - which is why a five-model chain is still correct.
MAX_MODELS_ARRAY = 3

# Completion budget for the hosted arm. NOT `ReporterConfig.max_new_tokens` (1600), which belongs
# to the measured local Qwen arm and must not move - every figure in `results/evaluation.md` was
# produced under it.
#
# 8192 because EVERY MODEL IN THE CHAIN IS A REASONING MODEL (checked against
# `/api/v1/models`: all of them advertise `reasoning`), so part of the budget goes on thinking
# before the object is emitted. Measured: 1600 truncated nemotron-3-super on every clip, then 4096
# truncated dots-3-note-preview at 47.5 s with `unterminated JSON object`. 8192 is the FLOOR of the
# chain's `max_completion_tokens` (liquid/lfm-2.5-2.6b caps there; the others allow 235k-460k), so
# it is the largest value every model accepts.
#
# `max_tokens` is a CEILING, not a target - a model that needs 900 tokens still emits 900 and
# returns. So raising it costs nothing on a report that already fitted, and buys the reasoning
# models room on one that did not. What it does cost is worst-case latency, which is why
# `deadline_s` moved with it.
HOSTED_MAX_TOKENS = 8192

# The ceiling a single report's completion budget may escalate to. A 200 whose
# `finish_reason` is `length` was cut by the BUDGET, not the model - so one transport-level
# escalation (doubling) is honest: the retry produces a genuinely complete generation, unlike a
# parse repair. 16,384 is the stop: at the observed ~90 tok/s that is already ~3 minutes, past
# the 120 s POST timeout, so a model still truncating there cannot be served at all.
MAX_TOKENS_ESCALATION = 16_384

# Substrings identifying a 400/422 as the model refusing the SCHEMA, as opposed to the request being
# malformed for everyone. The distinction earned itself: a 400 about the `models` array was read as
# a schema refusal, so enforcement was silently dropped and the request retried - degrading the one
# guarantee `constrained=True` is for, to work around something that had nothing to do with it.
SCHEMA_REFUSAL_MARKERS = ("response_format", "json_schema", "structured output",
                          "structured_outputs", "json schema", "schema")


def _incomplete_json(content: str) -> bool:
    """True when `content` opens a JSON object and never closes it.

    THE SECOND TRUNCATION SIGNAL, and the one that actually caught dots-3-note-preview. The
    escalation below keys off `finish_reason == "length"`, which is the authoritative report - but
    a provider is free not to send it. Measured: a 200 whose `finish_reason` was not `length` and
    whose body was an unterminated object, so nothing escalated and the cut surfaced two layers
    downstream as `unterminated JSON object in response (truncated generation)` with the whole
    report lost.

    Sound only because `constrained=True` means the endpoint is enforcing a JSON schema
    server-side: under that promise an unbalanced object cannot be a legitimate completion, so it
    is evidence the generation was cut. The caller therefore checks this ONLY when constrained.

    This is a well-formedness question, not a semantic one, and it repairs nothing: the answer is
    used to decide whether to re-request with a larger budget. `CaregiverReporter` still owns
    parsing, `repair_claim`, and the hallucination measurement - and content that IS complete is
    handed back byte-for-byte.
    """
    depth, in_str, esc, opened = 0, False, False, False
    for ch in content:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
            opened = True
        elif ch in "}]":
            depth -= 1
    # `in_str` catches a cut inside a string literal, which leaves depth looking balanced.
    return opened and (depth > 0 or in_str)


def _mask(key: str) -> str:
    """Keys never appear in logs, errors or stats. Last four characters identify them enough."""
    return f"...{key[-4:]}" if len(key) > 4 else "...."


def _redact(body: str) -> str:
    """Strip the account id OpenRouter attaches to errors, before the body reaches a page.

    A 400 arrives as `{"error":{...},"user_id":"user_3IM..."}` and that body is quoted verbatim on
    the demo page, which is where the useful part of it belongs. The account id is not a credential
    and cannot authenticate anything, but it identifies the operator and it is not diagnostic, so
    it does not need to be on a screen shown to other people. Nothing else is removed: the message
    is exactly how the `models` cap was found.
    """
    return re.sub(r',?\s*"user_id"\s*:\s*"[^"]*"', "", body or "")


def classify_429(body: str) -> str:
    """`"upstream"`, `"quota"` or `"unknown"` - which axis to change after a 429.

    Explicit markers win, and a quota marker only counts when no upstream marker is present,
    because the observed upstream body also carries the word "rate". When neither appears the
    answer is `"unknown"` and the caller HEDGES - a short key cooldown *and* a move to the next
    model - so an ambiguous 429 cannot reproduce either failure already seen: parking twenty
    innocent keys for a minute each, or hammering one genuinely spent key.
    """
    low = (body or "").lower()
    up = any(m in low for m in UPSTREAM_MARKERS)
    if any(m in low for m in QUOTA_MARKERS) and not up:
        return "quota"
    return "upstream" if up else "unknown"


def discover_keys(env: dict[str, str] | None = None, *, prefix: str = "OPENROUTER_API_KEY",
                  max_n: int = 64) -> list[str]:
    """Collect keys from `PREFIX_1..N`, then `PREFIX`, then a comma-separated `PREFIX_LIST`.

    Three sources because the deployment has two shapes. Locally they are numbered environment
    variables. On Kaggle each secret is a separate named entry, so twenty of them is twenty
    clicks - `OPENROUTER_API_KEY_LIST` holds them comma-separated in one secret instead.

    Order is preserved and duplicates are dropped, so a key present both numbered and in the list
    is tried once. Nothing here validates a key: that costs a request, and the caller finds out
    on first use anyway.
    """
    src = os.environ if env is None else env
    out: list[str] = []
    for i in range(1, max_n + 1):
        v = (src.get(f"{prefix}_{i}") or "").strip()
        if v:
            out.append(v)
    single = (src.get(prefix) or "").strip()
    if single:
        out.append(single)
    for v in (src.get(f"{prefix}_LIST") or "").split(","):
        if v.strip():
            out.append(v.strip())
    seen: set[str] = set()
    return [k for k in out if not (k in seen or seen.add(k))]


@dataclass
class _KeyState:
    """Per-key counters. The evidence for whether rotation actually buys anything.

    `rate_limited` and `upstream` are separate columns on purpose: a run where `upstream`
    dominates is a run where rotation was never the lever, and that is exactly what the first
    measurement showed. Collapsing them into one number is how the wrong fix got built.
    """

    key: str
    ok: int = 0
    rate_limited: int = 0
    upstream: int = 0
    empty: int = 0
    errors: int = 0
    rejected: int = 0
    cooling_until: float = 0.0
    last_status: int | None = None

    def summary(self) -> dict[str, Any]:
        return {"key": _mask(self.key), "ok": self.ok, "429_quota": self.rate_limited,
                "429_upstream": self.upstream, "empty": self.empty, "errors": self.errors,
                "rejected": self.rejected, "last_status": self.last_status,
                "cooling_s": max(0.0, round(self.cooling_until - time.time(), 1))}


@dataclass
class _ModelState:
    """Per-model counters - the other axis, and the one that turned out to matter.

    `upstream` here is a property of the PROVIDER behind the model. When it is non-zero for the
    primary and zero for a fallback, the fallback chain did its job and the counters say so.
    """

    model: str
    calls: int = 0
    ok: int = 0
    upstream: int = 0
    empty: int = 0
    errors: int = 0
    schema_refused: bool = False
    cooling_until: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {"model": self.model, "calls": self.calls, "ok": self.ok,
                "429_upstream": self.upstream, "empty": self.empty, "errors": self.errors,
                "schema_refused": self.schema_refused,
                "cooling_s": max(0.0, round(self.cooling_until - time.time(), 1))}


def _post(url: str, payload: dict[str, Any], key: str,
          timeout: float) -> tuple[int, str, float | None]:
    """One POST. Returns (status, body, retry_after_s). Never raises for an HTTP status.

    `Retry-After` is returned because the endpoint tells us how long to wait and ignoring it is
    how six attempts got spent in 4.19 s against a provider that had said "retry shortly".

    `urllib` rather than `requests`: this runs inside a notebook whose dependency install order
    has already broken twice, and a zero-install transport cannot be the thing that fails. It is
    one request per report, so nothing here needs connection pooling.
    """
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 # OpenRouter asks for these for attribution; they are not secrets.
                 "HTTP-Referer": "https://github.com/PARTHG0106/BehaviorSense",
                 "X-Title": "BehaviorSense"})

    def _retry_after(headers) -> float | None:
        raw = (headers.get("Retry-After") or "").strip() if headers else ""
        try:
            return max(0.0, float(raw)) if raw else None
        except ValueError:
            return None            # the HTTP-date form; the caller's backoff covers it

    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", errors="replace"), _retry_after(r.headers)
    except urllib.error.HTTPError as e:
        return (e.code, e.read().decode("utf-8", errors="replace"),
                _retry_after(getattr(e, "headers", None)))
    except Exception as e:                                          # noqa: BLE001
        # Timeouts and DNS failures are indistinguishable from a 503 to the caller, and both
        # should rotate. 599 is not a real status; it is a local marker.
        return 599, f"{type(e).__name__}: {e}", None


class OpenRouterLLM:
    """`ReportLLM` over OpenRouter, with a provider-diverse model chain and key rotation.

        llm = OpenRouterLLM()                       # keys and fallback chain from defaults
        out = CaregiverReporter(llm).report(state)  # C1-C5 unchanged

    The retry policy has TWO axes because a 429 has two causes (see `classify_429`):

        sweep 0:  model A over keys ... | model B over keys ... | ...   <- no delay anywhere
        (back off)
        sweep 1:  the same, with cooled keys and cooled models skipped

    Waiting only ever happens *between* sweeps. Within one, a fresh key and a fresh provider are
    both free, and the first version of this file spent a 150 s budget discovering otherwise.

    `constrained=True` sends the schema as `response_format` so the endpoint enforces it
    server-side. That replaces `outlines`' local FSM with the same guarantee and no 152k-token
    index in host memory - and the measured difference between constrained and free decoding was
    8.2% vs 6.5% unfaithful at p=0.29, so neither arm depends on winning this argument. That is
    also why a model which *rejects* the schema is retried without it rather than dropped.
    """

    def __init__(self, model: str = DEFAULT_MODEL, *, keys: list[str] | None = None,
                 fallback_models: Sequence[str] = FALLBACK_MODELS,
                 max_tokens: int = HOSTED_MAX_TOKENS,
                 # PER REQUEST, and it has to cover a full escalated generation. Measured at
                 # ~90 tok/s on these free endpoints (4,096 tokens in ~45 s), so 8,192 is ~90 s and
                 # a 16,384 escalation is ~180 s. The old 120 s would have killed the escalated
                 # attempt itself - a timeout cancelling the fix for a truncation.
                 timeout: float = 300.0,
                 attempts: int = 0,
                 sweeps: int = 4, cooldown_s: float = 60.0, soft_cooldown_s: float = 5.0,
                 model_cooldown_s: float = 12.0, temperature: float = 0.0,
                 backoff_s: float = 2.0, max_backoff_s: float = 16.0,
                 # TOTAL wall clock for one report, and it had to grow with the token budget: at
                 # ~90 tok/s an 8,192 attempt is ~90 s and its 16,384 escalation another ~180 s, so
                 # the old 75 s cut the run off before the escalation could land. Affordable
                 # because Agent 4 runs LOCALLY (no tunnel cap) and `local_backend._blocking`
                 # heartbeats every 5 s, so the browser's watchdog - which measures SILENCE, not
                 # duration - stays fed however long generation takes.
                 deadline_s: float = 420.0, sleep: Callable[[float], None] = time.sleep,
                 post: Callable[..., tuple[int, str, float | None]] = _post,
                 route_server_side: bool = True, url: str = OPENROUTER_URL) -> None:
        self.model = model
        # Provider diversity is the whole value of this list, so it is a chain of ids on distinct
        # providers rather than "more of the same model". Also sent as OpenRouter's `models` array
        # (see `route_server_side`) so the router can do it in ONE round trip when it will.
        self.fallback_models = tuple(m for m in fallback_models if m and m != model)
        self.max_tokens = max_tokens
        self.timeout = timeout
        # `attempts` is a hard ceiling on POSTs per report, not a count of retries: with 5 models
        # and 20 keys the two loops could otherwise reach 100 requests. 0 derives it.
        self.attempts = attempts
        self.sweeps = sweeps
        self.cooldown_s = cooldown_s
        self.soft_cooldown_s = soft_cooldown_s
        self.model_cooldown_s = model_cooldown_s
        self.temperature = temperature
        self.backoff_s = backoff_s
        self.max_backoff_s = max_backoff_s
        self.deadline_s = deadline_s
        self.route_server_side = route_server_side
        self._sleep = sleep
        self._post = post
        self.url = url
        self.waited_s = 0.0
        self._t0 = time.time()
        self.calls = 0
        self.rotations = 0
        # Per-model completion budgets, escalated on `finish_reason: length` and REMEMBERED: a
        # model that needed 8,192 once should not re-pay a failed 4,096 on every report.
        self._budgets: dict[str, int] = {}
        self.token_escalations = 0
        self.served_by: str | None = None
        self.served_constrained: bool | None = None
        self._no_schema: set[str] = set()
        self._calls_now = 0
        found = keys if keys is not None else discover_keys()
        if not found:
            raise ValueError(
                "no OpenRouter keys found. Set OPENROUTER_API_KEY_1..N, or OPENROUTER_API_KEY, "
                "or OPENROUTER_API_KEY_LIST as a comma-separated list. On Kaggle these come from "
                "Add-ons -> Secrets; a key pasted into a cell is published by Save Version."
            )
        self._states = [_KeyState(k) for k in found]
        self._models = {m: _ModelState(m) for m in (self.model, *self.fallback_models)}
        self._cursor = 0

    @property
    def plan(self) -> list[str]:
        """The models tried, primary first. Public because the notebook prints it at startup."""
        return [self.model, *self.fallback_models]

    @property
    def name(self) -> str:
        # The page renders this. A report written by a hosted model must not be mistaken for one
        # written by the measured local checkpoint - and when the chain fell through to a fallback,
        # or when the schema had to be dropped, the label has to say so rather than imply the
        # primary answered under enforcement.
        served = self.served_by or self.model
        via = "" if served == self.model else f" via {served}"
        note = ", schema not enforced" if self.served_constrained is False else ""
        return f"{self.model}{via} (openrouter, {len(self._states)} key(s){note})"

    def stats(self) -> dict[str, Any]:
        """Per-key AND per-model outcomes - the evidence for which axis a failure was on.

        The first version of this reported keys only, and that is precisely why the wrong fix got
        built: thirteen keys each with one 429 looks like a key problem until you can also see
        that one model accounted for all thirteen. Both columns are here now.
        """
        return {"model": self.model, "served_by": self.served_by,
                "constrained": self.served_constrained, "plan": self.plan,
                "calls": self.calls, "rotations": self.rotations,
                "token_escalations": self.token_escalations,
                "budgets": {m: b for m, b in self._budgets.items() if b != self.max_tokens},
                "waited_s": round(self.waited_s, 1),
                "models": [s.summary() for s in self._models.values()],
                "keys": [s.summary() for s in self._states]}

    # ---- budget and cursor -------------------------------------------------------------------

    def _cap(self) -> int:
        """POST ceiling for one report. Derived so 20 keys are reachable but 100 requests are not.

        `2 * len(keys)` gives every key a second chance after a cooldown; the `4 * models` floor
        keeps a single-key deployment able to walk the whole chain, which is the case where the
        chain is the only thing that can help.
        """
        return self.attempts if self.attempts > 0 else max(2 * len(self._states),
                                                           4 * len(self._models), 8)

    def _spent(self) -> bool:
        return self._calls_now >= self._cap()

    def _left(self, t_start: float) -> float:
        # Deadline from BOTH the wall clock and the accumulated waits. A test with an injected
        # `sleep` never advances the clock, so wall-time alone let a 40 s deadline spend 48 s of
        # intended waiting - and a production `time.sleep` returning early would do the same.
        return self.deadline_s - max(time.time() - t_start, self.waited_s)

    def _now(self) -> float:
        """The clock every cooldown is measured against: `max(wall, start + waited)`.

        Same reason the deadline uses that form, applied consistently. An injected `sleep` never
        advances the wall clock, so a cooldown compared against `time.time()` alone is still in
        force after the code has "waited" well past it - which means no test could ever observe
        what happens AFTER a backoff. That is the vacuous-test shape this project has already been
        bitten by twice, and the region those two 429 bugs both lived in.
        """
        return max(time.time(), self._t0 + self.waited_s)

    def _wake_in(self) -> float:
        """Seconds until at least one model AND one key are eligible again.

        Without this the sweeps degenerate: a 12 s model cooldown against a 2 s backoff meant
        sweeps 1-3 woke up, found every model still cooling, skipped every one of them and slept
        again - three sweeps that issued no requests at all. A sweep should wake when there is
        something for it to do, so the wait is the larger of the formula and this.
        """
        now = self._now()
        models = [s.cooling_until for s in self._models.values()]
        keys = [s.cooling_until for s in self._states]
        need_m = 0.0 if any(c <= now for c in models) else min(models) - now
        need_k = 0.0 if any(c <= now for c in keys) else min(keys) - now
        return max(0.0, need_m, need_k)

    def _next_state(self) -> _KeyState | None:
        """The next key not cooling down, starting from the cursor. None when all are cooling."""
        now = self._now()
        n = len(self._states)
        for off in range(n):
            st = self._states[(self._cursor + off) % n]
            if st.cooling_until <= now:
                self._cursor = (self._cursor + off) % n
                return st
        return None

    def _advance(self) -> None:
        self._cursor = (self._cursor + 1) % len(self._states)
        self.rotations += 1

    def _route(self, model: str) -> list[str]:
        """The `models` array for one request: `model` first, then the chain onward, capped at 3.

        Two details earn their keep. It walks the chain FORWARD from `model` and wraps, so the
        truncated array never spends a slot re-offering something already tried this sweep. And a
        model currently cooling is dropped, because with only three slots, offering the router a
        provider that just returned `rate-limited upstream` wastes a third of the hint.
        """
        plan = self.plan
        i = plan.index(model)
        ordered = plan[i:] + plan[:i]
        now = self._now()
        eligible = [m for m in ordered
                    if m == model or self._models[m].cooling_until <= now]
        return eligible[:MAX_MODELS_ARRAY]

    def _body(self, model: str, prompt: str, *, constrained: bool,
              schema: dict[str, Any], max_tokens: int | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            # Greedy. Agent 4's job is to restate numbers it was given; sampling only adds ways
            # to paraphrase a figure wrongly, and C5 exists because that already happens.
            "temperature": self.temperature,
            # REASONING MODELS THINK IN THE COMPLETION BUDGET, and the fallback chain is full of
            # them (nemotron-3-super, dots-3-note-preview). Measured twice: nemotron returned
            # `unterminated JSON (truncated generation)` at 1,600 tokens, and after the budget
            # was raised, dots-3-note did the same at 4,096 - both burned the entire budget
            # before emitting the object. `effort: low` is OpenRouter's NORMALISED control:
            # supported models think less, unsupported ones ignore it. The report restates
            # twenty-seven numbers; there is nothing here that needs deep reasoning.
            "reasoning": {"effort": "low"},
        }
        if self.route_server_side and self.fallback_models:
            # OpenRouter tries these in order when the primary is unavailable, inside ONE round
            # trip, and it knows provider health better than this file can. The Python loop below
            # is the net beneath that, not a replacement for it - which is what makes the 3-item
            # cap survivable: the array is a hint, the sweep is the guarantee.
            route = self._route(model)
            if len(route) > 1:
                body["models"] = route
        if constrained and model not in self._no_schema:
            # Server-side JSON-schema enforcement. `strict` is what makes it a guarantee rather
            # than a hint; without it the field is advisory and the response can still drift.
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "caregiver_report", "strict": True, "schema": schema},
            }
        return body

    # ---- one model, every eligible key, no waiting --------------------------------------------

    def _try_model(self, model: str, prompt: str, *, constrained: bool,
                   schema: dict[str, Any]) -> tuple[str | None, int, str, float | None]:
        """Walk the keys for one model. Returns (content|None, last_status, last_body, retry_after).

        Returning early with `None` is how "change the MODEL" is expressed: an upstream 429 means
        every remaining key reaches the same saturated provider, so spending them here is exactly
        the mistake the first version made.
        """
        ms = self._models[model]
        budget = self._budgets.get(model, self.max_tokens)
        body = self._body(model, prompt, constrained=constrained, schema=schema,
                          max_tokens=budget)
        status, text, hint = 0, "", None
        for _ in range(len(self._states) + 1 + MAX_TOKENS_ESCALATION.bit_length()):
            if self._spent():
                break
            st = self._next_state()
            if st is None:
                break
            self.calls += 1
            self._calls_now += 1
            ms.calls += 1
            status, text, hint = self._post(self.url, body, st.key, self.timeout)
            st.last_status = status

            if status == 200:
                try:
                    doc = json.loads(text)
                    content = doc["choices"][0]["message"]["content"]
                    # Read the model back rather than assume it: with a `models` array OpenRouter
                    # may have routed elsewhere, and the page names whoever wrote the report.
                    served = str(doc.get("model") or model)
                    finish = (doc.get("choices") or [{}])[0].get("finish_reason")
                except Exception as e:                              # noqa: BLE001
                    # A 200 whose body is not the expected shape is an upstream change, not a
                    # model failure, and must not be mistaken for an unparseable report.
                    raise RuntimeError(
                        f"OpenRouter returned 200 with an unexpected body "
                        f"({type(e).__name__}: {e}). First 300 chars: {text[:300]!r}"
                    ) from None
                # TWO SIGNALS FOR ONE CONDITION. `finish_reason == "length"` is the endpoint's own
                # report and is authoritative when present; `_incomplete_json` is the fallback for
                # a provider that does not send it. dots-3-note-preview returned a 200 with an
                # unterminated object and some other finish reason, so the first signal alone left
                # the cut to surface two layers downstream as a lost report.
                cut = finish == "length" or (constrained and _incomplete_json(content or ""))
                if cut and budget < MAX_TOKENS_ESCALATION:
                    # THE CUT WAS THE BUDGET, not a model, provider or key failure - it is this
                    # file's own `max_tokens` truncating a generation that was otherwise
                    # proceeding. Measured on dots-3-note-preview: a 200 that was all reasoning
                    # and no object, which surfaced downstream as `unterminated JSON (truncated
                    # generation)` and cost the whole report. Escalating the budget by doubling
                    # and retrying is honest - the retry yields a genuinely complete generation,
                    # which a parse repair never does - and it stays here in the transport, where
                    # the evidence for it was read.
                    budget = min(budget * 2, MAX_TOKENS_ESCALATION)
                    self._budgets[model] = budget
                    self.token_escalations += 1
                    body = self._body(model, prompt, constrained=constrained, schema=schema,
                                      max_tokens=budget)
                    continue            # same key, same model: the provider answered fine
                if cut:
                    # At the ceiling and still cutting: this model cannot finish inside the
                    # budget this file is willing to spend. Say so in the body the caller may
                    # report, and CHANGE MODEL - burning the remaining keys on a model that
                    # cannot fit is the same mistake the upstream-429 branch exists to avoid.
                    # The model is also PARKED for the rest of this call: without that, every
                    # later sweep re-asked it and burned a POST to be told the same thing
                    # (measured: four identical 4,096-length 200s from four sweeps).
                    text = (f"truncated at max_tokens={budget} "
                            f"(finish_reason={finish or 'not reported'}): the model never reached "
                            f"the end of its own output")
                    ms.errors += 1
                    # PARKED FOR THE REST OF THIS CALL, expressed as such rather than as a guessed
                    # number of seconds. A flat 300 s stopped covering the call the moment
                    # `deadline_s` grew to 420, so a model that had already proved it cannot fit
                    # inside the maximum budget was re-asked mid-report and burned another POST
                    # (measured: two identical ceiling replies in one generate()). Deadline+1 is
                    # exactly "no later sweep of THIS report", and leaves the next report free to
                    # try it again on a possibly shorter prompt.
                    ms.cooling_until = self._now() + self.deadline_s + 1.0
                    return None, status, text, hint
                if not (content or "").strip():
                    # An empty completion is a real outcome on a free endpoint. Rotate rather
                    # than hand `CaregiverReporter` an empty string to call a parse failure.
                    st.empty += 1
                    ms.empty += 1
                    self._advance()
                    continue
                st.ok += 1
                ms.ok += 1
                self.served_by = served
                self.served_constrained = bool(constrained) and model not in self._no_schema
                return content, status, text, hint

            if status == 429:
                kind = classify_429(text)
                if kind == "upstream":
                    # THE KEY IS INNOCENT. Do not cool it - thirteen keys were parked for a minute
                    # each on exactly this body, and the counters proved it wrong.
                    st.upstream += 1
                    ms.upstream += 1
                    ms.cooling_until = self._now() + self.model_cooldown_s
                    return None, status, text, hint
                if kind == "quota":
                    st.rate_limited += 1
                    st.cooling_until = self._now() + self.cooldown_s
                    self._advance()
                    continue
                # Ambiguous: hedge on both axes, cheaply. A short key cooldown plus the next model
                # covers whichever cause it was without committing to either.
                st.rate_limited += 1
                st.cooling_until = self._now() + self.soft_cooldown_s
                ms.cooling_until = self._now() + self.soft_cooldown_s
                self._advance()
                return None, status, text, hint

            if status == 402:
                st.errors += 1
                # Out of credit is not a transient state; park the key for the session.
                st.cooling_until = self._now() + 86_400.0
                self._advance()
                continue

            if status == 404:
                # The model id is gone. Park the MODEL, not the key: a fallback chain pinned in
                # source outlives the free tier it names, and one dead id must not cost 20 keys.
                ms.errors += 1
                ms.cooling_until = self._now() + 86_400.0
                return None, status, text, hint

            if status in (400, 422):
                low = (text or "").lower()
                refused = any(m in low for m in SCHEMA_REFUSAL_MARKERS)
                if refused and constrained and model not in self._no_schema:
                    # The model refused our SCHEMA. Retry it once without `response_format` rather
                    # than losing a working provider: `CaregiverReporter` owns parsing and
                    # `repair_claim`, and constrained vs free measured 8.2% vs 6.5% unfaithful at
                    # p=0.29. `served_constrained` records that enforcement was dropped, so the
                    # page can say so instead of implying a guarantee it did not get.
                    self._no_schema.add(model)
                    ms.schema_refused = True
                    body = self._body(model, prompt, constrained=False, schema=schema)
                    continue
                # Any other 400 is the request SHAPE, wrong for every model and every key alike.
                # This branch exists because the alternative was measured: `'models' array must
                # have 3 items or fewer` was read as a schema refusal, so enforcement was dropped
                # and the same malformed body sent again - a silent downgrade of the one guarantee
                # `constrained=True` buys, to work around something unrelated to it. Failing here
                # is how that limit got found, and it belongs in code rather than in a retry.
                st.errors += 1
                ms.errors += 1
                raise RuntimeError(
                    f"OpenRouter {status} with key {_mask(st.key)} on {model}: "
                    f"{_redact(text)[:300]} - this is the request itself, so no other key or model "
                    f"will fix it."
                )

            if status in (401, 403):
                # This CREDENTIAL is bad, not this request. Park it for the session and carry on:
                # with twenty keys, one revoked one must not end a run. If they are all bad, the
                # final message says exactly that rather than blaming the model.
                st.rejected += 1
                st.errors += 1
                st.cooling_until = self._now() + 86_400.0
                self._advance()
                continue

            if status in ROTATE_ON:
                st.errors += 1
                ms.errors += 1
                self._advance()
                continue

            # Anything left is about the request itself and no other key or model will fix it.
            st.errors += 1
            raise RuntimeError(
                f"OpenRouter {status} with key {_mask(st.key)} on {model}: {_redact(text)[:300]}")
        return None, status, text, hint

    # ---- sweeps ------------------------------------------------------------------------------

    def generate(self, prompt: str, *, constrained: bool, schema: dict[str, Any]) -> str:
        """Prompt in, JSON string out. Sweeps the model chain over the keys; backs off per sweep.

        Returns the model's raw content, NOT a repaired object: `CaregiverReporter` owns parsing
        and `repair_claim`, and a transport that quietly fixed malformed output would hide exactly
        the failure the hallucination measurement is about.
        """
        t_start = self._t0 = time.time()
        self.waited_s = 0.0
        self._calls_now = 0
        last: tuple[str, int, str] | None = None
        hint: float | None = None
        for sweep in range(max(1, self.sweeps)):
            if self._spent() or self._left(t_start) <= 0:
                break
            if sweep:
                # Waiting happens HERE and nowhere else: a repeat of something that already failed
                # is the only thing that needs to wait. Growth is exponential and capped;
                # `Retry-After` overrides the formula because the endpoint knows better than it,
                # and `_wake_in()` overrides both so a sweep never wakes with nothing to do.
                wait = min(self.max_backoff_s, self.backoff_s * (2 ** (sweep - 1)))
                if hint is not None:
                    wait = max(wait, hint)
                wait = max(wait, self._wake_in())
                left = self._left(t_start)
                if left <= 0 or wait > left:
                    # Sleeping to the deadline and then finding everything still cooling wastes
                    # the rest of the budget; stopping now reports the real state instead.
                    break
                self.waited_s += wait
                self._sleep(wait)
            for model in self.plan:
                if self._models[model].cooling_until > self._now():
                    continue
                content, status, text, rh = self._try_model(
                    model, prompt, constrained=constrained, schema=schema)
                if rh is not None:
                    hint = rh if hint is None else max(hint, rh)
                if content is not None:
                    return content
                if status:
                    last = (model, status, text)
                if self._spent() or self._left(t_start) <= 0:
                    break

        now = self._now()
        cooling = [s for s in self._states if s.cooling_until > now]
        rejected = [s for s in self._states if s.rejected]
        model, status, text = last if last else (self.model, 0, "")

        if rejected and len(rejected) == len(self._states):
            raise RuntimeError(
                f"every one of {len(self._states)} OpenRouter key(s) was rejected with 401/403, so "
                f"this is the credentials, not the endpoint. Check that the keys are current and "
                f"that nothing truncated them - last body: {_redact(text)[:200]}. stats(): {self.stats()}"
            )
        if cooling and len(cooling) == len(self._states):
            raise RuntimeError(
                f"all {len(self._states)} OpenRouter key(s) are rate limited or out of credit; the "
                f"earliest recovers in {max(0.0, min(s.cooling_until for s in cooling) - now):.0f}s. "
                f"Free endpoints allow 20 requests/minute and 50/day below 10 purchased credits "
                f"(1,000/day at or above it). stats(): {self.stats()}"
            )

        upstream = sum(s.upstream for s in self._models.values())
        walked = [m for m, s in self._models.items() if s.calls]
        raise RuntimeError(
            f"OpenRouter gave up after {self._calls_now} request(s) across {len(walked)} of "
            f"{len(self._models)} model(s) in {time.time() - t_start:.0f}s "
            f"({self.waited_s:.0f}s of it waiting); last was {model} -> {status}: "
            f"{_redact(text)[:200]}. "
            f"{upstream} of those were 'rate-limited upstream', which is the free PROVIDER "
            f"saturated and not your keys - measured once as 13 keys returning 13 one-each 429s "
            f"while the dashboards showed several of them with no requests that day. Rotation "
            f"cannot help that; a different provider can, which is what the model chain is for. "
            f"If every provider in the chain is saturated at the same moment, the fixes that "
            f"remain are a BYOK provider key at openrouter.ai/settings/integrations for dedicated "
            f"limits, or 10 purchased credits to lift 50 requests/day to 1,000. "
            f"stats(): {self.stats()}"
        )


__all__ = ["DEFAULT_MODEL", "FALLBACK_MODELS", "HOSTED_MAX_TOKENS", "MAX_MODELS_ARRAY",
           "MAX_TOKENS_ESCALATION",
           "OPENROUTER_URL",
           "QUOTA_MARKERS", "SCHEMA_REFUSAL_MARKERS", "UPSTREAM_MARKERS", "OpenRouterLLM",
           "classify_429", "discover_keys"]
