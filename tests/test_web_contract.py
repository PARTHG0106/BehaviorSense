"""The `/video` contract: notebook 05, `web/dev_backend.py`, and the front end must agree.

Three implementations describe one payload. The Kaggle notebook produces it on a GPU, the
dev stub produces it on a laptop so the page can be built without one, and `stages.js` reads
it. Any two of those drifting apart is a defect that only shows up during a demo:

  - stub drifts from notebook -> the page is developed against a shape the GPU never sends
  - notebook drifts from JS     -> the live run renders blanks where figures should be
  - a renamed key               -> `undefined` in the DOM, which looks like a model that
                                   returned nothing rather than like a typo

This suite reads the notebook's own source with `ast`, so it tests the shape the notebook
actually returns rather than a copy of it maintained here. It caught the drift it was written
for: `upload.js` was still reading `n_windows`, `fall_posterior`, `peak_fall_posterior` and
`report_unavailable_because` months after `/video` stopped emitting any of them.

Run: python tests/test_web_contract.py
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "05_serve_inference_online.ipynb"
WEB = ROOT / "web"

# What `stages.js` and `upload.js` actually dereference. Hand-listed on purpose: it is short,
# and a generated list would drift with the same silence it is meant to detect.
JS_TOP_LEVEL = ("fps", "width", "height", "frames_kept", "truncated", "reid", "tracks",
                "n_people", "stages", "checks", "timing", "tau", "report_provenance")
JS_STAGE = ("agent", "name", "status", "elapsed_s", "payload")
JS_CHECK = ("claim_id", "text", "evidence_ref", "claimed_value", "claimed_pct_change",
            "direction", "C1_ref_exists", "C2_value_matches", "C3_pct_matches",
            "C4_direction_consistent", "C5_prose_quoted_value", "faithful", "notes")
JS_TIMING = ("pose_s", "classify_s", "behaviour_s", "report_s")
JS_TRACK = ("track_id", "role", "segments", "n_segments", "fall_segments")
JS_SEGMENT = ("name", "t0", "t1", "confidence")
JS_STAGE3 = ("baseline_provenance", "baseline_days", "real_days_from_this_video",
             "observed_hours", "is_reliable", "features_from_video", "deviations", "alerts",
             "caveats")
JS_STAGE4 = ("model", "constrained_decoding", "summary", "recommendation", "escalate",
             "claims_emitted", "claims_scorable", "hallucination_rate", "parse_failed")


def _video_cell() -> str:
    doc = json.loads(NB.read_text(encoding="utf-8"))
    cells = [c["source"] for c in doc["cells"] if c["cell_type"] == "code"
             and '@app.post("/video")' in c["source"]]
    assert len(cells) == 1, f"expected exactly one /video cell, found {len(cells)}"
    return cells[0]


def _dict_keys(node: ast.Dict) -> set[str]:
    return {k.value for k in node.keys if isinstance(k, ast.Constant)
            and isinstance(k.value, str)}


def notebook_shape() -> dict[str, set[str]]:
    """Key sets the notebook's `/video` emits, read out of its own AST."""
    tree = ast.parse(_video_cell())
    # The staging generator, not the `video` endpoint. `/video` now returns a StreamingResponse
    # and the payload is the last thing `_video_stages` yields, because the tunnel drops a
    # request whose origin has not answered in ~2 minutes and this one takes longer.
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
               and n.name == "_video_stages"), None)
    assert fn is not None, "no `_video_stages` generator in the /video cell"

    # `yield "result", {...}` - the tuple's second element is the top-level payload. Matched on
    # the "result" tag rather than on position, so adding a stage yield cannot silently
    # redefine what this test thinks the payload is.
    # There are now TWO `yield "result"` sites: the full payload and the truncated one for
    # `?stages=12`, which stops at the Agent 2 -> Agent 3 seam so a caller can run Agents 3-4
    # where its LLM keys live. The CANONICAL shape is the richer of the two - taking whichever
    # `ast.walk` yields last would silently make the truncated payload the contract.
    final = None
    for n in ast.walk(fn):
        if (isinstance(n, ast.Yield) and isinstance(n.value, ast.Tuple)
                and len(n.value.elts) == 2
                and isinstance(n.value.elts[0], ast.Constant)
                and n.value.elts[0].value == "result"
                and isinstance(n.value.elts[1], ast.Dict)):
            if final is None or len(_dict_keys(n.value.elts[1])) > len(_dict_keys(final)):
                final = n.value.elts[1]
    assert final is not None, (
        'no `yield "result", {...}` literal in `_video_stages` - the payload shape cannot be '
        "read, so W1/W2 would compare the stub against nothing")
    top = _dict_keys(final)

    # `timing` is a nested literal in the same payload.
    timing: set[str] = set()
    for k, v in zip(final.keys, final.values):
        if isinstance(k, ast.Constant) and k.value == "timing" and isinstance(v, ast.Dict):
            timing = _dict_keys(v)
    assert timing, "no `timing` dict literal in the /video payload"

    # `_stage(...)` is defined in the same cell; its literal is the per-stage record.
    stage_fn = next((n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name == "_stage"), None)
    assert stage_fn is not None, "no `_stage` helper in the /video cell"
    stage = _dict_keys(next(n.value for n in ast.walk(stage_fn)
                            if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict)))

    # The per-claim record is the dict passed to `checks.append(...)`.
    check: set[str] = set()
    for n in ast.walk(fn):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "append" and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "checks" and n.args
                and isinstance(n.args[0], ast.Dict)):
            check = _dict_keys(n.args[0])
    assert check, "no `checks.append({...})` literal found"

    # Stage 3 and 4 payloads are assigned as literals to `stage3` / `stage4`.
    payloads: dict[str, set[str]] = {}
    for n in ast.walk(fn):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in ("stage3", "stage4")
                and isinstance(n.value, ast.Dict)):
            keys = _dict_keys(n.value)
            if len(keys) > 2:                    # skip the `{"error": ...}` fallback
                payloads[n.targets[0].id] = keys

    return {"top": top, "timing": timing, "stage": stage, "check": check,
            "stage3": payloads.get("stage3", set()), "stage4": payloads.get("stage4", set())}


def stub_payload() -> dict:
    spec = importlib.util.spec_from_file_location("bs_dev", WEB / "dev_backend.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._video_payload()


def test_w1_stub_returns_every_key_the_notebook_does():
    """Top-level parity. The stub may not be missing anything the GPU will send."""
    nb = notebook_shape()
    stub = set(stub_payload())
    missing = sorted(nb["top"] - stub - {"frames"})   # `frames` is present in both; belt
    extra = sorted(stub - nb["top"])
    assert not missing, (
        f"dev_backend omits {missing}, which notebook 05 returns. The page would be built "
        "against a shape the GPU never sends.")
    assert not extra, (
        f"dev_backend invents {extra}, which notebook 05 does not return. The page would "
        "render fields that vanish the moment it talks to a real backend.")
    print(f"  W1 {len(nb['top'])} top-level keys, exact match between notebook and stub")


def test_w2_nested_records_match():
    """`stages`, `timing` and `checks` are where a rename hides."""
    nb = notebook_shape()
    p = stub_payload()

    assert set(p["timing"]) == nb["timing"], (
        f"timing keys differ: stub {sorted(p['timing'])} vs notebook {sorted(nb['timing'])}")
    for s in p["stages"]:
        assert set(s) == nb["stage"], (
            f"stage {s.get('agent')} keys {sorted(s)} != notebook {sorted(nb['stage'])}")
    for c in p["checks"]:
        assert set(c) == nb["check"], (
            f"check keys {sorted(c)} != notebook {sorted(nb['check'])}")

    s3 = next(s["payload"] for s in p["stages"] if s["agent"] == 3)
    s4 = next(s["payload"] for s in p["stages"] if s["agent"] == 4)
    assert set(s3) == nb["stage3"], (
        f"Agent 3 payload {sorted(set(s3) ^ nb['stage3'])} differs from the notebook")
    assert set(s4) == nb["stage4"], (
        f"Agent 4 payload {sorted(set(s4) ^ nb['stage4'])} differs from the notebook")
    print(f"  W2 stage record ({len(nb['stage'])} keys), timing ({len(nb['timing'])}), "
          f"check ({len(nb['check'])}), Agent 3 ({len(nb['stage3'])}) and Agent 4 "
          f"({len(nb['stage4'])}) payloads all match")


def test_w3_every_field_the_front_end_reads_exists():
    """The half the AST cannot see: what the JS actually dereferences."""
    p = stub_payload()
    for k in JS_TOP_LEVEL:
        assert k in p, f"stages.js/upload.js read `{k}`, which the payload does not carry"
    for k in JS_TIMING:
        assert k in p["timing"], f"upload.js reads timing.{k}"
    assert len(p["stages"]) == 4, f"the page draws four agent cards, got {len(p['stages'])}"
    for i, s in enumerate(p["stages"], start=1):
        assert s["agent"] == i, f"stages are out of order at position {i}: agent {s['agent']}"
        for k in JS_STAGE:
            assert k in s, f"stages.js reads stage.{k}"
    for k in JS_STAGE3:
        assert k in p["stages"][2]["payload"], f"stages.js reads Agent 3's `{k}`"
    for k in JS_STAGE4:
        assert k in p["stages"][3]["payload"], f"stages.js reads Agent 4's `{k}`"
    for c in p["checks"]:
        for k in JS_CHECK:
            assert k in c, f"stages.js reads check.{k}"
    for t in p["stages"][1]["payload"]["per_track"]:
        for k in JS_TRACK:
            assert k in t, f"timelineMarkup reads track.{k}"
        for s in t["segments"]:
            for k in JS_SEGMENT:
                assert k in s, f"timelineMarkup reads segment.{k}"
    print(f"  W3 all {len(JS_TOP_LEVEL)} top-level, {len(JS_STAGE3)} Agent 3, "
          f"{len(JS_STAGE4)} Agent 4, {len(JS_CHECK)} check and {len(JS_TRACK)} track "
          "fields the JS reads are present")


def test_w3b_every_page_module_parses_as_an_es_module():
    """The page's JS must actually load in a browser, and `node --check` does not prove it.

    MEASURED: a ternary nested inside a template literal was written with two double-quoted
    strings spanning line breaks - invalid JavaScript. `node --check web/scripts/stages.js`
    exited 0 anyway, because it checks the file as a CommonJS script; the browser parses it
    as an ES module and would have rendered a blank page. The difference is not academic:
    every module here uses `import`, so the browser's parser is the only one that counts.

    `web/scripts/_parse_check.mjs` imports each module for real and treats ONLY SyntaxError
    as failure - a module that parses and then throws for want of `document` or an import map
    is fine here. Skipped rather than failed when node is absent, because a Python-only
    machine should still be able to run this suite.
    """
    import shutil
    import subprocess

    if not shutil.which("node"):
        print("  W3b SKIP: node is not on PATH, so the browser parser cannot be consulted")
        return
    checker = WEB / "scripts" / "_parse_check.mjs"
    assert checker.is_file(), f"{checker} is missing - the parse check cannot run"
    r = subprocess.run(["node", str(checker)], cwd=ROOT, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, (
        "a page module does not parse as an ES module, so the browser would render a blank "
        "page: " + r.stdout + r.stderr)
    last = [ln for ln in r.stdout.splitlines() if ln.strip()][-1]
    print(f"  W3b {last.strip()} (as ES modules - `node --check` treats them as CommonJS "
          "and accepts strings that span line breaks)")


def test_w4_no_front_end_file_reads_a_retired_field():
    """The exact drift this suite was written for, pinned so it cannot come back."""
    retired = ("n_windows", "fall_posterior", "peak_fall_posterior",
               "report_unavailable_because")
    hits = []
    for path in sorted((WEB / "scripts").glob("*.js")):
        src = path.read_text(encoding="utf-8")
        for name in retired:
            if name in src:
                hits.append(f"{path.name}: {name}")
    assert not hits, (
        f"front end still reads retired /video fields: {hits}. `/video` stopped emitting "
        "these when the staged contract landed; reading them yields `undefined`, which "
        "renders as a model that returned nothing rather than as a typo.")
    # And the stub must not resurrect them either.
    body = json.dumps(stub_payload())
    for name in retired:
        assert f'"{name}"' not in body, f"dev_backend still emits retired field `{name}`"
    print(f"  W4 none of the {len(retired)} retired fields appear in web/scripts/*.js or "
          "in the stub payload")


def test_w5_the_stub_still_fails_one_claim_with_real_arithmetic():
    """A stub where everything passes proves only that the happy path renders."""
    p = stub_payload()
    failing = [c for c in p["checks"] if not c["faithful"]]
    assert len(failing) == 1, (
        f"{len(failing)} claims fail; the fixture is built so exactly one does, because the "
        "page has to be seen REJECTING something")
    c = failing[0]
    assert c["C1_ref_exists"] and c["C2_value_matches"], (
        "the fault should pass C1 and C2 - a claim that fails everything does not "
        "demonstrate that C4 is the check doing the work")
    assert not c["C4_direction_consistent"], "the injected fault is an inverted direction"
    assert c["notes"], (
        "the failing claim carries no notes, so the withheld margin renders the generic "
        "fallback instead of the arithmetic that failed - the weaker version of the feature")
    assert any("direction field wrong" in n for n in c["notes"]), c["notes"]
    assert p["checks"] and all(isinstance(x["notes"], list) for x in p["checks"])

    # C5 consistency: every claim the stub calls faithful must actually echo its value in
    # its own prose. A stub claim that passed C1-C4 but failed C5 would render as
    # "five cells, one dashed" on a claim marked shown - the stub contradicting itself.
    for x in p["checks"]:
        if x["faithful"] and x["claimed_value"] is not None:
            assert x["C5_prose_quoted_value"], (
                f"{x['claim_id']} is marked faithful but its prose never quotes "
                f"{x['claimed_value']} - the stub fails its own C5")
    print(f"  W5 claim {c['claim_id']} passes C1-C2, fails C4, and its margin carries "
          f"{len(c['notes'])} real note(s): {c['notes'][0][:52]}...; every faithful claim "
          "echoes its value (C5-consistent)")


def _video_tail_callable(ns: dict) -> object:
    """Compile `_video_stages` out of the notebook cell, as a callable generator.

    Everything upstream of it needs a GPU, an ONNX session and a real upload. `_video_stages`
    itself is ordinary Python over schema objects, and that is where the bugs have actually
    been. It exists as a separate function precisely so transport can be skipped here: the
    streaming wrapper knows about HTTP, this does not.

    `_blocking` is pulled from the endpoints cell rather than stubbed. It runs the slow calls on
    a worker thread and is the only thing bounding stream silence, so a stub would leave the
    mechanism that caused the reported failure untested.
    """
    doc = json.loads(NB.read_text(encoding="utf-8"))
    blocking = next(c["source"] for c in doc["cells"] if c["cell_type"] == "code"
                    and "def _blocking(" in c["source"])
    bmark = re.search(r'^def _blocking\(.*?(?=^\S|\Z)', blocking, re.M | re.S)
    assert bmark, "could not isolate `_blocking` from the endpoints cell"
    exec(compile("import time\n" + bmark.group(0), "<notebook05:_blocking>", "exec"), ns)

    src = _video_cell()
    # Signature-agnostic: it grew a `stages` parameter for the seam split, and pinning the exact
    # argument list made this fail for a reason that had nothing to do with the payload.
    mark = re.search(r'^def _video_stages\(', src, re.M)
    assert mark, ("`def _video_stages(poses, t_start, t_pose)` is gone from the /video cell, so "
                  "this test can no longer reach the staging code. Re-point it.")
    # To the end of the cell: it is the last function in it.
    exec(compile(src[mark.start():], "<notebook05:_video_stages>", "exec"), ns)
    return ns["_video_stages"]


def _w6_namespace(reporter=None):
    """Everything `_video_stages` closes over, with stubs for the two models."""
    import time

    import numpy as np

    sys.path.insert(0, str(ROOT / "src"))
    from behaviorsense.agents.activity import (
        CLASS_NAMES,
        FALLEN,
        FALLING,
        N_CLASSES,
        ActivityConfig,
    )
    from behaviorsense.agents.reasoning.reporter import CaregiverReporter, FaithfulStubLLM
    from behaviorsense.eval.activity_eval import logit_adjust
    from behaviorsense.pipeline import ActivityPipeline, observed_hours_from_frames
    from behaviorsense.schemas import Role
    from behaviorsense.video import rebuild_observations

    class StubClf:
        # Confident on one class so Viterbi yields real segments rather than all-abstain -
        # an empty `segments` list would skip the staging loop and vacuously pass.
        def logits(self, windows):
            k = len(windows) if hasattr(windows, "__len__") else windows.shape[0]
            lg = np.random.default_rng(0).normal(0, 1, (k, N_CLASSES)).astype(np.float32)
            lg[:, 3] += 6.0
            return lg

    # THE REAL `class_prior`, not a re-implementation. The stub here took only `y`, so when the
    # notebook started passing `n_classes=lg.shape[1]` - the fix for a 22-class head - this
    # namespace raised `unexpected keyword argument` while the notebook was correct. A stub that
    # drifts from the signature it stands in for turns an execution test back into a shape test.
    from behaviorsense.eval.activity_eval import class_prior

    # The REAL subject decision, not a re-implementation. It lives in `service/staging.py` and the
    # notebook imports it in a cell W6 does not execute. Supplying the real one is the whole point:
    # the merge rule decides which tracks' activity survives into Agent 3's features, and a stub
    # that always picked the most-present track would let the discard bug back in unnoticed.
    from behaviorsense.service.staging import assert_subject as _assert_subject

    return {
        # SAMPLE_FPS and CLASS_NAMES_SERVED are set by cells W6 does not execute, so they are
        # supplied here. Both are real serving parameters: the rate the checkpoint was trained at,
        # and which name list a 20- vs 22-class head indexes.
        "time": time, "np": np, "TAU": 0.25, "TEMPERATURE": 0.6, "SAMPLE_FPS": 15.0,
        "_assert_subject": _assert_subject,
        # `_rss_mb` lives in the endpoints cell W6 does not execute. Returning None exercises the
        # branch a non-Linux host takes, which is the honest default for a laptop.
        "_rss_mb": lambda: None,
        "CLASS_NAMES": CLASS_NAMES,
        # `class_names_served()` lives in the /video cell above `_video_stages` and is resolved per
        # request from the loaded classifier. W6 executes only the staging function, so the
        # resolver is supplied here - deliberately as a callable, because the bug it fixes was a
        # value captured before `STATE` was populated.
        "class_names_served": lambda: CLASS_NAMES,
        "FALLING": FALLING, "FALLEN": FALLEN,
        "ActivityPipeline": ActivityPipeline, "ActivityConfig": ActivityConfig,
        "observed_hours_from_frames": observed_hours_from_frames,
        "rebuild_observations": rebuild_observations, "Role": Role,
        "logit_adjust": logit_adjust, "class_prior": class_prior,
        "STATE": {"clf": StubClf(), "streams": ["bone"],
                  "reporter": reporter or CaregiverReporter(FaithfulStubLLM())},
    }


def _w6_poses(n=120, people=2, fps=15.0):
    def wire(tid, role, cx, cy):
        # `role_confidence` 0 for an unknown role: PersonObservation refuses a confident
        # UNKNOWN, because open-set rejection reporting confident identity is contradictory.
        return {"track_id": tid, "role": role,
                "role_confidence": 0.0 if role == "unknown" else 0.8,
                "box": [cx - 30, cy - 60, cx + 30, cy + 80],
                "kp": [[cx + j * 1.5, cy + j * 2.0, 0.9] for j in range(17)]}

    def crowd(i):
        # Role `unknown` for everyone, which is what OSNet returns on an uploaded clip: there
        # is no enrolled gallery to match a stranger against.
        out = [wire(0, "unknown", 400 + i, 300)]
        if people > 1:
            out.append(wire(1, "unknown", 800 - i, 320))
        return out

    return {"fps": fps, "width": 1280, "height": 720, "frames_kept": n, "truncated": False,
            "providers": ["CPUExecutionProvider"], "reid": True,
            "frames": [{"i": i, "t": round(i / fps, 3), "people": crowd(i)}
                       for i in range(n)]}


def test_w6_the_video_handler_actually_runs_on_real_schema_objects():
    """EXECUTE `/video`'s staging code. W1-W5 compare keys and cannot see a wrong attribute.

    This is the test that was missing. `/video` 500'd on every upload with
    `'ActivitySegment' object has no attribute 'subject_role'` - the role field on a segment
    is `role`; `subject_role` belongs to BehaviourState. W1-W5 all passed throughout, because
    a dict key spelled correctly says nothing about the expression that fills it, and an AST
    walk cannot know which pydantic model a loop variable holds.

    Four properties, because each of the three bugs found here rendered as four green stages:

      - every attribute resolves and the payload is JSON-encodable
      - all four stages report `done` - most of this runs inside `except Exception` and
        degrades a stage instead of raising, which is right in production and silent in a test
      - Agent 4 emitted claims - zero claims is what the 2026-01-01 baseline collision produced
      - the features are NOT all zero - which is what the role gate produced, while Agent 2 was
        simultaneously reporting real segments
    """
    import time

    t = time.time()
    ns = _w6_namespace()
    poses = _w6_poses()
    events = list(_video_tail_callable(ns)(poses, t - 2, t - 1))

    # Heartbeats may interleave (the stub is fast, so usually none). Order is asserted on the
    # events that carry state: a `result` before a `stage` would leave the cards half-drawn
    # under a finished report.
    spine = [(k, o) for k, o in events if k != "heartbeat"]
    kinds = [k for k, _ in spine]
    assert kinds == ["stage"] * 4 + ["result"], kinds
    agents = [o["agent"] for k, o in spine if k == "stage"]
    assert agents == [1, 2, 3, 4], f"stages streamed out of order: {agents}"
    payload = spine[-1][1]

    # FastAPI has to encode it. A datetime or a numpy scalar left in here is a 500.
    json.dumps(payload)

    stages = {s["agent"]: s for s in payload["stages"]}
    assert sorted(stages) == [1, 2, 3, 4], sorted(stages)
    broken = {n_: (s["status"], (s["payload"] or {}).get("error"))
              for n_, s in stages.items()
              if s["status"] != "done" or "error" in (s["payload"] or {})}
    assert not broken, (
        f"stages did not all complete: {broken}. Most of this handler runs inside "
        "`except Exception` and degrades to a failed stage instead of raising, so a real "
        "AttributeError in Agent 3 or 4 shows up here and nowhere else.")

    # The streamed stage records must be the ones the final payload reports. Two sources for
    # one fact is how a page ends up showing a stage that "passed" live and failed in the
    # result, or vice versa.
    for k, obj in spine[:-1]:
        assert obj == stages[obj["agent"]], (
            f"streamed Agent {obj['agent']} record differs from the one in result.stages")

    # Two tracked people, real segments. Roles stay as the RE-IDENTIFIER reported them, even
    # though one track is asserted as the subject downstream - the cards must not claim OSNet
    # recognised somebody it did not.
    roles = {t_["track_id"]: t_["role"] for t_ in payload["tracks"]}
    assert roles == {0: "unknown", 1: "unknown"}, (
        f"{roles} - a track's displayed role changed. It must be what Agent 1's ReID decided; "
        "the subject assertion travels in `is_subject`/`subject_provenance`, not in `role`.")
    assert all(t_["n_segments"] >= 1 for t_ in payload["tracks"]), payload["tracks"]
    assert sum(1 for t_ in payload["tracks"] if t_.get("is_subject")) == 1, (
        "exactly one track must be marked the subject")

    # Agent 1 counts people TRACKED, which is >= people classified. Reporting Agent 2's count
    # on Agent 1's card hid anyone present but too occluded to label - the state the page
    # draws as a dashed "pose unusable" box.
    assert stages[1]["payload"]["n_people"] == payload["n_people"] == 2, (
        f"{stages[1]['payload']['n_people']} / {payload['n_people']}")

    # THE ALL-ZERO BUG. Every track on an uploaded clip is `unknown`, and
    # aggregate_daily_features attributes personal features only to the subject role, so the
    # subject filter matched nothing and all 27 features read 0.0 - while Agent 2 was reporting
    # real segments in the same response.
    s3 = stages[3]["payload"]
    feats = s3["features_from_video"]
    nonzero = {k: v for k, v in feats.items() if v}
    assert nonzero, (
        f"all {len(feats)} features are zero. Every track is `unknown` on an uploaded clip, so "
        "the subject filter in aggregate_daily_features matched nothing. Agent 4 then writes a "
        "confident report about a person who did nothing.")
    assert s3["subject_provenance"] == "asserted_most_present_track", s3["subject_provenance"]
    assert s3["subject_track"] in (0, 1)
    assert any("ASSERTED" in c for c in s3["caveats"]), (
        "the identity assertion must be declared in the caveats, like the baseline is")

    # Agent 4 produced claims and every one carries all five checks, resolved.
    assert payload["checks"], (
        "Agent 4 reported done but emitted no checks, so the C1-C5 table would render empty "
        "on a page that says every claim is verified.")
    for c in payload["checks"]:
        missing = [k for k in JS_CHECK if k not in c]
        assert not missing, f"{c.get('claim_id')} lacks {missing}"

    assert s3["baseline_provenance"] != "resident_history", (
        "one clip cannot supply this person's 14-day history; the banner depends on this "
        "field declaring the reference")
    # THE DEVIATION TABLE IS WITHHELD, AND THAT IS THE ASSERTION. This clip observes 0.002 h;
    # the reference days observe ~22.9 h. A ten-minute total against a 23-hour median reads as a
    # collapse whatever the person did (measured: walking rate 62.7 s/h vs the baseline's ~67 s/h
    # rendered as robust_z -10.00), so the numbers are withheld from the PAGE exactly as they are
    # withheld from Agent 4's prompt, and `deviations_withheld` is what the page renders instead.
    # The old assertion demanded deviations non-empty so claims would have something to cite -
    # but the verifier's evidence index is built from the state, not from this display list, and
    # the checks below pass regardless.
    assert s3["deviations"] == [], s3["deviations"]
    assert s3["deviations_withheld"] is True, (
        "a partial window must declare the withholding, or the empty table reads as 'no "
        "deviations computed' rather than 'not comparable'")
    assert payload["checks"], "claims must still cite OBSERVED values on a partial window"

    # `is_reliable` must carry the VALUE, not merely the key. `DailyFeatures.is_reliable` is a
    # METHOD, and `bool(getattr(today, "is_reliable", False))` evaluates a bound method object -
    # always truthy - so a 29-second clip was published as a reliable day. `stages.js` keys its
    # "far below the 8 h a day needs to be called reliable" warning off this exact field, which
    # means the bug silently deleted the one sentence telling a caregiver not to act on the
    # numbers. W1 checked the key was present and passed throughout, which is the same
    # key-comparison failure this whole suite was written after.
    assert s3["observed_hours"] < 8.0, s3["observed_hours"]
    assert s3["is_reliable"] is False, (
        f"a {s3['observed_hours']} h clip is reported as a reliable day. `is_reliable` is a "
        "method: it has to be CALLED, or every short clip claims a full day's confidence.")
    print(f"  W6 is_reliable={s3['is_reliable']} for observed_hours={s3['observed_hours']} "
          f"(a bound method would have made this True)")
    print(f"  W6 executed /video's staging: streamed {agents} then result, 4/4 done, "
          f"{len(payload['tracks'])} tracks (roles {sorted(set(roles.values()))}, subject "
          f"{s3['subject_track']} asserted), {len(nonzero)}/{len(feats)} features non-zero "
          f"({', '.join(f'{k}={v}' for k, v in list(nonzero.items())[:3])}), deviations withheld "
          f"({s3['observed_hours']} h vs full-day baselines), "
          f"{len(payload['checks'])} claims x 5 checks")


def test_w6b_a_slow_agent_emits_heartbeats_so_silence_stays_bounded():
    """The reported failure: "No data from the backend for 90 s."

    Streaming one line per AGENT made the longest silence in the stream equal to the slowest
    stage. Agent 3's line went out, then nothing at all while Qwen generated, and the browser's
    silence watchdog fired and reported a wedged GPU on a run that was proceeding normally.

    A generator cannot yield from inside a blocking call, so `_blocking` moves the work to a
    worker thread and beats while it waits. This drives it with a reporter that sleeps longer
    than the beat interval, which is the only way to observe the mechanism at all - with the
    fast stub, zero heartbeats fire and the bug is invisible.
    """
    import time

    from behaviorsense.agents.reasoning.reporter import CaregiverReporter, FaithfulStubLLM

    class SlowReporter:
        # Sleeps past the heartbeat interval, then delegates. Deliberately not a mock of the
        # return value: the payload still has to be real or the assertions below prove nothing.
        def __init__(self):
            self.inner = CaregiverReporter(FaithfulStubLLM())

        def report(self, state):
            time.sleep(0.75)
            return self.inner.report(state)

    ns = _w6_namespace(reporter=SlowReporter())
    # A 0.25 s beat against a 0.75 s call: at least two beats, and the test still runs in under
    # a second. Patched on the extracted function's own default, so the production default
    # (10 s) is what ships.
    real_blocking = None
    t = time.time()
    gen = _video_tail_callable(ns)
    real_blocking = ns["_blocking"]
    ns["_blocking"] = lambda fn, label, every=0.25: real_blocking(fn, label, every=0.25)

    events = list(gen(_w6_poses(), t - 2, t - 1))
    beats = [o for k, o in events if k == "heartbeat"]
    assert beats, (
        "a stage that took 3x the heartbeat interval emitted no heartbeat, so the stream is "
        "silent for the whole of generation and the client cannot distinguish a slow model "
        "from a dead session")
    assert all(b["step"] == "qwen_generating" for b in beats), beats
    assert all(b["elapsed_s"] >= 0 for b in beats), beats

    # The gap between consecutive lines is what the watchdog measures. Every beat must land
    # between Agent 3's stage line and Agent 4's, or it is not covering the silent window.
    kinds = [k for k, _ in events]
    i3, i4 = None, None
    seen = 0
    for i, (k, o) in enumerate(events):
        if k == "stage":
            seen += 1
            if seen == 3:
                i3 = i
            if seen == 4:
                i4 = i
    assert i3 is not None and i4 is not None
    beat_positions = [i for i, (k, _) in enumerate(events) if k == "heartbeat"]
    assert all(i3 < p < i4 for p in beat_positions), (
        f"heartbeats at {beat_positions} but the silent window is between {i3} and {i4}")

    # And the run still completes normally - the thread hand-off must not change the payload.
    payload = events[-1][1]
    assert kinds[-1] == "result" and payload["checks"], "the slow path lost its claims"
    print(f"  W6b {len(beats)} heartbeat(s) during a 0.75 s Agent 4 at a 0.25 s interval, all "
          f"between the Agent 3 and Agent 4 stage lines (positions {beat_positions} in "
          f"{len(events)} events); payload intact with {len(payload['checks'])} claims")


def test_w7_both_backends_speak_the_same_ndjson_envelope():
    """The stream envelope is a contract too, and it is newer than every other one here.

    `/video` and `/demo` stream NDJSON because the Cloudflare quick tunnel drops a request
    whose origin has not answered in about two minutes - measured, /demo was cut at 125.8 s
    with a 524 while the T4 was still generating. Neither endpoint can be a plain JSON reply.

    That makes the event names load-bearing in a way a key name is not: `api.js` dispatches on
    them, and a stub that says `done` where the notebook says `result` would leave the page
    waiting forever on a stream that had already finished. Asserted from both sources rather
    than from a list maintained here.
    """
    nb = _video_cell()
    doc = json.loads(NB.read_text(encoding="utf-8"))
    demo_cell = next(c["source"] for c in doc["cells"] if c["cell_type"] == "code"
                     and '@app.get("/demo")' in c["source"])
    stub = (WEB / "dev_backend.py").read_text(encoding="utf-8")
    js = (WEB / "scripts" / "api.js").read_text(encoding="utf-8")

    def events(text: str) -> set[str]:
        return set(re.findall(r'["\']event["\']\s*:\s*["\'](\w+)["\']', text))

    nb_video, nb_demo = events(nb), events(demo_cell)
    stub_ev = events(stub)
    # `api.js` compares rather than constructs: `ev.event === "result"`.
    js_ev = set(re.findall(r'ev\.event\s*===\s*["\'](\w+)["\']', js))

    # Both notebook endpoints must send the two the client cannot do without.
    for name, got in (("/video", nb_video), ("/demo", nb_demo)):
        assert {"accepted", "result", "error"} <= got, f"{name} emits only {sorted(got)}"

    # Anything the notebook can send, the stub must be able to send, or the page is developed
    # against a narrower stream than the GPU produces.
    missing = (nb_video | nb_demo) - stub_ev - {"error"}   # the stub has no way to fail mid-run
    assert not missing, (
        f"dev_backend.py never emits {sorted(missing)}, which notebook 05 does - the page "
        "would be built against a stream that is missing events")

    # And every event the client branches on must actually be produced by someone.
    unknown = js_ev - (nb_video | nb_demo | stub_ev)
    assert not unknown, f"api.js dispatches on {sorted(unknown)}, which no backend sends"
    assert "result" in js_ev and "error" in js_ev, sorted(js_ev)

    # The stub must really stream: chunked, flushed per line, and HTTP/1.1 or the chunk
    # framing is delivered as body bytes.
    assert "x-ndjson" in stub, "the stub does not declare application/x-ndjson"
    assert 'protocol_version = "HTTP/1.1"' in stub, (
        "chunked transfer needs HTTP/1.1; on HTTP/1.0 the chunk headers arrive as content")
    assert "ThreadingTCPServer" in stub, (
        "a single-threaded stub blocks /health polling behind an open stream, so the page "
        "reports a dead backend while it is streaming from one")
    print(f"  W7 /video emits {sorted(nb_video)}, /demo {sorted(nb_demo)}, stub "
          f"{sorted(stub_ev)}, api.js dispatches {sorted(js_ev)}; stub streams chunked "
          "NDJSON over HTTP/1.1 on a threaded server")


def test_w8_one_person_split_into_three_tracks_keeps_all_of_their_activity():
    """Fragmented track ids must not silently delete the same person's activity.

    MEASURED, on a 45 s upload of one woman in a kitchen: RTMO detected her in 69% of frames, and
    every gap longer than `max_age_frames` (30 frames = 1.5 s at 20 Hz) killed the track, so she
    arrived at Agent 2 as THREE track ids. `assert_subject` then kept the most-present one and
    `aggregate_daily_features` attributes personal features to the subject only - so the page
    showed `cooking_food_prep 8.6s` on Agent 2's card and `cooking_duration_s 0.00` on Agent 3's,
    with half the housework gone too.

    The rule that fixes it is not a heuristic. Two tracks seen in the SAME frame are two people,
    so an overlap is decisive against merging; tracks whose frame spans never overlap cannot be
    separated without a gallery, and with re-identification off there is none. This test pins both
    directions, because a merge that fired on genuinely co-present people would corrupt every
    downstream feature in the opposite direction - and that is the worse failure of the two.
    """
    import datetime as dt

    from behaviorsense.agents.activity import EXTENDED_CLASS_NAMES
    from behaviorsense.schemas import ActivitySegment, Role
    from behaviorsense.service.staging import (assert_subject, behaviour_stage, disjoint_chain,
                                               reachable)

    ids = {n: i for i, n in enumerate(EXTENDED_CLASS_NAMES)}
    base = dt.datetime(2026, 1, 1, 9, 0, 0)

    def seg(tid, name, t0, t1):
        return ActivitySegment(
            segment_id=f"t{tid}-{name}-{t0}", track_id=tid, role=Role.UNKNOWN,
            activity_id=ids[name], activity_name=name,
            start_time=base + dt.timedelta(seconds=t0),
            end_time=base + dt.timedelta(seconds=t1), confidence=0.8)

    # The clip's actual segments, to the tenth of a second.
    segs = [seg(0, "cleaning_housework", 0, 3.2), seg(0, "cleaning_housework", 4, 11.3),
            seg(1, "cooking_food_prep", 13, 21.6),
            seg(2, "cleaning_housework", 24, 28.6), seg(2, "cleaning_housework", 30, 35.9)]
    fpt = {0: 240, 1: 180, 2: 300}
    spans = {0: (0, 230), 1: (260, 440), 2: (470, 899)}          # sequential: one person

    def features(sp):
        kept, primary, prov, tracks = assert_subject(segs, fpt, spans=sp)
        payload, _ = behaviour_stage(kept, day=dt.date(2026, 1, 22), observed_hours=0.0125,
                                     subject_track=primary, subject_provenance=prov,
                                     subject_tracks=tracks)
        return payload, prov, tracks

    # WITHOUT spans: the old behaviour, kept as the fallback and pinned so the regression is
    # visible rather than remembered.
    old, prov_old, tracks_old = features(None)
    assert prov_old == "asserted_most_present_track" and tracks_old == [2], tracks_old
    assert old["features_from_video"]["cooking_duration_s"] == 0.0, (
        "the old path is supposed to lose the cooking segment; if it does not, this test is "
        "measuring the wrong thing")

    new, prov_new, tracks_new = features(spans)
    assert prov_new == "asserted_disjoint_track_chain", prov_new
    assert tracks_new == [0, 1, 2], tracks_new
    f = new["features_from_video"]
    assert f["cooking_duration_s"] == 8.6, (
        f"cooking is {f['cooking_duration_s']} - Agent 2 emitted an 8.6 s cooking segment on "
        "track 1 and Agent 3 must not report zero for it")
    assert f["housework_duration_s"] == 21.0, f["housework_duration_s"]
    assert f["housework_duration_s"] > old["features_from_video"]["housework_duration_s"] * 1.9, (
        "the merge recovered less than the discarded half")
    assert any("MERGED" in c for c in new["caveats"]), (
        "a merge changes who the numbers are about, so it must be declared exactly as the "
        "asserted subject and the simulated baseline already are")
    assert new["subject_tracks"] == [0, 1, 2], new["subject_tracks"]

    # THE OTHER DIRECTION, and the one that matters more. Three people in the room at once share
    # frames, so nothing may merge and the most-present track wins as before.
    _, prov_co, tracks_co = features({0: (0, 500), 1: (100, 600), 2: (470, 899)})
    assert prov_co == "asserted_most_present_track" and tracks_co == [2], (prov_co, tracks_co)

    # And partially: track 1 overlaps 0, so 1 is a second person and only 0 and 2 are the subject.
    _, prov_part, tracks_part = features({0: (0, 230), 1: (200, 440), 2: (470, 899)})
    assert prov_part == "asserted_disjoint_track_chain" and tracks_part == [0, 2], tracks_part

    # The chain maximises FRAMES OBSERVED, not fragment count: given the choice between one long
    # track and two short ones that together see less, it keeps the long one.
    assert disjoint_chain({0: (0, 100), 1: (0, 40), 2: (60, 100)},
                          {0: 90, 1: 20, 2: 20}) == [0]
    assert disjoint_chain({0: (0, 100), 1: (0, 40), 2: (60, 100)},
                          {0: 30, 1: 40, 2: 40}) == [1, 2]

    # THE TELEVISION. Both of the screenshots this merge was built from contain a television
    # showing people, and RTMO detects them - so a screen in the corner is a track that never
    # shares a frame with the room and looks EXACTLY like a person who stepped out of detection.
    # Disjoint spans alone fold it into the resident and attribute a broadcast's motion to her.
    #
    # The guard is arithmetic rather than recognition: a person walks about 1.2 body-heights per
    # second, so a fragment is only reachable if the walk from the previous fragment's last box to
    # its first box is a walk someone could have made. Here the resident is at the counter and the
    # screen's person is in the far corner, 14 body-heights away with a 4 s gap.
    resident = (0, 200, 60, 320)          # by the counter
    tv_person = (900, 300, 940, 360)      # tiny, across the room, fixed
    assert not reachable(resident, tv_person, 4.0), "a TV person must be out of walking range"
    # ...while the same person re-detected near where they were IS reachable.
    assert reachable(resident, (10, 210, 70, 330), 4.0), "a nearby re-detection must merge"
    # A slow walk across a kitchen over a long gap stays reachable: the constraint must not be so
    # tight that it re-creates the discarded activity the merge was built to stop.
    assert reachable(resident, (150, 220, 210, 340), 6.0), "a walk across the room is reachable"

    segs_tv = [seg(0, "cooking_food_prep", 5, 10), seg(7, "cleaning_housework", 20, 25)]
    fpt_tv = {0: 300, 7: 100}
    spans_tv = {0: (0, 200), 7: (280, 380)}
    boxes_tv = {0: (list(resident), list(resident)), 7: (list(tv_person), list(tv_person))}
    _, _, prov_tv, tracks_tv = assert_subject(segs_tv, fpt_tv, spans=spans_tv,
                                              boxes=boxes_tv, fps=20.0)
    assert prov_tv == "asserted_most_present_track" and tracks_tv == [0], (
        f"the television was merged into the resident: {prov_tv} {tracks_tv}")
    # Without boxes the old merge still fires - the guard is what makes the difference, not luck.
    assert disjoint_chain(spans_tv, fpt_tv) == [0, 7], "the unguarded chain should merge these"

    print(f"  W8 one person as 3 fragmented ids: most-present kept "
          f"housework={old['features_from_video']['housework_duration_s']:.2f} "
          f"cooking={old['features_from_video']['cooking_duration_s']:.2f}; the disjoint chain "
          f"{tracks_new} keeps housework={f['housework_duration_s']:.2f} "
          f"cooking={f['cooking_duration_s']:.2f} and declares the merge; co-present tracks "
          f"({tracks_co}) and a partial overlap ({tracks_part}) are never merged; a person on a "
          f"television {tracks_tv} is rejected as out of walking range")


def test_w9_upstream_lines_are_forwarded_as_they_arrive_not_batched():
    """The local half must not swallow the GPU half's heartbeats.

    Reported failure, with a ten-minute upload: "No data from the backend for 90 s. The session may
    have ended, or the GPU is wedged." The session was fine. Notebook 05 WAS beating, and the fix
    for exactly this symptom had already been written and tested there (W6b) - the bytes were being
    hoarded one process further down, in `web/local_backend.py`:

        for chunk in iter(lambda: r.read(65536), b""):   # blocks until 64 KB or EOF

    `read(n)` does not return what is available; it waits for n bytes or the end of the stream. A
    heartbeat line is ~80 bytes, so at one per 10 s that loop would sit silent for over two hours
    before forwarding anything. Measured against a stub emitting four 1 s-spaced heartbeats: all
    four arrived together at t+4.0 s under `read(65536)`, and at t+0/1/2/3 s under `readline()`.

    It only became visible when pose grew past the watchdog. At 35 s the whole response landed
    before anything timed out, so this ran green through 207 tests while being wrong the whole
    time - the reason this test drives a slow upstream rather than a fast one.
    """
    import http.server
    import importlib.util
    import json
    import socketserver
    import threading
    import time

    # Loaded BY PATH, not by `import web.local_backend`. `web/` is a directory of scripts with no
    # `__init__.py` and is deliberately not importable as a package - every other test in this file
    # reads it as text. Importing it by name only works when the repo root happens to be on
    # `sys.path`, which is true running this file directly and false under `run_tests.py`: W9
    # passed standalone and died with ModuleNotFoundError in the suite.
    spec = importlib.util.spec_from_file_location("_bs_local_backend", WEB / "local_backend.py")
    lb = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = lb                 # dataclasses/typing resolve against sys.modules
    spec.loader.exec_module(lb)
    Upstream = lb.Upstream

    gaps: list[float] = []

    class Slow(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):                                          # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            # Four small lines, spaced like real heartbeats and far below any read buffer.
            for i in range(4):
                line = (json.dumps({"event": "heartbeat", "step": "rtmo_decoding",
                                    "elapsed_s": i}) + "\n").encode()
                self.wfile.write(f"{len(line):X}\r\n".encode() + line + b"\r\n")
                self.wfile.flush()
                if i < 3:
                    time.sleep(0.4)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def log_message(self, *a):                                  # noqa: D102
            pass

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True
        allow_reuse_address = True

    srv = Server(("127.0.0.1", 0), Slow)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        up = Upstream(f"http://127.0.0.1:{port}")
        t0 = time.time()
        for ev in up.video_stages_12(b"x", "multipart/form-data"):
            gaps.append(time.time() - t0)
            assert ev["event"] == "heartbeat"
    finally:
        srv.shutdown()
        srv.server_close()

    assert len(gaps) == 4, f"expected 4 heartbeats, got {len(gaps)}"
    # THE ASSERTION THAT MATTERS: the first line must arrive almost immediately, not after all
    # four have been written. A buffering read passes every other check here.
    assert gaps[0] < 0.35, (
        f"the first heartbeat arrived at t+{gaps[0]:.2f}s, so the reader is waiting on a byte "
        "count rather than forwarding what has arrived - the page will see silence for the whole "
        "stage and report a wedged GPU")
    assert gaps[-1] >= 1.0, gaps
    spread = gaps[-1] - gaps[0]
    assert spread >= 1.0, (
        f"all lines arrived within {spread:.2f}s of each other, which is the batched signature of "
        "read(65536) rather than one line at a time")

    print(f"  W9 four heartbeats 0.4 s apart reached the local half at "
          f"{[round(g, 2) for g in gaps]}s - streamed, not batched at EOF (read(65536) delivered "
          "all four together at t+1.6s)")


def test_w10_the_gallery_round_trips_and_travels_by_boundary_surgery():
    """The enrolled-resident gallery: local file in, appended multipart part out, enrolment back.

    The design this pins (agreed before it was built): biometric templates live in ONE plain
    file on the operator's machine - never a Kaggle dataset, never under web/ where the static
    server would serve it to anyone with the page URL - and travel to the GPU half as an extra
    multipart part appended by BOUNDARY SURGERY, so the video bytes are never re-parsed or
    re-encoded by the proxy that claims to pass them through byte-for-byte.

    Round-trip exactness matters more than anything else here: the record IS the centroid the
    matcher uses, and enrol() normalises then averages, so one stored normalised vector must
    rebuild to itself. If it did not, every re-upload would drift the resident's identity.
    """
    import http.server
    import importlib.util
    import json as _json
    import socketserver
    import tempfile
    import threading

    import numpy as np

    spec = importlib.util.spec_from_file_location("_bs_local_backend_w10",
                                                  WEB / "local_backend.py")
    lb = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = lb
    spec.loader.exec_module(lb)

    with tempfile.TemporaryDirectory() as td:
        gpath = Path(td) / "gallery.json"

        # 1. Enrol from a clip's crops: normalise-before-average, n_updates accumulates,
        #    the file is written, and a reload sees exactly what was written.
        store = lb.GalleryStore(gpath)
        # 8-dim, not 4: GalleryStore refuses embeddings below 8 dimensions as garbage
        # (a real OSNet vector is 512), and the test must play by the same rule.
        emb = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        used = store.enrol("Mary", emb)
        assert used == 3, used
        assert gpath.is_file()
        reloaded = lb.GalleryStore(gpath)
        assert "Mary" in reloaded.residents
        c1 = np.asarray(store.residents["Mary"]["centroid"])
        c2 = np.asarray(reloaded.residents["Mary"]["centroid"])
        assert np.allclose(c1, c2, atol=1e-6), "the file did not round-trip"
        assert reloaded.residents["Mary"]["n_updates"] == 3

        # 2. Re-enrolment ACCUMULATES rather than replacing the count.
        store2 = lb.GalleryStore(gpath)
        store2.enrol("Mary", [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        assert store2.residents["Mary"]["n_updates"] == 4

        # 3. Garbage is refused, not enrolled: a name with no usable embeddings must not
        #    create a resident who can never be matched.
        assert store2.enrol("", emb) == 0
        assert store2.enrol("X", []) == 0
        assert store2.enrol("Y", [[1.0, 2.0]]) == 0          # inconsistent dimensions
        assert "X" not in store2.residents and "Y" not in store2.residents

        # 4. BOUNDARY SURGERY: a well-formed multipart with one part gains a second gallery
        #    part, the video bytes are untouched, and the result still parses as multipart.
        CRLF = chr(13) + chr(10)
        boundary = "----WebKitFormBoundaryTest123"
        video_bytes = bytes(range(256)) * 4          # binary payload incl. CR/LF/0x2d bytes
        original = (
            "--" + boundary + CRLF
            + 'Content-Disposition: form-data; name="file"; filename="clip.mp4"' + CRLF
            + "Content-Type: video/mp4" + CRLF + CRLF
        ).encode() + video_bytes + (CRLF + "--" + boundary + "--" + CRLF).encode()
        ctype = f"multipart/form-data; boundary={boundary}"
        gallery_json = store.as_request_json()

        received: dict = {}

        class Capture(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):                                        # noqa: N802, D102
                n = int(self.headers.get("Content-Length") or 0)
                received["body"] = self.rfile.read(n)
                received["type"] = self.headers.get("Content-Type")
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.write(b"0" + CRLF.encode() + CRLF.encode())

            def log_message(self, *a):                                # noqa: D102
                pass

        class S(socketserver.ThreadingTCPServer):
            daemon_threads = True
            allow_reuse_address = True

        srv = S(("127.0.0.1", 0), Capture)
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            up2 = lb.Upstream(f"http://127.0.0.1:{port}")
            list(up2.video_stages_12(original, ctype, gallery_json=gallery_json))
        finally:
            srv.shutdown()

        body = received["body"]
        assert received["type"] == ctype, "the content type must pass through unchanged"

        # PARSE IT, do not merely look for the bytes. The first version of this test asserted
        # the gallery part was PRESENT and the body CLOSED - both true of a body whose gallery
        # part sat after the closing `--boundary--` delimiter, i.e. in the epilogue that every
        # multipart parser is required to ignore. So `gallery` never reached FastAPI, OSNet
        # never matched, and re-identification silently stayed off through four live runs
        # while this test passed. A real parser is the only assertion that means anything.
        from email.parser import BytesParser
        from email.policy import default as _policy

        msg = BytesParser(policy=_policy).parsebytes(
            ("Content-Type: " + ctype + CRLF + "MIME-Version: 1.0" + CRLF + CRLF).encode()
            + body)
        parts = {p.get_param("name", header="content-disposition"): p.get_payload(decode=True)
                 for p in msg.iter_parts()}
        assert "gallery" in parts, (
            f"a multipart parser sees only {sorted(parts)} - the gallery part is present in "
            "the bytes but not in a position the parser reads (epilogue after --boundary--)")
        assert parts["file"] == video_bytes, "the video payload was altered"
        assert _json.loads(parts["gallery"])["residents"]["Mary"]["n_updates"] == 3

        # And the ordinary browser shape - a text field already beside the file - survives
        # too, because that is what a real enrolling upload looks like.
        with_enrol = (
            ("--" + boundary + CRLF
             + 'Content-Disposition: form-data; name="file"; filename="clip.mp4"' + CRLF
             + "Content-Type: video/mp4" + CRLF + CRLF).encode()
            + video_bytes
            + (CRLF + "--" + boundary + CRLF
               + 'Content-Disposition: form-data; name="enrol"' + CRLF + CRLF + "Mary"
               + CRLF + "--" + boundary + "--" + CRLF).encode())
        received.clear()
        srv3 = S(("127.0.0.1", 0), Capture)
        port3 = srv3.server_address[1]
        threading.Thread(target=srv3.serve_forever, daemon=True).start()
        try:
            list(lb.Upstream(f"http://127.0.0.1:{port3}").video_stages_12(
                with_enrol, ctype, gallery_json=gallery_json))
        finally:
            srv3.shutdown()
        msg2 = BytesParser(policy=_policy).parsebytes(
            ("Content-Type: " + ctype + CRLF + "MIME-Version: 1.0" + CRLF + CRLF).encode()
            + received["body"])
        got = {p.get_param("name", header="content-disposition"): p.get_payload(decode=True)
               for p in msg2.iter_parts()}
        assert sorted(got) == ["enrol", "file", "gallery"], sorted(got)
        assert got["enrol"] == b"Mary" and got["file"] == video_bytes
        # The body still closes properly.
        assert body.rstrip(CRLF.encode()).endswith(
            ("--" + boundary + "--").encode())
        # And without a gallery, the body passes through completely untouched.
        received.clear()
        srv2 = S(("127.0.0.1", 0), Capture)
        port2 = srv2.server_address[1]
        threading.Thread(target=srv2.serve_forever, daemon=True).start()
        try:
            up3 = lb.Upstream(f"http://127.0.0.1:{port2}")
            list(up3.video_stages_12(original, ctype, gallery_json=""))
        finally:
            srv2.shutdown()
        assert received["body"] == original, (
            "an empty gallery must not touch the body - the no-enrolment path is the "
            "common one and must remain byte-for-byte")

    print(f"  W10 gallery round-trips exactly (centroid preserved to 1e-6, n_updates "
          f"accumulates 3->4); garbage enrolments refused; after boundary surgery a REAL "
          f"multipart parser reads ['enrol','file','gallery'] with {len(video_bytes)} video "
          f"bytes intact; an empty gallery leaves the body byte-for-byte identical")


def test_w11_every_video_function_binds_the_names_it_reads():
    """No function in the /video cell may read a name nothing binds. Static, whole-cell.

    MEASURED, twice in one session: `use_osnet` was computed in `video()` (where an HTTP
    status is still available to reject a bad request) and read inside `_video_stream()`, a
    separate generator - so every upload died with `NameError: name 'use_osnet' is not
    defined`. The whole 215-test suite passed while that was live, because W6/W6b execute
    `_video_stages` and nothing executes `_video_stream`; its body only runs behind a real
    GPU, an ONNX session and a real upload.

    Executing it here is not the answer - that would need the GPU. But the defect class is
    static and cheap to catch: a name read in a function's body must be bound as a parameter,
    assigned locally, imported, defined in the cell, or be a builtin. This walks every
    function in the cell and checks exactly that, so a parameter that stops being threaded
    through fails here in 20 ms instead of on the next upload.
    """
    import ast
    import builtins

    src = _video_cell()
    tree = ast.parse(src)

    # Module-level names the cell itself binds, plus the notebook globals every cell sees.
    # The latter are listed rather than discovered: this test's job is to catch a name that
    # NOTHING binds, and an over-broad allowlist would defeat it.
    cell_level = set(dir(builtins))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            cell_level.add(node.name)
        elif isinstance(node, ast.Assign):
            cell_level |= {t.id for t in ast.walk(node) if isinstance(t, ast.Name)
                           and isinstance(t.ctx, ast.Store)}
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            cell_level |= {(a.asname or a.name).split(".")[0] for a in node.names}
    # Globals defined in EARLIER cells of the same notebook (STATE, RTMO, OSNET, MAX_FRAMES...).
    doc = json.loads(NB.read_text(encoding="utf-8"))
    for c in doc["cells"]:
        if c["cell_type"] != "code" or '@app.post("/video")' in c["source"]:
            continue
        try:
            other = ast.parse(c["source"])
        except SyntaxError:
            continue                     # a cell with notebook magics; not our concern
        for node in ast.walk(other):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                cell_level.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                cell_level.add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                cell_level |= {(a.asname or a.name).split(".")[0] for a in node.names}

    def _params(args) -> set:
        out = {a.arg for a in args.args} | {a.arg for a in args.kwonlyargs}
        out |= {a.arg for a in args.posonlyargs}
        if args.vararg:
            out.add(args.vararg.arg)
        if args.kwarg:
            out.add(args.kwarg.arg)
        return out

    def bound_in(fn) -> set:
        """Names this function binds, plus every nested binder's own names.

        The nested part matters and its absence was this test's own first bug: LAMBDA
        parameters, comprehension targets, and the `self` of a class defined inside the
        function are all bound where they are read, but a walk that only collects
        `FunctionDef` params reports them as unbound. Over-reporting would make this test
        noise a reader learns to ignore, which is worse than not having it - so the
        collector is deliberately generous while the READ side stays exact.
        """
        out = _params(fn.args)
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                out.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.add(node.name)
                out |= _params(node.args)          # nested defs' own params
            elif isinstance(node, ast.Lambda):
                out |= _params(node.args)          # lambda params, e.g. key=lambda kv: ...
            elif isinstance(node, ast.ClassDef):
                out.add(node.name)
                # A method's `self`/`cls` and its parameters are bound at call time.
                for sub in ast.walk(node):
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        out |= _params(sub.args)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                out |= {(a.asname or a.name).split(".")[0] for a in node.names}
            elif isinstance(node, ast.ExceptHandler) and node.name:
                out.add(node.name)
        return out

    functions = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert functions, "no functions found in the /video cell"
    problems = []
    for fn in functions:
        # A nested function may legitimately close over its parent's locals, so a function's
        # own scope is itself plus every enclosing function in this cell.
        enclosing = set()
        for other in functions:
            if other is not fn and any(n is fn for n in ast.walk(other)):
                enclosing |= bound_in(other)
        scope = bound_in(fn) | enclosing | cell_level
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in scope:
                    problems.append(f"{fn.name}() reads {node.id!r} (line {node.lineno})")
    joined = "; ".join(sorted(set(problems)))
    assert not problems, (
        "the /video cell reads names nothing binds - every upload would die with a "
        "NameError: " + joined)

    print(f"  W11 {len(functions)} functions in the /video cell; every name read is bound "
          f"as a parameter, assignment, import or cell-level global (the `use_osnet` "
          f"NameError class, which W6 cannot reach because `_video_stream` needs a GPU)")


def test_w12_the_local_backends_demo_endpoint_speaks_the_ledgers_language():
    """`Generate live` must work against the local backend, not just the notebook.

    MEASURED: the chip is shown whenever the page is live, and the page is live against the
    LOCAL backend in the split configuration it is actually used in - but the local backend
    had no /demo route, so the chip failed with "no route /demo". A broken advertised feature
    that only fires in the deployed configuration is the exact shape of bug this suite exists
    for.

    Two layers: the payload must carry every field `renderLive` dereferences (claims,
    verifications, evidence, model, emitted, schema_rejected, day...), and the HTTP endpoint
    must speak the same NDJSON envelope as the notebook's /demo, because `askStream` keys on
    the `accepted`/`result`/`error` events.
    """
    import importlib.util
    import json as _json
    import threading
    import urllib.request

    from behaviorsense.agents.reasoning.reporter import CaregiverReporter, FaithfulStubLLM

    spec = importlib.util.spec_from_file_location("_bs_local_backend_w12",
                                                  WEB / "local_backend.py")
    lb = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = lb
    spec.loader.exec_module(lb)

    # 1. THE PAYLOAD CONTRACT, against the ledger's own field list.
    payload = lb.demo_payload(CaregiverReporter(FaithfulStubLLM()), 0)
    need = ("day", "subject", "summary", "recommendation", "escalate", "claims",
            "verifications", "evidence", "alerts", "hallucination_rate", "emitted",
            "schema_rejected", "model")
    missing = [k for k in need if k not in payload]
    assert not missing, missing
    for c in payload["claims"]:
        assert c["evidence_ref"] in payload["evidence"], (
            f"{c['evidence_ref']} does not resolve - the browser verifier would score it "
            "as a fabrication")
        ev = payload["evidence"][c["evidence_ref"]]
        for f in ("observed_value", "baseline_median", "delta", "pct_change", "robust_z"):
            assert f in ev, (c["evidence_ref"], f)
    for v in payload["verifications"]:
        for f in ("claim_id", "ref_exists", "value_matches", "pct_matches",
                  "direction_consistent", "prose_quoted_value", "notes"):
            assert f in v, f

    # 2. THE HTTP ENDPOINT, over the wire, in the envelope askStream parses.
    import http.server
    import socketserver

    # Drive the REAL local backend handler: mount it on a socket with the stub reporter.
    Handler = lb.Handler
    Handler.reporter = CaregiverReporter(FaithfulStubLLM())

    class S(socketserver.ThreadingTCPServer):
        daemon_threads = True
        allow_reuse_address = True

    srv = S(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/demo?scenario=0",
                                    timeout=30) as r:
            lines = [ln for ln in r.read().decode("utf-8").splitlines() if ln.strip()]
        events = [_json.loads(ln) for ln in lines]
        kinds = [e.get("event") for e in events]
        assert kinds[0] == "accepted" and "result" in kinds, kinds
        result = next(e for e in events if e.get("event") == "result")
        assert result["payload"]["claims"], "the /demo result carried no claims"
        # And a route that does not exist still 404s, so the router is not a catch-all.
        # INSIDE the try: the finally shuts the server down, and a request after shutdown
        # hangs on the OS backlog until it times out - which is how this test first failed,
        # against a server that was working perfectly.
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/nope", timeout=10)
            raise AssertionError("/nope should have 404'd")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        srv.shutdown()

    print(f"  W12 demo_payload carries all {len(need)} fields the ledger dereferences with "
          f"every ref resolving; GET /demo streams {kinds} with "
          f"{len(result['payload']['claims'])} claims over the wire; unknown routes still 404")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            print(f"{fn.__name__}:")
            fn()
            print("  PASS")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
