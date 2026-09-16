"""Partial-label training across corpora that label different halves of the taxonomy.

M1 is the one to read. Every other test here supports it.

The failure being prevented: concatenating Toyota with Charades under ordinary cross-entropy
makes every one of 365,492 Toyota windows a statement that the resident is *not* sitting,
because Toyota annotates `Sit_down` as a transition and never labels the sustained posture.
A third of those windows are the unannotated gaps where sitting is most likely. That is not a
small mislabelling - it trains the model to suppress exactly the classes Charades is there to
teach.

Run: python tests/test_multicorpus.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behaviorsense.data.multicorpus import (  # noqa: E402
    CHARADES_TRUSTED,
    CORPORA,
    FALL_TRUSTED,
    N_COARSE,
    collapse_for,
    fine_spaces,
    masked_cross_entropy,
    pooled_coarse_logits,
    spec_for,
    supervision_mask,
)
from behaviorsense.data.toyota import (  # noqa: E402
    COARSE_V11,
    TSM_CLASSES,
    TSM_TO_COARSE,
    TSU_CLASSES,
    TSU_TO_COARSE,
)

SIT = COARSE_V11.index("sitting")
FALL = COARSE_V11.index("falling")
READ = COARSE_V11.index("reading")


def test_m1_an_unsupervised_class_receives_exactly_zero_gradient():
    """THE test. Masking must remove the class from the softmax, not just from the loss.

    Zeroing a masked class's loss afterwards is not equivalent and is the tempting wrong
    implementation: the softmax would still normalise over that class, so predicting it would
    still be penalised, and the model would still learn "Toyota windows are not sitting".
    Masking to -inf before the softmax removes it from the normaliser, so the gradient is
    identically zero - which is what abstention means.
    """
    tags = ["toyota_trimmed", "charades"]
    mask = torch.as_tensor(supervision_mask(tags))
    assert not bool(mask[0, SIT]), "Toyota must not be allowed to supervise `sitting`"
    assert bool(mask[1, SIT]), "Charades is in the mix precisely to supervise `sitting`"

    logits = torch.zeros(2, N_COARSE, requires_grad=True)
    targets = torch.tensor([READ, SIT])
    masked_cross_entropy(logits, targets, mask).backward()
    grad = logits.grad

    assert grad[0, SIT].item() == 0.0, (
        f"a Toyota window put {grad[0, SIT].item():+.6f} of gradient on `sitting`. Any "
        "non-zero value here is the model being taught that sitting did not happen, over "
        "365,492 windows that cannot say either way")
    assert grad[1, SIT].item() != 0.0, (
        "the Charades window put no gradient on `sitting`, so the mask is over-broad and the "
        "posture classes have no supervision at all")
    # And the same for every class the corpus cannot label, not just the one we picked.
    for cid in range(N_COARSE):
        if not mask[0, cid]:
            assert grad[0, cid].item() == 0.0, f"gradient leaked onto {COARSE_V11[cid]}"
    print(f"  M1 unsupervised gradient is exactly 0.0 on all "
          f"{int((~mask[0]).sum())} classes Toyota cannot label; Charades still gets "
          f"{grad[1, SIT].item():+.3f} on `sitting`")


def test_m2_an_unsatisfiable_target_raises_now_not_nan_later():
    """A masked target cannot be satisfied by any prediction: the loss is +inf.

    Left alone it produces NaN weights several minutes into a run with nothing in the log
    pointing at the label that caused it, so it is refused at the call.
    """
    mask = torch.as_tensor(supervision_mask(["toyota_trimmed"]))
    logits = torch.zeros(1, N_COARSE)
    try:
        masked_cross_entropy(logits, torch.tensor([SIT]), mask)
    except ValueError as exc:
        assert "outside their own corpus" in str(exc), str(exc)
    else:
        raise AssertionError(
            "a target masked out of its own softmax was accepted; the loss is +inf and the "
            "weights go NaN later")
    # The satisfiable case must still work.
    loss = masked_cross_entropy(logits, torch.tensor([READ]), mask)
    assert torch.isfinite(loss), loss
    print(f"  M2 masked target refused at the call; a supervised target gives a finite "
          f"loss ({loss.item():.4f})")


def test_m3_every_shard_tag_declares_what_it_can_supervise():
    """An unknown corpus must raise rather than default to supervising everything."""
    for tag in ("toyota_trimmed", "toyota_untrimmed", "charades",
                "gmdcsa", "urfd", "caucafall", "le2i"):
        spec = spec_for(tag)
        assert spec.supervised, f"{tag} declares no supervised classes"
        assert max(spec.supervised) < N_COARSE, f"{tag} names a class outside the taxonomy"
    try:
        spec_for("some_new_corpus")
    except KeyError as exc:
        assert "Add a CorpusSpec" in str(exc), str(exc)
    else:
        raise AssertionError(
            "an unknown tag resolved. Defaulting to 'supervise everything' reintroduces the "
            "all-classes-are-negative bug for whichever corpus was added last")
    print(f"  M3 {len(CORPORA)} corpora each declare a supervised set; an unknown tag raises")


def test_m4_the_union_covers_the_taxonomy_and_falls_are_not_orphaned():
    """Between them the corpora must reach every class, or one is unlearnable."""
    covered: set[int] = set()
    for spec in CORPORA.values():
        covered.update(spec.supervised)
    missing = [COARSE_V11[i] for i in range(N_COARSE) if i not in covered]
    assert not missing, f"no corpus can supervise {missing} - those classes cannot be learned"

    # falling/fallen_on_ground must come from the fall corpora and ONLY from them among the
    # Toyota entries, since Toyota is an ADL corpus with no falls in it at all.
    assert FALL in FALL_TRUSTED
    for tag in ("toyota_trimmed", "toyota_untrimmed"):
        assert FALL not in spec_for(tag).supervised, (
            f"{tag} claims it can supervise `falling`, but Toyota Smarthome contains no "
            "falls; that would teach the model falls never happen across 365,492 windows")
    print(f"  M4 the union covers all {N_COARSE} coarse classes; `falling` comes only from "
          "the fall corpora, never from Toyota")


def test_m5_charades_is_trusted_only_where_it_measured_well():
    """The trusted set is narrower than what the shards nominally contain, on purpose.

    `bending_reaching` measured 26 val windows at F1 0.000 and `standing` 123 at 0.005,
    because the Charades keyword map has no verb for either. They stay in the trusted set
    because no other corpus covers them at all - but the reason is recorded here so that a
    future reader sees a decision rather than an oversight.
    """
    trusted = {COARSE_V11[i] for i in CHARADES_TRUSTED}
    # The postures are the reason Charades is still in the mix.
    for name in ("sitting", "lying_down", "standing", "bending_reaching"):
        assert name in trusted, f"{name} has no other source and must stay trusted"
    # And it must NOT be trusted on falls, which it does not contain.
    assert "falling" not in trusted and "fallen_on_ground" not in trusted, (
        "Charades has no fall label; trusting it there would contradict the fall corpora")
    # Toyota-only classes must not be claimed by Charades, whose taxonomy predates them.
    for name in ("using_device", "object_interaction"):
        assert name not in trusted, (
            f"{name} is a v1.1 class appended after the Charades shards were extracted, so "
            "those shards cannot carry a label for it")
    print(f"  M5 Charades trusted on {len(trusted)} classes: postures kept (sole source), "
          "falls excluded, v1.1 appendages excluded")


def test_m6_fine_spaces_stay_separate():
    """31 trimmed and 51 untrimmed classes are DIFFERENT id spaces, not one with gaps."""
    spaces = fine_spaces()
    assert spaces == {"tsm31": 31, "tsu51": 51}, spaces
    assert spec_for("toyota_trimmed").fine_space != spec_for("toyota_untrimmed").fine_space, (
        "the two halves must not share an auxiliary head: `Readbook` and `Read` are the same "
        "activity under two spellings, but `Cook.Cleanup` has no exact untrimmed counterpart, "
        "and inventing one is another hand-written label map of the kind that produced "
        "macro-F1 0.128")
    assert spec_for("charades").fine_space == "", "Charades shards carry no fine_labels"
    # Same activity, different id in each space - the reason they cannot be merged blindly.
    assert TSM_CLASSES.index("Readbook") != TSU_CLASSES.index("Read")
    print(f"  M6 two separate fine heads {spaces}; `Readbook`="
          f"{TSM_CLASSES.index('Readbook')} vs `Read`={TSU_CLASSES.index('Read')}")


def test_m7_pooling_reinforces_a_split_class_instead_of_diluting_it():
    """log-sum-exp, not mean: a coarse class with 11 members must not be penalised for it."""
    for tag, mapping, names in (("toyota_trimmed", TSM_TO_COARSE, TSM_CLASSES),
                                ("toyota_untrimmed", TSU_TO_COARSE, TSU_CLASSES)):
        C = collapse_for(tag)
        assert C is not None and C.shape == (len(names), N_COARSE)
        assert np.all(C.sum(axis=1) == 1), "every fine class maps to exactly one coarse class"

        cook = COARSE_V11.index("cooking_food_prep")
        n = int(C[:, cook].sum())
        assert n > 1, "cooking_food_prep should have several fine members"
        pooled = pooled_coarse_logits(torch.zeros(1, len(names)), C)
        assert abs(pooled[0, cook].item() - float(np.log(n))) < 1e-5, (
            f"{n} equal members pooled to {pooled[0, cook].item():.4f}, expected log({n}) = "
            f"{np.log(n):.4f}. A mean would give 0.0 and systematically under-predict every "
            "class that happens to be split into many members")

        # A class with no members must stay at -inf rather than becoming 0.
        absent = [i for i in range(N_COARSE) if C[:, i].sum() == 0]
        assert absent, "some coarse class should be unrepresented in this fine space"
        assert pooled[0, absent[0]].item() == float("-inf")
    assert collapse_for("charades") is None, "a corpus with no fine space has no matrix"
    print("  M7 pooling gives log(n_members) for equal logits and -inf for an absent class, "
          "on both fine spaces")


def test_m8_the_head_size_belongs_to_the_shards_not_to_the_training_script():
    """A 22-class corpus must not be silently trained through a 20-way head.

    `activity.CLASS_NAMES` has 20 entries and `COARSE_V11` has 22 - ids 20 (`using_device`) and
    21 (`object_interaction`) exist only in the Toyota mapping. Both `train_adl.py` and
    `SkeletonWindowDataset` read `N_CLASSES` from module scope, so the 365k Toyota RTMO windows
    could not be trained at all, and the failure mode depended on where it surfaced: the loader's
    bound check catches it loudly, but a loader that had let it through would hand
    `cross_entropy` a target outside the logit range.

    So the bound is now a parameter with the 20-class default unchanged, and BOTH directions are
    asserted here. The negative control is the point: a test that only proved 22 works would pass
    equally well if the guard had simply been deleted.
    """
    from behaviorsense.agents.activity import N_CLASSES
    from behaviorsense.data.skeleton_dataset import SkeletonWindowDataset

    assert N_CLASSES == 20 and len(COARSE_V11) == 22, (N_CLASSES, len(COARSE_V11))
    assert COARSE_V11[20:22] == ("using_device", "object_interaction"), COARSE_V11[20:22]

    import tempfile

    n, T, M, V, C = 66, 30, 2, 17, 3
    rng = np.random.default_rng(0)
    labels = np.array([i % 22 for i in range(n)], dtype=np.int64)
    skels = rng.normal(0.0, 0.4, (n, T, M, V, C)).astype(np.float16)
    tmp = Path(tempfile.mkdtemp()) / "toyota_trimmed_20hz_0000.npz"
    np.savez_compressed(
        tmp, skeletons=skels, labels=labels,
        subjects=np.array([f"p{i % 6:02d}" for i in range(n)], dtype="<U32"),
        datasets=np.array(["toyota_trimmed"] * n, dtype="<U32"))

    # NEGATIVE CONTROL: the 20-class default must refuse, and must NAME the offending ids.
    try:
        SkeletonWindowDataset([str(tmp)], n_frames=T)
        raise AssertionError(
            "a 22-class label set was accepted through the 20-class default; ids 20 and 21 would "
            "reach cross_entropy as targets outside the logit range")
    except ValueError as exc:
        assert "outside [0,20)" in str(exc) and "20, 21" in str(exc), str(exc)

    # And with the bound declared, the same shard loads with every class intact.
    ds = SkeletonWindowDataset([str(tmp)], n_frames=T, n_classes=len(COARSE_V11))
    assert ds.n_classes == 22 and len(ds) == n
    assert sorted(set(ds.labels.tolist())) == list(range(22)), sorted(set(ds.labels.tolist()))

    # Charades shards must be unaffected: the default is still 20, and a 20-class set loads.
    lab20 = np.array([i % 20 for i in range(n)], dtype=np.int64)
    tmp20 = tmp.with_name("adl_0000.npz")
    np.savez_compressed(
        tmp20, skeletons=skels, labels=lab20,
        subjects=np.array([f"p{i % 6:02d}" for i in range(n)], dtype="<U32"),
        datasets=np.array(["charades"] * n, dtype="<U32"))
    assert SkeletonWindowDataset([str(tmp20)], n_frames=T).n_classes == 20

    # The training entry point must expose it and thread it, or the loader's parameter is
    # unreachable from the command line that actually runs.
    src = (ROOT / "scripts" / "train_adl.py").read_text(encoding="utf-8")
    assert '"--n-classes"' in src, "train_adl.py has no --n-classes flag"
    assert "STGCNpp(n_classes=N_CLASSES)" not in src, (
        "the model head is still built from the module constant, so --n-classes would size the "
        "loader and the network differently")
    assert src.count("n_classes=args.n_classes") >= 5, (
        f"only {src.count('n_classes=args.n_classes')} call sites thread the flag; the probe, "
        "both datasets, both model constructions and the prior all need it")

    print(f"  M8 20-class default REFUSES ids [20, 21] by name; n_classes=22 loads all "
          f"{len(ds)} windows over {len(set(ds.labels.tolist()))} classes; Charades shards still "
          f"default to {SkeletonWindowDataset([str(tmp20)], n_frames=T).n_classes}; "
          f"train_adl.py threads the flag at "
          f"{src.count('n_classes=args.n_classes')} sites")


def test_m9_taxonomy_v11_parity_and_a_22_class_head_can_emit_a_segment():
    """Two files define the 22-class taxonomy, and a segment must survive ids 20 and 21.

    `data/toyota.py` owns `COARSE_V11` and `agents/activity.py` owns `EXTENDED_CLASS_NAMES`. They
    are duplicated on purpose - `agents` must not import `data` - which means nothing at import
    time stops them drifting. Asserted here instead, because drift renames activities silently:
    a segment would carry `object_interaction` where the model meant `using_device`.

    The rest is the serving path for a 22-class head, which had three hard stops:

      * `ActivitySegment.activity_id` was `Field(ge=0, le=19)` - ids 20/21 failed validation
      * `activity_name=CLASS_NAMES[label]` would IndexError on a 20-name tuple
      * `EnsembleClassifier` built `STGCNpp(n_classes=N_CLASSES)`, so a 22-class checkpoint
        raised on load

    The bound is RAISED, not removed. An id outside the taxonomy is a relabelling bug, and
    passing it through would name a segment by the wrong activity rather than fail.
    """
    from datetime import datetime, timedelta

    from behaviorsense.agents.activity import (
        CLASS_NAMES,
        EXTENDED_CLASS_NAMES,
        N_CLASSES,
        N_EXTENDED_CLASSES,
    )
    from behaviorsense.schemas import ActivitySegment, Role

    # PARITY. The whole point of the test.
    assert tuple(EXTENDED_CLASS_NAMES) == tuple(COARSE_V11), (
        "activity.EXTENDED_CLASS_NAMES and toyota.COARSE_V11 have drifted; a segment would be "
        f"named by the wrong activity.\n  activity: {EXTENDED_CLASS_NAMES}\n  toyota:   "
        f"{tuple(COARSE_V11)}")
    assert tuple(EXTENDED_CLASS_NAMES[:N_CLASSES]) == tuple(CLASS_NAMES), (
        "the 20-class prefix changed, so every Charades checkpoint's head now means something "
        "different from what results/evaluation.md measured")
    assert N_EXTENDED_CLASSES == 22 and N_CLASSES == 20

    # A segment at each new id validates and is named correctly.
    t0 = datetime(2026, 1, 1, 9, 0, 0)
    for label in (20, 21):
        seg = ActivitySegment(
            segment_id=f"t0-x-{label}", track_id=0, role=Role.RESIDENT,
            activity_id=label, activity_name=EXTENDED_CLASS_NAMES[label],
            start_time=t0, end_time=t0 + timedelta(seconds=2), confidence=0.8)
        assert seg.activity_name == COARSE_V11[label], (seg.activity_name, COARSE_V11[label])
    assert EXTENDED_CLASS_NAMES[20] == "using_device"
    assert EXTENDED_CLASS_NAMES[21] == "object_interaction"

    # NEGATIVE CONTROL: the bound is raised, not deleted. 22 is still out of the taxonomy.
    try:
        ActivitySegment(
            segment_id="t0-x-22", track_id=0, role=Role.RESIDENT,
            activity_id=22, activity_name="nonexistent",
            start_time=t0, end_time=t0 + timedelta(seconds=2), confidence=0.8)
        raise AssertionError(
            "activity_id=22 was accepted; the bound has been removed rather than raised, so a "
            "relabelling bug would pass through and name a segment by the wrong activity")
    except Exception as exc:                                   # pydantic ValidationError
        assert "activity_id" in str(exc) or "less than or equal to 21" in str(exc), str(exc)

    # And the loader takes its head size from the checkpoint rather than a module constant.
    ens = (ROOT / "src" / "behaviorsense" / "models" / "ensemble.py").read_text(encoding="utf-8")
    assert "STGCNpp(n_classes=N_CLASSES)" not in ens, (
        "the ensemble still hard-codes the head size, so a 22-class checkpoint cannot load")
    assert 'ck.get("args", {}).get("n_classes")' in ens, (
        "the head size is not read from the checkpoint that records it")

    print(f"  M9 EXTENDED_CLASS_NAMES == COARSE_V11 ({N_EXTENDED_CLASSES} names, first "
          f"{N_CLASSES} unchanged); segments validate and name correctly at ids 20 "
          f"({EXTENDED_CLASS_NAMES[20]}) and 21 ({EXTENDED_CLASS_NAMES[21]}); id 22 still "
          f"refused; ensemble reads its head size from the checkpoint")


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
