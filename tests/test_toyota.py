"""Toyota Smarthome ingestion: protocol, parser, mapping, and the semantics of a gap.

Every constant in `data/toyota.py` was copied out of an official artefact, so the tests that
matter here are the ones that would catch a *plausible* transcription: a subject list that is
one id short, an annotation bound read as seconds instead of frames, a class tuple sorted for
tidiness, a fine class quietly defaulting to the reject class.

T7 is the one to read first if you only read one. It is the test that would have caught the
mistake this corpus exists to fix.

Run: python tests/test_toyota.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behaviorsense.data.toyota import (  # noqa: E402
    ALL_SUBJECTS,
    COARSE_V11,
    CS_TEST_SUBJECTS,
    CS_TRAIN_SUBJECTS,
    CV1_TEST_CAMERAS,
    CV1_TRAIN_CAMERAS,
    TSM_CLASSES,
    TSM_CV_CLASSES,
    TSM_FPS,
    TSM_FPS_DOCUMENTED,
    TSM_RGB_OFFSET,
    TSM_RGB_UNMATCHED,
    TSM_SPLIT_SIZES,
    TSM_TO_COARSE,
    TSU_ALIASES,
    TSU_CLASSES,
    TSU_FPS,
    TSU_TO_COARSE,
    TsuVideo,
    VideoId,
    assert_coarse_parity,
    canonical_tsu_class,
    collapse_matrix,
    cross_check_annotations,
    frame_counts,
    frame_mask,
    frame_multilabel,
    iter_videos,
    load_trimmed_split,
    load_tsu_annotations,
    load_tsu_annotations_from_csv,
    parse_tsm_name,
    parse_tsu_filename,
    parse_tsu_name,
    protocol_side,
    read_annotation_csv,
    slug,
    supervised_coarse,
    survey_annotation_csvs,
    to_coarse,
    tsm_id,
    tsm_name,
    tsu_id,
)

# A three-video stand-in with the real JSON shape: one train subject, one test subject, and
# one cross-view camera. Bounds are FRAMES, which is the whole point of T3.
FIXTURE = {
    "P03T01C01": {"subset": "training", "duration": 1000,
                  "actions": [[19, 100, 400], [20, 400, 450], [1, 0, 100]]},
    "P25T02C05": {"subset": "training", "duration": 500,
                  "actions": [[26, 0, 500]]},
    "P10T09C02": {"subset": "testing", "duration": 800,
                  "actions": [[27, 200, 300], [48, 300, 800]]},
}


def _fixture_path(tmp: Path, data: dict | None = None) -> Path:
    p = tmp / "smarthome_CS_51.json"
    p.write_text(json.dumps(data if data is not None else FIXTURE), encoding="utf-8")
    return p


def test_t1_class_tables_are_the_official_ones():
    """Counts, order-sensitivity, and the two spellings of the same activities."""
    assert len(TSU_CLASSES) == 51, f"{len(TSU_CLASSES)} untrimmed classes, expected 51"
    assert len(TSM_CLASSES) == 31, f"{len(TSM_CLASSES)} trimmed classes, expected 31"
    assert len(TSM_CV_CLASSES) == 19, f"{len(TSM_CV_CLASSES)} cross-view, expected 19"
    assert len(set(TSU_CLASSES)) == 51, "duplicate name in TSU_CLASSES"

    # Order is the id space. Sorting it for tidiness relabels the whole corpus, so pin the
    # anchors rather than the whole list.
    for name, idx in (("Enter", 0), ("Walk", 1), ("Read", 19), ("Take_pills", 20),
                      ("Watch_TV", 26), ("Use_tablet", 48), ("Pour.From_can", 50)):
        assert tsu_id(name) == idx, f"{name} is id {tsu_id(name)}, official id is {idx}"
    assert TSU_CLASSES != tuple(sorted(TSU_CLASSES)), (
        "TSU_CLASSES is alphabetically sorted, which the official Action_list is not - "
        "somebody tidied the id space")

    # The trimmed half spells them differently. If these ever coincide, one table was
    # pasted over the other.
    assert "Readbook" in TSM_CLASSES and "Read" in TSU_CLASSES
    assert "Takepills" in TSM_CLASSES and "Take_pills" in TSU_CLASSES
    assert slug("Stir_coffee/tea") == "stir_coffee_tea", "slash must not survive slugging"
    print(f"  T1 51 untrimmed / 31 trimmed / 19 cross-view classes, id order pinned at "
          f"7 anchors, both spellings present")


def test_t2_cross_subject_partition_is_disjoint_and_complete():
    """11 + 7 = 18 subjects, no overlap, and the id gaps are preserved."""
    train, test = set(CS_TRAIN_SUBJECTS), set(CS_TEST_SUBJECTS)
    assert len(train) == 11, f"{len(train)} train subjects, official CS has 11"
    assert len(test) == 7, f"{len(test)} test subjects, official CS has 7"
    assert not train & test, f"subject leakage across the CS boundary: {sorted(train & test)}"
    assert train | test == set(ALL_SUBJECTS), (
        f"partition does not cover ALL_SUBJECTS; missing "
        f"{sorted(set(ALL_SUBJECTS) - (train | test))}")
    # The numbering has gaps. `range(1, 19)` looks right, invents p1 and p5, and drops p25.
    assert set(ALL_SUBJECTS) != set(range(1, 19)), (
        "ALL_SUBJECTS is 1..18, but the real ids skip 1, 5, 8 and 21-24 and include 25")
    assert 25 in train and 1 not in set(ALL_SUBJECTS)
    print(f"  T2 CS partition {sorted(train)} / {sorted(test)} - disjoint, covers all "
          f"{len(ALL_SUBJECTS)} real subject ids, gaps preserved")


def test_t3_annotation_bounds_are_frames_not_seconds():
    """The unit trap, asserted numerically.

    `duration` is a frame count at 25 Hz. Read as seconds, a 1000-frame video becomes 1000 s
    instead of 40 s and every label span shrinks by 25x - which does not crash, does not
    look wrong, and silently destroys the supervision.
    """
    with tempfile.TemporaryDirectory() as td:
        v = load_tsu_annotations(_fixture_path(Path(td)))["P03T01C01"]
    assert v.duration == 1000
    assert abs(v.seconds - 40.0) < 1e-9, (
        f"1000 frames at {TSU_FPS:g} Hz is 40 s, got {v.seconds}")

    # `Read` covers frames 100..400 of 1000. At 100 steps that is steps 10..40 exclusive.
    lab = frame_multilabel(v, 100)
    read = lab[:, tsu_id("Read")]
    assert read[15] == 1.0 and read[5] == 0.0 and read[45] == 0.0
    assert 28 <= read.sum() <= 30, (
        f"Read covers {read.sum()} of 100 steps; frames 100-400 of 1000 is ~29 after the "
        "official strict bounds. A seconds/frames mix-up lands nowhere near this.")
    print(f"  T3 1000 frames = {v.seconds:.0f} s at {TSU_FPS:g} Hz; Read spans "
          f"{int(read.sum())}/100 steps from frames 100-400")


def test_t4_official_loader_parity_on_strict_bounds():
    """Reproduce the reference implementation exactly, off-by-one included."""
    with tempfile.TemporaryDirectory() as td:
        v = load_tsu_annotations(_fixture_path(Path(td)))["P25T02C05"]

    # `Watch_TV` covers the whole video, 0..500 of 500. The official strict `>` drops step 0.
    official = frame_multilabel(v, 50, official=True)[:, tsu_id("Watch_TV")]
    inclusive = frame_multilabel(v, 50, official=False)[:, tsu_id("Watch_TV")]
    assert official[0] == 0.0, "official semantics label a boundary step as negative"
    assert inclusive[0] == 1.0, "inclusive semantics must close the interval"
    assert inclusive.sum() == official.sum() + 1, (
        f"the two modes differ by more than the boundary: {inclusive.sum()} vs "
        f"{official.sum()}")
    print(f"  T4 official mode drops the boundary step ({int(official.sum())}/50) and "
          f"inclusive keeps it ({int(inclusive.sum())}/50) - published mAP stays comparable")


def test_t5_protocol_side_derives_from_the_id_not_the_json_field():
    """A CV run that filtered on `subset` would silently report the CS split."""
    with tempfile.TemporaryDirectory() as td:
        V = load_tsu_annotations(_fixture_path(Path(td)))

    cs_train = [v.vid for v in iter_videos(V, "CS", "train")]
    cs_test = [v.vid for v in iter_videos(V, "CS", "test")]
    assert cs_train == ["P03T01C01", "P25T02C05"], cs_train
    assert cs_test == ["P10T09C02"], cs_test

    # Same three videos, different protocol, different answer. P03T01C01 is on camera 1
    # (CV1 train) while P10T09C02 is on camera 2 (CV1 test) - and P10 is a CS TEST subject,
    # so a `subset`-based filter would have put it on the wrong side here.
    cv1_train = [v.vid for v in iter_videos(V, "CV1", "train")]
    cv1_test = [v.vid for v in iter_videos(V, "CV1", "test")]
    assert cv1_train == ["P03T01C01"], cv1_train
    assert cv1_test == ["P10T09C02"], cv1_test
    assert V["P10T09C02"].subset == "testing" and "P10T09C02" in cv1_test

    assert protocol_side(VideoId(3, 1, 1), "CS") == "train"
    assert protocol_side(VideoId(10, 1, 1), "CS") == "test"
    assert protocol_side(VideoId(3, 1, 4), "CV1") == "unused", (
        f"camera 4 is not in CV1 train {CV1_TRAIN_CAMERAS} / test {CV1_TEST_CAMERAS}, so it "
        "must be 'unused' rather than silently landing in train")
    try:
        protocol_side(VideoId(3, 1, 1), "xsub")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown protocol name must raise, not default to CS")
    print("  T5 CS and CV1 disagree on the same three videos, both derived from the id; "
          "unknown protocol raises; out-of-protocol camera is 'unused'")


def test_t6_a_broken_split_file_is_refused():
    """The partition check is containment, so it catches leakage at any file size.

    Equality against the full 18-subject partition is what confirmed both halves of the
    dataset share one CS boundary, but enforcing only equality would reject every
    legitimately filtered subset - so a stray subject is the error, not a small file.
    """
    broken = {k: dict(v) for k, v in FIXTURE.items()}
    broken["P10T09C02"]["subset"] = "training"      # a test subject marked for training
    with tempfile.TemporaryDirectory() as td:
        try:
            load_tsu_annotations(_fixture_path(Path(td), broken))
        except ValueError as exc:
            assert "official partition" in str(exc) and "[10]" in str(exc), str(exc)
        else:
            raise AssertionError(
                "a CS file whose `subset` disagrees with the subject partition loaded "
                "without complaint - the leak this project already paid for once")

    # And the subset must still load. This is the case the first version of the check broke.
    with tempfile.TemporaryDirectory() as td:
        subset = {"P03T01C01": FIXTURE["P03T01C01"]}
        assert len(load_tsu_annotations(_fixture_path(Path(td), subset))) == 1, (
            "a one-video subset was rejected; the check must be containment, not equality")

    bad_class = {"P03T01C01": {"subset": "training", "duration": 100,
                               "actions": [[51, 0, 50]]}}
    with tempfile.TemporaryDirectory() as td:
        try:
            load_tsu_annotations(_fixture_path(Path(td), bad_class))
        except ValueError as exc:
            assert "outside [0,51)" in str(exc), str(exc)
        else:
            raise AssertionError("class id 51 must be refused; there are only 0..50")
    print("  T6 a test subject marked `training` is refused, a filtered subset loads, "
          "out-of-range class id is refused")


def test_t7_no_fine_class_falls_through_to_the_reject_class():
    """The test that would have caught the mistake this whole corpus exists to fix.

    The Charades map's keyword fallback swallowed a large part of the class list into
    `other_idle`, and the reject class became the largest thing the model was trained on.
    Here, `collapse_matrix` REFUSES an unmapped class instead of defaulting it, and no
    Toyota class is allowed to point at `other_idle` at all.
    """
    for label, fine, mapping in (("untrimmed", TSU_CLASSES, TSU_TO_COARSE),
                                 ("trimmed", TSM_CLASSES, TSM_TO_COARSE)):
        M = collapse_matrix(fine, mapping)
        assert M.shape == (len(fine), len(COARSE_V11)), M.shape
        assert np.all(M.sum(axis=1) == 1), (
            f"{label}: {int((M.sum(axis=1) != 1).sum())} classes map to != 1 coarse target")
        idle = [n for n in fine if mapping[n] == "other_idle"]
        assert not idle, (
            f"{label}: {len(idle)} classes routed to the reject class ({idle[:4]}). Every "
            "Toyota class is a real activity - a fallback here is the Charades failure again")

    # And the refusal, proved rather than assumed.
    try:
        collapse_matrix(TSU_CLASSES, {k: v for k, v in list(TSU_TO_COARSE.items())[:-3]})
    except KeyError as exc:
        assert "no coarse target" in str(exc), str(exc)
    else:
        raise AssertionError("an incomplete map must raise, not default the remainder")

    # The two classes that exist because of this: 18% of the corpus would otherwise be idle.
    for name in ("Use_laptop", "Use_tablet"):
        assert TSU_TO_COARSE[name] == "using_device", name
    for name in ("Use_Drawer", "Use_cupboard", "Use_fridge",
                 "Put_something_on_table", "Take_something_off_table"):
        assert TSU_TO_COARSE[name] == "object_interaction", name
    print("  T7 all 51 + 31 classes map to exactly one coarse class, NONE to other_idle; "
          "an incomplete map raises; the 7 classes that motivated ids 20-21 are pinned")


def test_t8_taxonomy_is_appended_not_renumbered():
    """Ids 0..19 must still be `activity.CLASS_NAMES`, or every existing shard is relabelled."""
    from behaviorsense.agents.activity import CLASS_NAMES, N_CLASSES

    assert_coarse_parity()
    assert COARSE_V11[:N_CLASSES] == CLASS_NAMES, "ids 0..19 drifted from the trained head"
    assert len(COARSE_V11) == 22 and COARSE_V11[20:] == ("using_device", "object_interaction")
    # The shards carry only the coarse label id, so an id that MOVED cannot be repaired by
    # remapping - it needs the ~6 h re-extraction. Appending is the affordable shape.
    assert COARSE_V11.index("other_idle") == 19, (
        "other_idle moved off id 19; every existing shard now means something else")
    print(f"  T8 {len(CLASS_NAMES)} trained ids unchanged, 2 appended at 20-21, "
          "other_idle still 19 - existing shards stay readable")


def test_t9_gaps_are_masked_not_labelled_negative():
    """A third of the real corpus is gap, and the two heads must read it differently."""
    with tempfile.TemporaryDirectory() as td:
        V = load_tsu_annotations(_fixture_path(Path(td)))
    v = V["P10T09C02"]           # annotated 200..800 of 800 -> 25% gap

    mask = frame_mask(v, 80)
    assert mask[:19].sum() == 0.0, "frames before the first annotation must be masked out"
    assert mask[30:].min() == 1.0, "annotated frames must be unmasked"
    assert 0.7 < mask.mean() < 0.8, f"gap fraction {1 - mask.mean():.2f}, expected ~0.25"

    # The distinction that matters: for the 51-class head a gap is a true negative; for the
    # unified head it is unknown, because `sitting` and `standing` are exactly what a gap
    # most likely contains and Toyota never labels them.
    lab = frame_multilabel(v, 80)
    assert lab[0].sum() == 0.0, "a gap row is all-zero, which the benchmark head wants"
    absent = [COARSE_V11[i] for i in range(len(COARSE_V11))
              if i not in supervised_coarse(TSU_TO_COARSE)]
    assert set(absent) == {"standing", "sitting", "bending_reaching", "falling",
                           "fallen_on_ground", "interacting_with_person", "other_idle"}, absent
    print(f"  T9 mask covers {mask.mean():.2f} of steps; the 7 classes Toyota cannot "
          f"supervise are exactly {', '.join(sorted(absent))}")


def test_t10_coarse_target_is_an_or_not_a_sum():
    """Two fine classes in one coarse group on the same frame must give 1.0, never 2.0."""
    both = {"P03T01C01": {"subset": "training", "duration": 100,
                          # Cook.Cut and Cook.Stir overlap: same coarse class, same frames.
                          "actions": [[43, 0, 100], [45, 0, 100]]}}
    with tempfile.TemporaryDirectory() as td:
        v = load_tsu_annotations(_fixture_path(Path(td), both))["P03T01C01"]
    M = collapse_matrix(TSU_CLASSES, TSU_TO_COARSE)
    fine = frame_multilabel(v, 20)
    assert fine[10].sum() == 2.0, "fixture should put two fine classes on the same frame"
    coarse = to_coarse(fine, M)
    assert coarse.max() == 1.0, (
        f"coarse target reached {coarse.max()} - a BCE target above 1 is not a probability; "
        "`label @ M` counts members instead of taking their union")
    assert coarse[10, COARSE_V11.index("cooking_food_prep")] == 1.0
    print("  T10 overlapping members of one coarse class collapse to 1.0, not 2.0")


def test_t11_filenames_parse_and_bad_ones_raise():
    ident = parse_tsm_name("Cook.Cut_p03_r00_v02_c03.mp4")
    assert (ident.subject, ident.camera, ident.activity) == (3, 3, "Cook.Cut"), ident
    assert parse_tsm_name("WatchTV_p25_r01_v14_c07.json").activity == "WatchTV"
    assert parse_tsm_name("Drink.Fromcup_p10_r02_02_c05.npz").subject == 10
    assert parse_tsu_name("P11T15C01") == VideoId(11, 15, 1)

    for bad, fn in (("P11T15", parse_tsu_name), ("p11t15c01", parse_tsu_name),
                    ("Walk_p3_r0_v2_c3.mp4", parse_tsm_name), ("Walk.mp4", parse_tsm_name)):
        try:
            fn(bad)
        except ValueError:
            continue
        raise AssertionError(
            f"{bad!r} parsed instead of raising. A silently-unparsed id is assigned to no "
            "protocol side and vanishes from both train and test - which reads as a smaller "
            "dataset, not as a bug")
    print("  T11 both naming conventions parse; 4 malformed names raise rather than vanish")


def test_t12_frame_counts_measure_the_tail():
    with tempfile.TemporaryDirectory() as td:
        V = load_tsu_annotations(_fixture_path(Path(td)))
    c = frame_counts(V.values())
    assert c.shape == (51,)
    assert c[tsu_id("Read")] == 300 and c[tsu_id("Take_pills")] == 50
    assert c[tsu_id("Watch_TV")] == 500
    assert c[tsu_id("Lay_down")] == 0, "an unannotated class must count 0, not go missing"
    coarse = c @ collapse_matrix(TSU_CLASSES, TSU_TO_COARSE)
    assert coarse[COARSE_V11.index("other_idle")] == 0.0
    # Use_laptop (100 frames) and Use_tablet (500) are separate fine classes that share one
    # coarse class, so the coarse count must be their SUM - the collapse is a pooling, not a
    # pick-one.
    assert coarse[COARSE_V11.index("using_device")] == 600, (
        f"using_device got {coarse[COARSE_V11.index('using_device')]}, expected 100 + 500 "
        "from Use_laptop and Use_tablet")
    print(f"  T12 per-class frame counts sum to {int(c.sum())}; reject class gets 0; "
          "two fine classes pool into one coarse count")


def test_t13_the_two_official_trimmed_id_spaces_differ_by_one():
    """The most dangerous fact in the corpus, pinned.

    `2s-AGCN-For-Daily-Living` (skeleton) numbers the 31 trimmed classes 0..30.
    `i3d_smarthome` (RGB) numbers the same names 1..31 and reserves 0 for a name its
    if/elif chain did not match. Both are official and both are linked from the dataset's
    project page. Reading an RGB label file with the skeleton map shifts every class by one
    and adds a silent catch-all - which does not crash and produces a plausible number.
    """
    assert tsm_id("Cook.Cleandishes", space="skeleton") == 0
    assert tsm_id("Cook.Cleandishes", space="rgb") == 1
    assert tsm_id("WatchTV", space="skeleton") == 30
    assert tsm_id("WatchTV", space="rgb") == 31
    for name in TSM_CLASSES:
        assert tsm_id(name, space="rgb") - tsm_id(name, space="skeleton") == TSM_RGB_OFFSET

    # Cross-view is 1-indexed the same way, over its own 19 names.
    assert tsm_id("Cutbread", space="rgb", cross_view=True) == 1
    assert tsm_id("Walk", space="rgb", cross_view=True) == 19
    assert tsm_id("Walk", space="skeleton", cross_view=True) == 18

    # Round-trip, and the sentinel.
    for space in ("skeleton", "rgb"):
        for name in TSM_CLASSES:
            assert tsm_name(tsm_id(name, space=space), space=space) == name
    try:
        tsm_name(TSM_RGB_UNMATCHED, space="rgb")
    except KeyError as exc:
        assert "unmatched-name fallback" in str(exc), str(exc)
    else:
        raise AssertionError(
            "id 0 in the RGB space was resolved to a class. It is `_name_to_int`'s "
            "fall-through for a name it did not recognise, and treating it as a class is "
            "how an unrecognised filename becomes a confident wrong label")
    try:
        tsm_id("Readbook", space="i3d")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown id space must raise, not silently pick one")
    print(f"  T13 skeleton 0..30 vs RGB 1..31 for all {len(TSM_CLASSES)} names, cross-view "
          "1..19, round-trips both ways, RGB id 0 refused as the unmatched sentinel")


def test_t14_the_two_halves_have_different_frame_rates():
    """20 fps trimmed (measured), 25 untrimmed. And the README says 30 for trimmed.

    All three numbers are in the code because the conflict is real: `i3d_smarthome` documents
    30 fps for the trimmed clips and the shipped mp4s report 20. Anything converting trimmed
    frames to seconds must read the file; these constants exist so a mismatch is REPORTED
    rather than silently rescaling every duration by 1.5x.
    """
    assert TSU_FPS == 25.0
    assert TSM_FPS == 20.0, "measured from the shipped mp4s on Kaggle"
    assert TSM_FPS_DOCUMENTED == 30.0, "what i3d_smarthome's README claims"
    assert TSM_FPS != TSU_FPS, (
        "the two halves must not share one constant - untrimmed ships at x1.25 speed and "
        "is deframed at 25, trimmed is a different rate entirely")
    assert TSM_FPS != TSM_FPS_DOCUMENTED, (
        "if these ever agree the conflict has been resolved and the comment should say how")
    # The concrete case from the preflight: 163 frames of `Walk`.
    assert abs(163 / TSM_FPS - 8.15) < 0.05, "163 frames at 20 fps is 8.2 s"
    assert abs(163 / TSM_FPS_DOCUMENTED - 5.43) < 0.05, "and 5.4 s at 30"
    print(f"  T14 trimmed {TSM_FPS:g} fps measured vs {TSM_FPS_DOCUMENTED:g} documented, "
          f"untrimmed {TSU_FPS:g} - a 163-frame clip is 8.2s or 5.4s depending which")


def test_t15_trimmed_split_files_are_validated_not_trusted():
    """A truncated split file must fail loudly, and a CS validation set must hold TRAIN subjects."""
    good = ["Walk_p03_r01_v16_c06.mp4", "Laydown_p03_r00_v07_c04.mp4",
            "Drink.Fromcup_p17_r02_v09_c01.mp4"]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "train_CS.txt"
        p.write_text("\n".join(good) + "\n", encoding="utf-8")
        got = load_trimmed_split(p, expect=3)
        assert [v.subject for v in got] == [3, 3, 17]
        assert got[0].activity == "Walk" and got[1].activity == "Laydown"

        # The released file has 8,831 lines. A short one is a truncated download.
        try:
            load_trimmed_split(p)
        except ValueError as exc:
            assert "expected 8831" in str(exc), str(exc)
        else:
            raise AssertionError(
                "a 3-line train_CS.txt loaded as if it were the released 8,831-clip file")

        # A CS validation file containing a TEST subject is the leak worth refusing.
        leak = Path(td) / "validation_CS.txt"
        leak.write_text("Walk_p10_r01_v16_c06.mp4\n", encoding="utf-8")
        try:
            load_trimmed_split(leak, expect=1)
        except ValueError as exc:
            assert "[10]" in str(exc) and "TRAIN subjects" in str(exc), str(exc)
        else:
            raise AssertionError(
                "a CS validation file built from test subject p10 was accepted - tuning on "
                "it means the reported test number is not held out")
    assert TSM_SPLIT_SIZES["train_CS"] + TSM_SPLIT_SIZES["validation_CS"] \
        + TSM_SPLIT_SIZES["test_CS"] == 16129
    assert TSM_SPLIT_SIZES["validation_CV1"] == TSM_SPLIT_SIZES["validation_CV2"], (
        "CV1 and CV2 share one validation camera (c05), so their validation files must be "
        "the same size")
    print("  T15 split files parse with activity + subject, a truncated file raises, a CS "
          "validation set built from test subjects is refused")


def test_t16_real_kaggle_filenames_all_parse():
    """Every filename shape actually present in the uploaded Kaggle mounts.

    Taken verbatim from the mount listing, because the archives decorate the same ids three
    different ways and one of them broke the parser: `Walk_p25_r12_v15_c06_pose3d.json` is
    the entire refined-skeleton set, and an anchored pattern with no tag group rejected all
    of it while reporting "0 files matched" - which reads as a bad mount, not a bad regex.
    """
    trimmed = {
        "mp4/Walk_p03_r01_v15_c07.mp4": (3, 1, 7, "Walk", None),
        "json/Drink.Fromcup_p20_r02_v02_c05.json": (20, 2, 5, "Drink.Fromcup", None),
        "Walk_p25_r12_v15_c06_pose3d.json": (25, 12, 6, "Walk", "pose3d"),
        "depth/Walk_p03_r01_v15_c07.mp4": (3, 1, 7, "Walk", None),
    }
    for path, (sub, take, cam, act, tag) in trimmed.items():
        v = parse_tsm_name(path)
        assert (v.subject, v.take, v.camera, v.activity, v.tag) == (sub, take, cam, act, tag), \
            f"{path} -> {v}"

    untrimmed = {
        "Annotation/P18/P18T13C07.csv": ("P18T13C07", "test"),
        "Videos_mp4/P15T17C03.mp4": ("P15T17C03", "train"),
        "Skeleton/results_P17T07C02_lcrnet+v3d.json": ("P17T07C02", "train"),
        "Depth/P15T17C03.mp4": ("P15T17C03", "train"),
    }
    for path, (vid, side) in untrimmed.items():
        v = parse_tsu_filename(path)
        assert v.tsu == vid, f"{path} -> {v.tsu}, expected {vid}"
        assert protocol_side(v, "CS") == side, f"{path} -> {protocol_side(v, 'CS')}"

    # `tsu` must zero-pad. `P2T2C3` matches no file on disk.
    assert VideoId(2, 2, 3).tsu == "P02T02C03"
    # Ambiguity is refused rather than resolved by picking the first.
    try:
        parse_tsu_filename("P02T02C03_vs_P04T05C06.json")
    except ValueError as exc:
        assert "more than one video id" in str(exc), str(exc)
    else:
        raise AssertionError("two ids in one filename must raise, not silently pick one")
    print(f"  T16 all {len(trimmed)} trimmed and {len(untrimmed)} untrimmed real filename "
          "shapes parse; tsu zero-pads; a two-id filename is refused")


def test_t17_annotation_csv_schema_is_inferred_then_asserted():
    """The archive ships per-video CSVs with no documented schema, so both plausible
    layouts must parse identically and everything else must raise.

    Why this matters more than it looks: the aggregated JSON that every published baseline
    trains against is a DERIVED artefact. The CSVs are what Inria actually ships. If the
    schema is misread, the label tensor still has the right shape.
    """
    want = ((19, 100, 400), (20, 400, 450), (1, 0, 100))
    variants = {
        "index, 3 columns": "19,100,400\n20,400,450\n1,0,100\n",
        "index, 4 columns (event-map)": "19,100,400,1\n20,400,450,1\n1,0,100,1\n",
        "class names": "Read,100,400\nTake_pills,400,450\nWalk,0,100\n",
        "with header": "label,start,end\nRead,100,400\nTake_pills,400,450\nWalk,0,100\n",
        "semicolons": "19;100;400\n20;400;450\n1;0;100\n",
        "blank lines and BOM": "﻿19,100,400\n\n20,400,450\n1,0,100\n\n",
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "P03T01C01.csv"
        for label, body in variants.items():
            p.write_text(body, encoding="utf-8")
            got = read_annotation_csv(p)
            assert got == want, f"{label}: {got} != {want}"

        for body, fragment in (
            ("Read,abc,400\n", "not numeric"),
            ("Nonsense,1,2\n", "neither an integer class id"),
            ("19,400,100\n", "precedes start"),
            ("19,100\n", "at least 3 fields"),
            ("51,0,10\n", "outside [0,51)"),
        ):
            p.write_text(body, encoding="utf-8")
            try:
                read_annotation_csv(p)
            except ValueError as exc:
                assert fragment in str(exc), f"{body!r}: {exc}"
            else:
                raise AssertionError(f"{body!r} was accepted; it must raise ({fragment})")
    print(f"  T17 {len(variants)} CSV layouts parse to the same spans; 5 malformed inputs "
          "raise with the offending line named")


def test_t18_csv_tree_needs_durations_and_is_cross_checked():
    """`duration` has no default, and the two annotation sources are compared, not trusted.

    The CSVs carry spans but not the video length, and `frame_multilabel` scales every step
    by `duration / n_steps` - so a guessed duration rescales the whole label tensor while
    leaving its shape correct. That is why the parameter is required rather than optional.
    """
    with tempfile.TemporaryDirectory() as td:
        ann = Path(td) / "Annotation"
        (ann / "P03").mkdir(parents=True)
        (ann / "P03" / "P03T01C01.csv").write_text("19,100,400\n1,0,100\n", encoding="utf-8")
        (ann / "P10").mkdir(parents=True)
        (ann / "P10" / "P10T09C02.csv").write_text("27,200,300\n", encoding="utf-8")

        # A missing duration is refused, naming the videos.
        try:
            load_tsu_annotations_from_csv(ann, {"P03T01C01": 1000})
        except ValueError as exc:
            assert "P10T09C02" in str(exc) and "duration" in str(exc), str(exc)
        else:
            raise AssertionError("a video with annotations but no duration must raise")

        got = load_tsu_annotations_from_csv(
            ann, {"P03T01C01": 1000, "P10T09C02": 800})
        assert set(got) == {"P03T01C01", "P10T09C02"}
        # Sides derived from the official CS partition, not from any file.
        assert got["P03T01C01"].subset == "training" and got["P10T09C02"].subset == "testing"

        # Cross-check against a JSON that agrees, then one that does not.
        same = dict(got)
        rep = cross_check_annotations(got, same)
        assert rep["compared"] == 2 and rep["agree"] == 2 and not rep["disagree"]

        shifted = {
            k: TsuVideo(vid=v.vid, subject=v.subject, take=v.take, camera=v.camera,
                        subset=v.subset, duration=v.duration,
                        actions=tuple((c, s + 1, e) for c, s, e in v.actions))
            for k, v in got.items()
        }
        rep = cross_check_annotations(got, shifted)
        assert len(rep["disagree"]) == 2, (
            "a one-frame shift in every span went unreported; the cross-check is the only "
            "thing standing between an inferred schema and a 6 h extraction run")
        assert rep["disagree"][0]["only_in_csv"], "the report must name what differs"
        print(f"  T18 missing duration refused by name; sides from the CS partition; a "
              f"1-frame shift in every span is caught on {len(rep['disagree'])}/2 videos")


def test_t19_the_real_csv_schema_and_the_documented_merges():
    """The schema the Kaggle run confirmed, and the two merges that killed it.

    Header and rows are verbatim from the first real run's output:

        event,start_frame,end_frame
        Enter,170,187
        Make_coffee.Get_water,236,502

    `Make_coffee.Get_water` is not one of the 51 classes. The dataset README says to merge
    it with `Get_water`, and `Make_tea.Insert_tea_bag` with `Insert_tea_bag` - the CSVs ship
    a PRE-MERGE 53-name vocabulary while every published number is over the merged 51.
    Without the aliases the parser dies on file 1 of 536; without the merge the corpus is a
    53-class problem comparable to no baseline.
    """
    real = (
        "event,start_frame,end_frame\n"
        "Enter,170,187\n"
        "Walk,181,222\n"
        "Make_coffee,233,1615\n"
        "Make_coffee.Get_water,236,502\n"
        "Make_tea.Insert_tea_bag,600,640\n"
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "P02T01C06.csv"
        p.write_text(real, encoding="utf-8")
        got = read_annotation_csv(p)

    assert [c for c, _, _ in got] == [
        tsu_id("Enter"), tsu_id("Walk"), tsu_id("Make_coffee"),
        tsu_id("Get_water"), tsu_id("Insert_tea_bag"),
    ], got
    assert got[3] == (tsu_id("Get_water"), 236, 502), (
        "Make_coffee.Get_water must land on Get_water with its span intact")
    assert canonical_tsu_class("Make_tea.Insert_tea_bag") == "Insert_tea_bag"
    assert canonical_tsu_class("Walk") == "Walk", "a known name must pass through unchanged"
    # Every alias target must be a real class, or the alias table is the new bug.
    for raw, target in TSU_ALIASES.items():
        assert target in TSU_CLASSES, f"alias {raw!r} -> {target!r}, which is not a class"
    print(f"  T19 real header + 5 rows parse; {len(TSU_ALIASES)} documented merges applied, "
          "every alias target is a real class")


def test_t20_unknown_names_are_surveyed_not_fatal():
    """One pass must report EVERY unrecognised name, not die on the first.

    This is the test for the 93-minute failure. The parser examined one file of 536, raised
    on `Make_coffee.Get_water`, and told us nothing about `Make_tea.Insert_tea_bag` - which
    was in the same file, four lines later. A survey costs the same as a fatal parse.
    """
    with tempfile.TemporaryDirectory() as td:
        ann = Path(td) / "Annotation"
        (ann / "P03").mkdir(parents=True)
        (ann / "P03" / "P03T01C01.csv").write_text(
            "event,start_frame,end_frame\nWalk,0,10\nMystery_One,10,20\n", encoding="utf-8")
        (ann / "P10").mkdir(parents=True)
        (ann / "P10" / "P10T09C02.csv").write_text(
            "event,start_frame,end_frame\nMystery_Two,0,5\n"
            "Make_coffee.Get_water,5,9\n", encoding="utf-8")

        # Collect mode: unknown rows are skipped, known ones still parse.
        unk: list[str] = []
        rows = read_annotation_csv(ann / "P03" / "P03T01C01.csv", unknown=unk)
        assert len(rows) == 1 and unk == ["Mystery_One"], (rows, unk)

        rep = survey_annotation_csvs(ann)
        assert rep["n_files"] == 2 and rep["n_rows"] == 4, rep
        assert set(rep["unknown"]) == {"Mystery_One", "Mystery_Two"}, (
            f"the survey must report BOTH unknown names in one pass, got {rep['unknown']}")
        assert rep["unknown"]["Mystery_Two"] == "P10T09C02.csv", "report where each was seen"
        assert rep["aliased"] == {"Make_coffee.Get_water": 1}, rep["aliased"]

        # Default mode still raises, and names the escape hatch.
        try:
            read_annotation_csv(ann / "P03" / "P03T01C01.csv")
        except ValueError as exc:
            assert "unknown=[]" in str(exc), (
                f"the error must point at the survey mode; got {exc}")
        else:
            raise AssertionError("without `unknown=`, an unrecognised name must still raise")
    print("  T20 survey reports both unknown names with their files in one pass; "
          "collect mode skips them; default mode raises and names the escape hatch")


def test_t21_cross_check_scores_frames_not_only_spans():
    """The disagreements the first real run found, scored the way the model sees them.

    536 videos compared, 82 span-different - and the examples showed why span equality
    over-reports: the CSVs carry finer spans and the JSON merges adjacent same-class
    intervals. A pure split describes IDENTICAL frames and must score 1.0; only a moved
    boundary may cost anything.
    """
    def V(vid, dur, actions):
        return TsuVideo(vid=vid, subject=3, take=1, camera=1, subset="training",
                        duration=dur, actions=tuple(actions))

    # Pure split: CSV finer, JSON merged, same coverage. Benign.
    a = V("P0XT01C01", 4000, [(2, 1340, 2960), (2, 2960, 2963)])
    b = V("P0XT01C01", 4000, [(2, 1340, 2963)])
    rep = cross_check_annotations({a.vid: a}, {b.vid: b})
    assert rep["disagree"], "the span sets do differ, and that must still be reported"
    assert rep["frame_agreement_mean"] == 1.0, (
        f"a pure split describes identical frames; got {rep['frame_agreement_mean']}")
    assert rep["disagree"][0]["cells_differing"] == 0
    assert rep["videos_below_999"] == 0

    # P03T16C03 from the real run: split AND a boundary moved 140 frames.
    a = V("P03T16C03", 4000, [(2, 1340, 2960), (2, 2960, 2963)])
    b = V("P03T16C03", 4000, [(2, 1480, 2963)])
    rep = cross_check_annotations({a.vid: a}, {b.vid: b})
    assert 0.998 < rep["frame_agreement_mean"] < 1.0, rep["frame_agreement_mean"]
    assert rep["disagree"][0]["cells_differing"] == 140, (
        f"a 140-frame boundary shift must cost exactly 140 cells, got "
        f"{rep['disagree'][0]['cells_differing']}")

    # P02T11C02: a 4,500-frame difference is a real content disagreement.
    a = V("P02T11C02", 20000, [(35, 200, 220), (35, 220, 18540)])
    b = V("P02T11C02", 20000, [(35, 200, 14040)])
    rep = cross_check_annotations({a.vid: a}, {b.vid: b})
    assert rep["disagree"][0]["cells_differing"] == 4500, rep["disagree"][0]
    assert rep["videos_below_999"] == 1, "this one must be flagged as below 0.999"
    print("  T21 pure split scores 1.00000 with 0 cells differing; a 140-frame boundary "
          "costs exactly 140; a 4,500-frame difference is flagged below 0.999")


def test_t22_pose3d_reader_matches_the_measured_schema_and_refuses_a_scrambled_layout():
    """INRIA's own skeletons, read against the schema MEASURED from all three archives.

    Probed 2026-08-26 on Kaggle. All three releases share one layout:

        {"K": int, "njts": 13, "frames": [[{pose2d: 26 floats, pose3d: 39 floats}, ...], ...]}

    trimmed V1.1 (`Drink.Fromcup_p20_r02_v02_c05.json`, K=20, plus `cumscore`), trimmed V1.2
    (`Walk_p25_r12_v15_c06_pose3d.json`, K=5, no score), untrimmed
    (`results_P17T07C02_lcrnet+v3d.json`, K=5, 19,143 frames).

    The dangerous part is the FLAT array. 39 floats is 13 joints x 3, and both
    `[x0..x12, y0..y12, z0..z12]` and `[x0,y0,z0, x1,y1,z1, ...]` reshape without error - one
    builds a body, the other swaps joints for coordinates and trains perfectly well on nonsense.
    LCR-Net's visualiser plots joint `i` at `(pose2d[i], pose2d[i+njts])`, so it is
    coordinate-major; this test proves the reader agrees AND that it rejects the other ordering
    rather than accepting it silently. That negative control is the whole value here - a wrong
    layout is invisible in a loss curve.

    Two alignment properties also matter: a frame where LCR-Net detected nobody is an empty
    LIST, and it must survive as a zero-filled frame with `present=False`, because dropping it
    shortens the clip and desynchronises every frame-indexed annotation.
    """
    import json
    import pathlib
    import tempfile

    from behaviorsense.data.toyota import (
        AGCN_JOINTS,
        LCRNET_JOINTS,
        LCRNET_NJTS,
        derive_agcn_joints,
        read_pose3d,
        unflatten_lcrnet,
        verify_lcrnet_layout,
    )

    J = LCRNET_NJTS
    assert J == 13 and len(LCRNET_JOINTS) == 13 and len(AGCN_JOINTS) == 15

    # An UPRIGHT person in image coordinates: ankles at the bottom, head at the top.
    ys = {"right_ankle": 400, "left_ankle": 398, "right_knee": 320, "left_knee": 318,
          "right_hip": 250, "left_hip": 248, "right_wrist": 260, "left_wrist": 258,
          "right_elbow": 230, "left_elbow": 228, "right_shoulder": 180,
          "left_shoulder": 178, "head": 120}
    xs = {n: 300 + (5 if n.startswith("right") else -5) for n in ys}
    xy = np.array([[xs[n] for n in LCRNET_JOINTS], [ys[n] for n in LCRNET_JOINTS]],
                  dtype=np.float32)                       # [2, J]
    xyz = np.vstack([xy, np.full((1, J), 2.5, np.float32)])
    flat2d = xy.reshape(-1).tolist()                       # coordinate-major, as released
    flat3d = xyz.reshape(-1).tolist()

    assert np.allclose(unflatten_lcrnet(flat2d, dim=2), xy.T), "coordinate-major round-trip"

    check = verify_lcrnet_layout([flat2d] * 20)
    assert check["winner"] == "coordinate_major" and check["matches_constant"], check
    assert check["decisive"] is True, check
    assert check["coordinate_major_score"] > check["joint_major_score"], check

    def write(doc, name="Walk_p25_r12_v15_c06_pose3d.json"):
        f = pathlib.Path(tempfile.mkdtemp()) / name
        f.write_text(json.dumps(doc), encoding="utf-8")
        return f

    # V1.2 shape, with one frame where nobody was detected.
    frames = [[{"pose2d": flat2d, "pose3d": flat3d}] for _ in range(30)]
    frames[7] = []
    rec = read_pose3d(write({"K": 5, "njts": J, "frames": frames}))
    assert rec["n_frames"] == 30 and rec["n_missing"] == 1, (rec["n_frames"], rec["n_missing"])
    assert not rec["present"][7] and rec["present"][6] and rec["present"][8]
    assert np.allclose(rec["xyz"][7], 0.0), "an undetected frame must be zero-filled, not dropped"
    assert rec["joints"] == 15 and rec["njts"] == 13 and rec["K"] == 5
    assert rec["xy"].shape == (30, J, 2), rec["xy"].shape
    assert rec["has_cumscore"] is False, "V1.2 carries no per-detection score"

    # V1.1 shape: same schema plus `cumscore`, which exists in ONE archive only. A confidence
    # rule that silently applies to half a corpus is a distribution shift, so its presence is
    # reported rather than assumed either way.
    v11 = read_pose3d(write(
        {"K": 20, "njts": J,
         "frames": [[{"cumscore": 2.97, "pose2d": flat2d, "pose3d": flat3d}] for _ in range(5)]},
        name="Drink.Fromcup_p20_r02_v02_c05.json"))
    assert v11["has_cumscore"] is True and v11["K"] == 20

    # An INCONCLUSIVE check must not reject the file. A seated subject (knees at hip height,
    # ankles above knees) scores near zero under BOTH readings, so anatomy says nothing about the
    # format - and rejecting on the bare winner threw out 295 of 16,115 real clips that way,
    # concentrated in counter activities with occluded legs.
    seated_y = dict(ys, right_ankle=250, left_ankle=248, right_knee=240, left_knee=238)
    sxy = np.array([[xs[n] for n in LCRNET_JOINTS], [seated_y[n] for n in LCRNET_JOINTS]],
                   dtype=np.float32)
    sflat2 = sxy.reshape(-1).tolist()
    sflat3 = np.vstack([sxy, np.full((1, J), 2.5, np.float32)]).reshape(-1).tolist()
    sc = verify_lcrnet_layout([sflat2] * 10)
    assert sc["decisive"] is False, sc
    kept = read_pose3d(write({"K": 5, "njts": J,
                              "frames": [[{"pose2d": sflat2, "pose3d": sflat3}]] * 10},
                             name="Cook.Cut_p15_r02_v15_c03_pose3d.json"))
    assert kept["layout_decisive"] is False and kept["n_frames"] == 10, kept["layout_decisive"]

    # A scrambled FILE is read, not rejected - its verdict is recorded for audit instead. The
    # rejection belongs at corpus level (T27): a transposition is a property of the release, and
    # testing it per file threw out 73 ordinary clips.
    scr = read_pose3d(write({"K": 5, "njts": J, "frames": [
        [{"pose2d": xy.T.reshape(-1).tolist(),
          "pose3d": xyz.T.reshape(-1).tolist()}] for _ in range(30)]},
        name="scrambled_pose3d.json"))
    assert scr["n_frames"] == 30 and scr["layout_decisive"] is True, scr["layout_decisive"]
    assert scr["layout_scores"][1] > scr["layout_scores"][0], scr["layout_scores"]

    # A different joint count is a data-format change, not a reshape.
    try:
        read_pose3d(write({"K": 5, "njts": 17, "frames": [[{"pose2d": flat2d,
                                                           "pose3d": flat3d}]]}))
        raise AssertionError("njts=17 was accepted against a 13-joint table")
    except ValueError as exc:
        assert "njts=17" in str(exc), str(exc)

    # The midpoints are INRIA's, derived from NAMED joints, and only derivable once.
    d = derive_agcn_joints(np.stack([xy.T, xy.T]))
    exp_neck = (xy.T[LCRNET_JOINTS.index("right_shoulder")]
                + xy.T[LCRNET_JOINTS.index("left_shoulder")]) / 2
    assert np.allclose(d[0, AGCN_JOINTS.index("neck")], exp_neck), d[0, 13]
    try:
        derive_agcn_joints(d)
        raise AssertionError("a 15-joint array was accepted for derivation")
    except ValueError as exc:
        assert "13" in str(exc), str(exc)

    print(f"  T22 measured schema (K/njts/frames, njts=13, pose2d 26 + pose3d 39 flat) parses "
          f"for V1.1/V1.2/untrimmed; coordinate-major confirmed anatomically "
          f"({check['coordinate_major_score']} vs {check['joint_major_score']}); joint-major "
          f"file and njts=17 both REFUSED; undetected frame kept zero-filled with "
          f"present=False; cumscore present in V1.1 only; a seated clip is INCONCLUSIVE "
          f"({sc['coordinate_major_score']} vs {sc['joint_major_score']}) and kept, not rejected")


def test_t23_coverage_gate_rejects_the_videos_where_the_pose_estimator_failed():
    """A per-VIDEO gate, because the corpus average hides the failure.

    Measured 2026-08-26 over 120 untrimmed videos: 15.30% of 3,042,437 frames carry no
    detection. That reads like tolerable noise. Per video it is bimodal - 0.0%, 1.1%, 1.8%,
    5.6%, then 62.1% and 99.6%.

    `results_P09T14C07` is 17,385 empty frames out of 17,457. Undetected frames are zero-filled
    so frame indices stay aligned with the annotations, so that video contributes ~17k windows
    of all-zero skeletons carrying real activity labels - it teaches the model that `Cook` looks
    like nothing. A global tolerance cannot express that; only a per-video decision can.

    The threshold is asserted to sit in the GAP between the two modes rather than at a round
    number that happens to work, because a cut inside a continuum is not defensible and this
    one has to survive a reviewer asking why it is 0.60.
    """
    from behaviorsense.data.toyota import MIN_POSE_COVERAGE, coverage_report

    def present(n_frames, n_missing):
        return np.r_[np.zeros(n_missing, bool), np.ones(n_frames - n_missing, bool)]

    # The real per-file numbers from the verification run.
    measured = [
        ("results_P20T01C05", 29473, 0, True),
        ("results_P13T24C04", 26367, 284, True),
        ("results_P17T07C02", 19143, 347, True),
        ("results_P03T18C03", 14204, 797, True),
        ("results_P25T05C05", 32004, 19866, False),
        ("results_P09T14C07", 17457, 17385, False),
    ]
    for name, frames, missing, want in measured:
        rep = coverage_report(present(frames, missing))
        assert rep["usable"] is want, (
            f"{name}: coverage {rep['coverage']:.3f} -> usable={rep['usable']}, wanted {want}")
        assert rep["n_frames"] == frames and rep["n_detected"] == frames - missing
        if not want:
            assert rep["reason"], "a dropped video must carry the reason, not just False"
            assert f"{MIN_POSE_COVERAGE:.0%}" in rep["reason"], rep["reason"]

    kept = [1 - m / f for _n, f, m, w in measured if w]
    cut = [1 - m / f for _n, f, m, w in measured if not w]
    assert MIN_POSE_COVERAGE < min(kept) and MIN_POSE_COVERAGE > max(cut), (
        f"the threshold {MIN_POSE_COVERAGE} does not sit in the empty gap between the modes "
        f"(worst kept {min(kept):.3f}, best cut {max(cut):.3f}) - a cut inside a continuum is "
        "not defensible")

    # The longest blind stretch is reported beside the average, because a video can pass on
    # average and still be blind through the one activity that matters.
    gappy = np.r_[np.ones(5000, bool), np.zeros(3000, bool), np.ones(5000, bool)]
    rep = coverage_report(gappy)
    assert rep["usable"] and rep["longest_gap"] == 3000, rep
    assert coverage_report(np.zeros(0, bool))["reason"] == "no frames"

    print(f"  T23 gate at {MIN_POSE_COVERAGE:.0%} keeps {len(kept)}/{len(measured)} measured "
          f"videos (worst kept {min(kept):.1%}) and drops {len(cut)} (best cut {max(cut):.1%}); "
          f"threshold sits in the empty gap between the modes; a 3,000-frame blind stretch is "
          f"reported even when the average passes")


def test_t24_skeleton_clips_are_a_valid_tree_resampled_by_proportion():
    """The shared half of INRIA's skeleton pipeline: pose records -> fixed-length clips.

    The bone tree is what this test exists for. `bone_stream` computes `child - parent`, so a
    pair written the other way round produces an inverted bone AND leaves the real child with
    none. Written as an anatomical edge list, the first version of `SKELETON_BONES` was wrong on
    eight of fourteen bones: `head` and both wrists came out identically zero while `neck` was
    overwritten three times and the root got a bone it should not have. Nothing raised - it would
    simply have trained on a body whose arms do not exist.

    Naming the joints did not prevent that. Only the structural property does, so it is asserted
    here as well as at import: every non-root joint has exactly one parent, and the root has none.

    Two other properties matter as much:

    - **Resampling preserves PROPORTION, not rate.** Trimmed clips run 35 to 1,170 frames. A
      stride can only produce rates that divide T, so it changes how much of the action a clip
      covers - the defect behind notebook 07's 10 Hz corpus and serving's 3-second windows.
    - **Gaps are bridged, not zero-filled.** `read_pose3d` zero-fills to keep annotation indices
      aligned; a zero skeleton fed to a graph net is a body collapsed at the origin and reads as
      violent motion.
    """
    from behaviorsense.data.toyota import AGCN_JOINTS
    from behaviorsense.data.toyota_skeleton import (
        BONE_EDGES,
        CLIP_FRAMES,
        ROOT_JOINT,
        SKELETON_BONES,
        bone_adjacency,
        bone_stream,
        bridge_missing,
        build_clip,
        centre_on_root,
        resample_time,
    )

    # 1. A tree: 14 bones over 15 joints, each non-root joint parented exactly once.
    children = [c for c, _ in SKELETON_BONES]
    assert len(children) == len(set(children)) == len(AGCN_JOINTS) - 1, children
    assert AGCN_JOINTS[ROOT_JOINT] == "mid_hip"
    assert set(children) == set(AGCN_JOINTS) - {"mid_hip"}, (
        f"unparented: {sorted(set(AGCN_JOINTS) - {'mid_hip'} - set(children))}")

    A = bone_adjacency()
    assert np.allclose(A, A.T), "adjacency must be symmetric"
    assert np.all(np.diag(A) == 1), "self-loops missing: a node could not see its own features"
    assert int((A.sum() - len(AGCN_JOINTS)) / 2) == len(BONE_EDGES)

    # 2. Every non-root joint gets a REAL bone; only the root's is zero. This is the assertion
    # the broken version failed - three leaves were silently all-zero.
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 0.3, (20, len(AGCN_JOINTS), 3)).astype(np.float32)
    b = bone_stream(x)
    zero = [AGCN_JOINTS[j] for j in range(len(AGCN_JOINTS)) if np.allclose(b[:, j], 0.0)]
    assert zero == ["mid_hip"], f"joints with no bone: {zero}"
    for name, parent in (("right_ankle", "right_knee"), ("head", "neck"),
                         ("left_wrist", "left_elbow")):
        i, p = AGCN_JOINTS.index(name), AGCN_JOINTS.index(parent)
        assert np.allclose(b[:, i], x[:, i] - x[:, p]), f"{name} bone is not {name}-{parent}"

    # Bones are translation-invariant; joints are not. That is why two streams beat one.
    assert np.allclose(b, bone_stream(x + np.float32([5.0, -2.0, 1.0])), atol=1e-5)
    assert np.allclose(centre_on_root(x)[:, ROOT_JOINT], 0.0, atol=1e-6)

    # 3. Proportion, not rate: the first and last frame of the action survive every input length.
    for t in (35, 100, 1170):
        ramp = np.zeros((t, len(AGCN_JOINTS), 3), np.float32)
        ramp[:, 0, 0] = np.linspace(0.0, 1.0, t)
        r = resample_time(ramp, CLIP_FRAMES)
        assert r.shape == (CLIP_FRAMES, len(AGCN_JOINTS), 3), r.shape
        assert abs(r[0, 0, 0]) < 1e-5 and abs(r[-1, 0, 0] - 1.0) < 1e-5, (t, r[0, 0, 0])

    # 4. Bridging fills a gap from its neighbours instead of leaving a collapsed body.
    flat = np.tile(np.arange(len(AGCN_JOINTS), dtype=np.float32)[None, :, None], (10, 1, 3))
    flat[4:7] = 0.0
    present = np.ones(10, bool)
    present[4:7] = False
    filled, share = bridge_missing(flat, present)
    assert abs(share - 0.3) < 1e-6 and abs(filled[5, 7, 0] - 7.0) < 1e-5, (share, filled[5, 7, 0])
    assert bridge_missing(flat, np.zeros(10, bool))[1] == 1.0, "an empty clip must not be invented"

    # 5. The coverage gate returns None rather than raising - a corpus pass must report drops.
    full = {"xyz": x, "present": np.ones(20, bool)}
    clip = build_clip(full)
    assert clip["joint"].shape == (3, CLIP_FRAMES, len(AGCN_JOINTS)), clip["joint"].shape
    assert clip["bone"].shape == clip["joint"].shape and clip["normalised"] is True
    for share, keep in ((11 / 20, False), (13 / 20, True)):
        p = np.zeros(20, bool)
        p[:int(round(share * 20))] = True
        got = build_clip({"xyz": x, "present": p})
        assert (got is not None) is keep, f"coverage {share:.0%} -> kept={got is not None}"
    assert build_clip({"xyz": x, "present": np.zeros(20, bool)}) is None

    print(f"  T24 15-joint tree valid ({len(BONE_EDGES)} bones, root {AGCN_JOINTS[ROOT_JOINT]}, "
          f"only the root has no bone); bone stream translation-invariant; 35/100/1170-frame "
          f"clips all resample to {CLIP_FRAMES} preserving proportion; a 30% gap bridged from "
          f"neighbours; gate keeps 65% coverage and drops 55%")


def test_t25_the_graph_network_accepts_the_15_node_toyota_layout():
    """ST-GCN++ was hard-wired to COCO-17. Toyota's released poses are a 15-node tree.

    `build_adjacency` read `N_JOINTS` and `COCO_EDGES` from module scope and hard-coded
    `centre = 0`, so there was no way to build a graph for another layout - and `STGCNpp` sized
    `data_bn` from `N_JOINTS` while the blocks used whatever `A` it had just built. Both are now
    parameters, with the COCO defaults unchanged.

    The centre is worth noting: COCO-17 uses the NOSE because it has no spine joint at all, which
    makes "distance to body centre" - the thing the three-partition strategy is built on - a
    stand-in. Toyota's layout has a real pelvis, so the partitions mean what Yan et al. intended.

    The orphan check is the load-bearing assertion. A joint disconnected from the centre has
    infinite hop distance, falls into no partition, and therefore receives no messages from
    anywhere - it trains as a constant. Nothing about that is visible in a loss curve.
    """
    import torch

    from behaviorsense.data.toyota import AGCN_JOINTS
    from behaviorsense.data.toyota_skeleton import BONE_EDGES, CLIP_FRAMES, ROOT_JOINT
    from behaviorsense.models.stgcnpp import N_JOINTS, STGCNpp, build_adjacency

    # The COCO default must be untouched by the generalisation.
    A_coco = build_adjacency()
    assert A_coco.shape == (3, N_JOINTS, N_JOINTS), A_coco.shape

    V = len(AGCN_JOINTS)
    A = build_adjacency(edges=BONE_EDGES, n_nodes=V, centre=ROOT_JOINT)
    assert A.shape == (3, V, V), A.shape
    assert all(A[k].any() for k in range(3)), (
        "an empty partition means one of self/centripetal/centrifugal carries no edges, so the "
        "conv learns nothing for that direction")
    assert np.isfinite(A).all()

    model = STGCNpp(n_classes=len(TSM_CLASSES), n_joints=V, adjacency=A, n_person=1)
    # [N, C, T, V, M]: `build_clip` emits [C, T, V] and the person axis is appended.
    out = model(torch.randn(2, 3, CLIP_FRAMES, V, 1))
    assert out.shape == (2, len(TSM_CLASSES)), out.shape

    # A graph that disagrees with n_joints must be refused: data_bn would normalise over a
    # different feature count than the blocks consume, and the shapes still broadcast.
    try:
        STGCNpp(n_classes=31, n_joints=V, adjacency=A_coco, n_person=1)
        raise AssertionError("a 17-node adjacency was accepted for a 15-joint model")
    except ValueError as exc:
        assert "expected (3, 15, 15)" in str(exc), str(exc)

    # A disconnected joint must raise rather than train as a constant.
    try:
        build_adjacency(edges=BONE_EDGES[:-1], n_nodes=V, centre=ROOT_JOINT)
        raise AssertionError("an orphaned joint was accepted")
    except ValueError as exc:
        assert "not connected" in str(exc), str(exc)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  T25 15-node graph centred on {AGCN_JOINTS[ROOT_JOINT]} (COCO uses the nose as a "
          f"stand-in); 3 non-empty partitions; STGCNpp({V} joints, 1 person) forwards "
          f"[2,3,{CLIP_FRAMES},{V},1] -> {tuple(out.shape)} with {n_params:,} params; "
          f"mismatched adjacency and orphaned joints both refused")


def test_t26_shard_builder_runs_and_reports_every_rejection_by_reason():
    """EXECUTE the shard builder. A script that only prints a total cannot be audited.

    Written after `/video` 500'd on every upload while five key-comparison tests passed: a
    pipeline stage has to be RUN, not described. This drives the real script over a synthetic
    archive holding one clip of each interesting kind - normal, below the coverage gate, and an
    activity name no id space contains.

    The properties that matter are all about honesty of the report rather than throughput:

    - an unmapped activity is DROPPED AND NAMED, never defaulted. Charades was mapped through 157
      keyword rules whose fallback quietly absorbed a large part of the class list.
    - drops are counted BY REASON, so "did the gate do this or the label mapping?" has an answer
    - the CS split comes from the subject id and no subject may appear on both sides
    - classes with zero clips are named, so a per-class table cannot report them as scoring zero
    - the manifest is re-read FROM DISK, because a printed running total is a claim about memory
    """
    import json
    import pathlib
    import subprocess
    import tempfile

    from behaviorsense.data.toyota import LCRNET_JOINTS

    J = len(LCRNET_JOINTS)
    ys = {"right_ankle": 400, "left_ankle": 398, "right_knee": 320, "left_knee": 318,
          "right_hip": 250, "left_hip": 248, "right_wrist": 260, "left_wrist": 258,
          "right_elbow": 230, "left_elbow": 228, "right_shoulder": 180,
          "left_shoulder": 178, "head": 120}
    xs = {n: 300 + (5 if n.startswith("right") else -5) for n in ys}
    xy = np.array([[xs[n] for n in LCRNET_JOINTS], [ys[n] for n in LCRNET_JOINTS]], np.float32)
    xyz = np.vstack([xy / 500.0, np.full((1, J), 2.5, np.float32)])
    det = {"pose2d": xy.reshape(-1).tolist(), "pose3d": xyz.reshape(-1).tolist()}

    base = pathlib.Path(tempfile.mkdtemp())
    arch = base / "datasets" / "owner" / "toyota-smarthome-skeleton-v1-2"
    arch.mkdir(parents=True)
    train3, test2 = list(CS_TRAIN_SUBJECTS)[:3], list(CS_TEST_SUBJECTS)[:2]
    for act in ("Walk", "Sitdown", "Cook.Stir", "WatchTV"):
        for subj in train3 + test2:
            n = 40 + subj
            frames = [[det] for _ in range(n)]
            if act == "WatchTV":                     # 70% undetected: below the gate
                frames[: int(n * 0.7)] = [[] for _ in range(int(n * 0.7))]
            (arch / f"{act}_p{subj:02d}_r00_v01_c02_pose3d.json").write_text(
                json.dumps({"K": 5, "njts": J, "frames": frames}), encoding="utf-8")
    (arch / "Teleporting_p03_r00_v01_c02_pose3d.json").write_text(
        json.dumps({"K": 5, "njts": J, "frames": [[det]] * 50}), encoding="utf-8")

    out = base / "shards"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_toyota_skeleton_shards.py"),
         "--root", str(base), "--out", str(out)],
        capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-800:] + proc.stderr[-800:]

    man = json.loads(next(out.glob("*_manifest.json")).read_text(encoding="utf-8"))
    n_files = len(train3 + test2) * 4 + 1
    assert man["n_files_seen"] == n_files, (man["n_files_seen"], n_files)
    assert man["n_clips"] == len(train3 + test2) * 3, man["n_clips"]

    # The unmapped name is named, not absorbed.
    assert man["unknown_activities"] == {"Teleporting": 1}, man["unknown_activities"]
    assert man["drops"]["unknown_activity"] == 1
    assert man["drops"]["below_coverage_gate"] == len(train3 + test2), man["drops"]

    # The split is the official one, derived from subject ids, with no leak.
    assert man["subjects_by_split"]["train"] == sorted(train3), man["subjects_by_split"]
    assert man["subjects_by_split"]["test"] == sorted(test2), man["subjects_by_split"]
    assert not set(man["subjects_by_split"]["train"]) & set(man["subjects_by_split"]["test"])

    # Classes with no clips are named rather than left to score zero in a per-class table.
    assert "WatchTV" in man["classes_with_no_clips"], "the gated class must be reported absent"
    assert set(man["per_class"]) == set(TSM_CLASSES), "the manifest must cover the id space"
    assert sum(man["per_class"].values()) == man["n_clips"]

    # Only the joint stream is stored; the bone tree travels as data so it cannot drift.
    assert man["streams_stored"] == ["joint"] and man["joints"] == 15
    assert len(man["bone_parents"]) == 14, man["bone_parents"]

    # And the arrays on disk are the shape the model consumes.
    with np.load(next(out.glob("*_0000.npz")), allow_pickle=False) as z:
        assert z["joint"].shape[1:] == (3, man["clip_frames"], 15), z["joint"].shape
        assert z["joint"].dtype == np.float16, z["joint"].dtype
        assert len(z["labels"]) == len(z["subjects"]) == len(z["coverage"])

    print(f"  T26 builder ran on {n_files} synthetic clips: kept {man['n_clips']}, dropped "
          f"{sum(man['drops'].values())} by reason {dict(man['drops'])}; 'Teleporting' named "
          f"not defaulted; CS split {man['subjects_by_split']} with no leak; "
          f"{len(man['classes_with_no_clips'])} classes named as having no clips; "
          f"joint fp16 [N,3,{man['clip_frames']},15] on disk")


def test_t27_the_layout_verdict_is_corpus_level_not_per_file():
    """Where the flat-layout check belongs, established by two rounds of getting it wrong.

    A coordinate/joint transposition is a property of how a RELEASE was written, not of one clip.
    Testing it per file inherits every posture the estimator saw:

      * strict per-file rejection threw out **295 of 16,115** trimmed clips whose subject is
        simply not upright - concentrated in `Cook.Cut`, `Cutbread` and `Drink.From*`
      * adding a confidence gate recovered 222 and left **73** scoring "decisively joint-major"
        while being ordinary files. Those are the heuristic being fooled, not a mixed archive:
        reading coordinate-major 2D as joint-major builds "joints 0-5" from x-values alone and
        "joints 7-12" from y-values alone, which satisfies hips > shoulders > head structurally
        and leaves the ankle/knee links as coin flips on millimetre x-differences.

    So `read_pose3d` records and never rejects, and one corpus verdict guards everything. The
    threshold is a MAJORITY (90%), not unanimity, because 0.45% contamination would otherwise
    abort a good build whenever a sample happened to contain one of those files.

    Three failure modes must still stop a build, and all three are asserted: an archive written
    the other way round, too little evidence to judge at all, and - the negative control - the
    healthy case passing so the test cannot be vacuous.
    """
    import json
    import pathlib
    import tempfile

    from behaviorsense.data.toyota import (
        LCRNET_JOINTS,
        MIN_LAYOUT_AGREEMENT,
        assert_corpus_layout,
    )

    J = len(LCRNET_JOINTS)
    ys = {"right_ankle": 400, "left_ankle": 398, "right_knee": 320, "left_knee": 318,
          "right_hip": 250, "left_hip": 248, "right_wrist": 260, "left_wrist": 258,
          "right_elbow": 230, "left_elbow": 228, "right_shoulder": 180,
          "left_shoulder": 178, "head": 120}
    xs = {n: 300 + (5 if n.startswith("right") else -5) for n in ys}
    up = np.array([[xs[n] for n in LCRNET_JOINTS], [ys[n] for n in LCRNET_JOINTS]], np.float32)
    seated_y = dict(ys, right_ankle=250, left_ankle=248, right_knee=240, left_knee=238)
    seat = np.array([[xs[n] for n in LCRNET_JOINTS], [seated_y[n] for n in LCRNET_JOINTS]],
                    np.float32)

    def archive(n_up=0, n_seat=0, n_scrambled=0):
        d = pathlib.Path(tempfile.mkdtemp())
        def det(a):
            z = np.vstack([a, np.full((1, J), 2.5, np.float32)])
            return {"pose2d": a.reshape(-1).tolist(), "pose3d": z.reshape(-1).tolist()}
        items = ([det(up)] * n_up + [det(seat)] * n_seat
                 + [det(up.T.reshape(2, J))] * 0 + [{"pose2d": up.T.reshape(-1).tolist(),
                     "pose3d": np.vstack([up, np.full((1, J), 2.5, np.float32)]).T.reshape(
                         -1).tolist()}] * n_scrambled)
        out = []
        for i, dd in enumerate(items):
            f = d / f"Walk_p03_r{i:02d}_v01_c02_pose3d.json"
            f.write_text(json.dumps({"K": 5, "njts": J, "frames": [[dd]] * 20}), encoding="utf-8")
            out.append(f)
        return out

    # HEALTHY: mostly upright with a third seated, which is Toyota's real posture mix.
    ok = assert_corpus_layout(archive(n_up=40, n_seat=20))
    assert ok["assumed"] == "coordinate_major" and ok["agreement"] == 1.0, ok
    assert ok["n_decisive"] == 40, ok            # the seated files abstain rather than vote
    assert ok["n_read"] == 60, ok

    # A few fooled files must NOT abort the build - that is the whole point of a majority bar.
    tol = assert_corpus_layout(archive(n_up=40, n_seat=10, n_scrambled=2))
    assert tol["agreement"] >= MIN_LAYOUT_AGREEMENT, tol
    assert len(tol["dissenters"]) == 2, tol["dissenters"]

    # AN ARCHIVE WRITTEN THE OTHER WAY ROUND must stop everything.
    try:
        assert_corpus_layout(archive(n_scrambled=30))
        raise AssertionError("a joint-major archive was accepted; every skeleton would be built "
                             "from scrambled joints and would train without complaint")
    except ValueError as exc:
        assert "0.0%" in str(exc) and "scrambled joints" in str(exc), str(exc)

    # TOO LITTLE EVIDENCE must also stop it, rather than asserting the layout on nothing.
    try:
        assert_corpus_layout(archive(n_seat=30))
        raise AssertionError("the layout was asserted with no decisive file to support it")
    except ValueError as exc:
        assert "decisive layout verdict" in str(exc), str(exc)

    print(f"  T27 corpus verdict from {ok['n_decisive']}/{ok['n_read']} decisive files "
          f"({ok['agreement']:.0%} agree, bar {MIN_LAYOUT_AGREEMENT:.0%}); 2 fooled files "
          f"tolerated; a joint-major archive and an all-seated archive both abort the build")


def test_t28_the_training_loop_learns_and_reports_support_beside_every_number():
    """EXECUTE the training script on a tiny learnable set. Two epochs, seconds.

    Not a check that accuracy is good - it is synthetic data - but that the loop is wired: loss
    reaches the optimiser, the graph accepts 15 joints, the stream is derived on the fly, and the
    metrics land on disk. A training script that runs and learns nothing looks identical to one
    that runs and learns, until a 10-hour session has been spent.

    Two reporting properties are asserted because they are the ones that make a number quotable:

    - **mean per-class is recorded alongside top-1.** On an 88x imbalance they disagree by
      construction, and a single headline would be a silent choice about which classes matter.
    - **every per-class accuracy carries its test support.** Over 3 clips, accuracy is quantised
      to steps of 33 points; that has to sit beside the value, not be inferred afterwards.
    """
    import json
    import pathlib
    import subprocess
    import tempfile

    rng = np.random.default_rng(0)
    V, T, C = 15, 64, 3
    # Each class gets its own constant joint offset, so a working loop must beat chance. One
    # class deliberately has 3 clips, to exercise the thin-support path.
    per = {0: 30, 1: 30, 2: 30, 3: 6}
    joints, labels, subjects, splits = [], [], [], []
    for cls, n in per.items():
        for k in range(n):
            base = np.zeros((C, T, V), np.float32)
            base[:, :, cls % V] = cls + 1
            joints.append(base + rng.normal(0, 0.2, (C, T, V)).astype(np.float32))
            labels.append(cls)
            is_train = k % 3 != 0
            subjects.append(int(CS_TRAIN_SUBJECTS[k % 11] if is_train
                                else CS_TEST_SUBJECTS[k % 7]))
            splits.append("train" if is_train else "test")

    tmp = pathlib.Path(tempfile.mkdtemp())
    shards = tmp / "shards"
    shards.mkdir()
    np.savez_compressed(
        shards / "tsm_v12_64f_0000.npz",
        joint=np.stack(joints).astype(np.float16),
        labels=np.asarray(labels, np.int16), subjects=np.asarray(subjects, np.int16),
        split=np.asarray(splits), clips=np.asarray([f"c{i}.json" for i in range(len(labels))]),
        coverage=np.ones(len(labels), np.float32),
        bridged_share=np.zeros(len(labels), np.float32))

    out = tmp / "run"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_tsm_skeleton.py"),
         "--shards", str(shards), "--out", str(out), "--stream", "joint",
         "--epochs", "2", "--batch", "16"],
        capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-1200:] + proc.stderr[-1200:]

    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert m["protocol"] == "CS" and m["joints"] == 15 and m["clip_frames"] == T
    assert m["class_balance"].startswith("effective_number"), m["class_balance"]
    assert not set(m["train_subjects"]) & set(m["test_subjects"]), "CS leak"

    # It learned something: 4 classes means chance is 0.25, and a broken loop sits at chance.
    assert m["mean_class"] > 0.30, f"mean per-class {m['mean_class']} is at or below chance"
    assert m["top1"] > 0.30, m["top1"]
    assert m["mean_class"] != m["top1"], (
        "identical top-1 and mean per-class on an imbalanced split means one of them is not "
        "being computed as claimed")

    # Support travels with every per-class number, and thin classes are named.
    assert set(m["per_class"]) <= set(TSM_CLASSES)
    for name, rec in m["per_class"].items():
        assert rec["support"] > 0 and 0.0 <= rec["acc"] <= 1.0, (name, rec)
    assert sum(r["support"] for r in m["per_class"].values()) == m["n_test"]
    assert m["thin_test_classes"], "a 6-clip class must be reported as thin"
    assert "granularity" in proc.stdout, "the per-class table must state its own granularity"

    # The checkpoint is selected on mean per-class, not top-1: top-1 would pick the epoch that
    # best fits the majority class, which is the opposite of what the benchmark measures.
    ck = (out / "best.pt")
    assert ck.is_file() and ck.stat().st_size > 1000
    assert m["best_epoch"] == max(m["history"], key=lambda h: h["mean_class"])["epoch"]

    print(f"  T28 trained 2 epochs on {m['n_train']} synthetic clips: top-1 {m['top1']:.3f}, "
          f"mean per-class {m['mean_class']:.3f} over {m['n_classes_scored']} classes (chance "
          f"0.25); support printed per class, {len(m['thin_test_classes'])} thin class(es) "
          f"named; checkpoint selected on mean per-class at epoch {m['best_epoch']}")


def test_t29_post_hoc_sweep_is_wired_and_temperature_cannot_move_a_fine_argmax():
    """The free lever: sweep tau and T over saved logits instead of retraining per point.

    The load-bearing assertion is an INVARIANCE, and the script checks it on itself. Temperature
    divides the logits, scaling is monotone, so it cannot change the fine argmax - every fine
    number in a tau row must be bit-identical across T. If it varies, T is being applied
    somewhere it should not and none of the coarse numbers mean anything either. A sweep that
    silently reports a temperature effect on an argmax is worse than no sweep, because the numbers
    look like a finding.

    Temperature DOES matter to coarse pooling, which is the whole reason to sweep it here:
    `logsumexp(z/T)` tends to `max(z)/T` as T falls, so low T collapses pooling onto the argmax
    rule and the unearned `log(k)` bonus a large family gets disappears. That is the specific test
    of whether the -0.145 pooling loss on the real checkpoint is a calibration artefact.
    """
    import json
    import pathlib
    import subprocess
    import tempfile

    from behaviorsense.data.toyota import TSM_TO_COARSE, coarse_id

    rng = np.random.default_rng(0)
    K = len(TSM_CLASSES)
    n = 900
    f2c = np.array([coarse_id(TSM_TO_COARSE[c]) for c in TSM_CLASSES])
    # Long-tailed labels and logits that know the FAMILY but are confused inside it - the real
    # model's failure mode, so the sweep is exercised on the shape it exists for.
    p = np.array([0.24 if c == "Walk" else (0.14 if c == "Drink.Fromcup" else 0.62 / 29)
                  for c in TSM_CLASSES])
    p = p / p.sum()
    y = rng.choice(K, size=n, p=p)
    z = rng.normal(0.0, 1.0, (n, K))
    for i in range(n):
        z[i, f2c == f2c[y[i]]] += 2.2
        z[i, y[i]] += 0.5
        z[i] += 1.4 * np.log(p)                 # head bias, which tau exists to undo

    tmp = pathlib.Path(tempfile.mkdtemp())
    np.savez_compressed(tmp / "test_logits.npz", logits=z.astype(np.float32),
                        targets=y.astype(np.int16), classes=np.asarray(TSM_CLASSES))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "rescore_tsm.py"),
         "--logits", str(tmp / "test_logits.npz")],
        capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-900:] + proc.stderr[-900:]

    rep = json.loads((tmp / "rescore.json").read_text(encoding="utf-8"))
    grid = rep["grid"]
    taus = sorted({r["tau"] for r in grid})
    temps = sorted({r["T"] for r in grid})
    assert len(grid) == len(taus) * len(temps), (len(grid), len(taus), len(temps))
    assert rep["n_fine"] == K and rep["n_coarse"] == len(COARSE_V11)

    # THE INVARIANCE. Also asserted inside the script, and again here so a regression there
    # cannot pass silently.
    for tau in taus:
        vals = {r["fine_mean_class"] for r in grid if r["tau"] == tau}
        assert len(vals) == 1, f"fine mean-class moved with T at tau={tau}: {sorted(vals)}"

    # tau must actually do something, or the lever is being reported without being applied.
    by_tau = {r["tau"]: r["fine_mean_class"] for r in grid if r["T"] == 1.0}
    assert len(set(by_tau.values())) > 1, f"tau had no effect at all: {by_tau}"
    assert max(by_tau.values()) > by_tau[0.0], (
        f"no tau beat the unadjusted baseline on a 110x-imbalanced set: {by_tau}")

    # Low temperature must pull pooling TOWARD the argmax rule - that is the mechanism.
    lo = min(grid, key=lambda r: (r["T"], r["tau"]))
    assert abs(lo["coarse_lse_mean_class"] - lo["coarse_argmax_mean_class"]) < 0.05, (
        f"at T={lo['T']} logsumexp should approach max and therefore the argmax collapse, "
        f"got {lo['coarse_lse_mean_class']} vs {lo['coarse_argmax_mean_class']}")

    # The prior must come from predictions, not from the test labels - otherwise the adjustment
    # is fitted on the split it is scored on.
    assert "not test labels" in rep["prior_source"], rep["prior_source"]

    print(f"  T29 swept {len(taus)}x{len(temps)} = {len(grid)} points on {rep['n_test']:,} "
          f"synthetic clips: fine mean-class invariant to T at every tau (asserted twice), "
          f"tau moves it {by_tau[0.0]:.3f} -> {max(by_tau.values()):.3f}, low T pulls pooling "
          f"onto the argmax rule; logsumexp ever wins = {rep['logsumexp_ever_wins']}")


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
