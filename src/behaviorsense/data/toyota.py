"""Toyota Smarthome ingestion — the corpus this project should have been training on.

Everything in this module was read out of the official artefacts, not remembered:
`pipline/data/Action_list` and `pipline/data/smarthome_CS_51.json` from
`dairui01/Toyota_Smarthome`, and `data_gen/smarthome_gendata.py` from
`YangDi666/2s-AGCN-For-Daily-Living` (both linked from the dataset's own project page).
The two repositories agree on the cross-subject partition, which is why it is asserted
below rather than trusted.

**Why this corpus replaces Charades for the ADL half of the label space.** Charades gave
us actor-disjoint macro-F1 0.128 across 20 classes, and the diagnosis was never the model:
crowdsourced clips of young adults acting scripted prompts to a handheld camera, mapped to
our taxonomy through 157 hand-written keyword rules whose fallback swallowed a large part
of the class list. Toyota Smarthome is the same *task* recorded properly — 18 subjects aged
60-80, seven fixed cameras in one apartment, 640x480, which is the geometry a CCTV install
actually has. Its untrimmed half carries dense per-frame multi-label annotation, so the
segment decoder finally has real supervision and a published metric instead of an internal
fragmentation ratio.

**What it does NOT carry, stated up front.** There are no falls: it is an ADL corpus, so
the fall head stays on the fall corpora and its transfer numbers stand as measured. And
there are no *sustained posture* labels — `Sit_down`, `Get_up` and `Lay_down` are annotated
as transitions, while the sitting and standing that fill the gaps between them are not
annotated at all. Treating those gaps as our `other_idle` would actively teach the model to
call sitting "other", which is why `frame_mask()` exists next to `frame_multilabel()` and
why the unified head has to be trained with a partial-label mask.

Units are the trap here and they have already been mis-read once upstream. `duration` and
every `[class, start, end]` triple in the annotation JSON are **frame counts at 25 Hz**,
not seconds - the untrimmed videos ship at x1.25 speed and the dataset README says to
deframe at 25. The official loader computes `fps = num_features / duration`, which is
features-per-frame and not a frame rate at all; the name has misled readers into treating
the annotation bounds as seconds, which silently compresses every label to 1/25th of its
span.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

# --- the corpus, as the dataset defines it ------------------------------------------------

TSU_FPS = 25.0
"""Deframe rate mandated by the dataset README, because the videos are stored at x1.25."""

TSM_FPS = 20.0
"""The TRIMMED clips' container frame rate, MEASURED, and it contradicts the documentation.

`srijandas07/i3d_smarthome` says the trimmed videos "are decoded into frames at 30 fps".
The shipped mp4s disagree: two probed at random on Kaggle both report **20 fps**
(`Walk_p03_r01_v15_c07.mp4`, 163 frames; `Sitdown_p20_r07_v09_c01.mp4`, 53 frames). At 20 fps
that Walk clip is 8.2 s, at 30 fps it is 5.4 s - and 8 s is the plausible one for a walk.

The container wins over the README here, but neither is trusted blindly: `CAP_PROP_FPS` is
known to lie on some encoders, and `video.py` already carries a comment saying so. Anything
that converts trimmed frames to seconds must call `probe_fps()` on the actual file and
compare; this constant is the expected value that a mismatch is reported against, not a
substitute for reading the file.

The untrimmed half is a separate 25 (see `TSU_FPS`) and the two must never share a constant.
"""

TSM_FPS_DOCUMENTED = 30.0
"""What `i3d_smarthome`'s README claims. Kept so the conflict is visible in code rather than
rediscovered by whoever next wonders why a duration is 1.5x off."""

TSU_CLASSES: tuple[str, ...] = (
    "Enter", "Walk", "Make_coffee", "Get_water", "Make_coffee.Pour_water", "Use_Drawer",
    "Make_coffee.Pour_grains", "Use_telephone", "Leave", "Put_something_on_table",
    "Take_something_off_table", "Pour.From_kettle", "Stir_coffee/tea", "Drink.From_cup",
    "Dump_in_trash", "Make_tea", "Make_tea.Boil_water", "Use_cupboard", "Insert_tea_bag",
    "Read", "Take_pills", "Use_fridge", "Clean_dishes",
    "Clean_dishes.Put_something_in_sink", "Eat_snack", "Sit_down", "Watch_TV",
    "Use_laptop", "Get_up", "Drink.From_bottle", "Pour.From_bottle", "Drink.From_glass",
    "Lay_down", "Drink.From_can", "Write", "Breakfast",
    "Breakfast.Spread_jam_or_butter", "Breakfast.Cut_bread", "Breakfast.Eat_at_table",
    "Breakfast.Take_ham", "Clean_dishes.Dry_up", "Wipe_table", "Cook", "Cook.Cut",
    "Cook.Use_stove", "Cook.Stir", "Cook.Use_oven", "Clean_dishes.Clean_with_water",
    "Use_tablet", "Use_glasses", "Pour.From_can",
)
"""The 51 untrimmed classes, index == official id. Order is `Action_list`'s, verbatim.

Order is load-bearing and is NOT alphabetical or grouped: the annotation JSON stores bare
integers, so re-sorting this tuple relabels the entire corpus. `Stir_coffee/tea` contains a
slash on purpose - it is the official name and `slug()` is what makes it path-safe.
"""

TSM_CLASSES: tuple[str, ...] = (
    "Cook.Cleandishes", "Cook.Cleanup", "Cook.Cut", "Cook.Stir", "Cook.Usestove",
    "Cutbread", "Drink.Frombottle", "Drink.Fromcan", "Drink.Fromcup", "Drink.Fromglass",
    "Eat.Attable", "Eat.Snack", "Enter", "Getup", "Laydown", "Leave",
    "Makecoffee.Pourgrains", "Makecoffee.Pourwater", "Maketea.Boilwater",
    "Maketea.Insertteabag", "Pour.Frombottle", "Pour.Fromcan", "Pour.Fromkettle",
    "Readbook", "Sitdown", "Takepills", "Uselaptop", "Usetelephone", "Usetablet",
    "Walk", "WatchTV",
)
"""The 31 trimmed classes, index == the id in `smarthome_gendata.py`'s `action_classes`.

Spelled differently from `TSU_CLASSES` for the same activities (`Readbook` vs `Read`,
`Takepills` vs `Take_pills`) because the two halves of the dataset were annotated
separately. Both spellings therefore need their own entry in the taxonomy map; folding
them together by normalising case and punctuation would be one regex away from silently
mapping a class that does not exist.
"""

TSM_CV_CLASSES: tuple[str, ...] = (
    "Cutbread", "Drink.Frombottle", "Drink.Fromcan", "Drink.Fromcup", "Drink.Fromglass",
    "Eat.Attable", "Eat.Snack", "Enter", "Getup", "Leave", "Pour.Frombottle",
    "Pour.Fromcan", "Readbook", "Sitdown", "Takepills", "Uselaptop", "Usetablet",
    "Usetelephone", "Walk",
)
"""The 19 classes the cross-view protocols use. Not a prefix of `TSM_CLASSES` - a
cross-view run reports over its own id space, and mixing the two is a silent relabel."""

TSM_SKELETON_JOINTS = 15
"""13 LCR-Net joints plus two derived midpoints (neck, mid-hip), as `read_xyz` builds them.

Recorded so the joint-count mismatch with our COCO-17 pipeline is explicit rather than
discovered at training time. We do not consume these skeletons as the model's input - see
the module docstring's note on train/serve skew - but the count is needed to read them at
all, for the pose-agreement check.
"""

# --- protocols ---------------------------------------------------------------------------

ALL_SUBJECTS: tuple[int, ...] = (2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                19, 20, 25)
"""The 18 subject ids actually present. Not 1..18: the numbering has gaps (no 1, 5, 8,
21-24), so `range(1, 19)` looks right and quietly drops p25 while inventing p1."""

CS_TRAIN_SUBJECTS: tuple[int, ...] = (3, 4, 6, 7, 9, 12, 13, 15, 17, 19, 25)
CS_TEST_SUBJECTS: tuple[int, ...] = (2, 10, 11, 14, 16, 18, 20)
"""The official cross-subject partition: 11 train / 7 test.

Cross-checked two ways before being written down. `training_subjects` in the trimmed
repo's `smarthome_gendata.py` lists exactly these 11 ids, and grouping
`smarthome_CS_51.json` by its own `subset` field yields the same 11 for `training` and
these 7 for `testing` (351 / 185 videos). The trimmed and untrimmed halves therefore share
one subject partition, which is what makes a model trained on one measurable on the other
without leaking a person across the boundary.
"""

CV1_TRAIN_CAMERAS: tuple[int, ...] = (1,)
CV1_VAL_CAMERAS: tuple[int, ...] = (5,)
CV1_TEST_CAMERAS: tuple[int, ...] = (2,)
CV2_TRAIN_CAMERAS: tuple[int, ...] = (1, 3, 4, 6, 7)
CV2_VAL_CAMERAS: tuple[int, ...] = (5,)
CV2_TEST_CAMERAS: tuple[int, ...] = (2,)
"""Cross-view: train on one or five cameras, validate on c05, test on c02.

`smarthome_gendata.py` implements CV2's train side as
`istraining = (subject_id in training_cameras2)` - a subject id tested against a camera
list. That is an upstream bug, it silently selects subjects 1/3/4/6/7 instead of cameras,
and any CV2 number produced with it is not the protocol it claims. We implement the
protocol as documented and keep the camera comparison on camera ids.
"""

PROTOCOLS = ("CS", "CV1", "CV2")


# --- names, ids, filenames ----------------------------------------------------------------

def slug(name: str) -> str:
    """Official class name -> path- and YAML-safe key. `Stir_coffee/tea` -> `stir_coffee_tea`."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def tsu_id(name: str) -> int:
    """Class name -> official untrimmed id, raising rather than returning -1."""
    try:
        return TSU_CLASSES.index(name)
    except ValueError:
        raise KeyError(f"{name!r} is not one of the 51 TSU classes") from None


# The trimmed half has TWO official id spaces for the same 31 class names, and they differ
# by one. This is the single most dangerous thing in the corpus.
#
#   YangDi666/2s-AGCN-For-Daily-Living  data_gen/smarthome_gendata.py `action_classes`
#       Cook.Cleandishes: 0  ...  WatchTV: 30        (0-indexed, 31 values)
#   srijandas07/i3d_smarthome           _name_to_int.py
#       Cook.Cleandishes: 1  ...  WatchTV: 31        (1-indexed, 0 = unmatched name)
#
# Both are official, both are linked from the dataset's own project page - one for the
# skeleton baseline, one for the RGB baseline. Mixing a label file from one with the id map
# from the other shifts EVERY class by one and leaves a silent catch-all at 0, which does not
# crash, does not look wrong, and produces a plausible accuracy that is measuring nothing.
# The cross-view maps are 1-indexed the same way over their own 19 names.
#
# `TSM_CLASSES` is canonical here and is 0-indexed. Anything reading a released RGB label
# file or the released `rgb_SH_CS.pt` head must go through `tsm_id(..., space="rgb")`.
TSM_ID_SPACES = ("skeleton", "rgb")
TSM_RGB_OFFSET = 1
TSM_RGB_UNMATCHED = 0
"""`_name_to_int` initialises `integer = 0` and falls through, so an unrecognised class name
returns 0 instead of raising. In the RGB space 0 therefore means "no name matched", not a
class - which is why `tsm_id` raises on an unknown name rather than reproducing that."""


def tsm_id(name: str, *, space: str = "skeleton", cross_view: bool = False) -> int:
    """Trimmed class name -> id in the requested official space.

    `space="skeleton"` is 0-indexed (the 2s-AGCN generator). `space="rgb"` is 1-indexed (the
    I3D repo's `_name_to_int`). Always name the space at the call site; the default exists so
    the canonical tuple index and the default agree, not so the choice can be skipped.
    """
    if space not in TSM_ID_SPACES:
        raise ValueError(f"space must be one of {TSM_ID_SPACES}, got {space!r}")
    table = TSM_CV_CLASSES if cross_view else TSM_CLASSES
    try:
        base = table.index(name)
    except ValueError:
        raise KeyError(
            f"{name!r} is not one of the {len(table)} trimmed "
            f"{'cross-view' if cross_view else 'cross-subject'} classes"
        ) from None
    return base + (TSM_RGB_OFFSET if space == "rgb" else 0)


def tsm_name(idx: int, *, space: str = "skeleton", cross_view: bool = False) -> str:
    """Inverse of `tsm_id`, refusing the RGB space's unmatched-name sentinel."""
    table = TSM_CV_CLASSES if cross_view else TSM_CLASSES
    if space == "rgb":
        if idx == TSM_RGB_UNMATCHED:
            raise KeyError(
                "id 0 in the RGB space is `_name_to_int`'s unmatched-name fallback, not a "
                "class. A 0 in a released RGB label file means the name was not recognised."
            )
        idx -= TSM_RGB_OFFSET
    if not 0 <= idx < len(table):
        raise KeyError(f"id {idx} outside [0,{len(table)}) for space={space!r}")
    return table[idx]


_TSU_NAME = re.compile(r"^P(\d{2})T(\d{2})C(\d{2})$")
_TSU_IN_FILENAME = re.compile(r"P(\d{2})T(\d{2})C(\d{2})")
_TSM_NAME = re.compile(
    r"^(?P<activity>[A-Za-z.]+)_p(?P<subject>\d{2})_r(?P<take>\d{2})_"
    r"(?P<view>[a-z]?\d{2})_c(?P<camera>\d{2})"
    # Derived files append a tag after the camera field, and the released archives use at
    # least two: `Walk_p25_r12_v15_c06_pose3d.json` in the refined-skeleton set and
    # `..._LCRNet2d.npz` in the visualisation tools. An anchored regex without this group
    # rejects the entire V1.2 skeleton archive - the one modality whose whole purpose is to
    # be the occlusion-robust reference - and the symptom is "0 files matched", which reads
    # as a bad mount rather than a bad pattern.
    r"(?:_(?P<tag>[A-Za-z0-9+.]+))?$"
)


@dataclass(frozen=True)
class VideoId:
    """A parsed video identifier. `take` is the recording session, not a repetition index."""

    subject: int
    take: int
    camera: int
    activity: str | None = None      # trimmed clips name their class; untrimmed do not
    tag: str | None = None           # derived-file suffix, e.g. `pose3d`, `LCRNet2d`

    @property
    def tsu(self) -> str:
        """Canonical untrimmed id, `P02T02C03`. Zero-padded, because `P2T2C3` matches nothing."""
        return f"P{self.subject:02d}T{self.take:02d}C{self.camera:02d}"


def parse_tsu_name(vid: str) -> VideoId:
    """`P11T15C01` -> VideoId(subject=11, take=15, camera=1).

    Strict by design. A silently-unparsed id would be assigned to no protocol side and
    dropped from both train and test, which reads as a smaller dataset rather than as a
    bug.
    """
    m = _TSU_NAME.match(vid.strip())
    if not m:
        raise ValueError(f"not a TSU video id: {vid!r} (expected P<dd>T<dd>C<dd>)")
    return VideoId(subject=int(m.group(1)), take=int(m.group(2)), camera=int(m.group(3)))


def parse_tsu_filename(path: str | Path) -> VideoId:
    """Pull the video id out of any untrimmed filename, however it has been decorated.

    The released archives decorate the same id three different ways, and all three are in
    the Kaggle mounts right now:

        Annotation/P18/P18T13C07.csv                    bare
        Videos_mp4/P15T17C03.mp4                        bare
        Skeleton/results_P17T07C02_lcrnet+v3d.json      prefixed AND suffixed

    So a search rather than a full match, with the id required to appear exactly once - two
    matches means the filename is ambiguous and guessing which one is the video would be
    the kind of silent mis-association that survives all the way to a wrong label.
    """
    stem = Path(path).name
    found = _TSU_IN_FILENAME.findall(stem)
    if not found:
        raise ValueError(
            f"no TSU video id in {stem!r}; expected P<dd>T<dd>C<dd> somewhere in the name")
    if len({tuple(f) for f in found}) > 1:
        raise ValueError(f"{stem!r} contains more than one video id: {found}")
    s, t, c = found[0]
    return VideoId(subject=int(s), take=int(t), camera=int(c))


def parse_tsm_name(filename: str) -> VideoId:
    """`Cook.Cut_p03_r00_v02_c03.mp4` -> VideoId(3, 0, 3, activity='Cook.Cut').

    The trimmed convention is `Activityname_p[id]_r[XX]_[XX]_c[0-7]`, where the fourth
    field is a view/repetition code that the official generator reads positionally and
    ignores. Extensions (`.mp4`, `.json`, `.npz`) are stripped so the same parser serves
    video, skeleton and derived-feature paths, and a derived tag after the camera field
    (`_pose3d`, `_LCRNet2d`) is captured rather than rejected.
    """
    stem = Path(filename).name
    for ext in (".mp4", ".json", ".npz", ".npy", ".avi", ".png", ".jpg", ".csv", ".txt"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    m = _TSM_NAME.match(stem)
    if not m:
        raise ValueError(
            f"not a trimmed Smarthome name: {filename!r} "
            "(expected Activity_p<dd>_r<dd>_<dd>_c<dd>[_tag])"
        )
    return VideoId(subject=int(m.group("subject")), take=int(m.group("take")),
                   camera=int(m.group("camera")), activity=m.group("activity"),
                   tag=m.group("tag"))


def protocol_side(vid: VideoId, protocol: str = "CS") -> str:
    """Which side of a protocol a video falls on: 'train', 'val', 'test' or 'unused'.

    CS has no validation camera, so it returns only train/test; carving a validation set
    out of CS means holding out TRAIN SUBJECTS, never test ones, and that belongs to the
    caller because the choice has to be recorded next to the numbers it produces.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")
    if protocol == "CS":
        if vid.subject in CS_TRAIN_SUBJECTS:
            return "train"
        if vid.subject in CS_TEST_SUBJECTS:
            return "test"
        return "unused"
    train = CV1_TRAIN_CAMERAS if protocol == "CV1" else CV2_TRAIN_CAMERAS
    val = CV1_VAL_CAMERAS if protocol == "CV1" else CV2_VAL_CAMERAS
    test = CV1_TEST_CAMERAS if protocol == "CV1" else CV2_TEST_CAMERAS
    if vid.camera in train:
        return "train"
    if vid.camera in val:
        return "val"
    if vid.camera in test:
        return "test"
    return "unused"


# --- annotations --------------------------------------------------------------------------

@dataclass(frozen=True)
class TsuVideo:
    """One untrimmed video's annotation. `duration` and action bounds are FRAMES at 25 Hz."""

    vid: str
    subject: int
    take: int
    camera: int
    subset: str                                  # 'training' | 'testing', as shipped
    duration: int                                # frames
    actions: tuple[tuple[int, int, int], ...]    # (class_id, start_frame, end_frame)

    @property
    def seconds(self) -> float:
        return self.duration / TSU_FPS

    def annotated_frames(self) -> int:
        """Frames covered by at least one action, so gap fraction is measurable."""
        if not self.actions:
            return 0
        covered = np.zeros(self.duration + 1, dtype=bool)
        for _, s, e in self.actions:
            covered[max(0, s):min(self.duration, e) + 1] = True
        return int(covered.sum())


def load_tsu_annotations(path: str | Path) -> dict[str, TsuVideo]:
    """Parse `smarthome_CS_51.json` / `smarthome_CV_51.json`.

    Validated on read rather than on use: an id whose embedded subject disagrees with the
    `subset` field it was shipped with means the file and the protocol have drifted apart,
    and that is a defect worth failing on. The check is what caught that both halves of the
    dataset share one subject partition instead of two.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, TsuVideo] = {}
    for vid, rec in data.items():
        ident = parse_tsu_name(vid)
        subset = str(rec["subset"])
        actions = tuple(
            (int(a[0]), int(a[1]), int(a[2])) for a in rec.get("actions", [])
        )
        bad = [a for a in actions if not 0 <= a[0] < len(TSU_CLASSES)]
        if bad:
            raise ValueError(f"{vid}: class ids outside [0,51): {bad[:3]}")
        out[vid] = TsuVideo(vid=vid, subject=ident.subject, take=ident.take,
                            camera=ident.camera, subset=subset,
                            duration=int(rec["duration"]), actions=actions)
    if not out:
        raise ValueError(f"{path} contained no videos")
    _assert_subset_matches_protocol(out)
    return out


def _assert_subset_matches_protocol(videos: dict[str, TsuVideo]) -> None:
    """The shipped `subset` must agree with the subject partition for a CS file.

    Containment, not equality, is the load-bearing check: every subject marked `training`
    has to be a CS train subject and every subject marked otherwise a CS test subject. That
    catches the defect worth catching - a held-out person appearing on the training side -
    on a file of any size, so a legitimately filtered subset still gets checked.

    Equality is additionally required when the file covers all 18 subjects, because then it
    IS claiming to be the complete official split and a missing subject is a truncated
    download rather than a deliberate subset.

    Skipped entirely for a cross-view file, which legitimately puts the same subject on both
    sides; applying the subject rule there would fail on correct data.
    """
    train_subj = {v.subject for v in videos.values() if v.subset == "training"}
    test_subj = {v.subject for v in videos.values() if v.subset != "training"}
    if train_subj & test_subj:
        return                                   # cross-view file: subjects span both sides

    stray_train = sorted(train_subj - set(CS_TRAIN_SUBJECTS))
    stray_test = sorted(test_subj - set(CS_TEST_SUBJECTS))
    if stray_train or stray_test:
        raise ValueError(
            "cross-subject split does not match the official partition: "
            f"subjects {stray_train} marked `training` are CS TEST subjects; "
            f"subjects {stray_test} marked `testing` are CS TRAIN subjects. "
            f"Official train={list(CS_TRAIN_SUBJECTS)}, test={list(CS_TEST_SUBJECTS)}."
        )
    if train_subj | test_subj == set(ALL_SUBJECTS) and (
            train_subj != set(CS_TRAIN_SUBJECTS) or test_subj != set(CS_TEST_SUBJECTS)):
        raise ValueError(
            "file covers all 18 subjects but the sides are wrong: "
            f"train={sorted(train_subj)} expected {list(CS_TRAIN_SUBJECTS)}; "
            f"test={sorted(test_subj)} expected {list(CS_TEST_SUBJECTS)}"
        )


def frame_multilabel(video: TsuVideo, n_steps: int, *,
                     n_classes: int = len(TSU_CLASSES),
                     official: bool = True) -> np.ndarray:
    """Dense [n_steps, n_classes] 0/1 target for one untrimmed video.

    `n_steps` is however many temporal positions the feature extractor produced, which is
    NOT the frame count: an I3D-style extractor with a stride emits one vector per snippet.
    The mapping from step to annotation units is therefore
    `position = step * duration / n_steps`, exactly the official loader's
    `fr / (num_feat/duration)`.

    `official=True` reproduces the reference implementation bit for bit, including its
    strict inequalities - a step landing exactly on a boundary is labelled negative. That
    loses at most one step per boundary, and it is kept as the default anyway: a published
    mAP is only comparable if the target tensor is built the same way, and quietly fixing
    an off-by-one would make every number in the literature an unfair comparison in our
    favour. `official=False` closes the interval, for our own training where nobody is
    comparing.
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    label = np.zeros((n_steps, n_classes), dtype=np.float32)
    scale = video.duration / float(n_steps)
    pos = np.arange(n_steps, dtype=np.float64) * scale
    for cls, start, end in video.actions:
        if cls >= n_classes:
            continue
        hit = (pos > start) & (pos < end) if official else (pos >= start) & (pos <= end)
        label[hit, cls] = 1.0
    return label


def frame_mask(video: TsuVideo, n_steps: int) -> np.ndarray:
    """[n_steps] 1.0 where at least one class is annotated, 0.0 in the gaps.

    The gaps are the whole reason this function exists. TSU is densely annotated for the 51
    activities it defines, and the resident spends the rest of the time sitting, standing or
    out of shot - none of which is a class here. For the 51-class benchmark head those gaps
    are legitimate negatives (the resident genuinely is not reading), so the official loader
    is right to leave them as zeros. For our UNIFIED head, which also carries `sitting`,
    `standing` and `lying_down` supervised by other corpora, a zero in the gap is a lie: it
    asserts "not sitting" over precisely the frames where sitting is most likely.

    So the cross-corpus loss is masked, and this is the mask. Every corpus supervises the
    classes it defines and abstains on the rest, which is the honest version of "use every
    dataset" - the alternative teaches the model that the classes another corpus owns are
    absent whenever this corpus is the source.
    """
    lab = frame_multilabel(video, n_steps, official=False)
    return (lab.sum(axis=1) > 0).astype(np.float32)


def iter_videos(videos: dict[str, TsuVideo], protocol: str = "CS",
                side: str = "train") -> Iterator[TsuVideo]:
    """Videos on one side of one protocol, derived from the ID rather than the JSON field.

    Deliberately not `v.subset == 'training'`. The shipped field is correct for the file it
    came in and meaningless for any other protocol, so a CV run that filtered on it would
    silently train on the CS split and report a cross-view number.
    """
    for v in sorted(videos.values(), key=lambda x: x.vid):
        ident = VideoId(subject=v.subject, take=v.take, camera=v.camera)
        if protocol_side(ident, protocol) == side:
            yield v


# Clip counts in the released trimmed split files, recorded so a truncated download is
# visible as a number rather than as a slightly worse result.
TSM_SPLIT_SIZES: dict[str, int] = {
    "train_CS": 8831, "validation_CS": 1854, "test_CS": 5444,
    "train_CV1": 1877, "validation_CV1": 3539, "test_CV1": 1901,
    "train_CV2": 7806, "validation_CV2": 3539, "test_CV2": 1901,
}


def load_trimmed_split(path: str | Path, *, expect: int | None = None) -> list[VideoId]:
    """Parse `splits/train_CS.txt` and friends from `srijandas07/i3d_smarthome`.

    Use these files rather than deriving the trimmed split ourselves, for one reason that is
    easy to miss: they ship a **validation** set. The skeleton generator does not - its
    `part='val'` branch is `not istraining`, which for the cross-subject protocol IS the test
    set. Following it means every hyper-parameter is tuned on the numbers being reported, and
    the result is not a held-out measurement at all. With `validation_CS.txt` we tune on 1,854
    clips and report on 5,444 we never touched.

    Every line is validated against the protocol implied by the filename, so a file that has
    been edited or concatenated fails here instead of leaking a subject later.
    """
    p = Path(path)
    names = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not names:
        raise ValueError(f"{p} is empty")
    out = [parse_tsm_name(n) for n in names]

    want = expect if expect is not None else TSM_SPLIT_SIZES.get(p.stem)
    if want is not None and len(out) != want:
        raise ValueError(
            f"{p.name} has {len(out)} clips, expected {want}. Either the download is "
            "truncated or the file is not the released one - both change every number "
            "computed from it."
        )

    stem = p.stem
    if stem.endswith("_CS"):
        side = "train" if stem.startswith(("train", "validation")) else "test"
        stray = sorted({v.subject for v in out if protocol_side(v, "CS") != side})
        if stray:
            raise ValueError(
                f"{p.name} is a CS {side} file but contains subjects {stray} from the other "
                "side. Note that `validation_CS.txt` must hold TRAIN subjects - a validation "
                "set drawn from test subjects is the leak this check exists to catch."
            )
    return out


# --- the per-video CSV annotations, as the official archive actually ships them ----------
#
# `Annotation_v1.0.tar.gz` does NOT contain `smarthome_CS_51.json`. It contains one CSV per
# video, foldered by subject:
#
#     Annotation/P18/P18T13C07.csv
#
# The aggregated JSON that every published baseline trains against is a DERIVED artefact,
# mirrored in `dairui01/Toyota_Smarthome/pipline/data/`. Both are used here and neither is
# trusted alone: the CSVs are the authoritative annotation, the JSON supplies `duration`
# (which the CSVs do not carry) and an independent copy of the same spans to check against.
# If they disagree, something is wrong with one of them and training on either would be
# guessing which.

_CSV_HEADER_WORDS = frozenset({"label", "class", "action", "activity", "name", "start",
                               "end", "start_frame", "end_frame", "begin", "stop", "id",
                               "action_index", "confidence", "score", "event"})

TSU_ALIASES: dict[str, str] = {
    # The shipped CSVs use a PRE-MERGE vocabulary; `Action_list` is post-merge at 51.
    # The dataset README states the merge as a requirement, not a suggestion:
    #
    #   "In this annotation, please merge `Make_coffee.Get_water` & `Get Water` and
    #    `Insert_tea_bag` & `Make_tea.Insert_tea_bag` to have 51 action classes."
    #
    # Without it the raw annotations carry 53 names, two of which resolve to no class, and
    # every published number is computed over the merged 51. Skipping the merge would give a
    # 53-class model that is not comparable to any baseline; dropping the unmerged spans
    # would silently delete real activity. So they are aliased onto their post-merge target.
    "Make_coffee.Get_water": "Get_water",
    "Make_tea.Insert_tea_bag": "Insert_tea_bag",
    # The README writes "Get Water" with a space in the same sentence. Whether that spelling
    # appears in any CSV is unknown; accepting it costs nothing and refusing it would fail a
    # 536-file run on one file.
    "Get Water": "Get_water",
    "Get water": "Get_water",
}


def canonical_tsu_class(name: str) -> str:
    """Raw annotation class name -> the post-merge name present in `TSU_CLASSES`."""
    return TSU_ALIASES.get(name.strip(), name.strip())


def read_annotation_csv(
    path: str | Path,
    *,
    classes: Sequence[str] = TSU_CLASSES,
    unknown: list[str] | None = None,
) -> tuple[tuple[int, int, int], ...]:
    """One per-video CSV -> ((class_id, start_frame, end_frame), ...).

    Confirmed schema, from the shipped archive: a header line `event,start_frame,end_frame`
    followed by rows naming the class, e.g. `Enter,170,187`. The sniffer also accepts the
    integer-id and 4-column event-map layouts, because `TSU_evaluation`'s own ground truth
    uses the latter and a reader that only handles one of them breaks on the other.

    Class names go through `canonical_tsu_class` first: the CSVs carry the PRE-MERGE
    53-name vocabulary and `TSU_CLASSES` is the post-merge 51 the README mandates.

    `unknown` changes the failure mode, and that is the point. Passing a list makes this
    COLLECT unrecognised class names into it and skip those rows instead of raising, so one
    pass over 536 files reports the complete vocabulary gap. Raising on the first unknown
    name cost a 93-minute Kaggle session to learn one name - and there were two.
    """
    p = Path(path)
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8-sig").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ()

    first = [c.strip() for c in re.split(r"[,;\t]", lines[0])]
    if any(c.lower() in _CSV_HEADER_WORDS for c in first) and not first[1].lstrip("-").isdigit():
        lines = lines[1:]                      # a header, not a row

    out: list[tuple[int, int, int]] = []
    for n, ln in enumerate(lines, start=1):
        parts = [c.strip() for c in re.split(r"[,;\t]", ln) if c.strip() != ""]
        if len(parts) < 3:
            raise ValueError(
                f"{p.name} line {n}: expected at least 3 fields "
                f"(class, start, end), got {len(parts)}: {ln!r}")
        head, start_s, end_s = parts[0], parts[1], parts[2]
        if head.lstrip("-").isdigit():
            cls = int(head)
        else:
            canon = canonical_tsu_class(head)
            try:
                cls = list(classes).index(canon)
            except ValueError:
                if unknown is not None:
                    unknown.append(head)
                    continue
                raise ValueError(
                    f"{p.name} line {n}: {head!r} (canonical {canon!r}) is neither an "
                    f"integer class id nor one of the {len(classes)} known class names. "
                    f"Line: {ln!r}. Pass `unknown=[]` to survey every file instead of "
                    "failing on the first."
                ) from None
        if not 0 <= cls < len(classes):
            raise ValueError(f"{p.name} line {n}: class id {cls} outside [0,{len(classes)})")
        try:
            start, end = int(float(start_s)), int(float(end_s))
        except ValueError:
            raise ValueError(
                f"{p.name} line {n}: start/end are not numeric ({start_s!r}, {end_s!r}). "
                "If this file quotes seconds rather than frames every span is 25x wrong; "
                "check against the JSON before using it."
            ) from None
        if end < start:
            raise ValueError(f"{p.name} line {n}: end {end} precedes start {start}")
        out.append((cls, start, end))
    return tuple(out)


def survey_annotation_csvs(root: str | Path) -> dict[str, Any]:
    """Walk every per-video CSV and report the COMPLETE vocabulary before parsing for real.

    Exists because of how the first Kaggle run failed: the parser raised on the first
    unrecognised class name, 93 minutes in, having surveyed one file out of 536. The
    vocabulary gap turned out to be two documented merges - and a run that reports both at
    once costs the same as a run that reports one.

    Returns the name histogram, the unknown names with an example file each, and the row
    count, so the notebook can print a go/no-go on the label space alone.
    """
    root = Path(root)
    files = sorted(root.glob("**/P*T*C*.csv"))
    if not files:
        raise FileNotFoundError(f"no per-video annotation CSVs under {root}")

    known = set(TSU_CLASSES)
    seen: dict[str, int] = {}
    unknown_where: dict[str, str] = {}
    aliased: dict[str, int] = {}
    rows = 0

    for f in files:
        lines = [ln.strip() for ln in f.read_text(encoding="utf-8-sig").splitlines()]
        lines = [ln for ln in lines if ln]
        if not lines:
            continue
        head = [c.strip() for c in re.split(r"[,;\t]", lines[0])]
        if any(c.lower() in _CSV_HEADER_WORDS for c in head) and not head[1].lstrip("-").isdigit():
            lines = lines[1:]
        for ln in lines:
            parts = [c.strip() for c in re.split(r"[,;\t]", ln) if c.strip() != ""]
            if not parts:
                continue
            rows += 1
            raw = parts[0]
            seen[raw] = seen.get(raw, 0) + 1
            if raw.lstrip("-").isdigit():
                continue
            canon = canonical_tsu_class(raw)
            if canon != raw:
                aliased[raw] = aliased.get(raw, 0) + 1
            if canon not in known:
                unknown_where.setdefault(raw, f.name)

    return {
        "n_files": len(files),
        "n_rows": rows,
        "n_distinct_names": len(seen),
        "histogram": dict(sorted(seen.items(), key=lambda kv: -kv[1])),
        "aliased": aliased,
        "unknown": {k: unknown_where[k] for k in sorted(unknown_where)},
    }


def load_tsu_annotations_from_csv(
    root: str | Path,
    durations: dict[str, int],
    *,
    subsets: dict[str, str] | None = None,
    unknown: list[str] | None = None,
) -> dict[str, TsuVideo]:
    """Walk `Annotation/P*/P*T*C*.csv` into the same `TsuVideo` objects as the JSON path.

    `durations` is required and has no default. The CSVs carry spans but not the video
    length, and every downstream computation needs it: `frame_multilabel` maps feature step
    to annotation position as `step * duration / n_steps`, so a wrong duration rescales the
    entire label tensor while leaving its shape correct. Passing it in forces the caller to
    say where it came from - the mirrored JSON, or a decode of the video itself.

    `subsets` defaults to deriving each video's side from the official CS subject partition,
    which is what makes the result usable without the JSON present at all.
    """
    root = Path(root)
    files = sorted(root.glob("**/P*T*C*.csv"))
    if not files:
        raise FileNotFoundError(
            f"no per-video annotation CSVs under {root}. Expected the layout the archive "
            "ships: Annotation/P<dd>/P<dd>T<dd>C<dd>.csv")

    out: dict[str, TsuVideo] = {}
    missing_duration: list[str] = []
    for f in files:
        ident = parse_tsu_filename(f)
        vid = ident.tsu
        if vid not in durations:
            missing_duration.append(vid)
            continue
        side = (subsets or {}).get(
            vid, "training" if ident.subject in CS_TRAIN_SUBJECTS else "testing")
        out[vid] = TsuVideo(vid=vid, subject=ident.subject, take=ident.take,
                            camera=ident.camera, subset=side,
                            duration=int(durations[vid]),
                            actions=read_annotation_csv(f, unknown=unknown))
    if missing_duration:
        raise ValueError(
            f"{len(missing_duration)} videos have annotations but no duration "
            f"(e.g. {missing_duration[:4]}). Supply them from the mirrored JSON or by "
            "decoding the video - a guessed duration rescales every label."
        )
    _assert_subset_matches_protocol(out)
    return out


def cross_check_annotations(csv_side: dict[str, TsuVideo],
                            json_side: dict[str, TsuVideo],
                            *, n_steps: int = 0) -> dict[str, Any]:
    """Compare the two annotation sources, at the SPAN level and at the FRAME level.

    This exists because the official archive and the artefact every published baseline
    trains on are two different files, and we now hold both.

    Span-set equality alone over-reports. The first real run found 82 of 536 videos
    "disagreeing", and the pattern in the examples was benign: the CSVs carry finer spans and
    the JSON merges adjacent same-class intervals. `(2, 1340, 2960)` + `(2, 2960, 2963)` in
    the CSV against `(2, 1480, 2963)` in the JSON is two different span lists describing
    almost the same frames - and frames are what the model sees. A boundary that moved by 140
    frames matters; a split that merges into the identical interval does not.

    So the frame-level agreement is computed too: per video, the fraction of (step, class)
    cells on which the two dense targets agree, plus the count that differ. `n_steps=0` (the
    default) uses each video's own duration capped at 20,000 steps, which is exact for the
    frame-rate the annotations are written in without allocating a 50,000 x 51 array.
    """
    shared = sorted(set(csv_side) & set(json_side))
    disagree: list[dict[str, Any]] = []
    frame_agree: list[float] = []

    for vid in shared:
        a, b = csv_side[vid], json_side[vid]
        sa, sb = set(a.actions), set(b.actions)
        steps = n_steps or min(max(a.duration, 1), 20_000)
        la = frame_multilabel(a, steps, official=False)
        lb = frame_multilabel(b, steps, official=False)
        same = float((la == lb).mean())
        frame_agree.append(same)
        if sa != sb or a.duration != b.duration:
            disagree.append({
                "vid": vid,
                "csv_spans": len(sa), "json_spans": len(sb),
                "only_in_csv": sorted(sa - sb)[:3],
                "only_in_json": sorted(sb - sa)[:3],
                "duration_csv": a.duration, "duration_json": b.duration,
                "frame_agreement": round(same, 5),
                "cells_differing": int((la != lb).sum()),
            })

    fa = np.asarray(frame_agree) if frame_agree else np.zeros(0)
    return {
        "compared": len(shared),
        "csv_only_videos": sorted(set(csv_side) - set(json_side)),
        "json_only_videos": sorted(set(json_side) - set(csv_side)),
        "agree": len(shared) - len(disagree),
        "disagree": disagree,
        "frame_agreement_mean": float(fa.mean()) if fa.size else 0.0,
        "frame_agreement_min": float(fa.min()) if fa.size else 0.0,
        "videos_below_999": int((fa < 0.999).sum()) if fa.size else 0,
    }


# --- fine -> coarse -----------------------------------------------------------------------

#
# Two classes are APPENDED at ids 20 and 21 rather than folded into `other_idle`, and
# nothing between 0 and 19 is renumbered. Both halves of that sentence are deliberate.
#
# Appending is what makes the change affordable. The shards store only the coarse label id
# (`labels` int64) and not the source class, so any change that RE-PARTITIONS ids 0..19
# cannot be applied by remapping - it needs the ~6 h re-extraction, because the information
# needed to un-merge was thrown away at extraction time. Appending leaves every existing
# label value correct: old shards simply contain no 20s or 21s, the range check still
# passes, and only the head width changes - which a retrain on a new corpus was going to do
# anyway.
#
# Not folding them into `other_idle` is the accuracy argument. Under the obvious map,
# `Use_laptop` (6.7% of annotated frames), `Use_tablet` (5.2%), `Use_Drawer` (1.6%),
# `Put_something_on_table` (1.3%), `Take_something_off_table` (1.1%) and `Use_cupboard`
# (1.1%) all land in the reject class - about 18% of the corpus, poured into the one class
# that is supposed to mean "nothing identifiable is happening". That is the same mistake the
# Charades keyword fallback made, and it is a large part of why macro-F1 came out at 0.128:
# a reject class trained on real, distinct, high-frequency activities stops being a reject
# class and starts competing with every other label.
#
# `using_device` earns its place through `cognitive_engagement` and `sedentary_index`, and
# `object_interaction` through `iadl_index` - purposeful handling of storage and surfaces is
# the instrumental-ADL signal that declines before basic ADLs do. Neither is dead weight
# under the taxonomy's own contract.

COARSE_V11: tuple[str, ...] = (
    "walking", "standing", "sitting", "lying_down", "standing_up", "sitting_down",
    "bending_reaching", "falling", "fallen_on_ground", "eating", "drinking",
    "cooking_food_prep", "taking_medication", "watching_tv", "reading", "using_phone",
    "cleaning_housework", "personal_hygiene", "interacting_with_person", "other_idle",
    "using_device", "object_interaction",
)


def assert_coarse_parity() -> None:
    """Ids 0..19 of `COARSE_V11` must be `activity.CLASS_NAMES`, in order.

    A parity assertion, not a second definition. `agents/activity.py` stays at 20 classes on
    purpose while the trained checkpoints are 20 wide - bumping it today would stop
    `EnsembleClassifier` loading `best.pt` and take the live demo down to buy nothing. This
    function is what guarantees the two lists cannot silently disagree in the meantime.
    """
    from behaviorsense.agents.activity import CLASS_NAMES     # noqa: PLC0415

    if tuple(COARSE_V11[:len(CLASS_NAMES)]) != tuple(CLASS_NAMES):
        raise AssertionError(
            "COARSE_V11 is not an extension of activity.CLASS_NAMES; ids 0..19 must match "
            f"exactly. Got {COARSE_V11[:len(CLASS_NAMES)]} vs {CLASS_NAMES}"
        )


TSU_TO_COARSE: dict[str, str] = {
    # locomotion. Enter/Leave are annotated as doorway traversals, which is walking with a
    # room transition attached - and `walking`'s behaviour_role already names
    # `room_transition`, so the signal has somewhere to go.
    "Enter": "walking", "Walk": "walking", "Leave": "walking",
    # transitions and recumbency. TSU annotates the TRANSITION only; the sustained sitting
    # and standing between them are unannotated, which is what `frame_mask` is for.
    "Sit_down": "sitting_down", "Get_up": "standing_up", "Lay_down": "lying_down",
    # nutrition: intake, kept apart from preparation
    "Eat_snack": "eating", "Breakfast": "eating", "Breakfast.Eat_at_table": "eating",
    "Drink.From_cup": "drinking", "Drink.From_bottle": "drinking",
    "Drink.From_glass": "drinking", "Drink.From_can": "drinking",
    # preparation. The four `Drink.From_*` classes above and the `Pour.From_*` classes here
    # are the same arm-to-mouth or arm-to-surface motion; separating them is an APPEARANCE
    # problem, not a pose one, which is the whole case for the crop-tube stream.
    "Make_coffee": "cooking_food_prep", "Get_water": "cooking_food_prep",
    "Make_coffee.Pour_water": "cooking_food_prep",
    "Make_coffee.Pour_grains": "cooking_food_prep",
    "Make_tea": "cooking_food_prep", "Make_tea.Boil_water": "cooking_food_prep",
    "Insert_tea_bag": "cooking_food_prep", "Stir_coffee/tea": "cooking_food_prep",
    "Pour.From_kettle": "cooking_food_prep", "Pour.From_bottle": "cooking_food_prep",
    "Pour.From_can": "cooking_food_prep", "Cook": "cooking_food_prep",
    "Cook.Cut": "cooking_food_prep", "Cook.Use_stove": "cooking_food_prep",
    "Cook.Stir": "cooking_food_prep", "Cook.Use_oven": "cooking_food_prep",
    "Breakfast.Cut_bread": "cooking_food_prep",
    "Breakfast.Spread_jam_or_butter": "cooking_food_prep",
    "Breakfast.Take_ham": "cooking_food_prep",
    # health. 383 instances and 60k frames of real supervision for the class Charades could
    # only reach through a single keyword rule.
    "Take_pills": "taking_medication",
    # housework
    "Clean_dishes": "cleaning_housework",
    "Clean_dishes.Put_something_in_sink": "cleaning_housework",
    "Clean_dishes.Dry_up": "cleaning_housework",
    "Clean_dishes.Clean_with_water": "cleaning_housework",
    "Wipe_table": "cleaning_housework", "Dump_in_trash": "cleaning_housework",
    # leisure and cognition. `Write` joins `reading` because both carry exactly one
    # behavioural signal, `cognitive_engagement`, and no Agent 3 feature distinguishes them.
    # Splitting them would add a class no downstream consumer reads.
    "Read": "reading", "Write": "reading", "Watch_TV": "watching_tv",
    "Use_telephone": "using_phone",
    "Use_laptop": "using_device", "Use_tablet": "using_device",
    # storage and surfaces -> the new instrumental class
    "Use_Drawer": "object_interaction", "Use_cupboard": "object_interaction",
    "Use_fridge": "object_interaction",
    "Put_something_on_table": "object_interaction",
    "Take_something_off_table": "object_interaction",
    # self-care. Putting glasses on is dressing, which is what `personal_hygiene` covers
    # coarsely; the taxonomy notes say coarse only, and this is the coarse reading.
    "Use_glasses": "personal_hygiene",
}

TSM_TO_COARSE: dict[str, str] = {
    # The trimmed half spells the same activities differently, so it gets its own table
    # rather than a normalising regex. A regex that turned `Readbook` into `Read` would also
    # turn a typo into a valid class, and the failure would be a silent relabel.
    "Cook.Cleandishes": "cleaning_housework", "Cook.Cleanup": "cleaning_housework",
    "Cook.Cut": "cooking_food_prep", "Cook.Stir": "cooking_food_prep",
    "Cook.Usestove": "cooking_food_prep", "Cutbread": "cooking_food_prep",
    "Makecoffee.Pourgrains": "cooking_food_prep",
    "Makecoffee.Pourwater": "cooking_food_prep",
    "Maketea.Boilwater": "cooking_food_prep",
    "Maketea.Insertteabag": "cooking_food_prep",
    "Pour.Frombottle": "cooking_food_prep", "Pour.Fromcan": "cooking_food_prep",
    "Pour.Fromkettle": "cooking_food_prep",
    "Drink.Frombottle": "drinking", "Drink.Fromcan": "drinking",
    "Drink.Fromcup": "drinking", "Drink.Fromglass": "drinking",
    "Eat.Attable": "eating", "Eat.Snack": "eating",
    "Enter": "walking", "Leave": "walking", "Walk": "walking",
    "Getup": "standing_up", "Sitdown": "sitting_down", "Laydown": "lying_down",
    "Readbook": "reading", "Takepills": "taking_medication", "WatchTV": "watching_tv",
    "Usetelephone": "using_phone",
    "Uselaptop": "using_device", "Usetablet": "using_device",
}


def coarse_id(name: str) -> int:
    return COARSE_V11.index(name)


def collapse_matrix(fine: Sequence[str], mapping: dict[str, str],
                    coarse: Sequence[str] = COARSE_V11) -> np.ndarray:
    """[n_fine, n_coarse] 0/1 membership, for pooling fine logits into coarse ones.

    The model is trained on the FINE labels and the coarse posterior is derived as
    `logsumexp` over each coarse class's members, rather than training directly on the
    coarse labels. That ordering is the point.

    Collapsing before the loss destroys the supervision that separates the members. All five
    `Cook.*` classes become one `cooking_food_prep` target, so the network is asked to give
    one answer for cutting, stirring, using a stove and using an oven while receiving no
    signal that they differ - a deliberately multi-modal class, which is the condition under
    which a softmax head does worst. Collapsing AFTER keeps every discriminative gradient and
    still produces exactly the 22 numbers Agent 3 consumes. It also lets the same checkpoint
    report the 51-class benchmark and the 22-class behaviour taxonomy without retraining.
    """
    M = np.zeros((len(fine), len(coarse)), dtype=np.float32)
    unmapped = []
    for i, f in enumerate(fine):
        target = mapping.get(f)
        if target is None:
            unmapped.append(f)
            continue
        M[i, list(coarse).index(target)] = 1.0
    if unmapped:
        raise KeyError(
            f"{len(unmapped)} fine classes have no coarse target: {unmapped[:6]}. "
            "Every class must be mapped explicitly - an unmapped class defaulting to "
            "`other_idle` is how 18% of this corpus nearly ended up in the reject class."
        )
    return M


def supervised_coarse(mapping: dict[str, str]) -> tuple[int, ...]:
    """Coarse ids a corpus can actually supervise. The complement is what must be masked.

    For Toyota this returns 15 of the 22 classes. Absent, and the reason each one is absent:
    `standing`/`sitting` (annotated only as transitions, never as sustained posture),
    `bending_reaching` (not in the label set), `falling`/`fallen_on_ground` (ADL corpus, no
    falls), `interacting_with_person` (one resident per recording), `other_idle` (a reject
    class has no positive examples to donate). Every one of those has to come from another
    corpus, and until it does the loss must abstain on them rather than score them as
    negative.
    """
    return tuple(sorted({COARSE_V11.index(v) for v in mapping.values()}))


def to_coarse(fine_label: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Dense fine target [T, n_fine] -> coarse target [T, n_coarse], as an OR not a sum.

    `label @ M` would count: a frame annotated both `Cook.Cut` and `Cook.Stir` would get a
    2 in `cooking_food_prep`, and a BCE target of 2 is not a probability. Clipping is the
    correct reduction for a membership matrix over multi-label data.
    """
    return np.clip(fine_label.astype(np.float32) @ M, 0.0, 1.0)


def frame_counts(videos: Iterable[TsuVideo],
                 n_classes: int = len(TSU_CLASSES)) -> np.ndarray:
    """Annotated frames per class, for the long-tail policy.

    Measured on the shipped CS file: 10,705,053 annotated frames, `Read` at 33.8% of them
    and `Drink.From_glass` at 0.04% - a 780x span. Effective-number balancing saturates by
    design well below that (it did on Charades at ~145x of ~942x), so the plan is balancing
    for the sampler plus post-hoc logit adjustment on the output, with tau swept and tau=0
    asserted to be an exact no-op.
    """
    counts = np.zeros(n_classes, dtype=np.int64)
    for v in videos:
        for cls, start, end in v.actions:
            if 0 <= cls < n_classes:
                counts[cls] += max(0, end - start)
    return counts


LCRNET_JOINTS: tuple[str, ...] = (
    "right_ankle", "left_ankle", "right_knee", "left_knee", "right_hip", "left_hip",
    "right_wrist", "left_wrist", "right_elbow", "left_elbow", "right_shoulder",
    "left_shoulder", "head",
)
"""The 13 joints LCR-Net emits, in its own order. Index positions are load-bearing.

This is the order INRIA's `read_xyz` indexes when it derives the two midpoints, so getting it
wrong does not raise - it silently builds a neck from an elbow and a hip. Recorded here so the
derivation below can be read against a name rather than against a number.
"""

# Indices into LCRNET_JOINTS, not into the 15-joint result. Named because `(10, 11)` in the
# middle of an expression is exactly the kind of constant that survives a refactor incorrectly.
_SHOULDERS = (LCRNET_JOINTS.index("right_shoulder"), LCRNET_JOINTS.index("left_shoulder"))
_HIPS = (LCRNET_JOINTS.index("right_hip"), LCRNET_JOINTS.index("left_hip"))

AGCN_JOINTS: tuple[str, ...] = (*LCRNET_JOINTS, "neck", "mid_hip")
"""The 15 joints INRIA's 2s-AGCN pipeline actually consumes: LCR-Net's 13 plus two midpoints.

The midpoints are DERIVED, not measured. That matters for any occlusion argument: a neck whose
two shoulders were both hallucinated is not evidence of a neck, and the confidence of a derived
joint is only as good as the pair it came from.
"""


def derive_agcn_joints(xyz: np.ndarray) -> np.ndarray:
    """[T, 13, C] LCR-Net joints -> [T, 15, C], appending neck and mid-hip.

    Reproduces the midpoint derivation in INRIA's own `read_xyz` rather than inventing a
    skeleton layout: the released 2s-AGCN graph has 15 nodes and its adjacency is indexed
    against exactly this order, so a different ordering trains a model on a body whose bones
    connect the wrong joints.
    """
    arr = np.asarray(xyz, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] != len(LCRNET_JOINTS):
        raise ValueError(
            f"expected [T, {len(LCRNET_JOINTS)}, C] LCR-Net joints, got {arr.shape}. "
            "Passing an already-derived 15-joint array here would append a second pair of "
            "midpoints computed from the wrong indices."
        )
    neck = arr[:, _SHOULDERS, :].mean(axis=1, keepdims=True)
    mid_hip = arr[:, _HIPS, :].mean(axis=1, keepdims=True)
    return np.concatenate([arr, neck, mid_hip], axis=1)


LCRNET_NJTS = 13
"""`njts` in every released pose file. MEASURED 2026-08-26 across all three archives.

Confirmed identical in trimmed V1.1 (`Drink.Fromcup_p20_r02_v02_c05.json`), trimmed V1.2
(`Walk_p25_r12_v15_c06_pose3d.json`) and untrimmed (`results_P17T07C02_lcrnet+v3d.json`):
`{"K": int, "njts": 13, "frames": [[{pose2d: 26 floats, pose3d: 39 floats}, ...], ...]}`.
Read from the document rather than assumed - a release shipping a different count must fail
loudly, because every index in `LCRNET_JOINTS` and every bone in the graph depends on it.
"""

LCRNET_FLAT_COORDINATE_MAJOR = True
"""`pose3d` is `[x0..x12, y0..y12, z0..z12]` - NOT `[x0,y0,z0, x1,y1,z1, ...]`.

The single most dangerous fact about this format: both readings have length 39, both reshape
without error, and only one builds a body. The other silently swaps joints for coordinates and
produces a skeleton that trains perfectly well on nonsense.

LCR-Net's own visualiser is the authority - it plots joint `i` as
`(pose2d[i], pose2d[i + njts])`, so y of joint i sits `njts` positions later, which is
coordinate-major. `verify_lcrnet_layout()` re-derives it from the data anatomically so the
constant is checked rather than trusted.
"""

# Vertical anatomy of an upright person, in IMAGE coordinates (y increases downward). Used to
# decide the flat layout from the data instead of from a remembered convention.
_UPRIGHT_CHAIN = (
    ("right_ankle", "left_ankle"),
    ("right_knee", "left_knee"),
    ("right_hip", "left_hip"),
    ("right_shoulder", "left_shoulder"),
    ("head",),
)


def unflatten_lcrnet(flat: Any, *, njts: int = LCRNET_NJTS, dim: int,
                     coordinate_major: bool = LCRNET_FLAT_COORDINATE_MAJOR) -> np.ndarray:
    """`[njts*dim]` flat -> `[njts, dim]`, honouring LCR-Net's coordinate-major layout."""
    arr = np.asarray(flat, dtype=np.float32).reshape(-1)
    if arr.size != njts * dim:
        raise ValueError(
            f"expected {njts * dim} floats for {njts} joints x {dim}D, got {arr.size}. "
            "`njts` is in the document; read it rather than assuming this constant."
        )
    return arr.reshape(dim, njts).T if coordinate_major else arr.reshape(njts, dim)


def upright_chain_score(xy: np.ndarray) -> float:
    """Fraction of frames whose vertical joint order is that of an upright person.

    Ankles below knees below hips below shoulders below head, in image y. Toyota subjects are
    upright for most of every clip, so a correct layout scores high and a joint-for-coordinate
    transposition scores near chance. This is the evidence `verify_lcrnet_layout` uses.
    """
    arr = np.asarray(xy, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] != LCRNET_NJTS or arr.shape[2] < 2:
        raise ValueError(f"expected [T, {LCRNET_NJTS}, >=2], got {arr.shape}")
    levels = [arr[:, [LCRNET_JOINTS.index(n) for n in group], 1].mean(axis=1)
              for group in _UPRIGHT_CHAIN]
    ok = np.ones(len(arr), dtype=bool)
    for above, below in zip(levels, levels[1:]):
        ok &= above > below                     # y grows downward: ankles have the largest y
    return float(ok.mean())


def verify_lcrnet_layout(flat_frames: Sequence[Any], *,
                         njts: int = LCRNET_NJTS) -> dict[str, Any]:
    """Decide coordinate-major vs joint-major FROM THE DATA, and say by how much.

    Returns the score of each hypothesis and which one won. A caller that assumed the wrong one
    gets a number to look at instead of a plausible-looking skeleton, which is the whole point:
    this is the one error in the format that cannot be seen by eye in a training curve.
    """
    pose2d = [np.asarray(f, dtype=np.float32) for f in flat_frames]
    if not pose2d:
        raise ValueError("no frames given to verify the layout with")
    cm = np.stack([unflatten_lcrnet(f, njts=njts, dim=2, coordinate_major=True)
                   for f in pose2d])
    jm = np.stack([unflatten_lcrnet(f, njts=njts, dim=2, coordinate_major=False)
                   for f in pose2d])
    s_cm, s_jm = upright_chain_score(cm), upright_chain_score(jm)
    # DECISIVE means the winner is positive evidence, not just the larger of two near-zero
    # numbers. `upright_chain_score` asks whether the body looks like a standing person, so it
    # says nothing at all about a clip where the subject is seated, lying, or has their legs
    # behind a kitchen counter - both readings then score near zero and noise picks a winner.
    #
    # This is not hypothetical: rejecting on the bare winner threw out 295 of 16,115 trimmed
    # clips, concentrated in `Cook.Cut`, `Cutbread`, `Cook.Cleanup` and `Drink.From*` - counter
    # activities with occluded legs - and clustered by camera, which is a viewing-geometry
    # signature rather than a file-format one. A real transposition scores ~1.0 against ~0.0.
    best, worst = max(s_cm, s_jm), min(s_cm, s_jm)
    decisive = best >= 0.5 and (best - worst) >= 0.2
    return {
        "coordinate_major_score": round(s_cm, 4),
        "joint_major_score": round(s_jm, 4),
        "winner": "coordinate_major" if s_cm >= s_jm else "joint_major",
        "margin": round(abs(s_cm - s_jm), 4),
        # False = "anatomy cannot tell", which is a different answer from "the layout differs".
        "decisive": bool(decisive),
        "n_frames": len(pose2d),
        "matches_constant": (s_cm >= s_jm) == LCRNET_FLAT_COORDINATE_MAJOR,
    }


MIN_LAYOUT_AGREEMENT = 0.90
"""Share of DECISIVE files that must agree before an archive's flat layout is accepted.

Not unanimity, and the number is measured rather than chosen for tidiness. Over the whole
trimmed archive 73 of 16,115 files (0.45%) score "decisively joint-major" while being ordinary
coordinate-major clips - `upright_chain_score` can be satisfied by the wrong reading when the
body is roughly symmetric. A unanimity bar would therefore abort a perfectly good build whenever
a sample happened to contain one of them.

0.90 is far above that contamination and far below what an archive-wide transposition looks
like, which is ~0% agreement. The gap between 0.45% and 10% is the whole margin of safety.
"""


def assert_corpus_layout(paths: Sequence[str | Path], *, njts: int = LCRNET_NJTS,
                         sample: int = 64, coordinate_major: bool = LCRNET_FLAT_COORDINATE_MAJOR,
                         min_agreement: float = MIN_LAYOUT_AGREEMENT) -> dict[str, Any]:
    """Decide an archive's flat layout ONCE, from the files where anatomy can speak.

    This is where the layout check belongs. A transposition is a property of how a release was
    written, so one corpus-level answer protects every file - whereas a per-file test inherits
    every posture the estimator saw and rejects clips for being seated.

    Reads up to `sample` files, keeps the DECISIVE verdicts, and requires `min_agreement` of them
    to match the assumed layout. Raises otherwise: an archive written the other way round must
    stop a build before it produces a corpus of scrambled skeletons that trains without
    complaint.
    """
    votes: list[tuple[str, dict[str, Any]]] = []
    for p in list(paths)[:sample]:
        path = Path(p)
        try:
            doc = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            flat = [d[0]["pose2d"] for d in (_frame_detections(f) for f in _pose_frames(doc))
                    if d and "pose2d" in d[0]]
            if flat:
                votes.append((path.name, verify_lcrnet_layout(flat[:400], njts=njts)))
        except Exception:                                          # noqa: BLE001, S112
            continue                    # unreadable files are the builder's problem, not this one

    want = "coordinate_major" if coordinate_major else "joint_major"
    decisive = [(n, v) for n, v in votes if v["decisive"]]
    agree = [n for n, v in decisive if v["winner"] == want]
    dissent = [(n, v["coordinate_major_score"], v["joint_major_score"])
               for n, v in decisive if v["winner"] != want]
    share = len(agree) / len(decisive) if decisive else 0.0

    report = {
        "n_read": len(votes),
        "n_decisive": len(decisive),
        "n_agree": len(agree),
        "agreement": round(share, 4),
        "assumed": want,
        "min_agreement": min_agreement,
        "dissenters": dissent[:10],
    }
    if len(decisive) < 8:
        raise ValueError(
            f"only {len(decisive)} of {len(votes)} sampled files gave a decisive layout verdict, "
            "which is too few to confirm the archive. Every subject may be seated, or `pose2d` "
            "may not be in image coordinates. Raise `sample` or inspect the files directly - "
            "proceeding would mean asserting the layout on no evidence."
        )
    if share < min_agreement:
        raise ValueError(
            f"only {share:.1%} of {len(decisive)} decisive files read as {want} (needed "
            f"{min_agreement:.0%}). This archive is probably written the other way round, and "
            f"reading it as {want} would build every skeleton out of scrambled joints and train "
            f"on it without complaint. Dissenting files and their (coordinate, joint) scores: "
            f"{dissent[:5]}"
        )
    return report


MIN_POSE_COVERAGE = 0.60
"""Fraction of a video's frames that must carry a detection for the video to be usable.

The threshold exists because the corpus average hides the shape of the failure. Measured
2026-08-26 over 120 untrimmed videos: 15.30% of 3,042,437 frames carry no detection, which reads
like tolerable noise. Per video it is bimodal:

    results_P20T01C05   29,473 frames    0.0% empty
    results_P13T24C04   26,367 frames    1.1%
    results_P17T07C02   19,143 frames    1.8%
    results_P03T18C03   14,204 frames    5.6%
    results_P25T05C05   32,004 frames   62.1%
    results_P09T14C07   17,457 frames   99.6%

`P09T14C07` is 17,385 empty frames out of 17,457. Undetected frames are zero-filled to keep frame
indices aligned with the annotations, so that video contributes ~17k windows of all-zero skeletons
carrying real activity labels - it teaches the model that `Cook` looks like nothing. Averaged in,
it is invisible; per video it is a hard reject.

0.60 is a JUDGEMENT, not a measurement: it sits in the empty gap between the two modes (5.6% and
62.1% missing) rather than being fitted to anything. `coverage_report` prints what it drops so the
cut can be argued with, and the deciles from `scripts/verify_toyota_poses.py` are what would move
it. It is deliberately loose - a video at 55% coverage still has 45% real skeletons and those
windows are fine; what is excluded is videos where the pose estimator essentially failed.
"""


def _longest_run(mask: np.ndarray) -> int:
    """Longest run of True in a 1-D boolean array. 0 for none."""
    best = run = 0
    for v in np.asarray(mask, dtype=bool).reshape(-1):
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def coverage_report(present: Any, *,
                    min_coverage: float = MIN_POSE_COVERAGE) -> dict[str, Any]:
    """Per-VIDEO usability from detection coverage, with the reason attached.

    Returns `usable` plus the numbers behind it. A boolean alone would make a dropped video
    indistinguishable from a missing one in a log, and this corpus has already shown that a
    silently skipped input is the expensive kind of bug.
    """
    seen = np.asarray(present, dtype=bool).reshape(-1)
    if seen.size == 0:
        return {"usable": False, "n_frames": 0, "n_detected": 0, "coverage": 0.0,
                "min_coverage": float(min_coverage), "longest_gap": 0,
                "reason": "no frames"}
    cov = float(seen.mean())
    ok = cov >= min_coverage
    return {
        "usable": ok,
        "n_frames": int(seen.size),
        "n_detected": int(seen.sum()),
        "coverage": round(cov, 4),
        "min_coverage": float(min_coverage),
        # Longest unbroken run of empty frames. A video can pass on average and still be blind
        # through the one activity that matters, so the gap is reported beside the average.
        "longest_gap": _longest_run(~seen),
        "reason": "" if ok else (
            f"detection coverage {cov:.1%} is below {min_coverage:.0%}; its windows would be "
            "mostly zero-filled skeletons carrying real labels"),
    }


def _pose_frames(doc: Any) -> list[Any]:
    """Find the per-frame list inside a `*_pose3d.json`, whatever it is wrapped in.

    SNIFFED, not assumed. The same reasoning as `read_annotation_csv`: the released archives
    are not uniform, the README does not specify this file, and the first run of notebook 06
    died on an unlisted class name because a format was taken on trust. A reader that reports
    what it found beats one that raises `KeyError: 'frames'` from three frames deep.
    """
    if isinstance(doc, list):
        return doc
    if isinstance(doc, dict):
        for key in ("frames", "poses", "pose3d", "annotations", "data", "results"):
            val = doc.get(key)
            if isinstance(val, list):
                return val
        # A dict keyed by frame index, which some releases use. Sorted NUMERICALLY: string
        # order puts frame 10 before frame 2 and silently shuffles the clip.
        if doc and all(str(k).lstrip("-").isdigit() for k in doc):
            return [doc[k] for k in sorted(doc, key=lambda k: int(k))]
    raise ValueError(
        f"cannot find a per-frame list in this pose document. Top level is "
        f"{type(doc).__name__}"
        + (f" with keys {sorted(doc)[:12]}" if isinstance(doc, dict) else "")
        + ". The measured schema is {'K': int, 'njts': 13, 'frames': [[{pose2d, pose3d}]]}; "
          "run `scripts/verify_toyota_poses.py` and paste its report - this reader is extended "
          "from measured layouts, never from a guess."
    )


def _frame_detections(frame: Any) -> list[dict]:
    """One entry of `frames` -> its list of person dicts. `[]` when nobody was detected.

    `frames[i]` is a LIST, not a dict: LCR-Net emits one entry per detection per frame, and an
    empty list is a real observation (nobody found), not a parse failure. Dropping such a frame
    would shorten the clip and desynchronise every frame-indexed annotation - the alignment
    `frame_multilabel` exists to hold.
    """
    if frame is None:
        return []
    if isinstance(frame, dict):
        return [frame]                          # a release that unwrapped the single detection
    if isinstance(frame, (list, tuple)):
        return [d for d in frame if isinstance(d, dict)]
    raise ValueError(f"a frame entry is {type(frame).__name__}, not a list of detections")


def _frame_pose(frame: Any, njts: int, *, key: str, dim: int,
                coordinate_major: bool) -> np.ndarray | None:
    """One frame -> `[njts, dim]` for the FIRST detection, or None when there is none.

    First detection only, deliberately. Toyota Smarthome is single-subject by construction, and
    "pick the largest" or "pick the highest score" would introduce a selection rule INRIA's own
    loader does not have - reproducing their pipeline means reproducing this too. Note that a
    score is not even available in V1.2: `cumscore` is present in V1.1 and absent from the
    refined release, so a confidence-based rule could not be applied uniformly anyway.
    """
    dets = _frame_detections(frame)
    if not dets:
        return None
    flat = dets[0].get(key)
    if flat is None:
        raise ValueError(
            f"a detection carries {sorted(dets[0])} and no {key!r}. The measured schema is "
            f"{{pose2d: {njts * 2} floats, pose3d: {njts * 3} floats}}; V1.1 adds `cumscore`."
        )
    return unflatten_lcrnet(flat, njts=njts, dim=dim, coordinate_major=coordinate_major)


def read_pose3d(path: str | Path, *, derive: bool = True,
                coordinate_major: bool = LCRNET_FLAT_COORDINATE_MAJOR,
                verify_layout: bool = True) -> dict[str, Any]:
    """One released Toyota pose file -> `{xyz, xy, present, n_frames, ...}`.

    Handles all three archives, which share one measured schema:
    `{"K": int, "njts": 13, "frames": [[{pose2d, pose3d}, ...], ...]}` - trimmed V1.1 (which
    also carries `cumscore`), trimmed V1.2 (`*_pose3d.json`, no score), and untrimmed
    (`results_P*T*C*_lcrnet+v3d.json`).

    `xyz` is [T, 15, 3] with `derive=True` (INRIA's 2s-AGCN input) or [T, 13, 3] raw. `xy` is
    the 2D pose in image coordinates, kept because it is the only part of this file that can be
    checked against the video - and therefore the only cheap cross-reference for RTMO.

    `present` is [T] bool: False where LCR-Net detected nobody. Those frames are ZERO-FILLED and
    flagged rather than dropped, so `xyz[i]` still corresponds to source frame `i` and the
    frame-indexed annotations continue to line up.

    `njts` is read from the document, not assumed. `verify_layout` re-derives the flat array's
    coordinate-major ordering from the data and raises on disagreement, because that is the one
    error here that produces a trainable skeleton made of scrambled joints.

    Consuming these commits to something worth stating: 15-joint 3D from a 2017 estimator, which
    no webcam produces. A model trained on this input cannot be served from an RGB upload without
    also shipping LCR-Net. The RTMO COCO-17 path (`behaviorsense.video`) stays the deployable
    one; this exists so INRIA's published protocol can run on their published inputs.
    """
    p = Path(path)
    doc = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    njts = int(doc.get("njts", LCRNET_NJTS)) if isinstance(doc, dict) else LCRNET_NJTS
    if njts != len(LCRNET_JOINTS):
        raise ValueError(
            f"{p.name} declares njts={njts}, but LCRNET_JOINTS names {len(LCRNET_JOINTS)}. "
            "Every joint index and every bone in the 2s-AGCN graph is keyed to that order, so "
            "this is a data-format change, not a reshape - the joint table is what must be "
            "updated."
        )
    frames = _pose_frames(doc)

    check: dict[str, Any] | None = None
    if verify_layout:
        flat2d = [d[0]["pose2d"] for d in (_frame_detections(f) for f in frames)
                  if d and "pose2d" in d[0]]
        if flat2d:
            # RECORDED, NEVER REJECTED. The flat layout is a property of how the ARCHIVE was
            # written, not of one clip, so it is decided once by `assert_corpus_layout` and every
            # file is then read with the confirmed constant.
            #
            # Rejecting per file was wrong twice over. First it threw out 295 of 16,115 clips
            # whose subject simply is not upright. Then, with the confidence gate added, 73
            # survived as "decisively joint-major" - and those are the heuristic being fooled,
            # not a mixed archive: reading coordinate-major 2D as joint-major yields "joints 0-5"
            # from x-values alone and "joints 7-12" from y-values alone, which satisfies
            # hips > shoulders > head structurally and reduces the ankle/knee links to coin flips
            # on millimetre x-differences. A roughly symmetric body can therefore score high
            # under the wrong reading by construction. The residue clustered in `Drink.From*` and
            # adjacent takes of one subject, which is a posture-and-viewpoint signature.
            check = verify_lcrnet_layout(flat2d[:600], njts=njts)

    out3, out2, present = [], [], []
    for f in frames:
        a3 = _frame_pose(f, njts, key="pose3d", dim=3, coordinate_major=coordinate_major)
        a2 = _frame_pose(f, njts, key="pose2d", dim=2, coordinate_major=coordinate_major)
        present.append(a3 is not None)
        out3.append(np.zeros((njts, 3), np.float32) if a3 is None else a3)
        out2.append(np.zeros((njts, 2), np.float32) if a2 is None else a2)
    if not out3:
        raise ValueError(f"{p.name} parsed but contained 0 frames")
    xyz = np.stack(out3)
    if derive:
        xyz = derive_agcn_joints(xyz)
    seen = np.asarray(present, dtype=bool)
    vis = xyz[seen].reshape(-1, 3) if seen.any() else None
    return {
        "path": str(p),
        "xyz": xyz,
        "xy": np.stack(out2),
        "present": seen,
        "njts": njts,
        "K": doc.get("K") if isinstance(doc, dict) else None,
        # V1.1 only. Recorded because a confidence filter can be applied to one archive and not
        # the other, and a rule that silently applies to half a corpus is a distribution shift.
        "has_cumscore": bool(frames) and "cumscore" in (_frame_detections(frames[0]) or [{}])[0],
        # The layout evidence for THIS file, so an inconclusive read is auditable afterwards
        # instead of being indistinguishable from a confident one.
        "layout_decisive": None if check is None else check["decisive"],
        "layout_scores": None if check is None else
        [check["coordinate_major_score"], check["joint_major_score"]],
        "n_frames": int(len(seen)),
        "n_missing": int((~seen).sum()),
        "joints": int(xyz.shape[1]),
        # Coordinate RANGE, because it is the only cheap way to tell metres from pixels from
        # normalised units, and all three occur in released pose data somewhere.
        "coord_min": None if vis is None else [round(float(v), 3) for v in vis.min(axis=0)],
        "coord_max": None if vis is None else [round(float(v), 3) for v in vis.max(axis=0)],
    }


__all__ = [
    "assert_corpus_layout",
    "MIN_LAYOUT_AGREEMENT",
    "coverage_report",
    "MIN_POSE_COVERAGE",
    "verify_lcrnet_layout",
    "upright_chain_score",
    "unflatten_lcrnet",
    "LCRNET_FLAT_COORDINATE_MAJOR",
    "LCRNET_NJTS",
    "read_pose3d",
    "derive_agcn_joints",
    "LCRNET_JOINTS",
    "AGCN_JOINTS",
    "COARSE_V11",
    "CS_TEST_SUBJECTS",
    "CS_TRAIN_SUBJECTS",
    "CV1_TEST_CAMERAS",
    "CV1_TRAIN_CAMERAS",
    "CV1_VAL_CAMERAS",
    "CV2_TEST_CAMERAS",
    "CV2_TRAIN_CAMERAS",
    "CV2_VAL_CAMERAS",
    "PROTOCOLS",
    "TSM_CLASSES",
    "TSM_CV_CLASSES",
    "TSM_FPS",
    "TSM_FPS_DOCUMENTED",
    "TSM_ID_SPACES",
    "TSM_RGB_OFFSET",
    "TSM_RGB_UNMATCHED",
    "TSM_SKELETON_JOINTS",
    "TSM_SPLIT_SIZES",
    "TSM_TO_COARSE",
    "TSU_ALIASES",
    "TSU_CLASSES",
    "TSU_FPS",
    "TSU_TO_COARSE",
    "TsuVideo",
    "VideoId",
    "assert_coarse_parity",
    "canonical_tsu_class",
    "coarse_id",
    "collapse_matrix",
    "cross_check_annotations",
    "frame_counts",
    "frame_mask",
    "frame_multilabel",
    "iter_videos",
    "load_trimmed_split",
    "load_tsu_annotations",
    "load_tsu_annotations_from_csv",
    "parse_tsm_name",
    "parse_tsu_filename",
    "parse_tsu_name",
    "protocol_side",
    "read_annotation_csv",
    "slug",
    "supervised_coarse",
    "survey_annotation_csvs",
    "to_coarse",
    "tsm_id",
    "tsm_name",
    "tsu_id",
]
