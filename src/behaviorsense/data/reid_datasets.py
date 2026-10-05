"""Open-set ReID protocol construction for Market-1501 / MSMT17 layouts.

Why this file exists
--------------------
`PerceptionConfig.reid_match_threshold` was a guess. It is the single most consequential
number in Agent 1: too high and a stranger is labelled RESIDENT, contaminating every
behavioural feature Agent 3 consumes; too low and the resident fragments into several
identities. A guessed value is not defensible, so it has to be fitted on real data and
reported as an operating point with its false-accept rate.

That requires an *open-set* protocol, which standard ReID benchmarks do not provide.
Market-1501 and MSMT17 evaluate closed-set retrieval: every query identity exists in the
gallery, and the metric is "is the top-1 correct". Home monitoring asks a different
question - "is this person anyone we enrolled, or nobody?" - and the answer must
sometimes be *nobody*. So we rebuild the split:

    enrolled ids   -> a few crops become the gallery centroid (like enrolling a resident)
                      their remaining crops, FROM OTHER CAMERAS, are genuine probes
    impostor ids   -> never enrolled; all their crops are impostor probes

A genuine probe should match its own centroid; an impostor probe must be REJECTED. That
gives a two-distribution problem with a threshold sweep, which is what an ROC needs.

Three ways this protocol could be rigged, and what stops each
-------------------------------------------------------------
1. *Identity leakage* - an impostor id also appearing as enrolled. Then impostor probes
   are genuine probes in disguise and the false-accept rate is meaningless. Enforced
   disjoint by construction and asserted in `validate()`.
2. *Same-camera probing* - probing an enrolled id with a crop from the same camera in the
   same few seconds. Consecutive frames of one tracklet are near-duplicates, so this
   measures JPEG similarity rather than re-identification, and inflates accuracy
   massively. `enrol_cameras` and probe cameras are disjoint per identity.
3. *Threshold fitted and reported on the same probes.* Split into fit/test halves by
   identity, fit on one, report on the other.

Filename convention (identical in both datasets after the standard conversion):

    0043_c12_0033.jpg  ->  pid 0043, camera c12, frame 0033

Market-1501 additionally uses pid `-1` (junk) and `0000` (distractor); both are dropped.
In MSMT17 `0000` is a legitimate identity, so the rule is applied per dataset.
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

DatasetName = Literal["market1501", "msmt17"]

# Market-1501 reserves these pids; MSMT17 does not.
_MARKET_RESERVED = {"-1", "0000"}

_LAYOUTS: dict[str, tuple[str, ...]] = {
    "market1501": ("bounding_box_train", "bounding_box_test", "query"),
    "msmt17": ("bounding_box_train", "bounding_box_test", "query"),
}


@dataclass(frozen=True)
class Crop:
    """One person crop on disk. Metadata comes from the filename; no image is decoded."""

    path: str
    pid: str
    camera: str
    frame: int

    @property
    def key(self) -> str:
        return f"{self.pid}/{self.camera}/{self.frame}"


@dataclass
class OpenSetSplit:
    """An open-set evaluation protocol instance.

    `enrolment[pid]` are the crops used to build that identity's gallery centroid.
    `genuine[pid]` are probes of an enrolled identity from *other* cameras.
    `impostors` are probes whose identity was never enrolled - the correct answer for
    every one of them is UNKNOWN.
    """

    name: str
    enrolment: dict[str, list[Crop]] = field(default_factory=dict)
    genuine: dict[str, list[Crop]] = field(default_factory=dict)
    impostors: list[Crop] = field(default_factory=list)
    enrol_cameras: dict[str, set[str]] = field(default_factory=dict)

    @property
    def n_enrolled(self) -> int:
        return len(self.enrolment)

    @property
    def n_genuine(self) -> int:
        return sum(len(v) for v in self.genuine.values())

    @property
    def n_impostor(self) -> int:
        return len(self.impostors)

    def all_crops(self) -> list[Crop]:
        out: list[Crop] = []
        for crops in self.enrolment.values():
            out.extend(crops)
        for crops in self.genuine.values():
            out.extend(crops)
        out.extend(self.impostors)
        return out

    def validate(self) -> None:
        """Fail loudly if the protocol is rigged in any of the three known ways.

        Called by `build_open_set_split`, so a leaked split cannot silently produce a
        flattering ROC. Every assertion here corresponds to a way the reported
        false-accept rate would be a lie.
        """
        enrolled = set(self.enrolment)
        if not enrolled:
            raise ValueError(f"{self.name}: no enrolled identities")
        if not self.impostors:
            raise ValueError(
                f"{self.name}: no impostor probes - a false-accept rate cannot be "
                "estimated without them, and the ROC would be meaningless"
            )

        impostor_ids = {c.pid for c in self.impostors}
        leaked = enrolled & impostor_ids
        if leaked:
            raise ValueError(
                f"{self.name}: {len(leaked)} identities are both enrolled and impostor "
                f"(e.g. {sorted(leaked)[:5]}); impostor probes would be genuine probes "
                "in disguise"
            )

        for pid, probes in self.genuine.items():
            if pid not in self.enrolment:
                raise ValueError(f"{self.name}: genuine probes for unenrolled id {pid}")
            bad_pid = [c for c in probes if c.pid != pid]
            if bad_pid:
                raise ValueError(f"{self.name}: probe of id {bad_pid[0].pid} filed under {pid}")
            overlap = {c.camera for c in probes} & self.enrol_cameras[pid]
            if overlap:
                raise ValueError(
                    f"{self.name}: id {pid} is probed from its own enrolment camera(s) "
                    f"{sorted(overlap)}; near-duplicate frames measure JPEG similarity, "
                    "not re-identification"
                )

        enrol_paths = {c.path for crops in self.enrolment.values() for c in crops}
        probe_paths = {c.path for crops in self.genuine.values() for c in crops}
        probe_paths |= {c.path for c in self.impostors}
        shared = enrol_paths & probe_paths
        if shared:
            raise ValueError(
                f"{self.name}: {len(shared)} crops are used both to enrol and to probe; "
                "an identical image trivially matches its own centroid"
            )

    def summary(self) -> str:
        gpi = self.n_genuine / max(1, self.n_enrolled)
        epi = sum(len(v) for v in self.enrolment.values()) / max(1, self.n_enrolled)
        return (
            f"{self.name}: {self.n_enrolled} enrolled ids ({epi:.1f} crops/id), "
            f"{self.n_genuine} genuine probes ({gpi:.1f}/id), "
            f"{self.n_impostor} impostor probes from "
            f"{len({c.pid for c in self.impostors})} unseen ids"
        )


def parse_crop(path: str, name: str, dataset: DatasetName) -> Crop | None:
    """Parse `0043_c12_0033.jpg`. Returns None for junk, distractors and non-images."""
    if not name.endswith((".jpg", ".jpeg", ".png")) or name.startswith("."):
        return None
    stem = name.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    pid, cam = parts[0], parts[1]
    if dataset == "market1501" and pid in _MARKET_RESERVED:
        # -1 is junk, 0000 is a distractor. Keeping them would put unlabelled people into
        # the genuine set. MSMT17 has a real identity 0000, hence the per-dataset rule.
        return None
    if not cam.startswith("c"):
        return None
    try:
        frame = int(parts[2])
    except ValueError:
        frame = 0
    return Crop(path=path, pid=pid, camera=cam, frame=frame)


def scan_split(root: Path, split: str, dataset: DatasetName) -> list[Crop]:
    """List crops in one split directory. Filenames only - no image is opened."""
    d = root / split
    if not d.is_dir():
        raise FileNotFoundError(f"missing split directory: {d}")
    crops: list[Crop] = []
    with os.scandir(d) as it:
        for entry in it:
            crop = parse_crop(entry.path, entry.name, dataset)
            if crop is not None:
                crops.append(crop)
    return crops


def detect_dataset(root: Path) -> DatasetName:
    """Distinguish MSMT17 from Market-1501 by identity count.

    Both use the same directory layout and filename convention after conversion, so the
    discriminator is scale: MSMT17 has 1,041 train ids, Market-1501 has 751.
    """
    for split in ("bounding_box_train", "bounding_box_test"):
        if (root / split).is_dir():
            ids = {c.pid for c in scan_split(root, split, "msmt17")}
            return "msmt17" if len(ids) > 900 else "market1501"
    raise FileNotFoundError(f"{root} does not look like a Market-1501/MSMT17 layout")


def build_open_set_split(
    root: Path | str,
    *,
    dataset: DatasetName | None = None,
    split: str = "bounding_box_test",
    n_enrolled: int = 100,
    n_impostor_ids: int = 400,
    enrol_per_id: int = 4,
    enrol_cameras_per_id: int = 1,
    max_genuine_per_id: int = 10,
    max_impostor_per_id: int = 4,
    min_cameras: int = 2,
    seed: int = 0,
    name: str | None = None,
) -> OpenSetSplit:
    """Construct an open-set protocol from a Market-1501/MSMT17 directory.

    Defaults model the deployment: a handful of enrolment crops per resident (you do not
    ask a family to label 200 images), taken from one camera, then recognition from other
    rooms. `n_impostor_ids` is large on purpose - a false-accept rate of 1% cannot be
    estimated from 20 impostors.
    """
    root = Path(root)
    dataset = dataset or detect_dataset(root)
    rng = random.Random(seed)

    crops = scan_split(root, split, dataset)
    by_id: defaultdict[str, list[Crop]] = defaultdict(list)
    for c in crops:
        by_id[c.pid].append(c)

    def pick_enrolment(cs: list[Crop]) -> tuple[list[Crop], set[str]] | None:
        """Choose enrolment cameras that yield enough crops AND leave probes elsewhere.

        Selecting a camera before checking it has `enrol_per_id` crops is what made an
        earlier version silently enrol 60 identities when asked for 100. A split that
        quietly shrinks changes the experiment with no error, so the eligibility test now
        happens *here* and `build_open_set_split` raises if the request cannot be met.
        """
        per_cam: defaultdict[str, list[Crop]] = defaultdict(list)
        for c in cs:
            per_cam[c.camera].append(c)
        cams = sorted(per_cam)
        rng.shuffle(cams)
        chosen: set[str] = set()
        pool: list[Crop] = []
        for cam in cams:
            if len(chosen) >= enrol_cameras_per_id:
                break
            # Never take a camera that would leave no other camera to probe from.
            if not any(c.camera not in chosen | {cam} for c in cs):
                continue
            chosen.add(cam)
            pool.extend(per_cam[cam])
        if len(pool) < enrol_per_id or not chosen:
            return None
        rng.shuffle(pool)
        enrol = pool[:enrol_per_id]
        if not any(c.camera not in chosen for c in cs):
            return None
        return enrol, chosen

    # Eligibility is decided by actually attempting the selection, not by a proxy count.
    eligible: list[str] = []
    prepared: dict[str, tuple[list[Crop], set[str]]] = {}
    for pid in sorted(by_id):
        cs = by_id[pid]
        if len({c.camera for c in cs}) < min_cameras:
            continue
        got = pick_enrolment(cs)
        if got is None:
            continue
        prepared[pid] = got
        eligible.append(pid)

    if len(eligible) < n_enrolled:
        raise ValueError(
            f"{dataset}:{split}: only {len(eligible)} identities can supply "
            f"{enrol_per_id} enrolment crops from {enrol_cameras_per_id} camera(s) and "
            f"still leave cross-camera probes; cannot enrol {n_enrolled}. Lower "
            "n_enrolled or enrol_per_id."
        )

    rng.shuffle(eligible)
    enrolled_ids = eligible[:n_enrolled]
    # Impostors are drawn from ids NOT enrolled. This slice is what guarantees
    # disjointness; validate() then re-checks it rather than trusting this line.
    enrolled_set = set(enrolled_ids)
    remaining = [pid for pid in sorted(by_id) if pid not in enrolled_set]
    rng.shuffle(remaining)
    impostor_ids = remaining[:n_impostor_ids]
    if not impostor_ids:
        raise ValueError(f"{dataset}: no identities left over to act as impostors")
    if len(impostor_ids) < n_impostor_ids:
        raise ValueError(
            f"{dataset}:{split}: only {len(impostor_ids)} identities available as "
            f"impostors, requested {n_impostor_ids}; a FAR of 1% cannot be estimated "
            "from too few impostors"
        )

    out = OpenSetSplit(name=name or f"{dataset}:{split}")

    for pid in enrolled_ids:
        enrol, enrol_cams = prepared[pid]
        probes = [c for c in by_id[pid] if c.camera not in enrol_cams]
        rng.shuffle(probes)
        out.enrolment[pid] = enrol
        out.enrol_cameras[pid] = enrol_cams
        out.genuine[pid] = probes[:max_genuine_per_id]

    if len(out.enrolment) != n_enrolled:  # pragma: no cover - guarded above
        raise AssertionError(
            f"asked for {n_enrolled} enrolled ids, built {len(out.enrolment)}"
        )

    for pid in impostor_ids:
        cs = list(by_id[pid])
        rng.shuffle(cs)
        out.impostors.extend(cs[:max_impostor_per_id])

    out.validate()
    return out


def split_by_identity(
    split: OpenSetSplit, frac: float = 0.5, seed: int = 0
) -> tuple[OpenSetSplit, OpenSetSplit]:
    """Partition into fit/test halves with disjoint identities.

    A threshold chosen on the same probes it is reported on is fitted to noise. Splitting
    by *identity* rather than by probe is what makes the test half independent: two probes
    of the same person are not independent samples.
    """
    rng = random.Random(seed)
    enrolled = sorted(split.enrolment)
    impostor_ids = sorted({c.pid for c in split.impostors})
    rng.shuffle(enrolled)
    rng.shuffle(impostor_ids)

    n_e = max(1, int(len(enrolled) * frac))
    n_i = max(1, int(len(impostor_ids) * frac))
    halves = []
    for tag, e_ids, i_ids in (
        ("fit", set(enrolled[:n_e]), set(impostor_ids[:n_i])),
        ("test", set(enrolled[n_e:]), set(impostor_ids[n_i:])),
    ):
        part = OpenSetSplit(name=f"{split.name}[{tag}]")
        for pid in e_ids:
            part.enrolment[pid] = list(split.enrolment[pid])
            part.enrol_cameras[pid] = set(split.enrol_cameras[pid])
            part.genuine[pid] = list(split.genuine.get(pid, []))
        part.impostors = [c for c in split.impostors if c.pid in i_ids]
        part.validate()
        halves.append(part)
    return halves[0], halves[1]


__all__ = [
    "Crop",
    "OpenSetSplit",
    "DatasetName",
    "parse_crop",
    "scan_split",
    "detect_dataset",
    "build_open_set_split",
    "split_by_identity",
]
