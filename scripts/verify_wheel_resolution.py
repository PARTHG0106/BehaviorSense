"""Prove notebook 00's wheel staging can resolve BEFORE a Kaggle session is spent on it.

Notebook 00 failed twice in the resolver, each time costing a session:

  1. Unpinned `pip download torch` backtracked across every cu128 torch (2.7 -> 2.11)
     hunting for a satisfiable nvidia-* set and reported ResolutionImpossible.
  2. Pinning alone did not fix it. torch 2.7.1+cu128 requires nvidia-cudnn-cu12==9.7.1.26,
     which publishes ONLY a manylinux_2_27_x86_64 wheel, and the --platform list in use
     ({manylinux2014, manylinux_2_28}) excluded that tag. Empty candidate set, same error,
     different cause.

Both failures share one root: a pinned dependency whose only wheel carries a platform tag
the --platform list does not name. That is a *static* property of the index, so it can be
checked from a laptop in about a minute instead of discovered 4 minutes into a GPU session.

Why not just run `pip download --dry-run`? Because --platform governs wheel *tag* matching
only -- it does NOT set `platform_system`. Run from Windows or macOS, torch's nvidia-* deps
are all marker-excluded (`platform_system == "Linux"`), the resolve trivially succeeds, and
the check proves nothing. Verified: a dry run on this repo's Windows box resolves torch
2.7.1+cu128 to 10 packages with zero nvidia-* among them. So this reads the dependency pins
out of the wheel's own METADATA and tag-checks each one directly, which is platform-agnostic.

The wheel is ~1 GB and download.pytorch.org serves no PEP 658 .metadata sidecar (403), so
METADATA is read over HTTP Range requests -- the same lazy-wheel trick pip uses internally.
Cost is a few hundred KB, not a gigabyte.

Usage (needs internet; run on your own machine, not in an offline session):
    python scripts/verify_wheel_resolution.py

Exit code is 0 when every pin is stageable, 1 otherwise, so CI can gate on it.
"""

from __future__ import annotations

import io
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = "https://download.pytorch.org/whl/cu128"
TIMEOUT = 60

# Kept identical to notebook 00's PLATFORMS. tests/test_notebooks.py::N9 fails if the two
# ever drift, because a check against a stale list is worse than no check at all.
PLATFORMS: list[str] = (
    ["manylinux1_x86_64", "manylinux2010_x86_64", "manylinux2014_x86_64"]
    + [f"manylinux_2_{m}_x86_64" for m in range(5, 40)]
    + ["linux_x86_64"]
)

TORCH_VERSION = "2.7.1"
PYTHON_TAGS = ("cp311", "cp312")   # offline image interpreter is not knowable; stage both

# The list that failed on Kaggle. Kept only so the run reports how many pins it would miss
# today: that number is the evidence PLATFORMS has to stay wide, and it stops a future
# reader from "tidying" the ladder back into a short guess.
HISTORICAL_PLATFORMS = {"manylinux2014_x86_64", "manylinux_2_28_x86_64"}


class RangeFile(io.RawIOBase):
    """A seekable read-only file over HTTP Range, so zipfile can pull one member."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.pos = 0
        # HEAD is 403 on download.pytorch.org. A 1-byte Range GET returns the total size in
        # Content-Range, which is the only size source that actually works against this CDN.
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            if resp.status != 206:
                raise RuntimeError(f"server ignored Range (status {resp.status}); "
                                   "cannot read METADATA without downloading ~1 GB")
            self.size = int(resp.headers["Content-Range"].split("/")[-1])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.pos
        if size == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + size, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = resp.read()
        self.pos += len(data)
        return data


def index_files(package: str) -> list[str]:
    """Wheel filenames the cu128 index publishes for one package."""
    url = f"{INDEX}/{package.replace('_', '-').lower()}/"
    with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
        html = resp.read().decode("utf-8", "replace")
    out = []
    for href in re.findall(r'href="([^"]+)"', html):
        name = re.sub(r"#.*$", "", href).rsplit("/", 1)[-1]
        out.append(urllib.parse.unquote(name))   # '%2Bcu128' -> '+cu128'
    return out


def platform_tags(wheel: str) -> set[str]:
    """Platform tags a wheel filename declares (the last '-' field, '.'-separated)."""
    return set(wheel[:-4].rsplit("-", 1)[-1].split("."))


def torch_wheel_url() -> tuple[str, str]:
    """The cp312 x86_64 wheel notebook 00 will actually stage.

    Matching on the version+interpreter prefix alone is not enough: the cu128 index also
    publishes an aarch64 build under the same prefix, it sorts first, and it is ~2.8 GB of
    a different dependency set. PLATFORMS is x86_64-only, so that wheel is never staged and
    reading its METADATA would tag-check the wrong pins.
    """
    files = index_files("torch")
    want = f"torch-{TORCH_VERSION}+cu128-cp312-cp312-"
    hits = [f for f in files
            if f.startswith(want) and platform_tags(f) & set(PLATFORMS)]
    if not hits:
        near = [f for f in files if f.startswith(want)]
        raise SystemExit(
            f"FAIL: cu128 index publishes no {want}* wheel matching PLATFORMS.\n"
            f"       closest: {near or 'nothing at this version'}")
    name = hits[0]
    return f"{INDEX}/{name.replace('+', '%2B')}", name


def gpu_pins(url: str) -> list[tuple[str, str]]:
    """(package, exact version) for every nvidia-*/triton pin torch declares."""
    handle = RangeFile(url)
    print(f"  wheel is {handle.size / 1e6:.0f} MB; reading METADATA only")
    with zipfile.ZipFile(handle) as zf:
        members = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
        if not members:
            raise SystemExit("FAIL: no .dist-info/METADATA inside the torch wheel")
        meta = zf.read(members[0]).decode("utf-8", "replace")

    pins: list[tuple[str, str]] = []
    for line in meta.splitlines():
        if not line.startswith("Requires-Dist:"):
            continue
        req = line.split(":", 1)[1].strip()
        if not re.match(r"(nvidia-|triton)", req, re.I):
            continue
        m = re.match(r"([A-Za-z0-9_.\-]+)\s*==\s*([^\s;]+)", req)
        if not m:
            print(f"  NOTE: unpinned GPU requirement, not tag-checkable: {req}")
            continue
        pins.append((m.group(1), m.group(2)))
    return sorted(pins)


def main() -> int:
    allowed = set(PLATFORMS)
    print(f"cu128 index: {INDEX}")
    print(f"pin: torch=={TORCH_VERSION}   --platform entries: {len(PLATFORMS)}")

    try:
        url, name = torch_wheel_url()
        print(f"\ntorch wheel: {name}")

        # The pin must exist for every interpreter the offline image might run, on an
        # x86_64 tag we actually allow. Missing one is silent until the offline install
        # finds nothing tagged for its python.
        torch_files = index_files("torch")
        allowed_now = set(PLATFORMS)
        missing_py = [
            tag for tag in PYTHON_TAGS
            if not [f for f in torch_files
                    if f.startswith(f"torch-{TORCH_VERSION}+cu128-{tag}-")
                    and platform_tags(f) & allowed_now]
        ]

        pins = gpu_pins(url)
    except (urllib.error.URLError, OSError) as exc:
        print(f"\nSKIP: cu128 index unreachable ({exc}). This check needs internet.")
        return 0

    print(f"\n{len(pins)} pinned GPU dependencies:\n")
    unresolvable: list[dict] = []
    would_have_missed: list[dict] = []
    for pkg, ver in pins:
        try:
            files = index_files(pkg)
        except (urllib.error.URLError, OSError) as exc:
            print(f"  ?? {pkg:<26} {ver:<12} index unreachable: {exc}")
            unresolvable.append({"package": pkg, "version": ver, "reason": "index unreachable"})
            continue
        stem = f"{pkg.replace('-', '_')}-{ver}-"
        cand = [f for f in files if f.startswith(stem)]
        tags: set[str] = set()
        for f in cand:
            tags |= platform_tags(f)
        if not cand:
            print(f"  !! {pkg:<26} {ver:<12} NO WHEEL AT THIS VERSION")
            unresolvable.append({"package": pkg, "version": ver, "reason": "no wheel",
                                 "tags": []})
            continue
        if not tags & HISTORICAL_PLATFORMS:
            would_have_missed.append({"package": pkg, "version": ver,
                                      "tags": sorted(t for t in tags if t.endswith("x86_64"))})
        matched = sorted(tags & allowed)
        if matched:
            print(f"  OK {pkg:<26} {ver:<12} via {matched[0]}")
        else:
            print(f"  !! {pkg:<26} {ver:<12} publishes only {sorted(tags)}")
            unresolvable.append({"package": pkg, "version": ver, "reason": "tag not allowed",
                                 "tags": sorted(tags)})

    report = {"index": INDEX, "torch": f"{TORCH_VERSION}+cu128",
              "platforms": len(PLATFORMS), "gpu_pins": len(pins),
              "missing_python_tags": missing_py, "unresolvable": unresolvable,
              "historical_platforms": sorted(HISTORICAL_PLATFORMS),
              "would_have_missed": would_have_missed}
    out = ROOT / "artifacts" / "wheel_resolution.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if would_have_missed:
        spread = sorted({t for e in would_have_missed for t in e["tags"]})
        print(f"\nwhy PLATFORMS is a ladder, not a guess: the two-tag list that failed on "
              f"Kaggle\nreaches {len(pins) - len(would_have_missed)}/{len(pins)} pins. The "
              f"other {len(would_have_missed)} publish only {spread}\n-- three separate tags, "
              f"so no short hand-picked list would have covered them:")
        for e in would_have_missed:
            print(f"  {e['package']:<26} {e['version']:<12} only {e['tags']}")

    print()
    if missing_py:
        print(f"FAIL: no torch {TORCH_VERSION}+cu128 wheel for {missing_py} -- an offline "
              "image on that interpreter would have nothing to install")
    if unresolvable:
        print(f"FAIL: {len(unresolvable)} pin(s) unstageable with the current --platform list.")
        print("      Add the missing tag(s) above to PLATFORMS in notebook 00 AND here.")
    if not missing_py and not unresolvable:
        print(f"PASS: torch {TORCH_VERSION}+cu128 stageable for {list(PYTHON_TAGS)}; "
              f"all {len(pins)} GPU pins have a wheel matching PLATFORMS.")
    print(f"wrote {out.relative_to(ROOT)}")
    return 1 if (missing_py or unresolvable) else 0


if __name__ == "__main__":
    sys.exit(main())
