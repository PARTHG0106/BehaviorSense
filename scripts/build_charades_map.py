"""Generate the full Charades -> 20-class taxonomy mapping, for human review.

`configs/taxonomy.yaml` embeds a hand-checked subset of the 157 Charades classes and
points here for the full table. This script exists because hand-typing 157 mappings
invites transcription errors, but silently auto-mapping them invites label noise that is
unrecoverable after training - so the output is a REVIEWABLE artifact (YAML + a TSV with
the rule that fired per class), not a direct edit of the taxonomy.

Design decisions:

  - First-match-wins ordered rules. Charades names are short and formulaic
    ("Walking through a doorway", "Putting a blanket somewhere"), so keyword rules are
    reliable; ordering resolves conflicts explicitly ("Sitting down" must match the
    transition rule before the generic "sitting" posture rule).
  - Hand-object manipulations (holding/putting/taking/throwing X) map to `other_idle`.
    They carry no signal any behavioural feature consumes, and the taxonomy's reject
    class exists precisely so ambiguous windows have somewhere honest to go.
  - Camera-performance classes (taking a picture, playing with a camera) are DROPPED:
    they are artefacts of how Charades was crowdsourced, not activities an elderly
    resident performs, and training on them teaches the model the dataset, not the task.
  - Every class that only matched the fallback is listed on stdout for eye review;
    the exit code is nonzero if the input did not contain exactly 157 classes.

Usage:
    python scripts/build_charades_map.py --classes Charades_v1_classes.txt \
        --out configs/charades_map.yaml
    python scripts/build_charades_map.py --selftest      # rule-engine check, no dataset
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# (pattern, target, why). First match wins. `None` target = drop from training.
RULES: tuple[tuple[str, str | None, str], ...] = (
    # -- transitions before postures: "sitting down" must not match "sitting" ----------
    (r"\bsitting down\b|\bsitting in\b.*\bbed\b$", "sitting_down", "explicit transition"),
    (r"\bstanding up\b|\bgetting up\b", "standing_up", "explicit transition"),
    # -- critical / posture ------------------------------------------------------------
    (r"\bfall|\bfalling\b|\btripping\b", "falling", "fall event"),
    (r"\blying\b|\blaying\b|\bawakening\b|\bwaking\b|\basleep\b|\bnap\b",
     "lying_down", "recumbent posture"),
    (r"\bsitting\b|\bsit\b", "sitting", "seated posture"),
    (r"\bstanding\b(?! up)", "standing", "upright posture"),
    # -- mobility ----------------------------------------------------------------------
    (r"\bwalk|\brunning\b|\bjogging\b", "walking", "locomotion"),
    (r"\breaching\b|\bbending\b|\bcrouching\b|\bkneeling\b|\bstretching\b",
     "bending_reaching", "flexion movement"),
    # -- nutrition / health ------------------------------------------------------------
    (r"\beating\b|\bsnack|\bsandwich\b|\bmeal\b", "eating", "food intake"),
    (r"\bdrink|\bpouring\b.*\b(glass|cup|water)\b|\bglass of\b", "drinking", "fluid intake"),
    (r"\bcook|\bstove\b|\bfood\b.*\bprepar|\bkitchen\b.*\bprepar", "cooking_food_prep",
     "food preparation"),
    (r"\bmedicine\b|\bmedication\b|\bpill\b", "taking_medication", "medication"),
    # -- leisure / social --------------------------------------------------------------
    (r"\btelevision\b|\btv\b|\bwatching something\b", "watching_tv", "screen leisure"),
    (r"\bbook\b|\breading\b|\bmagazine\b|\bnotebook\b", "reading", "reading material"),
    (r"\bphone\b|\btexting\b", "using_phone", "phone use"),
    (r"\blaughing\b|\bsmiling\b|\btalking\b|\bhugging\b(?!.*\bpillow\b)",
     "interacting_with_person", "social behaviour"),
    # -- iadl / selfcare ---------------------------------------------------------------
    (r"\btidying\b|\bcleaning\b|\bwashing\b(?!.*\b(face|hands|themselves)\b)|\bvacuum|"
     r"\bbroom\b|\blaundry\b|\bdishes\b|\bmopping\b|\bdusting\b|\bfixing\b",
     "cleaning_housework", "housework"),
    (r"\bgrooming\b|\bhair\b|\bteeth\b|\bbrushing\b|\btowel\b|\bwashing\b.*"
     r"\b(face|hands|themselves)\b|\bdressing\b|\bundressing\b|\bshoes\b|\bclothes\b",
     "personal_hygiene", "self-care"),
    # -- dataset artefacts: drop, do not teach -----------------------------------------
    (r"\bcamera\b|\btaking a picture\b|\bphotograph", None, "crowdsourcing artefact"),
    (r"\bsneezing\b|\bcoughing\b", None, "transient reflex, not an activity"),
)

FALLBACK = "other_idle"

# Small embedded sample for --selftest: real Charades classes with known-correct targets.
# NOT the full list - the real file ships with the dataset and is verified by count.
SELFTEST_CASES: tuple[tuple[str, str, str | None], ...] = (
    ("c093", "Walking through a doorway", "walking"),
    ("c106", "Standing up from somewhere", "standing_up"),
    ("c107", "Sitting down in a chair", "sitting_down"),
    ("c108", "Sitting in a chair", "sitting"),
    ("c059", "Lying on a bed", "lying_down"),
    ("c063", "Eating a sandwich", "eating"),
    ("c070", "Drinking from a cup", "drinking"),
    ("c027", "Cooking something", "cooking_food_prep"),
    ("c077", "Taking medicine", "taking_medication"),
    ("c151", "Watching television", "watching_tv"),
    ("c082", "Reading a book", "reading"),
    ("c110", "Playing with a phone", "using_phone"),
    ("c130", "Tidying up a table", "cleaning_housework"),
    ("c009", "Washing their hands", "personal_hygiene"),
    ("c141", "Taking a picture", None),
    ("c020", "Sneezing", None),
    ("c096", "Holding a pillow", "other_idle"),
    ("c132", "Putting a blanket somewhere", "other_idle"),
    ("c008", "Laughing at a picture", "interacting_with_person"),
    ("c150", "Someone is awakening in bed", "lying_down"),
)


def map_class(name: str) -> tuple[str | None, str]:
    """Return (target, rule_description) for one Charades class name."""
    low = name.lower()
    for pattern, target, why in RULES:
        if re.search(pattern, low):
            return target, why
    return FALLBACK, "fallback (REVIEW)"


def slugify(cid: str, name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"{cid}_{slug}"


def parse_classes(path: Path) -> list[tuple[str, str]]:
    """Parse Charades_v1_classes.txt: lines of `c093 Walking through a doorway`."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(c\d{3})\s+(.+)$", line)
        if not m:
            raise SystemExit(f"unparseable line in {path}: {line!r}")
        out.append((m.group(1), m.group(2)))
    return out


def selftest() -> int:
    """Verify the rule engine on classes with known-correct targets."""
    failures = []
    for cid, name, expected in SELFTEST_CASES:
        got, why = map_class(name)
        status = "ok" if got == expected else "WRONG"
        if got != expected:
            failures.append((cid, name, expected, got, why))
        print(f"  [{status}] {cid} {name!r} -> {got} ({why})")
    if failures:
        print(f"\n{len(failures)} rule failures:")
        for cid, name, exp, got, why in failures:
            print(f"  {cid} {name!r}: expected {exp}, got {got} via {why}")
        return 1
    print(f"\nself-test passed: {len(SELFTEST_CASES)}/{len(SELFTEST_CASES)} known classes "
          "mapped correctly")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--classes", default=None, help="path to Charades_v1_classes.txt")
    ap.add_argument("--out", default="configs/charades_map.yaml")
    ap.add_argument("--review-tsv", default="configs/charades_map_review.tsv")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.classes:
        raise SystemExit("--classes is required (or use --selftest)")

    classes = parse_classes(Path(args.classes))
    if len(classes) != 157:
        print(f"WARNING: expected 157 Charades classes, found {len(classes)} - "
              "wrong or truncated file?", file=sys.stderr)

    rows = []
    for cid, name in classes:
        target, why = map_class(name)
        rows.append((cid, name, slugify(cid, name), target, why))

    counts = Counter(t if t is not None else "(dropped)" for *_, t, _ in rows)
    defaulted = [(cid, name) for cid, name, _, t, why in rows if "REVIEW" in why]

    yaml_lines = [
        "# Charades -> BehaviorSense taxonomy. GENERATED by scripts/build_charades_map.py.",
        "# Review before use: every `other_idle` from the fallback rule is listed in",
        f"# {args.review_tsv} with the rule that fired. Do not edit by hand; edit RULES.",
        "charades:",
    ]
    for cid, name, slug, target, _ in sorted(rows):
        if target is None:
            yaml_lines.append(f"  {slug}: {{drop: true}}   # {name}")
        else:
            yaml_lines.append(f"  {slug}: {target}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    tsv = Path(args.review_tsv)
    tsv.write_text(
        "charades_id\tname\ttarget\trule\n"
        + "\n".join(f"{cid}\t{name}\t{t or 'DROP'}\t{why}" for cid, name, _, t, why in rows)
        + "\n",
        encoding="utf-8",
    )

    print(f"wrote {out} and {tsv}")
    print("\nclass distribution:")
    for target, n in counts.most_common():
        print(f"  {target:<24} {n:>4}")
    if defaulted:
        print(f"\n{len(defaulted)} classes hit the fallback - review these by eye:")
        for cid, name in defaulted[:20]:
            print(f"  {cid} {name}")
        if len(defaulted) > 20:
            print(f"  ... and {len(defaulted) - 20} more (full list in the TSV)")

        # The fallback list alone was not enough. `walking` came out of the real extraction
        # with 545 of 35,698 validation windows (1.5%) at F1 0.093 - implausible for the
        # most pose-separable activity in a corpus of people moving around their homes - and
        # nothing in the existing output pointed at why. Group the fallback by candidate
        # theme so a misrouted FAMILY of classes is visible rather than 60 lines of prose.
        #
        # These probes are diagnostic only: they never assign a label. Adding a rule is a
        # human decision, because a wrong rule teaches the model a wrong label and then
        # costs a 6-hour re-extraction to undo.
        PROBES = (
            ("locomotion?", r"\bgoing\b|\bentering\b|\bleaving\b|\bexiting\b|\bstairs?\b|"
                            r"\bdoorway\b|\bsomewhere\b.*\bwalk|\bthrough a door"),
            ("posture?", r"\bstand|\bsit\b|\blie\b|\blying\b|\bcrouch|\bkneel"),
            ("object-hold?", r"\bholding\b|\bputting\b|\btaking\b|\bgrasping\b|\bcarrying\b"),
            ("door/window?", r"\bdoor\b|\bwindow\b|\bcloset\b|\bcabinet\b|\bdrawer\b"),
            ("light/switch?", r"\blight\b|\bswitch\b|\blamp\b"),
        )
        print("\nfallback grouped by candidate theme (diagnostic only - no rule is implied):")
        matched: set[str] = set()
        for label, pat in PROBES:
            hits = [(c, n) for c, n in defaulted if re.search(pat, n, re.I)]
            matched.update(c for c, _ in hits)
            if hits:
                print(f"  {label:<14} {len(hits):>3} class(es)")
                for cid, name in hits[:6]:
                    print(f"                 {cid} {name}")
                if len(hits) > 6:
                    print(f"                 ... and {len(hits) - 6} more")
        rest = [(c, n) for c, n in defaulted if c not in matched]
        print(f"  {'unthemed':<14} {len(rest):>3} class(es)")
        print("\nIf 'locomotion?' is non-empty, `walking` is being starved by the rules and")
        print("every one of those windows is also inflating other_idle. Changing RULES means")
        print("RE-EXTRACTING the shards (labels are baked in at extraction, ~6 h), so decide")
        print("once, with this list in front of you.")


if __name__ == "__main__":
    main()
