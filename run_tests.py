"""Run every test suite and report a single verdict.

Exists because "all tests pass" is a claim that should be reproducible with one command,
and because the suites have different runtimes: the CPU logic suites finish in seconds
while the training suite spawns real subprocess training runs and takes minutes. Splitting
them lets the fast set gate ordinary edits.

    python run_tests.py           # fast suites (seconds)
    python run_tests.py --all     # includes the training suite (~10-20 min)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

FAST = [
    ("Agent 1 perception", "tests/test_perception.py"),
    ("Agent 2 activity", "tests/test_activity.py"),
    ("Agent 3 behaviour", "tests/test_behaviour.py"),
    ("Agent 4 verifier", "tests/test_verifier.py"),
    ("Agent 4 reporter", "tests/test_reporter.py"),
    ("Open-set ReID", "tests/test_reid_eval.py"),
    ("Service API", "tests/test_service.py"),
    ("Pipeline + ensemble", "tests/test_pipeline.py"),
    ("Toyota Smarthome ingestion", "tests/test_toyota.py"),
    # M1 is the load-bearing one: it proves a class the source corpus cannot label
    # receives exactly zero gradient, which is what stops 365,492 Toyota windows
    # teaching the model that sitting never happens.
    ("Partial-label multi-corpus", "tests/test_multicorpus.py"),
    # Reads notebook 05's own AST and compares it to web/dev_backend.py. Fast, and it is the
    # only thing standing between the front end and a shape the GPU never sends.
    ("Web /video contract", "tests/test_web_contract.py"),
    ("Notebook validation", "tests/test_notebooks.py"),
    # Executes notebook 04's real cell sources. Listed here, not in SLOW, because every
    # failure it has caught cost a 10-minute GPU session; a ~40 s local run that gates
    # ordinary edits is the whole point.
    ("Notebook 04 end-to-end", "tests/test_notebook04_e2e.py"),
]
SLOW = [("Training stack", "tests/test_training.py")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="include the slow training suite")
    args = ap.parse_args()

    suites = FAST + (SLOW if args.all else [])
    results: list[tuple[str, bool, str, float]] = []

    for name, path in suites:
        print(f"\n{'=' * 70}\n{name}  ({path})\n{'=' * 70}")
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(ROOT / path)], cwd=ROOT, capture_output=True, text=True
        )
        dt = time.time() - t0
        tail = proc.stdout.strip().splitlines()
        summary = next((l for l in reversed(tail) if "passed" in l), "no summary")
        for line in tail:
            if line.startswith("  FAIL") or line.startswith("  ERROR"):
                print(line)
        print(f"{summary}  [{dt:.1f}s]")
        if proc.returncode != 0 and not tail:
            print(proc.stderr[-800:])
        results.append((name, proc.returncode == 0, summary, dt))

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    width = max(len(n) for n, *_ in results)
    total_pass = total = 0
    for name, ok, summary, dt in results:
        counts = summary.split()[0] if "/" in summary else "?"
        if "/" in counts:
            p, t = counts.split("/")
            total_pass += int(p)
            total += int(t)
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {counts:>8}  {dt:6.1f}s")

    failed = [n for n, ok, *_ in results if not ok]
    print(f"\n{total_pass}/{total} individual tests passed across {len(results)} suites")
    if not args.all:
        print("(training suite skipped; run with --all)")
    if failed:
        print(f"FAILING SUITES: {', '.join(failed)}")
        sys.exit(1)
    print("ALL SUITES PASSED")


if __name__ == "__main__":
    main()
