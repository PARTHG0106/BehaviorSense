"""Run the caregiver state/translation UI contract tests with Node's built-in runner."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    result = subprocess.run(
        ["node", "--test", "--test-reporter=tap", "tests/test_caregiver.mjs", "tests/test_report_data.mjs"],
        cwd=ROOT, text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=60,
    )
    print(result.stdout.encode("ascii", errors="backslashreplace").decode("ascii"))
    counts = re.search(r"# tests (\d+)", result.stdout)
    failures = re.search(r"# fail (\d+)", result.stdout)
    if counts and failures:
        total, failed = int(counts[1]), int(failures[1])
        print(f"{total - failed}/{total} passed")
    sys.exit(result.returncode if counts and failures else 1)
