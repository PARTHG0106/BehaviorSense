"""Regression checks for source isolation, subprocess cleanup and source inclusion."""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_tests import (SUITE_TIMEOUT, copy_test_source, print_summary, run_suite,  # noqa: E402
                       summarize_suite, test_environment)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class RunnerTests(unittest.TestCase):
    def test_snapshot_includes_data_code_and_excludes_operator_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, snapshot = Path(temporary) / "repo", Path(temporary) / "snapshot"
            files = {
                "src/behaviorsense/data/simulator.py": "SOURCE",
                "tests/test_example.py": "TEST",
                "configs/taxonomy.yaml": "classes: []",
                "data/residents.json": "PRIVATE",
                "runs/fall/best.pt": "EXISTING CHECKPOINT",
                "weights/model.pth": "WEIGHTS",
                "gallery.json": "PRIVATE GALLERY",
                "kaggle.json": "PRIVATE CREDENTIAL",
                "configs/local.yaml": "PRIVATE CONFIG",
                "configs/site.local.yaml": "PRIVATE CONFIG",
                "src/behaviorsense/__pycache__/cached.py": "CACHE",
                "web/node_modules/vendor/index.js": "DEPENDENCY",
            }
            for name, content in files.items():
                write(source / name, content)
            copy_test_source(source, snapshot)
            copied = {p.relative_to(snapshot).as_posix() for p in snapshot.rglob("*") if p.is_file()}
            self.assertEqual(copied, {"src/behaviorsense/data/simulator.py",
                                      "tests/test_example.py", "configs/taxonomy.yaml"})
            # A test following a trainer's default output path can only overwrite its copy.
            write(snapshot / "tests/test_example.py", """from pathlib import Path
root = Path(__file__).resolve().parents[1]
assert Path.cwd() == root
out = root / 'runs/fall/best.pt'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text('SYNTHETIC', encoding='utf-8')
print('1/1 passed')
""")
            code, output = run_suite(snapshot, "tests/test_example.py", threads=1, timeout=20)
            self.assertEqual(code, 0, output)
            self.assertEqual((source / "runs/fall/best.pt").read_text(), "EXISTING CHECKPOINT")
            self.assertEqual((snapshot / "runs/fall/best.pt").read_text(), "SYNTHETIC")

    def test_snapshot_rejects_a_destination_inside_the_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            with self.assertRaises(ValueError):
                copy_test_source(source, source / "nested")

    def test_snapshot_does_not_follow_external_source_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source, snapshot = base / "repo", base / "snapshot"
            write(base / "outside/private.py", "PRIVATE")
            (source / "src").mkdir(parents=True)
            try:
                (source / "src/external").symlink_to(base / "outside", target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlinks unavailable: {exc}")
            copy_test_source(source, snapshot)
            self.assertFalse((snapshot / "src/external/private.py").exists())

    def test_timeout_stops_training_descendants(self):
        self.check_descendant_cleanup(parent_exits=False)

    def test_timeout_stops_orphans_after_the_suite_exits(self):
        # The descendant still holds stdout open after the suite process disappears.
        # Looking up that dead PID with taskkill cannot clean up this process tree.
        self.check_descendant_cleanup(parent_exits=True)

    def check_descendant_cleanup(self, parent_exits: bool):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary)
            write(snapshot / "tests/child.py", """from pathlib import Path
import time
with Path('heartbeat.txt').open('a') as stream:
    while True:
        stream.write('alive\\n')
        stream.flush()
        time.sleep(0.05)
""")
            parent = """from pathlib import Path
import subprocess, sys, time
subprocess.Popen([sys.executable, str(Path(__file__).with_name('child.py'))])
"""
            if not parent_exits:
                parent += "time.sleep(60)\n"
            write(snapshot / "tests/parent.py", parent)
            code, output = run_suite(snapshot, "tests/parent.py", threads=1, timeout=2)
            self.assertEqual(code, SUITE_TIMEOUT)
            self.assertIn("process tree stopped", output)
            heartbeat = snapshot / "heartbeat.txt"
            self.assertTrue(heartbeat.exists(), "child never started; cleanup was not exercised")
            size = heartbeat.stat().st_size
            time.sleep(0.2)
            self.assertEqual(heartbeat.stat().st_size, size, "child continued after timeout")

    def test_suite_status_requires_both_completed_assertions_and_a_clean_exit(self):
        cases = [
            (0, "3/3 passed (2 skipped)", "PASS", True),
            (0, "0/0 passed (2 skipped)", "SKIP", True),
            (0, "0/0 passed", "INCOMPLETE", False),
            (0, "fixture setup only", "INCOMPLETE", False),
            (0, "4/3 passed", "INCOMPLETE", False),
            (1, "fixture setup failed", "FAIL", False),
            (1, "3/3 passed\ncrashed during teardown", "FAIL", True),
            (0, "2/3 passed", "FAIL", True),
            (SUITE_TIMEOUT, "3/3 passed", "TIMEOUT", False),
        ]
        for code, output, status, has_counts in cases:
            with self.subTest(code=code, output=output):
                result = summarize_suite("fixture", code, output, 1.0)
                self.assertEqual(result.status, status)
                self.assertEqual(result.has_counts, has_counts)

    def test_summary_distinguishes_incomplete_totals_failures_timeouts_and_skips(self):
        results = [
            summarize_suite("passed", 0, "3/3 passed (2 skipped)", 1.0),
            summarize_suite("failed", 1, "1/2 passed", 1.0),
            summarize_suite("timed out", SUITE_TIMEOUT, "99/99 passed", 1.0),
            summarize_suite("no assertions", 0, "", 1.0),
        ]
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertFalse(print_summary(results))
        text = output.getvalue()
        self.assertIn("Suites: 1/4 passed; 1 failed; 1 timed out; 1 incomplete; 0 skipped", text)
        self.assertIn("Reported test counts from 2/4 suites: 4/5 passed; 2 skipped", text)
        self.assertIn("Test totals are incomplete", text)
        self.assertIn("FAILING SUITES: failed", text)
        self.assertIn("TIMED OUT SUITES: timed out", text)
        self.assertIn("INCOMPLETE SUITES: no assertions", text)
        self.assertNotIn("99/99", text)
        self.assertNotIn("ALL SUITES PASSED", text)

    def test_summary_does_not_count_an_entirely_skipped_suite_as_passed(self):
        results = [summarize_suite("unsupported", 0, "0/0 passed (2 skipped)", 1.0)]
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertTrue(print_summary(results))
        text = output.getvalue()
        self.assertIn("Suites: 0/1 passed; 0 failed; 0 timed out; 0 incomplete; 1 skipped", text)
        self.assertIn("0/0 passed; 2 skipped", text)
        self.assertNotIn("ALL SUITES PASSED", text)

    def test_runner_cli_fails_when_a_suite_times_out_after_printing_a_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            shutil.copy2(ROOT / "run_tests.py", source / "run_tests.py")
            write(source / "tests/test_perception.py", "print('3/3 passed (2 skipped)')\n")
            write(source / "tests/test_activity.py", "import time\n"
                  "print('99/99 passed', flush=True)\ntime.sleep(60)\n")
            result = subprocess.run(
                [sys.executable, str(source / "run_tests.py"), "--suite", "test_perception",
                 "--suite", "test_activity", "--timeout", "2"],
                cwd=source, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            summary = result.stdout.split("\nSUMMARY\n", 1)[-1]
            self.assertIn("Suites: 1/2 passed; 0 failed; 1 timed out", summary)
            self.assertIn("Reported test counts from 1/2 suites: 3/3 passed; 2 skipped", summary)
            self.assertIn("TIMED OUT SUITES: Agent 2 activity", summary)
            self.assertNotIn("99/99", summary)
            self.assertNotIn("ALL SUITES PASSED", summary)

    def test_heartbeat_keeps_output_and_the_original_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary)
            write(snapshot / "tests/slow.py", "import time\n"
                  "print('before waiting', flush=True)\ntime.sleep(60)\n")
            progress = io.StringIO()
            started = time.monotonic()
            with patch("run_tests.HEARTBEAT_SECONDS", 0.05), redirect_stdout(progress):
                code, output = run_suite(snapshot, "tests/slow.py", threads=1, timeout=1)
            self.assertEqual(code, SUITE_TIMEOUT)
            self.assertIn("before waiting", output)
            self.assertIn("slow still running", progress.getvalue())
            self.assertLess(time.monotonic() - started, 10, "heartbeat reset the suite timeout")

    def test_environment_uses_snapshot_imports_and_bounded_threads(self):
        env = test_environment(Path("snapshot"), threads=2)
        self.assertEqual(env["PYTHONPATH"], str(Path("snapshot/src")))
        self.assertEqual(env["PYTHONUNBUFFERED"], "1")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            self.assertEqual(env[name], "2")

    @unittest.skipUnless(shutil.which("git"), "Git is needed to verify ignore rules")
    def test_gitignore_keeps_python_data_package_visible(self):
        with tempfile.TemporaryDirectory() as temporary:
            probe = Path(temporary)
            shutil.copy2(ROOT / ".gitignore", probe / ".gitignore")
            write(probe / "empty-global-ignore", "")
            candidates = ["src/behaviorsense/data/simulator.py", "data/shards/example.npz",
                          "gallery.json", "kaggle.json", "runs/fall/best.pt"]
            for name in candidates:
                write(probe / name, "fixture")
            subprocess.run(["git", "init", "--quiet", str(probe)], check=True,
                           capture_output=True)
            command = ["git", "-C", str(probe), "-c",
                       f"core.excludesFile={probe / 'empty-global-ignore'}",
                       "check-ignore", "--no-index", "--", *candidates]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(set(result.stdout.splitlines()), set(candidates[1:]))


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(RunnerTests))
    passed = result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
    total = result.testsRun - len(result.skipped)
    print(f"{passed}/{total} passed ({len(result.skipped)} skipped)")
    sys.exit(0 if result.wasSuccessful() else 1)
