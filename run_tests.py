"""Run CPU suites from a temporary copy of the project source.

No local datasets, checkpoints, credentials or gallery are copied. Tests that derive paths
from __file__ or their working directory therefore write into the disposable copy. Tests
must still relocate absolute notebook paths explicitly; a source copy is not an OS sandbox.

    python run_tests.py                     # default CPU suites
    python run_tests.py --all               # also train and resume synthetic models
    python run_tests.py --suite test_runner # one suite, still isolated
"""

from __future__ import annotations

import argparse
import math
import os
import re
import runpy
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Negative values beyond the signal range cannot be normal subprocess exit codes.
SUITE_TIMEOUT = -1000
HEARTBEAT_SECONDS = 30

FAST = [
    ("Test runner isolation", "tests/test_runner.py"),
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
    ("Caregiver report translation", "tests/test_translation.py"),
    ("Caregiver language UI", "tests/test_web_localization.py"),
    ("Notebook validation", "tests/test_notebooks.py"),
    # Executes notebook 04's real cell sources. Listed here, not in SLOW, because every
    # failure it has caught cost a 10-minute GPU session; a ~40 s local run that gates
    # ordinary edits is the whole point.
    ("Notebook 04 end-to-end", "tests/test_notebook04_e2e.py"),
]
SLOW = [("Training stack", "tests/test_training.py")]

# Explicit source-only snapshot. In particular, never copy root data/, runs/, weights/,
# gallery.json, credentials, or config overrides from an operator's machine.
SOURCE_TREES = {
    "src": {".py"}, "scripts": {".py"}, "tests": {".py", ".mjs"},
    "notebooks": {".py", ".ipynb"}, "configs": {".yaml", ".tsv"},
    "web": {".py", ".html", ".js", ".mjs", ".css", ".md"}, "docs": {".md"},
}
SOURCE_FILES = ("run_tests.py", "README.md", "LICENSE", "Dockerfile", ".gitignore",
                ".kaggleignore", "dataset-metadata.json", "requirements-serve.txt",
                "requirements-train.txt", "requirements-test.txt")
SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".pytest_cache"}


def copy_test_source(source: Path, destination: Path) -> None:
    """Copy testable code, including the data package, without local runtime assets."""
    source, destination = source.resolve(), destination.resolve()
    if destination.is_relative_to(source):
        raise ValueError("test snapshot must be outside the source tree")
    destination.mkdir(parents=True, exist_ok=True)

    def copy(path: Path) -> None:
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            return
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    for name in SOURCE_FILES:
        path = source / name
        if path.is_file():
            copy(path)
    for top, suffixes in SOURCE_TREES.items():
        base = source / top
        if not base.is_dir() or base.is_symlink() or not base.resolve().is_relative_to(source):
            continue
        for directory, directories, files in os.walk(base, followlinks=False):
            parent = Path(directory)
            directories[:] = [name for name in directories if name not in SKIP_DIRS
                              and not (parent / name).is_symlink()
                              and (parent / name).resolve().is_relative_to(source)]
            for name in files:
                path = parent / name
                if path.suffix not in suffixes or name == "local.yaml" or name.endswith(".local.yaml"):
                    continue
                copy(path)


def test_environment(snapshot: Path, threads: int) -> dict[str, str]:
    env = {**os.environ, "PYTHONPATH": str(snapshot / "src"),
           "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"}
    # Small CPU fixtures slow dramatically when each convolution fans out over all cores.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[name] = str(threads)
    return env


def windows_kernel():
    """Typed Win32 calls; imported lazily so the runner also works on Linux."""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
        "SetInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                     wintypes.DWORD], wintypes.BOOL),
        "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
        "GetCurrentProcess": ([], wintypes.HANDLE),
        "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(kernel, name)
        function.argtypes, function.restype = arguments, result
    return kernel


class WindowsJob:
    """Own a suite's descendants even if the original suite process exits first."""

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                        ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
                        ("max_working_set", ctypes.c_size_t), ("active_processes", wintypes.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                        ("scheduling", wintypes.DWORD)]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io_counters", ctypes.c_ulonglong * 6),
                        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                        ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

        self.kernel = windows_kernel()
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits),
                                                  ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def run_windows_child(path: Path, job_handle: int) -> None:
    """Join the inherited job before executing any test or spawning its subprocesses."""
    import ctypes

    kernel = windows_kernel()
    try:
        if not kernel.AssignProcessToJobObject(job_handle, kernel.GetCurrentProcess()):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        # Only the parent retains the job handle, so its cleanup also kills orphans.
        kernel.CloseHandle(job_handle)
    sys.argv = [str(path)]
    sys.path[0] = str(path.parent)
    runpy.run_path(str(path), run_name="__main__")


def stop_process_tree(proc: subprocess.Popen, job=None) -> None:
    """Stop a timed-out suite and the training subprocesses it owns."""
    if os.name == "nt":
        job.close()
        # The launcher might time out before it has joined the job. No test code can
        # have run in that case, and the only remaining process is the launcher itself.
        if proc.poll() is None:
            proc.kill()
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_suite(snapshot: Path, path: str, threads: int, timeout: float) -> tuple[int, str]:
    job = WindowsJob() if os.name == "nt" else None
    command = [sys.executable, "-u", str(snapshot / path)]
    options = {"start_new_session": True}
    if job:
        startup = subprocess.STARTUPINFO()
        startup.lpAttributeList = {"handle_list": [job.handle]}
        os.set_handle_inheritable(job.handle, True)
        # This launcher attaches itself before loading the test, eliminating the race
        # between assigning a job from the parent and the test spawning descendants.
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   "--_child-suite", str(snapshot / path), "--_job-handle", str(job.handle)]
        options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
                   "startupinfo": startup, "close_fds": True}
    try:
        proc = subprocess.Popen(command, cwd=snapshot, env=test_environment(snapshot, threads),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", **options)
    except BaseException:
        if job:
            job.close()
        raise
    finally:
        if job and job.handle:
            os.set_handle_inheritable(job.handle, False)
    try:
        started = time.monotonic()
        deadline = started + timeout
        while True:
            try:
                output, _ = proc.communicate(
                    timeout=min(HEARTBEAT_SECONDS, max(0, deadline - time.monotonic())))
                return proc.returncode, output
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                if elapsed >= timeout:
                    raise
                print(f"  {Path(path).stem} still running: {elapsed:.0f}s "
                      f"(limit {timeout:g}s)", flush=True)
    except subprocess.TimeoutExpired:
        stop_process_tree(proc, job)
        try:
            output, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            output = "Process output did not close after cleanup.\n"
        return SUITE_TIMEOUT, output + f"\nSuite exceeded {timeout:g} seconds; process tree stopped.\n"
    except BaseException:
        stop_process_tree(proc, job)
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        raise
    finally:
        if job:
            job.close()


@dataclass(frozen=True)
class SuiteResult:
    name: str
    status: str
    summary: str
    elapsed: float
    passed: int = 0
    total: int = 0
    skipped: int = 0
    has_counts: bool = False


def summarize_suite(name: str, code: int, output: str, elapsed: float) -> SuiteResult:
    """Keep process completion separate from any assertion counts it printed."""
    matches = list(re.finditer(r"(?m)^\s*(\d+)/(\d+)\s+passed\b([^\n]*)$", output))
    match = matches[-1] if matches else None
    passed, total = (int(match[1]), int(match[2])) if match else (0, 0)
    skip_match = re.search(r"\b(\d+)\s+skipped\b", match[3]) if match else None
    skipped = int(skip_match[1]) if skip_match else 0
    valid = match is not None and 0 <= passed <= total
    summary = match[0].strip() if match else "no test summary"
    if code == SUITE_TIMEOUT:
        status = "TIMEOUT"
        # A child process or an earlier phase may have emitted this summary.
        summary = "no completed test summary"
    elif code != 0:
        status = "FAIL"
        summary += f" (exit {code})"
    elif not valid or (total == 0 and skipped == 0):
        status = "INCOMPLETE"
    elif passed != total:
        status = "FAIL"
    else:
        status = "SKIP" if total == 0 else "PASS"
    return SuiteResult(name, status, summary, elapsed, passed, total, skipped,
                       has_counts=valid and (total > 0 or skipped > 0) and code != SUITE_TIMEOUT)


def print_summary(results: list[SuiteResult]) -> bool:
    """Print honest suite coverage even when a suite never reaches its summary."""
    print("\nSUMMARY")
    width = max(len(result.name) for result in results)
    for result in results:
        counts = f"{result.passed}/{result.total}" if result.has_counts else "unknown"
        print(f"  {result.status:<10} {result.name:<{width}}  {counts:>8}  "
              f"{result.elapsed:6.1f}s  {result.skipped if result.has_counts else '?'} skipped")

    totals = {status: sum(result.status == status for result in results)
              for status in ("PASS", "FAIL", "TIMEOUT", "INCOMPLETE", "SKIP")}
    print(f"\nSuites: {totals['PASS']}/{len(results)} passed; {totals['FAIL']} failed; "
          f"{totals['TIMEOUT']} timed out; {totals['INCOMPLETE']} incomplete; "
          f"{totals['SKIP']} skipped")
    counted = [result for result in results if result.has_counts]
    print(f"Reported test counts from {len(counted)}/{len(results)} suites: "
          f"{sum(result.passed for result in counted)}/{sum(result.total for result in counted)} "
          f"passed; {sum(result.skipped for result in counted)} skipped")
    if len(counted) != len(results):
        print("Test totals are incomplete; suites without completed counts are excluded.")
    failed = [result.name for result in results if result.status == "FAIL"]
    timed_out = [result.name for result in results if result.status == "TIMEOUT"]
    incomplete = [result.name for result in results if result.status == "INCOMPLETE"]
    for label, names in (("FAILING SUITES", failed), ("TIMED OUT SUITES", timed_out),
                         ("INCOMPLETE SUITES", incomplete)):
        if names:
            print(f"{label}: {', '.join(names)}")
    ok = not (failed or timed_out or incomplete)
    if ok:
        print("NO SUITE FAILURES (some suites skipped)" if totals["SKIP"] else "ALL SUITES PASSED")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="include the slow training suite")
    ap.add_argument("--suite", action="append", choices=[Path(p).stem for _, p in FAST + SLOW],
                    help="run only this suite; repeat to select several")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads per test process (default: 2)")
    ap.add_argument("--timeout", type=float, default=900,
                    help="seconds allowed per suite, including subprocesses (default: 900)")
    ap.add_argument("--list", action="store_true", help="list suites without running tests")
    ap.add_argument("--_child-suite", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--_job-handle", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args._child_suite is not None:
        if os.name != "nt" or args._job_handle is None:
            ap.error("the internal Windows launcher needs an inherited job handle")
        run_windows_child(args._child_suite, args._job_handle)
        return
    if args.threads < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        ap.error("--threads and --timeout must be positive")
    if args.list:
        for name, path in FAST + SLOW:
            print(f"{Path(path).stem}: {name}")
        return
    suites = ([item for item in FAST + SLOW if Path(item[1]).stem in args.suite]
              if args.suite else FAST + (SLOW if args.all else []))
    results: list[SuiteResult] = []
    with tempfile.TemporaryDirectory(prefix="behaviorsense-tests-") as temporary:
        snapshot = Path(temporary) / "source"
        copy_test_source(ROOT, snapshot)
        print(f"Running {len(suites)} suites in a temporary source copy; "
              f"{args.threads} CPU threads per process.", flush=True)
        for name, path in suites:
            print(f"\n{name} ({path})", flush=True)
            t0 = time.monotonic()
            code, output = run_suite(snapshot, path, args.threads, args.timeout)
            dt = time.monotonic() - t0
            result = summarize_suite(name, code, output, dt)
            if result.status not in {"PASS", "SKIP"}:
                print(output[-20000:].rstrip(), flush=True)
            print(f"{result.status}: {result.summary} [{dt:.1f}s]", flush=True)
            results.append(result)

    ok = print_summary(results)
    if not args.all and not args.suite:
        print("(training suite not selected; run with --all)")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
