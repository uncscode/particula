#!/usr/bin/env python3
"""Pytest runner with authoritative validation and coverage controls for ADW.

Runs pytest with coverage reporting and validation that prevents success-style
output when pytest fails or when requested coverage data is unusable. Supports
scoped tests, deterministic coverage source validation, a shared timeout cap of
3600 seconds, process-group-aware timeout cleanup, and multiple output formats
for both interactive and programmatic use.

Key features:
    - Coverage reporting with configurable source and thresholds
    - Hard-failure handling for unusable pytest-cov diagnostics
    - Validation of minimum test counts to catch collection errors
    - Fail-fast mode for quick development feedback
    - Same-worktree coverage locking to avoid shared .coverage collisions
    - Duration profiling for performance optimization
    - Worktree-aware PYTHONPATH handling for isolated execution

Usage:
    python3 run_pytest.py
    python3 run_pytest.py adw/core/tests/ --min-tests 1
    python3 run_pytest.py --coverage-threshold 80

Examples:
    # Run full test suite (expects ~1700 tests)
    python3 .opencode/tools/run_pytest.py --min-tests 1700

    # Run scoped tests (always set min-tests=1 for scoped)
    python3 .opencode/tools/run_pytest.py adw/core/tests/ --min-tests 1

    # With coverage threshold enforcement
    python3 .opencode/tools/run_pytest.py --coverage-threshold 80

    # Fail fast during development
    python3 .opencode/tools/run_pytest.py --fail-fast adw/core/tests/

    # In worktree for isolated execution
    python3 .opencode/tools/run_pytest.py --cwd /path/to/trees/abc12345

    # Show slowest tests for optimization
    python3 .opencode/tools/run_pytest.py --durations 10

    # Skip slow tests
    python3 .opencode/tools/run_pytest.py -m 'not slow and not performance'
"""

import argparse
import errno
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Tuple, Union
from zlib import crc32

SECTION_HEADER_PATTERN = re.compile(r"^=+\s*.+\s*=+\s*$")
DURATIONS_HEADER_PATTERN = re.compile(
    r"^=+\s*slowest(?:\s+\d+)?\s+durations\s*=+\s*$",
    re.IGNORECASE,
)
FAILURES_HEADER_PATTERN = re.compile(r"^=+\s*FAILURES\s*=+\s*$", re.IGNORECASE)

COVERAGE_ADDOPT_PATTERN = re.compile(r"^(--cov(?:=|\b)|--cov-report=|--cov-fail-under=)")
COVERAGE_PYTEST_ARG_PATTERN = re.compile(
    r"^(--cov(?:=|\b)|--cov-report(?:=|\b)|--cov-fail-under(?:=|\b)|"
    r"--cov-config(?:=|\b)|--cov-context(?:=|\b))"
)
COVERAGE_HEADER_PATTERN = re.compile(r"^-+\s+coverage:.*-+$", re.IGNORECASE)
MAX_COVERAGE_FILES = 500
COVERAGE_LOCK_FILENAME = ".run_pytest_coverage.lock"
MAX_TIMEOUT_SECONDS = 3600
PYTEST_TIMEOUT_KILL_GRACE_SECONDS = 1.0
UNUSABLE_COVERAGE_FRAGMENTS = (
    "no data collected",
    "no data was collected",
    "no data to report",
    "module was never imported",
)


def _candidate_tool_dirs(cwd: Optional[str]) -> List[Path]:
    """Return likely executable directories that may be absent from tool PATH."""

    dirs: List[Path] = []
    if cwd:
        current = Path(cwd).resolve(strict=False)
        while True:
            dirs.extend([current / ".venv" / "bin", current / "venv" / "bin"])
            if current.parent == current:
                break
            current = current.parent
    dirs.append(Path(sys.executable).resolve(strict=False).parent)
    dirs.append(Path.home() / ".local" / "bin")
    return dirs


def _resolve_python_tool_command(tool_name: str, module_name: str, cwd: Optional[str]) -> List[str]:
    """Resolve a Python CLI robustly for non-login tool subprocess environments."""

    resolved = shutil.which(tool_name)
    if resolved:
        return [resolved]
    for directory in _candidate_tool_dirs(cwd):
        candidate = directory / tool_name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return [str(candidate)]
    if importlib.util.find_spec(module_name) is not None:
        return [sys.executable, "-m", module_name]
    return [tool_name]


class CoverageSourceValidationError(ValueError):
    """Raised when coverage source input violates the wrapper contract."""


class CoverageLockError(RuntimeError):
    """Raised when a same-worktree coverage run is already in progress."""


class PytestTimeoutValidationError(ValueError):
    """Raised when a timeout argument violates the wrapper contract."""


@dataclass(frozen=True)
class PytestTimeoutDetails:
    """Structured timeout details for deterministic wrapper diagnostics."""

    timeout_seconds: float
    elapsed_seconds: float
    pid: int
    process_group_id: int
    cwd: str
    command: List[str]
    sigkill_escalated: bool


class PytestTimedOutError(RuntimeError):
    """Raised when pytest exceeds the configured timeout."""

    def __init__(self, details: PytestTimeoutDetails) -> None:
        super().__init__("pytest timed out")
        self.details = details


@dataclass(frozen=True)
class PytestSubprocessResult:
    """Captured subprocess result for pytest execution."""

    returncode: int
    stdout: str
    stderr: str


def _format_timeout_number(value: float) -> str:
    """Format timeout-related numeric values deterministically."""

    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _validate_timeout_seconds(timeout: object) -> float:
    """Validate timeout input before any subprocess launch.

    Args:
        timeout: Caller-provided timeout value from the API or CLI.

    Returns:
        Normalized timeout value as a float in seconds.

    Raises:
        PytestTimeoutValidationError: The timeout is not numeric, is not finite,
            is non-positive, or exceeds the shared 3600-second cap.
    """

    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise PytestTimeoutValidationError(
            "timeout must be a positive finite number in seconds and must not exceed "
            "3600 seconds (1 hour)."
        )
    timeout_value = float(timeout)
    if (
        not math.isfinite(timeout_value)
        or timeout_value <= 0
        or timeout_value > MAX_TIMEOUT_SECONDS
    ):
        raise PytestTimeoutValidationError(
            "timeout must be a positive finite number in seconds and must not exceed "
            "3600 seconds (1 hour)."
        )
    return timeout_value


def _format_timeout_error(details: PytestTimeoutDetails) -> str:
    """Render deterministic timeout diagnostics in a fixed field order.

    Args:
        details: Structured timeout metadata captured during subprocess cleanup.

    Returns:
        Wrapper-safe ``ERROR:`` string with stable timeout, process, and command
        fields for downstream tooling and regression tests.
    """

    return (
        "ERROR: pytest timed out; "
        f"timeout_seconds={_format_timeout_number(details.timeout_seconds)}; "
        f"elapsed_seconds={_format_timeout_number(details.elapsed_seconds)}; "
        f"pid={details.pid}; "
        f"process_group_id={details.process_group_id}; "
        f"cwd={_redact_timeout_cwd(details.cwd)}; "
        f"command={_redact_timeout_command(details.command)}; "
        f"sigkill_escalated={'true' if details.sigkill_escalated else 'false'}"
    )


def _redact_timeout_cwd(cwd: str) -> str:
    """Return a stable, non-absolute cwd token for timeout diagnostics."""

    cwd_name = Path(cwd).name
    return cwd_name or "."


def _redact_timeout_command(command: List[str]) -> str:
    """Return a stable, argument-redacted command summary."""

    if not command:
        return "<unknown>"
    executable = Path(command[0]).name or command[0]
    redacted_arg_count = max(len(command) - 1, 0)
    if redacted_arg_count == 0:
        return executable
    return f"{executable} [args_redacted:{redacted_arg_count}]"


def _process_group_exists(process_group_id: int) -> bool:
    """Return whether the process group still exists."""

    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(process: subprocess.Popen[str], process_group_id: int) -> bool:
    """Terminate a timed-out pytest process group.

    Sends ``SIGTERM`` to the full process group first, waits briefly for a
    graceful exit, and escalates to ``SIGKILL`` only when the group remains
    alive. Missing process groups are treated as already exited.

    Args:
        process: Running pytest subprocess handle.
        process_group_id: Process-group identifier associated with ``process``.

    Returns:
        ``True`` when cleanup required ``SIGKILL`` escalation, otherwise
        ``False``.
    """

    sigkill_escalated = False
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return sigkill_escalated

    try:
        process.wait(timeout=PYTEST_TIMEOUT_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass

    if not _process_group_exists(process_group_id):
        return sigkill_escalated

    sigkill_escalated = True

    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return sigkill_escalated

    try:
        process.wait(timeout=PYTEST_TIMEOUT_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    return sigkill_escalated


def _filter_non_coverage_addopts(addopts: str) -> List[str]:
    """Return non-coverage addopts from a PYTEST_ADDOPTS string."""
    return [arg for arg in addopts.split() if arg and not COVERAGE_ADDOPT_PATTERN.match(arg)]


def _normalize_coverage_source(coverage_source: Optional[object]) -> List[str]:
    """Normalize coverage source inputs into a clean list.

    Args:
        coverage_source: Coverage source from CLI or API. Accepts None, string,
            or list of strings.

    Returns:
        List of normalized coverage sources. Returns an empty list when
        coverage should fall back to the default configuration.

    Raises:
        CoverageSourceValidationError: The input contains empty comma-separated
            segments or absolute paths.

    Notes:
        The special value ``all`` clears explicit sources so pytest-cov falls
        back to the repository coverage configuration.
    """
    if coverage_source is None:
        return []

    sources: List[str] = []
    if isinstance(coverage_source, str):
        sources = coverage_source.split(",")
    elif isinstance(coverage_source, (list, tuple)):
        for entry in coverage_source:
            if entry is None:
                continue
            if isinstance(entry, str):
                sources.extend(entry.split(","))
    else:
        return []

    cleaned: List[str] = []
    for source in sources:
        stripped = source.strip()
        if not stripped:
            raise CoverageSourceValidationError(
                "coverageSource must not contain empty comma-separated entries"
            )
        cleaned.append(stripped)

    if any(source.lower() == "all" for source in cleaned):
        return []

    normalized: List[str] = []
    for source in cleaned:
        if Path(source).is_absolute():
            raise CoverageSourceValidationError(
                f"coverageSource must not contain absolute paths: {source}"
            )
        normalized.append(source)
    return normalized


def _resolve_repo_root_for_coverage(cwd: Optional[str]) -> Path:
    """Resolve the trusted repository/worktree root used for coverage paths."""

    if cwd:
        return Path(cwd).resolve()

    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists() or (current / ".git").exists():
            return current
        current = current.parent
    return Path.cwd().resolve()


def _validate_coverage_source_scope(source: str, repo_root: Path) -> None:
    """Reject path-like coverage sources that resolve outside the repo/worktree."""

    if (
        "/" not in source
        and "\\" not in source
        and not source.startswith(".")
        and not source.endswith(".py")
    ):
        return

    resolved_source = (repo_root / Path(source)).resolve(strict=False)
    try:
        resolved_source.relative_to(repo_root)
    except ValueError as exc:
        raise CoverageSourceValidationError(
            f"coverageSource must stay within the repository/worktree root: {source}"
        ) from exc


def _contains_coverage_pytest_args(args: List[str]) -> bool:
    """Return True when passthrough pytest args request coverage behavior."""

    return any(COVERAGE_PYTEST_ARG_PATTERN.match(arg) for arg in args)


def _coverage_source_to_rcfile(sources: List[str], cwd: Optional[str] = None) -> Optional[str]:
    """Create a temporary coveragerc file for explicit coverage sources.

    Args:
        sources: Normalized coverage sources (module names or paths) that should
            be injected into the coverage configuration.
        cwd: Base directory used to resolve repo-relative path sources before
            writing the temporary coverage config.

    Returns:
        Absolute path to the generated temporary coveragerc file when sources
        are provided, otherwise ``None``.
    """

    if not sources:
        return None
    resolved_sources: List[str] = []
    base_dir = Path(cwd) if cwd else Path.cwd()
    for source in sources:
        path = Path(source)
        if "/" in source or "\\" in source or source.endswith(".py"):
            resolved_sources.append(str((base_dir / path).resolve()))
        else:
            resolved_sources.append(source)
    temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".coveragerc")
    temp_file.write("[run]\n")
    temp_file.write("source =\n")
    for source in resolved_sources:
        temp_file.write(f"    {source}\n")
    temp_file.close()
    return temp_file.name


def _load_pyproject_addopts(root_dir: Path) -> List[str]:
    """Load default pytest addopts defined in pyproject.toml.

    Args:
        root_dir: Project root used to locate ``pyproject.toml``.

    Returns:
        List of addopts parsed from the configuration file. Returns an empty
        list when the file is missing, unreadable, or does not define addopts.
    """

    pyproject_path = root_dir / "pyproject.toml"
    if not pyproject_path.exists():
        return []
    try:
        data = tomllib.loads(pyproject_path.read_text())
    except tomllib.TOMLDecodeError:
        return []
    addopts = data.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("addopts", "")
    if not isinstance(addopts, str) or not addopts:
        return []
    try:
        return shlex.split(addopts)
    except ValueError:
        return addopts.split()


def _should_apply_coverage_threshold(
    *, coverage_threshold: Optional[int], cov_args: List[str], pytest_args: List[str]
) -> bool:
    """Determine if coverage threshold enforcement should run.

    Args:
        coverage_threshold: Configured minimum coverage percentage. ``None``
            disables enforcement.
        cov_args: Coverage arguments generated by this wrapper.
        pytest_args: Original pytest args passed by caller.

    Returns:
        True when validation should enforce the coverage threshold, False
        otherwise.
    """

    if coverage_threshold is None:
        return False
    for arg in cov_args + pytest_args:
        match = re.match(r"--cov(?:=([^\s]+))?$", arg)
        if match and match.group(1):
            return True
    return False


def _coverage_request_has_file_target(sources: List[str]) -> bool:
    """Return True when coverage sources include a repo-relative file target."""

    return any(Path(source).suffix == ".py" for source in sources)


def _detect_unusable_coverage_diagnostics(output: str) -> Optional[str]:
    """Return a stable error for pytest-cov diagnostics that invalidate coverage.

    Args:
        output: Combined pytest stdout and stderr.

    Returns:
        A reviewer-actionable validation message when pytest-cov reports known
        unusable coverage diagnostics, otherwise ``None``.
    """

    lowered_output = output.lower()
    for fragment in UNUSABLE_COVERAGE_FRAGMENTS:
        if fragment in lowered_output:
            return (
                "Coverage data is unusable: pytest-cov reported "
                f"'{fragment}'. Review coverageSource/import targeting."
            )
    return None


def _extract_section(
    lines: List[str],
    header_pattern: Pattern[str],
    *,
    stop_on_blank: bool,
    max_lines: Optional[int] = None,
) -> List[str]:
    """Extract a section of output starting at a header line.

    Args:
        lines: Output lines to scan.
        header_pattern: Compiled regex matching the header line.
        stop_on_blank: Whether to stop at the first blank line.
        max_lines: Optional cap for the number of lines returned.

    Returns:
        List of lines including the header. Returns an empty list if the header is not found.
    """
    start_index = next(
        (index for index, line in enumerate(lines) if header_pattern.match(line)),
        None,
    )
    if start_index is None:
        return []

    collected: List[str] = []
    for index in range(start_index, len(lines)):
        line = lines[index]
        if index != start_index:
            if stop_on_blank and not line.strip():
                break
            if SECTION_HEADER_PATTERN.match(line) and not header_pattern.match(line):
                break
        collected.append(line)
        if max_lines is not None and len(collected) >= max_lines:
            break
    return collected


def parse_pytest_output(output: str) -> Dict:
    """Parse pytest output to extract key metrics.

    Extracts test counts, runtime, coverage percentage, duration profiling data,
    and failure details from pytest's terminal output using regex patterns.

    Args:
        output: The full pytest output text including summary line
            (e.g., "===== 1630 passed, 8 skipped in 35.20s =====").

    Returns:
        Dictionary with parsed metrics:
            - passed/failed/errors/skipped/warnings: Test counts
            - total: Sum of passed + failed + errors
            - duration: Test run time in seconds
            - coverage_pct: Coverage percentage (0-100) if reported
            - durations: List of slowest test entries with duration, phase, test
            - has_failures/has_errors: Boolean flags
            - failed_tests/error_tests: Lists of test names
            - exit_code: Will be set by caller
    """
    result: Dict[str, Any] = {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "warnings": 0,
        "total": 0,
        "exit_code": None,
        "duration": None,
        "coverage_pct": None,
        "has_failures": False,
        "has_errors": False,
        "failed_tests": [],
        "error_tests": [],
    }

    # Parse test counts from summary line
    # Example: "===== 1630 passed, 8 skipped in 35.20s ====="
    # Or: "===== 1 failed, 1880 passed, 9 skipped in 26.35s ====="
    # Extract all counts from the summary line (order can vary)
    summary_line_pattern = r"=+\s*(.*?)\s+in\s+([\d.]+)s?(?:\s*\([^)]*\))?\s*=+"
    summary_match = re.search(summary_line_pattern, output)

    if summary_match:
        summary_text = summary_match.group(1)
        result["duration"] = float(summary_match.group(2))

        # Extract individual counts (order-independent)
        passed_match = re.search(r"(\d+)\s+passed", summary_text)
        if passed_match:
            result["passed"] = int(passed_match.group(1))

        failed_match = re.search(r"(\d+)\s+failed", summary_text)
        if failed_match:
            result["failed"] = int(failed_match.group(1))

        error_match = re.search(r"(\d+)\s+errors?", summary_text)
        if error_match:
            result["errors"] = int(error_match.group(1))

        skipped_match = re.search(r"(\d+)\s+skipped", summary_text)
        if skipped_match:
            result["skipped"] = int(skipped_match.group(1))

        warning_match = re.search(r"(\d+)\s+warnings?", summary_text)
        if warning_match:
            result["warnings"] = int(warning_match.group(1))

        result["total"] = result["passed"] + result["failed"] + result["errors"]

    # Check for FAILED marker
    failed_pattern = r"^(.*?)\s+FAILED"
    for line in output.split("\n"):
        if " FAILED " in line:
            result["has_failures"] = True
            match = re.match(failed_pattern, line.strip())
            if match:
                result["failed_tests"].append(match.group(1))

    # Check for ERROR marker
    error_pattern = r"^(.*?)\s+ERROR"
    for line in output.split("\n"):
        if " ERROR " in line:
            result["has_errors"] = True
            match = re.match(error_pattern, line.strip())
            if match:
                result["error_tests"].append(match.group(1))

    # Parse coverage percentage
    # Example: "TOTAL        6956   6956     0%"
    coverage_pattern = r"TOTAL\s+\d+\s+\d+\s+(\d+)%"
    match = re.search(coverage_pattern, output)
    if match:
        result["coverage_pct"] = int(match.group(1))

    coverage_files: List[Dict[str, object]] = []

    def _coverage_sort_key(entry: Dict[str, object]) -> int:
        value = entry.get("coverage_pct")
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return 0

    coverage_file_pattern = re.compile(r"^(\S+)\s+(\d+)\s+(\d+)\s+(\d+)%\s*(?:\|\s*)?(.+)?$")
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("-"):
            continue
        if stripped.startswith("Name") or stripped.startswith("Stmts"):
            continue
        if stripped.startswith("TOTAL") or COVERAGE_HEADER_PATTERN.match(stripped):
            continue
        match = coverage_file_pattern.match(stripped)
        if match:
            coverage_files.append(
                {
                    "file": match.group(1),
                    "statements": int(match.group(2)),
                    "missing": int(match.group(3)),
                    "coverage_pct": int(match.group(4)),
                    "missing_lines": (match.group(5) or "").strip(),
                }
            )

    if coverage_files:
        coverage_files_sorted = sorted(coverage_files, key=_coverage_sort_key)
        if len(coverage_files_sorted) > MAX_COVERAGE_FILES:
            result["coverage_files_total"] = len(coverage_files_sorted)
            result["coverage_files_truncated"] = len(coverage_files_sorted) - MAX_COVERAGE_FILES
            coverage_files_sorted = coverage_files_sorted[:MAX_COVERAGE_FILES]
        result["coverage_files"] = coverage_files_sorted

    lines = output.splitlines()
    durations_section = _extract_section(lines, DURATIONS_HEADER_PATTERN, stop_on_blank=True)
    if durations_section:
        entry_pattern = re.compile(r"^([\d.]+)s\s+(\w+)\s+(.+)$")
        durations_entries: List[Dict[str, object]] = []
        for line in durations_section[1:]:
            stripped = line.strip()
            if not stripped or (
                stripped.startswith("(") and "hidden" in stripped and "durations" in stripped
            ):
                continue
            match = entry_pattern.match(stripped)
            if match:
                durations_entries.append(
                    {
                        "duration": float(match.group(1)),
                        "phase": match.group(2),
                        "test": match.group(3),
                    }
                )
        result["durations"] = durations_entries

    return result


def format_summary(
    metrics: Dict, validation_errors: List[str], coverage_threshold: Optional[int] = None
) -> str:
    """Format a human-readable summary of test results.

    Generates a structured summary with test counts, runtime, coverage,
    failed test names, slowest test metrics, and validation status.

    Args:
        metrics: Parsed metrics from pytest output including test counts,
            duration, coverage, and failure details.
        validation_errors: List of validation error messages (empty if passed).
        coverage_threshold: Optional minimum coverage percentage for display.
            Shows PASSED/FAILED status relative to threshold.

    Returns:
        Multi-line formatted string with visual separators, test counts,
        coverage status, failed test previews, slowest test data, and
        validation result.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("PYTEST SUMMARY")
    lines.append("=" * 60)

    # Test counts
    lines.append(f"\nTests Run: {metrics['total']}")
    lines.append(f"  Passed:  {metrics['passed']}")
    if metrics["failed"] > 0:
        lines.append(f"  Failed:  {metrics['failed']}")
    if metrics["errors"] > 0:
        lines.append(f"  Errors:  {metrics['errors']}")
    if metrics["skipped"] > 0:
        lines.append(f"  Skipped: {metrics['skipped']}")

    # Duration
    if metrics["duration"]:
        lines.append(f"\nDuration: {metrics['duration']:.2f}s")

    # Coverage
    if metrics["coverage_pct"] is not None:
        coverage_status = ""
        if coverage_threshold is not None:
            if metrics["coverage_pct"] >= coverage_threshold:
                coverage_status = f" (threshold: {coverage_threshold}% PASSED)"
            else:
                coverage_status = f" (threshold: {coverage_threshold}% FAILED)"
        lines.append(f"Coverage: {metrics['coverage_pct']}%{coverage_status}")

    coverage_files = metrics.get("coverage_files", [])
    if coverage_files:

        def coverage_entry_pct(entry: Dict[str, object]) -> int:
            value = entry.get("coverage_pct")
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
            return 0

        lines.append("\nCoverage by File:")
        threshold = coverage_threshold
        if threshold is not None:
            below = [entry for entry in coverage_files if coverage_entry_pct(entry) < threshold]
            remaining = [entry for entry in coverage_files if entry not in below]
            remaining_sorted = sorted(remaining, key=coverage_entry_pct)
            ordered = below + remaining_sorted
        else:
            ordered = sorted(coverage_files, key=coverage_entry_pct)
        for entry in ordered[:15]:
            missing_lines = entry.get("missing_lines") or ""
            missing_info = f" (missing: {missing_lines})" if missing_lines else ""
            coverage_pct = coverage_entry_pct(entry)
            lines.append(f"  {entry.get('file', '')} — {coverage_pct}%{missing_info}")
        if len(ordered) > 15:
            lines.append(f"  ... and {len(ordered) - 15} more files")

    # Failed tests
    if metrics["failed_tests"]:
        lines.append(f"\nFailed Tests ({len(metrics['failed_tests'])}):")
        for test in metrics["failed_tests"][:10]:  # Show first 10
            lines.append(f"  - {test}")
        if len(metrics["failed_tests"]) > 10:
            lines.append(f"  ... and {len(metrics['failed_tests']) - 10} more")

    # Error tests
    if metrics["error_tests"]:
        lines.append(f"\nError Tests ({len(metrics['error_tests'])}):")
        for test in metrics["error_tests"][:10]:  # Show first 10
            lines.append(f"  - {test}")
        if len(metrics["error_tests"]) > 10:
            lines.append(f"  ... and {len(metrics['error_tests']) - 10} more")

    # Slowest tests
    durations = metrics.get("durations", [])
    if durations:
        lines.append("\nSlowest Tests:")
        for entry in durations[:30]:
            duration = entry.get("duration")
            phase = entry.get("phase", "")
            test_name = entry.get("test", "")
            lines.append(f"  {duration:>7.2f}s  {phase:<8} {test_name}")
        if len(durations) > 30:
            lines.append(f"  ... and {len(durations) - 30} more")

    # Validation results
    lines.append("\n" + "=" * 60)
    if validation_errors:
        lines.append("VALIDATION: FAILED")
        lines.append("=" * 60)
        for error in validation_errors:
            lines.append(f"  - {error}")
    else:
        lines.append("VALIDATION: PASSED")
        lines.append("=" * 60)
        lines.append("  All validation checks passed")

    return "\n".join(lines)


def validate_results(
    metrics: Dict,
    min_test_count: int = 1,
    coverage_threshold: Optional[int] = None,
) -> List[str]:
    """Validate pytest results against expected criteria.

    Performs multiple validation checks to catch common issues:
    - Test failures or errors
    - Insufficient passing test count (catches collection issues)
    - No tests ran (empty test suite)
    - Coverage below threshold

    Args:
        metrics: Parsed metrics from pytest output including test counts
            and coverage percentage.
        min_test_count: Minimum expected number of passing tests (default: 1).
            Set to ~1700 for full suite, 1 for scoped tests.
        coverage_threshold: Minimum required coverage percentage (0-100),
            or None to skip coverage validation.

    Returns:
        List of validation error messages. Empty list indicates all
        checks passed. Non-empty list triggers exit code 1.
    """
    errors = []

    # Check for failures
    if metrics["has_failures"]:
        errors.append(f"Found {metrics['failed']} failed test(s)")

    # Check for errors
    if metrics["has_errors"]:
        errors.append(f"Found {metrics['errors']} test error(s)")

    # Check test count
    if metrics["passed"] < min_test_count:
        errors.append(
            f"Expected at least {min_test_count} passing tests, but only {metrics['passed']} passed"
        )

    # Check if no tests ran
    if metrics["total"] == 0:
        errors.append("No tests were collected or run")

    # Check coverage threshold
    if coverage_threshold is not None and metrics["coverage_pct"] is not None:
        if metrics["coverage_pct"] < coverage_threshold:
            errors.append(
                f"Coverage {metrics['coverage_pct']}% is below threshold of {coverage_threshold}%"
            )

    return errors


def _resolve_normalized_sources(
    cwd: Optional[str], coverage_source: Optional[Union[str, List[str]]]
) -> List[str]:
    """Resolve normalized coverage sources for the current invocation.

    Args:
        cwd: Requested pytest working directory. Included for call-site parity;
            normalization validates inputs but does not rewrite them.
        coverage_source: Optional coverage source configuration from caller.

    Returns:
        Normalized list of coverage sources that preserves module names,
        repo-relative directories, and repo-relative file targets.
    """

    normalized_sources = _normalize_coverage_source(coverage_source)
    repo_root = _resolve_repo_root_for_coverage(cwd)
    for source in normalized_sources:
        _validate_coverage_source_scope(source, repo_root)
    return normalized_sources


def _recover_stale_coverage_lock(lock_path: Path) -> bool:
    """Recover a stale same-worktree coverage lock when safe to do so."""

    try:
        contents = lock_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise CoverageLockError(
            "coverage lock exists but could not be inspected for stale recovery: "
            f"{lock_path} ({exc})"
        ) from exc

    pid_text = contents.removeprefix("pid=").strip()
    if not pid_text.isdigit():
        try:
            lock_path.unlink()
        except OSError as exc:
            raise CoverageLockError(
                f"coverage lock exists with invalid metadata and could not be removed: {lock_path}"
            ) from exc
        return True

    pid = int(pid_text)
    try:
        os.kill(pid, 0)
    except OSError as exc:
        if exc.errno != errno.ESRCH:
            raise CoverageLockError(
                "coverage-enabled pytest runs in the same worktree must be serialized; "
                f"coverage lock holder pid={pid} could not be verified"
            ) from exc
        try:
            lock_path.unlink()
        except OSError as unlink_exc:
            raise CoverageLockError(
                f"stale coverage lock could not be removed: {lock_path}"
            ) from unlink_exc
        return True

    return False


def _get_coverage_lock_path(cwd: str) -> Path:
    """Return the repo-local runtime lock path for the given worktree.

    Args:
        cwd: Worktree root used for the pytest run.

    Returns:
        Absolute path to the deterministic lock file under ``adforge_local/state``.
    """

    worktree_root = Path(cwd).resolve()
    runtime_state_dir = worktree_root / "adforge_local" / "state"
    runtime_state_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{crc32(str(worktree_root).encode('utf-8')):08x}"
    return runtime_state_dir / f"{suffix}-{COVERAGE_LOCK_FILENAME}"


def _acquire_coverage_lock(cwd: str) -> str:
    """Acquire an exclusive same-worktree coverage lock.

    Args:
        cwd: Worktree root where the lock file should be created.

    Returns:
        Absolute path to the created lock file.

    Raises:
        CoverageLockError: Another coverage-enabled pytest run is already active
            in the same worktree.
    """

    lock_path = _get_coverage_lock_path(cwd)
    for _ in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError as exc:
            if _recover_stale_coverage_lock(lock_path):
                continue
            raise CoverageLockError(
                "coverage-enabled pytest runs in the same worktree must be serialized; "
                "another coverage run is already active"
            ) from exc
    else:
        raise CoverageLockError(
            f"failed to acquire coverage lock after stale-lock recovery: {lock_path}"
        )

    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()}\n")
    return str(lock_path)


def _run_pytest_subprocess(
    cmd: List[str], *, cwd: str, requested_cwd: Optional[str], timeout: float | int
) -> PytestSubprocessResult:
    """Execute pytest with worktree-aware PYTHONPATH and timeout cleanup.

    Starts pytest in a new process group so timeout handling can terminate the
    entire pytest tree rather than only the direct child process. When a
    timeout occurs, the function records deterministic diagnostics and raises a
    structured timeout error after cleanup.

    Args:
        cmd: Fully resolved pytest command to execute.
        cwd: Working directory used for the subprocess.
        requested_cwd: Original caller-provided worktree path to prepend to
            ``PYTHONPATH`` for isolated imports.
        timeout: Maximum runtime in seconds. Values must satisfy the shared
            timeout validation contract.

    Returns:
        Captured subprocess result containing return code, stdout, and stderr.

    Raises:
        PytestTimedOutError: Pytest exceeded ``timeout`` and the process group
            was terminated.
    """

    env = os.environ.copy()
    if requested_cwd:
        existing_pythonpath = env.get("PYTHONPATH") or ""
        env["PYTHONPATH"] = (
            f"{requested_cwd}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else requested_cwd
        )

    timeout_seconds = _validate_timeout_seconds(timeout)
    started_at = time.monotonic()
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return PytestSubprocessResult(
            returncode=process.returncode or 0, stdout=stdout, stderr=stderr
        )
    except subprocess.TimeoutExpired:
        elapsed_seconds = time.monotonic() - started_at
        try:
            process_group_id = os.getpgid(process.pid)
        except ProcessLookupError:
            process_group_id = process.pid
        sigkill_escalated = _terminate_process_group(process, process_group_id)
        raise PytestTimedOutError(
            PytestTimeoutDetails(
                timeout_seconds=timeout_seconds,
                elapsed_seconds=elapsed_seconds,
                pid=process.pid,
                process_group_id=process_group_id,
                cwd=cwd,
                command=cmd,
                sigkill_escalated=sigkill_escalated,
            )
        ) from None


def _build_pytest_command(
    *,
    args: List[str],
    fail_fast: bool,
    durations: Optional[int],
    durations_min: Optional[float],
    coverage: bool,
    normalized_sources: List[str],
    coverage_rcfile: Optional[str],
    cov_report: str,
    override_ini: Optional[List[str]],
) -> tuple[List[str], List[str], List[str], str]:
    """Build the pytest command and related derived coverage/ini state."""

    cmd = ["pytest", "-v", "--tb=short"]

    if fail_fast:
        cmd.append("-x")

    if durations is not None:
        cmd.append(f"--durations={durations}")
        if durations_min is not None:
            cmd.append(f"--durations-min={durations_min}")

    effective_override_ini = list(override_ini or [])
    if not any(entry.startswith("addopts=") for entry in effective_override_ini):
        if not coverage or normalized_sources:
            effective_override_ini.append("addopts=")

    if effective_override_ini:
        cmd.extend([f"--override-ini={entry}" for entry in effective_override_ini])

    cov_args: List[str] = []
    effective_cov_report = cov_report
    if coverage and not any("--cov" in arg for arg in args):
        if normalized_sources:
            for source in normalized_sources:
                cov_args.append(f"--cov={source}")
            if coverage_rcfile:
                cov_args.append(f"--cov-config={coverage_rcfile}")
            else:
                cov_args.append("--cov-config=/dev/null")
            cov_args.append("--cov-context=test")
            cov_args.extend(_filter_non_coverage_addopts(os.environ.get("PYTEST_ADDOPTS", "")))
            if effective_cov_report.strip() == "term-missing":
                cov_args.append("--cov-report=term-missing")
                effective_cov_report = ""
        else:
            cov_args.append("--cov")
        for report_format in effective_cov_report.split(","):
            if report_format.strip():
                cov_args.append(f"--cov-report={report_format.strip()}")

    if cov_args:
        cmd.extend(cov_args)

    cmd.extend(args)
    return cmd, cov_args, effective_override_ini, effective_cov_report


def run_pytest(
    args: List[str],
    output_mode: str = "summary",
    min_test_count: int = 1,
    cwd: Optional[str] = None,
    timeout: Union[int, float] = 600,
    coverage: bool = True,
    coverage_source: Optional[Union[str, List[str]]] = None,
    coverage_threshold: Optional[int] = None,
    cov_report: str = "term-missing",
    fail_fast: bool = False,
    durations: Optional[int] = None,
    durations_min: Optional[float] = None,
    override_ini: Optional[List[str]] = None,
) -> Tuple[int, str]:
    """Run pytest with coverage and validation.

    Executes pytest with the specified options, parses results, and validates
    against expected criteria. Automatically handles worktree PYTHONPATH for
    isolated execution environments. Non-zero pytest exits always fail
    validation, and coverage-enabled runs also fail when pytest-cov reports
    unusable data such as missing imports or no collected coverage.

    Note:
        -v and --tb=short are always included. Do NOT pass these in args.

    Args:
        args: Additional pytest arguments passed through (e.g., test paths,
            markers like ['-m', 'not slow'], specific tests).
        output_mode: Output format for results. One of:
            - "summary": Human-readable with key metrics (default)
            - "full": Complete pytest output + summary (truncated if >500 lines)
            - "json": Structured data for programmatic use
        min_test_count: Minimum expected passing tests (default: 1).
            Set to ~1700 for full suite validation, 1 for scoped tests.
        cwd: Working directory for pytest execution. If provided, prepends
            to PYTHONPATH for worktree isolation. Defaults to project root.
        timeout: Maximum execution time in seconds (default: 600 = 10 min,
            maximum: 3600 = 1 hour).
        coverage: Enable coverage reporting (default: True). Uses pytest-cov.
        coverage_source: Source module/path for coverage (for example, ``adw``,
            ``adw.core``, ``adw/``, or a repo-relative ``.py`` file target).
            Comma-separated sources are supported. ``None`` or ``all`` uses the
            repository coverage configuration.
        coverage_threshold: Minimum coverage percentage (0-100) to enforce.
            None skips threshold validation.
        cov_report: Coverage report format(s), comma-separated (default: "term-missing").
            Examples: "html", "xml", "term-missing,html:coverage_html".
        fail_fast: Stop on first failure with -x flag (default: False).
        durations: Show N slowest test durations. Use 0 for all, None to skip.
        durations_min: Minimum duration in seconds for inclusion (default: 0.005).
        override_ini: Optional list of ini overrides passed as
            ``--override-ini=<option>=<value>``. When coverage sources are
            provided, addopts are cleared to avoid pyproject overrides.

    Returns:
        Tuple of (exit_code, output_string) where exit_code is 0 if pytest
        and validation pass, 1 otherwise. File-target coverage requests preserve
        explicit ``coverage_files = null`` semantics when per-file numeric detail
        is not authoritative.

    Raises:
        Does not raise; errors are captured and returned in output_string.
    """
    # Determine working directory
    requested_cwd = cwd
    if cwd is None:
        # Try to find project root
        current = Path.cwd()
        while current != current.parent:
            if (current / "pyproject.toml").exists() or (current / ".git").exists():
                cwd = str(current)
                break
            current = current.parent
        if cwd is None:
            cwd = str(Path.cwd())

    coverage_rcfile: Optional[str] = None
    coverage_lock_path: Optional[str] = None

    try:
        _validate_timeout_seconds(timeout)
        normalized_sources = _resolve_normalized_sources(cwd, coverage_source)
        file_scoped_coverage = _coverage_request_has_file_target(normalized_sources)
        coverage_rcfile = _coverage_source_to_rcfile(normalized_sources, cwd) if coverage else None

        if not coverage and _contains_coverage_pytest_args(args):
            raise CoverageSourceValidationError(
                "coverage-related pytest arguments are not allowed when coverage is disabled"
            )

        cmd, cov_args, _, _ = _build_pytest_command(
            args=args,
            fail_fast=fail_fast,
            durations=durations,
            durations_min=durations_min,
            coverage=coverage,
            normalized_sources=normalized_sources,
            coverage_rcfile=coverage_rcfile,
            cov_report=cov_report,
            override_ini=override_ini,
        )
        cmd = [*_resolve_python_tool_command("pytest", "pytest", cwd), *cmd[1:]]

        if coverage:
            coverage_lock_path = _acquire_coverage_lock(cwd)

        result = _run_pytest_subprocess(
            cmd,
            cwd=cwd,
            requested_cwd=requested_cwd,
            timeout=timeout,
        )

        # Combine stdout and stderr
        full_output = result.stdout
        if result.stderr:
            full_output += "\n\nSTDERR:\n" + result.stderr

        # Parse output
        metrics = parse_pytest_output(full_output)
        metrics["exit_code"] = result.returncode
        if file_scoped_coverage:
            metrics["coverage_files"] = None

        # Validate results (including coverage threshold)
        threshold = coverage_threshold
        if not _should_apply_coverage_threshold(
            coverage_threshold=coverage_threshold,
            cov_args=cov_args,
            pytest_args=args,
        ):
            threshold = None
        validation_errors = validate_results(metrics, min_test_count, threshold)
        if result.returncode != 0:
            validation_errors.append(
                "pytest exited with code "
                f"{result.returncode}; inspect failed tests and stderr output"
            )
        if coverage:
            unusable_coverage_error = _detect_unusable_coverage_diagnostics(full_output)
            if unusable_coverage_error:
                validation_errors.append(unusable_coverage_error)
            elif metrics["coverage_pct"] is None:
                validation_errors.append(
                    "Coverage data is unavailable: pytest-cov did not report a TOTAL "
                    "coverage percentage. "
                    "Review coverageSource/import targeting."
                )

        # Determine final exit code (fail if validation fails)
        exit_code = result.returncode
        if validation_errors:
            exit_code = 1

        # Format output based on mode
        if output_mode == "summary":
            output = format_summary(metrics, validation_errors, coverage_threshold)
        elif output_mode == "json":
            output = json.dumps(
                {
                    "metrics": metrics,
                    "durations": metrics.get("durations", []),
                    "validation_errors": validation_errors,
                    "success": len(validation_errors) == 0,
                    "coverage_threshold": coverage_threshold,
                },
                indent=2,
            )
        else:  # full
            # Include summary at the end of full output
            summary = format_summary(metrics, validation_errors, coverage_threshold)
            output = f"{full_output}\n\n{summary}"

            # Fall back to smart truncation if full output is too long (>500 lines)
            max_lines = 500
            line_count = len(output.splitlines())
            if line_count > max_lines:
                lines = full_output.splitlines()
                failures_section = _extract_section(
                    lines, FAILURES_HEADER_PATTERN, stop_on_blank=False, max_lines=200
                )
                durations_section = _extract_section(
                    lines, DURATIONS_HEADER_PATTERN, stop_on_blank=True, max_lines=200
                )
                coverage_section = _extract_section(
                    lines, COVERAGE_HEADER_PATTERN, stop_on_blank=True, max_lines=200
                )
                truncated_lines = [
                    f"[Output truncated: {line_count} lines exceeded {max_lines} line limit. "
                    "Showing failures/durations/coverage sections + summary only.]"
                ]
                if failures_section:
                    truncated_lines.append("")
                    truncated_lines.extend(failures_section)
                if durations_section:
                    truncated_lines.append("")
                    truncated_lines.extend(durations_section)
                if coverage_section:
                    truncated_lines.append("")
                    truncated_lines.extend(coverage_section)
                truncated_lines.append("")
                truncated_lines.append(summary)
                output = "\n".join(truncated_lines)

        return exit_code, output

    except CoverageSourceValidationError as exc:
        return 1, f"ERROR: {exc}"
    except CoverageLockError as exc:
        return 1, f"ERROR: {exc}"
    except PytestTimeoutValidationError as exc:
        return 1, f"ERROR: {exc}"
    except PytestTimedOutError as exc:
        return 1, _format_timeout_error(exc.details)
    except FileNotFoundError:
        return 1, "ERROR: pytest command not found. Is pytest installed?"
    except Exception as e:
        return 1, f"ERROR: Unexpected error running pytest: {e}"
    finally:
        if coverage_lock_path:
            try:
                os.unlink(coverage_lock_path)
            except OSError:
                pass
        if coverage_rcfile:
            try:
                os.unlink(coverage_rcfile)
            except OSError:
                pass


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for CLI usage.

    Parses command-line arguments and executes pytest with validation.

    Args:
        argv: Optional argument list. When omitted, arguments are read from
            ``sys.argv``.

    Returns:
        Exit code (0 if pytest and validation pass, 1 otherwise).
    """
    parser = argparse.ArgumentParser(
        description="Run pytest with coverage and validation",
        epilog="""
Examples:
  %(prog)s                                    Run all tests with coverage
  %(prog)s adw/core/tests/ --min-tests 1      Run scoped tests
  %(prog)s --coverage-threshold 80            Enforce 80%% coverage
  %(prog)s --fail-fast adw/core/tests/        Stop on first failure
  %(prog)s --durations 10                     Show 10 slowest tests
  %(prog)s --cwd /path/to/worktree            Run in worktree
  %(prog)s -m 'not slow'                      Skip slow tests

NOTE: -v and --tb=short are always included. Do NOT pass these.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        choices=["summary", "full", "json"],
        default="summary",
        help=(
            "Output mode: summary (default, key metrics), full (complete output), json (structured)"
        ),
    )
    parser.add_argument(
        "--min-tests",
        type=int,
        default=1,
        help="Minimum expected test count (default: 1). Use ~1700 for full suite, 1 for scoped.",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        help="Working directory (defaults to project root). Use for worktree isolation.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Timeout in seconds (default: 600 = 10 minutes, maximum: 3600 = 1 hour)",
    )
    # Coverage options
    parser.add_argument(
        "--coverage",
        action="store_true",
        default=True,
        help="Enable coverage reporting (default: enabled)",
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage for faster runs",
    )
    parser.add_argument(
        "--coverage-source",
        action="append",
        default=None,
        help=(
            "Source module for coverage (e.g., 'adw'). Can be repeated or comma-separated. "
            "Omit or pass 'all' to use pyproject.toml config."
        ),
    )
    parser.add_argument(
        "--coverage-files-only",
        action="store_true",
        help=(
            "Suppress printing pytest output and only return the exit code "
            "(for tooling/test helpers). Not intended for general CLI usage."
        ),
    )
    parser.add_argument(
        "--coverage-threshold",
        type=int,
        help="Minimum coverage percentage (0-100). Fails validation if below threshold.",
    )
    parser.add_argument(
        "--cov-report",
        type=str,
        default="term-missing",
        help="Coverage report format(s), comma-separated. Examples: 'term-missing', 'html,xml'",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure (-x flag). Good for quick dev feedback.",
    )
    parser.add_argument(
        "--durations",
        type=int,
        default=None,
        help="Show N slowest tests (0 for all). Useful for optimization.",
    )
    parser.add_argument(
        "--durations-min",
        type=float,
        default=None,
        help="Minimum duration in seconds for slowest list (default: 0.005)",
    )
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="Additional pytest arguments (test paths, markers, etc.)",
    )
    parser.add_argument(
        "--override-ini",
        action="append",
        default=[],
        help=(
            "Override ini option (passed through to pytest). Can be repeated,"
            " e.g., --override-ini=addopts=."
        ),
    )

    args, unknown_args = parser.parse_known_args(argv)

    # Determine coverage setting (--no-coverage overrides --coverage)
    coverage_enabled = not args.no_coverage

    pytest_args = list(args.pytest_args) + list(unknown_args)

    exit_code, output = run_pytest(
        pytest_args,
        output_mode=args.output,
        min_test_count=args.min_tests,
        cwd=args.cwd,
        timeout=args.timeout,
        coverage=coverage_enabled,
        coverage_source=args.coverage_source,
        coverage_threshold=args.coverage_threshold,
        cov_report=args.cov_report,
        fail_fast=args.fail_fast,
        durations=args.durations,
        durations_min=args.durations_min,
        override_ini=args.override_ini,
    )

    if args.coverage_files_only:
        return exit_code

    print(output)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
