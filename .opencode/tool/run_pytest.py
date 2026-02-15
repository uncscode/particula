#!/usr/bin/env python3
"""Pytest Runner Tool with Coverage and Validation for ADW.

Runs pytest with coverage reporting and result validation to prevent false
positives. Supports scoped tests, coverage thresholds, and multiple output
formats for both interactive and programmatic use.

Key features:
    - Coverage reporting with configurable source and thresholds
    - Validation of minimum test counts to catch collection errors
    - Fail-fast mode for quick development feedback
    - Duration profiling for performance optimization
    - Worktree-aware PYTHONPATH handling for isolated execution

Usage:
    python3 run_pytest.py
    python3 run_pytest.py adw/core/tests/ --min-tests 1
    python3 run_pytest.py --coverage-threshold 80

Examples:
    # Run full test suite (expects ~1700 tests)
    python3 .opencode/tool/run_pytest.py --min-tests 1700

    # Run scoped tests (always set min-tests=1 for scoped)
    python3 .opencode/tool/run_pytest.py adw/core/tests/ --min-tests 1

    # With coverage threshold enforcement
    python3 .opencode/tool/run_pytest.py --coverage-threshold 80

    # Fail fast during development
    python3 .opencode/tool/run_pytest.py --fail-fast adw/core/tests/

    # In worktree for isolated execution
    python3 .opencode/tool/run_pytest.py --cwd /path/to/trees/abc12345

    # Show slowest tests for optimization
    python3 .opencode/tool/run_pytest.py --durations 10

    # Skip slow tests
    python3 .opencode/tool/run_pytest.py -m 'not slow and not performance'
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Tuple

SECTION_HEADER_PATTERN = re.compile(r"^=+\s*.+\s*=+\s*$")
DURATIONS_HEADER_PATTERN = re.compile(
    r"^=+\s*slowest(?:\s+\d+)?\s+durations\s*=+\s*$",
    re.IGNORECASE,
)
FAILURES_HEADER_PATTERN = re.compile(r"^=+\s*FAILURES\s*=+\s*$", re.IGNORECASE)

COVERAGE_ADDOPT_PATTERN = re.compile(r"^(--cov(?:=|\b)|--cov-report=|--cov-fail-under=)")
COVERAGE_HEADER_PATTERN = re.compile(r"^-+\s+coverage:.*-+$", re.IGNORECASE)
MAX_COVERAGE_FILES = 500


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
    """
    if coverage_source is None:
        return []

    sources: List[str] = []
    if isinstance(coverage_source, str):
        sources = coverage_source.split(",")
    elif isinstance(coverage_source, list):
        for entry in coverage_source:
            if entry is None:
                continue
            if isinstance(entry, str):
                sources.extend(entry.split(","))
    else:
        return []

    cleaned = [source.strip() for source in sources if source and source.strip()]
    if any(source.lower() == "all" for source in cleaned):
        return []

    normalized: List[str] = []
    for source in cleaned:
        path = Path(source)
        if path.suffix == ".py" and path.exists():
            normalized.append(str(path.resolve()))
        else:
            normalized.append(source)
    return normalized


def _coverage_source_to_rcfile(sources: List[str]) -> Optional[str]:
    """Create a temporary coveragerc file for explicit coverage sources.

    Args:
        sources: Normalized coverage sources (module names or paths) that should
            be injected into the coverage configuration.

    Returns:
        Absolute path to the generated temporary coveragerc file when sources
        are provided, otherwise ``None``.
    """

    if not sources:
        return None
    resolved_sources: List[str] = []
    for source in sources:
        path = Path(source)
        if path.is_absolute() or "/" in source or path.exists():
            resolved_sources.append(str(path.resolve()))
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
    *, coverage_threshold: Optional[int], cov_args: List[str]
) -> bool:
    """Determine if coverage threshold enforcement should run.

    Args:
        coverage_threshold: Configured minimum coverage percentage. ``None``
            disables enforcement.
        cov_args: Coverage arguments passed to pytest, used to detect whether at
            least one explicit ``--cov`` target is present.

    Returns:
        True when validation should enforce the coverage threshold, False
        otherwise.
    """

    if coverage_threshold is None:
        return False
    for arg in cov_args:
        match = re.match(r"--cov(?:=([^\s]+))?$", arg)
        if match and match.group(1):
            return True
    return False


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
    result = {
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


def run_pytest(
    args: List[str],
    output_mode: str = "summary",
    min_test_count: int = 1,
    cwd: Optional[str] = None,
    timeout: int = 600,
    coverage: bool = True,
    coverage_source: Optional[object] = None,
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
    isolated execution environments.

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
        timeout: Maximum execution time in seconds (default: 600 = 10 min).
        coverage: Enable coverage reporting (default: True). Uses pytest-cov.
        coverage_source: Source module/path for coverage (e.g., 'adw' or 'adw.core').
            Comma-separated sources are supported. If None, uses pyproject.toml
            [tool.coverage.run].source config.
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
        and validation pass, 1 otherwise.

    Raises:
        Does not raise; errors are captured and returned in output_string.
    """
    # Build pytest command
    # NOTE: -v and --tb=short are always included. Do not pass these in pytestArgs.
    cmd = ["pytest", "-v", "--tb=short"]

    # Add fail-fast if requested
    if fail_fast:
        cmd.append("-x")

    # Add durations if requested
    if durations is not None:
        cmd.append(f"--durations={durations}")
        if durations_min is not None:
            cmd.append(f"--durations-min={durations_min}")

    if cwd:
        # When running in a specific cwd (e.g., a worktree), keep coverage sources
        # relative so they are resolved in the same directory pytest runs in.
        if coverage_source is None:
            normalized_sources: List[str] = []
        elif isinstance(coverage_source, (list, tuple)):
            normalized_sources = list(coverage_source)
        else:
            normalized_sources = [coverage_source]
    else:
        # Fallback to default normalization when no explicit cwd override is used.
        normalized_sources = _normalize_coverage_source(coverage_source)
    coverage_rcfile = _coverage_source_to_rcfile(normalized_sources) if coverage else None
    effective_override_ini = list(override_ini or [])
    if (
        coverage
        and normalized_sources
        and not any(entry.startswith("addopts=") for entry in effective_override_ini)
    ):
        effective_override_ini.append("addopts=")

    if effective_override_ini:
        cmd.extend([f"--override-ini={entry}" for entry in effective_override_ini])

    # Add coverage if enabled and not already specified in args
    cov_args: List[str] = []
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
            if cov_report.strip() == "term-missing":
                cov_args.append("--cov-report=term-missing")
                cov_report = ""
        else:
            cov_args.append("--cov")  # Enable coverage without specifying source
        # Add each report format separately
        for report_format in cov_report.split(","):
            cov_args.append(f"--cov-report={report_format.strip()}")
    if cov_args:
        cmd.extend(cov_args)

    # Add user arguments
    cmd.extend(args)

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

    # Run pytest
    # Ensure the worktree is prioritized for imports when sharing a venv with the main repo.
    # Copy the environment so we can safely prepend cwd to PYTHONPATH without side effects.
    env = os.environ.copy()
    if requested_cwd:
        existing_pythonpath = env.get("PYTHONPATH") or ""
        env["PYTHONPATH"] = (
            f"{requested_cwd}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else requested_cwd
        )

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if coverage_rcfile:
            try:
                os.unlink(coverage_rcfile)
            except OSError:
                pass

        # Combine stdout and stderr
        full_output = result.stdout
        if result.stderr:
            full_output += "\n\nSTDERR:\n" + result.stderr

        # Parse output
        metrics = parse_pytest_output(full_output)
        metrics["exit_code"] = result.returncode

        # Validate results (including coverage threshold)
        threshold = coverage_threshold
        if not _should_apply_coverage_threshold(
            coverage_threshold=coverage_threshold,
            cov_args=cov_args,
        ):
            threshold = None
        validation_errors = validate_results(metrics, min_test_count, threshold)

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

    except subprocess.TimeoutExpired:
        return 1, f"ERROR: pytest timed out after {timeout} seconds"
    except FileNotFoundError:
        return 1, "ERROR: pytest command not found. Is pytest installed?"
    except Exception as e:
        return 1, f"ERROR: Unexpected error running pytest: {e}"


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for CLI usage.

    Parses command-line arguments and executes pytest with validation.

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
        type=int,
        default=600,
        help="Timeout in seconds (default: 600 = 10 minutes)",
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
        coverage_source=_normalize_coverage_source(args.coverage_source),
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
