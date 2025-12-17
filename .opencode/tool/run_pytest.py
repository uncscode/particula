#!/usr/bin/env python3
"""Pytest Runner Tool with Coverage.

Runs pytest with coverage and returns either full output or a summary.
This tool validates test results to prevent false positives.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict


class PytestMetrics(TypedDict):
    """Container for parsed pytest metrics."""

    passed: int
    failed: int
    errors: int
    skipped: int
    warnings: int
    total: int
    exit_code: int | None
    duration: float | None
    coverage_pct: int | None
    has_failures: bool
    has_errors: bool
    failed_tests: List[str]
    error_tests: List[str]


def parse_pytest_output(output: str) -> PytestMetrics:  # noqa: C901
    """Parse pytest output to extract key metrics.

    Args:
        output: The full pytest output text

    Returns:
        Dictionary with parsed metrics
    """
    result: PytestMetrics = {
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

    return result


def format_summary(  # noqa: C901
    metrics: PytestMetrics,
    validation_errors: List[str],
    coverage_threshold: Optional[int] = None,
) -> str:
    """Format a human-readable summary of test results.

    Args:
        metrics: Parsed metrics from pytest output
        validation_errors: List of validation errors
        coverage_threshold: Optional coverage threshold for display

    Returns:
        Formatted summary string
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
    metrics: PytestMetrics,
    min_test_count: int = 1,
    coverage_threshold: Optional[int] = None,
) -> List[str]:
    """Validate pytest results against expected criteria.

    Args:
        metrics: Parsed metrics from pytest output.
        min_test_count: Minimum expected number of passing tests (default: 1).
        coverage_threshold: Minimum required coverage percentage (0-100), or
            None to skip the check.

    Returns:
        List of validation errors (empty if all checks pass).
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
            "Expected at least "
            f"{min_test_count} passing tests, but only "
            f"{metrics['passed']} passed"
        )

    # Check if no tests ran
    if metrics["total"] == 0:
        errors.append("No tests were collected or run")

    # Check coverage threshold
    if coverage_threshold is not None and metrics["coverage_pct"] is not None:
        if metrics["coverage_pct"] < coverage_threshold:
            errors.append(
                "Coverage "
                f"{metrics['coverage_pct']}% is below threshold of "
                f"{coverage_threshold}%"
            )

    return errors


def run_pytest(  # noqa: C901
    args: List[str],
    output_mode: str = "summary",
    min_test_count: int = 1,
    cwd: Optional[str] = None,
    timeout: int = 600,
    coverage: bool = True,
    coverage_source: Optional[str] = None,
    coverage_threshold: Optional[int] = None,
    cov_report: str = "term-missing",
    fail_fast: bool = False,
) -> Tuple[int, str]:
    """Run pytest with the specified arguments.

    Prepends ``cwd`` to ``PYTHONPATH`` (when provided) so git worktrees using a
    shared virtual environment import code from the worktree before any
    installed package copies.

    Args:
        args: Additional pytest arguments.
        output_mode: Either "summary", "full", or "json".
        min_test_count: Minimum expected test count for validation (default: 1).
        cwd: Working directory (defaults to project root).
        timeout: Timeout in seconds (default: 600 = 10 minutes).
        coverage: Enable coverage reporting (default: True).
        coverage_source: Source module for coverage. If None, uses
            ``pyproject.toml`` config.
        coverage_threshold: Minimum coverage percentage (0-100), or None to
            skip.
        cov_report: Coverage report format(s), comma-separated (default:
            "term-missing").
        fail_fast: Stop on first failure (default: False).

    Returns:
        Tuple of (exit_code, output_string).
    """
    # Build pytest command
    # NOTE: -v and --tb=short are always included.
    # Do not pass these in pytest_args.
    cmd = ["pytest", "-v", "--tb=short"]

    # Add fail-fast if requested
    if fail_fast:
        cmd.append("-x")

    # Add coverage if enabled and not already specified in args
    if coverage and not any("--cov" in arg for arg in args):
        # Only pass --cov=<source> if explicitly provided.
        # Otherwise let pytest-cov use pyproject.toml
        # [tool.coverage.run].source.
        if coverage_source:
            cmd.append(f"--cov={coverage_source}")
        else:
            cmd.append("--cov")  # Enable coverage without specifying source
        # Add each report format separately
        for report_format in cov_report.split(","):
            cmd.append(f"--cov-report={report_format.strip()}")

    # Add user arguments
    cmd.extend(args)

    # Determine working directory
    requested_cwd = cwd
    if cwd is None:
        # Try to find project root
        current = Path.cwd()
        while current != current.parent:
            if (current / "pyproject.toml").exists() or (
                current / ".git"
            ).exists():
                cwd = str(current)
                break
            current = current.parent
        if cwd is None:
            cwd = str(Path.cwd())

    # Run pytest
    # Ensure the worktree is prioritized for imports when sharing a venv with
    # the main repository.
    # Copy the environment so we can safely prepend cwd to PYTHONPATH without
    # side effects.
    env = os.environ.copy()
    if requested_cwd:
        existing_pythonpath = env.get("PYTHONPATH") or ""
        env["PYTHONPATH"] = (
            f"{requested_cwd}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else requested_cwd
        )

    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Combine stdout and stderr
        full_output = result.stdout
        if result.stderr:
            full_output += "\n\nSTDERR:\n" + result.stderr

        # Parse output
        metrics = parse_pytest_output(full_output)
        metrics["exit_code"] = result.returncode

        # Validate results (including coverage threshold)
        validation_errors = validate_results(
            metrics, min_test_count, coverage_threshold
        )

        # Determine final exit code (fail if validation fails)
        exit_code = result.returncode
        if validation_errors:
            exit_code = 1

        # Format output based on mode
        if output_mode == "summary":
            output = format_summary(
                metrics, validation_errors, coverage_threshold
            )
        elif output_mode == "json":
            output = json.dumps(
                {
                    "metrics": metrics,
                    "validation_errors": validation_errors,
                    "success": len(validation_errors) == 0,
                    "coverage_threshold": coverage_threshold,
                },
                indent=2,
            )
        else:  # full
            # Include summary at the end of full output
            summary = format_summary(
                metrics, validation_errors, coverage_threshold
            )
            output = f"{full_output}\n\n{summary}"

        return exit_code, output

    except subprocess.TimeoutExpired:
        return 1, f"ERROR: pytest timed out after {timeout} seconds"
    except FileNotFoundError:
        return 1, "ERROR: pytest command not found. Is pytest installed?"
    except Exception as e:
        return 1, f"ERROR: Unexpected error running pytest: {e}"


def main():
    """Parse CLI arguments and execute pytest runner.

    Returns:
        Process exit code indicating success (0) or failure (non-zero).
    """
    parser = argparse.ArgumentParser(
        description="Run pytest with coverage and validation"
    )
    parser.add_argument(
        "--output",
        choices=["summary", "full", "json"],
        default="summary",
        help="Output mode: summary (default), full output, or JSON",
    )
    parser.add_argument(
        "--min-tests",
        type=int,
        default=1,
        help="Minimum expected test count (default: 1 for scoped tests)",
    )
    parser.add_argument(
        "--cwd", type=str, help="Working directory (defaults to project root)"
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
        help="Disable coverage reporting for faster runs",
    )
    parser.add_argument(
        "--coverage-source",
        type=str,
        default=None,
        help=(
            "Source module for coverage. If omitted, uses pyproject.toml "
            "[tool.coverage.run].source. Examples: 'adw', 'src/my_package'"
        ),
    )
    parser.add_argument(
        "--coverage-threshold",
        type=int,
        help=(
            "Minimum coverage percentage required (0-100). Fails if below "
            "threshold."
        ),
    )
    parser.add_argument(
        "--cov-report",
        type=str,
        default="term-missing",
        help=(
            "Coverage report format(s), comma-separated (default: "
            "'term-missing'). Examples: 'term-missing', 'html,xml', "
            "'term-missing,html:coverage_html'"
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure (-x flag)",
    )
    parser.add_argument(
        "pytest_args", nargs="*", help="Additional arguments to pass to pytest"
    )

    args = parser.parse_args()

    # Determine coverage setting (--no-coverage overrides --coverage)
    coverage_enabled = not args.no_coverage

    exit_code, output = run_pytest(
        args.pytest_args,
        output_mode=args.output,
        min_test_count=args.min_tests,
        cwd=args.cwd,
        timeout=args.timeout,
        coverage=coverage_enabled,
        coverage_source=args.coverage_source,
        coverage_threshold=args.coverage_threshold,
        cov_report=args.cov_report,
        fail_fast=args.fail_fast,
    )

    print(output)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
