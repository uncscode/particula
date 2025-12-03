#!/usr/bin/env python3
"""Pytest Runner Tool with Coverage

Runs pytest with coverage and returns either full output or a summary.
This tool validates test results to prevent false positives.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_pytest_output(output: str) -> Dict:
    """Parse pytest output to extract key metrics.

    Args:
        output: The full pytest output text

    Returns:
        Dictionary with parsed metrics
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

    return result


def format_summary(metrics: Dict, validation_errors: List[str]) -> str:
    """Format a human-readable summary of test results.

    Args:
        metrics: Parsed metrics from pytest output
        validation_errors: List of validation errors

    Returns:
        Formatted summary string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("PYTEST SUMMARY")
    lines.append("=" * 60)

    # Test counts
    lines.append(f"\nTests Run: {metrics['total']}")
    lines.append(f"  ✓ Passed:  {metrics['passed']}")
    if metrics["failed"] > 0:
        lines.append(f"  ✗ Failed:  {metrics['failed']}")
    if metrics["errors"] > 0:
        lines.append(f"  ⚠ Errors:  {metrics['errors']}")
    if metrics["skipped"] > 0:
        lines.append(f"  ⊘ Skipped: {metrics['skipped']}")

    # Duration
    if metrics["duration"]:
        lines.append(f"\nDuration: {metrics['duration']:.2f}s")

    # Coverage
    if metrics["coverage_pct"] is not None:
        lines.append(f"Coverage: {metrics['coverage_pct']}%")

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
            lines.append(f"  ✗ {error}")
    else:
        lines.append("VALIDATION: PASSED")
        lines.append("=" * 60)
        lines.append("  ✓ All validation checks passed")

    return "\n".join(lines)


def validate_results(metrics: Dict, min_test_count: int = 500) -> List[str]:
    """Validate pytest results against expected criteria.

    Args:
        metrics: Parsed metrics from pytest output
        min_test_count: Minimum expected number of passing tests

    Returns:
        List of validation errors (empty if all checks pass)
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

    return errors


def run_pytest(
    args: List[str],
    output_mode: str = "summary",
    min_test_count: int = 500,
    cwd: Optional[str] = None,
    timeout: int = 600,
) -> Tuple[int, str]:
    """Run pytest with the specified arguments.

    Args:
        args: Additional pytest arguments
        output_mode: Either "summary" or "full"
        min_test_count: Minimum expected test count for validation
        cwd: Working directory (defaults to project root)
        timeout: Timeout in seconds (default: 600 = 10 minutes)

    Returns:
        Tuple of (exit_code, output_string)
    """
    # Build pytest command
    cmd = ["pytest", "-v", "--tb=short"]

    # Add coverage if not already specified
    if not any("--cov" in arg for arg in args):
        cmd.extend(["--cov=particula", "--cov-report=term-missing"])

    # Add user arguments
    cmd.extend(args)

    # Determine working directory
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
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
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

        # Validate results
        validation_errors = validate_results(metrics, min_test_count)

        # Determine final exit code (fail if validation fails)
        exit_code = result.returncode
        if validation_errors:
            exit_code = 1

        # Format output based on mode
        if output_mode == "summary":
            output = format_summary(metrics, validation_errors)
        elif output_mode == "json":
            output = json.dumps(
                {
                    "metrics": metrics,
                    "validation_errors": validation_errors,
                    "success": len(validation_errors) == 0,
                },
                indent=2,
            )
        else:  # full
            # Include summary at the end of full output
            summary = format_summary(metrics, validation_errors)
            output = f"{full_output}\n\n{summary}"

        return exit_code, output

    except subprocess.TimeoutExpired:
        return 1, f"ERROR: pytest timed out after {timeout} seconds"
    except FileNotFoundError:
        return 1, "ERROR: pytest command not found. Is pytest installed?"
    except Exception as e:
        return 1, f"ERROR: Unexpected error running pytest: {e}"


def main():
    """Main entry point for CLI usage."""
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
        default=500,
        help="Minimum expected test count (default: 500)",
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
    parser.add_argument(
        "pytest_args", nargs="*", help="Additional arguments to pass to pytest"
    )

    args = parser.parse_args()

    exit_code, output = run_pytest(
        args.pytest_args,
        output_mode=args.output,
        min_test_count=args.min_tests,
        cwd=args.cwd,
        timeout=args.timeout,
    )

    print(output)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
