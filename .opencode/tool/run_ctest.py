#!/usr/bin/env python3
"""CTest Runner Tool for ADW.

Runs CTest with parsing, validation, and structured outputs
modeled after the run_pytest tool. Supports filtering, parallel
execution, timeouts, and multiple output modes (summary/full/json).

Usage:
    python3 run_ctest.py --build-dir build
    python3 run_ctest.py --build-dir build -R "MyTest.*" --min-tests 5
    python3 run_ctest.py --build-dir build -E "slow" --output json

Examples:
    # Run all tests from build directory
    python3 .opencode/tool/run_ctest.py --build-dir example_cpp_dev/build

    # Filter tests with regex pattern
    python3 .opencode/tool/run_ctest.py --build-dir build -R "test_add"

    # Exclude tests with regex pattern
    python3 .opencode/tool/run_ctest.py --build-dir build -E "slow"

    # Parallel execution
    python3 .opencode/tool/run_ctest.py --build-dir build -j 4

    # With minimum test count validation
    python3 .opencode/tool/run_ctest.py --build-dir build --min-tests 5

    # JSON output for programmatic use
    python3 .opencode/tool/run_ctest.py --build-dir build --output json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

OUTPUT_LINE_LIMIT = 500
OUTPUT_BYTE_LIMIT = 50_000
DEFAULT_TIMEOUT = 300


def _truncate_output(output: str) -> Tuple[str, bool, str]:
    """Truncate output to bounded lines/bytes with a notice.

    Args:
        output: Raw combined stdout and stderr from CTest.

    Returns:
        Tuple of the possibly truncated output, whether truncation occurred,
        and a truncation notice string.
    """

    lines = output.splitlines()
    truncated = False
    notice_parts: List[str] = []

    if len(lines) > OUTPUT_LINE_LIMIT:
        lines = lines[:OUTPUT_LINE_LIMIT]
        truncated = True
        notice_parts.append(f"Output truncated to {OUTPUT_LINE_LIMIT} lines")

    joined = "\n".join(lines)
    if len(joined.encode("utf-8")) > OUTPUT_BYTE_LIMIT:
        encoded = joined.encode("utf-8")[:OUTPUT_BYTE_LIMIT]
        joined = encoded.decode("utf-8", errors="ignore")
        truncated = True
        notice_parts.append(f"Output truncated to {OUTPUT_BYTE_LIMIT // 1024}KB")

    notice = "; ".join(notice_parts) if truncated else ""
    if truncated:
        joined = f"{joined}\n...\n{notice}"
    return joined, truncated, notice


def parse_ctest_output(output: str) -> Dict[str, Any]:
    """Parse CTest output to extract key metrics.

    Args:
        output: Combined stdout and stderr emitted by CTest.

    Returns:
        Metrics dictionary containing test counts, duration, failed test names,
        control flags, and placeholders for caller-set values such as
        ``exit_code``.
    """

    metrics: Dict[str, Any] = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "total": 0,
        "duration": None,
        "failed_tests": [],
        "exit_code": None,
        "timeout": False,
        "ctest_missing": False,
        "build_dir_error": False,
        "timeout_seconds": None,
    }

    summary_pattern = re.compile(
        r"(\d+)%\s+tests\s+passed,\s+(\d+)\s+tests?\s+failed\s+out\s+of\s+(\d+)",
        re.IGNORECASE,
    )
    summary_match = summary_pattern.search(output)
    if summary_match:
        total = int(summary_match.group(3))
        failed = int(summary_match.group(2))
        passed = total - failed
        metrics.update({"total": total, "failed": failed, "passed": passed})

    if "No tests were found" in output:
        metrics.update({"total": 0, "failed": 0, "passed": 0})

    duration_match = re.search(r"Total Test time \(real\)\s*=\s*([\d.]+)\s*sec", output)
    if duration_match:
        try:
            metrics["duration"] = float(duration_match.group(1))
        except ValueError:
            metrics["duration"] = None

    failed_tests: List[str] = []
    failed_line_pattern = re.compile(r"Test\s+#\d+:\s+([^\s]+)\s+.*\*\*\*Failed", re.IGNORECASE)
    for match in failed_line_pattern.finditer(output):
        failed_tests.append(match.group(1))

    section_pattern = re.compile(r"^\s*\d+\s+-\s+(.+?)(?:\s+\(Failed\))?$", re.MULTILINE)
    for match in section_pattern.finditer(output):
        failed_tests.append(match.group(1).strip())

    if failed_tests:
        metrics["failed_tests"] = list(dict.fromkeys(failed_tests))

    return metrics


def validate_results(metrics: Dict[str, Any], min_test_count: int = 1) -> List[str]:
    """Validate CTest results against expected criteria.

    Args:
        metrics: Parsed metrics produced by ``parse_ctest_output`` or augmented
            by ``run_ctest``.
        min_test_count: Minimum number of tests expected to run.

    Returns:
        List of validation error messages. Empty list indicates success.
    """

    errors: List[str] = []

    if metrics.get("failed", 0) > 0:
        errors.append(f"Found {metrics['failed']} failed test(s)")

    total = metrics.get("total", 0)
    if total == 0:
        errors.append("No tests were collected or run")
    if total < min_test_count:
        errors.append(
            f"Expected at least {min_test_count} tests to run, but only {total} were collected"
        )

    if metrics.get("timeout"):
        seconds = metrics.get("timeout_seconds")
        suffix = f" after {seconds} seconds" if seconds is not None else ""
        errors.append(f"CTest timed out{suffix}")

    if metrics.get("ctest_missing"):
        errors.append("ctest command not found")

    if metrics.get("build_dir_error"):
        errors.append("Build directory is missing CTestTestfile.cmake")

    return errors


def format_summary(metrics: Dict[str, Any], validation_errors: List[str]) -> str:
    """Format a human-readable summary of CTest results.

    Args:
        metrics: Metrics dictionary containing counts, duration, and failures.
        validation_errors: Validation errors produced by ``validate_results``.

    Returns:
        Multi-line summary string styled after the pytest runner output.
    """

    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("CTEST SUMMARY")
    lines.append("=" * 60)

    lines.append(f"\nTests Run: {metrics['total']}")
    lines.append(f"  Passed:  {metrics['passed']}")
    if metrics["failed"] > 0:
        lines.append(f"  Failed:  {metrics['failed']}")
    if metrics["skipped"] > 0:
        lines.append(f"  Skipped: {metrics['skipped']}")

    if metrics.get("duration") is not None:
        lines.append(f"\nDuration: {metrics['duration']:.2f}s")

    if metrics.get("failed_tests"):
        lines.append(f"\nFailed Tests ({len(metrics['failed_tests'])}):")
        for test in metrics["failed_tests"][:10]:
            lines.append(f"  - {test}")
        if len(metrics["failed_tests"]) > 10:
            remaining = len(metrics["failed_tests"]) - 10
            lines.append(f"  ... and {remaining} more")

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


def run_ctest(
    build_dir: Path,
    include_filter: Optional[str] = None,
    exclude_filter: Optional[str] = None,
    parallel: int = 0,
    timeout: int = DEFAULT_TIMEOUT,
    min_test_count: int = 1,
    output_mode: str = "summary",
) -> Tuple[int, str]:
    """Run CTest with parsing and validation.

    Args:
        build_dir: Directory containing ``CTestTestfile.cmake``.
        include_filter: Regex pattern passed to ``-R`` to include tests.
        exclude_filter: Regex pattern passed to ``-E`` to exclude tests.
        parallel: Number of parallel jobs (``-j``) when > 0.
        timeout: Timeout in seconds for subprocess execution.
        min_test_count: Minimum expected number of tests to run.
        output_mode: One of ``summary``, ``full``, or ``json``.

    Returns:
        Tuple of (exit_code, output_string) where exit_code is 0 on success and
        1 on validation or execution failure.
    """

    metrics = parse_ctest_output("")
    metrics["timeout_seconds"] = timeout
    validation_errors: List[str] = []

    build_path = Path(build_dir)
    ctest_file = build_path / "CTestTestfile.cmake"
    if not build_path.exists() or not ctest_file.exists():
        metrics["build_dir_error"] = True
        validation_errors = validate_results(metrics, min_test_count=min_test_count)
        summary = format_summary(metrics, validation_errors)
        if output_mode == "json":
            payload = {
                "metrics": metrics,
                "validation_errors": validation_errors,
                "success": False,
                "output": summary,
                "truncated": False,
                "truncation_notice": "",
            }
            return 1, json.dumps(payload, indent=2)
        return 1, summary

    cmd: List[str] = ["ctest", "--output-on-failure"]
    if output_mode == "full":
        cmd.append("-V")
    if include_filter:
        cmd.extend(["-R", include_filter])
    if exclude_filter:
        cmd.extend(["-E", exclude_filter])
    if parallel > 0:
        cmd.extend(["-j", str(parallel)])

    def _to_text(value: Optional[Union[str, bytes]]) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    try:
        process = subprocess.run(
            cmd,
            cwd=str(build_path),
            capture_output=True,
            text=False,
            timeout=timeout,
        )
        stdout_decoded = _to_text(process.stdout)
        stderr_decoded = _to_text(process.stderr)
        combined_output = stdout_decoded + stderr_decoded
        metrics.update(parse_ctest_output(combined_output))
        metrics["exit_code"] = process.returncode
        validation_errors = validate_results(metrics, min_test_count=min_test_count)
    except subprocess.TimeoutExpired as exc:
        partial_out = _to_text(exc.stdout)
        partial_err = _to_text(exc.stderr)
        timeout_message = f"ERROR: CTest timed out after {timeout} seconds"
        combined_output = "\n".join(
            part for part in [partial_out.strip(), partial_err.strip(), timeout_message] if part
        )
        metrics.update(parse_ctest_output(combined_output))
        metrics["timeout"] = True
        metrics["exit_code"] = 1
        validation_errors = validate_results(metrics, min_test_count=min_test_count)
    except FileNotFoundError:
        metrics["ctest_missing"] = True
        metrics["exit_code"] = 1
        combined_output = "ERROR: ctest command not found"
        validation_errors = validate_results(metrics, min_test_count=min_test_count)

    success = (
        metrics.get("failed", 0) == 0
        and not validation_errors
        and metrics.get("exit_code", 1) == 0
        and not metrics.get("timeout")
        and not metrics.get("ctest_missing")
        and not metrics.get("build_dir_error")
    )

    # Handle output modes
    if output_mode == "json":
        truncated_output, truncated, notice = _truncate_output(combined_output)
        payload = {
            "metrics": metrics,
            "validation_errors": validation_errors,
            "success": success,
            "output": truncated_output,
            "truncated": truncated,
            "truncation_notice": notice,
        }
        return (0 if success else 1), json.dumps(payload, indent=2)
    elif output_mode == "full":
        # _truncate_output already includes the notice in the output
        truncated_output, truncated, notice = _truncate_output(combined_output)
        return (0 if success else 1), truncated_output
    else:  # output_mode == "summary"
        summary_output = format_summary(metrics, validation_errors)
        return (0 if success else 1), summary_output


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the CTest runner tool.

    Args:
        argv: Optional list of arguments to use instead of ``sys.argv``.

    Returns:
        Parsed namespace containing the CLI options.
    """
    parser = argparse.ArgumentParser(
        description="Run CTest with ADW-style parsing and validation",
        epilog=(
            "Examples:\n"
            "  python3 .opencode/tool/run_ctest.py --build-dir example_cpp_dev/build\n"
            "  python3 .opencode/tool/run_ctest.py --build-dir build -R 'MyTest.*'\n"
            "  python3 .opencode/tool/run_ctest.py --build-dir build -E 'slow' -j 4 --output json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--build-dir", type=Path, required=True, help="CMake build directory")
    parser.add_argument("-R", "--include", dest="include", help="Regex to include tests")
    parser.add_argument("-E", "--exclude", dest="exclude", help="Regex to exclude tests")
    parser.add_argument("-j", "--parallel", type=int, default=0, help="Number of parallel jobs")
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout in seconds (default: 300)"
    )
    parser.add_argument(
        "--min-tests", type=int, default=1, help="Minimum expected number of tests (default: 1)"
    )
    parser.add_argument(
        "--output",
        choices=["summary", "full", "json"],
        default="summary",
        help="Output format",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Run the CTest tool with parsed arguments and exit with its status.

    Args:
        argv: Optional list of arguments to override ``sys.argv`` when invoking the tool.

    Raises:
        SystemExit: Always raised with the exit code returned by ``run_ctest``.
    """
    args = _parse_args(argv)
    exit_code, output = run_ctest(
        build_dir=args.build_dir,
        include_filter=args.include,
        exclude_filter=args.exclude,
        parallel=args.parallel,
        timeout=args.timeout,
        min_test_count=args.min_tests,
        output_mode=args.output,
    )
    print(output)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
