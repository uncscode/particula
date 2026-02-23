#!/usr/bin/env python3
"""Bun Test Runner Tool for ADW.

Runs bun test with parsing, validation, and structured outputs
modeled after the run_ctest tool. Supports filtering, timeouts,
and multiple output modes (summary/full/json).

Usage:
    python3 .opencode/tool/run_bun_test.py
    python3 .opencode/tool/run_bun_test.py --test-path __tests__/foo.test.ts
    python3 .opencode/tool/run_bun_test.py --filter "My suite" --output json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

OUTPUT_LINE_LIMIT = 500
OUTPUT_BYTE_LIMIT = 50_000
DEFAULT_TIMEOUT = 300


def _truncate_output(output: str) -> Tuple[str, bool, str]:
    """Truncate output to bounded lines/bytes with a notice.

    Args:
        output: Raw combined stdout and stderr from bun test.

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


def parse_bun_output(output: str) -> Dict[str, Any]:
    """Parse bun test output to extract key metrics.

    Args:
        output: Combined stdout and stderr emitted by bun test.

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
        "bun_missing": False,
        "test_path_error": False,
        "timeout_seconds": None,
    }

    pass_matches = re.findall(r"(\d+)\s+pass", output)
    fail_matches = re.findall(r"(\d+)\s+fail", output)
    skip_matches = re.findall(r"(\d+)\s+skip", output)

    if pass_matches:
        metrics["passed"] = int(pass_matches[-1])
    if fail_matches:
        metrics["failed"] = int(fail_matches[-1])
    if skip_matches:
        metrics["skipped"] = int(skip_matches[-1])

    if pass_matches or fail_matches or skip_matches:
        metrics["total"] = metrics["passed"] + metrics["failed"] + metrics["skipped"]

    duration_matches = re.findall(r"\[(\d+\.?\d*)ms\]", output)
    if duration_matches:
        try:
            metrics["duration"] = float(duration_matches[-1]) / 1000.0
        except ValueError:
            metrics["duration"] = None

    failed_tests: List[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("✗") or stripped.startswith("✕"):
            name = stripped.lstrip("✗✕").strip()
        elif stripped.startswith("x "):
            name = stripped[2:].strip()
        else:
            continue

        if "[" in name:
            name = name.split("[", 1)[0].strip()
        if name:
            failed_tests.append(name)

    if failed_tests:
        metrics["failed_tests"] = list(dict.fromkeys(failed_tests))

    return metrics


def validate_results(metrics: Dict[str, Any], min_test_count: int = 1) -> List[str]:
    """Validate bun test results against expected criteria.

    Args:
        metrics: Parsed metrics produced by ``parse_bun_output`` or augmented
            by ``run_bun_test``.
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
        errors.append(f"Bun test timed out{suffix}")

    if metrics.get("bun_missing"):
        errors.append("bun command not found")

    if metrics.get("test_path_error"):
        errors.append("Test path does not exist")

    return errors


def format_summary(metrics: Dict[str, Any], validation_errors: List[str]) -> str:
    """Format a human-readable summary of bun test results.

    Args:
        metrics: Metrics dictionary containing counts, duration, and failures.
        validation_errors: Validation errors produced by ``validate_results``.

    Returns:
        Multi-line summary string styled after the CTest runner output.
    """

    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("BUN TEST SUMMARY")
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


def run_bun_test(
    test_path: Optional[str] = None,
    test_filter: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    min_test_count: int = 1,
    output_mode: str = "summary",
    bail: bool = False,
    cwd: Optional[str] = None,
) -> Tuple[int, str]:
    """Run bun test with parsing and validation.

    Args:
        test_path: Optional test file or directory path to run.
        test_filter: Optional test name pattern passed to bun.
        timeout: Timeout in seconds for subprocess execution.
        min_test_count: Minimum expected number of tests to run.
        output_mode: One of ``summary``, ``full``, or ``json``.
        bail: Whether to stop after first failure (``--bail``).
        cwd: Optional working directory for running bun.

    Returns:
        Tuple of (exit_code, output_string) where exit_code is 0 on success and
        1 on validation or execution failure.
    """

    metrics = parse_bun_output("")
    metrics["timeout_seconds"] = timeout
    validation_errors: List[str] = []

    default_cwd = Path(__file__).resolve().parent
    cwd_path = Path(cwd) if cwd else default_cwd

    if test_path:
        test_path_obj = Path(test_path)
        if not test_path_obj.is_absolute():
            test_path_obj = cwd_path / test_path_obj
        if not test_path_obj.exists():
            metrics["test_path_error"] = True
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

    cmd: List[str] = ["bun", "test"]
    if test_path:
        cmd.append(test_path)
    cmd.extend(["--timeout", str(timeout)])
    if bail:
        cmd.append("--bail")
    if test_filter:
        cmd.extend(["--test-name-pattern", test_filter])

    try:
        process = subprocess.run(
            cmd,
            cwd=str(cwd_path),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        combined_output = "".join([process.stdout or "", process.stderr or ""])
        metrics.update(parse_bun_output(combined_output))
        metrics["exit_code"] = process.returncode
        validation_errors = validate_results(metrics, min_test_count=min_test_count)
    except subprocess.TimeoutExpired:
        metrics["timeout"] = True
        metrics["exit_code"] = 1
        combined_output = f"ERROR: Bun test timed out after {timeout} seconds"
        validation_errors = validate_results(metrics, min_test_count=min_test_count)
    except FileNotFoundError:
        metrics["bun_missing"] = True
        metrics["exit_code"] = 1
        combined_output = "ERROR: bun command not found"
        validation_errors = validate_results(metrics, min_test_count=min_test_count)

    success = (
        metrics.get("failed", 0) == 0
        and not validation_errors
        and metrics.get("exit_code", 1) == 0
        and not metrics.get("timeout")
        and not metrics.get("bun_missing")
        and not metrics.get("test_path_error")
    )

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
    if output_mode == "full":
        truncated_output, _, _ = _truncate_output(combined_output)
        return (0 if success else 1), truncated_output
    summary_output = format_summary(metrics, validation_errors)
    return (0 if success else 1), summary_output


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the bun test runner tool.

    Args:
        argv: Optional list of arguments to use instead of ``sys.argv``.

    Returns:
        Parsed namespace containing the CLI options.
    """

    parser = argparse.ArgumentParser(
        description="Run bun test with ADW-style parsing and validation",
        epilog=(
            "Examples:\n"
            "  python3 .opencode/tool/run_bun_test.py\n"
            "  python3 .opencode/tool/run_bun_test.py --test-path __tests__/foo.test.ts\n"
            "  python3 .opencode/tool/run_bun_test.py --filter 'My suite' --output json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test-path", help="Test file or directory path to run")
    parser.add_argument("--filter", dest="filter", help="Test name pattern filter")
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
    parser.add_argument("--bail", action="store_true", help="Stop after the first failure")
    parser.add_argument("--cwd", help="Working directory for bun test")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Run the bun test tool with parsed arguments and exit with its status.

    Args:
        argv: Optional list of arguments to override ``sys.argv`` when invoking the tool.

    Raises:
        SystemExit: Always raised with the exit code returned by ``run_bun_test``.
    """

    args = _parse_args(argv)
    exit_code, output = run_bun_test(
        test_path=args.test_path,
        test_filter=args.filter,
        timeout=args.timeout,
        min_test_count=args.min_tests,
        output_mode=args.output,
        bail=args.bail,
        cwd=args.cwd,
    )
    print(output)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
