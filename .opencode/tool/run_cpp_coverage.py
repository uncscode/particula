#!/usr/bin/env python3
"""C++ Coverage Runner Tool for ADW.

Generates coverage reports via gcovr (gcov or llvm-cov backends),
validates coverage thresholds, and provides summary/full/json outputs
with optional HTML reports and path filtering.

Usage:
    python3 .opencode/tool/run_cpp_coverage.py --build-dir build
    python3 .opencode/tool/run_cpp_coverage.py --build-dir build --threshold 80
    python3 .opencode/tool/run_cpp_coverage.py --build-dir build --html coverage_html --output json

Examples:
    # Generate coverage report
    python3 .opencode/tool/run_cpp_coverage.py --build-dir example_cpp_dev/build/coverage

    # Validate 80% line coverage threshold
    python3 .opencode/tool/run_cpp_coverage.py --build-dir build --threshold 80

    # Generate HTML report
    python3 .opencode/tool/run_cpp_coverage.py --build-dir build --html coverage_html

    # Filter to specific directory
    python3 .opencode/tool/run_cpp_coverage.py --build-dir build --filter src/ --output json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

OUTPUT_LINE_LIMIT = 500
OUTPUT_BYTE_LIMIT = 50_000
DEFAULT_TIMEOUT = 300


def _truncate_output(output: str) -> Tuple[str, bool, str]:
    """Truncate output to bounded lines/bytes with a notice.

    Args:
        output: Raw stdout or stderr text to be truncated.

    Returns:
        A tuple of:
            str: The possibly truncated output, with a trailing notice if
                truncated.
            bool: True if truncation occurred, otherwise False.
            str: A human-readable truncation notice, or an empty string if not
                truncated.
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


@dataclass
class CoverageMetrics:
    """Coverage metrics for a file or aggregated totals.

    Attributes:
        lines_covered: Number of lines executed during coverage run.
        lines_total: Total number of executable lines in the file.
        branches_covered: Number of branches taken during coverage run.
        branches_total: Total number of branches in the file.
        functions_covered: Number of functions executed during coverage run.
        functions_total: Total number of functions in the file.
    """

    lines_covered: int
    lines_total: int
    branches_covered: int
    branches_total: int
    functions_covered: int
    functions_total: int

    @property
    def line_percent(self) -> float:
        return 100.0 * self.lines_covered / self.lines_total if self.lines_total else 0.0

    @property
    def branch_percent(self) -> float:
        return 100.0 * self.branches_covered / self.branches_total if self.branches_total else 0.0

    @property
    def function_percent(self) -> float:
        return (
            100.0 * self.functions_covered / self.functions_total if self.functions_total else 0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "line_percent": self.line_percent,
                "branch_percent": self.branch_percent,
                "function_percent": self.function_percent,
            }
        )
        return payload


def parse_gcovr_output(output: str) -> Dict[str, CoverageMetrics]:
    """Parse gcovr JSON output into coverage metrics per file and total.

    Args:
        output: Raw JSON string produced by gcovr.

    Returns:
        A dictionary mapping file paths (and the special key ``"__total__"``) to
        :class:`CoverageMetrics` instances describing line, branch, and function
        coverage.

    Raises:
        RuntimeError: If the ``output`` string cannot be parsed as JSON.
    """

    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:  # pragma: no cover - surfaced via runtime error path
        raise RuntimeError(f"Failed to parse gcovr JSON: {exc}") from exc

    metrics: Dict[str, CoverageMetrics] = {}

    def build(entry: Dict[str, Any]) -> CoverageMetrics:
        return CoverageMetrics(
            lines_covered=int(entry.get("line_covered", 0) or 0),
            lines_total=int(entry.get("line_total", 0) or 0),
            branches_covered=int(entry.get("branch_covered", 0) or 0),
            branches_total=int(entry.get("branch_total", 0) or 0),
            functions_covered=int(entry.get("function_covered", 0) or 0),
            functions_total=int(entry.get("function_total", 0) or 0),
        )

    total_entry = {
        "line_covered": data.get("line_covered", 0),
        "line_total": data.get("line_total", 0),
        "branch_covered": data.get("branch_covered", 0),
        "branch_total": data.get("branch_total", 0),
        "function_covered": data.get("function_covered", 0),
        "function_total": data.get("function_total", 0),
    }
    metrics["__total__"] = build(total_entry)

    for file_entry in data.get("files", []) or []:
        filename = file_entry.get("filename")
        if not filename:
            continue
        metrics[filename] = build(file_entry)

    return metrics


def validate_threshold(metrics: CoverageMetrics, threshold: Optional[float]) -> Tuple[bool, str]:
    """Validate line coverage against a threshold (percentage).

    Args:
        metrics: Coverage metrics whose ``line_percent`` value will be
            validated.
        threshold: Minimum required line coverage percentage. If ``None``,
            validation is skipped.

    Returns:
        A tuple of:
            bool: ``True`` if the line coverage meets or exceeds the threshold,
                or if no threshold is provided.
            str: Human-readable message describing the validation result.
    """

    if threshold is None:
        return True, "No threshold provided; skipping validation"

    percent = metrics.line_percent
    passed = percent >= threshold
    comparator = "meets" if passed else "below"
    message = f"Line coverage {percent:.1f}% {comparator} threshold {threshold:.1f}%"
    return passed, message


def format_summary(
    metrics: CoverageMetrics,
    threshold: Optional[float],
    files_below_threshold: List[Tuple[str, float]],
    duration: Optional[float],
    tool: str,
    validation_errors: List[str],
) -> str:
    """Format human-readable summary of coverage results.

    Args:
        metrics: Aggregated coverage metrics including lines, branches, and
            functions.
        threshold: Optional line coverage percentage threshold used for
            validation. If ``None``, no threshold is applied.
        files_below_threshold: List of file paths and their line coverage
            percentages for files that are below the specified coverage
            threshold.
        duration: Optional duration in seconds for running the coverage tool. If
            ``None``, duration is omitted from the summary.
        tool: Name of the coverage tool/backend used (for example, ``"gcov"`` or
            ``"llvm-cov"``).
        validation_errors: List of validation error messages produced during
            coverage validation.

    Returns:
        A multi-line, human-readable summary string describing overall coverage,
        threshold status, any files below the threshold, execution duration
        (when provided), and validation results.
    """

    lines: List[str] = []
    lines.append("=" * 60)
    lines.append(f"C++ COVERAGE SUMMARY ({tool} via gcovr)")
    lines.append("=" * 60)

    lines.append("\nOverall Coverage:")
    lines.append(
        f"  Lines:     {metrics.lines_covered}/{metrics.lines_total} ({metrics.line_percent:.1f}%)"
    )
    branch_pct = f"{metrics.branch_percent:.1f}%"
    lines.append(f"  Branches:  {metrics.branches_covered}/{metrics.branches_total} ({branch_pct})")
    func_pct = f"{metrics.function_percent:.1f}%"
    lines.append(f"  Functions: {metrics.functions_covered}/{metrics.functions_total} ({func_pct})")

    if threshold is not None:
        threshold_status = "PASSED" if metrics.line_percent >= threshold else "FAILED"
        lines.append(f"\nThreshold: {threshold:.1f}% lines ({threshold_status})")
    else:
        lines.append("\nThreshold: not set")

    lines.append("\nFiles Below Threshold:")
    if threshold is None or not files_below_threshold:
        lines.append("  None")
    else:
        for path, percent in files_below_threshold:
            lines.append(f"  - {path}: {percent:.1f}% (target: {threshold:.1f}%)")

    if duration is not None:
        lines.append(f"\nDuration: {duration:.2f}s")

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


def run_coverage(
    build_dir: Path,
    threshold: Optional[float] = None,
    tool: str = "gcov",
    filter_path: Optional[str] = None,
    html_dir: Optional[Path] = None,
    timeout: int = DEFAULT_TIMEOUT,
    output_mode: str = "summary",
) -> Tuple[int, str]:
    """Run gcovr to generate coverage metrics and render output.

    This is the main entry point used by the CLI wrapper. It invokes gcovr with
    the requested backend, parses the JSON metrics, optionally enforces a
    minimum coverage threshold, and returns formatted output in summary,
    full-text, or JSON form.

    Args:
        build_dir: Path to the compiled C/C++ build directory that contains
            coverage data (for example, ``.gcda`` files). This directory is
            passed to gcovr as the project root via ``--root``.
        threshold: Optional minimum line coverage percentage that must be met by
            the overall project and each file. When provided and the threshold
            is not met, validation errors are recorded and the returned exit
            code is non-zero.
        tool: Coverage backend to use. Supported values are ``"gcov"`` and
            ``"llvm-cov"``. The default is ``"gcov"``.
        filter_path: Optional path or pattern used to restrict coverage
            collection to a subset of files. When set, it is passed to gcovr via
            the ``--filter`` option.
        html_dir: Optional directory where an HTML coverage report should be
            written. When provided, gcovr is invoked with the appropriate HTML
            output flags and the report is written into this directory.
        timeout: Maximum number of seconds to allow the gcovr process to run
            before it is terminated and treated as a timeout. A timeout is
            reported in the output and results in a non-zero exit code.
        output_mode: Output format selector. ``"summary"`` returns a short
            textual summary, ``"full"`` returns a detailed per-file report, and
            ``"json"`` returns a JSON payload suitable for machine consumption.

    Returns:
        A tuple ``(exit_code, output)`` where:

        * ``exit_code`` is the status code that should be propagated by the CLI
          (``0`` on success, non-zero on validation or execution failure).
        * ``output`` is the rendered coverage result as a string. For
          ``output_mode == "json"`` this is a JSON document containing metrics
          and validation details; otherwise it is human-readable text.
    """

    start_time = time.perf_counter()
    validation_errors: List[str] = []
    metrics_map: Dict[str, CoverageMetrics] = {"__total__": CoverageMetrics(0, 0, 0, 0, 0, 0)}
    files_below_threshold: List[Tuple[str, float]] = []
    combined_output = ""
    timed_out = False
    exit_code = 1

    build_path = Path(build_dir)
    if not build_path.exists():
        validation_errors.append(f"Build directory does not exist: {build_path}")
        summary = format_summary(
            metrics_map["__total__"],
            threshold,
            files_below_threshold,
            None,
            tool,
            validation_errors,
        )
        if output_mode == "json":
            payload = {
                "metrics": {k: v.to_dict() for k, v in metrics_map.items()},
                "files_below_threshold": files_below_threshold,
                "validation_errors": validation_errors,
                "success": False,
                "output": summary,
                "truncated": False,
                "truncation_notice": "",
                "tool": tool,
                "html_dir": str(html_dir) if html_dir else None,
                "threshold": threshold,
                "duration": None,
            }
            return 1, json.dumps(payload, indent=2)
        return 1, summary

    cmd: List[str] = ["gcovr", "--json", "-", "--root", str(build_path)]
    if tool == "gcov":
        cmd.append("--gcov-executable=gcov")
    elif tool == "llvm-cov":
        cmd.extend(["--gcov-executable=llvm-cov", "--llvm-cov", "llvm-cov"])

    if filter_path:
        cmd.extend(["--filter", str(filter_path)])

    html_output = None
    if html_dir:
        html_dir.mkdir(parents=True, exist_ok=True)
        html_output = Path(html_dir) / "index.html"
        cmd.extend(["--html", "--html-details", "--output", str(html_output)])

    try:
        process = subprocess.run(
            cmd,
            cwd=str(build_path),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        combined_output = "".join([process.stdout or "", process.stderr or ""])
        exit_code = process.returncode
        metrics_map = parse_gcovr_output(process.stdout or "{}")
    except subprocess.TimeoutExpired:
        timed_out = True
        validation_errors.append(f"gcovr timed out after {timeout} seconds")
    except FileNotFoundError:
        validation_errors.append("gcovr command not found")
    except RuntimeError as exc:
        validation_errors.append(str(exc))

    overall_metrics = metrics_map.get("__total__", CoverageMetrics(0, 0, 0, 0, 0, 0))

    if threshold is not None:
        files_below_threshold = [
            (path, metric.line_percent)
            for path, metric in metrics_map.items()
            if path != "__total__" and metric.line_percent < threshold
        ]

    threshold_passed, threshold_message = validate_threshold(overall_metrics, threshold)
    if threshold is not None and not threshold_passed:
        validation_errors.append(threshold_message)

    if not timed_out and exit_code != 0:
        validation_errors.append(f"gcovr exited with code {exit_code}")

    duration = time.perf_counter() - start_time

    success = threshold_passed and not validation_errors and exit_code == 0 and not timed_out

    truncated_output, truncated, truncation_notice = _truncate_output(combined_output)

    if output_mode == "json":
        payload = {
            "metrics": {key: value.to_dict() for key, value in metrics_map.items()},
            "files_below_threshold": files_below_threshold,
            "validation_errors": validation_errors,
            "success": success,
            "output": truncated_output,
            "truncated": truncated,
            "truncation_notice": truncation_notice,
            "tool": tool,
            "html_dir": str(html_dir) if html_dir else None,
            "threshold": threshold,
            "duration": duration,
        }
        return (0 if success else 1), json.dumps(payload, indent=2)

    if output_mode == "full":
        return (0 if success else 1), truncated_output

    summary = format_summary(
        overall_metrics,
        threshold,
        files_below_threshold,
        duration,
        tool,
        validation_errors,
    )
    return (0 if success else 1), summary


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the coverage tool.

    Args:
        argv: Optional list of command-line arguments. If None, uses
            ``sys.argv[1:]``.

    Returns:
        argparse.Namespace: Parsed arguments for the C++ coverage runner.
    """

    parser = argparse.ArgumentParser(
        description="Generate C++ coverage reports with threshold validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--build-dir", type=Path, required=True, help="CMake build directory")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum line coverage percentage (0-100)",
    )
    parser.add_argument(
        "--tool",
        choices=["gcov", "llvm-cov"],
        default="gcov",
        help="Coverage backend to use (default: gcov)",
    )
    parser.add_argument("--filter", default=None, help="Filter coverage to path/pattern")
    parser.add_argument("--html", type=Path, default=None, help="Generate HTML report in directory")
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--output",
        choices=["summary", "full", "json"],
        default="summary",
        help="Output mode",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Run the C++ coverage CLI entry point.

    Args:
        argv: Optional list of command-line arguments. If None, arguments are
            read from sys.argv[1:].
    """
    args = _parse_args(argv)
    exit_code, output = run_coverage(
        build_dir=args.build_dir,
        threshold=args.threshold,
        tool=args.tool,
        filter_path=args.filter,
        html_dir=args.html,
        timeout=args.timeout,
        output_mode=args.output,
    )
    print(output)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
