#!/usr/bin/env python3
"""C++ Linter Runner Tool for ADW.

Runs configured C++ linters (clang-format, clang-tidy, cppcheck)
with optional auto-fixing and bounded outputs suitable for agents
without shell access. Mirrors the structure of ``run_linters.py``
with summary/full/json output modes and predictable resource usage.

Usage:
    python3 .opencode/tool/run_cpp_linters.py --source-dir example_cpp_dev
    python3 .opencode/tool/run_cpp_linters.py --source-dir src --auto-fix
    python3 .opencode/tool/run_cpp_linters.py --source-dir src --linters clang-format
    python3 .opencode/tool/run_cpp_linters.py --source-dir src --build-dir build --linters clang-tidy
    python3 .opencode/tool/run_cpp_linters.py --source-dir src --output json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

OUTPUT_LINE_LIMIT = 500
OUTPUT_BYTE_LIMIT = 50_000
ISSUE_CAPTURE_LIMIT = 200
CLANG_TIDY_BATCH_SIZE = 25
DEFAULT_TIMEOUTS: Dict[str, int] = {
    "clang-format": 120,
    "clang-tidy": 300,
    "cppcheck": 240,
}


@dataclass
class LinterResult:
    """Store results from a single C++ linter run."""

    name: str
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    errors: int = 0
    warnings: int = 0
    files_checked: int = 0
    files_with_issues: int = 0
    success: bool = True
    skipped: bool = False
    error_message: Optional[str] = None
    issues: List[str] = field(default_factory=list)
    duration: Optional[float] = None
    truncated: bool = False
    timeout: bool = False


def _truncate_output(output: str) -> Tuple[str, bool, str]:
    """Truncate output to bounded lines/bytes with notice.

    Args:
        output: Combined stdout and stderr text to constrain.

    Returns:
        A tuple of truncated output, whether truncation occurred, and a
        human-readable notice describing truncation.
    """

    lines = output.splitlines()
    truncated = False
    notices: List[str] = []

    if len(lines) > OUTPUT_LINE_LIMIT:
        lines = lines[:OUTPUT_LINE_LIMIT]
        truncated = True
        notices.append(f"Output truncated to {OUTPUT_LINE_LIMIT} lines")

    joined = "\n".join(lines)
    if len(joined.encode("utf-8")) > OUTPUT_BYTE_LIMIT:
        encoded = joined.encode("utf-8")[:OUTPUT_BYTE_LIMIT]
        joined = encoded.decode("utf-8", errors="ignore")
        truncated = True
        notices.append(f"Output truncated to {OUTPUT_BYTE_LIMIT // 1024}KB")

    notice = "; ".join(notices) if truncated else ""
    if truncated:
        joined = f"{joined}\n...\n{notice}"
    return joined, truncated, notice


def _bounded_append(collection: List[str], value: str, limit: int = ISSUE_CAPTURE_LIMIT) -> None:
    """Append while respecting a maximum size."""

    if len(collection) < limit:
        collection.append(value)


def check_linter_available(linter: str) -> bool:
    """Check if linter command is available on PATH."""

    return shutil.which(linter) is not None


def get_cpp_files(source_dir: str) -> List[Path]:
    """Collect C/C++ source and header files under a directory.

    Args:
        source_dir: Root directory to search for C/C++ files.

    Returns:
        Sorted list of discovered C/C++ file paths.
    """

    root = Path(source_dir)
    extensions = {".cpp", ".cc", ".cxx", ".c", ".hpp", ".h"}
    files = {
        path for path in root.rglob("*") if path.suffix.lower() in extensions and path.is_file()
    }
    return sorted(files, key=str)


def _run_subprocess(cmd: Sequence[str], timeout: int) -> Tuple[int, str, str, bool, Optional[str]]:
    """Run subprocess with timeout handling.

    Args:
        cmd: Command sequence to execute.
        timeout: Timeout in seconds for the subprocess.

    Returns:
        Tuple containing exit code, stdout, stderr, timeout flag, and an
        optional error message when execution failed before running the
        command.
    """

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr, False, None
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - exercised via unit test
        stdout = str(exc.stdout or "")
        stderr = str(exc.stderr or "")
        return 1, stdout, stderr, True, None
    except FileNotFoundError:
        return 1, "", "", False, f"Command not found: {cmd[0]}"
    except Exception as exc:  # pragma: no cover - defensive
        return 1, "", str(exc), False, str(exc)


def run_clang_format(files: List[Path], auto_fix: bool, timeout: int) -> LinterResult:
    """Run clang-format over the provided file list.

    Args:
        files: Precomputed list of files to format.
        auto_fix: Whether to apply in-place fixes (`-i`).
        timeout: Timeout in seconds for the command.

    Returns:
        Populated :class:`LinterResult` for clang-format execution.
    """

    result = LinterResult("clang-format")

    if not check_linter_available("clang-format"):
        result.skipped = True
        result.error_message = "clang-format not found - skipping"
        return result

    if not files:
        result.success = False
        result.error_message = "No C++ files found to format"
        return result

    result.files_checked = len(files)

    cmd = ["clang-format"]
    if auto_fix:
        cmd.append("-i")
    else:
        cmd.extend(["--dry-run", "--Werror"])
    cmd.extend(str(path) for path in files)

    exit_code, stdout, stderr, timed_out, error_message = _run_subprocess(cmd, timeout)
    result.exit_code = exit_code
    result.timeout = timed_out

    combined_output = (stdout + "\n" + stderr).strip()
    truncated_output, was_truncated, notice = _truncate_output(combined_output)
    result.stdout = truncated_output
    result.truncated = was_truncated
    if notice and not combined_output:
        result.stderr = notice

    if error_message:
        result.success = False
        result.error_message = error_message
        return result

    if timed_out:
        result.success = False
        result.error_message = f"clang-format timed out after {timeout} seconds"
        return result

    if exit_code != 0:
        result.success = False
        issue_files = set()
        for line in combined_output.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            _bounded_append(result.issues, stripped)
            for path in files:
                if str(path) in stripped:
                    issue_files.add(path)
        result.files_with_issues = len(issue_files) or len(files)
        if not result.issues:
            _bounded_append(result.issues, "clang-format reported formatting differences")
    else:
        result.success = True

    return result


def run_clang_tidy(
    files: List[Path], build_dir: Optional[str], auto_fix: bool, timeout: int
) -> LinterResult:
    """Run clang-tidy with compile_commands.json enforcement and batching.

    Args:
        files: Precomputed list of files to analyze.
        build_dir: Directory containing compile_commands.json (required).
        auto_fix: Whether to apply fixes using ``--fix``.
        timeout: Timeout in seconds for each clang-tidy batch.

    Returns:
        Populated :class:`LinterResult` detailing clang-tidy execution.
    """

    result = LinterResult("clang-tidy")

    if not check_linter_available("clang-tidy"):
        result.skipped = True
        result.error_message = "clang-tidy not found - skipping"
        return result

    if not build_dir:
        result.success = False
        result.error_message = (
            "--build-dir with compile_commands.json is required for clang-tidy"
        )
        return result

    compile_commands = Path(build_dir) / "compile_commands.json"
    if not compile_commands.exists():
        result.success = False
        result.error_message = (
            f"compile_commands.json not found in {build_dir}. "
            "Run cmake with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
        )
        return result

    if not files:
        result.success = False
        result.error_message = "No C++ files found to analyze"
        return result

    result.files_checked = len(files)

    for start in range(0, len(files), CLANG_TIDY_BATCH_SIZE):
        chunk = files[start : start + CLANG_TIDY_BATCH_SIZE]
        cmd = ["clang-tidy", f"-p={build_dir}"]
        if auto_fix:
            cmd.append("--fix")
        cmd.extend(str(path) for path in chunk)

        exit_code, stdout, stderr, timed_out, error_message = _run_subprocess(cmd, timeout)
        combined_output = (stdout + "\n" + stderr).strip()
        truncated_output, was_truncated, _ = _truncate_output(combined_output)
        result.stdout = "\n".join(filter(None, [result.stdout, truncated_output])).strip()
        result.truncated = result.truncated or was_truncated

        if error_message:
            result.success = False
            result.error_message = error_message
            return result

        if timed_out:
            result.success = False
            result.timeout = True
            result.error_message = f"clang-tidy timed out after {timeout} seconds"
            return result

        result.exit_code = max(result.exit_code, exit_code)

        errors = len(re.findall(r":\s*error:\s", combined_output))
        warnings = len(re.findall(r":\s*warning:\s", combined_output))
        result.errors += errors
        result.warnings += warnings

        if errors > 0 or warnings > 0 or exit_code != 0:
            result.files_with_issues += len(chunk)
            for line in combined_output.splitlines():
                if ": warning:" in line or ": error:" in line:
                    _bounded_append(result.issues, line.strip())

    result.success = result.errors == 0 and result.exit_code == 0
    return result


def run_cppcheck(files: List[Path], timeout: int) -> LinterResult:
    """Run cppcheck static analysis over provided files.

    Args:
        files: Precomputed list of files to analyze.
        timeout: Timeout in seconds for cppcheck.

    Returns:
        Populated :class:`LinterResult` for cppcheck execution.
    """

    result = LinterResult("cppcheck")

    if not check_linter_available("cppcheck"):
        result.skipped = True
        result.error_message = "cppcheck not found - skipping"
        return result

    if not files:
        result.success = False
        result.error_message = "No C++ files found to analyze"
        return result

    result.files_checked = len(files)

    cmd = [
        "cppcheck",
        "--enable=all",
        "--error-exitcode=1",
        "--inline-suppr",
        "--quiet",
    ]
    cmd.extend(str(path) for path in files)

    exit_code, stdout, stderr, timed_out, error_message = _run_subprocess(cmd, timeout)
    result.exit_code = exit_code
    result.timeout = timed_out

    combined_output = (stdout + "\n" + stderr).strip()
    truncated_output, was_truncated, notice = _truncate_output(combined_output)
    result.stdout = truncated_output
    result.truncated = was_truncated
    if notice and not combined_output:
        result.stderr = notice

    if error_message:
        result.success = False
        result.error_message = error_message
        return result

    if timed_out:
        result.success = False
        result.error_message = f"cppcheck timed out after {timeout} seconds"
        return result

    result.errors = len(re.findall(r"\(error\)", combined_output))
    result.warnings = len(re.findall(r"\(warning\)", combined_output))

    issue_files = set()
    for line in combined_output.splitlines():
        if "(error)" in line or "(warning)" in line:
            _bounded_append(result.issues, line.strip())
            for path in files:
                if str(path) in line:
                    issue_files.add(path)

    result.files_with_issues = len(issue_files)
    result.success = exit_code == 0 and result.errors == 0 and not timed_out
    return result


def format_summary(results: List[LinterResult], duration: float, all_skipped: bool) -> str:
    """Format human-readable summary for linter results.

    Args:
        results: Linter results to summarize.
        duration: Total runtime across linters.
        all_skipped: Whether every linter was skipped.

    Returns:
        Multi-line summary string with validation banner.
    """

    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("C++ LINTERS SUMMARY")
    lines.append("=" * 60)

    warnings_present = False

    for result in results:
        if result.skipped:
            status = "SKIPPED"
        elif not result.success and result.errors > 0:
            status = "FAILED"
        elif result.warnings > 0:
            status = "WARNINGS"
            warnings_present = True
        elif result.success:
            status = "PASSED"
        else:
            status = "FAILED"

        lines.append(f"\n{result.name}: {status}")
        lines.append(f"  Files checked: {result.files_checked}")
        if not result.skipped:
            lines.append(f"  Files with issues: {result.files_with_issues}")
            lines.append(f"  Errors: {result.errors}")
            lines.append(f"  Warnings: {result.warnings}")
        if result.error_message:
            lines.append(f"  Message: {result.error_message}")
        if result.timeout:
            lines.append("  Message: timed out")
        if result.issues:
            lines.append("  Issues:")
            for issue in result.issues[:5]:
                lines.append(f"    - {issue}")
            if len(result.issues) > 5:
                lines.append(f"    ... and {len(result.issues) - 5} more")
        if result.truncated:
            lines.append("  Output truncated")

    lines.append(f"\nDuration: {duration:.2f}s")
    lines.append("\n" + "=" * 60)

    if all_skipped:
        validation = "FAILED (all linters skipped)"
    elif any(not r.success and not r.skipped for r in results):
        validation = "FAILED"
    elif warnings_present:
        validation = "PASSED (warnings only)"
    else:
        validation = "PASSED"

    lines.append(f"VALIDATION: {validation}")
    lines.append("=" * 60)

    return "\n".join(lines)


def format_full_output(results: List[LinterResult], summary: str) -> str:
    """Format full output with per-linter stdout/stderr and summary."""

    lines: List[str] = []
    for result in results:
        lines.append("=" * 60)
        lines.append(f"{result.name} Output")
        lines.append("=" * 60)
        if result.stdout:
            lines.append(result.stdout)
        if result.stderr:
            lines.append("\nStderr:")
            lines.append(result.stderr)
        if result.truncated:
            lines.append("\n(Note: output truncated)")
        lines.append("")

    lines.append(summary)
    return "\n".join(lines)


def format_json_output(results: List[LinterResult], duration: float, all_skipped: bool) -> str:
    """Format JSON output with bounded content."""

    payload = {
        "duration": duration,
        "all_skipped": all_skipped,
        "results": [
            {
                "name": r.name,
                "success": r.success,
                "skipped": r.skipped,
                "exit_code": r.exit_code,
                "errors": r.errors,
                "warnings": r.warnings,
                "files_checked": r.files_checked,
                "files_with_issues": r.files_with_issues,
                "issues": r.issues,
                "error_message": r.error_message,
                "stdout": r.stdout,
                "stderr": r.stderr,
                "truncated": r.truncated,
                "timeout": r.timeout,
            }
            for r in results
        ],
    }
    payload["success"] = (
        not any(not r.success and not r.skipped for r in results) and not all_skipped
    )
    return json.dumps(payload, indent=2)


def run_cpp_linters(
    source_dir: str,
    build_dir: Optional[str],
    linters: List[str],
    auto_fix: bool = False,
    output_mode: str = "summary",
    timeout: Optional[int] = None,
) -> Tuple[int, str]:
    """Run configured C++ linters and return exit code + formatted output.

    Args:
        source_dir: Directory containing C/C++ sources.
        build_dir: Directory containing compile_commands.json for clang-tidy.
        linters: List of linters to run (clang-format, clang-tidy, cppcheck).
        auto_fix: Whether to enable auto-fix for applicable linters.
        output_mode: One of ``summary``, ``full``, or ``json``.
        timeout: Optional global timeout applied per linter; falls back to
            tool-specific defaults when omitted.

    Returns:
        Tuple of exit code (0 on success) and formatted output string.
    """

    normalized_linters = [l.strip() for l in linters if l.strip()]
    if not normalized_linters:
        normalized_linters = ["clang-format", "clang-tidy", "cppcheck"]

    file_list = get_cpp_files(source_dir)
    timeouts = {
        "clang-format": timeout or DEFAULT_TIMEOUTS["clang-format"],
        "clang-tidy": timeout or DEFAULT_TIMEOUTS["clang-tidy"],
        "cppcheck": timeout or DEFAULT_TIMEOUTS["cppcheck"],
    }

    results: List[LinterResult] = []
    start_time = time.perf_counter()

    if "clang-format" in normalized_linters:
        results.append(
            run_clang_format(file_list, auto_fix=auto_fix, timeout=timeouts["clang-format"])
        )

    if "clang-tidy" in normalized_linters:
        results.append(
            run_clang_tidy(
                file_list, build_dir=build_dir, auto_fix=auto_fix, timeout=timeouts["clang-tidy"]
            )
        )

    if "cppcheck" in normalized_linters:
        results.append(run_cppcheck(file_list, timeout=timeouts["cppcheck"]))

    duration = time.perf_counter() - start_time

    all_skipped = all(result.skipped for result in results) if results else False
    any_failures = any((not r.success) and not r.skipped for r in results) or all_skipped

    summary = format_summary(results, duration, all_skipped)

    if output_mode == "summary":
        output = summary
    elif output_mode == "full":
        output = format_full_output(results, summary)
    elif output_mode == "json":
        output = format_json_output(results, duration, all_skipped)
    else:
        raise ValueError(f"Unsupported output mode: {output_mode}")

    exit_code = 0 if not any_failures else 1
    return exit_code, output


def parse_linters_arg(raw: str) -> List[str]:
    """Parse comma-separated linters argument with defaults.

    Args:
        raw: Raw comma-separated linter string from CLI.

    Returns:
        Normalized list of linter names, or defaults when empty.
    """

    entries = [item.strip() for item in raw.split(",") if item.strip()]
    return entries or ["clang-format", "clang-tidy", "cppcheck"]


def main() -> int:
    """CLI entrypoint for running C++ linters."""

    parser = argparse.ArgumentParser(
        description="Run C++ linters with optional auto-fix and bounded output",
        epilog="""
Examples:
  %(prog)s --source-dir example_cpp_dev
  %(prog)s --source-dir src --auto-fix
  %(prog)s --source-dir src --linters clang-format
  %(prog)s --source-dir src --build-dir build --linters clang-tidy
  %(prog)s --source-dir src --output json
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="example_cpp_dev",
        help="Directory containing C/C++ sources (default: example_cpp_dev)",
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default=None,
        help="Build directory containing compile_commands.json (required for clang-tidy)",
    )
    parser.add_argument(
        "--linters",
        type=str,
        default="clang-format,clang-tidy,cppcheck",
        help="Comma-separated linters to run",
    )
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        default=False,
        help="Apply auto-fixes when supported (clang-format -i, clang-tidy --fix)",
    )
    parser.add_argument(
        "--no-auto-fix",
        action="store_false",
        dest="auto_fix",
        help="Disable auto-fixing (default)",
    )
    parser.add_argument(
        "--output",
        choices=["summary", "full", "json"],
        default="summary",
        help="Output mode: summary (default), full, or json",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds per linter (defaults vary by linter)",
    )

    args = parser.parse_args()

    linters = parse_linters_arg(args.linters)

    exit_code, output = run_cpp_linters(
        source_dir=args.source_dir,
        build_dir=args.build_dir,
        linters=linters,
        auto_fix=args.auto_fix,
        output_mode=args.output,
        timeout=args.timeout,
    )

    print(output)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
