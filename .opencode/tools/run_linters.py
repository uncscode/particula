#!/usr/bin/env python3
"""Run repository linters in mutating or validation-only modes.

This tool runs the configured Ruff and mypy commands for the repository while
mirroring the lint workflow in `.github/workflows/lint.yml`.

When auto-fix is enabled, the Ruff path is mutating:
    1. ruff check --fix
    2. ruff format
    3. ruff check

When auto-fix is disabled, the Ruff path is validation-only and non-mutating:
    1. ruff check

Mypy remains validation-only in both modes. Ruff and mypy subprocesses can also
run from an explicit working directory via `cwd`.

Usage:
    python3 run_linters.py
    python3 run_linters.py --output json
    python3 run_linters.py --target-dir adw/core --linters ruff

Examples:
    # Run all linters with auto-fix (default)
    python3 .opencode/tools/run_linters.py

    # Get JSON output for programmatic use
    python3 .opencode/tools/run_linters.py --output json

    # Lint specific directory
    python3 .opencode/tools/run_linters.py --target-dir adw/workflows

    # Run only ruff without mypy
    python3 .opencode/tools/run_linters.py --linters ruff

    # Disable auto-fix for validation-only, non-mutating checks
    python3 .opencode/tools/run_linters.py --no-auto-fix
"""

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


class LinterResult:
    """Store results from a single linter run.

    Tracks execution results including output, issue counts, and error details
    for a specific linter tool (ruff, mypy, etc.).

    Attributes:
        name: Linter identifier (e.g., "ruff_check", "mypy").
        exit_code: Process exit code from linter execution.
        stdout: Standard output captured from linter.
        stderr: Standard error captured from linter.
        issues_found: Count of issues detected by linter.
        issues_fixed: Count of issues automatically fixed.
        success: Whether linter passed without remaining issues.
        error_message: Error details if linter failed to run.
    """

    def __init__(self, name: str) -> None:
        """Initialize linter result with defaults.

        Args:
            name: Identifier for the linter tool.
        """
        self.name = name
        self.exit_code = 0
        self.stdout = ""
        self.stderr = ""
        self.issues_found = 0
        self.issues_fixed = 0
        self.success = True
        self.error_message: Optional[str] = None


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


def _count_ruff_issues(output: str) -> int:
    """Extract the Ruff issue count from command output when available.

    Args:
        output: Ruff stdout to inspect.

    Returns:
        Parsed issue count, or ``0`` when Ruff did not emit a matching summary.
    """
    if "Found" not in output:
        return 0

    found_match = re.search(r"Found\s+(\d+)\s+error", output)
    if found_match:
        return int(found_match.group(1))
    return 0


def _apply_process_result(result: LinterResult, proc: subprocess.CompletedProcess[str]) -> None:
    """Populate a linter result from a completed subprocess run.

    Args:
        result: Mutable result object to update.
        proc: Completed subprocess result from a linter invocation.

    Returns:
        None.
    """
    result.exit_code = proc.returncode
    result.stdout = proc.stdout
    result.stderr = proc.stderr
    result.success = proc.returncode == 0


def _resolve_target_arg(target_dir: Optional[str], cwd: str) -> str:
    """Validate and resolve a linter target relative to ``cwd``.

    Args:
        target_dir: Optional directory argument supplied by the caller.
        cwd: Resolved subprocess working directory.

    Returns:
        Target argument to forward to linters.

    Raises:
        ValueError: If the target begins with ``-`` and could be interpreted as
            a subprocess option, or if it resolves outside ``cwd``.
    """
    if not target_dir:
        return "."
    if target_dir.startswith("-"):
        raise ValueError(f"target_dir must not start with '-': {target_dir}")

    cwd_path = Path(cwd).resolve()
    target_path = Path(target_dir)
    resolved_target = (
        target_path if target_path.is_absolute() else cwd_path / target_path
    ).resolve()
    if resolved_target != cwd_path and cwd_path not in resolved_target.parents:
        raise ValueError(f"target_dir resolves outside cwd: {target_dir}")
    return f"{target_dir}/"


def _resolve_cwd(cwd: Optional[str]) -> str:
    """Resolve and validate the working directory for linter subprocesses.

    Args:
        cwd: Optional working directory override.

    Returns:
        Existing directory path to use as subprocess ``cwd``. When ``cwd`` is
        omitted, the nearest ancestor containing ``pyproject.toml`` or ``.git``
        is used, with the current directory as the fallback.

    Raises:
        ValueError: If an explicit ``cwd`` does not exist or is not a directory.
    """
    if cwd is None:
        current = Path.cwd()
        while current != current.parent:
            if (current / "pyproject.toml").exists() or (current / ".git").exists():
                return str(current)
            current = current.parent
        return str(Path.cwd())

    cwd_path = Path(cwd)
    if not cwd_path.exists():
        raise ValueError(f"cwd does not exist: {cwd}")
    if not cwd_path.is_dir():
        raise ValueError(f"cwd is not a directory: {cwd}")
    return str(cwd_path)


def run_ruff_check(
    target_dir: Optional[str] = None,
    auto_fix: bool = True,
    timeout: int = 120,
    cwd: Optional[str] = None,
) -> LinterResult:
    """Run Ruff checks in mutating or validation-only mode.

    When ``auto_fix`` is true, this follows the mutating Ruff workflow used by
    the lint pipeline: ``ruff check --fix``, then ``ruff format``, then a final
    ``ruff check``. Each step is stop-on-failure. When ``auto_fix`` is false,
    it performs only the final ``ruff check`` so the run stays validation-only
    and does not edit files.

    Args:
        target_dir: Directory to lint. If None, uses pyproject.toml config from
            the project root.
        auto_fix: Whether to automatically fix issues. When False, this path is
            validation-only and must not modify files.
        timeout: Timeout in seconds for each Ruff command.
        cwd: Working directory for subprocess execution.

    Returns:
        LinterResult describing the Ruff run outcome.
    """
    result = LinterResult("ruff_check")

    try:
        resolved_cwd = _resolve_cwd(cwd)
        target_arg = _resolve_target_arg(target_dir, resolved_cwd)

        if auto_fix:
            # Step 1: Apply fixes. Match CI's `ruff check --fix || true` behavior:
            # keep diagnostics from a non-zero fix pass, but continue to format
            # and let the final validation check determine success/failure.
            ruff_cmd = _resolve_python_tool_command("ruff", "ruff", resolved_cwd)
            fix_cmd = [*ruff_cmd, "check", "--fix", target_arg]
            fix_proc = subprocess.run(
                fix_cmd, capture_output=True, text=True, timeout=timeout, cwd=resolved_cwd
            )
            if fix_proc.returncode != 0:
                result.stderr = fix_proc.stderr
                result.stdout = fix_proc.stdout
                result.issues_found = _count_ruff_issues(fix_proc.stdout)

            # Step 2: Format code
            format_cmd = [*ruff_cmd, "format", target_arg]
            format_proc = subprocess.run(
                format_cmd, capture_output=True, text=True, timeout=timeout, cwd=resolved_cwd
            )
            if format_proc.returncode != 0:
                _apply_process_result(result, format_proc)
                return result

        # Step 3: Final check (this determines success/failure)
        ruff_cmd = _resolve_python_tool_command("ruff", "ruff", resolved_cwd)
        check_cmd = [*ruff_cmd, "check", target_arg]
        proc = subprocess.run(
            check_cmd, capture_output=True, text=True, timeout=timeout, cwd=resolved_cwd
        )

        _apply_process_result(result, proc)

        # Parse output to count issues
        # Ruff outputs: "Found X errors" or "All checks passed"
        result.issues_found = _count_ruff_issues(result.stdout)

        # If we auto-fixed, assume some issues were fixed
        # (we don't get exact count from ruff check --fix output)
        if auto_fix and result.success:
            result.issues_fixed = result.issues_found  # Rough estimate

    except subprocess.TimeoutExpired:
        result.success = False
        result.error_message = f"Timeout after {timeout} seconds"
    except ValueError as exc:
        result.success = False
        result.error_message = str(exc)
    except FileNotFoundError:
        result.success = False
        result.error_message = "ruff not found - is it installed?"
    except Exception as e:
        result.success = False
        result.error_message = str(e)

    return result


def run_ruff_format(
    target_dir: Optional[str] = None, timeout: int = 120, cwd: Optional[str] = None
) -> LinterResult:
    """Run ``ruff format`` for explicit formatting-only execution.

    This helper is primarily used by the mutating ``run_ruff_check()`` workflow
    when ``auto_fix`` is enabled, but it remains available for isolated testing
    of formatting behavior.

    Args:
        target_dir: Directory to format. If None, uses pyproject.toml config
            from the project root.
        timeout: Timeout in seconds.
        cwd: Working directory for subprocess execution.

    Returns:
        LinterResult describing the formatting run outcome.
    """
    result = LinterResult("ruff_format")

    try:
        resolved_cwd = _resolve_cwd(cwd)
        target_arg = _resolve_target_arg(target_dir, resolved_cwd)
        cmd = [*_resolve_python_tool_command("ruff", "ruff", resolved_cwd), "format", target_arg]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=resolved_cwd
        )

        result.exit_code = proc.returncode
        result.stdout = proc.stdout
        result.stderr = proc.stderr
        result.success = proc.returncode == 0

        # Count formatted files
        if "file" in result.stdout.lower():
            # Ruff outputs: "X files reformatted"
            match = re.search(r"(\d+)\s+files?\s+reformatted", result.stdout)
            if match:
                result.issues_fixed = int(match.group(1))

    except subprocess.TimeoutExpired:
        result.success = False
        result.error_message = f"Timeout after {timeout} seconds"
    except ValueError as exc:
        result.success = False
        result.error_message = str(exc)
    except FileNotFoundError:
        result.success = False
        result.error_message = "ruff not found - is it installed?"
    except Exception as e:
        result.success = False
        result.error_message = str(e)

    return result


def run_mypy(
    target_dir: Optional[str] = None, timeout: int = 180, cwd: Optional[str] = None
) -> LinterResult:
    """Run mypy in validation-only mode.

    Args:
        target_dir: Directory to type check. If None, uses pyproject.toml
            config from the project root.
        timeout: Timeout in seconds.
        cwd: Working directory for subprocess execution.

    Returns:
        LinterResult describing the mypy run outcome.
    """
    result = LinterResult("mypy")

    try:
        resolved_cwd = _resolve_cwd(cwd)
        target_arg = _resolve_target_arg(target_dir, resolved_cwd)
        cmd = [
            *_resolve_python_tool_command("mypy", "mypy", resolved_cwd),
            target_arg,
            "--ignore-missing-imports",
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=resolved_cwd
        )

        result.exit_code = proc.returncode
        result.stdout = proc.stdout
        result.stderr = proc.stderr
        result.success = proc.returncode == 0

        # Count errors in output
        # Mypy outputs errors on individual lines
        error_lines = [line for line in result.stdout.split("\n") if ": error:" in line]
        result.issues_found = len(error_lines)

    except subprocess.TimeoutExpired:
        result.success = False
        result.error_message = f"Timeout after {timeout} seconds"
    except ValueError as exc:
        result.success = False
        result.error_message = str(exc)
    except FileNotFoundError:
        result.success = False
        result.error_message = "mypy not found - is it installed?"
    except Exception as e:
        result.success = False
        result.error_message = str(e)

    return result


def format_summary(results: List[LinterResult], all_passed: bool) -> str:
    """Format a human-readable summary of linting results.

    Generates a structured summary with status indicators, issue counts,
    and error previews for each linter run.

    Args:
        results: List of LinterResult objects from each linter execution.
        all_passed: Whether all linters completed without remaining issues.

    Returns:
        Multi-line formatted string with visual status indicators (✓/✗),
        issue counts, and overall pass/fail result.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("LINTING SUMMARY")
    lines.append("=" * 60)

    for result in results:
        status = "✓" if result.success else "✗"
        name = result.name.replace("_", " ").title()

        lines.append(f"\n{status} {name}")

        if result.error_message:
            lines.append(f"  Error: {result.error_message}")
        elif result.success:
            if result.issues_fixed > 0:
                lines.append(f"  Fixed: {result.issues_fixed} issues")
            if result.issues_found > 0:
                lines.append(f"  Remaining: {result.issues_found} issues")
            if result.issues_fixed == 0 and result.issues_found == 0:
                lines.append("  No issues found")
        else:
            if result.issues_found > 0:
                lines.append(f"  Found: {result.issues_found} issues")

            # Show first few errors from output
            if result.stdout:
                error_lines = [
                    line
                    for line in result.stdout.split("\n")
                    if line.strip() and not line.startswith("Found")
                ][:5]
                if error_lines:
                    lines.append("  Preview:")
                    for line in error_lines:
                        lines.append(f"    {line[:80]}")

    lines.append("\n" + "=" * 60)
    if all_passed:
        lines.append("RESULT: ALL LINTERS PASSED ✓")
    else:
        lines.append("RESULT: LINTING FAILED ✗")
    lines.append("=" * 60)

    return "\n".join(lines)


def format_full_output(results: List[LinterResult], all_passed: bool) -> str:
    """Format full linter output with complete details.

    Includes the complete stdout/stderr from each linter followed by
    the summary. Useful for debugging and understanding all issues.

    Args:
        results: List of LinterResult objects from each linter execution.
        all_passed: Whether all linters completed without remaining issues.

    Returns:
        Multi-line string with complete linter output followed by summary.
    """
    lines = []

    for result in results:
        name = result.name.replace("_", " ").title()
        lines.append("=" * 60)
        lines.append(f"{name} Output")
        lines.append("=" * 60)

        if result.stdout:
            lines.append(result.stdout)

        if result.stderr:
            lines.append("\nStderr:")
            lines.append(result.stderr)

        lines.append("")

    # Add summary at the end
    summary = format_summary(results, all_passed)
    lines.append(summary)

    return "\n".join(lines)


def run_linters(
    target_dir: Optional[str],
    auto_fix: bool,
    linters: List[str],
    output_mode: str = "summary",
    cwd: Optional[str] = None,
    ruff_timeout: int = 120,
    mypy_timeout: int = 180,
) -> Tuple[int, str]:
    """Run configured linters and format the combined result.

    Executes linters in the repository's lint workflow order. When
    ``auto_fix`` is true, Ruff runs the mutating ``check --fix`` and
    ``format`` steps before a final validation check. When ``auto_fix`` is
    false, Ruff runs only the validation check and must not modify files.
    Mypy remains validation-only in both modes.

    Args:
        target_dir: Directory to lint. If None, uses pyproject.toml config
            which typically lints the entire project from root.
        auto_fix: Whether to automatically fix issues. When True, runs
            Ruff's mutating fix and format steps before the final check. When
            False, runs validation-only checks.
        linters: List of linters to run. Valid values: ["ruff", "mypy"].
            Can also use "ruff_check" or "ruff_format" for granular control.
        output_mode: Output format for results. One of:
            - "summary": Human-readable with status indicators
            - "full": Complete linter output with summary
            - "json": Structured data for programmatic use
        cwd: Working directory for linter execution. Defaults to project root
            (found by traversing up to pyproject.toml or .git).
        ruff_timeout: Timeout in seconds for each ruff command (default: 120).
        mypy_timeout: Timeout in seconds for mypy command (default: 180).

    Returns:
        Tuple of (exit_code, output_string) where exit_code is 0 if all
        linters passed, 1 otherwise.
    """
    resolved_cwd = _resolve_cwd(cwd)
    _resolve_target_arg(target_dir, resolved_cwd)

    # Run linters following .github/workflows/lint.yml
    results = []

    # Ruff check includes format when auto_fix=True (matching CI workflow)
    if "ruff" in linters or "ruff_check" in linters:
        results.append(run_ruff_check(target_dir, auto_fix, ruff_timeout, cwd=resolved_cwd))

    # Allow running format separately for testing
    elif "ruff_format" in linters:
        results.append(run_ruff_format(target_dir, ruff_timeout, cwd=resolved_cwd))

    if "mypy" in linters:
        results.append(run_mypy(target_dir, mypy_timeout, cwd=resolved_cwd))

    # Determine overall success
    all_passed = all(result.success for result in results)
    exit_code = 0 if all_passed else 1

    # Format output
    if output_mode == "summary":
        output = format_summary(results, all_passed)
    elif output_mode == "json":
        output = json.dumps(
            {
                "results": [
                    {
                        "name": r.name,
                        "success": r.success,
                        "exit_code": r.exit_code,
                        "issues_found": r.issues_found,
                        "issues_fixed": r.issues_fixed,
                        "error_message": r.error_message,
                    }
                    for r in results
                ],
                "all_passed": all_passed,
            },
            indent=2,
        )
    else:  # full
        output = format_full_output(results, all_passed)

    return exit_code, output


def main() -> int:
    """Main entry point for CLI usage.

    Parses command-line arguments and executes linter suite.

    Returns:
        Exit code (0 if all linters pass, 1 otherwise).
    """
    parser = argparse.ArgumentParser(
        description="Run linters in mutating auto-fix mode or validation-only no-auto-fix mode.",
        epilog="""
Examples:
  %(prog)s                              Run all linters with auto-fix
  %(prog)s --output json                Get JSON output for scripting
  %(prog)s --target-dir adw/core        Lint specific directory
  %(prog)s --linters ruff               Run only ruff
  %(prog)s --no-auto-fix                Validation-only check without fixes or formatting
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        choices=["summary", "full", "json"],
        default="summary",
        help="Output mode: summary (default), full (complete output), json",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        help="Directory to lint. If omitted, uses pyproject.toml config (lints from project root).",
    )
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        default=True,
        help=(
            "Automatically fix issues (default: True). Runs ruff check --fix, "
            "then ruff format, then a final ruff check, stopping early if a "
            "step fails."
        ),
    )
    parser.add_argument(
        "--no-auto-fix",
        action="store_false",
        dest="auto_fix",
        help="Disable auto-fixing for validation-only mode that must not edit files.",
    )
    parser.add_argument(
        "--linters",
        type=str,
        default="ruff,mypy",
        help="Comma-separated linters: ruff,mypy (default matches CI workflow)",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        help="Working directory (defaults to project root found via pyproject.toml or .git)",
    )
    parser.add_argument(
        "--ruff-timeout",
        type=int,
        default=120,
        help="Timeout for ruff commands in seconds (default: 120 = 2 minutes)",
    )
    parser.add_argument(
        "--mypy-timeout",
        type=int,
        default=180,
        help="Timeout for mypy command in seconds (default: 180 = 3 minutes)",
    )

    args = parser.parse_args()

    linters = [name.strip() for name in args.linters.split(",")]

    exit_code, output = run_linters(
        target_dir=args.target_dir,
        auto_fix=args.auto_fix,
        linters=linters,
        output_mode=args.output,
        cwd=args.cwd,
        ruff_timeout=args.ruff_timeout,
        mypy_timeout=args.mypy_timeout,
    )

    print(output)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
