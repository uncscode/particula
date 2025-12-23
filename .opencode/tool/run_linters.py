#!/usr/bin/env python3
"""Linter Runner Tool

Runs configured linters (ruff, mypy) for the Agent repository.
Automatically fixes issues where possible and reports remaining problems.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


class LinterResult:
    """Store results from a single linter run."""

    def __init__(self, name: str):
        self.name = name
        self.exit_code = 0
        self.stdout = ""
        self.stderr = ""
        self.issues_found = 0
        self.issues_fixed = 0
        self.success = True
        self.error_message: Optional[str] = None


def run_ruff_check(
    target_dir: Optional[str] = None, auto_fix: bool = True, timeout: int = 120
) -> LinterResult:
    """Run ruff check with optional auto-fixing.

    Follows .github/workflows/lint.yml workflow:
    1. ruff check --fix (apply fixes, don't fail)
    2. ruff format (format code)
    3. ruff check (final check, fail if issues remain)

    Args:
        target_dir: Directory to lint. If None, uses pyproject.toml config from project root.
        auto_fix: Whether to automatically fix issues
        timeout: Timeout in seconds for each ruff command (default: 120)

    Returns:
        LinterResult with check results
    """
    result = LinterResult("ruff_check")

    # Build target argument - use "." if no specific target
    target_arg = f"{target_dir}/" if target_dir else "."

    try:
        if auto_fix:
            # Step 1: Apply fixes (don't fail on errors)
            fix_cmd = ["ruff", "check", "--fix", target_arg]
            subprocess.run(
                fix_cmd, capture_output=True, text=True, timeout=timeout
            )

            # Step 2: Format code
            format_cmd = ["ruff", "format", target_arg]
            subprocess.run(
                format_cmd, capture_output=True, text=True, timeout=timeout
            )

        # Step 3: Final check (this determines success/failure)
        check_cmd = ["ruff", "check", target_arg]
        proc = subprocess.run(
            check_cmd, capture_output=True, text=True, timeout=timeout
        )

        result.exit_code = proc.returncode
        result.stdout = proc.stdout
        result.stderr = proc.stderr
        result.success = proc.returncode == 0

        # Parse output to count issues
        # Ruff outputs: "Found X errors" or "All checks passed"
        if "Found" in result.stdout:
            # Extract numbers from output
            found_match = re.search(r"Found\s+(\d+)\s+error", result.stdout)
            if found_match:
                result.issues_found = int(found_match.group(1))

        # If we auto-fixed, assume some issues were fixed
        # (we don't get exact count from ruff check --fix output)
        if auto_fix and result.success:
            result.issues_fixed = result.issues_found  # Rough estimate

    except subprocess.TimeoutExpired:
        result.success = False
        result.error_message = f"Timeout after {timeout} seconds"
    except FileNotFoundError:
        result.success = False
        result.error_message = "ruff not found - is it installed?"
    except Exception as e:
        result.success = False
        result.error_message = str(e)

    return result


def run_ruff_format(
    target_dir: Optional[str] = None, timeout: int = 120
) -> LinterResult:
    """Run ruff format to auto-format code.

    Note: This is now called as part of run_ruff_check() workflow,
    but kept separate for individual linter testing.

    Args:
        target_dir: Directory to format. If None, uses pyproject.toml config from project root.
        timeout: Timeout in seconds (default: 120)

    Returns:
        LinterResult with format results
    """
    result = LinterResult("ruff_format")

    # Build target argument - use "." if no specific target
    target_arg = f"{target_dir}/" if target_dir else "."
    cmd = ["ruff", "format", target_arg]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
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
    except FileNotFoundError:
        result.success = False
        result.error_message = "ruff not found - is it installed?"
    except Exception as e:
        result.success = False
        result.error_message = str(e)

    return result


def run_mypy(
    target_dir: Optional[str] = None, timeout: int = 180
) -> LinterResult:
    """Run mypy for type checking.

    Args:
        target_dir: Directory to type check. If None, uses pyproject.toml config from project root.
        timeout: Timeout in seconds (default: 180 = 3 minutes)

    Returns:
        LinterResult with type check results
    """
    result = LinterResult("mypy")

    # Build target argument - use "." if no specific target
    target_arg = f"{target_dir}/" if target_dir else "."
    cmd = ["mypy", target_arg, "--ignore-missing-imports"]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )

        result.exit_code = proc.returncode
        result.stdout = proc.stdout
        result.stderr = proc.stderr
        result.success = proc.returncode == 0

        # Count errors in output
        # Mypy outputs errors on individual lines
        error_lines = [
            line for line in result.stdout.split("\n") if ": error:" in line
        ]
        result.issues_found = len(error_lines)

    except subprocess.TimeoutExpired:
        result.success = False
        result.error_message = f"Timeout after {timeout} seconds"
    except FileNotFoundError:
        result.success = False
        result.error_message = "mypy not found - is it installed?"
    except Exception as e:
        result.success = False
        result.error_message = str(e)

    return result


def format_summary(results: List[LinterResult], all_passed: bool) -> str:
    """Format a human-readable summary of linting results.

    Args:
        results: List of LinterResult objects
        all_passed: Whether all linters passed

    Returns:
        Formatted summary string
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
    """Format full linter output with all details.

    Args:
        results: List of LinterResult objects
        all_passed: Whether all linters passed

    Returns:
        Formatted full output string
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
    """Run configured linters.

    Args:
        target_dir: Directory to lint. If None, uses pyproject.toml config from project root.
        auto_fix: Whether to auto-fix issues
        linters: List of linters to run
        output_mode: Output format (summary, full, json)
        cwd: Working directory (defaults to project root)
        ruff_timeout: Timeout for ruff commands in seconds (default: 120)
        mypy_timeout: Timeout for mypy command in seconds (default: 180)

    Returns:
        Tuple of (exit_code, output_string)
    """
    # Determine working directory
    if cwd is None:
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

    # Run linters following .github/workflows/lint.yml
    results = []

    # Ruff check includes format when auto_fix=True (matching CI workflow)
    if "ruff" in linters or "ruff_check" in linters:
        results.append(run_ruff_check(target_dir, auto_fix, ruff_timeout))

    # Allow running format separately for testing
    elif "ruff_format" in linters:
        results.append(run_ruff_format(target_dir, ruff_timeout))

    if "mypy" in linters:
        results.append(run_mypy(target_dir, mypy_timeout))

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


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(description="Run linters with auto-fixing")
    parser.add_argument(
        "--output",
        choices=["summary", "full", "json"],
        default="summary",
        help="Output mode: summary (default), full output, or JSON",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        help="Target directory to lint. If omitted, uses pyproject.toml config from project root.",
    )
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        default=True,
        help="Automatically fix issues where possible (default: True)",
    )
    parser.add_argument(
        "--no-auto-fix",
        action="store_false",
        dest="auto_fix",
        help="Disable auto-fixing",
    )
    parser.add_argument(
        "--linters",
        type=str,
        default="ruff,mypy",
        help="Comma-separated list of linters to run (default: ruff,mypy matching CI)",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        help="Working directory (defaults to project root)",
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

    linters = [l.strip() for l in args.linters.split(",")]

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
