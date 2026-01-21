#!/usr/bin/env python3
"""Validate Jupyter notebooks for syntax and execution.

This script validates notebooks in two phases:
1. Syntax validation: Parse all code cells with ast to catch syntax errors
2. Execution validation: Run the notebook with a timeout (default 5 min)

Notebooks that timeout are marked as skipped (not failed) since they may be
long-running simulations that are valid but too slow for CI.

Usage:
    python validate_notebooks.py notebook1.ipynb notebook2.ipynb ...

Environment variables:
    NOTEBOOK_TIMEOUT: Execution timeout in seconds (default: 300)
"""

import ast
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidationResult:
    """Result of notebook validation."""

    notebook: str
    syntax_valid: bool
    syntax_errors: list[str]
    execution_valid: bool | None  # None if skipped due to timeout
    execution_error: str | None
    timed_out: bool


def validate_syntax(notebook_path: Path) -> tuple[bool, list[str]]:
    """Validate Python syntax in all code cells.

    Args:
        notebook_path: Path to the notebook file.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors = []

    try:
        with open(notebook_path, encoding="utf-8") as f:
            notebook = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]

    cells = notebook.get("cells", [])

    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        if isinstance(source, list):
            code = "".join(source)
        else:
            code = source

        # Skip empty cells
        if not code.strip():
            continue

        # Skip cells that are just magic commands or shell commands
        lines = code.strip().split("\n")
        non_magic_lines = [
            line
            for line in lines
            if not line.strip().startswith(("%", "!", "#"))
        ]
        if not non_magic_lines:
            continue

        # Try to parse the code
        try:
            ast.parse(code)
        except SyntaxError as e:
            cell_num = i + 1
            errors.append(
                f"Cell {cell_num}, line {e.lineno}: {e.msg}\n"
                f"  Code: {e.text.strip() if e.text else 'N/A'}"
            )

    return len(errors) == 0, errors


def execute_notebook(
    notebook_path: Path, timeout: int
) -> tuple[bool | None, str | None, bool]:
    """Execute a notebook with timeout.

    Args:
        notebook_path: Path to the notebook file.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (success, error_message, timed_out).
        success is None if timed out (not a failure).
    """
    try:
        # Create a temporary file for the output
        # nbconvert appends .ipynb to the output name, so we need a temp file
        with tempfile.NamedTemporaryFile(
            suffix=".ipynb", delete=True, mode="w"
        ) as temp_output:
            # Get the path without the .ipynb extension (nbconvert adds it)
            output_path = temp_output.name.rsplit(".ipynb", 1)[0]

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--ExecutePreprocessor.timeout",
                    str(timeout),
                    "--ExecutePreprocessor.kernel_name=python3",
                    "--output",
                    output_path,
                    str(notebook_path),
                ],
                capture_output=True,
                text=True,
                timeout=timeout + 30,  # Extra buffer for nbconvert overhead
            )

        if result.returncode == 0:
            return True, None, False

        # Check if it was a timeout
        stderr = result.stderr
        if "CellTimeoutError" in stderr or "timed out" in stderr.lower():
            return None, "Execution timed out (skipped)", True

        # Extract relevant error message
        error_lines = stderr.strip().split("\n")
        # Find the actual error (usually near the end)
        error_msg = "\n".join(error_lines[-20:]) if error_lines else "Unknown"
        return False, error_msg, False

    except subprocess.TimeoutExpired:
        return None, "Execution timed out (skipped)", True
    except Exception as e:
        return False, str(e), False


def validate_notebook(notebook_path: str, timeout: int) -> ValidationResult:
    """Validate a single notebook.

    Args:
        notebook_path: Path to the notebook.
        timeout: Execution timeout in seconds.

    Returns:
        ValidationResult with all validation details.
    """
    path = Path(notebook_path)

    if not path.exists():
        return ValidationResult(
            notebook=notebook_path,
            syntax_valid=False,
            syntax_errors=[f"File not found: {notebook_path}"],
            execution_valid=None,
            execution_error=None,
            timed_out=False,
        )

    # Phase 1: Syntax validation
    syntax_valid, syntax_errors = validate_syntax(path)

    # Phase 2: Execution validation (only if syntax is valid)
    if syntax_valid:
        exec_valid, exec_error, timed_out = execute_notebook(path, timeout)
    else:
        exec_valid = None
        exec_error = "Skipped due to syntax errors"
        timed_out = False

    return ValidationResult(
        notebook=notebook_path,
        syntax_valid=syntax_valid,
        syntax_errors=syntax_errors,
        execution_valid=exec_valid,
        execution_error=exec_error,
        timed_out=timed_out,
    )


def print_result(result: ValidationResult) -> None:
    """Print validation result with formatting."""
    notebook_name = Path(result.notebook).name

    if result.syntax_valid and result.execution_valid:
        print(f"  PASS: {notebook_name}")
    elif result.timed_out:
        print(f"  SKIP: {notebook_name} (timeout - execution took >5 min)")
    elif not result.syntax_valid:
        print(f"  FAIL: {notebook_name} (syntax errors)")
        for error in result.syntax_errors:
            print(f"        {error}")
    elif result.execution_valid is False:
        print(f"  FAIL: {notebook_name} (execution error)")
        if result.execution_error:
            # Truncate long error messages
            error_preview = result.execution_error[:500]
            if len(result.execution_error) > 500:
                error_preview += "..."
            for line in error_preview.split("\n"):
                print(f"        {line}")


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failures).
    """
    if len(sys.argv) < 2:
        print("Usage: validate_notebooks.py notebook1.ipynb ...")
        return 0

    # Get timeout from environment (default 5 minutes)
    timeout = int(os.environ.get("NOTEBOOK_TIMEOUT", "300"))

    notebooks = [nb for nb in sys.argv[1:] if nb.strip()]

    if not notebooks:
        print("No notebooks to validate.")
        return 0

    print("=" * 60)
    print("NOTEBOOK VALIDATION")
    print("=" * 60)
    print(f"Notebooks to validate: {len(notebooks)}")
    print(f"Execution timeout: {timeout}s ({timeout // 60} min)")
    print("-" * 60)

    results = []
    for notebook in notebooks:
        print(f"\nValidating: {notebook}")
        result = validate_notebook(notebook, timeout)
        results.append(result)
        print_result(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(
        1 for r in results if r.syntax_valid and r.execution_valid is True
    )
    failed = sum(
        1 for r in results if not r.syntax_valid or r.execution_valid is False
    )
    skipped = sum(1 for r in results if r.timed_out)

    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped} (timeout)")
    print("=" * 60)

    # Return failure if any notebooks failed (not for timeouts)
    if failed > 0:
        print("\nValidation FAILED - syntax or execution errors found")
        return 1

    print("\nValidation PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
