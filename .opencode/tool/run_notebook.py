#!/usr/bin/env python3
"""Notebook Execution Tool for ADW.

Executes Jupyter notebooks and validates outputs. Supports single notebook or
directories (optional recursion), per-notebook timeouts, output validation, and
multiple output modes (summary/full/json). Mirrors run_pytest/run_ctest patterns.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

DEFAULT_TIMEOUT = 600
TOTAL_GUARD_PADDING = 60
FULL_OUTPUT_CHAR_LIMIT = 4000


class NotebookToolError(Exception):
    """Predictable error for user-facing failures."""


class NotebookDependencyError(Exception):
    """Raised when notebook execution dependencies are missing."""


def _load_executor():
    """Import notebook executor, surfacing dependency errors clearly."""

    try:
        from adw.utils.notebook import NotebookExecutionResult, execute_notebook
    except ImportError as exc:  # pragma: no cover - exercised via CLI path
        raise NotebookDependencyError(
            "Missing dependency for notebook execution (nbconvert/nbclient). "
            "Install with: pip install nbconvert nbclient"
        ) from exc

    return NotebookExecutionResult, execute_notebook


@contextmanager
def pushd(path: Optional[Path]) -> Iterator[None]:
    """Temporarily change cwd and prepend to PYTHONPATH.

    Args:
        path: Target working directory. If None, no-op.
    """

    prev_cwd = Path.cwd()
    prev_pythonpath = os.environ.get("PYTHONPATH", "")

    if path is None:
        yield
        return

    try:
        os.chdir(path)
        if prev_pythonpath:
            os.environ["PYTHONPATH"] = f"{path}{os.pathsep}{prev_pythonpath}"
        else:
            os.environ["PYTHONPATH"] = str(path)
        yield
    finally:
        os.chdir(prev_cwd)
        os.environ["PYTHONPATH"] = prev_pythonpath


@contextmanager
def time_limit(seconds: int) -> Iterator[None]:
    """Enforce a wall-clock guard for the enclosed block."""

    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        if elapsed > seconds:
            raise TimeoutError(f"Total execution exceeded {seconds} seconds")


def _collect_notebooks(target: Path, recursive: bool) -> List[Path]:
    """Collect notebook paths from a file or directory."""

    if target.is_file() and target.suffix == ".ipynb":
        return [target]

    if not target.exists():
        raise NotebookToolError(f"Notebook path not found: {target}")

    if not target.is_dir():
        raise NotebookToolError(f"Notebook path must be a .ipynb file or directory: {target}")

    notebooks: List[Path] = []
    if recursive:
        for root, _, files in os.walk(target):
            for name in files:
                if name.endswith(".ipynb"):
                    notebooks.append(Path(root) / name)
    else:
        notebooks = list(target.glob("*.ipynb"))

    if not notebooks:
        raise NotebookToolError(f"No notebooks found under {target}")

    return sorted(notebooks)


def _validate_output(executed_path: Path, expected_strings: Sequence[str]) -> List[str]:
    """Validate expected substrings are present in executed notebook outputs."""

    import nbformat

    nb_json = nbformat.read(executed_path, as_version=4)
    outputs: List[str] = []

    for cell in nb_json.get("cells", []):
        for output in cell.get("outputs", []):
            if "text" in output:
                value = output["text"]
                outputs.extend(value if isinstance(value, list) else [value])
            if "data" in output:
                data_value = output["data"]
                if isinstance(data_value, dict):
                    outputs.extend(str(v) for v in data_value.values())
    joined = "\n".join(outputs)

    missing: List[str] = []
    for expected in expected_strings:
        if expected not in joined:
            missing.append(expected)
    return missing


def _truncate(text: str, limit: int = FULL_OUTPUT_CHAR_LIMIT) -> Tuple[str, bool]:
    """Truncate text with a notice when exceeding limit."""

    if len(text) <= limit:
        return text, False
    return f"{text[:limit]}\n...\n[output truncated to {limit} chars]", True


def _format_summary(
    results: List[Any],
    validation_errors: Dict[str, List[str]],
    total_time: float,
) -> str:
    """Build human-readable summary output."""

    passed = sum(1 for r in results if r.success)
    failed = len(results) - passed

    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("NOTEBOOK EXECUTION SUMMARY")
    lines.append("=" * 60)
    lines.append(f"\nNotebooks Run: {len(results)}")
    lines.append(f"  Passed:  {passed}")
    if failed:
        lines.append(f"  Failed:  {failed}")
    lines.append(f"\nTotal Time: {total_time:.2f}s")

    failures = [r for r in results if not r.success]
    if failures:
        lines.append(f"\nFailures ({len(failures)}):")
        for res in failures[:10]:
            detail = res.error_message or "Unknown error"
            prefix = f"  - {Path(res.notebook_path).name}"
            if res.failed_cell_index is not None:
                prefix += f" (cell {res.failed_cell_index})"
            lines.append(f"{prefix}: {detail}")
        if len(failures) > 10:
            lines.append(f"  ... and {len(failures) - 10} more")

    if validation_errors:
        lines.append("\nValidation Errors:")
        for nb, missing in validation_errors.items():
            lines.append(f"  - {Path(nb).name}: missing expectations: {', '.join(missing)}")

    lines.append("\n" + "=" * 60)
    lines.append("VALIDATION: FAILED" if (failures or validation_errors) else "VALIDATION: PASSED")
    lines.append("=" * 60)
    return "\n".join(lines)


def _serialize_result(result: Any, include_output: bool) -> Dict[str, object]:
    """Serialize a single notebook result for JSON output."""

    payload: Dict[str, object] = {
        "notebook": result.notebook_path,
        "success": result.success,
        "execution_time": result.execution_time,
        "error_message": result.error_message,
        "failed_cell_index": result.failed_cell_index,
        "output_path": result.output_path,
    }

    if include_output and result.output_path and Path(result.output_path).exists():
        try:
            payload["output_contents"] = Path(result.output_path).read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            payload["output_contents"] = f"[failed to read executed notebook: {exc}]"

    return payload


def run_notebooks(
    notebooks: List[Path],
    timeout: int,
    expect_output: Sequence[str],
    output_mode: str,
    write_executed: Optional[Path],
) -> Tuple[int, str]:
    """Execute notebooks and format output for the requested mode."""

    NotebookExecutionResult, execute_notebook = _load_executor()

    temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    validation_dir = write_executed
    if expect_output and validation_dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        validation_dir = Path(temp_dir.name)

    results: List[NotebookExecutionResult] = []
    validation_errors: Dict[str, List[str]] = {}
    start = time.time()

    try:
        for nb_path in notebooks:
            output_path = None
            if validation_dir is not None:
                output_path = validation_dir / nb_path.name

            result = execute_notebook(nb_path, output_path=output_path, timeout=timeout)
            results.append(result)

            if result.success and expect_output and output_path:
                missing = _validate_output(output_path, expect_output)
                if missing:
                    validation_errors[result.notebook_path] = missing
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    total_time = time.time() - start
    overall_success = all(r.success for r in results) and not validation_errors

    if output_mode == "json":
        payload = {
            "notebooks_executed": len(results),
            "notebooks_passed": sum(1 for r in results if r.success),
            "notebooks_failed": sum(1 for r in results if not r.success),
            "total_execution_time": total_time,
            "results": [_serialize_result(r, include_output=False) for r in results],
            "validation_errors": validation_errors,
            "success": overall_success,
            "truncated": False,
        }
        return (0 if overall_success else 1), json.dumps(payload, indent=2)

    if output_mode == "full":
        lines: List[str] = []
        for res in results:
            lines.append("=" * 60)
            lines.append(f"Notebook: {res.notebook_path}")
            lines.append("=" * 60)
            lines.append(f"Success: {res.success}")
            lines.append(f"Execution Time: {res.execution_time:.2f}s")
            if res.error_message:
                lines.append(f"Error: {res.error_message}")
            if res.failed_cell_index is not None:
                lines.append(f"Failed Cell Index: {res.failed_cell_index}")
            if res.output_path:
                lines.append(f"Executed Copy: {res.output_path}")
            lines.append("")

        summary = _format_summary(results, validation_errors, total_time)
        lines.append(summary)
        combined = "\n".join(lines)
        truncated_text, truncated = _truncate(combined)
        suffix = "\n[output was truncated]" if truncated else ""
        return (0 if overall_success else 1), f"{truncated_text}{suffix}"

    summary = _format_summary(results, validation_errors, total_time)
    return (0 if overall_success else 1), summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for the notebook execution tool."""

    parser = argparse.ArgumentParser(
        description="Execute Jupyter notebooks with validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 .opencode/tool/run_notebook.py docs/Examples/setup-template-init-tutorial.ipynb\n"
            "  python3 .opencode/tool/run_notebook.py docs/Examples/ --recursive\n"
            "  python3 .opencode/tool/run_notebook.py notebook.ipynb --output json --timeout 300\n"
            "  python3 .opencode/tool/run_notebook.py notebook.ipynb --expect-output DataFrame plot\n"
        ),
    )
    parser.add_argument("notebook_path", help="Path to notebook file or directory")
    parser.add_argument(
        "--output",
        choices=["summary", "full", "json"],
        default="summary",
        help="Output mode: summary (default), full (per-notebook details), json (structured)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Per-notebook timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When notebook_path is a directory, search recursively for .ipynb files",
    )
    parser.add_argument(
        "--expect-output",
        nargs="*",
        default=[],
        help="Expected output substrings; missing values fail validation",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Working directory for execution; also prepends to PYTHONPATH",
    )
    parser.add_argument(
        "--write-executed",
        type=str,
        default=None,
        help="Optional directory to write executed notebook copies",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    target = Path(args.notebook_path)
    recursive = bool(args.recursive)
    timeout = int(args.timeout)
    output_mode = args.output
    expect_output = list(args.expect_output)
    cwd = Path(args.cwd) if args.cwd else None
    write_executed = Path(args.write_executed) if args.write_executed else None

    try:
        if timeout <= 0:
            raise NotebookToolError("Timeout must be positive")

        if cwd is not None and (not cwd.exists() or not cwd.is_dir()):
            raise NotebookToolError(f"cwd must be an existing directory: {cwd}")

        if write_executed is not None:
            write_executed.mkdir(parents=True, exist_ok=True)
            if not write_executed.is_dir():
                raise NotebookToolError(f"write-executed must be a directory: {write_executed}")

        notebooks = _collect_notebooks(target, recursive=recursive)
        total_timeout = timeout * len(notebooks) + TOTAL_GUARD_PADDING

        with pushd(cwd):
            with time_limit(total_timeout):
                exit_code, output = run_notebooks(
                    notebooks=notebooks,
                    timeout=timeout,
                    expect_output=expect_output,
                    output_mode=output_mode,
                    write_executed=write_executed,
                )
    except NotebookDependencyError as exc:
        print(f"ERROR: {exc}")
        return 1
    except NotebookToolError as exc:
        print(f"ERROR: {exc}")
        return 1
    except TimeoutError as exc:
        print(f"ERROR: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Unexpected failure: {exc}")
        return 1

    print(output)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
