#!/usr/bin/env python3
"""Notebook Validation Tool for ADW.

Validates Jupyter notebook structure and syntax without executing code and
adds Jupytext-powered conversion and sync operations. Supports validation for
single files or directories (optional recursion) plus conversion (`--convert-to-py`),
bidirectional sync (`--sync`), and CI read-only sync checks (`--check-sync`).

Exit codes:
  - 0: success (valid notebooks / conversions / sync / in-sync)
  - 1: functional failure (invalid notebooks, conversion failures, out-of-sync)
  - 2: tool/argument error

Examples:
  # Validation
  python3 .opencode/tool/validate_notebook.py docs/Examples --recursive --output json
  # Conversion
  python3 .opencode/tool/validate_notebook.py notebook.ipynb --convert-to-py
  python3 .opencode/tool/validate_notebook.py docs/Examples --recursive --convert-to-py --output-dir scripts
  # Sync
  python3 .opencode/tool/validate_notebook.py notebook.ipynb --sync
  # Check-sync (read-only)
  python3 .opencode/tool/validate_notebook.py docs/Examples --recursive --check-sync
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Lazy imports to avoid unnecessary dependencies for conversion/sync modes
if TYPE_CHECKING:
    from adw.utils.notebook_validation import NotebookValidationResult

MAX_ERRORS_PER_NOTEBOOK = 5
FULL_OUTPUT_CHAR_LIMIT = 4000
CHECK_SYNC_MTIME_TOLERANCE = 0.0


class ValidationToolError(Exception):
    """Predictable error for user-facing failures."""


def _is_hidden(path: Path) -> bool:
    """Check whether any segment of the path is hidden.

    Args:
        path: Filesystem path to inspect.

    Returns:
        True when any segment starts with a dot, otherwise False.
    """

    return any(part.startswith(".") for part in path.parts)


def _collect_notebooks(target: Path, recursive: bool) -> List[Path]:
    """Collect notebook paths from a file or directory.

    Skips hidden paths and .ipynb_checkpoints, returns deterministically sorted list.
    Hidden paths are defined by any segment starting with a dot.
    """

    if target.is_file() and target.suffix == ".ipynb":
        return [target]

    if not target.exists():
        raise ValidationToolError(f"Path not found: {target}")

    if not target.is_dir():
        raise ValidationToolError(f"Path must be .ipynb file or directory: {target}")

    candidates: Iterable[Path]
    if recursive:
        candidates = target.rglob("*.ipynb")
    else:
        candidates = target.glob("*.ipynb")

    notebooks = [p for p in candidates if ".ipynb_checkpoints" not in p.parts and not _is_hidden(p)]

    if not notebooks:
        raise ValidationToolError(f"No notebooks found under {target}")

    return sorted(notebooks)


def _truncate(text: str, limit: int = FULL_OUTPUT_CHAR_LIMIT) -> tuple[str, bool]:
    """Truncate text to the limit, returning text and truncation flag.

    Args:
        text: Content to truncate.
        limit: Maximum allowed characters; defaults to FULL_OUTPUT_CHAR_LIMIT.

    Returns:
        Tuple of truncated text and a boolean indicating whether truncation occurred.
    """

    if len(text) <= limit:
        return text, False
    return f"{text[:limit]}\n...\n[output truncated to {limit} chars]", True


def _format_summary(results: List[NotebookValidationResult]) -> str:
    """Return summary text for a batch of notebook validation results.

    Args:
        results: List of notebook validation results.

    Returns:
        Human-readable summary string with counts and failures.
    """

    checked = len(results)
    valid = sum(1 for r in results if r.valid)
    invalid = checked - valid

    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("NOTEBOOK VALIDATION SUMMARY")
    lines.append("=" * 60)
    lines.append(f"\nNotebooks Checked: {checked}")
    lines.append(f"  Valid:   {valid}")
    if invalid:
        lines.append(f"  Invalid: {invalid}")
        lines.append(f"\nFailures ({invalid}):")
        for result in results:
            if result.valid:
                continue
            lines.append(f"  - {Path(result.notebook_path).name}")
            for error in result.errors[:MAX_ERRORS_PER_NOTEBOOK]:
                lines.append(f"      {error.error_type}: {error.message}")
            remaining = max(len(result.errors) - MAX_ERRORS_PER_NOTEBOOK, 0)
            if remaining:
                lines.append(f"      ... and {remaining} more errors")

    lines.append("\n" + "=" * 60)
    lines.append("VALIDATION: PASSED" if invalid == 0 else "VALIDATION: FAILED")
    lines.append("=" * 60)
    return "\n".join(lines)


def _format_full(results: List[NotebookValidationResult], truncated_notice: bool = True) -> str:
    """Return detailed text for each notebook, optionally noting truncation.

    Args:
        results: List of notebook validation results.
        truncated_notice: Whether to append a note when output is truncated.

    Returns:
        String containing detailed notebook validation information.
    """

    lines: List[str] = []
    for result in results:
        lines.append("=" * 60)
        lines.append(f"Notebook: {result.notebook_path}")
        lines.append("=" * 60)
        lines.append(f"Valid: {result.valid}")
        if result.warnings:
            lines.append("Warnings:")
            for warning in result.warnings:
                lines.append(f"  - {warning}")
        if result.errors:
            lines.append("Errors:")
            for error in result.errors:
                lines.append(f"  - {error.error_type}: {error.message}")
                if error.cell_index is not None:
                    lines.append(f"      cell_index: {error.cell_index}")
                if error.details:
                    for key, value in error.details.items():
                        lines.append(f"      {key}: {value}")
        lines.append("")

    combined = "\n".join(lines)
    truncated_text, truncated = _truncate(combined)
    if truncated and truncated_notice:
        return f"{truncated_text}\n[output was truncated]"
    return truncated_text


def _format_json(results: List[NotebookValidationResult], skip_syntax: bool) -> str:
    """Return JSON string with validation results and meta fields.

    Args:
        results: List of notebook validation results.
        skip_syntax: Whether syntax errors were converted to warnings.

    Returns:
        JSON-formatted string describing validation outcomes.
    """

    payload = {
        "notebooks_checked": len(results),
        "notebooks_valid": sum(1 for r in results if r.valid),
        "notebooks_invalid": sum(1 for r in results if not r.valid),
        "skip_syntax": skip_syntax,
        "results": [],
    }

    serialized = []
    for result in results:
        serialized_errors = [
            {
                "error_type": err.error_type,
                "message": err.message,
                "cell_index": err.cell_index,
                "details": err.details,
            }
            for err in result.errors
        ]
        serialized.append(
            {
                "notebook_path": result.notebook_path,
                "valid": result.valid,
                "errors": serialized_errors,
                "warnings": result.warnings,
            }
        )
    payload["results"] = serialized
    return json.dumps(payload, indent=2)


def _validate_output_dir_usage(args: argparse.Namespace) -> None:
    if args.output_dir and not args.convert_to_py:
        raise ValidationToolError("--output-dir is only allowed with --convert-to-py")


def _ensure_validation_only_flags(args: argparse.Namespace) -> None:
    if (args.output != "summary" or args.skip_syntax) and (
        args.convert_to_py or args.sync or args.check_sync
    ):
        raise ValidationToolError(
            "--output/--skip-syntax can only be used with validation (omit convert/sync flags)"
        )


def _handle_convert(args: argparse.Namespace) -> int:
    from adw.utils.notebook_jupytext import SyncResult, notebook_to_script

    target = Path(args.notebook_path)
    notebooks = _collect_notebooks(target, recursive=args.recursive)
    output_dir = Path(args.output_dir) if args.output_dir else None

    failures: list[SyncResult] = []
    for notebook in notebooks:
        output_path = None
        if output_dir:
            output_path = output_dir / notebook.with_suffix(".py").name
        result = notebook_to_script(notebook, output_path)
        status = "✓" if result.success else "✗"
        action = result.action
        message = f"{status} {notebook} -> {result.target_path} ({action})"
        if result.error_message:
            message = f"{message}: {result.error_message}"
        print(message)
        if not result.success:
            failures.append(result)

    if failures:
        return 1
    return 0


def _handle_sync(args: argparse.Namespace) -> int:
    from adw.utils.notebook_jupytext import SyncResult, sync_notebook_script

    if args.output_dir:
        raise ValidationToolError("--output-dir is only allowed with --convert-to-py")

    target = Path(args.notebook_path)
    notebooks = _collect_notebooks(target, recursive=args.recursive)

    failures: list[SyncResult] = []
    for notebook in notebooks:
        result = sync_notebook_script(notebook)
        status = "✓" if result.success else "✗"
        message = f"{status} {notebook} -> {result.target_path} ({result.action})"
        if result.error_message:
            message = f"{message}: {result.error_message}"
        print(message)
        if not result.success:
            failures.append(result)

    if failures:
        return 1
    return 0


def _handle_check_sync(args: argparse.Namespace) -> int:
    if args.output_dir:
        raise ValidationToolError("--output-dir is only allowed with --convert-to-py")

    target = Path(args.notebook_path)
    notebooks = _collect_notebooks(target, recursive=args.recursive)

    out_of_sync: list[str] = []
    for notebook in notebooks:
        script_path = notebook.with_suffix(".py")
        if not script_path.exists():
            out_of_sync.append(f"{notebook}: missing script")
            continue
        try:
            nb_mtime = notebook.stat().st_mtime_ns
            py_mtime = script_path.stat().st_mtime_ns
        except FileNotFoundError:
            out_of_sync.append(f"{notebook}: failed to stat files")
            continue

        mtime_diff = nb_mtime - py_mtime
        if abs(mtime_diff) <= CHECK_SYNC_MTIME_TOLERANCE:
            continue
        if mtime_diff > 0:
            out_of_sync.append(f"{notebook}: notebook is newer")
        else:
            out_of_sync.append(f"{notebook}: script is newer")

    if out_of_sync:
        print("Out of sync:")
        for line in out_of_sync:
            print(f"  ✗ {line}")
        return 1

    print(f"All {len(notebooks)} notebooks are in sync with their scripts")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint with validation, conversion, sync, and check-sync modes."""

    parser = argparse.ArgumentParser(
        description=("Validate notebooks or perform Jupytext conversion/sync operations."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Validation\n"
            "  python3 .opencode/tool/validate_notebook.py notebook.ipynb\n"
            "  python3 .opencode/tool/validate_notebook.py docs/Examples --recursive --output json\n"
            "\n"
            "  # Conversion\n"
            "  python3 .opencode/tool/validate_notebook.py notebook.ipynb --convert-to-py\n"
            "  python3 .opencode/tool/validate_notebook.py docs/Examples --recursive --convert-to-py --output-dir scripts\n"
            "\n"
            "  # Sync\n"
            "  python3 .opencode/tool/validate_notebook.py notebook.ipynb --sync\n"
            "  python3 .opencode/tool/validate_notebook.py docs/Examples --recursive --check-sync\n"
        ),
    )

    parser.add_argument("notebook_path", help="Path to notebook file or directory")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search directories recursively for .ipynb files",
    )

    parser.add_argument(
        "--output",
        choices=["summary", "full", "json"],
        default="summary",
        help="Output mode (validation only; default: summary)",
    )
    parser.add_argument(
        "--skip-syntax",
        action="store_true",
        help=("Skip Python syntax checking (validation only); syntax issues become warnings"),
    )

    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--convert-to-py",
        dest="convert_to_py",
        action="store_true",
        help="Convert notebooks to .py:percent format",
    )
    action_group.add_argument(
        "--sync",
        action="store_true",
        help="Bidirectional sync between notebook and script (newer file wins)",
    )
    action_group.add_argument(
        "--check-sync",
        dest="check_sync",
        action="store_true",
        help="Check notebook/script sync state (read-only, CI friendly)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for converted files (only with --convert-to-py)",
    )

    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 2

    try:
        _validate_output_dir_usage(args)
        _ensure_validation_only_flags(args)

        if args.convert_to_py:
            return _handle_convert(args)
        if args.sync:
            return _handle_sync(args)
        if args.check_sync:
            return _handle_check_sync(args)

        # Validation path
        from adw.utils.notebook_validation import validate_notebook_json

        target = Path(args.notebook_path)
        notebooks = _collect_notebooks(target, recursive=args.recursive)
        results = [validate_notebook_json(nb, skip_syntax=args.skip_syntax) for nb in notebooks]

        if args.output == "json":
            output = _format_json(results, skip_syntax=args.skip_syntax)
        elif args.output == "full":
            output = _format_full(results)
        else:
            output = _format_summary(results)

        print(output)
        all_valid = all(r.valid for r in results)
        return 0 if all_valid else 1
    except ValidationToolError as exc:
        print(f"ERROR: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Unexpected failure: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
