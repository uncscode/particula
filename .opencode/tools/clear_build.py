#!/usr/bin/env python3
"""Build directory cleanup tool for ADW.

Safely removes build directories with validation to prevent accidental
selection outside the project root. Supports dry-run mode for previewing
deletions and a force flag to perform deletion without interactivity.

Usage:
    python3 clear_build.py --build-dir build --dry-run
    python3 clear_build.py --build-dir build --force
    python3 clear_build.py --build-dir build/debug --force
    python3 clear_build.py --project-root /repo --build-dir build

WARNING: This tool permanently deletes files when run with ``--force``.
Run with ``--dry-run`` first to review what would be deleted.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

SEPARATOR = "=" * 60


def find_project_root(start: Optional[Path] = None) -> Path:
    """Locate the project root by walking upward.

    The first ancestor containing ``.git`` or ``pyproject.toml`` is treated as
    the project root. If neither is found, the current working directory is
    returned.

    Args:
        start: Optional starting path. Defaults to ``Path.cwd()``.

    Returns:
        Resolved ``Path`` representing the project root.
    """
    current = (start or Path.cwd()).resolve()
    for ancestor in [current] + list(current.parents):
        if (ancestor / ".git").exists() or (
            ancestor / "pyproject.toml"
        ).exists():
            return ancestor
    return current


def validate_path(
    build_dir: str | Path, project_root: Path | None = None
) -> Path:
    """Validate that the build directory is within the project root.

    Resolves both paths and ensures the build directory is contained within the
    project root but is not the root itself. Symlink escapes and traversal
    attempts are rejected because ``Path.resolve()`` follows symlinks.

    Args:
        build_dir: Target build directory to clear.
        project_root: Explicit project root. If ``None``, it is discovered via
            :func:`find_project_root`.

    Returns:
        Resolved build directory path.

    Raises:
        ValueError: If the path is empty, outside the project root, or equals
            the root itself.
    """
    if not str(build_dir).strip():
        raise ValueError("Build directory path must not be empty")

    root = (project_root or find_project_root()).resolve()
    resolved_build = Path(build_dir).expanduser().resolve()

    if not resolved_build.is_relative_to(root):
        raise ValueError(
            f"Path '{resolved_build}' is outside project root '{root}'."
        )

    if resolved_build == root:
        raise ValueError("Refusing to operate on the project root directory.")

    return resolved_build


def get_directory_size(path: Path) -> Tuple[int, int]:
    """Calculate total size and file count for a directory without symlinks.

    Args:
        path: Directory to inspect.

    Returns:
        A tuple of ``(total_bytes, file_count)``. Returns ``(0, 0)`` when the
        path does not exist.
    """
    if not path.exists():
        return 0, 0

    total_bytes = 0
    file_count = 0

    for root, dirnames, filenames in os.walk(path, followlinks=False):
        # Remove symlinked directories to avoid walking them
        dirnames[:] = [d for d in dirnames if not (Path(root) / d).is_symlink()]

        for name in filenames:
            candidate = Path(root) / name
            if candidate.is_symlink():
                continue
            try:
                total_bytes += candidate.stat().st_size
                file_count += 1
            except (FileNotFoundError, PermissionError, OSError):
                # Skip files that disappear or are unreadable
                continue

    return total_bytes, file_count


def format_size(bytes_size: int) -> str:
    """Convert a byte count into a human-readable string.

    Args:
        bytes_size: Size in bytes.

    Returns:
        Human-readable size using B, KB, MB, GB, or TB units.
    """
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(bytes_size)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


def _append_validation_footer(
    lines: List[str], validation_errors: List[str]
) -> None:
    lines.append("\n" + SEPARATOR)
    if validation_errors:
        lines.append("VALIDATION: FAILED")
        lines.append(SEPARATOR)
        for error in validation_errors:
            lines.append(f"  - {error}")
    else:
        lines.append("VALIDATION: PASSED")
        lines.append(SEPARATOR)


def clear_build(
    build_dir: str | Path,
    *,
    dry_run: bool = False,
    force: bool = False,
    project_root: Path | None = None,
) -> Tuple[int, str]:
    """Clear a build directory with safety checks.

    Validates the target path, reports size/file counts, supports dry-run mode,
    and requires ``--force`` to perform deletion.

    Args:
        build_dir: Directory to delete.
        dry_run: If ``True``, report what would be deleted without removing
            anything.
        force: If ``True``, proceed with deletion without additional prompts.
        project_root: Optional project root override for validation.

    Returns:
        Tuple of ``(exit_code, summary_text)``. Exit code ``0`` on success or
        safe no-op; ``1`` on validation or deletion failure.
    """
    lines: List[str] = [SEPARATOR, "CLEAR BUILD", SEPARATOR]
    validation_errors: List[str] = []

    try:
        resolved_root = (project_root or find_project_root()).resolve()
        resolved_build = validate_path(build_dir, resolved_root)
    except ValueError as exc:  # Validation failure
        lines.append(f"\nBuild Directory: {build_dir}")
        validation_errors.append(str(exc))
        _append_validation_footer(lines, validation_errors)
        return 1, "\n".join(lines)

    lines.append(f"\nProject Root: {resolved_root}")
    lines.append(f"Build Directory: {resolved_build}")

    if not resolved_build.exists():
        lines.append("Status: Directory does not exist (nothing to delete)")
        _append_validation_footer(lines, validation_errors)
        return 0, "\n".join(lines)

    total_bytes, file_count = get_directory_size(resolved_build)
    lines.append(f"Files: {file_count}")
    lines.append(f"Size: {format_size(total_bytes)}")

    if dry_run:
        lines.append("\nMode: DRY RUN (no files deleted)")
        lines.append(
            f"Would delete: {file_count} files ({format_size(total_bytes)})"
        )
        _append_validation_footer(lines, validation_errors)
        return 0, "\n".join(lines)

    if not force:
        validation_errors.append(
            "Deletion not performed. Re-run with --force to delete."
        )
        _append_validation_footer(lines, validation_errors)
        return 1, "\n".join(lines)

    try:
        shutil.rmtree(resolved_build, ignore_errors=False)
        lines.append("\nMode: FORCE DELETE")
        lines.append(
            f"Deleted: {file_count} files ({format_size(total_bytes)})"
        )
        _append_validation_footer(lines, validation_errors)
        return 0, "\n".join(lines)
    except PermissionError as exc:
        validation_errors.append(f"Permission denied while deleting: {exc}")
    except Exception as exc:  # pragma: no cover - defensive
        validation_errors.append(f"Failed to delete build directory: {exc}")

    _append_validation_footer(lines, validation_errors)
    return 1, "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    description = (
        "Safely clear build directories with validation, dry-run previews, "
        "and a force flag to perform deletion."
    )
    epilog = (
        "Examples:\n"
        "  %(prog)s --build-dir build --dry-run    Preview deletion\n"
        "  %(prog)s --build-dir build --force      Delete without prompt\n"
        "  %(prog)s --build-dir build/debug        Clear a specific variant\n\n"
        "WARNING: This tool permanently deletes files. Use --dry-run first."
    )

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default="build",
        help="Build directory to clear (default: 'build').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview deletion without removing files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Required to perform deletion (safety guard).",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Optional project root override (defaults to discovered root).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for the clear_build tool."""
    parser = build_parser()
    args = parser.parse_args(argv)

    exit_code, summary = clear_build(
        build_dir=args.build_dir,
        dry_run=args.dry_run,
        force=args.force,
        project_root=Path(args.project_root) if args.project_root else None,
    )

    print(summary)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
