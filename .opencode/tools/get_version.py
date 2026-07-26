#!/usr/bin/env python3
"""Read version metadata from repository files."""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

VERSION_PATTERN = re.compile(r'__version__\s*=\s*["\']([^"\']+)["\']')


def _ensure_confined_path(path: Path, *, root: Path) -> Path:
    """Resolve a path and fail closed when it escapes the allowed root.

    Args:
        path: Candidate path to normalize.
        root: Repository/worktree root that the path must remain under.

    Returns:
        Canonical resolved path under ``root``.

    Raises:
        ValueError: If the resolved path escapes the allowed root.
    """
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"Path resolves outside allowed root: {resolved_path} (root: {resolved_root})"
        ) from exc
    return resolved_path


def _read_pyproject_version(path: Path, *, allowed_root: Path) -> str:
    """Read a version string from a ``pyproject.toml`` file.

    Supports PEP 621 ``project.version``, Hatch dynamic version paths, and
    Poetry ``tool.poetry.version`` fields.

    Args:
        path: Path to the ``pyproject.toml`` file.

    Returns:
        Resolved version string.

    Raises:
        ValueError: If no supported version field can be resolved.
    """
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    project = data.get("project")
    if isinstance(project, dict):
        version = project.get("version")
        if isinstance(version, str) and version.strip():
            return version.strip()

        dynamic = project.get("dynamic")
        if isinstance(dynamic, list) and "version" in dynamic:
            hatch = data.get("tool", {}).get("hatch", {}).get("version", {})
            version_path = hatch.get("path") if isinstance(hatch, dict) else None
            if isinstance(version_path, str) and version_path.strip():
                target = _ensure_confined_path(
                    path.parent / version_path.strip(),
                    root=allowed_root,
                )
                return _read_python_dunder_version(target)

    poetry = data.get("tool", {}).get("poetry", {})
    if isinstance(poetry, dict):
        version = poetry.get("version")
        if isinstance(version, str) and version.strip():
            return version.strip()

    raise ValueError(f"No version field found in {path}")


def _read_python_dunder_version(path: Path) -> str:
    """Read a ``__version__`` assignment from a Python source file.

    Args:
        path: Path to the Python file to inspect.

    Returns:
        Version string captured from the assignment.

    Raises:
        ValueError: If the file does not contain a supported ``__version__`` assignment.
    """
    match = VERSION_PATTERN.search(path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"No __version__ assignment found in {path}")
    return match.group(1)


def _read_package_json_version(path: Path) -> str:
    """Read a version string from a ``package.json`` file.

    Args:
        path: Path to the ``package.json`` file.

    Returns:
        Version string from the package metadata.

    Raises:
        ValueError: If the file does not contain a non-empty ``version`` field.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    version = data.get("version")
    if isinstance(version, str) and version.strip():
        return version.strip()
    raise ValueError(f"No version field found in {path}")


def resolve_target_path(raw_path: str | None, cwd: Path | None = None) -> Path:
    """Resolve the target metadata file for version lookup.

    Args:
        raw_path: Optional user-provided file path.
        cwd: Optional base directory used for path resolution.

    Returns:
        Absolute path to the version metadata file.

    Raises:
        FileNotFoundError: If no default metadata file can be found.
    """
    base = (cwd or Path.cwd()).resolve()
    if raw_path and raw_path.strip():
        return _ensure_confined_path(base / raw_path.strip(), root=base)

    for candidate in (base / "pyproject.toml", base / "package.json"):
        if candidate.exists():
            return _ensure_confined_path(candidate, root=base)
    raise FileNotFoundError(
        "Could not find pyproject.toml or package.json in the current directory"
    )


def get_version(target_path: Path, *, allowed_root: Path | None = None) -> str:
    """Return the version string stored in a supported metadata file.

    Args:
        target_path: Path to ``pyproject.toml`` or ``package.json``.

    Returns:
        Resolved version string.

    Raises:
        FileNotFoundError: If the target path does not exist.
        ValueError: If the target is not a supported metadata file.
    """
    if allowed_root is not None:
        target_path = _ensure_confined_path(target_path, root=allowed_root)
    if not target_path.exists():
        raise FileNotFoundError(f"File not found: {target_path}")
    if not target_path.is_file():
        raise ValueError(f"Path is not a file: {target_path}")

    if target_path.name == "pyproject.toml":
        root = allowed_root if allowed_root is not None else target_path.parent
        return _read_pyproject_version(target_path, allowed_root=root)
    if target_path.name == "package.json":
        return _read_package_json_version(target_path)

    raise ValueError(
        "Unsupported file type for version lookup: "
        f"{target_path.name}. Expected pyproject.toml or package.json"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the command-line version lookup tool.

    Args:
        argv: Optional CLI arguments used instead of ``sys.argv``.

    Returns:
        Process exit code.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) > 1:
        print("Usage: get_version.py [file]", file=sys.stderr)
        return 2

    raw_path = args[0] if args else None

    try:
        cwd = Path.cwd().resolve()
        target_path = resolve_target_path(raw_path, cwd=cwd)
        print(get_version(target_path, allowed_root=cwd))
        return 0
    except Exception as exc:  # pragma: no cover - wrapper owns formatting precedence
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
