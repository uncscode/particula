"""Locate version-matched documentation without importing Particula.

Run ``particula-docs`` or ``python -m particula_docs`` in the selected Python
installation. Only the standard library is needed, even without NumPy or Warp.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path

DOC_ROOTS = ("guide", "theory", "contribute", "examples", "data", "images")
DOC_EXCLUDED_PARTS = {
    "__pycache__",
    "node_modules",
    "API",
    "api",
    "build",
    "dist",
    "site",
}
DOC_EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".bak", ".tmp", ".log"}
MANIFEST = "bundle-manifest.json"


def source_files(docs: Path) -> list[Path]:
    """Select maintained docs and assets, excluding generated/local content."""
    candidates = [docs / "index.md"]
    for name in DOC_ROOTS:
        candidates.extend((docs / name).rglob("*"))
    selected = []
    for path in sorted(candidates):
        relative = path.relative_to(docs)
        if (
            any(
                part.startswith(".") or part in DOC_EXCLUDED_PARTS
                for part in relative.parts
            )
            or path.suffix in DOC_EXCLUDED_SUFFIXES
        ):
            continue
        if path.is_symlink():
            raise ValueError(f"Documentation symlink is not supported: {path}")
        if path.is_file():
            selected.append(path)
    if (docs / "guide/index.md") not in selected:
        raise FileNotFoundError(f"Maintained documentation is missing: {docs}")
    return selected


def source_version(package: Path) -> str:
    """Read literal package metadata without executing its initializer."""
    for node in ast.parse((package / "__init__.py").read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str) and value:
                return value
    raise ValueError("Particula has no literal version")


def documentation_path() -> Path:
    """Return installed bundled docs or the maintained editable source tree."""
    spec = importlib.util.find_spec("particula")
    if spec is None or spec.origin is None:
        raise FileNotFoundError(
            "Particula is not installed in this interpreter"
        )
    package = Path(spec.origin).resolve().parent
    bundled = package / "docs"
    if (bundled / MANIFEST).is_file():
        return bundled
    # PEP 660 editable installs resolve the package to the maintained checkout.
    source = package.parent / "docs"
    if (package.parent / "pyproject.toml").is_file() and (
        source / "guide/index.md"
    ).is_file():
        return source
    raise FileNotFoundError("This Particula installation has no bundled docs")


def documentation_info() -> dict:
    """Describe source/version and uncompressed documentation payload size."""
    docs = documentation_path()
    manifest = docs / MANIFEST
    if manifest.is_file():
        info = json.loads(manifest.read_text())
        return {
            "path": str(docs),
            "mode": "bundled",
            "version": info["version"],
            "file_count": len(info["files"]),
            "size_bytes": info["size_bytes"],
            "manifest": str(manifest),
        }
    files = source_files(docs)
    return {
        "path": str(docs),
        "mode": "editable",
        "version": source_version(docs.parent / "particula"),
        "file_count": len(files),
        "size_bytes": sum(path.stat().st_size for path in files),
        "manifest": None,
    }


def main() -> int:
    """Print a path by default, or machine-readable provenance with --json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="Show bundle metadata"
    )
    args = parser.parse_args()
    try:
        print(
            json.dumps(documentation_info(), sort_keys=True)
            if args.json
            else documentation_path()
        )
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"particula-docs: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
