"""
Generate API reference stub pages for mkdocstrings using mkdocs-gen-files.

Mirrors the old Handsdown layout:
  - Writes to API/...
  - Only includes files under particula/particula
  - Excludes tests/* and build/*
  - Creates:
      API/index.md
      API/particula.md
      API/particula/<subpkg>/.../<module>.md
"""

from __future__ import annotations
import os
from pathlib import Path
import mkdocs_gen_files
import logging

logger = logging.getLogger("mkdocs")

# Get the repository root - when running under mkdocs, cwd should be the docs root
REPO_PATH = Path.cwd()
PACKAGE = "particula"

# Source root is just the particula package directory
SOURCE_ROOT = REPO_PATH / PACKAGE

# If SOURCE_ROOT doesn't exist, we might be in a subdirectory
if not SOURCE_ROOT.exists():
    # Try going up one level
    REPO_PATH = REPO_PATH.parent
    SOURCE_ROOT = REPO_PATH / PACKAGE

logger.info(f"Repository path: {REPO_PATH}")
logger.info(f"Source root: {SOURCE_ROOT}")

EXCLUDE_DIR_NAMES = {"tests", "build"}
EXCLUDE_FILE_PATTERNS = ["*_test.py", "*_tests.py", "test_*.py"]
INCLUDE_SUFFIX = ".py"


def is_excluded(path: Path) -> bool:
    # Exclude if any parent directory matches EXCLUDE_DIR_NAMES or contains "test"
    for part in path.parts:
        if part in EXCLUDE_DIR_NAMES or "test" in part.lower():
            return True

    # Exclude test files by pattern
    for pattern in EXCLUDE_FILE_PATTERNS:
        if path.match(pattern):
            return True

    return False


def is_in_package(py_file: Path) -> bool:
    """Check if a Python file is in a proper package (has __init__.py in all parent dirs)."""
    # Check all parent directories up to SOURCE_ROOT
    current = py_file.parent
    while current != SOURCE_ROOT and current.is_relative_to(SOURCE_ROOT):
        if not (current / "__init__.py").exists():
            return False
        current = current.parent
    return True


def module_name_from_file(py_file: Path) -> str | None:
    """
    Convert a file path like:
        particula/a/b/c.py
    into a module name:
        particula.a.b.c
    and for __init__.py, use the package name:
        particula.a.b
    Returns None if py_file is outside SOURCE_ROOT.
    """
    try:
        rel = py_file.relative_to(SOURCE_ROOT)
    except ValueError:
        return None

    parts = list(rel.parts)
    if not parts:
        return None

    # Handle __init__.py -> package page
    if parts[-1] == "__init__.py":
        mod_parts = parts[:-1]
    else:
        mod_parts = parts
        mod_parts[-1] = mod_parts[-1].removesuffix(".py")

    return ".".join([PACKAGE] + mod_parts)


def get_display_name(modname: str, category: str = None) -> str:
    """
    Get a cleaner display name for the TOC with parent folder context.
    Returns the module name with one level of parent folder for clarity.

    E.g., 'particula.activity.gibbs' -> 'gibbs' (top-level in category)
    E.g., 'particula.particles.properties.activity_module' ->
          'properties/activity_module'
    """
    # Remove the package prefix
    if modname.startswith(f"{PACKAGE}."):
        rel_path = modname[len(PACKAGE) + 1:]
    else:
        rel_path = modname

    parts = rel_path.split(".")

    # If it's a top-level module or has only one level, return just the name
    if len(parts) <= 1:
        return parts[-1]

    # If category is provided and module is directly under category,
    # just return the module name
    if category and len(parts) == 2 and parts[0] == category.lower():
        return parts[-1]

    # For deeper modules, show parent/module_name
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"

    return parts[-1]


def organize_modules(modules: list[str]) -> dict[str, list[str]]:
    """
    Organize modules into a hierarchical structure.
    Returns a dict with top-level categories as keys and module lists as
    values.
    """
    organized = {}
    root_modules = []

    for mod in sorted(modules):
        # Remove package prefix for organization
        if mod == PACKAGE:
            root_modules.append(mod)
            continue

        if not mod.startswith(f"{PACKAGE}."):
            continue

        rel_mod = mod[len(PACKAGE) + 1:]
        parts = rel_mod.split(".")

        if len(parts) == 1:
            # Top-level module (e.g., particula.aerosol)
            root_modules.append(mod)
        else:
            # Submodule (e.g., particula.activity.gibbs)
            category = parts[0]
            if category not in organized:
                organized[category] = []
            organized[category].append(mod)

    # Add root modules at the beginning
    if root_modules:
        organized = {"Core": root_modules, **organized}

    return organized


def main() -> None:
    print(f"mk_generator.py starting - SOURCE_ROOT: {SOURCE_ROOT}",
          flush=True)
    if os.environ.get("MKGEN_DEBUG") == "1":
        with open("/tmp/mkgen_debug.txt", "w") as f:
            f.write(f"Script ran! SOURCE_ROOT: {SOURCE_ROOT}\n")
            f.write(f"SOURCE_ROOT exists: {SOURCE_ROOT.exists()}\n")

    out_dir = Path("API")  # keep same top-level directory name

    # Generate folder structure, if needed
    os.makedirs(REPO_PATH / "docs/API/", exist_ok=True)

    # Walk filesystem and collect all modules first
    if not SOURCE_ROOT.exists():
        return

    # Collect module names to avoid duplicates
    seen: set[str] = set()
    all_modules: list[str] = []

    logger.info(f"Scanning for Python files in {SOURCE_ROOT}")

    for py_file in SOURCE_ROOT.rglob(f"*{INCLUDE_SUFFIX}"):
        if is_excluded(py_file):
            logger.debug(f"Excluding {py_file}")
            continue

        if not is_in_package(py_file):
            print(f"Skipping {py_file} - not in a proper package")
            logger.debug(f"Skipping {py_file} - not in proper package")
            continue

        modname = module_name_from_file(py_file)
        if not modname:
            logger.warning(f"Could not determine module name for {py_file}")
            continue
        if modname in seen:
            logger.debug(f"Skipping duplicate module {modname}")
            continue
        seen.add(modname)
        all_modules.append(modname)
        logger.info(f"Generating docs for {modname} from {py_file}")

        # Map module name to API/<module>.md path
        # e.g., particula.a.b -> API/particula/a/b.md
        rel_md = out_dir / (modname.replace(".", "/") + ".md")

        # Create stub with mkdocstrings directive
        with mkdocs_gen_files.open(rel_md, "w") as f:
            f.write(f"# `{modname}`\n\n")
            f.write(f"::: {modname}\n")

        # Optional: make "Edit this page" point to the source file
        try:
            edit_rel = py_file.relative_to(REPO_PATH)
            mkdocs_gen_files.set_edit_path(rel_md, str(edit_rel))
        except ValueError:
            pass

    # Create organized index page
    index_path = out_dir / "index.md"
    organized = organize_modules(all_modules)

    with mkdocs_gen_files.open(index_path, "w") as f:
        f.write("# API Reference\n\n")

        for category, modules in organized.items():
            # Write category header
            f.write(f"## {category.title()}\n\n")

            for modname in sorted(modules):
                display = get_display_name(modname, category)
                link = modname.replace(".", "/") + ".md"
                f.write(f"- [{display}]({link})\n")

            f.write("\n")

    # Create SUMMARY.md for literate-nav navigation
    # summary_path = out_dir / "SUMMARY.md"
    # with mkdocs_gen_files.open(summary_path, "w") as f:
    #     f.write("# API Reference\n\n")

    #     for category, modules in organized.items():
    #         # Write category as expandable section
    #         f.write(f"* {category}\n")

    #         for modname in sorted(modules):
    #             display = get_display_name(modname, category)
    #             link = modname.replace(".", "/") + ".md"
    #             # Use indentation for literate-nav hierarchy
    #             f.write(f"    * [{display}]({link})\n")


# Always run main() when imported by mkdocs-gen-files
main()