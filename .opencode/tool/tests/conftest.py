from __future__ import annotations

import shutil
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[3]
tool_root = Path(__file__).resolve().parents[1]
example_cpp_dir = repo_root / "example_cpp_dev"


def cmake_available() -> bool:
    """Return True when CMake is available on PATH."""

    return shutil.which("cmake") is not None


def ninja_available() -> bool:
    """Return True when Ninja is available on PATH."""

    return shutil.which("ninja") is not None


def gcovr_available() -> bool:
    """Return True when gcovr is available on PATH."""

    return shutil.which("gcovr") is not None


def sanitizers_available() -> bool:
    """Return True when a sanitizer-capable compiler is available."""

    return shutil.which("clang++") is not None or shutil.which("g++") is not None


requires_cmake = pytest.mark.skipif(not cmake_available(), reason="CMake not installed")
requires_ninja = pytest.mark.skipif(not ninja_available(), reason="Ninja not installed")
requires_gcovr = pytest.mark.skipif(not gcovr_available(), reason="gcovr not installed")
requires_sanitizers = pytest.mark.skipif(
    not sanitizers_available(), reason="Sanitizer-capable compiler not available"
)

# Default mark for integration suites in this directory.
INTEGRATION_MARK = pytest.mark.integration


def _coverage_targets_from_tests(test_files: list[Path]) -> list[str]:
    targets: set[str] = set()
    for test_file in test_files:
        if test_file.suffix != ".py":
            continue
        stem = test_file.stem
        if stem.endswith("_integration_test"):
            stem = stem[: -len("_integration_test")]
        elif stem.endswith("_test"):
            stem = stem[: -len("_test")]
        module_path = tool_root / f"{stem}.py"
        if module_path.exists():
            targets.add(str(module_path))
    return sorted(targets)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Scope coverage collection to tool modules during local tool tests."""
    root_path = getattr(config, "rootpath", None)
    if root_path is None:
        root_path = Path(str(getattr(config, "rootdir", repo_root)))
    is_tool_test = Path(str(root_path)).resolve() == repo_root
    if not is_tool_test:
        return

    test_paths = [Path(item.fspath) for item in items]
    coverage_targets = _coverage_targets_from_tests(test_paths)
    if coverage_targets:
        config.option.cov = coverage_targets
        config.option.cov_source = coverage_targets
    else:
        config.option.cov = [str(tool_root)]
        config.option.cov_source = [str(tool_root)]
    config.option.cov_config = "/dev/null"
    config.option.cov_fail_under = 0


__all__ = [
    "repo_root",
    "example_cpp_dir",
    "cmake_available",
    "ninja_available",
    "gcovr_available",
    "sanitizers_available",
    "requires_cmake",
    "requires_ninja",
    "requires_gcovr",
    "requires_sanitizers",
    "INTEGRATION_MARK",
]
