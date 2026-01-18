from __future__ import annotations

import shutil
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[3]
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
    return (
        shutil.which("clang++") is not None or shutil.which("g++") is not None
    )


requires_cmake = pytest.mark.skipif(
    not cmake_available(), reason="CMake not installed"
)
requires_ninja = pytest.mark.skipif(
    not ninja_available(), reason="Ninja not installed"
)
requires_gcovr = pytest.mark.skipif(
    not gcovr_available(), reason="gcovr not installed"
)
requires_sanitizers = pytest.mark.skipif(
    not sanitizers_available(),
    reason="Sanitizer-capable compiler not available",
)

# Default mark for integration suites in this directory.
INTEGRATION_MARK = pytest.mark.integration

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
