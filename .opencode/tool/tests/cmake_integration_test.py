"""Integration tests for run_cmake and clear_build using example_cpp_dev.

These tests exercise the real tooling end-to-end against the sample CMake
project. They require:
- CMake 3.21+ installed
- Ninja installed (optional; Ninja-specific test skips otherwise)
- The example_cpp_dev project with CMakePresets.json available

Usage:
    pytest -m integration .opencode/tool/tests/cmake_integration_test.py
    pytest -m "integration and not requires_ninja" \\
        .opencode/tool/tests/cmake_integration_test.py

All tests create unique build directories under example_cpp_dev/build and clean
up after themselves even when assertions fail.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import uuid
from pathlib import Path
from types import ModuleType
from typing import Generator

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_PROJECT = REPO_ROOT / "example_cpp_dev"
RUN_CMAKE_PATH = Path(__file__).resolve().parent.parent / "run_cmake.py"
CLEAR_BUILD_PATH = Path(__file__).resolve().parent.parent / "clear_build.py"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def cmake_available() -> bool:
    return shutil.which("cmake") is not None


def ninja_available() -> bool:
    return shutil.which("ninja") is not None


PYTEST_MARKS = [
    pytest.mark.integration,
    pytest.mark.requires_cmake,
    pytest.mark.skipif(not cmake_available(), reason="CMake not installed"),
]
pytestmark = PYTEST_MARKS


@pytest.fixture(scope="session")
def run_cmake_module() -> ModuleType:
    return _load_module("run_cmake", RUN_CMAKE_PATH)


@pytest.fixture(scope="session")
def clear_build_module() -> ModuleType:
    return _load_module("clear_build", CLEAR_BUILD_PATH)


@pytest.fixture(scope="session")
def example_project() -> Path:
    if not EXAMPLE_PROJECT.exists():
        pytest.skip("example_cpp_dev project not found (E12-F1 prerequisite)")
    if not (EXAMPLE_PROJECT / "CMakeLists.txt").exists():
        pytest.skip("example_cpp_dev/CMakeLists.txt missing")
    if not (EXAMPLE_PROJECT / "CMakePresets.json").exists():
        pytest.skip("CMakePresets.json missing for example_cpp_dev")
    return EXAMPLE_PROJECT


@pytest.fixture()
def build_dir(example_project: Path) -> Generator[Path, None, None]:
    root = example_project / "build" / f"integration_{uuid.uuid4().hex[:8]}"
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)


@pytest.fixture()
def in_example_project(monkeypatch: pytest.MonkeyPatch, example_project: Path) -> None:
    """Run commands from the example project directory for preset invocations."""

    monkeypatch.chdir(example_project)


# --- run_cmake integrations ---


def test_configure_with_preset_default_summary(
    run_cmake_module: ModuleType, example_project: Path, in_example_project: None
) -> None:
    expected_build_dir = example_project / "build" / "default"
    if expected_build_dir.exists():
        shutil.rmtree(expected_build_dir, ignore_errors=True)

    try:
        exit_code, output = run_cmake_module.run_cmake(
            source_dir=str(example_project),
            build_dir=str(expected_build_dir),
            preset="default",
            output_mode="summary",
        )

        assert exit_code == 0
        assert expected_build_dir.exists()
        assert "CMAKE SUMMARY" in output
        assert "=" * 60 in output
        assert "VALIDATION: PASSED" in output
        assert str(expected_build_dir) in output
    finally:
        if expected_build_dir.exists():
            shutil.rmtree(expected_build_dir, ignore_errors=True)


@pytest.mark.requires_ninja
@pytest.mark.skipif(not ninja_available(), reason="Ninja not installed")
def test_configure_with_ninja_generator(
    run_cmake_module: ModuleType, example_project: Path, build_dir: Path
) -> None:
    # Ninja fallback is allowed by the tool; we assert success
    # and mention of Ninja or success summary.
    exit_code, output = run_cmake_module.run_cmake(
        source_dir=str(example_project),
        build_dir=str(build_dir),
        ninja=True,
        output_mode="summary",
    )

    assert exit_code == 0
    assert build_dir.exists()
    assert "ninja" in output.lower() or "VALIDATION: PASSED" in output


def test_summary_output_format(
    run_cmake_module: ModuleType, example_project: Path, build_dir: Path
) -> None:
    exit_code, output = run_cmake_module.run_cmake(
        source_dir=str(example_project),
        build_dir=str(build_dir),
        output_mode="summary",
    )

    assert exit_code == 0
    for required in [
        "Configuration:",
        "Source Dir:",
        "Build Dir:",
        "Warnings:",
        "Errors:",
        "VALIDATION:",
    ]:
        assert required in output


def test_json_output_contains_metrics(
    run_cmake_module: ModuleType, example_project: Path, build_dir: Path
) -> None:
    exit_code, output = run_cmake_module.run_cmake(
        source_dir=str(example_project),
        build_dir=str(build_dir),
        output_mode="json",
    )

    data = json.loads(output)
    metrics = data.get("metrics", {})
    assert metrics.get("exit_code") == exit_code
    assert "success" in metrics
    assert "timeout" in metrics
    assert "ninja_fallback" in metrics
    assert isinstance(metrics.get("targets", []), list)
    assert data.get("output") is not None
    assert str(build_dir) in json.dumps(data)
    assert str(example_project) in json.dumps(data)


@pytest.mark.timeout(120)
def test_timeout_flagged(
    run_cmake_module: ModuleType,
    example_project: Path,
    build_dir: Path,
    in_example_project: None,
) -> None:
    exit_code, output = run_cmake_module.run_cmake(
        source_dir=str(example_project),
        build_dir=str(build_dir),
        preset="default",
        timeout=1,
        output_mode="json",
    )

    payload = json.loads(output)
    metrics = payload.get("metrics", {})
    if exit_code != 0:
        assert metrics.get("timeout") is True or "timeout" in payload.get("output", "").lower()
    else:
        # Fast hosts may finish within 1s; timeout should be explicitly false in that case.
        assert metrics.get("timeout") is False


# --- clear_build integrations ---


def test_clear_build_dry_run_preserves_dir(
    clear_build_module: ModuleType, example_project: Path, build_dir: Path
) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "test.txt").write_text("content")

    exit_code, output = clear_build_module.clear_build(
        build_dir=str(build_dir),
        dry_run=True,
        project_root=example_project,
    )

    assert exit_code == 0
    assert "DRY RUN" in output
    assert "VALIDATION: PASSED" in output
    assert build_dir.exists()


def test_clear_build_force_deletes(
    clear_build_module: ModuleType, example_project: Path, build_dir: Path
) -> None:
    nested = build_dir / "subdir"
    nested.mkdir(parents=True, exist_ok=True)
    (build_dir / "CMakeCache.txt").write_text("cache")
    (nested / "file.o").write_bytes(b"data")

    exit_code, output = clear_build_module.clear_build(
        build_dir=str(build_dir),
        force=True,
        project_root=example_project,
    )

    assert exit_code == 0
    assert "Deleted" in output
    assert not build_dir.exists()


def test_clear_build_nonexistent_directory(
    clear_build_module: ModuleType, example_project: Path
) -> None:
    missing = example_project / "build" / f"missing_{uuid.uuid4().hex[:6]}"

    exit_code, output = clear_build_module.clear_build(
        build_dir=str(missing),
        project_root=example_project,
    )

    assert exit_code == 0
    assert "does not exist" in output.lower()


def test_clear_build_rejects_outside_project(
    clear_build_module: ModuleType, example_project: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside_build"
    outside.mkdir(parents=True, exist_ok=True)
    (outside / "file.bin").write_bytes(b"data")

    exit_code, output = clear_build_module.clear_build(
        build_dir=str(outside),
        project_root=example_project,
    )

    assert exit_code == 1
    assert "FAILED" in output
    assert outside.exists()


# --- combined workflow ---


def test_configure_clear_reconfigure(
    run_cmake_module: ModuleType,
    clear_build_module: ModuleType,
    example_project: Path,
    build_dir: Path,
) -> None:
    exit_code1, output1 = run_cmake_module.run_cmake(
        source_dir=str(example_project),
        build_dir=str(build_dir),
        output_mode="summary",
    )
    assert exit_code1 == 0, output1
    assert build_dir.exists()

    exit_code2, output2 = clear_build_module.clear_build(
        build_dir=str(build_dir),
        force=True,
        project_root=example_project,
    )
    assert exit_code2 == 0, output2
    assert not build_dir.exists()

    exit_code3, output3 = run_cmake_module.run_cmake(
        source_dir=str(example_project),
        build_dir=str(build_dir),
        output_mode="summary",
    )
    assert exit_code3 == 0, output3
    assert build_dir.exists()
