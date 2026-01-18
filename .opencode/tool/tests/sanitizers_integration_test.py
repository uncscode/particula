from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import (
    INTEGRATION_MARK,
    example_cpp_dir,
    requires_cmake,
    requires_ninja,
    requires_sanitizers,
)

RUN_SANITIZERS_PATH = (
    Path(__file__).resolve().parent.parent / "run_sanitizers.py"
)
EXECUTABLE_NAME = "test_sanitizer_demo"

pytestmark = [
    INTEGRATION_MARK,
    requires_cmake,
    requires_ninja,
    requires_sanitizers,
]


def _configure_and_build(preset: str) -> Path:
    """Configure and build the example project for a given preset."""
    subprocess.run(
        ["cmake", "--preset", preset],
        cwd=str(example_cpp_dir),
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    subprocess.run(
        ["cmake", "--build", "--preset", preset],
        cwd=str(example_cpp_dir),
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return example_cpp_dir / "build" / preset


def _run_sanitizer(
    sanitizer: str,
    build_dir: Path,
    scenario: str,
    *,
    output_mode: str = "summary",
    suppressions: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke run_sanitizers.py for the demo executable."""
    executable = build_dir / EXECUTABLE_NAME
    cmd = [
        sys.executable,
        str(RUN_SANITIZERS_PATH),
        "--sanitizer",
        sanitizer,
        "--executable",
        str(executable),
        "--build-dir",
        str(build_dir),
        "--timeout",
        "60",
        "--output-mode",
        output_mode,
    ]
    if suppressions:
        cmd.extend(["--suppressions", str(suppressions)])
    cmd.append("--")
    cmd.append(scenario)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


@pytest.fixture(scope="session")
def asan_build_dir() -> Path:
    return _configure_and_build("asan")


@pytest.fixture(scope="session")
def tsan_build_dir() -> Path:
    return _configure_and_build("tsan")


@pytest.fixture(scope="session")
def ubsan_build_dir() -> Path:
    return _configure_and_build("ubsan")


def test_asan_detects_heap_overflow(asan_build_dir: Path) -> None:
    result = _run_sanitizer("asan", asan_build_dir, "asan", output_mode="json")
    payload = json.loads(result.stdout or "{}")

    assert result.returncode != 0
    assert payload.get("errors") not in (None, [])
    assert payload.get("truncated") is False


def test_tsan_detects_data_race(tsan_build_dir: Path) -> None:
    result = _run_sanitizer("tsan", tsan_build_dir, "tsan", output_mode="json")
    payload = json.loads(result.stdout or "{}")

    assert result.returncode != 0
    assert payload.get("errors") not in (None, [])


def test_ubsan_detects_undefined_behavior(ubsan_build_dir: Path) -> None:
    result = _run_sanitizer(
        "ubsan", ubsan_build_dir, "ubsan", output_mode="json"
    )
    payload = json.loads(result.stdout or "{}")

    assert result.returncode != 0
    assert payload.get("errors") not in (None, [])


def test_json_output_includes_errors_and_truncation_flag(
    asan_build_dir: Path,
) -> None:
    result = _run_sanitizer("asan", asan_build_dir, "asan", output_mode="json")
    payload = json.loads(result.stdout or "{}")

    assert result.returncode != 0
    assert payload.get("sanitizer") == "asan"
    assert isinstance(payload.get("errors"), list)
    assert payload.get("truncated") is False
    assert payload.get("exit_code") != 0


def test_suppressions_reduce_reported_errors(
    tmp_path: Path, asan_build_dir: Path
) -> None:
    suppression_file = tmp_path / "suppressions.txt"
    suppression_file.write_text("heap-buffer-overflow\n")

    baseline = _run_sanitizer(
        "asan", asan_build_dir, "asan", output_mode="json"
    )
    suppressed = _run_sanitizer(
        "asan",
        asan_build_dir,
        "asan",
        output_mode="json",
        suppressions=suppression_file,
    )

    baseline_payload = json.loads(baseline.stdout or "{}")
    suppressed_payload = json.loads(suppressed.stdout or "{}")

    assert baseline_payload.get("errors") not in (None, [])
    assert suppressed_payload.get("suppressed_count", 0) >= 1
    assert len(suppressed_payload.get("errors", [])) <= len(
        baseline_payload.get("errors", [])
    )
