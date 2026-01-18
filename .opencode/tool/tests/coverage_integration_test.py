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
    requires_gcovr,
    requires_ninja,
)

RUN_COVERAGE_PATH = (
    Path(__file__).resolve().parent.parent / "run_cpp_coverage.py"
)

pytestmark = [INTEGRATION_MARK, requires_cmake, requires_ninja, requires_gcovr]


def _configure_and_build() -> Path:
    """Configure, build, and run tests for coverage instrumentation."""
    subprocess.run(
        ["cmake", "--preset", "coverage"],
        cwd=str(example_cpp_dir),
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    subprocess.run(
        ["cmake", "--build", "--preset", "coverage"],
        cwd=str(example_cpp_dir),
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    build_dir = example_cpp_dir / "build" / "coverage"
    subprocess.run(
        ["ctest", "--test-dir", str(build_dir)],
        cwd=str(example_cpp_dir),
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return build_dir


def _run_coverage(
    build_dir: Path,
    *,
    threshold: float | None = None,
    html_dir: Path | None = None,
    filter_path: str | None = None,
    output: str = "summary",
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(RUN_COVERAGE_PATH),
        "--build-dir",
        str(build_dir),
        "--output",
        output,
    ]
    if threshold is not None:
        cmd.extend(["--threshold", str(threshold)])
    if html_dir is not None:
        cmd.extend(["--html", str(html_dir)])
    if filter_path is not None:
        cmd.extend(["--filter", filter_path])

    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


@pytest.fixture(scope="session")
def coverage_build_dir() -> Path:
    return _configure_and_build()


def test_threshold_passes_with_low_requirement(
    coverage_build_dir: Path,
) -> None:
    result = _run_coverage(
        coverage_build_dir,
        threshold=0,
        filter_path=str(example_cpp_dir),
        output="json",
    )
    payload = json.loads(result.stdout or "{}")

    assert result.returncode == 0
    assert payload.get("success") is True
    assert payload.get("validation_errors") == []


def test_threshold_failure_reports_files(coverage_build_dir: Path) -> None:
    result = _run_coverage(
        coverage_build_dir,
        threshold=95,
        filter_path=str(example_cpp_dir),
        output="json",
    )
    payload = json.loads(result.stdout or "{}")

    assert result.returncode != 0
    assert payload.get("success") is False
    assert payload.get("validation_errors") not in (None, [])


def test_html_report_is_generated(
    tmp_path: Path, coverage_build_dir: Path
) -> None:
    html_dir = tmp_path / "coverage_html"
    result = _run_coverage(
        coverage_build_dir, html_dir=html_dir, output="summary"
    )

    assert result.returncode == 0
    assert (html_dir / "index.html").exists()


def test_filter_option_limits_scope(coverage_build_dir: Path) -> None:
    result = _run_coverage(
        coverage_build_dir, filter_path="src/", output="summary"
    )

    assert result.returncode == 0
    assert len((result.stdout or "").splitlines()) < 100


def test_json_output_contains_metrics_and_truncation_flag(
    coverage_build_dir: Path,
) -> None:
    result = _run_coverage(coverage_build_dir, output="json")
    payload = json.loads(result.stdout or "{}")

    assert "metrics" in payload
    assert payload.get("truncated") is False
    assert payload.get("success") in {True, False}
