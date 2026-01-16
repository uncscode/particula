"""Integration tests for run_ctest using the real example_cpp_dev project.

Prerequisites:
- CMake and CTest binaries installed and on PATH
- example_cpp_dev project present with CMakeLists.txt and CMakePresets.json

Usage:
    pytest -m integration .opencode/tool/tests/run_ctest_integration_test.py
    pytest -m "integration and requires_ctest" \
        .opencode/tool/tests/run_ctest_integration_test.py

The module builds the example project once per session, reuses the build
directory across tests, and verifies happy-path, filter, parallel, timeout,
and error conditions.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from types import ModuleType
from typing import Generator, TypedDict, cast

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_PROJECT = REPO_ROOT / "example_cpp_dev"
RUN_CTEST_PATH = Path(__file__).resolve().parent.parent / "run_ctest.py"
CMAKE_TIMEOUT = 600


class CTestMetrics(TypedDict):
    total: int
    failed: int
    passed: int
    ctest_missing: bool
    timeout: bool
    build_dir_error: bool
    exit_code: int


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def cmake_available() -> bool:
    return shutil.which("cmake") is not None


def ctest_available() -> bool:
    return shutil.which("ctest") is not None


PYTEST_MARKS = [
    pytest.mark.integration,
    pytest.mark.requires_cmake,
    pytest.mark.requires_ctest,
    pytest.mark.skipif(not cmake_available(), reason="CMake not installed"),
    pytest.mark.skipif(not ctest_available(), reason="CTest not installed"),
]
pytestmark = PYTEST_MARKS


@pytest.fixture(scope="session")
def run_ctest_module() -> ModuleType:
    return _load_module("run_ctest", RUN_CTEST_PATH)


@pytest.fixture(scope="session")
def example_project() -> Path:
    if not EXAMPLE_PROJECT.exists():
        pytest.skip("example_cpp_dev project not found (E12-F3 prerequisite)")
    if not (EXAMPLE_PROJECT / "CMakeLists.txt").exists():
        pytest.skip("example_cpp_dev/CMakeLists.txt missing")
    if not (EXAMPLE_PROJECT / "CMakePresets.json").exists():
        pytest.skip("CMakePresets.json missing for example_cpp_dev")
    return EXAMPLE_PROJECT


@pytest.fixture(scope="session")
def build_dir(example_project: Path) -> Generator[Path, None, None]:
    root = example_project / "build" / f"integration_run_ctest_{uuid.uuid4().hex[:8]}"
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="session")
def ensure_built(example_project: Path, build_dir: Path) -> Path:
    if not cmake_available():
        pytest.skip("CMake not installed")
    build_dir.parent.mkdir(parents=True, exist_ok=True)

    def run_command(cmd: list[str]) -> None:
        subprocess.run(
            cmd,
            cwd=str(example_project),
            check=True,
            capture_output=True,
            text=True,
            timeout=CMAKE_TIMEOUT,
        )

    if not (build_dir / "CTestTestfile.cmake").exists():
        try:
            run_command(["cmake", "-S", str(example_project), "-B", str(build_dir)])
            run_command(["cmake", "--build", str(build_dir)])
        except subprocess.CalledProcessError as exc:  # pragma: no cover - integration only
            output = exc.stderr or exc.stdout or str(exc)
            pytest.skip(f"Failed to configure/build example_cpp_dev: {output.strip()}")
        except subprocess.TimeoutExpired:  # pragma: no cover - integration only
            pytest.skip("cmake configure/build step timed out")
        except FileNotFoundError:  # pragma: no cover - integration only
            pytest.skip("cmake executable missing")

    ctest_file = build_dir / "CTestTestfile.cmake"
    if not ctest_file.exists():
        pytest.skip("CTestTestfile.cmake missing after build")
    return build_dir


def _parse_json(output: str) -> dict[str, object]:
    return json.loads(output)


# --- Integration scenarios --------------------------------------------------


def test_run_all_summary_output(run_ctest_module: ModuleType, ensure_built: Path) -> None:
    """Summary output runs every discovered test and reports validation success."""

    exit_code, output = run_ctest_module.run_ctest(
        build_dir=ensure_built,
        output_mode="summary",
    )

    assert exit_code == 0
    assert "CTEST SUMMARY" in output
    assert "VALIDATION: PASSED" in output
    assert "Tests Run:" in output


def test_full_output_includes_raw_ctest(run_ctest_module: ModuleType, ensure_built: Path) -> None:
    """Full output retains the raw CTest log and reports the 100% pass summary."""

    exit_code, output = run_ctest_module.run_ctest(
        build_dir=ensure_built,
        output_mode="full",
    )

    assert exit_code == 0
    assert "Example Library Test Suite" in output
    assert "100% tests passed" in output


def test_json_output_metrics_and_flags(run_ctest_module: ModuleType, ensure_built: Path) -> None:
    """JSON output surfaces structured metrics, flags, and validation metadata."""

    exit_code, output = run_ctest_module.run_ctest(
        build_dir=ensure_built,
        output_mode="json",
    )
    payload = _parse_json(output)
    metrics = cast(CTestMetrics, payload["metrics"])
    validation_errors = cast(list[str], payload["validation_errors"])

    assert exit_code == 0
    assert payload["success"] is True
    assert validation_errors == []
    assert metrics["total"] > 0
    assert metrics["failed"] == 0
    assert metrics["ctest_missing"] is False
    assert metrics["timeout"] is False
    assert metrics["build_dir_error"] is False


def test_include_filter_runs_known_test(run_ctest_module: ModuleType, ensure_built: Path) -> None:
    """Include filters still execute the expected example test."""

    exit_code, output = run_ctest_module.run_ctest(
        build_dir=ensure_built,
        include_filter="test_example_lib",
        output_mode="json",
    )
    payload = _parse_json(output)
    metrics = cast(CTestMetrics, payload["metrics"])

    assert exit_code == 0
    assert payload["success"] is True
    assert metrics["total"] == 1


def test_exclude_filter_non_matching_still_runs(
    run_ctest_module: ModuleType, ensure_built: Path
) -> None:
    """Exclude filters that match nothing do not prevent a successful run."""

    exit_code, output = run_ctest_module.run_ctest(
        build_dir=ensure_built,
        exclude_filter="__no_match__",
        output_mode="json",
    )
    payload = _parse_json(output)
    metrics = cast(CTestMetrics, payload["metrics"])

    assert exit_code == 0
    assert payload["success"] is True
    assert metrics["total"] >= 1


def test_parallel_execution(run_ctest_module: ModuleType, ensure_built: Path) -> None:
    """Running with multiple jobs still succeeds and reports the same metrics."""

    jobs = min(4, max(1, os.cpu_count() or 1))
    exit_code, output = run_ctest_module.run_ctest(
        build_dir=ensure_built,
        parallel=jobs,
        output_mode="json",
    )
    payload = _parse_json(output)
    metrics = cast(CTestMetrics, payload["metrics"])

    assert exit_code == 0
    assert payload["success"] is True
    assert metrics["total"] > 0


def test_min_test_count_validation(run_ctest_module: ModuleType, ensure_built: Path) -> None:
    """Requesting more tests than exist yields a validation failure."""

    exit_code, output = run_ctest_module.run_ctest(
        build_dir=ensure_built,
        min_test_count=5,
        output_mode="json",
    )
    payload = _parse_json(output)
    validation_errors = cast(list[str], payload["validation_errors"])

    assert exit_code == 1
    assert payload["success"] is False
    assert any("Expected at least 5 tests" in err for err in validation_errors)


def test_no_matching_tests_reports_zero(run_ctest_module: ModuleType, ensure_built: Path) -> None:
    """Include filters that match nothing report zero tests and fail validation."""

    exit_code, output = run_ctest_module.run_ctest(
        build_dir=ensure_built,
        include_filter="__no_such_test__",
        output_mode="json",
    )
    payload = _parse_json(output)
    metrics = cast(CTestMetrics, payload["metrics"])
    validation_errors = cast(list[str], payload["validation_errors"])

    assert exit_code == 1
    assert metrics["total"] == 0
    assert payload["success"] is False
    assert any("No tests were collected" in err for err in validation_errors)


def test_timeout_flagged(run_ctest_module: ModuleType, ensure_built: Path) -> None:
    """Timeout enforcement either flags the timeout or passes when the host is fast."""

    exit_code, output = run_ctest_module.run_ctest(
        build_dir=ensure_built,
        timeout=1,
        output_mode="json",
    )
    payload = _parse_json(output)
    metrics = cast(CTestMetrics, payload["metrics"])
    validation_errors = cast(list[str], payload["validation_errors"])

    if exit_code != 0:
        assert metrics["timeout"] is True
        assert any("timed out" in err.lower() for err in validation_errors)
    else:
        assert metrics["timeout"] is False
        assert payload["success"] is True


def test_invalid_build_directory_error(run_ctest_module: ModuleType, tmp_path: Path) -> None:
    """Pointing at a non-existent build directory reports build_dir_error."""

    missing = tmp_path / "missing_ctest_build"
    exit_code, output = run_ctest_module.run_ctest(
        build_dir=missing,
        output_mode="json",
    )
    payload = _parse_json(output)
    metrics = cast(CTestMetrics, payload["metrics"])
    validation_errors = cast(list[str], payload["validation_errors"])

    assert exit_code == 1
    assert metrics["build_dir_error"] is True
    assert payload["success"] is False
    assert any("CTestTestfile" in err for err in validation_errors)


def test_ctest_missing_flag(
    run_ctest_module: ModuleType, ensure_built: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate a missing ctest binary to ensure the flag is exposed."""

    def raise_missing(*_: object, **__: object) -> None:  # type: ignore[override]
        raise FileNotFoundError()

    monkeypatch.setattr(run_ctest_module.subprocess, "run", raise_missing)

    exit_code, output = run_ctest_module.run_ctest(
        build_dir=ensure_built,
        output_mode="json",
    )
    payload = _parse_json(output)
    metrics = cast(CTestMetrics, payload["metrics"])
    validation_errors = cast(list[str], payload["validation_errors"])

    assert exit_code == 1
    assert metrics["ctest_missing"] is True
    assert payload["success"] is False
    assert any("ctest command not found" in err.lower() for err in validation_errors)


# --- Additional coverage ----------------------------------------------------


def test_truncate_output_limits_size(run_ctest_module: ModuleType) -> None:
    long_line = "x" * 200
    payload = "\n".join(long_line for _ in range(run_ctest_module.OUTPUT_LINE_LIMIT + 10))

    truncated_output, truncated, notice = run_ctest_module._truncate_output(payload)

    assert truncated is True
    assert "truncated" in notice.lower()
    assert "..." in truncated_output


def test_parse_ctest_output_handles_invalid_duration(run_ctest_module: ModuleType) -> None:
    raw_output = (
        "50% tests passed, 1 tests failed out of 2\n"
        "Total Test time (real) = abc sec\n"
        "Test #1: sample_case ***Failed\n"
        "1 - integration_case (Failed)\n"
    )

    metrics = run_ctest_module.parse_ctest_output(raw_output)

    assert metrics["total"] == 2
    assert metrics["failed"] == 1
    assert metrics["passed"] == 1
    assert metrics["duration"] is None
    assert set(metrics["failed_tests"]) == {"sample_case", "integration_case"}


def test_validate_results_captures_all_flags(run_ctest_module: ModuleType) -> None:
    metrics = {
        "failed": 2,
        "total": 1,
        "timeout": True,
        "timeout_seconds": 7,
        "ctest_missing": True,
        "build_dir_error": True,
    }

    errors = run_ctest_module.validate_results(metrics, min_test_count=3)

    assert any("failed test" in err for err in errors)
    assert any("at least 3" in err for err in errors)
    assert any("timed out" in err for err in errors)
    assert any("ctest command not found" in err for err in errors)
    assert any("CTestTestfile" in err or "Build directory" in err for err in errors)


def test_format_summary_truncates_failed_tests(run_ctest_module: ModuleType) -> None:
    metrics = {
        "total": 12,
        "passed": 10,
        "failed": 2,
        "skipped": 0,
        "duration": 1.5,
        "failed_tests": [f"test_{i}" for i in range(15)],
    }

    summary = run_ctest_module.format_summary(metrics, ["some validation error"])

    assert "Failed Tests (15):" in summary
    assert "... and 5 more" in summary
    assert "VALIDATION: FAILED" in summary


def test_run_ctest_timeout_sets_flags(
    run_ctest_module: ModuleType, ensure_built: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_timeout(*_: object, **__: object) -> None:  # type: ignore[override]
        raise subprocess.TimeoutExpired(cmd="ctest", timeout=1)

    monkeypatch.setattr(run_ctest_module.subprocess, "run", raise_timeout)

    exit_code, output = run_ctest_module.run_ctest(
        build_dir=ensure_built,
        timeout=1,
        output_mode="json",
    )
    payload = _parse_json(output)
    metrics = cast(CTestMetrics, payload["metrics"])
    validation_errors = cast(list[str], payload["validation_errors"])

    assert exit_code == 1
    assert metrics["timeout"] is True
    assert metrics["exit_code"] == 1
    assert any("timed out" in err for err in validation_errors)


def test_run_ctest_full_output_truncates(
    run_ctest_module: ModuleType, ensure_built: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    long_output = (
        "100% tests passed, 0 tests failed out of 1\n"
        "Total Test time (real) = 1.00 sec\n"
        + "\n".join(f"line {i}" for i in range(run_ctest_module.OUTPUT_LINE_LIMIT + 50))
    )

    class FakeCompletedProcess:
        def __init__(self) -> None:
            self.stdout = long_output
            self.stderr = ""
            self.returncode = 0

    monkeypatch.setattr(run_ctest_module.subprocess, "run", lambda *_, **__: FakeCompletedProcess())

    exit_code, output = run_ctest_module.run_ctest(
        build_dir=ensure_built,
        output_mode="full",
    )

    assert exit_code == 0
    assert "Output truncated" in output


def test_parse_args_maps_cli_options(run_ctest_module: ModuleType, tmp_path: Path) -> None:
    args = run_ctest_module._parse_args(
        [
            "--build-dir",
            str(tmp_path),
            "-R",
            "foo",
            "-E",
            "bar",
            "-j",
            "2",
            "--timeout",
            "10",
            "--min-tests",
            "3",
            "--output",
            "json",
        ]
    )

    assert args.build_dir == tmp_path
    assert args.include == "foo"
    assert args.exclude == "bar"
    assert args.parallel == 2
    assert args.timeout == 10
    assert args.min_tests == 3
    assert args.output == "json"


def test_main_propagates_run_ctest_exit_code(
    run_ctest_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    called: dict[str, object] = {}

    def fake_run_ctest(**kwargs: object) -> tuple[int, str]:
        called.update({"build_dir": kwargs["build_dir"]})
        return 0, "ok"

    monkeypatch.setattr(run_ctest_module, "run_ctest", fake_run_ctest)

    with pytest.raises(SystemExit) as excinfo:
        run_ctest_module.main(["--build-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert excinfo.value.code == 0
    assert called["build_dir"] == tmp_path
    assert captured.out.strip() == "ok"
