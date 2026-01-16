"""Unit tests for run_cpp_coverage tool."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType
from typing import List

import pytest

RUN_CPP_COVERAGE_PATH = Path(__file__).resolve().parent.parent / "run_cpp_coverage.py"


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_cpp_coverage", RUN_CPP_COVERAGE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


@pytest.fixture(scope="module")
def coverage_module() -> ModuleType:
    return load_module()


GCOVR_OUTPUT = json.dumps(
    {
        "files": [
            {
                "filename": "src/example_lib.cpp",
                "line_covered": 45,
                "line_total": 52,
                "branch_covered": 12,
                "branch_total": 16,
                "function_covered": 8,
                "function_total": 10,
            },
            {
                "filename": "src/edge_cases.cpp",
                "line_covered": 34,
                "line_total": 50,
                "branch_covered": 5,
                "branch_total": 10,
                "function_covered": 3,
                "function_total": 5,
            },
        ],
        "line_covered": 79,
        "line_total": 102,
        "branch_covered": 17,
        "branch_total": 26,
        "function_covered": 11,
        "function_total": 15,
    }
)

LOW_COVERAGE_OUTPUT = json.dumps(
    {
        "files": [
            {
                "filename": "src/low.cpp",
                "line_covered": 40,
                "line_total": 100,
                "branch_covered": 10,
                "branch_total": 20,
                "function_covered": 2,
                "function_total": 5,
            }
        ],
        "line_covered": 40,
        "line_total": 100,
        "branch_covered": 10,
        "branch_total": 20,
        "function_covered": 2,
        "function_total": 5,
    }
)


class DummyProcess:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_coverage_metrics_percentages(coverage_module: ModuleType) -> None:
    metrics = coverage_module.CoverageMetrics(
        lines_covered=80,
        lines_total=100,
        branches_covered=15,
        branches_total=20,
        functions_covered=9,
        functions_total=10,
    )

    assert metrics.line_percent == 80.0
    assert metrics.branch_percent == 75.0
    assert metrics.function_percent == 90.0


def test_coverage_metrics_zero_totals(coverage_module: ModuleType) -> None:
    metrics = coverage_module.CoverageMetrics(0, 0, 0, 0, 0, 0)
    assert metrics.line_percent == 0.0
    assert metrics.branch_percent == 0.0
    assert metrics.function_percent == 0.0


def test_parse_gcovr_output_parses_totals_and_files(coverage_module: ModuleType) -> None:
    metrics = coverage_module.parse_gcovr_output(GCOVR_OUTPUT)

    assert "__total__" in metrics
    total = metrics["__total__"]
    assert total.lines_covered == 79
    assert total.lines_total == 102

    assert "src/example_lib.cpp" in metrics
    assert metrics["src/example_lib.cpp"].branches_total == 16


def test_parse_gcovr_output_invalid_json_raises(coverage_module: ModuleType) -> None:
    with pytest.raises(RuntimeError):
        coverage_module.parse_gcovr_output("{not-json}")


def test_validate_threshold_pass_and_fail(coverage_module: ModuleType) -> None:
    metrics = coverage_module.CoverageMetrics(90, 100, 0, 0, 0, 0)
    passed, msg = coverage_module.validate_threshold(metrics, 80.0)
    assert passed is True
    assert "meets" in msg

    metrics_low = coverage_module.CoverageMetrics(50, 100, 0, 0, 0, 0)
    passed_low, msg_low = coverage_module.validate_threshold(metrics_low, 80.0)
    assert passed_low is False
    assert "below" in msg_low


def test_format_summary_includes_threshold_and_files(coverage_module: ModuleType) -> None:
    metrics = coverage_module.CoverageMetrics(50, 100, 10, 20, 5, 10)
    summary = coverage_module.format_summary(
        metrics,
        threshold=80.0,
        files_below_threshold=[("src/low.cpp", 50.0)],
        duration=1.23,
        tool="gcov",
        validation_errors=[],
    )

    assert "C++ COVERAGE SUMMARY" in summary
    assert "Lines:     50/100 (50.0%)" in summary
    assert "Threshold: 80.0% lines (FAILED)" in summary
    assert "src/low.cpp" in summary
    assert "Duration" in summary
    assert "VALIDATION: PASSED" in summary


def test_run_coverage_success_with_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    commands: List[List[str]] = []

    def fake_run(cmd: List[str], **_: object) -> DummyProcess:  # type: ignore[override]
        commands.append(list(cmd))
        return DummyProcess(stdout=GCOVR_OUTPUT, stderr="", returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    html_dir = tmp_path / "html"
    exit_code, output = module.run_coverage(
        build_dir=build_dir,
        threshold=70.0,
        tool="llvm-cov",
        filter_path="src/",
        html_dir=html_dir,
        timeout=5,
        output_mode="json",
    )

    payload = json.loads(output)
    assert exit_code == 0
    assert payload["success"] is True
    assert payload["tool"] == "llvm-cov"
    assert payload["html_dir"] == str(html_dir)
    assert payload["metrics"]["__total__"]["line_percent"] == pytest.approx(77.45098, rel=1e-3)
    assert payload["metrics"]["src/example_lib.cpp"]["branch_percent"] == pytest.approx(75.0)
    assert any("--llvm-cov" in cmd for cmd in commands)
    assert any("--filter" in cmd for cmd in commands)
    assert any("--html" in cmd for cmd in commands)


def test_run_coverage_threshold_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    def fake_run(*_: object, **__: object) -> DummyProcess:  # type: ignore[override]
        return DummyProcess(stdout=LOW_COVERAGE_OUTPUT, stderr="", returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code, output = module.run_coverage(
        build_dir=build_dir,
        threshold=80.0,
        output_mode="json",
    )

    payload = json.loads(output)
    assert exit_code == 1
    assert payload["success"] is False
    assert payload["files_below_threshold"]
    assert any("below" in err or "code" in err for err in payload["validation_errors"])


def test_run_coverage_missing_build_dir_json(tmp_path: Path) -> None:
    module = load_module()
    missing_dir = tmp_path / "missing"

    exit_code, output = module.run_coverage(
        build_dir=missing_dir, threshold=50.0, output_mode="json"
    )

    payload = json.loads(output)
    assert exit_code == 1
    assert payload["success"] is False
    assert payload["threshold"] == 50.0
    assert payload["metrics"]["__total__"]["lines_total"] == 0
    assert any("does not exist" in err for err in payload["validation_errors"])


def test_run_coverage_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    def raise_timeout(*_: object, **__: object) -> None:  # type: ignore[override]
        raise subprocess.TimeoutExpired(cmd="gcovr", timeout=1)

    monkeypatch.setattr(module.subprocess, "run", raise_timeout)

    exit_code, output = module.run_coverage(build_dir=build_dir, timeout=1, output_mode="summary")

    assert exit_code == 1
    assert "timed out" in output.lower()


def test_run_coverage_missing_gcovr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    def raise_missing(*_: object, **__: object) -> None:  # type: ignore[override]
        raise FileNotFoundError()

    monkeypatch.setattr(module.subprocess, "run", raise_missing)

    exit_code, output = module.run_coverage(build_dir=build_dir)

    assert exit_code == 1
    assert "command not found" in output


def test_run_coverage_non_zero_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    def fake_run(*_: object, **__: object) -> DummyProcess:  # type: ignore[override]
        return DummyProcess(stdout=GCOVR_OUTPUT, stderr="error", returncode=2)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code, output = module.run_coverage(build_dir=build_dir, output_mode="json")

    payload = json.loads(output)
    assert exit_code == 1
    assert payload["success"] is False
    assert any("exited" in err for err in payload["validation_errors"])


def test_run_coverage_full_output_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    def fake_run(cmd: List[str], **_: object) -> DummyProcess:  # type: ignore[override]
        return DummyProcess(stdout=GCOVR_OUTPUT, stderr="stderr-text", returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code, output = module.run_coverage(build_dir=build_dir, output_mode="full")

    assert exit_code == 0
    assert "line_covered" in output
    assert "stderr-text" in output


def test_run_coverage_truncates_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    long_output = "\n".join(["line"] * 700)

    def fake_run(*_: object, **__: object) -> DummyProcess:  # type: ignore[override]
        return DummyProcess(stdout=long_output, stderr="", returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code, output = module.run_coverage(build_dir=build_dir, output_mode="json")

    payload = json.loads(output)
    assert exit_code == 1  # validation fails because totals are zero
    assert payload["truncated"] is True
    assert "truncated" in payload["truncation_notice"]


def test_main_wires_arguments(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    module = load_module()
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    captured_args = {}

    def fake_run_coverage(**kwargs: object) -> tuple[int, str]:
        captured_args.update(kwargs)
        return 0, "ok"

    monkeypatch.setattr(module, "run_coverage", fake_run_coverage)

    with pytest.raises(SystemExit) as excinfo:
        module.main(
            [
                "--build-dir",
                str(build_dir),
                "--threshold",
                "85",
                "--tool",
                "gcov",
                "--filter",
                "src/",
                "--html",
                str(tmp_path / "html"),
                "--timeout",
                "10",
                "--output",
                "json",
            ]
        )

    assert excinfo.value.code == 0

    captured = capsys.readouterr()
    assert "ok" in captured.out
    assert captured_args["threshold"] == 85.0
    assert captured_args["tool"] == "gcov"
    assert captured_args["filter_path"] == "src/"
    assert captured_args["output_mode"] == "json"
