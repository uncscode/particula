"""Unit tests for run_pytest tool parsing and validation logic."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType
from typing import List

import pytest

RUN_PYTEST_PATH = Path(__file__).resolve().parent.parent / "run_pytest.py"


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_pytest", RUN_PYTEST_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


@pytest.fixture(scope="module")
def run_pytest_module() -> ModuleType:
    return load_module()


class DummyProcess:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_parse_pytest_output_extracts_counts(run_pytest_module: ModuleType) -> None:
    output = """===== 3 passed, 1 skipped in 0.50s =====\nTOTAL        4     0    75%"""
    metrics = run_pytest_module.parse_pytest_output(output)

    assert metrics["passed"] == 3
    assert metrics["skipped"] == 1
    assert metrics["duration"] == 0.50
    assert metrics["coverage_pct"] == 75


def test_validate_results_flags_all_issues(run_pytest_module: ModuleType) -> None:
    metrics = {
        "passed": 0,
        "failed": 1,
        "errors": 1,
        "skipped": 0,
        "warnings": 0,
        "total": 0,
        "duration": None,
        "coverage_pct": 50,
        "has_failures": True,
        "has_errors": True,
        "failed_tests": [],
        "error_tests": [],
    }

    errors = run_pytest_module.validate_results(metrics, min_test_count=1, coverage_threshold=80)

    assert "Found 1 failed test(s)" in errors
    assert "Found 1 test error(s)" in errors
    assert any("Expected at least" in err for err in errors)
    assert any("Coverage 50% is below threshold of 80%" in err for err in errors)


def test_format_summary_reports_validation_and_coverage(run_pytest_module: ModuleType) -> None:
    metrics = {
        "passed": 2,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "warnings": 0,
        "total": 2,
        "duration": 1.23,
        "coverage_pct": 90,
        "failed_tests": [],
        "error_tests": [],
    }
    validation_errors = ["sample validation error"]

    summary = run_pytest_module.format_summary(metrics, validation_errors, coverage_threshold=80)

    assert "Coverage: 90% (threshold: 80% PASSED)" in summary
    assert "VALIDATION: FAILED" in summary
    assert "sample validation error" in summary


def test_run_pytest_successful_run(
    monkeypatch: pytest.MonkeyPatch, run_pytest_module: ModuleType
) -> None:
    commands: List[List[str]] = []

    def fake_run(cmd: List[str], **_: object) -> DummyProcess:  # type: ignore[override]
        commands.append(cmd)
        stdout = "===== 2 passed in 0.10s =====\nTOTAL        2     0   100%"
        return DummyProcess(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(run_pytest_module.subprocess, "run", fake_run)

    exit_code, output = run_pytest_module.run_pytest(
        ["tests"],
        output_mode="summary",
        min_test_count=1,
        coverage=True,
        coverage_threshold=80,
        fail_fast=True,
    )

    assert exit_code == 0
    assert any("--cov" in arg for arg in commands[-1])
    assert "VALIDATION: PASSED" in output


def test_run_pytest_enforces_coverage_threshold(
    monkeypatch: pytest.MonkeyPatch, run_pytest_module: ModuleType
) -> None:
    def fake_run(cmd: List[str], **_: object) -> DummyProcess:  # type: ignore[override]
        stdout = "===== 1 passed in 0.05s =====\nTOTAL        1     0    50%"
        return DummyProcess(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(run_pytest_module.subprocess, "run", fake_run)

    exit_code, output = run_pytest_module.run_pytest(
        ["tests"], coverage=True, coverage_threshold=80, output_mode="summary"
    )

    assert exit_code == 1
    assert "Coverage: 50% (threshold: 80% FAILED)" in output


def test_run_pytest_handles_timeout(
    monkeypatch: pytest.MonkeyPatch, run_pytest_module: ModuleType
) -> None:
    def raise_timeout(*_: object, **__: object) -> None:  # type: ignore[override]
        raise subprocess.TimeoutExpired(cmd="pytest", timeout=1)

    monkeypatch.setattr(run_pytest_module.subprocess, "run", raise_timeout)

    exit_code, output = run_pytest_module.run_pytest(["tests"], timeout=1)

    assert exit_code == 1
    assert "pytest timed out" in output


def test_run_pytest_handles_missing_command(
    monkeypatch: pytest.MonkeyPatch, run_pytest_module: ModuleType
) -> None:
    def raise_filenotfound(*_: object, **__: object) -> None:  # type: ignore[override]
        raise FileNotFoundError

    monkeypatch.setattr(run_pytest_module.subprocess, "run", raise_filenotfound)

    exit_code, output = run_pytest_module.run_pytest(["tests"], coverage=False)

    assert exit_code == 1
    assert "pytest command not found" in output
