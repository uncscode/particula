"""Unit tests for run_pytest tool parsing and validation logic."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, List, Mapping, cast

import pytest

RUN_PYTEST_PATH = Path(__file__).resolve().parent.parent / "run_pytest.py"
CONFTEXT_PATH = Path(__file__).resolve().parent / "conftest.py"

DURATIONS_BLOCK = """
============================= slowest 3 durations =============================
0.12s call tests/test_alpha.py::test_one
0.08s setup tests/test_alpha.py::test_one
0.05s teardown tests/test_alpha.py::test_one
"""

DURATIONS_BLOCK_NO_ENTRIES = """
============================= slowest durations =============================

"""

COVERAGE_BLOCK = """
----------- coverage: platform linux, python 3.12 -----------
Name                                   Stmts   Miss  Cover   Missing
adw/utils/progress.py                     28      0   100%
adw/workflows/operations/status.py       452    114    75%   73, 77-78
TOTAL                                   480    114    76%
"""

COVERAGE_BLOCK_NO_MISSING = """
----------- coverage: platform linux, python 3.12 -----------
Name                                   Stmts   Miss  Cover
adw/core/constants.py                     28      0   100%
TOTAL                                     28      0   100%
"""


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_pytest", RUN_PYTEST_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def load_conftest_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tool_conftest", CONFTEXT_PATH)
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


def test_parse_pytest_output_includes_coverage_files(run_pytest_module: ModuleType) -> None:
    output = "===== 3 passed in 0.10s =====\n" + COVERAGE_BLOCK
    metrics = run_pytest_module.parse_pytest_output(output)

    assert metrics["coverage_pct"] == 76
    assert metrics["coverage_files"] == [
        {
            "file": "adw/workflows/operations/status.py",
            "statements": 452,
            "missing": 114,
            "coverage_pct": 75,
            "missing_lines": "73, 77-78",
        },
        {
            "file": "adw/utils/progress.py",
            "statements": 28,
            "missing": 0,
            "coverage_pct": 100,
            "missing_lines": "",
        },
    ]


def test_parse_pytest_output_coverage_files_without_missing_column(
    run_pytest_module: ModuleType,
) -> None:
    output = "===== 1 passed in 0.10s =====\n" + COVERAGE_BLOCK_NO_MISSING
    metrics = run_pytest_module.parse_pytest_output(output)

    assert metrics["coverage_pct"] == 100
    assert metrics["coverage_files"] == [
        {
            "file": "adw/core/constants.py",
            "statements": 28,
            "missing": 0,
            "coverage_pct": 100,
            "missing_lines": "",
        }
    ]


def test_format_summary_includes_coverage_files_section(run_pytest_module: ModuleType) -> None:
    metrics = {
        "passed": 2,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "warnings": 0,
        "total": 2,
        "duration": 1.23,
        "coverage_pct": 80,
        "failed_tests": [],
        "error_tests": [],
        "coverage_files": [
            {
                "file": "adw/workflows/operations/status.py",
                "statements": 452,
                "missing": 114,
                "coverage_pct": 75,
                "missing_lines": "73, 77-78",
            },
            {
                "file": "adw/utils/progress.py",
                "statements": 28,
                "missing": 0,
                "coverage_pct": 100,
                "missing_lines": "",
            },
        ],
    }

    summary = run_pytest_module.format_summary(metrics, [], coverage_threshold=80)

    assert "Coverage by File:" in summary
    assert "status.py" in summary
    assert "missing: 73, 77-78" in summary


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


def test_run_pytest_coverage_source_multiple_modules(
    monkeypatch: pytest.MonkeyPatch, run_pytest_module: ModuleType
) -> None:
    commands: List[List[str]] = []

    def fake_run(cmd: List[str], **_: object) -> DummyProcess:  # type: ignore[override]
        commands.append(cmd)
        stdout = "===== 1 passed in 0.10s =====\nTOTAL        1     0   100%"
        return DummyProcess(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(run_pytest_module.subprocess, "run", fake_run)

    exit_code, output = run_pytest_module.run_pytest(
        ["tests"],
        output_mode="summary",
        coverage=True,
        coverage_source="adw.core,adw.utils",
        coverage_threshold=80,
    )

    assert exit_code == 0
    assert "VALIDATION: PASSED" in output
    cmd = commands[-1]
    assert "--cov=adw.core" in cmd
    assert "--cov=adw.utils" in cmd


def test_run_pytest_coverage_source_whitespace_split(
    monkeypatch: pytest.MonkeyPatch, run_pytest_module: ModuleType
) -> None:
    commands: List[List[str]] = []

    def fake_run(cmd: List[str], **_: object) -> DummyProcess:  # type: ignore[override]
        commands.append(cmd)
        stdout = "===== 1 passed in 0.10s =====\nTOTAL        1     0   100%"
        return DummyProcess(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(run_pytest_module.subprocess, "run", fake_run)

    exit_code, output = run_pytest_module.run_pytest(
        ["tests"],
        output_mode="summary",
        coverage=True,
        coverage_source="adw.core, adw.utils",
    )

    assert exit_code == 0
    assert "VALIDATION: PASSED" in output
    cmd = commands[-1]
    assert "--cov=adw.core" in cmd
    assert "--cov=adw.utils" in cmd


def test_run_pytest_enforces_coverage_threshold(
    monkeypatch: pytest.MonkeyPatch, run_pytest_module: ModuleType
) -> None:
    def fake_run(cmd: List[str], **_: object) -> DummyProcess:  # type: ignore[override]
        stdout = "===== 1 passed in 0.05s =====\nTOTAL        1     0    50%"
        return DummyProcess(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(run_pytest_module.subprocess, "run", fake_run)

    exit_code, output = run_pytest_module.run_pytest(
        ["tests"],
        coverage=True,
        coverage_source="adw",
        coverage_threshold=80,
        output_mode="summary",
    )

    assert exit_code == 1
    assert "Coverage: 50% (threshold: 80% FAILED)" in output


def test_run_pytest_coverage_source_all_uses_default(
    monkeypatch: pytest.MonkeyPatch, run_pytest_module: ModuleType
) -> None:
    commands: List[List[str]] = []

    def fake_run(cmd: List[str], **_: object) -> DummyProcess:  # type: ignore[override]
        commands.append(cmd)
        stdout = "===== 1 passed in 0.05s =====\nTOTAL        1     0   100%"
        return DummyProcess(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(run_pytest_module.subprocess, "run", fake_run)

    exit_code, output = run_pytest_module.run_pytest(
        ["tests"],
        coverage=True,
        coverage_source="all",
        coverage_threshold=80,
        output_mode="summary",
    )

    assert exit_code == 0
    assert "VALIDATION: PASSED" in output
    cmd = commands[-1]
    assert "--cov=all" not in cmd
    assert "--cov" in cmd


def test_run_pytest_coverage_source_all_overrides_other_sources(
    monkeypatch: pytest.MonkeyPatch, run_pytest_module: ModuleType
) -> None:
    commands: List[List[str]] = []

    def fake_run(cmd: List[str], **_: object) -> DummyProcess:  # type: ignore[override]
        commands.append(cmd)
        stdout = "===== 1 passed in 0.05s =====\nTOTAL        1     0   100%"
        return DummyProcess(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(run_pytest_module.subprocess, "run", fake_run)

    exit_code, output = run_pytest_module.run_pytest(
        ["tests"],
        coverage=True,
        coverage_source=["adw", "all"],
        output_mode="summary",
    )

    assert exit_code == 0
    assert "VALIDATION: PASSED" in output
    cmd = commands[-1]
    assert "--cov=adw" not in cmd
    assert "--cov" in cmd


def test_run_pytest_coverage_source_none_or_empty_uses_default(
    monkeypatch: pytest.MonkeyPatch, run_pytest_module: ModuleType
) -> None:
    commands: List[List[str]] = []

    def fake_run(cmd: List[str], **_: object) -> DummyProcess:  # type: ignore[override]
        commands.append(cmd)
        stdout = "===== 1 passed in 0.05s =====\nTOTAL        1     0   100%"
        return DummyProcess(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(run_pytest_module.subprocess, "run", fake_run)

    exit_code, output = run_pytest_module.run_pytest(
        ["tests"],
        coverage=True,
        coverage_source="",
        output_mode="summary",
    )

    assert exit_code == 0
    assert "VALIDATION: PASSED" in output
    cmd = commands[-1]
    assert "--cov" in cmd
    assert "--cov=" not in cmd


def test_run_pytest_allows_override_ini(
    monkeypatch: pytest.MonkeyPatch, run_pytest_module: ModuleType
) -> None:
    commands: List[List[str]] = []

    def fake_run(cmd: List[str], **_: object) -> DummyProcess:  # type: ignore[override]
        commands.append(cmd)
        stdout = "===== 1 passed in 0.01s =====\nTOTAL        1     0   100%"
        return DummyProcess(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(run_pytest_module.subprocess, "run", fake_run)

    exit_code, _ = run_pytest_module.run_pytest(
        ["tests"],
        coverage=True,
        coverage_source="tool",
        override_ini=["addopts="],
    )

    assert exit_code == 0
    assert "--override-ini=addopts=" in commands[-1]


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


def test_parse_pytest_output_durations_multiple_entries(run_pytest_module: ModuleType) -> None:
    output = """===== 1 passed in 0.10s =====\n""" + DURATIONS_BLOCK
    metrics = run_pytest_module.parse_pytest_output(output)

    assert metrics["durations"] == [
        {
            "duration": 0.12,
            "phase": "call",
            "test": "tests/test_alpha.py::test_one",
        },
        {
            "duration": 0.08,
            "phase": "setup",
            "test": "tests/test_alpha.py::test_one",
        },
        {
            "duration": 0.05,
            "phase": "teardown",
            "test": "tests/test_alpha.py::test_one",
        },
    ]


def test_parse_pytest_output_durations_empty_block(run_pytest_module: ModuleType) -> None:
    output = """===== 1 passed in 0.10s =====\n""" + DURATIONS_BLOCK_NO_ENTRIES
    metrics = run_pytest_module.parse_pytest_output(output)

    assert metrics["durations"] == []


def test_parse_pytest_output_durations_header_variants(run_pytest_module: ModuleType) -> None:
    output = (
        "===== 1 passed in 0.10s =====\n"
        "============================= slowest durations =============================\n"
        "0.20s call tests/test_alpha.py::test_two\n"
    )
    metrics = run_pytest_module.parse_pytest_output(output)

    assert metrics["durations"] == [
        {
            "duration": 0.20,
            "phase": "call",
            "test": "tests/test_alpha.py::test_two",
        }
    ]


def test_parse_pytest_output_durations_hidden_line_ignored(
    run_pytest_module: ModuleType,
) -> None:
    output = (
        "===== 1 passed in 0.10s =====\n"
        "============================= slowest 2 durations =============================\n"
        "(1 durations < 0.005s hidden. Use -vv to show these durations.)\n"
        "0.30s call tests/test_alpha.py::test_three\n"
    )
    metrics = run_pytest_module.parse_pytest_output(output)

    assert metrics["durations"] == [
        {
            "duration": 0.30,
            "phase": "call",
            "test": "tests/test_alpha.py::test_three",
        }
    ]


def test_parse_pytest_output_no_durations(run_pytest_module: ModuleType) -> None:
    output = """===== 1 passed in 0.10s =====\n"""
    metrics = run_pytest_module.parse_pytest_output(output)

    assert "durations" not in metrics


def test_format_summary_includes_slowest_tests(run_pytest_module: ModuleType) -> None:
    metrics = {
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "warnings": 0,
        "total": 1,
        "duration": 0.1,
        "coverage_pct": None,
        "failed_tests": [],
        "error_tests": [],
        "durations": [
            {
                "duration": 0.5,
                "phase": "call",
                "test": "tests/test_alpha.py::test_four",
            }
        ],
    }

    summary = run_pytest_module.format_summary(metrics, [], coverage_threshold=None)

    assert "Slowest Tests:" in summary
    assert "0.50s" in summary
    assert "tests/test_alpha.py::test_four" in summary


def test_format_summary_omits_slowest_tests_when_empty(run_pytest_module: ModuleType) -> None:
    metrics = {
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "warnings": 0,
        "total": 1,
        "duration": 0.1,
        "coverage_pct": None,
        "failed_tests": [],
        "error_tests": [],
        "durations": [],
    }

    summary = run_pytest_module.format_summary(metrics, [], coverage_threshold=None)

    assert "Slowest Tests" not in summary


def test_format_summary_slowest_tests_cap(run_pytest_module: ModuleType) -> None:
    metrics = {
        "passed": 31,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "warnings": 0,
        "total": 31,
        "duration": 1.0,
        "coverage_pct": None,
        "failed_tests": [],
        "error_tests": [],
        "durations": [
            {
                "duration": 1.0,
                "phase": "call",
                "test": f"tests/test_alpha.py::test_{idx}",
            }
            for idx in range(31)
        ],
    }

    summary = run_pytest_module.format_summary(metrics, [], coverage_threshold=None)

    assert summary.count("tests/test_alpha.py::test_") == 30
    assert "... and 1 more" in summary


def test_run_pytest_full_mode_no_truncation_under_limit(
    monkeypatch: pytest.MonkeyPatch, run_pytest_module: ModuleType
) -> None:
    stdout = "\n".join([f"test_line_{index}" for index in range(10)])
    stdout += "\n===== 1 passed in 0.10s ====="
    stdout += f"\n{DURATIONS_BLOCK.strip()}"
    monkeypatch.setattr(
        run_pytest_module.subprocess,
        "run",
        lambda *_args, **_kwargs: DummyProcess(stdout=stdout, stderr="", returncode=0),
    )

    exit_code, output = run_pytest_module.run_pytest(["tests"], output_mode="full")

    assert exit_code == 0
    assert "test_line_0" in output
    assert "Slowest Tests" in output


def test_run_pytest_full_mode_truncation_preserves_sections(
    monkeypatch: pytest.MonkeyPatch, run_pytest_module: ModuleType
) -> None:
    failures_block = """
============================= FAILURES =============================
______________________________ test_fail ______________________________
E   AssertionError: boom
"""
    long_output = "\n".join([f"PASSED tests/test_alpha.py::test_{i}" for i in range(600)])
    stdout = (
        f"{long_output}\n{failures_block.strip()}\n"
        f"{DURATIONS_BLOCK.strip()}\n{COVERAGE_BLOCK.strip()}\n===== 1 failed in 1.00s ====="
    )
    monkeypatch.setattr(
        run_pytest_module.subprocess,
        "run",
        lambda *_args, **_kwargs: DummyProcess(stdout=stdout, stderr="", returncode=1),
    )

    exit_code, output = run_pytest_module.run_pytest(["tests"], output_mode="full")

    assert exit_code == 1
    assert "Output truncated" in output
    assert "FAILURES" in output
    assert "slowest" in output
    assert "coverage:" in output
    assert "PASSED tests/test_alpha.py::test_0" not in output


def test_run_pytest_json_includes_durations(
    monkeypatch: pytest.MonkeyPatch, run_pytest_module: ModuleType
) -> None:
    stdout = """===== 1 passed in 0.10s =====\n""" + DURATIONS_BLOCK
    monkeypatch.setattr(
        run_pytest_module.subprocess,
        "run",
        lambda *_args, **_kwargs: DummyProcess(stdout=stdout, stderr="", returncode=0),
    )

    exit_code, output = run_pytest_module.run_pytest(["tests"], output_mode="json")

    payload = run_pytest_module.json.loads(output)

    assert exit_code == 0
    assert payload["durations"] == payload["metrics"]["durations"]


def test_run_pytest_json_includes_coverage_files(
    monkeypatch: pytest.MonkeyPatch, run_pytest_module: ModuleType
) -> None:
    stdout = """===== 1 passed in 0.10s =====\n""" + COVERAGE_BLOCK
    monkeypatch.setattr(
        run_pytest_module.subprocess,
        "run",
        lambda *_args, **_kwargs: DummyProcess(stdout=stdout, stderr="", returncode=0),
    )

    exit_code, output = run_pytest_module.run_pytest(["tests"], output_mode="json")

    payload = run_pytest_module.json.loads(output)

    assert exit_code == 0
    metrics = payload["metrics"]
    assert metrics["coverage_files"][0]["file"] == "adw/workflows/operations/status.py"
    assert payload.get("coverage_files_total") is None
    assert payload.get("coverage_files_truncated") is None


def test_run_pytest_coverage_source_preserves_non_coverage_addopts(
    monkeypatch: pytest.MonkeyPatch,
    run_pytest_module: ModuleType,
) -> None:
    commands: List[List[str]] = []

    def fake_run(cmd: List[str], **_: object) -> DummyProcess:  # type: ignore[override]
        commands.append(cmd)
        stdout = "===== 1 passed in 0.10s =====\nTOTAL        1     0   100%"
        return DummyProcess(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setenv("PYTEST_ADDOPTS", "--cov=adw --maxfail=2 --color=yes")
    monkeypatch.setattr(run_pytest_module.subprocess, "run", fake_run)

    exit_code, output = run_pytest_module.run_pytest(
        ["tests"],
        output_mode="summary",
        coverage=True,
        coverage_source="adw",
        coverage_threshold=80,
    )

    assert exit_code == 0
    assert "VALIDATION: PASSED" in output
    assert commands
    cmd = commands[-1]
    assert cmd.count("--cov=adw") == 1
    assert "--maxfail=2" in cmd
    assert "--color=yes" in cmd
    assert "--cov-report=term-missing" in cmd


def test_main_repeated_coverage_source_flags_normalize(
    monkeypatch: pytest.MonkeyPatch, run_pytest_module: ModuleType
) -> None:
    captured: dict[str, Any] = {}

    def fake_run_pytest(*args: object, **kwargs: object) -> tuple[int, str]:
        captured["coverage_source"] = kwargs.get("coverage_source")
        return 0, "ok"

    monkeypatch.setattr(run_pytest_module, "run_pytest", fake_run_pytest)
    monkeypatch.setattr(run_pytest_module, "print", lambda *_args, **_kwargs: None, raising=False)

    exit_code = run_pytest_module.main(
        [
            "--coverage-source=adw.core",
            "--coverage-source=adw.utils",
            "--coverage-files-only",
            ".opencode/tool/tests/run_pytest_test.py",
        ]
    )

    assert exit_code == 0
    assert captured["coverage_source"] == ["adw.core", "adw.utils"]


def test_run_pytest_ignores_addopts_without_coverage_source(
    monkeypatch: pytest.MonkeyPatch,
    run_pytest_module: ModuleType,
) -> None:
    commands: List[List[str]] = []

    def fake_run(cmd: List[str], **_: object) -> DummyProcess:  # type: ignore[override]
        commands.append(cmd)
        stdout = "===== 1 passed in 0.10s =====\nTOTAL        1     0   100%"
        return DummyProcess(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setenv("PYTEST_ADDOPTS", "--maxfail=1 --color=yes")
    monkeypatch.setattr(run_pytest_module.subprocess, "run", fake_run)

    exit_code, output = run_pytest_module.run_pytest(
        ["tests"],
        output_mode="summary",
        coverage=True,
        coverage_source=None,
        coverage_threshold=80,
    )

    assert exit_code == 0
    assert "VALIDATION: PASSED" in output
    assert commands
    cmd = commands[-1]
    assert "--cov" in cmd
    assert "--maxfail=1" not in cmd
    assert "--color=yes" not in cmd


def test_run_pytest_ignores_addopts_when_coverage_disabled(
    monkeypatch: pytest.MonkeyPatch,
    run_pytest_module: ModuleType,
) -> None:
    commands: List[List[str]] = []

    def fake_run(cmd: List[str], **_: object) -> DummyProcess:  # type: ignore[override]
        commands.append(cmd)
        stdout = "===== 1 passed in 0.10s =====\nTOTAL        1     0   100%"
        return DummyProcess(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setenv("PYTEST_ADDOPTS", "--maxfail=1 --color=yes")
    monkeypatch.setattr(run_pytest_module.subprocess, "run", fake_run)

    exit_code, output = run_pytest_module.run_pytest(
        ["tests"],
        output_mode="summary",
        coverage=False,
    )

    assert exit_code == 0
    assert "VALIDATION: PASSED" in output
    assert commands
    cmd = commands[-1]
    assert "--cov" not in cmd
    assert "--maxfail=1" not in cmd
    assert "--color=yes" not in cmd


def test_run_pytest_combines_addopts_with_override_ini(
    monkeypatch: pytest.MonkeyPatch,
    run_pytest_module: ModuleType,
) -> None:
    commands: List[List[str]] = []

    def fake_run(cmd: List[str], **_: object) -> DummyProcess:  # type: ignore[override]
        commands.append(cmd)
        stdout = "===== 1 passed in 0.10s =====\nTOTAL        1     0   100%"
        return DummyProcess(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setenv("PYTEST_ADDOPTS", "--maxfail=1 --color=yes")
    monkeypatch.setattr(run_pytest_module.subprocess, "run", fake_run)

    exit_code, output = run_pytest_module.run_pytest(
        ["tests"],
        output_mode="summary",
        coverage=True,
        coverage_source="adw",
        override_ini=["addopts=--capture=no"],
    )

    assert exit_code == 0
    assert "VALIDATION: PASSED" in output
    assert commands
    cmd = commands[-1]
    assert "--cov=adw" in cmd
    assert "--maxfail=1" in cmd
    assert "--color=yes" in cmd
    assert "--override-ini=addopts=--capture=no" in cmd


def test_run_pytest_uses_cwd_in_env_and_command(
    monkeypatch: pytest.MonkeyPatch,
    run_pytest_module: ModuleType,
    tmp_path: Path,
) -> None:
    captured_env: dict[str, str] = {}
    worktree = tmp_path / "worktree"
    worktree.mkdir()

    def fake_run(cmd: List[str], **kwargs: object) -> DummyProcess:  # type: ignore[override]
        env = cast(Mapping[str, str], kwargs.get("env", {}))
        captured_env.clear()
        captured_env.update(env)
        stdout = "===== 1 passed in 0.10s =====\nTOTAL        1     0   100%"
        return DummyProcess(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(run_pytest_module.subprocess, "run", fake_run)

    exit_code, output = run_pytest_module.run_pytest(
        ["tests"],
        output_mode="summary",
        cwd=str(worktree),
        coverage=False,
    )

    assert exit_code == 0
    assert "VALIDATION: PASSED" in output
    pythonpath = captured_env.get("PYTHONPATH", "")
    assert pythonpath.split(run_pytest_module.os.pathsep)[0] == str(worktree)


def test_run_pytest_uses_project_root_when_cwd_missing(
    monkeypatch: pytest.MonkeyPatch,
    run_pytest_module: ModuleType,
    tmp_path: Path,
) -> None:
    commands: List[List[str]] = []
    cwd_values: List[str] = []
    marker = tmp_path / "project"
    marker.mkdir()
    (marker / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")

    def fake_run(cmd: List[str], **kwargs: object) -> DummyProcess:  # type: ignore[override]
        commands.append(cmd)
        cwd_values.append(str(kwargs.get("cwd")))
        stdout = "===== 1 passed in 0.10s =====\nTOTAL        1     0   100%"
        return DummyProcess(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(run_pytest_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_pytest_module.Path,
        "cwd",
        lambda: marker,
    )

    exit_code, output = run_pytest_module.run_pytest(
        ["tests"],
        output_mode="summary",
        coverage=False,
    )

    assert exit_code == 0
    assert "VALIDATION: PASSED" in output
    assert commands
    assert cwd_values[-1] == str(marker)
    assert "--cov" not in commands[-1]


def test_run_pytest_addopts_uses_tool_coverage_when_defaults_set(
    monkeypatch: pytest.MonkeyPatch,
    run_pytest_module: ModuleType,
) -> None:
    commands: List[List[str]] = []

    def fake_run(cmd: List[str], **_: object) -> DummyProcess:  # type: ignore[override]
        commands.append(cmd)
        stdout = "===== 1 passed in 0.10s =====\nTOTAL        1     0   100%"
        return DummyProcess(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setenv("PYTEST_ADDOPTS", "--cov=adw")
    monkeypatch.setattr(run_pytest_module.subprocess, "run", fake_run)

    exit_code, output = run_pytest_module.run_pytest(
        [str(CONFTEXT_PATH)],
        output_mode="summary",
        coverage=True,
        coverage_source=".opencode/tool",
        coverage_threshold=80,
    )

    assert exit_code == 0
    assert "VALIDATION: PASSED" in output
    assert commands
    cmd = commands[-1]
    assert "--cov=.opencode/tool" in cmd
    assert "--cov=adw" not in cmd


def test_coverage_targets_from_tests_prefers_existing_modules(
    run_pytest_module: ModuleType,
) -> None:
    conftest = load_conftest_module()
    conftest_module = cast(Any, conftest)
    fake_tool_root = Path("/workspace/.opencode/tool")
    fake_paths = [
        fake_tool_root / "run_pytest_test.py",
        fake_tool_root / "run_pytest_integration_test.py",
        fake_tool_root / "run_missing_test.py",
        fake_tool_root / "not_python.txt",
    ]

    def fake_exists(path: Path) -> bool:
        return path.name in {"run_pytest.py", "run_pytest_integration.py"}

    original_exists = Path.exists

    try:
        Path.exists = fake_exists  # type: ignore[assignment]
        conftest_module.tool_root = fake_tool_root
        targets = conftest_module._coverage_targets_from_tests(fake_paths)
    finally:
        Path.exists = original_exists  # type: ignore[assignment]

    assert targets == [
        str(fake_tool_root / "run_pytest.py"),
    ]


def test_pytest_collection_modifyitems_sets_cov_for_tool_tests(
    run_pytest_module: ModuleType,
) -> None:
    conftest = load_conftest_module()
    conftest_module = cast(Any, conftest)
    fake_tool_root = Path("/workspace/.opencode/tool")
    conftest_module.tool_root = fake_tool_root
    conftest_module.repo_root = Path("/workspace")

    class FakeConfig:
        def __init__(self, rootpath: Path) -> None:
            self.rootpath = rootpath
            self.rootdir = rootpath
            self.option = SimpleNamespace(
                cov=None,
                cov_source=None,
                cov_config=None,
                cov_fail_under=None,
            )

    config = FakeConfig(conftest_module.repo_root)
    items = [SimpleNamespace(fspath=fake_tool_root / "run_pytest_test.py")]

    def fake_exists(path: Path) -> bool:
        return path.name == "run_pytest.py"

    original_exists = Path.exists

    try:
        Path.exists = fake_exists  # type: ignore[assignment]
        conftest.pytest_collection_modifyitems(config, items)
    finally:
        Path.exists = original_exists  # type: ignore[assignment]

    assert config.option.cov == [str(fake_tool_root / "run_pytest.py")]
    assert config.option.cov_source == [str(fake_tool_root / "run_pytest.py")]
    assert config.option.cov_config == "/dev/null"
    assert config.option.cov_fail_under == 0


def test_pytest_collection_modifyitems_noop_for_non_tool_root(
    run_pytest_module: ModuleType,
) -> None:
    conftest = load_conftest_module()
    conftest_module = cast(Any, conftest)
    conftest_module.repo_root = Path("/workspace")

    class FakeConfig:
        def __init__(self, rootpath: Path) -> None:
            self.rootpath = rootpath
            self.rootdir = rootpath
            self.option = SimpleNamespace(
                cov=["existing"],
                cov_source=["existing"],
                cov_config="existing",
                cov_fail_under=50,
            )

    config = FakeConfig(Path("/other"))
    items = [SimpleNamespace(fspath=Path("/other/test.py"))]

    conftest.pytest_collection_modifyitems(config, items)

    assert config.option.cov == ["existing"]
    assert config.option.cov_source == ["existing"]
    assert config.option.cov_config == "existing"
    assert config.option.cov_fail_under == 50
