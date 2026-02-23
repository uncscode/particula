from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import List

import pytest

RUN_BUN_TEST_PATH = Path(__file__).resolve().parent.parent / "run_bun_test.py"


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_bun_test", RUN_BUN_TEST_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_bun_test"] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


@pytest.fixture(scope="module")
def run_bun_module() -> ModuleType:
    return load_module()


class DummyProcess:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_parse_bun_output_pass_and_duration(run_bun_module: ModuleType) -> None:
    output = (
        "bun test v1.0.0\n\n"
        "__tests__/get_datetime.test.ts:\n"
        "  get_datetime tool\n"
        "    ✓ returns UTC date by default (no args) [0.12ms]\n\n"
        " 1 pass\n"
        "Ran 1 tests across 1 files. [12.00ms]\n"
    )
    metrics = run_bun_module.parse_bun_output(output)

    assert metrics["passed"] == 1
    assert metrics["failed"] == 0
    assert metrics["total"] == 1
    assert metrics["duration"] == 0.012
    assert metrics["failed_tests"] == []


def test_parse_bun_output_mixed_failures(run_bun_module: ModuleType) -> None:
    output = (
        "__tests__/sample.test.ts:\n"
        "  suite\n"
        "    ✓ ok [0.01ms]\n"
        "    ✗ fails on invalid input [0.03ms]\n"
        "    ✕ other failure [0.02ms]\n"
        "    x legacy failure [0.01ms]\n\n"
        " 2 pass\n"
        " 2 fail\n"
        "Ran 4 tests across 1 files. [10.00ms]\n"
    )
    metrics = run_bun_module.parse_bun_output(output)

    assert metrics["passed"] == 2
    assert metrics["failed"] == 2
    assert metrics["total"] == 4
    assert "fails on invalid input" in metrics["failed_tests"]
    assert "other failure" in metrics["failed_tests"]
    assert "legacy failure" in metrics["failed_tests"]


def test_parse_bun_output_all_failing(run_bun_module: ModuleType) -> None:
    output = " 0 pass\n 2 fail\nRan 2 tests across 1 files. [5.00ms]"
    metrics = run_bun_module.parse_bun_output(output)

    assert metrics["passed"] == 0
    assert metrics["failed"] == 2
    assert metrics["total"] == 2


def test_parse_bun_output_with_skips(run_bun_module: ModuleType) -> None:
    output = " 1 pass\n 1 skip\nRan 2 tests across 1 files. [3.00ms]"
    metrics = run_bun_module.parse_bun_output(output)

    assert metrics["skipped"] == 1
    assert metrics["total"] == 2


def test_parse_bun_output_empty(run_bun_module: ModuleType) -> None:
    metrics = run_bun_module.parse_bun_output("")

    assert metrics["total"] == 0
    assert metrics["failed_tests"] == []


def test_validate_results_flags_errors(run_bun_module: ModuleType) -> None:
    metrics = {
        "passed": 0,
        "failed": 2,
        "total": 0,
        "timeout": True,
        "bun_missing": True,
        "test_path_error": True,
        "timeout_seconds": 10,
    }

    errors = run_bun_module.validate_results(metrics, min_test_count=3)

    assert any("failed" in err for err in errors)
    assert any("No tests" in err for err in errors)
    assert any("Expected at least" in err for err in errors)
    assert any("timed out" in err for err in errors)
    assert any("bun command not found" in err for err in errors)
    assert any("Test path" in err for err in errors)


def test_format_summary_includes_failures_and_validation(run_bun_module: ModuleType) -> None:
    metrics = {
        "passed": 3,
        "failed": 1,
        "skipped": 0,
        "total": 4,
        "duration": 1.5,
        "failed_tests": ["bad_test"],
    }
    summary = run_bun_module.format_summary(metrics, ["failure present"])

    assert "BUN TEST SUMMARY" in summary
    assert "Tests Run: 4" in summary
    assert "Failed Tests (1):" in summary
    assert "VALIDATION: FAILED" in summary
    assert "failure present" in summary


def test_truncate_output_limits(run_bun_module: ModuleType) -> None:
    long_lines = "\n".join(["x" * 200 for _ in range(600)])

    truncated_output, truncated, notice = run_bun_module._truncate_output(long_lines)

    assert truncated is True
    assert "Output truncated to 500 lines" in notice
    assert "Output truncated to 48KB" in notice
    assert truncated_output.rstrip().endswith(notice)


def test_run_bun_test_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    cwd = tmp_path / "tool"
    cwd.mkdir()
    tests_dir = cwd / "__tests__"
    tests_dir.mkdir()
    (tests_dir / "foo.test.ts").write_text("// dummy test")

    commands: List[List[str]] = []

    def fake_run(cmd: List[str], **_: object) -> DummyProcess:  # type: ignore[override]
        commands.append(cmd)
        stdout = " 1 pass\nRan 1 tests across 1 files. [10.00ms]"
        return DummyProcess(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code, output = module.run_bun_test(
        test_path="__tests__/foo.test.ts",
        test_filter="suite",
        timeout=42,
        min_test_count=1,
        output_mode="summary",
        bail=True,
        cwd=str(cwd),
    )

    assert exit_code == 0
    assert "VALIDATION: PASSED" in output
    assert [
        "bun",
        "test",
        "__tests__/foo.test.ts",
        "--timeout",
        "42",
        "--bail",
        "--test-name-pattern",
        "suite",
    ] in commands


def test_run_bun_test_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()

    def raise_timeout(*_: object, **__: object) -> None:  # type: ignore[override]
        raise subprocess.TimeoutExpired(cmd="bun", timeout=1)

    monkeypatch.setattr(module.subprocess, "run", raise_timeout)

    exit_code, output = module.run_bun_test(timeout=1, output_mode="summary")

    assert exit_code == 1
    assert "timed out" in output.lower()


def test_run_bun_test_missing_command(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()

    def raise_missing(*_: object, **__: object) -> None:  # type: ignore[override]
        raise FileNotFoundError()

    monkeypatch.setattr(module.subprocess, "run", raise_missing)

    exit_code, output = module.run_bun_test()

    assert exit_code == 1
    assert "command not found" in output


def test_run_bun_test_missing_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()

    def should_not_run(*_: object, **__: object) -> None:  # type: ignore[override]
        raise AssertionError("subprocess.run should not be called when path is missing")

    monkeypatch.setattr(module.subprocess, "run", should_not_run)

    exit_code, output = module.run_bun_test(
        test_path="missing.test.ts",
        output_mode="json",
        cwd=str(tmp_path),
    )

    payload = json.loads(output)
    assert exit_code == 1
    assert payload["metrics"]["test_path_error"] is True
    assert payload["success"] is False


def test_run_bun_test_json_output_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()

    def fake_run(*_: object, **__: object) -> DummyProcess:  # type: ignore[override]
        stdout = " 1 pass\nRan 1 tests across 1 files. [1.00ms]"
        return DummyProcess(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code, output = module.run_bun_test(output_mode="json")
    payload = json.loads(output)

    assert exit_code == 0
    assert payload["metrics"]["passed"] == 1
    assert payload["output"]
    assert "success" in payload


def test_main_wires_cli_arguments(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_module()
    captured_args = {}

    def fake_run_bun_test(**kwargs: object) -> tuple[int, str]:
        captured_args.update(kwargs)
        return 0, "ok"

    monkeypatch.setattr(module, "run_bun_test", fake_run_bun_test)

    with pytest.raises(SystemExit) as excinfo:
        module.main(
            [
                "--test-path",
                "__tests__/foo.test.ts",
                "--filter",
                "suite",
                "--timeout",
                "12",
                "--min-tests",
                "2",
                "--output",
                "summary",
                "--bail",
                "--cwd",
                "/tmp",
            ]
        )

    captured = capsys.readouterr()

    assert excinfo.value.code == 0
    assert captured.out.strip() == "ok"
    assert captured_args["test_path"] == "__tests__/foo.test.ts"
    assert captured_args["test_filter"] == "suite"
    assert captured_args["timeout"] == 12
    assert captured_args["min_test_count"] == 2
    assert captured_args["output_mode"] == "summary"
    assert captured_args["bail"] is True
    assert captured_args["cwd"] == "/tmp"


def test_parse_args_defaults(run_bun_module: ModuleType) -> None:
    """Test _parse_args returns expected defaults when no arguments are provided."""

    args = run_bun_module._parse_args([])

    assert args.test_path is None
    assert args.filter is None
    assert args.timeout == run_bun_module.DEFAULT_TIMEOUT
    assert args.min_tests == 1
    assert args.output == "summary"
    assert args.bail is False
    assert args.cwd is None
