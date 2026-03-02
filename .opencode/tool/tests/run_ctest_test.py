"""Unit tests for run_ctest.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType
from typing import List

import pytest

RUN_CTEST_PATH = Path(__file__).resolve().parent.parent / "run_ctest.py"


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_ctest", RUN_CTEST_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


@pytest.fixture(scope="module")
def run_ctest_module() -> ModuleType:
    return load_module()


class DummyProcess:
    def __init__(
        self,
        stdout: bytes | None = b"",
        stderr: bytes | None = b"",
        returncode: int = 0,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_parse_ctest_output_pass_and_duration(run_ctest_module: ModuleType) -> None:
    output = (
        "Test project /tmp/build\n"
        "1/1 Test #1: test_example ................   Passed    0.01 sec\n\n"
        "100% tests passed, 0 tests failed out of 1\n\n"
        "Total Test time (real) =   0.02 sec"
    )
    metrics = run_ctest_module.parse_ctest_output(output)

    assert metrics["passed"] == 1
    assert metrics["failed"] == 0
    assert metrics["total"] == 1
    assert metrics["duration"] == 0.02
    assert metrics["failed_tests"] == []


def test_parse_ctest_output_failures_and_section(run_ctest_module: ModuleType) -> None:
    output = (
        "1/2 Test #1: ok_test ................   Passed    0.01 sec\n"
        "2/2 Test #2: bad_test ...............***Failed    0.01 sec\n\n"
        "50% tests passed, 1 tests failed out of 2\n\n"
        "Total Test time (real) =   0.05 sec\n\n"
        "The following tests FAILED:\n"
        "  2 - bad_test (Failed)\n"
    )
    metrics = run_ctest_module.parse_ctest_output(output)

    assert metrics["failed"] == 1
    assert metrics["passed"] == 1
    assert metrics["total"] == 2
    assert "bad_test" in metrics["failed_tests"]


def test_validate_results_flags_all_errors(run_ctest_module: ModuleType) -> None:
    metrics = {
        "passed": 0,
        "failed": 2,
        "total": 0,
        "timeout": True,
        "ctest_missing": True,
        "build_dir_error": True,
        "timeout_seconds": 10,
    }

    errors = run_ctest_module.validate_results(metrics, min_test_count=3)

    assert any("failed" in err for err in errors)
    assert any("No tests" in err for err in errors)
    assert any("Expected at least" in err for err in errors)
    assert any("timed out" in err for err in errors)
    assert any("ctest command not found" in err for err in errors)
    assert any("CTestTestfile" in err or "Build directory" in err for err in errors)


def test_format_summary_includes_failed_tests(run_ctest_module: ModuleType) -> None:
    metrics = {
        "passed": 3,
        "failed": 1,
        "skipped": 0,
        "total": 4,
        "duration": 1.5,
        "failed_tests": ["bad_test"],
    }
    summary = run_ctest_module.format_summary(metrics, ["failure present"])

    assert "CTEST SUMMARY" in summary
    assert "Tests Run: 4" in summary
    assert "Failed Tests (1):" in summary
    assert "VALIDATION: FAILED" in summary
    assert "failure present" in summary


def test_truncate_output_limits(run_ctest_module: ModuleType) -> None:
    long_lines = "\n".join(["x" * 200 for _ in range(600)])

    truncated_output, truncated, notice = run_ctest_module._truncate_output(long_lines)

    assert truncated is True
    assert "Output truncated to 500 lines" in notice
    assert "Output truncated to 48KB" in notice
    assert truncated_output.rstrip().endswith(notice)


def test_run_ctest_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CTestTestfile.cmake").write_text("# dummy")

    commands: List[List[str]] = []

    def fake_run(cmd: List[str], **_: object) -> DummyProcess:  # type: ignore[override]
        commands.append(cmd)
        stdout = (
            "1/1 Test #1: sample ................   Passed    0.01 sec\n\n"
            "100% tests passed, 0 tests failed out of 1\n\n"
            "Total Test time (real) =   0.02 sec"
        ).encode("utf-8")
        return DummyProcess(stdout=stdout, stderr=b"", returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code, output = module.run_ctest(
        build_dir=build_dir,
        include_filter="foo",
        exclude_filter="bar",
        parallel=4,
        min_test_count=1,
        output_mode="summary",
    )

    assert exit_code == 0
    assert "VALIDATION: PASSED" in output
    assert ["ctest", "--output-on-failure", "-R", "foo", "-E", "bar", "-j", "4"] in commands


def test_run_ctest_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CTestTestfile.cmake").write_text("# dummy")

    def raise_timeout(*_: object, **__: object) -> None:  # type: ignore[override]
        raise subprocess.TimeoutExpired(cmd="ctest", timeout=1)

    monkeypatch.setattr(module.subprocess, "run", raise_timeout)

    exit_code, output = module.run_ctest(build_dir=build_dir, timeout=1, output_mode="summary")

    assert exit_code == 1
    assert "timed out" in output.lower()


def test_run_ctest_binary_output_decodes_with_replace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CTestTestfile.cmake").write_text("# dummy")

    def fake_run(*_: object, **__: object) -> DummyProcess:  # type: ignore[override]
        stdout = b"1/1 Test #1: sample .... Passed 0.01 sec\n\xff\n"
        stderr = b"binary-error\xfe"
        return DummyProcess(stdout=stdout, stderr=stderr, returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code, output = module.run_ctest(build_dir=build_dir, output_mode="json")

    payload = json.loads(output)
    assert exit_code == 1  # validation fails because total=0
    assert "\ufffd" in payload["output"]


def test_run_ctest_binary_output_none_handling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CTestTestfile.cmake").write_text("# dummy")

    def fake_run(*_: object, **__: object) -> DummyProcess:  # type: ignore[override]
        return DummyProcess(stdout=None, stderr=None, returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code, output = module.run_ctest(build_dir=build_dir, output_mode="json")

    payload = json.loads(output)
    assert exit_code == 1  # validation fails because total=0
    assert payload["output"] == ""


def test_run_ctest_timeout_binary_partial_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CTestTestfile.cmake").write_text("# dummy")

    def raise_timeout(*_: object, **__: object) -> None:  # type: ignore[override]
        raise subprocess.TimeoutExpired(
            cmd="ctest",
            timeout=1,
            output=b"partial\xff",
            stderr=b"stderr\xfe",
        )

    monkeypatch.setattr(module.subprocess, "run", raise_timeout)

    exit_code, output = module.run_ctest(build_dir=build_dir, timeout=1, output_mode="full")

    assert exit_code == 1
    assert "timed out" in output.lower()
    assert "\ufffd" in output


def test_run_ctest_missing_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CTestTestfile.cmake").write_text("# dummy")

    def raise_missing(*_: object, **__: object) -> None:  # type: ignore[override]
        raise FileNotFoundError()

    monkeypatch.setattr(module.subprocess, "run", raise_missing)

    exit_code, output = module.run_ctest(build_dir=build_dir)

    assert exit_code == 1
    assert "command not found" in output


def test_run_ctest_missing_build_dir(tmp_path: Path) -> None:
    module = load_module()
    build_dir = tmp_path / "missing"

    exit_code, output = module.run_ctest(build_dir=build_dir, output_mode="json")

    payload = json.loads(output)
    assert exit_code == 1
    assert payload["metrics"]["build_dir_error"] is True
    assert payload["success"] is False
    assert any(
        "CTestTestfile" in err or "Build directory" in err for err in payload["validation_errors"]
    )


def test_run_ctest_full_and_json_truncate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CTestTestfile.cmake").write_text("# dummy")

    long_output = "\n".join([f"line {i}" for i in range(700)])

    def fake_run(*_: object, **__: object) -> DummyProcess:  # type: ignore[override]
        return DummyProcess(stdout=long_output.encode("utf-8"), stderr=b"", returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code_full, full_output = module.run_ctest(build_dir=build_dir, output_mode="full")
    assert exit_code_full == 1  # validation fails because total=0
    assert "Output truncated" in full_output

    exit_code_json, json_output = module.run_ctest(build_dir=build_dir, output_mode="json")
    payload = json.loads(json_output)

    assert exit_code_json == 1
    assert payload["truncated"] is True
    assert "Output truncated" in payload["truncation_notice"]


def test_run_ctest_min_test_enforced(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CTestTestfile.cmake").write_text("# dummy")

    def fake_run(*_: object, **__: object) -> DummyProcess:  # type: ignore[override]
        stdout = "100% tests passed, 0 tests failed out of 1\nTotal Test time (real) =   0.01 sec"
        return DummyProcess(stdout=stdout.encode("utf-8"), stderr=b"", returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code, output = module.run_ctest(build_dir=build_dir, min_test_count=3)

    assert exit_code == 1
    assert "Expected at least 3 tests" in output


def test_parse_args_accepts_overrides(tmp_path: Path) -> None:
    module = load_module()
    build_dir = tmp_path / "build"

    parsed = module._parse_args(
        [
            "--build-dir",
            str(build_dir),
            "-R",
            "include",
            "-E",
            "exclude",
            "-j",
            "2",
            "--timeout",
            "42",
            "--min-tests",
            "3",
            "--output",
            "full",
        ]
    )

    assert parsed.build_dir == build_dir
    assert parsed.include == "include"
    assert parsed.exclude == "exclude"
    assert parsed.parallel == 2
    assert parsed.timeout == 42
    assert parsed.min_tests == 3
    assert parsed.output == "full"


def test_main_wires_cli_arguments(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    module = load_module()
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CTestTestfile.cmake").write_text("# dummy")

    captured_args = {}

    def fake_run_ctest(**kwargs: object) -> tuple[int, str]:
        captured_args.update(kwargs)
        return 0, "ok"

    monkeypatch.setattr(module, "run_ctest", fake_run_ctest)

    with pytest.raises(SystemExit) as excinfo:
        module.main(
            [
                "--build-dir",
                str(build_dir),
                "-R",
                "foo",
                "-E",
                "bar",
                "-j",
                "3",
                "--timeout",
                "12",
                "--min-tests",
                "2",
                "--output",
                "summary",
            ]
        )

    captured = capsys.readouterr()

    assert excinfo.value.code == 0
    assert captured.out.strip() == "ok"
    assert captured_args["build_dir"] == build_dir
    assert captured_args["include_filter"] == "foo"
    assert captured_args["exclude_filter"] == "bar"
    assert captured_args["parallel"] == 3
    assert captured_args["timeout"] == 12
    assert captured_args["min_test_count"] == 2
    assert captured_args["output_mode"] == "summary"
