"""Unit tests for run_cpp_linters tool."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import List, Tuple

import pytest

RUN_CPP_LINTERS_PATH = Path(__file__).resolve().parent.parent / "run_cpp_linters.py"


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_cpp_linters", RUN_CPP_LINTERS_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module  # ensure dataclass resolution sees module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


@pytest.fixture(scope="module")
def run_cpp_linters_module() -> ModuleType:
    return load_module()


def create_cpp_files(tmp_path: Path, count: int = 1) -> List[Path]:
    files = []
    for idx in range(count):
        file_path = tmp_path / f"file_{idx}.cpp"
        file_path.write_text("int main() { return 0; }\n")
        files.append(file_path)
    return files


def test_get_cpp_files_filters_and_sorts(
    tmp_path: Path, run_cpp_linters_module: ModuleType
) -> None:
    cpp1 = tmp_path / "b_file.cpp"
    cpp2 = tmp_path / "a_file.hpp"
    other = tmp_path / "ignore.txt"
    cpp1.write_text("int a() { return 0; }\n")
    cpp2.write_text("// header\n")
    other.write_text("noop")

    found = run_cpp_linters_module.get_cpp_files(str(tmp_path))

    assert found == [cpp2, cpp1]


def test_check_linter_available_true_false(
    monkeypatch: pytest.MonkeyPatch, run_cpp_linters_module: ModuleType
) -> None:
    monkeypatch.setattr(run_cpp_linters_module.shutil, "which", lambda *_: "/usr/bin/tool")
    assert run_cpp_linters_module.check_linter_available("dummy") is True

    monkeypatch.setattr(run_cpp_linters_module.shutil, "which", lambda *_: None)
    assert run_cpp_linters_module.check_linter_available("dummy") is False


def test_helper_truncate_and_boundaries(run_cpp_linters_module: ModuleType) -> None:
    long_lines = "\n".join(["line"] * (run_cpp_linters_module.OUTPUT_LINE_LIMIT + 5))
    truncated, was_truncated, notice = run_cpp_linters_module._truncate_output(long_lines)

    assert was_truncated is True
    assert "truncated" in notice
    assert "..." in truncated

    items: List[str] = []
    for _ in range(3):
        run_cpp_linters_module._bounded_append(items, "item", limit=2)
    assert items == ["item", "item"]


def test_run_subprocess_file_not_found(
    monkeypatch: pytest.MonkeyPatch, run_cpp_linters_module: ModuleType
) -> None:
    def fake_run(*_: object, **__: object) -> None:
        raise FileNotFoundError("missing")

    monkeypatch.setattr(run_cpp_linters_module.subprocess, "run", fake_run)

    (
        exit_code,
        stdout,
        stderr,
        timed_out,
        error_message,
    ) = run_cpp_linters_module._run_subprocess(["missing-cmd"], timeout=1)

    assert exit_code == 1
    assert error_message == "Command not found: missing-cmd"
    assert stdout == ""
    assert stderr == ""
    assert timed_out is False


def test_parse_linters_arg_defaults_and_trim(run_cpp_linters_module: ModuleType) -> None:
    assert run_cpp_linters_module.parse_linters_arg("") == [
        "clang-format",
        "clang-tidy",
        "cppcheck",
    ]
    assert run_cpp_linters_module.parse_linters_arg(" clang-format , cppcheck ") == [
        "clang-format",
        "cppcheck",
    ]


def test_clang_format_skips_when_missing(
    monkeypatch: pytest.MonkeyPatch, run_cpp_linters_module: ModuleType
) -> None:
    monkeypatch.setattr(run_cpp_linters_module, "check_linter_available", lambda *_: False)
    result = run_cpp_linters_module.run_clang_format([], auto_fix=False, timeout=10)
    assert result.skipped is True
    assert "not found" in (result.error_message or "")


def test_clang_format_reports_issues(
    monkeypatch: pytest.MonkeyPatch, run_cpp_linters_module: ModuleType, tmp_path: Path
) -> None:
    files = create_cpp_files(tmp_path)
    monkeypatch.setattr(run_cpp_linters_module, "check_linter_available", lambda *_: True)

    commands: List[List[str]] = []

    def fake_run(cmd: List[str], timeout: int) -> Tuple[int, str, str, bool, str | None]:
        commands.append(list(cmd))
        return 1, f"{files[0]}: warning: format", "", False, None

    monkeypatch.setattr(run_cpp_linters_module, "_run_subprocess", fake_run)

    result = run_cpp_linters_module.run_clang_format(files, auto_fix=False, timeout=5)

    assert result.success is False
    assert result.files_with_issues > 0
    assert result.issues
    assert commands[-1][0] == "clang-format"


def test_clang_tidy_requires_compile_commands(
    monkeypatch: pytest.MonkeyPatch, run_cpp_linters_module: ModuleType, tmp_path: Path
) -> None:
    files = create_cpp_files(tmp_path)
    monkeypatch.setattr(run_cpp_linters_module, "check_linter_available", lambda *_: True)

    result = run_cpp_linters_module.run_clang_tidy(
        files, build_dir=str(tmp_path / "build"), auto_fix=False, timeout=5
    )

    assert result.success is False
    assert "compile_commands.json" in (result.error_message or "")


def test_clang_tidy_parses_warnings(
    monkeypatch: pytest.MonkeyPatch, run_cpp_linters_module: ModuleType, tmp_path: Path
) -> None:
    files = create_cpp_files(tmp_path)
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "compile_commands.json").write_text("[]")
    monkeypatch.setattr(run_cpp_linters_module, "check_linter_available", lambda *_: True)

    def fake_run(*_: object, **__: object) -> Tuple[int, str, str, bool, str | None]:
        return 0, f"{files[0]}:10: warning: test", "", False, None

    monkeypatch.setattr(run_cpp_linters_module, "_run_subprocess", fake_run)

    result = run_cpp_linters_module.run_clang_tidy(
        files, build_dir=str(build_dir), auto_fix=False, timeout=5
    )

    assert result.warnings == 1
    assert result.success is True
    assert result.files_with_issues == 1


def test_clang_tidy_batching(
    monkeypatch: pytest.MonkeyPatch, run_cpp_linters_module: ModuleType, tmp_path: Path
) -> None:
    file_count = run_cpp_linters_module.CLANG_TIDY_BATCH_SIZE + 5
    files = create_cpp_files(tmp_path, count=file_count)
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "compile_commands.json").write_text("[]")
    monkeypatch.setattr(run_cpp_linters_module, "check_linter_available", lambda *_: True)

    commands: List[List[str]] = []

    def fake_run(cmd: List[str], timeout: int) -> Tuple[int, str, str, bool, str | None]:
        commands.append(list(cmd))
        return 0, "", "", False, None

    monkeypatch.setattr(run_cpp_linters_module, "_run_subprocess", fake_run)

    result = run_cpp_linters_module.run_clang_tidy(
        files, build_dir=str(build_dir), auto_fix=False, timeout=5
    )

    expected_calls = (
        file_count + run_cpp_linters_module.CLANG_TIDY_BATCH_SIZE - 1
    ) // run_cpp_linters_module.CLANG_TIDY_BATCH_SIZE
    assert len(commands) == expected_calls
    assert result.success is True


def test_cppcheck_parses_warnings_and_errors(
    monkeypatch: pytest.MonkeyPatch, run_cpp_linters_module: ModuleType, tmp_path: Path
) -> None:
    files = create_cpp_files(tmp_path)
    monkeypatch.setattr(run_cpp_linters_module, "check_linter_available", lambda *_: True)

    def fake_run(*_: object, **__: object) -> Tuple[int, str, str, bool, str | None]:
        stderr = (
            f"{files[0]}:10: (warning) thing\n"
            f"{files[0]}:12: (error) boom"
        )
        return 1, "", stderr, False, None

    monkeypatch.setattr(run_cpp_linters_module, "_run_subprocess", fake_run)

    result = run_cpp_linters_module.run_cppcheck(files, timeout=5)

    assert result.warnings == 1
    assert result.errors == 1
    assert result.files_with_issues == 1
    assert result.success is False


def test_run_cpp_linters_all_skipped(
    monkeypatch: pytest.MonkeyPatch, run_cpp_linters_module: ModuleType, tmp_path: Path
) -> None:
    create_cpp_files(tmp_path)
    monkeypatch.setattr(run_cpp_linters_module, "get_cpp_files", lambda *_: [tmp_path / "file.cpp"])
    monkeypatch.setattr(run_cpp_linters_module, "check_linter_available", lambda *_: False)

    exit_code, summary = run_cpp_linters_module.run_cpp_linters(
        source_dir=str(tmp_path),
        build_dir=None,
        linters=["clang-format"],
        auto_fix=False,
        output_mode="summary",
        timeout=5,
    )

    assert exit_code == 1
    assert "VALIDATION: FAILED" in summary
    assert "SKIPPED" in summary


def test_run_cpp_linters_no_files_found(
    monkeypatch: pytest.MonkeyPatch, run_cpp_linters_module: ModuleType, tmp_path: Path
) -> None:
    monkeypatch.setattr(run_cpp_linters_module, "get_cpp_files", lambda *_: [])
    monkeypatch.setattr(run_cpp_linters_module, "check_linter_available", lambda *_: True)

    exit_code, summary = run_cpp_linters_module.run_cpp_linters(
        source_dir=str(tmp_path),
        build_dir=None,
        linters=["clang-format"],
        auto_fix=False,
        output_mode="summary",
        timeout=5,
    )

    assert exit_code == 1
    assert "No C++ files" in summary


def test_run_cpp_linters_missing_build_dir(
    monkeypatch: pytest.MonkeyPatch, run_cpp_linters_module: ModuleType, tmp_path: Path
) -> None:
    files = create_cpp_files(tmp_path)
    monkeypatch.setattr(run_cpp_linters_module, "get_cpp_files", lambda *_: files)
    monkeypatch.setattr(run_cpp_linters_module, "check_linter_available", lambda *_: True)

    exit_code, summary = run_cpp_linters_module.run_cpp_linters(
        source_dir=str(tmp_path),
        build_dir=None,
        linters=["clang-tidy"],
        auto_fix=False,
        output_mode="summary",
        timeout=5,
    )

    assert exit_code == 1
    assert "compile_commands.json" in summary or "--build-dir" in summary


def test_run_cpp_linters_success_with_warnings(
    monkeypatch: pytest.MonkeyPatch, run_cpp_linters_module: ModuleType, tmp_path: Path
) -> None:
    files = create_cpp_files(tmp_path)
    monkeypatch.setattr(run_cpp_linters_module, "get_cpp_files", lambda *_: files)
    monkeypatch.setattr(run_cpp_linters_module, "check_linter_available", lambda *_: True)

    def fake_run(cmd: List[str], timeout: int) -> Tuple[int, str, str, bool, str | None]:
        if cmd[0] == "clang-tidy":
            return 0, f"{files[0]}:10: warning: test", "", False, None
        return 0, "", "", False, None

    monkeypatch.setattr(run_cpp_linters_module, "_run_subprocess", fake_run)

    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "compile_commands.json").write_text("[]")

    exit_code, summary = run_cpp_linters_module.run_cpp_linters(
        source_dir=str(tmp_path),
        build_dir=str(build_dir),
        linters=["clang-format", "clang-tidy"],
        auto_fix=False,
        output_mode="summary",
        timeout=5,
    )

    assert exit_code == 0
    assert "WARNINGS" in summary
    assert "VALIDATION: PASSED" in summary


def test_run_cpp_linters_failure_on_errors(
    monkeypatch: pytest.MonkeyPatch, run_cpp_linters_module: ModuleType, tmp_path: Path
) -> None:
    files = create_cpp_files(tmp_path)
    monkeypatch.setattr(run_cpp_linters_module, "get_cpp_files", lambda *_: files)
    monkeypatch.setattr(run_cpp_linters_module, "check_linter_available", lambda *_: True)

    def fake_run(*_: object, **__: object) -> Tuple[int, str, str, bool, str | None]:
        return 1, "", f"{files[0]}:10: (error) fail", False, None

    monkeypatch.setattr(run_cpp_linters_module, "_run_subprocess", fake_run)

    exit_code, summary = run_cpp_linters_module.run_cpp_linters(
        source_dir=str(tmp_path),
        build_dir=str(tmp_path),
        linters=["cppcheck"],
        auto_fix=False,
        output_mode="summary",
        timeout=5,
    )

    assert exit_code == 1
    assert "FAILED" in summary


def test_run_cpp_linters_json_includes_truncated(
    monkeypatch: pytest.MonkeyPatch, run_cpp_linters_module: ModuleType, tmp_path: Path
) -> None:
    files = create_cpp_files(tmp_path)
    long_output = "\n".join(["line"] * (run_cpp_linters_module.OUTPUT_LINE_LIMIT + 10))
    monkeypatch.setattr(run_cpp_linters_module, "get_cpp_files", lambda *_: files)
    monkeypatch.setattr(run_cpp_linters_module, "check_linter_available", lambda *_: True)

    def fake_run(*_: object, **__: object) -> Tuple[int, str, str, bool, str | None]:
        return 0, long_output, "", False, None

    monkeypatch.setattr(run_cpp_linters_module, "_run_subprocess", fake_run)

    exit_code, output = run_cpp_linters_module.run_cpp_linters(
        source_dir=str(tmp_path),
        build_dir=str(tmp_path),
        linters=["clang-format"],
        auto_fix=False,
        output_mode="json",
        timeout=5,
    )

    payload = json.loads(output)
    assert exit_code == 0
    assert payload["results"][0]["truncated"] is True


def test_format_full_output_includes_truncated_notice(
    run_cpp_linters_module: ModuleType,
) -> None:
    result_ok = run_cpp_linters_module.LinterResult("clang-format")
    result_ok.stdout = "out"
    result_ok.stderr = "err"
    result_ok.truncated = True
    result_empty = run_cpp_linters_module.LinterResult("cppcheck")

    combined = run_cpp_linters_module.format_full_output(
        [result_ok, result_empty], "SUMMARY SECTION"
    )

    assert "clang-format Output" in combined
    assert "Stderr:" in combined
    assert "(Note: output truncated)" in combined
    assert "SUMMARY SECTION" in combined


def test_timeout_handling(
    monkeypatch: pytest.MonkeyPatch, run_cpp_linters_module: ModuleType, tmp_path: Path
) -> None:
    files = create_cpp_files(tmp_path)
    monkeypatch.setattr(run_cpp_linters_module, "get_cpp_files", lambda *_: files)
    monkeypatch.setattr(run_cpp_linters_module, "check_linter_available", lambda *_: True)

    def fake_run(*_: object, **__: object) -> Tuple[int, str, str, bool, str | None]:
        return 1, "", "", True, None

    monkeypatch.setattr(run_cpp_linters_module, "_run_subprocess", fake_run)

    result = run_cpp_linters_module.run_cppcheck(files, timeout=1)

    assert result.success is False
    assert "timed out" in (result.error_message or "")


def test_auto_fix_flags_passed(
    monkeypatch: pytest.MonkeyPatch, run_cpp_linters_module: ModuleType, tmp_path: Path
) -> None:
    files = create_cpp_files(tmp_path)
    monkeypatch.setattr(run_cpp_linters_module, "check_linter_available", lambda *_: True)

    commands: List[List[str]] = []

    def fake_run(cmd: List[str], timeout: int) -> Tuple[int, str, str, bool, str | None]:
        commands.append(list(cmd))
        return 0, "", "", False, None

    monkeypatch.setattr(run_cpp_linters_module, "_run_subprocess", fake_run)

    run_cpp_linters_module.run_clang_format(files, auto_fix=True, timeout=5)

    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "compile_commands.json").write_text("[]")
    run_cpp_linters_module.run_clang_tidy(
        files, build_dir=str(build_dir), auto_fix=True, timeout=5
    )

    assert any("-i" in cmd for cmd in commands)
    assert any("--fix" in cmd for cmd in commands)
