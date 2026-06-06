import importlib.util
import subprocess
from pathlib import Path

import pytest

TOOL_PATH = Path(__file__).resolve().parents[1] / "run_linters.py"
SPEC = importlib.util.spec_from_file_location("run_linters_tool", TOOL_PATH)
assert SPEC is not None
assert SPEC.loader is not None
run_linters_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_linters_tool)


def _completed_process(args: list[str], returncode: int = 0, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(
        args=args,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_count_ruff_issues_extracts_found_error_count():
    assert run_linters_tool._count_ruff_issues("Found 3 errors.\n") == 3


def test_count_ruff_issues_returns_zero_when_summary_missing():
    assert run_linters_tool._count_ruff_issues("All checks passed\n") == 0


def test_count_ruff_issues_returns_zero_when_found_summary_is_not_parseable():
    assert run_linters_tool._count_ruff_issues("Found errors but no count\n") == 0


def test_apply_process_result_copies_process_fields_and_success_state():
    result = run_linters_tool.LinterResult("ruff_check")
    proc = _completed_process(
        ["ruff", "check", "adw/core/"],
        returncode=1,
        stdout="lint output",
        stderr="lint error",
    )

    run_linters_tool._apply_process_result(result, proc)

    assert result.exit_code == 1
    assert result.stdout == "lint output"
    assert result.stderr == "lint error"
    assert result.success is False


def test_run_ruff_check_no_auto_fix_runs_only_validation_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls: list[dict[str, object]] = []

    def fake_run(args, **kwargs):
        calls.append({"args": list(args), **kwargs})
        return _completed_process(list(args), stdout="All checks passed\n")

    monkeypatch.setattr(run_linters_tool.subprocess, "run", fake_run)

    result = run_linters_tool.run_ruff_check(
        target_dir="adw/core",
        auto_fix=False,
        timeout=33,
        cwd=str(tmp_path),
    )

    assert result.success is True
    assert [call["args"] for call in calls] == [["ruff", "check", "adw/core/"]]
    assert calls[0]["timeout"] == 33
    assert calls[0]["cwd"] == str(tmp_path)


def test_run_ruff_check_auto_fix_runs_fix_format_then_final_check(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return _completed_process(list(args), stdout="All checks passed\n")

    monkeypatch.setattr(run_linters_tool.subprocess, "run", fake_run)

    result = run_linters_tool.run_ruff_check(target_dir="adw/core", auto_fix=True)

    assert result.success is True
    assert calls == [
        ["ruff", "check", "--fix", "adw/core/"],
        ["ruff", "format", "adw/core/"],
        ["ruff", "check", "adw/core/"],
    ]


def test_run_linters_no_auto_fix_reports_lint_failure_without_mutating_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    target_file = tmp_path / "sample.py"
    original_content = "x=1\n"
    target_file.write_text(original_content)

    def fake_run(args, **kwargs):
        assert "--fix" not in args
        assert "format" not in args
        return _completed_process(
            list(args),
            returncode=1,
            stdout=f"{target_file}:1:1: E225 missing whitespace around operator\nFound 1 error.\n",
        )

    monkeypatch.setattr(run_linters_tool.subprocess, "run", fake_run)

    exit_code, output = run_linters_tool.run_linters(
        target_dir=str(tmp_path),
        auto_fix=False,
        linters=["ruff"],
        cwd=str(tmp_path),
    )

    assert exit_code == 1
    assert "RESULT: LINTING FAILED" in output
    assert target_file.read_text() == original_content


def test_run_linters_passes_isolated_cwd_to_subprocess_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls: list[dict[str, object]] = []

    def fake_run(args, **kwargs):
        calls.append({"args": list(args), **kwargs})
        return _completed_process(list(args), stdout="All checks passed\n")

    monkeypatch.setattr(run_linters_tool.subprocess, "run", fake_run)

    exit_code, _ = run_linters_tool.run_linters(
        target_dir="adw/core",
        auto_fix=False,
        linters=["ruff", "mypy"],
        cwd=str(tmp_path),
        ruff_timeout=40,
        mypy_timeout=50,
    )

    assert exit_code == 0
    assert [call["args"] for call in calls] == [
        ["ruff", "check", "adw/core/"],
        ["mypy", "adw/core/", "--ignore-missing-imports"],
    ]
    assert [call["cwd"] for call in calls] == [str(tmp_path), str(tmp_path)]
    assert [call["timeout"] for call in calls] == [40, 50]


def test_run_ruff_check_timeout_sets_error_without_followup_mutation(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[:3] == ["ruff", "check", "--fix"]:
            raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs["timeout"])
        return _completed_process(list(args), stdout="All checks passed\n")

    monkeypatch.setattr(run_linters_tool.subprocess, "run", fake_run)

    result = run_linters_tool.run_ruff_check(target_dir="adw/core", auto_fix=True, timeout=12)

    assert result.success is False
    assert result.error_message == "Timeout after 12 seconds"
    assert calls == [["ruff", "check", "--fix", "adw/core/"]]


def test_run_ruff_check_fix_failure_continues_to_format_and_final_check(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[:3] == ["ruff", "check", "--fix"]:
            return _completed_process(
                list(args),
                returncode=1,
                stdout="sample.py:1:1: F401 unused import\nFound 1 error.\n",
            )
        return _completed_process(list(args), stdout="All checks passed\n")

    monkeypatch.setattr(run_linters_tool.subprocess, "run", fake_run)

    result = run_linters_tool.run_ruff_check(target_dir="adw/core", auto_fix=True)

    assert result.success is True
    assert result.exit_code == 0
    assert calls == [
        ["ruff", "check", "--fix", "adw/core/"],
        ["ruff", "format", "adw/core/"],
        ["ruff", "check", "adw/core/"],
    ]


def test_run_linters_rejects_target_dir_outside_resolved_cwd(tmp_path: Path):
    outside = tmp_path.parent / "outside-target"

    with pytest.raises(ValueError, match="target_dir resolves outside cwd"):
        run_linters_tool.run_linters(
            target_dir=str(outside),
            auto_fix=False,
            linters=["ruff"],
            cwd=str(tmp_path),
        )


def test_run_ruff_check_format_failure_stops_before_final_check(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[:2] == ["ruff", "format"]:
            return _completed_process(list(args), returncode=2, stderr="format failed")
        return _completed_process(list(args), stdout="All checks passed\n")

    monkeypatch.setattr(run_linters_tool.subprocess, "run", fake_run)

    result = run_linters_tool.run_ruff_check(target_dir="adw/core", auto_fix=True)

    assert result.success is False
    assert result.exit_code == 2
    assert result.stderr == "format failed"
    assert calls == [
        ["ruff", "check", "--fix", "adw/core/"],
        ["ruff", "format", "adw/core/"],
    ]


def test_run_ruff_check_missing_executable_reports_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_run(args, **kwargs):
        raise FileNotFoundError("ruff")

    monkeypatch.setattr(run_linters_tool.subprocess, "run", fake_run)

    result = run_linters_tool.run_ruff_check(auto_fix=False)

    assert result.success is False
    assert result.error_message == "ruff not found - is it installed?"


def test_run_ruff_check_rejects_option_like_target_dir():
    result = run_linters_tool.run_ruff_check(target_dir="--config", auto_fix=False)

    assert result.success is False
    assert result.error_message == "target_dir must not start with '-': --config"


def test_run_ruff_check_reports_invalid_cwd_before_spawn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    def fake_run(args, **kwargs):
        raise AssertionError("subprocess.run should not be reached for invalid cwd")

    monkeypatch.setattr(run_linters_tool.subprocess, "run", fake_run)

    missing_cwd = tmp_path / "missing-dir"
    result = run_linters_tool.run_ruff_check(auto_fix=False, cwd=str(missing_cwd))

    assert result.success is False
    assert result.error_message == f"cwd does not exist: {missing_cwd}"


def test_run_ruff_check_unexpected_exception_is_reported(monkeypatch: pytest.MonkeyPatch):
    def fake_run(args, **kwargs):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(run_linters_tool.subprocess, "run", fake_run)

    result = run_linters_tool.run_ruff_check(auto_fix=False)

    assert result.success is False
    assert result.error_message == "unexpected failure"


def test_run_linters_selected_linters_preserve_existing_orchestrator_behavior(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return _completed_process(list(args), stdout="All checks passed\n")

    monkeypatch.setattr(run_linters_tool.subprocess, "run", fake_run)

    exit_code_ruff, _ = run_linters_tool.run_linters(
        target_dir="adw/core",
        auto_fix=False,
        linters=["ruff"],
        cwd=str(tmp_path),
    )
    assert exit_code_ruff == 0
    assert calls == [["ruff", "check", "adw/core/"]]

    calls.clear()

    exit_code_mypy, _ = run_linters_tool.run_linters(
        target_dir="adw/core",
        auto_fix=False,
        linters=["mypy"],
        cwd=str(tmp_path),
    )
    assert exit_code_mypy == 0
    assert calls == [["mypy", "adw/core/", "--ignore-missing-imports"]]


def test_run_ruff_format_counts_reformatted_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def fake_run(args, **kwargs):
        return _completed_process(list(args), stdout="2 files reformatted\n")

    monkeypatch.setattr(run_linters_tool.subprocess, "run", fake_run)

    result = run_linters_tool.run_ruff_format(
        target_dir="adw/core",
        timeout=21,
        cwd=str(tmp_path),
    )

    assert result.success is True
    assert result.issues_fixed == 2


def test_run_ruff_format_timeout_reports_error(monkeypatch: pytest.MonkeyPatch):
    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs["timeout"])

    monkeypatch.setattr(run_linters_tool.subprocess, "run", fake_run)

    result = run_linters_tool.run_ruff_format(timeout=9)

    assert result.success is False
    assert result.error_message == "Timeout after 9 seconds"


def test_run_ruff_format_unexpected_exception_is_reported(monkeypatch: pytest.MonkeyPatch):
    def fake_run(args, **kwargs):
        raise RuntimeError("format boom")

    monkeypatch.setattr(run_linters_tool.subprocess, "run", fake_run)

    result = run_linters_tool.run_ruff_format()

    assert result.success is False
    assert result.error_message == "format boom"


def test_run_mypy_counts_error_lines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def fake_run(args, **kwargs):
        return _completed_process(
            list(args),
            returncode=1,
            stdout="a.py:1: error: bad\nb.py:2: error: worse\n",
        )

    monkeypatch.setattr(run_linters_tool.subprocess, "run", fake_run)

    result = run_linters_tool.run_mypy(target_dir="adw/core", timeout=17, cwd=str(tmp_path))

    assert result.success is False
    assert result.issues_found == 2
    assert result.exit_code == 1


def test_run_mypy_missing_executable_reports_helpful_error(monkeypatch: pytest.MonkeyPatch):
    def fake_run(args, **kwargs):
        raise FileNotFoundError("mypy")

    monkeypatch.setattr(run_linters_tool.subprocess, "run", fake_run)

    result = run_linters_tool.run_mypy()

    assert result.success is False
    assert result.error_message == "mypy not found - is it installed?"


def test_run_mypy_timeout_reports_error(monkeypatch: pytest.MonkeyPatch):
    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs["timeout"])

    monkeypatch.setattr(run_linters_tool.subprocess, "run", fake_run)

    result = run_linters_tool.run_mypy(timeout=14)

    assert result.success is False
    assert result.error_message == "Timeout after 14 seconds"


def test_run_mypy_unexpected_exception_is_reported(monkeypatch: pytest.MonkeyPatch):
    def fake_run(args, **kwargs):
        raise RuntimeError("mypy boom")

    monkeypatch.setattr(run_linters_tool.subprocess, "run", fake_run)

    result = run_linters_tool.run_mypy()

    assert result.success is False
    assert result.error_message == "mypy boom"


def test_format_summary_reports_fixed_remaining_and_preview_lines():
    passing_result = run_linters_tool.LinterResult("ruff_check")
    passing_result.success = True
    passing_result.issues_fixed = 2
    passing_result.issues_found = 1

    failing_result = run_linters_tool.LinterResult("mypy")
    failing_result.success = False
    failing_result.issues_found = 1
    failing_result.stdout = "a.py:1: error: bad\nFound 1 error\n"

    summary = run_linters_tool.format_summary([passing_result, failing_result], all_passed=False)

    assert "Fixed: 2 issues" in summary
    assert "Remaining: 1 issues" in summary
    assert "Found: 1 issues" in summary
    assert "Preview:" in summary
    assert "RESULT: LINTING FAILED ✗" in summary


def test_format_summary_reports_explicit_error_messages():
    result = run_linters_tool.LinterResult("ruff_check")
    result.success = False
    result.error_message = "Timeout after 10 seconds"

    summary = run_linters_tool.format_summary([result], all_passed=False)

    assert "Error: Timeout after 10 seconds" in summary


def test_format_full_output_includes_stderr_and_summary():
    result = run_linters_tool.LinterResult("ruff_format")
    result.stdout = "formatted output"
    result.stderr = "warning output"

    output = run_linters_tool.format_full_output([result], all_passed=True)

    assert "Ruff Format Output" in output
    assert "formatted output" in output
    assert "Stderr:" in output
    assert "warning output" in output
    assert "RESULT: ALL LINTERS PASSED ✓" in output


def test_run_linters_supports_ruff_format_json_and_discovers_cwd(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_run_ruff_format(target_dir, timeout, cwd):
        captured["target_dir"] = target_dir
        captured["timeout"] = timeout
        captured["cwd"] = cwd
        result = run_linters_tool.LinterResult("ruff_format")
        result.success = True
        result.issues_fixed = 1
        return result

    monkeypatch.setattr(run_linters_tool, "run_ruff_format", fake_run_ruff_format)

    exit_code, output = run_linters_tool.run_linters(
        target_dir="adw/core",
        auto_fix=False,
        linters=["ruff_format"],
        output_mode="json",
        cwd=None,
        ruff_timeout=23,
    )

    assert exit_code == 0
    assert '"name": "ruff_format"' in output
    assert '"all_passed": true' in output
    assert captured["target_dir"] == "adw/core"
    assert captured["timeout"] == 23
    assert captured["cwd"] == str(Path.cwd())


def test_run_linters_full_output_uses_formatter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def fake_run_ruff_check(target_dir, auto_fix, timeout, cwd):
        result = run_linters_tool.LinterResult("ruff_check")
        result.success = True
        return result

    monkeypatch.setattr(run_linters_tool, "run_ruff_check", fake_run_ruff_check)

    exit_code, output = run_linters_tool.run_linters(
        target_dir="adw/core",
        auto_fix=False,
        linters=["ruff"],
        output_mode="full",
        cwd=str(tmp_path),
    )

    assert exit_code == 0
    assert "Ruff Check Output" in output
    assert "RESULT: ALL LINTERS PASSED ✓" in output


def test_main_parses_args_and_prints_tool_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    captured: dict[str, object] = {}

    def fake_run_linters(**kwargs):
        captured.update(kwargs)
        return 0, "mock lint output"

    monkeypatch.setattr(run_linters_tool, "run_linters", fake_run_linters)
    monkeypatch.setattr(
        run_linters_tool.sys,
        "argv",
        [
            "run_linters.py",
            "--output",
            "json",
            "--target-dir",
            "adw/core",
            "--no-auto-fix",
            "--linters",
            "ruff",
            "--cwd",
            "/tmp/worktree",
            "--ruff-timeout",
            "11",
            "--mypy-timeout",
            "22",
        ],
    )

    exit_code = run_linters_tool.main()

    assert exit_code == 0
    assert captured == {
        "target_dir": "adw/core",
        "auto_fix": False,
        "linters": ["ruff"],
        "output_mode": "json",
        "cwd": "/tmp/worktree",
        "ruff_timeout": 11,
        "mypy_timeout": 22,
    }
    assert capsys.readouterr().out.strip() == "mock lint output"
