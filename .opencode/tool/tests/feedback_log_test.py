"""Tests for feedback_log CLI tool."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import adw.utils.feedback as feedback

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "feedback_log.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("feedback_log_cli", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - sanity check
        raise AssertionError("Failed to load CLI module")
    module = importlib.util.module_from_spec(spec)
    sys.modules["feedback_log_cli"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _run_cli(args: list[str]) -> SimpleNamespace:
    module = _load_cli_module()
    stdout_buffer: io.StringIO = io.StringIO()
    stderr_buffer: io.StringIO = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        returncode = module.main(args)
    return SimpleNamespace(
        args=args,
        returncode=returncode,
        stdout=stdout_buffer.getvalue(),
        stderr=stderr_buffer.getvalue(),
    )


def _setup_logger(monkeypatch, tmp_path: Path) -> None:
    original_get_logger = feedback.get_feedback_logger
    monkeypatch.setattr(feedback, "get_feedback_logger", lambda: original_get_logger(tmp_path))


def _read_feedback_entries(tmp_path: Path) -> list[dict[str, object]]:
    log_path = tmp_path / "agents" / "feedback" / "feedback.log"
    assert log_path.exists(), "Expected feedback log file to exist"
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]


def test_main_importable() -> None:
    module = _load_cli_module()

    assert hasattr(module, "main")


def test_build_parser_has_required_args() -> None:
    module = _load_cli_module()
    parser = module._build_parser()

    action_map = {action.dest: action for action in parser._actions}

    assert action_map["category"].required is True
    assert action_map["severity"].required is True
    assert action_map["description"].required is True
    assert parser.get_default("agent_type") == "unknown"


def test_truncate_description_flattens_and_limits() -> None:
    module = _load_cli_module()

    description = "Line one\nLine two"
    assert module._truncate_description(description, limit=50) == "Line one Line two"

    long_description = "x" * 10
    assert module._truncate_description(long_description, limit=3) == "xxx"
    assert module._truncate_description(long_description, limit=5) == "xx..."


def test_validation_error_includes_allowed_lists() -> None:
    module = _load_cli_module()

    message = module._validation_error("Invalid category 'oops'")

    assert message.startswith("Feedback error: Invalid category 'oops'.")
    assert "Valid categories" in message
    assert "Valid severities" in message


def test_log_error_strips_trailing_period() -> None:
    module = _load_cli_module()

    assert module._log_error("disk full.") == "Feedback error: disk full."
    assert module._log_error("disk full") == "Feedback error: disk full."


def test_valid_args_logs_feedback(tmp_path: Path, monkeypatch) -> None:
    _setup_logger(monkeypatch, tmp_path)
    feedback.reset_rate_limiter()

    proc = _run_cli(
        [
            "--category",
            "bug",
            "--severity",
            "high",
            "--description",
            "Crash in tool",
            "--adw-id",
            "adw-123",
            "--agent-type",
            "builder",
        ]
    )

    assert proc.returncode == 0
    assert "Feedback logged, thank you." in proc.stdout

    entries = _read_feedback_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["category"] == "bug"
    assert entries[0]["severity"] == "high"
    assert entries[0]["description"] == "Crash in tool"
    assert entries[0]["adw_id"] == "adw-123"
    assert entries[0]["agent_type"] == "builder"


def test_invalid_category_exits_1() -> None:
    proc = _run_cli(
        [
            "--category",
            "invalid",
            "--severity",
            "low",
            "--description",
            "Nope",
        ]
    )

    assert proc.returncode == 1
    assert "Feedback error: Invalid category" in proc.stdout
    assert "Valid categories" in proc.stdout
    assert "Valid severities" in proc.stdout


def test_invalid_severity_exits_1() -> None:
    proc = _run_cli(
        [
            "--category",
            "bug",
            "--severity",
            "invalid",
            "--description",
            "Nope",
        ]
    )

    assert proc.returncode == 1
    assert "Feedback error: Invalid severity" in proc.stdout
    assert "Valid categories" in proc.stdout
    assert "Valid severities" in proc.stdout


def test_missing_required_args_exits_with_usage() -> None:
    proc = _run_cli(["--category", "bug"])

    assert proc.returncode in {1, 2}
    assert "usage" in proc.stderr.lower()


def test_description_with_quotes_and_newlines(tmp_path: Path, monkeypatch) -> None:
    _setup_logger(monkeypatch, tmp_path)
    feedback.reset_rate_limiter()

    description = 'Line one\n"quoted" line two'
    proc = _run_cli(
        [
            "--category",
            "friction",
            "--severity",
            "low",
            "--description",
            description,
            "--adw-id",
            "adw-456",
            "--agent-type",
            "runner",
        ]
    )

    assert proc.returncode == 0

    entries = _read_feedback_entries(tmp_path)
    assert entries[0]["description"] == description


def test_success_output_format(tmp_path: Path, monkeypatch) -> None:
    _setup_logger(monkeypatch, tmp_path)
    feedback.reset_rate_limiter()

    long_description = "x" * 150
    proc = _run_cli(
        [
            "--category",
            "feature",
            "--severity",
            "medium",
            "--description",
            long_description,
            "--adw-id",
            "adw-789",
            "--agent-type",
            "builder",
        ]
    )

    assert proc.returncode == 0
    assert "Feedback logged, thank you. [feature/medium]" in proc.stdout
    assert proc.stdout.rstrip().endswith("...")


def test_rate_limited_output_format_and_exit_zero(tmp_path: Path, monkeypatch) -> None:
    _setup_logger(monkeypatch, tmp_path)
    feedback.reset_rate_limiter()
    monkeypatch.setattr(feedback.time, "time", lambda: 1234.0)

    args = [
        "--category",
        "bug",
        "--severity",
        "low",
        "--description",
        "Rate limit test",
        "--adw-id",
        "adw-rate",
        "--agent-type",
        "builder",
    ]

    first = _run_cli(args)
    second = _run_cli(args)

    assert first.returncode == 0
    assert second.returncode == 0
    assert "Rate limited — feedback already logged" in second.stdout
    assert "no action needed" in second.stdout.lower()


def test_write_error_exits_2(monkeypatch) -> None:
    module = _load_cli_module()

    def _raise(_entry):
        raise OSError("disk full")

    monkeypatch.setattr(module, "log_feedback", _raise)

    stdout_buffer: io.StringIO = io.StringIO()
    stderr_buffer: io.StringIO = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        returncode = module.main(
            [
                "--category",
                "bug",
                "--severity",
                "high",
                "--description",
                "Failure",
            ]
        )

    assert returncode == 2
    assert "Feedback error: disk full." in stdout_buffer.getvalue()


def test_log_feedback_failure_exits_2(monkeypatch) -> None:
    module = _load_cli_module()

    def _fail(_entry):
        return False, "Failed to write feedback: disk full"

    monkeypatch.setattr(module, "log_feedback", _fail)

    stdout_buffer: io.StringIO = io.StringIO()
    stderr_buffer: io.StringIO = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        returncode = module.main(
            [
                "--category",
                "bug",
                "--severity",
                "high",
                "--description",
                "Failure",
            ]
        )

    assert returncode == 2
    assert "Feedback error: Failed to write feedback: disk full." in stdout_buffer.getvalue()
