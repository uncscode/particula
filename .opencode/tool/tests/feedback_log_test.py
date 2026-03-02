"""Tests for feedback_log CLI tool."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import adw.utils.feedback as feedback

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "feedback_log.py"


def _load_cli_module():
    sys.modules.pop("feedback_log_cli", None)
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


def _reset_fallback_cache(module) -> None:
    module._fallback_last_entries.clear()


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

    message = module._validation_error(
        "Invalid category 'oops'",
        ["bug"],
        ["low"],
    )

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


def test_load_last_log_entry_missing_file(tmp_path: Path) -> None:
    module = _load_cli_module()
    log_path = tmp_path / "feedback.log"

    assert module._load_last_log_entry(log_path) is None


def test_load_last_log_entry_empty_file(tmp_path: Path) -> None:
    module = _load_cli_module()
    log_path = tmp_path / "feedback.log"
    log_path.write_text("", encoding="utf-8")

    assert module._load_last_log_entry(log_path) is None


def test_load_last_log_entry_oserror(monkeypatch, tmp_path: Path) -> None:
    module = _load_cli_module()
    log_path = tmp_path / "feedback.log"
    log_path.write_text("{}", encoding="utf-8")

    def _raise(_self, *args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr(Path, "read_text", _raise)

    assert module._load_last_log_entry(log_path) is None


def test_load_last_log_entry_all_corrupt(tmp_path: Path) -> None:
    module = _load_cli_module()
    log_path = tmp_path / "feedback.log"
    log_path.write_text('not json\n{}\n{"timestamp": "bad"}\n', encoding="utf-8")

    assert module._load_last_log_entry(log_path) is None


def test_is_rate_limited_false_for_different_adw_id() -> None:
    module = _load_cli_module()
    now = datetime.now(timezone.utc)
    last_entry = module._FallbackFeedbackEntry(
        timestamp=now - timedelta(seconds=30),
        adw_id="adw-1",
        agent_type="builder",
        category="bug",
        severity="low",
        description="first",
    )
    new_entry = module._FallbackFeedbackEntry(
        timestamp=now,
        adw_id="adw-2",
        agent_type="builder",
        category="bug",
        severity="low",
        description="second",
    )

    assert not module._is_rate_limited(
        last_entry,
        new_entry,
        module.FALLBACK_RATE_LIMIT_WINDOW,
    )


def test_is_rate_limited_false_for_different_agent_type() -> None:
    module = _load_cli_module()
    now = datetime.now(timezone.utc)
    last_entry = module._FallbackFeedbackEntry(
        timestamp=now - timedelta(seconds=30),
        adw_id="adw-1",
        agent_type="builder",
        category="bug",
        severity="low",
        description="first",
    )
    new_entry = module._FallbackFeedbackEntry(
        timestamp=now,
        adw_id="adw-1",
        agent_type="planner",
        category="bug",
        severity="low",
        description="second",
    )

    assert not module._is_rate_limited(
        last_entry,
        new_entry,
        module.FALLBACK_RATE_LIMIT_WINDOW,
    )


def test_is_rate_limited_false_outside_window() -> None:
    module = _load_cli_module()
    now = datetime.now(timezone.utc)
    last_entry = module._FallbackFeedbackEntry(
        timestamp=now - timedelta(seconds=120),
        adw_id="adw-1",
        agent_type="builder",
        category="bug",
        severity="low",
        description="first",
    )
    new_entry = module._FallbackFeedbackEntry(
        timestamp=now,
        adw_id="adw-1",
        agent_type="builder",
        category="bug",
        severity="low",
        description="second",
    )

    assert not module._is_rate_limited(
        last_entry,
        new_entry,
        module.FALLBACK_RATE_LIMIT_WINDOW,
    )


def test_is_rate_limited_true_within_window() -> None:
    module = _load_cli_module()
    now = datetime.now(timezone.utc)
    last_entry = module._FallbackFeedbackEntry(
        timestamp=now - timedelta(seconds=10),
        adw_id="adw-1",
        agent_type="builder",
        category="bug",
        severity="low",
        description="first",
    )
    new_entry = module._FallbackFeedbackEntry(
        timestamp=now,
        adw_id="adw-1",
        agent_type="builder",
        category="bug",
        severity="low",
        description="second",
    )

    assert module._is_rate_limited(
        last_entry,
        new_entry,
        module.FALLBACK_RATE_LIMIT_WINDOW,
    )


def test_write_error_exits_2(monkeypatch) -> None:
    module = _load_cli_module()

    def _raise(_entry):
        raise OSError("disk full")

    backend = module.FeedbackBackend(
        categories=module.FALLBACK_FEEDBACK_CATEGORIES,
        severities=module.FALLBACK_FEEDBACK_SEVERITIES,
        entry_type=module._FallbackFeedbackEntry,
        log_feedback=_raise,
        source="test",
    )
    monkeypatch.setattr(module, "_load_feedback_backend", lambda: backend)

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
        return module.FeedbackLogResult(False, "Failed to write feedback: disk full")

    backend = module.FeedbackBackend(
        categories=module.FALLBACK_FEEDBACK_CATEGORIES,
        severities=module.FALLBACK_FEEDBACK_SEVERITIES,
        entry_type=module._FallbackFeedbackEntry,
        log_feedback=_fail,
        source="test",
    )
    monkeypatch.setattr(module, "_load_feedback_backend", lambda: backend)

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


def test_get_repo_root_matches_feedback_log_location() -> None:
    """Ensure repository root resolves relative to feedback_log.py."""
    module = _load_cli_module()

    expected = SCRIPT_PATH.resolve().parents[2]

    assert module._get_repo_root() == expected


def test_ensure_repo_root_on_path_inserts_root(monkeypatch) -> None:
    """Ensure repo root is inserted at the front of sys.path."""
    module = _load_cli_module()
    monkeypatch.setattr(sys, "path", ["/tmp"], raising=False)

    repo_root = module._ensure_repo_root_on_path()

    assert sys.path[0] == str(repo_root)


def test_normalize_timestamp_handles_naive_and_aware() -> None:
    """Normalize timestamps to timezone-aware UTC values."""
    module = _load_cli_module()
    naive = datetime(2024, 1, 1, 12, 0, 0)
    aware = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone(timedelta(hours=-5)))

    naive_normalized = module._normalize_timestamp(naive)
    aware_normalized = module._normalize_timestamp(aware)

    assert naive_normalized.tzinfo == timezone.utc
    assert aware_normalized.tzinfo == timezone.utc
    assert aware_normalized.hour == 17


def test_fallback_feedback_entry_rejects_invalid_category() -> None:
    """Reject fallback entries with invalid categories."""
    module = _load_cli_module()

    with pytest.raises(ValueError, match="Invalid category"):
        module._FallbackFeedbackEntry(
            timestamp=datetime.now(timezone.utc),
            adw_id="adw-1",
            agent_type="builder",
            category="oops",
            severity="low",
            description="bad",
        )


def test_fallback_feedback_entry_rejects_invalid_severity() -> None:
    """Reject fallback entries with invalid severities."""
    module = _load_cli_module()

    with pytest.raises(ValueError, match="Invalid severity"):
        module._FallbackFeedbackEntry(
            timestamp=datetime.now(timezone.utc),
            adw_id="adw-1",
            agent_type="builder",
            category="bug",
            severity="oops",
            description="bad",
        )


def test_fallback_feedback_entry_rejects_non_datetime_timestamp() -> None:
    """Reject fallback entries with non-datetime timestamps."""
    module = _load_cli_module()

    with pytest.raises(ValueError, match="Timestamp must be a datetime"):
        module._FallbackFeedbackEntry(
            timestamp="not-a-datetime",
            adw_id="adw-1",
            agent_type="builder",
            category="bug",
            severity="low",
            description="bad",
        )


def test_coerce_fallback_entry_returns_existing_instance() -> None:
    """Coerce returns the same fallback entry instance."""
    module = _load_cli_module()
    entry = module._FallbackFeedbackEntry(
        timestamp=datetime.now(timezone.utc),
        adw_id="adw-1",
        agent_type="builder",
        category="bug",
        severity="low",
        description="ok",
    )

    assert module._coerce_fallback_entry(entry) is entry


def test_coerce_fallback_entry_from_dict() -> None:
    """Coerce converts dictionary payloads into fallback entries."""
    module = _load_cli_module()
    payload = {
        "timestamp": datetime.now(timezone.utc),
        "adw_id": "adw-1",
        "agent_type": "builder",
        "category": "bug",
        "severity": "low",
        "description": "ok",
    }

    entry = module._coerce_fallback_entry(payload)

    assert isinstance(entry, module._FallbackFeedbackEntry)
    assert entry.adw_id == "adw-1"


def test_fallback_log_feedback_writes_entry(tmp_path: Path) -> None:
    """Fallback logging writes JSONL and stores the last entry in memory."""
    module = _load_cli_module()
    _reset_fallback_cache(module)

    entry = module._FallbackFeedbackEntry(
        timestamp=datetime.now(timezone.utc),
        adw_id="adw-1",
        agent_type="builder",
        category="bug",
        severity="low",
        description="ok",
    )

    result = module._fallback_log_feedback(entry, log_dir=tmp_path / "agents" / "feedback")

    assert result.success is True

    log_path = tmp_path / "agents" / "feedback" / "feedback.log"
    payloads = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert payloads[0]["adw_id"] == "adw-1"
    assert module._fallback_last_entries[("adw-1", "builder")] is entry


def test_fallback_log_feedback_rate_limits_within_window(tmp_path: Path) -> None:
    """Fallback logging rate limits entries with the same key within the window."""
    module = _load_cli_module()
    _reset_fallback_cache(module)

    base_time = datetime.now(timezone.utc)
    entry = module._FallbackFeedbackEntry(
        timestamp=base_time,
        adw_id="adw-1",
        agent_type="builder",
        category="bug",
        severity="low",
        description="ok",
    )
    newer_entry = module._FallbackFeedbackEntry(
        timestamp=base_time + timedelta(seconds=30),
        adw_id="adw-1",
        agent_type="builder",
        category="bug",
        severity="low",
        description="ok",
    )

    log_dir = tmp_path / "agents" / "feedback"
    first = module._fallback_log_feedback(entry, log_dir=log_dir)
    second = module._fallback_log_feedback(newer_entry, log_dir=log_dir)

    assert first.success is True
    assert second.success is False
    assert second.message == module.FALLBACK_RATE_LIMIT_MESSAGE

    log_path = log_dir / "feedback.log"
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 1


def test_wrap_feedback_logger_returns_structured_result() -> None:
    """Wrap tuple-returning feedback loggers in FeedbackLogResult."""
    module = _load_cli_module()

    def _logger(entry):
        assert entry == "payload"
        return True, "ok"

    wrapped = module._wrap_feedback_logger(_logger)
    result = wrapped("payload")

    assert result.success is True
    assert result.message == "ok"


def test_load_feedback_backend_fallback_when_module_missing(monkeypatch, tmp_path: Path) -> None:
    """Fallback backend loads when adw.utils.feedback is unavailable."""
    module = _load_cli_module()

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("missing")),
    )
    monkeypatch.setattr(module, "_get_repo_root", lambda: tmp_path)

    backend = module._load_feedback_backend()

    assert backend.source == "fallback"
    assert backend.categories == module.FALLBACK_FEEDBACK_CATEGORIES
    assert backend.severities == module.FALLBACK_FEEDBACK_SEVERITIES
    assert backend.entry_type is module._FallbackFeedbackEntry

    entry = module._FallbackFeedbackEntry(
        timestamp=datetime.now(timezone.utc),
        adw_id="adw-1",
        agent_type="builder",
        category="bug",
        severity="low",
        description="ok",
    )
    result = backend.log_feedback(entry)
    assert result.success is True


def test_load_feedback_backend_uses_adw_module(monkeypatch) -> None:
    """Backend loads real module when adw.utils.feedback is importable."""
    module = _load_cli_module()

    @dataclass(frozen=True)
    class DummyEntry:
        timestamp: datetime
        adw_id: str
        agent_type: str
        category: str
        severity: str
        description: str

    def _log_feedback(_entry):
        return True, "ok"

    dummy_module = SimpleNamespace(
        FEEDBACK_CATEGORIES=["bug"],
        FEEDBACK_SEVERITIES=["low"],
        FeedbackEntry=DummyEntry,
        log_feedback=_log_feedback,
    )
    monkeypatch.setattr(importlib, "import_module", lambda _name: dummy_module)

    backend = module._load_feedback_backend()

    assert backend.source == "adw"
    assert backend.entry_type is DummyEntry
    result = backend.log_feedback(
        DummyEntry(
            timestamp=datetime.now(timezone.utc),
            adw_id="adw-1",
            agent_type="builder",
            category="bug",
            severity="low",
            description="ok",
        )
    )
    assert result.success is True
