#!/usr/bin/env python3
"""Python CLI backend for logging agent feedback.

Validates CLI arguments, constructs a feedback entry, and delegates logging to
``adw.utils.feedback.log_feedback``. Intended for subprocess invocation by
``.opencode/tool/feedback_log.ts``.

Exit codes:
    0: Feedback logged successfully or rate limited.
    1: Validation error (invalid category or severity).
    2: Write error or unexpected failure.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

MAX_DESCRIPTION_LENGTH = 100
RATE_LIMIT_PREFIX = "Rate limited"
RATE_LIMIT_MESSAGE = (
    "Rate limited — feedback already logged for this agent within the last 60s. "
    "Thank you, no action needed."
)
FALLBACK_FEEDBACK_CATEGORIES = ["bug", "feature", "friction", "performance"]
FALLBACK_FEEDBACK_SEVERITIES = ["low", "medium", "high", "critical"]
FALLBACK_RATE_LIMIT_MESSAGE = "Rate limited — please wait before submitting more feedback."
FALLBACK_RATE_LIMIT_WINDOW = timedelta(seconds=60)


@dataclass(frozen=True)
class FeedbackLogResult:
    """Structured result from feedback logging."""

    success: bool
    message: str


@dataclass(frozen=True)
class FeedbackBackend:
    """Container describing the loaded feedback backend."""

    categories: Sequence[str]
    severities: Sequence[str]
    entry_type: type[Any]
    log_feedback: Callable[[Any], FeedbackLogResult]
    source: str


@dataclass(frozen=True)
class _FallbackFeedbackEntry:
    """Fallback feedback entry when adw.utils.feedback is unavailable."""

    timestamp: datetime
    adw_id: str = ""
    workflow_step: str = ""
    agent_type: str = ""
    category: str = ""
    severity: str = ""
    tool_name: str = ""
    description: str = ""
    suggested_fix: str = ""
    context: str = ""

    def __post_init__(self) -> None:
        """Validate fallback entry fields."""
        if self.category not in FALLBACK_FEEDBACK_CATEGORIES:
            raise ValueError(
                "Invalid category "
                f"'{self.category}'. Must be one of: {FALLBACK_FEEDBACK_CATEGORIES}"
            )
        if self.severity not in FALLBACK_FEEDBACK_SEVERITIES:
            raise ValueError(
                "Invalid severity "
                f"'{self.severity}'. Must be one of: {FALLBACK_FEEDBACK_SEVERITIES}"
            )
        if not isinstance(self.timestamp, datetime):
            raise ValueError("Timestamp must be a datetime instance")


_fallback_last_entries: dict[tuple[str, str], _FallbackFeedbackEntry] = {}


def _truncate_description(description: str, limit: int = MAX_DESCRIPTION_LENGTH) -> str:
    """Truncate a description for output messages.

    Args:
        description: Description text to truncate.
        limit: Maximum length for the output text.

    Returns:
        Truncated description with ellipsis when needed.
    """
    flattened = " ".join(description.splitlines()).strip()
    if len(flattened) <= limit:
        return flattened
    if limit <= 3:
        return flattened[:limit]
    return f"{flattened[: limit - 3].rstrip()}..."


def _validation_error(
    reason: str,
    categories: Sequence[str],
    severities: Sequence[str],
) -> str:
    """Build a validation error message.

    Args:
        reason: Human-readable validation failure reason.
        categories: Valid feedback categories.
        severities: Valid feedback severities.

    Returns:
        Complete validation error message.
    """
    return (
        f"Feedback error: {reason}. "
        f"Valid categories: {list(categories)}. "
        f"Valid severities: {list(severities)}."
    )


def _log_error(message: str) -> str:
    """Format a feedback logging error message.

    Args:
        message: Error details from the feedback logger.

    Returns:
        Formatted error message.
    """
    reason = message.rstrip().rstrip(".")
    return f"Feedback error: {reason}."


def _get_repo_root() -> Path:
    """Return the repository root for this tool."""
    return Path(__file__).resolve().parents[2]


def _ensure_repo_root_on_path() -> Path:
    """Ensure the repository root is on sys.path for imports."""
    repo_root = _get_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def _normalize_timestamp(timestamp: datetime) -> datetime:
    """Normalize timestamps to UTC for comparisons."""
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _load_last_log_entry(log_path: Path) -> _FallbackFeedbackEntry | None:
    """Load the last valid feedback entry from the fallback log file.

    Args:
        log_path: Path to the fallback feedback log file.

    Returns:
        The most recent valid fallback feedback entry, or None if unavailable.
    """
    if not log_path.exists():
        return None

    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue

        timestamp_value = payload.get("timestamp")
        if not timestamp_value:
            continue
        try:
            payload["timestamp"] = datetime.fromisoformat(timestamp_value)
        except (TypeError, ValueError):
            continue

        try:
            return _FallbackFeedbackEntry(**payload)
        except (TypeError, ValueError):
            continue
    return None


def _is_rate_limited(
    last_entry: _FallbackFeedbackEntry | None,
    new_entry: _FallbackFeedbackEntry,
    window: timedelta,
) -> bool:
    """Return True if the new entry should be rate limited.

    Args:
        last_entry: Most recent logged entry, if any.
        new_entry: Entry being evaluated.
        window: Rate limit interval.

    Returns:
        True when the entry is within the rate limit window for the same key.
    """
    if last_entry is None:
        return False
    if last_entry.adw_id != new_entry.adw_id:
        return False
    if last_entry.agent_type != new_entry.agent_type:
        return False

    last_timestamp = _normalize_timestamp(last_entry.timestamp)
    new_timestamp = _normalize_timestamp(new_entry.timestamp)
    return new_timestamp - last_timestamp < window


def _coerce_fallback_entry(
    entry: _FallbackFeedbackEntry | dict[str, Any],
) -> _FallbackFeedbackEntry:
    """Convert raw fallback payloads into a typed fallback entry."""
    if isinstance(entry, _FallbackFeedbackEntry):
        return entry
    return _FallbackFeedbackEntry(**entry)


def _fallback_log_feedback(
    entry: _FallbackFeedbackEntry | dict[str, Any],
    *,
    log_dir: Path,
) -> FeedbackLogResult:
    """Fallback logger that writes JSONL feedback locally.

    Args:
        entry: Fallback entry payload to log.
        log_dir: Directory to write feedback log files into.

    Returns:
        FeedbackLogResult describing success or failure.
    """
    entry_obj = _coerce_fallback_entry(entry)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "feedback.log"

    key = (entry_obj.adw_id, entry_obj.agent_type)
    last_entry = _fallback_last_entries.get(key) or _load_last_log_entry(log_path)
    if _is_rate_limited(last_entry, entry_obj, FALLBACK_RATE_LIMIT_WINDOW):
        return FeedbackLogResult(False, FALLBACK_RATE_LIMIT_MESSAGE)

    payload = {
        "timestamp": _normalize_timestamp(entry_obj.timestamp).isoformat(),
        "adw_id": entry_obj.adw_id,
        "workflow_step": entry_obj.workflow_step,
        "agent_type": entry_obj.agent_type,
        "category": entry_obj.category,
        "severity": entry_obj.severity,
        "tool_name": entry_obj.tool_name,
        "description": entry_obj.description,
        "suggested_fix": entry_obj.suggested_fix,
        "context": entry_obj.context,
    }

    try:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload))
            handle.write("\n")
    except OSError as exc:
        return FeedbackLogResult(False, f"Failed to write feedback: {exc}")
    except Exception as exc:  # pragma: no cover - defensive
        return FeedbackLogResult(False, f"Failed to write feedback: {exc}")

    _fallback_last_entries[key] = entry_obj
    return FeedbackLogResult(True, "Feedback logged, thank you.")


def _wrap_feedback_logger(
    func: Callable[[Any], tuple[bool, str]],
) -> Callable[[Any], FeedbackLogResult]:
    """Wrap a tuple-returning feedback logger with a structured result."""

    def _wrapped(entry: Any) -> FeedbackLogResult:
        success, message = func(entry)
        return FeedbackLogResult(success, message)

    return _wrapped


def _load_feedback_backend() -> FeedbackBackend:
    """Load the feedback backend, falling back to a local implementation.

    Attempts to import ``adw.utils.feedback`` after ensuring the repo root is on
    ``sys.path``. If import fails, returns a fallback backend that writes JSONL
    feedback to ``agents/feedback/feedback.log`` and enforces rate limiting based
    on the most recent logged entry.

    Returns:
        FeedbackBackend describing the active feedback backend.
    """
    _ensure_repo_root_on_path()
    try:
        feedback_module = importlib.import_module("adw.utils.feedback")
    except ModuleNotFoundError:
        log_dir = _get_repo_root() / "agents" / "feedback"
        return FeedbackBackend(
            categories=FALLBACK_FEEDBACK_CATEGORIES,
            severities=FALLBACK_FEEDBACK_SEVERITIES,
            entry_type=_FallbackFeedbackEntry,
            log_feedback=lambda entry: _fallback_log_feedback(entry, log_dir=log_dir),
            source="fallback",
        )

    return FeedbackBackend(
        categories=feedback_module.FEEDBACK_CATEGORIES,
        severities=feedback_module.FEEDBACK_SEVERITIES,
        entry_type=feedback_module.FeedbackEntry,
        log_feedback=_wrap_feedback_logger(feedback_module.log_feedback),
        source="adw",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for feedback logging.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description="Log feedback for an ADW agent run.")
    parser.add_argument("--category", required=True, help="Feedback category")
    parser.add_argument("--severity", required=True, help="Feedback severity")
    parser.add_argument("--description", required=True, help="Issue description")
    parser.add_argument("--suggested-fix", default="", help="Suggested fix")
    parser.add_argument("--tool-name", default="", help="Tool that triggered feedback")
    parser.add_argument("--workflow-step", default="", help="Workflow step name")
    parser.add_argument("--agent-type", default="unknown", help="Agent type")
    parser.add_argument("--adw-id", default="", help="ADW workflow ID")
    parser.add_argument("--context", default="", help="Additional context")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for feedback logging.

    Args:
        argv: Optional CLI argument sequence for testing.

    Returns:
        Exit code for the CLI execution.
    """
    backend = _load_feedback_backend()
    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 2

    if args.category not in backend.categories:
        print(
            _validation_error(
                f"Invalid category '{args.category}'", backend.categories, backend.severities
            )
        )
        return 1
    if args.severity not in backend.severities:
        print(
            _validation_error(
                f"Invalid severity '{args.severity}'", backend.categories, backend.severities
            )
        )
        return 1

    entry = backend.entry_type(
        timestamp=datetime.now(timezone.utc),
        adw_id=args.adw_id,
        workflow_step=args.workflow_step,
        agent_type=args.agent_type,
        category=args.category,
        severity=args.severity,
        tool_name=args.tool_name,
        description=args.description,
        suggested_fix=args.suggested_fix,
        context=args.context,
    )

    try:
        result = backend.log_feedback(entry)
    except Exception as exc:  # noqa: BLE001
        print(_log_error(str(exc)))
        return 2

    if result.success:
        truncated = _truncate_description(args.description)
        print(f"Feedback logged, thank you. [{args.category}/{args.severity}] {truncated}")
        return 0

    if result.message.startswith(RATE_LIMIT_PREFIX):
        print(RATE_LIMIT_MESSAGE)
        return 0

    print(_log_error(result.message))
    return 2


if __name__ == "__main__":
    sys.exit(main())
