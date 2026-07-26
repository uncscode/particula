#!/usr/bin/env python3
"""Python CLI backend for logging and reading agent feedback.

This backend powers ``.opencode/tools/feedback_log.ts``. Write mode validates
the required feedback contract fields (category, severity, description,
workflow step, agent type, and ADW ID), constructs a backend entry, and
delegates to ``adw.utils.feedback.log_feedback`` when available. Read mode
returns paginated JSON envelopes from the canonical feedback log with optional
severity filtering. When the shared backend cannot be imported, the CLI falls
back to verified local JSONL logging with deterministic rate limiting,
rotation, and exit codes.

Exit codes:
    0: Feedback logged successfully, was rate limited, or read-mode succeeded.
    1: Validation error.
    2: Write error or unexpected failure.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib
import json
import os
import stat
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

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
LOCK_ACQUIRE_TIMEOUT_SECONDS = 2.0
LOCK_ACQUIRE_RETRY_INTERVAL_SECONDS = 0.05
FALLBACK_WRITE_PREFIX = "Failed to write feedback: "
READ_DEFAULT_PAGE = 1
READ_DEFAULT_PAGE_SIZE = 50
READ_MAX_PAGE_SIZE = 500
FALLBACK_MAX_LOG_BYTES = 1_000_000
FALLBACK_MAX_ROTATED_BACKUPS = 4
FALLBACK_MAX_ENTRY_BYTES = FALLBACK_MAX_LOG_BYTES


@dataclass(frozen=True)
class FeedbackLogResult:
    """Structured result returned by a feedback logging backend.

    Attributes:
        success: Whether the backend accepted the feedback entry.
        message: Human-readable backend status or error message.
    """

    success: bool
    message: str


@dataclass(frozen=True)
class FeedbackBackend:
    """Container describing the active feedback backend.

    Attributes:
        categories: Accepted feedback categories for validation.
        severities: Accepted feedback severities for validation.
        entry_type: Backend-specific feedback entry type to instantiate.
        log_feedback: Callable that persists an entry and returns a structured
            result.
        source: Identifier describing whether the backend is native or fallback.
    """

    categories: Sequence[str]
    severities: Sequence[str]
    entry_type: type[Any]
    log_feedback: Callable[[Any], FeedbackLogResult]
    source: str


@dataclass(frozen=True)
class _FallbackFeedbackEntry:
    """Fallback feedback entry when ``adw.utils.feedback`` is unavailable.

    Attributes:
        timestamp: UTC timestamp for the feedback submission.
        adw_id: Workflow identifier used for scoping and rate limiting.
        workflow_step: Workflow step associated with the feedback item.
        agent_type: Agent name or role that submitted the feedback.
        category: Feedback category validated against fallback allowlists.
        severity: Feedback severity validated against fallback allowlists.
        tool_name: Optional tool name associated with the issue.
        description: Human-readable description of the issue.
        suggested_fix: Optional proposed resolution.
        context: Optional extra execution context.
    """

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
    """Return the repository root for this tool.

    Returns:
        Repository root resolved from this file location.
    """
    return Path(__file__).resolve().parents[2]


def _ensure_repo_root_on_path() -> Path:
    """Ensure the repository root is on ``sys.path`` for imports.

    Returns:
        Repository root that was ensured on ``sys.path``.
    """
    repo_root = _get_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def _normalize_timestamp(timestamp: datetime) -> datetime:
    """Normalize timestamps to UTC for comparisons.

    Args:
        timestamp: Timestamp to normalize.

    Returns:
        UTC-aware timestamp suitable for comparison operations.
    """
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


@contextlib.contextmanager
def _exclusive_lock(lock_path: Path):
    """Acquire an exclusive advisory lock for fallback feedback writes.

    Args:
        lock_path: Path to the lock file.

    Yields:
        None while the exclusive lock is held.

    Raises:
        OSError: If the lock path is unsafe or cannot be opened.
        TimeoutError: If the lock cannot be acquired before the timeout.
    """
    lock_path = _normalize_and_validate_log_path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    lock_fd = os.open(lock_path, flags, 0o644)
    try:
        descriptor_stat = os.fstat(lock_fd)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise OSError(f"feedback lock path must be a regular file: {lock_path}")
        with os.fdopen(lock_fd, "a", encoding="utf-8") as lock_handle:
            lock_fd = -1
            deadline = time.monotonic() + LOCK_ACQUIRE_TIMEOUT_SECONDS
            while True:
                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Failed to acquire feedback log lock "
                            f"within {LOCK_ACQUIRE_TIMEOUT_SECONDS:.1f}s"
                        )
                    time.sleep(LOCK_ACQUIRE_RETRY_INTERVAL_SECONDS)
            try:
                yield
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    finally:
        if lock_fd >= 0:
            os.close(lock_fd)


def _read_tail_lines_bytes(log_path: Path, max_bytes: int = 64 * 1024) -> list[bytes]:
    """Read trailing log lines using byte offsets for robust tail parsing.

    Args:
        log_path: Path to the log file.
        max_bytes: Maximum number of trailing bytes to inspect.

    Returns:
        Log lines from the file tail as raw bytes.
    """
    with log_path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        file_size = handle.tell()
        if file_size <= 0:
            return []

        read_size = min(file_size, max_bytes)
        handle.seek(file_size - read_size, os.SEEK_SET)
        chunk = handle.read(read_size)

    if read_size < file_size:
        _, _sep, chunk = chunk.partition(b"\n")

    return chunk.splitlines()


def _atomic_append_json_line(log_path: Path, payload_line: str) -> None:
    """Append one JSONL record using append-only write + fsync.

    The caller holds the exclusive file lock. The final open uses ``O_NOFOLLOW``
    when available and validates the opened descriptor before writing to close
    symlink replacement races between path validation and append.

    Args:
        log_path: Canonical feedback log path to append to.
        payload_line: Serialized JSON object without a trailing newline.

    Raises:
        OSError: If the target cannot be opened safely or written successfully.
    """
    encoded_line = payload_line.encode("utf-8") + b"\n"
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(log_path, flags, 0o644)
    try:
        descriptor_stat = os.fstat(fd)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise OSError(f"feedback log append target must be a regular file: {log_path}")
        os.write(fd, encoded_line)
        os.fsync(fd)
    finally:
        os.close(fd)


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
        log_path = _validate_existing_log_file(log_path)
        lines = _read_tail_lines_bytes(log_path)
    except OSError:
        return None

    for line in reversed(lines):
        if not line.strip():
            continue
        decoded = line.decode("utf-8", errors="replace")
        try:
            payload = json.loads(decoded)
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


def _load_last_log_entry_for_key(
    log_path: Path,
    *,
    adw_id: str,
    agent_type: str,
) -> _FallbackFeedbackEntry | None:
    """Load the latest fallback log entry matching ``(adw_id, agent_type)``.

    Args:
        log_path: Path to the fallback feedback log file.
        adw_id: Workflow identifier to match.
        agent_type: Agent type to match.

    Returns:
        Most recent matching fallback feedback entry, or None if unavailable.
    """
    if not log_path.exists():
        return None

    try:
        log_path = _validate_existing_log_file(log_path)
        lines = _read_tail_lines_bytes(log_path)
    except OSError:
        return None

    for line in reversed(lines):
        if not line.strip():
            continue
        decoded = line.decode("utf-8", errors="replace")
        try:
            payload = json.loads(decoded)
        except json.JSONDecodeError:
            continue

        if payload.get("adw_id", "") != adw_id:
            continue
        if payload.get("agent_type", "") != agent_type:
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


def _latest_entry_for_key(
    in_memory_entry: _FallbackFeedbackEntry | None,
    persisted_entry: _FallbackFeedbackEntry | None,
) -> _FallbackFeedbackEntry | None:
    """Choose the latest entry for a key between memory cache and persisted log.

    Args:
        in_memory_entry: Cached in-memory entry for the key.
        persisted_entry: Most recent persisted entry for the key.

    Returns:
        Newest available entry, or None when neither source has one.
    """
    if in_memory_entry is None:
        return persisted_entry
    if persisted_entry is None:
        return in_memory_entry

    in_memory_ts = _normalize_timestamp(in_memory_entry.timestamp)
    persisted_ts = _normalize_timestamp(persisted_entry.timestamp)
    if in_memory_ts >= persisted_ts:
        return in_memory_entry
    return persisted_entry


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
    """Convert raw fallback payloads into a typed fallback entry.

    Args:
        entry: Existing fallback entry or raw mapping payload.

    Returns:
        Normalized fallback entry instance.
    """
    if isinstance(entry, _FallbackFeedbackEntry):
        return entry
    return _FallbackFeedbackEntry(**entry)


def _format_fallback_write_error(exc: BaseException) -> str:
    """Return a normalized fallback write error with a single prefix.

    Args:
        exc: Exception raised during fallback logging.

    Returns:
        User-facing error message with canonical prefix handling.
    """
    reason = str(exc).strip() or exc.__class__.__name__
    lowered_prefix = FALLBACK_WRITE_PREFIX.lower()
    if reason.lower().startswith(lowered_prefix):
        reason = reason[len(FALLBACK_WRITE_PREFIX) :].lstrip()
    return f"{FALLBACK_WRITE_PREFIX}{reason}"


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

    lock_path = log_dir / "feedback.log.lock"

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
        with _exclusive_lock(lock_path):
            last_entry = _latest_entry_for_key(
                _fallback_last_entries.get(key),
                _load_last_log_entry_for_key(log_path, adw_id=key[0], agent_type=key[1]),
            )
            if _is_rate_limited(last_entry, entry_obj, FALLBACK_RATE_LIMIT_WINDOW):
                return FeedbackLogResult(False, FALLBACK_RATE_LIMIT_MESSAGE)
            _append_json_line(log_path, json.dumps(payload))
            _verify_appended_payload(log_path, json.dumps(payload))
    except (OSError, TimeoutError) as exc:
        return FeedbackLogResult(False, _format_fallback_write_error(exc))
    except Exception as exc:  # pragma: no cover - defensive
        return FeedbackLogResult(False, _format_fallback_write_error(exc))

    _fallback_last_entries[key] = entry_obj
    return FeedbackLogResult(True, "Feedback logged, thank you.")


def _wrap_feedback_logger(
    func: Callable[[Any], tuple[bool, str]],
) -> Callable[[Any], FeedbackLogResult]:
    """Wrap a tuple-returning feedback logger with a structured result.

    Args:
        func: Logger callable that returns ``(success, message)``.

    Returns:
        Wrapper that converts tuple results into ``FeedbackLogResult``.
    """

    def _wrapped(entry: Any) -> FeedbackLogResult:
        """Convert tuple logger results into ``FeedbackLogResult``.

        Args:
            entry: Feedback entry object passed through to the backend logger.

        Returns:
            Structured feedback logging result.
        """
        success, message = func(entry)
        return FeedbackLogResult(success, message)

    return _wrapped


def _load_feedback_backend() -> FeedbackBackend:
    """Load the feedback backend, falling back to a local implementation.

    Attempts to import ``adw.utils.feedback`` after ensuring the repo root is on
    ``sys.path``. If import fails, returns a fallback backend that writes JSONL
    feedback to ``adforge_local/agents/feedback/feedback.log`` and enforces rate
    limiting based on the most recent logged entry.

    Returns:
        FeedbackBackend describing the active feedback backend.
    """
    _ensure_repo_root_on_path()
    try:
        feedback_module = importlib.import_module("adw.utils.feedback")
    except ModuleNotFoundError as exc:
        if exc.name != "adw.utils.feedback":
            raise
        log_dir = _get_repo_root() / "adforge_local" / "agents" / "feedback"
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

    The parser supports read and write modes. Write-mode validation is completed
    in ``main()`` so the CLI can return deterministic error messages for the
    required contract fields ``--workflow-step``, ``--agent-type``, and
    ``--adw-id`` alongside the existing required write inputs.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description="Log feedback for an ADW agent run.")
    parser.add_argument("--command", choices=["write", "read"], default="write")
    parser.add_argument("--category", help="Feedback category")
    parser.add_argument("--severity", help="Feedback severity")
    parser.add_argument("--description", help="Issue description")
    parser.add_argument("--suggested-fix", default="", help="Suggested fix")
    parser.add_argument("--tool-name", default="", help="Tool that triggered feedback")
    parser.add_argument("--workflow-step", default="", help="Workflow step name")
    parser.add_argument("--agent-type", default="", help="Agent type")
    parser.add_argument("--adw-id", default="", help="ADW workflow ID")
    parser.add_argument("--context", default="", help="Additional context")
    parser.add_argument("--page", type=int, default=READ_DEFAULT_PAGE, help="Read mode page number")
    parser.add_argument(
        "--page-size",
        type=int,
        default=READ_DEFAULT_PAGE_SIZE,
        help="Read mode page size",
    )
    parser.add_argument(
        "--severity-filter",
        choices=FALLBACK_FEEDBACK_SEVERITIES,
        help="Optional severity filter for read mode",
    )
    return parser


def _resolve_project_root() -> Path:
    """Resolve the project root, preferring shared path helpers when available.

    Returns:
        Canonical project root for the current worktree.
    """
    _ensure_repo_root_on_path()
    try:
        from adw.utils.paths import get_project_root

        return Path(get_project_root()).resolve()
    except Exception:
        return _get_repo_root().resolve()


def _resolve_feedback_log_path() -> Path:
    """Resolve the canonical feedback log path.

    Prefers ``adforge_local/agents/feedback/feedback.log`` via shared path
    helpers when available, falling back to the same canonical path rooted at
    the resolved project root.

    Returns:
        Resolved path to the feedback log file.
    """
    _ensure_repo_root_on_path()
    try:
        from adw.utils.paths import get_adforge_local_agents_dir

        return get_adforge_local_agents_dir() / "feedback" / "feedback.log"
    except Exception:
        return _resolve_project_root() / "adforge_local" / "agents" / "feedback" / "feedback.log"


def _normalize_and_validate_log_path(log_path: Path) -> Path:
    """Normalize and validate that a log path is safe and repo-confined.

    Args:
        log_path: Candidate feedback log path.

    Returns:
        Normalized absolute path confined to the repository root.

    Raises:
        OSError: If the path is symlinked or resolves outside the repository.
    """
    root = _resolve_project_root().resolve()
    candidate_raw = log_path.expanduser()
    if not candidate_raw.is_absolute():
        candidate_raw = root / candidate_raw

    try:
        parts = candidate_raw.relative_to(root).parts
    except ValueError:
        parts = ()

    cursor = root
    for part in parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise OSError(f"feedback log path component must not be symlinked: {cursor}")

    candidate = candidate_raw.resolve()

    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise OSError(f"feedback log path resolves outside repository root: {candidate}") from exc

    return candidate


def _validate_existing_log_file(log_path: Path) -> Path:
    """Validate an existing feedback log path before reads or rotation.

    Args:
        log_path: Feedback log path to validate.

    Returns:
        Normalized validated log path.

    Raises:
        OSError: If the existing path is not a regular file.
    """
    normalized = _normalize_and_validate_log_path(log_path)
    if normalized.exists():
        path_stat = normalized.stat()
        if not stat.S_ISREG(path_stat.st_mode):
            raise OSError(f"feedback log path must be a regular file: {normalized}")
    return normalized


def _rotate_feedback_logs(log_path: Path) -> None:
    """Shift numbered backups down and rotate active log to ``.1``.

    Args:
        log_path: Active feedback log path.
    """
    candidates: list[tuple[int, Path]] = []
    for rotated in log_path.parent.glob(f"{log_path.name}.*"):
        suffix = rotated.name.removeprefix(f"{log_path.name}.")
        if suffix.isdigit():
            validated = _validate_existing_log_file(rotated)
            candidates.append((int(suffix), validated))

    for index, rotated in sorted(candidates, key=lambda item: item[0], reverse=True):
        os.replace(rotated, log_path.parent / f"{log_path.name}.{index + 1}")

    os.replace(log_path, log_path.parent / f"{log_path.name}.1")
    _prune_rotated_backups(log_path)


def _prune_rotated_backups(log_path: Path, *, max_backups: int | None = None) -> None:
    """Prune rotated backups beyond the configured retention count.

    Args:
        log_path: Active feedback log path whose numbered backups are pruned.
        max_backups: Maximum number of rotated backups to retain.
    """
    if max_backups is None:
        max_backups = FALLBACK_MAX_ROTATED_BACKUPS
    if max_backups < 1:
        return

    candidates: list[tuple[int, Path]] = []
    for rotated in log_path.parent.glob(f"{log_path.name}.*"):
        suffix = rotated.name.removeprefix(f"{log_path.name}.")
        if suffix.isdigit():
            validated = _validate_existing_log_file(rotated)
            candidates.append((int(suffix), validated))

    for index, rotated in sorted(candidates, key=lambda item: item[0]):
        if index > max_backups:
            rotated.unlink(missing_ok=True)


def _append_json_line(log_path: Path, payload_line: str) -> None:
    """Append a JSONL line and rotate when size budget would be exceeded.

    Args:
        log_path: Target feedback log path.
        payload_line: Serialized JSON line to append.

    """
    normalized = _normalize_and_validate_log_path(log_path)
    normalized.parent.mkdir(parents=True, exist_ok=True)

    new_line_size = len(payload_line.encode("utf-8")) + 1
    if new_line_size > FALLBACK_MAX_ENTRY_BYTES:
        raise OSError(
            "feedback log entry exceeds maximum size: "
            f"{new_line_size} bytes > {FALLBACK_MAX_ENTRY_BYTES} bytes"
        )

    existing_size = normalized.stat().st_size if normalized.exists() else 0
    if existing_size + new_line_size > FALLBACK_MAX_LOG_BYTES and normalized.exists():
        _rotate_feedback_logs(normalized)

    _atomic_append_json_line(normalized, payload_line)


def _verify_appended_payload(log_path: Path, payload_line: str) -> None:
    """Verify that a just-written payload is present in the log tail.

    Args:
        log_path: Target feedback log path.
        payload_line: Serialized JSON line that should have been appended.

    Raises:
        OSError: If the payload cannot be observed after append.
    """
    normalized = _validate_existing_log_file(log_path)
    expected = payload_line.encode("utf-8")
    for line in reversed(_read_tail_lines_bytes(normalized)):
        if line.strip() == expected:
            return
    raise OSError(f"feedback log write verification failed for: {normalized}")


def _iter_feedback_log_files(log_path: Path) -> list[Path]:
    """Return rotated log files oldest-to-newest, ending with the current log file.

    Args:
        log_path: Active feedback log path.

    Returns:
        Ordered list of rotated backups followed by the active log path.
    """
    base = _normalize_and_validate_log_path(log_path)
    candidates: list[tuple[int, Path]] = []
    for rotated in base.parent.glob(f"{base.name}.*"):
        suffix = rotated.name.removeprefix(f"{base.name}.")
        if suffix.isdigit():
            validated = _validate_existing_log_file(rotated)
            candidates.append((int(suffix), validated))

    ordered = [path for _idx, path in sorted(candidates, key=lambda item: item[0], reverse=True)]
    if base.exists():
        ordered.append(_validate_existing_log_file(base))
    else:
        ordered.append(base)
    return ordered


def _iter_feedback_entries(log_path: Path) -> Iterable[dict[str, Any]]:
    """Yield JSONL feedback entries from disk, skipping malformed records.

    Args:
        log_path: Active feedback log path.

    Yields:
        Parsed feedback entry dictionaries from the log files.
    """
    for path in _iter_feedback_log_files(log_path):
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    with contextlib.suppress(json.JSONDecodeError):
                        payload = json.loads(line)
                        if isinstance(payload, dict):
                            yield payload
        except OSError:
            continue


def _read_feedback_entries(log_path: Path) -> list[dict[str, Any]]:
    """Read JSONL feedback entries from disk, skipping malformed records.

    Args:
        log_path: Active feedback log path.

    Returns:
        Parsed feedback entry dictionaries in on-disk iteration order.
    """
    return list(_iter_feedback_entries(log_path))


def _read_feedback_page(
    log_path: Path,
    *,
    page: int,
    page_size: int,
    severity_filter: str | None,
) -> dict[str, Any]:
    """Read one page of feedback entries without materializing full datasets.

    Args:
        log_path: Active feedback log path.
        page: One-based page number to read.
        page_size: Maximum number of entries to return.
        severity_filter: Optional severity filter applied case-insensitively.

    Returns:
        Deterministic paginated response envelope.
    """
    start_offset = (page - 1) * page_size
    end_offset = start_offset + page_size
    total_entries = 0
    page_entries: list[dict[str, Any]] = []

    normalized_filter = severity_filter.lower() if severity_filter else None

    for entry in _iter_feedback_entries(log_path):
        if normalized_filter and str(entry.get("severity", "")).lower() != normalized_filter:
            continue

        if start_offset <= total_entries < end_offset:
            page_entries.append(entry)
        total_entries += 1

    total_pages = (total_entries + page_size - 1) // page_size if total_entries > 0 else 0
    returned = len(page_entries)
    start_index = start_offset + 1 if returned > 0 else 0
    end_index = start_offset + returned if returned > 0 else 0

    return {
        "entries": page_entries,
        "page": page,
        "page_size": page_size,
        "returned": returned,
        "total_entries": total_entries,
        "total_pages": total_pages,
        "has_next": total_pages > 0 and page < total_pages,
        "has_previous": page > 1,
        "start_index": start_index,
        "end_index": end_index,
    }


def _build_read_response(
    entries: list[dict[str, Any]],
    *,
    page: int,
    page_size: int,
) -> dict[str, Any]:
    """Build a deterministic read-mode response envelope.

    Args:
        entries: All entries to paginate.
        page: One-based page number to render.
        page_size: Maximum number of entries to include.

    Returns:
        Paginated response envelope for read-mode callers.
    """
    total_entries = len(entries)
    total_pages = (total_entries + page_size - 1) // page_size if total_entries > 0 else 0
    start_offset = (page - 1) * page_size
    page_entries = (
        entries[start_offset : start_offset + page_size] if start_offset < total_entries else []
    )
    returned = len(page_entries)

    start_index = start_offset + 1 if returned > 0 else 0
    end_index = start_offset + returned if returned > 0 else 0

    return {
        "entries": page_entries,
        "page": page,
        "page_size": page_size,
        "returned": returned,
        "total_entries": total_entries,
        "total_pages": total_pages,
        "has_next": total_pages > 0 and page < total_pages,
        "has_previous": page > 1,
        "start_index": start_index,
        "end_index": end_index,
    }


def _normalize_cli_text(value: Any) -> str:
    """Normalize CLI string-like values by trimming surrounding whitespace.

    Args:
        value: Raw CLI value to normalize.

    Returns:
        Trimmed string representation, or an empty string for None.
    """
    if value is None:
        return ""
    return str(value).strip()


def _print_read_error(exc: BaseException) -> int:
    """Print a deterministic read-mode failure message.

    Args:
        exc: Exception raised while serving a read-mode request.

    Returns:
        Canonical read-mode failure exit code.
    """
    print(_log_error(str(exc) or exc.__class__.__name__))
    return 2


def _build_feedback_entry(backend: FeedbackBackend, args: argparse.Namespace) -> Any:
    """Build a backend feedback entry from normalized CLI arguments.

    Args:
        backend: Active backend contract used for validation and entry creation.
        args: Parsed CLI arguments.

    Returns:
        Backend-specific feedback entry instance.

    Raises:
        ValueError: If required arguments are missing or validation fails.
    """
    category = _normalize_cli_text(args.category)
    severity = _normalize_cli_text(args.severity)
    description = _normalize_cli_text(args.description)
    workflow_step = _normalize_cli_text(args.workflow_step)
    agent_type = _normalize_cli_text(args.agent_type)
    adw_id = _normalize_cli_text(args.adw_id)
    suggested_fix = _normalize_cli_text(args.suggested_fix)
    tool_name = _normalize_cli_text(args.tool_name)
    context = _normalize_cli_text(args.context)

    missing_args: list[str] = []
    if not category:
        missing_args.append("category")
    if not severity:
        missing_args.append("severity")
    if not description:
        missing_args.append("description")
    if not workflow_step:
        missing_args.append("workflow_step")
    if not agent_type:
        missing_args.append("agent_type")
    if not adw_id:
        missing_args.append("adw_id")
    if missing_args:
        missing_flags = ", ".join(f"--{item.replace('_', '-')}" for item in missing_args)
        raise ValueError(f"Missing required arguments for write command: {missing_flags}")

    if category not in backend.categories:
        raise ValueError(
            _validation_error(
                f"Invalid category '{category}'",
                backend.categories,
                backend.severities,
            )
        )
    if severity not in backend.severities:
        raise ValueError(
            _validation_error(
                f"Invalid severity '{severity}'",
                backend.categories,
                backend.severities,
            )
        )

    return backend.entry_type(
        timestamp=datetime.now(timezone.utc),
        adw_id=adw_id,
        workflow_step=workflow_step,
        agent_type=agent_type,
        category=category,
        severity=severity,
        tool_name=tool_name,
        description=description,
        suggested_fix=suggested_fix,
        context=context,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for feedback logging.

    Write mode requires category, severity, description, workflow step, agent
    type, and ADW ID so wrapper/backend validation stays aligned. Read mode
    returns paginated JSON envelopes from the fallback feedback log.

    Args:
        argv: Optional CLI argument sequence for testing.

    Returns:
        Exit code for the CLI execution.
    """
    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 2

    if args.command == "read":
        if args.page < 1:
            print(_validation_error("Invalid page. Must be >= 1", [], []))
            return 1
        if args.page_size < 1 or args.page_size > READ_MAX_PAGE_SIZE:
            print(
                _validation_error(
                    (
                        f"Invalid page_size '{args.page_size}'. Must be between 1 "
                        f"and {READ_MAX_PAGE_SIZE}"
                    ),
                    [],
                    [],
                )
            )
            return 1

        try:
            log_path = _resolve_feedback_log_path()
            payload = _read_feedback_page(
                log_path,
                page=args.page,
                page_size=args.page_size,
                severity_filter=args.severity_filter,
            )
        except Exception as exc:  # noqa: BLE001
            return _print_read_error(exc)

        print(json.dumps(payload, ensure_ascii=False))
        return 0

    try:
        backend = _load_feedback_backend()
        entry = _build_feedback_entry(backend, args)
    except ValueError as exc:
        message = str(exc)
        if message.startswith("Feedback error:"):
            print(message)
        else:
            print(_validation_error(message, [], []))
        return 1
    except Exception as exc:  # noqa: BLE001
        print(_log_error(str(exc) or exc.__class__.__name__))
        return 2

    category = _normalize_cli_text(args.category)
    severity = _normalize_cli_text(args.severity)
    description = _normalize_cli_text(args.description)

    try:
        result = backend.log_feedback(entry)
    except Exception as exc:  # noqa: BLE001
        print(_log_error(str(exc)))
        return 2

    if result.success:
        truncated = _truncate_description(description)
        print(f"Feedback logged, thank you. [{category}/{severity}] {truncated}")
        return 0

    if result.message.startswith(RATE_LIMIT_PREFIX):
        print(RATE_LIMIT_MESSAGE)
        return 0

    print(_log_error(result.message))
    return 2


if __name__ == "__main__":
    sys.exit(main())
