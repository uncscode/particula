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
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adw.utils.feedback import (  # noqa: E402
    FEEDBACK_CATEGORIES,
    FEEDBACK_SEVERITIES,
    FeedbackEntry,
    log_feedback,
)

MAX_DESCRIPTION_LENGTH = 100
RATE_LIMIT_PREFIX = "Rate limited"
RATE_LIMIT_MESSAGE = (
    "Rate limited — feedback already logged for this agent within the last 60s. "
    "Thank you, no action needed."
)


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


def _validation_error(reason: str) -> str:
    """Build a validation error message.

    Args:
        reason: Human-readable validation failure reason.

    Returns:
        Complete validation error message.
    """
    return (
        f"Feedback error: {reason}. "
        f"Valid categories: {FEEDBACK_CATEGORIES}. "
        f"Valid severities: {FEEDBACK_SEVERITIES}."
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
    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 2

    if args.category not in FEEDBACK_CATEGORIES:
        print(_validation_error(f"Invalid category '{args.category}'"))
        return 1
    if args.severity not in FEEDBACK_SEVERITIES:
        print(_validation_error(f"Invalid severity '{args.severity}'"))
        return 1

    entry = FeedbackEntry(
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
        success, message = log_feedback(entry)
    except Exception as exc:  # noqa: BLE001
        print(_log_error(str(exc)))
        return 2

    if success:
        truncated = _truncate_description(args.description)
        print(f"Feedback logged, thank you. [{args.category}/{args.severity}] {truncated}")
        return 0

    if message.startswith(RATE_LIMIT_PREFIX):
        print(RATE_LIMIT_MESSAGE)
        return 0

    print(_log_error(message))
    return 2


if __name__ == "__main__":
    sys.exit(main())
