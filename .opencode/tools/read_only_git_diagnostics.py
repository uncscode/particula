# ruff: noqa: E402, I001
"""One-request JSON-lines adapter for local read-only Git diagnostics.

The adapter owns bounded JSON transport and explicit trusted-worktree context
construction only. It imports neither ``adw`` nor an ``adforge`` CLI root.
Operation-specific Git admission, fixed local command execution, and bounded
diagnostic projection remain solely in ``adforge_core.runtime.git_tools``.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from adforge_core.context.models import (  # noqa: E402
    FolderAccessPolicy,
    LocalInternetAccess,
    PolicyContext,
    ProjectContext,
    WorkspaceTrust,
)
from adforge_core.runtime.git_tools import execute_git_tool  # noqa: E402


_MAX_REQUEST_BYTES = 65_536
_COMMANDS = {"status": "git_status", "diff": "git_diff", "log": "git_log", "show": "git_show"}
_ALLOWED = {
    "status": {"command", "worktree_path"},
    "diff": {"command", "worktree_path", "base", "target", "path"},
    "log": {"command", "worktree_path", "ref", "max_count", "path"},
    "show": {"command", "worktree_path", "ref", "path"},
}
_TRUSTED_WORKTREE = _REPOSITORY_ROOT.resolve()


def _is_regular_file(path: Path) -> bool:
    """Return whether a path is a non-symlink regular file."""
    try:
        return stat.S_ISREG(path.stat(follow_symlinks=False).st_mode)
    except OSError:
        return False


def _is_directory(path: Path) -> bool:
    """Return whether a path is a non-symlink directory."""
    try:
        return stat.S_ISDIR(path.stat(follow_symlinks=False).st_mode)
    except OSError:
        return False


def _read_pointer(path: Path, *, prefix: str | None = None) -> Path | None:
    """Read one bounded metadata pointer without following a symlink."""
    if not hasattr(os, "O_NOFOLLOW"):
        return None
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 4096:
            return None
        payload = os.read(descriptor, 4097)
    except OSError:
        return None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(payload) > 4096:
        return None
    try:
        value = payload.decode("utf-8").strip()
    except UnicodeDecodeError:
        return None
    if not value or "\x00" in value or "\n" in value or "\r" in value:
        return None
    if prefix is not None:
        if not value.startswith(prefix):
            return None
        value = value.removeprefix(prefix).strip()
        if not value:
            return None
    target = Path(value).expanduser()
    if not target.is_absolute():
        target = path.parent / target
    try:
        return target.resolve(strict=True)
    except OSError:
        return None


def _derive_common_git_dir(root: Path) -> Path | None:
    """Derive one checkout's canonical Git common directory without cwd."""
    git_entry = root / ".git"
    if _is_directory(git_entry):
        try:
            return git_entry.resolve(strict=True)
        except OSError:
            return None
    git_directory = _read_pointer(git_entry, prefix="gitdir:")
    if git_directory is None or not _is_directory(git_directory):
        return None
    return _read_pointer(git_directory / "commondir")


_COMMON_GIT_DIR = _derive_common_git_dir(_TRUSTED_WORKTREE)


def _is_registered_worktree(root: Path) -> bool:
    """Validate one linked worktree against this checkout's Git registry."""
    if _COMMON_GIT_DIR is None:
        return False
    git_entry = root / ".git"
    git_directory = _read_pointer(git_entry, prefix="gitdir:")
    if git_directory is None or not _is_directory(git_directory):
        return False

    worktrees_directory = _COMMON_GIT_DIR / "worktrees"
    if not _is_directory(worktrees_directory) or git_directory.parent != worktrees_directory:
        return False
    if not _is_regular_file(git_directory / "HEAD"):
        return False

    common_directory = _read_pointer(git_directory / "commondir")
    if common_directory != _COMMON_GIT_DIR:
        return False
    backlink = _read_pointer(git_directory / "gitdir")
    return backlink == git_entry


def _error(kind: str, message: str, operation: str | None = None) -> dict[str, Any]:
    """Build one stable bounded adapter error envelope.

    Args:
        kind: Stable adapter or runtime-compatible failure classification.
        message: Safe human-readable summary without traceback details.
        operation: Optional canonical Git operation for the failed request.

    Returns:
        A JSON-serializable failure envelope.
    """
    result: dict[str, Any] = {"ok": False, "error": {"type": kind, "message": message}}
    if operation is not None:
        result["operation"] = operation
    return result


def _read_request() -> dict[str, Any] | None:
    """Read exactly one bounded UTF-8 JSON object from standard input.

    Returns:
        The decoded request object, or a bounded invalid-request envelope when
        input is empty, oversized, malformed, non-object, or has trailing data.
    """
    payload = sys.stdin.buffer.read(_MAX_REQUEST_BYTES + 1)
    if not payload:
        return _error("invalid_request", "request body is required")
    if len(payload) > _MAX_REQUEST_BYTES:
        return _error("invalid_request", "request body is too large")
    try:
        text = payload.decode("utf-8")
        decoder = json.JSONDecoder()
        value, offset = decoder.raw_decode(text.lstrip())
        if text.lstrip()[offset:].strip():
            return _error("invalid_request", "input must be one bounded UTF-8 JSON object")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _error("invalid_request", "input must be one bounded UTF-8 JSON object")
    if not isinstance(value, dict):
        return _error("invalid_request", "input must be one bounded UTF-8 JSON object")
    return value


def _context_for_worktree(raw_worktree: object) -> Any | dict[str, Any]:
    """Construct an explicit local-only context for an admitted Git worktree.

    This function never selects authority from the process current directory.

    Args:
        raw_worktree: Untrusted absolute worktree path supplied by the request.

    Returns:
        A repository-scoped ``ProjectContext`` with denied local internet, or a
        bounded invalid-request envelope when worktree admission fails.
    """
    if (
        not isinstance(raw_worktree, str)
        or not raw_worktree.strip()
        or raw_worktree.lstrip().startswith("-")
    ):
        return _error("invalid_request", "worktree_path must be an admitted absolute directory")
    candidate = Path(raw_worktree).expanduser()
    if not candidate.is_absolute():
        return _error("invalid_request", "worktree_path must be an admitted absolute directory")
    try:
        root = candidate.resolve(strict=True)
    except OSError:
        return _error("invalid_request", "worktree_path must be an admitted existing directory")
    if not _is_directory(root):
        return _error("invalid_request", "worktree_path must be an admitted existing directory")
    if root == _TRUSTED_WORKTREE:
        if _COMMON_GIT_DIR is None or _derive_common_git_dir(root) != _COMMON_GIT_DIR:
            return _error("invalid_request", "worktree_path must be an admitted git worktree")
    elif not _is_registered_worktree(root):
        return _error("invalid_request", "worktree_path is not a registered linked worktree")
    if not (root / ".git").exists():
        return _error("invalid_request", "worktree_path must be an admitted git worktree")

    return ProjectContext(
        workspace_id="read-only-git-diagnostics",
        root=root,
        config_path=root / ".adforge" / "config.json",
        trust=WorkspaceTrust.INTERNAL,
        policy=PolicyContext(
            local_internet_access=LocalInternetAccess.DENY,
            folder_access=FolderAccessPolicy.ALLOW,
            folder_roots=[root],
            tools=[],
        ),
        agent_name="read-only-git-diagnostics",
    )


def handle(request: dict[str, Any]) -> dict[str, Any]:
    """Validate one adapter request and dispatch it once to the core seam.

    The adapter rejects unsupported presentation fields and constructs explicit
    worktree authority before dispatch. It does not recreate Git validation or
    inspect output; the runtime seam produces the final bounded envelope.

    Args:
        request: Decoded untrusted request object from the JSON-lines transport.

    Returns:
        A stable adapter failure envelope or the single core diagnostic response.
    """
    command = request.get("command")
    if not isinstance(command, str) or command not in _COMMANDS:
        return _error("invalid_request", "command must be one of status, diff, log, or show")
    operation = _COMMANDS[command]
    if set(request) - _ALLOWED[command]:
        return _error("invalid_request", "request contains unsupported fields", operation)
    if any(field in request for field in {"porcelain", "stat", "oneline", "help"}):
        return _error("invalid_request", "legacy presentation fields are not supported", operation)
    context = _context_for_worktree(request.get("worktree_path"))
    if isinstance(context, dict):
        return context
    if shutil.which("git") is None:
        return _error("unavailable", "git executable not available", operation)
    arguments = {key: value for key, value in request.items() if key != "command"}
    return execute_git_tool(context, operation, arguments)


def handle_request() -> dict[str, Any]:
    """Read, validate, and dispatch one standard-input JSON-lines request.

    Returns:
        A bounded request-validation failure or the core diagnostic response.
    """
    request = _read_request()
    if request is None or request.get("ok") is False:
        return request or _error("invalid_request", "input must be one bounded UTF-8 JSON object")
    return handle(request)


def main() -> None:
    """Write one newline-terminated JSON response without stderr diagnostics.

    Unexpected adapter failures are converted to a bounded response so callers
    receive neither tracebacks nor unbounded process details.
    """
    try:
        response = handle_request()
    except Exception:
        response = _error("adapter_failed", "git diagnostics adapter failed")
    sys.stdout.write(json.dumps(response, separators=(",", ":"), sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
