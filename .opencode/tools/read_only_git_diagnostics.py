"""One-request JSON-lines adapter for local read-only Git diagnostics.

The adapter owns bounded transport, trusted-worktree admission, fixed local Git
execution, and bounded projection. It is deliberately standard-library-only so
the copied OpenCode tool runtime does not depend on ``adw`` or ``adforge_core``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

_MAX_REQUEST_BYTES = 65_536
_COMMANDS = {"status": "git_status", "diff": "git_diff", "log": "git_log", "show": "git_show"}
_ALLOWED = {
    "status": {"command", "worktree_path"},
    "diff": {"command", "worktree_path", "base", "target", "path"},
    "log": {"command", "worktree_path", "ref", "max_count", "path"},
    "show": {"command", "worktree_path", "ref", "path"},
}
_READ_ONLY_FIELDS = {
    "git_status": {"worktree_path"},
    "git_diff": {"worktree_path", "base", "target", "path"},
    "git_log": {"worktree_path", "ref", "max_count", "path"},
    "git_show": {"worktree_path", "ref", "path"},
}
_EVIDENCE_IDENTITY = {"contract": "e37-m2-validation-git", "version": 1}
_GIT_REF_PATTERN = re.compile(r"^[A-Za-z0-9._/\-~^]+$")
_MAX_STDOUT_CHARS = 12_000
_MAX_STDERR_CHARS = 4_000
_MAX_SUMMARY_CHARS = 240
_MAX_MAX_COUNT = 1_000
_GIT_TIMEOUT_SECONDS = 15
_TRUNCATION_SUFFIX = "... [truncated]"
_PATH_TOKEN = "<path>"
_WORKTREE_TOKEN = "<worktree>"
_REDACTED_TOKEN = "<redacted>"
_GIT_ENVIRONMENT_ALLOWLIST = ("HOME", "LANG", "LC_ALL", "LC_CTYPE", "PATH", "SYSTEMROOT")
_READ_ONLY_GIT_CONFIG = ("-c", "core.fsmonitor=false")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_TRUSTED_WORKTREE = _REPOSITORY_ROOT.resolve()


class _ProjectContext:
    """Minimal wrapper-local repository context."""

    __slots__ = ("root",)

    def __init__(self, *, root: Path) -> None:
        self.root = root


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


def _derive_primary_worktree(root: Path, common_git_dir: Path | None) -> Path:
    """Return the primary checkout for a standard linked-worktree topology."""
    if common_git_dir is None:
        return root
    candidate = common_git_dir.parent
    try:
        if _is_directory(candidate / ".git") and (candidate / ".git").resolve() == common_git_dir:
            return candidate.resolve(strict=True)
    except OSError:
        pass
    return root


_TRUSTED_WORKTREE = _derive_primary_worktree(_TRUSTED_WORKTREE, _COMMON_GIT_DIR)


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


def _context_for_worktree(raw_worktree: object) -> _ProjectContext | dict[str, Any]:
    """Construct an explicit local-only context for an admitted Git worktree.

    This function never selects authority from the process current directory.

    Args:
        raw_worktree: Untrusted absolute worktree path supplied by the request.

    Returns:
        A minimal repository-scoped context, or a bounded invalid-request
        envelope when worktree admission fails.
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

    return _ProjectContext(root=root)


def _with_identity(result: dict[str, Any]) -> dict[str, Any]:
    """Attach a fresh portable evidence-identity projection."""
    return result | {"evidence_identity": dict(_EVIDENCE_IDENTITY)}


def _invalid_request(operation: str, message: str) -> dict[str, Any]:
    """Build one identity-bearing runtime validation failure."""
    return _with_identity(
        {
            "ok": False,
            "operation": operation,
            "error": {"type": "invalid_request", "message": message},
        }
    )


def _sanitized_git_environment() -> dict[str, str]:
    """Return a local-only deterministic Git subprocess environment."""
    environment = {
        name: os.environ[name] for name in _GIT_ENVIRONMENT_ALLOWLIST if name in os.environ
    }
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _bound_text(value: str, *, limit: int) -> tuple[str, bool]:
    """Bound diagnostic text to a deterministic maximum length."""
    if len(value) <= limit:
        return value, False
    if limit <= len(_TRUNCATION_SUFFIX):
        return _TRUNCATION_SUFFIX[:limit], True
    return f"{value[: limit - len(_TRUNCATION_SUFFIX)].rstrip()}{_TRUNCATION_SUFFIX}", True


def _sanitize_text(value: object, *, project_root: Path) -> str:
    """Redact absolute paths and common secret-like values."""
    if not isinstance(value, str):
        return ""
    sanitized = value.replace(str(project_root.resolve(strict=False)), _WORKTREE_TOKEN)
    sanitized = re.sub(r"\bBearer\s+[A-Za-z0-9._\-]+", f"Bearer {_REDACTED_TOKEN}", sanitized)
    sanitized = re.sub(
        r"(?i)\b(token|secret|password|api[_-]?key|authorization)\b\s*[:=]\s*([^\s,;]+)",
        lambda match: f"{match.group(1)}={_REDACTED_TOKEN}",
        sanitized,
    )
    return re.sub(r"(?<!\w)/(?:[^\s'\"]+/?)+", _PATH_TOKEN, sanitized)


def _run_git(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Execute one fixed Git argv with capped concurrent pipe capture."""
    executable = shutil.which(command[0])
    if executable is None:
        raise FileNotFoundError(command[0])
    argv = [executable, *_READ_ONLY_GIT_CONFIG, "--literal-pathspecs", *command[1:]]
    process = subprocess.Popen(
        argv,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        env=_sanitized_git_environment(),
    )
    captured: dict[str, tuple[bytes, bool]] = {}

    def _capture(name: str, stream: Any, limit: int) -> None:
        retained = stream.read(limit + 1)
        overflowed = len(retained) > limit
        while stream.read(8_192):
            overflowed = True
        captured[name] = (retained[:limit], overflowed)

    readers = [
        threading.Thread(target=_capture, args=("stdout", process.stdout, _MAX_STDOUT_CHARS)),
        threading.Thread(target=_capture, args=("stderr", process.stderr, _MAX_STDERR_CHARS)),
    ]
    for reader in readers:
        reader.start()
    try:
        process.wait(timeout=_GIT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise
    finally:
        for reader in readers:
            reader.join()
    stdout, stdout_truncated = captured.get("stdout", (b"", False))
    stderr, stderr_truncated = captured.get("stderr", (b"", False))
    completed = subprocess.CompletedProcess(
        argv,
        process.returncode,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )
    completed.stdout_truncated = stdout_truncated  # type: ignore[attr-defined]
    completed.stderr_truncated = stderr_truncated  # type: ignore[attr-defined]
    return completed


def _validate_ref(value: object) -> str | None:
    """Return one admitted Git ref or ``None`` when invalid."""
    if not isinstance(value, str) or not value.strip():
        return None
    ref = value.strip()
    if (
        ref != value
        or any(character.isspace() for character in ref)
        or not _GIT_REF_PATTERN.fullmatch(ref)
        or ref.startswith(("-", "/"))
        or ref.endswith("/")
        or "//" in ref
        or "@{" in ref
        or ".." in ref
        or any(part in {"", ".", ".."} or part.endswith(".lock") for part in ref.split("/"))
    ):
        return None
    return ref


def _normalize_path(root: Path, raw_value: object) -> str | None | dict[str, Any]:
    """Normalize one confined repository-relative literal path."""
    if raw_value is None:
        return None
    if not isinstance(raw_value, str) or not raw_value.strip():
        return _error("invalid_request", "path must be a non-empty string when provided")
    value = raw_value.strip()
    if (
        value.startswith(("-", ":("))
        or "\x00" in value
        or Path(value).is_absolute()
        or value in {".", "./"}
    ):
        return _error("invalid_request", "path must be a confined repository-relative literal path")
    resolved = (root / value).resolve(strict=False)
    try:
        relative = resolved.relative_to(root.resolve(strict=False)).as_posix()
    except ValueError:
        return _error("invalid_request", "path must stay within the repository root")
    if relative in {"", "."}:
        return _error("invalid_request", "path must be a confined repository-relative literal path")
    return relative


def _verify_ref(root: Path, ref: str, *, require_commit: bool) -> dict[str, Any] | None:
    """Verify one admitted ref through a fixed local-only command."""
    suffix = "^{commit}" if require_commit else "^{object}"
    try:
        result = _run_git(["git", "rev-parse", "--verify", f"{ref}{suffix}"], cwd=root)
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": {
                "type": "execution_timeout",
                "message": "git ref verification timed out",
            },
        }
    except OSError:
        return {
            "ok": False,
            "error": {
                "type": "unavailable",
                "message": "git ref verification unavailable",
            },
        }
    if result.returncode != 0:
        expected = "a commit" if require_commit else "a git object"
        return {
            "ok": False,
            "error": {"type": "invalid_request", "message": f"ref must resolve to {expected}"},
        }
    return None


def _build_command(
    operation: str,
    *,
    base: str | None,
    target: str | None,
    ref: str | None,
    max_count: int,
    path: str | None,
) -> list[str]:
    """Build one admitted fixed read-only Git command."""
    if operation == "git_status":
        return ["git", "status", "--short"]
    if operation == "git_diff":
        command = ["git", "diff", "--no-ext-diff", "--no-textconv"]
        if base and target:
            command.extend([base, target])
        elif base:
            command.append(base)
    elif operation == "git_log":
        command = ["git", "log", "--oneline", f"--max-count={max_count}"]
        if ref:
            command.append(ref)
    else:
        command = [
            "git",
            "show",
            "--no-ext-diff",
            "--no-textconv",
            "--stat",
            "--format=medium",
            ref or "",
        ]
    if path is not None:
        command.extend(["--", path])
    return command


def _command_summary(operation: str, command: list[str], root: Path, path: str | None) -> str:
    """Build a bounded transcript-safe command summary."""
    if operation == "git_diff" and path is not None:
        value = f"git diff --no-ext-diff -- {_PATH_TOKEN}"
    elif operation == "git_show" and path is not None:
        value = f"git show --stat --format=medium <ref> -- {_PATH_TOKEN}"
    else:
        value = _sanitize_text(" ".join(command), project_root=root)
    return _bound_text(value, limit=_MAX_SUMMARY_CHARS)[0]


def _relative_worktree_text(root: Path, worktree: Path) -> str:
    """Return a tokenized worktree identifier."""
    try:
        relative = worktree.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
    except ValueError:
        return _PATH_TOKEN
    return _WORKTREE_TOKEN if relative in {"", "."} else f"{_WORKTREE_TOKEN}/{relative}"


def _data(
    root: Path,
    path: str | None,
    summary: str,
    completed: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    """Project one bounded Git subprocess result."""
    stdout, stdout_truncated = _bound_text(
        _sanitize_text(completed.stdout, project_root=root), limit=_MAX_STDOUT_CHARS
    )
    stderr, stderr_truncated = _bound_text(
        _sanitize_text(completed.stderr, project_root=root), limit=_MAX_STDERR_CHARS
    )
    stdout_truncated = stdout_truncated or bool(getattr(completed, "stdout_truncated", False))
    stderr_truncated = stderr_truncated or bool(getattr(completed, "stderr_truncated", False))
    if stdout_truncated and not stdout.endswith(_TRUNCATION_SUFFIX):
        stdout = _bound_text(f"{stdout}{_TRUNCATION_SUFFIX}", limit=_MAX_STDOUT_CHARS)[0]
    if stderr_truncated and not stderr.endswith(_TRUNCATION_SUFFIX):
        stderr = _bound_text(f"{stderr}{_TRUNCATION_SUFFIX}", limit=_MAX_STDERR_CHARS)[0]
    return {
        "command_summary": summary,
        "return_code": completed.returncode,
        "stdout": stdout,
        "stdout_truncated": stdout_truncated,
        "stderr": stderr,
        "stderr_truncated": stderr_truncated,
        "worktree_path": _relative_worktree_text(root, root),
        "path": path,
    }


def _failure_data(root: Path, path: str | None, summary: str) -> dict[str, Any]:
    """Build bounded details for a command that failed before completion."""
    return {
        "command_summary": summary,
        "return_code": -1,
        "stdout": "",
        "stdout_truncated": False,
        "stderr": "",
        "stderr_truncated": False,
        "worktree_path": _relative_worktree_text(root, root),
        "path": path,
    }


def _upstream_divergence(root: Path) -> str:
    """Classify local upstream divergence without network access."""
    try:
        upstream = _run_git(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
            cwd=root,
        )
        if upstream.returncode != 0:
            return "not_configured"
        local = _run_git(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
        counts = _run_git(
            [
                "git",
                "rev-list",
                "--left-right",
                "--count",
                f"{local.stdout.strip()}...{upstream.stdout.strip()}",
            ],
            cwd=root,
        )
        if local.returncode != 0 or counts.returncode != 0:
            return "unavailable"
        ahead, behind = (int(value) for value in counts.stdout.split())
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return "unavailable"
    if ahead and behind:
        return "diverged"
    if ahead:
        return "ahead"
    if behind:
        return "behind"
    return "in_sync"


def execute_git_tool(
    context: _ProjectContext, operation: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Validate and execute one wrapper-local read-only Git operation."""
    allowed = _READ_ONLY_FIELDS.get(operation)
    if allowed is None:
        return _invalid_request(operation, "unsupported git tool")
    if set(arguments) - allowed:
        return _invalid_request(operation, "unknown arguments are not allowed")
    if arguments.get("worktree_path") != str(context.root):
        return _invalid_request(operation, "worktree_path does not match admitted worktree")

    max_count = arguments.get("max_count", 50)
    if type(max_count) is not int or not 1 <= max_count <= _MAX_MAX_COUNT:
        return _invalid_request(operation, "max_count must be an integer between 1 and 1000")

    raw_fields = {name: arguments.get(name) for name in ("base", "target", "ref")}
    normalized: dict[str, str | None] = {}
    for name, value in raw_fields.items():
        if value is None:
            normalized[name] = None
            continue
        normalized[name] = _validate_ref(value)
        if normalized[name] is None:
            return _invalid_request(operation, f"{name} must use a safe git ref shape")
    base, target, ref = normalized["base"], normalized["target"], normalized["ref"]
    if operation == "git_diff" and target is not None and base is None:
        return _invalid_request(operation, "target requires base")
    if operation == "git_show" and ref is None:
        return _invalid_request(operation, "ref is required")

    path = _normalize_path(context.root, arguments.get("path"))
    if isinstance(path, dict):
        path["operation"] = operation
        return _with_identity(path)
    for candidate in (base, target, ref):
        if candidate is None:
            continue
        verification = _verify_ref(
            context.root,
            candidate,
            require_commit=operation != "git_show" or path is not None,
        )
        if verification is not None:
            verification["operation"] = operation
            return _with_identity(verification)

    command = _build_command(
        operation,
        base=base,
        target=target,
        ref=ref,
        max_count=max_count,
        path=path,
    )
    summary = _command_summary(operation, command, context.root, path)
    try:
        completed = _run_git(command, cwd=context.root)
    except subprocess.TimeoutExpired:
        return _with_identity(
            {
                "ok": False,
                "operation": operation,
                "error": {
                    "type": "execution_timeout",
                    "message": "git inspection timed out",
                    "details": _failure_data(context.root, path, summary),
                },
            }
        )
    except FileNotFoundError:
        return _with_identity(
            {
                "ok": False,
                "operation": operation,
                "error": {
                    "type": "unavailable",
                    "message": "git executable not available",
                    "details": _failure_data(context.root, path, summary),
                },
            }
        )
    except OSError:
        return _with_identity(
            {
                "ok": False,
                "operation": operation,
                "error": {
                    "type": "execution_failed",
                    "message": "git inspection failed",
                    "details": _failure_data(context.root, path, summary),
                },
            }
        )
    data = _data(context.root, path, summary, completed)
    if completed.returncode != 0:
        return _with_identity(
            {
                "ok": False,
                "operation": operation,
                "error": {
                    "type": "execution_failed",
                    "message": "git inspection failed",
                    "details": data,
                },
            }
        )
    if operation == "git_status":
        data["status"] = "clean" if not data["stdout"].strip() else "changed"
        data["divergence"] = _upstream_divergence(context.root)
    elif operation == "git_diff":
        data["diff"] = "no_diff" if not data["stdout"].strip() else "diff_present"
    return _with_identity({"ok": True, "operation": operation, "data": data})


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
