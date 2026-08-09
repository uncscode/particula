#!/usr/bin/env python3
"""Run pytest through validated, runner-owned controls for ADforge.

The runner evaluates test assertions and coverage evidence independently. It
preserves the repository coverage policy, including its minimum coverage floor,
while accepting only normalized, repository-confined explicit coverage sources.
Disabled coverage is reported explicitly rather than as a passing coverage
result. Coverage-enabled runs acquire an ownership-safe, no-wait lease scoped to
the canonical worktree; the lease prevents concurrent ``.coverage`` writes
without disclosing paths, holders, or tokens. Every runner-owned JSON outcome
also carries the canonical E37-M2 evidence identity; summary and full text
outputs do not. The advanced route transports caller-owned pytest tokens only as
a strict JSON string array; runner-owned controls include singular or ordered
plural targets, filtering, coverage, timeout, collection, and output.

Key features:
    - Independent assertion and coverage status projections
    - Coverage reporting with validated sources and retained policy thresholds
    - Hard-failure handling for unusable pytest-cov diagnostics or missing totals
    - Validation of minimum test counts to catch collection errors
    - Fail-fast mode for quick development feedback
    - Canonical-worktree coverage leases to avoid shared ``.coverage`` collisions
    - Ordered, repository-confined plural test targets
    - Explicit collect-only projections with collected and zero-executed counts
    - Duration profiling for performance optimization
    - Worktree-aware PYTHONPATH handling for isolated execution

Usage:
    python3 run_pytest.py
    python3 run_pytest.py adw/core/tests/ --min-tests 1
    python3 run_pytest.py --coverage-threshold 80

Examples:
    # Run full test suite (expects ~1700 tests)
    python3 .opencode/tools/run_pytest.py --min-tests 1700

    # Run scoped tests (always set min-tests=1 for scoped)
    python3 .opencode/tools/run_pytest.py adw/core/tests/ --min-tests 1

    # With coverage threshold enforcement
    python3 .opencode/tools/run_pytest.py --coverage-threshold 80

    # Fail fast during development
    python3 .opencode/tools/run_pytest.py --fail-fast adw/core/tests/

    # In worktree for isolated execution
    python3 .opencode/tools/run_pytest.py --cwd /path/to/trees/abc12345

    # Show slowest tests for optimization
    python3 .opencode/tools/run_pytest.py --durations 10

    # Select tests with a runner-owned filter
    python3 .opencode/tools/run_pytest.py --test-filter 'not slow and not performance'
"""

import argparse
import errno
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Tuple, Union

_EVIDENCE_CONTRACT = "e37-m2-validation-git"
_EVIDENCE_VERSION = 1


def evidence_identity_projection() -> dict[str, object]:
    """Return the portable validation-evidence identity.

    The OpenCode pytest backend is copied into repositories that do not ship
    ADforge's Pydantic runtime, so this fixed compatibility marker is owned
    locally rather than imported from :mod:`adforge_core`.
    """

    return {"contract": _EVIDENCE_CONTRACT, "version": _EVIDENCE_VERSION}


SECTION_HEADER_PATTERN = re.compile(r"^=+\s*.+\s*=+\s*$")
DURATIONS_HEADER_PATTERN = re.compile(
    r"^=+\s*slowest(?:\s+\d+)?\s+durations\s*=+\s*$",
    re.IGNORECASE,
)
FAILURES_HEADER_PATTERN = re.compile(r"^=+\s*FAILURES\s*=+\s*$", re.IGNORECASE)

COVERAGE_ADDOPT_PATTERN = re.compile(
    r"^(--cov(?:=|\b)|--cov-report(?:=|\b)|--cov-fail-under(?:=|\b)|"
    r"--cov-config(?:=|\b)|--cov-context(?:=|\b))"
)
COVERAGE_PYTEST_ARG_PATTERN = re.compile(
    r"^(--cov(?:=|\b)|--cov-report(?:=|\b)|--cov-fail-under(?:=|\b)|"
    r"--cov-config(?:=|\b)|--cov-context(?:=|\b))"
)
COVERAGE_HEADER_PATTERN = re.compile(r"^-+\s+coverage:.*-+$", re.IGNORECASE)
MAX_COVERAGE_FILES = 500
COVERAGE_LOCK_FILENAME = ".run_pytest_coverage.lock"
MAX_TIMEOUT_SECONDS = 1200
PYTEST_TIMEOUT_KILL_GRACE_SECONDS = 1.0
MAX_DIAGNOSTIC_TEXT = 2000
MAX_FULL_OUTPUT_TEXT = 20_000
MAX_DIAGNOSTIC_NODE_IDS = 50
MAX_DIAGNOSTIC_NODE_ID_LENGTH = 512
MAX_FAILURE_SCAN_TEXT = 20_000
PYTEST_ARG_VALUE_OPTIONS = {"-k", "-m"}
PYTEST_ARG_STANDALONE_OPTIONS = {"--collect-only", "-q", "-v", "--verbose"}
PYTEST_ARG_TB_VALUES = {"short", "long", "line", "native", "no"}
PYTEST_ARG_RESERVED_PREFIXES = (
    "--output",
    "--min-tests",
    "--timeout",
    "--cwd",
    "--test-path",
    "--test-filter",
    "--coverage",
    "--no-coverage",
    "--coverage-source",
    "--coverage-threshold",
    "--cov-report",
    "--fail-fast",
    "--durations",
    "--durations-min",
    "--pytest-argv-json",
    "--override-ini-json",
    "--override-ini",
    "--coverage-files-only",
)
UNUSABLE_COVERAGE_FRAGMENTS = (
    "no data collected",
    "no data was collected",
    "no data to report",
    "module was never imported",
)
COVERAGE_SOURCE_INFO = (
    "INFO: coverageSource supports only 'all' or existing repository-relative directories; "
    "dotted modules, file targets, and other non-directory entries are ignored."
)


class PytestArgumentValidationError(ValueError):
    """Raised when caller-owned advanced pytest argv violates the fixed grammar."""


def _bounded_diagnostic(value: str, limit: int = MAX_DIAGNOSTIC_TEXT) -> tuple[str, int]:
    """Return a bounded diagnostic tail and its omitted-character count.

    Args:
        value: Diagnostic text to bound.
        limit: Maximum number of trailing characters to retain.

    Returns:
        A pair containing the retained tail and the number of omitted characters.
    """

    if len(value) <= limit:
        return value, 0
    return value[-limit:], len(value) - limit


def _redact_diagnostic(value: str) -> str:
    """Redact absolute paths and JSON transport payloads from diagnostics.

    Args:
        value: Raw diagnostic text.

    Returns:
        Diagnostic text safe to render in a bounded failure outcome.
    """

    value = re.sub(r"(?<![\w.])/(?:[^\s'\"]+/?)+", "<path>", value)
    return re.sub(r"--(?:pytest-argv-json|override-ini-json)=\S+", "<transport>", value)


def _failure_outcome(
    *,
    returncode: int,
    validation_errors: list[str],
    stdout: str,
    stderr: str,
    elapsed_seconds: float,
    resolved_target: Optional[str],
) -> dict[str, Any]:
    """Build the canonical bounded failure projection without command/path leakage.

    Args:
        returncode: Pytest subprocess exit code.
        validation_errors: Validation failures found after pytest completes.
        stdout: Captured pytest standard output.
        stderr: Captured pytest standard error.
        elapsed_seconds: Monotonic execution duration in seconds.
        resolved_target: Runner-owned test target, when one was supplied.

    Returns:
        Ordered canonical failure data with a classification, reason, bounded
        diagnostics, and truncation metadata.
    """

    scan_text, scan_omitted = _bounded_diagnostic(f"{stdout}\n{stderr}", MAX_FAILURE_SCAN_TEXT)
    combined = scan_text.lower()
    if "error collecting" in combined or "collection" in combined and returncode == 2:
        classification = "collection"
    elif "usage:" in combined or "unrecognized arguments" in combined:
        classification = "usage"
    elif "plugin" in combined or "coverage" in combined and "error" in combined:
        classification = "plugin"
    elif "failed" in combined or "assert" in combined:
        classification = "assertion"
    else:
        classification = "invocation"
    bounded_stdout, stdout_omitted = _bounded_diagnostic(stdout)
    bounded_stderr, stderr_omitted = _bounded_diagnostic(stderr)
    stdout_tail = _redact_diagnostic(bounded_stdout)
    stderr_tail = _redact_diagnostic(bounded_stderr)
    excerpt, excerpt_omitted = _bounded_diagnostic("\n".join(validation_errors) or stdout_tail)
    raw_node_ids = re.findall(r"[^\s]+::[^\s]+", scan_text)
    omitted_nodes = max(0, len(raw_node_ids) - MAX_DIAGNOSTIC_NODE_IDS)
    node_ids = [
        "<absolute-node-id>"
        if os.path.isabs(node_id.split("::", 1)[0])
        else _redact_diagnostic(node_id[:MAX_DIAGNOSTIC_NODE_ID_LENGTH])
        for node_id in raw_node_ids[:MAX_DIAGNOSTIC_NODE_IDS]
    ]
    return {
        "classification": classification,
        "reason": validation_errors[0]
        if validation_errors
        else f"pytest exited with code {returncode}",
        "exit_code": returncode,
        "resolved_target": resolved_target,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "phase": "execution",
        "node_ids": node_ids,
        "excerpt": excerpt,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "truncation": {
            "node_ids_omitted": omitted_nodes,
            "excerpt_omitted": excerpt_omitted,
            "stdout_tail_omitted": stdout_omitted,
            "stderr_tail_omitted": stderr_omitted,
            "overall_truncated": any(
                (omitted_nodes, excerpt_omitted, stdout_omitted, stderr_omitted, scan_omitted)
            ),
            "scan_omitted": scan_omitted,
        },
    }


def _repository_root(cwd: str) -> Path:
    """Return the nearest repository root for a requested runner directory.

    Args:
        cwd: Requested runner working directory.

    Returns:
        The nearest ancestor containing ``pyproject.toml`` or ``.git``, or the
        resolved requested directory when no ancestor qualifies.
    """

    current = Path(cwd).resolve(strict=False)
    while current.parent != current:
        if (current / "pyproject.toml").exists() or (current / ".git").exists():
            return current
        current = current.parent
    return Path(cwd).resolve(strict=False)


def _validate_confined_target(value: str, cwd: str, name: str) -> str:
    """Validate a repository-relative pytest target without changing its bytes.

    Args:
        value: Candidate path or node-id target.
        cwd: Requested runner working directory.
        name: Argument name used in validation errors.

    Returns:
        The original validated target.

    Raises:
        PytestArgumentValidationError: If the target is empty, option-like,
            absolute, or resolves outside the repository root.
    """

    if not value or value.startswith("-") or Path(value).is_absolute():
        raise PytestArgumentValidationError(
            f"{name} must be a non-empty repository-relative target"
        )
    root = _repository_root(cwd)
    resolved = (root / value.split("::", 1)[0]).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError:
        raise PytestArgumentValidationError(
            f"{name} must stay within the repository/worktree root"
        ) from None
    return value


def _validate_pytest_argv(pytest_argv: object, cwd: str) -> list[str]:
    """Validate the ordered caller-owned argv suffix for the advanced route.

    Args:
        pytest_argv: Decoded JSON value expected to be a list of strings.
        cwd: Requested runner working directory used for target confinement.

    Returns:
        A byte-preserving validated copy of the accepted caller token sequence.

    Raises:
        PytestArgumentValidationError: If the array shape, token grammar, value
            pairing, reserved controls, or target confinement is invalid.
    """

    if not isinstance(pytest_argv, list) or any(not isinstance(item, str) for item in pytest_argv):
        raise PytestArgumentValidationError("pytest argv JSON must decode to an array of strings")
    validated: list[str] = []
    index = 0
    while index < len(pytest_argv):
        token = pytest_argv[index]
        if token == "--" or token.startswith("--cov"):
            raise PytestArgumentValidationError(f"pytest argument {token!r} is not permitted")
        if token in PYTEST_ARG_STANDALONE_OPTIONS:
            validated.append(token)
        elif token.startswith("--tb="):
            if token.removeprefix("--tb=") not in PYTEST_ARG_TB_VALUES:
                raise PytestArgumentValidationError(f"pytest argument {token!r} is not permitted")
            validated.append(token)
        elif token.startswith("--override-ini="):
            raise PytestArgumentValidationError(f"pytest argument {token!r} is not permitted")
        elif token in PYTEST_ARG_VALUE_OPTIONS:
            if index + 1 >= len(pytest_argv):
                raise PytestArgumentValidationError(f"pytest argument {token!r} requires a value")
            value = pytest_argv[index + 1]
            if value.startswith("-") or not value:
                raise PytestArgumentValidationError(
                    f"pytest argument {token!r} has an invalid value"
                )
            if token == "-o":
                raise PytestArgumentValidationError("pytest argument '-o' is not permitted")
            validated.extend((token, value))
            index += 1
        elif token.startswith("-") or token.startswith(PYTEST_ARG_RESERVED_PREFIXES):
            raise PytestArgumentValidationError(f"pytest argument {token!r} is not permitted")
        else:
            validated.append(_validate_confined_target(token, cwd, "pytest argument"))
        index += 1
    return validated


def _decode_string_array(raw: Optional[str], name: str) -> Optional[list[str]]:
    """Decode one compact JSON string array without coercion.

    Args:
        raw: JSON text supplied by a named transport control, if any.
        name: Human-readable transport name used in validation errors.

    Returns:
        The decoded string array, or ``None`` when the control was omitted.

    Raises:
        PytestArgumentValidationError: If ``raw`` is malformed JSON or does not
            decode to a list containing only strings.
    """

    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PytestArgumentValidationError(f"{name} must be valid JSON array") from exc
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise PytestArgumentValidationError(f"{name} must decode to an array of strings")
    return value


def _validate_test_paths(test_paths: object, cwd: str) -> list[str]:
    """Validate one through seven ordered canonical repository targets.

    Args:
        test_paths: Candidate plural target payload expected to be a list of
            strings.
        cwd: Requested runner working directory used to locate the repository
            root for confinement checks.

    Returns:
        A byte-preserving copy of the validated targets in caller order.

    Raises:
        PytestArgumentValidationError: If the payload is not a one-through-seven
            string list, or a target is option-like, noncanonical, or outside the
            repository root.
    """
    if not isinstance(test_paths, list) or not test_paths:
        raise PytestArgumentValidationError("testPaths must be a non-empty array of strings")
    if len(test_paths) > 7:
        raise PytestArgumentValidationError("testPaths must contain at most 7 entries")
    root = _repository_root(cwd)
    for index, value in enumerate(test_paths):
        label = f"testPaths[{index}]"
        if not isinstance(value, str) or not value or value.startswith("-"):
            raise PytestArgumentValidationError(
                f"{label} must be a non-empty repository-relative target"
            )
        path_part = value.split("::", 1)[0]
        if (
            "\\" in path_part
            or Path(path_part).is_absolute()
            or any(not part or part in {".", ".."} for part in path_part.split("/"))
        ):
            raise PytestArgumentValidationError(
                f"{label} must be a canonical relative POSIX target"
            )
        try:
            (root / path_part).resolve(strict=False).relative_to(root)
        except ValueError:
            raise PytestArgumentValidationError(
                f"{label} must stay within the repository/worktree root"
            ) from None
    return list(test_paths)


def _candidate_tool_dirs(cwd: Optional[str]) -> List[Path]:
    """Return likely executable directories that may be absent from tool PATH."""

    dirs: List[Path] = []
    if cwd:
        current = Path(cwd).resolve(strict=False)
        while True:
            dirs.extend([current / ".venv" / "bin", current / "venv" / "bin"])
            if current.parent == current:
                break
            current = current.parent
    dirs.append(Path(sys.executable).resolve(strict=False).parent)
    dirs.append(Path.home() / ".local" / "bin")
    return dirs


def _resolve_python_tool_command(tool_name: str, module_name: str, cwd: Optional[str]) -> List[str]:
    """Resolve a Python CLI robustly for non-login tool subprocess environments."""

    resolved = shutil.which(tool_name)
    if resolved:
        return [resolved]
    for directory in _candidate_tool_dirs(cwd):
        candidate = directory / tool_name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return [str(candidate)]
    if importlib.util.find_spec(module_name) is not None:
        return [sys.executable, "-m", module_name]
    return [tool_name]


class CoverageSourceValidationError(ValueError):
    """Raised when coverage source input violates the wrapper contract."""


class CoverageLockError(RuntimeError):
    """Raised when a same-worktree coverage run is already in progress."""


@dataclass(frozen=True)
class CoverageLease:
    """Opaque ownership record for one canonical-worktree coverage lease.

    Attributes:
        token: Fresh opaque value used to verify lease ownership before removal.
        pid: Process identifier used only to verify stale-holder liveness.
        worktree_id: Non-reversible identifier binding the lease to one canonical
            worktree.
    """

    token: str
    pid: int
    worktree_id: str

    def as_record(self) -> dict[str, object]:
        """Return the strict on-disk lease representation.

        Returns:
            JSON-serializable record containing the complete lease identity.
        """

        return {"token": self.token, "pid": self.pid, "worktree_id": self.worktree_id}


class PytestTimeoutValidationError(ValueError):
    """Raised when a timeout argument violates the wrapper contract."""


@dataclass(frozen=True)
class PytestTimeoutDetails:
    """Structured timeout details for deterministic wrapper diagnostics.

    Attributes:
        timeout_seconds: Configured timeout limit in seconds.
        elapsed_seconds: Monotonic duration before timeout handling began.
        pid: Direct pytest process identifier.
        process_group_id: Process group terminated during cleanup.
        cwd: Subprocess working directory before redaction.
        command: Full subprocess command before argument redaction.
        sigkill_escalated: Whether graceful termination required ``SIGKILL``.
    """

    timeout_seconds: float
    elapsed_seconds: float
    pid: int
    process_group_id: int
    cwd: str
    command: List[str]
    sigkill_escalated: bool


class PytestTimedOutError(RuntimeError):
    """Raised when pytest exceeds the configured timeout."""

    def __init__(self, details: PytestTimeoutDetails) -> None:
        super().__init__("pytest timed out")
        self.details = details


@dataclass(frozen=True)
class PytestSubprocessResult:
    """Captured pytest subprocess result.

    Attributes:
        returncode: Pytest process exit code.
        stdout: Captured standard output.
        stderr: Captured standard error.
    """

    returncode: int
    stdout: str
    stderr: str


def _format_timeout_number(value: float) -> str:
    """Format timeout-related numeric values deterministically."""

    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _validate_timeout_seconds(timeout: object) -> float:
    """Validate timeout input before any subprocess launch.

    Args:
        timeout: Caller-provided timeout value from the API or CLI.

    Returns:
        Normalized timeout value as a float in seconds.

    Raises:
        PytestTimeoutValidationError: The timeout is not numeric, is not finite,
            is non-positive, or exceeds the shared 1200-second cap.
    """

    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise PytestTimeoutValidationError(
            "timeout must be a positive finite number in seconds and must not exceed "
            "1200 seconds (20 minutes)."
        )
    timeout_value = float(timeout)
    if (
        not math.isfinite(timeout_value)
        or timeout_value <= 0
        or timeout_value > MAX_TIMEOUT_SECONDS
    ):
        raise PytestTimeoutValidationError(
            "timeout must be a positive finite number in seconds and must not exceed "
            "1200 seconds (20 minutes)."
        )
    return timeout_value


def _format_timeout_error(details: PytestTimeoutDetails) -> str:
    """Render deterministic timeout diagnostics in a fixed field order.

    Args:
        details: Structured timeout metadata captured during subprocess cleanup.

    Returns:
        Wrapper-safe ``ERROR:`` string with stable timeout, process, and command
        fields for downstream tooling and regression tests.
    """

    return (
        "ERROR: pytest timed out; "
        f"timeout_seconds={_format_timeout_number(details.timeout_seconds)}; "
        f"elapsed_seconds={_format_timeout_number(details.elapsed_seconds)}; "
        f"pid={details.pid}; "
        f"process_group_id={details.process_group_id}; "
        f"cwd={_redact_timeout_cwd(details.cwd)}; "
        f"command={_redact_timeout_command(details.command)}; "
        f"sigkill_escalated={'true' if details.sigkill_escalated else 'false'}"
    )


def _redact_timeout_cwd(cwd: str) -> str:
    """Return a stable, non-absolute cwd token for timeout diagnostics."""

    cwd_name = Path(cwd).name
    return cwd_name or "."


def _redact_timeout_command(command: List[str]) -> str:
    """Return a stable, argument-redacted command summary."""

    if not command:
        return "<unknown>"
    executable = Path(command[0]).name or command[0]
    redacted_arg_count = max(len(command) - 1, 0)
    if redacted_arg_count == 0:
        return executable
    return f"{executable} [args_redacted:{redacted_arg_count}]"


def _process_group_exists(process_group_id: int) -> bool:
    """Return whether the process group still exists."""

    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(process: subprocess.Popen[str], process_group_id: int) -> bool:
    """Terminate a timed-out pytest process group.

    Sends ``SIGTERM`` to the full process group first, waits briefly for a
    graceful exit, and escalates to ``SIGKILL`` only when the group remains
    alive. Missing process groups are treated as already exited.

    Args:
        process: Running pytest subprocess handle.
        process_group_id: Process-group identifier associated with ``process``.

    Returns:
        ``True`` when cleanup required ``SIGKILL`` escalation, otherwise
        ``False``.
    """

    sigkill_escalated = False
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return sigkill_escalated

    try:
        process.wait(timeout=PYTEST_TIMEOUT_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass

    if not _process_group_exists(process_group_id):
        return sigkill_escalated

    sigkill_escalated = True

    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return sigkill_escalated

    try:
        process.wait(timeout=PYTEST_TIMEOUT_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    return sigkill_escalated


def _filter_non_coverage_addopts(addopts: Union[str, List[str]]) -> List[str]:
    """Remove coverage controls and their operands from configured pytest addopts.

    Args:
        addopts: Shell-style addopts text or an already-tokenized addopts list.

    Returns:
        Retained non-coverage pytest tokens in their original order.
    """
    tokens = shlex.split(addopts) if isinstance(addopts, str) else list(addopts)
    filtered: List[str] = []

    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--cov=") or token.startswith("--cov-report="):
            index += 1
            continue
        if token.startswith("--cov-fail-under=") or token.startswith("--cov-config="):
            index += 1
            continue
        if token == "--cov":
            next_token = tokens[index + 1] if index + 1 < len(tokens) else None
            if next_token is not None and not next_token.startswith("-"):
                index += 2
            else:
                index += 1
            continue
        if token in {"--cov-report", "--cov-fail-under", "--cov-config", "--cov-context"}:
            next_token = tokens[index + 1] if index + 1 < len(tokens) else None
            if next_token is None or next_token.startswith("-"):
                raise CoverageSourceValidationError(
                    f"configured addopts token '{token}' must be followed by a value"
                )
            index += 2
            continue
        if COVERAGE_ADDOPT_PATTERN.match(token):
            index += 1
            continue
        filtered.append(token)
        index += 1
    return filtered


def _normalize_coverage_source(
    coverage_source: Optional[object],
    repo_root: Path,
    info_messages: Optional[List[str]] = None,
) -> List[str]:
    """Normalize coverage sources against the resolved repository root.

    Args:
        coverage_source: Coverage source from CLI or API. Accepts ``None``, a
            comma-separated string, or a list or tuple of such strings.
        repo_root: Resolved repository root that bounds path-form sources.

    Returns:
        Canonical root-relative POSIX directory paths in caller order.
        Returns an empty list when coverage should use repository defaults.

    Raises:
        CoverageSourceValidationError: The input has an invalid shape or an
            unsafe path form.

    Notes:
        The case-insensitive special value ``all`` must be the sole source and
        selects the repository coverage configuration.
    """
    if coverage_source is None:
        return []

    sources: List[str] = []
    if isinstance(coverage_source, str):
        sources = coverage_source.split(",")
    elif isinstance(coverage_source, (list, tuple)):
        for entry in coverage_source:
            if not isinstance(entry, str):
                raise CoverageSourceValidationError("coverageSource entries must be strings")
            sources.extend(entry.split(","))
    else:
        raise CoverageSourceValidationError(
            "coverageSource must be a string or an array of strings"
        )

    cleaned: List[str] = []
    for source in sources:
        stripped = source.strip()
        if not stripped:
            raise CoverageSourceValidationError(
                "coverageSource must not contain empty comma-separated entries"
            )
        cleaned.append(stripped)

    if any(source.lower() == "all" for source in cleaned):
        if len(cleaned) != 1:
            raise CoverageSourceValidationError("coverageSource 'all' must be the sole source")
        return []

    normalized: List[str] = []
    for source in cleaned:
        if "\\" in source or Path(source).is_absolute():
            raise CoverageSourceValidationError(
                f"coverageSource must be a relative POSIX path: {source}"
            )
        parts = source.split("/")
        if any(not part or part in {".", ".."} for part in parts):
            raise CoverageSourceValidationError(
                f"coverageSource has noncanonical path components: {source}"
            )
        candidate = repo_root / source
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(repo_root)
        except FileNotFoundError:
            if info_messages is not None and COVERAGE_SOURCE_INFO not in info_messages:
                info_messages.append(COVERAGE_SOURCE_INFO)
            continue
        except (OSError, ValueError) as exc:
            raise CoverageSourceValidationError(
                "coverageSource must be an existing safe path within the "
                f"repository/worktree root: {source}"
            ) from exc
        if candidate.is_symlink():
            raise CoverageSourceValidationError(
                f"coverageSource must be an existing safe directory: {source}"
            )
        if not resolved.is_dir():
            if info_messages is not None and COVERAGE_SOURCE_INFO not in info_messages:
                info_messages.append(COVERAGE_SOURCE_INFO)
            continue
        normalized.append(resolved.relative_to(repo_root).as_posix())
    return normalized


def _resolve_repo_root_for_coverage(cwd: Optional[str]) -> Path:
    """Resolve the trusted repository/worktree root used for coverage paths.

    Args:
        cwd: Requested pytest working directory, when explicitly supplied.

    Returns:
        Resolved requested directory or the nearest current-directory ancestor
        containing repository metadata.
    """

    current = Path(cwd).resolve() if cwd else Path.cwd().resolve()
    while True:
        if (current / "pyproject.toml").exists() or (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            return Path(cwd).resolve() if cwd else Path.cwd().resolve()
        current = parent


def _contains_coverage_pytest_args(args: List[str]) -> bool:
    """Determine whether passthrough pytest arguments request coverage behavior.

    Args:
        args: Caller-owned pytest argument suffix.

    Returns:
        ``True`` when an argument is a recognized coverage pytest control.
    """

    return any(COVERAGE_PYTEST_ARG_PATTERN.match(arg) for arg in args)


def _load_pyproject_addopts(root_dir: Path) -> List[str]:
    """Load default pytest addopts defined in pyproject.toml.

    Args:
        root_dir: Project root used to locate ``pyproject.toml``.

    Returns:
        List of addopts parsed from the configuration file. Returns an empty
        list when the file is missing, unreadable, or does not define addopts.
    """

    pyproject_path = root_dir / "pyproject.toml"
    if not pyproject_path.exists():
        return []
    try:
        data = tomllib.loads(pyproject_path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return []
    addopts = data.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("addopts", "")
    if not isinstance(addopts, str) or not addopts:
        return []
    try:
        return shlex.split(addopts)
    except ValueError:
        return addopts.split()


def _repository_coverage_floor(root_dir: Path, addopts: List[str]) -> float:
    """Return the effective repository coverage floor without weakening policy.

    Args:
        root_dir: Repository root containing optional ``pyproject.toml`` policy.
        addopts: Parsed pytest addopts to inspect for ``--cov-fail-under``.

    Returns:
        The configured floor, strengthened to the wrapper's minimum of 80.

    Raises:
        CoverageSourceValidationError: Configured floor values are nonnumeric or
            conflict between supported configuration locations.
    """
    values: List[float] = []

    def append_finite(value: object, description: str) -> None:
        try:
            parsed = float(str(value))
        except (TypeError, ValueError) as exc:
            raise CoverageSourceValidationError(
                f"configured {description} must be numeric"
            ) from exc
        if not math.isfinite(parsed):
            raise CoverageSourceValidationError(f"configured {description} must be finite")
        values.append(parsed)

    for index, token in enumerate(addopts):
        value: Optional[str] = None
        if token.startswith("--cov-fail-under="):
            value = token.split("=", 1)[1]
        elif token == "--cov-fail-under":
            if index + 1 >= len(addopts):
                raise CoverageSourceValidationError(
                    "configured --cov-fail-under must be followed by a value"
                )
            value = addopts[index + 1]
        if value is not None:
            append_finite(value, "--cov-fail-under")
    pyproject = root_dir / "pyproject.toml"
    if pyproject.exists():
        try:
            data = tomllib.loads(pyproject.read_text())
            value = data.get("tool", {}).get("coverage", {}).get("report", {}).get("fail_under")
            if value is not None:
                append_finite(value, "[tool.coverage.report].fail_under")
        except tomllib.TOMLDecodeError:
            pass
        except OSError as exc:
            raise CoverageSourceValidationError("coverage policy could not be read") from exc
    if len(set(values)) > 1:
        raise CoverageSourceValidationError("configured coverage floors are inconsistent")
    return max(values[0], 80.0) if values else 80.0


def _effective_coverage_floor(root_dir: Path, caller_threshold: Optional[object]) -> float:
    """Combine repository policy with a caller threshold that can only strengthen it.

    Args:
        root_dir: Repository root used to load coverage policy.
        caller_threshold: Optional caller-requested minimum coverage percentage.

    Returns:
        Effective coverage floor that is at least the repository policy floor.

    Raises:
        CoverageSourceValidationError: The caller threshold is invalid or would
            weaken repository policy.
    """
    floor = _repository_coverage_floor(root_dir, _load_pyproject_addopts(root_dir))
    if caller_threshold is None:
        return floor
    if isinstance(caller_threshold, bool) or not isinstance(caller_threshold, (int, float)):
        raise CoverageSourceValidationError("coverageThreshold must be a finite number")
    if not math.isfinite(caller_threshold) or caller_threshold < floor:
        raise CoverageSourceValidationError(
            f"coverageThreshold must be a finite number no lower than repository policy ({floor})"
        )
    return float(caller_threshold)


def _should_apply_coverage_threshold(
    *, coverage_threshold: Optional[float], cov_args: List[str], pytest_args: List[str]
) -> bool:
    """Determine if coverage threshold enforcement should run.

    Args:
        coverage_threshold: Configured minimum coverage percentage. ``None``
            disables enforcement.
        cov_args: Coverage arguments generated by this wrapper.
        pytest_args: Original pytest args passed by caller.

    Returns:
        True when validation should enforce the coverage threshold, False
        otherwise.
    """

    if coverage_threshold is None:
        return False
    for arg in cov_args + pytest_args:
        match = re.match(r"--cov(?:=([^\s]+))?$", arg)
        if match and match.group(1):
            return True
    return False


def _detect_unusable_coverage_diagnostics(output: str) -> Optional[str]:
    """Return a stable error for pytest-cov diagnostics that invalidate coverage.

    Args:
        output: Combined pytest stdout and stderr.

    Returns:
        A reviewer-actionable validation message when pytest-cov reports known
        unusable coverage diagnostics, otherwise ``None``.
    """

    lowered_output = output.lower()
    for fragment in UNUSABLE_COVERAGE_FRAGMENTS:
        if fragment in lowered_output:
            return (
                "Coverage data is unusable: pytest-cov reported "
                f"'{fragment}'. Review coverageSource/import targeting."
            )
    return None


def _extract_section(
    lines: List[str],
    header_pattern: Pattern[str],
    *,
    stop_on_blank: bool,
    max_lines: Optional[int] = None,
) -> List[str]:
    """Extract a section of output starting at a header line.

    Args:
        lines: Output lines to scan.
        header_pattern: Compiled regex matching the header line.
        stop_on_blank: Whether to stop at the first blank line.
        max_lines: Optional cap for the number of lines returned.

    Returns:
        List of lines including the header. Returns an empty list if the header is not found.
    """
    start_index = next(
        (index for index, line in enumerate(lines) if header_pattern.match(line)),
        None,
    )
    if start_index is None:
        return []

    collected: List[str] = []
    for index in range(start_index, len(lines)):
        line = lines[index]
        if index != start_index:
            if stop_on_blank and not line.strip():
                break
            if SECTION_HEADER_PATTERN.match(line) and not header_pattern.match(line):
                break
        collected.append(line)
        if max_lines is not None and len(collected) >= max_lines:
            break
    return collected


def parse_pytest_output(output: str) -> Dict:
    """Parse pytest output to extract key metrics.

    Extracts test counts, runtime, coverage percentage, duration profiling data,
    and failure details from pytest's terminal output using regex patterns.

    Args:
        output: The full pytest output text including summary line
            (e.g., "===== 1630 passed, 8 skipped in 35.20s =====").

    Returns:
        Dictionary with parsed metrics:
            - passed/failed/errors/skipped/warnings: Test counts
            - total: Sum of passed + failed + errors
            - duration: Test run time in seconds
            - coverage_pct: Coverage percentage (0-100) if reported
            - durations: List of slowest test entries with duration, phase, test
            - has_failures/has_errors: Boolean flags
            - failed_tests/error_tests: Lists of test names
            - exit_code: Will be set by caller
    """
    result: Dict[str, Any] = {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "warnings": 0,
        "total": 0,
        "exit_code": None,
        "duration": None,
        "coverage_pct": None,
        "has_failures": False,
        "has_errors": False,
        "failed_tests": [],
        "error_tests": [],
    }

    # Parse test counts from summary line
    # Example: "===== 1630 passed, 8 skipped in 35.20s ====="
    # Or: "===== 1 failed, 1880 passed, 9 skipped in 26.35s ====="
    # Extract all counts from the summary line (order can vary)
    summary_line_pattern = r"=+\s*(.*?)\s+in\s+([\d.]+)s?(?:\s*\([^)]*\))?\s*=+"
    summary_match = re.search(summary_line_pattern, output)

    if summary_match:
        summary_text = summary_match.group(1)
        result["duration"] = float(summary_match.group(2))

        # Extract individual counts (order-independent)
        passed_match = re.search(r"(\d+)\s+passed", summary_text)
        if passed_match:
            result["passed"] = int(passed_match.group(1))

        failed_match = re.search(r"(\d+)\s+failed", summary_text)
        if failed_match:
            result["failed"] = int(failed_match.group(1))

        error_match = re.search(r"(\d+)\s+errors?", summary_text)
        if error_match:
            result["errors"] = int(error_match.group(1))

        skipped_match = re.search(r"(\d+)\s+skipped", summary_text)
        if skipped_match:
            result["skipped"] = int(skipped_match.group(1))

        warning_match = re.search(r"(\d+)\s+warnings?", summary_text)
        if warning_match:
            result["warnings"] = int(warning_match.group(1))

        result["total"] = result["passed"] + result["failed"] + result["errors"]

    # Check for FAILED marker
    failed_pattern = r"^(.*?)\s+FAILED"
    for line in output.split("\n"):
        if " FAILED " in line:
            result["has_failures"] = True
            match = re.match(failed_pattern, line.strip())
            if match:
                result["failed_tests"].append(match.group(1))

    # Check for ERROR marker
    error_pattern = r"^(.*?)\s+ERROR"
    for line in output.split("\n"):
        if " ERROR " in line:
            result["has_errors"] = True
            match = re.match(error_pattern, line.strip())
            if match:
                result["error_tests"].append(match.group(1))

    # Parse coverage percentage
    # Example: "TOTAL        6956   6956     0%"
    coverage_pattern = r"TOTAL\s+\d+\s+\d+\s+(\d+)%"
    match = re.search(coverage_pattern, output)
    if match:
        result["coverage_pct"] = int(match.group(1))

    coverage_files: List[Dict[str, object]] = []

    def _coverage_sort_key(entry: Dict[str, object]) -> int:
        value = entry.get("coverage_pct")
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return 0

    coverage_file_pattern = re.compile(r"^(\S+)\s+(\d+)\s+(\d+)\s+(\d+)%\s*(?:\|\s*)?(.+)?$")
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("-"):
            continue
        if stripped.startswith("Name") or stripped.startswith("Stmts"):
            continue
        if stripped.startswith("TOTAL") or COVERAGE_HEADER_PATTERN.match(stripped):
            continue
        match = coverage_file_pattern.match(stripped)
        if match:
            coverage_files.append(
                {
                    "file": match.group(1),
                    "statements": int(match.group(2)),
                    "missing": int(match.group(3)),
                    "coverage_pct": int(match.group(4)),
                    "missing_lines": (match.group(5) or "").strip(),
                }
            )

    if coverage_files:
        coverage_files_sorted = sorted(coverage_files, key=_coverage_sort_key)
        if len(coverage_files_sorted) > MAX_COVERAGE_FILES:
            result["coverage_files_total"] = len(coverage_files_sorted)
            result["coverage_files_truncated"] = len(coverage_files_sorted) - MAX_COVERAGE_FILES
            coverage_files_sorted = coverage_files_sorted[:MAX_COVERAGE_FILES]
        result["coverage_files"] = coverage_files_sorted

    lines = output.splitlines()
    durations_section = _extract_section(lines, DURATIONS_HEADER_PATTERN, stop_on_blank=True)
    if durations_section:
        entry_pattern = re.compile(r"^([\d.]+)s\s+(\w+)\s+(.+)$")
        durations_entries: List[Dict[str, object]] = []
        for line in durations_section[1:]:
            stripped = line.strip()
            if not stripped or (
                stripped.startswith("(") and "hidden" in stripped and "durations" in stripped
            ):
                continue
            match = entry_pattern.match(stripped)
            if match:
                durations_entries.append(
                    {
                        "duration": float(match.group(1)),
                        "phase": match.group(2),
                        "test": match.group(3),
                    }
                )
        result["durations"] = durations_entries

    return result


def format_summary(
    metrics: Dict,
    validation_errors: List[str],
    coverage_threshold: Optional[float] = None,
    assertion: Optional[Dict[str, Any]] = None,
    coverage: Optional[Dict[str, Any]] = None,
    collection: Optional[dict[str, int]] = None,
) -> str:
    """Format a human-readable summary of test results.

    Generates a structured summary with test counts, runtime, coverage,
    failed test names, slowest test metrics, and validation status.

    Args:
        metrics: Parsed metrics from pytest output including test counts,
            duration, coverage, and failure details.
        validation_errors: List of validation error messages (empty if passed).
        coverage_threshold: Optional minimum coverage percentage for display.
            Shows PASSED/FAILED status relative to threshold.

    Returns:
        Multi-line formatted string with visual separators, test counts,
        coverage status, failed test previews, slowest test data, and
        validation result.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("PYTEST SUMMARY")
    lines.append("=" * 60)
    if assertion:
        lines.append(f"Assertions: {assertion['status'].upper()}")
    if coverage:
        lines.append(f"Coverage: {coverage['status'].upper()}")
    if collection is not None:
        lines.append(
            "Collection: "
            f"{collection['collected_count']} collected, {collection['executed_count']} executed"
        )

    # Test counts
    lines.append(f"\nTests Run: {metrics['total']}")
    lines.append(f"  Passed:  {metrics['passed']}")
    if metrics["failed"] > 0:
        lines.append(f"  Failed:  {metrics['failed']}")
    if metrics["errors"] > 0:
        lines.append(f"  Errors:  {metrics['errors']}")
    if metrics["skipped"] > 0:
        lines.append(f"  Skipped: {metrics['skipped']}")

    # Duration
    if metrics["duration"]:
        lines.append(f"\nDuration: {metrics['duration']:.2f}s")

    # Coverage
    if metrics["coverage_pct"] is not None:
        coverage_status = ""
        if coverage_threshold is not None:
            if metrics["coverage_pct"] >= coverage_threshold:
                coverage_status = f" (threshold: {coverage_threshold}% PASSED)"
            else:
                coverage_status = f" (threshold: {coverage_threshold}% FAILED)"
        lines.append(f"Coverage: {metrics['coverage_pct']}%{coverage_status}")

    coverage_files = metrics.get("coverage_files", [])
    if coverage_files:

        def coverage_entry_pct(entry: Dict[str, object]) -> int:
            value = entry.get("coverage_pct")
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
            return 0

        lines.append("\nCoverage by File:")
        threshold = coverage_threshold
        if threshold is not None:
            below = [entry for entry in coverage_files if coverage_entry_pct(entry) < threshold]
            remaining = [entry for entry in coverage_files if entry not in below]
            remaining_sorted = sorted(remaining, key=coverage_entry_pct)
            ordered = below + remaining_sorted
        else:
            ordered = sorted(coverage_files, key=coverage_entry_pct)
        for entry in ordered[:15]:
            missing_lines = entry.get("missing_lines") or ""
            missing_info = f" (missing: {missing_lines})" if missing_lines else ""
            coverage_pct = coverage_entry_pct(entry)
            lines.append(f"  {entry.get('file', '')} — {coverage_pct}%{missing_info}")
        if len(ordered) > 15:
            lines.append(f"  ... and {len(ordered) - 15} more files")

    # Failed tests
    if metrics["failed_tests"]:
        lines.append(f"\nFailed Tests ({len(metrics['failed_tests'])}):")
        for test in metrics["failed_tests"][:10]:  # Show first 10
            lines.append(f"  - {test}")
        if len(metrics["failed_tests"]) > 10:
            lines.append(f"  ... and {len(metrics['failed_tests']) - 10} more")

    # Error tests
    if metrics["error_tests"]:
        lines.append(f"\nError Tests ({len(metrics['error_tests'])}):")
        for test in metrics["error_tests"][:10]:  # Show first 10
            lines.append(f"  - {test}")
        if len(metrics["error_tests"]) > 10:
            lines.append(f"  ... and {len(metrics['error_tests']) - 10} more")

    # Slowest tests
    durations = metrics.get("durations", [])
    if durations:
        lines.append("\nSlowest Tests:")
        for entry in durations[:30]:
            duration = entry.get("duration")
            phase = entry.get("phase", "")
            test_name = entry.get("test", "")
            lines.append(f"  {duration:>7.2f}s  {phase:<8} {test_name}")
        if len(durations) > 30:
            lines.append(f"  ... and {len(durations) - 30} more")

    # Validation results
    lines.append("\n" + "=" * 60)
    if validation_errors:
        lines.append("VALIDATION: FAILED")
        lines.append("=" * 60)
        for error in validation_errors:
            lines.append(f"  - {error}")
    else:
        lines.append("VALIDATION: PASSED")
        lines.append("=" * 60)
        lines.append("  All validation checks passed")

    return "\n".join(lines)


def validate_results(
    metrics: Dict,
    min_test_count: int = 1,
    coverage_threshold: Optional[int] = None,
) -> List[str]:
    """Validate pytest results against expected criteria.

    Performs multiple validation checks to catch common issues:
    - Test failures or errors
    - Insufficient passing test count (catches collection issues)
    - No tests ran (empty test suite)
    - Coverage below threshold

    Args:
        metrics: Parsed metrics from pytest output including test counts
            and coverage percentage.
        min_test_count: Minimum expected number of passing tests (default: 1).
            Set to ~1700 for full suite, 1 for scoped tests.
        coverage_threshold: Minimum required coverage percentage (0-100),
            or None to skip coverage validation.

    Returns:
        List of validation error messages. Empty list indicates all
        checks passed. Non-empty list triggers exit code 1.
    """
    errors = []

    # Check for failures
    if metrics["has_failures"]:
        errors.append(f"Found {metrics['failed']} failed test(s)")

    # Check for errors
    if metrics["has_errors"]:
        errors.append(f"Found {metrics['errors']} test error(s)")

    # Check test count
    if metrics["passed"] < min_test_count:
        errors.append(
            f"Expected at least {min_test_count} passing tests, but only {metrics['passed']} passed"
        )

    # Check if no tests ran
    if metrics["total"] == 0:
        errors.append("No tests were collected or run")

    # Check coverage threshold
    if coverage_threshold is not None and metrics["coverage_pct"] is not None:
        if metrics["coverage_pct"] < coverage_threshold:
            errors.append(
                f"Coverage {metrics['coverage_pct']}% is below threshold of {coverage_threshold}%"
            )

    return errors


def _evaluate_assertions(
    metrics: Dict[str, Any],
    min_test_count: int,
    returncode: int,
    *,
    coverage_only: bool = False,
) -> Dict[str, Any]:
    """Produce the bounded assertion projection independently of coverage.

    Args:
        metrics: Parsed pytest counts and failure indicators.
        min_test_count: Minimum required number of passing tests.
        returncode: Pytest subprocess exit code.

    Returns:
        Assertion status and ordered failure reasons suitable for structured
        output. A nonzero pytest exit produces a failed status unless it is
        independently classified as a coverage-only failure.
    """
    reasons: List[str] = []
    if metrics["has_failures"]:
        reasons.append(f"Found {metrics['failed']} failed test(s)")
    if metrics["has_errors"]:
        reasons.append(f"Found {metrics['errors']} test error(s)")
    if metrics["passed"] < min_test_count:
        reasons.append(
            f"Expected at least {min_test_count} passing tests, but only {metrics['passed']} passed"
        )
    if metrics["total"] == 0:
        reasons.append("No tests were collected or run")
    if returncode != 0 and not coverage_only:
        reasons.append(
            f"pytest exited with code {returncode}; inspect failed tests and stderr output"
        )
    return {"status": "failed" if reasons else "passed", "reasons": reasons}


def _evaluate_coverage(
    metrics: Dict[str, Any], *, enabled: bool, floor: Optional[float], output: str
) -> Dict[str, Any]:
    """Produce the bounded coverage projection independently of assertions.

    Args:
        metrics: Parsed pytest metrics, including any ``TOTAL`` percentage.
        enabled: Whether the invocation requested coverage collection.
        floor: Effective minimum coverage percentage when coverage is enabled.
        output: Combined pytest standard output and standard error.

    Returns:
        Coverage status, ordered reasons, and a percentage when numeric evidence
        is available. Disabled coverage is reported as ``disabled``, not passed.
    """
    if not enabled:
        return {"status": "disabled", "reasons": []}
    reasons: List[str] = []
    unusable = _detect_unusable_coverage_diagnostics(output)
    percentage = metrics.get("coverage_pct")
    if unusable:
        reasons.append(unusable)
    elif percentage is None:
        reasons.append(
            "Coverage data is unavailable: pytest-cov did not report a TOTAL coverage percentage."
        )
    elif floor is not None and percentage < floor:
        reasons.append(f"Coverage {percentage}% is below threshold of {floor}%")
    result: Dict[str, Any] = {"status": "failed" if reasons else "passed", "reasons": reasons}
    if percentage is not None:
        result["percentage"] = percentage
    return result


def _collection_projection(output: str) -> Optional[dict[str, int]]:
    """Parse exactly one pytest collect-only summary without assertion inference.

    Args:
        output: Combined pytest standard output and standard error.

    Returns:
        Collected and zero-executed counts when exactly one valid collection
        summary is present; otherwise, ``None``.
    """

    matches = re.findall(r"(?:(\d+) tests? collected|(no tests collected))", output, re.I)
    if len(matches) != 1:
        return None
    count, empty = matches[0]
    return {"collected_count": 0 if empty else int(count), "executed_count": 0}


def _resolve_normalized_sources(
    cwd: Optional[str], coverage_source: Optional[Union[str, List[str]]]
) -> List[str]:
    """Resolve normalized coverage sources for the current invocation.

    Args:
        cwd: Requested pytest working directory used to resolve the repository
            root for path-form source validation.
        coverage_source: Optional coverage source configuration from caller.

    Returns:
        Normalized list of repo-relative coverage directories.
    """

    repo_root = _resolve_repo_root_for_coverage(cwd)
    return _normalize_coverage_source(coverage_source, repo_root)


def _coverage_worktree_id(root: Path | str) -> str:
    """Return a non-reversible identifier for the canonical worktree root.

    Args:
        root: Worktree root to canonicalize before deriving the identifier.

    Returns:
        SHA-256 digest of the canonical root path for lease-record comparison.
    """

    return hashlib.sha256(str(Path(root).resolve()).encode("utf-8")).hexdigest()


def _get_coverage_lock_path(root: Path | str) -> Path:
    """Return the repo-local runtime lock path for the given worktree.

    Args:
        root: Canonical worktree root used for the pytest run.

    Returns:
        Absolute path to the deterministic lock file under ``adforge_local/state``.
    """

    runtime_state_dir = Path(root).resolve() / "adforge_local" / "state"
    for directory in (runtime_state_dir.parent, runtime_state_dir):
        try:
            directory.mkdir(exist_ok=True)
            if directory.is_symlink() or not directory.is_dir():
                raise CoverageLockError("coverage coordination state is unavailable; retry later")
        except OSError as exc:
            raise CoverageLockError(
                "coverage coordination state is unavailable; retry later"
            ) from exc
    return runtime_state_dir / COVERAGE_LOCK_FILENAME


def _coverage_lease_guard(lock_path: Path):
    """Return an exclusive no-follow mutex for lease record transitions."""

    guard_path = lock_path.with_suffix(".guard")
    try:
        fd = os.open(guard_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError as exc:
        raise CoverageLockError("coverage coordination state is unavailable; retry later") from exc
    return fd


def _close_coverage_lease_guard(fd: int) -> None:
    """Release a lease transition mutex without surfacing cleanup failures."""

    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _read_coverage_lease(lock_path: Path, worktree_id: str) -> CoverageLease:
    """Read and strictly validate an existing lease without leaking its details.

    Args:
        lock_path: Lease record location within the canonical worktree runtime
            state directory.
        worktree_id: Expected non-reversible canonical-worktree identifier.

    Returns:
        Complete validated ownership record.

    Raises:
        CoverageLockError: The record is unreadable, malformed, or does not
            belong to the expected worktree.
    """

    try:
        fd = os.open(lock_path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CoverageLockError(
            "coverage coordination record could not be verified; retry later"
        ) from exc
    if (
        not isinstance(record, dict)
        or set(record) != {"token", "pid", "worktree_id"}
        or not isinstance(record["token"], str)
        or len(record["token"]) < 16
        or not isinstance(record["pid"], int)
        or isinstance(record["pid"], bool)
        or record["pid"] <= 0
        or record["worktree_id"] != worktree_id
    ):
        raise CoverageLockError("coverage coordination record could not be verified; retry later")
    return CoverageLease(record["token"], record["pid"], record["worktree_id"])


def _recover_stale_coverage_lock(lock_path: Path, worktree_id: str) -> bool:
    """Remove only a reread-verified stale lease.

    Args:
        lock_path: Lease record location to inspect.
        worktree_id: Expected canonical-worktree identifier.

    Returns:
        ``True`` if a verified stale lease was removed; ``False`` if its holder
        is still live.

    Raises:
        CoverageLockError: The record cannot be verified, its liveness is
            indeterminate, or it changes during stale recovery.
    """

    guard_fd = _coverage_lease_guard(lock_path)
    try:
        lease = _read_coverage_lease(lock_path, worktree_id)
        try:
            os.kill(lease.pid, 0)
        except OSError as exc:
            if exc.errno != errno.ESRCH:
                raise CoverageLockError(
                    "coverage coordination record liveness could not be verified; retry later"
                ) from exc
        else:
            return False
        # Every contender holds the same transition guard, so a cooperating
        # contender cannot replace this verified record between check and unlink.
        lock_path.unlink()
        return True
    except OSError as exc:
        raise CoverageLockError(
            "stale coverage coordination record could not be removed; retry later"
        ) from exc
    finally:
        _close_coverage_lease_guard(guard_fd)


def _acquire_coverage_lock(root: Path | str) -> tuple[Path, CoverageLease]:
    """Acquire an exclusive same-worktree coverage lock.

    Args:
        root: Canonical worktree root where the lease should be created.

    Returns:
        The created lock path and its opaque ownership lease.

    Raises:
        CoverageLockError: Another coverage-enabled pytest run is already active
            in the same worktree.
    """

    canonical_root = Path(root).resolve()
    lock_path = _get_coverage_lock_path(canonical_root)
    worktree_id = _coverage_worktree_id(canonical_root)
    lease = CoverageLease(secrets.token_hex(32), os.getpid(), worktree_id)
    for _ in range(2):
        temporary_path = lock_path.with_name(f".{lock_path.name}.{secrets.token_hex(16)}.tmp")
        try:
            fd = os.open(
                temporary_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                0o600,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(lease.as_record(), handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            # link() publishes a fully-written record without replacing an owner.
            os.link(temporary_path, lock_path, follow_symlinks=False)
            temporary_path.unlink()
            break
        except FileExistsError as exc:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            if _recover_stale_coverage_lock(lock_path, worktree_id):
                continue
            raise CoverageLockError(
                "coverage-enabled pytest runs in the same worktree must be serialized; "
                "another coverage run is active; retry after it completes"
            ) from exc
        except OSError as exc:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise CoverageLockError(
                "coverage coordination state is unavailable; retry later"
            ) from exc
    else:
        raise CoverageLockError("coverage coordination could not acquire a lease; retry later")
    return lock_path, lease


def _release_coverage_lock(lock_path: Path, lease: CoverageLease) -> None:
    """Release a lease only when the complete record still belongs to this caller.

    Args:
        lock_path: Lease record location to remove when ownership still matches.
        lease: Complete lease identity acquired by this invocation.
    """

    guard_fd: Optional[int] = None
    try:
        guard_fd = _coverage_lease_guard(lock_path)
        if _read_coverage_lease(lock_path, lease.worktree_id) != lease:
            return
        lock_path.unlink()
    except (CoverageLockError, OSError):
        return
    finally:
        if guard_fd is not None:
            _close_coverage_lease_guard(guard_fd)


def _run_pytest_subprocess(
    cmd: List[str],
    *,
    cwd: str,
    requested_cwd: Optional[str],
    timeout: float | int,
    coverage: bool = True,
) -> PytestSubprocessResult:
    """Execute pytest with worktree-aware PYTHONPATH and timeout cleanup.

    Starts pytest in a new process group so timeout handling can terminate the
    entire pytest tree rather than only the direct child process. When a
    timeout occurs, the function records deterministic diagnostics and raises a
    structured timeout error after cleanup.

    Args:
        cmd: Fully resolved pytest command to execute.
        cwd: Working directory used for the subprocess.
        requested_cwd: Original caller-provided worktree path to prepend to
            ``PYTHONPATH`` for isolated imports.
        timeout: Maximum runtime in seconds. Values must satisfy the shared
            timeout validation contract.
        coverage: Whether coverage controls are enabled. When disabled, inherited
            coverage addopts are removed from the subprocess environment.

    Returns:
        Captured subprocess result containing return code, stdout, and stderr.

    Raises:
        PytestTimedOutError: Pytest exceeded ``timeout`` and the process group
            was terminated.
    """

    env = os.environ.copy()
    if not coverage and "PYTEST_ADDOPTS" in env:
        retained_addopts = _filter_non_coverage_addopts(env["PYTEST_ADDOPTS"])
        if retained_addopts:
            env["PYTEST_ADDOPTS"] = shlex.join(retained_addopts)
        else:
            env.pop("PYTEST_ADDOPTS")
    if requested_cwd:
        existing_pythonpath = env.get("PYTHONPATH") or ""
        env["PYTHONPATH"] = (
            f"{requested_cwd}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else requested_cwd
        )

    timeout_seconds = _validate_timeout_seconds(timeout)
    started_at = time.monotonic()
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return PytestSubprocessResult(
            returncode=process.returncode or 0, stdout=stdout, stderr=stderr
        )
    except subprocess.TimeoutExpired:
        elapsed_seconds = time.monotonic() - started_at
        try:
            process_group_id = os.getpgid(process.pid)
        except ProcessLookupError:
            process_group_id = process.pid
        sigkill_escalated = _terminate_process_group(process, process_group_id)
        raise PytestTimedOutError(
            PytestTimeoutDetails(
                timeout_seconds=timeout_seconds,
                elapsed_seconds=elapsed_seconds,
                pid=process.pid,
                process_group_id=process_group_id,
                cwd=cwd,
                command=cmd,
                sigkill_escalated=sigkill_escalated,
            )
        ) from None


def _build_pytest_command(
    *,
    args: List[str],
    fail_fast: bool,
    durations: Optional[int],
    durations_min: Optional[float],
    coverage: bool,
    normalized_sources: List[str],
    cov_report: str,
    override_ini: Optional[List[str]],
    coverage_floor: Optional[float] = None,
    root_dir: Optional[Path] = None,
    test_path: Optional[str] = None,
    test_paths: Optional[list[str]] = None,
    test_filter: Optional[str] = None,
) -> tuple[List[str], List[str], List[str], str]:
    """Build the pytest command and derived coverage and ini state.

    Runner-owned controls are added before the caller-owned ``args`` suffix,
    which is appended unchanged exactly once.

    Args:
        args: Previously validated caller-owned pytest token suffix.
        fail_fast: Whether to add pytest's stop-on-first-failure control.
        durations: Number of slowest tests to report, if requested.
        durations_min: Minimum duration for reported slow tests, if requested.
        coverage: Whether runner-owned coverage controls are enabled.
        normalized_sources: Validated coverage source names or paths.
        cov_report: Comma-separated runner-owned coverage report formats.
        override_ini: Validated runner-owned pytest ini override entries.
        coverage_floor: Effective runner-owned minimum coverage percentage.
        root_dir: Repository root used to preserve non-coverage pytest addopts.
        test_path: Validated runner-owned test target, if supplied.
        test_paths: Validated ordered runner-owned targets, if supplied.
        test_filter: Runner-owned pytest selection expression, if supplied.

    Returns:
        The command, generated coverage arguments, effective ini overrides, and
        effective coverage report configuration. Coverage controls appear only
        when ``coverage`` is enabled.
    """

    cmd = ["pytest", "-v", "--tb=short"]

    if fail_fast:
        cmd.append("-x")

    if durations is not None:
        cmd.append(f"--durations={durations}")
        if durations_min is not None:
            cmd.append(f"--durations-min={durations_min}")

    effective_override_ini = list(override_ini or [])
    root_dir = root_dir or Path.cwd()
    # Replace configured addopts only after retaining their non-coverage policy.
    configured_addopts = _filter_non_coverage_addopts(_load_pyproject_addopts(root_dir))
    if configured_addopts:
        # ``addopts`` is parsed by pytest as a shell-style argument string.  Re-quote
        # retained tokens so values such as the configured marker expression remain
        # one argument instead of becoming accidental positional test targets.
        effective_override_ini.append("addopts=" + shlex.join(configured_addopts))

    if effective_override_ini:
        cmd.extend([f"--override-ini={entry}" for entry in effective_override_ini])

    cov_args: List[str] = []
    effective_cov_report = cov_report
    if coverage:
        if normalized_sources:
            for source in normalized_sources:
                cov_args.append(f"--cov={source}")
        else:
            cov_args.append("--cov")
        if coverage_floor is not None:
            cov_args.append(f"--cov-fail-under={coverage_floor:g}")
        for report_format in effective_cov_report.split(","):
            if report_format.strip():
                cov_args.append(f"--cov-report={report_format.strip()}")

    if cov_args:
        cmd.extend(cov_args)

    if test_filter:
        cmd.extend(["-k", test_filter])
    if test_path:
        cmd.append(test_path)
    if test_paths:
        cmd.extend(test_paths)
    # ``args`` is the caller-owned, already validated suffix. Keep it intact.
    cmd.extend(args)
    return cmd, cov_args, effective_override_ini, effective_cov_report


def _execution_target(target: str, *, cwd: str) -> str:
    """Convert a validated repository-relative target to the execution cwd."""

    root = _repository_root(cwd)
    path_part, separator, node_id = target.partition("::")
    relative = os.path.relpath(root / path_part, cwd)
    return f"{relative}{separator}{node_id}" if separator else relative


def _execution_pytest_args(args: list[str], *, cwd: str) -> list[str]:
    """Relativize only validated caller target tokens for a nested execution cwd."""

    converted: list[str] = []
    value_follows = False
    for arg in args:
        if value_follows:
            converted.append(arg)
            value_follows = False
        elif arg in PYTEST_ARG_VALUE_OPTIONS:
            converted.append(arg)
            value_follows = True
        elif not arg.startswith("-"):
            converted.append(_execution_target(arg, cwd=cwd))
        else:
            converted.append(arg)
    return converted


def _json_prelaunch_failure(
    reason: str,
    *,
    resolved_target: Optional[str],
    phase: str,
    classification: str = "invocation",
) -> str:
    """Render a canonical identity-bearing JSON failure before pytest starts.

    Args:
        reason: Safe diagnostic reason to include in the bounded outcome.
        resolved_target: Runner-owned test target, when one was resolved.
        phase: Execution phase associated with the failure.
        classification: Stable bounded failure category for the outcome.

    Returns:
        Serialized JSON failure payload with the canonical top-level E37-M2
        evidence identity and a redacted bounded outcome.
    """

    return json.dumps(
        {
            "success": False,
            "evidence_identity": evidence_identity_projection(),
            "outcome": {
                "classification": classification,
                "reason": _redact_diagnostic(reason),
                "exit_code": 1,
                "resolved_target": resolved_target,
                "elapsed_seconds": 0.0,
                "phase": phase,
                "node_ids": [],
                "excerpt": _redact_diagnostic(reason),
                "stdout_tail": "",
                "stderr_tail": "",
                "truncation": {
                    "node_ids_omitted": 0,
                    "excerpt_omitted": 0,
                    "stdout_tail_omitted": 0,
                    "stderr_tail_omitted": 0,
                    "overall_truncated": False,
                    "scan_omitted": 0,
                },
            },
        }
    )


def run_pytest(
    args: List[str],
    output_mode: str = "summary",
    min_test_count: int = 1,
    cwd: Optional[str] = None,
    timeout: Union[int, float] = 600,
    coverage: bool = True,
    coverage_source: Optional[Union[str, List[str]]] = None,
    coverage_threshold: Optional[int] = None,
    cov_report: str = "term-missing",
    fail_fast: bool = False,
    durations: Optional[int] = None,
    durations_min: Optional[float] = None,
    override_ini: Optional[List[str]] = None,
    test_path: Optional[str] = None,
    test_paths: Optional[list[str]] = None,
    test_filter: Optional[str] = None,
) -> Tuple[int, str]:
    """Run pytest with independently evaluated assertions and coverage.

    Executes pytest with the specified options, parses results, and validates
    against expected criteria. Automatically handles worktree PYTHONPATH for
    isolated execution environments. Coverage-enabled runs retain repository
    policy, acquire a no-wait canonical-worktree lease, and fail when pytest-cov
    reports unusable data, omits ``TOTAL``, or falls below the effective floor.
    Disabled coverage is returned as ``disabled``. Collect-only runs require
    disabled coverage and validate collected tests separately from assertions.

    Note:
        -v and --tb=short are always included. Do NOT pass these in args.

    Args:
        args: Caller-owned pytest suffix. Advanced callers must validate and
            transport this ordered sequence through ``--pytest-argv-json``;
            accepted tokens are appended exactly once without normalization.
        output_mode: Output format for results. JSON results carry the canonical
            top-level E37-M2 evidence identity; summary and full text do not.
            One of:
            - "summary": Human-readable with key metrics (default)
            - "full": Complete pytest output + summary (truncated if >500 lines)
            - "json": Structured data for programmatic use
        min_test_count: Minimum expected passing tests (default: 1).
            Set to ~1700 for full suite validation, 1 for scoped tests.
        cwd: Working directory for pytest execution. If provided, prepends
            to PYTHONPATH for worktree isolation. Defaults to project root.
        timeout: Maximum execution time in seconds (default: 600 = 10 min,
            maximum: 1200 = 20 minutes).
        coverage: Enable coverage reporting and policy validation (default:
            ``True``). Disabled coverage rejects coverage-specific controls.
        coverage_source: Existing repo-relative directories for coverage (for
            example, ``adw`` or ``adw/core``). Comma-separated directories are
            supported. ``None`` or ``all`` uses repository coverage configuration;
            dotted modules and file targets are ignored with an informational message.
        coverage_threshold: Optional minimum coverage percentage. It must be
            finite and at least the retained repository policy floor.
        cov_report: Coverage report format(s), comma-separated (default: "term-missing").
            Examples: "html", "xml", "term-missing,html:coverage_html".
        fail_fast: Stop on first failure with -x flag (default: False).
        durations: Show N slowest test durations. Use 0 for all, None to skip.
        durations_min: Minimum duration in seconds for inclusion (default: 0.005).
        override_ini: Optional list of ini overrides passed as
            ``--override-ini=<option>=<value>``. Repository non-coverage
            addopts are retained while coverage addopts are runner-owned.
        test_path: Runner-owned repository-relative test path or node-id target.
        test_paths: Ordered runner-owned repository-relative test path or node-id
            targets. Supply one through seven canonical, confined POSIX targets;
            this is mutually exclusive with ``test_path``.
        test_filter: Runner-owned pytest ``-k`` selection expression.

    Returns:
        Tuple of exit code and rendered output. JSON output includes the
        canonical top-level E37-M2 evidence identity but does not by itself
        establish execution success. Ordinary execution exits zero only when
        assertions pass and coverage either passes or is disabled.
        Collect-only execution instead requires valid collection evidence that
        satisfies ``min_test_count`` and reports collected and zero-executed
        counts without assertion or coverage success evidence.

    Raises:
        Does not raise; errors are captured and returned in output_string.
    """
    # Determine working directory
    requested_cwd = cwd
    if cwd is None:
        # Try to find project root
        current = Path.cwd()
        while current != current.parent:
            if (current / "pyproject.toml").exists() or (current / ".git").exists():
                cwd = str(current)
                break
            current = current.parent
        if cwd is None:
            cwd = str(Path.cwd())

    coverage_lease: Optional[tuple[Path, CoverageLease]] = None
    coverage_info: List[str] = []
    started_at = time.monotonic()

    try:
        _validate_timeout_seconds(timeout)

        if not coverage and (
            _contains_coverage_pytest_args(args)
            or coverage_source is not None
            or coverage_threshold is not None
            or cov_report != "term-missing"
        ):
            raise CoverageSourceValidationError(
                "coverage-specific controls are not allowed when coverage is disabled"
            )
        collect_only = "--collect-only" in args
        if collect_only and coverage:
            raise PytestArgumentValidationError("--collect-only requires coverage: false")
        if test_path is not None and test_paths is not None:
            raise PytestArgumentValidationError("testPath and testPaths cannot be combined")
        root_dir = _resolve_repo_root_for_coverage(cwd)
        normalized_sources = _normalize_coverage_source(coverage_source, root_dir, coverage_info)
        effective_floor = (
            _effective_coverage_floor(root_dir, coverage_threshold) if coverage else None
        )

        if test_path is not None:
            _validate_confined_target(test_path, cwd, "testPath")
            test_path = _execution_target(test_path, cwd=cwd)
        execution_test_paths: Optional[list[str]] = None
        if test_paths is not None:
            execution_test_paths = [
                _execution_target(value, cwd=cwd) for value in _validate_test_paths(test_paths, cwd)
            ]
        args = _execution_pytest_args(args, cwd=cwd)
        if test_filter is not None and (not isinstance(test_filter, str) or not test_filter):
            raise PytestArgumentValidationError("testFilter must be a non-empty string")
        cmd, cov_args, _, _ = _build_pytest_command(
            args=args,
            fail_fast=fail_fast,
            durations=durations,
            durations_min=durations_min,
            coverage=coverage,
            normalized_sources=normalized_sources,
            cov_report=cov_report,
            override_ini=override_ini,
            coverage_floor=effective_floor,
            root_dir=root_dir,
            test_path=test_path,
            test_paths=execution_test_paths,
            test_filter=test_filter,
        )
        cmd = [*_resolve_python_tool_command("pytest", "pytest", cwd), *cmd[1:]]

        if coverage:
            coverage_lease = _acquire_coverage_lock(root_dir)

        result = _run_pytest_subprocess(
            cmd,
            cwd=cwd,
            requested_cwd=requested_cwd,
            timeout=timeout,
            coverage=coverage,
        )

        # Combine stdout and stderr
        full_output = result.stdout
        if result.stderr:
            full_output += "\n\nSTDERR:\n" + result.stderr

        # Parse output
        metrics = parse_pytest_output(full_output)
        metrics["exit_code"] = result.returncode

        coverage_result = _evaluate_coverage(
            metrics, enabled=coverage, floor=effective_floor, output=full_output
        )
        assertion = _evaluate_assertions(
            metrics,
            min_test_count,
            result.returncode,
            coverage_only=(
                result.returncode != 0
                and coverage_result["status"] == "failed"
                and not metrics["has_failures"]
                and not metrics["has_errors"]
                and metrics["passed"] >= min_test_count
                and metrics["total"] > 0
            ),
        )
        validation_errors = [*assertion["reasons"], *coverage_result["reasons"]]
        collection: Optional[dict[str, int]] = None
        collection_outcome: Optional[dict[str, str]] = None
        if collect_only:
            collection = _collection_projection(full_output)
            collection_errors: list[str] = []
            if result.returncode != 0:
                collection_errors.append(
                    f"pytest exited with code {result.returncode} during collection"
                )
            if collection is None:
                collection_errors.append("pytest collection summary is missing or ambiguous")
            elif collection["collected_count"] < min_test_count:
                collection_errors.append(
                    f"Expected at least {min_test_count} collected tests, but only "
                    f"{collection['collected_count']} collected"
                )
            validation_errors = collection_errors
            if not collection_errors:
                collection_outcome = {"classification": "collection", "status": "completed"}

        # Determine final exit code (fail if validation fails)
        exit_code = result.returncode
        if validation_errors:
            exit_code = 1

        outcome = None
        if exit_code != 0:
            outcome = _failure_outcome(
                returncode=result.returncode,
                validation_errors=validation_errors,
                stdout=result.stdout,
                stderr=result.stderr,
                elapsed_seconds=time.monotonic() - started_at,
                resolved_target=test_path,
            )
            if collect_only:
                outcome["classification"] = "collection"

        # Format output based on mode
        if output_mode == "summary":
            output = format_summary(
                metrics,
                validation_errors,
                None if collect_only else effective_floor,
                None if collect_only else assertion,
                None if collect_only else coverage_result,
                collection,
            )
            if outcome is not None:
                output = f"{output}\nOutcome: {json.dumps(outcome, separators=(',', ':'))}"
        elif output_mode == "json":
            payload: dict[str, Any] = {
                "metrics": metrics,
                "durations": metrics.get("durations", []),
                "validation_errors": validation_errors,
                "success": (collection is not None and not validation_errors)
                if collect_only
                else assertion["status"] == "passed"
                and coverage_result["status"] in {"disabled", "passed"},
                "coverage_threshold": effective_floor,
                "evidence_identity": evidence_identity_projection(),
            }
            if collect_only:
                payload.pop("coverage_threshold")
                if collection is not None:
                    payload["collection"] = collection
                if collection_outcome is not None:
                    payload["outcome"] = collection_outcome
            else:
                payload["assertion"] = assertion
                payload["coverage"] = coverage_result
            if collection is not None and not collect_only:
                payload["collection"] = collection
            if outcome is not None:
                payload["success"] = False
                payload["outcome"] = outcome
            if coverage_info:
                payload["info"] = coverage_info
            output = json.dumps(payload, indent=2)
        else:  # full
            # Include summary at the end of full output
            summary = format_summary(
                metrics,
                validation_errors,
                None if collect_only else effective_floor,
                None if collect_only else assertion,
                None if collect_only else coverage_result,
                collection,
            )
            if outcome is not None:
                summary = f"{summary}\nOutcome: {json.dumps(outcome, separators=(',', ':'))}"
            safe_full_output, full_output_omitted = _bounded_diagnostic(
                _redact_diagnostic(full_output), MAX_FULL_OUTPUT_TEXT
            )
            if full_output_omitted:
                safe_full_output = (
                    f"[Output truncated: {full_output_omitted} characters omitted.]\n"
                    f"{safe_full_output}"
                )
            output = f"{safe_full_output}\n\n{summary}"

            # Fall back to smart truncation if full output is too long (>500 lines)
            max_lines = 500
            line_count = len(output.splitlines())
            if line_count > max_lines:
                lines = safe_full_output.splitlines()
                failures_section = _extract_section(
                    lines, FAILURES_HEADER_PATTERN, stop_on_blank=False, max_lines=200
                )
                durations_section = _extract_section(
                    lines, DURATIONS_HEADER_PATTERN, stop_on_blank=True, max_lines=200
                )
                coverage_section = _extract_section(
                    lines, COVERAGE_HEADER_PATTERN, stop_on_blank=True, max_lines=200
                )
                truncated_lines = [
                    f"[Output truncated: {line_count} lines exceeded {max_lines} line limit. "
                    "Showing failures/durations/coverage sections + summary only.]"
                ]
                if failures_section:
                    truncated_lines.append("")
                    truncated_lines.extend(failures_section)
                if durations_section:
                    truncated_lines.append("")
                    truncated_lines.extend(durations_section)
                if coverage_section:
                    truncated_lines.append("")
                    truncated_lines.extend(coverage_section)
                truncated_lines.append("")
                truncated_lines.append(summary)
                output = "\n".join(truncated_lines)

        if coverage_info and output_mode != "json":
            output = f"{' '.join(coverage_info)}\n{output}"
        return exit_code, output

    except (
        CoverageSourceValidationError,
        CoverageLockError,
        PytestTimeoutValidationError,
        PytestArgumentValidationError,
    ) as exc:
        if output_mode == "json":
            return 1, _json_prelaunch_failure(
                str(exc),
                resolved_target=test_path,
                phase="pre_spawn",
                classification="coordination"
                if isinstance(exc, CoverageLockError)
                else "invocation",
            )
        return 1, f"ERROR: {exc}"
    except PytestTimedOutError as exc:
        if output_mode == "json":
            return 1, _json_prelaunch_failure(
                "pytest timed out", resolved_target=test_path, phase="execution"
            )
        return 1, _format_timeout_error(exc.details)
    except FileNotFoundError:
        if output_mode == "json":
            return 1, _json_prelaunch_failure(
                "pytest command not found", resolved_target=test_path, phase="pre_spawn"
            )
        return 1, "ERROR: pytest command not found. Is pytest installed?"
    except Exception:
        if output_mode == "json":
            return 1, _json_prelaunch_failure(
                "pytest runner failed before a safe result could be produced",
                resolved_target=test_path,
                phase="pre_spawn",
                classification="runner",
            )
        return 1, "ERROR: Unexpected error running pytest"
    finally:
        if coverage_lease:
            _release_coverage_lock(*coverage_lease)


def main(argv: Optional[List[str]] = None) -> int:
    """Parse runner controls and execute pytest.

    The advanced transport accepts caller tokens only through
    ``--pytest-argv-json`` as a JSON string array. Plural runner-owned targets
    use ``--test-paths-json`` and remain mutually exclusive with ``--test-path``.
    It rejects invalid transport or target combinations before coverage leasing
    or subprocess spawn.

    Parses command-line arguments and executes pytest with validation.

    Args:
        argv: Optional CLI argument sequence. When omitted, arguments are read
            from ``sys.argv``.

    Returns:
        Exit code (0 if pytest and validation pass, 1 otherwise).
    """
    parser = argparse.ArgumentParser(
        description="Run pytest with coverage and validation",
        epilog="""
Examples:
  %(prog)s                                    Run all tests with coverage
  %(prog)s adw/core/tests/ --min-tests 1      Run scoped tests
  %(prog)s --coverage-threshold 80            Enforce 80%% coverage
  %(prog)s --fail-fast adw/core/tests/        Stop on first failure
  %(prog)s --durations 10                     Show 10 slowest tests
  %(prog)s --cwd /path/to/worktree            Run in worktree

NOTE: -v and --tb=short are always included. Do NOT pass these.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        choices=["summary", "full", "json"],
        default="summary",
        help=(
            "Output mode: summary (default, key metrics), full (complete output), json (structured)"
        ),
    )
    parser.add_argument(
        "--min-tests",
        type=int,
        default=1,
        help="Minimum expected test count (default: 1). Use ~1700 for full suite, 1 for scoped.",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        help="Working directory (defaults to project root). Use for worktree isolation.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Timeout in seconds (default: 600 = 10 minutes, maximum: 1200 = 20 minutes)",
    )
    # Coverage options
    parser.add_argument(
        "--coverage",
        action="store_true",
        default=True,
        help="Enable coverage reporting (default: enabled)",
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage for faster runs",
    )
    parser.add_argument(
        "--coverage-source",
        action="append",
        default=None,
        help=(
            "Existing repo-relative directory for coverage (e.g., 'adw/core'). "
            "Can be repeated or comma-separated. Omit or pass 'all' to use "
            "pyproject.toml config; dotted modules and file targets are ignored."
        ),
    )
    parser.add_argument(
        "--coverage-files-only",
        action="store_true",
        help=(
            "Suppress printing pytest output and only return the exit code "
            "(for tooling/test helpers). Not intended for general CLI usage."
        ),
    )
    parser.add_argument(
        "--coverage-threshold",
        type=int,
        help="Minimum coverage percentage (0-100). Fails validation if below threshold.",
    )
    parser.add_argument(
        "--cov-report",
        type=str,
        default="term-missing",
        help="Coverage report format(s), comma-separated. Examples: 'term-missing', 'html,xml'",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure (-x flag). Good for quick dev feedback.",
    )
    parser.add_argument(
        "--durations",
        type=int,
        default=None,
        help="Show N slowest tests (0 for all). Useful for optimization.",
    )
    parser.add_argument(
        "--durations-min",
        type=float,
        default=None,
        help="Minimum duration in seconds for slowest list (default: 0.005)",
    )
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="Legacy repository-relative pytest targets only (use --test-filter for filtering).",
    )
    parser.add_argument(
        "--override-ini",
        action="append",
        default=[],
        help=(
            "Override ini option (passed through to pytest). Can be repeated,"
            " e.g., --override-ini=addopts=."
        ),
    )
    parser.add_argument(
        "--pytest-argv-json",
        help="Compact JSON array containing the validated caller-owned pytest argv suffix.",
    )
    parser.add_argument(
        "--override-ini-json",
        help="Compact JSON array containing runner-owned override-ini entries.",
    )
    parser.add_argument("--test-path", help="Repository-relative test target owned by the runner.")
    parser.add_argument(
        "--test-paths-json", help="Compact JSON array of runner-owned test targets."
    )
    parser.add_argument("--test-filter", help="Test filter owned by the runner.")

    args = parser.parse_args(argv)

    # Determine coverage setting (--no-coverage overrides --coverage)
    coverage_enabled = not args.no_coverage

    try:
        json_pytest_args = _decode_string_array(args.pytest_argv_json, "pytest argv JSON")
        json_override_ini = _decode_string_array(args.override_ini_json, "override ini JSON")
        json_test_paths = _decode_string_array(args.test_paths_json, "testPaths JSON")
        if json_pytest_args is not None and args.pytest_args:
            raise PytestArgumentValidationError(
                "--pytest-argv-json cannot be combined with legacy positional pytest arguments"
            )
        cwd_for_validation = args.cwd or str(Path.cwd())
        pytest_args = (
            _validate_pytest_argv(json_pytest_args, cwd_for_validation)
            if json_pytest_args is not None
            else list(args.pytest_args)
        )
        override_ini = json_override_ini if json_override_ini is not None else args.override_ini
        if args.test_path is not None and json_test_paths is not None:
            raise PytestArgumentValidationError("testPath and testPaths cannot be combined")
        if json_test_paths is not None:
            _validate_test_paths(json_test_paths, cwd_for_validation)
        if override_ini:
            raise PytestArgumentValidationError("override ini controls are not permitted")
    except PytestArgumentValidationError as exc:
        if not args.coverage_files_only:
            if args.output == "json":
                print(
                    _json_prelaunch_failure(
                        str(exc), resolved_target=None, phase="argument_validation"
                    )
                )
            else:
                print(f"ERROR: {exc}")
        return 1

    exit_code, output = run_pytest(
        pytest_args,
        output_mode=args.output,
        min_test_count=args.min_tests,
        cwd=args.cwd,
        timeout=args.timeout,
        coverage=coverage_enabled,
        coverage_source=args.coverage_source,
        coverage_threshold=args.coverage_threshold,
        cov_report=args.cov_report,
        fail_fast=args.fail_fast,
        durations=args.durations,
        durations_min=args.durations_min,
        override_ini=override_ini,
        test_path=args.test_path,
        test_paths=json_test_paths,
        test_filter=args.test_filter,
    )

    if args.coverage_files_only:
        return exit_code

    print(output)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
