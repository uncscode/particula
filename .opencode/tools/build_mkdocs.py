#!/usr/bin/env python3
"""Bounded, validate-only MkDocs build runner."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

DEFAULT_TIMEOUT = 600
OUTPUT_LINE_LIMIT = 500
OUTPUT_BYTE_LIMIT = 50_000
OUTPUT_LINE_BYTE_LIMIT = 4_096
EVENT_QUEUE_LIMIT = 64
STREAM_READ_SIZE = 4_096
TERMINATE_GRACE_SECONDS = 0.2
DRAIN_REAP_SECONDS = 0.2
WARNING_LIMIT = 50
PROGRESS_LIMIT = 25
REDACTION_MARKER = "[REDACTED]"
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SECRET_PATTERNS = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{10,}\b"),
    re.compile(r"\b(token|api[_-]?key|secret|password)\s*[:=]\s*([^\s\"']+)", re.I),
    re.compile(r"\bAuthorization\s*:\s*Bearer\s+([^\s\"']+)", re.I),
)
PROGRESS_PATTERN = re.compile(
    r"\b(?:Building documentation|Cleaning site directory|Documentation built)\b", re.I
)
# MkDocs emits warning records with an explicit leading ``WARNING -`` or
# ``WARNING:`` prefix.  Do not treat arbitrary diagnostic prose containing the
# word "warning" as a structured MkDocs warning.
WARNING_PATTERN = re.compile(r"^\s*WARNING\s*(?:-|:)\s*(.+)$", re.I)
MARKDOWN_PATH_PATTERN = re.compile(r"(?<![\w/])([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.md)(?![\w/])")


@dataclass(frozen=True)
class Truncation:
    """Bounded diagnostic truncation state."""

    output_lines: bool = False
    output_bytes: bool = False
    warnings: bool = False
    progress: bool = False


@dataclass
class MkdocsResult:
    """The single bounded model used by every output mode."""

    outcome: str
    success: bool
    exit_code: Optional[int]
    stage: str
    progress: list[str] = field(default_factory=list)
    output: str = ""
    truncation: Truncation = field(default_factory=Truncation)
    warnings: list[dict[str, Optional[str]]] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


def sanitize(value: str) -> str:
    """Sanitize control characters, secret-like values, and whitespace."""
    sanitized = CONTROL_CHARS.sub(" ", value or "")
    sanitized = SECRET_PATTERNS[0].sub(REDACTION_MARKER, sanitized)
    sanitized = SECRET_PATTERNS[1].sub(
        lambda match: f"{match.group(1)}: {REDACTION_MARKER}", sanitized
    )
    sanitized = SECRET_PATTERNS[2].sub(f"Authorization: Bearer {REDACTION_MARKER}", sanitized)
    return re.sub(r"[\t\r ]+", " ", sanitized).strip()


def resolve_cwd(cwd: Optional[str]) -> Path:
    """Resolve explicit cwd, retaining compatibility for direct runner users."""
    return Path(cwd).resolve() if cwd else Path.cwd().resolve()


def resolve_config_path(config_file: str, cwd: Path) -> Path:
    """Resolve a config relative to the supplied working directory."""
    candidate = Path(config_file)
    return (candidate if candidate.is_absolute() else cwd / candidate).resolve()


def build_command(
    *,
    strict: bool = False,
    clean: bool = True,
    config_file: str = "mkdocs.yml",
    validate_only: bool = False,
    site_dir: Optional[str] = None,
) -> list[str]:
    """Build an MkDocs command, requiring a temporary site directory when requested."""
    command = ["mkdocs", "build"]
    if strict:
        command.append("--strict")
    if clean:
        command.append("--clean")
    if config_file != "mkdocs.yml":
        command.extend(["--config-file", config_file])
    if validate_only:
        if not site_dir:
            raise ValueError("site_dir is required when validate_only is True")
        command.extend(["--site-dir", site_dir])
    return command


def _bounded_output(lines: list[str]) -> tuple[str, Truncation]:
    retained: list[str] = []
    byte_count = 0
    line_capped = byte_capped = False
    for line in lines:
        encoded = (line + "\n").encode("utf-8")
        if len(retained) >= OUTPUT_LINE_LIMIT:
            line_capped = True
            continue
        if byte_count + len(encoded) > OUTPUT_BYTE_LIMIT:
            byte_capped = True
            continue
        retained.append(line)
        byte_count += len(encoded)
    return "\n".join(retained), Truncation(output_lines=line_capped, output_bytes=byte_capped)


def _docs_dir(config: Path) -> Path:
    """Read docs_dir from MkDocs where available without widening runner output."""
    try:
        from mkdocs.config import load_config

        loaded = load_config(config_file=str(config))
        return Path(loaded["docs_dir"]).resolve()
    except Exception:
        return (config.parent / "docs").resolve()


def _warning(line: str, docs_dir: Path) -> Optional[dict[str, Optional[str]]]:
    match = WARNING_PATTERN.search(line)
    if not match:
        return None
    candidates: list[Path] = []
    for raw in MARKDOWN_PATH_PATTERN.findall(match.group(1)):
        candidate = (docs_dir / raw).resolve()
        if candidate.is_relative_to(docs_dir):
            candidates.append(candidate)
    unique = list(dict.fromkeys(candidates))
    source = str(unique[0].relative_to(docs_dir)) if len(unique) == 1 else None
    return {
        "message": sanitize(match.group(1)),
        "source": source,
        "attribution": "attributed" if source else "unattributed",
    }


def _render_summary(result: MkdocsResult) -> str:
    lines = [
        "MKDOCS VALIDATION SUMMARY",
        f"Outcome: {result.outcome}",
        f"Stage: {result.stage}",
        f"Exit Code: {result.exit_code}",
    ]
    if result.output:
        lines.extend(["Output:", result.output])
    if any(asdict(result.truncation).values()):
        lines.append(f"Truncation: {json.dumps(asdict(result.truncation), sort_keys=True)}")
    return "\n".join(lines)


def _render_full(result: MkdocsResult) -> str:
    return (
        _render_summary(result)
        + "\nProgress: "
        + json.dumps(result.progress)
        + "\nWarnings: "
        + json.dumps(result.warnings)
    )


def _render(result: MkdocsResult, mode: str) -> str:
    if mode == "json":
        return json.dumps(asdict(result), sort_keys=True)
    return _render_full(result) if mode == "full" else _render_summary(result)


def _drain(
    pipe: Any,
    stream: str,
    events: queue.Queue[tuple[str, Optional[str]]],
    stop: threading.Event,
) -> None:
    """Drain a stream in bounded chunks, applying backpressure through a bounded queue."""
    try:
        while chunk := pipe.read(STREAM_READ_SIZE):
            while not stop.is_set():
                try:
                    events.put((stream, chunk), timeout=0.05)
                    break
                except queue.Full:
                    continue
    finally:
        while not stop.is_set():
            try:
                events.put((stream, None), timeout=0.05)
                break
            except queue.Full:
                continue


def _wait_briefly(process: Any, timeout: float) -> None:
    """Reap a child only for a bounded interval; fake processes may lack timeout support."""
    try:
        process.wait(timeout=timeout)
    except TypeError:
        # Test doubles without a timeout-aware wait are already reaped when poll succeeds.
        return
    except subprocess.TimeoutExpired:
        return


def _terminate_process(process: Any) -> None:
    """Request graceful termination without assuming process-group support from test doubles."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (AttributeError, OSError):
        process.terminate()


def _kill_process(process: Any) -> None:
    """Escalate a still-running process without blocking on held pipe descriptors."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (AttributeError, OSError):
        process.kill()


def run_mkdocs(
    *,
    output_mode: str = "summary",
    timeout: int = DEFAULT_TIMEOUT,
    cwd: Optional[str] = None,
    strict: bool = False,
    clean: bool = True,
    config_file: str = "mkdocs.yml",
    validate_only: bool = False,
    process_factory: Callable[..., Any] = subprocess.Popen,
    clock: Callable[[], float] = time.monotonic,
    terminate_process: Callable[[Any], None] = _terminate_process,
    kill_process: Callable[[Any], None] = _kill_process,
) -> tuple[int, str]:
    """Run MkDocs with concurrent bounded output draining and stable diagnostics."""
    resolved_cwd = resolve_cwd(cwd)
    resolved_config = resolve_config_path(config_file, resolved_cwd)
    options: dict[str, Any] = {
        "cwd": str(resolved_cwd),
        "timeout": timeout,
        "strict": strict,
        "clean": clean,
        "config_file": str(resolved_config),
        "validate_only": validate_only,
        "site_dir": None,
    }
    if not resolved_config.exists():
        result = MkdocsResult(
            "failure",
            False,
            1,
            "config",
            options=options,
            error="mkdocs config file not found",
            output="mkdocs config file not found",
        )
        return 1, _render(result, output_mode)
    if timeout <= 0:
        result = MkdocsResult(
            "failure", False, 1, "admission", options=options, error="timeout must be positive"
        )
        return 1, _render(result, output_mode)

    lines: list[str] = []
    output_bytes = 0
    output_lines_capped = output_bytes_capped = False
    progress: list[str] = []
    warnings: list[dict[str, Optional[str]]] = []
    progress_capped = warnings_capped = False
    outcome = "failure"
    exit_code: Optional[int] = 1
    stage = "launch"
    error: Optional[str] = None
    docs_dir = _docs_dir(resolved_config)
    command_config = str(resolved_config) if config_file != "mkdocs.yml" else "mkdocs.yml"
    try:
        with tempfile.TemporaryDirectory() as site_dir:
            options["site_dir"] = "<temporary>"
            command = build_command(
                strict=strict,
                clean=clean,
                config_file=command_config,
                validate_only=True,
                site_dir=site_dir,
            )
            process = process_factory(
                command,
                cwd=str(resolved_cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            events: queue.Queue[tuple[str, Optional[str]]] = queue.Queue(maxsize=EVENT_QUEUE_LIMIT)
            stop_draining = threading.Event()
            threads = [
                threading.Thread(
                    target=_drain, args=(pipe, label, events, stop_draining), daemon=True
                )
                for pipe, label in ((process.stdout, "stdout"), (process.stderr, "stderr"))
            ]
            for thread in threads:
                thread.start()
            deadline, closed = clock() + timeout, 0
            timed_out = False
            terminate_deadline = drain_deadline = None
            partial_lines = {"stdout": "", "stderr": ""}

            def record(raw_line: str) -> None:
                """Classify one bounded complete stream line without retaining discarded data."""
                nonlocal output_bytes, output_bytes_capped, output_lines_capped
                nonlocal progress_capped, warnings_capped, stage
                line = sanitize(raw_line[:OUTPUT_LINE_BYTE_LIMIT])
                if len(raw_line.encode("utf-8")) > OUTPUT_LINE_BYTE_LIMIT:
                    output_bytes_capped = True
                if not line:
                    return
                encoded = (line + "\n").encode("utf-8")
                if len(lines) >= OUTPUT_LINE_LIMIT:
                    output_lines_capped = True
                elif output_bytes + len(encoded) > OUTPUT_BYTE_LIMIT:
                    output_bytes_capped = True
                else:
                    lines.append(line)
                    output_bytes += len(encoded)
                if PROGRESS_PATTERN.search(line):
                    if len(progress) < PROGRESS_LIMIT:
                        progress.append(line)
                        stage = "build"
                    else:
                        progress_capped = True
                warning = _warning(line, docs_dir)
                if warning:
                    if len(warnings) < WARNING_LIMIT:
                        warnings.append(warning)
                    else:
                        warnings_capped = True

            while True:
                remaining = deadline - clock()
                if remaining <= 0 and not timed_out:
                    timed_out = True
                    stage = "timeout"
                    terminate_process(process)
                    terminate_deadline = clock() + TERMINATE_GRACE_SECONDS
                if timed_out and terminate_deadline is not None and clock() >= terminate_deadline:
                    if process.poll() is None:
                        kill_process(process)
                    terminate_deadline = None
                    drain_deadline = clock() + DRAIN_REAP_SECONDS
                if closed >= 2:
                    break
                if drain_deadline is not None and clock() >= drain_deadline:
                    break
                try:
                    stream, raw = events.get(timeout=max(0.01, min(0.05, max(remaining, 0.01))))
                except queue.Empty:
                    if process.poll() is not None and all(
                        not thread.is_alive() for thread in threads
                    ):
                        break
                    continue
                if raw is None:
                    record(partial_lines[stream])
                    partial_lines[stream] = ""
                    closed += 1
                    continue
                partial_lines[stream] += raw
                complete_lines = partial_lines[stream].splitlines(keepends=True)
                partial_lines[stream] = ""
                if complete_lines and not complete_lines[-1].endswith(("\n", "\r")):
                    partial_lines[stream] = complete_lines.pop()
                for complete_line in complete_lines:
                    record(complete_line.rstrip("\r\n"))
                if len(partial_lines[stream].encode("utf-8")) > OUTPUT_LINE_BYTE_LIMIT:
                    record(partial_lines[stream])
                    partial_lines[stream] = ""
                    output_bytes_capped = True
            if timed_out and process.poll() is None:
                kill_process(process)
            stop_draining.set()
            for pipe in (process.stdout, process.stderr):
                try:
                    pipe.close()
                except (AttributeError, OSError):
                    pass
            for thread in threads:
                thread.join(timeout=DRAIN_REAP_SECONDS)
            _wait_briefly(process, DRAIN_REAP_SECONDS)
            if timed_out:
                outcome, exit_code, error = (
                    "timeout",
                    None,
                    f"mkdocs build timed out after {timeout} seconds",
                )
            else:
                exit_code = process.returncode
                outcome = "success" if exit_code == 0 else "failure"
                stage = "complete" if exit_code == 0 else "build"
    except FileNotFoundError:
        error, lines, stage = (
            "mkdocs not found - install with: pip install mkdocs",
            ["mkdocs not found"],
            "launch",
        )
    except Exception as exc:
        error, lines, stage = (
            f"unexpected mkdocs runner error: {sanitize(str(exc))}",
            ["mkdocs runner failed"],
            "launch",
        )

    output = "\n".join(lines)
    truncation = Truncation(
        output_lines_capped,
        output_bytes_capped,
        warnings_capped,
        progress_capped,
    )
    result = MkdocsResult(
        outcome,
        outcome == "success",
        exit_code,
        stage,
        progress,
        output,
        truncation,
        warnings,
        options,
        error,
    )
    status = 0 if outcome == "success" else 1
    return status, _render(result, output_mode)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded MkDocs validation")
    parser.add_argument("--output", choices=["summary", "full", "json"], default="summary")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--cwd")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--clean", action="store_true", default=True)
    parser.add_argument("--no-clean", action="store_false", dest="clean")
    parser.add_argument("--config-file", default="mkdocs.yml")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    code, output = run_mkdocs(
        output_mode=args.output,
        timeout=args.timeout,
        cwd=args.cwd,
        strict=args.strict,
        clean=args.clean,
        config_file=args.config_file,
        validate_only=args.validate_only,
    )
    print(output)
    sys.exit(code)


if __name__ == "__main__":
    main()
