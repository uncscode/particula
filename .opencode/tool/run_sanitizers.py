#!/usr/bin/env python3
"""Sanitizer Runner Tool for ADW.

Executes sanitizer-enabled binaries (ASAN, TSAN, UBSAN), parses their outputs
into structured ``SanitizerError`` objects, and renders summary/full/JSON
reports similar to the run_ctest/run_pytest tools. Supports output truncation,
suppressions, timeout handling, and environment option propagation.

Usage:
    python3 .opencode/tool/run_sanitizers.py --sanitizer asan \\
        --executable ./a.out
    python3 .opencode/tool/run_sanitizers.py --sanitizer tsan \\
        --executable ./race --normal-duration 1.2
    python3 .opencode/tool/run_sanitizers.py --sanitizer ubsan \\
        --executable ./ubsan_target --output-mode json

Notes:
    - Output is truncated to 500 lines or 50KB with a notice to bound memory.
    - Error lists are capped to the first 50 findings to avoid bloat; a note is
      appended when capping occurs.
    - Suppressions filter errors whose location or type contains a suppression
      token (comments/blank lines ignored). Suppressed count is surfaced in
      summary and JSON outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

OUTPUT_LINE_LIMIT = 500
OUTPUT_BYTE_LIMIT = 50_000
DEFAULT_TIMEOUT = 600
ERROR_LIMIT = 50

ASAN_ENV = "ASAN_OPTIONS"
TSAN_ENV = "TSAN_OPTIONS"
UBSAN_ENV = "UBSAN_OPTIONS"

ASAN_HEADER_PATTERN = re.compile(r"AddressSanitizer:\s*(?P<type>[^\n]+)", re.IGNORECASE)
ASAN_ACCESS_PATTERN = re.compile(r"\b(?:READ|WRITE)[^\n]*", re.IGNORECASE)
STACK_FRAME_PATTERN = re.compile(r"#\d+\s+.*")
LOCATION_PATTERN = re.compile(r"(?P<location>[^\s:]+:\d+(?::\d+)?)")
TSAN_HEADER_PATTERN = re.compile(r"ThreadSanitizer:\s*(?P<type>[^\n]+)", re.IGNORECASE)
TSAN_ACCESS_PATTERN = re.compile(r"\b(?:Read|Write) of size [^\n]+", re.IGNORECASE)
TSAN_BLOCK_SEPARATOR = re.compile(r"^=+")
UBSAN_PATTERN = re.compile(
    r"^(?P<location>.+?:\d+(?::\d+)?):\s*runtime error:\s*(?P<type>.+)$",
    re.IGNORECASE,
)


@dataclass
class SanitizerError:
    """Represents a parsed sanitizer error.

    Attributes:
        error_type: Error category or description from the sanitizer.
        location: File:line:col where the error occurred.
        access_info: Optional read/write details (e.g., "READ of size 4").
        stack_trace: List of stack frames showing the call path.
    """

    error_type: str
    location: str
    access_info: Optional[str]
    stack_trace: List[str]


def _truncate_output(output: str) -> Tuple[str, bool, str]:
    """Truncate output to bounded lines/bytes with a notice.

    Args:
        output: Combined stdout and stderr text to bound.

    Returns:
        Tuple containing the bounded output, whether truncation occurred, and
        a human-readable truncation notice (empty string when untouched).
    """

    lines = output.splitlines()
    truncated = False
    notice_parts: List[str] = []

    if len(lines) > OUTPUT_LINE_LIMIT:
        lines = lines[:OUTPUT_LINE_LIMIT]
        truncated = True
        notice_parts.append(f"Output truncated to {OUTPUT_LINE_LIMIT} lines")

    joined = "\n".join(lines)
    if len(joined.encode("utf-8")) > OUTPUT_BYTE_LIMIT:
        encoded = joined.encode("utf-8")[:OUTPUT_BYTE_LIMIT]
        joined = encoded.decode("utf-8", errors="ignore")
        truncated = True
        notice_parts.append(f"Output truncated to {OUTPUT_BYTE_LIMIT // 1024}KB")

    notice = "; ".join(notice_parts) if truncated else ""
    if truncated:
        joined = f"{joined}\n...\n{notice}"
    return joined, truncated, notice


def _extract_location(text: str) -> str:
    """Extract a file:line(:col) location from a stack frame line.

    Args:
        text: Stack frame line to parse.

    Returns:
        Location string when present, otherwise an empty string.
    """

    match = LOCATION_PATTERN.search(text)
    return match.group("location") if match else ""


def _load_suppressions(path: Optional[Path]) -> List[str]:
    """Load suppression tokens from a file.

    Args:
        path: Optional path to a suppressions file. Ignored when None.

    Returns:
        List of non-empty, non-comment tokens. Missing files yield an empty
        list without raising.
    """

    if not path:
        return []
    tokens: List[str] = []
    try:
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens.append(stripped)
    except OSError:
        return []
    return tokens


def _apply_suppressions(
    errors: List[SanitizerError], suppressions: Sequence[str]
) -> Tuple[List[SanitizerError], int]:
    """Filter errors using suppression tokens.

    Args:
        errors: Parsed sanitizer errors.
        suppressions: Tokens to suppress when found in error type or location.

    Returns:
        Tuple of filtered errors and the suppressed count.
    """

    if not suppressions:
        return errors, 0

    filtered: List[SanitizerError] = []
    suppressed_count = 0
    for error in errors:
        haystacks = [error.location, error.error_type]
        if any(token in hay for hay in haystacks for token in suppressions):
            suppressed_count += 1
            continue
        filtered.append(error)
    return filtered, suppressed_count


def parse_asan_output(output: str) -> List[SanitizerError]:
    """Parse AddressSanitizer output into structured errors.

    Args:
        output: Raw (already truncated) sanitizer stdout/stderr text.

    Returns:
        List of ``SanitizerError`` entries extracted from the output.
    """

    errors: List[SanitizerError] = []
    if "AddressSanitizer" not in output:
        return errors

    error_type: Optional[str] = None
    location = ""
    access_info: Optional[str] = None
    stack_trace: List[str] = []

    def flush_current() -> None:
        nonlocal error_type, location, access_info, stack_trace
        if error_type is None:
            return
        errors.append(
            SanitizerError(
                error_type=error_type,
                location=location,
                access_info=access_info,
                stack_trace=list(stack_trace),
            )
        )
        error_type = None
        location = ""
        access_info = None
        stack_trace = []

    for raw_line in output.splitlines():
        line = raw_line.strip("\n")
        header = ASAN_HEADER_PATTERN.search(line)
        if header:
            flush_current()
            error_type = header.group("type").strip()
            continue

        if error_type is None:
            continue

        if access_info is None:
            access_match = ASAN_ACCESS_PATTERN.search(line)
            if access_match:
                access_info = access_match.group(0).strip()

        if STACK_FRAME_PATTERN.match(line.strip()):
            frame = line.strip()
            stack_trace.append(frame)
            if not location:
                loc = _extract_location(frame)
                if loc:
                    location = loc

    flush_current()
    return errors


def parse_tsan_output(output: str) -> List[SanitizerError]:
    """Parse ThreadSanitizer output into structured errors.

    Args:
        output: Raw (already truncated) sanitizer stdout/stderr text.

    Returns:
        List of SanitizerError entries extracted from the output.
    """

    errors: List[SanitizerError] = []
    if "ThreadSanitizer" not in output:
        return errors

    lines = output.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        header = TSAN_HEADER_PATTERN.search(line)
        if not header:
            idx += 1
            continue

        block_lines: List[str] = []
        idx += 1
        while idx < len(lines) and not TSAN_BLOCK_SEPARATOR.match(lines[idx]):
            block_lines.append(lines[idx])
            idx += 1

        access_info = None
        stack_trace: List[str] = []
        location = ""

        for block_line in block_lines:
            if access_info is None:
                access_match = TSAN_ACCESS_PATTERN.search(block_line)
                if access_match:
                    access_info = access_match.group(0).strip()
            if STACK_FRAME_PATTERN.match(block_line.strip()):
                frame = block_line.strip()
                stack_trace.append(frame)
                if not location:
                    loc = _extract_location(frame)
                    if loc:
                        location = loc

        if not location and stack_trace:
            location = _extract_location(stack_trace[0])

        errors.append(
            SanitizerError(
                error_type=header.group("type").strip(),
                location=location,
                access_info=access_info,
                stack_trace=stack_trace,
            )
        )
    return errors


def parse_ubsan_output(output: str) -> List[SanitizerError]:
    """Parse UBSan output into structured errors.

    Args:
        output: Raw (already truncated) sanitizer stdout/stderr text.

    Returns:
        List of SanitizerError entries extracted from the output.
    """

    errors: List[SanitizerError] = []
    if "runtime error" not in output:
        return errors

    lines = output.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        match = UBSAN_PATTERN.match(line)
        if not match:
            idx += 1
            continue

        error_type = match.group("type").strip()
        location = match.group("location").strip()
        stack_trace: List[str] = []
        idx += 1
        while idx < len(lines) and lines[idx].strip():
            frame_line = lines[idx].strip()
            if STACK_FRAME_PATTERN.match(frame_line):
                stack_trace.append(frame_line)
            idx += 1

        errors.append(
            SanitizerError(
                error_type=error_type,
                location=location,
                access_info=None,
                stack_trace=stack_trace,
            )
        )
    return errors


def _select_parser(sanitizer: str) -> Optional[Callable[[str], List[SanitizerError]]]:
    """Return the parser implementation for a sanitizer type.

    Args:
        sanitizer: Sanitizer type string (asan, tsan, or ubsan).

    Returns:
        Parser function for the sanitizer type, or None if unsupported.
    """

    return {
        "asan": parse_asan_output,
        "tsan": parse_tsan_output,
        "ubsan": parse_ubsan_output,
    }.get(sanitizer)


def format_summary(
    sanitizer: str,
    errors: List[SanitizerError],
    duration: Optional[float],
    normal_duration: Optional[float],
    truncated: bool,
    suppressions_info: Tuple[int, bool],
    timed_out: bool,
    exit_code: Optional[int],
    error_cap: bool,
    overhead_ratio: Optional[float],
    message: Optional[str],
) -> str:
    """Create a concise human-readable summary for sanitizer results.

    Args:
        sanitizer: Sanitizer name (asan, tsan, ubsan).
        errors: Filtered sanitizer errors to display.
        duration: Observed execution duration in seconds.
        normal_duration: Optional baseline duration for overhead ratio.
        truncated: Whether output truncation occurred.
        suppressions_info: Tuple of (suppressed_count, suppressions_present flag).
        timed_out: True when execution exceeded timeout.
        exit_code: Process exit code, if available.
        error_cap: True when errors were capped to ERROR_LIMIT.
        overhead_ratio: Computed overhead ratio or None when not provided.
        message: Optional single-line message extracted from raw output.

    Returns:
        Summary string limited to key details for CLI presentation.
    """

    lines: List[str] = []
    lines.append("=" * 60)
    lines.append(f"{sanitizer.upper()} SANITIZER SUMMARY")
    lines.append("=" * 60)

    status = "PASSED"
    if timed_out:
        status = "TIMEOUT"
    elif exit_code not in (0, None):
        status = "FAILED"
    if errors:
        status = "FAILED"

    lines.append(f"Status: {status}")
    if exit_code is not None:
        lines.append(f"Exit Code: {exit_code}")

    if duration is not None:
        lines.append(f"Duration: {duration:.2f}s")
    if overhead_ratio is not None:
        lines.append(f"Overhead Ratio: {overhead_ratio:.2f}x")

    lines.append(f"Errors: {len(errors)}")

    for idx, error in enumerate(errors[:5], start=1):
        lines.append(f"\nError {idx}: {error.error_type}")
        if error.location:
            lines.append(f"  Location: {error.location}")
        if error.access_info:
            lines.append(f"  Access: {error.access_info}")
        if error.stack_trace:
            lines.append("  Stack:")
            for frame in error.stack_trace[:5]:
                lines.append(f"    {frame}")
            if len(error.stack_trace) > 5:
                lines.append("    ...")

    if message:
        lines.append(f"\nMessage: {message}")

    suppressed_count, suppressions_present = suppressions_info
    if suppressions_present:
        lines.append(f"\nSuppressions: {suppressed_count} error(s) filtered")

    if error_cap and len(errors) >= ERROR_LIMIT:
        lines.append(f"\nNote: Errors capped at first {ERROR_LIMIT}")

    if truncated:
        lines.append("\nOutput was truncated for brevity")

    return "\n".join(lines)


def _env_var_for_sanitizer(sanitizer: str) -> Optional[str]:
    """Return the environment variable name for a sanitizer type.

    Args:
        sanitizer: Sanitizer type string (asan, tsan, or ubsan).

    Returns:
        Environment variable name for the sanitizer, or None if unknown.
    """
    if sanitizer == "asan":
        return ASAN_ENV
    if sanitizer == "tsan":
        return TSAN_ENV
    if sanitizer == "ubsan":
        return UBSAN_ENV
    return None


def run_sanitizer(
    *,
    build_dir: Optional[Path],
    executable: Path,
    sanitizer: str,
    timeout: int = DEFAULT_TIMEOUT,
    suppressions: Optional[Path] = None,
    options: Optional[str] = None,
    output_mode: str = "summary",
    normal_duration: Optional[float] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> Tuple[int, str]:
    """Execute a sanitizer-enabled binary and format its results.

    Args:
        build_dir: Optional working directory for the executable.
        executable: Path to the sanitizer-enabled binary.
        sanitizer: Sanitizer type to parse (asan, tsan, ubsan).
        timeout: Subprocess timeout in seconds (default: 600).
        suppressions: Optional path to suppressions file.
        options: Extra sanitizer options appended to the relevant env var.
        output_mode: One of ``summary``, ``full``, or ``json``.
        normal_duration: Baseline duration in seconds for overhead ratio.
        extra_args: Additional arguments passed through to the executable.

    Returns:
        Tuple of (exit_code, rendered_output) following the selected mode.
    """

    parser = _select_parser(sanitizer)
    if parser is None:
        return 1, f"Unsupported sanitizer: {sanitizer}"

    cmd = [str(executable)]
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env_var = _env_var_for_sanitizer(sanitizer)
    if env_var and options:
        existing = env.get(env_var)
        env[env_var] = f"{existing}:{options}" if existing else options

    start = time.monotonic()
    timed_out = False
    combined_output = ""

    try:
        result = subprocess.run(
            cmd,
            cwd=str(build_dir) if build_dir else None,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        combined_output = (result.stdout or "") + (result.stderr or "")
        exit_code = result.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        parts: List[str] = []
        for part in (exc.output, exc.stderr):
            if part:
                if isinstance(part, bytes):
                    parts.append(part.decode("utf-8", errors="ignore"))
                else:
                    parts.append(str(part))
        combined_output = "".join(parts)
        if not combined_output:
            combined_output = f"ERROR: Execution timed out after {timeout} seconds"
        exit_code = 1
    except FileNotFoundError:
        combined_output = f"ERROR: Executable not found: {executable}"
        exit_code = 1

    duration = time.monotonic() - start
    truncated_output, truncated, truncation_notice = _truncate_output(combined_output)

    parsed_errors = parser(truncated_output)
    suppressions_tokens = _load_suppressions(suppressions)
    filtered_errors, suppressed_count = _apply_suppressions(parsed_errors, suppressions_tokens)

    error_cap = False
    if len(filtered_errors) > ERROR_LIMIT:
        filtered_errors = filtered_errors[:ERROR_LIMIT]
        error_cap = True

    overhead_ratio = None
    if normal_duration and normal_duration > 0:
        overhead_ratio = duration / normal_duration

    success = exit_code == 0 and not filtered_errors and not timed_out

    message = None
    if not success:
        stripped_output = truncated_output.strip()
        if stripped_output:
            message = stripped_output.splitlines()[0][:200]

    summary = format_summary(
        sanitizer=sanitizer,
        errors=filtered_errors,
        duration=duration,
        normal_duration=normal_duration,
        truncated=truncated,
        suppressions_info=(suppressed_count, bool(suppressions_tokens)),
        timed_out=timed_out,
        exit_code=exit_code,
        error_cap=error_cap,
        overhead_ratio=overhead_ratio,
        message=message,
    )

    if output_mode == "json":
        payload = {
            "sanitizer": sanitizer,
            "exit_code": exit_code,
            "success": success,
            "timed_out": timed_out,
            "duration": duration,
            "normal_duration": normal_duration,
            "overhead_ratio": overhead_ratio,
            "truncated": truncated,
            "truncation_notice": truncation_notice,
            "suppressed_count": suppressed_count,
            "errors": [asdict(err) for err in filtered_errors],
            "error_cap": error_cap,
        }
        return (0 if success else 1), json.dumps(payload, indent=2)

    if output_mode == "full":
        return (0 if success else 1), truncated_output

    return (0 if success else 1), summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the sanitizer runner."""

    parser = argparse.ArgumentParser(
        description="Run a sanitizer-enabled binary with ADW parsing and formatting",
        epilog=(
            "Examples:\n"
            "  python3 .opencode/tool/run_sanitizers.py --sanitizer asan \\\n"
            "      --executable ./a.out\n"
            "  python3 .opencode/tool/run_sanitizers.py --sanitizer tsan \\\n"
            "      --executable ./race --normal-duration 1.2\n"
            "  python3 .opencode/tool/run_sanitizers.py --sanitizer ubsan \\\n"
            "      --executable ./ubsan_target --output-mode json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sanitizer",
        required=True,
        choices=["asan", "tsan", "ubsan"],
        help="Sanitizer to parse (asan, tsan, ubsan)",
    )
    parser.add_argument(
        "--executable",
        required=True,
        type=Path,
        help="Path to sanitizer-enabled executable",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        help="Working directory for the executable (optional)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--suppressions",
        type=Path,
        help="Path to suppressions file (one token per line; # comments supported)",
    )
    parser.add_argument(
        "--options",
        type=str,
        help="Additional sanitizer options appended to the relevant *_OPTIONS env var",
    )
    parser.add_argument(
        "--normal-duration",
        type=float,
        help="Baseline duration in seconds for overhead ratio calculations",
    )
    parser.add_argument(
        "--output-mode",
        choices=["summary", "full", "json"],
        default="summary",
        help="Output rendering mode",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file to write output to (otherwise prints to stdout)",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to the executable (prefix with --)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for the sanitizer runner."""

    args = _parse_args(argv)
    passthrough = [arg for arg in (args.extra_args or []) if arg != "--"]
    exit_code, output = run_sanitizer(
        build_dir=args.build_dir,
        executable=args.executable,
        sanitizer=args.sanitizer,
        timeout=args.timeout,
        suppressions=args.suppressions,
        options=args.options,
        output_mode=args.output_mode,
        normal_duration=args.normal_duration,
        extra_args=passthrough,
    )

    if args.output:
        args.output.write_text(output)
    else:
        print(output)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
