#!/usr/bin/env python3
"""MkDocs build runner tool for ADW.

Runs ``mkdocs build`` with configurable options and returns structured output.
Follows the two-layer tool pattern used across ADW backing scripts.

Usage:
    python3 .opencode/tool/build_mkdocs.py
    python3 .opencode/tool/build_mkdocs.py --output json
    python3 .opencode/tool/build_mkdocs.py --strict --clean
    python3 .opencode/tool/build_mkdocs.py --config-file docs/mkdocs.yml
    python3 .opencode/tool/build_mkdocs.py --validate-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_TIMEOUT = 120
OUTPUT_LINE_LIMIT = 500
OUTPUT_BYTE_LIMIT = 50_000


def _truncate_output(output: str) -> Tuple[str, bool, str]:
    """Truncate output to bounded lines/bytes with a notice.

    Args:
        output: Raw combined stdout and stderr from mkdocs build.

    Returns:
        Tuple of the possibly truncated output, whether truncation occurred,
        and a truncation notice string.
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


def resolve_cwd(cwd: Optional[str]) -> Path:
    """Resolve the working directory for mkdocs execution.

    Args:
        cwd: Optional explicit working directory.

    Returns:
        Path to use as working directory, walking up to find mkdocs.yml or .git
        when cwd is not provided.
    """

    if cwd:
        return Path(cwd)

    current = Path.cwd()
    while True:
        if (current / "mkdocs.yml").exists() or (current / ".git").exists():
            return current
        if current == current.parent:
            return Path.cwd()
        current = current.parent


def resolve_config_path(config_file: str, cwd: Path) -> Path:
    """Resolve the mkdocs config file path against the working directory.

    Args:
        config_file: Config file path from CLI arguments.
        cwd: Working directory to resolve relative paths.

    Returns:
        Absolute path to the configuration file.
    """

    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = cwd / config_path
    return config_path.resolve()


def build_command(
    *,
    strict: bool = False,
    clean: bool = True,
    config_file: str = "mkdocs.yml",
    validate_only: bool = False,
    site_dir: Optional[str] = None,
) -> List[str]:
    """Construct mkdocs build command from parameters.

    Args:
        strict: Whether to enable strict mode.
        clean: Whether to clean the output directory before building.
        config_file: Config file path to pass to mkdocs.
        validate_only: Whether to build to a temporary site directory.
        site_dir: Temporary site directory for validate-only mode.

    Returns:
        List of command arguments for subprocess execution.

    Raises:
        ValueError: If validate_only is True but site_dir is not provided.
    """

    cmd = ["mkdocs", "build"]
    if strict:
        cmd.append("--strict")
    if clean:
        cmd.append("--clean")
    if config_file != "mkdocs.yml":
        cmd.extend(["--config-file", config_file])
    if validate_only:
        if not site_dir:
            raise ValueError("site_dir is required when validate_only is True")
        cmd.extend(["--site-dir", site_dir])
    return cmd


def _combine_output(stdout: str, stderr: str) -> str:
    """Combine stdout and stderr into a single output string.

    Args:
        stdout: Captured standard output.
        stderr: Captured standard error.

    Returns:
        Combined output including labeled stderr when present.
    """
    combined = stdout or ""
    if stderr:
        combined += "\n\nSTDERR:\n" + stderr
    return combined


def format_summary(
    *,
    exit_code: int,
    stdout: str,
    stderr: str,
    error_message: Optional[str] = None,
) -> str:
    """Format a human-readable summary of mkdocs build results.

    Args:
        exit_code: Exit code from the mkdocs process.
        stdout: Captured standard output.
        stderr: Captured standard error.
        error_message: Optional error message from exception handling.

    Returns:
        Multi-line summary string with status and output.
    """

    status = "PASSED" if exit_code == 0 else "FAILED"
    lines: List[str] = ["=" * 60, "MKDOCS BUILD SUMMARY", "=" * 60]
    lines.append(f"\nStatus: {status}")
    lines.append(f"Exit Code: {exit_code}")

    if error_message:
        lines.append(f"Error: {error_message}")

    combined = _combine_output(stdout, stderr)
    if combined.strip():
        truncated_output, _, _ = _truncate_output(combined)
        lines.append("\nOutput:")
        lines.append(truncated_output)
    else:
        lines.append("\nOutput: (none)")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def format_full_output(
    *,
    stdout: str,
    stderr: str,
    error_message: Optional[str] = None,
) -> str:
    """Format full mkdocs output with minimal framing.

    Args:
        stdout: Captured standard output.
        stderr: Captured standard error.
        error_message: Optional error message from exception handling.

    Returns:
        Full output string, truncated if it exceeds size limits.
    """

    combined = _combine_output(stdout, stderr)
    if error_message:
        if combined:
            combined = f"ERROR: {error_message}\n{combined}"
        else:
            combined = f"ERROR: {error_message}"
    truncated_output, _, _ = _truncate_output(combined)
    return truncated_output


def _format_json_output(
    *,
    exit_code: int,
    stdout: str,
    stderr: str,
    options: Dict[str, Any],
    error_message: Optional[str] = None,
) -> str:
    """Format mkdocs output as structured JSON.

    Args:
        exit_code: Exit code from mkdocs execution.
        stdout: Captured standard output.
        stderr: Captured standard error.
        options: Options used to construct the mkdocs command.
        error_message: Optional error message from exception handling.

    Returns:
        JSON string containing structured results.
    """

    combined = _combine_output(stdout, stderr)
    truncated_output, truncated, notice = _truncate_output(combined)
    payload: Dict[str, Any] = {
        "success": exit_code == 0,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "output": truncated_output,
        "truncated": truncated,
        "truncation_notice": notice,
        "options": options,
    }
    if error_message:
        payload["error"] = {"message": error_message}
    return json.dumps(payload, indent=2)


def run_mkdocs(
    *,
    output_mode: str = "summary",
    timeout: int = DEFAULT_TIMEOUT,
    cwd: Optional[str] = None,
    strict: bool = False,
    clean: bool = True,
    config_file: str = "mkdocs.yml",
    validate_only: bool = False,
) -> Tuple[int, str]:
    """Run mkdocs build and return (exit_code, formatted_output).

    Args:
        output_mode: One of ``summary``, ``full``, or ``json``.
        timeout: Timeout in seconds for mkdocs execution.
        cwd: Optional working directory override.
        strict: Whether to enable mkdocs strict mode.
        clean: Whether to clean the output directory before building.
        config_file: Path to mkdocs configuration file.
        validate_only: Whether to build into a temporary directory.

    Returns:
        Tuple of (exit_code, output_string).
    """

    resolved_cwd = resolve_cwd(cwd)
    resolved_config = resolve_config_path(config_file, resolved_cwd)
    if not resolved_config.exists():
        error_message = f"mkdocs config file not found: {resolved_config}"
        options = {
            "cwd": str(resolved_cwd),
            "timeout": timeout,
            "strict": strict,
            "clean": clean,
            "config_file": str(resolved_config),
            "validate_only": validate_only,
        }
        if output_mode == "json":
            return 1, _format_json_output(
                exit_code=1,
                stdout="",
                stderr=error_message,
                options=options,
                error_message=error_message,
            )
        if output_mode == "full":
            return 1, format_full_output(
                stdout="", stderr=error_message, error_message=error_message
            )
        return 1, format_summary(
            exit_code=1,
            stdout="",
            stderr=error_message,
            error_message=error_message,
        )

    stdout = ""
    stderr = ""
    build_error: Optional[str] = None
    site_dir: Optional[str] = None
    if config_file != "mkdocs.yml":
        command_config_file = str(resolved_config)
    else:
        command_config_file = "mkdocs.yml"

    try:
        if validate_only:
            with tempfile.TemporaryDirectory() as tmpdir:
                site_dir = tmpdir
                cmd = build_command(
                    strict=strict,
                    clean=clean,
                    config_file=command_config_file,
                    validate_only=True,
                    site_dir=tmpdir,
                )
                process = subprocess.run(
                    cmd,
                    cwd=str(resolved_cwd),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
        else:
            cmd = build_command(
                strict=strict,
                clean=clean,
                config_file=command_config_file,
                validate_only=False,
            )
            process = subprocess.run(
                cmd,
                cwd=str(resolved_cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
            )

        stdout = process.stdout or ""
        stderr = process.stderr or ""
        exit_code = process.returncode
    except subprocess.TimeoutExpired:
        exit_code = 1
        build_error = f"mkdocs build timed out after {timeout} seconds"
        stderr = build_error
    except FileNotFoundError:
        exit_code = 1
        build_error = "mkdocs not found - is it installed? Install with: pip install mkdocs"
        stderr = build_error
    except Exception as exc:  # pragma: no cover - generic safety net
        exit_code = 1
        build_error = f"Unexpected error running mkdocs: {exc}"
        stderr = build_error

    options = {
        "cwd": str(resolved_cwd),
        "timeout": timeout,
        "strict": strict,
        "clean": clean,
        "config_file": str(resolved_config),
        "validate_only": validate_only,
        "site_dir": site_dir,
    }

    if output_mode == "json":
        return exit_code, _format_json_output(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            options=options,
            error_message=build_error,
        )
    if output_mode == "full":
        return exit_code, format_full_output(
            stdout=stdout, stderr=stderr, error_message=build_error
        )
    return exit_code, format_summary(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        error_message=build_error,
    )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the mkdocs build tool.

    Args:
        argv: Optional list of arguments to use instead of ``sys.argv``.

    Returns:
        Parsed namespace containing the CLI options.
    """

    parser = argparse.ArgumentParser(
        description="Run mkdocs build with ADW-style output handling",
        epilog=(
            "Examples:\n"
            "  python3 .opencode/tool/build_mkdocs.py\n"
            "  python3 .opencode/tool/build_mkdocs.py --output json\n"
            "  python3 .opencode/tool/build_mkdocs.py --strict --clean\n"
            "  python3 .opencode/tool/build_mkdocs.py --config-file docs/mkdocs.yml\n"
            "  python3 .opencode/tool/build_mkdocs.py --validate-only"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        choices=["summary", "full", "json"],
        default="summary",
        help="Output format",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument("--cwd", help="Working directory for mkdocs build")
    parser.add_argument("--strict", action="store_true", help="Enable strict mode")
    parser.add_argument(
        "--clean",
        action="store_true",
        default=True,
        help="Clean build directory before building (default: true)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_false",
        dest="clean",
        help="Disable cleaning the build directory",
    )
    parser.add_argument(
        "--config-file",
        default="mkdocs.yml",
        help="Path to mkdocs configuration file (default: mkdocs.yml)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Build to a temporary directory and discard output",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Run mkdocs build tool with parsed arguments and exit with its status.

    Args:
        argv: Optional list of arguments to override ``sys.argv`` when invoking the tool.

    Raises:
        SystemExit: Always raised with the exit code returned by ``run_mkdocs``.
    """

    args = _parse_args(argv)
    exit_code, output = run_mkdocs(
        output_mode=args.output,
        timeout=args.timeout,
        cwd=args.cwd,
        strict=args.strict,
        clean=args.clean,
        config_file=args.config_file,
        validate_only=args.validate_only,
    )
    print(output)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
