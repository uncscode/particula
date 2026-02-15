#!/usr/bin/env python3
"""CMake Configuration Tool for ADW.

Runs CMake configuration with preset/generator support, summary parsing,
and multiple output modes modeled after ``run_pytest.py``.

Usage:
    python3 run_cmake.py --preset default
    python3 run_cmake.py --ninja --source-dir example_cpp_dev
    python3 run_cmake.py --preset debug --output full

Examples:
    # Configure with preset
    python3 .opencode/tool/run_cmake.py --preset ninja-release

    # Configure with Ninja generator
    python3 .opencode/tool/run_cmake.py --ninja --source-dir example_cpp_dev

    # Full output mode for debugging
    python3 .opencode/tool/run_cmake.py --preset debug --output full
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

OUTPUT_LINE_LIMIT = 500
OUTPUT_BYTE_LIMIT = 50_000
TARGET_SUMMARY_LIMIT = 10
MESSAGE_CAPTURE_LIMIT = 50


def _truncate_output(output: str) -> Tuple[str, bool, str]:
    """Truncate output to bounded lines/bytes with a notice.

    Args:
        output: Raw combined stdout/stderr string.

    Returns:
        Tuple of (truncated_output, was_truncated, notice).
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


def _bounded_append(collection: List[str], value: str, limit: int) -> None:
    """Append to a list while respecting a maximum size."""
    if len(collection) < limit:
        collection.append(value)


def parse_cmake_output(output: str, exit_code: Optional[int] = None) -> Dict[str, Any]:
    """Parse CMake output to extract key metrics.

    Processes the output line-by-line to collect generator, build type,
    targets, warnings, errors, duration, and success flag.

    Args:
        output: Combined stdout/stderr from CMake.
        exit_code: Optional exit code to propagate into metrics.

    Returns:
        Metrics dictionary including generator, build_type, targets,
        warnings/errors, message lists, duration, success, and exit_code.
    """
    metrics: Dict[str, Any] = {
        "generator": None,
        "build_type": None,
        "targets": [],
        "warnings": 0,
        "errors": 0,
        "warning_messages": [],
        "error_messages": [],
        "duration": None,
        "success": True,
        "exit_code": exit_code,
        "timeout": False,
        "ninja_fallback": False,
        "truncated_targets": False,
    }

    target_pattern = re.compile(r"Target\s+(.+?)\s+\(([^)]+)\)")
    built_target_pattern = re.compile(r"Built target\s+(.+)")
    generator_pattern = re.compile(r"CMake\s+Generator:\s*(.+)")
    build_type_pattern = re.compile(r"CMAKE_BUILD_TYPE[\s:=\"]+([A-Za-z0-9_+-]+)")
    duration_pattern = re.compile(r"([0-9]+(?:\.[0-9]+)?)s(?:ec|econds)?")

    targets_list = cast(List[Dict[str, str]], metrics["targets"])
    warning_messages = cast(List[str], metrics["warning_messages"])
    error_messages = cast(List[str], metrics["error_messages"])

    for line in output.splitlines():
        stripped = line.strip()

        if not metrics["generator"]:
            gen_match = generator_pattern.search(stripped)
            if gen_match:
                metrics["generator"] = gen_match.group(1).strip()

        if not metrics["build_type"]:
            bt_match = build_type_pattern.search(stripped)
            if bt_match:
                metrics["build_type"] = bt_match.group(1).strip()

        tgt_match = target_pattern.search(stripped)
        if tgt_match:
            if len(targets_list) >= MESSAGE_CAPTURE_LIMIT:
                metrics["truncated_targets"] = True
            else:
                targets_list.append(
                    {"name": tgt_match.group(1).strip(), "type": tgt_match.group(2).strip()}
                )

        built_match = built_target_pattern.search(stripped)
        if built_match:
            if len(targets_list) < MESSAGE_CAPTURE_LIMIT:
                targets_list.append({"name": built_match.group(1).strip(), "type": "unknown"})
            else:
                metrics["truncated_targets"] = True

        if stripped.lower().startswith("cmake warning"):
            metrics["warnings"] += 1
            _bounded_append(warning_messages, stripped, MESSAGE_CAPTURE_LIMIT)

        if stripped.lower().startswith("cmake error"):
            metrics["errors"] += 1
            _bounded_append(error_messages, stripped, MESSAGE_CAPTURE_LIMIT)

        if metrics["duration"] is None:
            dur_match = duration_pattern.search(stripped)
            if dur_match:
                try:
                    metrics["duration"] = float(dur_match.group(1))
                except ValueError:
                    metrics["duration"] = None

    if metrics["errors"] > 0:
        metrics["success"] = False

    if exit_code is not None:
        metrics["exit_code"] = exit_code
        if exit_code != 0:
            metrics["success"] = False

    return metrics


def format_summary(metrics: Dict[str, Any], source_dir: str, build_dir: str) -> str:
    """Format human-readable summary of CMake configuration.

    Args:
        metrics: Parsed metrics dictionary produced by ``parse_cmake_output``.
        source_dir: Source directory passed to CMake.
        build_dir: Build directory passed to CMake.

    Returns:
        A formatted multi-line string summarizing configuration results, warnings, errors,
        and validation status.
    """
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("CMAKE SUMMARY")
    lines.append("=" * 60)

    status = "Success" if metrics.get("success") else "Failed"
    lines.append(f"\nConfiguration: {status}")

    if metrics.get("generator"):
        lines.append(f"Generator: {metrics['generator']}")
    if metrics.get("build_type"):
        lines.append(f"Build Type: {metrics['build_type']}")
    lines.append(f"Source Dir: {source_dir}")
    lines.append(f"Build Dir: {build_dir}")

    targets: Iterable[Dict[str, str]] = metrics.get("targets") or []
    targets_list = list(targets)
    if targets_list:
        lines.append(f"\nTargets Configured: {len(targets_list)}")
        for target in targets_list[:TARGET_SUMMARY_LIMIT]:
            lines.append(f"  - {target.get('name')} ({target.get('type')})")
        remaining = len(targets_list) - TARGET_SUMMARY_LIMIT
        if remaining > 0 or metrics.get("truncated_targets"):
            extra = remaining if remaining > 0 else "more"
            lines.append(f"  ... and {extra} (truncated)")

    lines.append(f"\nWarnings: {metrics.get('warnings', 0)}")
    lines.append(f"Errors: {metrics.get('errors', 0)}")

    warning_messages = metrics.get("warning_messages") or []
    if warning_messages:
        lines.append("\nWarning Messages:")
        for message in warning_messages[:5]:
            lines.append(f"  - {message}")
        if len(warning_messages) > 5:
            lines.append(f"  ... and {len(warning_messages) - 5} more")

    error_messages = metrics.get("error_messages") or []
    if error_messages:
        lines.append("\nError Messages:")
        for message in error_messages[:5]:
            lines.append(f"  - {message}")
        if len(error_messages) > 5:
            lines.append(f"  ... and {len(error_messages) - 5} more")

    if metrics.get("duration") is not None:
        lines.append(f"\nDuration: {metrics['duration']:.2f}s")

    if metrics.get("ninja_fallback"):
        lines.append("\nNote: Ninja requested but not available; used default generator")

    lines.append("\n" + "=" * 60)
    validation = "PASSED" if metrics.get("success") and metrics.get("errors", 0) == 0 else "FAILED"
    lines.append(f"VALIDATION: {validation}")
    lines.append("=" * 60)

    return "\n".join(lines)


def _load_presets(source_dir: str) -> Dict[str, Any]:
    """Load configure presets from CMake preset files.

    Args:
        source_dir: Directory containing CMakePresets.json and optional CMakeUserPresets.json.

    Returns:
        Parsed preset data with merged ``configurePresets`` entries from both files.

    Raises:
        FileNotFoundError: If CMakePresets.json is missing.
        json.JSONDecodeError: If CMakePresets.json is not valid JSON.
        TypeError: If either configurePresets value is not a list.
    """
    presets_path = Path(source_dir) / "CMakePresets.json"
    if not presets_path.exists():
        raise FileNotFoundError(f"CMakePresets.json not found at {presets_path}")
    preset_data = json.loads(presets_path.read_text())

    configure_presets = preset_data.get("configurePresets")
    if configure_presets is None:
        configure_presets = []
    if not isinstance(configure_presets, list):
        raise TypeError("CMakePresets.json configurePresets is not a list")

    user_presets_path = Path(source_dir) / "CMakeUserPresets.json"
    if user_presets_path.exists():
        # CMake convention: merge user configure presets into the standard preset list.
        try:
            user_data = json.loads(user_presets_path.read_text())
        except json.JSONDecodeError:
            user_data = None

        if isinstance(user_data, dict):
            user_configure_presets = user_data.get("configurePresets")
            if user_configure_presets is None:
                user_configure_presets = []
            if not isinstance(user_configure_presets, list):
                raise TypeError("CMakeUserPresets.json configurePresets is not a list")
            configure_presets.extend(user_configure_presets)

    preset_data["configurePresets"] = configure_presets
    return preset_data


def _validate_preset_name(preset: str, preset_data: Dict[str, Any]) -> None:
    """Validate a configure preset name across merged preset files.

    Args:
        preset: Preset name to validate.
        preset_data: Preset data returned by ``_load_presets``.

    Raises:
        ValueError: If the preset name is not present in merged configure presets.
    """
    configure_presets = preset_data.get("configurePresets") or []
    names = {
        item.get("name") for item in configure_presets if isinstance(item, dict) and "name" in item
    }
    if preset not in names:
        raise ValueError(
            f"Preset '{preset}' not found in CMakePresets.json or CMakeUserPresets.json"
        )


def run_cmake(
    source_dir: str,
    build_dir: str,
    preset: Optional[str] = None,
    ninja: bool = False,
    cmake_args: Optional[List[str]] = None,
    timeout: int = 300,
    output_mode: str = "summary",
) -> Tuple[int, str]:
    """Run CMake configuration with preset and Ninja support.

    Args:
        source_dir: Path to the CMake source directory.
        build_dir: Path to the build directory (created when not using presets).
        preset: Optional preset name to pass to ``cmake --preset``; validated when set.
        ninja: Whether to request the Ninja generator (ignored when using presets).
        cmake_args: Additional arguments forwarded to the CMake invocation.
        timeout: Timeout in seconds for the CMake process.
        output_mode: Output format, one of ``summary``, ``full``, or ``json``.

    Returns:
        A tuple of (exit_code, output_text) where ``output_text`` is formatted according
        to ``output_mode`` and ``exit_code`` is 0 on success and 1 on failure.
    """
    cmake_args = cmake_args or []
    cmd: List[str] = ["cmake"]
    metrics: Dict[str, Any]
    preset_data: Optional[Dict[str, Any]] = None
    ninja_requested = bool(preset is None and ninja)

    try:
        if preset:
            preset_data = _load_presets(source_dir)
            _validate_preset_name(preset, preset_data)
            cmd.extend(["--preset", preset])
        else:
            Path(build_dir).mkdir(parents=True, exist_ok=True)
            cmd.extend(["-S", source_dir, "-B", build_dir])
            if ninja_requested:
                if shutil.which("ninja"):
                    cmd.extend(["-G", "Ninja"])
                else:
                    ninja_requested = False
        cmd.extend(cmake_args)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        metrics = parse_cmake_output("", exit_code=1)
        metrics["success"] = False
        metrics["errors"] = metrics.get("errors", 0) + 1
        _bounded_append(cast(List[str], metrics["error_messages"]), str(exc), MESSAGE_CAPTURE_LIMIT)
        output = format_summary(metrics, source_dir, build_dir)
        return 1, output

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        combined_output = result.stdout + ("\n" + result.stderr if result.stderr else "")
        metrics = parse_cmake_output(combined_output, exit_code=result.returncode)

        if ninja and not ninja_requested:
            metrics["ninja_fallback"] = True
            metrics["warnings"] = metrics.get("warnings", 0) + 1
            _bounded_append(
                cast(List[str], metrics["warning_messages"]),
                "Ninja requested but not available; used default generator",
                MESSAGE_CAPTURE_LIMIT,
            )

        if preset and result.stderr:
            lower_err = result.stderr.lower()
            if "preset" in lower_err and "not supported" in lower_err:
                metrics["success"] = False
                metrics["errors"] = metrics.get("errors", 0) + 1
                _bounded_append(
                    cast(List[str], metrics["error_messages"]),
                    "CMake preset not supported by this CMake version",
                    MESSAGE_CAPTURE_LIMIT,
                )

        truncated_output, was_truncated, notice = _truncate_output(combined_output)

        if output_mode == "summary":
            output_text = format_summary(metrics, source_dir, build_dir)
        elif output_mode == "full":
            parts = [truncated_output]
            if was_truncated and notice:
                parts.append(notice)
            parts.append("")
            parts.append(format_summary(metrics, source_dir, build_dir))
            output_text = "\n".join(part for part in parts if part)
        elif output_mode == "json":
            payload = {
                "metrics": metrics,
                "output": truncated_output,
                "truncated": was_truncated,
                "truncation_notice": notice,
            }
            output_text = json.dumps(payload, indent=2)
        else:
            raise ValueError(f"Unsupported output mode: {output_mode}")

        exit_code = 0 if metrics.get("success") else 1
        return exit_code, output_text

    except subprocess.TimeoutExpired as exc:
        partial_out = str(exc.stdout or "")
        partial_err = str(exc.stderr or "")
        combined = (partial_out + "\n" + partial_err).strip()
        metrics = parse_cmake_output(combined, exit_code=1)
        metrics["success"] = False
        metrics["timeout"] = True
        metrics["errors"] = metrics.get("errors", 0) + 1
        _bounded_append(
            cast(List[str], metrics["error_messages"]),
            f"CMake timed out after {timeout} seconds",
            MESSAGE_CAPTURE_LIMIT,
        )
        truncated_output, was_truncated, notice = _truncate_output(combined)
        if output_mode == "json":
            payload = {
                "metrics": metrics,
                "output": truncated_output,
                "truncated": was_truncated,
                "truncation_notice": notice,
            }
            return 1, json.dumps(payload, indent=2)
        output_text = format_summary(metrics, source_dir, build_dir)
        return 1, output_text
    except FileNotFoundError:
        message = "ERROR: cmake command not found. Is CMake installed?"
        metrics = parse_cmake_output("", exit_code=1)
        metrics["success"] = False
        metrics["errors"] = metrics.get("errors", 0) + 1
        _bounded_append(cast(List[str], metrics["error_messages"]), message, MESSAGE_CAPTURE_LIMIT)
        if output_mode == "json":
            payload = {
                "metrics": metrics,
                "output": message,
                "truncated": False,
                "truncation_notice": "",
            }
            return 1, json.dumps(payload, indent=2)
        return 1, format_summary(metrics, source_dir, build_dir)
    except Exception as exc:  # pragma: no cover - safety net
        message = f"ERROR: {exc}"
        metrics = parse_cmake_output("", exit_code=1)
        metrics["success"] = False
        metrics["errors"] = metrics.get("errors", 0) + 1
        _bounded_append(cast(List[str], metrics["error_messages"]), message, MESSAGE_CAPTURE_LIMIT)
        if output_mode == "json":
            payload = {
                "metrics": metrics,
                "output": message,
                "truncated": False,
                "truncation_notice": "",
            }
            return 1, json.dumps(payload, indent=2)
        return 1, format_summary(metrics, source_dir, build_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI.

    Returns:
        Configured ``ArgumentParser`` for the run_cmake command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="Run CMake configuration with preset and Ninja support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 .opencode/tool/run_cmake.py --preset debug\n"
            "  python3 .opencode/tool/run_cmake.py --ninja --source-dir example_cpp_dev\n"
            "  python3 .opencode/tool/run_cmake.py --preset release --output json\n"
        ),
    )
    parser.add_argument("--output", choices=["summary", "full", "json"], default="summary")
    parser.add_argument("--preset", type=str, help="CMake preset name")
    parser.add_argument("--source-dir", type=str, default=".", help="Source directory")
    parser.add_argument("--build-dir", type=str, default="build", help="Build directory")
    parser.add_argument("--ninja", action="store_true", help="Use Ninja generator")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    parser.add_argument("cmake_args", nargs="*", help="Additional CMake arguments")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entrypoint for running CMake configuration.

    Args:
        argv: Optional list of command-line arguments; defaults to ``sys.argv``.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    exit_code, output = run_cmake(
        source_dir=args.source_dir,
        build_dir=args.build_dir,
        preset=args.preset,
        ninja=args.ninja,
        cmake_args=args.cmake_args,
        timeout=args.timeout,
        output_mode=args.output,
    )

    print(output)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
