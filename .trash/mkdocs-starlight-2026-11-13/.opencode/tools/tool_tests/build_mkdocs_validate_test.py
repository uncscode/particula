"""Focused bounded-model tests for the MkDocs validation runner."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import threading
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "build_mkdocs.py"
SPEC = importlib.util.spec_from_file_location("build_mkdocs", MODULE_PATH)
assert SPEC and SPEC.loader
build_mkdocs = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = build_mkdocs
SPEC.loader.exec_module(build_mkdocs)


class FakeProcess:
    """Minimal reaped process with independent stdout/stderr streams."""

    def __init__(self, stdout: str, stderr: str, returncode: int = 0) -> None:
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self.returncode = returncode
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.returncode


class UncooperativeProcess:
    """Process double that ignores termination until kill and keeps pipes open."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.stdout = _HeldStream(self.release)
        self.stderr = _HeldStream(self.release)
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self.wait_timeouts: list[float] = []

    def terminate(self) -> None:
        self.terminated = True

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.release.set()

    def wait(self, timeout: float | None = None) -> int:
        if timeout is not None:
            self.wait_timeouts.append(timeout)
        return self.returncode if self.returncode is not None else 0


class _HeldStream:
    """A stream that holds EOF until its owner has been killed."""

    def __init__(self, release: threading.Event) -> None:
        self.release = release

    def read(self, _size: int) -> str:
        self.release.wait(timeout=1)
        return ""


def _config(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    config = tmp_path / "mkdocs.yml"
    config.write_text("site_name: test\ndocs_dir: docs\n", encoding="utf-8")
    return config


def test_json_result_is_bounded_redacted_and_has_no_raw_streams(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def factory(*_args: object, **_kwargs: object) -> FakeProcess:
        return FakeProcess(
            "Building documentation...\ntoken=super-secret", "WARNING - bad.md: warning"
        )

    code, output = build_mkdocs.run_mkdocs(
        output_mode="json",
        cwd=str(tmp_path),
        config_file=str(config),
        strict=True,
        process_factory=factory,
    )
    payload = json.loads(output)
    assert code == 0
    assert payload["outcome"] == "success"
    assert payload["exit_code"] == 0
    assert "stdout" not in payload and "stderr" not in payload
    assert "super-secret" not in payload["output"]
    assert payload["warnings"][0]["source"] == "bad.md"


def test_nonzero_build_is_failure_and_every_mode_uses_same_model(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def factory(*_args: object, **_kwargs: object) -> FakeProcess:
        return FakeProcess("Building documentation", "WARNING - ambiguous a.md b.md", 2)

    code, json_output = build_mkdocs.run_mkdocs(
        output_mode="json", cwd=str(tmp_path), config_file=str(config), process_factory=factory
    )
    _, summary = build_mkdocs.run_mkdocs(
        output_mode="summary", cwd=str(tmp_path), config_file=str(config), process_factory=factory
    )
    _, full = build_mkdocs.run_mkdocs(
        output_mode="full", cwd=str(tmp_path), config_file=str(config), process_factory=factory
    )
    payload = json.loads(json_output)
    assert code == 1 and payload["outcome"] == "failure" and payload["exit_code"] == 2
    assert payload["warnings"][0]["attribution"] == "unattributed"
    assert "Outcome: failure" in summary
    assert '"attribution": "unattributed"' in full


def test_temporary_site_directory_is_always_injected_and_removed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    commands: list[list[str]] = []

    def factory(command: list[str], **_kwargs: object) -> FakeProcess:
        commands.append(command)
        return FakeProcess("Documentation built", "")

    build_mkdocs.run_mkdocs(
        cwd=str(tmp_path), config_file=str(config), clean=False, process_factory=factory
    )
    build_mkdocs.run_mkdocs(
        cwd=str(tmp_path), config_file=str(config), clean=True, process_factory=factory
    )
    site_dirs = [Path(command[command.index("--site-dir") + 1]) for command in commands]
    assert all(not site_dir.exists() for site_dir in site_dirs)
    assert site_dirs[0] != site_dirs[1]
    assert "--clean" not in commands[0]
    assert "--clean" in commands[1]


def test_runner_admission_and_missing_config_are_bounded_failures(tmp_path: Path) -> None:
    """Invalid timeout and configuration fail before process creation."""
    config = tmp_path / "missing.yml"

    code, output = build_mkdocs.run_mkdocs(
        output_mode="json", cwd=str(tmp_path), config_file=str(config)
    )
    payload = json.loads(output)
    assert code == 1
    assert payload["outcome"] == "failure"
    assert payload["stage"] == "config"
    assert payload["exit_code"] == 1

    config = _config(tmp_path)
    code, output = build_mkdocs.run_mkdocs(
        output_mode="json", timeout=0, cwd=str(tmp_path), config_file=str(config)
    )
    payload = json.loads(output)
    assert code == 1
    assert payload["stage"] == "admission"
    assert payload["error"] == "timeout must be positive"


def test_bounded_helpers_redact_clip_and_conservatively_attribute_warnings(tmp_path: Path) -> None:
    """Retained diagnostics cap output and only attribute one safe Markdown path."""
    output, truncation = build_mkdocs._bounded_output(
        ["x" * build_mkdocs.OUTPUT_BYTE_LIMIT, "tail"]
    )
    assert output == "tail"
    assert truncation.output_bytes is True

    sanitized = build_mkdocs.sanitize("Authorization: Bearer hidden\x00 token=also-hidden")
    assert "hidden" not in sanitized
    assert sanitized.count("[REDACTED]") == 2

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    assert build_mkdocs._warning("WARNING - guide/page.md: bad", docs_dir) == {
        "message": "guide/page.md: bad",
        "source": "guide/page.md",
        "attribution": "attributed",
    }
    assert (
        build_mkdocs._warning("WARNING - one.md two.md", docs_dir)["attribution"] == "unattributed"
    )
    assert build_mkdocs._warning("not a warning", docs_dir) is None


def test_runner_bounds_streaming_capture_while_draining_both_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Oversized dual-stream output is discarded during capture rather than after collection."""
    config = _config(tmp_path)
    monkeypatch.setattr(build_mkdocs, "STREAM_READ_SIZE", 8)
    monkeypatch.setattr(build_mkdocs, "OUTPUT_LINE_LIMIT", 2)
    monkeypatch.setattr(build_mkdocs, "OUTPUT_BYTE_LIMIT", 100)
    monkeypatch.setattr(build_mkdocs, "OUTPUT_LINE_BYTE_LIMIT", 6)

    def factory(*_args: object, **_kwargs: object) -> FakeProcess:
        return FakeProcess("abcdefgh" * 4, "ijklmnop" * 4)

    _, output = build_mkdocs.run_mkdocs(
        output_mode="json", cwd=str(tmp_path), config_file=str(config), process_factory=factory
    )
    payload = json.loads(output)
    assert len(payload["output"].encode("utf-8")) <= 100
    assert payload["truncation"]["output_bytes"] is True
    assert payload["truncation"]["output_lines"] is True


def test_timeout_escalates_and_reaps_without_waiting_for_held_pipe_eof(tmp_path: Path) -> None:
    """Timeout cleanup terminates, kills, and boundedly reaps an uncooperative child."""
    config = _config(tmp_path)
    process = UncooperativeProcess()
    ticks = iter(range(100))

    code, output = build_mkdocs.run_mkdocs(
        output_mode="json",
        timeout=1,
        cwd=str(tmp_path),
        config_file=str(config),
        process_factory=lambda *_args, **_kwargs: process,
        clock=lambda: float(next(ticks)),
    )

    payload = json.loads(output)
    assert code == 1
    assert payload["outcome"] == "timeout"
    assert process.terminated and process.killed
    assert process.wait_timeouts == [build_mkdocs.DRAIN_REAP_SECONDS]


def test_build_command_requires_ephemeral_site_dir_for_validate_only() -> None:
    """Command construction retains compatibility flags but rejects missing validation output."""
    assert build_mkdocs.build_command(
        strict=True, clean=False, config_file="custom.yml", validate_only=True, site_dir="/tmp/site"
    ) == ["mkdocs", "build", "--strict", "--config-file", "custom.yml", "--site-dir", "/tmp/site"]
    try:
        build_mkdocs.build_command(validate_only=True)
    except ValueError as error:
        assert "site_dir is required" in str(error)
    else:
        raise AssertionError("validate-only command unexpectedly accepted no site directory")
