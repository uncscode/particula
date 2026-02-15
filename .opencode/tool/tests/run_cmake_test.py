"""Unit tests for run_cmake.py.

Covers parsing, formatting, CLI handling, and error scenarios without
invoking real CMake.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Tuple

import pytest

RUN_CMAKE_PATH = Path(__file__).resolve().parent.parent / "run_cmake.py"


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_cmake", RUN_CMAKE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


@pytest.fixture(scope="module")
def run_cmake_module() -> ModuleType:
    return load_module()


class DummyResult:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_parse_cmake_output_success(run_cmake_module: ModuleType) -> None:
    output = """
CMake Generator: Ninja
Preset CMake variables:
  CMAKE_BUILD_TYPE="Debug"
Target example_app (EXECUTABLE)
-- Configuring done
-- Generating done
-- Build files have been written to: /tmp/build
"""
    metrics = run_cmake_module.parse_cmake_output(output, exit_code=0)
    assert metrics["success"] is True
    assert metrics["generator"] == "Ninja"
    assert metrics["build_type"] == "Debug"
    assert metrics["warnings"] == 0
    assert metrics["errors"] == 0
    assert len(metrics["targets"]) == 1


def test_parse_cmake_output_errors_and_warnings(run_cmake_module: ModuleType) -> None:
    output = """
CMake Warning at CMakeLists.txt:10 (message):
  Deprecated feature
CMake Error at CMakeLists.txt:20 (find_package):
  Missing dependency
"""
    metrics = run_cmake_module.parse_cmake_output(output, exit_code=1)
    assert metrics["success"] is False
    assert metrics["warnings"] == 1
    assert metrics["errors"] == 1
    assert len(metrics["warning_messages"]) == 1
    assert len(metrics["error_messages"]) == 1


def test_parse_cmake_output_truncates_targets(run_cmake_module: ModuleType) -> None:
    targets = "\n".join([f"Target target_{i} (EXECUTABLE)" for i in range(60)])
    metrics = run_cmake_module.parse_cmake_output(targets, exit_code=0)
    assert len(metrics["targets"]) == 50
    assert metrics["truncated_targets"] is True
    summary = run_cmake_module.format_summary(metrics, "src", "build")
    assert "truncated" in summary


def test_format_summary_handles_missing_fields(run_cmake_module: ModuleType) -> None:
    metrics = {
        "success": False,
        "generator": None,
        "build_type": None,
        "targets": [],
        "warnings": 0,
        "errors": 1,
        "duration": None,
        "ninja_fallback": False,
    }
    summary = run_cmake_module.format_summary(metrics, "src", "build")
    assert "Configuration: Failed" in summary
    assert "VALIDATION: FAILED" in summary


@pytest.fixture()
def preset_file(tmp_path: Path) -> Tuple[Path, Path]:
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    presets = {
        "version": 3,
        "cmakeMinimumRequired": {"major": 3, "minor": 21, "patch": 0},
        "configurePresets": [
            {"name": "debug", "binaryDir": "build"},
            {"name": "release", "binaryDir": "build-release"},
        ],
    }
    (source_dir / "CMakePresets.json").write_text(json.dumps(presets))
    build_dir = tmp_path / "build"
    return source_dir, build_dir


def test_run_cmake_with_preset_builds_command(
    monkeypatch: pytest.MonkeyPatch, preset_file: Tuple[Path, Path]
) -> None:
    module = load_module()
    source_dir, build_dir = preset_file

    def fake_run(cmd: list, capture_output: bool, text: bool, timeout: int) -> DummyResult:  # type: ignore[override]
        assert "--preset" in cmd
        assert "debug" in cmd
        return DummyResult(returncode=0, stdout="-- Configuring done\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    exit_code, output = module.run_cmake(
        source_dir=str(source_dir),
        build_dir=str(build_dir),
        preset="debug",
        output_mode="summary",
    )
    assert exit_code == 0
    assert "VALIDATION: PASSED" in output


def test_run_cmake_ninja_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    source_dir = tmp_path / "src"
    build_dir = tmp_path / "build"
    source_dir.mkdir()

    def fake_which(_: str) -> None:
        return None

    def fake_run(cmd: list, capture_output: bool, text: bool, timeout: int) -> DummyResult:  # type: ignore[override]
        # Ensure -G Ninja not present when fallback triggered
        assert "-G" not in cmd
        return DummyResult(returncode=0, stdout="-- Configuring done\n", stderr="")

    monkeypatch.setattr(module.shutil, "which", fake_which)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code, output = module.run_cmake(
        source_dir=str(source_dir),
        build_dir=str(build_dir),
        ninja=True,
        output_mode="summary",
    )
    assert exit_code == 0
    assert "Ninja requested" in output


def test_run_cmake_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    source_dir = tmp_path / "src"
    build_dir = tmp_path / "build"
    source_dir.mkdir()

    def fake_run(*_, **__):
        raise subprocess.TimeoutExpired(cmd="cmake", timeout=1, output="partial", stderr="late")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    exit_code, output = module.run_cmake(
        source_dir=str(source_dir),
        build_dir=str(build_dir),
        timeout=1,
        output_mode="summary",
    )
    assert exit_code == 1
    assert "timed out" in output.lower()


def test_run_cmake_not_found(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    source_dir = tmp_path / "src"
    build_dir = tmp_path / "build"
    source_dir.mkdir()

    def fake_run(*_, **__):
        raise FileNotFoundError()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    exit_code, output = module.run_cmake(source_dir=str(source_dir), build_dir=str(build_dir))
    assert exit_code == 1
    assert "not available" not in output


def test_run_cmake_invalid_preset_name(preset_file: Tuple[Path, Path]) -> None:
    module = load_module()
    source_dir, build_dir = preset_file
    exit_code, output = module.run_cmake(
        source_dir=str(source_dir),
        build_dir=str(build_dir),
        preset="nonexistent",
    )
    assert exit_code == 1
    assert "Preset 'nonexistent' not found" in output
    assert "CMakeUserPresets.json" in output


def test_run_cmake_missing_preset_file(tmp_path: Path) -> None:
    module = load_module()
    source_dir = tmp_path / "src"
    build_dir = tmp_path / "build"
    source_dir.mkdir()
    exit_code, output = module.run_cmake(
        source_dir=str(source_dir),
        build_dir=str(build_dir),
        preset="debug",
    )
    assert exit_code == 1
    assert "CMakePresets.json" in output


def test_load_presets_missing_user_file(tmp_path: Path) -> None:
    module = load_module()
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    presets = {
        "version": 3,
        "configurePresets": [
            {"name": "debug", "binaryDir": "build"},
        ],
    }
    (source_dir / "CMakePresets.json").write_text(json.dumps(presets))

    loaded = module._load_presets(str(source_dir))
    names = [item["name"] for item in loaded.get("configurePresets", [])]
    assert names == ["debug"]


def test_load_presets_merges_user_presets(tmp_path: Path) -> None:
    module = load_module()
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    presets = {
        "version": 3,
        "configurePresets": [
            {"name": "debug", "binaryDir": "build"},
        ],
    }
    user_presets = {
        "version": 3,
        "configurePresets": [
            {"name": "user", "binaryDir": "build-user"},
        ],
    }
    (source_dir / "CMakePresets.json").write_text(json.dumps(presets))
    (source_dir / "CMakeUserPresets.json").write_text(json.dumps(user_presets))

    loaded = module._load_presets(str(source_dir))
    names = [item["name"] for item in loaded.get("configurePresets", [])]
    assert names == ["debug", "user"]


def test_validate_preset_name_accepts_user_preset(tmp_path: Path) -> None:
    module = load_module()
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    presets = {"version": 3, "configurePresets": []}
    user_presets = {
        "version": 3,
        "configurePresets": [
            {"name": "user", "binaryDir": "build-user"},
        ],
    }
    (source_dir / "CMakePresets.json").write_text(json.dumps(presets))
    (source_dir / "CMakeUserPresets.json").write_text(json.dumps(user_presets))

    loaded = module._load_presets(str(source_dir))
    module._validate_preset_name("user", loaded)


def test_load_presets_skips_malformed_user_file(tmp_path: Path) -> None:
    module = load_module()
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    presets = {"version": 3, "configurePresets": [{"name": "debug"}]}
    (source_dir / "CMakePresets.json").write_text(json.dumps(presets))
    (source_dir / "CMakeUserPresets.json").write_text("{bad json")

    loaded = module._load_presets(str(source_dir))
    names = [item["name"] for item in loaded.get("configurePresets", [])]
    assert names == ["debug"]


def test_load_presets_skips_missing_user_configure_presets(tmp_path: Path) -> None:
    module = load_module()
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    presets = {"version": 3, "configurePresets": [{"name": "debug"}]}
    user_presets = {"version": 3}
    (source_dir / "CMakePresets.json").write_text(json.dumps(presets))
    (source_dir / "CMakeUserPresets.json").write_text(json.dumps(user_presets))

    loaded = module._load_presets(str(source_dir))
    names = [item["name"] for item in loaded.get("configurePresets", [])]
    assert names == ["debug"]


def test_load_presets_none_user_configure_presets(tmp_path: Path) -> None:
    module = load_module()
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    presets = {"version": 3, "configurePresets": [{"name": "debug"}]}
    (source_dir / "CMakePresets.json").write_text(json.dumps(presets))
    (source_dir / "CMakeUserPresets.json").write_text(
        json.dumps({"version": 3, "configurePresets": None})
    )

    loaded = module._load_presets(str(source_dir))
    names = [item["name"] for item in loaded.get("configurePresets", [])]
    assert names == ["debug"]


def test_load_presets_invalid_user_configure_presets(tmp_path: Path) -> None:
    module = load_module()
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    presets = {"version": 3, "configurePresets": [{"name": "debug"}]}
    (source_dir / "CMakePresets.json").write_text(json.dumps(presets))
    (source_dir / "CMakeUserPresets.json").write_text(
        json.dumps({"version": 3, "configurePresets": {"name": "user"}})
    )

    with pytest.raises(TypeError, match="CMakeUserPresets.json configurePresets"):
        module._load_presets(str(source_dir))


def test_run_cmake_preset_not_supported(
    monkeypatch: pytest.MonkeyPatch, preset_file: Tuple[Path, Path]
) -> None:
    module = load_module()
    source_dir, build_dir = preset_file

    def fake_run(*_, **__):  # type: ignore[override]
        return DummyResult(
            returncode=1,
            stdout="",
            stderr="Presets are not supported by this CMake version",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    exit_code, output = module.run_cmake(
        source_dir=str(source_dir),
        build_dir=str(build_dir),
        preset="debug",
        output_mode="summary",
    )
    assert exit_code == 1
    assert "not supported" in output.lower()


def test_run_cmake_json_truncation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    source_dir = tmp_path / "src"
    build_dir = tmp_path / "build"
    source_dir.mkdir()

    long_output = "\n".join([f"line {i}" for i in range(600)])

    def fake_run(*_, **__):  # type: ignore[override]
        return DummyResult(returncode=0, stdout=long_output, stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    exit_code, output = module.run_cmake(
        source_dir=str(source_dir),
        build_dir=str(build_dir),
        output_mode="json",
    )
    assert exit_code == 0
    payload = json.loads(output)
    assert payload["truncated"] is True
    assert "truncated" in payload["truncation_notice"]


def test_cli_parses_arguments(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_module()

    def fake_run_cmake(**kwargs):  # type: ignore[override]
        assert kwargs["output_mode"] == "summary"
        return 0, "CLI OK"

    monkeypatch.setattr(module, "run_cmake", fake_run_cmake)

    with pytest.raises(SystemExit) as excinfo:
        module.main(["--source-dir", "src", "--build-dir", "build", "--timeout", "10"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr().out
    assert "CLI OK" in captured


def test_cli_json_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_module()

    def fake_run_cmake(**kwargs):  # type: ignore[override]
        assert kwargs["output_mode"] == "json"
        return 0, json.dumps({"ok": True})

    monkeypatch.setattr(module, "run_cmake", fake_run_cmake)

    with pytest.raises(SystemExit) as excinfo:
        module.main(["--output", "json"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr().out
    assert '"ok"' in captured


def test_truncate_output_byte_limit(run_cmake_module: ModuleType) -> None:
    long_output = "x" * 60_000
    truncated, was_truncated, notice = run_cmake_module._truncate_output(long_output)
    assert was_truncated is True
    assert "48KB" in notice
    assert notice in truncated


def test_run_cmake_full_output_truncation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    source_dir = tmp_path / "src"
    build_dir = tmp_path / "build"
    source_dir.mkdir()

    long_output = "\n".join([f"line {i}" for i in range(600)])

    def fake_run(*_, **__):  # type: ignore[override]
        return DummyResult(returncode=0, stdout=long_output, stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    exit_code, output = module.run_cmake(
        source_dir=str(source_dir), build_dir=str(build_dir), output_mode="full"
    )

    assert exit_code == 0
    assert "Output truncated" in output
    assert "CMAKE SUMMARY" in output


def test_run_cmake_ninja_generator(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    source_dir = tmp_path / "src"
    build_dir = tmp_path / "build"
    source_dir.mkdir()

    def fake_which(_: str) -> str:
        return "/usr/bin/ninja"

    def fake_run(cmd: list, capture_output: bool, text: bool, timeout: int) -> DummyResult:  # type: ignore[override]
        assert ["-G", "Ninja"] in [cmd[i : i + 2] for i in range(len(cmd) - 1)]
        return DummyResult(returncode=0, stdout="-- Configuring done\n", stderr="")

    monkeypatch.setattr(module.shutil, "which", fake_which)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code, output = module.run_cmake(
        source_dir=str(source_dir), build_dir=str(build_dir), ninja=True, output_mode="summary"
    )

    assert exit_code == 0
    assert "Ninja requested" not in output
