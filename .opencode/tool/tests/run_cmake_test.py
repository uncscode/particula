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
from typing import Any, Tuple

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
    def __init__(
        self,
        returncode: int = 0,
        stdout: bytes | str | None = b"",
        stderr: bytes | str | None = b"",
    ) -> None:
        self.returncode = returncode
        self.stdout = self._coerce_bytes(stdout)
        self.stderr = self._coerce_bytes(stderr)

    @staticmethod
    def _coerce_bytes(value: bytes | str | None) -> bytes | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            return value
        return value.encode("utf-8")


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


def test_parse_compiler_errors(run_cmake_module: ModuleType) -> None:
    output = """
src/main.cc:10: error: missing include
src/lib.cc:5: Fatal Error: header not found
"""
    metrics = run_cmake_module.parse_cmake_output(output, exit_code=1)
    assert metrics["errors"] == 2
    assert any("missing include" in msg for msg in metrics["error_messages"])
    assert any("header not found" in msg for msg in metrics["error_messages"])


def test_parse_compiler_warnings(run_cmake_module: ModuleType) -> None:
    output = """
src/main.cc:20: Warning: unused variable
"""
    metrics = run_cmake_module.parse_cmake_output(output, exit_code=0)
    assert metrics["warnings"] == 1
    assert any("unused variable" in msg for msg in metrics["warning_messages"])


def test_parse_cmake_output_truncates_targets(run_cmake_module: ModuleType) -> None:
    targets = "\n".join([f"Target target_{i} (EXECUTABLE)" for i in range(60)])
    metrics = run_cmake_module.parse_cmake_output(targets, exit_code=0)
    assert len(metrics["targets"]) == 50
    assert metrics["truncated_targets"] is True
    summary = run_cmake_module.format_summary(metrics, "src", "build")
    assert "truncated" in summary


def test_bounded_append_limits_list(run_cmake_module: ModuleType) -> None:
    items: list[str] = []
    run_cmake_module._bounded_append(items, "first", limit=1)
    run_cmake_module._bounded_append(items, "second", limit=1)
    assert items == ["first"]


def test_format_summary_handles_missing_fields(run_cmake_module: ModuleType) -> None:
    metrics: dict[str, Any] = {
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


def test_build_arg_parser_defaults(run_cmake_module: ModuleType) -> None:
    parser = run_cmake_module.build_arg_parser()
    args = parser.parse_args([])
    assert args.output == "summary"
    assert args.preset is None
    assert args.source_dir == "."
    assert args.build_dir == "build"
    assert args.ninja is False
    assert args.timeout == 300
    assert args.build is False
    assert args.jobs == 0
    assert args.build_timeout == 1800
    assert args.cmake_args == []


def test_cli_build_args(run_cmake_module: ModuleType) -> None:
    parser = run_cmake_module.build_arg_parser()
    args = parser.parse_args(["--build", "--jobs", "8", "--build-timeout", "3600"])
    assert args.build is True
    assert args.jobs == 8
    assert args.build_timeout == 3600


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
        assert text is False
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
        assert text is False
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


def test_binary_timeout_handling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    source_dir = tmp_path / "src"
    build_dir = tmp_path / "build"
    source_dir.mkdir()

    def fake_run(*_, **__):
        raise subprocess.TimeoutExpired(
            cmd="cmake",
            timeout=1,
            output=b"partial\xff",
            stderr=None,
        )

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
    assert "CMakePresets.json" in output
    assert "CMakeUserPresets.json" in output
    assert "checked both files" in output


def test_binary_output_decode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    source_dir = tmp_path / "src"
    build_dir = tmp_path / "build"
    source_dir.mkdir()

    def fake_run(*_, **__):
        return DummyResult(returncode=0, stdout=None, stderr=b"warn\xfe")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    exit_code, output = module.run_cmake(
        source_dir=str(source_dir),
        build_dir=str(build_dir),
        output_mode="full",
    )
    assert exit_code == 0
    assert "\ufffd" in output


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


def test_load_presets_merges_build_presets(tmp_path: Path) -> None:
    module = load_module()
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    presets = {
        "version": 3,
        "configurePresets": [{"name": "debug", "binaryDir": "build"}],
        "buildPresets": [{"name": "build-debug", "configurePreset": "debug"}],
    }
    user_presets = {
        "version": 3,
        "buildPresets": [{"name": "build-user", "configurePreset": "debug"}],
    }
    (source_dir / "CMakePresets.json").write_text(json.dumps(presets))
    (source_dir / "CMakeUserPresets.json").write_text(json.dumps(user_presets))

    loaded = module._load_presets(str(source_dir))
    build_names = [item["name"] for item in loaded.get("buildPresets", [])]
    assert build_names == ["build-debug", "build-user"]


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


def test_build_cmake_success(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()

    def fake_run(cmd: list, capture_output: bool, text: bool, timeout: int) -> DummyResult:  # type: ignore[override]
        assert cmd[:2] == ["cmake", "--build"]
        assert text is False
        return DummyResult(returncode=0, stdout="Built target app\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    exit_code, output, metrics = module.build_cmake(build_dir="build", output_mode="summary")
    assert exit_code == 0
    assert metrics["success"] is True
    assert "BUILD SUMMARY" in output


def test_build_cmake_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()

    def fake_run(cmd: list, capture_output: bool, text: bool, timeout: int) -> DummyResult:  # type: ignore[override]
        assert cmd[:2] == ["cmake", "--build"]
        return DummyResult(returncode=1, stdout="", stderr="src/main.cc: error: fail")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    exit_code, output, metrics = module.build_cmake(build_dir="build", output_mode="summary")
    assert exit_code == 1
    assert metrics["success"] is False
    assert "Errors: 1" in output


def test_build_cmake_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()

    def fake_run(*_, **__):
        raise subprocess.TimeoutExpired(cmd="cmake --build", timeout=1, output="partial")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    exit_code, output, metrics = module.build_cmake(build_dir="build", build_timeout=1)
    assert exit_code == 1
    assert metrics["timeout"] is True
    assert "timed out" in output.lower()


def test_build_cmake_parallel_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()

    def fake_run(cmd: list, capture_output: bool, text: bool, timeout: int) -> DummyResult:  # type: ignore[override]
        assert ["--parallel", "4"] in [cmd[i : i + 2] for i in range(len(cmd) - 1)]
        return DummyResult(returncode=0, stdout="Built target app\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    exit_code, _, _ = module.build_cmake(build_dir="build", jobs=4)
    assert exit_code == 0


def test_build_cmake_no_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()

    def fake_run(cmd: list, capture_output: bool, text: bool, timeout: int) -> DummyResult:  # type: ignore[override]
        assert "--parallel" not in cmd
        return DummyResult(returncode=0, stdout="Built target app\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    exit_code, _, _ = module.build_cmake(build_dir="build", jobs=0)
    assert exit_code == 0


def test_build_cmake_uses_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()

    def fake_run(cmd: list, capture_output: bool, text: bool, timeout: int) -> DummyResult:  # type: ignore[override]
        assert cmd[:4] == ["cmake", "--build", "--preset", "release"]
        assert "build" not in cmd
        return DummyResult(returncode=0, stdout="Built target app\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    exit_code, _, _ = module.build_cmake(build_dir="build", build_preset="release")
    assert exit_code == 0


def test_resolve_build_dir_direct(run_cmake_module: ModuleType) -> None:
    preset_data = {"configurePresets": [{"name": "debug", "binaryDir": "build"}]}
    resolved = run_cmake_module._resolve_build_dir_from_preset("debug", preset_data)
    assert resolved == "build"


def test_resolve_build_dir_inherited(run_cmake_module: ModuleType) -> None:
    preset_data = {
        "configurePresets": [
            {"name": "base", "binaryDir": "build"},
            {"name": "child", "inherits": "base"},
        ]
    }
    resolved = run_cmake_module._resolve_build_dir_from_preset("child", preset_data)
    assert resolved == "build"


def test_resolve_build_dir_missing_binary_dir(run_cmake_module: ModuleType) -> None:
    preset_data = {"configurePresets": [{"name": "debug"}]}
    resolved = run_cmake_module._resolve_build_dir_from_preset("debug", preset_data)
    assert resolved is None


def test_resolve_build_dir_cycle_returns_none(run_cmake_module: ModuleType) -> None:
    preset_data = {
        "configurePresets": [
            {"name": "A", "inherits": "B"},
            {"name": "B", "inherits": "A"},
        ]
    }
    resolved = run_cmake_module._resolve_build_dir_from_preset("A", preset_data)
    assert resolved is None


def test_find_build_preset_match(run_cmake_module: ModuleType) -> None:
    preset_data = {
        "buildPresets": [
            {"name": "build-debug", "configurePreset": "debug"},
            {"name": "build-release", "configurePreset": "release"},
        ]
    }
    match = run_cmake_module._find_build_preset("debug", preset_data)
    assert match == "build-debug"


def test_find_build_preset_no_match(run_cmake_module: ModuleType) -> None:
    preset_data = {"buildPresets": [{"name": "build-release", "configurePreset": "release"}]}
    match = run_cmake_module._find_build_preset("debug", preset_data)
    assert match is None


def test_build_summary_format(run_cmake_module: ModuleType) -> None:
    metrics = {
        "success": False,
        "warnings": 2,
        "errors": 1,
        "warning_messages": ["warn1", "warn2"],
        "error_messages": ["err1"],
        "timeout": False,
        "duration": 1.2,
    }
    summary = run_cmake_module.format_build_summary(metrics, "build")
    assert "BUILD SUMMARY" in summary
    assert "Warnings: 2" in summary
    assert "Errors: 1" in summary


def test_run_cmake_with_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    source_dir = tmp_path / "src"
    build_dir = tmp_path / "build"
    source_dir.mkdir()

    calls: list[list[str]] = []

    def fake_run(cmd: list, capture_output: bool, text: bool, timeout: int) -> DummyResult:  # type: ignore[override]
        calls.append(cmd)
        if "--build" in cmd:
            return DummyResult(returncode=0, stdout="Built target app\n", stderr="")
        return DummyResult(returncode=0, stdout="-- Configuring done\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    exit_code, output = module.run_cmake(
        source_dir=str(source_dir),
        build_dir=str(build_dir),
        build=True,
        output_mode="summary",
    )
    assert exit_code == 0
    assert any("--build" in cmd for cmd in calls)
    assert "BUILD SUMMARY" in output


def test_run_cmake_build_uses_build_preset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    source_dir = tmp_path / "src"
    build_dir = tmp_path / "build"
    source_dir.mkdir()
    presets = {
        "version": 3,
        "configurePresets": [{"name": "debug", "binaryDir": "build"}],
        "buildPresets": [{"name": "build-debug", "configurePreset": "debug"}],
    }
    (source_dir / "CMakePresets.json").write_text(json.dumps(presets))

    def fake_run(cmd: list, capture_output: bool, text: bool, timeout: int) -> DummyResult:  # type: ignore[override]
        if "--build" in cmd:
            assert "--preset" in cmd
            assert "build-debug" in cmd
            return DummyResult(returncode=0, stdout="Built target app\n", stderr="")
        assert "--preset" in cmd
        return DummyResult(returncode=0, stdout="-- Configuring done\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    exit_code, output = module.run_cmake(
        source_dir=str(source_dir),
        build_dir=str(build_dir),
        preset="debug",
        build=True,
        output_mode="summary",
    )
    assert exit_code == 0
    assert "BUILD SUMMARY" in output


def test_run_cmake_build_skipped_on_config_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    source_dir = tmp_path / "src"
    build_dir = tmp_path / "build"
    source_dir.mkdir()

    def fake_run(cmd: list, capture_output: bool, text: bool, timeout: int) -> DummyResult:  # type: ignore[override]
        assert "--build" not in cmd
        return DummyResult(returncode=1, stdout="", stderr="CMake Error: fail")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    exit_code, output = module.run_cmake(
        source_dir=str(source_dir),
        build_dir=str(build_dir),
        build=True,
        output_mode="summary",
    )
    assert exit_code == 1
    assert "BUILD SUMMARY" not in output


def test_run_cmake_build_dir_fallback_when_binary_dir_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    source_dir = tmp_path / "src"
    build_dir = tmp_path / "build"
    source_dir.mkdir()
    presets = {
        "version": 3,
        "configurePresets": [
            {"name": "debug"},
        ],
    }
    (source_dir / "CMakePresets.json").write_text(json.dumps(presets))

    def fake_run(cmd: list, capture_output: bool, text: bool, timeout: int) -> DummyResult:  # type: ignore[override]
        if "--build" in cmd:
            assert str(build_dir) in cmd
            return DummyResult(returncode=0, stdout="Built target app\n", stderr="")
        assert "--preset" in cmd
        return DummyResult(returncode=0, stdout="-- Configuring done\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    exit_code, output = module.run_cmake(
        source_dir=str(source_dir),
        build_dir=str(build_dir),
        preset="debug",
        build=True,
        output_mode="summary",
    )
    assert exit_code == 0
    assert "BUILD SUMMARY" in output


def test_run_cmake_json_includes_build_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    source_dir = tmp_path / "src"
    build_dir = tmp_path / "build"
    source_dir.mkdir()

    def fake_run(cmd: list, capture_output: bool, text: bool, timeout: int) -> DummyResult:  # type: ignore[override]
        if "--build" in cmd:
            return DummyResult(returncode=0, stdout="Built target app\n", stderr="")
        return DummyResult(returncode=0, stdout="-- Configuring done\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    exit_code, output = module.run_cmake(
        source_dir=str(source_dir),
        build_dir=str(build_dir),
        build=True,
        output_mode="json",
    )
    assert exit_code == 0
    payload = json.loads(output)
    assert payload["build_metrics"]["success"] is True
    assert payload.get("build_output") is not None


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
