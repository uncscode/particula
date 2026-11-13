import importlib.util
import json
import shutil
from pathlib import Path

import pytest

TOOL_PATH = Path(__file__).resolve().parents[1] / "run_cmake.py"
SPEC = importlib.util.spec_from_file_location("run_cmake_tool", TOOL_PATH)
assert SPEC is not None
assert SPEC.loader is not None
run_cmake_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_cmake_tool)


def _make_repo_local_dir(name: str) -> Path:
    base = run_cmake_tool.REPO_ROOT / "adforge_local" / "opencode" / "tmp" / "tool-tests" / name
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    return base


def test_resolve_preset_build_dir_confines_relative_escape():
    source_dir = _make_repo_local_dir("run-cmake-escape")
    preset_data = {
        "configurePresets": [{"name": "debug", "binaryDir": "../../../../../../outside-build"}],
        "buildPresets": [],
    }

    with pytest.raises(ValueError, match="outside repository root"):
        run_cmake_tool._resolve_preset_build_dir("debug", preset_data, str(source_dir))


def test_resolve_preset_build_dir_accepts_in_repo_relative_path():
    source_dir = _make_repo_local_dir("run-cmake-in-repo")
    preset_data = {
        "configurePresets": [{"name": "debug", "binaryDir": "build/debug"}],
        "buildPresets": [],
    }

    resolved = run_cmake_tool._resolve_preset_build_dir("debug", preset_data, str(source_dir))

    assert resolved == str((source_dir / "build" / "debug").resolve())


def test_run_cmake_json_reports_preset_build_dir_escape():
    source_dir = _make_repo_local_dir("run-cmake-json-escape")
    (source_dir / "CMakePresets.json").write_text(
        json.dumps(
            {
                "version": 3,
                "configurePresets": [
                    {
                        "name": "debug",
                        "binaryDir": "../../../../../../outside-build",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code, output = run_cmake_tool.run_cmake(
        source_dir=str(source_dir),
        build_dir="build",
        preset="debug",
        output_mode="json",
    )

    assert exit_code == 1
    assert "outside repository root" in output
