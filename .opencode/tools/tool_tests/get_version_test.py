import importlib.util
import json
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

TOOL_PATH = Path(__file__).resolve().parents[1] / "get_version.py"
SPEC = importlib.util.spec_from_file_location("get_version_tool", TOOL_PATH)
assert SPEC is not None
assert SPEC.loader is not None
get_version_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(get_version_tool)


def test_resolve_target_path_prefers_pyproject_then_package_json(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")
    (tmp_path / "package.json").write_text('{"version":"9.9.9"}\n', encoding="utf-8")

    assert get_version_tool.resolve_target_path(None, cwd=tmp_path) == pyproject.resolve()


def test_get_version_reads_dynamic_hatch_version_from_pyproject(tmp_path: Path):
    package_dir = tmp_path / "sample"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text('__version__ = "2.3.4"\n', encoding="utf-8")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[tool.hatch.version]\npath = "sample/__init__.py"\n\n[project]\ndynamic = ["version"]\n',
        encoding="utf-8",
    )

    assert get_version_tool.get_version(pyproject, allowed_root=tmp_path) == "2.3.4"


def test_get_version_reads_package_json_version(tmp_path: Path):
    package_json = tmp_path / "package.json"
    package_json.write_text(json.dumps({"version": "4.5.6"}), encoding="utf-8")

    assert get_version_tool.get_version(package_json, allowed_root=tmp_path) == "4.5.6"


def test_get_version_reads_poetry_version_from_pyproject(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[tool.poetry]\nversion = "6.7.8"\n', encoding="utf-8")

    assert get_version_tool.get_version(pyproject, allowed_root=tmp_path) == "6.7.8"


def test_resolve_target_path_rejects_explicit_path_outside_allowed_root(tmp_path: Path):
    outside = tmp_path.parent / "outside-package.json"
    outside.write_text('{"version":"1.2.3"}\n', encoding="utf-8")

    try:
        get_version_tool.resolve_target_path(str(Path("..") / outside.name), cwd=tmp_path)
    except ValueError as exc:
        assert "outside allowed root" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected explicit outside-root path rejection")


def test_get_version_rejects_dynamic_hatch_target_outside_allowed_root(tmp_path: Path):
    escaped = tmp_path.parent / "escaped_version.py"
    escaped.write_text('__version__ = "9.9.9"\n', encoding="utf-8")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        f'[tool.hatch.version]\npath = "../{escaped.name}"\n\n[project]\ndynamic = ["version"]\n',
        encoding="utf-8",
    )

    try:
        get_version_tool.get_version(pyproject, allowed_root=tmp_path)
    except ValueError as exc:
        assert "outside allowed root" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected dynamic Hatch path confinement failure")


def test_cli_auto_detects_current_directory_package_json(tmp_path: Path):
    package_json = tmp_path / "package.json"
    package_json.write_text(json.dumps({"version": "7.8.9"}), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(TOOL_PATH)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "7.8.9"
    assert result.stderr == ""


def test_cli_reads_explicit_pyproject_argument(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "8.9.0"\n', encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(TOOL_PATH), str(pyproject)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "8.9.0"
    assert result.stderr == ""


def test_main_returns_usage_error_for_multiple_arguments():
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()

    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        exit_code = get_version_tool.main(["pyproject.toml", "package.json"])

    assert exit_code == 2
    assert stdout_buffer.getvalue() == ""
    assert stderr_buffer.getvalue().strip() == "Usage: get_version.py [file]"


def test_main_reports_runtime_error_to_stderr(tmp_path: Path):
    missing_file = "missing.toml"

    result = subprocess.run(
        [sys.executable, str(TOOL_PATH), missing_file],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert f"File not found: {(tmp_path / missing_file).resolve()}" in result.stderr


def test_get_version_rejects_unsupported_file_type(tmp_path: Path):
    version_file = tmp_path / "VERSION"
    version_file.write_text("1.0.0\n", encoding="utf-8")

    try:
        get_version_tool.get_version(version_file, allowed_root=tmp_path)
    except ValueError as exc:
        assert "Unsupported file type" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected unsupported file type failure")


def test_resolve_target_path_raises_when_no_supported_files_exist(tmp_path: Path):
    try:
        get_version_tool.resolve_target_path(None, cwd=tmp_path)
    except FileNotFoundError as exc:
        assert "Could not find pyproject.toml or package.json" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected auto-detect failure")


def test_get_version_rejects_missing_file(tmp_path: Path):
    missing_file = tmp_path / "package.json"

    try:
        get_version_tool.get_version(missing_file, allowed_root=tmp_path)
    except FileNotFoundError as exc:
        assert f"File not found: {missing_file}" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected missing file failure")


def test_get_version_rejects_directory_path(tmp_path: Path):
    try:
        get_version_tool.get_version(tmp_path, allowed_root=tmp_path)
    except ValueError as exc:
        assert f"Path is not a file: {tmp_path}" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected directory path failure")
