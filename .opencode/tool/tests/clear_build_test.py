"""Unit tests for clear_build.py.

Covers path validation, size calculation, dry-run and force deletion
behavior, CLI invocation, and failure handling without performing
unsafe filesystem operations.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Tuple

import pytest

CLEAR_BUILD_PATH = Path(__file__).resolve().parent.parent / "clear_build.py"


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "clear_build", CLEAR_BUILD_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


@pytest.fixture(scope="module")
def clear_build_module() -> ModuleType:
    return load_module()


def test_find_project_root_prefers_git_marker(
    clear_build_module: ModuleType, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    nested = project / "src" / "app"
    nested.mkdir(parents=True)
    (project / ".git").mkdir()

    assert clear_build_module.find_project_root(nested) == project.resolve()


def test_find_project_root_falls_back_to_start(
    clear_build_module: ModuleType, tmp_path: Path
) -> None:
    start = tmp_path / "lonely"
    start.mkdir(parents=True)

    assert clear_build_module.find_project_root(start) == start.resolve()


def test_validate_path_accepts_inside_root(
    clear_build_module: ModuleType, tmp_path: Path
) -> None:
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    resolved = clear_build_module.validate_path(build_dir, tmp_path)
    assert resolved == build_dir.resolve()


def test_validate_path_rejects_outside_root(
    clear_build_module: ModuleType, tmp_path: Path
) -> None:
    outside = tmp_path.parent / "outside"
    with pytest.raises(ValueError):
        clear_build_module.validate_path(outside, tmp_path)


def test_validate_path_rejects_project_root(
    clear_build_module: ModuleType, tmp_path: Path
) -> None:
    with pytest.raises(ValueError):
        clear_build_module.validate_path(tmp_path, tmp_path)


def test_validate_path_rejects_symlink_escape(
    clear_build_module: ModuleType, tmp_path: Path
) -> None:
    outside = tmp_path.parent / "real-outside"
    outside.mkdir()
    escape_link = tmp_path / "build-link"
    escape_link.symlink_to(outside)
    with pytest.raises(ValueError):
        clear_build_module.validate_path(escape_link, tmp_path)


def test_get_directory_size_counts_files(
    clear_build_module: ModuleType, tmp_path: Path
) -> None:
    build_dir = tmp_path / "build"
    nested = build_dir / "nested"
    nested.mkdir(parents=True)
    (build_dir / "file1.o").write_bytes(b"x" * 100)
    (nested / "file2.o").write_bytes(b"y" * 200)

    total, count = clear_build_module.get_directory_size(build_dir)
    assert total == 300
    assert count == 2


def test_get_directory_size_ignores_missing(
    clear_build_module: ModuleType, tmp_path: Path
) -> None:
    total, count = clear_build_module.get_directory_size(tmp_path / "missing")
    assert (total, count) == (0, 0)


def test_format_size_units(clear_build_module: ModuleType) -> None:
    assert clear_build_module.format_size(512) == "512 B"
    assert "KB" in clear_build_module.format_size(2 * 1024)
    assert "MB" in clear_build_module.format_size(3 * 1024 * 1024)
    assert "GB" in clear_build_module.format_size(4 * 1024 * 1024 * 1024)


def test_build_parser_defaults(clear_build_module: ModuleType) -> None:
    parser = clear_build_module.build_parser()
    args = parser.parse_args([])

    assert args.build_dir == "build"
    assert args.dry_run is False
    assert args.force is False
    assert args.project_root is None


@pytest.fixture()
def populated_build(tmp_path: Path) -> Tuple[Path, Path]:
    root = tmp_path / "root"
    build_dir = root / "build"
    build_dir.mkdir(parents=True)
    (build_dir / "file.bin").write_bytes(b"data" * 10)
    return root, build_dir


def test_clear_build_dry_run_preserves_directory(
    clear_build_module: ModuleType, populated_build: Tuple[Path, Path]
) -> None:
    project_root, build_dir = populated_build
    code, output = clear_build_module.clear_build(
        build_dir,
        dry_run=True,
        force=False,
        project_root=project_root,
    )
    assert code == 0
    assert "DRY RUN" in output
    assert build_dir.exists()
    assert "VALIDATION: PASSED" in output


def test_clear_build_requires_force_for_delete(
    clear_build_module: ModuleType, populated_build: Tuple[Path, Path]
) -> None:
    project_root, build_dir = populated_build
    code, output = clear_build_module.clear_build(
        build_dir,
        dry_run=False,
        force=False,
        project_root=project_root,
    )
    assert code == 1
    assert build_dir.exists()
    assert "force" in output.lower()
    assert "VALIDATION: FAILED" in output


def test_clear_build_deletes_with_force(
    clear_build_module: ModuleType, populated_build: Tuple[Path, Path]
) -> None:
    project_root, build_dir = populated_build
    code, output = clear_build_module.clear_build(
        build_dir,
        dry_run=False,
        force=True,
        project_root=project_root,
    )
    assert code == 0
    assert "Deleted" in output
    assert not build_dir.exists()


def test_clear_build_handles_missing_directory(
    clear_build_module: ModuleType, tmp_path: Path
) -> None:
    project_root = tmp_path / "root"
    project_root.mkdir()
    missing_dir = project_root / "nope"
    code, output = clear_build_module.clear_build(
        missing_dir,
        project_root=project_root,
    )
    assert code == 0
    assert "does not exist" in output.lower()


def test_clear_build_validation_failure(
    clear_build_module: ModuleType, tmp_path: Path
) -> None:
    project_root = tmp_path / "root"
    project_root.mkdir()
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    code, output = clear_build_module.clear_build(
        outside,
        project_root=project_root,
    )
    assert code == 1
    assert "VALIDATION: FAILED" in output


def test_clear_build_handles_permission_error(
    clear_build_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    populated_build: Tuple[Path, Path],
) -> None:
    project_root, build_dir = populated_build

    def fake_rmtree(_path: Path, ignore_errors: bool = False) -> None:  # type: ignore[unused-argument]
        raise PermissionError("denied")

    monkeypatch.setattr(clear_build_module.shutil, "rmtree", fake_rmtree)
    code, output = clear_build_module.clear_build(
        build_dir,
        force=True,
        project_root=project_root,
    )
    assert code == 1
    assert "Permission denied" in output


def test_cli_dry_run(
    clear_build_module: ModuleType,
    populated_build: Tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    project_root, build_dir = populated_build
    exit_code = clear_build_module.main(
        [
            "--build-dir",
            str(build_dir),
            "--dry-run",
            "--project-root",
            str(project_root),
        ]
    )
    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "DRY RUN" in captured
    assert "VALIDATION: PASSED" in captured


def test_cli_force_delete(
    clear_build_module: ModuleType,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "root"
    build_dir = project_root / "build"
    build_dir.mkdir(parents=True)
    (build_dir / "file.bin").write_bytes(b"data")

    exit_code = clear_build_module.main(
        [
            "--build-dir",
            str(build_dir),
            "--force",
            "--project-root",
            str(project_root),
        ]
    )
    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "Deleted" in captured
    assert not build_dir.exists()


def test_cli_validation_failure(
    clear_build_module: ModuleType,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "root"
    project_root.mkdir()
    outside = tmp_path.parent / "outside-cli"
    outside.mkdir(exist_ok=True)

    exit_code = clear_build_module.main(
        ["--build-dir", str(outside), "--project-root", str(project_root)]
    )
    captured = capsys.readouterr().out
    assert exit_code == 1
    assert "VALIDATION: FAILED" in captured
