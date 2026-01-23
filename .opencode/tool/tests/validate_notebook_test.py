"""Tests for validate_notebook CLI tool."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import nbformat

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "validate_notebook.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("validate_notebook_cli", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - sanity check
        raise AssertionError("Failed to load CLI module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _make_notebook(path: Path, cells: list[dict]) -> None:
    nb = nbformat.v4.new_notebook()
    nb["cells"] = cells
    path.write_text(nbformat.writes(nb), encoding="utf-8")


def _run_cli(args: list[str]) -> SimpleNamespace:
    module = _load_cli_module()
    stdout_buffer: io.StringIO = io.StringIO()
    stderr_buffer: io.StringIO = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        returncode = module.main(args)
    return SimpleNamespace(
        args=args,
        returncode=returncode,
        stdout=stdout_buffer.getvalue(),
        stderr=stderr_buffer.getvalue(),
    )


def test_main_importable() -> None:
    module = _load_cli_module()

    assert hasattr(module, "main")


def test_valid_notebook_exit_zero(tmp_path: Path) -> None:
    nb_path = tmp_path / "ok.ipynb"
    _make_notebook(nb_path, [nbformat.v4.new_code_cell("x = 1\n")])

    proc = _run_cli([str(nb_path)])

    assert proc.returncode == 0
    assert "VALIDATION: PASSED" in proc.stdout
    assert proc.stderr == ""
    assert "Notebook: " not in proc.stdout  # summary mode should not include full details


def test_invalid_notebook_exit_one(tmp_path: Path) -> None:
    nb_path = tmp_path / "bad.ipynb"
    _make_notebook(nb_path, [nbformat.v4.new_code_cell("x =")])

    proc = _run_cli([str(nb_path)])

    assert proc.returncode == 1
    assert "VALIDATION: FAILED" in proc.stdout
    assert "syntax" in proc.stdout.lower()


def test_missing_file_exit_two(tmp_path: Path) -> None:
    missing = tmp_path / "missing.ipynb"

    proc = _run_cli([str(missing)])

    assert proc.returncode == 2
    assert proc.stdout.startswith("ERROR:")


def test_empty_directory_exit_two(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    proc = _run_cli([str(empty_dir)])

    assert proc.returncode == 2
    assert "No notebooks found" in proc.stdout


def test_recursive_discovers_nested(tmp_path: Path) -> None:
    nested = tmp_path / "nested" / "deeper"
    nested.mkdir(parents=True)
    nb_path = nested / "deep.ipynb"
    _make_notebook(nb_path, [nbformat.v4.new_code_cell("x = 1")])

    proc = _run_cli([str(tmp_path), "--recursive"])

    assert proc.returncode == 0
    assert "Notebooks Checked: 1" in proc.stdout


def test_output_modes_json_and_full(tmp_path: Path) -> None:
    nb_path = tmp_path / "ok.ipynb"
    _make_notebook(nb_path, [nbformat.v4.new_code_cell("x = 1")])

    json_proc = _run_cli([str(nb_path), "--output", "json"])
    data = json.loads(json_proc.stdout)
    assert data["notebooks_checked"] == 1
    assert data["notebooks_invalid"] == 0
    assert data["skip_syntax"] is False

    full_proc = _run_cli([str(nb_path), "--output", "full"])
    assert "Notebook: " in full_proc.stdout
    assert data["results"][0]["errors"] == []
    assert data["results"][0]["warnings"] == []


def test_skip_syntax_turns_errors_into_warnings(tmp_path: Path) -> None:
    nb_path = tmp_path / "syntax.ipynb"
    _make_notebook(nb_path, [nbformat.v4.new_code_cell("x =")])

    proc = _run_cli([str(nb_path), "--skip-syntax", "--output", "json"])
    data = json.loads(proc.stdout)

    assert proc.returncode == 0
    assert data["notebooks_invalid"] == 0
    assert data["skip_syntax"] is True
    warnings = data["results"][0]["warnings"]
    assert any("syntax" in w.lower() for w in warnings)
    assert data["results"][0]["errors"] == []


def test_mixed_batch_exit_one(tmp_path: Path) -> None:
    valid_nb = tmp_path / "ok.ipynb"
    invalid_nb = tmp_path / "bad.ipynb"
    _make_notebook(valid_nb, [nbformat.v4.new_code_cell("x = 1")])
    _make_notebook(invalid_nb, [nbformat.v4.new_code_cell("x =")])

    proc = _run_cli([str(tmp_path)])

    assert proc.returncode == 1
    assert "Invalid: 1" in proc.stdout
    assert "bad.ipynb" in proc.stdout


def test_convert_creates_py_sibling(tmp_path: Path) -> None:
    nb_path = tmp_path / "nb.ipynb"
    _make_notebook(nb_path, [nbformat.v4.new_code_cell("x = 1")])

    proc = _run_cli([str(nb_path), "--convert-to-py"])

    assert proc.returncode == 0
    py_path = nb_path.with_suffix(".py")
    assert py_path.exists()
    assert "✓" in proc.stdout


def test_convert_respects_output_dir(tmp_path: Path) -> None:
    nb_path = tmp_path / "nb.ipynb"
    out_dir = tmp_path / "scripts"
    _make_notebook(nb_path, [nbformat.v4.new_code_cell("x = 1")])

    proc = _run_cli([str(nb_path), "--convert-to-py", "--output-dir", str(out_dir)])

    assert proc.returncode == 0
    py_path = out_dir / "nb.py"
    assert py_path.exists()
    assert "✓" in proc.stdout


def test_convert_requires_flag_for_output_dir(tmp_path: Path) -> None:
    nb_path = tmp_path / "nb.ipynb"
    _make_notebook(nb_path, [nbformat.v4.new_code_cell("x = 1")])

    proc = _run_cli([str(nb_path), "--output-dir", "scripts"])

    assert proc.returncode == 2
    assert "--output-dir is only allowed with --convert-to-py" in proc.stdout


def test_sync_updates_newer_notebook(tmp_path: Path) -> None:
    nb_path = tmp_path / "nb.ipynb"
    py_path = tmp_path / "nb.py"
    _make_notebook(nb_path, [nbformat.v4.new_code_cell("x = 1")])
    py_path.write_text("# script\nprint('old')\n", encoding="utf-8")

    # Make notebook newer
    nb_path.touch()

    proc = _run_cli([str(nb_path), "--sync"])

    assert proc.returncode == 0
    assert "updated" in proc.stdout
    assert py_path.read_text(encoding="utf-8").strip()


def test_sync_updates_newer_script(tmp_path: Path) -> None:
    nb_path = tmp_path / "nb.ipynb"
    py_path = tmp_path / "nb.py"
    _make_notebook(nb_path, [nbformat.v4.new_code_cell("x = 1")])
    py_path.write_text("# script\nprint('newer')\n", encoding="utf-8")

    # Make script newer
    py_path.touch()

    proc = _run_cli([str(nb_path), "--sync"])

    assert proc.returncode == 0
    assert "updated" in proc.stdout
    assert nb_path.exists()


def test_sync_creates_missing_peer(tmp_path: Path) -> None:
    nb_path = tmp_path / "nb.ipynb"
    _make_notebook(nb_path, [nbformat.v4.new_code_cell("x = 1")])

    proc = _run_cli([str(nb_path), "--sync"])

    assert proc.returncode == 0
    assert nb_path.with_suffix(".py").exists()


def test_check_sync_in_sync(tmp_path: Path) -> None:
    nb_path = tmp_path / "nb.ipynb"
    py_path = nb_path.with_suffix(".py")
    _make_notebook(nb_path, [nbformat.v4.new_code_cell("x = 1")])
    py_path.write_text("# script\n", encoding="utf-8")

    # Align mtimes so files appear synced (use nanosecond precision)
    nb_mtime_ns = nb_path.stat().st_mtime_ns
    os.utime(py_path, ns=(nb_mtime_ns, nb_mtime_ns))

    proc = _run_cli([str(nb_path), "--check-sync"])

    assert proc.returncode == 0
    assert "in sync" in proc.stdout


def test_check_sync_out_of_sync_states(tmp_path: Path) -> None:
    nb_path = tmp_path / "nb.ipynb"
    py_path = nb_path.with_suffix(".py")
    _make_notebook(nb_path, [nbformat.v4.new_code_cell("x = 1")])
    py_path.write_text("# script\n", encoding="utf-8")

    # Make script newer by explicitly setting a future mtime
    nb_mtime_ns = nb_path.stat().st_mtime_ns
    os.utime(py_path, ns=(nb_mtime_ns + 1000000, nb_mtime_ns + 1000000))

    proc = _run_cli([str(nb_path), "--check-sync"])

    assert proc.returncode == 1
    assert "script is newer" in proc.stdout

    # Missing script case
    py_path.unlink()
    proc_missing = _run_cli([str(nb_path), "--check-sync"])
    assert proc_missing.returncode == 1
    assert "missing script" in proc_missing.stdout


def test_check_sync_is_read_only(tmp_path: Path) -> None:
    nb_path = tmp_path / "nb.ipynb"
    py_path = nb_path.with_suffix(".py")
    _make_notebook(nb_path, [nbformat.v4.new_code_cell("x = 1")])
    py_path.write_text("# script\n", encoding="utf-8")

    # Align mtimes so check-sync sees them as in sync (use nanosecond precision)
    nb_mtime_ns = nb_path.stat().st_mtime_ns
    os.utime(py_path, ns=(nb_mtime_ns, nb_mtime_ns))

    nb_before = nb_path.stat().st_mtime
    py_before = py_path.stat().st_mtime

    proc = _run_cli([str(nb_path), "--check-sync"])

    nb_after = nb_path.stat().st_mtime
    py_after = py_path.stat().st_mtime

    assert proc.returncode == 0
    assert nb_before == nb_after
    assert py_before == py_after


def test_mutual_exclusion_and_validation_only_flags(tmp_path: Path) -> None:
    nb_path = tmp_path / "nb.ipynb"
    _make_notebook(nb_path, [nbformat.v4.new_code_cell("x = 1")])

    # validation-only flags with convert should fail
    proc_output_with_convert = _run_cli([str(nb_path), "--convert-to-py", "--output", "json"])
    assert proc_output_with_convert.returncode == 2
    assert "--output/--skip-syntax" in proc_output_with_convert.stdout

    # output-dir without convert should fail
    proc_output_dir_without_convert = _run_cli([str(nb_path), "--output-dir", "scripts"])
    assert proc_output_dir_without_convert.returncode == 2
    assert (
        "--output-dir is only allowed with --convert-to-py"
        in proc_output_dir_without_convert.stdout
    )

    # multiple action flags should be blocked by argparse mutual exclusion (exit 2)
    proc_conflict = _run_cli([str(nb_path), "--convert-to-py", "--sync"])
    assert proc_conflict.returncode == 2


def test_no_notebooks_found_exit_two(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    proc = _run_cli([str(empty_dir), "--convert-to-py"])

    assert proc.returncode == 2
    assert "No notebooks found" in proc.stdout
