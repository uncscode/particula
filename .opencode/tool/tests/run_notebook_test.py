"""Tests for run_notebook.py tool.

Validates pre-execution validation gating, skip-validation flag behavior,
graceful degradation when validation module is missing, expected-output
validation coexistence, and structured output reporting across modes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path
from types import ModuleType
from typing import Callable

import nbformat
import pytest

from adw.utils.notebook import NotebookExecutionResult
from adw.utils.notebook_validation import (
    NotebookValidationError,
    NotebookValidationResult,
    validate_notebook_json,
)
import adw.utils.notebook_validation as validation_module

RUN_NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "run_notebook.py"


def write_notebook(path: Path, cells: list[str]) -> None:
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source=cell) for cell in cells])
    nbformat.write(nb, path)


def load_module() -> ModuleType:
    module_name = f"run_notebook_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, RUN_NOTEBOOK_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute_and_expect(
    module: ModuleType,
    notebooks: list[Path],
    *,
    expect_output: list[str] | None = None,
    write_executed: Path | None = None,
    no_overwrite: bool = False,
    no_backup: bool = False,
) -> tuple[int, str]:
    return module.run_notebooks(
        notebooks=notebooks,
        timeout=1,
        expect_output=expect_output or [],
        output_mode="summary",
        write_executed=write_executed,
        skip_validation=False,
        no_overwrite=no_overwrite,
        no_backup=no_backup,
    )


def make_validation_result(
    notebook_path: Path | str,
    *,
    valid: bool,
    cell_index: int | None = None,
    message: str = "Validation failed",
) -> NotebookValidationResult:
    errors: list[NotebookValidationError] = []
    if not valid:
        errors.append(
            NotebookValidationError(error_type="schema", message=message, cell_index=cell_index)
        )

    return NotebookValidationResult(valid=valid, notebook_path=str(notebook_path), errors=errors)


def make_successful_execute(
    call_log: list[str],
    *,
    output_contents: str | None = None,
) -> Callable[[Path, Path | None, int], NotebookExecutionResult]:
    def execute(
        notebook_path: Path, output_path: Path | None = None, timeout: int = 0
    ) -> NotebookExecutionResult:
        call_log.append(str(notebook_path))

        nb = nbformat.v4.new_notebook()
        cell = nbformat.v4.new_code_cell(source=output_contents or "print('ok')")
        cell.outputs = [
            nbformat.v4.new_output(
                output_type="stream",
                name="stdout",
                text=output_contents or "ok",
            )
        ]
        nb.cells = [cell]
        if output_path is not None:
            nbformat.write(nb, output_path)

        return NotebookExecutionResult(
            success=True,
            notebook_path=str(notebook_path),
            output_path=str(output_path) if output_path is not None else None,
            execution_time=0.1,
            error_message=None,
            failed_cell_index=None,
        )

    return execute


def test_validation_runs_before_execution_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    notebook = tmp_path / "broken.ipynb"
    notebook.write_text("not a notebook")

    def fake_validate(path: Path | str) -> NotebookValidationResult:
        return make_validation_result(path, valid=False, message="Malformed JSON", cell_index=1)

    monkeypatch.setattr(validation_module, "validate_notebook_json", fake_validate)

    def fake_load_executor() -> tuple[type, Callable[..., NotebookExecutionResult]]:
        def execute(*_: object, **__: object) -> NotebookExecutionResult:
            raise RuntimeError("Execution should be skipped on validation failure")

        return NotebookExecutionResult, execute

    monkeypatch.setattr(module, "_load_executor", fake_load_executor)

    exit_code, output = module.run_notebooks(
        notebooks=[notebook],
        timeout=1,
        expect_output=[],
        output_mode="full",
        write_executed=None,
    )

    assert exit_code == 1
    assert "Validation Failures" in output
    assert "Malformed JSON" in output


def test_corrupted_notebook_returns_validation_error_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    notebook = tmp_path / "corrupted.ipynb"
    notebook.write_text("bad")

    def fake_validate(path: Path | str) -> NotebookValidationResult:
        return make_validation_result(path, valid=False, message="Corrupted structure")

    monkeypatch.setattr(validation_module, "validate_notebook_json", fake_validate)

    def fake_load_executor() -> tuple[type, Callable[..., NotebookExecutionResult]]:
        def execute(*_: object, **__: object) -> NotebookExecutionResult:
            raise RuntimeError("Execution should not run when validation fails")

        return NotebookExecutionResult, execute

    monkeypatch.setattr(module, "_load_executor", fake_load_executor)

    exit_code, output = module.run_notebooks(
        notebooks=[notebook],
        timeout=1,
        expect_output=[],
        output_mode="json",
        write_executed=None,
    )

    payload = json.loads(output)

    assert exit_code == 1
    assert payload["success"] is False
    assert str(notebook) in payload["validation_failures"]
    failure = payload["validation_failures"][str(notebook)]
    assert "Corrupted structure" in failure["message"]


def test_skip_validation_flag_bypasses_precheck(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    notebook = tmp_path / "intentional.ipynb"
    notebook.write_text("broken")

    def fake_validate(*_: object, **__: object) -> NotebookValidationResult:
        raise AssertionError("Validation should not run when skip_validation is enabled")

    monkeypatch.setattr(validation_module, "validate_notebook_json", fake_validate)

    executed: list[str] = []

    def fake_load_executor() -> tuple[type, Callable[..., NotebookExecutionResult]]:
        return NotebookExecutionResult, make_successful_execute(executed)

    monkeypatch.setattr(module, "_load_executor", fake_load_executor)

    exit_code, output = module.run_notebooks(
        notebooks=[notebook],
        timeout=1,
        expect_output=[],
        output_mode="summary",
        write_executed=None,
        skip_validation=True,
    )

    assert exit_code == 0
    assert executed == [str(notebook)]
    assert "Validation Failures" not in output


def test_graceful_when_validation_module_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing_module = ModuleType("adw.utils.notebook_validation")
    original = sys.modules.get("adw.utils.notebook_validation")
    sys.modules["adw.utils.notebook_validation"] = missing_module

    module = load_module()
    executed: list[str] = []

    def fake_load_executor() -> tuple[type, Callable[..., NotebookExecutionResult]]:
        return NotebookExecutionResult, make_successful_execute(executed)

    monkeypatch.setattr(module, "_load_executor", fake_load_executor)

    try:
        exit_code, output = module.run_notebooks(
            notebooks=[tmp_path / "ok.ipynb"],
            timeout=1,
            expect_output=[],
            output_mode="summary",
            write_executed=None,
        )
    finally:
        if original is not None:
            sys.modules["adw.utils.notebook_validation"] = original
        else:
            sys.modules.pop("adw.utils.notebook_validation", None)

    assert exit_code == 0
    assert "Validation module unavailable" in output
    assert "executed without pre-validation" in output


def test_expected_output_validation_still_applies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    notebook = tmp_path / "valid.ipynb"
    notebook.write_text("ok")

    def fake_validate(path: Path | str) -> NotebookValidationResult:
        return make_validation_result(path, valid=True)

    monkeypatch.setattr(validation_module, "validate_notebook_json", fake_validate)

    executed: list[str] = []

    def fake_load_executor() -> tuple[type, Callable[..., NotebookExecutionResult]]:
        return NotebookExecutionResult, make_successful_execute(executed)

    monkeypatch.setattr(module, "_load_executor", fake_load_executor)

    exit_code, output = module.run_notebooks(
        notebooks=[notebook],
        timeout=1,
        expect_output=["missing-string"],
        output_mode="summary",
        write_executed=None,
    )

    assert exit_code == 1
    assert "Validation Errors" in output


def test_validation_failure_in_full_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    notebook = tmp_path / "failure.ipynb"
    notebook.write_text("bad")

    def fake_validate(path: Path | str) -> NotebookValidationResult:
        return make_validation_result(path, valid=False, cell_index=3, message="Cell issue")

    monkeypatch.setattr(validation_module, "validate_notebook_json", fake_validate)

    def fake_load_executor() -> tuple[type, Callable[..., NotebookExecutionResult]]:
        def execute(*_: object, **__: object) -> NotebookExecutionResult:
            raise RuntimeError("Should not run when validation fails")

        return NotebookExecutionResult, execute

    monkeypatch.setattr(module, "_load_executor", fake_load_executor)

    exit_code, output = module.run_notebooks(
        notebooks=[notebook],
        timeout=1,
        expect_output=[],
        output_mode="full",
        write_executed=None,
    )

    assert exit_code == 1
    assert "Validation Failures" in output
    assert "(cell 3)" in output
    assert "Cell issue" in output


def test_valid_notebook_executes_and_reports_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    notebook = tmp_path / "good.ipynb"
    notebook.write_text("ok")

    def fake_validate(path: Path | str) -> NotebookValidationResult:
        return make_validation_result(path, valid=True)

    monkeypatch.setattr(validation_module, "validate_notebook_json", fake_validate)

    executed: list[str] = []

    def fake_load_executor() -> tuple[type, Callable[..., NotebookExecutionResult]]:
        return NotebookExecutionResult, make_successful_execute(executed)

    monkeypatch.setattr(module, "_load_executor", fake_load_executor)

    exit_code, output = module.run_notebooks(
        notebooks=[notebook],
        timeout=1,
        expect_output=[],
        output_mode="json",
        write_executed=None,
    )

    payload = json.loads(output)

    assert exit_code == 0
    assert payload["success"] is True
    assert payload["validation_failures"] == {}
    assert payload["validation_errors"] == {}


def test_default_overwrite_creates_backup_and_updates_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    notebook = tmp_path / "nb.ipynb"
    write_notebook(notebook, ["x = 1", "x = 2"])

    def fake_validate(path: Path | str) -> NotebookValidationResult:
        return make_validation_result(path, valid=True)

    monkeypatch.setattr(validation_module, "validate_notebook_json", fake_validate)

    executed: list[str] = []

    def fake_load_executor() -> tuple[type, Callable[..., NotebookExecutionResult]]:
        return NotebookExecutionResult, make_successful_execute(
            executed, output_contents="print('ran')"
        )

    monkeypatch.setattr(module, "_load_executor", fake_load_executor)

    exit_code, _ = execute_and_expect(module, [notebook])

    backup = notebook.with_suffix(".ipynb.bak")
    assert exit_code == 0
    assert backup.exists()
    assert executed == [str(notebook)]
    backup_nb = nbformat.read(backup, as_version=4)
    assert backup_nb.cells[0].source == "x = 1"
    overwritten = nbformat.read(notebook, as_version=4)
    assert "print('ran')" in overwritten.cells[0].source


def test_no_overwrite_skips_backup_and_preserves_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    notebook = tmp_path / "nb.ipynb"
    write_notebook(notebook, ["a = 1"])
    write_executed = tmp_path / "executed"

    def fake_validate(path: Path | str) -> NotebookValidationResult:
        return make_validation_result(path, valid=True)

    monkeypatch.setattr(validation_module, "validate_notebook_json", fake_validate)

    executed: list[str] = []

    def fake_load_executor() -> tuple[type, Callable[..., NotebookExecutionResult]]:
        return NotebookExecutionResult, make_successful_execute(
            executed, output_contents="print('ran')"
        )

    monkeypatch.setattr(module, "_load_executor", fake_load_executor)

    exit_code, _ = execute_and_expect(
        module,
        [notebook],
        write_executed=write_executed,
        no_overwrite=True,
    )

    assert exit_code == 0
    assert not notebook.with_suffix(".ipynb.bak").exists()
    original = nbformat.read(notebook, as_version=4)
    assert original.cells[0].source == "a = 1"
    executed_nb = nbformat.read(write_executed / notebook.name, as_version=4)
    assert "print('ran')" in executed_nb.cells[0].source


def test_no_backup_flag_skips_backup_but_overwrites_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    notebook = tmp_path / "nb.ipynb"
    write_notebook(notebook, ["b = 1"])

    def fake_validate(path: Path | str) -> NotebookValidationResult:
        return make_validation_result(path, valid=True)

    monkeypatch.setattr(validation_module, "validate_notebook_json", fake_validate)

    executed: list[str] = []

    def fake_load_executor() -> tuple[type, Callable[..., NotebookExecutionResult]]:
        return NotebookExecutionResult, make_successful_execute(
            executed, output_contents="print('ran')"
        )

    monkeypatch.setattr(module, "_load_executor", fake_load_executor)

    exit_code, _ = execute_and_expect(module, [notebook], no_backup=True)

    assert exit_code == 0
    assert not notebook.with_suffix(".ipynb.bak").exists()
    overwritten = nbformat.read(notebook, as_version=4)
    assert "print('ran')" in overwritten.cells[0].source


def test_write_executed_and_overwrite_reuse_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    notebook = tmp_path / "nb.ipynb"
    write_notebook(notebook, ["c = 1"])
    write_executed = tmp_path / "executed"

    def fake_validate(path: Path | str) -> NotebookValidationResult:
        return make_validation_result(path, valid=True)

    monkeypatch.setattr(validation_module, "validate_notebook_json", fake_validate)

    executed: list[str] = []

    def fake_load_executor() -> tuple[type, Callable[..., NotebookExecutionResult]]:
        return NotebookExecutionResult, make_successful_execute(
            executed, output_contents="print('ran')"
        )

    monkeypatch.setattr(module, "_load_executor", fake_load_executor)

    exit_code, _ = execute_and_expect(
        module,
        [notebook],
        write_executed=write_executed,
        no_overwrite=False,
    )

    assert exit_code == 0
    assert executed == [str(notebook)]
    backup = notebook.with_suffix(".ipynb.bak")
    assert backup.exists()
    executed_copy = nbformat.read(write_executed / notebook.name, as_version=4)
    overwritten = nbformat.read(notebook, as_version=4)
    assert "print('ran')" in executed_copy.cells[0].source
    assert "print('ran')" in overwritten.cells[0].source


def test_backup_failure_warns_and_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    module = load_module()
    notebook = tmp_path / "nb.ipynb"
    write_notebook(notebook, ["d = 1"])

    def fake_validate(path: Path | str) -> NotebookValidationResult:
        return make_validation_result(path, valid=True)

    monkeypatch.setattr(validation_module, "validate_notebook_json", fake_validate)

    executed: list[str] = []

    # pragma: no cover - behavior asserted via log
    def fake_copy2(src: Path, dest: Path) -> None:
        raise OSError("cannot copy")

    def fake_load_executor() -> tuple[type, Callable[..., NotebookExecutionResult]]:
        return NotebookExecutionResult, make_successful_execute(
            executed, output_contents="print('ran')"
        )

    monkeypatch.setattr(module.shutil, "copy2", fake_copy2)
    monkeypatch.setattr(module, "_load_executor", fake_load_executor)

    exit_code, output = execute_and_expect(module, [notebook])

    assert exit_code == 0
    assert not notebook.with_suffix(".ipynb.bak").exists()
    assert "Failed to create notebook backup" in caplog.text
    overwritten = nbformat.read(notebook, as_version=4)
    assert "print('ran')" in overwritten.cells[0].source


def test_expect_output_reads_executed_file_when_overwriting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    notebook = tmp_path / "nb.ipynb"
    write_notebook(notebook, ["print('hello')"])

    def fake_validate(path: Path | str) -> NotebookValidationResult:
        return make_validation_result(path, valid=True)

    monkeypatch.setattr(validation_module, "validate_notebook_json", fake_validate)

    executed: list[str] = []

    def fake_load_executor() -> tuple[type, Callable[..., NotebookExecutionResult]]:
        return NotebookExecutionResult, make_successful_execute(
            executed, output_contents="print('hello')"
        )

    monkeypatch.setattr(module, "_load_executor", fake_load_executor)

    exit_code, _ = execute_and_expect(module, [notebook], expect_output=["hello"])

    assert exit_code == 0
    assert executed == [str(notebook)]


def test_expect_output_uses_validation_temp_when_no_overwrite_and_no_write_executed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    notebook = tmp_path / "nb.ipynb"
    write_notebook(notebook, ["print('temp')"])

    def fake_validate(path: Path | str) -> NotebookValidationResult:
        return make_validation_result(path, valid=True)

    monkeypatch.setattr(validation_module, "validate_notebook_json", fake_validate)

    executed: list[str] = []

    def fake_load_executor() -> tuple[type, Callable[..., NotebookExecutionResult]]:
        return NotebookExecutionResult, make_successful_execute(
            executed, output_contents="print('temp')"
        )

    monkeypatch.setattr(module, "_load_executor", fake_load_executor)

    exit_code, _ = execute_and_expect(
        module,
        [notebook],
        expect_output=["temp"],
        no_overwrite=True,
    )

    assert exit_code == 0
    assert executed == [str(notebook)]
    assert not (tmp_path / "nb.ipynb.bak").exists()
