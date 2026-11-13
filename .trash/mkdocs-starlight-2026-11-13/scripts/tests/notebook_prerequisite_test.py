"""Keep artifact-only viewer skips distinct from executable notebook failures."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / ".github/scripts/validate_notebooks.py"
)
SPEC = importlib.util.spec_from_file_location("notebook_validation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("case", ["missing", "present", "invalid", "ordinary"])
def test_only_known_viewer_with_valid_syntax_can_skip(
    tmp_path, monkeypatch, case
):
    """Existing artifacts and ordinary notebooks always reach execution."""
    monkeypatch.setattr(
        MODULE,
        "__file__",
        str(tmp_path / ".github/scripts/validate_notebooks.py"),
    )
    path = tmp_path / (
        "docs/contribute/validation/gpu/legacy-benchmarks/gpu_benchmarks.ipynb"
        if case != "ordinary"
        else "docs/examples/example.ipynb"
    )
    path.parent.mkdir(parents=True)
    code = "invalid (" if case == "invalid" else "value = 1"
    path.write_text(
        json.dumps({"cells": [{"cell_type": "code", "source": code}]})
    )
    if case == "present":
        (tmp_path / "gpu_benchmark_results.json").write_text("{}")
    calls = []

    def execute(*args):
        calls.append(args)
        return False, "invalid artifact or executable workload", False

    monkeypatch.setattr(MODULE, "execute_notebook", execute)
    result = MODULE.validate_notebook(str(path), timeout=10)
    assert bool(result.prerequisite_missing) == (case == "missing")
    assert bool(calls) == (case in {"present", "ordinary"})
    if case == "invalid":
        assert not result.syntax_valid
    if calls:
        assert result.execution_valid is False
