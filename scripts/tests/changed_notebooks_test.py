"""Regression coverage for notebook matrix selection against real Git history."""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / ".github/scripts/changed_notebooks.py"
)
SPEC = importlib.util.spec_from_file_location("changed_notebooks", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
GIT = shutil.which("git")
assert GIT is not None


def git(*args):
    """Run a Git command in the isolated fixture repository."""
    return subprocess.check_output(  # noqa: S603 -- fixture-owned Git argv
        [GIT, *args],
        text=True,
    ).strip()


@pytest.fixture
def history(tmp_path, monkeypatch):
    """Create changed, removed and space-containing notebook paths."""
    monkeypatch.chdir(tmp_path)
    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    Path("docs/examples").mkdir(parents=True)
    for name in ("changed", "deleted", "unchanged"):
        Path(f"docs/examples/{name}.ipynb").write_text("{}")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    Path("docs/examples/changed.ipynb").write_text('{"cells": []}')
    Path("docs/examples/deleted.ipynb").unlink()
    Path("docs/examples/space name.ipynb").write_text("{}")
    Path("docs/examples/code.py").write_text("pass")
    git("add", ".")
    git("commit", "-qm", "changes")
    return base, git("rev-parse", "HEAD")


@pytest.mark.parametrize("kind", ["push", "pull_request"])
def test_changed_paths_preserve_spaces_and_exclude_deleted(history, kind):
    """Select each executable changed notebook exactly once."""
    base, head = history
    event = {
        "before": base,
        "after": head,
        "pull_request": {"base": {"sha": base}, "head": {"sha": head}},
    }
    assert MODULE.changed_notebooks(event, kind) == [
        "docs/examples/changed.ipynb",
        "docs/examples/space name.ipynb",
    ]


def test_first_push_and_no_changes(history):
    """First pushes select all tracked notebooks; empty diffs select none."""
    _, head = history
    assert (
        len(
            MODULE.changed_notebooks(
                {"before": "0" * 40, "after": head}, "push"
            )
        )
        == 3
    )
    assert (
        MODULE.changed_notebooks({"before": head, "after": head}, "push") == []
    )
