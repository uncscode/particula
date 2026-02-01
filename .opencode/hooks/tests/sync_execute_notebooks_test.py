"""Tests for sync-execute-notebooks pre-commit hook."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List, Tuple

HOOK_PATH = Path(__file__).resolve().parent.parent / "sync-execute-notebooks.sh"


def make_stub_env(tmp_path: Path, mode: str = "ok") -> Tuple[dict, Path]:
    """Create stubbed python3/git and return env and log path."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    log_path = tmp_path / "calls.log"
    log_path.touch()

    python_stub = bin_dir / "python3"
    python_stub.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'echo "python3 $@" >> "$STUB_LOG"',
                'if [[ "${STUB_MODE:-ok}" == "fail_run" && "$1" == ".opencode/tool/run_notebook.py" ]]; then',
                "  exit 2",
                "fi",
                "exit 0",
                "",
            ]
        )
    )
    python_stub.chmod(0o755)

    git_stub = bin_dir / "git"
    git_stub.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'echo "git $@" >> "$STUB_LOG"',
                "exit 0",
                "",
            ]
        )
    )
    git_stub.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "STUB_LOG": str(log_path),
            "STUB_MODE": mode,
        }
    )
    return env, log_path


def setup_repo(tmp_path: Path) -> Tuple[Path, Path]:
    """Create isolated repo directory with hook script copied in place."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    hook_dst = repo_dir / ".opencode/hooks/sync-execute-notebooks.sh"
    hook_dst.parent.mkdir(parents=True)
    hook_dst.write_text(HOOK_PATH.read_text())
    hook_dst.chmod(0o755)

    tool_dir = repo_dir / ".opencode/tool"
    tool_dir.mkdir(parents=True)
    (tool_dir / "validate_notebook.py").write_text("")
    (tool_dir / "run_notebook.py").write_text("")

    return repo_dir, hook_dst


def run_hook(
    hook_path: Path, cwd: Path, files: List[str], env: dict
) -> subprocess.CompletedProcess:
    """Run the hook with provided files and environment."""
    return subprocess.run(
        [str(hook_path), *files],
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_happy_path_sync_execute_and_stage(tmp_path: Path) -> None:
    env, log_path = make_stub_env(tmp_path)
    repo_dir, hook_path = setup_repo(tmp_path)

    py_file = repo_dir / "docs/Examples/Activity/example.py"
    ipynb_file = repo_dir / "docs/Examples/Activity/example.ipynb"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("print('hello')\n")
    ipynb_file.write_text("{}\n")

    result = run_hook(
        hook_path,
        cwd=repo_dir,
        files=[str(py_file.relative_to(repo_dir))],
        env=env,
    )

    assert result.returncode == 0, result.stderr
    log_lines = log_path.read_text().splitlines()
    assert (
        "python3 .opencode/tool/validate_notebook.py docs/Examples/Activity/example.ipynb --sync"
        in log_lines
    )
    assert (
        "python3 .opencode/tool/run_notebook.py docs/Examples/Activity/example.ipynb"
        in log_lines
    )
    assert (
        "git add docs/Examples/Activity/example.py docs/Examples/Activity/example.ipynb"
        in log_lines
    )


def test_missing_notebook_fails_early(tmp_path: Path) -> None:
    env, log_path = make_stub_env(tmp_path)
    repo_dir, hook_path = setup_repo(tmp_path)

    py_file = repo_dir / "docs/Examples/Activity/example.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("print('hello')\n")

    result = run_hook(
        hook_path,
        cwd=repo_dir,
        files=[str(py_file.relative_to(repo_dir))],
        env=env,
    )

    assert result.returncode != 0
    assert "Missing paired notebook" in result.stderr
    assert log_path.read_text().strip() == ""


def test_execution_failure_propagates(tmp_path: Path) -> None:
    env, log_path = make_stub_env(tmp_path, mode="fail_run")
    repo_dir, hook_path = setup_repo(tmp_path)

    py_file = repo_dir / "docs/Examples/Activity/example.py"
    ipynb_file = repo_dir / "docs/Examples/Activity/example.ipynb"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("print('hello')\n")
    ipynb_file.write_text("{}\n")

    result = run_hook(
        hook_path,
        cwd=repo_dir,
        files=[str(py_file.relative_to(repo_dir))],
        env=env,
    )

    assert result.returncode != 0
    log_lines = log_path.read_text().splitlines()
    assert (
        "python3 .opencode/tool/validate_notebook.py docs/Examples/Activity/example.ipynb --sync"
        in log_lines
    )
    assert (
        "python3 .opencode/tool/run_notebook.py docs/Examples/Activity/example.ipynb"
        in log_lines
    )
    assert not any(line.startswith("git add") for line in log_lines)


def test_simulations_path_is_skipped(tmp_path: Path) -> None:
    env, log_path = make_stub_env(tmp_path)
    repo_dir, hook_path = setup_repo(tmp_path)

    sim_file = "docs/Examples/Simulations/Notebooks/cloud.py"

    result = run_hook(hook_path, cwd=repo_dir, files=[sim_file], env=env)

    assert result.returncode == 0, result.stderr
    assert "Skipping Simulations notebook" in result.stderr
    assert log_path.read_text().strip() == ""


def test_duplicate_files_processed_once(tmp_path: Path) -> None:
    env, log_path = make_stub_env(tmp_path)
    repo_dir, hook_path = setup_repo(tmp_path)

    py_file = repo_dir / "docs/Examples/Activity/example.py"
    ipynb_file = repo_dir / "docs/Examples/Activity/example.ipynb"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("print('hello')\n")
    ipynb_file.write_text("{}\n")

    file_arg = str(py_file.relative_to(repo_dir))
    result = run_hook(
        hook_path,
        cwd=repo_dir,
        files=[file_arg, file_arg],
        env=env,
    )

    assert result.returncode == 0, result.stderr
    log_lines = log_path.read_text().splitlines()
    assert (
        log_lines.count(
            "python3 .opencode/tool/validate_notebook.py docs/Examples/Activity/example.ipynb --sync"
        )
        == 1
    )
    assert (
        log_lines.count(
            "python3 .opencode/tool/run_notebook.py docs/Examples/Activity/example.ipynb"
        )
        == 1
    )
    assert (
        log_lines.count(
            "git add docs/Examples/Activity/example.py docs/Examples/Activity/example.ipynb"
        )
        == 1
    )
