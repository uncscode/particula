"""CLI-boundary regressions for advanced pytest argv transport."""

from __future__ import annotations

import errno
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "run_pytest.py"
IDENTITY = {"contract": "e37-m2-validation-git", "version": 1}


def _load_runner():
    """Load the runner module from its repository-relative file path.

    Returns:
        The imported ``run_pytest`` module used by these regression tests.
    """

    spec = importlib.util.spec_from_file_location("run_pytest_advanced_runner", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_payload_with_identity(output: str) -> dict[str, Any]:
    """Parse a runner JSON payload while asserting the canonical identity.

    Args:
        output: Serialized runner JSON output.

    Returns:
        Decoded payload mapping with the expected evidence identity.
    """

    payload = json.loads(output)
    assert payload["evidence_identity"] == IDENTITY
    return payload


def test_runner_cli_rejects_addopts_control_in_pytest_argv_json_before_spawn(monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.setattr(
        runner, "run_pytest", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError())
    )
    assert runner.main(["--pytest-argv-json", json.dumps(["-o", "addopts=", "-q"])]) == 1


def test_runner_cli_rejects_pytest_argv_json_with_legacy_passthrough_before_spawn(
    monkeypatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(
        runner, "run_pytest", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError())
    )

    assert runner.main(["--pytest-argv-json", "[]", "tests/"]) == 1


def test_runner_cli_rejects_invalid_json_and_empty_override_ini_before_spawn(monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.setattr(
        runner,
        "run_pytest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not spawn")),
    )

    assert runner.main(["--pytest-argv-json", "null"]) == 1
    assert runner.main(["--override-ini-json", '[""]']) == 1


@pytest.mark.parametrize(
    "argv",
    [
        ["--pytest-argv-json", "null"],
        ["--pytest-argv-json", "[]", "tests/"],
        ["--override-ini-json", '[""]'],
    ],
)
def test_runner_cli_json_argument_failures_use_canonical_prelaunch_envelope(
    monkeypatch, capsys, argv: list[str]
) -> None:
    """Runner-owned JSON transport validation failures remain typed evidence."""
    runner = _load_runner()
    monkeypatch.setattr(
        runner, "run_pytest", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError())
    )

    assert runner.main(["--output", "json", *argv]) == 1

    payload = _json_payload_with_identity(capsys.readouterr().out)
    assert payload["success"] is False
    assert payload["outcome"]["phase"] == "argument_validation"


def test_runner_cli_text_argument_failure_remains_plaintext(monkeypatch, capsys) -> None:
    """Non-JSON transport failures preserve the established text presentation."""
    runner = _load_runner()
    monkeypatch.setattr(
        runner, "run_pytest", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError())
    )

    assert runner.main(["--pytest-argv-json", "null"]) == 1

    assert capsys.readouterr().out.startswith("ERROR: ")


def test_runner_cli_forwards_named_target_and_filter_without_pytest_suffix(
    monkeypatch,
) -> None:
    runner = _load_runner()
    captured: dict[str, object] = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return 0, "ok"

    monkeypatch.setattr(runner, "run_pytest", fake_run)
    assert (
        runner.main(
            [
                "--test-path",
                "tests/run_pytest_default_test.py",
                "--test-filter",
                "transport",
            ]
        )
        == 0
    )

    assert captured["args"] == []
    assert captured["test_path"] == "tests/run_pytest_default_test.py"
    assert captured["test_filter"] == "transport"
    assert captured["override_ini"] == []


def test_runner_cli_forwards_ordered_plural_targets(monkeypatch) -> None:
    runner = _load_runner()
    captured: dict[str, object] = {}

    def fake_run(_args, **kwargs):
        captured.update(kwargs)
        return 0, "ok"

    monkeypatch.setattr(runner, "run_pytest", fake_run)
    assert runner.main(["--test-paths-json", '["tests/a_test.py", "tests/b_test.py"]']) == 0
    assert captured["test_paths"] == ["tests/a_test.py", "tests/b_test.py"]


def test_plural_targets_validate_before_spawn(monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.setattr(
        runner, "run_pytest", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError())
    )
    assert runner.main(["--test-path", "tests/a_test.py", "--test-paths-json", "[]"]) == 1
    assert runner.main(["--test-paths-json", '["../outside"]']) == 1


def test_coverage_lease_rejects_live_holder_without_disclosing_lease_details(
    tmp_path: Path,
) -> None:
    """A live canonical-worktree lease must be busy and keep ownership values private."""
    runner = _load_runner()
    root = tmp_path / "repo"
    root.mkdir()
    lock_path = runner._get_coverage_lock_path(root)
    lease = runner.CoverageLease(
        "token-value-that-must-not-leak", os.getpid(), runner._coverage_worktree_id(root)
    )
    lock_path.write_text(json.dumps(lease.as_record()))

    try:
        runner._acquire_coverage_lock(root)
    except runner.CoverageLockError as exc:
        message = str(exc)
        assert "retry after it completes" in message
        assert str(root) not in message
        assert str(lease.pid) not in message
        assert lease.token not in message
    else:
        raise AssertionError("expected a live lease to reject a second owner")


def test_coverage_lease_recovers_verified_stale_holder(monkeypatch, tmp_path: Path) -> None:
    """A verified stale record is replaced by a new ownership-safe lease."""
    runner = _load_runner()
    root = tmp_path / "repo"
    root.mkdir()
    lock_path = runner._get_coverage_lock_path(root)
    stale = runner.CoverageLease("stale-token-value", 99999, runner._coverage_worktree_id(root))
    lock_path.write_text(json.dumps(stale.as_record()))

    def stale_process(_pid: int, _signal: int) -> None:
        raise OSError(errno.ESRCH, "missing")

    monkeypatch.setattr(runner.os, "kill", stale_process)
    acquired_path, acquired = runner._acquire_coverage_lock(root)

    assert acquired_path == lock_path
    assert acquired != stale
    assert runner._read_coverage_lease(lock_path, acquired.worktree_id) == acquired


def test_coverage_lease_release_does_not_delete_replacement(tmp_path: Path) -> None:
    """Release must preserve a record replaced by another contender."""
    runner = _load_runner()
    root = tmp_path / "repo"
    root.mkdir()
    lock_path = runner._get_coverage_lock_path(root)
    owner = runner.CoverageLease("owner-token-value", 1, runner._coverage_worktree_id(root))
    replacement = runner.CoverageLease("replacement-token-value", 2, owner.worktree_id)
    lock_path.write_text(json.dumps(replacement.as_record()))

    runner._release_coverage_lock(lock_path, owner)

    assert runner._read_coverage_lease(lock_path, owner.worktree_id) == replacement


def test_collect_only_requires_explicit_disabled_coverage_before_spawn(
    monkeypatch, tmp_path: Path
) -> None:
    """Direct runner callers cannot collect while implicit coverage remains enabled."""
    runner = _load_runner()
    monkeypatch.setattr(
        runner,
        "_run_pytest_subprocess",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not spawn")),
    )

    code, output = runner.run_pytest(["--collect-only"], cwd=str(tmp_path), output_mode="json")

    assert code == 1
    assert json.loads(output)["outcome"]["reason"] == "--collect-only requires coverage: false"


def test_collect_only_reports_collected_and_zero_executed_tests(
    monkeypatch, tmp_path: Path
) -> None:
    """Collection success uses collection counts rather than assertion metrics."""
    runner = _load_runner()
    monkeypatch.setattr(
        runner,
        "_run_pytest_subprocess",
        lambda *_args, **_kwargs: runner.PytestSubprocessResult(
            0, "===== 3 tests collected in 0.01s =====", ""
        ),
    )

    code, output = runner.run_pytest(
        ["--collect-only"], coverage=False, cwd=str(tmp_path), output_mode="json", min_test_count=2
    )

    payload = json.loads(output)
    assert code == 0
    assert payload["collection"] == {"collected_count": 3, "executed_count": 0}
    assert payload["outcome"] == {"classification": "collection", "status": "completed"}
    assert "assertion" not in payload
    assert "coverage" not in payload


def test_collect_only_accepts_singular_collection_summary(monkeypatch, tmp_path: Path) -> None:
    """A one-test collect-only summary is valid collection evidence."""
    runner = _load_runner()
    monkeypatch.setattr(
        runner,
        "_run_pytest_subprocess",
        lambda *_args, **_kwargs: runner.PytestSubprocessResult(
            0, "===== 1 test collected in 0.01s =====", ""
        ),
    )

    code, output = runner.run_pytest(
        ["--collect-only"], coverage=False, cwd=str(tmp_path), output_mode="json"
    )

    assert code == 0
    assert json.loads(output)["collection"] == {"collected_count": 1, "executed_count": 0}


def test_collect_only_failure_is_classified_and_rendered_as_collection(
    monkeypatch, tmp_path: Path
) -> None:
    """Missing collection evidence must not be reported as an assertion outcome."""
    runner = _load_runner()
    monkeypatch.setattr(
        runner,
        "_run_pytest_subprocess",
        lambda *_args, **_kwargs: runner.PytestSubprocessResult(
            0, "pytest output without summary", ""
        ),
    )

    code, output = runner.run_pytest(
        ["--collect-only"], coverage=False, cwd=str(tmp_path), output_mode="json"
    )
    payload = json.loads(output)

    assert code == 1
    assert payload["outcome"]["classification"] == "collection"
    assert payload["outcome"]["reason"] == "pytest collection summary is missing or ambiguous"

    _, summary = runner.run_pytest(
        ["--collect-only"],
        coverage=False,
        cwd=str(tmp_path),
        output_mode="summary",
        min_test_count=1,
    )
    assert "Collection:" not in summary


def test_collect_only_summary_reports_collected_and_executed_counts(
    monkeypatch, tmp_path: Path
) -> None:
    """Successful collection summary explicitly distinguishes collection from execution."""
    runner = _load_runner()
    monkeypatch.setattr(
        runner,
        "_run_pytest_subprocess",
        lambda *_args, **_kwargs: runner.PytestSubprocessResult(
            0, "===== 2 tests collected in 0.01s =====", ""
        ),
    )

    code, output = runner.run_pytest(
        ["--collect-only"], coverage=False, cwd=str(tmp_path), output_mode="summary"
    )

    assert code == 0
    assert "Collection: 2 collected, 0 executed" in output


def test_coverage_coordination_json_failure_has_bounded_classification(
    monkeypatch, tmp_path: Path
) -> None:
    """Lease contention must retain its coordination classification in JSON output."""
    runner = _load_runner()
    monkeypatch.setattr(
        runner,
        "_acquire_coverage_lock",
        lambda *_args: (_ for _ in ()).throw(runner.CoverageLockError("busy; retry later")),
    )

    code, output = runner.run_pytest([], cwd=str(tmp_path), output_mode="json")
    payload = json.loads(output)

    assert code == 1
    assert payload["outcome"]["classification"] == "coordination"
    assert payload["outcome"]["reason"] == "busy; retry later"


def test_coverage_lease_rejects_symlink_state_path(tmp_path: Path) -> None:
    """Lease state must not follow a runtime-state symlink."""
    runner = _load_runner()
    root = tmp_path / "repo"
    root.mkdir()
    (root / "adforge_local").mkdir()
    (root / "adforge_local" / "state").symlink_to(tmp_path)

    try:
        runner._acquire_coverage_lock(root)
    except runner.CoverageLockError as exc:
        assert str(root) not in str(exc)
    else:
        raise AssertionError("expected symlinked state path to be rejected")


def test_coverage_lease_rejects_malformed_and_wrong_worktree_records(tmp_path: Path) -> None:
    """Malformed and foreign records fail closed without publishing a new lease."""
    runner = _load_runner()
    root = tmp_path / "repo"
    root.mkdir()
    lock_path = runner._get_coverage_lock_path(root)

    for content in (
        "not-json",
        json.dumps({"token": "x" * 16, "pid": 1, "worktree_id": "foreign"}),
    ):
        lock_path.write_text(content)
        try:
            runner._acquire_coverage_lock(root)
        except runner.CoverageLockError as exc:
            assert str(root) not in str(exc)
        else:
            raise AssertionError("expected invalid lease record rejection")


def test_coverage_lease_interrupted_publish_cleans_owned_temp(monkeypatch, tmp_path: Path) -> None:
    """A publication failure leaves neither a partial lease nor an owned temp artifact."""
    runner = _load_runner()
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(
        runner.os,
        "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError(errno.EPERM, "denied")),
    )

    try:
        runner._acquire_coverage_lock(root)
    except runner.CoverageLockError as exc:
        assert "denied" not in str(exc)
    else:
        raise AssertionError("expected publication failure")
    state = root / "adforge_local" / "state"
    assert not (state / runner.COVERAGE_LOCK_FILENAME).exists()
    assert not list(state.glob(f".{runner.COVERAGE_LOCK_FILENAME}.*.tmp"))


def test_runner_unexpected_json_failure_is_sanitized(monkeypatch, tmp_path: Path) -> None:
    """Unexpected runner failures retain the JSON envelope and redact OS details."""
    runner = _load_runner()
    monkeypatch.setattr(
        runner,
        "_run_pytest_subprocess",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError(13, f"{tmp_path}/secret")),
    )

    code, output = runner.run_pytest([], cwd=str(tmp_path), output_mode="json", coverage=False)
    payload = json.loads(output)
    assert code == 1
    assert payload["outcome"]["classification"] == "runner"
    assert str(tmp_path) not in output


@pytest.mark.parametrize("entry", ["addopts=/outside", "pythonpath=/outside"])
def test_runner_cli_rejects_all_override_ini_before_spawn(monkeypatch, entry: str) -> None:
    runner = _load_runner()
    monkeypatch.setattr(
        runner, "run_pytest", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError())
    )
    assert runner.main(["--override-ini-json", json.dumps([entry])]) == 1


@pytest.mark.parametrize("argv", [["-p", "unsafe_plugin"], ["--override-ini=pythonpath=/outside"]])
def test_runner_cli_rejects_plugin_and_ini_pytest_args_before_spawn(monkeypatch, argv) -> None:
    """Caller pytest argv cannot load plugins or override ini settings."""
    runner = _load_runner()
    monkeypatch.setattr(
        runner, "run_pytest", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError())
    )
    assert runner.main(["--pytest-argv-json", json.dumps(argv)]) == 1


def test_runner_relativizes_root_target_for_nested_execution_cwd(tmp_path: Path) -> None:
    runner = _load_runner()
    root = tmp_path / "repo"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    (root / "tests").mkdir()

    assert runner._execution_target("tests/sample_test.py::test_sample", cwd=str(nested)) == (
        "../tests/sample_test.py::test_sample"
    )


def test_runner_orders_converted_plural_targets_before_caller_suffix(
    monkeypatch, tmp_path: Path
) -> None:
    """Plural runner targets retain order and precede caller-owned pytest arguments."""
    runner = _load_runner()
    root = tmp_path / "repo"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    captured: dict[str, list[str]] = {}

    def fake_subprocess(command, **_kwargs):
        captured["command"] = command
        return runner.PytestSubprocessResult(0, "===== 1 passed in 0.01s =====", "")

    monkeypatch.setattr(runner, "_run_pytest_subprocess", fake_subprocess)
    code, _ = runner.run_pytest(
        ["-q"],
        coverage=False,
        cwd=str(nested),
        test_paths=["tests/a_test.py", "tests/b_test.py::test_b"],
    )

    assert code == 0
    command = captured["command"]
    assert command.index("../tests/a_test.py") < command.index("../tests/b_test.py::test_b")
    assert command.index("../tests/b_test.py::test_b") < command.index("-q")


def test_runner_json_prelaunch_failure_uses_canonical_outcome() -> None:
    runner = _load_runner()
    payload = _json_payload_with_identity(
        runner._json_prelaunch_failure("bad input", resolved_target="tests/x.py", phase="pre_spawn")
    )

    assert payload["success"] is False
    assert payload["outcome"]["phase"] == "pre_spawn"
    assert payload["outcome"]["resolved_target"] == "tests/x.py"


def test_failure_outcome_bounds_and_redacts_absolute_node_ids() -> None:
    runner = _load_runner()
    outcome = runner._failure_outcome(
        returncode=1,
        validation_errors=["failed"],
        stdout="/private/test.py::test_secret " * (runner.MAX_DIAGNOSTIC_NODE_IDS + 2),
        stderr="",
        elapsed_seconds=0.1,
        resolved_target="tests/x.py",
    )

    assert outcome["node_ids"] == ["<absolute-node-id>"] * runner.MAX_DIAGNOSTIC_NODE_IDS
    assert outcome["truncation"]["node_ids_omitted"] == 2


def test_coverage_source_normalizer_requires_safe_canonical_sources(tmp_path: Path) -> None:
    runner = _load_runner()
    (tmp_path / "pkg").mkdir()
    (tmp_path / "module.py").write_text("x = 1\n")

    assert runner._normalize_coverage_source("pkg,module.py,adw.core", tmp_path) == [
        "pkg",
        "module.py",
        "adw.core",
    ]
    for value in ("all,pkg", "pkg//child", "../outside", "module.txt", "pkg/tests.md", [None]):
        try:
            runner._normalize_coverage_source(value, tmp_path)
        except runner.CoverageSourceValidationError:
            pass
        else:
            raise AssertionError(f"expected source rejection for {value!r}")


def test_root_level_python_coverage_source_is_path_validated(tmp_path: Path) -> None:
    """Root-level Python filenames must not bypass source confinement checks."""
    runner = _load_runner()
    (tmp_path / "safe.py").write_text("x = 1\n")
    outside = tmp_path.parent / "outside.py"
    outside.write_text("x = 1\n")
    (tmp_path / "linked.py").symlink_to(outside)

    assert runner._normalize_coverage_source("safe.py", tmp_path) == ["safe.py"]
    for value in ("missing.py", "linked.py"):
        try:
            runner._normalize_coverage_source(value, tmp_path)
        except runner.CoverageSourceValidationError:
            pass
        else:
            raise AssertionError(f"expected source rejection for {value!r}")


def test_coverage_paths_are_resolved_from_repository_root_for_nested_cwd(tmp_path: Path) -> None:
    runner = _load_runner()
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    (tmp_path / "root_module.py").write_text("x = 1\n")

    assert runner._resolve_normalized_sources(str(nested), "root_module.py") == ["root_module.py"]


def test_disabled_coverage_returns_disabled_projection_without_spawn_coverage_args(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\naddopts = '--cov=bad -q'\n"
    )
    captured: dict[str, object] = {}

    def fake_subprocess(command, **_kwargs):
        captured["command"] = command
        return runner.PytestSubprocessResult(0, "===== 1 passed in 0.01s =====", "")

    monkeypatch.setattr(runner, "_run_pytest_subprocess", fake_subprocess)
    code, output = runner.run_pytest([], output_mode="json", cwd=str(tmp_path), coverage=False)
    payload = json.loads(output)

    assert code == 0
    assert payload["assertion"]["status"] == "passed"
    assert payload["coverage"] == {"status": "disabled", "reasons": []}
    assert all(not str(arg).startswith("--cov") for arg in cast(list[object], captured["command"]))


def test_disabled_coverage_removes_inherited_coverage_addopts(monkeypatch, tmp_path: Path) -> None:
    runner = _load_runner()
    captured: dict[str, object] = {}

    class FakeProcess:
        returncode = 0

        def communicate(self, timeout):
            return "===== 1 passed in 0.01s =====", ""

    def fake_popen(*_args, **kwargs):
        captured["env"] = kwargs["env"]
        return FakeProcess()

    monkeypatch.setenv("PYTEST_ADDOPTS", "--cov adw --cov-config coveragerc -q")
    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    code, _ = runner.run_pytest([], output_mode="json", cwd=str(tmp_path), coverage=False)

    assert code == 0
    assert cast(dict[str, str], captured["env"])["PYTEST_ADDOPTS"] == "-q"


def test_disabled_coverage_rejects_invalid_source_before_spawn(monkeypatch, tmp_path: Path) -> None:
    runner = _load_runner()
    monkeypatch.setattr(
        runner,
        "_run_pytest_subprocess",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not spawn")),
    )

    code, output = runner.run_pytest(
        [], output_mode="json", cwd=str(tmp_path), coverage=False, coverage_source="module.txt"
    )

    payload = json.loads(output)
    assert code == 1
    assert payload["success"] is False
    assert (
        payload["outcome"]["reason"]
        == "coverage-specific controls are not allowed when coverage is disabled"
    )


def test_coverage_floor_retains_policy_and_strengthens_only(tmp_path: Path) -> None:
    runner = _load_runner()
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\n"
        "addopts = '--cov adw --cov-report term --cov-fail-under 80 -q'\n"
        "[tool.coverage.report]\nfail_under = 80\n"
    )

    command, cov_args, overrides, _ = runner._build_pytest_command(
        args=[],
        fail_fast=False,
        durations=None,
        durations_min=None,
        coverage=True,
        normalized_sources=["adw", "adforge_core"],
        cov_report="term-missing",
        override_ini=None,
        coverage_floor=90,
        root_dir=tmp_path,
    )

    assert cov_args == [
        "--cov=adw",
        "--cov=adforge_core",
        "--cov-fail-under=90",
        "--cov-report=term-missing",
    ]
    assert command.count("--cov=adw") == 1
    assert overrides == ["addopts=-q"]
    assert runner._effective_coverage_floor(tmp_path, 90) == 90
    try:
        runner._effective_coverage_floor(tmp_path, 79)
    except runner.CoverageSourceValidationError:
        pass
    else:
        raise AssertionError("expected lower coverage threshold rejection")


def test_coverage_floor_never_drops_below_repository_minimum(tmp_path: Path) -> None:
    """Keep the wrapper's documented 80 percent minimum despite weaker local config."""
    runner = _load_runner()
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\naddopts = '--cov-fail-under=70'\n"
        "[tool.coverage.report]\nfail_under = 70\n"
    )

    assert runner._effective_coverage_floor(tmp_path, None) == 80


def test_coverage_floor_rejects_nonfinite_policy_values(tmp_path: Path) -> None:
    runner = _load_runner()
    for value in ("nan", "inf", "-inf"):
        (tmp_path / "pyproject.toml").write_text(
            f"[tool.pytest.ini_options]\naddopts = '--cov-fail-under={value}'\n"
        )
        try:
            runner._effective_coverage_floor(tmp_path, None)
        except runner.CoverageSourceValidationError as exc:
            assert str(exc) == "configured --cov-fail-under must be finite"
        else:
            raise AssertionError(f"expected rejection for {value}")


def test_coverage_policy_read_failure_is_bounded_and_path_free(monkeypatch, tmp_path: Path) -> None:
    runner = _load_runner()
    policy = tmp_path / "pyproject.toml"
    policy.write_text("[tool.coverage.report]\nfail_under = 80\n")
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("/secret/path")),
    )

    try:
        runner._effective_coverage_floor(tmp_path, None)
    except runner.CoverageSourceValidationError as exc:
        assert str(exc) == "coverage policy could not be read"
        assert "/secret/path" not in str(exc)
    else:
        raise AssertionError("expected bounded policy-read failure")


def test_disabled_command_strips_separated_coverage_addopts_without_orphans(tmp_path: Path) -> None:
    """Remove inherited coverage option operands when coverage is disabled."""
    runner = _load_runner()
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\n"
        "addopts = '--cov adw --cov-report term --cov-fail-under 80 -q --import-mode=importlib'\n"
    )

    command, cov_args, overrides, _ = runner._build_pytest_command(
        args=[],
        fail_fast=False,
        durations=None,
        durations_min=None,
        coverage=False,
        normalized_sources=[],
        cov_report="term-missing",
        override_ini=None,
        root_dir=tmp_path,
    )

    assert cov_args == []
    assert command.count("adw") == 0
    assert command.count("term") == 0
    assert "--cov-fail-under" not in command
    assert overrides == ["addopts=-q --import-mode=importlib"]


def test_disabled_command_keeps_following_noncoverage_option_after_bare_cov(tmp_path: Path) -> None:
    """A bare ``--cov`` must not swallow the next non-coverage addopts token."""
    runner = _load_runner()
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\naddopts = '--cov -q'\n")

    command, cov_args, overrides, _ = runner._build_pytest_command(
        args=[],
        fail_fast=False,
        durations=None,
        durations_min=None,
        coverage=False,
        normalized_sources=[],
        cov_report="term-missing",
        override_ini=None,
        root_dir=tmp_path,
    )

    assert cov_args == []
    assert command.count("-q") == 0
    assert overrides == ["addopts=-q"]


def test_repository_coverage_floor_rejects_missing_cov_fail_under_value(tmp_path: Path) -> None:
    runner = _load_runner()
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\naddopts = '--cov-fail-under'\n"
    )

    try:
        runner._repository_coverage_floor(tmp_path, ["--cov-fail-under"])
    except runner.CoverageSourceValidationError as exc:
        assert "must be followed by a value" in str(exc)
    else:
        raise AssertionError("expected missing value rejection")


def test_command_preserves_quoted_non_coverage_addopts_values(tmp_path: Path) -> None:
    """Retained marker expressions must not become positional pytest targets."""
    runner = _load_runner()
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\naddopts = \"-m 'not slow and not visual' --cov=adw -n auto\"\n"
    )

    _, _, overrides, _ = runner._build_pytest_command(
        args=[],
        fail_fast=False,
        durations=None,
        durations_min=None,
        coverage=False,
        normalized_sources=[],
        cov_report="term-missing",
        override_ini=None,
        root_dir=tmp_path,
    )

    assert overrides == ["addopts=-m 'not slow and not visual' -n auto"]


def test_command_does_not_inject_empty_addopts_override_when_repo_has_none(tmp_path: Path) -> None:
    """Avoid clearing repository addopts when no retained options exist."""
    runner = _load_runner()
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")

    _, _, overrides, _ = runner._build_pytest_command(
        args=[],
        fail_fast=False,
        durations=None,
        durations_min=None,
        coverage=False,
        normalized_sources=[],
        cov_report="term-missing",
        override_ini=None,
        root_dir=tmp_path,
    )

    assert overrides == []


def test_coverage_evaluation_marks_missing_and_unusable_evidence_failed() -> None:
    """Keep missing totals and either diagnostic stream independent from assertions."""
    runner = _load_runner()
    metrics = {"coverage_pct": None}

    missing_total = runner._evaluate_coverage(metrics, enabled=True, floor=80, output="")
    no_data_stdout = runner._evaluate_coverage(
        metrics, enabled=True, floor=80, output="WARNING: No data collected"
    )
    never_imported_stderr = runner._evaluate_coverage(
        metrics, enabled=True, floor=80, output="STDERR:\nModule was never imported"
    )

    assert missing_total == {
        "status": "failed",
        "reasons": [
            "Coverage data is unavailable: pytest-cov did not report a TOTAL coverage percentage."
        ],
    }
    assert no_data_stdout["status"] == "failed"
    assert "no data collected" in no_data_stdout["reasons"][0].lower()
    assert never_imported_stderr["status"] == "failed"
    assert "never imported" in never_imported_stderr["reasons"][0].lower()


def test_coverage_and_assertion_results_are_independent(monkeypatch, tmp_path: Path) -> None:
    runner = _load_runner()
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\naddopts = '--cov=adw --cov-fail-under=80'\n"
    )
    monkeypatch.setattr(
        runner,
        "_run_pytest_subprocess",
        lambda *_args, **_kwargs: runner.PytestSubprocessResult(
            1, "===== 1 failed in 0.01s =====\nTOTAL 10 1 90%", ""
        ),
    )
    code, output = runner.run_pytest([], output_mode="json", cwd=str(tmp_path))
    payload = json.loads(output)

    assert code == 1
    assert payload["assertion"]["status"] == "failed"
    assert payload["coverage"] == {"status": "passed", "reasons": [], "percentage": 90}


def test_coverage_only_failure_keeps_assertions_passed(monkeypatch, tmp_path: Path) -> None:
    runner = _load_runner()
    monkeypatch.setattr(
        runner,
        "_run_pytest_subprocess",
        lambda *_args, **_kwargs: runner.PytestSubprocessResult(
            1, "===== 1 passed in 0.01s =====\nTOTAL 10 1 10%", ""
        ),
    )

    code, output = runner.run_pytest([], output_mode="json", cwd=str(tmp_path))
    payload = _json_payload_with_identity(output)

    assert code == 1
    assert payload["assertion"] == {"status": "passed", "reasons": []}
    assert payload["coverage"]["status"] == "failed"


def test_runner_json_terminal_paths_publish_identity_and_text_does_not(
    monkeypatch, tmp_path: Path
) -> None:
    """Every runner-owned JSON terminal path publishes, but text never renders, identity."""
    runner = _load_runner()
    cases: tuple[tuple[int, str, list[str], bool, str], ...] = (
        (0, "===== 1 passed in 0.01s =====", [], False, "passed"),
        (1, "===== 1 failed in 0.01s =====", [], False, "failed"),
        (0, "===== 2 tests collected in 0.01s =====", ["--collect-only"], False, "collection"),
    )

    for returncode, stdout, args, coverage, expected in cases:
        monkeypatch.setattr(
            runner,
            "_run_pytest_subprocess",
            lambda *_args, returncode=returncode, stdout=stdout, **_kwargs: (
                runner.PytestSubprocessResult(returncode, stdout, "")
            ),
        )
        code, output = runner.run_pytest(
            args, cwd=str(tmp_path), coverage=coverage, output_mode="json"
        )
        payload = _json_payload_with_identity(output)
        assert code == (0 if expected in {"passed", "collection"} else 1)
        if expected == "collection":
            assert payload["outcome"]["classification"] == "collection"
        else:
            assert payload["assertion"]["status"] == expected

    prelaunch = _json_payload_with_identity(
        runner._json_prelaunch_failure("bad input", resolved_target=None, phase="pre_spawn")
    )
    assert prelaunch["outcome"]["classification"] == "invocation"

    monkeypatch.setattr(
        runner,
        "_run_pytest_subprocess",
        lambda *_args, **_kwargs: runner.PytestSubprocessResult(
            0, "===== 1 passed in 0.01s =====", ""
        ),
    )
    _, text_output = runner.run_pytest([], cwd=str(tmp_path), coverage=False, output_mode="summary")
    assert "evidence_identity" not in text_output
