from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterator, List

import pytest

BUILD_MKDOCS_PATH = Path(__file__).resolve().parent.parent / "build_mkdocs.py"
REPO_ROOT = Path(__file__).resolve().parents[3]
INTEGRATION_PYTEST_MARKS = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("mkdocs") is None, reason="mkdocs not installed"),
]


def _remove_site_dir(site_dir: Path) -> None:
    if site_dir.exists():
        shutil.rmtree(site_dir)


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_mkdocs", BUILD_MKDOCS_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_mkdocs"] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


@pytest.fixture(scope="module")
def build_mkdocs_module() -> ModuleType:
    return load_module()


class DummyProcess:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_build_command_defaults(build_mkdocs_module: ModuleType) -> None:
    cmd = build_mkdocs_module.build_command()
    assert cmd[:2] == ["mkdocs", "build"]
    assert "--clean" in cmd
    assert "--strict" not in cmd
    assert "--config-file" not in cmd


def test_build_command_strict_no_clean(build_mkdocs_module: ModuleType) -> None:
    cmd = build_mkdocs_module.build_command(strict=True, clean=False)
    assert "--strict" in cmd
    assert "--clean" not in cmd


def test_build_command_config_override(build_mkdocs_module: ModuleType) -> None:
    cmd = build_mkdocs_module.build_command(config_file="docs/mkdocs.yml")
    assert "--config-file" in cmd
    assert cmd[cmd.index("--config-file") + 1] == "docs/mkdocs.yml"


def test_build_command_validate_only_requires_site_dir(build_mkdocs_module: ModuleType) -> None:
    with pytest.raises(ValueError):
        build_mkdocs_module.build_command(validate_only=True)


def test_build_command_validate_only_with_site_dir(build_mkdocs_module: ModuleType) -> None:
    cmd = build_mkdocs_module.build_command(validate_only=True, site_dir="/tmp/site")
    assert "--site-dir" in cmd
    assert cmd[cmd.index("--site-dir") + 1] == "/tmp/site"


def test_parse_args_defaults(build_mkdocs_module: ModuleType) -> None:
    args = build_mkdocs_module._parse_args([])

    assert args.output == "summary"
    assert args.timeout == build_mkdocs_module.DEFAULT_TIMEOUT
    assert args.cwd is None
    assert args.strict is False
    assert args.clean is True
    assert args.config_file == "mkdocs.yml"
    assert args.validate_only is False


def test_format_summary_includes_status(build_mkdocs_module: ModuleType) -> None:
    summary = build_mkdocs_module.format_summary(exit_code=0, stdout="ok", stderr="")
    assert "MKDOCS BUILD SUMMARY" in summary
    assert "Status: PASSED" in summary
    assert "Exit Code: 0" in summary
    assert "ok" in summary


def test_format_full_output_includes_stderr(build_mkdocs_module: ModuleType) -> None:
    output = build_mkdocs_module.format_full_output(stdout="out", stderr="err")
    assert "out" in output
    assert "STDERR:" in output
    assert "err" in output


def test_truncate_output_limits_lines(build_mkdocs_module: ModuleType) -> None:
    lines = ["line"] * (build_mkdocs_module.OUTPUT_LINE_LIMIT + 1)
    output = "\n".join(lines)

    truncated_output, truncated, notice = build_mkdocs_module._truncate_output(output)

    assert truncated is True
    assert notice
    assert f"{build_mkdocs_module.OUTPUT_LINE_LIMIT} lines" in notice
    assert truncated_output.endswith(notice)


def test_truncate_output_limits_bytes(build_mkdocs_module: ModuleType) -> None:
    output = "a" * (build_mkdocs_module.OUTPUT_BYTE_LIMIT + 10)

    truncated_output, truncated, notice = build_mkdocs_module._truncate_output(output)

    assert truncated is True
    assert notice
    assert "Output truncated to" in notice
    assert truncated_output.endswith(notice)


def test_combine_output_includes_stderr_label(build_mkdocs_module: ModuleType) -> None:
    combined = build_mkdocs_module._combine_output("stdout", "stderr")

    assert "stdout" in combined
    assert "STDERR:" in combined
    assert "stderr" in combined


def test_format_json_output_includes_error_and_truncation(
    build_mkdocs_module: ModuleType,
) -> None:
    output = "b" * (build_mkdocs_module.OUTPUT_BYTE_LIMIT + 5)
    payload = json.loads(
        build_mkdocs_module._format_json_output(
            exit_code=1,
            stdout=output,
            stderr="",
            options={"cwd": "/tmp"},
            error_message="boom",
        )
    )

    assert payload["success"] is False
    assert payload["exit_code"] == 1
    assert payload["truncated"] is True
    assert payload["truncation_notice"]
    assert payload["error"]["message"] == "boom"


def test_resolve_cwd_respects_override(build_mkdocs_module: ModuleType, tmp_path: Path) -> None:
    resolved = build_mkdocs_module.resolve_cwd(str(tmp_path))
    assert resolved == tmp_path


def test_resolve_cwd_walks_up(
    build_mkdocs_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "mkdocs.yml").write_text("site_name: demo")
    nested = root / "nested" / "more"
    nested.mkdir(parents=True)

    monkeypatch.chdir(nested)
    resolved = build_mkdocs_module.resolve_cwd(None)
    assert resolved == root


def test_resolve_cwd_fallback_to_current(
    build_mkdocs_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    resolved = build_mkdocs_module.resolve_cwd(None)
    assert resolved == Path.cwd()


def test_resolve_config_path_relative(build_mkdocs_module: ModuleType, tmp_path: Path) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    resolved = build_mkdocs_module.resolve_config_path("docs/mkdocs.yml", cwd)
    assert resolved == (cwd / "docs/mkdocs.yml").resolve()


def test_resolve_config_path_absolute(build_mkdocs_module: ModuleType, tmp_path: Path) -> None:
    config = tmp_path / "mkdocs.yml"
    resolved = build_mkdocs_module.resolve_config_path(str(config), tmp_path)
    assert resolved == config.resolve()


def test_run_mkdocs_missing_config_skips_subprocess(
    build_mkdocs_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def should_not_run(*_: object, **__: object) -> None:  # type: ignore[override]
        raise AssertionError("subprocess.run should not be called when config is missing")

    monkeypatch.setattr(build_mkdocs_module.subprocess, "run", should_not_run)

    exit_code, output = build_mkdocs_module.run_mkdocs(
        cwd=str(tmp_path),
        output_mode="json",
    )

    payload = json.loads(output)
    assert exit_code == 1
    assert payload["success"] is False
    assert "config file not found" in payload["error"]["message"]


def test_run_mkdocs_missing_config_summary(build_mkdocs_module: ModuleType, tmp_path: Path) -> None:
    exit_code, output = build_mkdocs_module.run_mkdocs(
        cwd=str(tmp_path),
        output_mode="summary",
    )

    assert exit_code == 1
    assert "Status: FAILED" in output
    assert "config file not found" in output


def test_run_mkdocs_missing_config_full(build_mkdocs_module: ModuleType, tmp_path: Path) -> None:
    exit_code, output = build_mkdocs_module.run_mkdocs(
        cwd=str(tmp_path),
        output_mode="full",
    )

    assert exit_code == 1
    assert "ERROR:" in output
    assert "config file not found" in output


def test_run_mkdocs_success_json(
    build_mkdocs_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mkdocs.yml").write_text("site_name: demo")
    commands: List[List[str]] = []
    captured_kwargs = {}

    def fake_run(cmd: List[str], **kwargs: object) -> DummyProcess:  # type: ignore[override]
        commands.append(cmd)
        captured_kwargs.update(kwargs)
        return DummyProcess(stdout="Build complete", stderr="", returncode=0)

    monkeypatch.setattr(build_mkdocs_module.subprocess, "run", fake_run)

    exit_code, output = build_mkdocs_module.run_mkdocs(
        output_mode="json",
        cwd=str(tmp_path),
        strict=True,
        clean=False,
    )

    payload = json.loads(output)
    assert exit_code == 0
    assert payload["success"] is True
    assert payload["stdout"] == "Build complete"
    assert payload["options"]["strict"] is True
    assert payload["options"]["clean"] is False
    assert commands
    assert "--strict" in commands[0]
    assert "--clean" not in commands[0]
    assert captured_kwargs["cwd"] == str(tmp_path)


def test_run_mkdocs_timeout_json(
    build_mkdocs_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mkdocs.yml").write_text("site_name: demo")

    def raise_timeout(*_: object, **__: object) -> None:  # type: ignore[override]
        raise subprocess.TimeoutExpired(cmd="mkdocs", timeout=1)

    monkeypatch.setattr(build_mkdocs_module.subprocess, "run", raise_timeout)

    exit_code, output = build_mkdocs_module.run_mkdocs(
        output_mode="json",
        cwd=str(tmp_path),
        timeout=1,
    )

    payload = json.loads(output)
    assert exit_code == 1
    assert payload["success"] is False
    assert "timed out" in payload["error"]["message"]


def test_run_mkdocs_missing_binary_json(
    build_mkdocs_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mkdocs.yml").write_text("site_name: demo")

    def raise_missing(*_: object, **__: object) -> None:  # type: ignore[override]
        raise FileNotFoundError()

    monkeypatch.setattr(build_mkdocs_module.subprocess, "run", raise_missing)

    exit_code, output = build_mkdocs_module.run_mkdocs(
        output_mode="json",
        cwd=str(tmp_path),
    )

    payload = json.loads(output)
    assert exit_code == 1
    assert payload["success"] is False
    assert "mkdocs not found" in payload["error"]["message"]


def test_run_mkdocs_generic_exception_json(
    build_mkdocs_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mkdocs.yml").write_text("site_name: demo")

    def raise_generic(*_: object, **__: object) -> None:  # type: ignore[override]
        raise RuntimeError("boom")

    monkeypatch.setattr(build_mkdocs_module.subprocess, "run", raise_generic)

    exit_code, output = build_mkdocs_module.run_mkdocs(
        output_mode="json",
        cwd=str(tmp_path),
    )

    payload = json.loads(output)
    assert exit_code == 1
    assert payload["success"] is False
    assert "Unexpected error running mkdocs: boom" in payload["error"]["message"]


def test_run_mkdocs_validate_only_uses_tempdir(
    build_mkdocs_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mkdocs.yml").write_text("site_name: demo")
    captured_site_dir: List[str] = []

    def fake_run(cmd: List[str], **_: object) -> DummyProcess:  # type: ignore[override]
        assert "--site-dir" in cmd
        site_dir = cmd[cmd.index("--site-dir") + 1]
        captured_site_dir.append(site_dir)
        assert Path(site_dir).exists()
        return DummyProcess(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(build_mkdocs_module.subprocess, "run", fake_run)

    exit_code, output = build_mkdocs_module.run_mkdocs(
        output_mode="summary",
        cwd=str(tmp_path),
        validate_only=True,
    )

    assert exit_code == 0
    assert "Status: PASSED" in output
    assert captured_site_dir
    assert not Path(captured_site_dir[0]).exists()


def test_run_mkdocs_passes_timeout_and_cwd(
    build_mkdocs_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mkdocs.yml").write_text("site_name: demo")
    captured_kwargs = {}

    def fake_run(cmd: List[str], **kwargs: object) -> DummyProcess:  # type: ignore[override]
        captured_kwargs.update(kwargs)
        return DummyProcess(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(build_mkdocs_module.subprocess, "run", fake_run)

    exit_code, _ = build_mkdocs_module.run_mkdocs(
        output_mode="summary",
        cwd=str(tmp_path),
        timeout=12,
    )

    assert exit_code == 0
    assert captured_kwargs["cwd"] == str(tmp_path)
    assert captured_kwargs["timeout"] == 12


def test_main_wires_cli_arguments(
    build_mkdocs_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured_args = {}

    def fake_run_mkdocs(**kwargs: object) -> tuple[int, str]:
        captured_args.update(kwargs)
        return 0, "ok"

    monkeypatch.setattr(build_mkdocs_module, "run_mkdocs", fake_run_mkdocs)

    with pytest.raises(SystemExit) as excinfo:
        build_mkdocs_module.main(
            [
                "--output",
                "json",
                "--timeout",
                "45",
                "--cwd",
                "/tmp",
                "--strict",
                "--no-clean",
                "--config-file",
                "docs/mkdocs.yml",
                "--validate-only",
            ]
        )

    captured = capsys.readouterr()

    assert excinfo.value.code == 0
    assert captured.out.strip() == "ok"
    assert captured_args["output_mode"] == "json"
    assert captured_args["timeout"] == 45
    assert captured_args["cwd"] == "/tmp"
    assert captured_args["strict"] is True
    assert captured_args["clean"] is False
    assert captured_args["config_file"] == "docs/mkdocs.yml"
    assert captured_args["validate_only"] is True


class TestBuildMkdocsIntegration:
    pytestmark = INTEGRATION_PYTEST_MARKS

    @pytest.fixture(scope="class", autouse=True)
    def _require_mkdocs(self) -> None:
        pytest.importorskip("mkdocs")

    @pytest.fixture(autouse=True)
    def _clean_site_dir(self) -> Iterator[None]:
        site_dir = REPO_ROOT / "site"
        _remove_site_dir(site_dir)
        yield
        _remove_site_dir(site_dir)

    def test_default_build_succeeds(self, build_mkdocs_module: ModuleType) -> None:
        exit_code, output = build_mkdocs_module.run_mkdocs(
            output_mode="summary",
            cwd=str(REPO_ROOT),
        )
        assert exit_code == 0
        assert "MKDOCS BUILD SUMMARY" in output
        assert "Status: PASSED" in output

    def test_validate_only_leaves_no_site_dir(self, build_mkdocs_module: ModuleType) -> None:
        site_dir = REPO_ROOT / "site"

        exit_code, output = build_mkdocs_module.run_mkdocs(
            output_mode="summary",
            cwd=str(REPO_ROOT),
            validate_only=True,
        )

        assert exit_code == 0
        assert "Status: PASSED" in output
        assert not site_dir.exists()

    def test_strict_mode_json_contains_options(self, build_mkdocs_module: ModuleType) -> None:
        exit_code, output = build_mkdocs_module.run_mkdocs(
            output_mode="json",
            cwd=str(REPO_ROOT),
            strict=True,
        )

        payload = json.loads(output)
        assert payload["success"] == (payload["exit_code"] == 0)
        assert payload["exit_code"] in (0, 1)
        assert "stdout" in payload
        assert "stderr" in payload
        assert payload["options"]["strict"] is True
        assert exit_code in (0, 1)

    def test_cwd_override_builds_from_repo_root(
        self, build_mkdocs_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(REPO_ROOT / "adw-docs")

        exit_code, output = build_mkdocs_module.run_mkdocs(
            output_mode="summary",
            cwd=str(REPO_ROOT),
        )
        assert exit_code == 0
        assert "MKDOCS BUILD SUMMARY" in output
        assert "Status: PASSED" in output

    def test_json_output_valid(self, build_mkdocs_module: ModuleType) -> None:
        exit_code, output = build_mkdocs_module.run_mkdocs(
            output_mode="json",
            cwd=str(REPO_ROOT),
        )

        payload = json.loads(output)
        assert payload["success"] == (payload["exit_code"] == 0)
        assert payload["exit_code"] == exit_code
        assert "stdout" in payload
        assert "stderr" in payload
        assert "options" in payload

    def test_missing_config_returns_error(self, build_mkdocs_module: ModuleType) -> None:
        exit_code, output = build_mkdocs_module.run_mkdocs(
            output_mode="json",
            cwd=str(REPO_ROOT),
            config_file="nonexistent.yml",
        )

        payload = json.loads(output)
        assert exit_code != 0
        assert payload["success"] is False
        assert "mkdocs config file not found" in payload["error"]["message"]
