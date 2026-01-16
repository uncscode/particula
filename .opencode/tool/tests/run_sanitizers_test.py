"""Unit tests for run_sanitizers.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import List, Tuple

import pytest

RUN_SANITIZERS_PATH = Path(__file__).resolve().parent.parent / "run_sanitizers.py"


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_sanitizers", RUN_SANITIZERS_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


@pytest.fixture(scope="module")
def run_sanitizers_module() -> ModuleType:
    return load_module()


class DummyProcess:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


ASAN_ERROR_OUTPUT = (
    "=================================================================\n"
    "==12345==ERROR: AddressSanitizer: heap-use-after-free on address 0x602000000014\n"
    "READ of size 4 at 0x602000000014 thread T0\n"
    "    #0 0x4f5b6f in main /src/foo.cpp:42:7\n"
    "    #1 0x7f0 in __libc_start_main\n"
)

TSAN_ERROR_OUTPUT = (
    "==================\n"
    "WARNING: ThreadSanitizer: data race (pid=321)\n"
    "  Read of size 8 at 0x0 by thread T1:\n"
    "    #0 read_fn /src/tsan.cpp:10\n"
    "  Previous write of size 8 at 0x0 by thread T2:\n"
    "    #0 write_fn /src/tsan.cpp:5\n"
    "==================\n"
    "WARNING: ThreadSanitizer: mutex lock order violation\n"
    "    #0 lock_fn /src/tsan2.cpp:20\n"
)

UBSAN_ERROR_OUTPUT = (
    "/src/ubsan.c:12:5: runtime error: shift exponent 32 is too large\n"
    "    #0 0x4b5c in foo /src/ubsan.c:12:5\n"
)


def test_env_var_for_sanitizer_helpers(run_sanitizers_module: ModuleType) -> None:
    assert run_sanitizers_module._env_var_for_sanitizer("asan") == run_sanitizers_module.ASAN_ENV
    assert run_sanitizers_module._env_var_for_sanitizer("tsan") == run_sanitizers_module.TSAN_ENV
    assert run_sanitizers_module._env_var_for_sanitizer("ubsan") == run_sanitizers_module.UBSAN_ENV
    assert run_sanitizers_module._env_var_for_sanitizer("unknown") is None


def test_select_parser_mappings(run_sanitizers_module: ModuleType) -> None:
    assert run_sanitizers_module._select_parser("asan") is run_sanitizers_module.parse_asan_output
    assert run_sanitizers_module._select_parser("tsan") is run_sanitizers_module.parse_tsan_output
    assert run_sanitizers_module._select_parser("ubsan") is run_sanitizers_module.parse_ubsan_output
    assert run_sanitizers_module._select_parser("missing") is None


def test_run_sanitizer_rejects_unknown_sanitizer(run_sanitizers_module: ModuleType) -> None:
    exit_code, output = run_sanitizers_module.run_sanitizer(
        build_dir=None,
        executable=Path("/bin/true"),
        sanitizer="unknown",
    )

    assert exit_code == 1
    assert "Unsupported sanitizer" in output


def test_parse_asan_output_extracts_fields(run_sanitizers_module: ModuleType) -> None:
    errors = run_sanitizers_module.parse_asan_output(ASAN_ERROR_OUTPUT)

    assert len(errors) == 1
    err = errors[0]
    assert "heap-use-after-free" in err.error_type
    assert err.location.endswith("foo.cpp:42:7")
    assert err.access_info and err.access_info.startswith("READ")
    assert any("#0" in frame for frame in err.stack_trace)


def test_parse_tsan_output_multiple_blocks(run_sanitizers_module: ModuleType) -> None:
    errors = run_sanitizers_module.parse_tsan_output(TSAN_ERROR_OUTPUT)

    assert len(errors) == 2
    assert "data race" in errors[0].error_type
    assert errors[0].location.endswith("tsan.cpp:10")
    assert errors[1].location.endswith("tsan2.cpp:20")


def test_parse_ubsan_output_with_stack(run_sanitizers_module: ModuleType) -> None:
    errors = run_sanitizers_module.parse_ubsan_output(UBSAN_ERROR_OUTPUT)

    assert len(errors) == 1
    assert errors[0].error_type.startswith("shift exponent")
    assert errors[0].location.endswith("ubsan.c:12:5")
    assert len(errors[0].stack_trace) == 1


def test_extract_location_parses_file_and_line(run_sanitizers_module: ModuleType) -> None:
    frame = "#3 0x0 in fn /src/path/file.cc:99:13"

    assert run_sanitizers_module._extract_location(frame) == "/src/path/file.cc:99:13"
    assert run_sanitizers_module._extract_location("no match here") == ""


def test_truncate_output_limits(run_sanitizers_module: ModuleType) -> None:
    long_lines = "\n".join(["x" * 200 for _ in range(600)])

    truncated_output, truncated, notice = run_sanitizers_module._truncate_output(long_lines)

    assert truncated is True
    assert "Output truncated to 500 lines" in notice
    assert "Output truncated to 48KB" in notice
    assert truncated_output.rstrip().endswith(notice)


def test_run_sanitizer_success_env_and_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    captured: List[str] = []
    captured_env: dict = {}

    def fake_run(cmd: List[str], env: dict, **_: object) -> DummyProcess:  # type: ignore[override]
        captured.extend(cmd)
        captured_env.update(env)
        return DummyProcess(stdout="clean", stderr="", returncode=0)

    monkeypatch.setenv(module.ASAN_ENV, "orig")
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code, summary = module.run_sanitizer(
        build_dir=build_dir,
        executable=Path("/bin/true"),
        sanitizer="asan",
        options="detect_leaks=0",
        extra_args=["--foo", "bar"],
    )

    assert exit_code == 0
    assert summary.startswith("=")
    assert captured[-2:] == ["--foo", "bar"]
    assert captured_env[module.ASAN_ENV] == "orig:detect_leaks=0"


def test_run_sanitizer_failure_parses_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()

    def fake_run(*_: object, **__: object) -> DummyProcess:  # type: ignore[override]
        return DummyProcess(stdout=ASAN_ERROR_OUTPUT, stderr="", returncode=1)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code, summary = module.run_sanitizer(
        build_dir=tmp_path,
        executable=Path("/bin/fake"),
        sanitizer="asan",
    )

    assert exit_code == 1
    assert "FAILED" in summary
    assert "heap-use-after-free" in summary
    assert "foo.cpp:42:7" in summary


def test_run_sanitizer_nonzero_exit_no_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()

    call_order = iter([10.0, 12.0])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(call_order))

    def fake_run(*_: object, **__: object) -> DummyProcess:  # type: ignore[override]
        return DummyProcess(stdout="clean output", stderr="", returncode=5)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code, summary = module.run_sanitizer(
        build_dir=tmp_path,
        executable=Path("/bin/true"),
        sanitizer="asan",
        normal_duration=1.0,
    )

    assert exit_code == 1
    assert "Exit Code: 5" in summary
    assert "FAILED" in summary
    assert "Overhead Ratio: 2.00x" in summary


def test_run_sanitizer_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()

    def raise_timeout(*_: object, **__: object) -> None:  # type: ignore[override]
        raise subprocess.TimeoutExpired(
            cmd=["/bin/slow"], timeout=1, output="partial", stderr="err"
        )

    monkeypatch.setattr(module.subprocess, "run", raise_timeout)

    exit_code, summary = module.run_sanitizer(
        build_dir=tmp_path,
        executable=Path("/bin/slow"),
        sanitizer="tsan",
        timeout=1,
    )

    assert exit_code == 1
    assert "TIMEOUT" in summary


def test_run_sanitizer_file_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()

    def raise_missing(*_: object, **__: object) -> None:  # type: ignore[override]
        raise FileNotFoundError()

    monkeypatch.setattr(module.subprocess, "run", raise_missing)

    exit_code, summary = module.run_sanitizer(
        build_dir=None,
        executable=Path("/missing"),
        sanitizer="ubsan",
    )

    assert exit_code == 1
    assert "Executable not found" in summary


def test_run_sanitizer_suppressions_and_json_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()

    suppressions_file = tmp_path / "suppressions.txt"
    suppressions_file.write_text("# comment\n\nheap-use-after-free\n")

    def fake_run(*_: object, **__: object) -> DummyProcess:  # type: ignore[override]
        return DummyProcess(stdout=ASAN_ERROR_OUTPUT, stderr="", returncode=1)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code, payload = module.run_sanitizer(
        build_dir=tmp_path,
        executable=Path("/bin/fake"),
        sanitizer="asan",
        suppressions=suppressions_file,
        output_mode="json",
    )

    data = json.loads(payload)
    assert exit_code == 1
    assert data["suppressed_count"] == 1
    assert data["errors"] == []
    assert data["timed_out"] is False


def test_load_suppressions_missing_file(run_sanitizers_module: ModuleType, tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.txt"

    assert run_sanitizers_module._load_suppressions(missing_path) == []


def test_run_sanitizer_full_output_truncation_notice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module()
    long_output = "\n".join([f"line {i}" for i in range(700)])

    def fake_run(*_: object, **__: object) -> DummyProcess:  # type: ignore[override]
        return DummyProcess(stdout=long_output, stderr="", returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code, output = module.run_sanitizer(
        build_dir=tmp_path,
        executable=Path("/bin/true"),
        sanitizer="ubsan",
        output_mode="full",
    )

    assert exit_code == 0
    assert "Output truncated" in output


def test_run_sanitizer_error_cap_note(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()

    def fake_parser(_: str):
        return [module.SanitizerError("err", "loc", None, []) for _ in range(60)]

    monkeypatch.setattr(module, "_select_parser", lambda __: fake_parser)

    exit_code, summary = module.run_sanitizer(
        build_dir=tmp_path,
        executable=Path("/bin/true"),
        sanitizer="asan",
        output_mode="summary",
    )

    assert exit_code == 1
    assert f"Errors capped at first {module.ERROR_LIMIT}" in summary


def test_cli_parses_and_writes_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_module()
    output_file = tmp_path / "out.txt"

    captured_args: dict = {}

    def fake_run_sanitizer(**kwargs: object) -> Tuple[int, str]:
        captured_args.update(kwargs)
        return 0, "ok"

    monkeypatch.setattr(module, "run_sanitizer", fake_run_sanitizer)

    with pytest.raises(SystemExit) as excinfo:
        module.main(
            [
                "--sanitizer",
                "asan",
                "--executable",
                str(Path("/bin/true")),
                "--output",
                str(output_file),
            ]
        )

    assert excinfo.value.code == 0
    assert output_file.read_text() == "ok"
    assert captured_args["sanitizer"] == "asan"


def test_cli_passes_passthrough_args(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()

    captured_args: dict = {}

    def fake_run_sanitizer(**kwargs: object) -> Tuple[int, str]:
        captured_args.update(kwargs)
        return 0, "ok"

    monkeypatch.setattr(module, "run_sanitizer", fake_run_sanitizer)

    with pytest.raises(SystemExit) as excinfo:
        module.main(
            [
                "--sanitizer",
                "tsan",
                "--executable",
                str(Path("/bin/true")),
                "--output-mode",
                "summary",
                "--",
                "--flag",
                "value",
            ]
        )

    assert excinfo.value.code == 0
    assert captured_args["sanitizer"] == "tsan"
    assert captured_args["extra_args"] == ["--flag", "value"]


def test_cli_invalid_sanitizer_choice() -> None:
    module = load_module()

    with pytest.raises(SystemExit):
        module._parse_args(["--sanitizer", "foo", "--executable", "/bin/true"])
