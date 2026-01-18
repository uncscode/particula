"""Integration tests for run_cpp_linters using the example_cpp_dev project.

Prerequisites:
- example_cpp_dev present with CMakeLists.txt and CMakePresets.json
- CMake available to generate compile_commands.json for clang-tidy
- clang-format, clang-tidy, and cppcheck installed on PATH

The module mirrors the integration patterns used by run_ctest_integration_test.py.
It skips cleanly when prerequisites are missing and caches the C++ file list and
build artifacts to minimize repeated work across tests.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from types import ModuleType
from typing import Generator, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_CPP_DIR = REPO_ROOT / "example_cpp_dev"
SOURCE_DIR = EXAMPLE_CPP_DIR / "src"
BUILD_ROOT = EXAMPLE_CPP_DIR / "build"
RUN_CPP_LINTERS_PATH = (
    Path(__file__).resolve().parent.parent / "run_cpp_linters.py"
)


def _load_module(name: str, path: Path) -> ModuleType:
    """Load a module from a file path for integration testing."""
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def cmake_available() -> bool:
    """Return True when cmake is available on PATH."""
    return shutil.which("cmake") is not None


def clang_format_available() -> bool:
    """Return True when clang-format is available on PATH."""
    return shutil.which("clang-format") is not None


def clang_tidy_available() -> bool:
    """Return True when clang-tidy is available on PATH."""
    return shutil.which("clang-tidy") is not None


def cppcheck_available() -> bool:
    """Return True when cppcheck is available on PATH."""
    return shutil.which("cppcheck") is not None


PYTEST_MARKS = [
    pytest.mark.integration,
    pytest.mark.requires_cmake,
    pytest.mark.skipif(not cmake_available(), reason="CMake not installed"),
    pytest.mark.skipif(
        not clang_format_available(), reason="clang-format not installed"
    ),
    pytest.mark.skipif(
        not clang_tidy_available(), reason="clang-tidy not installed"
    ),
    pytest.mark.skipif(
        not cppcheck_available(), reason="cppcheck not installed"
    ),
]
pytestmark = PYTEST_MARKS


@pytest.fixture(scope="session")
def run_cpp_linters_module() -> ModuleType:
    """Load the run_cpp_linters module under test."""
    return _load_module("run_cpp_linters", RUN_CPP_LINTERS_PATH)


@pytest.fixture(scope="session")
def example_project() -> Path:
    """Provide the example_cpp_dev project root or skip when absent."""
    if not EXAMPLE_CPP_DIR.exists():
        pytest.skip("example_cpp_dev project not found (E12-F3 prerequisite)")
    if not (EXAMPLE_CPP_DIR / "CMakeLists.txt").exists():
        pytest.skip("example_cpp_dev/CMakeLists.txt missing")
    if not (EXAMPLE_CPP_DIR / "CMakePresets.json").exists():
        pytest.skip("CMakePresets.json missing for example_cpp_dev")
    return EXAMPLE_CPP_DIR


@pytest.fixture(scope="session")
def cpp_files(run_cpp_linters_module: ModuleType) -> list[Path]:
    """Resolve and cache the C/C++ sources under the example project."""
    return run_cpp_linters_module.get_cpp_files(str(SOURCE_DIR))


@pytest.fixture(scope="session")
def build_dir() -> Generator[Path, None, None]:
    """Create and clean a unique build directory for generating compile commands."""
    root = BUILD_ROOT / f"run_cpp_linters_{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="session")
def compile_commands_dir(
    example_project: Path,
    build_dir: Path,
    run_cpp_linters_module: ModuleType,
) -> Optional[Path]:
    """Configure and build once to produce compile_commands.json for clang-tidy.

    Returns the build directory when compile_commands.json is present; skips on
    cmake errors or timeouts and returns None when generation succeeds without
    producing compile commands.
    """
    if not cmake_available():
        pytest.skip("CMake not installed")

    timeout = run_cpp_linters_module.DEFAULT_TIMEOUTS["clang-tidy"]
    configure_cmd = [
        "cmake",
        "-S",
        str(example_project),
        "-B",
        str(build_dir),
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ]
    build_cmd = ["cmake", "--build", str(build_dir)]

    try:
        subprocess.run(
            configure_cmd,
            cwd=str(example_project),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        subprocess.run(
            build_cmd,
            cwd=str(example_project),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (
        subprocess.CalledProcessError
    ) as exc:  # pragma: no cover - integration
        output = exc.stderr or exc.stdout or str(exc)
        pytest.skip(
            f"Failed to configure/build example_cpp_dev: {output.strip()}"
        )
    except subprocess.TimeoutExpired:  # pragma: no cover - integration
        pytest.skip("cmake configure/build step timed out")
    except FileNotFoundError:  # pragma: no cover - integration
        pytest.skip("cmake executable missing")

    compile_commands = build_dir / "compile_commands.json"
    return build_dir if compile_commands.exists() else None


class TestClangFormatIntegration:
    """Validate clang-format runs in check and auto-fix modes."""

    def test_clang_format_check(
        self,
        run_cpp_linters_module: ModuleType,
        cpp_files: list[Path],
        tmp_path: Path,
    ) -> None:
        """Check mode reports success after formatting a temporary copy."""
        copied_files = []
        for path in cpp_files:
            target = tmp_path / path.name
            shutil.copy(path, target)
            copied_files.append(target)

        formatted = run_cpp_linters_module.run_clang_format(
            copied_files,
            auto_fix=True,
            timeout=run_cpp_linters_module.DEFAULT_TIMEOUTS["clang-format"],
        )
        check_result = run_cpp_linters_module.run_clang_format(
            copied_files,
            auto_fix=False,
            timeout=run_cpp_linters_module.DEFAULT_TIMEOUTS["clang-format"],
        )

        assert formatted.skipped is False
        assert check_result.skipped is False
        assert check_result.files_checked > 0
        assert check_result.success is True

    def test_clang_format_autofix(
        self, run_cpp_linters_module: ModuleType, tmp_path: Path
    ) -> None:
        """Auto-fix mode updates a misformatted file copy."""
        source_file = SOURCE_DIR / "example_lib.cpp"
        target = tmp_path / "example_lib.cpp"
        target.write_text(source_file.read_text())
        target.write_text(
            target.read_text() + "\nint   add(int a,int b){return a+b;}\n"
        )

        before = target.read_text()
        result = run_cpp_linters_module.run_clang_format(
            [target],
            auto_fix=True,
            timeout=run_cpp_linters_module.DEFAULT_TIMEOUTS["clang-format"],
        )
        after = target.read_text()

        assert result.skipped is False
        assert result.success is True
        assert after != before
        assert "add(int a, int b)" in after


class TestClangTidyIntegration:
    """Exercise clang-tidy with and without compile commands."""

    def test_clang_tidy_with_compile_commands(
        self,
        run_cpp_linters_module: ModuleType,
        cpp_files: list[Path],
        compile_commands_dir: Optional[Path],
    ) -> None:
        """Runs clang-tidy when compile_commands.json is available."""
        if compile_commands_dir is None:
            pytest.skip(
                "compile_commands.json not generated; skipping clang-tidy positive case"
            )

        result = run_cpp_linters_module.run_clang_tidy(
            cpp_files,
            build_dir=str(compile_commands_dir),
            auto_fix=False,
            timeout=run_cpp_linters_module.DEFAULT_TIMEOUTS["clang-tidy"],
        )

        assert result.skipped is False
        assert result.error_message is None
        assert result.success is True

    def test_clang_tidy_missing_compile_commands(
        self,
        run_cpp_linters_module: ModuleType,
        cpp_files: list[Path],
        tmp_path: Path,
    ) -> None:
        """Reports a clear error when compile_commands.json is absent."""
        missing = tmp_path / "missing_build"
        result = run_cpp_linters_module.run_clang_tidy(
            cpp_files,
            build_dir=str(missing),
            auto_fix=False,
            timeout=run_cpp_linters_module.DEFAULT_TIMEOUTS["clang-tidy"],
        )

        assert result.success is False
        assert result.skipped is False
        assert "compile_commands.json" in (result.error_message or "")


class TestCppcheckIntegration:
    """Validate cppcheck execution path."""

    def test_cppcheck_analysis(
        self, run_cpp_linters_module: ModuleType, cpp_files: list[Path]
    ) -> None:
        """Runs cppcheck and surfaces errors via result flags."""
        result = run_cpp_linters_module.run_cppcheck(
            cpp_files,
            timeout=run_cpp_linters_module.DEFAULT_TIMEOUTS["cppcheck"],
        )

        if result.skipped:
            pytest.skip(result.error_message or "cppcheck not available")

        assert result.error_message is None
        if result.exit_code == 0:
            assert result.success is True
        else:
            assert result.success is False


class TestCombinedLinters:
    """Cover combined linters and output modes."""

    def test_all_linters_summary(
        self,
        run_cpp_linters_module: ModuleType,
        compile_commands_dir: Optional[Path],
    ) -> None:
        """Summary output includes per-linter sections and validation banner."""
        linters = ["clang-format", "cppcheck"]
        build_dir = None
        if compile_commands_dir is not None:
            linters.append("clang-tidy")
            build_dir = str(compile_commands_dir)

        exit_code, output = run_cpp_linters_module.run_cpp_linters(
            source_dir=str(SOURCE_DIR),
            build_dir=build_dir,
            linters=linters,
            output_mode="summary",
        )

        assert "C++ LINTERS SUMMARY" in output
        for linter in linters:
            assert linter in output
        assert exit_code in {0, 1}
        if exit_code == 1:
            assert "VALIDATION: FAILED" in output

    def test_json_output_mode(self, run_cpp_linters_module: ModuleType) -> None:
        """JSON output parses and exposes expected keys."""
        exit_code, output = run_cpp_linters_module.run_cpp_linters(
            source_dir=str(SOURCE_DIR),
            build_dir=None,
            linters=["clang-format"],
            output_mode="json",
        )

        payload = json.loads(output)

        assert exit_code in {0, 1}
        assert "results" in payload
        assert "all_skipped" in payload

    def test_full_output_mode(self, run_cpp_linters_module: ModuleType) -> None:
        """Full output includes the summary banner and raw linter sections."""
        exit_code, output = run_cpp_linters_module.run_cpp_linters(
            source_dir=str(SOURCE_DIR),
            build_dir=None,
            linters=["clang-format"],
            output_mode="full",
        )

        assert exit_code in {0, 1}
        assert "C++ LINTERS SUMMARY" in output
        assert "clang-format" in output


class TestErrorHandling:
    """Edge-case handling for missing linters and empty projects."""

    def test_missing_linter_skipped(
        self,
        run_cpp_linters_module: ModuleType,
        cpp_files: list[Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Simulate clang-format missing and ensure skipped flag is set."""
        monkeypatch.setattr(
            run_cpp_linters_module, "check_linter_available", lambda *_: False
        )

        result = run_cpp_linters_module.run_clang_format(
            cpp_files,
            auto_fix=False,
            timeout=run_cpp_linters_module.DEFAULT_TIMEOUTS["clang-format"],
        )

        assert result.skipped is True
        assert "not found" in (result.error_message or "").lower()

    def test_no_cpp_files(
        self, run_cpp_linters_module: ModuleType, tmp_path: Path
    ) -> None:
        """Running without any C++ files surfaces validation errors."""
        exit_code, output = run_cpp_linters_module.run_cpp_linters(
            source_dir=str(tmp_path),
            build_dir=None,
            linters=["clang-format"],
            output_mode="summary",
        )

        assert exit_code == 1
        assert "No C++ files found" in output
        assert "VALIDATION: FAILED" in output


class TestHelperUtilities:
    """Unit-level coverage for helper functions and parsing."""

    def test_truncate_output_applies_limits(
        self, run_cpp_linters_module: ModuleType
    ) -> None:
        """Large output is truncated with notices for lines and bytes."""
        long_text = "\n".join(
            "line" for _ in range(run_cpp_linters_module.OUTPUT_LINE_LIMIT + 10)
        )
        expanded = long_text + (
            "x" * (run_cpp_linters_module.OUTPUT_BYTE_LIMIT)
        )

        truncated, was_truncated, notice = (
            run_cpp_linters_module._truncate_output(expanded)
        )

        assert was_truncated is True
        assert "Output truncated" in notice
        assert truncated.endswith(notice)

    def test_run_subprocess_handles_missing_command(
        self, run_cpp_linters_module: ModuleType
    ) -> None:
        """Command-not-found errors are surfaced gracefully."""
        exit_code, stdout, stderr, timed_out, error_message = (
            run_cpp_linters_module._run_subprocess(
                ["definitely_missing_command"], timeout=1
            )
        )

        assert exit_code == 1
        assert timed_out is False
        assert error_message and "Command not found" in error_message
        assert stdout == ""
        assert stderr == ""

    def test_parse_linters_arg_defaults_and_values(
        self, run_cpp_linters_module: ModuleType
    ) -> None:
        """Comma-separated linters are normalized and defaults are applied when empty."""
        assert run_cpp_linters_module.parse_linters_arg("") == [
            "clang-format",
            "clang-tidy",
            "cppcheck",
        ]
        assert run_cpp_linters_module.parse_linters_arg(
            " clang-format , cppcheck "
        ) == [
            "clang-format",
            "cppcheck",
        ]


class TestRunClangFormatUnit:
    """Unit coverage for clang-format error handling branches."""

    def test_clang_format_no_files(
        self,
        run_cpp_linters_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reports an error when invoked without any files."""
        monkeypatch.setattr(
            run_cpp_linters_module, "check_linter_available", lambda *_: True
        )

        result = run_cpp_linters_module.run_clang_format(
            [], auto_fix=False, timeout=1
        )

        assert result.success is False
        assert result.error_message == "No C++ files found to format"
        assert result.files_checked == 0

    def test_clang_format_failure_records_issues(
        self,
        run_cpp_linters_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-zero exit codes capture issues and affected files."""
        monkeypatch.setattr(
            run_cpp_linters_module, "check_linter_available", lambda *_: True
        )
        monkeypatch.setattr(
            run_cpp_linters_module,
            "_run_subprocess",
            lambda *_args, **_kwargs: (
                1,
                "/tmp/foo.cpp: needs format",
                "",
                False,
                None,
            ),
        )

        result = run_cpp_linters_module.run_clang_format(
            [Path("/tmp/foo.cpp")], auto_fix=False, timeout=1
        )

        assert result.success is False
        assert result.files_with_issues >= 1
        assert any("foo.cpp" in issue for issue in result.issues)


class TestRunClangTidyUnit:
    """Unit coverage for clang-tidy validation paths."""

    def test_clang_tidy_requires_build_dir(
        self,
        run_cpp_linters_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Missing build_dir surfaces a clear error message."""
        monkeypatch.setattr(
            run_cpp_linters_module, "check_linter_available", lambda *_: True
        )

        result = run_cpp_linters_module.run_clang_tidy(
            [Path("foo.cpp")], build_dir=None, auto_fix=False, timeout=1
        )

        assert result.success is False
        assert "build-dir" in (result.error_message or "")

    def test_clang_tidy_empty_files(
        self,
        run_cpp_linters_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Empty file lists report validation errors before invocation."""
        monkeypatch.setattr(
            run_cpp_linters_module, "check_linter_available", lambda *_: True
        )
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        (build_dir / "compile_commands.json").write_text("{}")

        result = run_cpp_linters_module.run_clang_tidy(
            [], build_dir=str(build_dir), auto_fix=False, timeout=1
        )

        assert result.success is False
        assert "No C++ files found" in (result.error_message or "")


class TestRunCppcheckUnit:
    """Unit coverage for cppcheck validation paths."""

    def test_cppcheck_no_files(
        self,
        run_cpp_linters_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reports an error when no source files are provided."""
        monkeypatch.setattr(
            run_cpp_linters_module, "check_linter_available", lambda *_: True
        )

        result = run_cpp_linters_module.run_cppcheck([], timeout=1)

        assert result.success is False
        assert result.error_message == "No C++ files found to analyze"


class TestFormatSummary:
    """Exercise summary formatting across result states."""

    def test_summary_captures_statuses(
        self, run_cpp_linters_module: ModuleType
    ) -> None:
        """Statuses include skipped, failed, warnings, and passed variants."""
        results = [
            run_cpp_linters_module.LinterResult(
                "clang-format", skipped=True, files_checked=0
            ),
            run_cpp_linters_module.LinterResult(
                "clang-tidy",
                success=False,
                errors=1,
                files_checked=2,
                files_with_issues=1,
            ),
            run_cpp_linters_module.LinterResult(
                "cppcheck",
                success=True,
                warnings=1,
                files_checked=1,
                files_with_issues=0,
            ),
            run_cpp_linters_module.LinterResult(
                "extra", success=True, files_checked=1, files_with_issues=0
            ),
        ]

        summary = run_cpp_linters_module.format_summary(
            results, duration=1.0, all_skipped=False
        )

        assert "SKIPPED" in summary
        assert "FAILED" in summary
        assert "WARNINGS" in summary
        assert "PASSED" in summary
        assert "VALIDATION: FAILED" in summary

    def test_summary_all_skipped(
        self, run_cpp_linters_module: ModuleType
    ) -> None:
        """All-skipped runs mark validation failed with explicit messaging."""
        results = [
            run_cpp_linters_module.LinterResult("clang-format", skipped=True)
        ]

        summary = run_cpp_linters_module.format_summary(
            results, duration=0.1, all_skipped=True
        )

        assert "FAILED (all linters skipped)" in summary


class TestRunCppLintersControlFlow:
    """Validate top-level runner behaviors without invoking real tools."""

    def test_run_cpp_linters_invalid_output_mode(
        self,
        run_cpp_linters_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unsupported output modes raise ValueError before returning output."""
        monkeypatch.setattr(
            run_cpp_linters_module,
            "get_cpp_files",
            lambda _dir: [Path("foo.cpp")],
        )
        monkeypatch.setattr(
            run_cpp_linters_module,
            "run_clang_format",
            lambda *_args, **_kwargs: run_cpp_linters_module.LinterResult(
                "clang-format"
            ),
        )
        monkeypatch.setattr(
            run_cpp_linters_module,
            "run_clang_tidy",
            lambda *_args, **_kwargs: run_cpp_linters_module.LinterResult(
                "clang-tidy"
            ),
        )
        monkeypatch.setattr(
            run_cpp_linters_module,
            "run_cppcheck",
            lambda *_args, **_kwargs: run_cpp_linters_module.LinterResult(
                "cppcheck"
            ),
        )

        with pytest.raises(ValueError):
            run_cpp_linters_module.run_cpp_linters(
                source_dir="src",
                build_dir=None,
                linters=["clang-format"],
                output_mode="invalid",
            )

    def test_run_cpp_linters_defaults(
        self,
        run_cpp_linters_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Empty linter lists fall back to all linters and keep validation passing."""
        calls: list[str] = []

        monkeypatch.setattr(
            run_cpp_linters_module,
            "get_cpp_files",
            lambda _dir: [Path("foo.cpp")],
        )

        def record_clang_format(*_args, **_kwargs):
            calls.append("clang-format")
            return run_cpp_linters_module.LinterResult("clang-format")

        def record_clang_tidy(*_args, **_kwargs):
            calls.append("clang-tidy")
            return run_cpp_linters_module.LinterResult("clang-tidy")

        def record_cppcheck(*_args, **_kwargs):
            calls.append("cppcheck")
            return run_cpp_linters_module.LinterResult("cppcheck")

        monkeypatch.setattr(
            run_cpp_linters_module, "run_clang_format", record_clang_format
        )
        monkeypatch.setattr(
            run_cpp_linters_module, "run_clang_tidy", record_clang_tidy
        )
        monkeypatch.setattr(
            run_cpp_linters_module, "run_cppcheck", record_cppcheck
        )

        exit_code, output = run_cpp_linters_module.run_cpp_linters(
            source_dir="src", build_dir=None, linters=[], output_mode="summary"
        )

        assert exit_code == 0
        assert set(calls) == {"clang-format", "clang-tidy", "cppcheck"}
        assert "VALIDATION: PASSED" in output
