"""Run the CPU-focused 0.2.x release suite against an installed package.

By default, build a wheel and test it in a fresh virtual environment. Conda
recipes use ``--installed`` to validate their already-installed package instead.
Only test sources, pytest configuration, and three CPU examples enter the
isolated workspace. Review the GPU exclusions for v0.3.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

CPU_EXAMPLES = (
    "docs/Examples/cpu_dilution.py",
    "docs/Examples/Nucleation/cpu_nucleation.py",
    "docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py",
)
RELEASE_MARKERS = (
    "not slow and not performance and not benchmark "
    "and not warp and not cuda and not gpu_parity"
)
# These modules/subtrees include GPU helpers or eager Warp imports. Exclude
# them before import, not just after collection. Mixed CPU execution and
# integration suites remain selected and use per-test GPU markers.
RELEASE_IGNORES = (
    "particula/gpu",
    "particula/execution/tests/diagnostics_test.py",
    "particula/execution/tests/gpu_resources_test.py",
)


def stage_test_inputs(source: Path, destination: Path) -> None:
    """Copy explicit inputs without copying importable application sources."""
    for path in (source / "particula").rglob("*"):
        relative = path.relative_to(source)
        if not path.is_file() or "__pycache__" in relative.parts:
            continue
        if not {"tests", "integration_tests"}.intersection(relative.parts) and (
            path.name != "conftest.py"
        ):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for relative in ("conftest.py", "pyproject.toml", *CPU_EXAMPLES):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)


class Report:
    """Record selection counts and require scientific coverage domains."""

    def __init__(self):
        """Initialize independent counters for this pytest session."""
        self.counts = dict(
            collected=0, deselected=0, passed=0, skipped=0, failed=0, errors=0
        )

    def pytest_deselected(self, items):
        """Count marker-deselected test cases."""
        self.counts["deselected"] += len(items)

    def pytest_collection_finish(self, session):
        """Require retained tests in scientific and integration domains."""
        import pytest

        self.counts["collected"] = (
            len(session.items) + self.counts["deselected"]
        )
        if not session.items:
            raise pytest.UsageError("Release selection is empty")
        for domain in (
            "dynamics/condensation/",
            "dynamics/coagulation/",
            "dynamics/nucleation/",
            "gas/",
            "particles/",
            "integration_tests/",
            "execution/",
        ):
            if not any(domain in item.nodeid for item in session.items):
                raise pytest.UsageError(f"No release tests for {domain}")

    def pytest_sessionfinish(self, session, exitstatus):
        """Use pytest's final categories, including module-level skips."""
        reporter = session.config.pluginmanager.getplugin("terminalreporter")
        for key, category in (
            ("passed", "passed"),
            ("skipped", "skipped"),
            ("failed", "failed"),
            ("errors", "error"),
        ):
            self.counts[key] = len(reporter.stats.get(category, ()))
        print("\nRelease test counts: " + json.dumps(self.counts), flush=True)


def installed_test_main(source: Path, workspace: Path) -> int:
    """Validate package provenance, then collect and run the release suite."""
    import importlib.metadata

    import particula
    import pytest

    origin = Path(particula.__file__).resolve()
    distribution = importlib.metadata.distribution("particula")
    expected = Path(distribution.locate_file("particula/__init__.py")).resolve()
    if origin != expected or any(
        origin.is_relative_to(root.resolve()) for root in (source, workspace)
    ):
        raise RuntimeError(f"Release tests imported source checkout: {origin}")
    print(f"Installed package: {origin}", flush=True)
    print(f"Version: {particula.__version__}", flush=True)
    if particula.__version__ != distribution.version:
        raise RuntimeError("Imported version differs from installed metadata")
    subprocess.run(  # noqa: S603 - current interpreter, fixed pip command
        [sys.executable, "-m", "pip", "check"], check=True
    )

    # Configuration and warning bootstrap stay identical to source CI.
    return int(
        pytest.main(
            [
                "-c",
                str(workspace / "pyproject.toml"),
                # Give test modules an inputs.* namespace so pytest cannot
                # synthesize empty particula.* parents over installed modules.
                "--rootdir",
                str(workspace.parent),
                "--import-mode=importlib",
                "--strict-markers",
                "-Werror",
                "-m",
                RELEASE_MARKERS,
                *(f"--ignore={path}" for path in RELEASE_IGNORES),
                "particula",
            ],
            plugins=[Report()],
        )
    )


def run_installed(source: Path) -> int:
    """Launch an isolated interpreter outside the source and build trees."""
    with tempfile.TemporaryDirectory(prefix="particula-release-tests-") as temp:
        workspace = Path(temp) / "inputs"
        workspace.mkdir()
        stage_test_inputs(source, workspace)
        environment = os.environ.copy()
        for key in ("PYTHONPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
            environment.pop(key, None)
        environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        command = [
            sys.executable,
            "-I",
            str(Path(__file__).resolve()),
            "--test-workspace",
            str(workspace),
            "--source",
            str(source),
        ]
        artifacts = source / ".artifacts"
        artifacts.mkdir(exist_ok=True)
        log_path = artifacts / "release-tests.log"
        with log_path.open("w", encoding="utf-8") as log:
            with subprocess.Popen(  # noqa: S603 - fixed runner, no shell
                command,
                cwd=workspace,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            ) as process:
                if process.stdout is None:
                    raise RuntimeError("Release runner output pipe unavailable")
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                return process.wait()


def main() -> int:
    """Build a wheel environment or test an installed conda package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--installed", action="store_true")
    parser.add_argument("--test-workspace", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    source = args.source.resolve()
    if args.test_workspace:
        return installed_test_main(source, args.test_workspace)
    if args.installed:
        return run_installed(source)
    with tempfile.TemporaryDirectory(prefix="particula-release-wheel-") as temp:
        root = Path(temp)
        wheels = root / "wheels"
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        subprocess.run(  # noqa: S603 - disposable interpreter, fixed pip args
            [
                str(python),
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--wheel-dir",
                str(wheels),
                str(source),
            ],
            check=True,
        )
        (wheel,) = wheels.glob("particula-*.whl")
        subprocess.run(  # noqa: S603 - install the freshly built wheel
            [
                str(python),
                "-m",
                "pip",
                "install",
                str(wheel),
                "pytest",
            ],
            check=True,
        )
        return subprocess.run(  # noqa: S603 - fixed isolated runner
            [
                str(python),
                "-I",
                str(Path(__file__).resolve()),
                "--installed",
                "--source",
                str(source),
            ],
            check=False,
        ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
