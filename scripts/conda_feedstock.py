"""Build and test the current source with the mirrored conda-feedstock recipe.

Requires conda and conda-build. The default invocation builds a clean staging
tree and saves the literal build/test log and package under .artifacts/.
``--check-version BASE HEAD`` emits a GitHub Actions output after comparing
literal version assignments without importing either revision's package.
"""

from __future__ import annotations

import argparse
import ast
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from run_release_tests import CPU_EXAMPLES

ROOT = Path(__file__).resolve().parents[1]
VERSION_PATH = "particula/__init__.py"


def read_version(source: str) -> str:
    """Read the single literal version assignment without executing source."""
    values = [
        node.value
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        )
    ]
    if len(values) != 1:
        raise ValueError("Expected exactly one __version__ assignment")
    value = ast.literal_eval(values[0])
    if not isinstance(value, str) or not value.strip():
        raise ValueError("__version__ must be a nonempty string literal")
    return value


def version_changed(base: str, head: str) -> bool:
    """Compare versions, ignoring other edits to the package initializer."""
    return read_version(base) != read_version(head)


def check_version(base: str, head: str) -> bool:
    """Compare PR revisions from Git and emit the build-gate output."""
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Git is required for the PR version comparison")
    # Match GitHub's three-dot PR diff: unrelated version changes on the base
    # branch must not make an unchanged PR version look like a release change.
    ancestor = subprocess.run(  # noqa: S603 - fixed Git subcommand
        [git, "merge-base", base, head],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    sources = [
        subprocess.run(  # noqa: S603 - git show, no shell
            [git, "show", f"{revision}:{VERSION_PATH}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for revision in (ancestor, head)
    ]
    changed = version_changed(*sources)
    print(f"Version: {read_version(sources[0])} -> {read_version(sources[1])}")
    output = f"changed={str(changed).lower()}\n"
    print(output, end="")
    if os.environ.get("GITHUB_OUTPUT"):
        output_path = Path(os.environ["GITHUB_OUTPUT"])
        with output_path.open("a", encoding="utf-8") as file:
            file.write(output)
    return changed


def stage_source(source: Path, destination: Path) -> None:
    """Create the recipe source without Git, documentation prose, or plans."""
    shutil.copytree(
        source / "particula",
        destination / "particula",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )
    for relative in (
        "pyproject.toml",
        "readme.md",
        "license",
        "conftest.py",
        "scripts/run_release_tests.py",
        *CPU_EXAMPLES,
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)


def build() -> int:
    """Run the real conda package build/test and retain its output."""
    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError(
            "conda is unavailable; install conda-build in a conda environment "
            "or dispatch the conda-feedstock GitHub Actions workflow."
        )
    artifacts = ROOT / ".artifacts" / "conda-feedstock"
    artifacts.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PARTICULA_VERSION"] = read_version(
        (ROOT / VERSION_PATH).read_text(encoding="utf-8")
    )
    with tempfile.TemporaryDirectory(prefix="particula-conda-") as temp:
        staging = Path(temp)
        stage_source(ROOT, staging / "source")
        shutil.copytree(ROOT / "conda/recipe", staging / "recipe")
        command = [
            conda,
            "build",
            str(staging / "recipe"),
            "--override-channels",
            "--channel",
            "conda-forge",
            "--output-folder",
            str(artifacts / "packages"),
        ]
        with (artifacts / "build.log").open("w", encoding="utf-8") as log:
            log.write(f"Version: {environment['PARTICULA_VERSION']}\n")
            log.write(f"Command: {command!r}\n")
            log.flush()
            with subprocess.Popen(  # noqa: S603 - fixed conda-build command
                command,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            ) as process:
                if process.stdout is None:
                    raise RuntimeError("Conda output pipe unavailable")
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                return process.wait()


def main() -> int:
    """Select the cheap PR version gate or the actual conda build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-version", nargs=2, metavar=("BASE", "HEAD"))
    args = parser.parse_args()
    if args.check_version:
        check_version(*args.check_version)
        return 0
    return build()


if __name__ == "__main__":
    raise SystemExit(main())
