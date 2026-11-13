"""Verify built documentation and an offline sdist-to-wheel round trip.

Requires build and flit_core in this interpreter. Installs wheels without
runtime dependencies into disposable virtual environments to prove the locator
works without loading simulation code. Never downloads dependencies.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
import venv
import zipfile
from email.parser import BytesParser
from pathlib import Path

DOC_PREFIX = "particula/docs/"
REPRESENTATIVES = (
    "guide/data-model/index.md",
    "theory/index.md",
    "contribute/CONTRIBUTING.md",
    "examples/fundamentals/native_aerosol.py",
    "data/gpu_benchmark_results.json",
    "images/icon_dark.svg",
)
# Independent inclusion policy: do not validate the backend using its selector.
SECTIONS = {"guide", "theory", "contribute", "examples", "data", "images"}
EXCLUDED = {
    "__pycache__",
    "API",
    "api",
    "node_modules",
    "build",
    "dist",
    "site",
}
OFFLINE_WRAPPER = """
import runpy, sys

def no_network_or_process(event, args):
    if event in {
        "socket.connect", "socket.getaddrinfo", "subprocess.Popen", "os.system"
    }:
        raise RuntimeError("Offline build attempted external access: " + event)

sys.addaudithook(no_network_or_process)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def hashes(source: Path) -> dict:
    """Compute source hashes independently of bundle metadata."""
    result = {}
    for path in sorted((source / "docs").rglob("*")):
        relative = path.relative_to(source / "docs")
        if (
            relative.parts[0] not in SECTIONS
            and relative.as_posix() != "index.md"
        ):
            continue
        if any(p.startswith(".") or p in EXCLUDED for p in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo", ".bak", ".tmp", ".log"}:
            continue
        if path.is_symlink():
            raise ValueError(f"Unexpected source symlink: {relative}")
        if path.is_file():
            data = path.read_bytes()
            result[relative.as_posix()] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
    if not result:
        raise ValueError("Empty documentation source inventory")
    return result


def verify_wheel(wheel: Path, expected: dict) -> dict:
    """Verify doc bytes, RECORD integrity and package-version match."""
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Wheel contains duplicate paths")
        actual = {
            name.removeprefix(DOC_PREFIX)
            for name in names
            if name.startswith(DOC_PREFIX)
        }
        if actual != {*expected, "bundle-manifest.json"}:
            raise ValueError(
                "Wheel documentation inventory differs from sources"
            )
        manifest = json.loads(archive.read(DOC_PREFIX + "bundle-manifest.json"))
        if manifest["files"] != expected:
            raise ValueError(
                "Documentation manifest differs from source hashes"
            )
        for name, record in expected.items():
            content = archive.read(DOC_PREFIX + name)
            if (
                len(content) != record["size_bytes"]
                or hashlib.sha256(content).hexdigest() != record["sha256"]
            ):
                raise ValueError(f"Bundled documentation differs: {name}")
        metadata_name = next(
            n for n in names if n.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        if manifest["version"] != metadata["Version"]:
            raise ValueError("Documentation and package versions differ")
        if manifest["size_bytes"] != sum(
            r["size_bytes"] for r in expected.values()
        ):
            raise ValueError("Documentation byte count differs")
        verify_record(archive)
        if "particula_docs.py" not in names:
            raise ValueError("Missing lightweight documentation locator")
    return manifest


def verify_record(archive: zipfile.ZipFile) -> None:
    """Validate the full wheel RECORD including the added locator module."""
    names = archive.namelist()
    record_name = next(n for n in names if n.endswith(".dist-info/RECORD"))
    records = list(csv.reader(io.StringIO(archive.read(record_name).decode())))
    if {r[0] for r in records} != set(names) or len(records) != len(names):
        raise ValueError("Wheel RECORD inventory differs")
    for name, digest, size in records:
        if name == record_name:
            continue
        content = archive.read(name)
        actual_digest = (
            base64.urlsafe_b64encode(hashlib.sha256(content).digest())
            .rstrip(b"=")
            .decode()
        )
        if digest != f"sha256={actual_digest}" or size != str(len(content)):
            raise ValueError(f"Invalid wheel RECORD: {name}")


def environment_python(environment: Path) -> Path:
    """Locate a virtual environment's interpreter on supported platforms."""
    return environment / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )


def install_and_check(wheel: Path, root: Path, expected: dict) -> dict:
    """Install without dependencies; check discovery and search."""
    environment = root / "venv"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment_python(environment)
    subprocess.run(  # noqa: S603 -- local wheel in disposable environment
        [
            str(python),
            "-I",
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            str(wheel),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    command = environment / (
        "Scripts/particula-docs.exe"
        if os.name == "nt"
        else "bin/particula-docs"
    )
    result = subprocess.run(  # noqa: S603 -- installed console entry point
        [str(command), "--json"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(result.stdout)
    docs = Path(info["path"])
    if not docs.is_relative_to(environment) or info["mode"] != "bundled":
        raise ValueError("Locator resolved outside the installed wheel")
    for relative in REPRESENTATIVES:
        if not (docs / relative).is_file():
            raise ValueError(
                f"Missing representative documentation: {relative}"
            )
    # Search actual installed text and follow an unchanged relative guide link.
    if not any(
        "ParticleData" in p.read_text() for p in (docs / "guide").rglob("*.md")
    ):
        raise ValueError("Installed guides are not searchable")
    guide = docs / "guide/migration/v0.3.0/index.md"
    if (
        "../../data-model/ownership.md" not in guide.read_text()
        or not (guide.parent / "../../data-model/ownership.md")
        .resolve()
        .is_file()
    ):
        raise ValueError("Bundled relative guide link is broken")
    for relative, record in expected.items():
        if (
            hashlib.sha256((docs / relative).read_bytes()).hexdigest()
            != record["sha256"]
        ):
            raise ValueError(f"Installed documentation changed: {relative}")
    probe = subprocess.run(  # noqa: S603 -- isolated import probe
        [
            str(python),
            "-I",
            "-c",
            "import sys, particula_docs; particula_docs.documentation_info(); "
            "assert not {'particula', 'numpy', 'scipy', 'warp'} "
            "& sys.modules.keys()",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    module = subprocess.run(  # noqa: S603 -- alternative module entry point
        [str(python), "-I", "-m", "particula_docs"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if module.stdout.strip() != str(docs) or probe.stderr:
        raise ValueError("Locator entry points disagree")
    return info


def offline_runner(command, cwd=None, extra_environ=None):
    """Run backend hooks without network access or subprocesses."""
    environment = {**os.environ, **(extra_environ or {})}
    environment.update(PATH="", PIP_NO_INDEX="1", PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(  # noqa: S603 -- build frontend's Python hook, audit guarded
        [command[0], "-c", OFFLINE_WRAPPER, *command[1:]],
        cwd=cwd,
        env=environment,
        check=True,
    )


def check(wheel: Path, sdist: Path, source: Path) -> dict:
    """Verify published artifacts and an offline, Git-free rebuilt wheel."""
    import build

    expected = hashes(source)
    manifest = verify_wheel(wheel, expected)
    with tempfile.TemporaryDirectory(prefix="particula-docs-check-") as temp:
        root = Path(temp)
        extracted = root / "source"
        extracted.mkdir()
        with tarfile.open(sdist) as archive:
            archive.extractall(extracted, filter="data")
        (project,) = extracted.iterdir()
        actual_docs = {
            p.relative_to(project / "docs").as_posix()
            for p in (project / "docs").rglob("*")
            if p.is_file()
        }
        if (
            (project / ".git").exists()
            or actual_docs != set(expected)
            or hashes(project) != expected
        ):
            raise ValueError(
                "Source archive differs from documentation sources"
            )
        for name in ("_particula_build.py", "particula_docs.py"):
            if not (project / name).is_file():
                raise ValueError(f"Source archive lacks build input: {name}")
        before = {
            str(p.relative_to(project)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in project.rglob("*")
            if p.is_file()
        }
        rebuilt = Path(
            build.ProjectBuilder(project, runner=offline_runner).build(
                "wheel", root / "rebuilt"
            )
        )
        after = {
            str(p.relative_to(project)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in project.rglob("*")
            if p.is_file()
        }
        if before != after:
            raise ValueError("Building a wheel changed the source archive tree")
        if verify_wheel(rebuilt, expected) != manifest:
            raise ValueError("Direct and sdist-built documentation differ")
        for name, candidate in (("direct", wheel), ("sdist", rebuilt)):
            installation = root / name
            installation.mkdir()
            info = install_and_check(candidate, installation, expected)
            if info["version"] != manifest["version"]:
                raise ValueError("Installed locator version mismatch")
    return {
        "version": manifest["version"],
        "file_count": len(expected),
        "documentation_size_bytes": manifest["size_bytes"],
        "wheel_size_bytes": wheel.stat().st_size,
        "sdist_size_bytes": sdist.stat().st_size,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "sdist_sha256": hashlib.sha256(sdist.read_bytes()).hexdigest(),
        "offline_sdist_rebuild": "passed",
        "isolated_installations": "passed",
    }


def main() -> None:
    """Check release files before upload and optionally retain a report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = check(
        args.wheel.resolve(), args.sdist.resolve(), args.source.resolve()
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)


if __name__ == "__main__":
    main()
