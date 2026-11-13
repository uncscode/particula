"""Build and install real documentation bundles outside the source checkout."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
import venv
import zipfile
from pathlib import Path

import build
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "check_doc_bundle", ROOT / "scripts/check_doc_bundle.py"
)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)
EXCLUDED = (
    "docs/API/generated.md",
    "docs/guide/api/generated.md",
    "docs/guide/build/generated.html",
    "docs/theory/site/index.html",
    "docs/examples/__pycache__/example.pyc",
    "docs/examples/backup.py.bak",
    "docs/data/cache.pyo",
    "docs/data/temporary.tmp",
    "docs/data/run.log",
    "docs/images/.cache/nested/sentinel.svg",
    "docs/.assets/temp/sentinel.md",
    "website/dist/index.html",
    ".trash/docs/retired.md",
)


def run(command, cwd):
    """Run an actual frontend or installed interpreter with isolated inputs."""
    return subprocess.run(  # noqa: S603 -- controlled local test commands
        [str(arg) for arg in command],
        cwd=cwd,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PIP_NO_INDEX": "1"},
        check=True,
        text=True,
        capture_output=True,
    )


def snapshot(root):
    """Record all source bytes, including sentinels and preexisting files."""
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    """Exercise real sources with excluded content and additional valid assets."""
    root = tmp_path_factory.mktemp("doc-bundle")
    bootstrap = Path(
        build.ProjectBuilder(ROOT, runner=CHECK.offline_runner).build(
            "sdist", root / "bootstrap"
        )
    )
    with tarfile.open(bootstrap) as archive:
        archive.extractall(root / "source", filter="data")
    (source,) = (root / "source").iterdir()
    for relative in EXCLUDED:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("EXCLUDED-PACKAGING-SENTINEL")
    data = source / "docs/data/nested/bundle_probe.csv"
    data.parent.mkdir(parents=True)
    data.write_text("diameter_m,number_m3\n1e-7,1e6\n")
    guide = source / "docs/guide/bundle_probe.md"
    guide.write_text("[Example data](../data/nested/bundle_probe.csv)\n")
    before = snapshot(source)
    # Use pip's normal wheel path, with already-installed build prerequisites.
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "--no-cache-dir",
            "--wheel-dir",
            root / "wheels",
            source,
        ],
        cwd=root,
    )
    (wheel,) = (root / "wheels").glob("*.whl")
    sdist = Path(
        build.ProjectBuilder(source, runner=CHECK.offline_runner).build(
            "sdist", root / "sdist"
        )
    )
    assert snapshot(source) == before
    assert not (source / "particula/docs").exists()
    return root, source, wheel, sdist


def test_wheel_and_offline_sdist_installations(artifacts):
    """Verify all hashes, isolation, version, search, data, links and exclusions."""
    _, source, wheel, sdist = artifacts
    report = CHECK.check(wheel, sdist, source)
    assert report["offline_sdist_rebuild"] == "passed"
    assert report["isolated_installations"] == "passed"
    assert report["documentation_size_bytes"] > 0
    expected = CHECK.hashes(source)
    assert report["file_count"] == len(expected)
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for relative in EXCLUDED:
            assert not any(name.endswith(relative) for name in names)
        assert not any(
            b"EXCLUDED-PACKAGING-SENTINEL" in archive.read(name)
            for name in names
        )
        assert (
            archive.read("particula/docs/data/nested/bundle_probe.csv")
            == (source / "docs/data/nested/bundle_probe.csv").read_bytes()
        )
        assert (
            archive.read("particula/docs/guide/bundle_probe.md")
            == (source / "docs/guide/bundle_probe.md").read_bytes()
        )


def test_editable_resolves_live_docs_without_simulation_imports(artifacts):
    """A real editable install sees maintained source edits immediately."""
    root, source, _, _ = artifacts
    before = snapshot(source)
    wheel = build.ProjectBuilder(source, runner=CHECK.offline_runner).build(
        "editable", root / "editable-wheel"
    )
    environment = root / "editable-env"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = CHECK.environment_python(environment)
    run([python, "-I", "-B", "-m", "pip", "install", "--no-deps", wheel], root)
    result = run([python, "-I", "-B", "-m", "particula_docs", "--json"], root)
    info = json.loads(result.stdout)
    assert info["mode"] == "editable"
    assert Path(info["path"]) == source / "docs"
    run(
        [
            python,
            "-I",
            "-B",
            "-c",
            "import sys, particula_docs; particula_docs.documentation_info(); "
            "assert not {'particula', 'numpy', 'scipy', 'warp'} & sys.modules.keys()",
        ],
        root,
    )
    assert snapshot(source) == before
    probe = source / "docs/guide/live-edit.md"
    try:
        probe.write_text("Live editable documentation")
        updated = json.loads(
            run(
                [python, "-I", "-B", "-m", "particula_docs", "--json"], root
            ).stdout
        )
        assert updated["file_count"] == info["file_count"] + 1
    finally:
        probe.unlink()


@pytest.mark.parametrize("target", ["guide/index.md", "bundle-manifest.json"])
def test_corrupt_bundles_fail_verification(artifacts, tmp_path, target):
    """Publication checks reject modified docs and mismatched version metadata."""
    _, source, wheel, _ = artifacts
    corrupted = tmp_path / wheel.name
    with (
        zipfile.ZipFile(wheel) as original,
        zipfile.ZipFile(corrupted, "w") as out,
    ):
        for entry in original.infolist():
            content = original.read(entry.filename)
            if entry.filename == CHECK.DOC_PREFIX + target:
                if target.endswith(".json"):
                    manifest = json.loads(content)
                    manifest["version"] = "0.0.0"
                    content = json.dumps(manifest).encode()
                else:
                    content += b"corrupted"
            out.writestr(entry, content)
    with pytest.raises(ValueError, match="differs|versions differ"):
        CHECK.verify_wheel(corrupted, CHECK.hashes(source))


@pytest.mark.parametrize(
    "operation",
    [
        "import socket; socket.getaddrinfo('example.com', 443)",
        "import subprocess; subprocess.run(['git', '--version'])",
    ],
)
def test_offline_runner_blocks_external_access(tmp_path, operation):
    """The source-archive check really disallows network and Git invocations."""
    probe = tmp_path / "probe.py"
    probe.write_text(operation)
    with pytest.raises(subprocess.CalledProcessError):
        CHECK.offline_runner([sys.executable, str(probe)], cwd=tmp_path)


def test_publication_requires_bundle_verification():
    """Both integration tests and exact-artifact checks gate the upload step."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/pypi.yml").read_text())
    steps = workflow["jobs"]["pyping"]["steps"]
    publish = next(
        i for i, s in enumerate(steps) if "pypi-publish" in s.get("uses", "")
    )
    for command in (
        "scripts/tests/doc_bundle_test.py",
        "scripts/check_doc_bundle.py",
    ):
        step = next(s for s in steps[:publish] if command in s.get("run", ""))
        assert not step.get("continue-on-error")
        assert "if" not in step
