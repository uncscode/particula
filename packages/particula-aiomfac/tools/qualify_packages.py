"""Build both distributions and check three isolated wheel installations."""

import argparse
import email
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]


def check_archives(dist, backend):
    [wheel] = dist.glob("*.whl")
    [sdist] = dist.glob("*.tar.gz")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        [meta] = [p for p in names if p.endswith(".dist-info/METADATA")]
        metadata = email.message_from_bytes(archive.read(meta))
        wheel_files = set(names)
    with tarfile.open(sdist) as archive:
        source_files = {
            p.split("/", 1)[1] for p in archive.getnames() if "/" in p
        }
    project = "particula-aiomfac" if backend else "particula"
    assert metadata["Name"] == project
    stem = project.replace("-", "_") + "-" + metadata["Version"]
    assert (
        wheel.name == stem + "-py3-none-any.whl"
        and sdist.name == stem + ".tar.gz"
    )
    with tarfile.open(sdist) as archive:
        pkg = email.message_from_bytes(
            archive.extractfile(stem + "/PKG-INFO").read()
        )
        assert (pkg["Name"], pkg["Version"]) == (project, metadata["Version"])
    if backend:
        required = {
            "LICENSE",
            "NOTICE",
            "tools/reference.py",
            "tools/sparse_study.py",
            "tools/sparse_trajectories.py",
            "tools/export_parameters.py",
            "examples/closed_water.py",
            "src/particula_aiomfac/data/parameters.npz",
            "src/particula_aiomfac/data/manifest.json",
            "reference/upstream.toml",
            "reference/cases/mixed.json",
            "reference/expected/mixed.json",
            "tests/test_reference.py",
            "reference/README.md",
            "reference/parameters.toml",
        }
        assert required <= source_files, required - source_files
        assert all(
            p.startswith(("particula_aiomfac/", "particula_aiomfac-"))
            for p in wheel_files
        )
        assert all(
            not any(
                segment in {"reference", "tools", "tests"}
                for segment in Path(p).parts
            )
            for p in wheel_files
        )
        assert {
            "particula_aiomfac/data/parameters.npz",
            "particula_aiomfac/data/manifest.json",
        } <= wheel_files
        assert all(
            "particula_aiomfac/kernels/" + name + ".inc" in wheel_files
            for name in ("activity", "equilibrium", "output")
        )
        for recipe in (
            "mixed-water",
            "mixed-two-vapor",
            "mixed-variable-acid",
            "sulfate",
            "carbonate",
            "shared-ion",
            "neutral",
        ):
            item = f"particula_aiomfac/data/sparse/{recipe}.npz"
            assert item in wheel_files and "src/" + item in source_files
        for notice in ("LICENSE", "NOTICE"):
            assert any(
                p.endswith(".dist-info/licenses/" + notice) for p in wheel_files
            )
        assert metadata["License-Expression"] == "GPL-3.0-or-later"
        assert not any(
            p.endswith((".f90", ".out")) or ".artifacts/" in p
            for p in source_files | wheel_files
        )
    else:
        assert "aiomfac" in metadata.get_all("Provides-Extra", [])
        assert any(
            "particula-aiomfac" in r and "extra ==" in r
            for r in metadata.get_all("Requires-Dist", [])
        )
        # Compare payload digests too, including files renamed under bundled docs.

        protected = {
            hashlib.sha256(p.read_bytes()).digest()
            for p in PACKAGE.rglob("*")
            if p.is_file()
            and not p.name.startswith(".")
            and "__pycache__" not in p.parts
        }

        def covered(name):
            return any(
                s in name
                for s in (
                    "packages/particula-aiomfac/",
                    "particula_aiomfac/",
                    "aiomfac-reference/",
                )
            )

        assert not any(map(covered, source_files | wheel_files))
        with zipfile.ZipFile(wheel) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    assert (
                        hashlib.sha256(archive.read(info)).digest()
                        not in protected
                    ), info.filename
        with tarfile.open(sdist) as archive:
            for member in archive:
                if member.isfile():
                    assert (
                        hashlib.sha256(
                            archive.extractfile(member).read()
                        ).digest()
                        not in protected
                    ), member.name
    return wheel


def write_manifest(base, backend, destination):
    """Record scientific identity and digests from the actual wheel payload."""
    identity = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "distributions": {},
    }
    for wheel in (base, backend):
        with zipfile.ZipFile(wheel) as z:
            meta = email.message_from_bytes(
                z.read(
                    next(
                        n
                        for n in z.namelist()
                        if n.endswith(".dist-info/METADATA")
                    )
                )
            )
            if wheel == backend:
                data = "particula_aiomfac/data/"
                identity["upstream_revision"] = json.loads(
                    z.read(data + "manifest.json")
                )["revision"]
                identity["parameters_sha256"] = hashlib.sha256(
                    z.read(data + "parameters.npz")
                ).hexdigest()
        artifacts = [wheel, *wheel.parent.glob("*.tar.gz")]
        identity["distributions"][meta["Name"]] = {
            "version": meta["Version"],
            "sha256": {
                file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                for file in artifacts
            },
        }
    destination.write_text(json.dumps(identity, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-isolation",
        action="store_true",
        help="use already installed build and flit_core",
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        help="retain audited artifacts in base/ and backend/",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="exercise standalone and combined wheels on an NVIDIA GPU",
    )
    parser.add_argument(
        "--from-dist",
        type=Path,
        help="qualify existing base/ and backend/ archive pairs without rebuilding",
    )
    parser.add_argument(
        "--sdists",
        action="store_true",
        help="also install each source archive in isolation",
    )
    parser.add_argument(
        "--manifest", type=Path, help="write the audited release manifest"
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="stop after inspecting both archive pairs",
    )
    args = parser.parse_args()
    if args.from_dist and args.dist_dir:
        parser.error("--from-dist and --dist-dir are mutually exclusive")
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        wheels = []
        for name, project in (("base", ROOT), ("backend", PACKAGE)):
            if args.from_dist:
                wheels.append(
                    check_archives(
                        args.from_dist.resolve() / name, name == "backend"
                    )
                )
                continue
            dist = (args.dist_dir.resolve() if args.dist_dir else work) / name
            cmd = [sys.executable, "-m", "build", "--outdir", str(dist)]
            if args.no_isolation:
                cmd.append("--no-isolation")
            subprocess.run([*cmd, str(project)], cwd=work, check=True)
            wheels.append(check_archives(dist, name == "backend"))
        base, backend = wheels
        if args.manifest:
            write_manifest(base, backend, args.manifest)
        if args.audit_only:
            return
        [archive] = backend.parent.glob("*.tar.gz")
        with tarfile.open(archive) as sdist:
            sdist.extractall(work / "source", filter="data")
        [unpacked] = (work / "source").iterdir()
        # Constrain the extra to this exact local build while exercising pip dependency resolution.
        constraint = work / "local-backend.txt"
        constraint.write_text(f"particula-aiomfac @ {backend.as_uri()}\n")
        for label, requirement in [
            ("base", str(base)),
            ("backend", str(backend)),
            ("combined", str(base) + "[aiomfac]"),
            *(
                [
                    ("base-source", str(next(base.parent.glob("*.tar.gz")))),
                    ("backend-source", str(archive)),
                ]
                if args.sdists
                else []
            ),
            *([("backend-cuda", str(backend) + "[cuda]")] if args.cuda else []),
        ]:
            env = work / ("env-" + label)
            venv.EnvBuilder(with_pip=True).create(env)
            python = env / "bin/python"
            subprocess.run(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    *(
                        ["--constraint", constraint]
                        if label == "combined"
                        else []
                    ),
                    "--find-links",
                    backend.parent,
                    requirement,
                ],
                cwd=work,
                check=True,
            )
            subprocess.run([python, "-m", "pip", "check"], cwd=work, check=True)
            probe = """
import importlib.util, importlib.metadata, sys
from pathlib import Path
for name,wanted in zip(("particula","particula_aiomfac"),sys.argv[1:],strict=True):
    spec=importlib.util.find_spec(name)
    assert (spec is not None)==(wanted=="yes"),name
    if spec:
        module=__import__(name)
        assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        assert module.__version__==importlib.metadata.version(name)
        print(name,module.__version__,module.__file__)
"""
            subprocess.run(
                [
                    python,
                    "-I",
                    "-c",
                    probe,
                    "no" if label.startswith("backend") else "yes",
                    "no" if label.startswith("base") else "yes",
                ],
                cwd=work,
                check=True,
            )
            subprocess.run(
                [
                    python,
                    "-I",
                    unpacked / "tools/installed_probe.py",
                    label.removesuffix("-source"),
                    *(
                        ["cuda"]
                        if args.cuda and label in {"combined", "backend-cuda"}
                        else []
                    ),
                ],
                env=dict(os.environ, PATH=str(python.parent), PYTHONPATH=""),
                cwd=work,
                check=True,
            )
        print(
            "Verified both archive pairs and all requested isolated installs."
        )


if __name__ == "__main__":
    main()
