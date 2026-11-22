"""Build both distributions and check three isolated wheel installations."""

import argparse
import email
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
        source_files = {p.split("/", 1)[1] for p in archive.getnames() if "/" in p}
    if backend:
        required = {
            "LICENSE",
            "NOTICE",
            "tools/reference.py",
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
                segment in {"reference", "tools", "tests"} for segment in Path(p).parts
            )
            for p in wheel_files
        )
        for notice in ("LICENSE", "NOTICE"):
            assert any(p.endswith(".dist-info/licenses/" + notice) for p in wheel_files)
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
        import hashlib

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
                        hashlib.sha256(archive.read(info)).digest() not in protected
                    ), info.filename
        with tarfile.open(sdist) as archive:
            for member in archive:
                if member.isfile():
                    assert (
                        hashlib.sha256(archive.extractfile(member).read()).digest()
                        not in protected
                    ), member.name
    return wheel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-isolation",
        action="store_true",
        help="use already installed build and flit_core",
    )
    args = parser.parse_args()
    cache = ROOT / ".artifacts/aiomfac-packages"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as work:
        work = Path(work)
        wheels = []
        for name, project in (("base", ROOT), ("backend", PACKAGE)):
            dist = work / name
            cmd = [sys.executable, "-m", "build", "--outdir", str(dist)]
            if args.no_isolation:
                cmd.append("--no-isolation")
            subprocess.run([*cmd, str(project)], cwd=work, check=True)
            wheels.append(check_archives(dist, name == "backend"))
        base, backend = wheels
        # Constrain the extra to this exact local build while exercising pip dependency resolution.
        constraint = work / "local-backend.txt"
        constraint.write_text(f"particula-aiomfac @ {backend.as_uri()}\n")
        for label, requirement in [
            ("base", str(base)),
            ("backend", str(backend)),
            ("combined", str(base) + "[aiomfac]"),
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
                    "--constraint",
                    constraint,
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
                    "no" if label == "backend" else "yes",
                    "no" if label == "base" else "yes",
                ],
                cwd=work,
                check=True,
            )
        print("Verified both archive pairs and three isolated installs.")


if __name__ == "__main__":
    main()
