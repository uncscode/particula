"""PEP 517/660 adapter: stage maintained docs, then let Flit build the package.

No Git, network, simulation imports or source-tree writes are required.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path

from flit_core import buildapi as flit
from particula_docs import MANIFEST, source_files, source_version

get_requires_for_build_wheel = flit.get_requires_for_build_wheel
get_requires_for_build_sdist = flit.get_requires_for_build_sdist
get_requires_for_build_editable = flit.get_requires_for_build_editable
prepare_metadata_for_build_wheel = flit.prepare_metadata_for_build_wheel
prepare_metadata_for_build_editable = flit.prepare_metadata_for_build_editable
build_sdist = flit.build_sdist
# Flit's editable .pth exposes the source root, including particula_docs.py.
build_editable = flit.build_editable


@contextmanager
def staged_source(config_settings):
    """Stage the same complete sources available from an offline sdist."""
    original = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="particula-docs-build-") as temp:
        root = Path(temp)
        archive = root / flit.build_sdist(str(root), config_settings)
        with tarfile.open(archive) as source:
            source.extractall(root, filter="data")
        (staging,) = (p for p in root.iterdir() if p.is_dir())
        try:
            os.chdir(staging)
            yield staging
        finally:
            os.chdir(original)


def stage_docs(root: Path) -> None:
    """Copy sources unchanged and record version, size and hashes."""
    source = root / "docs"
    destination = root / "particula/docs"
    if destination.exists():
        raise ValueError("particula/docs must be generated, not maintained")
    files = {}
    for path in source_files(source):
        relative = path.relative_to(source)
        content = path.read_bytes()
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        files[relative.as_posix()] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
    manifest = {
        "schema_version": 1,
        "version": source_version(root / "particula"),
        "size_bytes": sum(record["size_bytes"] for record in files.values()),
        "files": files,
    }
    (destination / MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def add_locator(wheel: Path, locator: bytes) -> None:
    """Add the independent lightweight module and its PEP 427 RECORD entry.

    Flit packages one import package. A separate top-level locator is necessary
    to avoid executing particula.__init__ when reading installed documentation.
    Preserve every Flit file and its metadata, updating only RECORD.
    """
    temporary = wheel.with_suffix(".tmp")
    with (
        zipfile.ZipFile(wheel) as original,
        zipfile.ZipFile(temporary, "w") as out,
    ):
        record_name = next(
            n for n in original.namelist() if n.endswith(".dist-info/RECORD")
        )
        records = list(
            csv.reader(io.StringIO(original.read(record_name).decode()))
        )
        for entry in original.infolist():
            if entry.filename != record_name:
                out.writestr(entry, original.read(entry.filename))
        entry = zipfile.ZipInfo(
            "particula_docs.py", original.getinfo(record_name).date_time
        )
        entry.compress_type = zipfile.ZIP_DEFLATED
        entry.external_attr = 0o100644 << 16
        out.writestr(entry, locator)
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(locator).digest())
            .rstrip(b"=")
            .decode()
        )
        records.append(
            ["particula_docs.py", f"sha256={digest}", str(len(locator))]
        )
        text = io.StringIO(newline="")
        csv.writer(text, lineterminator="\n").writerows(records)
        out.writestr(original.getinfo(record_name), text.getvalue().encode())
    temporary.replace(wheel)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build an ordinary wheel with documentation, without changing checkout."""
    output = Path(wheel_directory).resolve()
    metadata = (
        str(Path(metadata_directory).resolve()) if metadata_directory else None
    )
    with staged_source(config_settings) as root:
        stage_docs(root)
        name = flit.build_wheel(str(output), config_settings, metadata)
        add_locator(output / name, (root / "particula_docs.py").read_bytes())
    return name
