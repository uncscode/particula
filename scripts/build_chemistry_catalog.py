"""Build the GitHub Release catalog from the pinned, redistributable recipes."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from particula.dynamics.chemistry.catalog import ChemistryCatalog
from particula.dynamics.chemistry.catalog.bundle import (
    ASSET,
    DATABASE,
    FORMAT,
    RELEASE,
)
from particula.dynamics.chemistry.catalog.store import IMPORT_VERSION

NOTICES = Path(__file__).with_name("catalog_notices")
# Each recipe includes its upstream notice in the download.
CITATIONS = {
    "cmaq": "https://github.com/USEPA/CMAQ/tree/5.5/CCTM/src/MECHS/cracmm2",
    "e3sm": "https://github.com/E3SM-Project/E3SM/releases/tag/v3.0.0",
    "wrf-chem": "https://github.com/wrf-model/WRF/releases/tag/v4.6.1",
}


def build(output):
    """Create a fresh database, its provenance manifest and a checksum."""
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temp:
        work = Path(temp)
        catalog = ChemistryCatalog(work / DATABASE)
        snapshots = {}
        for key, citation in sorted(CITATIONS.items()):
            snapshot = catalog.show(catalog.fetch(key))
            snapshots[key] = {
                field: snapshot[field]
                for field in ("id", "count", "release", "mechanism", "files")
            }
            snapshots[key]["citation"] = citation
        manifest = {
            "format": FORMAT,
            "release": RELEASE,
            "importer_version": IMPORT_VERSION,
            "snapshots": snapshots,
        }
        with zipfile.ZipFile(work / ASSET, "w") as bundle:
            entries = {
                DATABASE: catalog.path.read_bytes(),
                "manifest.json": json.dumps(
                    manifest, indent=2, sort_keys=True
                ).encode(),
                **{
                    f"notices/{key}.txt": (NOTICES / f"{key}.txt").read_bytes()
                    for key in snapshots
                },
            }
            for name, data in sorted(entries.items()):
                # Fixed timestamps and permissions keep repeat builds identical.
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = 0o644 << 16
                bundle.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)
        with (work / ASSET).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        checksum = f"{ASSET}.sha256"
        (work / checksum).write_text(f"{digest}  {ASSET}\n", encoding="ascii")
        for name in (ASSET, checksum):
            (work / name).replace(output / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
