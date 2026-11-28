"""Convert pinned channels and package the reviewed native records."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from particula.dynamics.chemistry.catalog import ChemistryCatalog, sources
from particula.dynamics.chemistry.catalog.bundle import ASSET, FORMAT, RELEASE
from particula.dynamics.chemistry.catalog.native import compile_channel
from particula.dynamics.chemistry.catalog.store import IMPORT_VERSION
from particula.dynamics.chemistry.mechanism import ReactionMechanism

DATA = Path(__file__).resolve().parents[1] / "chemistry_catalog"
NOTICES = Path(__file__).with_name("catalog_notices")
CITATIONS = {
    "waccm": "https://www.cesm.ucar.edu/working-groups/whole-atmosphere/code",
    "mcm": "https://www.mcm.york.ac.uk/MCM/about/cite",
    "e3sm": "https://github.com/E3SM-Project/E3SM/releases/tag/v3.0.0",
    "wrf-chem": "https://github.com/wrf-model/WRF/releases/tag/v4.6.1",
    "cmaq": "https://github.com/USEPA/CMAQ/tree/5.5/CCTM/src/MECHS/cracmm2",
    "camx": "https://www.camx.com/download/source/",
    "pmcamx": "https://doi.org/10.5281/zenodo.4071362",
}


def _convert(record):
    """Return one validated executable channel, with any oxygen assumption."""
    try:
        reaction, species = compile_channel(record)
        closure = False
    except ValueError:
        try:
            reaction, species = compile_channel(record, close_oxygen=True)
        except ValueError:
            return None
        closure = True
    mechanism = dict(
        version=1, basis="mol/m^3-air", species=species, reactions=[reaction]
    )
    ReactionMechanism.from_dict(mechanism)
    aliases = {
        role: sorted(
            set(record[role + "s"])
            - {name.removeprefix("gas:") for name in reaction[role + "s"]}
        )
        for role in ("reactant", "product")
    }
    return dict(
        source_aliases={
            role: names for role, names in aliases.items() if names
        },
        label=record["label"],
        revision=record["revision"],
        line=record["line"],
        requires_closed_oxygen=closure,
        mechanism=mechanism,
    )


def refresh(catalog, directory):
    """Regenerate reviewable native records using checksum-pinned inputs."""
    directory.mkdir(parents=True, exist_ok=True)
    snapshots = {}
    for key in sorted(sources.recipes()):
        snapshot = catalog.show(catalog.fetch(key))
        records = catalog.selection([snapshot["id"]])["reactions"]
        converted = [item for r in records if (item := _convert(r))]
        data = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in converted
        ).encode()
        (directory / f"{key}.jsonl").write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        snapshots[key] = {
            field: snapshot[field]
            for field in ("release", "mechanism", "files", "source", "aerosol")
        }
        snapshots[key].update(
            id=f"{snapshot['id']}:native@{digest}",
            source_snapshot=snapshot["id"],
            importer_version=snapshot["importer_version"],
            source_count=snapshot["count"],
            count=len(converted),
            requires_closed_oxygen=sum(
                row["requires_closed_oxygen"] for row in converted
            ),
            sha256=digest,
            citation=CITATIONS[snapshot["source"]],
        )
    (directory / "snapshots.json").write_text(
        json.dumps(snapshots, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build(output, directory=DATA):
    """Package the reviewed native files offline with a release checksum."""
    output.mkdir(parents=True, exist_ok=True)
    snapshots = json.loads((directory / "snapshots.json").read_bytes())
    recipes = sources.recipes()
    if snapshots.keys() != recipes.keys():
        raise ValueError("native inventory differs from pinned recipe list")
    manifest = dict(
        format=FORMAT,
        release=RELEASE,
        importer_version=IMPORT_VERSION,
        snapshots=snapshots,
    )
    entries = {"manifest.json": json.dumps(manifest, sort_keys=True).encode()}
    for key, snapshot in snapshots.items():
        recipe = recipes[key]
        primary, *companions = recipe["files"]
        files = [primary, *sorted(companions, key=lambda f: f["name"])]
        name = f"{key}.jsonl"
        data = (directory / name).read_bytes()
        if (
            hashlib.sha256(data).hexdigest() != snapshot["sha256"]
            or len(data.splitlines()) != snapshot["count"]
            or snapshot["importer_version"] != IMPORT_VERSION
            or snapshot["files"] != files
            or any(
                snapshot[k] != recipe[k]
                for k in ("source", "release", "mechanism", "aerosol")
            )
        ):
            raise ValueError(f"stale or damaged native catalog for {key}")
        entries[name] = data
    for notice in sorted(NOTICES.glob("*.txt")):
        entries[f"notices/{notice.name}"] = notice.read_bytes()
    with TemporaryDirectory(dir=output) as temp:
        work = Path(temp)
        with zipfile.ZipFile(work / ASSET, "w") as bundle:
            for name, data in sorted(entries.items()):
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = 0o644 << 16
                bundle.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)
        digest = hashlib.sha256((work / ASSET).read_bytes()).hexdigest()
        checksum = f"{ASSET}.sha256"
        (work / checksum).write_text(f"{digest}  {ASSET}\n", encoding="ascii")
        for name in (ASSET, checksum):
            (work / name).replace(output / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="regenerate tracked native records from pinned recipes first",
    )
    args = parser.parse_args()
    if args.refresh:
        refresh(ChemistryCatalog(), DATA)
    build(args.output)
