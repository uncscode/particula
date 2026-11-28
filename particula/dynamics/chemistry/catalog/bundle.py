"""Install the prebuilt reaction catalog from the matching Particula release."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import urlopen

from particula import __version__
from particula.dynamics.chemistry.catalog.sources import _verify
from particula.dynamics.chemistry.catalog.store import IMPORT_VERSION

ASSET = "particula-chemistry-catalog.zip"
FORMAT = "particula.native-chemistry-catalog/1"
RELEASE = f"v{__version__}"
BASE_URL = f"https://github.com/uncscode/particula/releases/download/{RELEASE}"


def _download(archive):
    """Download both assets from the same library tag and verify the ZIP."""
    try:
        with urlopen(f"{BASE_URL}/{ASSET}.sha256", timeout=60) as src:  # noqa: S310
            checksum = src.read(256).decode("ascii").split()
        if len(checksum) != 2 or checksum[1] != ASSET:
            raise ValueError("invalid release catalog checksum file")
        with (
            urlopen(f"{BASE_URL}/{ASSET}", timeout=120) as src,  # noqa: S310
            archive.open("wb") as output,
        ):
            shutil.copyfileobj(src, output)
    except HTTPError as exc:
        exc.close()
        if exc.code == 404:
            raise ValueError(
                f"No chemistry catalog is published for {RELEASE}. "
                "Use 'particula chem fetch RECIPE' to import a source."
            ) from exc
        raise
    _verify(archive, checksum[0])


def _equation(terms):
    return " + ".join(
        (f"{n} " if n != 1 else "") + name.removeprefix("gas:")
        for name, n in terms.items()
    )


def _install_snapshot(db, snapshot, data):
    expected_id = f"{snapshot['source_snapshot']}:native@{snapshot['sha256']}"
    if (
        hashlib.sha256(data).hexdigest() != snapshot["sha256"]
        or len(data.splitlines()) != snapshot["count"]
        or snapshot["id"] != expected_id
    ):
        raise ValueError("release catalog inventory mismatch")
    if db.execute(
        "SELECT 1 FROM snapshots WHERE id=?", (snapshot["id"],)
    ).fetchone():
        return
    provenance = {k: v for k, v in snapshot.items() if k != "id"}
    db.execute(
        "INSERT INTO snapshots VALUES (?,?)",
        (snapshot["id"], json.dumps(provenance, sort_keys=True)),
    )
    for line in data.splitlines():
        record = json.loads(line)
        (reaction,) = record["mechanism"]["reactions"]
        rid = f"{snapshot['id']}/{record['label']}"
        record.update(
            id=rid,
            snapshot=snapshot["id"],
            source=snapshot["source"],
            equation=(
                _equation(reaction["reactants"])
                + " -> "
                + _equation(reaction["products"])
            ),
        )
        db.execute(
            "INSERT INTO reactions VALUES (?,?,?,?)",
            (rid, snapshot["id"], snapshot["source"], json.dumps(record)),
        )
        for role in ("reactant", "product"):
            names = {n.removeprefix("gas:") for n in reaction[role + "s"]}
            names.update(record["source_aliases"].get(role, ()))
            db.executemany(
                "INSERT INTO terms VALUES (?,?,?)",
                ((name, role, rid) for name in sorted(names)),
            )


def install_release(catalog):
    """Index the preconverted native records in one local transaction."""
    with TemporaryDirectory() as temp:
        archive = Path(temp) / ASSET
        _download(archive)
        try:
            with zipfile.ZipFile(archive) as bundle, catalog._db() as db:
                manifest = json.loads(bundle.read("manifest.json"))
                if (
                    not isinstance(manifest, dict)
                    or manifest.get("format") != FORMAT
                    or manifest.get("release") != RELEASE
                    or manifest.get("importer_version") != IMPORT_VERSION
                    or not isinstance(manifest.get("snapshots"), dict)
                    or not manifest["snapshots"]
                ):
                    raise ValueError("incompatible release catalog manifest")
                db.execute("BEGIN IMMEDIATE")
                for key, snapshot in manifest["snapshots"].items():
                    _install_snapshot(db, snapshot, bundle.read(f"{key}.jsonl"))
        except (KeyError, TypeError, zipfile.BadZipFile) as exc:
            raise ValueError("invalid release catalog archive") from exc
        return manifest
