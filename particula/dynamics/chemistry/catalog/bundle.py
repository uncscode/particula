"""Install the prebuilt reaction catalog from the matching Particula release."""

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
DATABASE = "chemistry.sqlite3"
FORMAT = "particula.chemistry-catalog/1"
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


def install_release(catalog):
    """Merge the verified release snapshots in one transaction."""
    with TemporaryDirectory() as temp:
        archive = Path(temp) / ASSET
        _download(archive)
        database = Path(temp) / DATABASE
        try:
            with zipfile.ZipFile(archive) as bundle:
                manifest = json.loads(bundle.read("manifest.json"))
                if (
                    not isinstance(manifest, dict)
                    or manifest.get("format") != FORMAT
                    or manifest.get("release") != RELEASE
                    or manifest.get("importer_version") != IMPORT_VERSION
                ):
                    raise ValueError("incompatible release catalog manifest")
                with (
                    bundle.open(DATABASE) as src_db,
                    database.open("wb") as out,
                ):
                    shutil.copyfileobj(src_db, out)
        except (KeyError, zipfile.BadZipFile) as exc:
            raise ValueError("invalid release catalog archive") from exc

        with catalog._db() as db:
            db.execute("ATTACH DATABASE ? AS bundled", (str(database),))
            db.execute("BEGIN IMMEDIATE")
            if db.execute("PRAGMA bundled.integrity_check").fetchall() != [
                ("ok",)
            ]:
                raise ValueError("invalid release catalog database")
            inventory = dict(
                db.execute(
                    "SELECT s.id, count(r.id) FROM bundled.snapshots s "
                    "LEFT JOIN bundled.reactions r ON r.snapshot=s.id "
                    "GROUP BY s.id"
                )
            )
            snapshots = manifest.get("snapshots")
            if not isinstance(snapshots, dict) or not snapshots:
                raise ValueError("missing release catalog inventory")
            try:
                expected = {
                    item["id"]: item["count"] for item in snapshots.values()
                }
            except (KeyError, TypeError) as exc:
                raise ValueError("invalid release catalog inventory") from exc
            if inventory != expected or len(expected) != len(snapshots):
                raise ValueError("release catalog inventory mismatch")
            for table in ("files", "snapshots", "reactions", "terms"):
                db.execute(
                    f"INSERT OR IGNORE INTO main.{table} "  # noqa: S608
                    f"SELECT * FROM bundled.{table}"
                )
        return manifest
