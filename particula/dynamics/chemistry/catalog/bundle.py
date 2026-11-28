"""Install the prebuilt reaction catalog from a versioned catalog release."""

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import urlopen

from particula.dynamics.chemistry.catalog.sources import _verify

ASSET = "particula-chemistry-catalog.zip"
FORMAT = "particula.chemistry-catalog/2"
TAG_PREFIX = "chemistry-catalog-v"
VERSION_PATTERN = r"(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){2}"
RELEASES_API = "https://api.github.com/repos/uncscode/particula/releases"
DOWNLOAD_URL = "https://github.com/uncscode/particula/releases/download"


def catalog_tag(version):
    """Return the stable GitHub release tag for a catalog data version."""
    if not re.fullmatch(VERSION_PATTERN, version):
        raise ValueError("catalog version must be MAJOR.MINOR.PATCH")
    return TAG_PREFIX + version


def latest_version():
    """Find the highest stable catalog version across release pages."""
    versions = []
    page = 1
    while True:
        url = f"{RELEASES_API}?per_page=100&page={page}"
        with urlopen(url, timeout=60) as response:  # noqa: S310
            releases = json.load(response)
        for item in releases:
            tag = item["tag_name"]
            if (
                item["draft"]
                or item["prerelease"]
                or not tag.startswith(TAG_PREFIX)
            ):
                continue
            version = tag.removeprefix(TAG_PREFIX)
            if re.fullmatch(VERSION_PATTERN, version):
                versions.append(tuple(map(int, version.split("."))))
        if len(releases) < 100:
            break
        page += 1
    if not versions:
        raise ValueError(
            "No chemistry catalog is published. "
            "Use 'particula chem fetch RECIPE' for a source."
        )
    return ".".join(map(str, max(versions)))


def _download(archive, tag):
    """Download both assets from the selected catalog tag and verify the ZIP."""
    base = f"{DOWNLOAD_URL}/{tag}"
    try:
        with urlopen(f"{base}/{ASSET}.sha256", timeout=60) as src:  # noqa: S310
            checksum = src.read(256).decode("ascii").split()
        if len(checksum) != 2 or checksum[1] != ASSET:
            raise ValueError("invalid release catalog checksum file")
        with (
            urlopen(f"{base}/{ASSET}", timeout=120) as src,  # noqa: S310
            archive.open("wb") as output,
        ):
            shutil.copyfileobj(src, output)
    except HTTPError as exc:
        exc.close()
        if exc.code == 404:
            raise ValueError(
                f"No chemistry catalog is published for {tag}. "
                "Use 'particula chem fetch RECIPE' to import a source."
            ) from exc
        raise
    _verify(archive, checksum[0])


def _install_snapshot(db, snapshot, data):
    expected_id = f"{snapshot['source_snapshot']}:catalog@{snapshot['sha256']}"
    if (
        hashlib.sha256(data).hexdigest() != snapshot["sha256"]
        or len(data.splitlines()) != snapshot["count"]
        or snapshot["count"] != snapshot["source_count"]
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
        rid = f"{snapshot['id']}/{record['label']}"
        record.update(
            id=rid,
            snapshot=snapshot["id"],
            source=snapshot["source"],
        )
        db.execute(
            "INSERT INTO reactions VALUES (?,?,?,?)",
            (rid, snapshot["id"], snapshot["source"], json.dumps(record)),
        )
        for role in ("reactant", "product"):
            names = set(record[role + "s"])
            if role == "product":
                names.update(record["metadata"].get("product_bins", ()))
            if record["mechanism"] is not None:
                (reaction,) = record["mechanism"]["reactions"]
                names.update(
                    n.removeprefix("gas:") for n in reaction[role + "s"]
                )
            db.executemany(
                "INSERT INTO terms VALUES (?,?,?)",
                ((name, role, rid) for name in sorted(names)),
            )


def install_release(catalog, version=None):
    """Index every released source entry and native conversion atomically."""
    tag = catalog_tag(latest_version() if version is None else version)
    with TemporaryDirectory() as temp:
        archive = Path(temp) / ASSET
        _download(archive, tag)
        try:
            with zipfile.ZipFile(archive) as bundle, catalog._db() as db:
                manifest = json.loads(bundle.read("manifest.json"))
                if (
                    not isinstance(manifest, dict)
                    or manifest.get("format") != FORMAT
                    or manifest.get("release") != tag
                    or not isinstance(manifest.get("snapshots"), dict)
                    or not manifest["snapshots"]
                ):
                    raise ValueError("incompatible release catalog manifest")
                db.execute("BEGIN IMMEDIATE")
                db.execute("DELETE FROM active_release_snapshots")
                for key, snapshot in manifest["snapshots"].items():
                    _install_snapshot(db, snapshot, bundle.read(f"{key}.jsonl"))
                    db.execute(
                        "INSERT INTO active_release_snapshots VALUES (?)",
                        (snapshot["id"],),
                    )
        except (KeyError, TypeError, zipfile.BadZipFile) as exc:
            raise ValueError("invalid release catalog archive") from exc
        return manifest
