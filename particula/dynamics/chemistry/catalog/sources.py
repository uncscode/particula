"""Retrieve the repository's checksum-pinned atmospheric mechanism sources."""

import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen


def recipes():
    """Return available recipes, their origins, and reviewed byte digests."""
    return json.loads(
        Path(__file__).with_name("sources.json").read_text("utf-8")
    )


def _verify(path, expected):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(
            f"checksum mismatch for {path.name}: expected {expected}, "
            f"received {actual}"
        )


def _member(archive, name):
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            return bundle.read(name)
    with tarfile.open(archive) as bundle:
        info = bundle.getmember(name)
        if not info.isfile():
            raise ValueError(f"expected regular source file {name}")
        stream = bundle.extractfile(info)
        if stream is None:
            raise ValueError(f"unreadable source member {name}")
        with stream:
            return stream.read()


def fetch(catalog, key):
    """Verify and import a published recipe using available cached files."""
    recipe = recipes()[key]
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        paths = []
        for spec in recipe["files"]:
            path = directory / spec["name"]
            paths.append(path)
            cached = catalog.source_bytes(spec["sha256"])
            if cached is not None:
                path.write_bytes(cached)
            else:
                download = (
                    directory / spec["archive_sha256"]
                    if "member" in spec
                    else path
                )
                if not download.exists():
                    # Only HTTPS URLs in the installed manifest are accepted.
                    if not spec["url"].startswith("https://"):
                        raise ValueError("source recipes require HTTPS")
                    with (
                        urlopen(spec["url"], timeout=120) as response,  # noqa: S310
                        download.open("wb") as out,
                    ):
                        shutil.copyfileobj(response, out)
                    _verify(
                        download, spec.get("archive_sha256", spec["sha256"])
                    )
                if "container" in spec:
                    inner = directory / spec["container_sha256"]
                    inner.write_bytes(_member(download, spec["container"]))
                    _verify(inner, spec["container_sha256"])
                    download = inner
                if "member" in spec:
                    path.write_bytes(_member(download, spec["member"]))
            _verify(path, spec["sha256"])
        return catalog.import_file(
            recipe["source"],
            paths[0],
            release=recipe["release"],
            mechanism=recipe["mechanism"],
            aerosol=recipe["aerosol"],
            companions=paths[1:],
            origins=recipe["files"],
        )
