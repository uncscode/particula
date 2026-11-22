"""Retrieve the author files used by the Dunne study and verify their digests."""

import argparse
import csv
import hashlib
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
COMPANION = {
    "donahue2026-figure-data.zip",
    "kuerten2019-data.zip",
    "kuerten2015-correction.pdf",
}


def inputs():
    """Yield local names, author URLs and expected hashes."""
    with (ROOT / "sources.csv").open(newline="") as stream:
        for record in csv.DictReader(stream):
            yield (
                record["file"],
                record["source_url"],
                "md5",
                record["publisher_md5"],
            )
    with (ROOT / "companion/sources.csv").open(newline="") as stream:
        for record in csv.DictReader(stream):
            if record["filename"] in COMPANION:
                yield (
                    "companion/" + record["filename"],
                    record["download_url"],
                    "sha256",
                    record["sha256_local"],
                )


def digest(path, algorithm):
    """Hash a file incrementally."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def fetch(cache):
    """Reuse verified inputs or install a checked download atomically."""
    for name, url, algorithm, expected in inputs():
        path = ROOT / name
        if path.is_file() and digest(path, algorithm) == expected:
            print("Verified", name)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
            temporary = Path(temp.name)
        try:
            with (
                (cache / name).open("rb")
                if cache
                else urlopen(url, timeout=120) as source
            ):
                with temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
            if digest(temporary, algorithm) != expected:
                raise ValueError(f"Checksum mismatch for {url}")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        print("Verified", name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, help="Copy from an existing study input tree."
    )
    fetch(parser.parse_args().cache)
