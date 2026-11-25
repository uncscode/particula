"""Retrieve and verify the two publisher files used by the reconstruction."""

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

from extract_tensions import extract

HERE = Path(__file__).resolve().parent
BASE = "https://acp.copernicus.org/articles/25/14301/2025/"
FILES = {
    "acp-25-14301-2025-supplement.zip": (
        "858763efe5f2383fa41b02bd4c789c1189480960a13fda7a75e9c95551fdd965"
    ),
    "acp-25-14301-2025.pdf": (
        "e6ec27719c9a78e00b62464142e6ec908dbe3e2c4c8f7a73b27bd7efa3f5837d"
    ),
}


def publisher_file(cache, name):
    """Cache publisher files after checking their SHA-256 digests."""
    target = cache / name
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != FILES[name]:
            raise ValueError(f"Checksum mismatch in cached {target}")
        return target
    with urlopen(BASE + name, timeout=90) as response:  # noqa: S310
        data = response.read()
    if hashlib.sha256(data).hexdigest() != FILES[name]:
        raise ValueError(f"Publisher file has changed: {name}")
    with tempfile.NamedTemporaryFile(dir=cache, delete=False) as temp:
        temp.write(data)
    Path(temp.name).replace(target)
    return target


def main():
    """Reconstruct the measured tensions and digitized growth envelope."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=HERE / "publisher")
    cache = parser.parse_args().cache
    cache.mkdir(exist_ok=True, parents=True)
    supplement, article = [publisher_file(cache, name) for name in FILES]
    extract(supplement)
    subprocess.run(  # noqa: S603 - fixed local script and interpreter
        [
            sys.executable,
            str(HERE / "digitize_growth.py"),
            str(article),
            "--output",
            str(HERE / "growth_envelope.csv"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
