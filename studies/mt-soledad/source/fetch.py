"""Download or verify the original UCSD products without duplicating them in Git."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

MANIFEST = Path(__file__).with_name("products.json")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    products = json.loads(MANIFEST.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "product", nargs="*", help="optional file names from products.json"
    )
    args = parser.parse_args()
    selected = args.product or list(products)
    unknown = set(selected) - set(products)
    if unknown:
        parser.error(f"Unknown products: {sorted(unknown)}")
    args.directory.mkdir(parents=True, exist_ok=True)
    for name in selected:
        if name == "filter-flow.zip" and (args.directory / "minute-flow.csv").is_file():
            if (
                digest(args.directory / "minute-flow.csv")
                != "8e779e92c897f34b4a2ccc7677a4fa99ddb5d018c1597ec6fbf66bcbd6ba3bf8"
            ):
                raise ValueError("Unexpected minute-series digest")
            continue
        destination = args.directory / name
        expected = products[name]["sha256"]
        if destination.exists():
            if expected and digest(destination) != expected:
                raise ValueError(f"Existing file differs from release: {destination}")
            continue
        with NamedTemporaryFile(
            dir=args.directory, prefix=name + ".", delete=False
        ) as temp:
            temp_path = Path(temp.name)
        try:
            with (
                urlopen(products[name]["url"], timeout=120) as remote,
                temp_path.open("wb") as out,
            ):
                shutil.copyfileobj(remote, out, length=1024 * 1024)
            if expected and digest(temp_path) != expected:
                raise ValueError(
                    f"Downloaded product differs from recorded SHA-256: {name}"
                )
            temp_path.replace(destination)
        finally:
            temp_path.unlink(missing_ok=True)
        print(destination)


if __name__ == "__main__":
    main()
