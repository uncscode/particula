"""Download Jakobsson et al.'s event archive and Tables 1, 3 and 4 by digest."""

import argparse
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

TABLE = "https://acp.copernicus.org/articles/22/6717/2022/acp-22-6717-2022-t"
INPUTS = {
    "lund-study-events.tar": (
        "https://web.nateko.lu.se/vp/inp_raw_data.tar",
        "cbc85471842794a0f3c16c9c58f7a8b05a5507b066871316d77bd7ac01ba9eb7",
    ),
    "cloud-sources/lund-table1.xlsx": (
        TABLE + "01.xlsx",
        "87cd1af2b3ae2f0d46e4932cbace489afc753ffa759471ae15e0d7195bfa206d",
    ),
    "lund-study-table3.xlsx": (
        TABLE + "03.xlsx",
        "16a664ab91ab5b7b173116d8c748e796e0a5af7c6a8dfc09adf30f74f62bf790",
    ),
    "lund-study-table4.xlsx": (
        TABLE + "04.xlsx",
        "eb5127e695ff2e77143b587916f9cecf58a9189b472b60676f29fcdd5355397c",
    ),
}


def fetch(directory):
    for name, (url, expected) in INPUTS.items():
        path = directory / name
        if path.exists():
            if sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(
                    f"Existing file has an unexpected digest: {path}"
                )
        else:
            with urlopen(url, timeout=90) as response:
                data = response.read()
            if sha256(data).hexdigest() != expected:
                raise ValueError(
                    f"Downloaded bytes differ from the study input: {url}"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
                temp.write(data)
            Path(temp.name).replace(path)
        print(f"Verified {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    fetch(parser.parse_args().directory)
