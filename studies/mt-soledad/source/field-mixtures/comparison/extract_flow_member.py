"""Extract the verified minute flow series from the UCSD ZIP or its complete leading members."""

import argparse
import hashlib
from pathlib import Path
import struct
import zlib

EXPECTED = "8e779e92c897f34b4a2ccc7677a4fa99ddb5d018c1597ec6fbf66bcbd6ba3bf8"


def extract(archive, destination):
    with archive.open("rb") as f:
        for _ in range(2):
            (
                sig,
                _,
                flags,
                method,
                _,
                _,
                crc,
                compressed,
                size,
                nname,
                nextra,
            ) = struct.unpack("<4s5H3I2H", f.read(30))
            if sig != b"PK\x03\x04" or flags:
                raise ValueError("Unexpected ZIP local header")
            name = f.read(nname).decode()
            f.read(nextra)
            data = f.read(compressed)
            if "/Minute_Mean_" not in name:
                continue
            if method != 8:
                raise ValueError("Expected deflate")
            out = zlib.decompress(data, -15)
            if (
                len(out) != size
                or zlib.crc32(out) != crc
                or hashlib.sha256(out).hexdigest() != EXPECTED
            ):
                raise ValueError("Minute-series checksum failed")
            destination.write_bytes(out)
            return
    raise ValueError("Minute series absent")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("destination", type=Path)
    args = p.parse_args()
    extract(args.archive, args.destination)
