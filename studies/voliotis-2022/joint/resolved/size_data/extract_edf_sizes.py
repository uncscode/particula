"""Read DMPS size summaries from the pinned EUROCHAMP source records."""

import argparse
import csv
import hashlib
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent


def source_bytes(record, raw_dirs):
    """Obtain one raw file and confirm its archived SHA-256."""
    run, file = int(record["run"]), record["file"]
    if raw_dirs:
        hits = [
            path
            for raw in raw_dirs
            for path in (raw / str(run)).glob("*/" + file)
        ]
        if len(hits) != 1:
            raise ValueError(f"Expected one archived copy of {file}")
        data = hits[0].read_bytes()
    else:
        with urlopen(record["url"], timeout=30) as stream:  # noqa: S310
            data = stream.read()
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError(f"Source checksum mismatch: {file}")
    return data


def main():
    """Reproduce size summaries from the public EDFs or a local raw archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, nargs="+")
    args = parser.parse_args()
    with (HERE / "edf_sources.csv").open() as src:
        records = list(csv.DictReader(src))
    modes, areas = [], []
    for record in records:
        run, file = int(record["run"]), record["file"]
        data = source_bytes(record, args.raw)
        header, body = data.decode("latin-1").split("&&&&&&&&&&", 1)
        if f"SHORT_NAME(1)={record['mode_header']}" not in header:
            raise ValueError(f"Unexpected mode field in {file}")
        area_column = "SHORT_NAME(2)=Total_surface" in header
        for line in body.strip().splitlines():
            row = line.split("\t")
            seconds, mode = float(row[0]), float(row[1])
            if -900 <= seconds <= 18000:
                if mode > 0:
                    modes.append((run, seconds, mode))
                if area_column and float(row[2]) > 0:
                    areas.append((run, seconds, float(row[2])))
    for name, field, rows in (
        ("dmps_modes.csv", "mode_nm", modes),
        ("dmps_surface.csv", "surface_nm2_cm3", areas),
    ):
        with (HERE / name).open("w") as out:
            writer = csv.writer(out, lineterminator="\n")
            writer.writerow(("run", "time_s", field))
            writer.writerows(sorted(rows))


if __name__ == "__main__":
    main()
