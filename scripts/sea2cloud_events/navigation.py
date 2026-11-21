"""Extract hourly ship positions from the NOAA TAN2003 underway deposit."""

import csv
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scripts.sea2cloud_events.figure_inputs import ROOT


def extract(path):
    """Use nearest reported fix within 15 minutes of each nominal UTC hour."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != (
        "cfb090e88389dd41d186b884216a6501247b531e14d5b74cf5ee7b5913819090"
    ):
        raise ValueError("Unexpected TAN2003 navigation file")
    with path.open() as stream:
        for _ in range(4):
            next(stream)
        rows = []
        for row in csv.DictReader(stream):
            t = datetime.strptime(
                row["DATE_UTC__ddmmyyyy"] + row["TIME_UTC_hh:mm:ss"],
                "%d%m%Y%H:%M:%S",
            )
            h = (t - datetime(2020, 3, 19)).total_seconds() / 3600
            if -4 <= h <= 90:
                rows.append(
                    [
                        h,
                        float(row["LAT_dec_degree"]),
                        float(row["LONG_dec_degree"]),
                    ]
                )
    data = np.asarray(rows)
    output = []
    for h in range(-3, 89):
        nearest = data[np.argmin(abs(data[:, 0] - h))]
        if abs(nearest[0] - h) < 0.25:
            output.append([h, *nearest])
    np.savetxt(
        ROOT / "track.csv",
        output,
        fmt="%.6f",
        delimiter=",",
        header="nominal_hour,actual_hour_from_19mar_00utc,latitude,longitude",
        comments="",
    )


if __name__ == "__main__":
    extract(Path(sys.argv[1]))
