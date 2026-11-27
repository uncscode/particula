"""Compare the Figure 4 time digitization to printed Figure 3 and S4 spectra legends."""

import csv
from pathlib import Path

here = Path(__file__).resolve().parent
legends = {
    ("pinene", "dry"): [0, 1, 28, 50, 100, 173, 1113],
    ("ocimene", "dry"): [0, 1, 25, 102, 143, 415, 1166, 1301],
    ("caryophyllene", "dry"): [0, 1, 32, 70, 95, 158, 214, 1146],
    ("farnesene", "dry"): [0, 1, 32, 62, 131, 208, 1207],
    ("pinene", "wet"): [0, 1, 22, 50, 110, 1042],
    ("ocimene", "wet"): [0, 1, 25, 72, 127, 399, 1138, 1265],
    ("caryophyllene", "wet"): [0, 1, 32, 64, 93, 154, 209, 1144],
    ("farnesene", "wet"): [0, 1, 30, 62, 128, 205, 1185],
}
with (here / "observations.csv").open() as stream:
    data = list(csv.DictReader(stream))
with (here / "time_verification.csv").open("w", newline="") as stream:
    w = csv.writer(stream)
    w.writerow(
        [
            "precursor",
            "rh",
            "figure4_time_min",
            "other_figure",
            "compatible_legend_times_min",
            "status",
            "unique_time_difference_min",
        ]
    )
    for point in data:
        key = point["precursor"], point["rh"]
        time = float(point["time_min"])
        hits = [t for t in legends[key] if abs(t - time) <= 4.58]
        w.writerow(
            [
                *key,
                time,
                "3" if key[1] == "dry" else "S4",
                "/".join(map(str, hits)),
                (
                    "unique"
                    if len(hits) == 1
                    else "overlap"
                    if hits
                    else "unmatched"
                ),
                round(time - hits[0], 2) if len(hits) == 1 else "",
            ]
        )
