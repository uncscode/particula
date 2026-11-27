"""Estimate remaining volume from paired initial and final plotted diameter modes."""

import csv
from pathlib import Path

here = Path(__file__).resolve().parent
# Manual center of top of peak. Columns give two labeled axis ticks in image
# pixels and nm, followed by the two peak abscissae, in original image pixels.
anchors = [
    ("pinene", "dry", 138, 813, 0, 400, 483, 365),
    ("ocimene", "dry", 873, 1552, 100, 400, 1280, 1144),
    ("farnesene", "dry", 875, 1556, 100, 300, 1103, 1038),
    ("pinene", "wet", 216, 619, 100, 300, 428, 273),
    ("ocimene", "wet", 710, 1112, 150, 400, 919, 802),
    ("caryophyllene", "wet", 215, 618, 100, 300, 403, 320),
    ("farnesene", "wet", 710, 1112, 100, 250, 889, 831),
]
with (here / "observations.csv").open() as f:
    last = {
        (r["precursor"], r["rh"]): float(r["vfr"]) for r in csv.DictReader(f)
    }
with (here / "peak_check.csv").open("w", newline="") as f:
    out = csv.writer(f)
    out.writerow(
        [
            "precursor",
            "rh",
            "x1_px",
            "x2_px",
            "d1_nm",
            "d2_nm",
            "x_initial_px",
            "x_last_px",
            "d_initial_nm",
            "d_last_nm",
            "volume_ratio",
            "figure4_vfr",
            "contrast",
        ]
    )
    for precursor, rh, x1, x2, d1, d2, x_initial, x_last in anchors:

        def diameter(x):
            return d1 + (d2 - d1) * (x - x1) / (x2 - x1)

        ratio = (diameter(x_last) / diameter(x_initial)) ** 3
        target = last[precursor, rh]
        out.writerow(
            [
                precursor,
                rh,
                x1,
                x2,
                d1,
                d2,
                x_initial,
                x_last,
                round(diameter(x_initial), 2),
                round(diameter(x_last), 2),
                round(ratio, 5),
                target,
                round(ratio - target, 5),
            ]
        )
