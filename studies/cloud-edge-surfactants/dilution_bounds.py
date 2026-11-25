"""Bracket unobserved pendant-drop tensions using adjacent measurements."""

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent


def limits(rows, recipe, rh):
    """Find the nearest lower and upper measured tensions."""
    points = sorted(
        (float(r["rh_percent"]), float(r["sigma_mN_m"]))
        for r in rows
        if r["recipe"] == recipe and r["method"] == "pendant"
    )
    return (
        max(p for p in points if p[0] <= rh)[1],
        min(p for p in points if p[0] >= rh)[1],
    )


def main():
    """Write monotone tension bounds at the selected humidities."""
    with (HERE / "figure5.csv").open() as f:
        rows = list(csv.DictReader(f))
    with (HERE / "dilution_bounds.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(
            [
                "rh_pct",
                "octyl_min",
                "octyl_max",
                "dodecyl_min",
                "dodecyl_max",
                "octyl_minus_dodecyl_min",
                "octyl_minus_dodecyl_max",
            ]
        )
        for rh in (98.41085, 99.5, 100):
            o0, o1 = limits(rows, "octyl", rh)
            d0, d1 = limits(rows, "dodecyl", rh)
            w.writerow([rh, o0, o1, d0, d1, o0 - d1, o1 - d0])


if __name__ == "__main__":
    main()
