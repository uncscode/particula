"""Paired calibration scenarios for the Dunne archive and cold forward runs."""

import csv
from pathlib import Path

from scipy.stats import qmc

HERE = Path(__file__).resolve().parent


def draws():
    """Four independent scrambled Sobol designs, 32 conditions each."""
    rows = []
    factors = dict(
        acid=1.5,
        base0=10.0,
        base1=2.0,
        base2=1.5,
        base3=1.2,
        jscale=2.5,
        wall=2.0,
    )
    for block, seed in enumerate((31971, 41983, 52999, 63103)):
        points = qmc.Sobol(d=16, scramble=True, seed=seed).random_base2(5)
        for k, point in enumerate(points):
            row = dict(id=32 * block + k, block=block)
            row.update(
                {
                    name: factor ** (2 * u - 1)
                    for (name, factor), u in zip(factors.items(), point)
                }
            )
            row.update(
                {f"u{i}": u for i, u in enumerate(point[len(factors) :])}
            )
            rows.append(row)
    return rows


if __name__ == "__main__":
    rows = draws()
    with (HERE / "draws.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
