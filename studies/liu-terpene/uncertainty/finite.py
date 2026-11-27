"""Check the 60 nm ocimene design with finite vapor removal and a common starting size."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transport"))
from finite_challenge import simulate
from rescreen import HERE, ROOT, compatible


def run(bounds, out):
    read = (
        pd.read_csv(bounds)
        .query(
            "precursor=='ocimene' and status=='paired' and resolution=='half_bin'"
        )
        .sort_values("time_min")
    )
    times = read.spectrum_min.to_numpy()
    mask = (times <= 300) | (times > 900)
    w = pd.read_csv(ROOT / "model/results/finite_weights.csv").query(
        "cyclic=='pinene' and rh=='wet'"
    )
    params = pd.DataFrame(
        [
            ("evolving_D", d, s, 0.0)
            for d in np.arange(-22, -14.9, 0.25)
            for s in np.arange(0, 16.1, 0.5)
        ]
        + [
            ("long_lived_fraction", np.inf, 0.0, v)
            for v in np.linspace(0.0, 0.8, 161)
        ],
        columns=["kind", "logD", "slope", "pool"],
    )
    rows = []
    for (tau, num), group in w.groupby(
        ["wall_lifetime_s", "cyclic_number_cm3"]
    ):
        f = group.sort_values("cstar_ug_m3").fitted_fraction.to_numpy()
        f /= f.sum()
        for count in (100.0, 200.0):
            native = simulate(f, "ocimene", 220.0, count, tau, times, params)
            valid = compatible(native, read, mask)
            for k, subset in params[valid].groupby("kind"):
                for d in (60, 90):
                    vals = simulate(
                        f, "ocimene", d, count, tau, [1200.0], subset
                    )[:, 0]
                    rows.append(
                        (
                            tau,
                            num,
                            count,
                            k,
                            d,
                            len(subset),
                            vals.min(),
                            vals.max(),
                        )
                    )
            for k in sorted(set(params.kind) - set(params[valid].kind)):
                rows.extend(
                    (tau, num, count, k, d, 0, np.nan, np.nan) for d in (60, 90)
                )
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows,
        columns="lifetime_s cyclic_number_cm3 acyclic_number_cm3 mechanism diameter_nm accepted low high".split(),
    ).to_csv(out / "finite_vapor.csv", index=False, float_format="%.10g")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--bounds", type=Path, default=HERE / "results/read_bounds.csv"
    )
    cli.add_argument("--out", type=Path, default=HERE / "results")
    a = cli.parse_args()
    run(a.bounds, a.out)
