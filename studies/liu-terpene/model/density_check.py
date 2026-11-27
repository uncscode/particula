# %%
"""Density sensitivity of the cubed vacuum aerodynamic diameter ratio.

For spherical particles d_va/rho = geometric diameter times a constant.
If eta=rho(t)/rho0 then apparent_VFR=physical_VFR*eta^3=mass_fraction*eta^2.
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from chamber import DENSITY, HERE


def run(out):
    data = pd.read_csv(out / "series.csv")
    matched = pd.read_csv(out / "matched.csv").set_index(
        ["model", "precursor", "rh"]
    )
    rows = []
    acyclic = data[
        (data.model == "primary")
        & (data.precursor.isin(["ocimene", "farnesene"]))
    ]
    for (name, rh), g in acyclic.groupby(["precursor", "rh"]):
        g = g.sort_values("time_min")
        series = list(
            zip(
                g.time_min,
                g.observed_vfr,
                g.cyclic_prediction,
                ["marker"] * len(g),
            )
        )
        point = np.interp(200.0, g.time_min, g.observed_vfr)
        series.append(
            (
                200.0,
                point,
                matched.loc[("primary", name, rh), "cyclic_native"],
                "interpolated",
            )
        )
        for t, obs, pred, kind in series:
            ratio = obs / pred
            rows.append(
                (
                    name,
                    rh,
                    t,
                    kind,
                    obs,
                    pred,
                    ratio ** (1 / 3),
                    ratio**0.5,
                    DENSITY[name] * ratio**0.5 / 1000,
                )
            )
    pd.DataFrame(
        rows,
        columns="precursor rh time_min source observed_vfr frozen_density_cyclic_vfr density_ratio_at_fixed_geometric_volume density_ratio_at_fixed_mass terminal_density_at_fixed_mass_g_cm3".split(),
    ).to_csv(out / "density_sensitivity.csv", index=False, float_format="%.10g")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--out", type=Path, default=HERE / "results")
    run(p.parse_args().out)
