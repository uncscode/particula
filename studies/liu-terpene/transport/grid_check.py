# %%
"""Refine the parameter region entering the proposed 90 nm humid measurement."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from internal_supply import HERE, ROOT, ref, simulate


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    obs = pd.read_csv(ROOT / "sources/observations.csv").query(
        "precursor=='ocimene' and rh=='wet'"
    )
    t = obs.time_min.to_numpy()
    y = obs.vfr.to_numpy()
    fits = pd.read_csv(ROOT / "model/results/fits.csv")
    for_write = []
    for setting in (
        "primary",
        "low_accommodation",
        "zero_curvature",
        "high_tension",
        "humid_water",
    ):
        f = (
            fits.query("model==@setting and precursor=='pinene' and rh=='wet'")
            .sort_values("cstar_ug_m3")
            .fitted_fraction.to_numpy()
        )
        f /= f.sum()
        for kind in ("evolving_D", "long_lived_fraction"):
            if kind == "evolving_D":
                ds, gs = np.meshgrid(
                    np.arange(-22.0, -14.9, 0.2), np.arange(0.0, 16.01, 0.25)
                )
                rows = [
                    (kind, d, g, 0.0) for d, g in zip(ds.ravel(), gs.ravel())
                ]
            else:
                rows = [
                    (kind, np.inf, 0.0, v) for v in np.linspace(0.0, 0.8, 161)
                ]
            cols = [
                "treatment",
                "log10_D0",
                "loss_sensitivity",
                "long_lived_fraction",
            ]

            def evaluate(params):
                runs = [
                    simulate(
                        f,
                        "ocimene",
                        220.0,
                        t,
                        params.iloc[i : i + 1500],
                        420,
                        ref.SCENARIOS[setting],
                        "wet",
                    )
                    for i in range(0, len(params), 1500)
                ]
                pred = np.vstack(runs)
                return np.sqrt(
                    ((pred[:, t <= 300] - y[t <= 300]) ** 2).mean(axis=1)
                ), np.max(abs(pred[:, t > 900] - y[t > 900]), axis=1)

            coarse = pd.DataFrame(rows, columns=cols)
            day, late = evaluate(coarse)
            if kind == "evolving_D":
                # Include a guard band outside both selection contours and
                # around the unconstrained early optimum, which sets the cut.
                near = coarse.loc[
                    ((day < day.min() + 0.035) & (late < 0.1))
                    | (day < day.min() + 0.003)
                ]
                cells = {
                    (
                        round(r.log10_D0 + di, 5),
                        round(r.loss_sensitivity + gi, 5),
                    )
                    for r in near.itertuples()
                    for di in np.arange(-0.2, 0.201, 0.05)
                    for gi in np.arange(-0.25, 0.251, 0.0625)
                }
                fine = pd.DataFrame(
                    [(kind, d, g, 0.0) for d, g in sorted(cells) if g >= 0],
                    columns=cols,
                )
            else:
                fine = pd.DataFrame(
                    [
                        (kind, np.inf, 0.0, v)
                        for v in np.linspace(0.0, 0.8, 1281)
                    ],
                    columns=cols,
                )
            for label, params in [("coarse", coarse), ("refined", fine)]:
                ed, el = evaluate(params)
                for limit in (0.025, 0.05):
                    keep = params.loc[(ed <= ed.min() + 0.01) & (el <= limit)]
                    vals = simulate(
                        f,
                        "ocimene",
                        90.0,
                        [1200.0],
                        keep,
                        500,
                        ref.SCENARIOS[setting],
                        "wet",
                    )[:, 0]
                    for_write.append(
                        dict(
                            setting=setting,
                            treatment=kind,
                            resolution=label,
                            night_limit=limit,
                            n_trials=len(params),
                            n_kept=len(keep),
                            low=vals.min(),
                            high=vals.max(),
                        )
                    )
    pd.DataFrame(for_write).to_csv(
        out / "grid_resolution.csv", index=False, float_format="%.8g"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=HERE / "results")
    run(p.parse_args().out)
