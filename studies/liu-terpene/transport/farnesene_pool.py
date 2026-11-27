"""Fit one added low-volatility fraction to daytime humid farnesene and forecast night."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from internal_supply import ROOT, ref, simulate
from scipy.optimize import brentq, minimize_scalar


def run(out):
    fits = pd.read_csv(ROOT / "model/results/fits.csv")
    obs = pd.read_csv(ROOT / "sources/observations.csv").query(
        "precursor=='farnesene' and rh=='wet'"
    )
    times = obs.time_min.to_numpy()
    y = obs.vfr.to_numpy()
    params = pd.DataFrame(
        dict(
            log10_D0=np.inf,
            loss_sensitivity=0.0,
            long_lived_fraction=np.linspace(0.0, 0.8, 1281),
        )
    )
    rows = []
    for setting in (
        "primary",
        "low_accommodation",
        "zero_curvature",
        "high_tension",
        "humid_water",
    ):
        w = (
            fits.query(
                "model==@setting and precursor=='caryophyllene' and rh=='wet'"
            )
            .sort_values("cstar_ug_m3")
            .fitted_fraction.to_numpy()
        )
        w /= w.sum()
        sc = ref.SCENARIOS[setting]
        native = simulate(w, "farnesene", 140.0, times, params, 650, sc, "wet")
        for earliest in (0.0, 30.0):
            mask = (times >= earliest) & (times <= 300.0)
            rmse = np.sqrt(((native[:, mask] - y[mask]) ** 2).mean(axis=1))
            near = rmse <= rmse.min() + 0.01

            def independent_curve(fraction):
                mixture = w * (1.0 - fraction)
                mixture[0] += fraction
                return ref.predict(
                    mixture, "farnesene", "wet", 140.0, times, sc, n=10400
                )

            def day_error(fraction):
                return np.sqrt(
                    np.mean((independent_curve(fraction)[mask] - y[mask]) ** 2)
                )

            optimum = minimize_scalar(
                day_error,
                bounds=(0.0, 0.8),
                method="bounded",
                options={"xatol": 1e-9},
            )
            cutoff = optimum.fun + 0.01
            lowest = brentq(lambda f: day_error(f) - cutoff, 0.0, optimum.x)
            assert np.all(np.diff(native[:, times > 900].ravel()) >= 0)
            continuous_min_error = (
                independent_curve(lowest)[times > 900] - y[times > 900]
            ).item()
            idx = rmse.argmin()
            errs = native[near][:, times > 900] - y[times > 900]
            # Keep a daytime-only forecast even when the overnight screen rejects it.
            for size in (60, 90, 120, 140, 165, 220):
                pred = simulate(
                    w,
                    "farnesene",
                    size,
                    [1200.0],
                    params.iloc[[idx]],
                    650,
                    sc,
                    "wet",
                )[0, 0]
                rows.append(
                    (
                        setting,
                        earliest,
                        size,
                        params.long_lived_fraction.iloc[idx],
                        rmse[idx],
                        errs.min(),
                        errs.max(),
                        continuous_min_error,
                        pred,
                    )
                )
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows,
        columns="setting first_day_min diameter_nm best_added_fraction best_day_rmse near_day_night_error_min near_day_night_error_max continuous_night_error_min day_only_20h".split(),
    ).to_csv(out, index=False, float_format="%.9g")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args().out)
