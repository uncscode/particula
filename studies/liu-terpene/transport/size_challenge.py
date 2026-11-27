# %%
"""Repeat the proposed same-fill size test with alternative cyclic calibrations."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from internal_supply import HERE, ROOT, ref, simulate


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    obs = pd.read_csv(ROOT / "sources/observations.csv")
    fits = pd.read_csv(ROOT / "model/results/fits.csv")
    result, viscosity = [], []
    # Widen the gamma range beyond the primary grid and include changes to
    # curvature, organic water uptake and accommodation with consistent refits.
    parameters = [
        ("evolving_D", d, g, 0.0)
        for d in np.arange(-22, -14.9, 0.2)
        for g in np.arange(0.0, 16.1, 0.25)
    ]
    parameters += [
        ("long_lived_fraction", np.inf, 0.0, f)
        for f in np.linspace(0.0, 0.8, 161)
    ]
    models = pd.DataFrame(
        parameters,
        columns=[
            "treatment",
            "log10_D0",
            "loss_sensitivity",
            "long_lived_fraction",
        ],
    )
    for name, rh, sets in [
        (
            "ocimene",
            "wet",
            [
                "primary",
                "low_accommodation",
                "zero_curvature",
                "high_tension",
                "humid_water",
            ],
        ),
        ("farnesene", "dry", ["primary"]),
    ]:
        data = obs.query("precursor==@name and rh==@rh").sort_values("time_min")
        t = data.time_min.to_numpy()
        y = data.vfr.to_numpy()
        family = ref.CYCLIC[name]
        all_times = np.r_[t, 15.0]
        for setting in sets:
            sc = ref.SCENARIOS[setting]
            f = (
                fits.query("model==@setting and precursor==@family and rh==@rh")
                .sort_values("cstar_ug_m3")
                .fitted_fraction.to_numpy()
            )
            f /= f.sum()
            pred = simulate(
                f, name, ref.SIZE[name], all_times, models, 360, sc, rh
            )
            at15 = pred[:, -1]
            pred = pred[:, :-1]
            early = np.sqrt(
                np.mean((pred[:, t <= 300] - y[t <= 300]) ** 2, axis=1)
            )
            late = np.max(np.abs(pred[:, t > 900] - y[t > 900]), axis=1)
            for kind, group in models.groupby("treatment"):
                idx = group.index.to_numpy()
                opt = idx[np.argmin(early[idx])]
                if name == "farnesene" and kind == "evolving_D":
                    d15 = 10.0 ** (
                        group.log10_D0.to_numpy()
                        - group.loss_sensitivity.to_numpy() * (1 - at15[idx])
                    )
                    for label, mask in [
                        ("best_day", idx == opt),
                        (
                            "day_and_night",
                            (early[idx] <= early[opt] + 0.01)
                            & (late[idx] <= 0.025),
                        ),
                        (
                            "measured_viscosity_SE",
                            (
                                d15
                                >= 1.380649e-23
                                * 297.15
                                / (6 * np.pi * 2.8e8 * 0.6e-9)
                            )
                            & (
                                d15
                                <= 1.380649e-23
                                * 297.15
                                / (6 * np.pi * 1.4e8 * 0.3e-9)
                            ),
                        ),
                    ]:
                        if np.any(mask):
                            viscosity.append(
                                dict(
                                    screen=label,
                                    n=sum(mask),
                                    D15_min=d15[mask].min(),
                                    D15_max=d15[mask].max(),
                                    min_early_rmse=early[idx][mask].min(),
                                    min_late_abs=late[idx][mask].min(),
                                )
                            )
                if name != "ocimene":
                    continue
                for band in (0.025, 0.05):
                    kept = idx[
                        (early[idx] <= early[opt] + 0.01) & (late[idx] <= band)
                    ]
                    if not len(kept):
                        continue
                    for d in (90.0, 120.0, 165.0, 220.0):
                        targets = [30.0, 200.0, 600.0, 1200.0]
                        additional = simulate(
                            f, name, d, targets, models.loc[kept], 460, sc, rh
                        )
                        for j, tj in enumerate(targets):
                            result.append(
                                dict(
                                    setting=setting,
                                    treatment=kind,
                                    night_tolerance=band,
                                    diameter_nm=d,
                                    time_min=tj,
                                    n=len(kept),
                                    min_vfr=additional[:, j].min(),
                                    max_vfr=additional[:, j].max(),
                                )
                            )
    pd.DataFrame(result).to_csv(
        out / "ocimene_size_challenge.csv", index=False, float_format="%.8g"
    )
    pd.DataFrame(viscosity).to_csv(
        out / "farnesene_viscosity.csv", index=False, float_format="%.8g"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    run(parser.parse_args().out)
