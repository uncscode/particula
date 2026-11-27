# %%
"""Forecast dry-organic VFR with a reduced particle-interior supply resistance."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "model"))
import chamber as ref


def candidates():
    rows = [
        (name, d, g, 0.0)
        for d in np.arange(-23.0, -14.99, 0.2)
        for g in np.arange(0.0, 10.01, 0.25)
        for name in ["constant_D" if g == 0 else "evolving_D"]
    ]
    rows += [
        ("long_lived_fraction", np.inf, 0.0, f)
        for f in np.linspace(0.0, 0.8, 161)
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "treatment",
            "log10_D0",
            "loss_sensitivity",
            "long_lived_fraction",
        ],
    )


def simulate(
    weights, name, diam, times, params, steps, scenario=None, rh="dry"
):
    """Apply midstep exponential depletion with the spherical relaxation time.

    The particle is a sphere of dry organic material. Interior resistance is
    an effective first-mode supply approximation with radius r(t), not a
    spatial diffusion solution. Surface/bulk composition and phase gradients
    are unobserved. Remaining particles shrink at constant dry density.
    """
    f = params.long_lived_fraction.to_numpy()[:, None]
    mass = weights[None, :] * (1 - f)
    mass[:, 0] += f[:, 0]
    times = np.array(times)
    grid = np.unique(np.r_[0.0, np.geomspace(0.0001, max(times), steps), times])
    records = np.empty((len(params), len(times)))
    events = {t: np.flatnonzero(times == t) for t in np.unique(times)}
    logd = params.log10_D0.to_numpy()
    g = params.loss_sensitivity.to_numpy()
    scenario = scenario or ref.Scenario()
    rho, water_ratio, organic_share = ref.particle_properties(
        name, rh, scenario
    )
    r0 = diam * 0.5e-9
    m0 = (4 * np.pi / 3) * rho * r0**3

    # Constant-coefficient gas/transition law matches the registered Particula
    # chamber reference. Evaluate K and Kelvin along the moving radius.
    def rate(pop):
        total = np.sum(pop, axis=1)
        r = r0 * np.cbrt(total * (1 + water_ratio))
        k, kelvin = ref.kinetics(r, rho, scenario)
        surface_time = (m0 * total / (k * kelvin * organic_share * 1e-9))[
            :, None
        ] / ref.CSTAR
        supply_time = r**2 * 10 ** (-logd + g * (1 - total)) / np.pi**2
        return 60.0 / (surface_time + supply_time[:, None])

    if 0.0 in events:
        records[:, events[0.0]] = 1.0
    for t0, t1 in zip(grid, grid[1:]):
        dt = t1 - t0
        mid = mass * np.exp(-0.5 * dt * rate(mass))
        mass *= np.exp(-dt * rate(mid))
        if t1 in events:
            records[:, events[t1]] = mass.sum(axis=1)[:, None]
    assert np.isfinite(records).all() and np.all(
        (records > 0) & (records < 1.0000001)
    )
    return records


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    fits = pd.read_csv(ROOT / "model/results/fits.csv")
    obs = pd.read_csv(ROOT / "sources/observations.csv")
    params = candidates()
    bests, hold, transfers, audit = [], [], [], []
    for name in ("ocimene", "farnesene"):
        for rh in ("dry", "wet"):
            family = ref.CYCLIC[name]
            weights = (
                fits.query(
                    "model=='primary' and precursor==@family and rh==@rh"
                )
                .sort_values("cstar_ug_m3")
                .fitted_fraction.to_numpy()
            )
            weights /= weights.sum()
            data = obs.query("precursor==@name and rh==@rh").sort_values(
                "time_min"
            )
            t = data.time_min.to_numpy()
            y = data.vfr.to_numpy()
            day = t <= 300
            night = t > 900
            pred = simulate(weights, name, ref.SIZE[name], t, params, 410)
            oracle = ref.predict(
                weights, name, rh, ref.SIZE[name], t, ref.Scenario(), n=5200
            )
            base = params.index[
                (params.treatment == "long_lived_fraction")
                & (params.long_lived_fraction == 0.0)
            ][0]
            assert np.max(abs(pred[base] - oracle)) < 0.0003
            early = np.sqrt(np.mean((pred[:, day] - y[day]) ** 2, axis=1))
            late = np.mean(abs(pred[:, night] - y[night]), axis=1)
            for method, subset in params.groupby("treatment"):
                idx = subset.index.to_numpy()
                i = idx[np.argmin(early[idx])]
                # Keep all parameters within one hundredth of the best day RMSE
                # for that family. This is a sensitivity screen, not a CI.
                close = idx[early[idx] <= early[i] + 0.01]
                # Separate posterior prediction of new diameters using all
                # existing information. Drop candidates >0.025 off at night.
                conditioned = close[
                    np.max(abs(pred[close][:, night] - y[night]), axis=1)
                    <= 0.025
                ]
                refined = simulate(
                    weights, name, ref.SIZE[name], t, params.loc[[i]], 1230
                )[0]
                check = np.max(abs(refined - pred[i]))
                assert check < 0.0004
                meta = dict(precursor=name, rh=rh, treatment=method)
                bests.append(
                    dict(
                        **meta,
                        **params.loc[i].drop("treatment").to_dict(),
                        training_rmse=early[i],
                        overnight_mae=late[i],
                        grid_count=len(idx),
                        near_count=len(close),
                        near_overnight_mae_min=late[close].min(),
                        near_overnight_mae_max=late[close].max(),
                        conditioned_count=len(conditioned),
                    )
                )
                for j in np.flatnonzero(night):
                    hold.append(
                        dict(
                            **meta,
                            time_min=t[j],
                            observed=y[j],
                            best=pred[i, j],
                            day_near_min=pred[close, j].min(),
                            day_near_max=pred[close, j].max(),
                        )
                    )
                audit.append(
                    dict(
                        **meta,
                        refinement_max=check,
                        zero_resistance_oracle=np.max(abs(pred[base] - oracle)),
                    )
                )
                if not len(conditioned):
                    continue
                for diameter in (90.0, 120.0, 165.0, 220.0):
                    targets = [30.0, 200.0, 600.0, 1200.0]
                    new = simulate(
                        weights,
                        name,
                        diameter,
                        targets,
                        params.loc[conditioned],
                        410,
                    )
                    for j, tj in enumerate(targets):
                        transfers.append(
                            dict(
                                **meta,
                                diameter_nm=diameter,
                                time_min=tj,
                                low=new[:, j].min(),
                                high=new[:, j].max(),
                                n=len(conditioned),
                            )
                        )
    for n, rows in [
        ("fits", bests),
        ("overnight", hold),
        ("transfer", transfers),
        ("verification", audit),
    ]:
        pd.DataFrame(rows).to_csv(
            out / (n + ".csv"), index=False, float_format="%.8g"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    run(parser.parse_args().out)
