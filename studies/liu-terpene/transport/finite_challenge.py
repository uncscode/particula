# %%
"""Test the proposed two-size discrimination under finite vapor buffering."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from internal_supply import HERE, ROOT, ref


def simulate(fractions, name, diameter, number, tau, times, params, steps=420):
    """Integrate particle ⇄ gas → wall in each volatility bin. Time is in s internally."""
    times = np.asarray(times)
    grid = np.unique(np.r_[0.0, np.geomspace(0.0001, max(times), steps), times])
    events = {t: i for i, t in enumerate(times)}
    result = np.empty((len(params), len(times)))
    f = params.pool.to_numpy()[:, None]
    p = fractions[None, :] * (1 - f)
    p[:, 0] += f[:, 0]
    g = np.zeros_like(p)
    d0 = params.logD.to_numpy()[:, None]
    slope = params.slope.to_numpy()[:, None]
    rho = ref.DENSITY[name]
    r0 = diameter * 0.5e-9
    initial_mass = (4 * np.pi / 3) * rho * r0**3
    w = 1 / tau

    def coefficients(pop):
        # Same 1-nm radius disappearance criterion as the finite-box reference.
        # An extinct population has no receiving surface for condensation.
        total = np.maximum(pop.sum(axis=1, keepdims=True), (1e-9 / r0) ** 3)
        r = r0 * np.cbrt(total)
        k, kelvin = ref.kinetics(r.ravel(), rho, ref.Scenario())
        a = (k * kelvin * 1e-9 / initial_mass)[:, None] * ref.CSTAR / total
        resistance = r * r * 10.0 ** (-d0 + slope * (1 - total)) / np.pi**2
        hindrance = 1 / (1 + resistance * a)
        return a * hindrance, k[:, None] * number * 1e6 * hindrance

    def step(a, b, dt, p, g):
        # Exact two-reservoir propagator at frozen radius and composition.
        diag = a + b + w
        fast = -0.5 * (diag + np.sqrt((a - w) ** 2 + b * (2 * (a + w) + b)))
        slow = a * w / fast
        exp_s = np.exp(slow * dt)
        off = exp_s * (-np.expm1((fast - slow) * dt)) / (slow - fast)
        newp = exp_s * p + off * ((-a - slow) * p + b * g)
        newg = exp_s * g + off * (a * p - (b + w + slow) * g)
        return newp, newg

    wall = np.zeros_like(p)
    initial = p.copy()
    min_deposited = 0.0
    for t0, t1 in zip(grid, grid[1:]):
        dt = 60 * (t1 - t0)
        extinct = p.sum(axis=1) <= (1e-9 / r0) ** 3
        pm, _ = step(*coefficients(p), dt * 0.5, p, g)
        pn, gn = step(*coefficients(pm), dt, p, g)
        extinct |= (pm.sum(axis=1) <= (1e-9 / r0) ** 3) | (
            pn.sum(axis=1) <= (1e-9 / r0) ** 3
        )
        if np.any(extinct):
            # Transfer the tiny residual particle mass to the gas on disappearance.
            gn[extinct] = (g[extinct] + p[extinct]) * np.exp(-w * dt)
            pn[extinct] = 0.0
        increment = p + g - pn - gn
        min_deposited = min(min_deposited, increment.min())
        wall += increment
        p, g = pn, gn
        if t1 in events:
            result[:, events[t1]] = p.sum(axis=1)
    assert min_deposited > -1e-12 and np.min(p) > -1e-12 and np.min(g) > -1e-12
    assert np.max(abs(p + g + wall - initial)) < 1e-10
    return result


def run(out, weights_path, diameter=90.0):
    out.mkdir(parents=True, exist_ok=True)
    weights = pd.read_csv(weights_path)
    obs = pd.read_csv(ROOT / "sources/observations.csv")
    data = obs.query("precursor=='ocimene' and rh=='wet'").sort_values(
        "time_min"
    )
    t = data.time_min.to_numpy()
    y = data.vfr.to_numpy()
    models = pd.DataFrame(
        [
            ("interior", d, s, 0.0)
            for d in np.arange(-22, -14.9, 0.25)
            for s in np.arange(0, 16.1, 0.5)
        ]
        + [("pool", np.inf, 0.0, f) for f in np.linspace(0, 0.8, 161)],
        columns=["model", "logD", "slope", "pool"],
    )
    transfer = []
    check = []
    for (tau, num), grp in weights.query(
        "cyclic=='pinene' and rh=='wet'"
    ).groupby(["wall_lifetime_s", "cyclic_number_cm3"]):
        f = grp.sort_values("cstar_ug_m3").fitted_fraction.to_numpy()
        f /= f.sum()
        for study_count in (100.0, 200.0):
            pred = simulate(f, "ocimene", 220.0, study_count, tau, t, models)
            early = np.sqrt(
                np.mean((pred[:, t <= 300] - y[t <= 300]) ** 2, axis=1)
            )
            late = np.max(abs(pred[:, t > 900] - y[t > 900]), axis=1)
            for treatment, rows in models.groupby("model"):
                idx = rows.index.to_numpy()
                best = idx[np.argmin(early[idx])]
                for band in (0.025, 0.05):
                    kept = idx[
                        (early[idx] <= early[best] + 0.01) & (late[idx] <= band)
                    ]
                    for count_basis in ("same number", "same mass"):
                        next_count = (
                            study_count
                            if count_basis == "same number"
                            else study_count * (220.0 / diameter) ** 3
                        )
                        forecast = (
                            simulate(
                                f,
                                "ocimene",
                                diameter,
                                next_count,
                                tau,
                                [1200.0],
                                models.loc[kept],
                            )
                            if len(kept)
                            else None
                        )
                        transfer.append(
                            dict(
                                wall_lifetime_s=tau,
                                cyclic_number_cm3=num,
                                acyclic_number_cm3=study_count,
                                next_test=count_basis,
                                new_number_cm3=next_count,
                                treatment=treatment,
                                band=band,
                                day_best_rmse=early[best],
                                night_best_day_max=late[best],
                                n=len(kept),
                                forecast_low=forecast.min()
                                if forecast is not None
                                else None,
                                forecast_high=forecast.max()
                                if forecast is not None
                                else None,
                            )
                        )
                fine = simulate(
                    f,
                    "ocimene",
                    220.0,
                    study_count,
                    tau,
                    t,
                    models.loc[[best]],
                    1260,
                )[0]
                assert max(abs(fine - pred[best])) < 0.0006
                check.append(
                    dict(
                        tau=tau,
                        nref=num,
                        ntarget=study_count,
                        treatment=treatment,
                        refinement=max(abs(fine - pred[best])),
                    )
                )
            zero = models.query("model=='pool' and pool==0").index[0]
            oracle, budget = ref.finite_sink(
                f, "ocimene", "wet", 220.0, t, ref.Scenario(), tau, study_count
            )
            assert max(abs(oracle - pred[zero])) < 0.0006
            check.append(
                dict(
                    tau=tau,
                    nref=num,
                    ntarget=study_count,
                    treatment="zero_resistance_LSODA",
                    refinement=max(abs(oracle - pred[zero])),
                )
            )
    assert len(transfer) == 96
    pd.DataFrame(transfer).to_csv(
        out / "finite_size_challenge.csv", index=False, float_format="%.8g"
    )
    pd.DataFrame(check).to_csv(
        out / "finite_verification.csv", index=False, float_format="%.8g"
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "model/results/finite_weights.csv",
    )
    cli.add_argument("--out", type=Path, default=HERE / "results")
    cli.add_argument(
        "--diameter", type=float, choices=(60.0, 90.0), default=90.0
    )
    args = cli.parse_args()
    run(args.out, args.weights, args.diameter)
