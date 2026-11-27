# %% Is the dry farnesene miss forced by viscosity or by the assumed inventory?
"""Train one parameter on three daytime sizes and forecast the overnight point."""

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from radial import KB, R0, T, gas_speeds, inputs, run, write
from scipy.integrate import quad
from scipy.optimize import minimize_scalar


def experiment(packet, out, training_n, check_n):
    f, t, y = inputs(packet)
    start = time.monotonic()
    totals = []
    observations = []
    calls = 0
    for eta, a in ((1.4e8, 0.3), (1.4e8, 0.6), (2.8e8, 0.6)):
        D = KB * T / (6 * np.pi * eta * a * 1.0e-9)
        for mode in ("surface_thickness_nm", "fraction_highest_volatility"):

            @lru_cache(None)
            def predict(value, n):
                nonlocal calls
                calls += 1
                weights = f.copy()
                h = 0.0
                if mode == "surface_thickness_nm":
                    h = value
                else:
                    weights *= 1 - value
                    weights[-1] += value
                tt, V, m = run(weights, t, D, n, h, 1)
                return np.interp(t, tt, V), weights, m

            def objective(param):
                pred, _, _ = predict(float(param), training_n)
                return float(np.mean((pred[t < 300] - y[t < 300]) ** 2))

            fit = minimize_scalar(
                objective,
                bounds=(0, 9.0 if mode.startswith("surface") else 0.65),
                method="bounded",
                options={
                    "xatol": 0.004 if mode.startswith("surface") else 0.0004
                },
            )
            val = float(fit.x)
            trained, _, _ = predict(val, training_n)
            pred, weights, m = predict(val, check_n)
            tags = dict(
                mode=mode, eta_Pas=eta, solute_radius_nm=a, D_m2_s=D, param=val
            )
            totals.append(
                dict(
                    **tags,
                    cyclic_cstar100_fraction=f[-1],
                    case_cstar100_fraction=weights[-1],
                    day_rmse=float(
                        np.mean((pred[t < 300] - y[t < 300]) ** 2) ** 0.5
                    ),
                    night_error=pred[-1] - y[-1],
                    training_shells=training_n,
                    forecast_shells=check_n,
                    max_mesh_shift=np.max(abs(trained - pred)),
                    **m,
                )
            )
            observations.extend(
                dict(
                    **tags,
                    time_min=ti,
                    observed=obs,
                    model=mod,
                    signed_error=mod - obs,
                )
                for ti, obs, mod in zip(t, y, pred)
            )
            write(out / "condition-metrics.csv", totals)
            write(out / "condition.csv", observations)

    # A homogeneous single-component droplet has no concentration gradient. Its
    # pure VBS-bin flux erodes the surface directly, irrespective of D. The
    # high-C* time to 1% initial volume stays above the continuum radius floor.
    def minutes_to(target, component):
        return (
            quad(
                lambda r: 1 / gas_speeds(r)[component],
                R0 * np.cbrt(target),
                R0,
                epsabs=1e-9,
                epsrel=1e-9,
            )[0]
            / 60
        )

    controls = [
        dict(component=bin, target_vfr=v, time_min=minutes_to(v, i))
        for i, bin in ((6, 100.0), (0, 0.0001))
        for v in (0.84654, 0.77633, 0.67603, 0.01)
    ]
    write(out / "pure-edge.csv", controls)
    sys.path.insert(0, str(packet / "model"))
    import chamber

    oracle = chamber.predict(
        f, "farnesene", "dry", 140.0, t, chamber.SCENARIOS["primary"], n=10000
    )
    tt, v, c = run(f, t, KB * T / (6 * np.pi * 1e3 * 0.45e-9), 32)
    calls += 1
    diff = np.interp(t, tt, v) - oracle
    write(
        out / "rapid-check.csv",
        [
            dict(
                time_min=ti,
                radial=yp,
                well_mixed=yp - delv,
                signed_difference=delv,
            )
            for ti, yp, delv in zip(t, np.interp(t, tt, v), diff)
        ],
    )
    (out / "condition-runtime.json").write_text(
        json.dumps(
            dict(radial_runs=calls, total_s=time.monotonic() - start), indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("packet", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--training-n", type=int, default=128)
    p.add_argument("--check-n", type=int, default=224)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    experiment(a.packet, a.out, a.training_n, a.check_n)
