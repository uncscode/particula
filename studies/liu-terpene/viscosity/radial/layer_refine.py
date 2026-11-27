# %% Refine the slowest-viscosity-corner shell forecast before using it.
"""Train the shell thickness only on daytime points for each grid, then predict night."""

import json
import time
from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path

import numpy as np
from radial import KB, T, inputs, run, write
from scipy.optimize import minimize_scalar


def study(packet, out):
    t0 = time.monotonic()
    f, t, y = inputs(packet)
    D = KB * T / (6 * np.pi * 2.8e8 * 0.6e-9)
    metrics = []
    rows = []
    nsolve = 0
    for n in (128, 224, 320):

        @lru_cache(None)
        def result(h, nr):
            nonlocal nsolve
            nsolve += 1
            tt, V, s = run(f, t, D, nr, h, 1)
            return np.interp(t, tt, V), s

        def objective(h):
            p, _ = result(float(h), n)
            return np.mean((p[:-1] - y[:-1]) ** 2)

        scan = minimize_scalar(
            objective,
            method="bounded",
            bounds=(1.0, 7.0),
            options={"xatol": 0.003},
        )
        h = float(scan.x)
        for m in (n, 480):
            forecast, s = result(h, m)
            meta = dict(training_shells=n, forecast_shells=m, thickness_nm=h)
            rows.extend(
                dict(
                    **meta,
                    time_min=ti,
                    observed=obs,
                    predicted=val,
                    signed_error=val - obs,
                )
                for ti, obs, val in zip(t, y, forecast)
            )
            metrics.append(
                dict(
                    **meta,
                    day_rmse=np.sqrt(np.mean((forecast[:-1] - y[:-1]) ** 2)),
                    night_error=forecast[-1] - y[-1],
                    **s,
                )
            )
            write(out / "layer-refine.csv", rows)
            write(out / "layer-refine-metrics.csv", metrics)
    (out / "layer-refine-runtime.json").write_text(
        json.dumps(
            dict(radial_runs=nsolve, total_s=time.monotonic() - t0), indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("packet", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    study(a.packet, a.out)
