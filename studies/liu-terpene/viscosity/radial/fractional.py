# %% Compare two independent viscosity-to-diffusion laws in resolved particles.
"""Apply fixed ξ=.93 with 10^-3 Pa s reference viscosity or classic ξ=1."""

import json
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from radial import KB, T, inputs, run, write


def campaign(p, out):
    start = time.monotonic()
    f, t, y = inputs(p)
    rows = []
    metrics = []
    for xi in (1.0, 0.93):
        for eta, a in (
            (1.4e8, 0.3),
            (1.4e8, 0.6),
            (2.1e8, 0.4),
            (2.8e8, 0.3),
            (2.8e8, 0.6),
        ):
            se = KB * T / (6 * np.pi * eta * a * 1.0e-9)
            D = se * (eta / 0.001) ** (1 - xi)
            for n in (80, 160):
                tt, V, stats = run(f, t, D, n)
                pred = np.interp(t, tt, V)
                tags = dict(
                    xi=xi,
                    eta_Pas=eta,
                    solute_radius_nm=a,
                    D_classical=se,
                    D_m2_s=D,
                    shells=n,
                )
                rows.extend(
                    dict(
                        **tags,
                        time_min=ti,
                        observed=obs,
                        computed=vv,
                        signed_error=vv - obs,
                    )
                    for ti, obs, vv in zip(t, y, pred)
                )
                metrics.append(
                    dict(
                        **tags,
                        day_rmse=np.mean((pred[:-1] - y[:-1]) ** 2) ** 0.5,
                        night_error=pred[-1] - y[-1],
                        max_abs_error=np.max(abs(pred - y)),
                        **stats,
                    )
                )
    write(out / "fractional.csv", rows)
    write(out / "fractional-metrics.csv", metrics)
    (out / "fractional-runtime.json").write_text(
        json.dumps(
            dict(runs=len(metrics), wall_seconds=time.monotonic() - start),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("packet", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    campaign(a.packet, a.out)
