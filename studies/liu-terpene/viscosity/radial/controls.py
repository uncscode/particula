# %% Check interior diffusivity with the exact release of a dilute tracer.
"""An absorbing surface in an inert sphere has a known transient solution."""

import json
import time
from argparse import ArgumentParser
from pathlib import Path
from unittest.mock import patch

import numpy as np
import radial


def oracle(out):
    start = time.monotonic()
    # A dilute tracer reduces interface motion to O(eps) and separates the
    # known diffusion solution from gas-transfer and interspecies effects.
    eps = 1.0e-4
    f = np.r_[1 - eps - 5.0e-14, np.full(5, 1.0e-14), eps]
    fo = np.array([0.0003, 0.003, 0.03, 0.15])
    D = 3.0e-21
    t = fo * radial.R0**2 / D / 60
    lam = np.arange(1, 1001, dtype=float) ** 2 * np.pi**2
    exact = 6 * np.sum(np.exp(-fo[:, None] * lam) / lam, axis=1)
    ans = []
    for n in (32, 80, 160, 320):
        # The dilute tracer's surface is removed instantaneously on the
        # diffusion timescale. Host and five trace lanes have no vapor flux.
        with patch.object(
            radial, "gas_speeds", lambda r: np.r_[np.zeros(6), 1.0e-4]
        ):
            tt, v, s = radial.run(f, t, D, n, rtol=1.0e-8, first_step=1.0e-4)
        remaining = 1 - (1 - np.interp(t, tt, v)) / eps
        for x, y, z in zip(fo, exact, remaining):
            ans.append(
                dict(
                    shells=n,
                    Fourier=x,
                    analytical_tracer_fraction=y,
                    simulation_tracer_fraction=z,
                    difference=z - y,
                )
            )
    radial.write(out / "absorbing-sphere.csv", ans)
    if max(abs(r["difference"]) for r in ans if r["shells"] == 320) > 0.0005:
        raise ValueError("Resolved dilute-tracer check failed")
    (out / "control-runtime.json").write_text(
        json.dumps(
            dict(runs=4, wall_seconds=time.monotonic() - start), indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    oracle(a.out)
