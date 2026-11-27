# %%
"""Post-prediction diagnostic: constant D that matches each marker separately."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fractional import (
    CROSSOVER_PA_S,
    DEFAULT_PACKET,
    FRACTIONAL_EXPONENT,
    diffusion,
)


def run(packet, out):
    sys.path.insert(0, str(packet / "transport"))
    from internal_supply import simulate

    data = (
        pd.read_csv(packet / "sources" / "observations.csv")
        .query("precursor=='farnesene' and rh=='dry'")
        .sort_values("time_min")
    )
    weights = pd.read_csv(
        out / "cyclic_composition.csv"
    ).fitted_fraction.to_numpy()
    weights /= weights.sum()
    logs = np.linspace(-22.0, -17.0, 501)
    params = pd.DataFrame(
        dict(log10_D0=logs, loss_sensitivity=0.0, long_lived_fraction=0.0)
    )
    pred = simulate(weights, "farnesene", 140.0, data.time_min, params, 800)
    assert np.all(np.diff(pred, axis=0) <= 0)
    results = []
    for j, (_, row) in enumerate(data.iterrows()):
        curve = pred[::-1, j]
        obs = row.vfr
        assert curve.min() < obs - 0.004 < obs + 0.004 < curve.max()
        # At a single time, infer the constant diffusivity which would yield
        # the integrated loss since t=0. Vertical ±.004 is a reading stress.
        logd = np.interp([obs - 0.004, obs, obs + 0.004], curve, logs[::-1])
        d = 10**logd
        etas = CROSSOVER_PA_S * (
            diffusion(CROSSOVER_PA_S, 0.4, FRACTIONAL_EXPONENT) / d
        ) ** (1 / FRACTIONAL_EXPONENT)
        results.append((row.time_min, obs, d[2], d[1], d[0], etas[1]))
    pd.DataFrame(
        results,
        columns="time_min apparent_vfr D_for_vfr_plus004 D_for_vfr D_for_vfr_minus004 equivalent_eta_Pa_s_for_0p4nm".split(),
    ).to_csv(out / "marker_equivalent.csv", index=False, float_format="%.11g")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    p.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    a = p.parse_args()
    run(a.packet, a.out)
