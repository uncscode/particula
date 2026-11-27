"""Calibrate against binary water and ammonium sulfate. No NaCl phase targets enter."""

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from trial import HERE, prepare, binary_prediction
from stability import grid

C2_GRID = (0.0, 0.003, 0.006081, 0.010)
TAIL_GRID = np.linspace(0, 1.5, 31)
NH4_GRID = np.r_[0.0, np.linspace(0.01, 0.30, 15), np.linspace(0.34, 1.0, 12)]
TEMPERATURE = 275.15


def fit_binary(tail, c2, point_scale=1.0):
    data = pd.read_csv(HERE / "binary-water.csv")
    scale = data.a_w_meter_accuracy.fillna(0.01 * point_scale).to_numpy() / data.a_w
    base = binary_prediction(tail, [0, 0, c2], data)
    design = np.stack(
        [binary_prediction(tail, v, data) - base for v in [[1, 0, c2], [0, 1, c2]]],
        axis=1,
    )
    y = (np.log(data.a_w) - base) / scale
    matrix = design / scale.to_numpy()[:, None]
    sol = lsq_linear(matrix, y, bounds=([0, -5], [2, 5]), tol=1e-12)
    return (*sol.x, c2), sum((matrix @ sol.x - y) ** 2)


def fast_inputs(g, h):
    a = g.energies[:, None] - g.trial_coordinates @ g.feed_mu.T
    b = (h.energies - g.energies)[:, None] - g.trial_coordinates @ (
        h.feed_mu - g.feed_mu
    ).T
    return a, b, g.feed_mu[:, 0], h.feed_mu[:, 0] - g.feed_mu[:, 0]


def fast_onset(cache, delta):
    a, b, aw, slope = cache
    values = a + delta * b
    # Look for the first off-feed phase with lower Gibbs energy.
    d = values.min(axis=0)
    hits = np.flatnonzero(d < -1e-5)
    if not len(hits) or hits[0] == 0:
        return np.nan
    j = hits[0]
    i = j - 1
    w = np.argmin(values[:, j])
    v0 = values[w, i]
    v1 = values[w, j]
    t = np.clip((v0 + 1e-5) / (v0 - v1), 0, 1)
    y = aw + delta * slope
    return 100 * np.exp((1 - t) * y[i] + t * y[j])


def main():
    targets = (
        pd.read_csv(HERE / "phase-data.csv").query("salt=='AS'").set_index("ovf_pct")
    )
    records = []
    for tail in TAIL_GRID:
        for c2 in C2_GRID:
            sodium, chi = fit_binary(tail, c2)
            models = [prepare("AS", tail, sodium, step) for step in [0.0, 1.0]]
            caches = [
                fast_inputs(
                    *[grid("AS", v, TEMPERATURE, m, maxmol=160.0) for m in models]
                )
                for v in targets.index
            ]
            for delta in NH4_GRID:
                estimates = np.array([fast_onset(c, delta) for c in caches])
                # The fit must produce a finite onset for all three measured AS mixtures.
                loss = np.inf
                if np.all(np.isfinite(estimates)):
                    loss = chi + np.sum(
                        ((estimates - targets.srh_pct) / targets.srh_sd_pct) ** 2
                    )
                records.append([tail, *sodium, delta, chi, loss, *estimates])
    df = pd.DataFrame(
        records,
        columns=[
            "tail",
            "B1",
            "B2",
            "C2",
            "NH4_delta",
            "binary_loss",
            "training_loss",
            "AS50",
            "AS70",
            "AS90",
        ],
    )
    df.to_csv(HERE / "calibration.csv", index=False)
    print(df.nsmallest(5, "training_loss").to_string(index=False))


if __name__ == "__main__":
    main()
