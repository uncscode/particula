"""Reconstruct the observed spectrum. Color is left-censored in occupied cells."""

import numpy as np
from scipy.special import factorial

from workspace import measurement

D = np.arange(10.5, 80, 1.0)
BANDS = np.array([12, 20, 22, 26, 30, 34, 38, 42, 46, 48, 50, 65.0])
MID = np.flatnonzero((BANDS[:-1] >= 22) & (BANDS[1:] <= 48))
TAIL = [0, len(BANDS) - 2]


def load(run, clip="occupancy"):
    c = measurement("cells.csv")
    c = c[c.run == run]
    t = measurement(f"times-{run}.csv").t_s.to_numpy()
    i = np.searchsorted(t, c.t_s.to_numpy())
    j = (c.D_um - 10.5).astype(int).to_numpy()
    mat = np.zeros((len(t), len(D)))
    mat[i, j] = c.density
    floor = np.zeros_like(mat, dtype=bool)
    floor[i, j] = c.floor.astype(bool)
    quantum = 2.03 / D
    max_n = np.maximum(1, np.floor(0.104 / quantum)).astype(int)
    for block in np.unique((t * 4).astype(int)):
        idx = (t * 4).astype(int) == block
        occ = (mat[idx] > 0).sum(axis=0)
        lam = -np.log(1 - (occ + 0.5) / (idx.sum() + 1))
        n = np.arange(1, max_n.max() + 1)[:, None]
        p = (lam**n / factorial(n)) * (n <= max_n)
        if clip == "low":
            mult = np.ones_like(D)
        elif clip == "high":
            mult = max_n
        else:
            mult = (p * n).sum(axis=0) / p.sum(axis=0)
        mat[idx] = np.where(floor[idx], quantum * mult, mat[idx])
    return t, mat


def average(t, mat, start, stop):
    rows = (t >= start) & (t < stop)
    if not rows.any():
        raise ValueError("empty time block")
    return mat[rows].mean(axis=0)
