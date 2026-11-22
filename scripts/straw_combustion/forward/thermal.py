"""Thermal histories for equal-flux annuli of the tubular reactor."""

from pathlib import Path

import numpy as np

INPUT = Path(__file__).resolve().parents[1] / "inputs"


def temperature_history(dt, annuli):
    """Return midpoint properties, time steps (s) and exit flux weights."""
    if not np.isfinite(dt) or dt <= 0 or not 1 <= annuli <= 64:
        raise ValueError(
            "Choose a positive finite maximum time step and 1–64 annuli"
        )
    wall, center = [
        np.loadtxt(INPUT / f"temperature_{name}.csv", delimiter=",", skiprows=1)
        for name in ("wall", "center")
    ]
    z_cm = np.unique(np.r_[wall[:, 0], center[:, 0]])
    z_cm = z_cm[(z_cm >= center[0, 0]) & (z_cm <= center[-1, 0])]
    z = z_cm / 100
    wall_k = np.interp(z_cm, *wall.T) + 273.15
    center_k = np.interp(z_cm, *center.T) + 273.15
    # q = (r/R)^2. A parabolic flow gives equal flux between these edges.
    edges = 1 - np.sqrt(1 - np.linspace(0, 1, annuli + 1))
    q = (edges[:-1] + edges[1:]) / 2
    temp = wall_k + (1 - q[:, None]) * (center_k - wall_k)
    alpha = 3.5e-3 / 60 / (np.pi * (0.025 / 2) ** 2 * 293.15)
    u_over_temp = 2 * (1 - q[:, None]) * alpha
    dtemp = np.diff(temp, axis=1) / temp[:, :-1]
    log_ratio = np.ones_like(dtemp)
    np.divide(np.log1p(dtemp), dtemp, where=dtemp != 0, out=log_ratio)
    travel = np.diff(z)[None, :] * log_ratio / (u_over_temp * temp[:, :-1])
    steps = int(np.ceil(travel.sum(axis=1).max() / dt))
    times = np.c_[np.zeros(annuli), np.cumsum(travel, axis=1)]
    ticks = times[:, -1] / steps
    t = np.empty((annuli, steps))
    for lane in range(annuli):
        mid = (0.5 + np.arange(steps)) * ticks[lane]
        j = np.searchsorted(times[lane, 1:], mid)
        change = np.log(temp[lane, j + 1] / temp[lane, j])
        t[lane] = temp[lane, j] * np.exp(
            change * (mid - times[lane, j]) / travel[lane, j]
        )
    grid, monomer, dimer = np.loadtxt(
        INPUT / "salt_vapor.csv", delimiter=",", skiprows=1
    ).T
    if t.min() < grid.min() or t.max() > grid.max():
        raise ValueError("Thermal path exceeds the vapor-pressure table")
    log_m = np.interp(-1 / t, -1 / grid, monomer)
    log_d = np.interp(-1 / t, -1 / grid, dimer)
    log_s = -19286 / t - 7.15 * np.log10(t) + 31.895
    return (
        np.stack(
            (t, 1e5 * 10**log_m, 1e5 * 10**log_d, 1e5 * 10**log_s), axis=-1
        ),
        ticks,
        np.full(annuli, 1 / annuli),
    )
