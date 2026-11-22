"""Apply the mobility and aerodynamic windows to the two origin modes."""

import numpy as np
from scipy.constants import R
from scipy.special import ndtr

from . import observations as obs

REF_MOLES = 101325 / (R * 293.15)


def windows(mu, s, lo, hi):
    """Truncate a lognormal mode and return its number and log moments."""
    a, b = (np.log(lo) - mu) / s, (np.log(hi) - mu) / s
    f = np.where(a.real > 0, ndtr(-a) - ndtr(-b), ndtr(b) - ndtr(a))
    mom = f * mu + s * (np.exp(-a * a / 2) - np.exp(-b * b / 2)) / np.sqrt(
        2 * np.pi
    )
    return f, mom


def observation(states, weight, gsd, ratio, packing=1.0, seed=0.0):
    """Return detected number, sizes, instrument masses and sulfate percent.

    States have axes (candidate, feed, path, state). Packing is a common
    material/outer mobility volume ratio, used only for the readout. Background
    cores carry a sulfate-equivalent volume with a separate inert mass ledger.
    """
    a = np.asarray(states)

    def field(x):
        return np.broadcast_to(np.atleast_1d(x), (len(a),)).reshape(-1, 1, 1)

    s = np.log(field(gsd))[..., None]
    ratio = field(ratio)
    w = np.asarray(weight)[None, None, :]
    ns, nc = a[..., 7], a[..., 6]
    nt = ns + nc
    ns = ns / np.maximum(nt, 1e-100)
    fractions = np.stack([ns, 1 - ns], axis=-1)
    ms, mc = (
        a[..., 5] * REF_MOLES * 1e-6 * obs.SULFATE_MW,
        a[..., 4] * REF_MOLES * 1e-6 * obs.CHLORIDE_MW,
    )
    core = (
        field(seed) * (0.157 / 0.086) * 1e-6
    )  # kg/m³, equivalent mass from Table 4, kept outside K/S
    mass = ms + mc + core
    v = (
        ms / obs.SULFATE_RHO + mc / obs.CHLORIDE_RHO + core / obs.SULFATE_RHO
    ) / field(packing)
    density = mass / np.maximum(v, 1e-100)
    density = np.where(mass > 1e-50, density, 2000.0)
    fac = np.stack(
        np.broadcast_arrays(np.ones_like(ns), ratio * np.ones_like(ns)), axis=-1
    )
    mf = fractions * fac**3
    denom = mf.sum(-1)
    mf = mf / denom[..., None]
    logmu = (
        np.log(
            np.maximum(
                6e15 * v / (np.pi * np.maximum(nt, 1e-100) * denom), 1e-300
            )
        )[..., None]
        / 3
        - 1.5 * s * s
        + np.log(fac)
    )
    number_frac, number_log = windows(logmu, s, *obs.SMPS)
    mu = logmu + 3 * s * s
    smps_frac, smps_log = windows(mu, s, *obs.SMPS)
    lmin = obs._rescale_density(obs.LPI_AERO[0], 1000.0 / density)[..., None]
    lmax = obs._rescale_density(obs.LPI_AERO[1], 1000.0 / density)[..., None]
    lf, _ = windows(mu, s, lmin, lmax)
    # Quadrature of converted mobility diameter exactly as in parent study.
    z0 = np.maximum((np.log(lmin) - mu) / s, -14.0)
    z1 = np.minimum((np.log(lmax) - mu) / s, 14.0)
    dz = np.maximum(0, z1 - z0)
    z = z0[..., None] + 0.5 * dz[..., None] * (1 + obs.GAUSS_X)
    diam = np.exp(np.clip(mu[..., None] + s[..., None] * z, -35, 35))
    reported = obs._rescale_density(
        diam, (density / obs.REPORTED_RHO)[..., None, None]
    )
    ll = (
        0.5
        * dz
        * np.sum(
            obs.GAUSS_W
            * np.log(reported)
            * np.exp(-0.5 * z * z)
            / np.sqrt(2 * np.pi),
            axis=-1,
        )
    )

    def add(x):
        return (x * w[..., None]).sum(axis=(-1, -2))

    n = nt[..., None] * fractions
    mode_mass = mass[..., None] * 1e6 * mf
    m = (obs.REPORTED_RHO * v * 1e6)[..., None] * mf
    count, impact, mob = (
        add(n * number_frac),
        add(mode_mass * lf),
        add(m * smps_frac),
    )

    def mean_top(x, y):
        return np.where(
            y > 1e-30, np.exp(np.clip(x / np.maximum(y, 1e-30), -40, 40)), 0.0
        )

    return np.stack(
        [
            count,
            mean_top(add(n * number_log), count),
            impact,
            mob,
            mean_top(add(mode_mass * ll), impact),
            mean_top(add(m * smps_log), mob),
            100
            * add(ms[..., None] * 1e6 * mf * lf)
            / np.maximum(impact, 1e-30),
        ],
        axis=-1,
    )
