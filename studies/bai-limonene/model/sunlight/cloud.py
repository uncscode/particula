"""Shared-water cloud dynamics and size-continuous counting by original seed cohort."""

import numpy as np
from scipy.special import ndtr
from particula.dynamics import parcel as par

P0, T0, RH0 = 90000.0, 285.0, 0.999
QREF = (
    par.RD
    / par.RV
    * RH0
    * par._saturation_pressure(T0)
    / (P0 - RH0 * par._saturation_pressure(T0))
)
VREF = (par.RD + par.RV * QREF) * T0 / P0


def wet_at(dry, k, q, T0=T0, P0=P0):
    rh = P0 * q / ((par.RD / par.RV + q) * par._saturation_pressure(T0))
    lo = dry.copy()
    hi = dry * np.cbrt(1 + k * rh / (1 - rh))
    for _ in range(45):
        mid = (hi + lo) / 2
        low = par._equilibrium(mid, dry, T0, k) < rh
        lo, hi = np.where(low, mid, lo), np.where(low, hi, mid)
    return (lo + hi) / 2


def margins(bins, wet, temp):
    dry, n, _, k = bins.T
    x = wet / dry
    u = x**3 - 1
    a = 2 * par.SIGMA / (par.RV * par.RHO_W * temp * dry)
    return np.log(np.maximum(1e-80, a * u * (u + k) / (3 * k * x**4)))


def cdf(pop, tag, x):
    if tag == 0:
        return (
            np.clip(
                (ndtr((x - np.log(pop.med)) / np.log(1.5)) - ndtr(-4))
                / (ndtr(4) - ndtr(-4)),
                0,
                1,
            )
            * pop.number[pop.tag == 0].sum()
        )
    return np.interp(
        x, np.log(pop.edges[1]), np.r_[0, np.cumsum(pop.number[pop.tag == 1])]
    )
