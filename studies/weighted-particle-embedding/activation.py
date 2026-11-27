"""Diagnose equilibrium CCN activation from dry size and volume-mixed kappa."""

import numpy as np

KELVIN_LENGTH = 2 * 0.072 * 0.01801528 / (1000 * 8.31446261815324 * 293)
SUPERSATURATION_PERCENT = np.geomspace(0.002, 1.0, 18)


def critical_percent(radius_m, kappa):
    """Find the maximum equilibrium supersaturation of each dry particle."""
    a = KELVIN_LENGTH / np.asarray(radius_m)
    if np.any(a <= 0) or np.any(kappa <= 0):
        raise ValueError(
            "Activation diagnostic requires positive radius and kappa"
        )
    lo = np.ones_like(a)
    hi = 2 + np.sqrt(6 * kappa / a)
    for _ in range(45):
        q = (lo + hi) / 2
        water = q**3 - 1
        increasing = 3 * kappa * q**4 > a * water * (water + kappa)
        lo = np.where(increasing, q, lo)
        hi = np.where(increasing, hi, q)
    q = (lo + hi) / 2
    return 100 * np.expm1(a / q - np.log1p(kappa / (q**3 - 1)))


def spectrum(mass, counts, density, volume):
    """Return equilibrium activated number concentration in particles per m³."""
    dry = mass[..., 1:] / density[1:]
    total = dry.sum(-1)
    if np.any((counts > 0) & (total <= 0)):
        raise ValueError("Active particle has zero dry volume")
    radius = np.cbrt(3 * np.where(total > 0, total, 1.0) / (4 * np.pi))
    kappa = (dry[..., 0] * 0.03 + dry[..., 1] * 0.75) / np.where(
        total > 0, total, 1.0
    )
    critical = critical_percent(radius, np.where(total > 0, kappa, 0.03))
    return (
        np.einsum(
            "...i,...ik->...k",
            counts,
            critical[..., None] <= SUPERSATURATION_PERCENT,
        )
        / volume
    )
