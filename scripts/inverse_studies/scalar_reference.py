"""Inverse-radius quadrature for smooth closed-box growth."""

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scripts import assess_native_condensation as case
from scripts.assess_native_adaptive import VISCOSITY


@lru_cache(maxsize=1)
def _quadrature():
    """Cache the fixed integration nodes on [-1, 1]."""
    return np.polynomial.legendre.leggauss(32)


def growth_time(initial_um: float, final_um: float) -> float:
    """Integrate dt/dr between radii in micrometres and return seconds.

    Applies to a monodisperse population with fixed initial vapor, zero surface
    vapor and positive remaining gas. Negative intervals extend the residual
    smoothly to trial initial radii above the observation.
    """
    if (
        not np.all(np.isfinite([initial_um, final_um]))
        or min(initial_um, final_um) <= 0
    ):
        raise ValueError("radii must be finite and positive")
    initial, final = np.array([initial_um, final_um]) * 1e-6
    half = (final - initial) / 2
    nodes, weights = _quadrature()
    radius = initial + half * (1 + nodes)
    transfer = 4 * np.pi * case.DENSITY * case.NUMBER / 3
    if case.VAPOR - transfer * (final**3 - initial**3) <= 0:
        raise ValueError("final radius exceeds the available vapor")
    vapor = case.VAPOR - transfer * (radius**3 - initial**3)
    mean_path = (
        2
        * VISCOSITY
        / case.PRESSURE
        / np.sqrt(
            8 * case.MOLAR_MASS / (np.pi * case.GAS_CONSTANT * case.TEMPERATURE)
        )
    )
    kn = mean_path / radius
    correction = 0.75 * (1 + kn) / (kn**2 + 1.283 * kn + 0.75)
    inverse_rate = case.DENSITY * radius / (case.DIFFUSION * correction * vapor)
    return float(half * (weights @ inverse_rate))


def recover_radius(final_um: float, duration: float = 1.4) -> float:
    """Fit one initial radius in [0.8, 1.2] µm by a bounded scalar search."""
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("duration must be finite and nonnegative")
    if not np.isfinite(final_um) or final_um <= 0:
        raise ValueError("observed radius must be finite and positive")

    def residual(initial_um):
        # An unreachable endpoint lies on the long-time side of the root.
        try:
            return growth_time(initial_um, final_um) - duration
        except ValueError:
            return np.inf

    if residual(0.8) <= 0:
        return 0.8
    if residual(1.2) >= 0:
        return 1.2
    return float(brentq(residual, 0.8, 1.2, xtol=1e-13))
