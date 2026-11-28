"""Independent forward wet-diameter reference for CPU and device tests."""

import numpy as np
from scipy.optimize import brentq, minimize_scalar

KELVIN_DIAMETER = 2.1e-9


def forward_peak(dry_diameter, kappa, log_growth_grid):
    """Find all local maxima on the wet-diameter axis and refine each one."""
    if kappa == 0:
        return KELVIN_DIAMETER / dry_diameter

    def log_saturation(log_relative_water_layer):
        # v = (D_wet - D_dry) / D_dry. Factoring the difference of
        # cubes retains the water volume of a nearly dry particle.
        v = np.exp(log_relative_water_layer)
        log_water_per_dry = log_relative_water_layer + np.logaddexp(
            np.log(3) + np.log1p(v), 2 * log_relative_water_layer
        )
        kelvin = (KELVIN_DIAMETER / dry_diameter) / (1 + v)
        return kelvin - np.logaddexp(0.0, np.log(kappa) - log_water_per_dry)

    values = log_saturation(log_growth_grid)
    maxima = (
        np.flatnonzero(
            (values[1:-1] >= values[:-2]) & (values[1:-1] > values[2:])
        )
        + 1
    )
    if not len(maxima):
        raise ArithmeticError("Forward peak falls outside the test search grid")
    peaks = [
        -minimize_scalar(
            lambda x: -log_saturation(x),
            bounds=(log_growth_grid[i - 1], log_growth_grid[i + 1]),
            method="bounded",
            options={"xatol": 1e-12},
        ).fun
        for i in maxima
    ]
    peak = max(peaks)
    if peak <= max(values[0], values[-1]):
        raise ArithmeticError("Forward peak is bounded by a search endpoint")
    return peak


def forward_threshold(kappa, log_target, grid):
    """Invert the independent wet-diameter maximum on a dry-diameter axis."""
    upper = np.log(KELVIN_DIAMETER / log_target)
    root = brentq(
        lambda log_d: forward_peak(np.exp(log_d), kappa, grid) - log_target,
        upper - 30,
        upper + 1,
        xtol=1e-14,
    )
    return np.exp(root)
