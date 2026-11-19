"""Independently refined radial energy profiles for validation experiments."""

import numpy as np
from particula.dynamics.coagulation._pair_potential import (
    dielectric_response,
    hamaker_energy,
)
from scipy.interpolate import PchipInterpolator


def potential_profile(
    fraction, charges, attraction, coupling, epsilon, resolution=256, order=160
):
    """Return continuous electrostatic and total thermal energy profiles."""
    x = np.sin(np.linspace(0, np.pi / 2, resolution + 1)) ** 2
    left, right = charges
    electric = coupling * left * right * x
    if epsilon != 1 and (left or right):
        coefficients = np.array(
            [
                dielectric_response(value, fraction, epsilon, order)
                for value in x
            ]
        )
        electric += (
            coupling
            * coefficients
            @ np.array([left**2, left * right, right**2])
        )
    spline = PchipInterpolator(x, electric)

    def potential(ratio):
        position = np.asarray(ratio)
        result = spline(position)
        if attraction:
            with np.errstate(divide="ignore", invalid="ignore"):
                result = np.where(
                    position < 1,
                    result + attraction * hamaker_energy(position, fraction),
                    -np.inf,
                )
        return result

    return spline, potential
