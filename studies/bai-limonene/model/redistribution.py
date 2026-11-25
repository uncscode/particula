"""A mass-conserving pre-cloud redistribution of photochemical products."""

import numpy as np
from scipy.integrate import solve_ivp
from particula.particles import get_vapor_transition_correction


def redeposit(diameter, weights, retained, returned_share, path_nm, alpha):
    """Integrate dry volume uptake in units of fresh ensemble volume.

    Prescribe gross removal of a uniform volume fraction from every size,
    followed by uptake from a common, effectively nonvolatile vapor. End at
    the observed net mass fraction. All material has the same final kappa
    and density. Gas supply fixes the endpoint. No elapsed time is inferred.
    """
    diameter_scale = 100e-9
    original = (diameter / diameter_scale) ** 3
    mean = weights @ original
    initial = original * retained * (1 - returned_share)

    def rhs(_, volume):
        radius = diameter_scale * 0.5 * np.cbrt(volume)
        beta = get_vapor_transition_correction(path_nm * 1e-9 / radius, alpha)
        conductance = radius * beta
        return mean * conductance / (weights @ conductance)

    out = solve_ivp(
        rhs, (0, retained * returned_share), initial, rtol=2e-10, atol=2e-13
    )
    if not out.success:
        raise ValueError(out.message)
    volume = out.y[:, -1]
    assert np.isclose(weights @ volume / mean, retained, rtol=1e-11)
    return volume / original
