"""Nonretarded dispersion and dielectric response of two compact spheres."""

from functools import lru_cache

import numpy as np
from scipy.linalg import solve
from scipy.special import comb


def hamaker_energy(x, fraction):
    """Return U/A for centre-distance ratio x=(a+b)/s, 0 <= x < 1.

    The symmetric radius fraction a/(a+b) may refer to either sphere.
    Hamaker (1937), doi:10.1016/S0031-8914(37)80203-7.
    """
    x = np.asarray(x, dtype=np.float64)
    u = 4 * fraction * (1 - fraction) * x**2 / ((1 - x) * (1 + x))
    # Hamaker's two rational terms cancel the first two log series terms.
    # Evaluate the leading small-separation-ratio terms without subtraction.
    small = np.minimum(u, 0.01)
    series = small**3 * (1 / 6 - small * (1 / 4 - small * (3 / 10 - small / 3)))
    direct = (u + u / (1 + u)) / 2 - np.log1p(u)
    return -np.where(u < 0.001, series, direct) / 6


@lru_cache(maxsize=8)
def _multipole_orders(order):
    degrees = np.arange(1, order + 1, dtype=np.float64)
    translations = comb(degrees[:, None] + degrees, degrees)
    return degrees, translations


def dielectric_response(x, fraction, relative_permittivity, order):
    """Return induced energy coefficients for q1², q1*q2 and q2².

    Lengths are normalized by contact distance R=a+b. Add x*q1*q2
    for the bare monopoles; multiply the result by 1/(4*pi*eps0*R)
    with charges in C to obtain energy in J. The exterior is vacuum.
    Free surface charge is uniform on each sphere; bound charge responds
    self-consistently. Both polar axes point toward the partner centre.

    Each degree l reflects a regular incident potential with coefficient
    -l*(eps-1)/(l*(eps+1)+1). Translation of an axial multipole k to
    the other centre supplies binomial(l+k,l). Energy is half the sum
    of free charges times the induced potentials at their centres.

    Reference: Bichoutskaia et al. (2010), doi:10.1063/1.3457157.
    """
    if relative_permittivity == 1 or x == 0:
        return np.zeros(3)
    degree, translation = _multipole_orders(order)
    reflect = (
        degree
        * (relative_permittivity - 1)
        / (degree * (relative_permittivity + 1) + 1)
    )
    left, right = fraction * x, (1 - fraction) * x
    left_power, right_power = left**degree, right**degree
    left_incident = reflect * left * left_power
    right_incident = reflect * right * right_power
    left_from_right = left_incident[:, None] * translation * right_power
    right_from_left = right_incident[:, None] * translation * left_power
    # Columns correspond to unit free charge on the left or right sphere.
    left_source = np.column_stack((np.zeros(order), -left_incident))
    right_source = np.column_stack((-right_incident, np.zeros(order)))
    left_moments = solve(
        np.eye(order) - left_from_right @ right_from_left,
        left_source - left_from_right @ right_source,
        check_finite=False,
    )
    right_moments = right_source - right_from_left @ left_moments
    at_left = right_power @ right_moments
    at_right = left_power @ left_moments
    return (
        0.5 * x * np.array([at_left[0], at_left[1] + at_right[0], at_right[1]])
    )
