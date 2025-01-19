"""
Calculate the radial distribution function g_{12} for particles in a
turbulent flow.
"""

from numpy.typing import NDArray
import numpy as np

from particula.util.validate_inputs import validate_inputs
from particula.util.constants import STANDARD_GRAVITY
from particula.util.self_broadcast import (
    get_pairwise_sum_matrix,
    get_pairwise_max_matrix,
    get_pairwise_diff_matrix,
)


@validate_inputs(
    {
        "particle_radius": "positive",
        "stokes_number": "positive",
        "kolmogorov_length_scale": "positive",
        "reynolds_lambda": "positive",
        "normalized_accel_variance": "positive",
        "kolmogorov_velocity": "positive",
        "kolmogorov_time": "positive",
    }
)
def get_g12_radial_distribution_ao2008(
    particle_radius: NDArray[np.float64],
    stokes_number: NDArray[np.float64],
    kolmogorov_length_scale: float,
    reynolds_lambda: float,
    normalized_accel_variance: float,
    kolmogorov_velocity: float,
    kolmogorov_time: float,
) -> NDArray[np.float64]:
    # pylint: disable=too-many-arguments
    """
    Compute the radial distribution function g_{12}.

    The radial distribution function describes the clustering of particles
    in a turbulent flow and is given by:

        g_{12} = ((η² + r_c²) / (R² + r_c²))^(C_1/2)

    - η (kolmogorov_length_scale) : Kolmogorov length scale [m]
    - R (collision_radius) : Collision radius (sum of particle radii) [m]
    - Stokes_1, Stokes_2 : Stokes numbers of the two particles [-]
    - R_λ (reynolds_lambda) : Taylor-microscale Reynolds number [-]
    - a_o (normalized_accel_variance) : Normalized acceleration variance [-]
    - v_k (kolmogorov_velocity) : Kolmogorov velocity scale [m/s]
    - τ_k (kolmogorov_time) : Kolmogorov timescale [s]
    - g (gravitational_acceleration) : Gravitational acceleration [m/s²]

    Arguments:
    ----------
    - particle_radius : Array of particle radii [m]
    - stokes_number : Array of Stokes numbers of particles [-]
    - kolmogorov_length_scale : Kolmogorov length scale [m]
    - reynolds_lambda : Taylor-microscale Reynolds number [-]
    - normalized_accel_variance : Normalized acceleration variance [-]
    - kolmogorov_velocity : Kolmogorov velocity scale [m/s]
    - kolmogorov_time : Kolmogorov timescale [s]

    Returns:
    --------
    - Radial distribution function g_{12} [-]

    References:
    -----------
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2.
        Theory and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """
    collision_radius = get_pairwise_sum_matrix(particle_radius)
    stokes_diff_matrix = get_pairwise_diff_matrix(stokes_number)
    stokes_max_matrix = get_pairwise_max_matrix(stokes_number)

    c1 = _calculate_c1(
        stokes_max_matrix,
        reynolds_lambda,
        kolmogorov_velocity,
        kolmogorov_time,
    )

    rc = _calculate_rc(
        stokes_diff_matrix,
        kolmogorov_length_scale,
        normalized_accel_variance,
        reynolds_lambda,
        kolmogorov_velocity,
        kolmogorov_time,
    )

    return (
        (kolmogorov_length_scale**2 + rc**2) / (collision_radius**2 + rc**2)
    ) ** (c1 / 2)


def _calculate_c1(
    stokes_number: NDArray[np.float64],
    reynolds_lambda: float,
    kolmogorov_velocity: float,
    kolmogorov_time: float,
) -> NDArray[np.float64]:
    """
    Compute C_1 based on Stokes number and turbulence properties.

    C_1 is given by:

        C_1 = y(St) / (|g| / (v_k / τ_k))^f_3(R_λ)

    - y(St) = -0.1988 St^4 + 1.5275 St^3 - 4.2942 St^2 + 5.3406 St
    - f_3(R_λ) = 0.1886 * exp(20.306 / R_λ)
    - |g| : Gravitational acceleration [m/s²]
    - v_k : Kolmogorov velocity scale [m/s]
    - τ_k : Kolmogorov timescale [s]
    """
    y_stokes = _compute_y_stokes(stokes_number)
    f3_lambda = _compute_f3_lambda(reynolds_lambda)
    gravity_term = np.abs(STANDARD_GRAVITY) / (
        kolmogorov_velocity / kolmogorov_time
    )

    return y_stokes / (gravity_term**f3_lambda)


def _compute_y_stokes(
    stokes_number: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute y(St), ensuring values remain non-negative.

    y(St) = -0.1988 St^4 + 1.5275 St^3 - 4.2942 St^2 + 5.3406 St

    Ensures y(St) ≥ 0 (if negative, sets to 0).
    """
    y_st = (
        -0.1988 * stokes_number**4
        + 1.5275 * stokes_number**3
        - 4.2942 * stokes_number**2
        + 5.3406 * stokes_number
    )
    return np.maximum(y_st, 0)  # Ensures y(St) is non-negative


def _compute_f3_lambda(reynolds_lambda: float) -> float:
    """Compute f_3(R_lambda), an empirical turbulence factor."""
    return 0.1886 * np.exp(20.306 / reynolds_lambda)


def _calculate_rc(
    stokes_diff_matrix: NDArray[np.float64],
    kolmogorov_length_scale: float,
    normalized_accel_variance: float,
    reynolds_lambda: float,
    kolmogorov_velocity: float,
    kolmogorov_time: float,
) -> NDArray[np.float64]:
    """
    Compute r_c, the turbulence-driven correction to the collision kernel.

    The equation is:

    (r_c / η)^2 = |St_2 - St_1| * F(a_Og, R_λ)
    """
    stokes_difference = np.abs(stokes_diff_matrix)
    a_og = _compute_a_og(
        normalized_accel_variance,
        kolmogorov_velocity,
        kolmogorov_time,
    )
    f_value = _compute_f(a_og, reynolds_lambda)

    return kolmogorov_length_scale * np.sqrt(stokes_difference * f_value)


def _compute_a_og(
    normalized_accel_variance: float,
    kolmogorov_velocity: float,
    kolmogorov_time: float,
) -> float:
    """
    Compute aOg, which accounts for the effect of gravity on
    turbulence-driven clustering.

    - a_Og = a_o + (π / 8) * (|g| / (v_k / τ_k))^2
    """
    gravity_term = np.abs(STANDARD_GRAVITY) / (
        kolmogorov_velocity / kolmogorov_time
    )
    return normalized_accel_variance + (np.pi / 8) * gravity_term**2


def _compute_f(
    a_og: float,
    reynolds_lambda: float,
) -> float:
    """
    Compute F(aOg, R_lambda), an empirical scaling factor for
    turbulence effects.

    - F(a_Og, R_λ) = 20.115 * (a_Og / R_λ)^0.5
    """
    return 20.115 * (a_og / reynolds_lambda) ** 0.5
