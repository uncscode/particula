"""
Calculate the radial distribution function g_{12} for particles in a
turbulent flow.
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.util.validate_inputs import validate_inputs
from particula.util.reduced_quantity import reduced_value

@validate_inputs(
    {
        "kolmogorov_length_scale": "positive",
        "collision_radius": "positive",
        "stokes_number_1": "positive",
        "stokes_number_2": "positive",
        "reynolds_lambda": "positive",
        "normalized_accel_variance": "positive",
        "kolmogorov_velocity": "positive",
        "kolmogorov_time": "positive",
        "gravitational_acceleration": "positive",
    }
)
def get_g12_radial_distribution_ao2008(
    kolmogorov_length_scale: NDArray[np.float64],
    collision_radius: NDArray[np.float64],
    stokes_number_1: NDArray[np.float64],
    stokes_number_2: NDArray[np.float64],
    reynolds_lambda: float,
    normalized_accel_variance: float,
    kolmogorov_velocity: float,
    kolmogorov_time: float,
    gravitational_acceleration: float,
) -> NDArray[np.float64]:
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
        - kolmogorov_length_scale : Kolmogorov length scale [m]
        - collision_radius : Collision radius (sum of particle radii) [m]
        - stokes_number_1 : Stokes number of first particle [-]
        - stokes_number_2 : Stokes number of second particle [-]
        - reynolds_lambda : Taylor-microscale Reynolds number [-]
        - normalized_accel_variance : Normalized acceleration variance [-]
        - kolmogorov_velocity : Kolmogorov velocity scale [m/s]
        - kolmogorov_time : Kolmogorov timescale [s]
        - gravitational_acceleration : Gravitational acceleration [m/s²]

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
    stokes_number = np.maximum(stokes_number_1, stokes_number_2)
    C1 = _calculate_C1(
        stokes_number,
        reynolds_lambda,
        kolmogorov_velocity,
        kolmogorov_time,
        gravitational_acceleration,
    )
    rc = _calculate_rc(
        kolmogorov_length_scale,
        stokes_number_1,
        stokes_number_2,
        normalized_accel_variance,
        reynolds_lambda,
        kolmogorov_velocity,
        kolmogorov_time,
        gravitational_acceleration,
    )

    return (
        (kolmogorov_length_scale**2 + rc**2) / (collision_radius**2 + rc**2)
    ) ** (C1 / 2)


def _calculate_C1(
    stokes_number: NDArray[np.float64],
    reynolds_lambda: float,
    kolmogorov_velocity: float,
    kolmogorov_time: float,
    gravitational_acceleration: float,
) -> NDArray[np.float64]:
    """Compute C_1 based on Stokes number and turbulence properties."""
    y_stokes = _compute_y_stokes(stokes_number)
    f3_lambda = _compute_f3_lambda(reynolds_lambda)
    gravity_term = np.abs(gravitational_acceleration) / (
        kolmogorov_velocity / kolmogorov_time
    )

    return y_stokes / (gravity_term**f3_lambda)


def _compute_y_stokes(
    stokes_number: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute y(St), ensuring values remain non-negative."""
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
    kolmogorov_length_scale: NDArray[np.float64],
    stokes_number_1: NDArray[np.float64],
    stokes_number_2: NDArray[np.float64],
    normalized_accel_variance: float,
    reynolds_lambda: float,
    kolmogorov_velocity: float,
    kolmogorov_time: float,
    gravitational_acceleration: float,
) -> NDArray[np.float64]:
    """
    Compute r_c, the turbulence-driven correction to the collision kernel.
    """
    stokes_difference = np.abs(stokes_number_2 - stokes_number_1)
    aOg = _compute_aOg(
        normalized_accel_variance,
        gravitational_acceleration,
        kolmogorov_velocity,
        kolmogorov_time,
    )
    F_value = _compute_F(aOg, reynolds_lambda)

    return kolmogorov_length_scale * np.sqrt(stokes_difference * F_value)


def _compute_aOg(
    normalized_accel_variance: float,
    gravitational_acceleration: float,
    kolmogorov_velocity: float,
    kolmogorov_time: float,
) -> float:
    """Compute aOg, which accounts for the effect of gravity on turbulence-driven clustering."""
    gravity_term = np.abs(gravitational_acceleration) / (
        kolmogorov_velocity / kolmogorov_time
    )
    return normalized_accel_variance + (np.pi / 8) * gravity_term**2


def _compute_F(
    aOg: float,
    reynolds_lambda: float,
) -> float:
    """Compute F(aOg, R_lambda), an empirical scaling factor for turbulence effects."""
    return 20.115 * (aOg / reynolds_lambda) ** 0.5
