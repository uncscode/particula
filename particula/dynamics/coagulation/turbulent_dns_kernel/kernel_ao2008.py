"""
Calculate the geometric collision kernel Γ₁₂ (or K₁₂) based on turbulent
DNS simulations.
"""

from typing import Union
import numpy as np
from numpy.typing import NDArray

from particula.util.validate_inputs import validate_inputs
from particula.dynamics.coagulation.turbulent_dns_kernel.radial_velocity_module import (
    get_radial_relative_velocity_ao2008,
)
from particula.dynamics.coagulation.turbulent_dns_kernel.g12_radial_distribution_ao2008 import (
    get_g12_radial_distribution_ao2008,
)


@validate_inputs(
    {
        "particle_radius": "positive",
        "velocity_dispersion": "positive",
        "particle_inertia_time": "positive",
    }
)
def get_kernel_ao2008(
    particle_radius: Union[float, NDArray[np.float64]],
    velocity_dispersion: Union[float, NDArray[np.float64]],
    particle_inertia_time: Union[float, NDArray[np.float64]],
    stokes_number: Union[float, NDArray[np.float64]],
    kolmogorov_length_scale: float,
    reynolds_lambda: float,
    normalized_accel_variance: float,
    kolmogorov_velocity: float,
    kolmogorov_time: float,
) -> Union[float, NDArray[np.float64]]:
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    """
    Get the geometric collision kernel Γ₁₂.

        Γ₁₂ = 2π R² ⟨ |w_r| ⟩ g₁₂

    - R = a₁ + a₂ (collision radius)
    - ⟨ |w_r| ⟩ : Radial relative velocity, computed using
        `get_radial_relative_velocity_ao2008`
    - g₁₂ : Radial distribution function, computed using
        `g12_radial_distribution`
    - radius << η (Kolmogorov length scale)
    - ρ_w >> ρ (water density much greater than air density)
    - Sv > 1 (Stokes number sufficiently large)

    Arguments:
    ----------
        - particle_radius : Particle radius [m].
        - velocity_dispersion : Velocity dispersion [m/s].
        - particle_inertia_time : Particle inertia time [s].
        - stokes_number : Stokes number [-].
        - kolmogorov_length_scale : Kolmogorov length scale [m].
        - reynolds_lambda : Reynolds number [-].
        - normalized_accel_variance : Normalized acceleration variance [-].
        - kolmogorov_velocity : Kolmogorov velocity [m/s].
        - kolmogorov_time : Kolmogorov time [s].

    Returns:
    --------
        - Collision kernel Γ₁₂ [m³/s].

    References:
    -----------
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2.
        Theory and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """
    collision_radius = (
        particle_radius[:, np.newaxis] + particle_radius[np.newaxis, :]
    )

    # Compute radial relative velocity ⟨ |w_r| ⟩
    wr = get_radial_relative_velocity_ao2008(
        velocity_dispersion, particle_inertia_time
    )

    # Compute radial distribution function g₁₂
    g12 = get_g12_radial_distribution_ao2008(
        particle_radius,
        stokes_number,
        kolmogorov_length_scale,
        reynolds_lambda,
        normalized_accel_variance,
        kolmogorov_velocity,
        kolmogorov_time,
    )

    # Compute collision kernel Γ₁₂
    gamma_12 = 2 * np.pi * collision_radius**2 * wr * g12

    return gamma_12
