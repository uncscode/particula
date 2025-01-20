from typing import Union
import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

from particula.util.validate_inputs import validate_inputs
from particula.util.constants import STANDARD_GRAVITY


@validate_inputs(
    {
        "velocity_dispersion": "positive",
        "particle_inertia_time_1": "positive",
        "particle_inertia_time_2": "positive",
    }
)
def get_radial_relative_velocity_dz2002(
    velocity_dispersion: Union[float, NDArray[np.float64]],
    particle_inertia_time_1: NDArray[np.float64],
) -> Union[float, NDArray[np.float64]]:
    """
    Compute the radial relative velocity based on the Dodin and Elperin (2002)
    formulation.

    The relative velocity is given by:

        ⟨ |w_r| ⟩ = sqrt(2 / π) * σ * f(b)

    - f(b) = (1/2) sqrt(π) (b + 0.5 / b) erf(b) + (1/2) exp(-b^2)
    - b = g |τ_p1 - τ_p2| / (sqrt(2) σ)
    - σ : Turbulence velocity dispersion [m/s]
    - τ_p1, τ_p2 : Inertia timescale of particles 1 and 2 [s]
    - g : Gravitational acceleration [m/s²]

    Arguments:
    ----------
        - velocity_dispersion : Turbulence velocity dispersion (σ) [m/s].
        - particle_inertia_time : Inertia timescale of particles (τ_p) [s].

    Returns:
    --------
        - Radial relative velocity ⟨ |w_r| ⟩ [m/s].

    References:
    -----------
    - Dodin, Z., & Elperin, T. (2002). Phys. Fluids, 14, 2921-24.
    """
    tau_diff = np.abs(particle_inertia_time_1 - particle_inertia_time_2)
    b = (STANDARD_GRAVITY * tau_diff) / (
        np.sqrt(2) * velocity_dispersion
    )

    # Compute f(b)
    sqrt_pi = np.sqrt(np.pi)
    erf_b = erf(b)
    exp_b2 = np.exp(-(b**2))
    f_b = (
        0.5 * sqrt_pi * (b + 0.5 / np.maximum(b, 1e-12)) * erf_b + 0.5 * exp_b2
    )

    return np.sqrt(2 / np.pi) * velocity_dispersion * f_b


@validate_inputs(
    {
        "velocity_dispersion": "positive",
        "particle_inertia_time_1": "positive",
        "particle_inertia_time_2": "positive",
    }
)
def get_radial_relative_velocity_ao2008(
    velocity_dispersion: Union[float, NDArray[np.float64]],
    particle_inertia_time_1: Union[float, NDArray[np.float64]],
    particle_inertia_time_2: Union[float, NDArray[np.float64]],
    gravitational_acceleration: float = STANDARD_GRAVITY,
) -> Union[float, NDArray[np.float64]]:
    """
    Compute the radial relative velocity based on the Ayala et al. (2008) formulation.

    The relative velocity is given by:

        ⟨ |w_r| ⟩ = sqrt(2 / π) * sqrt(σ² + (π/8) * (τ_p1 + τ_p2)² * |g|²)

    Arguments:
    ----------
        - velocity_dispersion : Turbulence velocity dispersion (σ) [m/s].
        - particle_inertia_time_1 : Inertia timescale of particle 1 (τ_p1) [s].
        - particle_inertia_time_2 : Inertia timescale of particle 2 (τ_p2) [s].
        - gravitational_acceleration : Gravitational acceleration (default: 9.81 m/s²).

    Returns:
    --------
        - Radial relative velocity ⟨ |w_r| ⟩ [m/s].

    References:
    -----------
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). New Journal of Physics, 10.
    """
    tau_sum = particle_inertia_time_1 + particle_inertia_time_2
    gravity_term = (np.pi / 8) * tau_sum**2 * gravitational_acceleration**2

    return np.sqrt(2 / np.pi) * np.sqrt(velocity_dispersion**2 + gravity_term)
