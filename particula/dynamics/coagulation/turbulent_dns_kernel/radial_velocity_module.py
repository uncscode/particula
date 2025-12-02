"""Radial relative velocity calculation module."""

from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf  # pylint: disable=no-name-in-module

from particula.util.constants import STANDARD_GRAVITY
from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "velocity_dispersion": "positive",
        "particle_inertia_time": "positive",
    }
)
def get_radial_relative_velocity_dz2002(
    velocity_dispersion: Union[float, NDArray[np.float64]],
    particle_inertia_time: NDArray[np.float64],
) -> Union[float, NDArray[np.float64]]:
    """Compute the radial relative velocity based on Dodin and Elperin (2002).

    This function calculates the radial relative velocity between pairs of
    particles under turbulent conditions, capturing the effects of different
    inertia timescales. The equation is:

    - ⟨|wᵣ|⟩ = √(2/π) × σ × f(b)
        - wᵣ is the radial relative velocity in m/s,
        - σ is the turbulence velocity dispersion in m/s,
        - b = (g × |τₚᵢ - τₚⱼ|) / (√2 × σ),
        - f(b) = ½√π (b + 0.5 / b) erf(b) + ½ exp(-b²).

    Arguments:
        - velocity_dispersion : Turbulence velocity dispersion (σ) in m/s.
        - particle_inertia_time : Inertia timescale(s) (τₚ) in seconds.

    Returns:
        - The radial relative velocity ⟨|wᵣ|⟩ in m/s.

    Examples:
        ```py
        import numpy as np
        import particula as par

        # Example with an array of inertia times
        result = par.dynamics.get_radial_relative_velocity_dz2002(
            1.0, np.array([0.1, 0.2, 0.3])
        )
        print(result)
        ```

    References:
        - Dodin, Z., & Elperin, T. (2002). Phys. Fluids, 14, 2921–2924.
    """
    tau_diff = np.abs(
        particle_inertia_time[:, np.newaxis]
        - particle_inertia_time[np.newaxis, :]
    )

    b = (STANDARD_GRAVITY * tau_diff) / (np.sqrt(2) * velocity_dispersion)

    # Compute f(b)
    sqrt_pi = np.sqrt(np.pi)
    erf_b = erf(b)
    exp_b2 = np.exp(-(b**2))
    f_b = (
        0.5 * sqrt_pi * (b + 0.5 / np.maximum(b, 1e-16)) * erf_b + 0.5 * exp_b2
    )

    return np.sqrt(2 / np.pi) * velocity_dispersion * f_b


@validate_inputs(
    {
        "velocity_dispersion": "positive",
        "particle_inertia_time": "positive",
    }
)
def get_radial_relative_velocity_ao2008(
    velocity_dispersion: Union[float, NDArray[np.float64]],
    particle_inertia_time: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Compute the radial relative velocity based on Ayala et al. (2008).

    This function estimates the radial relative velocity between pairs of
    particles considering both turbulent velocity dispersion and gravitational
    acceleration. The conceptual form is:

    - ⟨|wᵣ|⟩ = √(2/π) × √(σ² + (π/8) × (τₚ₁ - τₚ₂)² × g²)
        - wᵣ is the radial relative velocity in m/s,
        - σ is the turbulence velocity dispersion in m/s,
        - τₚ₁, τₚ₂ are the inertia timescales (s),
        - g is the gravitational acceleration (m/s²).

    Arguments:
        - velocity_dispersion : Turbulence velocity dispersion (σ) in m/s.
        - particle_inertia_time : Inertia timescale(s) of particles [s].

    Returns:
        - The radial relative velocity ⟨|wᵣ|⟩ in m/s.

    Examples:
        ```py
        import numpy as np
        import particula as par

        # Example usage (currently raises NotImplementedError)
        try:
            rv = par.dynamics.get_radial_relative_velocity_ao2008(
                1.0, np.array([0.05, 0.1])
            )
        except NotImplementedError as e:
            print(e)
        ```

    References:
        - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
          the geometric collision rate of sedimenting droplets. Part 2. Theory
          and parameterization. New Journal of Physics, 10.
    """
    # tau_delta = (
    #     particle_inertia_time[:, np.newaxis]
    #     - particle_inertia_time[np.newaxis, :]
    # )
    # print(f"tau_delta: {tau_delta}")

    # gravity_term = (np.pi / 8) * tau_delta**2 * STANDARD_GRAVITY**2

    # return np.sqrt(2 / np.pi) * np.sqrt(velocity_dispersion + gravity_term)
    raise NotImplementedError("This function is not yet right.")
