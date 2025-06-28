"""Particle inertia time calculation."""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "particle_radius": "nonnegative",
        "particle_density": "positive",
        "fluid_density": "positive",
        "kinematic_viscosity": "positive",
    }
)
def get_particle_inertia_time(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    fluid_density: Union[float, NDArray[np.float64]],
    kinematic_viscosity: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Compute the particle inertia time (τ_p).

    The particle inertia time represents the response time of a particle to
    changes in fluid velocity, given by:

    - τ_p = (2 / 9) × (ρ_p / ρ_f) × (r² / ν)
        - τ_p is the particle inertia time in seconds (s).
        - ρ_p is the particle density (kg/m³).
        - ρ_f is the surrounding fluid density (kg/m³).
        - r is the particle radius (m).
        - ν is the kinematic viscosity of the fluid (m²/s).

    Arguments:
        - particle_radius : Particle radius in meters (m).
        - particle_density : Density of the particle in kg/m³.
        - fluid_density : Density of the fluid in kg/m³.
        - kinematic_viscosity : Kinematic viscosity of the fluid in m²/s.

    Returns:
        - The particle inertia time in seconds (s). Returned as either a float
            or NDArray[np.float64].

    Examples:
        ```py title="Example"
        import particula as par
        par.particles.get_particle_inertia_time(
            particle_radius=1e-6,
            particle_density=1000.0,
            fluid_density=1.225,
            kinematic_viscosity=1.5e-5
        )
        # Output: ...
        ```

    References:
        - Ayala, O., Rosa, B., & Wang, L. P. (2008). "Effects of turbulence on
          the geometric collision rate of sedimenting droplets. Part 2. Theory
          and parameterization." New Journal of Physics, 10.
          https://doi.org/10.1088/1367-2630/10/7/075016
    """
    return (
        (2 / 9)
        * (particle_density / fluid_density)
        * (particle_radius**2 / (kinematic_viscosity))
    )
