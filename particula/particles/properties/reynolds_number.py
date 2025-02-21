"""
Calculate the particle Reynolds number.
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "particle_radius": "positive",
        "particle_velocity": "positive",
        "kinematic_viscosity": "positive",
    }
)
def get_particle_reynolds_number(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_velocity: Union[float, NDArray[np.float64]],
    kinematic_viscosity: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the Reynolds number (Reₚ) of a particle in a fluid.

    This dimensionless quantity characterizes the flow regime:

    - Reₚ = (2 × a × vₚ) / ν
        - a is the particle radius in meters (m).
        - vₚ is the particle velocity in meters/second (m/s).
        - ν is the kinematic viscosity in square meters/second (m²/s).

    Arguments:
        - particle_radius : Particle radius (m).
        - particle_velocity : Particle velocity relative to the fluid (m/s).
        - kinematic_viscosity : Kinematic viscosity of the fluid (m²/s).

    Returns:
        - Dimensionless Reynolds number (float or NDArray[np.float64]).

    Examples:
        ```py title="Example"
        import particula as par
        par.particles.get_particle_reynolds_number(
            particle_radius=1e-6,
            particle_velocity=0.1,
            kinematic_viscosity=1.5e-5
        )
        # Output: ...
        ```

    References:
        - [Reynolds number, Wikipedia](https://en.wikipedia.org/wiki/Reynolds_number)
        - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry and
            Physics,
        - **Stokes Flow (Viscous Dominated, Re_p < 1)**:
            - Particles follow the fluid closely (e.g., aerosols).
        - **Transitional Flow (1 < Re_p < 1000)**:
            - Both **viscous and inertial forces** contribute to flow behavior.
            - Intermediate drag corrections apply.
        - **Turbulent Flow (Re_p > 1000)**:
            - **Inertial forces dominate**, resulting in vortex shedding and
                wake formation.
            - Applies to **large, fast-moving particles**
                (e.g., raindrops, large sediment).
    """
    return (2 * particle_radius * particle_velocity) / kinematic_viscosity
