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
    Calculate the particle Reynolds number.

    The particle Reynolds number (Re_p) characterizes the flow behavior around
    a particle in a fluid and is given by:

        Re_p = (2 a v_p) / ν

        - Re_p : Particle Reynolds number [-]
        - a (particle_radius) : Particle radius [m]
        - v_p (particle_velocity) : Particle velocity relative to the fluid
            [m/s]
        - ν (kinematic_viscosity) : Kinematic viscosity of the fluid [m²/s]



    Arguments:
    ----------
        - particle_radius : Radius of the particle [m]
        - particle_velocity : Particle velocity relative to the fluid [m/s]
        - kinematic_viscosity : Kinematic viscosity of the surrounding fluid [m²/s]

    Returns:
    --------
        - Particle Reynolds number [-]

    References:
    -----------
        - Wikipedia https://en.wikipedia.org/wiki/Reynolds_number
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
