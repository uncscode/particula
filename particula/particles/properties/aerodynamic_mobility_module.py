"""Module for aerodynamic mobility of a particle in a fluid."""

from typing import Union
import logging
import numpy as np
from numpy.typing import NDArray

from particula.util.validate_inputs import validate_inputs

logger = logging.getLogger("particula")


@validate_inputs(
    {
        "particle_radius": "positive",
        "slip_correction_factor": "nonnegative",
        "dynamic_viscosity": "positive",
    }
)
def get_aerodynamic_mobility(
    particle_radius: Union[float, NDArray[np.float64]],
    slip_correction_factor: Union[float, NDArray[np.float64]],
    dynamic_viscosity: float,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the aerodynamic mobility of a particle using classical fluid
    mechanics.

    The aerodynamic mobility (B) can be determined by:

    - B = C / (6πμr)
        - B is the aerodynamic mobility (m²/s).
        - C is the slip correction factor (dimensionless).
        - μ is the dynamic viscosity of the fluid (Pa·s).
        - r is the radius of the particle (m).

    Arguments:
        - particle_radius : The radius of the particle in meters.
        - slip_correction_factor : Slip correction factor (dimensionless).
        - dynamic_viscosity : Dynamic viscosity of the fluid in Pa·s.

    Returns:
        - The particle aerodynamic mobility in m²/s.

    Examples:
        ``` py title="Example"
        import particula as par
        par.particles.get_particle_aerodynamic_mobility(
            particle_radius=0.00005,
            slip_correction_factor=1.1,
            dynamic_viscosity=0.0000181
        )

    References:
        - Wikipedia contributors, "Stokes' Law," Wikipedia,
        https://en.wikipedia.org/wiki/Stokes%27_law.
    """
    return slip_correction_factor / (
        6 * np.pi * dynamic_viscosity * particle_radius
    )
