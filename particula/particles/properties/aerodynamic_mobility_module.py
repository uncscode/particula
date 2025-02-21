"""Module for aerodynamic mobility of a particle in a fluid."""

from typing import Union
import logging
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger("particula")


def particle_aerodynamic_mobility(
    radius: Union[float, NDArray[np.float64]],
    slip_correction_factor: Union[float, NDArray[np.float64]],
    dynamic_viscosity: float,
) -> Union[float, NDArray[np.float64]]:
"""
Calculate the aerodynamic mobility of a particle using classical fluid mechanics.

The aerodynamic mobility (B) can be determined by:

- B = C / (6πμr)
    - B is the aerodynamic mobility (m²/s).
    - C is the slip correction factor (dimensionless).
    - μ is the dynamic viscosity of the fluid (Pa·s).
    - r is the radius of the particle (m).

Arguments:
    - radius : The radius of the particle in meters.
    - slip_correction_factor : Slip correction factor (dimensionless).
    - dynamic_viscosity : Dynamic viscosity of the fluid in Pa·s.

Returns:
    - The particle aerodynamic mobility in m²/s.

Examples:
    ``` py title="Example"
    from particula.particles.properties.aerodynamic_mobility_module import particle_aerodynamic_mobility
    aerodynamic_mobility = particle_aerodynamic_mobility(
        radius=0.00005,
        slip_correction_factor=1.1,
        dynamic_viscosity=0.0000181
    )
    # Output: ...
    ```

References:
    - Wikipedia contributors, "Stokes' Law," Wikipedia,
      https://en.wikipedia.org/wiki/Stokes%27_law.
"""
    """
    # Validate radius
    if (
        np.any(radius <= 0)
        or np.any(np.isnan(radius))
        or np.any(np.isinf(radius))
    ):
        message = "The radius must be a positive, finite number."
        logger.error(message)
        raise ValueError(message)

    return slip_correction_factor / (6 * np.pi * dynamic_viscosity * radius)
