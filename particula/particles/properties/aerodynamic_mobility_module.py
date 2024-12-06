"""Module for aerodynamic mobility of a particle in a fluid.
"""

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
    Calculate the aerodynamic mobility of a particle.

    This is defined as the ratio of the slip correction factor to the product
    of the dynamic viscosity of the fluid, the particle radius, and a slip
    correction constant derived. This mobility quantifies the ease with which
    a particle can move through a fluid.

    Args:
        - radius : The radius of the particle (m).
        - slip_correction_factor : The slip correction factor for the particle
             in the fluid (dimensionless).
        - dynamic_viscosity : The dynamic viscosity of the fluid (Pa.s).

    Returns:
        The particle aerodynamic mobility (m^2/s).

    Example:
        ``` py title="Example"
        aerodynamic_mobility = particle_aerodynamic_mobility(
            radius=0.00005,
            slip_correction_factor=1.1,
            dynamic_viscosity=0.0000181,
        )
        ```
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
