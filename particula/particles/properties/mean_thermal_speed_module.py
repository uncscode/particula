"""Module contains the function for calculating the mean thermal speed
of particles in a fluid.
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.util.constants import BOLTZMANN_CONSTANT
from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "particle_mass": "nonnegative",
        "temperature": "positive",
    }
)
def get_mean_thermal_speed(
    particle_mass: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Calculate the mean thermal speed of a particle in a fluid.

    The mean thermal speed (v) is derived from kinetic theory and is given by:

    - v = √( (8 × k_B × T) / (π × m) )
        - v is the mean thermal speed in m/s,
        - k_B is the Boltzmann constant in J/K,
        - T is the temperature in Kelvin (K),
        - m is the particle mass in kilograms (kg).

    Arguments:
        - particle_mass : The mass of the particle(s) in kg.
        - temperature : The temperature of the system in Kelvin (K).

    Returns:
        - The mean thermal speed in m/s, as either a float or an
            NDArray[np.float64].

    Examples:
        ``` py title="Example"
        import particula as par
        par.particles.get_mean_thermal_speed(1e-17, 298)
        # Output: ...
        ```

    References:
        - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry and
          Physics, Section 9.5.3 Mean Free Path of an Aerosol Particle,
          Equation 9.87.
    """
    return np.sqrt(
        (8 * BOLTZMANN_CONSTANT * temperature) / (np.pi * particle_mass)
    )
