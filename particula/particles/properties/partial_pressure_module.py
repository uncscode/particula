"""Module for calculating the partial pressure of a species in a
gas over particle phase.
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray


def get_partial_pressure_delta(
    partial_pressure_gas: Union[float, NDArray[np.float64]],
    partial_pressure_particle: Union[float, NDArray[np.float64]],
    kelvin_term: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Calculate the difference in partial pressure between gas and particle
    phase, considering the Kelvin effect.

    - Δp = p_gas − (p_particle × K)
        - p_gas is the partial pressure in the gas phase,
        - p_particle is the partial pressure in the particle phase,
        - K is the Kelvin term (dimensionless).

    Arguments:
        - partial_pressure_gas : Partial pressure of the species in the gas
            phase.
        - partial_pressure_particle : Partial pressure of the species in
            the particle phase.
        - kelvin_term : Dimensionless Kelvin effect factor due to particle
            curvature.

    Returns:
        - The difference in partial pressure, as either a float or
            NDArray[np.float64].

    Examples:
        ``` py title="Example"
        import particula as par
        par.particles.get_partial_pressure_delta(
            partial_pressure_gas=1000.0,
            partial_pressure_particle=900.0,
            kelvin_term=1.01
        )
        # Output: 1000.0 - (900.0 * 1.01) = 91.0
        ```

    References:
        - [Kelvin effect, Wikipedia](https://en.wikipedia.org/wiki/Kelvin_equation)
        - [Partial pressure, Wikipedia](https://en.wikipedia.org/wiki/Partial_pressure)
    """
    return partial_pressure_gas - partial_pressure_particle * kelvin_term
