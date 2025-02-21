"""Module for calculating the partial pressure of a species in a
gas over particle phase."""

from typing import Union
from numpy.typing import NDArray
import numpy as np


def get_partial_pressure_delta(
    partial_pressure_gas: Union[float, NDArray[np.float64]],
    partial_pressure_particle: Union[float, NDArray[np.float64]],
    kelvin_term: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the difference in partial pressure between gas and particle phase,
    considering the Kelvin effect.

    This function computes:
        Δp = p_gas − (p_particle × K)
    where:
        - p_gas is the partial pressure in the gas phase,
        - p_particle is the partial pressure in the particle phase,
        - K is the Kelvin term (dimensionless).

    Arguments:
        - partial_pressure_gas : Partial pressure of the species in the gas phase.
        - partial_pressure_particle : Partial pressure of the species in
          the particle phase.
        - kelvin_term : Dimensionless Kelvin effect factor due to particle curvature.

    Returns:
        - The difference in partial pressure, as either a float or NDArray[np.float64].

    Examples:
        ``` py title="Example"
        from particula.particles.properties.partial_pressure_module import get_partial_pressure_delta
        difference = get_partial_pressure_delta(
            partial_pressure_gas=1000.0,
            partial_pressure_particle=900.0,
            kelvin_term=1.01
        )
        print(difference)
        # Output: 1000.0 - (900.0 * 1.01) = 91.0
        ```

    References:
        - Kelvin effect, "Kelvin equation," Wikipedia,
          https://en.wikipedia.org/wiki/Kelvin_equation
    """
    return partial_pressure_gas - partial_pressure_particle * kelvin_term
