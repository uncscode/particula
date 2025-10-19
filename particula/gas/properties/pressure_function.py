"""Functions for calculating the partial pressure of a gas."""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.util.constants import GAS_CONSTANT
from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "concentration": "nonnegative",
        "molar_mass": "positive",
        "temperature": "positive",
    }
)
def get_partial_pressure(
    concentration: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Calculate partial pressure from concentration, molar mass, T.

    - p = (c × R × T) / M
        - p is Partial pressure [Pa].
        - c is Gas concentration [kg/m³].
        - R is Universal gas constant [J/(mol·K)].
        - T is Temperature [K].
        - M is Molar mass [kg/mol].

    Arguments:
        - concentration : Concentration of the gas [kg/m³].
        - molar_mass : Molar mass of the gas [kg/mol].
        - temperature : Temperature [K].

    Returns:
        - Partial pressure of the gas [Pa].

    Examples:
        ```py title="Example usage"
        import particula as par
        par.gas.get_partial_pressure(1.2, 0.02897, 298)
        # Output: ~986.4 Pa
        ```

    References:
        - Wikipedia contributors, "Ideal gas law," Wikipedia,
          https://en.wikipedia.org/wiki/Ideal_gas_law
    """
    # Calculate the partial pressure
    return (concentration * float(GAS_CONSTANT) * temperature) / molar_mass


@validate_inputs(
    {
        "partial_pressure": "positive",
        "pure_vapor_pressure": "positive",
    }
)
def get_saturation_ratio_from_pressure(
    partial_pressure: Union[float, NDArray[np.float64]],
    pure_vapor_pressure: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Calculate the saturation ratio of the gas at a given partial pressure and
    pure vapor pressure.

    The saturation ratio is defined as the ratio of partial pressure to the
    pure vapor pressure.

    - S = p / p_vap
        - S is Saturation ratio (dimensionless).
        - p is Partial pressure [Pa].
        - p_vap is Pure vapor pressure [Pa].

    Arguments:
        - partial_pressure : Partial pressure [Pa].
        - pure_vapor_pressure : Pure vapor pressure [Pa].

    Returns:
        - Saturation ratio of the gas (dimensionless).

    Examples:
        ```py title="Example usage"
        import particula as par
        par.gas.get_saturation_ratio_from_pressure(800.0, 1000.0)
        # Output: 0.8
        ```

    References:
        - Wikipedia contributors, "Relative humidity," Wikipedia,
          https://en.wikipedia.org/wiki/Relative_humidity
    """
    return partial_pressure / pure_vapor_pressure
