"""Module for retrieving saturation-vapor pressure of chemicals."""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from .thermo_import import Chemical


def get_chemical_vapor_pressure(
    chemical_identifier: str, temperature: Union[float, NDArray[np.float64]]
) -> NDArray[np.float64]:
    """Retrieve the saturation-vapor pressure of a chemical at temperature.

    The saturation-vapor pressure (Pₛₐₜ) is obtained by calling the correlation
    implemented in ``thermo.chemical.Chemical``:

    - Pₛₐₜ = Chemical(chemical_identifier).VaporPressure(T)
        - Pₛₐₜ is saturation-vapor pressure in Pascals (Pa),
        - T is temperature in Kelvin.

    Arguments:
        - chemical_identifier : Identifier accepted by
          ``thermo.chemical.Chemical`` (name, CAS number, or formula).
        - temperature : Temperature(s) in Kelvin. May be a scalar or a NumPy
          array.

    Returns:
        - Saturation-vapor pressure(s) in Pascals (Pa). For a scalar
          temperature, a 0-d NumPy array is returned for consistency.

    Examples:
        ``` py title="Example Usage – scalar input"
        from particula.util.materials.vapor_pressure import get_vapor_pressure
        p_sat = get_vapor_pressure("water", 298.15)
        # p_sat ≈ 3169.0  # Pa
        ```

        ``` py title="Example Usage – vectorised input"
        import numpy as np
        from particula.util.materials.vapor_pressure import get_vapor_pressure
        temps = np.array([280.0, 290.0, 300.0])
        p_sat = get_vapor_pressure("water", temps)
        # p_sat → array([...])  # Pa
        ```

    References:
        - "Vapour pressure,"
          [Wikipedia](https://en.wikipedia.org/wiki/Vapour_pressure)
        - R. H. Perry & D. W. Green, *Perry's Chemical Engineers' Handbook*,
          8th ed., McGraw-Hill, 2007.
    """
    if Chemical is None:
        raise ImportError(
            "The 'thermo' package is required for vapor pressure calculations."
            " Please install it using 'pip install thermo'."
        )

    temps = np.asarray(temperature, dtype=np.float64)
    chem = Chemical(chemical_identifier)

    return np.vectorize(
        lambda T: chem.VaporPressure(T=T),  # noqa: N803
        otypes=[np.float64],
    )(temps)
