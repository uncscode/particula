"""Module for calculating the surface tension of pure chemicals."""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from .thermo_import import Chemical


def get_chemical_surface_tension(
    chemical_identifier: str, temperature: Union[float, NDArray[np.float64]]
) -> NDArray[np.float64]:
    """Calculate the surface tension of a pure chemical in N/m.

    The calculation delegates to ``thermo.chemical.Chemical.SurfaceTension``,
    evaluated at the requested temperature(s):

    - σ = σ(T)
        - σ is the surface tension in newtons per metre (N/m),
        - T is the absolute temperature in kelvin (K).

    Arguments:
        - chemical_identifier : Identifier accepted by
          ``thermo.chemical.Chemical`` (name, CAS number, or formula).
        - temperature : Scalar or array of temperatures in kelvin (K).

    Returns:
        - Surface tension(s) in N/m with the same shape as ``temperature``.
          A scalar input returns a 0-d ``numpy.ndarray`` for consistency.

    Examples:
        ``` py title="Scalar temperature"
        from particula.util.materials.surface_tension import \
            get_surface_tension
        get_surface_tension("water", 298.15)
        # Output: 0.072 (≈ N/m)
        ```

        ``` py title="Array temperature"
        import numpy as np
        from particula.util.materials.surface_tension import \
            get_surface_tension
        temps = np.array([280.0, 298.15, 320.0])
        get_surface_tension("water", temps)
        # Output: array([0.076, 0.072, 0.065])
        ```

    References:
        - "Surface tension",
          [Wikipedia](https://en.wikipedia.org/wiki/Surface_tension)
        - D. R. Lide, *CRC Handbook of Chemistry and Physics*, 90th ed.,
          CRC Press, 2009.
    """
    if Chemical is None:
        raise ImportError(
            "The 'thermo' package is required for vapor pressure calculations."
            "Please install it using 'pip install thermo'."
        )

    temps = np.asarray(temperature, dtype=np.float64)
    chem = Chemical(chemical_identifier)

    return np.vectorize(
        lambda T: chem.SurfaceTension(T=T),  # noqa: N803
        otypes=[np.float64],
    )(temps)
