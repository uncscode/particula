"""Module for the vapor transition correction function, which accounts for the
intermediate regime between continuum and free molecular flow. This is the
Suchs and Futugin transition function.
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "knudsen_number": "nonnegative",
        "mass_accommodation": "nonnegative",
    }
)
def get_vapor_transition_correction(
    knudsen_number: Union[float, NDArray[np.float64]],
    mass_accommodation: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Calculate the Fuchs–Sutugin vapor transition correction factor.

    This correction factor (f) accounts for the transition regime between free
    molecular flow and continuum diffusion when computing mass or heat
    transport.

    Mathematically:

    - f(Kn, α) = [0.75·α·(1+Kn)] / [Kn² + Kn + 0.283·α·Kn + 0.75·α]
        - Kn is the Knudsen number (dimensionless),
        - α is the mass accommodation coefficient (dimensionless).

    Arguments:
        - knudsen_number : Dimensionless Knudsen number.
        - mass_accommodation : Mass accommodation coefficient (dimensionless).

    Returns:
        - Transition correction factor (float or NDArray[np.float64]).

    Examples:
        ``` py title="Example"
        import particula as par
        par.particles.get_vapor_transition_correction(
            knudsen_number=0.1, mass_accommodation=1.0
        )
        # Output: 0.73...
        ```

    References:
        - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry
          and Physics, Ch. 12. Equation 12.43.
        - Fuchs, N. A., & Sutugin, A. G. (1971). *High-Dispersed Aerosols*.
          In *Topics in Current Aerosol Research*, Elsevier, pp. 1–60.
    """
    return (0.75 * mass_accommodation * (1 + knudsen_number)) / (
        (knudsen_number**2 + knudsen_number)
        + 0.283 * mass_accommodation * knudsen_number
        + 0.75 * mass_accommodation
    )
