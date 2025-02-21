"""Module for calculate slip correction"""

from typing import Union
from numpy.typing import NDArray
import numpy as np


def cunningham_slip_correction(
    knudsen_number: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the Cunningham slip correction factor for small particles in a fluid.

    The slip correction factor (C_c) accounts for non-continuum effects on small
    particles, correcting for the no-slip assumption used in Stokes' law. It is
    calculated using:

    - C_c = 1 + Kn × (1.257 + 0.4 × exp(-1.1 / Kn))
        - Kn is the dimensionless Knudsen number.

    Arguments:
        - knudsen_number : Knudsen number (dimensionless).

    Returns:
        - Slip correction factor (dimensionless).

    Examples:
        ``` py title="Example"
        import numpy as np
        from particula.particles.properties.slip_correction_module import cunningham_slip_correction
        cc_factor = cunningham_slip_correction(0.1)
        print(cc_factor)
        # Output: ...
        ```

    References:
        - "Cunningham correction factor," Wikipedia,
          https://en.wikipedia.org/wiki/Cunningham_correction_factor
    """
    return 1 + knudsen_number * (1.257 + 0.4 * np.exp(-1.1 / knudsen_number))
