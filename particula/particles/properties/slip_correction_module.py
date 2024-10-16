""" Module for calculate slip correction
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np


def cunningham_slip_correction(
    knudsen_number: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the Cunningham slip correction factor. Accounts for
    non-continuum effects on small particles.

    Args:
    -----
    - knudsen_number: Knudsen number [unitless].

    Returns:
    --------
    - Slip correction factor [unitless].

    Reference:
    ----------
    - Dimensionless quantity accounting for non-continuum effects
    on small particles. It is a deviation from Stokes' Law.
    Stokes assumes a no-slip condition that is not correct at
    high Knudsen numbers. The slip correction factor is used to
    calculate the friction factor.
    Thus, the slip correction factor is about unity (1) for larger
    particles (Kn -> 0). Its behavior on the other end of the
    spectrum (smaller particles; Kn -> inf) is more nuanced, though
    it tends to scale linearly on a log-log scale, log Cc vs log Kn.
    - https://en.wikipedia.org/wiki/Cunningham_correction_factor
    """
    return 1 + knudsen_number * (1.257 + 0.4 * np.exp(-1.1 / knudsen_number))
