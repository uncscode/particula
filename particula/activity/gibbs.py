"""Calculates the Gibbs free energy of mixing for a binary solution."""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from particula.util.machine_limit import get_safe_log


def gibbs_free_engery(
    organic_mole_fraction: NDArray[np.float64],
    gibbs_mix: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate the gibbs free energy of the mixture. Ideal and non-ideal.

    Args:
        - organic_mole_fraction : A numpy array of organic mole
            fractions.
        - gibbs_mix : A numpy array of gibbs free energy of mixing.

    Returns:
        - gibbs_ideal : The ideal gibbs free energy of mixing.
        - gibbs_real : The real gibbs free energy of mixing.
    """
    gibbs_ideal = (1 - organic_mole_fraction) * get_safe_log(
        1 - organic_mole_fraction
    ) + organic_mole_fraction * get_safe_log(organic_mole_fraction)
    gibbs_real = gibbs_ideal + gibbs_mix
    return gibbs_ideal, gibbs_real
