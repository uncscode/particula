"""Calculates the Gibbs free energy of mixing for a binary solution."""

import warnings
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from particula.util.machine_limit import get_safe_log


def gibbs_free_energy(
    organic_mole_fraction: NDArray[np.float64],
    gibbs_mix: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute ideal and real Gibbs free energy of mixing.

    Args:
        organic_mole_fraction: Mole fraction of the organic component.
            Values should be between 0 and 1.
        gibbs_mix: Gibbs free energy of mixing (non-ideal contribution).

    Returns:
        Tuple containing:
            gibbs_ideal: Ideal Gibbs free energy of mixing.
            gibbs_real: Real Gibbs free energy of mixing (ideal + mix).

    Examples:
        >>> import numpy as np
        >>> from particula.activity.gibbs import gibbs_free_energy
        >>> x_org = np.array([0.2, 0.5, 0.8])
        >>> g_mix = np.array([0.1, 0.2, 0.1])
        >>> g_ideal, g_real = gibbs_free_energy(x_org, g_mix)
        >>> g_ideal.shape, g_real.shape
        ((3,), (3,))
    """
    gibbs_ideal = (1 - organic_mole_fraction) * get_safe_log(
        1 - organic_mole_fraction
    ) + organic_mole_fraction * get_safe_log(organic_mole_fraction)
    gibbs_real = gibbs_ideal + gibbs_mix
    return gibbs_ideal, gibbs_real


def gibbs_free_engery(
    organic_mole_fraction: NDArray[np.float64],
    gibbs_mix: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Deprecated alias for gibbs_free_energy.

    .. deprecated:: 0.3.0
        Use :func:`gibbs_free_energy` instead. This alias will be removed
        in a future version.
    """
    warnings.warn(
        "gibbs_free_engery is deprecated and will be removed in a future "
        "version. Use gibbs_free_energy instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return gibbs_free_energy(organic_mole_fraction, gibbs_mix)
