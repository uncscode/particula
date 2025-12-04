"""Fit coefficients for the Binary Activity Coefficient.

Gorkowski, K., Preston, T. C., &#38; Zuend, A. (2019).
Relative-humidity-dependent organic aerosol thermodynamics
Via an efficient reduced-complexity model.
Atmospheric Chemistry and Physics
https://doi.org/10.5194/acp-19-13383-2019
"""

from typing import List, NamedTuple, Union

import numpy as np
from numpy.typing import NDArray


class FitValues(NamedTuple):
    """Named tuple for the fit values for the activity model."""

    a1: List[float]
    a2: List[float]
    s: List[float]


G19_FIT_LOW = FitValues(
    a1=[7.089476e00, -7.711860e00, -3.885941e01, -1.000000e02],
    a2=[-6.226781e-01, -1.000000e02, 3.081244e-09, 6.188812e01],
    s=[-5.988895e00, 6.940689e00],
)
G19_FIT_MID = FitValues(
    a1=[5.872214e00, -4.535007e00, -5.129327e00, -2.809232e01],
    a2=[-9.740486e-01, -1.000000e02, 2.109751e00, -2.367683e01],
    s=[-1.219164e00, 4.742729e00],
)
G19_FIT_HIGH = FitValues(
    a1=[5.921550e00, -2.528295e00, -3.883017e00, -7.898128e00],
    a2=[-1.000000e02, -1.000000e02, 1.353916e00, -1.160145e01],
    s=[-7.868187e-02, 3.650860e00],
)
INTERPOLATE_WATER_FIT = 500
LOWEST_ORGANIC_MOLE_FRACTION = 1e-12


def coefficients_c(
    molar_mass_ratio: Union[float, NDArray[np.float64]],
    oxygen2carbon: Union[float, NDArray[np.float64]],
    fit_values: List[float],
) -> NDArray[np.float64]:
    """Coefficients for activity model, see Gorkowski (2019). equation S1 S2.

    Args:
        - molar_mass_ratio : The molar mass ratio of water to organic
          matter.
        - oxygen2carbon : The oxygen to carbon ratio.
        - fit_values : The fit values for the activity model.

    Returns:
        - The coefficients for the activity model.
    """
    # force to array
    molar_mass_ratio = np.asarray(molar_mass_ratio, dtype=np.float64)
    oxygen2carbon = np.asarray(oxygen2carbon, dtype=np.float64)
    fit_values_array: NDArray[np.float64] = np.asarray(
        fit_values, dtype=np.float64
    )

    return fit_values_array[0] * np.exp(
        fit_values_array[1] * oxygen2carbon
    ) + fit_values_array[2] * np.exp(fit_values_array[3] * molar_mass_ratio)
