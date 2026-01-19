"""Gibbs free energy of mixing for a binary mixture.

Gorkowski, K., Preston, T. C., &#38; Zuend, A. (2019).
Relative-humidity-dependent organic aerosol thermodynamics
Via an efficient reduced-complexity model.
Atmospheric Chemistry and Physics
https://doi.org/10.5194/acp-19-13383-2019
"""

from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from particula.activity.bat_blending import bat_blending_weights
from particula.activity.bat_coefficients import (
    G19_FIT_HIGH,
    G19_FIT_LOW,
    G19_FIT_MID,
    FitValues,
    coefficients_c,
)
from particula.activity.convert_functional_group import (
    convert_to_oh_equivalent,
)
from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "molar_mass_ratio": "positive",
        "organic_mole_fraction": "nonnegative",
        "oxygen2carbon": "nonnegative",
        "density": "positive",
    }
)
def gibbs_of_mixing(
    molar_mass_ratio: Union[float, NDArray[np.float64]],
    organic_mole_fraction: Union[float, NDArray[np.float64]],
    oxygen2carbon: Union[float, NDArray[np.float64]],
    density: Union[float, NDArray[np.float64]],
    fit_dict: FitValues,
) -> Tuple[
    Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]]
]:
    """Calculate the Gibbs free energy of mixing for a binary mixture.

    Args:
        - molar_mass_ratio : The molar mass ratio of water to organic
          matter.
        - organic_mole_fraction : The fraction of organic matter.
        - oxygen2carbon : The oxygen to carbon ratio.
        - density : The density of the mixture, in kg/m^3.
        - fit_dict : A dictionary of fit values for the low oxygen2carbon
            region

    Returns:
        - A tuple containing the Gibbs free energy of mixing and its
          derivative.
    """
    molar_mass_ratio = np.asarray(molar_mass_ratio, dtype=np.float64)
    organic_mole_fraction = np.asarray(organic_mole_fraction, dtype=np.float64)
    oxygen2carbon = np.asarray(oxygen2carbon, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)

    c1 = coefficients_c(molar_mass_ratio, oxygen2carbon, fit_dict.a1)
    c2 = coefficients_c(molar_mass_ratio, oxygen2carbon, fit_dict.a2)

    rhor = 997.0 / density  # assumes water is the other fluid

    scaled_molar_mass_ratio = (
        molar_mass_ratio
        * fit_dict.s[1]
        * (1.0 + oxygen2carbon) ** fit_dict.s[0]
    )

    phi2 = organic_mole_fraction / (
        organic_mole_fraction
        + (1.0 - organic_mole_fraction) * scaled_molar_mass_ratio / rhor
    )

    sum1 = c1 + c2 * (1 - 2 * phi2)
    gibbs_mix = phi2 * (1.0 - phi2) * sum1

    # Initialize result with zeros
    dphi2dx2 = np.zeros_like(organic_mole_fraction)
    non_zero = organic_mole_fraction != 0
    dphi2dx2[non_zero] = (scaled_molar_mass_ratio / rhor) * (
        phi2[non_zero] / organic_mole_fraction[non_zero]
    ) ** 2

    derivative_gibbs_mix = (
        (1.0 - 2.0 * phi2) * sum1 - 2 * c2 * phi2 * (1.0 - phi2)
    ) * dphi2dx2

    return gibbs_mix, derivative_gibbs_mix


# pylint: disable=too-many-locals
def gibbs_mix_weight(
    molar_mass_ratio: Union[float, NDArray[np.float64]],
    organic_mole_fraction: Union[float, NDArray[np.float64]],
    oxygen2carbon: Union[float, NDArray[np.float64]],
    density: Union[float, NDArray[np.float64]],
    functional_group: Optional[str] = None,
) -> Tuple[
    Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]]
]:
    """Calculate weighted Gibbs free energy of mixing for the BAT model.

    Blends Gibbs free energy fits across low, mid, and high oxygen-to-carbon
    (O:C) regions to avoid discontinuities in activity coefficient
    calculations. The blending weights come from :func:`bat_blending_weights`
    and transition smoothly between the Gorkowski et al. (2019) fits. Only one
    compound is processed per call; vector inputs are iterated elementwise.

    Args:
        molar_mass_ratio: Ratio of water to organic molar mass
            (:math:`MW_{water} / MW_{organic}`). Dimensionless.
        organic_mole_fraction: Mole fraction of the organic component.
            Range: [0, 1].
        oxygen2carbon: Oxygen-to-carbon atomic ratio of the organic compound.
        density: Density of the organic compound in kg/m^3.
        functional_group: Optional functional group for OH-equivalent
            conversion (e.g., "alcohol", "carboxylic_acid", "ether").

    Returns:
        Tuple containing:
            gibbs_mix: Gibbs energy of mixing normalized by :math:`RT`
                (dimensionless).
            derivative_gibbs: Derivative of Gibbs energy with respect to the
                organic mole fraction, normalized by :math:`RT`.

    Examples:
        >>> from particula.activity import gibbs_mix_weight
        >>> g_mix, dg_dx = gibbs_mix_weight(
        ...     molar_mass_ratio=0.09,
        ...     organic_mole_fraction=0.3,
        ...     oxygen2carbon=0.4,
        ...     density=1400.0,
        ... )

    References:
        Gorkowski et al. (2019), Section 2.2 and SI S2.
        https://doi.org/10.5194/acp-19-13383-2019
    """
    density = np.asarray(density, dtype=np.float64)

    oxygen2carbon, molar_mass_ratio = convert_to_oh_equivalent(
        oxygen2carbon=oxygen2carbon,
        molar_mass_ratio=molar_mass_ratio,
        functional_group=functional_group,
    )

    weights = bat_blending_weights(
        molar_mass_ratio=molar_mass_ratio, oxygen2carbon=oxygen2carbon
    )

    if np.size(oxygen2carbon) == 1:
        # Cast scalar arrays to float for single value case
        return _calculate_gibbs_mix_single(
            float(np.asarray(molar_mass_ratio).flat[0]),
            float(np.asarray(organic_mole_fraction).flat[0]),
            float(np.asarray(oxygen2carbon).flat[0]),
            float(density.flat[0]),
            weights,
        )

    # Cast to arrays for iteration
    oxygen2carbon_arr = np.asarray(oxygen2carbon, dtype=np.float64)
    molar_mass_ratio_arr = np.asarray(molar_mass_ratio, dtype=np.float64)
    organic_mole_fraction_arr = np.asarray(
        organic_mole_fraction, dtype=np.float64
    )
    density_arr = np.asarray(density, dtype=np.float64)

    gibbs_mix = np.zeros((len(oxygen2carbon_arr)))
    derivative_gibbs = np.zeros((len(oxygen2carbon_arr)))

    for i, o2c in enumerate(oxygen2carbon_arr):
        gibbs_mix[i], derivative_gibbs[i] = _calculate_gibbs_mix_single(
            molar_mass_ratio=float(molar_mass_ratio_arr[i]),
            organic_mole_fraction=float(organic_mole_fraction_arr[i]),
            oxygen2carbon=float(o2c),
            density=float(density_arr[i]),
            weights=weights[i],
        )

    return gibbs_mix, derivative_gibbs


def _calculate_gibbs_mix_single(
    molar_mass_ratio: float,
    organic_mole_fraction: float,
    oxygen2carbon: float,
    density: float,
    weights: NDArray[np.float64],
) -> Tuple[
    Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]]
]:
    """Compute Gibbs free energy for one compound using blended weights.

    Uses the precomputed blending weights to select the appropriate fit
    combinations (low/mid or mid/high O:C) from Gorkowski et al. (2019).

    Args:
        molar_mass_ratio: Ratio of water to organic molar mass.
        organic_mole_fraction: Mole fraction of the organic component.
        oxygen2carbon: Oxygen-to-carbon atomic ratio.
        density: Density of the organic compound in kg/m^3.
        weights: Three-element array of blending weights for low, mid, high
            O:C regions.

    Returns:
        gibbs_mix: Gibbs energy of mixing normalized by :math:`RT`.
        derivative_gibbs: Derivative of Gibbs energy with respect to the
            organic mole fraction, normalized by :math:`RT`.
    """
    if weights[1] > 0:  # if mid region is used
        gibbs_mix_mid, derivative_gibbs_mid = gibbs_of_mixing(
            molar_mass_ratio=molar_mass_ratio,
            organic_mole_fraction=organic_mole_fraction,
            oxygen2carbon=oxygen2carbon,
            density=density,
            fit_dict=G19_FIT_MID,
        )

        if weights[0] > 0:  # if paired with low oxygen2carbon region
            gibbs_mix_low, derivative_gibbs_low = gibbs_of_mixing(
                molar_mass_ratio=molar_mass_ratio,
                organic_mole_fraction=organic_mole_fraction,
                oxygen2carbon=oxygen2carbon,
                density=density,
                fit_dict=G19_FIT_LOW,
            )
            gibbs_mix = weights[0] * gibbs_mix_low + weights[1] * gibbs_mix_mid
            derivative_gibbs = (
                weights[0] * derivative_gibbs_low
                + weights[1] * derivative_gibbs_mid
            )
        else:  # else paired with high oxygen2carbon region
            gibbs_mix_high, derivative_gibbs_high = gibbs_of_mixing(
                molar_mass_ratio=molar_mass_ratio,
                organic_mole_fraction=organic_mole_fraction,
                oxygen2carbon=oxygen2carbon,
                density=density,
                fit_dict=G19_FIT_HIGH,
            )
            gibbs_mix = weights[2] * gibbs_mix_high + weights[1] * gibbs_mix_mid
            derivative_gibbs = (
                weights[2] * derivative_gibbs_high
                + weights[1] * derivative_gibbs_mid
            )
    else:  # when only high 2OC region is used
        gibbs_mix, derivative_gibbs = gibbs_of_mixing(
            molar_mass_ratio=molar_mass_ratio,
            organic_mole_fraction=organic_mole_fraction,
            oxygen2carbon=oxygen2carbon,
            density=density,
            fit_dict=G19_FIT_HIGH,
        )
    return gibbs_mix, derivative_gibbs
