"""
Gibbs free energy of mixing for a binary mixture.

Gorkowski, K., Preston, T. C., &#38; Zuend, A. (2019).
Relative-humidity-dependent organic aerosol thermodynamics
Via an efficient reduced-complexity model.
Atmospheric Chemistry and Physics
https://doi.org/10.5194/acp-19-13383-2019
"""

from typing import Union, List, Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from particula.util.validate_inputs import validate_inputs
from particula.activity.bat_coefficients import (
    FIT_LOW,
    FIT_MID,
    FIT_HIGH,
    coefficients_c,
)
from particula.activity.bat_blending import bat_blending_weights
from particula.activity.convert_functional_group import (
    convert_to_oh_equivalent,
)


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
    fit_dict: Dict[str, List[float]],
) -> Tuple[
    Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]]
]:
    """
    Calculate the Gibbs free energy of mixing for a binary mixture.

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

    c1 = coefficients_c(molar_mass_ratio, oxygen2carbon, fit_dict["a1"])
    c2 = coefficients_c(molar_mass_ratio, oxygen2carbon, fit_dict["a2"])

    rhor = 997.0 / density  # assumes water is the other fluid

    scaled_molar_mass_ratio = (
        molar_mass_ratio
        * fit_dict["s"][1]
        * (1.0 + oxygen2carbon) ** fit_dict["s"][0]
    )

    phi2 = organic_mole_fraction / (
        organic_mole_fraction
        + (1.0 - organic_mole_fraction) * scaled_molar_mass_ratio / rhor
    )

    sum1 = c1 + c2 * (1 - 2 * phi2)
    gibbs_mix = phi2 * (1.0 - phi2) * sum1

    dphi2dx2 = (scaled_molar_mass_ratio / rhor) * (
        phi2 / organic_mole_fraction
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gibbs free energy of mixing, see Gorkowski (2019), with weighted
    oxygen2carbon regions. Only can run one compound at a time.

    Args:
        - molar_mass_ratio : The molar mass ratio of water to organic
          matter.
        - organic_mole_fraction : The fraction of organic matter.
        - oxygen2carbon : The oxygen to carbon ratio.
        - density : The density of the mixture, in kg/m^3.
        - functional_group : Optional functional group(s) of the organic
          compound, if applicable.

    Returns:
        - gibbs_mix : Gibbs energy of mixing (including 1/RT)
        - derivative_gibbs : derivative of Gibbs energy with respect to
          mole fraction of organics (includes 1/RT)
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
        return _calculate_gibbs_mix_single(
            molar_mass_ratio,
            organic_mole_fraction,
            oxygen2carbon,
            density,
            weights,
        )

    gibbs_mix = np.zeros((len(oxygen2carbon)))
    derivative_gibbs = np.zeros((len(oxygen2carbon)))

    for i, (mmr, omf, o2c, d, w) in enumerate(
        zip(
            molar_mass_ratio,
            organic_mole_fraction,
            oxygen2carbon,
            density,
            weights,
        )
    ):
        gm, dg = _calculate_gibbs_mix_single(mmr, omf, o2c, d, w)
        gibbs_mix[i] = gm
        derivative_gibbs[i] = dg

    return gibbs_mix, derivative_gibbs


def _calculate_gibbs_mix_single(
    molar_mass_ratio: float,
    organic_mole_fraction: float,
    oxygen2carbon: float,
    density: float,
    weights: NDArray[np.float64],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Gibbs free energy of mixing for a single set of inputs.

    Args:
        - molar_mass_ratio : The molar mass ratio of water to organic
          matter.
        - organic_mole_fraction : The fraction of organic matter.
        - oxygen2carbon : The oxygen to carbon ratio.
        - density : The density of the mixture, in kg/m^3.
        - weights : Blending weights for the BAT model.

    Returns:
        - gibbs_mix : Gibbs energy of mixing (including 1/RT)
        - derivative_gibbs : derivative of Gibbs energy with respect to
          mole fraction of organics (includes 1/RT)
    """
    if weights[1] > 0:  # if mid region is used
        gibbs_mix_mid, derivative_gibbs_mid = gibbs_of_mixing(
            molar_mass_ratio=molar_mass_ratio,
            organic_mole_fraction=organic_mole_fraction,
            oxygen2carbon=oxygen2carbon,
            density=density,
            fit_dict=FIT_MID,
        )

        if weights[0] > 0:  # if paired with low oxygen2carbon region
            gibbs_mix_low, derivative_gibbs_low = gibbs_of_mixing(
                molar_mass_ratio=molar_mass_ratio,
                organic_mole_fraction=organic_mole_fraction,
                oxygen2carbon=oxygen2carbon,
                density=density,
                fit_dict=FIT_LOW,
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
                fit_dict=FIT_HIGH,
            )
            gibbs_mix = (
                weights[2] * gibbs_mix_high + weights[1] * gibbs_mix_mid
            )
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
            fit_dict=FIT_HIGH,
        )
    return gibbs_mix, derivative_gibbs
