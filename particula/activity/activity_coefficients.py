"""Activity coefficients for organic-water mixtures.

Gorkowski, K., Preston, T. C., &#38; Zuend, A. (2019).
Relative-humidity-dependent organic aerosol thermodynamics
Via an efficient reduced-complexity model.
Atmospheric Chemistry and Physics
https://doi.org/10.5194/acp-19-13383-2019
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from particula.activity.convert_functional_group import (
    convert_to_oh_equivalent,
)
from particula.activity.gibbs_mixing import gibbs_mix_weight
from particula.util.machine_limit import get_safe_exp
from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "molar_mass_ratio": "positive",
        "organic_mole_fraction": "nonnegative",
        "density": "positive",
    }
)
def bat_activity_coefficients(
    molar_mass_ratio: Union[float, NDArray[np.float64]],
    organic_mole_fraction: Union[float, NDArray[np.float64]],
    oxygen2carbon: Union[float, NDArray[np.float64]],
    density: Union[float, NDArray[np.float64]],
    functional_group: Optional[Union[str, List[str]]] = None,
) -> Tuple[
    Union[float, NDArray[np.float64]],
    Union[float, NDArray[np.float64]],
    Union[float, NDArray[np.float64]],
    Union[float, NDArray[np.float64]],
    Union[float, NDArray[np.float64]],
    Union[float, NDArray[np.float64]],
]:
    """Calculate the activity coefficients for water-organic mixtures.

    Args:
        - molar_mass_ratio : Ratio of the molecular weight of water to the
          molecular weight of organic matter.
        - organic_mole_fraction : Molar fraction of organic matter in the
          mixture.
        - oxygen2carbon : Oxygen to carbon ratio in the organic compound.
        - density : Density of the mixture, in kg/m^3.
        - functional_group : Optional functional group(s) of the organic
          compound, if applicable.

    Returns:
        - A tuple containing the activity of water, activity
          of organic matter, mass fraction of water, and mass
          fraction of organic matter, gamma_water (activity coefficient),
          and gamma_organic (activity coefficient).
    """
    oxygen2carbon, molar_mass_ratio = convert_to_oh_equivalent(
        oxygen2carbon=oxygen2carbon,
        molar_mass_ratio=molar_mass_ratio,
        functional_group=functional_group,
    )
    gibbs_mix, derivative_gibbs = gibbs_mix_weight(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction,
        oxygen2carbon=oxygen2carbon,
        density=density,
    )
    ln_gamma_water = gibbs_mix - organic_mole_fraction * derivative_gibbs
    ln_gamma_org = gibbs_mix + (1.0 - organic_mole_fraction) * derivative_gibbs

    gamma_water = get_safe_exp(ln_gamma_water)
    gamma_organic = get_safe_exp(ln_gamma_org)

    activity_water = gamma_water * (1.0 - organic_mole_fraction)
    activity_organic = gamma_organic * organic_mole_fraction

    mass_water = (
        (1.0 - organic_mole_fraction)
        * molar_mass_ratio
        / ((1.0 - organic_mole_fraction) * (molar_mass_ratio - 1.0) + 1.0)
    )
    mass_organic = 1.0 - mass_water

    return (
        activity_water,
        activity_organic,
        mass_water,
        mass_organic,
        gamma_water,
        gamma_organic,
    )
