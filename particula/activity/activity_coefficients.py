"""Compute activity coefficients for binary organic-water mixtures using BAT.

The Binary Activity Thermodynamics (BAT) model leverages AIOMFAC-derived
fits to describe thermodynamic activities of water and organics across realistic
O:C and concentration ranges. The helper optionally converts functional group
information to OH-equivalent form before blending Gibbs mixing weights and
translating them into activity coefficients.

References:
    Gorkowski, K., Preston, T. C., & Zuend, A. (2019).
    Relative-humidity-dependent organic aerosol thermodynamics via an efficient
    reduced-complexity model. Atmospheric Chemistry and Physics.
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
    r"""Calculate the activity coefficients for water-organic mixtures.

    Uses the BAT (Binary Activity Thermodynamics) model to compute activity
    coefficients for binary organic-water mixtures based on AIOMFAC-derived
    fits. Optionally converts functional groups to OH-equivalent form to align
    with the BAT parameterization.

    Args:
        molar_mass_ratio: Ratio of the molecular weight of water to the
            molecular weight of organic matter (dimensionless).
        organic_mole_fraction: Molar fraction of organic matter in the mixture.
            Range: [0, 1].
        oxygen2carbon: Oxygen-to-carbon ratio in the organic compound.
        density: Density of the mixture, in kg/m^3.
        functional_group: Optional functional group(s) of the organic compound
            (e.g., "alcohol", "carboxylic_acid", "ether").

    Returns:
        Tuple of six values:
            activity_water: Thermodynamic activity of water (:math:`a_w`).
                Range: [0, 1] for stable systems.
            activity_organic: Thermodynamic activity of the organic component
                (:math:`a_{org}`). Range: [0, 1] for stable systems.
            mass_water: Mass fraction of water in the mixture. Range: [0, 1].
            mass_organic: Mass fraction of organic; :math:`1 - mass_water`.
            gamma_water: Activity coefficient of water (:math:`\gamma_w`).
            gamma_organic: Activity coefficient of the organic component
                (:math:`\gamma_{org}`).

    Examples:
        >>> from particula.activity import bat_activity_coefficients
        >>> a_w, a_org, m_w, m_org, g_w, g_org = bat_activity_coefficients(
        ...     molar_mass_ratio=0.09,
        ...     organic_mole_fraction=0.3,
        ...     oxygen2carbon=0.4,
        ...     density=1400.0,
        ... )

    References:
        Gorkowski et al. (2019), Equations 1-6 and SI S1-S2.
        https://doi.org/10.5194/acp-19-13383-2019
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
