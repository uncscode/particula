"""Calculate the activity coefficients at fixed water activity.

This module contains functions to calculates the activity coefficients
at a fixed water activity or the water activity cross point.

Gorkowski, K., Preston, T. C., Zuend, A. (2019).
Relative-humidity-dependent organic aerosol thermodynamics
Via an efficient reduced-complexity model.
Atmospheric Chemistry and Physics
https://doi.org/10.5194/acp-19-13383-2019
"""

from typing import NamedTuple, Optional, Union

import numpy as np
from numpy.typing import NDArray

from particula.activity import phase_separation
from particula.activity.activity_coefficients import bat_activity_coefficients
from particula.activity.bat_coefficients import (
    INTERPOLATE_WATER_FIT,
    LOWEST_ORGANIC_MOLE_FRACTION,
)
from particula.particles.properties.organic_density_module import (
    get_organic_density_estimate,
)


def biphasic_water_activity_point(
    oxygen2carbon: Union[float, NDArray[np.float64]],
    hydrogen2carbon: Union[float, NDArray[np.float64]],
    molar_mass_ratio: Union[float, NDArray[np.float64]],
    functional_group: Optional[Union[list[str], str]] = None,
) -> np.ndarray:
    """Computes the biphasic to single phase water activity (RH*100).

    Args:
        oxygen2carbon: The oxygen to carbon ratio.
        hydrogen2carbon: The hydrogen to carbon ratio.
        molar_mass_ratio: The molar mass ratio of water to organic
            matter.
        functional_group: Optional functional group(s) of the organic
            compound, if applicable.

    Returns:
        The RH cross point array.
    """
    oxygen2carbon = np.asarray(oxygen2carbon, dtype=np.float64)
    hydrogen2carbon = np.asarray(hydrogen2carbon, dtype=np.float64)
    molar_mass_ratio = np.asarray(molar_mass_ratio, dtype=np.float64)
    if oxygen2carbon.ndim == 0:
        oxygen2carbon = np.expand_dims(oxygen2carbon, axis=0)
    if hydrogen2carbon.ndim == 0:
        hydrogen2carbon = np.expand_dims(hydrogen2carbon, axis=0)
    if molar_mass_ratio.ndim == 0:
        molar_mass_ratio = np.expand_dims(molar_mass_ratio, axis=0)

    water_activity_cross_point = np.zeros_like(oxygen2carbon)

    interpolate_step_numb = 200
    mole_frac = np.logspace(-6, 0, interpolate_step_numb + 1)

    for i, _ in enumerate(oxygen2carbon):
        density = get_organic_density_estimate(
            molar_mass_ratio[i],
            oxygen2carbon[i],
            hydrogen2carbon[i],
            mass_ratio_convert=True,
        )
        activities = bat_activity_coefficients(
            molar_mass_ratio=molar_mass_ratio[i],
            organic_mole_fraction=mole_frac,
            oxygen2carbon=oxygen2carbon[i],
            density=density,
            functional_group=functional_group,
        )

        if np.isnan(activities[0]).any():
            raise ValueError("water activity is NaN, check inputs")

        activities_water_seq = np.asarray(activities[0], dtype=np.float64)
        activities_organic_seq = np.asarray(activities[1], dtype=np.float64)

        phase_check = phase_separation.find_phase_separation(
            activities_water_seq, activities_organic_seq
        )

        if phase_check["phase_sep_check"] == 1:
            water_activity_cross_point[i] = phase_check["upper_seperation"]
        else:
            water_activity_cross_point[i] = 0

    water_activity_cross_point[water_activity_cross_point < 0] = 0
    water_activity_cross_point[water_activity_cross_point > 1] = 1

    return water_activity_cross_point


_PhaseProperties = tuple[float | NDArray[np.float64], ...] | NDArray[np.float64]


class FixedWaterActivityResult(NamedTuple):
    """Properties of the water-rich and optional organic-rich liquid branches.

    Each phase contains water and organic activities, water and organic mass
    fractions, then water and organic activity coefficients, in that order.
    The last axis of a sweep follows the requested water activities.
    """

    alpha: _PhaseProperties
    beta: _PhaseProperties | None
    q_alpha: float | NDArray[np.float64]


def fixed_water_activity(
    water_activity: Union[float, NDArray[np.float64]],
    molar_mass_ratio: Union[float, NDArray[np.float64]],
    oxygen2carbon: Union[float, NDArray[np.float64]],
    density: Union[float, NDArray[np.float64]],
) -> FixedWaterActivityResult:
    """Evaluate binary BAT liquid branches at prescribed water activities.

    Args:
        water_activity: Dimensionless target activity, scalar or 1-D sweep.
        molar_mass_ratio: Water molar mass divided by organic molar mass.
        oxygen2carbon: Organic atomic O:C ratio.
        density: Organic liquid density in kg/m³.

    Returns:
        Phase properties and the model's alpha-phase allocation fraction.
        The result supports tuple unpacking as (alpha, beta, q_alpha).
        A homogeneous solution has beta equal to None and q_alpha equal to 1.
    """
    # pylint: disable=too-many-locals

    # check types
    water_activity = np.asarray(water_activity, dtype=np.float64)
    molar_mass_ratio = np.asarray(molar_mass_ratio, dtype=np.float64)
    oxygen2carbon = np.asarray(oxygen2carbon, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)

    # must have activity of water in increasing order
    if water_activity.size > 1 and water_activity[0] > water_activity[-1]:
        water_activity = np.flip(water_activity)
        flip = True
    else:
        flip = False

    organic_mole_fraction_array = np.linspace(
        1,
        LOWEST_ORGANIC_MOLE_FRACTION,
        INTERPOLATE_WATER_FIT,
        dtype=np.float64,
    )

    # activity calculation
    activities = bat_activity_coefficients(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction_array,
        oxygen2carbon=oxygen2carbon,
        density=density,
    )
    # find phase separation
    phase_check = phase_separation.find_phase_separation(
        activities[0], activities[1]
    )
    # ensure water activity type is float
    activities_water = np.asarray(activities[0], dtype=np.float64)
    if phase_check["phase_sep_check"] == 0:
        alpha_organic_mole_fraction = np.interp(
            xp=activities_water,
            fp=organic_mole_fraction_array,
            x=water_activity,
            left=1.0,
            right=LOWEST_ORGANIC_MOLE_FRACTION,
        )
        # activity calculation for alpha phase
        activities_alpha = bat_activity_coefficients(
            molar_mass_ratio=molar_mass_ratio,
            organic_mole_fraction=alpha_organic_mole_fraction,
            oxygen2carbon=oxygen2carbon,
            density=density,
        )
        activities_beta = None
        q_alpha = np.ones(water_activity.size)
        # change back to original order
        if flip:
            activities_alpha = np.flip(activities_alpha, axis=-1)
            q_alpha = np.flip(q_alpha)
        return FixedWaterActivityResult(
            activities_alpha, activities_beta, q_alpha
        )

    # else phase separation occurs
    # split the activities into alpha and beta phases

    upper_index = int(phase_check["upper_seperation_index"])
    matching_upper_index = int(phase_check["matching_upper_seperation_index"])

    # alpha water rich phase
    alpha_water_activity = activities_water[upper_index:]
    alpha_organic_mole_fraction = organic_mole_fraction_array[upper_index:]
    # beta organic rich phase
    beta_water_activity = activities_water[:matching_upper_index]
    beta_organic_mole_fraction = organic_mole_fraction_array[
        :matching_upper_index
    ]

    # find the water activity of the alpha phase
    alpha_organic_mole_fraction_interp = np.interp(
        xp=alpha_water_activity,
        fp=alpha_organic_mole_fraction,
        x=water_activity,
        left=0.0,
        right=LOWEST_ORGANIC_MOLE_FRACTION,
    )
    # find the water activity of the beta phase
    beta_organic_mole_fraction_interp = np.interp(
        xp=beta_water_activity,
        fp=beta_organic_mole_fraction,
        x=water_activity,
        left=1,
        right=0.0,
    )
    # calculate the activity coefficients for the alpha phase
    activities_alpha = bat_activity_coefficients(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=alpha_organic_mole_fraction_interp,
        oxygen2carbon=oxygen2carbon,
        density=density,
    )
    # calculate the activity coefficients for the beta phase
    activities_beta = bat_activity_coefficients(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=beta_organic_mole_fraction_interp,
        oxygen2carbon=oxygen2carbon,
        density=density,
    )
    q_alpha = phase_separation.q_alpha(
        seperation_activity=phase_check["upper_seperation"],
        activities=water_activity,
    )
    # change back to original order
    if flip:
        activities_alpha = np.flip(activities_alpha, axis=-1)
        activities_beta = np.flip(activities_beta, axis=-1)
        q_alpha = np.flip(q_alpha)

    activities_alpha = np.asarray(activities_alpha, dtype=np.float64)
    activities_beta = np.asarray(activities_beta, dtype=np.float64)
    q_alpha = np.asarray(q_alpha, dtype=np.float64)

    return FixedWaterActivityResult(activities_alpha, activities_beta, q_alpha)
