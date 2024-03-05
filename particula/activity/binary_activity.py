"""Binary activity coefficient model for organic-water mixtures.

Gorkowski, K., Preston, T. C., &#38; Zuend, A. (2019).
Relative-humidity-dependent organic aerosol thermodynamics
Via an efficient reduced-complexity model.
Atmospheric Chemistry and Physics
https://doi.org/10.5194/acp-19-13383-2019
"""

from typing import Optional, Union, Tuple
from numpy.typing import ArrayLike
import numpy as np

from particula.util.machine_limit import safe_exp
from particula.activity import phase_separation
from particula.activity.species_density import organic_density_estimate


# the fit values for the activity model
FIT_LOW = {'a1': [7.089476E+00, -7.711860E+00, -3.885941E+01, -1.000000E+02],
           'a2': [-6.226781E-01, -1.000000E+02, 3.081244E-09, 6.188812E+01],
           's': [-5.988895E+00, 6.940689E+00]}
FIT_MID = {'a1': [5.872214E+00, -4.535007E+00, -5.129327E+00, -2.809232E+01],
           'a2': [-9.740486E-01, -1.000000E+02, 2.109751E+00, -2.367683E+01],
           's': [-1.219164E+00, 4.742729E+00]}
FIT_HIGH = {'a1': [5.921550E+00, -2.528295E+00, -3.883017E+00, -7.898128E+00],
            'a2': [-1.000000E+02, -1.000000E+02, 1.353916E+00, -1.160145E+01],
            's': [-7.868187E-02, 3.650860E+00]}
# interpolation points, could be done smarter
INTERPOLATE_WATER_FIT = 500
LOWEST_ORGANIC_MOLE_FRACTION = 1e-12


def activity_coefficients(
    molar_mass_ratio: ArrayLike,
    organic_mole_fraction: ArrayLike,
    oxygen2carbon: ArrayLike,
    density: ArrayLike,
    functional_group=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the activity coefficients for water and organic matter in
    organic-water mixtures.

    Args:
        - molar_mass_ratio: Ratio of the molecular weight of water to the
            molecular weight of organic matter.
        - organic_mole_fraction: Molar fraction of organic matter in the
            mixture.
        - oxygen2carbon: Oxygen to carbon ratio in the organic compound.
        - density: Density of the mixture.
        - functional_group: Optional functional group(s) of the organic
            compound, if applicable.

    Returns:
        A tuple containing the activity of water, activity
        of organic matter, mass fraction of water, and mass
        fraction of organic matter, gamma_water (activity coefficient),
        and gamma_organic (activity coefficient).
    """
    # check types
    organic_mole_fraction = np.asarray(organic_mole_fraction, dtype=np.float64)

    oxygen2carbon, molar_mass_ratio = convert_to_oh_equivalent(
        oxygen2carbon=oxygen2carbon,
        molar_mass_ratio=molar_mass_ratio,
        functional_group=functional_group
    )
    gibbs_mix, derivative_gibbs = gibbs_mix_weight(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction,
        oxygen2carbon=oxygen2carbon,
        density=density,
    )
    # equations S8 S10
    # the func value for component 1 = LOG(activity coeff. water)
    ln_gamma_water = gibbs_mix - organic_mole_fraction * derivative_gibbs

    # the func value of the component 2 = LOG(activity coeff. of the organic)
    ln_gamma_org = gibbs_mix + (1.0 - organic_mole_fraction) * derivative_gibbs

    gamma_water = safe_exp(ln_gamma_water)
    gamma_organic = safe_exp(ln_gamma_org)

    activity_water = gamma_water * (1.0 - organic_mole_fraction)
    activity_organic = gamma_organic * organic_mole_fraction

    mass_water = (1.0 - organic_mole_fraction) * molar_mass_ratio / (
        (1.0 - organic_mole_fraction) * (molar_mass_ratio - 1) + 1
    )
    mass_organic = 1 - mass_water

    return activity_water, activity_organic, mass_water, mass_organic, \
        gamma_water, gamma_organic


def gibbs_of_mixing(
    molar_mass_ratio: ArrayLike,
    organic_mole_fraction: ArrayLike,
    oxygen2carbon: ArrayLike,
    density: ArrayLike,
    fit_dict: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Gibbs free energy of mixing for a binary mixture.

    Args:
        - molar_mass_ratio: The molar mass ratio of water to organic
            matter.
        - organic_mole_fraction: The fraction of organic matter.
        - oxygen2carbon: The oxygen to carbon ratio.
        - density: The density of the mixture.
        - fit_dict: A dictionary of fit values for the low oxygen2carbon region

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the Gibbs free
        energy of mixing and its derivative.
    """
    # check types
    molar_mass_ratio = np.asarray(molar_mass_ratio, dtype=np.float64)
    organic_mole_fraction = np.asarray(organic_mole_fraction, dtype=np.float64)
    oxygen2carbon = np.asarray(oxygen2carbon, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)

    c1 = coefficients_c(molar_mass_ratio, oxygen2carbon, fit_dict['a1'])
    c2 = coefficients_c(molar_mass_ratio, oxygen2carbon, fit_dict['a2'])

    rhor = 0.997 / density  # assumes water is the other fluid

    # equation S3
    # the scaled molar mass ratio of this mixture's components.
    scaled_molar_mass_ratio = molar_mass_ratio * fit_dict['s'][1] \
        * (1.0 + oxygen2carbon) ** fit_dict['s'][0]

    # phi2 is a scaled volume fraction
    phi2 = organic_mole_fraction / (
        organic_mole_fraction + (1.0 - organic_mole_fraction)
        * scaled_molar_mass_ratio / rhor
    )

    # equation S4
    sum1 = c1 + c2 * (1 - 2 * phi2)
    gibbs_mix = phi2 * (1.0 - phi2) * sum1

    # equation s6 the derivative of phi2 with respect to organic x2
    dphi2dx2 = (scaled_molar_mass_ratio / rhor) \
        * (phi2 / organic_mole_fraction)**2

    # equation S7
    derivative_gibbs_mix = (
        (1.0 - 2.0 * phi2) * sum1 - 2 * c2 * phi2 * (1.0 - phi2)
    ) * dphi2dx2

    return gibbs_mix, derivative_gibbs_mix


def gibbs_mix_weight(
    molar_mass_ratio: ArrayLike,
    organic_mole_fraction: ArrayLike,
    oxygen2carbon: ArrayLike,
    density: ArrayLike,
    functional_group: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gibbs free energy of mixing, see Gorkowski (2019), with weighted
    oxygen2carbon regions. Only can run one compound at a time.

    Args:
        - molar_mass_ratio: The molar mass ratio of water to organic
            matter.
        - organic_mole_fraction: The fraction of organic matter.
        - oxygen2carbon: The oxygen to carbon ratio.
        - density: The density of the mixture.
        - functional_group: Optional functional group(s) of the organic
            compound, if applicable.

    Returns:
        - gibbs_mix : Gibbs energy of mixing (including 1/RT)
        - derivative_gibbs : derivative of Gibbs energy with respect to
        - mole fraction of organics (includes 1/RT)
    """
    # check types
    density = np.asarray(density, dtype=np.float64)

    oxygen2carbon, molar_mass_ratio = convert_to_oh_equivalent(
        oxygen2carbon=oxygen2carbon,
        molar_mass_ratio=molar_mass_ratio,
        functional_group=functional_group
    )

    weights = bat_blending_weights(
        molar_mass_ratio=molar_mass_ratio,
        oxygen2carbon=oxygen2carbon
    )

    if weights[1] > 0:  # if mid region is used
        gibbs_mix_mid, derivative_gibbs_mid = gibbs_of_mixing(
            molar_mass_ratio=molar_mass_ratio,
            organic_mole_fraction=organic_mole_fraction,
            oxygen2carbon=oxygen2carbon,
            density=density,
            fit_dict=FIT_MID
        )

        if weights[0] > 0:  # if paired with low oxygen2carbon region
            gibbs_mix_low, derivative_gibbs_low = gibbs_of_mixing(
                molar_mass_ratio=molar_mass_ratio,
                organic_mole_fraction=organic_mole_fraction,
                oxygen2carbon=oxygen2carbon,
                density=density,
                fit_dict=FIT_LOW
            )
            gibbs_mix = weights[0] * gibbs_mix_low + weights[1] * gibbs_mix_mid
            derivative_gibbs = weights[0] * derivative_gibbs_low \
                + weights[1] * derivative_gibbs_mid
        else:  # else paired with high oxygen2carbon region
            gibbs_mix_high, derivative_gibbs_high = gibbs_of_mixing(
                molar_mass_ratio=molar_mass_ratio,
                organic_mole_fraction=organic_mole_fraction,
                oxygen2carbon=oxygen2carbon,
                density=density,
                fit_dict=FIT_HIGH
            )
            gibbs_mix = weights[2] * gibbs_mix_high + \
                weights[1] * gibbs_mix_mid
            derivative_gibbs = weights[2] * derivative_gibbs_high \
                + weights[1] * derivative_gibbs_mid
    else:  # when only high 2OC region is used
        gibbs_mix, derivative_gibbs = gibbs_of_mixing(
            molar_mass_ratio=molar_mass_ratio,
            organic_mole_fraction=organic_mole_fraction,
            oxygen2carbon=oxygen2carbon,
            density=density,
            fit_dict=FIT_HIGH
        )
    return gibbs_mix, derivative_gibbs


def biphasic_water_activity_point(
    oxygen2carbon: ArrayLike,
    hydrogen2carbon: ArrayLike,
    molar_mass_ratio: ArrayLike,
    functional_group: Optional[Union[list[str], str]] = None
) -> np.ndarray:
    """
    This function computes the biphasic to single phase
    water activity (RH*100).

    Args:
        - oxygen2carbon: The oxygen to carbon ratio.
        - hydrogen2carbon: The hydrogen to carbon ratio.
        - molar_mass_ratio: The molar mass ratio of water to organic
            matter.
        - functional_group: Optional functional group(s) of the organic
            compound, if applicable.

    Returns:
        - np.array: The RH cross point array.
    """
    # check types
    oxygen2carbon = np.asarray(oxygen2carbon, dtype=np.float64)
    hydrogen2carbon = np.asarray(hydrogen2carbon, dtype=np.float64)
    molar_mass_ratio = np.asarray(molar_mass_ratio, dtype=np.float64)
    # if 0-d then expand to 1-d
    if oxygen2carbon.ndim == 0:
        oxygen2carbon = np.expand_dims(oxygen2carbon, axis=0)
    if hydrogen2carbon.ndim == 0:
        hydrogen2carbon = np.expand_dims(hydrogen2carbon, axis=0)
    if molar_mass_ratio.ndim == 0:
        molar_mass_ratio = np.expand_dims(molar_mass_ratio, axis=0)

    water_activity_cross_point = np.zeros_like(oxygen2carbon)

    interpolate_step_numb = 200  # interpolation points
    # mole_frac = np.linspace(1e-12, 1, interpolate_step_numb + 1)
    mole_frac = np.logspace(-6, 0, interpolate_step_numb + 1)

    for i, _ in enumerate(oxygen2carbon):
        density = organic_density_estimate(
            molar_mass_ratio[i],
            oxygen2carbon[i],
            hydrogen2carbon[i],
            mass_ratio_convert=True)
        activities = activity_coefficients(
            molar_mass_ratio=molar_mass_ratio[i],
            organic_mole_fraction=mole_frac,
            oxygen2carbon=oxygen2carbon[i],
            density=density,
            functional_group=functional_group
        )

        if np.isnan(activities[0]).any():
            raise ValueError('water activity is NaN, check inputs')

        phase_check = phase_separation.find_phase_separation(
            activities[0],
            activities[1])

        if phase_check['phase_sep_check'] == 1:
            water_activity_cross_point[i] = phase_check['upper_seperation']
        else:
            water_activity_cross_point[i] = 0  # no phase separation

    # Checks outputs with in physical limits
    # round to zero
    water_activity_cross_point[water_activity_cross_point < 0] = 0
    # round max to 1
    water_activity_cross_point[water_activity_cross_point > 1] = 1

    return water_activity_cross_point


def convert_to_oh_equivalent(
    oxygen2carbon: ArrayLike,
    molar_mass_ratio: ArrayLike,
    functional_group: Optional[Union[list[str], str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    just a pass through now, but will
    add the oh equivalent conversion
    """
    # check types
    oxygen2carbon = np.asarray(oxygen2carbon, dtype=np.float64)
    molar_mass_ratio = np.asarray(molar_mass_ratio, dtype=np.float64)

    # sourcery skip
    if functional_group is None:
        return oxygen2carbon, molar_mass_ratio
    if functional_group == 'alcohol':
        return oxygen2carbon + 1, molar_mass_ratio + 16  # fix this from SI
    raise ValueError('BAT functional group not recognized')


def bat_blending_weights(
        molar_mass_ratio: ArrayLike,
        oxygen2carbon: ArrayLike
) -> np.ndarray:
    """
    Function to estimate the blending weights for the BAT model.

    Args:
        - molar_mass_ratio: The molar mass ratio of water to organic
            matter.
        - oxygen2carbon: The oxygen to carbon ratio.

    Returns:
        - blending_weights : List of blending weights for the BAT model
        in the low, mid, and high oxygen2carbon regions.
    """
    # check types
    molar_mass_ratio = np.asarray(molar_mass_ratio, dtype=np.float64)
    oxygen2carbon = np.asarray(oxygen2carbon, dtype=np.float64)

    oxygen2carbon_ml = phase_separation.organic_water_single_phase(
        molar_mass_ratio=molar_mass_ratio)

    blending_weights = np.zeros(3)  # [low, mid, high] oxygen2carbon regions

    # lower to middle oxygen2carbon region
    if oxygen2carbon <= oxygen2carbon_ml * 0.75:
        b_ml = 0.189974476118418
        b_1 = 79.2606902175984
        b_2 = 0.0604293454322489

        oxygen2carbon_b = oxygen2carbon - oxygen2carbon_ml * b_ml
        weight_b = 1 / (1 + np.exp(
            - b_1 * (oxygen2carbon_b - b_2)
        ))  # logistic transfer function

        oxygen2carbon_b_norm = oxygen2carbon - (0.75 * oxygen2carbon_ml * b_ml)

        weight_norm = 1 / (1 + np.exp(
            - b_1 * (oxygen2carbon_b_norm - b_2)
        ))

        blending_weights[1] = weight_b / weight_norm
        blending_weights[0] = 1 - blending_weights[1]

    # middle to high oxygen2carbon region
    elif oxygen2carbon <= oxygen2carbon_ml * 2:
        b_1 = 75.0159268221068
        b_2 = 0.000947111285750515

        oxygen2carbon_b = oxygen2carbon - oxygen2carbon_ml
        blending_weights[2] = 1 / (1 + np.exp(
            - b_1 * (oxygen2carbon_b - b_2)
        ))  # logistic transfer function

        blending_weights[1] = 1 - blending_weights[2]

    else:  # high only region
        blending_weights[2] = 1

    return blending_weights


def coefficients_c(
    molar_mass_ratio: ArrayLike,
    oxygen2carbon: ArrayLike,
    fit_values: ArrayLike
) -> np.ndarray:
    """
    Coefficients for activity model, see Gorkowski (2019). equation S1 S2.

    Args:
        - molar_mass_ratio: The molar mass ratio of water to organic
            matter.
        - oxygen2carbon: The oxygen to carbon ratio.
        - fit_values: The fit values for the activity model.

    Returns:
        - np.ndarray: The coefficients for the activity model.
    """
    # check types
    molar_mass_ratio = np.asarray(molar_mass_ratio, dtype=np.float64)
    oxygen2carbon = np.asarray(oxygen2carbon, dtype=np.float64)
    fit_values = np.asarray(fit_values, dtype=np.float64)

    return fit_values[0] * np.exp(fit_values[1] * oxygen2carbon) + fit_values[
        2
    ] * np.exp(fit_values[3] * molar_mass_ratio)


def fixed_water_activity(
        water_activity: ArrayLike,
        molar_mass_ratio: ArrayLike,
        oxygen2carbon: ArrayLike,
        density: ArrayLike,
) -> Tuple:
    # pylint: disable=too-many-locals
    """
    Calculate the activity coefficients of water and organic matter in
    organic-water mixtures.

    This function assumes a fixed water activity value (e.g., RH = 75%
    corresponds to 0.75 water activity in equilibrium).
    It calculates the activity coefficients for different phases and
    determines phase separations if they occur.

    Parameters:
    water_activity (ArrayLike): An array of water activity values.
    molar_mass_ratio (ArrayLike): Array of molar mass ratios of the components.
    oxygen2carbon (ArrayLike): Array of oxygen-to-carbon ratios.
    density (ArrayLike): Array of densities of the mixture.

    Returns:
    Tuple: A tuple containing the activity coefficients for alpha and beta
            phases, and the alpha phase mole fraction.
           If no phase separation occurs, the beta phase values are None.
    """

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
        dtype=np.float64
    )

    # activity calculation
    activities = activity_coefficients(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction_array,
        oxygen2carbon=oxygen2carbon,
        density=density
    )
    # find phase separation
    phase_check = phase_separation.find_phase_separation(
        activities[0],
        activities[1]
    )
    # ensure water activity type is float
    activities_water = np.asarray(activities[0], dtype=np.float64)
    if phase_check['phase_sep_check'] == 0:
        alpha_organic_mole_fraction = np.interp(
            xp=activities_water,
            fp=organic_mole_fraction_array,
            x=water_activity,
            left=1.0,
            right=LOWEST_ORGANIC_MOLE_FRACTION,
        )
        # activity calculation for alpha phase
        activities_alpha = activity_coefficients(
            molar_mass_ratio=molar_mass_ratio,
            organic_mole_fraction=alpha_organic_mole_fraction,
            oxygen2carbon=oxygen2carbon,
            density=density
        )
        activities_beta = None
        q_alpha = np.ones(water_activity.size)
        # change back to original order
        if flip:
            activities_alpha = np.flip(activities_alpha)
            q_alpha = np.flip(q_alpha)
        return (activities_alpha, activities_beta, q_alpha)

    # else phase separation occurs
    # split the activities into alpha and beta phases

    # alpha water rich phase
    alpha_water_activity = activities_water[
        phase_check['upper_seperation_index']:]
    alpha_organic_mole_fraction = organic_mole_fraction_array[
        phase_check['upper_seperation_index']:]
    # beta organic rich phase
    beta_water_activity = activities_water[
        : phase_check['matching_upper_seperation_index']
    ]
    beta_organic_mole_fraction = organic_mole_fraction_array[
        : phase_check['matching_upper_seperation_index']
    ]

    # find the water activity of the alpha phase
    alpha_organic_mole_fraction_interp = np.interp(
        xp=alpha_water_activity,
        fp=alpha_organic_mole_fraction,
        x=water_activity,
        left=np.nan,
        right=LOWEST_ORGANIC_MOLE_FRACTION,
    )
    # find the water activity of the beta phase
    beta_organic_mole_fraction_interp = np.interp(
        xp=beta_water_activity,
        fp=beta_organic_mole_fraction,
        x=water_activity,
        left=1,
        right=np.nan,
    )
    # calculate the activity coefficients for the alpha phase
    activities_alpha = activity_coefficients(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=alpha_organic_mole_fraction_interp,
        oxygen2carbon=oxygen2carbon,
        density=density
    )
    # calculate the activity coefficients for the beta phase
    activities_beta = activity_coefficients(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=beta_organic_mole_fraction_interp,
        oxygen2carbon=oxygen2carbon,
        density=density
    )
    q_alpha = phase_separation.q_alpha(
        seperation_activity=phase_check['upper_seperation'],
        activities=water_activity,
    )
    # change back to original order
    if flip:
        activities_alpha = np.flip(activities_alpha)
        activities_beta = np.flip(activities_beta)
        q_alpha = np.flip(q_alpha)

    return (activities_alpha, activities_beta, q_alpha)
