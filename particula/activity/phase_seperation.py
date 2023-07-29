# linting disabled until reformatting of this file
# pylint: disable=all
# flake8: noqa
# pytype: skip-file

# %% 


import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'text.color': "#333333",
                     'axes.labelcolor': "#333333",
                     "figure.figsize": (6,4),
                     "font.size": 14,
                     "axes.edgecolor": "#333333",
                     "axes.labelcolor": "#333333",
                     "xtick.color": "#333333",
                     "ytick.color": "#333333",
                     "pdf.fonttype": 42,
                     "ps.fonttype": 42})

# Gorkowski, K., Preston, T. C., &#38; Zuend, A. (2019).
# Relative-humidity-dependent organic aerosol thermodynamics
# via an efficient reduced-complexity model.
# Atmospheric Chemistry and Physics
# https://doi.org/10.5194/acp-19-13383-2019

# %%


def organic_water_single_phase(molar_mass_ratio):
    """
    Convert the given molar mass ratio (MW water / MW organic) to a
    and O2C value were above is a single phase with water and below
    phase separation is possible.

    Parameters:
    molar_mass_ratio np.: The molar mass ratio with respect to water.

    Returns:
    float: The single phase cross point.
    """

    o2c_single_phase_cross_point = 0.205 / (
        1 + np.exp(26.6 * (molar_mass_ratio - 0.12))
        )**0.843 + 0.225
    return o2c_single_phase_cross_point

# %% Plot Example curve


organic_molecular_weight = np.linspace(50, 500, 500)

o2c_value = organic_water_single_phase(18.01528 / organic_molecular_weight)

fig, ax = plt.subplots()
ax.plot(
    organic_molecular_weight,
    o2c_value,
    label="initial",
    linestyle='dashed'
    )
ax.set_xlabel("Organic molecular weight (g/mol)")
ax.set_ylabel("O:C ratio")
ax.set_title("Organic Phase separation")
ax.legend()
fig.show

# %% activity calc

# function [ln_func1, ln_func2, ycalc1, ycalc2, activity_calc1, activity_calc2, mass_fraction1, mass_fraction2, Gibbs_RT, dGibbs_RTdx2]=BAT_properties_calculation_v1(...
#     org_mole_fraction, O2C, H2C, molarmass_ratio, BAT_functional_group, special_options, N2C_values_denistyOnly)
import numpy as np

def exp_wlimiter(x):
    # Define the limiting function for the exponent here
    return np.exp(x)

def convert_to_OH_eqivelent(O2C, molarmass_ratio, BAT_functional_group):
    # Define this function
    return O2C, molarmass_ratio

def organic_density_estimate(M, O2C, H2C=None, N2C=None):
    """
    Function to estimate the density of organic compounds based on the simple
    model by Girolami (1994). The input parameters include molar mass, O:C
    and H:C ratios. If the H:C ratio is unknown at input, enter a negative
    value. The actual H:C will then be estimated based on an initial assumption
    of H:C = 2. The model also estimates the number of carbon atoms per
    molecular structure based on molar mass, O:C, and H:C.
    The density is then approximated by the formula of Girolami.

    Reference:
    Girolami, G. S.: A Simple 'Back of the Envelope' Method for Estimating
    the Densities and Molecular Volumes of Liquids and Solids,
    J. Chem. Educ., 71(11), 962, doi:10.1021/ed071p962, 1994.

    Parameters:
        M (float): Molar mass.
        O2C (float): O:C ratio.
        H2C (float): H:C ratio. If unknown, provide a negative value.
        N2C (float, optional): N:C ratio. Defaults to None.

    Returns:
        densityEst (float): Estimated density in g/cm^3.
    """
    if N2C is None:
        N2C = M * 0
    if H2C is None:
        H2C = M * 0

    mass_C = 12.01  # the molar masses in [g/mol]
    mass_O = 16.0
    mass_H = 1.008
    mass_N = 14.0067

    # 1) Estimate the H2C value if not provided from input
    if H2C < 0.1:
        # Estimate H2C assuming an aliphatic compound with H2C = 2 in the absence of oxygen functional groups,
        # then correct for oxygen content assuming a -1 slope (Van Krevelen Diagram of typical SOA).
        H2Cest = 2.0 - O2C
    else:
        H2Cest = H2C

    # 2) Compute the approximate number of carbon atoms per organic molecule
    NC = M / (mass_C + H2Cest * mass_H + O2C * mass_O + N2C * mass_N)

    # 3) Compute density estimate based on method by Girolami (1994)
    # Here no correction is applied for rings and aromatic compounds
    # (due to limited info at input)
    rho1 = M / (5.0 * NC * (2.0 + H2Cest + O2C * 2.0 + N2C * 2.0))
    density = rho1 * (1.0 + min(NC * O2C * 0.1 + NC * N2C * 0.1, 0.3))
    # density in [g/cm^3];
    # Here it is scaled assuming that most of the oxygen atoms are able to
    # make H-bonds (donor or acceptor).

    return density


def single_phase_O2C_point_KGv3(Mr):
    # Define this function
    return Onephase_O2C

def replace_data_A_to_B_KGv1(x2, A, B):
    return np.where(x2 == A, B, x2)

# %%

org_mole_fraction = np.linspace(0, 1, 100)
molarmass_ratio = 18.016/250
O2C = 0.2
H2C = 0
N2C = 0


O2C, molarmass_ratio = convert_to_OH_eqivelent(
        O2C,
        molarmass_ratio,
        BAT_functional_group=None
    )

# check the limits of possible mole fractions
org_mole_fraction = np.where(org_mole_fraction>1, 1, org_mole_fraction)
org_mole_fraction = np.where(org_mole_fraction<=0, 10**-20, org_mole_fraction)

density = organic_density_estimate(18.016/molarmass_ratio, O2C, H2C, N2C)


#%%

def bat_blending_weights(molarmass_ratio, O2C):
    """
    Function to estimate the blending weights for the BAT model.

    Parameters:
    -----------
    molarmass_ratio (float): Molar mass ratio of the organic compound.

    Returns:
    --------
    blending_weights (array): List of blending weights for the BAT model
        in the low, mid, and high O2C regions.
    """

    O2C_ml = organic_water_single_phase(molarmass_ratio)

    blending_weights = np.zeros(3)  # [low, mid, high] O2C regions

    if O2C <= O2C_ml * 0.75:  # lower to mid O2C region
        b_ml = 0.189974476118418
        b_1 = 79.2606902175984
        b_2 = 0.0604293454322489

        O2C_b = O2C - O2C_ml * b_ml
        weight_b = 1 / (1 + np.exp(
            - b_1 * (O2C_b - b_2)
        ))  # logistic transfer function

        O2C_b_norm = O2C - (0.75 * O2C_ml * b_ml)

        weight_norm = 1 / (1 + np.exp(
            - b_1 * (O2C_b_norm - b_2)
        ))

        blending_weights[1] = weight_b / weight_norm
        blending_weights[0] = 1 - blending_weights[1]

    elif O2C <= O2C_ml * 2:  # mid to high O2C region
        b_1 = 75.0159268221068
        b_2 = 0.000947111285750515

        O2C_b = O2C - O2C_ml
        blending_weights[2] = 1 / (1 + np.exp(
            - b_1 * (O2C_b - b_2)
        ))  # logistic transfer function

        blending_weights[1] = 1 - blending_weights[2]

    else:  # high only region
        blending_weights[2] = 1

    return blending_weights

#%%

O2C_array = np.linspace(0, 0.6, 100)
weights_matrix = np.zeros((100, 3))

for index, O2C in enumerate(O2C_array):
    weights_matrix[index, :] = bat_blending_weights(molarmass_ratio, O2C)

fig, ax = plt.subplots()
ax.plot(
    O2C_array,
    weights_matrix,
    )

ax.set_xlabel("O:C")
ax.set_ylabel("weights")
# ax.legend()
fig.show


#%%
FIT_lowO2C = {'a1':[7.089476E+00, -7.711860E+00, -3.885941E+01, -1.000000E+02],
              'a2':[-6.226781E-01, -1.000000E+02, 3.081244E-09, 6.188812E+01],
              's':[-5.988895E+00, 6.940689E+00]}
FIT_midO2C = {'a1':[5.872214E+00, -4.535007E+00, -5.129327E+00, -2.809232E+01],
              'a2':[-9.740486E-01, -1.000000E+02, 2.109751E+00, -2.367683E+01,],
              's':[-1.219164E+00, 4.742729E+00]}
FIT_highO2C = {'a1':[5.921550E+00, -2.528295E+00, -3.883017E+00, -7.898128E+00,],
               'a2':[-1.000000E+02, -1.000000E+02, 1.353916E+00, -1.160145E+01,],
               's':[-7.868187E-02, 3.650860E+00]}

def coefficents_c(
        molarmass_ratio,
        fit_values
    ):

    """
    Coefficents for activity model, see Gorkowski (2019). equation S1 S2.

    Paramters:
    ---------
        molar mass ratio (float): water MW / orgniac MW
        fit_values (list): a_n1, a_n2, a_n3, a_n4
    """
    c = (fit_values[0] * np.exp(fit_values[1] * O2C)
        + fit_values[2] * np.exp(fit_values[3] * molarmass_ratio))
    return c


# %%

def exp_limited(value):
    """
    np.exp with limits for machine precision max input value of 690.

    Parameters:
    value (array): Input array.

    Returns:
    array: Exponential of the input array with a limit for machine precision.
    """
    return np.exp(np.where(value > 690, 690, value))

def log_limited(value):
    """
    np.log with limits for machine precision min input value of 1e-300.

    Parameters:
    value (array): Input array.

    Returns:
    array: Log of the input array with a limit for machine precision.
    """
    return np.log(np.where(value < 1e-300, 1e-300, value))

#%%
def gibbs_of_mixing(
        molarmass_ratio,
        org_mole_fraction,
        O2C,
        density,
        fit_dict
    ):
    """
    Gibbs free energy of mixing, see Gorkowski (2019). equation S4.

    Paramters:
    ---------
        molar mass ratio (float): water MW / orgniac MW
        org mole fraction (float): fraction of organic matter
        O2C (float): oxygen to carbon ratio
        density (float): density of mixture
        fit_coefficent (dict): dictionary of fit values for low O2C region
    """
    c1 = coefficents_c(molarmass_ratio, fit_dict['a1'])
    c2 = coefficents_c(molarmass_ratio, fit_dict['a2'])

    rhor = 0.997 / density  # assumes water is the other fluid

    # equation S3
    scaledMr = molarmass_ratio * fit_dict['s'][1] \
        * (1.0 + O2C) ** fit_dict['s'][0]  
    #the scaled molar mass ratio of this mixture's components.
    phi2 = org_mole_fraction / (
        org_mole_fraction + (1.0 - org_mole_fraction) * scaledMr / rhor
        )  # phi2 is a scaled volume fraction

    # equation S4
    sum1 = c1 + c2*(1-2*phi2)
    gibbs_mix = phi2 * (1.0 - phi2) * sum1

    # equation s6 the derivative of phi2 with respect to orgnaic x2
    dphi2dx2 = (scaledMr / rhor) * (phi2 / org_mole_fraction) ** 2

    # equation S7
    dervative_gibbs_mix = (
        (1.0 - 2.0 * phi2) * sum1 - 2*c2*phi2 * (1.0 - phi2)
        ) * dphi2dx2

    return gibbs_mix, dervative_gibbs_mix

gibbs_mix, dervative_gibbs = gibbs_of_mixing(
    molarmass_ratio,
    org_mole_fraction,
    O2C,
    density,
    FIT_lowO2C
)


# %%

def gibbs_mix_weight(
        molarmass_ratio,
        org_mole_fraction,
        O2C,
        density,
        BAT_functional_group=None,
    ):
    """
    Gibbs free energy of mixing, see Gorkowski (2019), with weighted
    O2C regions

    Paramters:
    ---------
        molar mass ratio (float): water MW / orgniac MW
        org mole fraction (float): fraction of organic matter
        O2C (float): oxygen to carbon ratio
        density (float): density of mixture
        fit_coefficent (dict): dictionary of fit values for low O2C region

    Returns:
    -------
        gibbs_mix (float): Gibbs energy of mixing (including 1/RT)
        dervative_gibbs (float): dervative of Gibbs energy with respect to 
        mole fraction of orgnaics (includes 1/RT)
    """
    O2C, molarmass_ratio = convert_to_OH_eqivelent(
        O2C,
        molarmass_ratio,
        BAT_functional_group=None
    )

    weights = bat_blending_weights(molarmass_ratio, O2C)

    if weights[1]>0:  # if mid region is used
        gibbs_mix_mid, dervative_gibbs_mid = gibbs_of_mixing(
            molarmass_ratio,
            org_mole_fraction,
            O2C,
            density,
            FIT_midO2C
        )

        if weights[0]>0: # if paired with low O2C region
            gibbs_mix_low, dervative_gibbs_low = gibbs_of_mixing(
                molarmass_ratio,
                org_mole_fraction,
                O2C,
                density,
                FIT_lowO2C
            )
            gibbs_mix = weights[0]*gibbs_mix_low + weights[1]*gibbs_mix_mid
            dervative_gibbs = weights[0]*dervative_gibbs_low \
            + weights[1]*dervative_gibbs_mid
        
        else:  #else paired with high O2C region
            gibbs_mix_high, dervative_gibbs_high = gibbs_of_mixing(
                molarmass_ratio,
                org_mole_fraction,
                O2C,
                density,
                FIT_highO2C
            )
            gibbs_mix = weights[2]*gibbs_mix_high + weights[1]*gibbs_mix_mid
            dervative_gibbs = weights[2]*dervative_gibbs_high \
                + weights[1]*dervative_gibbs_mid
    else:  # when only high 2OC region is used
        gibbs_mix, dervative_gibbs = gibbs_of_mixing(
            molarmass_ratio,
            org_mole_fraction,
            O2C,
            density,
            FIT_highO2C
        )
    return gibbs_mix, dervative_gibbs


# %%
O2C=0.225
molarmass_ratio = 18.016/100
density = organic_density_estimate(18.016/molarmass_ratio, O2C)

gibbs_mix, dervative_gibbs= gibbs_mix_weight(
        molarmass_ratio,
        org_mole_fraction,
        O2C,
        density,
    )

# equations S8 S10
# the func value for component 1 = LOG(activity coeff. water)
ln_gamma_water = gibbs_mix - org_mole_fraction * dervative_gibbs
# the func value of the component 2 = LOG(activity coeff. of the organic)
ln_gamma_org = gibbs_mix + (1.0 - org_mole_fraction) * dervative_gibbs

gamma_water = exp_limited(ln_gamma_water)
gamma_org = exp_limited(ln_gamma_org)

activity_water = gamma_water * (1.0 - org_mole_fraction)
activity_organic = gamma_org * org_mole_fraction

mass_water = (1.0 - org_mole_fraction) * molarmass_ratio / (
        (1.0 - org_mole_fraction) * (molarmass_ratio - 1) + 1
    )
mass_organic = 1 - mass_water

gibbs_ideal = (1-org_mole_fraction) * log_limited(1-org_mole_fraction) \
    +org_mole_fraction * log_limited(org_mole_fraction)
gibbs_real = gibbs_ideal + gibbs_mix

# %%
fig, ax = plt.subplots()
# ax.plot(
#     1-org_mole_fraction,
#     gibbs_real,
#     label="gibbs",
#     linestyle='dashed'
#     )
ax.plot(
    1-org_mole_fraction,
    activity_water,
    label="water",
    linestyle='dashed'
    )

ax.plot(
    1-org_mole_fraction,
    activity_organic,
    label="organic",
    )
ax.set_ylim((0,1))
ax.set_xlabel("water mole fraction")
ax.set_ylabel("activity")
ax.legend()
fig.show







# %%

def find_phase_sep_index(activity_data):
    """
    This function finds phase separation using activity>1 and
    inflections in the activity curve data.
    In physical systems activity can not be above one and 
    curve should be monotonic. Or else there will be phase separation.

    Parameters:
    activity_data (np.array): A numpy array of activity data.

    Returns:
    tuple: The phase separation via activity,
    phase separation via activity curvature,
    index phase separation starts, index phase separation end.
    """

    # Compute difference between consecutive elements in the array
    activity_diff = np.diff(activity_data)
    data_length = len(activity_data)

    # Check if the data length is more than 3
    if data_length > 3:
        min_value = np.min(activity_diff)
        max_value = np.max(activity_diff)

        # Check if the min and max differences have the same sign
        if np.sign(min_value) == np.sign(max_value):
            # If so, no phase separation via activity curvature
            phase_sep_curve = 0
            index_phase_sep_starts = np.nan
            index_phase_sep_end = np.nan
        else:
            # If signs differ, phase separation via activity curvature occurs
            phase_sep_curve = 1

            # Find where the sign changes in the activity difference
            activity_diff_sign_change = np.sign(
                np.concatenate(([activity_diff[0]], activity_diff))
                ) != np.sign(activity_diff[0])

            # Find the first change in sign
            index_start = np.where(activity_diff_sign_change)[0][0]
            # Find the last change in sign
            back_index = index_start - 1 + np.where(
                ~activity_diff_sign_change[index_start:])[0][0]

            # Find closest match to restart the process
            if back_index < data_length:
                activity_data_gap = np.argmin(
                    np.abs(
                    activity_data[back_index:] - activity_data[index_start]
                    ))
                restart_match_index = activity_data_gap + back_index - 1
            else:
                restart_match_index = data_length

            # Check if any activity data is greater than 1
            if sum(activity_data > 1):
                # Find minimum activity data and its corresponding index
                min_value_idilute = np.min(activity_data[index_start:])
                min_index_idilute = np.argmin(
                    activity_data[index_start:]) + index_start - 1

                # Find where activity data matches the minimum value
                activity_data_gap_start = np.argmin(
                    np.abs(
                    activity_data[:index_start] \
                        - activity_data[min_index_idilute]
                    ))

                # Assign appropriate indices for phase separation
                if activity_data_gap_start < index_start:
                    index_phase_sep_starts = activity_data_gap_start
                else:
                    index_phase_sep_starts = index_start

                if min_index_idilute < restart_match_index:
                    index_phase_sep_end = min_index_idilute
                else:
                    index_phase_sep_end = restart_match_index

            else:
                index_phase_sep_starts = index_start
                index_phase_sep_end = restart_match_index

    else:
        phase_sep_activity = activity_data
        phase_sep_curve = 0
        index_phase_sep_starts = np.nan
        index_phase_sep_end = np.nan

    # Assign phase separation via activity based on data being greater than 1
    if sum(activity_data > 1):
        phase_sep_activity = 1
    else:
        phase_sep_activity = 0

    return phase_sep_activity, phase_sep_curve, \
        index_phase_sep_starts, index_phase_sep_end


# out = finds_phase_sep_and_activity_curve_dips(activity_organic)

#%%
def find_phase_separation(activity_water, activity_org):
    """
    This function checks for phase separation in each activity curve.

    Parameters:
    activity_water (np.array): A numpy array of water activity values.
    activity_org (np.array): A numpy array of organic activity values.

    Returns:
    tuple: The phase separation check, lower a_w separation index,
    upper a_w separation index, matching upper a_w separation index.
    """

    # check for phase separation in each activity curve
    _, phase_sep_curve_w, index_phase_sep_starts_w, index_phase_sep_end_w = find_phase_sep_index(activity_water)
    _, phase_sep_curve_org, index_phase_sep_starts_org, index_phase_sep_end_org = find_phase_sep_index(activity_org)

    # gather all the indexes into a list for easier access
    indexes = [index_phase_sep_starts_w, index_phase_sep_end_w, index_phase_sep_starts_org, index_phase_sep_end_org]

    # If there is a phase separation curve in the water activity data
    if phase_sep_curve_w == 1:
        phase_sep_check = 1

        # Check for the direction of the curve (increasing or decreasing)
        if activity_water[0] < activity_water[-1]:  # increasing a_w with index
            # find the min and max indexes
            lower_a_w_sep_index = min(indexes)
            upper_a_w_sep_index = max(indexes)

            # calculate the mid index
            mid_sep_index = (lower_a_w_sep_index + upper_a_w_sep_index) // 2
            # slice the data upto mid index
            activity_water_beta = activity_water[:mid_sep_index]
            match_a_w = activity_water[upper_a_w_sep_index]
            # find the index where the difference is greater than 0
            match_index_prime = np.where((activity_water_beta - match_a_w) > 0)

            # if no such index found, assign the index where the max difference is located
            if len(match_index_prime[0]) == 0:
                match_index_prime = np.argmax(activity_water_beta - match_a_w)

            matching_upper_a_w_sep_index = match_index_prime[0][0] - 1

        else:  # decreasing a_w with index
            lower_a_w_sep_index = max(indexes)
            upper_a_w_sep_index = min(indexes)

            mid_sep_index = (lower_a_w_sep_index + upper_a_w_sep_index) // 2
            activity_water_beta = activity_water[mid_sep_index:]
            match_a_w = activity_water[upper_a_w_sep_index]
            match_index_prime = np.where(activity_water_beta <= match_a_w)

            matching_upper_a_w_sep_index = mid_sep_index + match_index_prime[0][0] - 1

    else:  # no phase separation
        lower_a_w_sep_index = 1
        upper_a_w_sep_index = 2
        matching_upper_a_w_sep_index = 2
        phase_sep_check = 0  # no phase sep

    return phase_sep_check, lower_a_w_sep_index, upper_a_w_sep_index, matching_upper_a_w_sep_index

phase_sep_check, lower_a_w_sep_index, upper_a_w_sep_index, matching_upper_a_w_sep_index = find_phase_separation(activity_water, activity_organic)
#%%


def phase_seperation_q_alpha(
        a_w_sep,
        aw_series,
        VBSBAT_options=None
    ):
    """
    This function makes a squeezed logistic function to transfer for 
    q_alpha ~0 to q_alpha ~1, 

    Parameters:
    a_w_sep (np.array): A numpy array of values.
    aw_series (np.array): A numpy array of values.
    VBSBAT_options (dict): A dictionary of options.

    Returns:
    np.array: The q_alpha value.
    """
    min_spread_in_aw = 10**-6
    q_alpha_at_1phase_aw = 0.99
    q_alpha_bounds=[10**6, 1]
    q_alpha_bounds_mean=[10**6, 1]

    mask_of_miscible_points = a_w_sep == 0  # values held for correction at the end

    # spread in transfer from 50/50 point
    delta_a_w_sep = 1 - a_w_sep

    # check min value allowed
    above_min_delta_a_w_sep_value = delta_a_w_sep > min_spread_in_aw
    delta_a_w_sep = delta_a_w_sep * above_min_delta_a_w_sep_value + \
        ~above_min_delta_a_w_sep_value * min_spread_in_aw

    # calculate curve parameter of sigmoid
    sigmoid_curve_parameter = log_limited(1 / (1 -q_alpha_at_1phase_aw) - 1) / delta_a_w_sep

    # calculate q_alpha value
    q_alpha_value = 1 - 1. / (1 + exp_limited(sigmoid_curve_parameter * (aw_series - a_w_sep + delta_a_w_sep)))

    # apply mask for complete miscibility, turns miscible organics to q_alpha=1 for all a_w
    q_alpha_value = q_alpha_value * ~mask_of_miscible_points + mask_of_miscible_points

    return q_alpha_value

q_alpha = phase_seperation_q_alpha()

#%%

def check_bat_functional_group_inputs_v1(O2C, shift_method):
    """
    This function checks the inputs of the BAT functional group.

    Parameters:
    O2C (np.array): A numpy array representing O2C values.
    shift_method (str/list): A string or list representing the shift method.

    Returns:
    list: The shift method with size equal to O2C.
    """

    max_dim = max(len(O2C), len(shift_method))
    
    if isinstance(shift_method, str):
        shift_method = [shift_method for _ in range(max_dim)]
    elif isinstance(shift_method, list):
        if len(shift_method) == 1:
            shift_method = [shift_method[0] for _ in range(max_dim)]
        elif len(shift_method) < max_dim:
            raise ValueError(f"shift_method has less points than O2C: {len(shift_method)} vs {max_dim}")
            
    return shift_method


def biphasic_to_single_phase_RH_master_v4(O2C, H2C, Mratio, BAT_functional_group):
    """
    This function computes the biphasic to single phase RH.

    Parameters:
    O2C (np.array): An array representing O2C values.
    H2C (np.array): An array representing H2C values.
    Mratio (np.array): An array representing molar mass ratio values.
    BAT_functional_group (str/list): The BAT functional group(s).

    Returns:
    np.array: The RH cross point array.
    """

    RH_cross_point = np.zeros_like(O2C)

    interpolate_step_numb = 500  # interpolation points
    mole_frac = np.linspace(0, 1, interpolate_step_numb + 1)

    for i in range(len(O2C)):  # loops through one compound at a time
        func1, func2, ycal_water, ycalc_org, activity_water, activity_org, mass_fraction1, mass_fraction2, Gibbs_RT, dGibbs_RTdx2 \
            = bat_properties_calculation_v1(mole_frac, O2C[i], H2C[i], Mratio[i], BAT_functional_group[i], [])

        if np.isnan(activity_water):
            raise ValueError('water activity is NaN, check inputs')

        phase_sep_check, _, upper_a_w_sep_index, _ = finds_phase_sep_w_and_org(activity_water, activity_org)  # finds a_w sep. point

        if phase_sep_check == 1:  # checks if there is phase separation.
            RH_cross_point[i] = activity_water[upper_a_w_sep_index]  # save phase sep RH
        else:
            RH_cross_point[i] = 0  # no phase separation

    # Checks outputs with in physical limits 
    #round to zero
    RH_cross_point[RH_cross_point < 0] = 0
    # round max to 1
    RH_cross_point[RH_cross_point > 1] = 1

    return RH_cross_point


# %%
