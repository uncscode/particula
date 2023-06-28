


import numpy as np

def single_phase_o2c_point_kgv3(molar_mass_ratio):
    """
    Convert the given molar mass ratio to a single phase cross point value using a sigmoid equivalent function.

    Parameters:
    molar_mass_ratio (float): The molar mass ratio.

    Returns:
    float: The single phase cross point value.
    """
    o2c_single_phase_cross_point = 0.205 / (1 + np.exp(26.6 * (molar_mass_ratio - 0.12))) ** 0.843 + 0.225
    return o2c_single_phase_cross_point


def q_alpha_transfer_vs_aw_calc_v1(a_w_sep, aw_series, VBSBAT_options):
    """
    This function makes a squeezed logistic function to transfer for q_alpha ~0 to q_alpha ~1, 
    described in VBSBAT_options.q_alpha.

    Parameters:
    a_w_sep (np.array): A numpy array of values.
    aw_series (np.array): A numpy array of values.
    VBSBAT_options (dict): A dictionary of options.

    Returns:
    np.array: The q_alpha value.
    """
    
    mask_of_miscible_points = a_w_sep == 0  # values held for correction at the end

    # spread in transfer from 50/50 point
    delta_a_w_sep = 1 - a_w_sep

    # check min value allowed
    above_min_delta_a_w_sep_value = delta_a_w_sep > VBSBAT_options['q_alpha']['min_spread_in_aw']
    delta_a_w_sep = delta_a_w_sep * above_min_delta_a_w_sep_value + \
        ~above_min_delta_a_w_sep_value * VBSBAT_options['q_alpha']['min_spread_in_aw']

    # calculate curve parameter of sigmoid
    sigmoid_curve_parameter = ln_zeropass(1. / (1 - VBSBAT_options['q_alpha']['q_alpha_at_1phase_aw']) - 1) / delta_a_w_sep

    # calculate q_alpha value
    q_alpha_value = 1 - 1. / (1 + exp_wlimiter(sigmoid_curve_parameter * (aw_series - a_w_sep + delta_a_w_sep)))

    # apply mask for complete miscibility, turns miscible organics to q_alpha=1 for all a_w
    q_alpha_value = q_alpha_value * ~mask_of_miscible_points + mask_of_miscible_points

    return q_alpha_value



def finds_phase_sep_w_and_org(activity_water, activity_org):
    """
    This function checks for phase separation in each activity curve.

    Parameters:
    activity_water (np.array): A numpy array of water activity values.
    activity_org (np.array): A numpy array of organic activity values.

    Returns:
    tuple: The phase separation check, lower a_w separation index, upper a_w separation index, matching upper a_w separation index.
    """

    # check for phase separation in each activity curve
    _, phase_sep_via_activity_curvature_w, index_phase_sep_starts_w, index_phase_sep_end_w = \
        finds_phase_sep_and_activity_curve_dips_v2(activity_water)
    _, phase_sep_via_activity_curvature_org, index_phase_sep_starts_org, index_phase_sep_end_org = \
        finds_phase_sep_and_activity_curve_dips_v2(activity_org)

    indexes = [index_phase_sep_starts_w, index_phase_sep_end_w, index_phase_sep_starts_org, index_phase_sep_end_org]

    if phase_sep_via_activity_curvature_w == 1:
        phase_sep_check = 1
        if activity_water[0] < activity_water[-1]:  # increasing a_w with index
            lower_a_w_sep_index = min(indexes)
            upper_a_w_sep_index = max(indexes)

            mid_sep_index = (lower_a_w_sep_index + upper_a_w_sep_index) // 2
            activity_water_beta = activity_water[:mid_sep_index]
            match_a_w = activity_water[upper_a_w_sep_index]
            match_index_prime = np.where((activity_water_beta - match_a_w) > 0)

            if len(match_index_prime[0]) == 0:
                match_index_prime = np.argmax(activity_water_beta - match_a_w)
            matching_Upper_a_w_sep_index = match_index_prime[0][0] - 1

        else:
            lower_a_w_sep_index = max(indexes)  # decreasing a_w with index
            upper_a_w_sep_index = min(indexes)

            mid_sep_index = (lower_a_w_sep_index + upper_a_w_sep_index) // 2
            activity_water_beta = activity_water[mid_sep_index:]
            match_a_w = activity_water[upper_a_w_sep_index]
            match_index_prime = np.where(activity_water_beta <= match_a_w)
            matching_Upper_a_w_sep_index = mid_sep_index + match_index_prime[0][0] - 1

    else:
        lower_a_w_sep_index = 1  # no phase sep
        upper_a_w_sep_index = 2
        matching_Upper_a_w_sep_index = 2
        phase_sep_check = 0  # no phase sep

    return phase_sep_check, lower_a_w_sep_index, upper_a_w_sep_index, matching_Upper_a_w_sep_index


def finds_phase_sep_and_activity_curve_dips_v2(activity_data):
    """
    This function finds phase separation and activity curve dips.

    Parameters:
    activity_data (np.array): A numpy array of activity data.

    Returns:
    tuple: The phase separation via activity, phase separation via activity curvature, index phase separation starts, index phase separation end.
    """

    activity_diff = np.diff(activity_data)
    L_m = len(activity_data)

    if L_m > 3:
        min_value = np.min(activity_diff)
        max_value = np.max(activity_diff)
        mean_sign = np.sign(np.mean(activity_diff))

        if np.sign(min_value) == np.sign(max_value):
            phase_sep_via_activity_curvature = 0
            index_phase_sep_starts = np.nan
            index_phase_sep_end = np.nan

        elif np.sign(min_value) != np.sign(max_value):
            phase_sep_via_activity_curvature = 1

            activity_calc1_diff_sign_change = np.sign(np.concatenate(([activity_diff[0]], activity_diff))) != np.sign(activity_diff[0])

            index_start = np.where(activity_calc1_diff_sign_change)[0][0]
            back_index = index_start - 1 + np.where(~activity_calc1_diff_sign_change[index_start:])[0][0]

            if back_index < L_m:
                activity_data_gap = np.argmin(np.abs(activity_data[back_index:] - activity_data[index_start]))
                restart_match_index = activity_data_gap + back_index - 1
            else:
                restart_match_index = L_m

            if sum(activity_data > 1):
                min_value_Idilute = np.min(activity_data[index_start:])
                min_index_Idilute = np.argmin(activity_data[index_start:]) + index_start - 1
                activity_data_gap_start = np.argmin(np.abs(activity_data[:index_start] - activity_data[min_index_Idilute]))

                if activity_data_gap_start < index_start:
                    index_phase_sep_starts = activity_data_gap_start
                else:
                    index_phase_sep_starts = index_start

                if min_index_Idilute < restart_match_index:
                    index_phase_sep_end = min_index_Idilute
                else:
                    index_phase_sep_end = restart_match_index

            else:
                index_phase_sep_starts = index_start
                index_phase_sep_end = restart_match_index

    else:
        phase_sep_via_activity = activity_data
        phase_sep_via_activity_curvature = 0
        index_phase_sep_starts = np.nan
        index_phase_sep_end = np.nan

    if sum(activity_data > 1):
        phase_sep_via_activity = 1
    else:
        phase_sep_via_activity = 0

    return phase_sep_via_activity, phase_sep_via_activity_curvature, index_phase_sep_starts, index_phase_sep_end


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


def mapminmax_apply(x, settings):
    return (x - settings['xoffset']) * settings['gain'] + settings['ymin']

def tansig_apply(n):
    return 2 / (1 + np.exp(-2*n)) - 1

def mapminmax_reverse(y, settings):
    return (y - settings['ymin']) / settings['gain'] + settings['xoffset']

def biphasic_to_single_phase_molfrac_org_NN_v5(x1):
    # Neural Network Constants

    # Input 1
    x1_step1 = {'xoffset': np.array([0, 0.03]), 
                'gain': np.array([4.85105461413554, 6.06060606060606]), 
                'ymin': -1}

    # Layer 1
    b1 = np.array([5.8514275487869635839, 5.7010903702309043695, -4.223155231828595646, 3.8774900021515046333, -1.325265064433274409, 1.7576973644493769644,
    -0.88991761674608715893, 3.1279829884292338349, 0.78794652837451617522, 0.43896968097298133538, 0.57708117017809135163, -3.0787608017900915947,
    -0.40581098864752368494, 0.64373309554603275195, 1.7016326841051676588, 1.4946559455903600799, -1.9151891347855638514, -0.36222536287456474913,
    1.7997779117047685293, 2.1894391519933469326, -4.8365290076330538227, -4.2267640672562194482, 11.037306029484762249, -12.465284279233749487])
    IW1_1 = np.array([
        [-2.9652369062270196309, 3.5456936036245476629],
        [-5.3003896191163066831, 1.6107048934053527223],
        [1.4490960942847532777, -4.4905963483778252865],
        [-5.3604492963083165691, -1.9038162660993562803],
        [3.5615591287827941258, -0.9760690759109916792],
        [-7.4323704665484466858, 3.9952918221278448385],
        [5.2436980487815443297, 0.26336330948747371794],
        [-9.7224092904321945952, -3.3963153832437282809],
        [-6.5939564893121698219, -0.86743278765059372848],
        [-3.9566478027540843421, 6.2043545317513464354],
        [-2.6933489710522580118, -5.3296776285599456457],
        [15.078615754750318345, 7.5357195857930081573],
        [8.5837138409899615965, 2.1781365767882316931],
        [4.2753681534309269097, -1.2376866082237749644],
        [-4.3694881192877721432, -4.1504032312908245572],
        [-5.1076333577305517153, -4.2573773952093914019],
        [3.9331831538436010653, 4.2383815452477779928],
        [-3.3051366335726815038, -2.1527745655619221488],
        [5.5102707313508449971, 0.6090289580711970574],
        [4.0590106228932549826, -1.4537733443192657479],
        [1.1330332610381823599, 7.987215295423046868],
        [-1.4690914445681406697, 3.46783846307828858],
        [1.7834488065629925391, 8.9409991055225059853],
        [-2.3985373019045113097, -10.018658523996098353]
    ])

    # Layer 2
    b2 = -0.23364831139900205104
    LW2_1 = np.array([0.61668049433913207924, -0.3593912575055407399, -0.3266392699738256411, -1.6045946666158248384, -1.1739361061623092564, 0.087334561642434180295,
    2.9371817317733741604, -0.38368295323950185605, 2.3343874738168426397, 0.031130301801809163315, -0.055017015768111438012, -0.2379118380584248349,
    0.51580586145584794711, -0.35767521938020213623, -7.6921394249634440499, 3.3355022461024885772, -4.3095896749403888037, 0.29571381132244778378,
    0.30910107924737867391, -0.25489989065848722705, 0.080905914998754310807, -0.17070806216025710689, 4.7037417262526615147, 3.0438085224819206864])

    # Output 1
    y1_step1 = {'ymin': -1, 'gain': 2.02204023860075, 'xoffset': 0.0109}

    # Simulation
    Q = x1.shape[1] # samples
    xp1 = mapminmax_apply(x1, x1_step1)
    a1 = tansig_apply(b1.reshape(-1, 1) + np.dot(IW1_1, xp1))
    a2 = b2 + np.dot(LW2_1, a1)
    y1 = mapminmax_reverse(a2, y1_step1)

    return y1
