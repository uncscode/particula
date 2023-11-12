"""data processing functions that can be used on datalake objects."""
# linting disabled until reformatting of this file
# pylint: disable=all
# %flake8: noqa
# pytype: skip-file

from typing import Optional, List

import numpy as np
from scipy.interpolate import interp1d

from particula.data.mie import kappa_fitting_caps_data
from particula.data import size_distribution
from particula.util import convert


def caps_processing(
        datalake: object,
        truncation_bsca: bool = True,
        truncation_interval_sec: int = 600,
        truncation_interp: bool = True,
        refractive_index: float = 1.45,
        calibration_wet=1,
        calibration_dry=1,
        kappa_fixed: float = None,
):  # sourcery skip
    """loader.
    Function to process the CAPS data, and smps for kappa fitting, and then add
    it to the datalake. Also applies truncation corrections to the CAPS data.

    Parameters
    ----------
    datalake : object
        DataLake object to add the processed data to.
    truncation_bsca : bool, optional
        Whether to calculate truncation corrections for the bsca data.
        The default is True.
    truncation_interval_sec : int, optional
        The interval to calculate the truncation corrections over.
        The default is 600. This can take around 10 sec per data point.
    truncation_interp : bool, optional
        Whether to interpolate the truncation corrections to the caps data.
    refractive_index : float, optional
        The refractive index of the aerosol. The default is 1.45.
    calibration_wet : float, optional
        The calibration factor for the wet data. The default is 1.
    calibration_dry : float, optional
        The calibration factor for the dry data. The default is 1.

    Returns
    -------
    datalake : object
        DataLake object with the processed data added.
    """
    # calc kappa and add to datalake
    print('CAPS kappa_HGF fitting')

    if kappa_fixed is None:
        kappa_fit, _, _ = kappa_fitting_caps_data(
            datalake=datalake,
            truncation_bsca=False,
            refractive_index=refractive_index
        )
    else:
        kappa_len = len(
            datalake.datastreams['CAPS_data'].return_time(
                datetime64=False))
        kappa_fit = np.ones((kappa_len, 3)) * kappa_fixed

    datalake.datastreams['CAPS_data'].add_processed_data(
        data_new=kappa_fit.T,
        time_new=datalake.datastreams['CAPS_data'].return_time(
            datetime64=False),
        time_new=datalake.datastreams['CAPS_data'].return_time(
            datetime64=False),
        header_new=['kappa_fit', 'kappa_fit_lower', 'kappa_fit_upper'],
    )
    orignal_average = datalake.datastreams['CAPS_data'].average_base_sec
    orignal_average = datalake.datastreams['CAPS_data'].average_base_sec

    # calc truncation corrections and add to datalake
    print('CAPS truncation corrections')
    if truncation_bsca:
        datalake.reaverage_datastreams(
            truncation_interval_sec,
            stream_keys=['CAPS_data', 'smps_1D', 'smps_2D'],
        )
        # epoch_start=epoch_start,
        # epoch_end=epoch_end

        _, bsca_truncation_dry, bsca_truncation_wet = kappa_fitting_caps_data(
            datalake=datalake,
            truncation_bsca=truncation_bsca,
            refractive_index=refractive_index
        )

        if truncation_interp:
            interp_dry = interp1d(
                datalake.datastreams['CAPS_data'].return_time(
                    datetime64=False),
                datalake.datastreams['CAPS_data'].return_time(
                    datetime64=False),
                bsca_truncation_dry,
                kind='linear',
                fill_value='extrapolate'
            )
            interp_wet = interp1d(
                datalake.datastreams['CAPS_data'].return_time(
                    datetime64=False),
                datalake.datastreams['CAPS_data'].return_time(
                    datetime64=False),
                bsca_truncation_wet,
                kind='linear',
                fill_value='extrapolate'
            )

            time = datalake.datastreams['CAPS_data'].return_time(
                datetime64=False,
                raw=True
            )
            bsca_truncation_dry = interp_dry(time)
            bsca_truncation_wet = interp_wet(time)
        else:
            time = datalake.datastreams['CAPS_data'].return_time(
                datetime64=False)

        datalake.datastreams['CAPS_data'].add_processed_data(
            data_new=bsca_truncation_dry.T,
            time_new=time,
            header_new=['bsca_truncation_dry'],
        )
        datalake.datastreams['CAPS_data'].add_processed_data(
            data_new=bsca_truncation_wet.T,
            time_new=time,
            header_new=['bsca_truncation_wet'],
        )
    else:
        bsca_truncation_wet = np.array([1])
        bsca_truncation_dry = np.array([1])
        time = datalake.datastreams['CAPS_data'].return_time(
            datetime64=False,
            raw=True
        )

    # index for Bsca wet and dry
    index_dic = datalake.datastreams['CAPS_data'].return_header_dict()
    index_dic = datalake.datastreams['CAPS_data'].return_header_dict()

    # check if raw in dict
    if 'raw_Bsca_dry_CAPS_450nm[1/Mm]' in index_dic:
        pass
    else:
        # save raw data
        datalake.datastreams['CAPS_data'].add_processed_data(
            data_new=datalake.datastreams['CAPS_data'].data_stream[
                index_dic['Bsca_wet_CAPS_450nm[1/Mm]'], :],
            time_new=time,
            header_new=['raw_Bsca_wet_CAPS_450nm[1/Mm]'],
        )
        datalake.datastreams['CAPS_data'].add_processed_data(
            data_new=datalake.datastreams['CAPS_data'].data_stream[
                index_dic['Bsca_dry_CAPS_450nm[1/Mm]'], :],
            time_new=time,
            header_new=['raw_Bsca_dry_CAPS_450nm[1/Mm]'],
        )
        index_dic = datalake.datastreams['CAPS_data'].return_header_dict()

    datalake.datastreams['CAPS_data'].data_stream[
        index_dic['Bsca_wet_CAPS_450nm[1/Mm]'], :] = \
        datalake.datastreams['CAPS_data'].data_stream[
            index_dic['raw_Bsca_wet_CAPS_450nm[1/Mm]'], :] \
        * bsca_truncation_wet.T * calibration_wet

    datalake.datastreams['CAPS_data'].data_stream[
        index_dic['Bsca_dry_CAPS_450nm[1/Mm]'], :] = \
        datalake.datastreams['CAPS_data'].data_stream[
            index_dic['raw_Bsca_dry_CAPS_450nm[1/Mm]'], :] \
        * bsca_truncation_dry.T * calibration_dry

    datalake.datastreams['CAPS_data'].reaverage(
        reaverage_base_sec=orignal_average
    )  # updates the averages to the original value

    return datalake


def albedo_processing(
    datalake,
    keys: list = None
):
    """
    Calculates the albedo from the CAPS data and updates the datastream.

    Parameters
    ----------
    datalake : object
        DataLake object with the processed data added.

    Returns
    -------
    datalake : object
        DataLake object with the processed data added.
    """

    ssa_wet = datalake.datastreams['CAPS_data'].return_data(
        keys=['Bsca_wet_CAPS_450nm[1/Mm]'])[0] \
        / datalake.datastreams['CAPS_data'].return_data(
        keys=['Bext_wet_CAPS_450nm[1/Mm]'])[0]
    ssa_dry = datalake.datastreams['CAPS_data'].return_data(
        keys=['Bsca_dry_CAPS_450nm[1/Mm]'])[0] \
        / datalake.datastreams['CAPS_data'].return_data(
        keys=['Bext_dry_CAPS_450nm[1/Mm]'])[0]
    ssa_wet = datalake.datastreams['CAPS_data'].return_data(
        keys=['Bsca_wet_CAPS_450nm[1/Mm]'])[0] \
        / datalake.datastreams['CAPS_data'].return_data(
        keys=['Bext_wet_CAPS_450nm[1/Mm]'])[0]
    ssa_dry = datalake.datastreams['CAPS_data'].return_data(
        keys=['Bsca_dry_CAPS_450nm[1/Mm]'])[0] \
        / datalake.datastreams['CAPS_data'].return_data(
        keys=['Bext_dry_CAPS_450nm[1/Mm]'])[0]

    babs_wet = datalake.datastreams['CAPS_data'].return_data(
        keys=['Bext_wet_CAPS_450nm[1/Mm]'])[0] \
        - datalake.datastreams['CAPS_data'].return_data(
        keys=['Bsca_wet_CAPS_450nm[1/Mm]'])[0]
    babs_dry = datalake.datastreams['CAPS_data'].return_data(
        keys=['Bext_dry_CAPS_450nm[1/Mm]'])[0] \
        - datalake.datastreams['CAPS_data'].return_data(
        keys=['Bsca_dry_CAPS_450nm[1/Mm]'])[0]
    babs_wet = datalake.datastreams['CAPS_data'].return_data(
        keys=['Bext_wet_CAPS_450nm[1/Mm]'])[0] \
        - datalake.datastreams['CAPS_data'].return_data(
        keys=['Bsca_wet_CAPS_450nm[1/Mm]'])[0]
    babs_dry = datalake.datastreams['CAPS_data'].return_data(
        keys=['Bext_dry_CAPS_450nm[1/Mm]'])[0] \
        - datalake.datastreams['CAPS_data'].return_data(
        keys=['Bsca_dry_CAPS_450nm[1/Mm]'])[0]

    time = datalake.datastreams['CAPS_data'].return_time(datetime64=False)
    time = datalake.datastreams['CAPS_data'].return_time(datetime64=False)

    datalake.datastreams['CAPS_data'].add_processed_data(
        data_new=ssa_wet,
        time_new=time,
        header_new=['SSA_wet_CAPS_450nm[1/Mm]'],
    )
    datalake.datastreams['CAPS_data'].add_processed_data(
        data_new=ssa_dry,
        time_new=time,
        header_new=['SSA_dry_CAPS_450nm[1/Mm]'],
    )
    datalake.datastreams['CAPS_data'].add_processed_data(
        data_new=babs_wet,
        time_new=time,
        header_new=['Babs_wet_CAPS_450nm[1/Mm]'],
    )
    datalake.datastreams['CAPS_data'].add_processed_data(
        data_new=babs_dry,
        time_new=time,
        header_new=['Babs_dry_CAPS_450nm[1/Mm]'],
    )
    return datalake


def ccnc_hygroscopicity(
        datalake,
        supersaturation_bounds=[0.3, 0.9],
        dp_crit_threshold=75
):
    """
    Calculate the hygroscopicity of CCNc using the activation diameter, and the
    smps data.

    Parameters
    ----------
    datalake : DataLake
        collection of datastreams.
    supersaturation_bounds : list, optional
        supersaturation bounds for the activation diameter. The default is
        [0.3, 0.9].
    dp_crit_threshold : float, optional
        dp_crit threshold for the activation diameter. The default is 75 nm.

    Returns
    -------
    datalake : object
        DataLake object with the processed data added.

    Petters, M. D.,; Kreidenweis, S. M. (2007). A single parameter
    representation of hygroscopic growth and cloud condensation nucleus
    activity, Atmospheric Chemistry and Physics, 7(8), 1961-1971.
    https://doi.org/10.5194/acp-7-1961-2007

    """

    time = datalake.datastreams['CCNc'].return_time(datetime64=False)
    ccnc_number = datalake.datastreams['CCNc'].return_data(
        keys=['CCN_Concentration_[#/cc]'])[0]
    super_sat_set = datalake.datastreams['CCNc'].return_data(
        keys=['CurrentSuperSaturationSet[%]'])[0]
    sizer_total_n = datalake.datastreams['smps_1D'].return_data(
        keys=['Total_Conc_(#/cc)'])[0]
    sizer_diameter = datalake.datastreams['smps_2D'].return_header_list(
    ).astype(float)
    sizer_diameter_fliped = np.flip(sizer_diameter)
    sizer_dndlogdp = np.nan_to_num(
        datalake.datastreams['smps_2D'].return_data())

    fitted_dp_crit = np.zeros_like(super_sat_set)
    activated_fraction = np.zeros_like(super_sat_set)

    for i in range(len(super_sat_set)):

        super_sat_set

        if ccnc_number[i] < sizer_total_n[i] \
           and ccnc_number[i] > 50 \
           and super_sat_set[i] > supersaturation_bounds[0] \
           and super_sat_set[i] <= supersaturation_bounds[1]:
            sizer_dn = convert.convert_sizer_dn(
                sizer_diameter, sizer_dndlogdp[:, i])
            sizer_dn = sizer_dn * sizer_total_n[i] / np.sum(sizer_dn)

            sizer_dn_cumsum = np.cumsum(np.flip(sizer_dn))

            fitted_dp_crit[i] = np.interp(
                ccnc_number[i],
                sizer_dn_cumsum,
                sizer_diameter_fliped,
                left=np.nan,
                right=np.nan)
            activated_fraction[i] = ccnc_number[i] / sizer_total_n[i]
        else:
            fitted_dp_crit[i] = np.nan
            activated_fraction[i] = np.nan

    kelvin_diameter = 1.06503 * 2  # nm # update to be a function of temp

    # Gohil, K., &#38; Asa-Awuku, A. A. (2022). Cloud condensation nuclei
    # (CCN) activity analysis of low-hygroscopicity aerosols using the
    # aerodynamic aerosol classifier (AAC). <i>Atmospheric Measurement
    # Techniques</i>, <i>15</i>(4), 1007–1019.
    # https://doi.org/10.5194/amt-15-1007-2022
    fitted_kappa = 4 * kelvin_diameter**3 / \
        (27 * fitted_dp_crit**3 * np.log(1 + super_sat_set / 100)**2)
    fitted_kappa_threshold = np.where(
        fitted_dp_crit > dp_crit_threshold, fitted_kappa, np.nan)

    datalake.add_processed_datastream(
        key='kappa_ccn',
        time_stream=time,
        data_stream=np.vstack(
            (fitted_dp_crit,
             activated_fraction,
             fitted_kappa,
             fitted_kappa_threshold)),
        data_stream_header=[
            'dp_critical',
            'activated_fraction',
            'kappa_CCNc',
            'kappa_CCNc_threshold'],
        average_times=[90],
        average_base=[90]
    )
    return datalake


def sizer_mean_properties(
    datalake: object,
    stream_key: str,
    new_key: str = 'sizer_mean_properties',
    sizer_limits: Optional[List[float]] = None,
    density: float = 1.5,
    diameter_multiplier_to_nm: float = 1.0
) -> object:
    """
    Calculates the mean properties of the size distribution. Adds the data to
    the datalake.

    Parameters
    ----------
    datalake : DataLake
        The datalake to process.
    stream_key : str
        The key for the 2d size distribution datastream.
    new_key : str, optional
        The key for the new datastream. The default is 'sizer_mean_properties'.
    sizer_limits : list, optional
        The lower and upper limits of the size of interest. The default is None
    density : float, optional
        The density of the particles. The default is 1.5.
    diameters_multiplier_to_nm : float, optional
        The multiplier to convert the diameters to nm. The default is 1.0.

    Returns
    -------
    datalake : DataLake
        The datalake with the mean properties added.
    """
    # check stream key is in datalake
    if stream_key not in datalake.list_datastreams():
        raise KeyError('stream_key not in datalake')

    time = datalake.datastreams[stream_key].return_time(datetime64=False)
    sizer_diameter_smps = np.array(
        datalake.datastreams[stream_key].return_header_list()
    ).astype(float) * diameter_multiplier_to_nm  # convert to nm
    sizer_dndlogdp_smps = np.nan_to_num(
        datalake.datastreams[stream_key].return_data())

    # # do: fix aps data to concentrations
    # sizer_dndlogdp_aps = datalake.datastreams['aps_2D'].return_data()/5

    total_concentration = np.zeros_like(time) * np.nan
    unit_mass_ugPm3 = np.zeros_like(time) * np.nan
    mean_diameter_nm = np.zeros_like(time) * np.nan
    mean_vol_diameter_nm = np.zeros_like(time) * np.nan
    geometric_mean_diameter_nm = np.zeros_like(time) * np.nan
    mode_diameter = np.zeros_like(time) * np.nan
    mode_diameter_mass = np.zeros_like(time) * np.nan

    total_concentration_PM01 = np.zeros_like(time) * np.nan
    unit_mass_ugPm3_PM01 = np.zeros_like(time) * np.nan

    total_concentration_PM1 = np.zeros_like(time) * np.nan
    unit_mass_ugPm3_PM1 = np.zeros_like(time) * np.nan

    total_concentration_PM25 = np.zeros_like(time) * np.nan
    unit_mass_ugPm3_PM25 = np.zeros_like(time) * np.nan

    total_concentration_PM10 = np.zeros_like(time) * np.nan
    unit_mass_ugPm3_PM10 = np.zeros_like(time) * np.nan

    for i in range(len(time)):
        total_concentration[i], unit_mass_ugPm3[i], mean_diameter_nm[i], \
            mean_vol_diameter_nm[i], geometric_mean_diameter_nm[i], \
            mode_diameter[i], mode_diameter_mass[i] = \
            size_distribution.mean_properties(
            sizer_dndlogdp_smps[:, i],
            sizer_diameter_smps,
            sizer_limits=sizer_limits
        )

        # total PM 100 nm concentration
        total_concentration_PM01[i], unit_mass_ugPm3_PM01[i], _, _, _, _, _ = \
            size_distribution.mean_properties(
            sizer_dndlogdp_smps[:, i],
            sizer_diameter_smps,
            sizer_limits=[0, 100]
        )

        # total PM1 um concentration
        total_concentration_PM1[i], unit_mass_ugPm3_PM1[i], _, _, _, _, _ = \
            size_distribution.mean_properties(
            sizer_dndlogdp_smps[:, i],
            sizer_diameter_smps,
            sizer_limits=[0, 1000]
        )
        # total PM <2.5 um concentration
        total_concentration_PM25[i], unit_mass_ugPm3_PM25[i], _, _, _, _, _ = \
            size_distribution.mean_properties(
            sizer_dndlogdp_smps[:, i],
            sizer_diameter_smps,
            sizer_limits=[0, 2500]
        )

        # total PM <10 um concentration
        total_concentration_PM10[i], unit_mass_ugPm3_PM10[i], _, _, _, _, _ = \
            size_distribution.mean_properties(
            sizer_dndlogdp_smps[:, i],
            sizer_diameter_smps,
            sizer_limits=[0, 10000]
        )

    mass_ugPm3 = unit_mass_ugPm3 * density
    mass_ugPm3_PM01 = unit_mass_ugPm3_PM01 * density
    mass_ugPm3_PM1 = unit_mass_ugPm3_PM1 * density
    mass_ugPm3_PM25 = unit_mass_ugPm3_PM25 * density
    mass_ugPm3_PM10 = unit_mass_ugPm3_PM10 * density

    # combine the data for datalake
    combinded = np.vstack((
        total_concentration,
        mean_diameter_nm,
        geometric_mean_diameter_nm,
        mode_diameter,
        mean_vol_diameter_nm,
        mode_diameter_mass,
        unit_mass_ugPm3,
        mass_ugPm3,
        total_concentration_PM01,
        unit_mass_ugPm3_PM01,
        mass_ugPm3_PM01,
        total_concentration_PM1,
        unit_mass_ugPm3_PM1,
        mass_ugPm3_PM1,
        total_concentration_PM25,
        unit_mass_ugPm3_PM25,
        mass_ugPm3_PM25,
        total_concentration_PM10,
        unit_mass_ugPm3_PM10,
        mass_ugPm3_PM10,
    ))
    header = [
        'Total_Conc_(#/cc)',
        'Mean_Diameter_(nm)',
        'Geometric_Mean_Diameter_(nm)',
        'Mode_Diameter_(nm)',
        'Mean_Diameter_Vol_(nm)',
        'Mode_Diameter_Vol_(nm)',
        'Unit_Mass_(ug/m3)',
        'Mass_(ug/m3)',
        'Total_Conc_(#/cc)_N100',
        'Unit_Mass_(ug/m3)_N100',
        'Mass_(ug/m3)_N100',
        'Total_Conc_(#/cc)_PM1',
        'Unit_Mass_(ug/m3)_PM1',
        'Mass_(ug/m3)_PM1',
        'Total_Conc_(#/cc)_PM2.5',
        'Unit_Mass_(ug/m3)_PM2.5',
        'Mass_(ug/m3)_PM2.5',
        'Total_Conc_(#/cc)_PM10',
        'Unit_Mass_(ug/m3)_PM10',
        'Mass_(ug/m3)_PM10',
    ]

    datalake.add_processed_datastream(
        key=new_key,
        time_stream=time,
        data_stream=combinded,
        data_stream_header=header,
        average_times=datalake.datastreams[stream_key].average_int_sec,
        average_base=datalake.datastreams[stream_key].average_base_sec
    )
    return datalake


def merge_distributions(
    concentration_lower: np.ndarray,
    diameters_lower: np.ndarray,
    concentration_upper: np.ndarray,
    diameters_upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge two particle size distributions using linear weighting.

    Parameters:
    concentration_lower:
        The concentration of particles in the lower
        distribution.
    diameters_lower:
        The diameters corresponding to the lower distribution.
        concentration_upper: The concentration of particles in the upper
        distribution.
    diameters_upper:
        The diameters corresponding to the upper distribution.

    Returns:
    new_2d: The merged concentration distribution.
    new_diameter: The merged diameter distribution.

    TODO: acount for the moblity vs aerodynamic diameters
    """
    # Define the linear weight function
    def weight_func(diameter, min_diameter, max_diameter):
        # Calculate the weight for each diameter
        weight = (diameter - min_diameter) / (max_diameter - min_diameter)

        # Clip the weights to the range [0, 1]
        weight = np.clip(weight, 0, 1)

        return weight

    # Find the overlapping range of diameters
    min_diameter = max(np.min(diameters_upper), np.min(diameters_lower))
    max_diameter = min(np.max(diameters_upper), np.max(diameters_lower))

    lower_min_overlap = np.argmin(np.abs(diameters_lower - min_diameter))
    upper_max_overlap = np.argmin(np.abs(diameters_upper - max_diameter))

    # Define the weighted grid
    weighted_diameter = diameters_lower[lower_min_overlap:]

    # Interpolate the lower and upper distributions onto the weighted grid
    lower_interp = concentration_lower[lower_min_overlap:]
    upper_interp = np.interp(
        weighted_diameter,
        diameters_upper,
        concentration_upper,
        left=0,
        right=0)

    # Apply the weights to the interpolated distributions
    weighted_lower = lower_interp * (
        1 - weight_func(weighted_diameter, min_diameter, max_diameter))
    weighted_upper = upper_interp * weight_func(
        weighted_diameter, min_diameter, max_diameter)

    # Add the weighted distributions together
    merged_2d = weighted_lower + weighted_upper

    # Combine the diameters
    new_diameter = np.concatenate((
        diameters_lower[:lower_min_overlap],
        weighted_diameter,
        diameters_upper[upper_max_overlap:]
    ))

    # Combine the concentrations
    new_2d = np.concatenate((
        concentration_lower[:lower_min_overlap],
        merged_2d,
        concentration_upper[upper_max_overlap:]))

    return new_2d, new_diameter


def iterate_merge_distributions(
    concentration_lower: np.ndarray,
    diameters_lower: np.ndarray,
    concentration_upper: np.ndarray,
    diameters_upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge two sets of particle size distributions using linear weighting.

    Parameters:
    concentration_lower: The concentration of particles in the
        lower distribution.
    diameters_lower: The diameters corresponding to the
        lower distribution.
    concentration_upper: The concentration of particles in the
        upper distribution.
    diameters_upper: The diameters corresponding to the upper distribution.

    Returns:
    A tuple containing the merged diameter distribution and the merged
        concentration distribution.
    """
    # Iterate over all columns in the concentration datastream
    merged_2d_list = []
    for i in range(concentration_lower.shape[1]):
        # Get the current column of the lower concentration distribution
        concentration_lower_col = concentration_lower[:, i]

        # Merge the current column of the lower and upper concentration
        merged_2d, merged_diameter = merge_distributions(
            concentration_lower_col,
            diameters_lower,
            concentration_upper[:, i],
            diameters_upper
        )

        # Add the merged concentration distribution to the list
        merged_2d_list.append(merged_2d)

    # Combine the merged concentration distributions into a single array
    merged_2d_array = np.column_stack(merged_2d_list)

    # Return the merged diameter distribution and the merged concentration
    return merged_diameter, merged_2d_array


def merge_smps_ops_datastreams(
    datalake: object,
    lower_key: str,
    upper_key: str,
    new_key='sizer_merged',
    scale_upper_dp=1000,
) -> object:
    """
    Merge two sets of particle size distributions using linear weighting.

    Parameters:
    datalake: The Lake object containing the datastreams.
    lower_key: The key for the lower range distribution (e.g. 'smps_2D').
    upper_key: The key for the upper range distribution (e.g. 'ops_2D').

    Returns:
    A datalake object

    TODO: add scaling of diamters to the import functions
    """

    # Get the datastreams from the Lake object
    lower_datastream = datalake.datastreams[lower_key]
    upper_datastream = datalake.datastreams[upper_key]

    # Get the concentration and diameter data from the datastreams
    concentration_lower = lower_datastream.return_data()
    diameters_lower = np.array(
        lower_datastream.return_header_list()).astype(float)

    concentration_upper = upper_datastream.return_data()
    diameters_upper = np.array(
        upper_datastream.return_header_list()).astype(float) * scale_upper_dp

    # Merge the datastreams
    merged_diameter, merged_2d = iterate_merge_distributions(
        concentration_lower=concentration_lower,
        diameters_lower=diameters_lower,
        concentration_upper=concentration_upper,
        diameters_upper=diameters_upper
    )

    datalake.add_processed_datastream(
        key=new_key,
        time_stream=lower_datastream.return_time(datetime64=False),
        data_stream=merged_2d,
        data_stream_header=list(merged_diameter.astype(str)),
        average_times=lower_datastream.average_int_sec,
        average_base=lower_datastream.average_base_sec
    )

    # Reaverage the datastream for the new data set
    datalake.reaverage_datastreams(
        average_base_sec=lower_datastream.average_base_sec,
        stream_keys=[new_key],
        epoch_start=lower_datastream.average_epoch_start,
        epoch_end=lower_datastream.average_epoch_end
    )
    # Return the merged diameter distribution and the merged concentration
    return datalake


def pass3_processing(
        datalake: object,
        babs_405_532_781=[1, 1, 1],
        bsca_405_532_781=[1, 1, 1],
):
    """
    Processing PASS3 data applying the calibration factors
    TODO: add the background correction

    Parameters
    ----------
    datalake : object
        DataLake object to add the processed data to.
    babs_405_532_781 : list, optional
        Calibration factors for the absorption channels. The default is [1,1,1]
    bsca_405_532_781 : list, optional
        Calibration factors for the scattering channels. The default is [1,1,1]

    Returns
    -------
    datalake : object
        DataLake object with the processed data added.
    """
    # index for Bsca wet and dry
    index_dic = datalake.datastreams['pass3'].return_header_dict()
    time = datalake.datastreams['pass3'].return_time(
        datetime64=False,
        raw=True
    )
    babs_list = ['Babs405nm[1/Mm]', 'Babs532nm[1/Mm]', 'Babs781nm[1/Mm]']
    bsca_list = ['Bsca405nm[1/Mm]', 'Bsca532nm[1/Mm]', 'Bsca781nm[1/Mm]']

    if 'raw_Babs405nm[1/Mm]' not in index_dic:
        print('Copy raw babs Pass-3')
        for babs in babs_list:
            raw_name = f'raw_{babs}'

            datalake.datastreams['pass3'].add_processed_data(
                data_new=datalake.datastreams['pass3'].data_stream[
                    index_dic[babs], :],
                time_new=time,
                header_new=[raw_name],
            )
    if 'raw_Bsca405nm[1/Mm]' not in index_dic:
        print('Copy raw bsca Pass-3')
        for bsca in bsca_list:
            raw_name = f'raw_{bsca}'

            datalake.datastreams['pass3'].add_processed_data(
                data_new=datalake.datastreams['pass3'].data_stream[
                    index_dic[bsca], :],
                time_new=time,
                header_new=[raw_name],
            )

    index_dic = datalake.datastreams['pass3'].return_header_dict()

    # calibration loop babs.
    print('Calibrated raw Pass-3')
    for i, babs in enumerate(babs_list):
        raw_name = f'raw_{babs}'
        datalake.datastreams['pass3'].data_stream[index_dic[babs], :] = \
            datalake.datastreams['pass3'].data_stream[index_dic[raw_name], :] \
            * babs_405_532_781[i]
    # calibration loop bsca
    for i, bsca in enumerate(bsca_list):
        raw_name = f'raw_{bsca}'
        datalake.datastreams['pass3'].data_stream[index_dic[bsca], :] = \
            datalake.datastreams['pass3'].data_stream[index_dic[raw_name], :] \
            * bsca_405_532_781[i]

    return datalake
