"""Size distribution processing functions to calculate mean properties and
merge distributions."""

# pylint: disable=too-many-locals

from typing import Optional, List, Tuple

from math import pi
import numpy as np
from scipy.stats.mstats import gmean

from particula.util.input_handling import convert_units
from particula.util import convert
from particula.data.stream import Stream


def mean_properties(
    sizer_dndlogdp: np.ndarray,
    sizer_diameter: np.ndarray,
    total_concentration: Optional[float] = None,
    sizer_limits: Optional[list] = None
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Calculates the mean properties of the size distribution.

    Parameters
    ----------
    sizer_dndlogdp : List[float]
        Concentration of particles in each bin.
    sizer_diameter : List[float]
        Bin centers
    total_concentration : Optional[float], default=None
        Total concentration of particles in the distribution.
    sizer_limits : Optional[Tuple[float, float]], default=None
        The lower and upper limits of the size of interest.

    Returns
    -------
    Tuple[float, float, float, float, float, float, float]
        Total concentration of particles in the distribution.
        Total mass of particles in the distribution.
        Mean diameter of the distribution by number.
        Mean diameter of the distribution by volume.
        Geometric mean diameter of the distribution.
        Mode diameter of the distribution by number.
        Mode diameter of the distribution by volume.
    """

    # convert to dn from dn/dlogDp
    sizer_dn = convert.convert_sizer_dn(sizer_diameter, sizer_dndlogdp)
    if total_concentration is not None:
        sizer_dn = sizer_dn * total_concentration / np.sum(sizer_dn)
    else:
        total_concentration = np.sum(sizer_dn)

    if sizer_limits is None:
        volume = 4 * pi / 3 * (sizer_diameter / 2)**3
    else:
        threshold_limits = (sizer_diameter >= sizer_limits[0]) & \
            (sizer_diameter <= sizer_limits[1])  # gets indexes to keep

        # if no bins are in the range, return nans
        if np.sum(threshold_limits) == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        sizer_dn = sizer_dn[threshold_limits]
        sizer_diameter = sizer_diameter[threshold_limits]

        volume = 4 * pi / 3 * (sizer_diameter / 2)**3
        total_concentration = np.sum(sizer_dn)

    # Calculate mass in ug_m3 assuming a density of 1.
    mass_ug_m3 = volume * sizer_dn * 1e-9
    unit_mass_ug_m3 = np.sum(mass_ug_m3)

    # Calculate mean diameter by number.
    normalized = sizer_dn / total_concentration
    diameter_weighted = normalized * sizer_diameter
    mean_diameter_nm = np.sum(diameter_weighted)

    # mean diameter by volume, unit density is assumed so mass=volume
    normalized_vol = mass_ug_m3 / unit_mass_ug_m3
    diameter_weighted_vol = normalized_vol * sizer_diameter
    mean_vol_diameter_nm = np.sum(diameter_weighted_vol)

    # Calculate geometric mean diameter of the distribution.
    geometric_mean_diameter_nm = gmean(sizer_diameter, weights=normalized)

    # Calculate mode diameter by number.
    sizer_dn_cumsum = np.cumsum(sizer_dn) / total_concentration
    mode_diameter = np.interp(
        0.5,
        sizer_dn_cumsum,
        sizer_diameter,
        left=np.nan,
        right=np.nan
    )

    # Calculate mode diameter by mass.
    sizer_dmass_cumsum = np.cumsum(mass_ug_m3) / unit_mass_ug_m3
    mode_diameter_mass = np.interp(
        0.5,
        sizer_dmass_cumsum,
        sizer_diameter,
        left=np.nan,
        right=np.nan
    )
    return total_concentration, unit_mass_ug_m3, mean_diameter_nm, \
        mean_vol_diameter_nm, geometric_mean_diameter_nm, \
        mode_diameter, mode_diameter_mass


def sizer_mean_properties(
    stream: Stream,
    sizer_limits: Optional[List[float]] = None,
    density: float = 1.5,
    diameter_units: str = 'nm',
) -> Stream:
    """
    Calculates the mean properties of the size distribution, and returns a
    stream.

    Parameters
    ----------
    stream : Stream
        The stream to process.
    sizer_limits : list, optional [in diameter_units]
        The lower and upper limits of the size of interest. The default is None
    density : float, optional
        The density of the particles. The default is 1.5 g/cm3.
    diameter_units : str
        The units of the diameter. The default is 'nm'. This will be converted
        to nm.

    Returns
    -------
    stream : Stream
        The stream with the mean properties added.
    """

    sizer_diameter_smps = np.array(stream.header).astype(float) \
        * convert_units(diameter_units, 'nm')
    if sizer_limits is not None:  # convert to nm
        sizer_limits[0] *= convert_units(diameter_units, 'nm')
        sizer_limits[1] *= convert_units(diameter_units, 'nm')
    sizer_dndlogdp_smps = np.nan_to_num(stream.data)

    total_concentration = np.zeros_like(stream.time) * np.nan
    unit_mass_ug_m3 = np.zeros_like(total_concentration) * np.nan
    mean_diameter_nm = np.zeros_like(total_concentration) * np.nan
    mean_vol_diameter_nm = np.zeros_like(total_concentration) * np.nan
    geometric_mean_diameter_nm = np.zeros_like(total_concentration) * np.nan
    mode_diameter = np.zeros_like(total_concentration) * np.nan
    mode_diameter_mass = np.zeros_like(total_concentration) * np.nan

    total_concentration_pm01 = np.zeros_like(total_concentration) * np.nan
    unit_mass_ug_m3_pm01 = np.zeros_like(total_concentration) * np.nan

    total_concentration_pm1 = np.zeros_like(total_concentration) * np.nan
    unit_mass_ug_m3_pm1 = np.zeros_like(total_concentration) * np.nan

    total_concentration_pm25 = np.zeros_like(total_concentration) * np.nan
    unit_mass_ug_m3_pm25 = np.zeros_like(total_concentration) * np.nan

    total_concentration_pm10 = np.zeros_like(total_concentration) * np.nan
    unit_mass_ug_m3_pm10 = np.zeros_like(total_concentration) * np.nan

    for i in range(len(stream.time)):
        total_concentration[i], unit_mass_ug_m3[i], mean_diameter_nm[i], \
            mean_vol_diameter_nm[i], geometric_mean_diameter_nm[i], \
            mode_diameter[i], mode_diameter_mass[i] = \
            mean_properties(
            sizer_dndlogdp_smps[i, :],
            sizer_diameter_smps,
            sizer_limits=sizer_limits
        )

        # total PM 100 nm concentration
        total_concentration_pm01[i], unit_mass_ug_m3_pm01[i], _, _, _, _, _ = \
            mean_properties(
            sizer_dndlogdp_smps[i, :],
            sizer_diameter_smps,
            sizer_limits=[0, 100]
        )

        # total PM1 um concentration
        total_concentration_pm1[i], unit_mass_ug_m3_pm1[i], _, _, _, _, _ = \
            mean_properties(
            sizer_dndlogdp_smps[i, :],
            sizer_diameter_smps,
            sizer_limits=[0, 1000]
        )
        # total PM <2.5 um concentration
        total_concentration_pm25[i], unit_mass_ug_m3_pm25[i], _, _, _, _, _ = \
            mean_properties(
            sizer_dndlogdp_smps[i, :],
            sizer_diameter_smps,
            sizer_limits=[0, 2500]
        )

        # total PM <10 um concentration
        total_concentration_pm10[i], unit_mass_ug_m3_pm10[i], _, _, _, _, _ = \
            mean_properties(
            sizer_dndlogdp_smps[i, :],
            sizer_diameter_smps,
            sizer_limits=[0, 10000]
        )

    mass_ug_m3 = unit_mass_ug_m3 * density
    mass_ug_m3_pm01 = unit_mass_ug_m3_pm01 * density
    mass_ug_m3_pm1 = unit_mass_ug_m3_pm1 * density
    mass_ug_m3_pm25 = unit_mass_ug_m3_pm25 * density
    mass_ug_m3_pm10 = unit_mass_ug_m3_pm10 * density

    # combine the data for datalake
    combinded = np.stack((
        total_concentration,
        mean_diameter_nm,
        geometric_mean_diameter_nm,
        mode_diameter,
        mean_vol_diameter_nm,
        mode_diameter_mass,
        unit_mass_ug_m3,
        mass_ug_m3,
        total_concentration_pm01,
        unit_mass_ug_m3_pm01,
        mass_ug_m3_pm01,
        total_concentration_pm1,
        unit_mass_ug_m3_pm1,
        mass_ug_m3_pm1,
        total_concentration_pm25,
        unit_mass_ug_m3_pm25,
        mass_ug_m3_pm25,
        total_concentration_pm10,
        unit_mass_ug_m3_pm10,
        mass_ug_m3_pm10,
    ), axis=1)
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

    return Stream(
        header=header,
        data=combinded,
        time=stream.time,
    )


def merge_distributions(
    concentration_lower: np.ndarray,
    diameters_lower: np.ndarray,
    concentration_upper: np.ndarray,
    diameters_upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge two particle size distributions using linear weighting.

    Args:
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

    add in an acount for the moblity vs aerodynamic diameters
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

    Args:
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


def merge_size_distribution(
    stream_lower: Stream,
    stream_upper: Stream,
    lower_units: str = 'nm',
    upper_units: str = 'um',
) -> object:
    """
    Merge two sets of particle size distributions using linear weighting.
    The concentration should be in dN/dlogDp.

    Args:
    stream_smaller: The stream with lower sizes, e.g. SMPS.
    stream_larger: The stream with larger sizes, e.g. OPS. or APS
    lower_units: The units of the lower distribution. The default is 'nm'.
    upper_units: The units of the upper distribution. The default is 'um'.

    Returns:
    Stream: A stream object containing the merged size distribution.
    """
    # Get the diameter data from the datastreams
    diameters_lower = np.array(stream_lower.header).astype(float) \
        * convert_units(lower_units, 'nm')

    diameters_upper = np.array(stream_upper.header).astype(float) \
        * convert_units(upper_units, 'nm')

    # Merge the datastreams
    merged_diameter, merged_2d = iterate_merge_distributions(
        concentration_lower=stream_lower.data,
        diameters_lower=diameters_lower,
        concentration_upper=stream_upper.data,
        diameters_upper=diameters_upper
    )
    return Stream(
        header=list(merged_diameter.astype(str)),
        data=merged_2d,
        time=stream_lower.time,
    )
