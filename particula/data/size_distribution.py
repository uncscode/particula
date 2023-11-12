# linting disabled until reformatting of this file
# pylint: disable=all
# pytype: skip-file
# flake8: noqa
"""functions for size distribution analysis"""

from typing import Optional, Tuple, List
from math import pi
from scipy.stats.mstats import gmean
import numpy as np
from particula.util import convert


def mean_properties(
    sizer_dndlogdp: List[float],
    sizer_diameter: List[float],
    total_concentration: Optional[float] = None,
    sizer_limits: Optional[Tuple[float, float]] = None
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
        volume = 4*pi/3 * (sizer_diameter/2)**3
    else:
        threshold_limits = (sizer_diameter >= sizer_limits[0]) & \
            (sizer_diameter <= sizer_limits[1])  # gets indexes to keep

        # if no bins are in the range, return nans
        if np.sum(threshold_limits) == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        sizer_dn = sizer_dn[threshold_limits]
        sizer_diameter = sizer_diameter[threshold_limits]

        volume = 4*pi/3 * (sizer_diameter/2)**3
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
    sizer_dn_cumsum = np.cumsum(sizer_dn)/total_concentration
    mode_diameter = np.interp(
            0.5,
            sizer_dn_cumsum,
            sizer_diameter,
            left=np.nan,
            right=np.nan
        )

    # Calculate mode diameter by mass.
    sizer_dmass_cumsum = np.cumsum(mass_ug_m3)/unit_mass_ug_m3
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
