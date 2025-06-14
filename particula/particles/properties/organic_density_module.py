"""Species density estimation functions."""

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from particula.activity.ratio import from_molar_mass_ratio

# molar masses in g/mol
MASS_C = 12.01
MASS_O = 16.0
MASS_H = 1.008
MASS_N = 14.0067


def get_organic_density_estimate(
    molar_mass: float,
    oxygen2carbon: float,
    hydrogen2carbon: Optional[float] = None,
    nitrogen2carbon: Optional[float] = None,
    mass_ratio_convert: bool = False,
) -> float:
    """Estimate the density of organic compounds using the Girolami method.

    The function follows a simplified approach described by Girolami (1994) to
    approximate the density of an organic species. The required inputs are the
    molar mass and the O:C ratio. Optionally the H:C and N:C ratios can also be
    provided. When the H:C ratio is unknown, supply a negative value and it will
    be estimated assuming H:C = 2. The number of carbon atoms per molecule is
    then determined from these ratios and the density is calculated accordingly.

    Reference
    ---------
    Girolami, G. S.: A Simple "Back of the Envelope" Method for Estimating the
    Densities and Molecular Volumes of Liquids and Solids, J. Chem. Educ.,
    71(11), 962, doi:10.1021/ed071p962, 1994.

    Parameters
    ----------
    molar_mass : float
        Molar mass of the compound.
    oxygen2carbon : float
        Atomic O:C ratio.
    hydrogen2carbon : float, optional
        Atomic H:C ratio. If negative, it is estimated within the function.
    nitrogen2carbon : float, optional
        Atomic N:C ratio.
    mass_ratio_convert : bool, default False
        If ``True``, convert ``molar_mass`` from a mass ratio.

    Returns
    -------
    float
        Estimated density in g/cm^3.
    """
    if nitrogen2carbon is None:
        nitrogen2carbon = oxygen2carbon * 0
    if hydrogen2carbon is None:
        hydrogen2carbon = oxygen2carbon * 0
    if mass_ratio_convert:
        molar_mass = from_molar_mass_ratio(molar_mass)

    hydrogen2carbon_est = (
        2.0 - oxygen2carbon if hydrogen2carbon < 0.1 else hydrogen2carbon
    )
    number_carbons = molar_mass / (
        MASS_C
        + hydrogen2carbon_est * MASS_H
        + oxygen2carbon * MASS_O
        + nitrogen2carbon * MASS_N
    )
    rho1 = molar_mass / (
        5.0
        * number_carbons
        * (
            2.0
            + hydrogen2carbon_est
            + oxygen2carbon * 2.0
            + nitrogen2carbon * 2.0
        )
    )
    return rho1 * (
        1.0
        + min(
            number_carbons * oxygen2carbon * 0.1
            + number_carbons * nitrogen2carbon * 0.1,
            0.3,
        )
    )


def get_organic_density_array(
    molar_mass: Union[list[float], NDArray[np.float64]],
    oxygen2carbon: Union[list[float], NDArray[np.float64]],
    hydrogen2carbon: Optional[Union[list[float], NDArray[np.float64]]] = None,
    nitrogen2carbon: Optional[Union[list[float], NDArray[np.float64]]] = None,
    mass_ratio_convert: bool = False,
) -> NDArray[np.float64]:
    """Return densities for an array of compounds."""
    mm = np.asarray(molar_mass, dtype=float)
    oc = np.asarray(oxygen2carbon, dtype=float)
    hc = None if hydrogen2carbon is None else np.asarray(hydrogen2carbon, dtype=float)
    nc = None if nitrogen2carbon is None else np.asarray(nitrogen2carbon, dtype=float)
    density = np.empty(mm.shape, dtype=float)
    for i, molar in enumerate(mm):
        hc_run = None if hc is None else hc[i]
        nc_run = None if nc is None else nc[i]
        density[i] = get_organic_density_estimate(
            molar_mass=molar,
            oxygen2carbon=oc[i],
            hydrogen2carbon=hc_run,
            nitrogen2carbon=nc_run,
            mass_ratio_convert=mass_ratio_convert,
        )
    return density
