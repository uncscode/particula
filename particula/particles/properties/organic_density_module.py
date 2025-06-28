"""Organic density estimation utilities.

Provides helper functions that implement the Girolami (1994) “back-of-the-
envelope” method for estimating the density of organic compounds from their
elemental composition.
"""

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
    """Estimate the density of an organic molecule via the Girolami method.

    The original paper proposes an empirical two-step approach:

    1.  Base density (ρ₀) is obtained from
        ρ₀ = M / (5 · n_C · (2 + H:C + 2 · O:C + 2 · N:C)) #[g cm⁻³]

    2.  A polar-functional correction is applied:
        ρ = ρ₀ × [1 + min(0.1 · n_C · (O:C + N:C), 0.3)]

        -  M is the molar mass [g mol⁻¹]
        -  n_C is the number of carbon atoms in the molecule
        -  H:C, O:C, N:C are atomic ratios.

    Arguments:
        - molar_mass : Molar mass of the compound in g mol⁻¹. If
          ``mass_ratio_convert`` is ``True`` this is interpreted as a mass
          ratio and is internally converted to molar mass.
        - oxygen2carbon : Atomic O:C ratio.
        - hydrogen2carbon : Atomic H:C ratio.  Supply a *negative* value (or
          ``None``) to assume H:C = 2 − O:C as suggested by Girolami.
        - nitrogen2carbon : Atomic N:C ratio.  ``None`` defaults to 0.
        - mass_ratio_convert : If ``True`` convert ``molar_mass`` from a mass
          ratio to molar mass using
          ``particula.activity.ratio.from_molar_mass_ratio``.

    Returns:
        - Estimated density of the compound in kg m⁻³.

    Examples:
        ```py title="Single compound"
        from particula.particles import (
            get_organic_density_estimate,
        )

        # Succinic acid (M = 118.09, O:C = 1, H:C = 1.333)
        rho = get_organic_density_estimate(118.09, oxygen2carbon=1.0,
                                           hydrogen2carbon=1.333)
        print(round(rho, 0))  # 1560
        ```

    References:
    - G. S. Girolami, “A Simple ‘Back of the Envelope’ Method for Estimating
        the Densities and Molecular Volumes of Liquids and Solids,” *J. Chem.
        Educ.*, 71 (11), 962 (1994).  DOI:10.1021/ed071p962
    """
    if nitrogen2carbon is None:
        nitrogen2carbon = 0
    if hydrogen2carbon is None:
        hydrogen2carbon = 0.1
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
    density_g_per_cm3 = rho1 * (
        1.0
        + min(
            number_carbons * oxygen2carbon * 0.1
            + number_carbons * nitrogen2carbon * 0.1,
            0.3,
        )
    )
    return density_g_per_cm3 * 1_000.0  # kg m⁻³


def get_organic_density_array(
    molar_mass: Union[list[float], NDArray[np.float64]],
    oxygen2carbon: Union[list[float], NDArray[np.float64]],
    hydrogen2carbon: Optional[Union[list[float], NDArray[np.float64]]] = None,
    nitrogen2carbon: Optional[Union[list[float], NDArray[np.float64]]] = None,
    mass_ratio_convert: bool = False,
) -> NDArray[np.float64]:
    """Vectorised wrapper around ``get_organic_density_estimate``.

    Applies the Girolami density estimate element-wise to one-dimensional NumPy
    arrays or lists of molecular properties.

    Arguments:
        - molar_mass : Sequence of molar masses (or mass ratios if
          ``mass_ratio_convert`` is ``True``) in g mol⁻¹.
        - oxygen2carbon : Sequence of atomic O:C ratios.
        - hydrogen2carbon : Sequence of atomic H:C ratios (may be ``None`` or
          negative to trigger estimation inside the helper function).
        - nitrogen2carbon : Sequence of atomic N:C ratios (optional).
        - mass_ratio_convert : Propagate conversion flag to the scalar helper.

    Returns:
        - NumPy array of estimated densities in kg m⁻³ having the same shape as
          the input arrays.

    Examples:
        ```py title="Batch calculation"
        import numpy as np
        from particula.particles import (
            get_organic_density_array,
        )

        mm  = np.array([118.09, 204.23])
        oc  = np.array([1.0, 0.5])
        hc  = np.array([1.333, 1.714])

        rho = get_organic_density_array(mm, oc, hc)
        print(np.round(rho, 0))  # array([1560., 1210.])
        ```

    References:
    - G. S. Girolami, “A Simple ‘Back of the Envelope’ Method for Estimating
        the Densities and Molecular Volumes of Liquids and Solids,” *J. Chem.
        Educ.*, 71 (11), 962 (1994).  DOI:10.1021/ed071p962
    """
    mm = np.asarray(molar_mass, dtype=float)
    oc = np.asarray(oxygen2carbon, dtype=float)
    hc = (
        None
        if hydrogen2carbon is None
        else np.asarray(hydrogen2carbon, dtype=float)
    )
    nc = (
        None
        if nitrogen2carbon is None
        else np.asarray(nitrogen2carbon, dtype=float)
    )
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
