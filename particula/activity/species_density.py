"""Species density estimation functions."""

import numpy as np
from particula.activity.ratio import from_molar_mass_ratio

MASS_C = 12.01  # the molar masses in [g/mol]
MASS_O = 16.0
MASS_H = 1.008
MASS_N = 14.0067


def organic_density_estimate(
    molar_mass,
    oxygen2carbon,
    hydrogen2carbon=None,
    nitrogen2carbon=None,
    mass_ratio_convert=False,
):
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

    Args:
        molar_mass(float): Molar mass.
        oxygen2carbon (float): O:C ratio.
        hydrogen2carbon (float): H:C ratio. If unknown, provide a negative
            value.
        nitrogen2carbon (float, optional): N:C ratio. Defaults to None.

    Returns:
        densityEst (float): Estimated density in g/cm^3.
    """
    if nitrogen2carbon is None:
        nitrogen2carbon = oxygen2carbon * 0
    if hydrogen2carbon is None:
        hydrogen2carbon = oxygen2carbon * 0
    if mass_ratio_convert:
        molar_mass = from_molar_mass_ratio(molar_mass)

    # 1) Estimate the hydrogen2carbon value if not provided from input
    # Assuming an aliphatic compound with hydrogen2carbon = 2.0 in the absence
    # of functional groups, then correct for oxygen content assuming a linear
    # -1 slope (Van Krevelen diagram for typical SOA)
    hydrogen2carbon_est = (
        2.0 - oxygen2carbon if hydrogen2carbon < 0.1 else hydrogen2carbon
    )

    # 2) Compute the approximate number of carbon atoms per organic molecule
    number_carbons = molar_mass / (
        MASS_C
        + hydrogen2carbon_est * MASS_H
        + oxygen2carbon * MASS_O
        + nitrogen2carbon * MASS_N
    )

    # 3) Compute density estimate based on method by Girolami (1994)
    # Here no correction is applied for rings and aromatic compounds
    # (due to limited info at input)
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

    # the returned denisty is in [g/cm^3]; and scaled assuming that most
    # that most of the oxygen atoms are able to make H-bonds
    # (donor or acceptor)
    return rho1 * (
        1.0
        + min(
            number_carbons * oxygen2carbon * 0.1
            + number_carbons * nitrogen2carbon * 0.1,
            0.3,
        )
    )


def organic_array(
    molar_mass,
    oxygen2carbon,
    hydrogen2carbon=None,
    nitrogen2carbon=None,
    mass_ratio_convert=False,
):
    # pylint: disable=too-many-positional-arguments, too-many-arguments
    """Get densities for an array."""
    density = np.empty([len(molar_mass), 1], dtype=float)
    for i, molar in enumerate(molar_mass):
        hydrogen2carbon_run = (
            None if hydrogen2carbon is None else hydrogen2carbon[i]
        )
        nitrogen2carbon_run = (
            None if nitrogen2carbon is None else nitrogen2carbon[i]
        )
        density[i] = organic_density_estimate(
            molar_mass=molar,
            oxygen2carbon=oxygen2carbon[i],
            hydrogen2carbon=hydrogen2carbon_run,
            nitrogen2carbon=nitrogen2carbon_run,
            mass_ratio_convert=mass_ratio_convert,
        )
    return density
