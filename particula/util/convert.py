"""conversion functions common for aerosol processing"""

# pylint: disable=duplicate-code

import warnings

import numpy as np


def convert_sizer_dn(
    diameter: np.ndarray, dn_dlogdp: np.ndarray, inverse: bool = False
) -> np.ndarray:
    """
    Converts the sizer data from dn/dlogdp to d_num.

    The bin width is defined as the  difference between the upper and lower
    diameter limits of each bin. This function calculates the bin widths
    based on the input diameter array. Assumes a log10 scale for dp edges.

    Args:
        diameter (np.ndarray): Array of particle diameters.
        dn_dlogdp (np.ndarray): Array of number concentration of particles per
        unit logarithmic diameter.
        inverse (bool): If True, converts from d_num to dn/dlogdp.

    Returns:
        np.ndarray: Array of number concentration of particles
        per unit diameter.

    References:
        Eq: dN/dlogD_p = dN/( log(D_{p-upper}) - log(D_{p-lower}) )
        https://tsi.com/getmedia/1621329b-f410-4dce-992b-e21e1584481a/
        PR-001-RevA_Aerosol-Statistics-AppNote?ext=.pdf
    """
    warnings.warn(
        "convert_sizer_dn() is deprecated and will be removed"
        " in a future release. "
        "Use size_distribution_convert.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    assert len(diameter) > 0, "Inputs must be non-empty arrays."
    # Compute the bin widths
    delta = np.zeros_like(diameter)
    delta[:-1] = np.diff(diameter)
    delta[-1] = delta[-2] ** 2 / delta[-3]

    # Compute the lower and upper bin edges
    lower = diameter - delta / 2
    upper = diameter + delta / 2

    # if dn_dlogdp.ndim == 2:
    #     # expand diameter by one dimension so it can be broadcast
    #     lower = np.expand_dims(lower, axis=1)
    #     upper = np.expand_dims(upper, axis=1)
    if inverse:
        # Convert from dn to dn/dlogdp
        return dn_dlogdp / np.log10(upper / lower)

    return dn_dlogdp * np.log10(upper / lower)


def distribution_convert_pdf_pms(
    x_array: np.ndarray, distribution: np.ndarray, to_pdf: bool = True
) -> np.ndarray:
    """
    Convert between a probability density function (PDF) and a probability
    mass spectrum (PMS) based on the specified direction.

    Args:
        x_array : An array of radii corresponding to the bins of the
            distribution, shape (m).
        distribution : The concentration values of the distribution
            (either PDF or PMS) at the given radii. Supports broadcasting
            across x_array (n,m).
        to_PDF : Direction of conversion. If True, converts PMS to PDF.
            If False, converts PDF to PMS.

    Returns:
        converted_distribution : The converted distribution array
            (either PDF or PMS).
    """
    warnings.warn(
        "distribution_convert_pdf_pms() is deprecated and will be removed"
        " in a future release. "
        "Use convert_size_distribution.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Calculate the differences between consecutive x_array values for bin
    # widths.
    delta_x_array = np.empty_like(x_array)
    delta_x_array[:-1] = np.diff(x_array)
    # For the last bin, extrapolate the width assuming constant growth
    # rate from the last two bins.
    delta_x_array[-1] = delta_x_array[-2] ** 2 / delta_x_array[-3]

    # Converting PMS to PDF by dividing the PMS values by the bin widths. or
    # Converting PDF to PMS by multiplying the PDF values by the bin widths.
    return (
        distribution / delta_x_array
        if to_pdf
        else distribution * delta_x_array
    )
