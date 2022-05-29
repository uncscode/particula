""" coagulation according to the hardsphere approx.
"""

import numpy as np
from particula.util.input_handling import in_scalar


def gk2008_coag_less(
        diff_knu=None,
        cpr=None,
    ):
    """ gk2008 approx.
        Dimensionless particle--particle coagulation kernel.
    """

    if diff_knu is None or cpr is None:
        raise ValueError(
            "Please provide an explicit value for diff_knu, cpr, cekl, and cecl")

    diff_knu = in_scalar(diff_knu)
    cpr = in_scalar(cpr)

    assert cpr.m >= 0  # gk2008 is only valid for positive cpr
    assert diff_knu > 0  # by definition

    cekl = 1 + cpr
    cecl = cpr / (1 - np.exp(-cpr)) if cpr.m > 0 else 1 * cpr.u

    return (4 * np.pi * diff_knu**2) * (
        1 -
        (1 + (np.sqrt(np.pi)*cecl*cpr*1.22) / (
            2*cekl*diff_knu)) *
        np.exp(-(np.sqrt(np.pi)*cecl*cpr*1.22) / (
            2*cekl*diff_knu))
    ) + (np.sqrt(8*np.pi)*diff_knu) * (
        0 +
        (1 + (2*np.sqrt(np.pi)*(1.22**3)*cecl*(cpr**3)) / (
            9*(cekl**2)*diff_knu)) *
        np.exp(-(np.sqrt(np.pi)*cecl*cpr*1.22) / (
            2*cekl*diff_knu))
    )
