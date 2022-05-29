""" coagulation according to the hardsphere approx.
"""

import numpy as np
from particula.util.input_handling import in_scalar


def dy2007_coag_less(
    diff_knu=None,
    cpr=None,
):
    """ dy2007 approx.
        Dimensionless particle--particle coagulation kernel.

        dy2007:
            https://aip.scitation.org/doi/10.1063/1.2713719
        cg2019: (keep gh2012 because using modified expression from it)
            https://www.tandfonline.com/doi/suppl/10.1080/02786826.2019.1614522
    """

    if diff_knu is None or cpr is None:
        raise ValueError(
            "Please provide explicit values: diff_knu, cpr")

    diff_knu = in_scalar(diff_knu)
    cpr = in_scalar(cpr)

    assert cpr.m >= 0  # gk2008 is only valid for positive cpr
    assert diff_knu > 0  # by definition

    cekl = 1 + cpr
    cecl = cpr / (1 - np.exp(-cpr)) if cpr.m > 0 else 1 * cpr.u

    return (4 * np.pi * diff_knu**2) / (
        np.sqrt(2*np.pi)*diff_knu*cekl *
        np.exp(-cpr/(1+diff_knu*cekl/cecl)) /
        (
            (1+diff_knu*cekl/cecl)**2 -
            (2+diff_knu*cekl/cecl)*diff_knu*cekl/cecl *
            np.exp(-cpr/(
                (1+diff_knu*cekl/cecl) *
                (2+diff_knu*cekl/cecl)
            ))
        ) +
        (1 - np.exp(-cpr/(1+diff_knu*cekl/cecl))) /
        (1 - np.exp(-cpr))
    )
