""" coagulation according to the hardsphere approx.
"""

import numpy as np
from particula.util.input_handling import in_scalar


def gh2012_coag_less(
    diff_knu=None,
    cpr=None,
):
    """ gh2012 approx.
        Dimensionless particle--particle coagulation kernel.

        gh2012:
            https://journals.aps.org/pre/abstract/10.1103/PhysRevE.85.026410

    """

    if diff_knu is None or cpr is None:
        raise ValueError(
            "Please provide an explicit value for diff_knu and cpr")

    diff_knu = in_scalar(diff_knu)
    cpr = in_scalar(cpr)

    cpr = cpr + 1e-32 if cpr.m == 0  # avoid division by zero
    return (4 * np.pi * diff_knu**2) / (
        1 + 1.598 * np.min([diff_knu.m, 3*diff_knu.m/(2*cpr.m)])**1.1709
    )
