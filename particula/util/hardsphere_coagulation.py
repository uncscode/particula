""" coagulation according to the hardsphere approx.
"""

import numpy as np
from particula.util.input_handling import in_scalar


def hardsphere_coag_less(
        diff_knu=None):
    """ hardsphere approx.
        Dimensionless particle--particle coagulation kernel.
    """

    if diff_knu is None:
        raise ValueError("Please provide an explicit value for diff_knu")

    diff_knu = in_scalar(diff_knu)
    hsa_consts = [25.836, 11.211, 3.502, 7.211]

    upstairs = (
        (4 * np.pi * diff_knu**2) +
        (hsa_consts[0] * diff_knu**3) +
        ((8 * np.pi)**(1/2) * hsa_consts[1] * diff_knu**4)
    )

    downstairs = (
        1 +
        (hsa_consts[2] * diff_knu) +
        (hsa_consts[3] * diff_knu**2) +
        (hsa_consts[1] * diff_knu**3)
    )

    return upstairs / downstairs
