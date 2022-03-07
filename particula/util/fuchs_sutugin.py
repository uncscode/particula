""" the Fuchs-Sutugin model transition regime correction
"""

import numpy as np
from particula.util.knudsen_number import knu
from particula.util.input_handling import in_scalar


def fsc(**kwargs):
    """ Returns the Fuchs-Sutugin model transition regime correction.

        Parameters:
            knu     (float)  [ ] (default: util)
            alpha   (float)  [ ] (default: 1)

        Returns:
                (float)  [ ]

        Notes:
            knu can be calculated using knu(**kwargs);
            refer to particula.util.knudsen_number.knu for more info.
    """

    knu_val = kwargs.get("knu", knu(**kwargs))
    alpha = kwargs.get("alpha", 1.0)

    knu_val = in_scalar(knu_val)
    alpha = np.transpose([in_scalar(alpha).m])*in_scalar(alpha).u

    return np.transpose(
        knu_val * alpha * (1 + knu_val) /
        (knu_val**2 + knu_val + 0.283 * knu_val * alpha + 0.75 * alpha)
    )
