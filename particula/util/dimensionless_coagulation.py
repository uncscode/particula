""" A class with methods for dimensionless coagulation
"""

import numpy as np
from particula.util.diffusive_knudsen import diff_knu as dknu


class DimensionlessCoagulation:

    """ dimensionless coagulation
    """

    def __init__(self, **kwargs):
        """ Dimensionless particle--particle coagulation kernel.

            Attributes:
                diff_knu        (float) [dimensionless]

            Notes:
                The dimensionless coagulation kernel is defined as
                a function of the diffusive knudsen number; for more info,
                please see the documentation of the respective function:
                    - particula.util.diffusive_knudsen.diff_knu(**kwargs)
        """

        self.diff_knu = dknu(**kwargs)
        self.authors = kwargs.get("authors", "hardsphere")

    def hardsphere_coag(self):
        """ Dimensionless particle--particle coagulation kernel.
        """

        hsa_consts = [25.836, 11.211, 3.502, 7.211]
        diff_knu = self.diff_knu

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

    def coag(self):
        """ Return the dimensionless coagulation kernel.
        """

        if self.authors == "hardsphere":
            result = self.hardsphere_coag()
        else:
            raise ValueError(f"{self.authors} not recognized!")

        return result


def hardsphere_coag(**kwargs):
    """ Return the dimensionless coagulation kernel.

        The dimensionless coagulation kernel is defined as
        a function of the diffusive knudsen number; for more info,
        please see the documentation of the respective function:
            - particula.util.diffusive_knudsen.diff_knu(**kwargs)

        Examples:
        ```
        >>> from particula import u
        >>> from particula.util.dimensionless_coagulation import hardsphere_coag as hsc
        >>> # with only one radius
        >>> hsc(radius=1e-9)
        <Quantity(147.877572, 'dimensionless')>
        >>> # with two radii
        >>> hsc(radius=1e-9, other_radius=1e-8)
        <Quantity(18.4245966, 'dimensionless')>
        >>> # with two radii and charges
        >>> hsc(radius=1e-9, other_radius=1e-8, charge=1, other_charge=-1)
        <Quantity(22.0727435, 'dimensionless')>
    """

    return DimensionlessCoagulation(**kwargs).hardsphere_coag()


def coag(**kwargs):
    """ Return the dimensionless coagulation kernel.
    """

    return DimensionlessCoagulation(**kwargs).coag()
