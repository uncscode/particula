""" A class with methods for dimensionless coagulation
"""

import numpy as np
from particula.util.diffusive_knudsen import diff_knu as dknu, red_mass, red_frifac, celimits, rxr


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
        self.kwargs = kwargs

    def hardsphere_coag_less(self):
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

    def coag_less(self):
        """ Return the dimensionless coagulation kernel.
        """

        if self.authors == "hardsphere":
            result = self.hardsphere_coag_less()
        else:
            raise ValueError(f"{self.authors} not recognized!")

        return result

    def coag_full(self):
        """ Retrun the dimensioned coagulation kernel
        """

        coag = self.coag_less()
        redff = red_frifac(**self.kwargs)
        redm = red_mass(**self.kwargs)
        cekl, cecl = celimits(**self.kwargs)
        xrxr = rxr(**self.kwargs)

        return (
            coag * redff * xrxr**3 * cekl**2 / (redm * cecl)
        )


def hsdl_coag_less(**kwargs):
    """ Return the dimensionless coagulation kernel.

        The dimensionless coagulation kernel is defined as
        a function of the diffusive knudsen number; for more info,
        please see the documentation of the respective function:
            - particula.util.diffusive_knudsen.diff_knu(**kwargs)

        Examples:
        ```
        >>> from particula import u
        >>> from particula.util.dimensionless_coagulation import hsdl_coag
        >>> # with only one radius
        >>> hsdl_coag(radius=1e-9)
        <Quantity(147.877572, 'dimensionless')>
        >>> # with two radii
        >>> hsdl_coag(radius=1e-9, other_radius=1e-8)
        <Quantity(18.4245966, 'dimensionless')>
        >>> # with two radii and charges
        >>> hsdl_coag(radius=1e-9, other_radius=1e-8, charge=1, other_charge=-1)
        <Quantity(22.0727435, 'dimensionless')>
    """

    return DimensionlessCoagulation(**kwargs).hardsphere_coag_less()


def less_coag(**kwargs):
    """ Return the dimensionless coagulation kernel.

        The dimensionless coagulation kernel is defined as
        a function of the diffusive knudsen number; for more info,
        please see the documentation of the respective function:
            - particula.util.diffusive_knudsen.diff_knu(**kwargs)

        Examples:
        ```
        >>> from particula import u
        >>> from particula.util.dimensionless_coagulation import less_coag
        >>> # only for hardsphere coagulation for now
        >>> # with only one radius
        >>> less_coag(radius=1e-9)
        <Quantity(147.877572, 'dimensionless')>
        >>> # with two radii
        >>> less_coag(radius=1e-9, other_radius=1e-8)
        <Quantity(18.4245966, 'dimensionless')>
        >>> # with two radii and charges
        >>> less_coag(radius=1e-9, other_radius=1e-8, charge=1, other_charge=-1)
        <Quantity(22.0727435, 'dimensionless')>
    """

    return DimensionlessCoagulation(**kwargs).coag_less()


def full_coag(**kwargs):
    """ Return the dimensioned coagulation kernel
    """

    return DimensionlessCoagulation(**kwargs).coag_full()
