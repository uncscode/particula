""" A class with methods for dimensionless coagulation
"""

from particula.util.input_handling import in_scalar
from particula.util.diffusive_knudsen import DiffusiveKnudsen as DKn
from particula.util.diffusive_knudsen import celimits
# from particula.util.diffusive_knudsen import diff_knu as dknu
from particula.util.diffusive_knudsen import red_frifac, red_mass, rxr
from particula.util.approx_coagulation import approx_coag_less


class DimensionlessCoagulation(DKn):

    """ dimensionless coagulation
    """

    def __init__(
        self,
        dkn_val=None,
        coag_approx="hardsphere",
        **kwargs
    ):
        """ Dimensionless particle--particle coagulation kernel.

            Attributes:
                diff_knu        (float) [dimensionless]

            Notes:
                The dimensionless coagulation kernel is defined as
                a function of the diffusive knudsen number; for more info,
                please see the documentation of the respective function:
                    - particula.util.diffusive_knudsen.diff_knu(**kwargs)
        """
        super().__init__(**kwargs)

        self.diff_knu = DKn(**kwargs).get_diff_knu() if dkn_val is None \
            else in_scalar(dkn_val)

        self.coag_approx = coag_approx

        self.kwargs = kwargs

    def coag_less(self):
        """ Return the dimensionless coagulation kernel.
        """

        impls = ["hardsphere", "gh2012", "cg2019", "dy2007", "gk2008"]

        if self.coag_approx not in impls:
            raise ValueError(f"{self.coag_approx} not recognized!")

        return approx_coag_less(
            diff_knu=self.diff_knu,
            cpr=self.coulomb_potential_ratio(),
            approx=self.coag_approx
        )

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
        >>> less_coag(
        ... radius=1e-9, other_radius=1e-8, charge=1, other_charge=-1
        ... )
        <Quantity(22.0727435, 'dimensionless')>
    """

    return DimensionlessCoagulation(**kwargs).coag_less()


def full_coag(**kwargs):
    """ Return the dimensioned coagulation kernel
    """

    return DimensionlessCoagulation(**kwargs).coag_full()
