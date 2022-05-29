""" dimensionless coagulation according to several approx.
"""

import numpy as np
from particula.util.input_handling import in_scalar


def approx_coag_less(
    diff_knu=None,
    cpr=None,
    approx="hardsphere"
):
    """ dy2007 approx.
        Dimensionless particle--particle coagulation kernel.

        gh2012:
        https://journals.aps.org/pre/abstract/10.1103/PhysRevE.85.026410

        gk2008:
        https://journals.aps.org/pre/abstract/10.1103/PhysRevE.78.046402

        dy2007:
        https://aip.scitation.org/doi/10.1063/1.2713719

        cg2019:
        https://www.tandfonline.com/doi/suppl/10.1080/02786826.2019.1614522
    """

    if diff_knu is None:
        raise ValueError(
            "Please provide explicit value for diff_knu")
    if approx != "hardsphere":
        if cpr is None:
            raise ValueError("Please provide explicit value for cpr")
        cpr = in_scalar(cpr)
        cpr = cpr + 1e-32 if cpr.m == 0 else cpr  # avoid division by zero

        cekl = 1 + cpr
        cecl = 1 * cpr.u if cpr.m == 0 else cpr / (1 - np.exp(-cpr))

    diff_knu = in_scalar(diff_knu)

    coag_clim = 4 * np.pi * diff_knu**2

    if approx == "hardsphere":

        hsa_consts = [25.836, 11.211, 3.502, 7.211]

        upstairs = (
            coag_clim +
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

    if approx == "gh2012":

        return (4 * np.pi * diff_knu**2) / (
            1 + 1.598 * np.min(
                [diff_knu.m, 3*diff_knu.m/(2*cpr.m)]
            )**1.1709
        )

    if approx == "gk2008":

        assert cpr.m >= 0
        assert diff_knu > 0

        return coag_clim * (
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

    if approx == "dy2007":

        assert diff_knu > 0

        return coag_clim / (
            np.sqrt(2*np.pi)*diff_knu*cekl *
            np.exp(-1*cpr/(1+diff_knu*cekl/cecl)) /
            (
                (1+diff_knu*cekl/cecl)**2 -
                (2+diff_knu*cekl/cecl)*diff_knu*cekl/cecl *
                np.exp(-1*cpr/(
                    (1+diff_knu*cekl/cecl) *
                    (2+diff_knu*cekl/cecl)
                ))
            ) +
            (1 - np.exp(-1*cpr/(1+diff_knu*cekl/cecl))) /
            (1 - np.exp(-1*cpr))
        )

    return None
