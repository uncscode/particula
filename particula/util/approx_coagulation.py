""" dimensionless coagulation according to several approx.
"""

import numpy as np
from particula.util.input_handling import in_scalar


def approx_coag_less(  # pylint: disable=too-many-locals
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

        todolater:
        - quick fixes for corner cases
        - examine better solutions?
    """

    if diff_knu is None:
        raise ValueError(
            "Please provide explicit value for diff_knu")
    if approx != "hardsphere":
        if cpr is None:
            raise ValueError("Please provide explicit value for cpr")
        cpr = in_scalar(cpr)
        cpr = cpr + 1e-16 if cpr.m == 0 else cpr  # avoid division by zero

        cekl = 1 + cpr
        cecl = 1 * cpr.u if cpr.m == 0 else cpr / (1 - np.exp(-cpr))

    diff_knu = in_scalar(diff_knu)

    coag_clim = 4 * np.pi * diff_knu**2

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

    hscoag = upstairs / downstairs

    if approx == "hardsphere":

        return hscoag

    if approx == "gh2012":

        min_fxn = np.min([diff_knu.m, 3*diff_knu.m/(2*cpr.m)])
        result = (4 * np.pi * diff_knu**2) / (
            1 + 1.598 * min_fxn**1.1709
        ) if cpr.m > 0.5 and min_fxn < 2.5 else hscoag

        return result

    if approx == "gk2008":

        result = coag_clim * (
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

        return (
            result.m * (cpr.m >= 0) + hscoag.m * (cpr.m < 0)
        ) * result.u

    if approx == "dy2007":

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

    if approx == "cg2019":
        # fix later: 1e-12 is arbitrary, pure numerics issue here
        diff_knu = diff_knu if diff_knu.m > 1e-12 else 1e-12

        tricky_corr = 1 if cpr.m <= 0 else (
            4.528*np.exp(-1.088*cpr)) + (.7091*np.log(1+1.527*cpr))

        tricky_corr_again = 0 if cpr.m <= 0 else (
            11.36*(cpr**0.272) - 10.33)

        corr = [
            2.5,
            tricky_corr,
            tricky_corr_again,
            -0.003533*cpr + 0.05971
        ]

        corr_mu = (corr[2]/corr[0])*(
            (1+corr[3]*(np.log(diff_knu)-corr[1])/corr[0])**(
                -1/corr[3]-1) *
            np.exp(-1*(1+corr[3]*(np.log(diff_knu)-corr[1])/corr[0])**(
                -1/corr[3]))
            )

        return (
            hscoag.m * (cpr.m <= 0) + np.exp(corr_mu.m) * hscoag.m * (cpr > 0)
        ) * hscoag.u

    return None
