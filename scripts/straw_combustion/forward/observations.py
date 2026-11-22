"""Instrument cutoffs, densities and Cunningham diameter conversion."""

import numpy as np

SULFATE_MW = 0.174259
CHLORIDE_MW = 0.0745513
SULFATE_RHO = 2660.0
CHLORIDE_RHO = 1980.0
REPORTED_RHO = 2000.0
SMPS = (14.0, 800.0)
LPI_AERO = np.array([30.0, 12700.0])
MFP_NM = 66.0
GAUSS_X, GAUSS_W = np.polynomial.legendre.leggauss(56)


def _slip_diameter_squared(d):
    """Return d² Cc(d) in nm² using Kn = 2λ/d and the Davies coefficients."""
    return d * (d + MFP_NM * (2.514 + 0.800 * np.exp(-0.55 * d / MFP_NM)))


def _rescale_density(d, ratio):
    """Solve F(new_d)=rho_old/rho_new * F(d) by Newton iteration."""
    goal = _slip_diameter_squared(d) * ratio
    a = 0.55 / MFP_NM
    x = d * np.sqrt(ratio)
    for _ in range(7):
        derivative = 2 * x + MFP_NM * (
            2.514 + 0.8 * np.exp(-a * x) * (1 - a * x)
        )
        x = x - (_slip_diameter_squared(x) - goal) / derivative
    return x
