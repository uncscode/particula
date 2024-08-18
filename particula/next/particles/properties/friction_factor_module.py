"""Module for friction factor of a particle in a fluid.

Zhang, C., Thajudeen, T., Larriba, C., Schwartzentruber, T. E., &
Hogan, C. J. (2012). Determination of the Scalar Friction Factor for
Nonspherical Particles and Aggregates Across the Entire Knudsen Number Range
by Direct Simulation Monte Carlo (DSMC). Aerosol Science and Technology,
46(10), 1065-1078. https://doi.org/10.1080/02786826.2012.690543
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np


def friction_factor(
    radius: Union[float, NDArray[np.float64]],
    dynamic_viscosity: float,
    slip_correction: Union[float, NDArray[np.float64]],
):
    """
    Returns a particle's friction factor. Property of the particle's size and
    surrounding medium. Multiplying the friction factor by the fluid velocity
    yields the drag force on the particle.

    Args:
    -----
    - radius : The radius of the particle [m].
    - dynamic_viscosity : The dynamic viscosity of the fluid [Pa s].
    - slip_correction : The slip correction factor [unitless].

    Returns:
    --------
    The friction factor of the particle [N s/m].

    References:
    ----------
    It is best thought of as an inverse of mobility or the ratio between
    thermal energy and diffusion coefficient. The modified Stoke's diffusion
    coefficient is defined as
    kT / (6 * np.pi * dyn_vis_air * radius / slip_corr)
    and thus the friction factor can be defined as
    (6 * np.pi * dyn_vis_air * radius / slip_corr).

    In the continuum limit (Kn -> 0; Cc -> 1):
    6 * np.pi * dyn_vis_air * radius

    In the kinetic limit (Kn -> inf):
    8.39 * (dyn_vis_air/mfp_air) * const * radius**2

    Zhang, C., Thajudeen, T., Larriba, C., Schwartzentruber, T. E., &
    Hogan, C. J. (2012). Determination of the Scalar Friction Factor for
    Nonspherical Particles and Aggregates Across the Entire Knudsen Number
    Range by Direct Simulation Monte Carlo (DSMC). Aerosol Science and
    Technology, 46(10), 1065-1078. https://doi.org/10.1080/02786826.2012.690543
    """

    return 6 * np.pi * dynamic_viscosity * radius / slip_correction
