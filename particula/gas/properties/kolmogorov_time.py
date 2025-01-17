"""
Get the Kolmogorov time of a gas particle.

References:
    Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on the
    geometric collision rate of sedimenting droplets. Part 2. Theory and
    parameterization. New Journal of Physics, 10.
    https://doi.org/10.1088/1367-2630/10/7/075016
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {"kinematic_viscosity": "positive", "turbulent_dissipation": "positive"}
)
def get_kolmogorov_time(
    kinematic_viscosity: Union[float, NDArray[np.float64]],
    turbulent_dissipation: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the Kolmogorov time of a fluid.

    The Kolmogorov time is the smallest timescale in turbulence where viscous
    forces dominate. It is calculated as the square root of kinematic
    viscosity divided by the rate of dissipation of turbulent kinetic
    energy.

    Arguments:
    ----------
        - kinematic_viscosity : Kinematic viscosity of the fluid [m^2/s]
        - turbulent_dissipation : Rate of dissipation of turbulent kinetic
            energy [m^2/s^3]

    Returns:
    --------
        - Kolmogorov time [s]

    References:
    ----------
        - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2. Theory
        and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """
    return np.sqrt(kinematic_viscosity / turbulent_dissipation)
