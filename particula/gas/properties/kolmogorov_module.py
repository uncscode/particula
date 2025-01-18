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

    The Kolmogorov time scale represents the smallest timescale in turbulence
    where viscous forces dominate over inertial effects. This timescale
    characterizes the turnover time of the smallest turbulent
    eddies. It is given by:

        τ_K = (v / ε)^(1/2)

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


@validate_inputs(
    {"kinematic_viscosity": "positive", "turbulent_dissipation": "positive"}
)
def get_kolmogorov_length(
    kinematic_viscosity: Union[float, NDArray[np.float64]],
    turbulent_dissipation: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the Kolmogorov length scale.

    The Kolmogorov length scale represents the smallest eddies in a turbulent
    flow where viscosity dominates. It is defined as:

        η = (ν³ / ε)^(1/4)

    Arguments:
    ----------
        - kinematic_viscosity : Kinematic viscosity of the fluid [m^2/s]
        - turbulent_dissipation : Rate of dissipation of turbulent kinetic
            energy [m^2/s^3]

    Returns:
    --------
        - Kolmogorov length scale [m]

    References:
    ----------
        - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2. Theory
        and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """
    return np.sqrt(np.sqrt(kinematic_viscosity**3 / turbulent_dissipation))


@validate_inputs(
    {"kinematic_viscosity": "positive", "turbulent_dissipation": "positive"}
)
def get_kolmogorov_velocity(
    kinematic_viscosity: Union[float, NDArray[np.float64]],
    turbulent_dissipation: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the Kolmogorov velocity scale.

    The Kolmogorov velocity scale characterizes the smallest turbulent velocity
    fluctuations and is given by:

        v_k = (v ε)^(1/4)

    Arguments:
    ----------
        - kinematic_viscosity : Kinematic viscosity of the fluid [m^2/s]
        - turbulent_dissipation : Rate of dissipation of turbulent kinetic
            energy [m^2/s^3]

    Returns:
    --------
        - Kolmogorov velocity scale [m/s]

    References:
    -----------
        - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2. Theory
        and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """
    return (kinematic_viscosity * turbulent_dissipation) ** 0.25
