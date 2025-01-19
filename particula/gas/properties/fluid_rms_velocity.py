"""
Fluid RMS fluctuation velocity calculation module.
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.util.validate_inputs import validate_inputs
from particula.gas.properties.kolmogorov_module import get_kolmogorov_velocity


@validate_inputs(
    {
        "re_lambda": "positive",
        "kinematic_viscosity": "positive",
        "turbulent_dissipation": "positive",
    }
)
def get_fluid_rms_velocity(
    re_lambda: Union[float, NDArray[np.float64]],
    kinematic_viscosity: Union[float, NDArray[np.float64]],
    turbulent_dissipation: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the fluid RMS fluctuation velocity.

    The fluid root-mean-square (RMS) velocity fluctuation quantifies
    turbulence intensity in a fluid flow. It is calculated as:

        u' = (R_λ^(1/2) v_K) / 15^(1/4)

    where:
        - u' : Fluid RMS fluctuation velocity [m/s]
        - R_λ (re_lambda) : Taylor-microscale Reynolds number [-]
        - v_K : Kolmogorov velocity scale, computed as v_K = ( ε)^(1/4) [m/s]
        - v (kinematic_viscosity) : Kinematic viscosity of the fluid [m²/s]
        - ε (turbulent_dissipation) : Turbulent energy dissipation rate [m²/s³]


    Arguments:
    ----------
        - re_lambda : Taylor-microscale Reynolds number [-]
        - kinematic_viscosity : Kinematic viscosity of the fluid [m²/s]
        - turbulent_dissipation : Rate of dissipation of turbulent kinetic
            energy [m²/s³]

    Returns:
    --------
        - Fluid RMS fluctuation velocity [m/s]
    """
    kolmogorov_velocity = get_kolmogorov_velocity(
        kinematic_viscosity, turbulent_dissipation
    )
    return (re_lambda**0.5 * kolmogorov_velocity) / (15**0.25)
