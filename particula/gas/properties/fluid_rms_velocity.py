"""Fluid RMS fluctuation velocity calculation module."""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.gas.properties.kolmogorov_module import get_kolmogorov_velocity
from particula.util.validate_inputs import validate_inputs


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
    """Calculate the fluid RMS fluctuation velocity.

    The fluid root-mean-square (RMS) velocity fluctuation quantifies
    turbulence intensity in a fluid flow. It is calculated as:

    - u' = (R_λ^(1/2) v_K) / 15^(1/4)
        - u' is Fluid RMS fluctuation velocity [m/s]
        - R_λ (re_lambda) is Taylor-microscale Reynolds number [-]
        - v_K is Kolmogorov velocity scale, computed as v_K = ( ε)^(1/4) [m/s]
        - v (kinematic_viscosity) is Kinematic viscosity of the fluid [m²/s]
        - ε (turbulent_dissipation) is Turbulent energy dissipation rate [m²/s³]


    Arguments:
        - re_lambda : Taylor-microscale Reynolds number [-]
        - kinematic_viscosity : Kinematic viscosity of the fluid [m²/s]
        - turbulent_dissipation : Rate of dissipation of turbulent kinetic
            energy [m²/s³]

    Returns:
        - Fluid RMS fluctuation velocity [m/s]

    Examples:
        ``` py title="Example Usage"
        velocity = get_fluid_rms_velocity(500, 1.5e-5, 0.1)
        # Output (example): 0.35
        ```

        ``` py title="Example Usage with Array Input"
        velocity = get_fluid_rms_velocity(
            np.array([500, 600]),
            np.array([1.5e-5, 1.7e-5]),
            np.array([0.1, 0.12])
        )
        # Output (example): array([0.35, 0.41])
        ```

    References:
        - H. Tennekes and J. L. Lumley, "A First Course in Turbulence,"
          MIT Press, 1972. [check this]
    """
    kolmogorov_velocity = get_kolmogorov_velocity(
        kinematic_viscosity, turbulent_dissipation
    )
    return (re_lambda**0.5 * kolmogorov_velocity) / (15**0.25)
