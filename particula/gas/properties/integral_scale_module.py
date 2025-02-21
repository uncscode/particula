"""
Calculate a fluids integral scale.
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {"fluid_rms_velocity": "positive", "turbulent_dissipation": "positive"}
)
def get_lagrangian_integral_time(
    fluid_rms_velocity: Union[float, NDArray[np.float64]],
    turbulent_dissipation: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    """
    Examples:
        ``` py title="Example Usage"
        T_L = get_lagrangian_integral_time(0.5, 0.1)
        # Output (example): 2.5
        ```

        ``` py title="Example Usage with Array Input"
        T_L = get_lagrangian_integral_time(
            np.array([0.5, 0.7]),
            np.array([0.1, 0.09])
        )
        # Output (example): array([2.5, 5.4444])
        ```

    References:
        - G. K. Batchelor, "An Introduction to Fluid Dynamics,"  
          Cambridge University Press, 1967.
    Calculate the Lagrangian integral timescale.

    The Lagrangian integral scale (T_L) characterizes the timescale of large
    eddies in turbulence. This represents the correlation time of fluid
    elements in turbulent motion. It is given by:

        T_L = u'^2 / ε

    where:
        - T_L : Lagrangian integral timescale [s]
        - u' (fluid_rms_velocity) : Fluid RMS fluctuation velocity [m/s]
        - ε (turbulent_dissipation) : Turbulent energy dissipation rate [m²/s³]

    Arguments:
        - fluid_rms_velocity : Fluid RMS fluctuation velocity [m/s]
        - turbulent_dissipation : Turbulent kinetic energy dissipation rate
            [m²/s³]

    Returns:
        - Lagrangian integral timescale [s]
    Examples:
        ``` py title="Example Usage"
        L_e = get_eulerian_integral_length(0.35, 0.1)
        # Output (example): 0.021437
        ```

        ``` py title="Example Usage with Array Input"
        L_e = get_eulerian_integral_length(
            np.array([0.35, 0.4]),
            np.array([0.1, 0.12])
        )
        # Output (example): array([0.021437, 0.026667])
        ```

    References:
        - S. B. Pope, "Turbulent Flows," Cambridge University Press, 2000.
    return (fluid_rms_velocity**2) / turbulent_dissipation


@validate_inputs(
    {"fluid_rms_velocity": "positive", "turbulent_dissipation": "positive"}
)
def get_eulerian_integral_length(
    fluid_rms_velocity: Union[float, NDArray[np.float64]],
    turbulent_dissipation: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the Eulerian integral length scale.

    The Eulerian integral length scale (L_e) represents the characteristic
    size of the largest turbulent eddies and provides insight into the spatial
    correlation of turbulence structures. It is given by:

        L_e = 0.5 * u'^3 / ε

    where:
        - L_e : Eulerian integral length scale [m]
        - u' (fluid_rms_velocity) : Fluid RMS fluctuation velocity [m/s]
        - ε (turbulent_dissipation) : Turbulent energy dissipation rate [m²/s³]

    Arguments:
    ----------
        - fluid_rms_velocity : Fluid RMS fluctuation velocity [m/s]
        - turbulent_dissipation : Turbulent kinetic energy dissipation rate
            [m²/s³]

    Returns:
    --------
        - Eulerian integral length scale [m]
    """
    return 0.5 * (fluid_rms_velocity**3) / turbulent_dissipation
