"""Calculate a fluids integral scale."""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {"fluid_rms_velocity": "positive", "turbulent_dissipation": "positive"}
)
def get_lagrangian_integral_time(
    fluid_rms_velocity: Union[float, NDArray[np.float64]],
    turbulent_dissipation: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Calculate the Lagrangian integral timescale.

    The Lagrangian integral timescale is a measure of the time it takes for
    a fluid particle to travel a distance equal to the integral length scale.

    - T_L = (u'²) / ε
        - T_L is Lagrangian integral timescale [s].
        - fluid_rms_velocity (u') is Fluid RMS fluctuation velocity [m/s].
        - turbulent_dissipation (ε) is Turbulent energy dissipation rate
            [m²/s³].

    Arguments:
        - fluid_rms_velocity : Fluid RMS fluctuation velocity [m/s].
        - turbulent_dissipation : Turbulent kinetic energy dissipation rate
            [m²/s³].

    Returns:
        - Lagrangian integral timescale [s].

    Examples:
        ``` py title="Example Usage"
        import particula as par
        par.gas.get_lagrangian_integral_time(0.3, 1e-4)
        # Output: 900.0
        ```

    References:
        - Townsend, A. A., "The Structure of Turbulent Shear Flow," 2nd ed.,
          Cambridge University Press, 1976. [Check this reference]
        - Wikipedia contributors, "Turbulence," Wikipedia.
    """
    return (fluid_rms_velocity**2) / turbulent_dissipation


@validate_inputs(
    {"fluid_rms_velocity": "positive", "turbulent_dissipation": "positive"}
)
def get_eulerian_integral_length(
    fluid_rms_velocity: Union[float, NDArray[np.float64]],
    turbulent_dissipation: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Calculate the Eulerian integral length scale.

    The Eulerian integral length scale is a measure of the size of the largest
    turbulent eddies in a fluid flow.

    - L_e = 0.5 × (u'³) / ε
        - L_e is Eulerian integral length scale [m].
        - fluid_rms_velocity (u') is Fluid RMS fluctuation velocity [m/s].
        - turbulent_dissipation (ε) is Turbulent energy dissipation rate
            [m²/s³].

    Arguments:
        - fluid_rms_velocity : Fluid RMS fluctuation velocity [m/s].
        - turbulent_dissipation : Turbulent kinetic energy dissipation rate
            [m²/s³].

    Returns:
        - Eulerian integral length scale [m].

    Examples:
        ``` py title="Example"
        import particula as par
        par.gas.get_eulerian_integral_length(0.3, 1e-4)
        # Output: 1350.0
        ```

    References:
        - Hinze, J. O., "Turbulence," McGraw-Hill, 1975. [Check this reference]
        - Wikipedia contributors, "Turbulence," Wikipedia.
    """
    return 0.5 * (fluid_rms_velocity**3) / turbulent_dissipation
