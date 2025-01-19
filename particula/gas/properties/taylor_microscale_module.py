"""
Taylor microscale module, for both the Lagrangian Taylor microscale time and
the Taylor microscale.
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "kolmogorov_time": "positive",
        "re_lambda": "positive",
        "accel_variance": "positive",
    }
)
def get_lagrangian_taylor_microscale_time(
    kolmogorov_time: Union[float, NDArray[np.float64]],
    re_lambda: Union[float, NDArray[np.float64]],
    accel_variance: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the Lagrangian Taylor microscale time.

    The Lagrangian Taylor microscale time (τ_T) represents the characteristic
    time for the decay of turbulent velocity correlations. It provides insight
    into the memory of turbulent fluid elements. It is given by:

        τ_T = τ_k * (2 R_λ / (15^(1/2) a_o))^(1/2)

    where:
        - τ_T : Lagrangian Taylor microscale time [s]
        - τ_k (kolmogorov_time) : Kolmogorov time scale [s]
        - R_λ (re_lambda) : Taylor-microscale Reynolds number [-]
        - a_o (ao) : Normalized acceleration variance in isotropic turbulence
            [-]

    Arguments:
    ----------
        - kolmogorov_time : Kolmogorov time scale [s]
        - re_lambda : Taylor-microscale Reynolds number [-]
        - accel_variance : Normalized acceleration variance in isotropic
            turbulence [-]

    Returns:
    --------
        - Lagrangian Taylor microscale time [s]
    """
    return kolmogorov_time * np.sqrt(
        (2 * re_lambda) / (15**0.5 * accel_variance)
    )


@validate_inputs(
    {
        "rms_velocity": "positive",
        "kinematic_viscosity": "positive",
        "turbulent_dissipation": "positive",
    }
)
def get_taylor_microscale(
    rms_velocity: Union[float, NDArray[np.float64]],
    kinematic_viscosity: Union[float, NDArray[np.float64]],
    turbulent_dissipation: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the Taylor microscale.

    The Taylor microscale (λ) represents an intermediate length scale in
    turbulence, linking the dissipative and energy-containing ranges of
    turbulence. It characterizes the smoothness of velocity fluctuations
    in turbulent flows. It is given by:

        λ = u' * (15 ν² / ε)^(1/2)

    where:
        - λ : Taylor microscale [m]
        - u' (rms_velocity) : Fluid RMS fluctuation velocity [m/s]
        - v (kinematic_viscosity) : Kinematic viscosity of the fluid [m²/s]
        - ε (turbulent_dissipation) : Turbulent kinetic energy dissipation
            rate [m²/s³]

    Arguments:
    ----------
        - rms_velocity : Fluid RMS fluctuation velocity [m/s]
        - kinematic_viscosity : Kinematic viscosity of the fluid [m²/s]
        - turbulent_dissipation : Turbulent kinetic energy dissipation rate
            [m²/s³]

    Returns:
    --------
        - Taylor microscale [m]
    """
    return rms_velocity * np.sqrt(
        (15 * kinematic_viscosity**2) / turbulent_dissipation
    )
