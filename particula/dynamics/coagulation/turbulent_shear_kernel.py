"""
Turbulent shear kernel for coagulation.

Seinfeld and Pandis (2016), Chapter 13, Equation 13A.2.
K(D1, D2) = (pi * e_k / 120 * v)**1/2 * (D1 + D2)**3

Where:
    - K(D1, D2) : Turbulent shear kernel for coagulation [m^3/s].
    - e_k : Turbulent kinetic energy [m^2/s^2].
    - v : Kinematic viscosity [m^2/s].
    - D1, D2 : Particle diameters [m].

Saffman, P. G., & Turner, J. S. (1956). On the collision of drops in turbulent
clouds. Journal of Fluid Mechanics, 1(1), 16-30.
https://doi.org/10.1017/S0022112056000020
"""

from numpy.typing import NDArray
import numpy as np

from particula.gas.properties.kinematic_viscosity import (
    get_kinematic_viscosity_via_system_state,
)


def get_turbulent_shear_kernel_st1956(
    particle_radius: NDArray[np.float64],
    turbulent_dissipation: float,
    kinematic_viscosity: float,
) -> NDArray[np.float64]:
    """
    Calculate the turbulent shear kernel for coagulation.

    Parameters:
        particle_radius : Array of particle radii [m].
        turbulent_dissipation : Turbulent kinetic energy [m^2/s^2].
        kinematic_viscosity : Kinematic viscosity [m^2/s].

    Returns:
        Turbulent shear kernel matrix for coagulation [m^3/s].

    References:
    - Equation 13A.2 : K(D1, D2) = (pi * e_k / 120 * v)**1/2 * (D1 + D2)**3
        - K(D1, D2) : Turbulent shear kernel for coagulation [m^3/s].
        - e_k : Turbulent kinetic energy [m^2/s^2].
        - v : Kinematic viscosity [m^2/s].
        - D1, D2 : Particle diameters [m].
    - Saffman, P. G., & Turner, J. S. (1956). On the collision of drops in
        turbulent clouds. Journal of Fluid Mechanics, 1(1), 16-30.
        https://doi.org/10.1017/S0022112056000020
    - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
        physics, Chapter 13, Equation 13A.2.
    """
    diameter_sum_matrix = (
        particle_radius[:, np.newaxis] + particle_radius[np.newaxis, :]
    ) * 2  # Convert from radius to diameter sum

    return (
        np.pi * turbulent_dissipation / (120 * kinematic_viscosity)
    ) ** 0.5 * diameter_sum_matrix**3


def get_turbulent_shear_kernel_st1956_via_system_state(
    particle_radius: NDArray[np.float64],
    turbulent_dissipation: float,
    temperature: float,
    fluid_density: float,
) -> NDArray[np.float64]:
    """
    Calculate the turbulent shear kernel for coagulation via system state.

    Arguments:
        particle_radius : Array of particle radii [m].
        turbulent_dissipation : Turbulent kinetic energy [m^2/s^2].
        temperature : Temperature of the system [K].
        fluid_density : Density of the fluid [kg/m^3].

    Returns:
        Turbulent shear kernel matrix for coagulation [m^3/s].

    References:
    - Saffman, P. G., & Turner, J. S. (1956). On the collision of drops in
        turbulent clouds. Journal of Fluid Mechanics, 1(1), 16-30.
        https://doi.org/10.1017/S0022112056000020
    """
    kinematic_viscosity = get_kinematic_viscosity_via_system_state(
        temperature=temperature, fluid_density=fluid_density
    )
    return get_turbulent_shear_kernel_st1956(
        particle_radius=particle_radius,
        turbulent_dissipation=turbulent_dissipation,
        kinematic_viscosity=kinematic_viscosity,
    )
