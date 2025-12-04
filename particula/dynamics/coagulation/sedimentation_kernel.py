"""Sedimentation kernel for aerosol particles.

Used for calculating collisions due to gravitational settling, where heavier
particles fall faster and may collide with slower ones. The kernel follows
Equation 13A.4 from Seinfeld and Pandis (2016), proportional to the difference
in settling velocities and the combined projected area of two particles.

Equation:
    - K(i, j) = (π / 4) × (Dᵢ + Dⱼ)² × |vᵢ - vⱼ| × Eᵢⱼ
      - Dᵢ, Dⱼ : diameters of particle i and j [m],
      - vᵢ, vⱼ : settling velocities [m/s],
      - Eᵢⱼ : collision efficiency (dimensionless).

References:
    - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry
      and physics (3rd ed.). John Wiley & Sons. Chapter 13, Equation 13A.4.
"""

import numpy as np
from numpy.typing import NDArray

from particula.particles.properties.settling_velocity import (
    get_particle_settling_velocity_via_system_state,
)
from particula.util.validate_inputs import validate_inputs


def calculate_collision_efficiency_function(
    radius1: float, radius2: float
) -> float:
    """Calculate the collision efficiency between two particles (placeholder).

    This function calculates the collision efficiency E for two particles
    of radii radius1 and radius2, which can depend on additional factors
    (e.g., fluid flow or electrostatic forces). Currently not implemented.

    Arguments:
        - radius1 : The radius of the first particle [m].
        - radius2 : The radius of the second particle [m].

    Returns:
        - Collision efficiency [dimensionless].

    Examples:
        ```py
        # Not implemented
        calculate_collision_efficiency_function(1e-7, 2e-7)
        # Raises NotImplementedError
        ```

    References:
        - Saffman, P. G., & Turner, J. S. (1956). On the collision of drops
          in turbulent clouds. Journal of Fluid Mechanics, 1(1), 16-30.
    """
    # Implement the actual collision efficiency calculation here
    raise NotImplementedError(
        "Collision efficiency calculation not implemented yet."
    )


def get_sedimentation_kernel_sp2016(
    particle_radius: NDArray[np.float64],
    settling_velocities: NDArray[np.float64],
    calculate_collision_efficiency: bool = True,
) -> NDArray[np.float64]:
    """Calculate sedimentation kernel for aerosol particles (Eq 13A.4).

    This function computes the coagulation kernel due to gravitational
    settling, where larger particles settle faster and overtake smaller
    ones. The kernel is based on the combined diameters, the settling
    velocity difference, and the collision efficiency.

    Equation:
    - K(i, j) = (π / 4) × (Dᵢ + Dⱼ)² × |vᵢ - vⱼ| × Eᵢⱼ
        - Dᵢ, Dⱼ : diameters of particle i and j [m],
        - vᵢ, vⱼ : settling velocities [m/s],
        - Eᵢⱼ : collision efficiency (dimensionless).

    Arguments:
        - particle_radius : Array of particle radii [m].
        - settling_velocities : Array of particle settling velocities [m/s].
        - calculate_collision_efficiency : Whether to calculate collision
          efficiency or use 1. Defaults to True.

    Returns:
        - Sedimentation kernel matrix [m³/s], shape (n, n).

    Examples:
        ```py title="Example"
        import numpy as np

        rads = np.array([1e-7, 2e-7])
        vels = np.array([5e-3, 1e-2])
        kernel = get_sedimentation_kernel_sp2016(rads, vels)
        print(kernel)
        ```

    References:
        - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry
          and physics (3rd ed.). John Wiley & Sons. Chapter 13, Equation 13A.4.
    """
    diameter_matrix = 2 * (
        particle_radius[:, np.newaxis] + particle_radius[np.newaxis, :]
    )
    velocity_diff_matrix = np.abs(
        settling_velocities[:, np.newaxis] - settling_velocities[np.newaxis, :]
    )

    if calculate_collision_efficiency:
        collision_efficiency_matrix = np.vectorize(
            calculate_collision_efficiency_function
        )(particle_radius[:, np.newaxis], particle_radius[np.newaxis, :])
    else:
        collision_efficiency_matrix = np.ones_like(diameter_matrix)

    return (
        np.pi
        / 4
        * diameter_matrix**2
        * velocity_diff_matrix
        * collision_efficiency_matrix
    )


@validate_inputs({"temperature": "positive", "pressure": "positive"})
def get_sedimentation_kernel_sp2016_via_system_state(
    particle_radius: NDArray[np.float64],
    particle_density: NDArray[np.float64],
    temperature: float,
    pressure: float,
    calculate_collision_efficiency: bool = True,
) -> NDArray[np.float64]:
    """Calculate the sedimentation kernel (Equation 13A.4) via system state.

    This function first derives settling velocities using the system state
    (particle radius, density, temperature, pressure), then calls
    get_sedimentation_kernel_sp2016.

    Arguments:
        - particle_radius : Array of particle radii [m].
        - particle_density : Array of particle densities [kg/m³].
        - temperature : Temperature [K].
        - pressure : Pressure [Pa].
        - calculate_collision_efficiency : Whether to calculate collision
          efficiency or use 1. Defaults to True.

    Returns:
        - Sedimentation kernel matrix [m³/s], shape (n, n).

    Examples:
        ```py title="Example"
        import numpy as np
        rads = np.array([1e-7, 2e-7])
        dens = np.array([1000, 1200])
        kernel = get_sedimentation_kernel_sp2016_via_system_state(
            particle_radius=rads,
            particle_density=dens,
            temperature=298,
            pressure=101325
        )
        print(kernel)
        ```

    References:
        - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry
          and physics (3rd ed.). John Wiley & Sons. Chapter 13, Equation 13A.4.
    """
    settling_velocities = get_particle_settling_velocity_via_system_state(
        particle_radius=particle_radius,
        particle_density=particle_density,
        temperature=temperature,
        pressure=pressure,
    )

    return get_sedimentation_kernel_sp2016(
        particle_radius=particle_radius,
        settling_velocities=np.asarray(settling_velocities, dtype=np.float64),
        calculate_collision_efficiency=calculate_collision_efficiency,
    )
