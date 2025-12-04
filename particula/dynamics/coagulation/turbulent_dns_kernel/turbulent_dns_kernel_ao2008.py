"""Calculate the geometric collision kernel Γ₁₂ (or K₁₂) based on turbulent
DNS simulations.

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
    the geometric collision rate of sedimenting droplets. Part 2.
    Theory and parameterization. New Journal of Physics, 10.
    https://doi.org/10.1088/1367-2630/10/7/075016
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula import gas, particles
from particula.util.validate_inputs import validate_inputs

from .g12_radial_distribution_ao2008 import get_g12_radial_distribution_ao2008
from .radial_velocity_module import get_radial_relative_velocity_dz2002
from .sigma_relative_velocity_ao2008 import get_relative_velocity_variance


@validate_inputs(
    {
        "particle_radius": "positive",
        "velocity_dispersion": "positive",
        "particle_inertia_time": "positive",
    }
)
def get_turbulent_dns_kernel_ao2008(
    particle_radius: Union[float, NDArray[np.float64]],
    velocity_dispersion: Union[float, NDArray[np.float64]],
    particle_inertia_time: Union[float, NDArray[np.float64]],
    stokes_number: Union[float, NDArray[np.float64]],
    kolmogorov_length_scale: float,
    reynolds_lambda: float,
    normalized_accel_variance: float,
    kolmogorov_velocity: float,
    kolmogorov_time: float,
) -> Union[float, NDArray[np.float64]]:
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    """Compute the geometric collision kernel Γ₁₂ from DNS simulations.

    Where the equation is

    - Γ₁₂ = 2π R² ⟨|wᵣ|⟩ g₁₂
        - Γ₁₂ is collision kernel [m³/s].
        - R is collision radius [m].
        - wᵣ is radial relative velocity [m/s].
        - g₁₂ is radial distribution function [dimensionless].

    Arguments:
        - particle_radius : Particle radius [m].
        - velocity_dispersion : Velocity dispersion [m/s].
        - particle_inertia_time : Particle inertia time [s].
        - stokes_number : Stokes number [-].
        - kolmogorov_length_scale : Kolmogorov length scale [m].
        - reynolds_lambda : Reynolds number [-].
        - normalized_accel_variance : Normalized acceleration variance [-].
        - kolmogorov_velocity : Kolmogorov velocity [m/s].
        - kolmogorov_time : Kolmogorov time [s].

    Returns:
        - Collision kernel Γ₁₂ [m³/s].



    References:
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2.
        Theory and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """
    # Type narrowing: ensure array for indexing operations
    radius_array = (
        particle_radius
        if isinstance(particle_radius, np.ndarray)
        else np.array([particle_radius])
    )

    collision_radius = radius_array[:, np.newaxis] + radius_array[np.newaxis, :]

    # Compute radial relative velocity ⟨ |w_r| ⟩
    wr = get_radial_relative_velocity_dz2002(
        velocity_dispersion, particle_inertia_time
    )

    # Compute radial distribution function g₁₂
    g12 = get_g12_radial_distribution_ao2008(
        particle_radius,
        stokes_number,
        kolmogorov_length_scale,
        reynolds_lambda,
        normalized_accel_variance,
        kolmogorov_velocity,
        kolmogorov_time,
    )

    # Compute collision kernel Γ₁₂
    return 2 * np.pi * collision_radius**2 * wr * g12


def get_turbulent_dns_kernel_ao2008_via_system_state(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    fluid_density: float,
    temperature: float,
    re_lambda: float,
    relative_velocity: Union[float, NDArray[np.float64]],
    turbulent_dissipation: float,
) -> Union[float, NDArray[np.float64]]:
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    # pylint: disable=too-many-locals
    """Compute the geometric collision kernel Γ₁₂ using AO2008 for system.

    This function orchestrates the calculation of the geometric collision
    kernel by deriving necessary fluid, turbulence, and particle parameters
    from the provided system state. The returned value (or array) represents
    the collision kernel, Γ₁₂ [m³/s], which describes collision frequency
    under turbulence.

    Arguments:
        - particle_radius : Radius of the particles [m]. If an array is given,
            it is assumed to represent multiple particle sizes.
        - particle_density : Density of the particles [kg/m³]. Must match the
            dimensionality of `particle_radius` if both are arrays.
        - fluid_density : Density of the surrounding fluid [kg/m³].
        - temperature : Temperature of the fluid [K].
        - re_lambda : Turbulent Reynolds number based on the Taylor microscale.
        - relative_velocity : Mean relative velocity between the particle and
            fluid [m/s]. Can be a single value or an array of the same
            dimensionality as `particle_radius`.
        - turbulent_dissipation : Turbulent kinetic energy dissipation rate
            [m²/s³].

    Returns:
        - Collision kernel Γ₁₂ [m³/s].

    Examples:
        ```py
        kernel_via_state = get_turbulent_dns_kernel_ao2008_via_system_state(
            particle_radius=np.array([1e-6, 2e-6]),
            particle_density=1000.0,
            fluid_density=1.225,
            temperature=298.15,
            re_lambda=100.0,
            relative_velocity=0.1,
            turbulent_dissipation=0.01,
        )
        # Output: array([...])
        ```

    References:
        - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
          the geometric collision rate of sedimenting droplets. Part 2.
          Theory and parameterization. New Journal of Physics, 10.
          https://doi.org/10.1088/1367-2630/10/7/075016
    """
    # 1. Basic fluid properties
    dynamic_viscosity = gas.get_dynamic_viscosity(temperature)
    kinematic_viscosity = gas.get_kinematic_viscosity(
        dynamic_viscosity=dynamic_viscosity, fluid_density=fluid_density
    )
    mean_free_path = gas.get_molecule_mean_free_path(
        temperature=temperature, dynamic_viscosity=dynamic_viscosity
    )

    # 2. Slip correction factors
    knudsen_number = particles.get_knudsen_number(
        mean_free_path=mean_free_path, particle_radius=particle_radius
    )
    slip_correction_factor = particles.get_cunningham_slip_correction(
        knudsen_number
    )

    # Handle radius addition properly for arrays
    collisional_radius = (
        particle_radius[:, np.newaxis] + particle_radius[np.newaxis, :]
        if isinstance(particle_radius, np.ndarray)
        else 2.0 * particle_radius
    )

    # 3. Particle inertia and settling velocity
    particle_inertia_time = particles.get_particle_inertia_time(
        particle_radius=particle_radius,
        particle_density=particle_density,
        fluid_density=fluid_density,
        kinematic_viscosity=kinematic_viscosity,
    )
    particle_settling_velocity = (
        particles.get_particle_settling_velocity_with_drag(
            particle_radius=particle_radius,
            particle_density=particle_density,
            fluid_density=fluid_density,
            dynamic_viscosity=dynamic_viscosity,
            slip_correction_factor=slip_correction_factor,
            re_threshold=0.1,
        )
    )
    particle_velocity = np.abs(particle_settling_velocity - relative_velocity)

    # 4. Turbulence scales
    fluid_rms_velocity = gas.get_fluid_rms_velocity(
        re_lambda=re_lambda,
        kinematic_viscosity=kinematic_viscosity,
        turbulent_dissipation=turbulent_dissipation,
    )
    taylor_microscale = gas.get_taylor_microscale(
        fluid_rms_velocity=fluid_rms_velocity,
        kinematic_viscosity=kinematic_viscosity,
        turbulent_dissipation=turbulent_dissipation,
    )
    eulerian_integral_length = gas.get_eulerian_integral_length(
        fluid_rms_velocity=fluid_rms_velocity,
        turbulent_dissipation=turbulent_dissipation,
    )
    lagrangian_integral_time = gas.get_lagrangian_integral_time(
        fluid_rms_velocity=fluid_rms_velocity,
        turbulent_dissipation=turbulent_dissipation,
    )

    # 6. Additional turbulence-based quantities
    kolmogorov_time = gas.get_kolmogorov_time(
        kinematic_viscosity=kinematic_viscosity,
        turbulent_dissipation=turbulent_dissipation,
    )
    stokes_number = particles.get_stokes_number(
        particle_inertia_time=particle_inertia_time,
        kolmogorov_time=kolmogorov_time,
    )
    kolmogorov_length_scale = gas.get_kolmogorov_length(
        kinematic_viscosity=kinematic_viscosity,
        turbulent_dissipation=turbulent_dissipation,
    )
    normalized_accel_variance = gas.get_normalized_accel_variance_ao2008(
        re_lambda=re_lambda,
    )
    kolmogorov_velocity = gas.get_kolmogorov_velocity(
        kinematic_viscosity=kinematic_viscosity,
        turbulent_dissipation=turbulent_dissipation,
    )
    lagrangian_taylor_microscale_time = (
        gas.get_lagrangian_taylor_microscale_time(
            kolmogorov_time=kolmogorov_time,
            re_lambda=re_lambda,
            accel_variance=normalized_accel_variance,
        )
    )

    # 5. Relative velocity variance
    velocity_dispersion = get_relative_velocity_variance(
        fluid_rms_velocity=fluid_rms_velocity,
        collisional_radius=collisional_radius,
        particle_inertia_time=particle_inertia_time,
        particle_velocity=particle_velocity,
        taylor_microscale=taylor_microscale,
        eulerian_integral_length=eulerian_integral_length,
        lagrangian_integral_time=lagrangian_integral_time,
        lagrangian_taylor_microscale_time=lagrangian_taylor_microscale_time,
    )

    # Compute Kernel Values
    return get_turbulent_dns_kernel_ao2008(
        particle_radius=particle_radius,
        velocity_dispersion=np.abs(velocity_dispersion),
        particle_inertia_time=particle_inertia_time,
        stokes_number=stokes_number,
        kolmogorov_length_scale=kolmogorov_length_scale,
        reynolds_lambda=re_lambda,
        normalized_accel_variance=normalized_accel_variance,
        kolmogorov_velocity=kolmogorov_velocity,
        kolmogorov_time=kolmogorov_time,
    )
