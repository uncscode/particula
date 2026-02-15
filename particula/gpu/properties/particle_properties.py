"""Warp GPU implementations of particle property functions.

These functions mirror the NumPy implementations in
``particula.particles.properties`` for use inside Warp kernels.
"""

import warp as wp


@wp.func
def knudsen_number_wp(
    mean_free_path: wp.float64,
    particle_radius: wp.float64,
) -> wp.float64:
    """Calculate the Knudsen number from mean free path and radius.

    Port of ``particula.particles.properties.knudsen_number_module``.

    Args:
        mean_free_path: Mean free path of the gas molecules [m].
        particle_radius: Particle radius [m].

    Returns:
        Knudsen number (dimensionless).
    """
    return mean_free_path / particle_radius


@wp.func
def cunningham_slip_correction_wp(
    knudsen_number: wp.float64,
) -> wp.float64:
    """Calculate the Cunningham slip correction factor.

    Port of
    ``particula.particles.properties.slip_correction_module.get_cunningham_slip_correction``.

    Args:
        knudsen_number: Knudsen number (dimensionless).

    Returns:
        Cunningham slip correction factor (dimensionless).
    """
    return wp.float64(1.0) + knudsen_number * (
        wp.float64(1.257)
        + wp.float64(0.4) * wp.exp(-wp.float64(1.1) / knudsen_number)
    )


@wp.func
def aerodynamic_mobility_wp(
    particle_radius: wp.float64,
    slip_correction_factor: wp.float64,
    dynamic_viscosity: wp.float64,
) -> wp.float64:
    """Calculate aerodynamic mobility of a particle.

    Port of
    ``particula.particles.properties.aerodynamic_mobility_module.get_aerodynamic_mobility``.

    Args:
        particle_radius: Particle radius [m].
        slip_correction_factor: Slip correction factor (dimensionless).
        dynamic_viscosity: Dynamic viscosity [Pa·s].

    Returns:
        Aerodynamic mobility [m²/s].
    """
    pi_value = wp.float64(3.141592653589793)
    return slip_correction_factor / (
        wp.float64(6.0) * pi_value * dynamic_viscosity * particle_radius
    )


@wp.func
def mean_thermal_speed_wp(
    particle_mass: wp.float64,
    temperature: wp.float64,
    boltzmann_constant: wp.float64,
) -> wp.float64:
    """Calculate the mean thermal speed of a particle.

    Port of
    ``particula.particles.properties.mean_thermal_speed_module.get_mean_thermal_speed``.

    Args:
        particle_mass: Particle mass [kg].
        temperature: Gas temperature [K].
        boltzmann_constant: Boltzmann constant [J/K].

    Returns:
        Mean thermal speed [m/s].
    """
    pi_value = wp.float64(3.141592653589793)
    return wp.sqrt(
        (wp.float64(8.0) * boltzmann_constant * temperature)
        / (pi_value * particle_mass)
    )


@wp.func
def friction_factor_wp(
    particle_radius: wp.float64,
    dynamic_viscosity: wp.float64,
    slip_correction: wp.float64,
) -> wp.float64:
    """Calculate the friction factor for a particle.

    Port of
    ``particula.particles.properties.friction_factor_module.get_friction_factor``.

    Args:
        particle_radius: Particle radius [m].
        dynamic_viscosity: Dynamic viscosity [Pa·s].
        slip_correction: Slip correction factor (dimensionless).

    Returns:
        Friction factor [N·s/m].
    """
    pi_value = wp.float64(3.141592653589793)
    return (
        wp.float64(6.0) * pi_value * dynamic_viscosity * particle_radius
    ) / slip_correction
