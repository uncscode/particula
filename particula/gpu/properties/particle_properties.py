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


@wp.func
def vapor_transition_correction_wp(
    knudsen_number: wp.float64,
    mass_accommodation: wp.float64,
) -> wp.float64:
    """Calculate the vapor transition correction factor.

    Port of
    ``particula.particles.properties.vapor_correction_module.get_vapor_transition_correction``.

    Args:
        knudsen_number: Knudsen number (dimensionless).
        mass_accommodation: Mass accommodation coefficient (dimensionless).

    Returns:
        Vapor transition correction factor (dimensionless).
    """
    numerator = (
        wp.float64(0.75)  # Fuchs-Sutugin model coefficient.
        * mass_accommodation
        * (wp.float64(1.0) + knudsen_number)
    )
    denominator = (
        knudsen_number * knudsen_number
        + knudsen_number
        + wp.float64(0.283)  # Fuchs-Sutugin model coefficient.
        * mass_accommodation
        * knudsen_number
        + wp.float64(0.75) * mass_accommodation
    )
    return numerator / denominator


@wp.func
def kelvin_radius_wp(
    effective_surface_tension: wp.float64,
    effective_density: wp.float64,
    molar_mass: wp.float64,
    temperature: wp.float64,
    gas_constant: wp.float64,
) -> wp.float64:
    """Calculate the Kelvin radius.

    Port of
    ``particula.particles.properties.kelvin_effect_module.get_kelvin_radius``.

    Args:
        effective_surface_tension: Effective surface tension [N/m].
        effective_density: Effective density [kg/m³].
        molar_mass: Molar mass [kg/mol].
        temperature: Temperature [K].
        gas_constant: Gas constant [J/(mol·K)].

    Returns:
        Kelvin radius [m].
    """
    numerator = wp.float64(2.0) * effective_surface_tension * molar_mass
    denominator = gas_constant * temperature * effective_density
    return numerator / denominator


@wp.func
def kelvin_term_wp(
    particle_radius: wp.float64,
    kelvin_radius_value: wp.float64,
) -> wp.float64:
    """Calculate the Kelvin term with safe clamping.

    Port of
    ``particula.particles.properties.kelvin_effect_module.get_kelvin_term``.

    Args:
        particle_radius: Particle radius [m].
        kelvin_radius_value: Kelvin radius [m].

    Returns:
        Kelvin term (dimensionless).
    """
    ratio = kelvin_radius_value / particle_radius
    clamped_ratio = wp.min(
        ratio,
        wp.float64(100.0),  # Matches MAX_KELVIN_RATIO in Python module.
    )
    return wp.exp(clamped_ratio)


@wp.func
def partial_pressure_delta_wp(
    partial_pressure_gas: wp.float64,
    partial_pressure_particle: wp.float64,
    kelvin_term: wp.float64,
) -> wp.float64:
    """Calculate the partial pressure delta.

    Port of
    ``particula.particles.properties.partial_pressure_module.get_partial_pressure_delta``.

    Args:
        partial_pressure_gas: Gas-phase partial pressure [Pa].
        partial_pressure_particle: Particle-phase partial pressure [Pa].
        kelvin_term: Kelvin term (dimensionless).

    Returns:
        Partial pressure delta [Pa].
    """
    return partial_pressure_gas - partial_pressure_particle * kelvin_term
