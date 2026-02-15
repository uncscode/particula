"""Warp GPU condensation composite functions.

These functions mirror NumPy implementations in
``particula.particles.properties`` and
``particula.dynamics.condensation.mass_transfer``.
"""

import warp as wp


@wp.func
def diffusion_coefficient_wp(
    temperature: wp.float64,
    aerodynamic_mobility: wp.float64,
    boltzmann_constant: wp.float64,
) -> wp.float64:
    """Calculate the diffusion coefficient via Stokes-Einstein.

    Port of ``particula.particles.properties.diffusion_coefficient``.

    Args:
        temperature: Gas temperature [K].
        aerodynamic_mobility: Aerodynamic mobility [m²/s].
        boltzmann_constant: Boltzmann constant [J/K].

    Returns:
        Diffusion coefficient [m²/s].
    """
    return boltzmann_constant * temperature * aerodynamic_mobility


@wp.func
def first_order_mass_transport_k_wp(
    particle_radius: wp.float64,
    vapor_transition: wp.float64,
    diffusion_coefficient: wp.float64,
) -> wp.float64:
    """Calculate the first-order mass transport coefficient.

    Port of
    ``particula.dynamics.condensation.mass_transfer.get_first_order_mass_transport_k``.

    Args:
        particle_radius: Particle radius [m].
        vapor_transition: Vapor transition correction factor (dimensionless).
        diffusion_coefficient: Diffusion coefficient [m²/s].

    Returns:
        First-order mass transport coefficient [m³/s].
    """
    pi_value = wp.float64(3.141592653589793)
    return (
        wp.float64(4.0)
        * pi_value
        * particle_radius
        * diffusion_coefficient
        * vapor_transition
    )


@wp.func
def mass_transfer_rate_wp(
    pressure_delta: wp.float64,
    first_order_mass_transport: wp.float64,
    temperature: wp.float64,
    molar_mass: wp.float64,
    gas_constant: wp.float64,
) -> wp.float64:
    """Calculate the condensation mass transfer rate.

    Port of
    ``particula.dynamics.condensation.mass_transfer.get_mass_transfer_rate``.

    Args:
        pressure_delta: Partial pressure difference [Pa].
        first_order_mass_transport: Mass transport coefficient [m³/s].
        temperature: Temperature [K].
        molar_mass: Molar mass [kg/mol].
        gas_constant: Universal gas constant [J/(mol·K)].

    Returns:
        Mass transfer rate [kg/s].
    """
    return (
        first_order_mass_transport
        * pressure_delta
        * molar_mass
        / (gas_constant * temperature)
    )
