"""Warp GPU coagulation composite functions.

These functions mirror the NumPy implementations in
``particula.dynamics.coagulation.brownian_kernel``.
"""

import warp as wp


@wp.func
def brownian_diffusivity_wp(
    temperature: wp.float64,
    aerodynamic_mobility: wp.float64,
    boltzmann_constant: wp.float64,
) -> wp.float64:
    """Calculate Brownian diffusivity via Stokes-Einstein scaling.

    Port of
    ``particula.dynamics.coagulation.brownian_kernel._brownian_diffusivity``
    using ``D = boltzmann_constant * temperature * aerodynamic_mobility``.

    Args:
        temperature: Gas temperature in kelvin.
        aerodynamic_mobility: Aerodynamic mobility with units consistent
            with the Stokes-Einstein relation.
        boltzmann_constant: Boltzmann constant in joules per kelvin.

    Returns:
        Brownian diffusivity in square meters per second.
    """
    return boltzmann_constant * temperature * aerodynamic_mobility


@wp.func
def particle_mean_free_path_wp(
    diffusivity_particle: wp.float64,
    mean_thermal_speed_particle: wp.float64,
) -> wp.float64:
    """Calculate the particle mean free path for coagulation.

    Port of ``particula.dynamics.coagulation.brownian_kernel._mean_free_path_l``
    using ``lambda = 8 * diffusivity_particle / (pi * mean_thermal_speed)``.

    Args:
        diffusivity_particle: Particle diffusivity in square meters per second.
        mean_thermal_speed_particle: Particle mean thermal speed in meters per
            second.

    Returns:
        Particle mean free path in meters.
    """
    pi_value = wp.float64(3.141592653589793)
    return (
        wp.float64(8.0)
        * diffusivity_particle
        / (pi_value * mean_thermal_speed_particle)
    )


@wp.func
def g_collection_term_wp(
    mean_free_path_particle: wp.float64,
    particle_radius: wp.float64,
) -> wp.float64:
    """Calculate the Brownian coagulation collection term ``g``.

    Port of
    ``particula.dynamics.coagulation.brownian_kernel._g_collection_term`` using
    the Fuchs form for the collection enhancement term.

    Args:
        mean_free_path_particle: Particle mean free path in meters.
        particle_radius: Particle radius in meters.

    Returns:
        Collection term ``g`` as a dimensionless quantity.
    """
    two_radius = wp.float64(2.0) * particle_radius
    numerator = wp.pow(
        two_radius + mean_free_path_particle,
        wp.float64(3.0),
    ) - wp.pow(
        wp.float64(4.0) * wp.pow(particle_radius, wp.float64(2.0))
        + wp.pow(mean_free_path_particle, wp.float64(2.0)),
        wp.float64(1.5),
    )
    denominator = wp.float64(6.0) * particle_radius * mean_free_path_particle
    return numerator / denominator - wp.float64(2.0) * particle_radius


@wp.func
def brownian_kernel_pair_wp(
    radius_i: wp.float64,
    radius_j: wp.float64,
    diff_i: wp.float64,
    diff_j: wp.float64,
    g_i: wp.float64,
    g_j: wp.float64,
    speed_i: wp.float64,
    speed_j: wp.float64,
    alpha: wp.float64,
) -> wp.float64:
    """Calculate the scalar Brownian coagulation kernel for a pair.

    Port of
    ``particula.dynamics.coagulation.brownian_kernel.get_brownian_kernel`` for
    a scalar particle pair using the Fuchs correction to the continuum kernel.

    Args:
        radius_i: Particle radius for particle i in meters.
        radius_j: Particle radius for particle j in meters.
        diff_i: Particle diffusivity for particle i in square meters per second.
        diff_j: Particle diffusivity for particle j in square meters per second.
        g_i: Collection term for particle i (dimensionless).
        g_j: Collection term for particle j (dimensionless).
        speed_i: Mean thermal speed for particle i in meters per second.
        speed_j: Mean thermal speed for particle j in meters per second.
        alpha: Collision efficiency (dimensionless).

    Returns:
        Brownian coagulation kernel for the pair in cubic meters per second.
    """
    pi_value = wp.float64(3.141592653589793)
    sum_radius = radius_i + radius_j
    sum_diffusivity = diff_i + diff_j
    g_term_sqrt = wp.sqrt(g_i * g_i + g_j * g_j)
    speed_sqrt = wp.sqrt(speed_i * speed_i + speed_j * speed_j)
    return (
        wp.float64(4.0)
        * pi_value
        * sum_diffusivity
        * sum_radius
        / (
            sum_radius / (sum_radius + g_term_sqrt)
            + wp.float64(4.0)
            * sum_diffusivity
            / (sum_radius * speed_sqrt * alpha)
        )
    )
