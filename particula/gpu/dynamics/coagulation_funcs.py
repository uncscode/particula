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

    Port of ``particula.dynamics.coagulation.brownian_kernel._brownian_diffusivity``.

    Args:
        temperature: Gas temperature [K].
        aerodynamic_mobility: Aerodynamic mobility [m²/s].
        boltzmann_constant: Boltzmann constant [J/K].

    Returns:
        Brownian diffusivity [m²/s].
    """
    return boltzmann_constant * temperature * aerodynamic_mobility


@wp.func
def particle_mean_free_path_wp(
    diffusivity_particle: wp.float64,
    mean_thermal_speed_particle: wp.float64,
) -> wp.float64:
    """Calculate the particle mean free path for coagulation.

    Port of ``particula.dynamics.coagulation.brownian_kernel._mean_free_path_l``.

    Args:
        diffusivity_particle: Particle diffusivity [m²/s].
        mean_thermal_speed_particle: Particle mean thermal speed [m/s].

    Returns:
        Particle mean free path [m].
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

    Port of ``particula.dynamics.coagulation.brownian_kernel._g_collection_term``.

    Args:
        mean_free_path_particle: Particle mean free path [m].
        particle_radius: Particle radius [m].

    Returns:
        Collection term ``g`` (dimensionless).
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

    Port of ``particula.dynamics.coagulation.brownian_kernel.get_brownian_kernel``
    for a scalar particle pair.

    Args:
        radius_i: Particle radius i [m].
        radius_j: Particle radius j [m].
        diff_i: Particle diffusivity i [m²/s].
        diff_j: Particle diffusivity j [m²/s].
        g_i: Collection term i (dimensionless).
        g_j: Collection term j (dimensionless).
        speed_i: Mean thermal speed i [m/s].
        speed_j: Mean thermal speed j [m/s].
        alpha: Collision efficiency (dimensionless).

    Returns:
        Brownian coagulation kernel for the pair [m³/s].
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
