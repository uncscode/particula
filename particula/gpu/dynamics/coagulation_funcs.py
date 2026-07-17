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


@wp.func
def coulomb_potential_ratio_wp(
    radius_i: wp.float64,
    radius_j: wp.float64,
    charge_i: wp.float64,
    charge_j: wp.float64,
    temperature: wp.float64,
    boltzmann_constant: wp.float64,
    elementary_charge_value: wp.float64,
    electric_permittivity: wp.float64,
) -> wp.float64:
    """Calculate the dimensionless pair Coulomb potential ratio.

    Computes ``-(q_i * q_j * e**2) / (4 * pi * epsilon_0 * (r_i + r_j)
    * k_B * T)``. A non-positive radius sum or temperature safely returns
    ``0.0``. Repulsion is lower-clipped at ``-200.0``.

    Args:
        radius_i: Particle i radius in meters.
        radius_j: Particle j radius in meters.
        charge_i: Particle i charge as an elementary-charge count.
        charge_j: Particle j charge as an elementary-charge count.
        temperature: Gas temperature in kelvin.
        boltzmann_constant: Boltzmann constant in joules per kelvin.
        elementary_charge_value: Elementary charge magnitude in coulombs.
        electric_permittivity: Vacuum electric permittivity in farads per
            meter.

    Returns:
        Dimensionless Coulomb potential ratio, or ``0.0`` for an invalid
        radius sum or temperature.

    References:
        Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced
        collisions in aerosols and dusty plasmas. *Physical Review E*, 85(2).
        https://doi.org/10.1103/PhysRevE.85.026410
    """
    sum_radius = radius_i + radius_j
    if sum_radius <= wp.float64(0.0) or temperature <= wp.float64(0.0):
        return wp.float64(0.0)

    pi_value = wp.float64(3.141592653589793)
    potential_ratio = -(
        charge_i * charge_j * elementary_charge_value * elementary_charge_value
    ) / (
        wp.float64(4.0)
        * pi_value
        * electric_permittivity
        * sum_radius
        * boltzmann_constant
        * temperature
    )
    return wp.max(potential_ratio, wp.float64(-200.0))


@wp.func
def reduced_value_wp(left: wp.float64, right: wp.float64) -> wp.float64:
    """Calculate a scalar reduced value for mass or friction inputs.

    Computes ``(left * right) / (left + right)``. Inputs retain their supplied
    units. A non-positive input sum safely returns ``0.0``.

    Args:
        left: First mass in kilograms or friction coefficient in kilograms per
            second.
        right: Second mass in kilograms or friction coefficient in kilograms
            per second.

    Returns:
        Reduced value in the input units, or ``0.0`` when the input sum is
        non-positive.

    References:
        Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free
        molecular regime Coulombic collisions in aerosols and dusty plasmas.
        *Aerosol Science and Technology*, 53(8), 933-957.
        https://doi.org/10.1080/02786826.2019.1614522
    """
    denominator = left + right
    if denominator <= wp.float64(0.0):
        return wp.float64(0.0)
    return left * right / denominator


@wp.func
def coulomb_kinetic_limit_wp(
    coulomb_potential_ratio: wp.float64,
) -> wp.float64:
    """Calculate the dimensionless kinetic Coulomb enhancement factor.

    Computes ``1 + phi`` for ``phi >= 0`` and ``exp(phi)`` otherwise. Finite
    scalar inputs have no fallback condition.

    Args:
        coulomb_potential_ratio: Dimensionless Coulomb potential ratio.

    Returns:
        Dimensionless kinetic-limit Coulomb enhancement factor.

    References:
        Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced
        collisions in aerosols and dusty plasmas. *Physical Review E*, 85(2).
        https://doi.org/10.1103/PhysRevE.85.026410
    """
    if coulomb_potential_ratio >= wp.float64(0.0):
        return wp.float64(1.0) + coulomb_potential_ratio
    return wp.exp(coulomb_potential_ratio)


@wp.func
def coulomb_continuum_limit_wp(
    coulomb_potential_ratio: wp.float64,
) -> wp.float64:
    """Calculate the dimensionless continuum Coulomb enhancement factor.

    Computes ``phi / (1 - exp(-phi))``. A neutral potential safely returns the
    limiting value ``1.0``.

    Args:
        coulomb_potential_ratio: Dimensionless Coulomb potential ratio.

    Returns:
        Dimensionless continuum-limit Coulomb enhancement factor.

    References:
        Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced
        collisions in aerosols and dusty plasmas. *Physical Review E*, 85(2).
        https://doi.org/10.1103/PhysRevE.85.026410
    """
    if coulomb_potential_ratio == wp.float64(0.0):
        return wp.float64(1.0)
    return coulomb_potential_ratio / (
        wp.float64(1.0) - wp.exp(-coulomb_potential_ratio)
    )


@wp.func
def diffusive_knudsen_number_wp(
    radius_i: wp.float64,
    radius_j: wp.float64,
    mass_i: wp.float64,
    mass_j: wp.float64,
    friction_i: wp.float64,
    friction_j: wp.float64,
    coulomb_potential_ratio: wp.float64,
    temperature: wp.float64,
    boltzmann_constant: wp.float64,
) -> wp.float64:
    """Calculate the dimensionless pair diffusive Knudsen number.

    Computes ``[sqrt(k_B * T * m_red) / f_red] / [(r_i + r_j) * Gamma_c /
    Gamma_k]``. A non-positive radius sum, temperature, reduced mass, or
    reduced friction safely returns ``0.0``. A kinetic enhancement below
    ``1e-80`` also returns ``0.0`` for extreme repulsion.

    Args:
        radius_i: Particle i radius in meters.
        radius_j: Particle j radius in meters.
        mass_i: Particle i mass in kilograms.
        mass_j: Particle j mass in kilograms.
        friction_i: Particle i friction coefficient in kilograms per second.
        friction_j: Particle j friction coefficient in kilograms per second.
        coulomb_potential_ratio: Dimensionless Coulomb potential ratio.
        temperature: Gas temperature in kelvin.
        boltzmann_constant: Boltzmann constant in joules per kelvin.

    Returns:
        Dimensionless diffusive Knudsen number, or ``0.0`` for an invalid
        scalar domain or a kinetic enhancement below ``1e-80``.

    References:
        Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free
        molecular regime Coulombic collisions in aerosols and dusty plasmas.
        *Aerosol Science and Technology*, 53(8), 933-957.
        https://doi.org/10.1080/02786826.2019.1614522
        Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced
        collisions in aerosols and dusty plasmas. *Physical Review E*, 85(2).
        https://doi.org/10.1103/PhysRevE.85.026410
    """
    sum_radius = radius_i + radius_j
    if sum_radius <= wp.float64(0.0) or temperature <= wp.float64(0.0):
        return wp.float64(0.0)

    reduced_mass = reduced_value_wp(mass_i, mass_j)
    reduced_friction = reduced_value_wp(friction_i, friction_j)
    if reduced_mass <= wp.float64(0.0) or reduced_friction <= wp.float64(0.0):
        return wp.float64(0.0)

    kinetic_limit = coulomb_kinetic_limit_wp(coulomb_potential_ratio)
    if kinetic_limit < wp.float64(1.0e-80):
        return wp.float64(0.0)

    continuum_limit = coulomb_continuum_limit_wp(coulomb_potential_ratio)
    numerator = (
        wp.sqrt(boltzmann_constant * temperature * reduced_mass)
        / reduced_friction
    )
    denominator = sum_radius * continuum_limit / kinetic_limit
    return numerator / denominator
