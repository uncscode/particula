"""Warp device functions for Brownian, sedimentation, and charged coagulation.

The Brownian helpers mirror ``particula.dynamics.coagulation.brownian_kernel``.
The internal sedimentation helpers calculate mixture effective density, Stokes
settling velocity with Cunningham slip, and the unit-efficiency SP2016 pair
rate using fp64 scalar device inputs.
The concrete-module-only charged hard-sphere helper supports the low-level,
charged-only and canonical Brownian-plus-charged particle-resolved GPU paths.
It does not add public exports or higher-level charged coagulation APIs.
"""

import warp as wp

from particula.gpu.properties.gas_properties import (
    dynamic_viscosity_wp,
    molecule_mean_free_path_wp,
)
from particula.gpu.properties.particle_properties import (
    cunningham_slip_correction_wp,
    friction_factor_wp,
    knudsen_number_wp,
)
from particula.util.constants import STANDARD_GRAVITY

_PI = wp.constant(wp.float64(3.141592653589793))
_STANDARD_GRAVITY = wp.constant(wp.float64(STANDARD_GRAVITY))
_MIN_NORMAL_FLOAT64 = wp.constant(wp.float64(2.2250738585072014e-308))
_COULOMB_SERIES_THRESHOLD = wp.constant(wp.float64(1.0e-5))
_KINETIC_LIMIT_CUTOFF = wp.constant(wp.float64(1.0e-80))
_HARD_SPHERE_LINEAR = wp.constant(wp.float64(25.836))
_HARD_SPHERE_QUARTIC = wp.constant(wp.float64(11.211))
_HARD_SPHERE_DENOMINATOR_LINEAR = wp.constant(wp.float64(3.502))
_HARD_SPHERE_DENOMINATOR_QUADRATIC = wp.constant(wp.float64(7.211))


@wp.func
def effective_density_wp(
    total_mass: wp.float64,
    total_volume: wp.float64,
) -> wp.float64:
    """Calculate an fp64 effective mixture density from scalar SI totals.

    This device-only final stage expects callers to sum component mass [kg] and
    volume [m³] before invocation. It returns exact ``0.0`` when an input or
    the quotient is non-finite, non-positive, or underflowed.

    Args:
        total_mass: Total particle mass [kg].
        total_volume: Total particle volume [m³].

    Returns:
        Effective mixture density [kg/m³], or exact safe zero for an invalid
        calculation.
    """
    zero = wp.float64(0.0)
    if (
        not wp.isfinite(total_mass)
        or total_mass <= zero
        or not wp.isfinite(total_volume)
        or total_volume <= zero
    ):
        return zero
    density = total_mass / total_volume
    if (
        not wp.isfinite(density)
        or density <= zero
        or density < _MIN_NORMAL_FLOAT64
    ):
        return zero
    return density


@wp.func
def settling_velocity_stokes_wp(
    particle_radius: wp.float64,
    effective_density: wp.float64,
    temperature: wp.float64,
    pressure: wp.float64,
    gas_constant: wp.float64,
    molecular_weight_air: wp.float64,
    ref_viscosity: wp.float64,
    ref_temperature: wp.float64,
    sutherland_constant: wp.float64,
) -> wp.float64:
    """Calculate fp64 Stokes settling velocity with Cunningham slip.

    This device-only helper derives gas viscosity, molecular mean free path,
    Knudsen number, and Cunningham slip once from the supplied SI gas state.
    It applies ``v = 2 r² ρ C_c g / (9 μ)`` with standard gravity and no
    fluid-density or non-Stokes drag correction. Non-finite, non-positive,
    underflowed, or overflowed inputs, intermediates, and results return exact
    ``0.0``.

    Args:
        particle_radius: Particle radius [m].
        effective_density: Effective particle density [kg/m³].
        temperature: Gas temperature [K].
        pressure: Gas pressure [Pa].
        gas_constant: Universal gas constant [J/(mol K)].
        molecular_weight_air: Air molar mass [kg/mol].
        ref_viscosity: Reference dynamic viscosity [Pa s].
        ref_temperature: Reference temperature [K].
        sutherland_constant: Sutherland temperature constant [K].

    Returns:
        Settling velocity [m/s], or exact ``0.0`` for an invalid calculation.

    References:
        Seinfeld, J. H., & Pandis, S. N. (2016). *Atmospheric Chemistry and
        Physics* (3rd ed.). John Wiley & Sons.
    """
    zero = wp.float64(0.0)
    if (
        not wp.isfinite(particle_radius)
        or particle_radius <= zero
        or not wp.isfinite(effective_density)
        or effective_density <= zero
        or not wp.isfinite(temperature)
        or temperature <= zero
        or not wp.isfinite(pressure)
        or pressure <= zero
        or not wp.isfinite(gas_constant)
        or gas_constant <= zero
        or not wp.isfinite(molecular_weight_air)
        or molecular_weight_air <= zero
        or not wp.isfinite(ref_viscosity)
        or ref_viscosity <= zero
        or not wp.isfinite(ref_temperature)
        or ref_temperature <= zero
        or not wp.isfinite(sutherland_constant)
        or sutherland_constant <= zero
    ):
        return zero
    dynamic_viscosity = dynamic_viscosity_wp(
        temperature,
        ref_viscosity,
        ref_temperature,
        sutherland_constant,
    )
    if not wp.isfinite(dynamic_viscosity) or dynamic_viscosity <= zero:
        return zero
    mean_free_path = molecule_mean_free_path_wp(
        molecular_weight_air,
        temperature,
        pressure,
        dynamic_viscosity,
        gas_constant,
    )
    if not wp.isfinite(mean_free_path) or mean_free_path <= zero:
        return zero
    return settling_velocity_stokes_from_transport_wp(
        particle_radius,
        effective_density,
        dynamic_viscosity,
        mean_free_path,
    )


@wp.func
def settling_velocity_stokes_from_transport_wp(
    particle_radius: wp.float64,
    effective_density: wp.float64,
    dynamic_viscosity: wp.float64,
    mean_free_path: wp.float64,
) -> wp.float64:
    """Calculate Stokes settling velocity from precomputed gas transport data.

    Args:
        particle_radius: Particle radius [m].
        effective_density: Effective particle density [kg/m³].
        dynamic_viscosity: Dynamic gas viscosity [Pa s].
        mean_free_path: Molecular mean free path [m].

    Returns:
        Settling velocity [m/s], or exact ``0.0`` for invalid inputs.
    """
    zero = wp.float64(0.0)
    if (
        not wp.isfinite(particle_radius)
        or particle_radius <= zero
        or not wp.isfinite(effective_density)
        or effective_density <= zero
        or not wp.isfinite(dynamic_viscosity)
        or dynamic_viscosity <= zero
        or not wp.isfinite(mean_free_path)
        or mean_free_path <= zero
    ):
        return zero
    knudsen_number = knudsen_number_wp(mean_free_path, particle_radius)
    if not wp.isfinite(knudsen_number) or knudsen_number <= zero:
        return zero
    slip_correction = cunningham_slip_correction_wp(knudsen_number)
    if not wp.isfinite(slip_correction) or slip_correction <= zero:
        return zero
    radius_squared = particle_radius * particle_radius
    if not wp.isfinite(radius_squared) or radius_squared <= zero:
        return zero
    numerator = (
        wp.float64(2.0)
        * radius_squared
        * effective_density
        * slip_correction
        * _STANDARD_GRAVITY
    )
    if not wp.isfinite(numerator) or numerator <= zero:
        return zero
    denominator = wp.float64(9.0) * dynamic_viscosity
    if not wp.isfinite(denominator) or denominator <= zero:
        return zero
    velocity = numerator / denominator
    if (
        not wp.isfinite(velocity)
        or velocity <= zero
        or velocity < _MIN_NORMAL_FLOAT64
    ):
        return zero
    return velocity


@wp.func
def sedimentation_sp2016_pair_rate_wp(
    radius_i: wp.float64,
    radius_j: wp.float64,
    velocity_i: wp.float64,
    velocity_j: wp.float64,
) -> wp.float64:
    """Calculate the fp64 SP2016 Eq. 13A.4 sedimentation pair rate.

    This device-only helper applies unit collision efficiency to
    ``K = π (r_i + r_j)² |v_i - v_j|``. Non-finite or negative inputs and
    non-finite, overflowed, or underflowed intermediates and results return
    exact ``0.0``. Equal finite velocities also return exact ``0.0``.

    Args:
        radius_i: Particle i radius [m].
        radius_j: Particle j radius [m].
        velocity_i: Particle i settling velocity [m/s].
        velocity_j: Particle j settling velocity [m/s].

    Returns:
        Sedimentation pair rate [m³/s], or exact safe zero for an invalid or
        zero-rate calculation.

    References:
        Seinfeld, J. H., & Pandis, S. N. (2016). *Atmospheric Chemistry and
        Physics* (3rd ed.), Eq. 13A.4.
    """
    zero = wp.float64(0.0)
    if (
        not wp.isfinite(radius_i)
        or radius_i <= zero
        or not wp.isfinite(radius_j)
        or radius_j <= zero
        or not wp.isfinite(velocity_i)
        or velocity_i < zero
        or not wp.isfinite(velocity_j)
        or velocity_j < zero
    ):
        return zero
    radius_sum = radius_i + radius_j
    if not wp.isfinite(radius_sum) or radius_sum <= zero:
        return zero
    area = _PI * radius_sum * radius_sum
    if not wp.isfinite(area) or area <= zero:
        return zero
    relative_velocity = wp.abs(velocity_i - velocity_j)
    if not wp.isfinite(relative_velocity):
        return zero
    rate = area * relative_velocity
    if not wp.isfinite(rate) or rate <= zero or rate < _MIN_NORMAL_FLOAT64:
        return zero
    return rate


@wp.func
def kinematic_viscosity_wp(
    dynamic_viscosity: wp.float64,
    fluid_density: wp.float64,
) -> wp.float64:
    """Calculate kinematic viscosity from device-native SI transport inputs.

    This device-only helper applies ``nu = mu / rho`` for positive finite
    dynamic viscosity ``mu`` [Pa s] and fluid density ``rho`` [kg/m³].
    Public-input validation is intentionally deferred to the calling boundary.

    Args:
        dynamic_viscosity: Dynamic viscosity ``mu`` [Pa s].
        fluid_density: Fluid density ``rho`` [kg/m³].

    Returns:
        Kinematic viscosity ``nu`` [m²/s].
    """
    return dynamic_viscosity / fluid_density


@wp.func
def turbulent_shear_st1956_pair_rate_wp(
    radius_i: wp.float64,
    radius_j: wp.float64,
    turbulent_dissipation: wp.float64,
    kinematic_viscosity: wp.float64,
) -> wp.float64:
    """Calculate the ST1956 turbulent-shear pair rate in SI units.

    This device-only helper applies
    ``K = sqrt(pi * epsilon / (120 * nu)) * (2 r_i + 2 r_j)**3`` for finite
    positive radii ``r`` [m] and kinematic viscosity ``nu`` [m²/s], with finite
    non-negative turbulent dissipation ``epsilon`` [m²/s³]. Zero dissipation
    returns exact ``0.0`` before the diameter sum, preventing ``0 * inf`` for
    finite extreme radii. Callers must preflight derived transport and radius
    state; they must not use rate sanitization to conceal invalid physics.

    Args:
        radius_i: Particle i radius [m].
        radius_j: Particle j radius [m].
        turbulent_dissipation: Turbulent energy dissipation ``epsilon`` [m²/s³].
        kinematic_viscosity: Fluid kinematic viscosity ``nu`` [m²/s].

    Returns:
        Turbulent-shear pair rate [m³/s].

    References:
        Saffman, P. G., & Turner, J. S. (1956). On the collision of drops in
        turbulent clouds. *Journal of Fluid Mechanics*, 1(1), 16-30.
        Seinfeld, J. H., & Pandis, S. N. (2016). *Atmospheric Chemistry and
        Physics* (3rd ed.), Eq. 13A.2.
    """
    zero = wp.float64(0.0)
    if turbulent_dissipation == zero:
        return zero
    return wp.sqrt(
        _PI * turbulent_dissipation / (wp.float64(120.0) * kinematic_viscosity)
    ) * wp.pow(
        wp.float64(2.0) * radius_i + wp.float64(2.0) * radius_j,
        wp.float64(3.0),
    )


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
    pi_value = _PI
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
    pi_value = _PI
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
    * k_B * T)``. A non-positive radius sum or temperature, or a non-finite
    intermediate from finite extreme inputs, safely returns ``0.0``. Repulsion
    is lower-clipped at ``-200.0``.

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
        radius sum, temperature, or non-finite intermediate.

    References:
        Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced
        collisions in aerosols and dusty plasmas. *Physical Review E*, 85(2).
        https://doi.org/10.1103/PhysRevE.85.026410
    """
    sum_radius = radius_i + radius_j
    if sum_radius <= wp.float64(0.0) or temperature <= wp.float64(0.0):
        return wp.float64(0.0)

    pi_value = _PI
    charge_product = charge_i * charge_j
    if (
        not wp.isfinite(charge_i)
        or not wp.isfinite(charge_j)
        or not wp.isfinite(charge_product)
    ):
        # The hard-sphere limits only need the bounded potential ratio. Preserve
        # the finite-input sign when the intermediate product overflows.
        if wp.isfinite(charge_i) and wp.isfinite(charge_j):
            if charge_i * charge_j < wp.float64(0.0):
                return wp.float64(200.0)
            return wp.float64(-200.0)
        return wp.float64(0.0)
    charge_scale = elementary_charge_value * elementary_charge_value
    numerator = -charge_product * charge_scale
    denominator = wp.float64(4.0) * pi_value * electric_permittivity
    denominator = denominator * sum_radius
    denominator = denominator * boltzmann_constant
    denominator = denominator * temperature
    if (
        not wp.isfinite(numerator)
        or not wp.isfinite(denominator)
        or denominator <= wp.float64(0.0)
    ):
        return wp.float64(0.0)
    potential_ratio = numerator / denominator
    if not wp.isfinite(potential_ratio):
        if charge_product < wp.float64(0.0):
            return wp.float64(200.0)
        return wp.float64(-200.0)
    return wp.max(
        wp.min(potential_ratio, wp.float64(200.0)), wp.float64(-200.0)
    )


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
    if (
        coulomb_potential_ratio > -_COULOMB_SERIES_THRESHOLD
        and coulomb_potential_ratio < _COULOMB_SERIES_THRESHOLD
    ):
        potential_squared = coulomb_potential_ratio * coulomb_potential_ratio
        return (
            wp.float64(1.0)
            + coulomb_potential_ratio / wp.float64(2.0)
            + potential_squared / wp.float64(12.0)
        )
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
    if kinetic_limit < _KINETIC_LIMIT_CUTOFF:
        return wp.float64(0.0)

    continuum_limit = coulomb_continuum_limit_wp(coulomb_potential_ratio)
    numerator = (
        wp.sqrt(boltzmann_constant * temperature * reduced_mass)
        / reduced_friction
    )
    denominator = sum_radius * continuum_limit / kinetic_limit
    return numerator / denominator


@wp.func
def charged_hard_sphere_wp(  # noqa: C901
    radius_i: wp.float64,
    radius_j: wp.float64,
    mass_i: wp.float64,
    mass_j: wp.float64,
    charge_i: wp.float64,
    charge_j: wp.float64,
    temperature: wp.float64,
    pressure: wp.float64,
    boltzmann_constant: wp.float64,
    elementary_charge_value: wp.float64,
    electric_permittivity: wp.float64,
    gas_constant: wp.float64,
    molecular_weight_air: wp.float64,
    ref_viscosity: wp.float64,
    ref_temperature: wp.float64,
    sutherland_constant: wp.float64,
) -> wp.float64:
    """Calculate an internal charged hard-sphere pair rate in SI units.

    This device-only, concrete-module-only helper is the pair-rate primitive for
    charged-only and canonical Brownian-plus-charged particle-resolved low-level
    GPU coagulation. It receives each particle's total mass across species and
    is used for candidate rates and compact-active majorants. It ports the
    charged CPU calculation from
    ``particula.dynamics.coagulation.charged_dimensional_kernel`` and the
    hard-sphere fit from
    ``particula.dynamics.coagulation.charged_dimensionless_kernel``. It does
    not change public exports.

    Non-finite or non-positive radii, masses, thermodynamic state, or physical
    constants return exact safe zero. Charges must be finite but may be signed
    or zero. Invalid intermediates, finite-input intermediate overflow,
    overflow-derived signed zero, and underflow return exact safe zero rather
    than a non-finite or negative rate. Finite extreme repulsion is clipped at
    a potential ratio of ``-200.0`` and returns safe zero through the kinetic
    cutoff when its enhancement is below ``1e-80``.

    Args:
        radius_i: Radius of particle i [m]. Must be finite and positive.
        radius_j: Radius of particle j [m]. Must be finite and positive.
        mass_i: Mass of particle i [kg]. Must be finite and positive.
        mass_j: Mass of particle j [kg]. Must be finite and positive.
        charge_i: Signed charge of particle i in elementary-charge counts.
        charge_j: Signed charge of particle j in elementary-charge counts.
        temperature: Gas temperature [K]. Must be finite and positive.
        pressure: Gas pressure [Pa]. Must be finite and positive.
        boltzmann_constant: Boltzmann constant [J/K]. Must be finite and
            positive.
        elementary_charge_value: Elementary-charge magnitude [C]. Must be
            finite and positive.
        electric_permittivity: Vacuum electric permittivity [F/m]. Must be
            finite and positive.
        gas_constant: Universal gas constant [J/(mol K)]. Must be finite and
            positive.
        molecular_weight_air: Molar mass of air [kg/mol]. Must be finite and
            positive.
        ref_viscosity: Reference gas dynamic viscosity [Pa s]. Must be finite
            and positive.
        ref_temperature: Reference temperature for Sutherland viscosity [K].
            Must be finite and positive.
        sutherland_constant: Sutherland temperature constant [K]. Must be
            finite and positive.

    Returns:
        Finite, non-negative charged hard-sphere pair rate [m³/s], or exact
        ``0.0`` for invalid inputs, invalid or overflowed intermediates,
        finite extreme-repulsion clipping, or the kinetic cutoff.

    References:
        Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007).
        Coagulation of particles in the transition regime: The effect of the
        Coulomb potential. *Journal of Chemical Physics*, 126(12).
        https://doi.org/10.1063/1.2713719
    """
    zero = wp.float64(0.0)
    if (
        not wp.isfinite(radius_i)
        or radius_i <= zero
        or not wp.isfinite(radius_j)
        or radius_j <= zero
        or not wp.isfinite(mass_i)
        or mass_i <= zero
        or not wp.isfinite(mass_j)
        or mass_j <= zero
        or not wp.isfinite(charge_i)
        or not wp.isfinite(charge_j)
        or not wp.isfinite(temperature)
        or temperature <= zero
        or not wp.isfinite(pressure)
        or pressure <= zero
        or not wp.isfinite(boltzmann_constant)
        or boltzmann_constant <= zero
        or not wp.isfinite(elementary_charge_value)
        or elementary_charge_value <= zero
        or not wp.isfinite(electric_permittivity)
        or electric_permittivity <= zero
        or not wp.isfinite(gas_constant)
        or gas_constant <= zero
        or not wp.isfinite(molecular_weight_air)
        or molecular_weight_air <= zero
        or not wp.isfinite(ref_viscosity)
        or ref_viscosity <= zero
        or not wp.isfinite(ref_temperature)
        or ref_temperature <= zero
        or not wp.isfinite(sutherland_constant)
        or sutherland_constant <= zero
    ):
        return zero

    dynamic_viscosity = dynamic_viscosity_wp(
        temperature,
        ref_viscosity,
        ref_temperature,
        sutherland_constant,
    )
    if not wp.isfinite(dynamic_viscosity) or dynamic_viscosity <= zero:
        return zero

    mean_free_path = molecule_mean_free_path_wp(
        molecular_weight_air,
        temperature,
        pressure,
        dynamic_viscosity,
        gas_constant,
    )
    if not wp.isfinite(mean_free_path) or mean_free_path <= zero:
        return zero

    knudsen_i = knudsen_number_wp(mean_free_path, radius_i)
    knudsen_j = knudsen_number_wp(mean_free_path, radius_j)
    if (
        not wp.isfinite(knudsen_i)
        or knudsen_i <= zero
        or not wp.isfinite(knudsen_j)
        or knudsen_j <= zero
    ):
        return zero

    slip_i = cunningham_slip_correction_wp(knudsen_i)
    slip_j = cunningham_slip_correction_wp(knudsen_j)
    if (
        not wp.isfinite(slip_i)
        or slip_i <= zero
        or not wp.isfinite(slip_j)
        or slip_j <= zero
    ):
        return zero

    friction_i = friction_factor_wp(radius_i, dynamic_viscosity, slip_i)
    friction_j = friction_factor_wp(radius_j, dynamic_viscosity, slip_j)
    if (
        not wp.isfinite(friction_i)
        or friction_i <= zero
        or not wp.isfinite(friction_j)
        or friction_j <= zero
    ):
        return zero

    sum_radius = radius_i + radius_j
    if not wp.isfinite(sum_radius) or sum_radius <= zero:
        return zero
    potential_ratio = coulomb_potential_ratio_wp(
        radius_i,
        radius_j,
        charge_i,
        charge_j,
        temperature,
        boltzmann_constant,
        elementary_charge_value,
        electric_permittivity,
    )
    if not wp.isfinite(potential_ratio):
        return zero

    reduced_mass = reduced_value_wp(mass_i, mass_j)
    reduced_friction = reduced_value_wp(friction_i, friction_j)
    if (
        not wp.isfinite(reduced_mass)
        or reduced_mass <= zero
        or not wp.isfinite(reduced_friction)
        or reduced_friction <= zero
    ):
        return zero
    thermal_mass_product = boltzmann_constant * temperature * reduced_mass
    if not wp.isfinite(thermal_mass_product) or thermal_mass_product <= zero:
        return zero

    kinetic_limit = coulomb_kinetic_limit_wp(potential_ratio)
    if not wp.isfinite(kinetic_limit) or kinetic_limit < _KINETIC_LIMIT_CUTOFF:
        return zero
    continuum_limit = coulomb_continuum_limit_wp(potential_ratio)
    if not wp.isfinite(continuum_limit) or continuum_limit <= zero:
        return zero
    diffusive_denominator = sum_radius * continuum_limit / kinetic_limit
    if not wp.isfinite(diffusive_denominator) or diffusive_denominator <= zero:
        return zero

    diffusive_knudsen = diffusive_knudsen_number_wp(
        radius_i,
        radius_j,
        mass_i,
        mass_j,
        friction_i,
        friction_j,
        potential_ratio,
        temperature,
        boltzmann_constant,
    )
    if not wp.isfinite(diffusive_knudsen) or diffusive_knudsen < zero:
        return zero

    pi_value = _PI
    continuum = (
        wp.float64(4.0) * pi_value * diffusive_knudsen * diffusive_knudsen
    )
    numerator = (
        continuum
        + _HARD_SPHERE_LINEAR * wp.pow(diffusive_knudsen, wp.float64(3.0))
        + wp.sqrt(wp.float64(8.0) * pi_value)
        * _HARD_SPHERE_QUARTIC
        * wp.pow(diffusive_knudsen, wp.float64(4.0))
    )
    denominator = (
        wp.float64(1.0)
        + _HARD_SPHERE_DENOMINATOR_LINEAR * diffusive_knudsen
        + _HARD_SPHERE_DENOMINATOR_QUADRATIC
        * diffusive_knudsen
        * diffusive_knudsen
        + _HARD_SPHERE_QUARTIC * wp.pow(diffusive_knudsen, wp.float64(3.0))
    )
    if (
        not wp.isfinite(numerator)
        or numerator < zero
        or not wp.isfinite(denominator)
        or denominator <= zero
    ):
        return zero
    dimensionless_kernel = numerator / denominator
    if not wp.isfinite(dimensionless_kernel) or dimensionless_kernel < zero:
        return zero

    result = (
        dimensionless_kernel
        * reduced_friction
        * wp.pow(sum_radius, wp.float64(3.0))
        * kinetic_limit
        * kinetic_limit
        / (reduced_mass * continuum_limit)
    )
    if not wp.isfinite(result) or result <= zero:
        return zero
    return result
