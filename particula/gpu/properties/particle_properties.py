"""Device-native fp64 particle transport and thermodynamic properties.

The ``@wp.func`` helpers provide scalar SI-unit property calculations for Warp
kernels. Alongside GPU counterparts of CPU particle properties, this module
owns the reusable neutral transport primitives and dimensionless wall-loss
geometry factors used by GPU dynamics.
"""

import warp as wp

from particula.gpu.properties.gas_properties import (
    dynamic_viscosity_wp,
    molecule_mean_free_path_wp,
)
from particula.util.constants import STANDARD_GRAVITY

_PI = wp.constant(wp.float64(3.141592653589793))
_CUNNINGHAM_CONSTANT = wp.constant(wp.float64(1.257))
_CUNNINGHAM_EXPONENTIAL_SCALE = wp.constant(wp.float64(0.4))
_CUNNINGHAM_EXPONENTIAL_RATE = wp.constant(wp.float64(1.1))
_STANDARD_GRAVITY = wp.constant(wp.float64(STANDARD_GRAVITY))
_MIN_NORMAL_FLOAT64 = wp.constant(wp.float64(2.2250738585072014e-308))
_PI_SQUARED_OVER_SIX = wp.constant(wp.float64(1.6449340668482264))


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
    """Calculate the Cunningham slip correction from Knudsen number.

    Evaluates ``C_c = 1 + Kn * (1.257 + 0.4 * exp(-1.1 / Kn))``. The
    continuum limit at ``Kn = 0`` is exactly ``1.0``; exponential underflow
    for small positive ``Kn`` is an intended path. Negative and non-finite
    inputs return exact ``0.0`` before division.

    Args:
        knudsen_number: Particle Knudsen number ``Kn`` (dimensionless).

    Returns:
        Dimensionless slip correction, or the documented exact limit or
        invalid-input sentinel.

    References:
        Seinfeld, J. H., & Pandis, S. N. (2016). *Atmospheric Chemistry and
        Physics* (3rd ed.). John Wiley & Sons.
    """
    if knudsen_number == wp.float64(0.0):
        return wp.float64(1.0)
    if not wp.isfinite(knudsen_number) or knudsen_number < wp.float64(0.0):
        return wp.float64(0.0)
    return wp.float64(1.0) + knudsen_number * (
        _CUNNINGHAM_CONSTANT
        + _CUNNINGHAM_EXPONENTIAL_SCALE
        * wp.exp(
            wp.float64(-1.0) * _CUNNINGHAM_EXPONENTIAL_RATE / knudsen_number
        )
    )


@wp.func
def particle_radius_from_volume_wp(total_volume: wp.float64) -> wp.float64:
    """Calculate a spherical particle radius from total volume.

    Evaluates ``r = (3 V / (4 pi))**(1/3)`` for volume ``V``. Non-finite and
    non-positive volumes return exact ``0.0`` before the cube root.

    Args:
        total_volume: Total spherical-equivalent particle volume ``V`` [m³].

    Returns:
        Spherical-equivalent particle radius [m], or exact ``0.0`` for an
        invalid volume.
    """
    if not wp.isfinite(total_volume) or total_volume <= wp.float64(0.0):
        return wp.float64(0.0)
    return wp.pow(
        wp.float64(3.0) * total_volume / (wp.float64(4.0) * _PI),
        wp.float64(1.0) / wp.float64(3.0),
    )


@wp.func
def diffusion_coefficient_wp(
    temperature: wp.float64,
    aerodynamic_mobility: wp.float64,
    boltzmann_constant: wp.float64,
) -> wp.float64:
    """Calculate Stokes-Einstein diffusion from mobility.

    Evaluates ``D = k_B T B``. This low-level device primitive intentionally
    performs no domain checks; callers must provide physically valid SI inputs.

    Args:
        temperature: Gas temperature ``T`` [K].
        aerodynamic_mobility: Particle mobility ``B`` [s/kg].
        boltzmann_constant: Boltzmann constant ``k_B`` [J/K].

    Returns:
        Particle diffusion coefficient ``D`` [m²/s].
    """
    return boltzmann_constant * temperature * aerodynamic_mobility


@wp.func
def effective_density_wp(
    total_mass: wp.float64,
    total_volume: wp.float64,
) -> wp.float64:
    """Calculate a safe spherical-equivalent mixture density.

    Evaluates ``rho_eff = m / V`` from total mass and volume. Non-finite,
    non-positive, subnormal, or overflowed inputs and results return exact
    ``0.0``.

    Args:
        total_mass: Total particle mass [kg].
        total_volume: Total particle volume [m³].

    Returns:
        Effective mixture density [kg/m³], or exact ``0.0`` for an invalid or
        non-normal calculation.
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
def settling_velocity_stokes_from_transport_wp(
    particle_radius: wp.float64,
    effective_density: wp.float64,
    dynamic_viscosity: wp.float64,
    mean_free_path: wp.float64,
) -> wp.float64:
    """Calculate safe slip-corrected Stokes settling velocity.

    Evaluates ``v_s = 2 r² rho_eff C_c g / (9 mu)`` from precomputed gas
    transport properties and standard gravity. Invalid, non-positive,
    subnormal, or non-finite intermediates and results return exact ``0.0``.

    Args:
        particle_radius: Particle radius [m].
        effective_density: Effective particle density [kg/m³].
        dynamic_viscosity: Dynamic gas viscosity [Pa s].
        mean_free_path: Molecular mean free path [m].

    Returns:
        Downward settling velocity [m/s], or exact ``0.0`` for an invalid or
        non-normal calculation.

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
    radius_squared = particle_radius * particle_radius
    numerator = (
        wp.float64(2.0)
        * radius_squared
        * effective_density
        * slip_correction
        * _STANDARD_GRAVITY
    )
    velocity = numerator / (wp.float64(9.0) * dynamic_viscosity)
    if (
        not wp.isfinite(slip_correction)
        or slip_correction <= zero
        or not wp.isfinite(radius_squared)
        or radius_squared <= zero
        or not wp.isfinite(numerator)
        or numerator <= zero
        or not wp.isfinite(velocity)
        or velocity <= zero
        or velocity < _MIN_NORMAL_FLOAT64
    ):
        return zero
    return velocity


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
    """Calculate safe slip-corrected Stokes settling velocity from gas state.

    Derives viscosity, molecular mean free path, Knudsen number, and slip
    correction from the supplied SI gas state before evaluating
    ``v_s = 2 r² rho_eff C_c g / (9 mu)``. Invalid inputs or derived transport
    return exact ``0.0``.

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
        Downward settling velocity [m/s], or exact ``0.0`` for invalid input
        or derived transport.

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
        temperature, ref_viscosity, ref_temperature, sutherland_constant
    )
    mean_free_path = molecule_mean_free_path_wp(
        molecular_weight_air,
        temperature,
        pressure,
        dynamic_viscosity,
        gas_constant,
    )
    if not wp.isfinite(dynamic_viscosity) or dynamic_viscosity <= zero:
        return zero
    if not wp.isfinite(mean_free_path) or mean_free_path <= zero:
        return zero
    return settling_velocity_stokes_from_transport_wp(
        particle_radius, effective_density, dynamic_viscosity, mean_free_path
    )


@wp.func
def _debye_integrand_wp(value: wp.float64) -> wp.float64:
    """Evaluate the removable Debye integrand ``t / (exp(t) - 1)``."""
    if value == wp.float64(0.0):
        return wp.float64(1.0)
    return value / (wp.exp(value) - wp.float64(1.0))


@wp.func
def _debye_gauss_pair_wp(
    half_x: wp.float64,
    midpoint: wp.float64,
    node: wp.float64,
    weight: wp.float64,
) -> wp.float64:
    """Return one symmetric Gauss--Legendre pair contribution."""
    return weight * (
        _debye_integrand_wp(midpoint - half_x * node)
        + _debye_integrand_wp(midpoint + half_x * node)
    )


@wp.func
def debye_1_wp(x: wp.float64) -> wp.float64:
    """Calculate the dimensionless first Debye wall-loss geometry factor.

    Evaluates ``D_1(x) = integral_0^x[t / (exp(t) - 1)] dt / x``. Its zero
    limit is exactly ``1.0``; negative and non-finite inputs return exact
    ``0.0``. The implementation uses a Bernoulli series for ``x <= 1``,
    32-node Gauss--Legendre quadrature for ``1 < x < 20``, and the
    finite-tail-corrected ``pi² / (6 x)`` asymptote at larger ``x``.

    Args:
        x: Dimensionless spherical wall-loss geometry argument.

    Returns:
        Dimensionless ``D_1(x)`` value, or the documented exact limit or
        invalid-input sentinel.

    References:
        Crump, J. G., Flagan, R. C., & Seinfeld, J. H. (1982). Particle wall
        loss rates in vessels. *Aerosol Science and Technology*, 2(3), 303-309.
        https://doi.org/10.1080/02786828308958636
    """
    zero = wp.float64(0.0)
    if x == zero:
        return wp.float64(1.0)
    if not wp.isfinite(x) or x < zero:
        return zero
    if x <= wp.float64(1.0):
        x2 = x * x
        return (
            wp.float64(1.0)
            - x / wp.float64(4.0)
            + x2 / wp.float64(36.0)
            - x2 * x2 / wp.float64(3600.0)
            + x2 * x2 * x2 / wp.float64(211680.0)
            - x2 * x2 * x2 * x2 / wp.float64(10886400.0)
            + x2 * x2 * x2 * x2 * x2 / wp.float64(526901760.0)
            - wp.float64(691.0)
            * x2
            * x2
            * x2
            * x2
            * x2
            * x2
            / wp.float64(16999766784000.0)
            + wp.float64(7.0)
            * x2
            * x2
            * x2
            * x2
            * x2
            * x2
            * x2
            / wp.float64(7846048170000.0)
        )
    if x >= wp.float64(20.0):
        return (_PI_SQUARED_OVER_SIX - (x + wp.float64(1.0)) * wp.exp(-x)) / x
    half_x = x / wp.float64(2.0)
    integral = _debye_gauss_pair_wp(
        half_x,
        half_x,
        wp.float64(0.0483076656877383),
        wp.float64(0.0965400885147278),
    )
    integral += _debye_gauss_pair_wp(
        half_x,
        half_x,
        wp.float64(0.1444719615827965),
        wp.float64(0.0956387200792749),
    )
    integral += _debye_gauss_pair_wp(
        half_x,
        half_x,
        wp.float64(0.2392873622521371),
        wp.float64(0.0938443990808046),
    )
    integral += _debye_gauss_pair_wp(
        half_x,
        half_x,
        wp.float64(0.3318686022821277),
        wp.float64(0.0911738786957639),
    )
    integral += _debye_gauss_pair_wp(
        half_x,
        half_x,
        wp.float64(0.4213512761306353),
        wp.float64(0.0876520930044038),
    )
    integral += _debye_gauss_pair_wp(
        half_x,
        half_x,
        wp.float64(0.5068999089322294),
        wp.float64(0.0833119242269468),
    )
    integral += _debye_gauss_pair_wp(
        half_x,
        half_x,
        wp.float64(0.5877157572407623),
        wp.float64(0.0781938957870703),
    )
    integral += _debye_gauss_pair_wp(
        half_x,
        half_x,
        wp.float64(0.6630442669302152),
        wp.float64(0.0723457941088485),
    )
    integral += _debye_gauss_pair_wp(
        half_x,
        half_x,
        wp.float64(0.7321821187402897),
        wp.float64(0.0658222227763618),
    )
    integral += _debye_gauss_pair_wp(
        half_x,
        half_x,
        wp.float64(0.7944837959679424),
        wp.float64(0.0586840934785355),
    )
    integral += _debye_gauss_pair_wp(
        half_x,
        half_x,
        wp.float64(0.84936761373257),
        wp.float64(0.0509980592623762),
    )
    integral += _debye_gauss_pair_wp(
        half_x,
        half_x,
        wp.float64(0.8963211557660521),
        wp.float64(0.0428358980222267),
    )
    integral += _debye_gauss_pair_wp(
        half_x,
        half_x,
        wp.float64(0.9349060759377397),
        wp.float64(0.0342738629130214),
    )
    integral += _debye_gauss_pair_wp(
        half_x,
        half_x,
        wp.float64(0.9647622555875064),
        wp.float64(0.0253920653092621),
    )
    integral += _debye_gauss_pair_wp(
        half_x,
        half_x,
        wp.float64(0.9856115115452684),
        wp.float64(0.0162743947309057),
    )
    integral += _debye_gauss_pair_wp(
        half_x,
        half_x,
        wp.float64(0.9972638618494816),
        wp.float64(0.0070186100094701),
    )
    return half_x * integral / x


@wp.func
def x_coth_x_wp(x: wp.float64) -> wp.float64:
    """Calculate the rectangular wall-loss factor ``x / tanh(x)``.

    The zero limit is exactly ``1.0``; negative and non-finite inputs return
    exact ``0.0``. A series through ``x**6`` for ``0 < x <= 1e-3`` avoids
    cancellation. The factor supports the rearranged rectangular term
    ``(4 * sqrt(K * D) / pi) * x_coth_x_wp(x)``.

    Args:
        x: Dimensionless rectangular wall-loss geometry argument.

    Returns:
        Dimensionless ``x / tanh(x)`` value, or the documented exact limit or
        invalid-input sentinel.

    References:
        Crump, J. G., Flagan, R. C., & Seinfeld, J. H. (1982). Particle wall
        loss rates in vessels. *Aerosol Science and Technology*, 2(3), 303-309.
        https://doi.org/10.1080/02786828308958636
    """
    zero = wp.float64(0.0)
    if x == zero:
        return wp.float64(1.0)
    if not wp.isfinite(x) or x < zero:
        return zero
    if x <= wp.float64(1.0e-3):
        x2 = x * x
        return (
            wp.float64(1.0)
            + x2 / wp.float64(3.0)
            - x2 * x2 / wp.float64(45.0)
            + wp.float64(2.0) * x2 * x2 * x2 / wp.float64(945.0)
        )
    return x / wp.tanh(x)


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
    pi_value = _PI
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
    pi_value = _PI
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
    pi_value = _PI
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
