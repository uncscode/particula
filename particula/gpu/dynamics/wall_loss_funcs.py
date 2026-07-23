"""Provide fp64 Warp wall-loss coefficient and private image-charge helpers.

The helpers calculate Crump-Seinfeld spherical and rectangular chamber
wall-loss coefficients in SI units [s^-1], plus private image-charge
enhancement primitives with dimensionless outputs. Coulomb self-potential
ratios are lower-clipped at -200, while image-charge exponents use the CPU
diagonal/self-pair absolute-value calculation and clip to [-50, 50]. P3 also
provides private electric-field drift and charged-coefficient composition
helpers. The helpers are device-only, have no public validation contract, and
direct-step integration remains deferred to P4; future step preflight owns
input validation.

Crump, J. G., & Seinfeld, J. H. (1981). Turbulent deposition and
gravitational sedimentation of an aerosol in a vessel of arbitrary shape.
*Journal of Aerosol Science*, 12(5).
https://doi.org/10.1016/0021-8502(81)90036-7

Crump, J. G., Flagan, R. C., & Seinfeld, J. H. (1982). Particle wall loss
rates in vessels. *Aerosol Science and Technology*, 2(3), 303-309.
https://doi.org/10.1080/02786828308958636
"""

import warp as wp

from particula.gpu.properties import (
    aerodynamic_mobility_wp,
    cunningham_slip_correction_wp,
    debye_1_wp,
    diffusion_coefficient_wp,
    dynamic_viscosity_wp,
    knudsen_number_wp,
    molecule_mean_free_path_wp,
    settling_velocity_stokes_from_transport_wp,
    x_coth_x_wp,
)

# Clip limits mirror the CPU Coulomb-ratio and image-charge authorities.
_PI = wp.constant(wp.float64(3.141592653589793))
_COULOMB_RATIO_LOWER_LIMIT = wp.constant(wp.float64(-200.0))
_IMAGE_CHARGE_EXPONENT_LOWER_LIMIT = wp.constant(wp.float64(-50.0))
_IMAGE_CHARGE_EXPONENT_UPPER_LIMIT = wp.constant(wp.float64(50.0))
_CHARGED_DRIFT_LOWER_GUARD = wp.constant(wp.float64(1.0e-30))
_FLOAT64_MAX = wp.constant(wp.float64(1.7976931348623157e308))


@wp.func
def spherical_wall_loss_coefficient_wp(
    wall_eddy_diffusivity: wp.float64,
    particle_radius: wp.float64,
    particle_density: wp.float64,
    temperature: wp.float64,
    pressure: wp.float64,
    chamber_radius: wp.float64,
    boltzmann_constant: wp.float64,
    gas_constant: wp.float64,
    molecular_weight_air: wp.float64,
    ref_viscosity: wp.float64,
    ref_temperature: wp.float64,
    sutherland_constant: wp.float64,
) -> wp.float64:
    """Calculate the neutral spherical wall-loss coefficient in s^-1.

    Combines diffusion-driven turbulent deposition with gravitational
    sedimentation using the Crump-Seinfeld spherical-chamber relation. This
    concrete Warp device helper assumes supported physical inputs and performs
    no public input validation or charged-particle correction.

    Args:
        wall_eddy_diffusivity: Wall eddy diffusivity in m^2/s.
        particle_radius: Particle radius in m.
        particle_density: Particle material density in kg/m^3.
        temperature: Gas temperature in K.
        pressure: Gas pressure in Pa.
        chamber_radius: Spherical chamber radius in m.
        boltzmann_constant: Boltzmann constant in J/K.
        gas_constant: Universal gas constant in J/(mol K).
        molecular_weight_air: Mean molecular weight of air in kg/mol.
        ref_viscosity: Reference dynamic viscosity in Pa s.
        ref_temperature: Reference temperature for viscosity in K.
        sutherland_constant: Sutherland temperature constant in K.

    Returns:
        Neutral particle wall-loss coefficient in s^-1.
    """
    dynamic_viscosity = dynamic_viscosity_wp(
        temperature,
        ref_viscosity,
        ref_temperature,
        sutherland_constant,
    )
    mean_free_path = molecule_mean_free_path_wp(
        molecular_weight_air,
        temperature,
        pressure,
        dynamic_viscosity,
        gas_constant,
    )
    knudsen_number = knudsen_number_wp(mean_free_path, particle_radius)
    slip_correction = cunningham_slip_correction_wp(knudsen_number)
    aerodynamic_mobility = aerodynamic_mobility_wp(
        particle_radius,
        slip_correction,
        dynamic_viscosity,
    )
    diffusion_coefficient = diffusion_coefficient_wp(
        temperature,
        aerodynamic_mobility,
        boltzmann_constant,
    )
    settling_velocity = settling_velocity_stokes_from_transport_wp(
        particle_radius,
        particle_density,
        dynamic_viscosity,
        mean_free_path,
    )
    transport_scale = wp.sqrt(wall_eddy_diffusivity * diffusion_coefficient)
    debye_argument = (
        wp.float64(3.141592653589793)
        * settling_velocity
        / (wp.float64(2.0) * transport_scale)
    )
    return wp.float64(6.0) * transport_scale / (
        wp.float64(3.141592653589793) * chamber_radius
    ) * debye_1_wp(debye_argument) + wp.float64(3.0) * settling_velocity / (
        wp.float64(4.0) * chamber_radius
    )


@wp.func
def rectangle_wall_loss_coefficient_wp(
    wall_eddy_diffusivity: wp.float64,
    particle_radius: wp.float64,
    particle_density: wp.float64,
    temperature: wp.float64,
    pressure: wp.float64,
    chamber_length: wp.float64,
    chamber_width: wp.float64,
    chamber_height: wp.float64,
    boltzmann_constant: wp.float64,
    gas_constant: wp.float64,
    molecular_weight_air: wp.float64,
    ref_viscosity: wp.float64,
    ref_temperature: wp.float64,
    sutherland_constant: wp.float64,
) -> wp.float64:
    """Calculate the neutral rectangular wall-loss coefficient in s^-1.

    Combines side-wall turbulent deposition with gravitational sedimentation
    using the Crump-Seinfeld rectangular-chamber relation. This concrete Warp
    device helper assumes supported physical inputs and performs no public
    input validation or charged-particle correction.

    Args:
        wall_eddy_diffusivity: Wall eddy diffusivity in m^2/s.
        particle_radius: Particle radius in m.
        particle_density: Particle material density in kg/m^3.
        temperature: Gas temperature in K.
        pressure: Gas pressure in Pa.
        chamber_length: Rectangular chamber length in m.
        chamber_width: Rectangular chamber width in m.
        chamber_height: Rectangular chamber height in m.
        boltzmann_constant: Boltzmann constant in J/K.
        gas_constant: Universal gas constant in J/(mol K).
        molecular_weight_air: Mean molecular weight of air in kg/mol.
        ref_viscosity: Reference dynamic viscosity in Pa s.
        ref_temperature: Reference temperature for viscosity in K.
        sutherland_constant: Sutherland temperature constant in K.

    Returns:
        Neutral particle wall-loss coefficient in s^-1.
    """
    dynamic_viscosity = dynamic_viscosity_wp(
        temperature,
        ref_viscosity,
        ref_temperature,
        sutherland_constant,
    )
    mean_free_path = molecule_mean_free_path_wp(
        molecular_weight_air,
        temperature,
        pressure,
        dynamic_viscosity,
        gas_constant,
    )
    knudsen_number = knudsen_number_wp(mean_free_path, particle_radius)
    slip_correction = cunningham_slip_correction_wp(knudsen_number)
    aerodynamic_mobility = aerodynamic_mobility_wp(
        particle_radius,
        slip_correction,
        dynamic_viscosity,
    )
    diffusion_coefficient = diffusion_coefficient_wp(
        temperature,
        aerodynamic_mobility,
        boltzmann_constant,
    )
    settling_velocity = settling_velocity_stokes_from_transport_wp(
        particle_radius,
        particle_density,
        dynamic_viscosity,
        mean_free_path,
    )
    transport_scale = wp.sqrt(wall_eddy_diffusivity * diffusion_coefficient)
    if transport_scale == wp.float64(0.0):
        return settling_velocity / chamber_height
    x = (
        wp.float64(3.141592653589793)
        * settling_velocity
        / (wp.float64(4.0) * transport_scale)
    )
    return (
        wp.float64(4.0)
        * transport_scale
        / wp.float64(3.141592653589793)
        * (
            wp.float64(1.0) / chamber_length
            + wp.float64(1.0) / chamber_width
            + x_coth_x_wp(x) / chamber_height
        )
    )


@wp.func
def _coulomb_self_potential_ratio_wp(
    particle_radius: wp.float64,
    particle_charge: wp.float64,
    temperature: wp.float64,
    elementary_charge_value: wp.float64,
    electric_permittivity: wp.float64,
    boltzmann_constant: wp.float64,
) -> wp.float64:
    """Calculate the dimensionless Coulomb self-potential ratio.

    Calculates the particle self-pair ratio and lower-clips it at -200. This
    private device helper expects finite physical constants and positive,
    finite radius and temperature. It performs no validation; a future direct
    wall-loss-step preflight owns validation.

    Args:
        particle_radius: Particle radius in m.
        particle_charge: Particle charge in elementary-charge units.
        temperature: Gas temperature in K.
        elementary_charge_value: Elementary charge in C.
        electric_permittivity: Electric permittivity in F/m.
        boltzmann_constant: Boltzmann constant in J/K.

    Returns:
        Dimensionless self-potential ratio, lower-clipped at -200.
    """
    raw_ratio = -(
        particle_charge
        * particle_charge
        * elementary_charge_value
        * elementary_charge_value
    ) / (
        wp.float64(4.0)
        * _PI
        * electric_permittivity
        * (particle_radius + particle_radius)
        * boltzmann_constant
        * temperature
    )
    return wp.max(raw_ratio, _COULOMB_RATIO_LOWER_LIMIT)


@wp.func
def _image_charge_enhancement_wp(
    particle_radius: wp.float64,
    particle_charge: wp.float64,
    temperature: wp.float64,
    elementary_charge_value: wp.float64,
    electric_permittivity: wp.float64,
    boltzmann_constant: wp.float64,
) -> wp.float64:
    """Calculate the dimensionless image-charge wall-loss enhancement.

    Returns exactly 1.0 for zero charge. Otherwise, lower-clips the Coulomb
    self-potential ratio at -200, takes its absolute value to match CPU
    diagonal/self-pair physics, clips the exponent to [-50, 50], and
    exponentiates it. This private device helper expects finite physical
    constants and positive, finite radius and temperature. It performs no
    validation; a future direct wall-loss-step preflight owns validation.

    Args:
        particle_radius: Particle radius in m.
        particle_charge: Particle charge in elementary-charge units.
        temperature: Gas temperature in K.
        elementary_charge_value: Elementary charge in C.
        electric_permittivity: Electric permittivity in F/m.
        boltzmann_constant: Boltzmann constant in J/K.

    Returns:
        Dimensionless image-charge enhancement factor with a clipped exponent.
    """
    if particle_charge == wp.float64(0.0):
        return wp.float64(1.0)
    ratio = _coulomb_self_potential_ratio_wp(
        particle_radius,
        particle_charge,
        temperature,
        elementary_charge_value,
        electric_permittivity,
        boltzmann_constant,
    )
    exponent = wp.min(
        wp.max(
            wp.abs(ratio),
            _IMAGE_CHARGE_EXPONENT_LOWER_LIMIT,
        ),
        _IMAGE_CHARGE_EXPONENT_UPPER_LIMIT,
    )
    return wp.exp(exponent)


@wp.func
def _geometry_scale_wp(
    geometry_mode: wp.int32,
    chamber_radius: wp.float64,
    chamber_length: wp.float64,
    chamber_width: wp.float64,
    chamber_height: wp.float64,
) -> wp.float64:
    """Return the unguarded characteristic chamber scale in m.

    Spherical chambers use their radius; rectangular chambers use their
    smallest dimension. Field resolution deliberately uses this unguarded
    scale to match the CPU charged-wall-loss strategy.

    Args:
        geometry_mode: Zero for spherical geometry; nonzero for rectangular.
        chamber_radius: Spherical chamber radius in m.
        chamber_length: Rectangular chamber length in m.
        chamber_width: Rectangular chamber width in m.
        chamber_height: Rectangular chamber height in m.

    Returns:
        Characteristic geometry scale in m without a lower-bound guard.
    """
    if geometry_mode == wp.int32(0):
        return chamber_radius
    return wp.min(chamber_length, wp.min(chamber_width, chamber_height))


@wp.func
def _resolve_spherical_electric_field_wp(
    wall_electric_field: wp.float64,
    wall_potential: wp.float64,
    geometry_scale: wp.float64,
) -> wp.float64:
    """Resolve a signed spherical electric field in V/m.

    Adds the potential-derived field only for a nonzero potential and positive
    geometry scale. The supplied scalar field remains signed.

    Args:
        wall_electric_field: Signed scalar electric field in V/m.
        wall_potential: Wall potential in V.
        geometry_scale: Unguarded characteristic chamber scale in m.

    Returns:
        Signed resolved electric field in V/m.
    """
    resolved_field = wall_electric_field
    if wall_potential != wp.float64(0.0) and geometry_scale > wp.float64(0.0):
        resolved_field = resolved_field + wall_potential / geometry_scale
    return resolved_field


@wp.func
def _resolve_rectangular_electric_field_wp(
    field_x: wp.float64,
    field_y: wp.float64,
    field_z: wp.float64,
    wall_potential: wp.float64,
    geometry_scale: wp.float64,
) -> wp.float64:
    """Resolve a rectangular electric-field magnitude in V/m.

    Computes the Euclidean magnitude of the three supplied field components,
    then conditionally adds the potential-derived field. A signed potential
    may therefore decrease the resolved magnitude.

    Args:
        field_x: X electric-field component in V/m.
        field_y: Y electric-field component in V/m.
        field_z: Z electric-field component in V/m.
        wall_potential: Wall potential in V.
        geometry_scale: Unguarded characteristic chamber scale in m.

    Returns:
        Resolved electric field in V/m.
    """
    resolved_field = wp.sqrt(
        field_x * field_x + field_y * field_y + field_z * field_z
    )
    if wall_potential != wp.float64(0.0) and geometry_scale > wp.float64(0.0):
        resolved_field = resolved_field + wall_potential / geometry_scale
    return resolved_field


@wp.func
def _electric_field_drift_wp(
    particle_radius: wp.float64,
    particle_charge: wp.float64,
    temperature: wp.float64,
    resolved_electric_field: wp.float64,
    geometry_scale: wp.float64,
    elementary_charge_value: wp.float64,
    ref_viscosity: wp.float64,
    ref_temperature: wp.float64,
    sutherland_constant: wp.float64,
) -> wp.float64:
    """Calculate signed electric-field drift contribution in s^-1.

    Uses Sutherland dynamic viscosity and charge mobility. Particle radius and
    the drift denominator are lower-guarded at 1e-30; NaN drift is mapped to
    zero while finite signed values and infinities are retained for final
    coefficient composition.

    Args:
        particle_radius: Particle radius in m.
        particle_charge: Particle charge in elementary-charge units.
        temperature: Gas temperature in K.
        resolved_electric_field: Geometry-resolved electric field in V/m.
        geometry_scale: Characteristic chamber scale in m.
        elementary_charge_value: Elementary charge in C.
        ref_viscosity: Reference dynamic viscosity in Pa s.
        ref_temperature: Reference temperature for viscosity in K.
        sutherland_constant: Sutherland temperature constant in K.

    Returns:
        Signed electric-field drift contribution in s^-1.
    """
    if particle_charge == wp.float64(
        0.0
    ) or resolved_electric_field == wp.float64(0.0):
        return wp.float64(0.0)
    viscosity = dynamic_viscosity_wp(
        temperature,
        ref_viscosity,
        ref_temperature,
        sutherland_constant,
    )
    diameter = wp.float64(2.0) * wp.max(
        particle_radius,
        _CHARGED_DRIFT_LOWER_GUARD,
    )
    mobility = (
        wp.abs(particle_charge)
        * elementary_charge_value
        / (wp.float64(3.0) * _PI * viscosity * diameter)
    )
    drift = (
        mobility
        * resolved_electric_field
        * wp.sign(particle_charge)
        / wp.max(geometry_scale, _CHARGED_DRIFT_LOWER_GUARD)
    )
    if wp.isnan(drift):
        return wp.float64(0.0)
    return drift


@wp.func
def _combine_charged_wall_loss_coefficient_wp(
    neutral_coefficient: wp.float64,
    electrostatic_factor: wp.float64,
    drift_term: wp.float64,
) -> wp.float64:
    """Combine and sanitize charged wall-loss coefficient contributions.

    Calculates ``neutral_coefficient * electrostatic_factor + drift_term``.
    NaN and nonpositive results map to zero, while positive overflow and
    infinity map to the largest finite float64 value.

    Args:
        neutral_coefficient: Neutral wall-loss coefficient in s^-1.
        electrostatic_factor: Dimensionless image-charge enhancement factor.
        drift_term: Signed electric-field drift contribution in s^-1.

    Returns:
        Finite nonnegative charged wall-loss coefficient in s^-1.
    """
    combined = neutral_coefficient * electrostatic_factor + drift_term
    if wp.isnan(combined) or combined <= wp.float64(0.0):
        return wp.float64(0.0)
    if wp.isinf(combined) or combined > _FLOAT64_MAX:
        return _FLOAT64_MAX
    return combined
