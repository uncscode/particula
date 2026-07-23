"""Neutral fp64 Warp wall-loss coefficient device helpers.

The helpers calculate Crump-Seinfeld spherical and rectangular chamber
wall-loss coefficients in SI units [s^-1]. They are concrete device helpers
only and intentionally exclude charged-particle physics and public validation.

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
