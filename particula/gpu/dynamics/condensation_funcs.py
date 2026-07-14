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


@wp.func
def _thermal_conductivity_wp(temperature: wp.float64) -> wp.float64:
    """Calculate air thermal conductivity [W/(m·K)].

    Port of Seinfeld and Pandis, Equation 17.54.
    """
    return wp.float64(1e-3) * (
        wp.float64(4.39) + wp.float64(0.071) * temperature
    )


@wp.func
def _thermal_resistance_factor_wp(
    diffusion_coefficient: wp.float64,
    latent_heat: wp.float64,
    vapor_pressure_surface: wp.float64,
    thermal_conductivity: wp.float64,
    temperature: wp.float64,
    molar_mass: wp.float64,
    gas_constant: wp.float64,
) -> wp.float64:
    """Calculate the latent-heat thermal resistance factor [J/kg].

    Port of Topping and Bane (2022), Equation 2.36.
    """
    r_specific = gas_constant / molar_mass
    diffusion_term = (
        diffusion_coefficient
        * latent_heat
        * vapor_pressure_surface
        / temperature
        / thermal_conductivity
    )
    correction_term = latent_heat / temperature / r_specific - wp.float64(1.0)
    return diffusion_term * correction_term + r_specific * temperature


@wp.func
def _mass_transfer_rate_latent_heat_wp(
    pressure_delta: wp.float64,
    first_order_mass_transport: wp.float64,
    temperature: wp.float64,
    molar_mass: wp.float64,
    latent_heat: wp.float64,
    thermal_conductivity: wp.float64,
    vapor_pressure_surface: wp.float64,
    diffusion_coefficient: wp.float64,
    gas_constant: wp.float64,
) -> wp.float64:
    """Calculate latent-corrected condensation mass transfer [kg/s].

    Port of Topping and Bane (2022), Equation 2.36. Zero latent heat returns
    the isothermal helper result exactly.
    """
    if latent_heat == wp.float64(0.0):
        return mass_transfer_rate_wp(
            pressure_delta,
            first_order_mass_transport,
            temperature,
            molar_mass,
            gas_constant,
        )
    thermal_factor = _thermal_resistance_factor_wp(
        diffusion_coefficient,
        latent_heat,
        vapor_pressure_surface,
        thermal_conductivity,
        temperature,
        molar_mass,
        gas_constant,
    )
    r_specific = gas_constant / molar_mass
    correction = thermal_factor / (r_specific * temperature)
    return (
        mass_transfer_rate_wp(
            pressure_delta,
            first_order_mass_transport,
            temperature,
            molar_mass,
            gas_constant,
        )
        / correction
    )


@wp.func
def particle_radius_from_volume_wp(total_volume: wp.float64) -> wp.float64:
    """Compute particle radius from total volume.

    Args:
        total_volume: Particle total volume [m^3].

    Returns:
        Particle radius [m].
    """
    pi_value = wp.float64(3.141592653589793)
    numerator = wp.float64(3.0) * total_volume
    denominator = wp.float64(4.0) * pi_value
    return wp.pow(numerator / denominator, wp.float64(1.0) / wp.float64(3.0))


@wp.func
def effective_surface_tension_wp(
    masses: wp.array3d[wp.float64],
    densities: wp.array[wp.float64],
    surface_tensions: wp.array[wp.float64],
    box_idx: int,
    particle_idx: int,
    requested_species_idx: int,
    composition_weighted: bool,
) -> wp.float64:
    """Return static or composition-weighted surface tension for one particle.

    In static mode, immediately return the requested species tension [N/m]
    without reading ``masses`` or ``densities``. In composition-weighted mode,
    ignore ``requested_species_idx`` and calculate the single-phase value as
    ``sum(surface_tension * mass / density) / sum(mass / density)`` across the
    fixed ``n_species`` axis. If the total volume is zero, return the
    unweighted arithmetic mean of all supplied species tensions instead.

    Args:
        masses: Species masses [kg] with fixed shape
            ``(n_boxes, n_particles, n_species)``.
        densities: Species densities [kg/m³] with fixed shape ``(n_species,)``.
        surface_tensions: Species surface tensions [N/m] with fixed shape
            ``(n_species,)``.
        box_idx: Index of the particle box.
        particle_idx: Index of the particle within ``box_idx``.
        requested_species_idx: Species index used only in static mode.
        composition_weighted: Selects composition-weighted mode when true;
            otherwise selects static mode.

    Returns:
        Effective surface tension [N/m].
    """
    if not composition_weighted:
        return surface_tensions[requested_species_idx]

    total_volume = wp.float64(0.0)
    tension_volume_sum = wp.float64(0.0)
    tension_sum = wp.float64(0.0)
    for species_idx in range(masses.shape[2]):  # type: ignore[attr-defined]
        species_volume = (
            masses[box_idx, particle_idx, species_idx]  # type: ignore[index]
            / densities[species_idx]
        )
        total_volume += species_volume
        tension_volume_sum += surface_tensions[species_idx] * species_volume
        tension_sum += surface_tensions[species_idx]

    if total_volume == wp.float64(0.0):
        return tension_sum / wp.float64(
            masses.shape[2]  # type: ignore[attr-defined]
        )
    return tension_volume_sum / total_volume


@wp.func
def water_activity_ideal_wp(
    masses: wp.array3d[wp.float64],
    molar_masses: wp.array[wp.float64],
    box_idx: int,
    particle_idx: int,
    water_index: int,
) -> wp.float64:
    """Calculate ideal water activity for one particle from its mole fraction.

    This Warp helper mirrors
    ``particula.particles.properties.activity_module.get_ideal_activity_molar``.
    It returns zero when the selected particle has zero total moles.

    Args:
        masses: Species masses [kg], shaped
            ``(n_boxes, n_particles, n_species)``.
        molar_masses: Species molar masses [kg/mol].
        box_idx: Index of the particle box (dimensionless).
        particle_idx: Particle index in the box.
        water_index: Index of the water species (dimensionless).

    Returns:
        Dimensionless water mole fraction, or zero for zero total moles.
    """
    total_moles = wp.float64(0.0)
    for species_idx in range(masses.shape[2]):  # type: ignore[attr-defined]
        total_moles += (
            masses[box_idx, particle_idx, species_idx]  # type: ignore[index]
            / molar_masses[species_idx]
        )

    if total_moles == wp.float64(0.0):
        return wp.float64(0.0)

    water_moles = (
        masses[box_idx, particle_idx, water_index]  # type: ignore[index]
        / molar_masses[water_index]
    )
    return water_moles / total_moles


@wp.func
def water_activity_kappa_wp(
    masses: wp.array3d[wp.float64],
    densities: wp.array[wp.float64],
    kappas: wp.array[wp.float64],
    box_idx: int,
    particle_idx: int,
    water_index: int,
) -> wp.float64:
    """Calculate κ-model water activity for one particle.

    This Warp helper mirrors
    ``particula.particles.properties.activity_module.get_kappa_activity``.
    It returns zero when water volume is zero and one when solute volume is
    zero.

    Args:
        masses: Species masses [kg], shaped
            ``(n_boxes, n_particles, n_species)``.
        densities: Species densities [kg/m³].
        kappas: Species hygroscopicity parameters (dimensionless).
        box_idx: Index of the particle box (dimensionless).
        particle_idx: Particle index in the box.
        water_index: Index of the water species (dimensionless).

    Returns:
        Dimensionless water activity.

    References:
        Petters, M. D., & Kreidenweis, S. M. (2007). A single parameter
        representation of hygroscopic growth and cloud condensation nucleus
        activity. Atmospheric Chemistry and Physics, 7(8), 1961-1971.
        https://doi.org/10.5194/acp-7-1961-2007
    """
    water_volume = wp.float64(0.0)
    solute_volume = wp.float64(0.0)
    kappa_times_solute_volume = wp.float64(0.0)
    for species_idx in range(masses.shape[2]):  # type: ignore[attr-defined]
        species_volume = (
            masses[box_idx, particle_idx, species_idx]  # type: ignore[index]
            / densities[species_idx]
        )
        if species_idx == water_index:
            water_volume = species_volume
        else:
            solute_volume += species_volume
            kappa_times_solute_volume += kappas[species_idx] * species_volume

    if water_volume == wp.float64(0.0):
        return wp.float64(0.0)
    if solute_volume == wp.float64(0.0):
        return wp.float64(1.0)
    return wp.float64(1.0) / (
        wp.float64(1.0) + kappa_times_solute_volume / water_volume
    )
