"""Warp GPU implementations of gas property functions.

These functions mirror the NumPy implementations in
``particula.gas.properties`` for use inside Warp kernels.
"""

import warp as wp


@wp.func
def dynamic_viscosity_wp(
    temperature: wp.float64,
    ref_viscosity: wp.float64,
    ref_temperature: wp.float64,
    sutherland_constant: wp.float64,
) -> wp.float64:
    """Calculate dynamic viscosity with Sutherland's formula.

    Port of
    ``particula.gas.properties.dynamic_viscosity.get_dynamic_viscosity``.

    Args:
        temperature: Gas temperature [K].
        ref_viscosity: Reference viscosity at ref_temperature [Pa·s].
        ref_temperature: Reference temperature [K].
        sutherland_constant: Sutherland constant [K].

    Returns:
        Dynamic viscosity [Pa·s].
    """
    return (
        ref_viscosity
        * wp.pow(temperature / ref_temperature, wp.float64(1.5))
        * (ref_temperature + sutherland_constant)
        / (temperature + sutherland_constant)
    )


@wp.func
def molecule_mean_free_path_wp(
    molar_mass: wp.float64,
    temperature: wp.float64,
    pressure: wp.float64,
    dynamic_viscosity: wp.float64,
    gas_constant: wp.float64,
) -> wp.float64:
    """Calculate mean free path for a gas molecule.

    Port of
    ``particula.gas.properties.mean_free_path.get_molecule_mean_free_path``.

    Args:
        molar_mass: Molar mass of the gas [kg/mol].
        temperature: Gas temperature [K].
        pressure: Gas pressure [Pa].
        dynamic_viscosity: Dynamic viscosity [Pa·s].
        gas_constant: Universal gas constant [J/(mol·K)].

    Returns:
        Mean free path of a gas molecule [m].
    """
    pi_value = wp.float64(3.141592653589793)
    numerator = wp.float64(8.0) * molar_mass
    denominator = wp.sqrt(numerator / (pi_value * gas_constant * temperature))
    return (wp.float64(2.0) * dynamic_viscosity / pressure) / denominator


@wp.func
def partial_pressure_wp(
    concentration: wp.float64,
    molar_mass: wp.float64,
    temperature: wp.float64,
    gas_constant: wp.float64,
) -> wp.float64:
    """Calculate partial pressure from concentration and temperature.

    Port of ``particula.gas.properties.pressure_function.get_partial_pressure``.

    Args:
        concentration: Gas concentration [kg/m³].
        molar_mass: Molar mass [kg/mol].
        temperature: Gas temperature [K].
        gas_constant: Universal gas constant [J/(mol·K)].

    Returns:
        Partial pressure [Pa].
    """
    return (concentration * gas_constant * temperature) / molar_mass
