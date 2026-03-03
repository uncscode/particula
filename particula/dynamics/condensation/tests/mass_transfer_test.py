"""Test the Condensation module."""

import numpy as np
import pytest
from particula.dynamics.condensation.mass_transfer import (
    get_first_order_mass_transport_k,
    get_mass_transfer,
    get_mass_transfer_of_multiple_species,
    get_mass_transfer_of_single_species,
    get_mass_transfer_rate,
    get_radius_transfer_rate,
    get_thermal_resistance_factor,
)
from particula.util.constants import GAS_CONSTANT


def test_first_order_mass_transport_k():
    """Test the first_order_mass_transport_k function."""
    radius = 1e-6
    vapor_transition = 0.6
    diffusion_coefficient = 2e-9
    expected_result = 1.5079644737231005e-14
    result = get_first_order_mass_transport_k(
        radius, vapor_transition, diffusion_coefficient
    )
    np.testing.assert_allclose(result, expected_result, rtol=1e-8)


def test_multi_radius_first_order_mass_transport_k():
    """Test the first_order_mass_transport_k function for multiple radii."""
    radius = np.array([1e-6, 2e-6, 3e-6])
    vapor_transition = 0.6
    diffusion_coefficient = 2e-9
    expected_result = np.array([1.50796447e-14, 3.01592895e-14, 4.52389342e-14])
    result = get_first_order_mass_transport_k(
        radius, vapor_transition, diffusion_coefficient
    )
    np.testing.assert_allclose(result, expected_result, rtol=1e-8)


def test_mass_transfer_rate():
    """Test the mass_transfer_rate function."""
    pressure_delta = 10.0
    first_order_mass_transport = 1e-17
    temperature = 300.0
    molar_mass = 0.02897
    expected_result = 1.16143004e-21
    result = get_mass_transfer_rate(
        pressure_delta, first_order_mass_transport, temperature, molar_mass
    )
    assert np.isclose(
        result,
        expected_result,
    )


def test_mass_transfer_mulit_particle_rate():
    """Test the mass_transfer_rate multi radii function."""
    pressure_delta = np.array([10.0, 15.0])
    first_order_mass_transport = np.array([1e-17, 2e-17])
    temperature = 300.0
    molar_mass = 0.02897
    expected_result = np.array([1.16143004e-21, 3.48429013e-21])
    result = get_mass_transfer_rate(
        pressure_delta, first_order_mass_transport, temperature, molar_mass
    )
    np.testing.assert_allclose(result, expected_result, rtol=1e-8)


def test_thermal_resistance_factor_isothermal_limit():
    """Ensure latent heat of zero returns the isothermal limit."""
    diffusion_coefficient = 2.5e-5
    latent_heat = 0.0
    vapor_pressure_surface = 2339.0
    thermal_conductivity = 0.0257
    temperature = 293.0
    molar_mass = 0.018015
    expected = GAS_CONSTANT / molar_mass * temperature
    result = get_thermal_resistance_factor(
        diffusion_coefficient=diffusion_coefficient,
        latent_heat=latent_heat,
        vapor_pressure_surface=vapor_pressure_surface,
        thermal_conductivity=thermal_conductivity,
        temperature=temperature,
        molar_mass=molar_mass,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-15)


def test_thermal_resistance_factor_water_293k():
    """Check known water-in-air values at 293 K."""
    diffusion_coefficient = 2.5e-5
    latent_heat = 2.454e6
    vapor_pressure_surface = 2339.0
    thermal_conductivity = 0.0257
    temperature = 293.0
    molar_mass = 0.018015
    expected = 461992.50781411503
    result = get_thermal_resistance_factor(
        diffusion_coefficient=diffusion_coefficient,
        latent_heat=latent_heat,
        vapor_pressure_surface=vapor_pressure_surface,
        thermal_conductivity=thermal_conductivity,
        temperature=temperature,
        molar_mass=molar_mass,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_thermal_resistance_factor_reference_value():
    """Lock a reference value to confirm sign and scale behavior."""
    diffusion_coefficient = 1.0
    latent_heat = 1.0
    vapor_pressure_surface = 1.0
    thermal_conductivity = 1.0
    temperature = 1.0
    molar_mass = 1.0
    expected = 7.43473497315324
    result = get_thermal_resistance_factor(
        diffusion_coefficient=diffusion_coefficient,
        latent_heat=latent_heat,
        vapor_pressure_surface=vapor_pressure_surface,
        thermal_conductivity=thermal_conductivity,
        temperature=temperature,
        molar_mass=molar_mass,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-9)


def test_thermal_resistance_factor_array_broadcasting():
    """Verify array inputs broadcast and preserve shape."""
    diffusion_coefficient = 2.5e-5
    latent_heat = np.array([0.0, 1.0e6])
    vapor_pressure_surface = np.array([1000.0, 2000.0])
    thermal_conductivity = 0.0257
    temperature = 293.0
    molar_mass = 0.018015
    result = get_thermal_resistance_factor(
        diffusion_coefficient=diffusion_coefficient,
        latent_heat=latent_heat,
        vapor_pressure_surface=vapor_pressure_surface,
        thermal_conductivity=thermal_conductivity,
        temperature=temperature,
        molar_mass=molar_mass,
    )
    assert np.asarray(result).shape == latent_heat.shape

    latent_heat_2d = np.array([[0.0, 1.0e6], [2.0e6, 3.0e6]])
    vapor_pressure_2d = np.array([[1000.0, 1500.0], [2000.0, 2500.0]])
    result_2d = get_thermal_resistance_factor(
        diffusion_coefficient=diffusion_coefficient,
        latent_heat=latent_heat_2d,
        vapor_pressure_surface=vapor_pressure_2d,
        thermal_conductivity=thermal_conductivity,
        temperature=temperature,
        molar_mass=molar_mass,
    )
    assert np.asarray(result_2d).shape == latent_heat_2d.shape


def test_thermal_resistance_factor_temperature_molar_mass_broadcasting():
    """Ensure temperature and molar mass arrays broadcast correctly."""
    diffusion_coefficient = 2.5e-5
    latent_heat = 2.454e6
    vapor_pressure_surface = 2339.0
    thermal_conductivity = 0.0257
    temperature = np.array([285.0, 293.0])
    molar_mass = np.array([[0.018015], [0.02897]])
    result = get_thermal_resistance_factor(
        diffusion_coefficient=diffusion_coefficient,
        latent_heat=latent_heat,
        vapor_pressure_surface=vapor_pressure_surface,
        thermal_conductivity=thermal_conductivity,
        temperature=temperature,
        molar_mass=molar_mass,
    )
    assert np.asarray(result).shape == (2, 2)


@pytest.mark.parametrize("latent_heat", [1.0e6, 2.0e6, 2.454e6])
def test_thermal_resistance_factor_positive_vs_baseline(latent_heat):
    """Thermal factor should be no less than the isothermal baseline."""
    diffusion_coefficient = 2.5e-5
    vapor_pressure_surface = 2339.0
    thermal_conductivity = 0.0257
    temperature = 293.0
    molar_mass = 0.018015
    baseline = GAS_CONSTANT / molar_mass * temperature
    result = get_thermal_resistance_factor(
        diffusion_coefficient=diffusion_coefficient,
        latent_heat=latent_heat,
        vapor_pressure_surface=vapor_pressure_surface,
        thermal_conductivity=thermal_conductivity,
        temperature=temperature,
        molar_mass=molar_mass,
    )
    assert result >= baseline


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("diffusion_coefficient", -1.0),
        ("latent_heat", -1.0),
        ("vapor_pressure_surface", -1.0),
        ("thermal_conductivity", 0.0),
        ("thermal_conductivity", -1.0),
        ("temperature", 0.0),
        ("temperature", -1.0),
        ("molar_mass", 0.0),
        ("molar_mass", -1.0),
    ],
)
def test_thermal_resistance_factor_rejects_invalid_inputs(field, value):
    """Invalid inputs should raise ValueError from validation."""
    kwargs = {
        "diffusion_coefficient": 2.5e-5,
        "latent_heat": 2.454e6,
        "vapor_pressure_surface": 2339.0,
        "thermal_conductivity": 0.0257,
        "temperature": 293.0,
        "molar_mass": 0.018015,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match="positive|nonnegative"):
        get_thermal_resistance_factor(**kwargs)


def test_multi_species_mass_transfer_rate():
    """Test the mass_transfer_rate function for multiple species."""
    pressure_delta = np.array([10.0, 15.0])
    first_order_mass_transport = np.array([1e-17, 2e-17])
    temperature = 300.0
    molar_mass = np.array([0.02897, 0.018015])
    expected_result = np.array([1.16143004e-21, 2.16670648e-21])
    result = get_mass_transfer_rate(
        pressure_delta, first_order_mass_transport, temperature, molar_mass
    )
    np.testing.assert_allclose(result, expected_result, rtol=1e-8)


def test_single_species_condensation_not_enough_gas_mass():
    """Test mass transfer for a single particle species where there is not
    enough gas mass for full condensation.
    """
    mass_rate = np.array([0.1, 0.5])  # kg/s (mass transfer rate per particle)
    time_step = 10  # seconds
    gas_mass = np.array([0.5])  # kg (not enough to satisfy both requests)
    particle_mass = np.array([1.0, 50])  # kg
    particle_concentration = np.array([1, 0.5])  # particles/m^3

    # Calculate the total mass to be transferred, accounting for particle
    # concentration
    total_mass_to_change = (
        mass_rate * time_step * particle_concentration
    )  # Total mass requested

    # Total requested mass is (0.1 * 10 * 1) + (0.5 * 10 * 0.5) = 1.0 kg,
    # but only 0.5 kg is available
    # Scaling factor: we need to scale the mass transfer so that the total
    # matches the available gas mass
    scaling_factor = gas_mass / total_mass_to_change.sum()

    # Expected mass transfer is the scaled version of the mass_to_change
    expected_mass_transfer = total_mass_to_change * scaling_factor

    # Calculate using the direct single species function
    result_direct = get_mass_transfer_of_single_species(
        mass_rate, time_step, gas_mass, particle_mass, particle_concentration
    )
    np.testing.assert_allclose(result_direct, expected_mass_transfer, rtol=1e-8)
    # second calc
    result_direct2 = get_mass_transfer_of_single_species(
        mass_rate, time_step, gas_mass, particle_mass, particle_concentration
    )
    np.testing.assert_allclose(
        result_direct2, expected_mass_transfer, rtol=1e-8
    )

    # Calculate using the general helper function
    result = get_mass_transfer(
        mass_rate, time_step, gas_mass, particle_mass, particle_concentration
    )
    np.testing.assert_allclose(result, expected_mass_transfer, rtol=1e-8)


def test_single_species_evaporation_not_enough_particle_mass():
    """Test mass transfer for a single particle species where there is not
    enough particle mass for full evaporation.
    """
    mass_rate = np.array([-0.2, -8])  # kg/s (negative for evaporation)
    time_step = 10  # seconds
    gas_mass = np.array([1.0])  # kg
    particle_mass = np.array([0.8, 0.3])  # kg per particle
    particle_concentration = np.array([1, 5])  # particles/m^3

    # ───── expected result ────────────────────────────────────────────
    requested = mass_rate * time_step * particle_concentration
    inventory = (particle_mass * particle_concentration).sum()

    # species-level down-scaling (evaporation)
    scale = 1.0
    if requested.sum() < 0.0 and -requested.sum() > inventory:
        scale = inventory / (-requested.sum())
    scaled = requested * scale

    # per-bin clip so we never evaporate more than each bin owns
    per_bin_limit = -particle_mass * particle_concentration
    expected_mass_transfer = np.maximum(scaled, per_bin_limit)

    result_direct = get_mass_transfer_of_single_species(
        mass_rate, time_step, gas_mass, particle_mass, particle_concentration
    )
    np.testing.assert_allclose(result_direct, expected_mass_transfer, rtol=1e-8)

    result = get_mass_transfer(
        mass_rate, time_step, gas_mass, particle_mass, particle_concentration
    )
    np.testing.assert_allclose(result, expected_mass_transfer, rtol=1e-8)


def test_multiple_species_condensation():
    """Test mass transfer for multiple particle and gas species (n=2, m=3)
    where there is not enough gas mass for full condensation, and particle
    concentration is greater than 1.
    """
    # mass_rate for 2 particles and 3 gas species
    mass_rate = np.array([[0.1, 0.05, 0.03], [0.2, 0.15, 0.07]])  # kg/s
    time_step = 10  # seconds

    # gas_mass for 3 gas species
    gas_mass = np.array([1.0, 0.8, 0.5])  # kg

    # particle_mass for 2 particles and 3 gas species
    particle_mass = np.array([[1.0, 0.9, 0.8], [1.2, 1.0, 0.7]])  # kg

    # particle_concentration for 2 particles, greater than 1
    particle_concentration = np.array([5, 4])  # particles/m^3

    # Step 1: Calculate the total mass to change (before scaling)
    mass_to_change = (
        mass_rate * time_step * particle_concentration[:, np.newaxis]
    )

    # Step 2: Calculate the total requested mass for each gas species
    total_requested_mass = mass_to_change.sum(axis=0)

    # Step 3: Apply scaling if requested mass exceeds available gas mass
    scaling_factor = np.ones_like(gas_mass)
    scaling_mask = total_requested_mass > gas_mass
    scaling_factor[scaling_mask] = (
        gas_mass[scaling_mask] / total_requested_mass[scaling_mask]
    )

    # Step 4: Calculate expected mass transfer (scaled)
    expected_mass_transfer = mass_to_change * scaling_factor

    # Test the direct multiple species function
    result_direct = get_mass_transfer_of_multiple_species(
        mass_rate, time_step, gas_mass, particle_mass, particle_concentration
    )
    # Check that the total mass transfer for each gas species is equal to the
    # total mass transfer for the gas phase
    np.testing.assert_allclose(result_direct.sum(axis=0), gas_mass, rtol=1e-8)
    # Check for each individual particle and species
    np.testing.assert_allclose(result_direct, expected_mass_transfer, rtol=1e-8)

    # Test the general helper function
    result = get_mass_transfer(
        mass_rate, time_step, gas_mass, particle_mass, particle_concentration
    )
    np.testing.assert_allclose(result, expected_mass_transfer, rtol=1e-8)


def test_multiple_species_evaporation_column_inventory_limit():
    """Evaporation across several species should never remove more mass
    than the *total particle inventory* of that species, even when
    the requested evaporation (rate × dt × conc) is far larger.

    Scenario
    --------
    * 2 particle‐size bins, 2 condensable species.
    * All fluxes are negative (evaporation).
    * Requested evaporation in each species exceeds its inventory
      so the routine must down-scale the column uniformly, then
      apply per-bin clipping.

    Expected behaviour
    ------------------
    * Column sum for every species equals **at most** the inventory.
    * No individual bin evaporates more mass than it owns.
    """
    # ──────────────────────────────────────────────────────────────
    # Inputs
    # ──────────────────────────────────────────────────────────────
    mass_rate = np.array([[-1.0, -0.2], [-1.5, -0.6]])  # kg s⁻¹  (shape 2×2)
    time_step = 10.0  # s

    # Gas mass is irrelevant for evaporation limit, pick generous values
    gas_mass = np.array([100.0, 100.0])  # kg  (shape 2,)

    particle_mass = np.array([[0.4, 0.2], [0.6, 0.3]])  # kg  (shape 2×2)

    particle_concentration = np.array([1.0, 1.0])  # # m⁻³  (shape 2,)

    # ──────────────────────────────────────────────────────────────
    # Manually compute the *expected* result following the algorithm
    # ──────────────────────────────────────────────────────────────
    requested = mass_rate * time_step * particle_concentration[:, None]
    # column sums (negative => evaporation)
    total_req = requested.sum(axis=0)  # [-25.0, -8.0]

    # inventories per species (positive)
    inventory = (particle_mass * particle_concentration[:, None]).sum(axis=0)
    # [1.0, 0.5]

    # species-level down-scaling
    scale = np.where(
        -total_req > inventory, inventory / (-total_req), 1.0
    )  # [0.04, 0.0625]
    scaled = requested * scale  # broadcast

    # per-bin clip so we never exceed its own mass
    per_bin_limit = -particle_mass * particle_concentration[:, None]
    expected = np.maximum(scaled, per_bin_limit)

    # ──────────────────────────────────────────────────────────────
    # Call the routine under test
    # ──────────────────────────────────────────────────────────────
    result = get_mass_transfer_of_multiple_species(
        mass_rate=mass_rate,
        time_step=time_step,
        gas_mass=gas_mass,
        particle_mass=particle_mass,
        particle_concentration=particle_concentration,
    )

    # ──────────────────────────────────────────────────────────────
    # Assertions
    # ──────────────────────────────────────────────────────────────
    # 1. element-wise equality
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_condensation_inventory_limit():
    """Requested condensation exceeds gas_mass → routine must down-scale so that
    the column sum equals the available gas.
    """
    # 2 size bins × 1 species (condensation only, values > 0)
    mass_rate = np.array([[0.5], [0.8]])  # kg s⁻¹
    time_step = 10.0  # s

    gas_mass = np.array([1.0])  # kg (scarce)
    particle_mass = np.array([[0.1], [0.1]])  # kg
    particle_conc = np.array([10.0, 10.0])  # # m⁻³

    # -------- expected result ------------------------------------------------
    requested = mass_rate * time_step * particle_conc[:, None]
    scale = gas_mass[0] / requested.sum()  # 1 / 130
    expected = requested * scale  # shape (2,1)

    # -------- routine under test --------------------------------------------
    res = get_mass_transfer_of_multiple_species(
        mass_rate, time_step, gas_mass, particle_mass, particle_conc
    )

    # elementwise equality
    np.testing.assert_allclose(res, expected, rtol=1e-12, atol=1e-12)
    # column-sum (mass conservation vs. gas reservoir)
    np.testing.assert_allclose(res.sum(axis=0), gas_mass, rtol=1e-12)


# ──────────────────────────────────────────────────────────────────────────
# Evaporation-limited case ─ scarce particle inventory, ample gas
# ──────────────────────────────────────────────────────────────────────────
def test_evaporation_inventory_limit():
    """Requested evaporation exceeds the particle inventory → routine must
    down-scale so that the column sum does not exceed −inventory and no bin
    evaporates more mass than it owns.
    """
    # 2 size bins × 1 species (evaporation only, values < 0)
    mass_rate = np.array([[-1.0], [-1.5]])  # kg s⁻¹
    time_step = 10.0  # s

    gas_mass = np.array([100.0])  # kg (irrelevant, ample)
    particle_mass = np.array([[0.1], [0.3]])  # kg
    particle_conc = np.array([10.0, 10.0])  # # m⁻³

    # inventories and per-bin limits
    inventory = (particle_mass * particle_conc[:, None]).sum(axis=0)  # [4.]
    per_bin_evap_limit = (
        -particle_mass * particle_conc[:, None]
    )  # [[-1.], [-3.]]

    # -------- expected result ------------------------------------------------
    requested = (
        mass_rate * time_step * particle_conc[:, None]
    )  # [[-100.], [-150.]]
    scale = inventory[0] / -requested.sum()  # 4 / 250
    scaled = requested * scale  # down-scaled
    expected = np.maximum(scaled, per_bin_evap_limit)  # per-bin clip

    # -------- routine under test --------------------------------------------
    res = get_mass_transfer_of_multiple_species(
        mass_rate, time_step, gas_mass, particle_mass, particle_conc
    )

    # elementwise equality
    np.testing.assert_allclose(res, expected, rtol=1e-12, atol=1e-12)
    # column-sum should equal −inventory or be slightly less (if per-bin clip)
    np.testing.assert_allclose(
        res.sum(axis=0), expected.sum(axis=0), rtol=1e-12
    )
    # per-bin safety: never exceed negative inventory
    assert np.all(res >= per_bin_evap_limit - 1e-12)


def test_zero_mass_transfer():
    """Test mass transfer when there is no mass transfer."""
    mass_rate = np.array([0.0])  # No mass transfer
    time_step = 10  # seconds
    gas_mass = np.array([1.0])  # kg
    particle_mass = np.array([1.0])  # kg
    particle_concentration = np.array([1])  # particles/m^3

    expected_mass_transfer = np.array([0.0])  # No mass transfer should occur

    result = get_mass_transfer(
        mass_rate, time_step, gas_mass, particle_mass, particle_concentration
    )
    np.testing.assert_array_almost_equal(result, expected_mass_transfer)


def test_radius_transfer_rate():
    """Test the radius_transfer_rate function."""
    # Test normal case
    result = get_radius_transfer_rate(1e-21, 1e-6, 1000)
    np.testing.assert_allclose(result, 7.95774715e-14, atol=1e-8)

    # Test edge cases
    np.testing.assert_allclose(get_radius_transfer_rate(0, 1e-6, 1000), 0)

    # Test with array inputs
    mass_rate = np.array([1e-21, 2e-21])  # kg/s
    radius = np.array([1e-6, 2e-6])  # m
    density = 1000  # kg/m^3
    expected_result = np.array([7.95774715e-14, 1.98943679e-14])  # m/s
    result = get_radius_transfer_rate(mass_rate, radius, density)
    np.testing.assert_allclose(result, expected_result, atol=1e-8)

    # Test with zero mass rate
    mass_rate = np.array([1e-21, 0])  # kg/s
    radius = np.array([1e-6, 2e-6])  # m
    density = 1000  # kg/m^3
    expected_result = np.array([7.95774715e-14, 0])  # m/s
    result = get_radius_transfer_rate(mass_rate, radius, density)
    np.testing.assert_allclose(result, expected_result, atol=1e-8)


def test_mixed_condensation_and_evaporation_inventory_limit():
    """Mixed-sign fluxes: one bin condenses while another evaporates.
    Only the *condensation* part must be down–scaled to respect the
    limited gas reservoir; the evaporation part must remain untouched
    except for the usual per-bin inventory clip.
    """
    # ───── scenario ───────────────────────────────────────────────
    # 2 size bins, 1 condensable species
    mass_rate = np.array([0.3, -0.1])  # kg s⁻¹  (+ condense, − evaporate)
    time_step = 10.0  # s
    gas_mass = np.array([1.0])  # kg  (scarce → scaling expected)
    particle_mass = np.array([0.5, 0.5])  # kg per particle
    particle_conc = np.array([1.0, 1.0])  # # m⁻³

    # ───── manual expectation ────────────────────────────────────
    requested = mass_rate * time_step * particle_conc  # [+3.0, −1.0]
    positive_sum = requested[requested > 0.0].sum()  # 3.0
    negative_sum = requested[requested < 0.0].sum()  # −1.0

    # scale *only* the condensation so that the column net equals gas_mass
    if positive_sum + negative_sum > gas_mass[0]:
        cond_scale = (gas_mass[0] - negative_sum) / positive_sum
    else:
        cond_scale = 1.0

    expected = np.where(requested > 0.0, requested * cond_scale, requested)

    # per-bin evaporation cannot exceed its inventory
    per_bin_limit = -particle_mass * particle_conc  # [−0.5, −0.5]
    expected = np.maximum(expected, per_bin_limit)

    # ───── routine under test ────────────────────────────────────
    result = get_mass_transfer_of_single_species(
        mass_rate=mass_rate,
        time_step=time_step,
        gas_mass=gas_mass,
        particle_mass=particle_mass,
        particle_concentration=particle_conc,
    )

    # ───── assertions ────────────────────────────────────────────
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
