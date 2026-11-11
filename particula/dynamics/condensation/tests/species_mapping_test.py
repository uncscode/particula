"""Independent native admission, mapping and inventory-weight assertions."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from particula.dynamics.condensation.condensation_factories import (
    CondensationFactory,
)
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
    CondensationIsothermalStaggered,
    CondensationLatentHeat,
)
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
    condensation_population_weights,
    validate_condensation_state,
)
from particula.gas.gas_data import GasData
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)
from particula.particles.activity_strategies import ActivityIdealMass
from particula.particles.particle_data import ParticleData
from particula.particles.surface_strategies import SurfaceStrategyVolume
from particula.util.constants import GAS_CONSTANT


def make_state(kind="particle_resolved", volume=0.25):
    """Construct native two-particle/three-gas-species state, without facades."""
    particle = ParticleData(
        masses=np.array([[[2e-18, 3e-18], [5e-18, 7e-18], [11e-18, 13e-18]]]),
        concentration=np.array([[2.0, 0.5, 4.0]]),
        charge=np.zeros((1, 3)),
        density=np.array([1000.0, 1200.0]),
        volume=np.array([volume]),
        distribution_type=kind,
        radius_grid=(
            np.array([1e-9, 2e-9, 4e-9]) if kind == "continuous_pdf" else None
        ),
    )
    gas = GasData(
        name=["inert", "B", "A"],
        molar_mass=np.array([0.028, 0.046, 0.018]),
        concentration=np.array([[9e-9, 2e-9, 3e-9]]),
        partitioning=np.array([False, True, True]),
    )
    mapping = CondensationSpeciesMap(("inert", "B", "A"), ((2, 0), (1, 1)))
    return particle, gas, mapping


def snapshot(particle, gas):
    """Retain identities and bytes, including directly malformed arrays."""
    arrays = [
        particle.masses,
        particle.concentration,
        particle.charge,
        particle.density,
        particle.volume,
        gas.concentration,
        gas.molar_mass,
        gas.partitioning,
    ]
    if particle.radius_grid is not None:
        arrays.append(particle.radius_grid)
    return (
        tuple(
            (id(value), value.shape, value.dtype.str, value.tobytes())
            for value in arrays
        ),
        tuple(gas.name),
        particle.distribution_type,
    )


def test_explicit_permutation_unequal_width_and_detached_indices():
    """Explicit permutation unequal width and detached indices."""
    particle, gas, mapping = make_state()
    before = snapshot(particle, gas)
    source, target = validate_condensation_state(
        particle, gas, mapping, 298, 101325
    )
    np.testing.assert_array_equal(source, [2, 1])
    np.testing.assert_array_equal(target, [0, 1])
    source[:] = 0
    assert mapping.gas_to_particle == ((2, 0), (1, 1))
    assert snapshot(particle, gas) == before


@pytest.mark.parametrize(
    "flags,skip,expected",
    [
        ([False, True, True], [1], ([2], [0])),
        ([False, False, False], None, ([], [])),
        ([True, True, True], None, ([2, 1], [0, 1])),
    ],
)
def test_full_gas_flags_and_skip_do_not_compact_storage(flags, skip, expected):
    """Full gas flags and skip do not compact storage."""
    particle, gas, mapping = make_state()
    gas.partitioning[:] = flags
    before = snapshot(particle, gas)
    source, target = validate_condensation_state(
        particle, gas, mapping, 298, 101325, skip
    )
    np.testing.assert_array_equal(source, expected[0])
    np.testing.assert_array_equal(target, expected[1])
    assert snapshot(particle, gas) == before


def test_explicit_empty_map_is_valid_and_omitted_map_is_not():
    """Explicit empty map is valid and omitted map is not."""
    particle, gas, mapping = make_state()
    empty = CondensationSpeciesMap(mapping.gas_names, ())
    assert all(
        value.size == 0
        for value in validate_condensation_state(
            particle, gas, empty, 298, 101325
        )
    )
    with pytest.raises(TypeError, match="CondensationSpeciesMap"):
        validate_condensation_state(particle, gas, None, 298, 101325)


@pytest.mark.parametrize(
    "names,pairs",
    [
        (("A", "A"), ((0, 0),)),
        (("A", " "), ((0, 0),)),
        ("AB", ((0, 0),)),
        (("A", "B"), ((0, 0), (0, 1))),
        (("A", "B"), ((0, 0), (1, 0))),
        (("A", "B"), ((2, 0),)),
        (("A", "B"), ((-1, 0),)),
        (("A", "B"), ((0, 0.0),)),
        (("A", "B"), ((False, 0),)),
        (("A", "B"), ((0, True),)),
        (("A", "B"), ((0, 1, 2),)),
        (("A", "B"), (0,)),
    ],
)
def test_map_rejects_ambiguous_provenance(names, pairs):
    """Map rejects ambiguous provenance."""
    with pytest.raises(ValueError):
        CondensationSpeciesMap(names, pairs)


def test_map_detaches_caller_sequences_and_is_immutable():
    """Map detaches caller sequences and is immutable."""
    names = ["A", "B"]
    pairs = [[np.int64(1), np.int64(0)]]
    mapping = CondensationSpeciesMap(names, pairs)
    names[1] = "changed"
    pairs[0][0] = 0
    assert mapping.gas_names == ("A", "B")
    assert mapping.gas_to_particle == ((1, 0),)
    with pytest.raises(FrozenInstanceError):
        mapping.gas_names = ("changed",)


@pytest.mark.parametrize(
    "field,value",
    [
        ("concentration", np.array([[1.0, -1.0, 1.0]])),
        ("concentration", np.array([[1.0, np.nan, 1.0]])),
        ("concentration", np.ones((2, 3))),
        ("concentration", np.ones((1, 2))),
        ("molar_mass", np.array([0.028, 0.0, 0.018])),
        ("molar_mass", np.array([0.028, np.inf, 0.018])),
        ("partitioning", np.ones(3, dtype=np.int32)),
        ("name", ["inert", "A", "B"]),
    ],
)
def test_mutated_gas_rejects_without_repair(field, value):
    """Mutated gas rejects without repair."""
    particle, gas, mapping = make_state()
    setattr(gas, field, value)
    before = snapshot(particle, gas)
    with pytest.raises(ValueError):
        validate_condensation_state(particle, gas, mapping, 298, 101325)
    assert snapshot(particle, gas) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("volume", np.array([0.0])),
        ("masses", np.full((1, 3, 2), -1.0)),
        ("charge", np.full((1, 3), np.inf)),
        ("concentration", np.ones((1, 2))),
        ("density", np.array([0.0, 1200.0])),
        ("distribution_type", None),
    ],
)
def test_mutated_particle_rejects_without_repair(field, value):
    """Mutated particle rejects without repair."""
    particle, gas, mapping = make_state()
    setattr(particle, field, value)
    before = snapshot(particle, gas)
    with pytest.raises(ValueError):
        validate_condensation_state(particle, gas, mapping, 298, 101325)
    assert snapshot(particle, gas) == before


@pytest.mark.parametrize(
    "temperature,pressure",
    [
        (0, 101325),
        (np.nan, 101325),
        (298, np.inf),
        (298, -1),
        (np.array([298.0]), 101325),
    ],
)
def test_invalid_environment(temperature, pressure):
    """Invalid environment."""
    particle, gas, mapping = make_state()
    before = snapshot(particle, gas)
    with pytest.raises(ValueError):
        validate_condensation_state(
            particle, gas, mapping, temperature, pressure
        )
    assert snapshot(particle, gas) == before


@pytest.mark.parametrize("skip", [[-1], [3], [1.0], [True]])
def test_invalid_skip_indices(skip):
    """Invalid skip indices."""
    particle, gas, mapping = make_state()
    with pytest.raises(ValueError, match="skip indices"):
        validate_condensation_state(particle, gas, mapping, 298, 101325, skip)


def test_mapping_validates_particle_bounds_even_when_all_gas_disabled():
    """Mapping validates particle bounds even when all gas disabled."""
    particle, gas, mapping = make_state()
    gas.partitioning[:] = False
    mapping = CondensationSpeciesMap(mapping.gas_names, ((1, 2),))
    with pytest.raises(ValueError, match="particle species index"):
        validate_condensation_state(particle, gas, mapping, 298, 101325)


@pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
def test_resolved_inventory_weights_have_one_volume_division(volume):
    """Resolved inventory weights have one volume division."""
    particle, _, _ = make_state(volume=volume)
    weights = condensation_population_weights(particle)
    np.testing.assert_allclose(
        weights, np.array([2.0, 0.5, 4.0]) / volume, rtol=1e-15, atol=0
    )
    expected_mass = np.array([50.5e-18, 61.5e-18])
    np.testing.assert_allclose(
        (weights[:, None] * particle.masses[0]).sum(axis=0) * volume,
        expected_mass,
        rtol=1e-14,
        atol=1e-30,
    )
    weights[:] = 0
    np.testing.assert_array_equal(particle.concentration, [[2, 0.5, 4]])


@pytest.mark.parametrize("kind", ["discrete", "continuous_pdf"])
@pytest.mark.parametrize("volume", [0.25, 4.0])
def test_nonunit_binned_volume_rejects(kind, volume):
    """Nonunit binned volume rejects."""
    particle, gas, mapping = make_state(kind, volume)
    before = snapshot(particle, gas)
    with pytest.raises(ValueError, match="exactly 1"):
        validate_condensation_state(particle, gas, mapping, 298, 101325)
    with pytest.raises(ValueError, match="exactly 1"):
        condensation_population_weights(particle)
    assert snapshot(particle, gas) == before


def test_pdf_integrates_mass_times_density_on_nonuniform_radius_grid():
    """Pdf integrates mass times density on nonuniform radius grid."""
    particle, _, _ = make_state("continuous_pdf", 1.0)
    particle.concentration[:] = [[1e9, 2e9, 1e9]]
    weights = condensation_population_weights(particle)
    # Independent two-interval trapezoids of mass*PDF (not a bare PDF sum).
    expected = (
        particle.masses[0, 0] * 1e9 + particle.masses[0, 1] * 2e9
    ) * 1e-9 / 2 + (
        particle.masses[0, 1] * 2e9 + particle.masses[0, 2] * 1e9
    ) * 2e-9 / 2
    np.testing.assert_allclose(weights, [0.5, 3.0, 1.0], rtol=1e-15, atol=0)
    np.testing.assert_allclose(
        (particle.masses[0] * weights[:, None]).sum(axis=0),
        expected,
        rtol=1e-14,
        atol=1e-30,
    )


def test_discrete_weights_are_already_density():
    """Discrete weights are already density."""
    particle, _, _ = make_state("discrete", 1.0)
    np.testing.assert_array_equal(
        condensation_population_weights(particle), [2, 0.5, 4]
    )


def test_valid_multibox_state_is_rejected_before_scientific_work():
    """Valid multibox state is rejected before scientific work."""
    particle, gas, mapping = make_state()
    for name in ("masses", "concentration", "charge", "volume"):
        setattr(particle, name, np.repeat(getattr(particle, name), 2, axis=0))
    before = snapshot(particle, gas)
    with pytest.raises(ValueError, match="n_boxes=1"):
        validate_condensation_state(particle, gas, mapping, 298, 101325)
    with pytest.raises(ValueError, match="n_boxes=1"):
        condensation_population_weights(particle)
    assert snapshot(particle, gas) == before


def native_strategy(strategy_type, gas, mapping, **kwargs):
    """Build an explicitly mapped process with full-gas transport parameters."""
    return strategy_type(
        molar_mass=gas.molar_mass.copy(),
        diffusion_coefficient=np.array([3e-5, 1e-5, 2e-5]),
        accommodation_coefficient=np.array([1.0, 0.7, 0.9]),
        activity_strategy=ActivityIdealMass(),
        surface_strategy=SurfaceStrategyVolume(density=1000.0),
        vapor_pressure_strategy=ConstantVaporPressureStrategy(
            vapor_pressure=np.array([0.0, 0.2, 0.1])
        ),
        species_map=mapping,
        **kwargs,
    )


@pytest.mark.parametrize(
    "strategy_type",
    [
        CondensationIsothermal,
        CondensationIsothermalStaggered,
        CondensationLatentHeat,
    ],
)
@pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
def test_native_mapped_rates_against_independent_equations(
    strategy_type, volume
):
    """Verify gas/particle permutation with an independent transport oracle."""
    particle, gas, mapping = make_state(volume=volume)
    strategy = native_strategy(strategy_type, gas, mapping)
    before = snapshot(particle, gas)
    temperature, pressure, viscosity = 298.0, 101325.0, 1.8e-5
    actual = strategy.mass_transfer_rate(
        particle, gas, temperature, pressure, dynamic_viscosity=viscosity
    )
    # The oracle uses raw arrays and explicit scalar equations, no production
    # pressure, radius, transfer, activity or population helper calls.
    radius = (
        3 * (particle.masses[0] / particle.density).sum(axis=1) / (4 * np.pi)
    ) ** (1 / 3)
    fractions = particle.masses[0] / particle.masses[0].sum(axis=1)[:, None]
    expected = np.zeros((3, 2))
    for source, target, pure, diffusion, alpha in (
        (2, 0, 0.1, 2e-5, 0.9),
        (1, 1, 0.2, 1e-5, 0.7),
    ):
        molar = gas.molar_mass[source]
        path = (2 * viscosity / pressure) / np.sqrt(
            8 * molar / (np.pi * GAS_CONSTANT * temperature)
        )
        kn = path / radius
        correction = (0.75 * alpha * (1 + kn)) / (
            kn**2 + kn + 0.283 * alpha * kn + 0.75 * alpha
        )
        kelvin = np.exp(
            2 * 0.072 * molar / (GAS_CONSTANT * temperature * 1000 * radius)
        )
        delta = (
            gas.concentration[0, source] * GAS_CONSTANT * temperature / molar
        )
        delta -= pure * fractions[:, target] * kelvin
        expected[:, target] = (
            4
            * np.pi
            * radius
            * diffusion
            * correction
            * delta
            * molar
            / (GAS_CONSTANT * temperature)
        )
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-35)
    assert snapshot(particle, gas) == before


@pytest.mark.parametrize(
    "strategy_type",
    [
        CondensationIsothermal,
        CondensationIsothermalStaggered,
        CondensationLatentHeat,
    ],
)
def test_native_rate_density_and_disabled_lanes(strategy_type):
    """Native population rates use density and preserve skipped gas ordering."""
    particle, gas, mapping = make_state(volume=4.0)
    strategy = native_strategy(
        strategy_type, gas, mapping, skip_partitioning_indices=[1]
    )
    per_particle = strategy.mass_transfer_rate(particle, gas, 298, 101325)
    rates = strategy.rate(particle, gas, 298, 101325)
    np.testing.assert_allclose(
        rates,
        per_particle * np.array([0.5, 0.125, 1.0])[:, None],
        rtol=1e-14,
        atol=1e-35,
    )
    np.testing.assert_array_equal(rates[:, 1], 0)
    gas.partitioning[:] = False
    np.testing.assert_array_equal(strategy.rate(particle, gas, 298, 101325), 0)


@pytest.mark.parametrize(
    "strategy_type",
    [
        CondensationIsothermal,
        CondensationIsothermalStaggered,
        CondensationLatentHeat,
    ],
)
@pytest.mark.parametrize(
    "field,value",
    [
        ("species_map", None),
        ("molar_mass", np.array([0.018, 0.046, 0.028])),
        ("diffusion_coefficient", np.ones(2)),
        ("accommodation_coefficient", np.array([1.0, np.nan, 1.0])),
        ("activity_strategy", None),
        ("surface_strategy", None),
        ("vapor_pressure_strategy", None),
    ],
)
def test_native_strategy_configuration_rejects_without_mutation(
    strategy_type,
    field,
    value,
):
    """Revalidate mutated process configuration at every rate call."""
    particle, gas, mapping = make_state()
    strategy = native_strategy(strategy_type, gas, mapping)
    setattr(strategy, field, value)
    before = snapshot(particle, gas)
    with pytest.raises((ValueError, TypeError)):
        strategy.mass_transfer_rate(particle, gas, 298, 101325)
    assert snapshot(particle, gas) == before


@pytest.mark.parametrize(
    "kind", ["isothermal", "isothermal_staggered", "latent_heat"]
)
def test_factory_and_fluent_builder_preserve_native_configuration(kind):
    """Every builder/factory transports process configuration by identity."""
    particle, gas, mapping = make_state()
    reference = native_strategy(CondensationIsothermal, gas, mapping)
    parameters = {
        "molar_mass": reference.molar_mass,
        "molar_mass_units": "kg/mol",
        "diffusion_coefficient": reference.diffusion_coefficient,
        "diffusion_coefficient_units": "m^2/s",
        "accommodation_coefficient": reference.accommodation_coefficient,
        "species_map": mapping,
        "activity_strategy": reference.activity_strategy,
        "surface_strategy": reference.surface_strategy,
        "vapor_pressure_strategy": reference.vapor_pressure_strategy,
        "skip_partitioning_indices": [1],
    }
    factory = CondensationFactory()
    strategy = factory.get_strategy(kind, parameters)
    builder = factory.get_builders()[kind]
    builder.set_molar_mass(reference.molar_mass, "kg/mol")
    builder.set_diffusion_coefficient(reference.diffusion_coefficient, "m^2/s")
    builder.set_accommodation_coefficient(reference.accommodation_coefficient)
    assert builder.set_species_map(mapping) is builder
    builder.set_activity_strategy(reference.activity_strategy)
    builder.set_surface_strategy(reference.surface_strategy)
    builder.set_vapor_pressure_strategy(reference.vapor_pressure_strategy)
    skipped = [1]
    builder.set_skip_partitioning_indices(skipped)
    skipped[:] = [2]
    fluent = builder.build()
    for built in (strategy, fluent):
        assert built.species_map is mapping
        assert built.activity_strategy is reference.activity_strategy
        assert built.surface_strategy is reference.surface_strategy
        assert (
            built.vapor_pressure_strategy is reference.vapor_pressure_strategy
        )
        assert built.skip_partitioning_indices == (1,)
        output = built.rate(particle, gas, 298, 101325)
        np.testing.assert_array_equal(output[:, 1], 0)
        assert np.all(output[:, 0] != 0)
    # Reloading a legacy dictionary clears optional native settings.
    builder.set_parameters(
        {
            key: value
            for key, value in parameters.items()
            if key
            in (
                "molar_mass",
                "molar_mass_units",
                "diffusion_coefficient",
                "diffusion_coefficient_units",
                "accommodation_coefficient",
            )
        }
    )
    rebuilt = builder.build()
    assert rebuilt.species_map is None
    assert rebuilt.activity_strategy is None
    assert rebuilt.surface_strategy is None
    assert rebuilt.vapor_pressure_strategy is None


@pytest.mark.parametrize(
    "key",
    [
        "species_map",
        "activity_strategy",
        "surface_strategy",
        "vapor_pressure_strategy",
    ],
)
def test_native_builder_type_failure_keeps_prior_configuration(key):
    """Invalid native configuration cannot partially overwrite the builder."""
    _, gas, mapping = make_state()
    builder = CondensationFactory().get_builders()["isothermal"]
    builder.set_species_map(mapping)
    with pytest.raises(TypeError):
        builder.set_parameters(
            {
                key: object(),
                "molar_mass": gas.molar_mass,
                "diffusion_coefficient": 2e-5,
                "accommodation_coefficient": 1.0,
            }
        )
    assert builder.species_map is mapping


def test_vapor_sequence_is_detached_and_keeps_gas_order():
    """Per-gas models preserve ordered identity without retaining caller lists."""
    particle, gas, mapping = make_state()
    strategy = native_strategy(CondensationIsothermal, gas, mapping)
    expected = strategy.rate(particle, gas, 298, 101325)
    models = [ConstantVaporPressureStrategy(value) for value in (0, 0.2, 0.1)]
    builder = CondensationFactory().get_builders()["isothermal"]
    builder.set_vapor_pressure_strategy(models)
    retained = builder.vapor_pressure_strategy
    models.reverse()
    strategy.vapor_pressure_strategy = retained
    np.testing.assert_allclose(
        strategy.rate(particle, gas, 298, 101325),
        expected,
        rtol=1e-14,
        atol=1e-35,
    )


def test_native_activity_retains_unmapped_particle_material():
    """An inert particle-only lane dilutes activity without acquiring transfer."""
    particle, gas, mapping = make_state()
    strategy = native_strategy(CondensationIsothermal, gas, mapping)
    original = strategy.mass_transfer_rate(particle, gas, 298, 101325)
    particle.masses = np.concatenate(
        (particle.masses, particle.masses.sum(axis=2, keepdims=True)), axis=2
    )
    particle.density = np.array([1000.0, 1200.0, 1000.0])
    before = snapshot(particle, gas)
    actual = strategy.mass_transfer_rate(particle, gas, 298, 101325)
    assert actual.shape == (3, 3)
    np.testing.assert_array_equal(actual[:, 2], 0)
    assert not np.allclose(actual[:, :2], original, rtol=1e-12, atol=1e-35)
    assert snapshot(particle, gas) == before


def test_inactive_nodes_have_no_native_mass_rate():
    """No number inventory means no transfer, even when slot masses remain."""
    particle, gas, mapping = make_state()
    particle.concentration[0, 1] = 0
    strategy = native_strategy(CondensationIsothermal, gas, mapping)
    np.testing.assert_array_equal(
        strategy.mass_transfer_rate(particle, gas, 298, 101325)[1], 0
    )
