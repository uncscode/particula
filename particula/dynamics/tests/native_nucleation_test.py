"""Native source count/volume and mapping regressions with independent oracles."""

from dataclasses import replace

import numpy as np
import pytest
from particula.aerosol import Aerosol
from particula.dynamics.nucleation import particle_source as source
from particula.dynamics.nucleation.nucleation_configuration import (
    NucleationSourceConfig,
)
from particula.dynamics.nucleation.nucleation_strategies import (
    ActivationNucleationStrategy,
    ClosedInterval,
    FormationMetadata,
    InjectionComposition,
    NucleationValidityDomain,
)
from particula.dynamics.nucleation.species_mapping import NucleationSpeciesMap
from particula.dynamics.particle_process import (
    Nucleation,
    NucleationCommitConfig,
)
from particula.gas.environment_data import EnvironmentData
from particula.gas.gas_data import GasData
from particula.particles.exhaustion import ExhaustionControls
from particula.particles.particle_data import ParticleData
from particula.util.constants import AVOGADRO_NUMBER


@pytest.fixture(autouse=True)
def forbid_legacy_construction(monkeypatch):
    """Native transactions never construct legacy facades."""
    from particula._legacy_aerosol import Aerosol as LegacyAerosol
    from particula.gas.atmosphere import Atmosphere
    from particula.gas.species import GasSpecies
    from particula.particles.representation import ParticleRepresentation

    def forbidden(*args, **kwargs):
        raise AssertionError("native source constructed a legacy facade")

    for carrier in (
        LegacyAerosol,
        Atmosphere,
        GasSpecies,
        ParticleRepresentation,
    ):
        monkeypatch.setattr(carrier, "__init__", forbidden)


def fixture(volume=0.25, capacity=4):
    """Return unequal-width native storage with a particle-only species."""
    masses = np.zeros((1, capacity, 3), dtype=np.float64)
    masses[0, 0] = [1e-20, 2e-20, 3e-20]
    counts = np.zeros((1, capacity), dtype=np.float64)
    counts[0, 0] = 2
    particles = ParticleData(
        masses=masses,
        concentration=counts,
        charge=np.zeros_like(counts),
        density=np.array([1000.0, 1200.0, 1400.0]),
        volume=np.array([volume]),
        distribution_type="particle_resolved",
    )
    gas = GasData(
        name=["inert", "B", "A", "spectator"],
        molar_mass=np.array([0.04, 0.1, 0.2, 0.03]),
        concentration=np.array([[9e-12, 2e-12, 3e-12, 7e-12]]),
        partitioning=np.array([False, True, True, False]),
    )
    environment = EnvironmentData(
        temperature=np.array([298.15]),
        pressure=np.array([101325.0]),
        saturation_ratio=np.ones((1, 4)),
    )
    mapping = NucleationSpeciesMap(tuple(gas.name), ((2, 0), (1, 1)))
    config = source.ParticleSourceCommitConfig(
        1e10,
        2.0,
        ExhaustionControls(),
        np.array([1.0]),
        np.array([0.1]),
        np.array([0.001]),
        species_map=mapping,
    )
    return Aerosol(
        particles=particles, gas=gas, environment=environment
    ), config


def records(aerosol, rate=1e9):
    """Finalize a density source independently of particle capacity."""
    return source.finalize_particle_source(
        source.PotentialEventData(np.array([rate]), 1.0),
        InjectionComposition((0, 2, 1, 0)),
        aerosol.gas,
    )


def inventory(aerosol):
    """Compute mapped mass density including the particle-only lane."""
    p = aerosol.particles
    result = np.sum(p.masses * p.concentration[..., None], axis=1)
    result /= p.volume[:, None]
    result[:, 0] += aerosol.gas.concentration[:, 2]
    result[:, 1] += aerosol.gas.concentration[:, 1]
    return result


@pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
def test_commit_converts_density_to_counts_and_maps_species(volume):
    """Source densities become represented counts at the held volume."""
    aerosol, config = fixture(volume)
    p, gas = aerosol.particles, aerosol.gas
    before = inventory(aerosol)
    gas_before = gas.concentration.copy()
    arrays = [p.masses, p.concentration, p.charge, p.volume, gas.concentration]
    result = source.commit_particle_source(*records(aerosol), p, gas, config)
    per_event = np.array([0, 2, 1, 0]) * gas.molar_mass / AVOGADRO_NUMBER
    assert result.event_count_units == "count"
    np.testing.assert_allclose(result.represented_event_count, [1e9 * volume])
    np.testing.assert_allclose(p.concentration[0, 1], 1e9 * volume)
    np.testing.assert_allclose(p.masses[0, 1], per_event[[2, 1, 0]])
    np.testing.assert_array_equal(p.charge[0], [0, 2, 0, 0])
    np.testing.assert_allclose(gas.concentration, gas_before - 1e9 * per_event)
    np.testing.assert_array_equal(
        gas.concentration[:, [0, 3]], gas_before[:, [0, 3]]
    )
    np.testing.assert_allclose(
        inventory(aerosol), before, rtol=1e-12, atol=1e-30
    )
    assert all(
        left is right
        for left, right in zip(
            arrays,
            [p.masses, p.concentration, p.charge, p.volume, gas.concentration],
            strict=True,
        )
    )


def test_representative_scaling_preserves_density_and_inert_gas():
    """Volume reduction preserves old density and leaves inert gas untouched."""
    aerosol, config = fixture(capacity=2)
    config = replace(
        config,
        maximum_slot_weight=2e8,
        exhaustion_controls=ExhaustionControls(False, True),
        requested_scale=np.array([0.5]),
    )
    before = inventory(aerosol)
    gas_before = aerosol.gas.concentration.copy()
    result = source.commit_particle_source(
        *records(aerosol),
        aerosol.particles,
        aerosol.gas,
        config,
    )
    np.testing.assert_array_equal(aerosol.particles.volume, [0.125])
    np.testing.assert_array_equal(aerosol.particles.concentration[0, :1], [1.0])
    np.testing.assert_allclose(result.represented_event_count, [1.25e8])
    np.testing.assert_allclose(
        result.representation_reduction_event_count, [1.25e8]
    )
    np.testing.assert_array_equal(
        aerosol.gas.concentration[:, [0, 3]], gas_before[:, [0, 3]]
    )
    np.testing.assert_allclose(
        inventory(aerosol), before, rtol=1e-12, atol=1e-30
    )


@pytest.mark.parametrize(
    "failure", ["names", "mapping", "partitioning", "readonly", "volume", "pmf"]
)
def test_rejection_is_write_free(failure):
    """Invalid source state rejects before caller-visible writes."""
    aerosol, config = fixture()
    demand, diagnostics = records(aerosol)
    p, gas = aerosol.particles, aerosol.gas
    if failure == "names":
        gas.name.reverse()
    elif failure == "mapping":
        config = replace(
            config, species_map=NucleationSpeciesMap(tuple(gas.name), ((2, 0),))
        )
    elif failure == "partitioning":
        gas.partitioning[1] = False
    elif failure == "readonly":
        gas.concentration.flags.writeable = False
    elif failure == "volume":
        p.volume[0] = 0
    elif failure == "pmf":
        p.distribution_type = "discrete"
        p.volume[:] = 1
    arrays = [p.masses, p.concentration, p.charge, p.volume, gas.concentration]
    snapshots = [array.copy() for array in arrays]
    with pytest.raises((TypeError, ValueError)):
        source.commit_particle_source(demand, diagnostics, p, gas, config)
    for array, snapshot in zip(arrays, snapshots, strict=True):
        np.testing.assert_array_equal(array, snapshot)


def test_partial_commit_failure_restores_all_arrays(monkeypatch):
    """A failing writer restores every attempted particle and gas write."""
    aerosol, config = fixture()
    p, gas = aerosol.particles, aerosol.gas
    arrays = [p.masses, p.concentration, p.charge, p.volume, gas.concentration]
    before = [array.copy() for array in arrays]
    copyto = np.copyto

    def fail(target, candidate):
        copyto(target, candidate)
        if target is gas.concentration:
            raise RuntimeError("gas writer failed after mutation")

    monkeypatch.setattr(source.np, "copyto", fail)
    with pytest.raises(RuntimeError, match="gas writer"):
        source.commit_particle_source(*records(aerosol), p, gas, config)
    for array, snapshot in zip(arrays, before, strict=True):
        np.testing.assert_array_equal(array, snapshot)


def runnable(aerosol, config):
    """Build the native public API with an explicit full-gas species map."""
    strategy = ActivationNucleationStrategy(
        0.01,
        NucleationValidityDomain(
            ClosedInterval(0.0, 1e30),
            ClosedInterval(200.0, 400.0),
            ClosedInterval(1.0, 3.0),
        ),
        InjectionComposition((0, 2, 1, 0)),
        FormationMetadata(1e-9),
    )
    return Nucleation(
        NucleationSourceConfig(strategy, 2, config.species_map),
        NucleationCommitConfig(
            1e20,
            2.0,
            ExhaustionControls(),
            np.array([1.0]),
            np.array([0.1]),
            np.array([0.001]),
        ),
    )


def test_runnable_reads_current_environment_and_repeated_gas():
    """The runnable follows current environment and preceding gas depletion."""
    aerosol, config = fixture(capacity=8)
    process = runnable(aerosol, config)
    before = inventory(aerosol)
    rate = process.rate(aerosol)
    assert rate > 0
    aerosol.environment = EnvironmentData(
        np.array([298.15]),
        np.array([101325.0]),
        np.full((1, 4), 0.5),
    )
    assert process.rate(aerosol) == 0
    aerosol.environment.saturation_ratio[:] = 2.0
    expected_gas = aerosol.gas.concentration.copy()
    # Two half-second activation substeps: precursor A loses one molecule/event.
    expected_gas[:, 2] *= (1 - 0.01 * 0.5) ** 2
    expected_gas[:, 1] -= aerosol.gas.concentration[:, 2] - expected_gas[:, 2]
    assert process.execute(aerosol, 1.0, 2) is aerosol
    np.testing.assert_allclose(
        aerosol.gas.concentration, expected_gas, rtol=1e-12, atol=1e-30
    )
    np.testing.assert_allclose(
        inventory(aerosol), before, rtol=1e-12, atol=1e-30
    )


def test_native_resampling_retains_existing_species_inventory():
    """Resampling releases one slot while preserving resolved count moments."""
    aerosol, config = fixture(capacity=2)
    p = aerosol.particles
    p.masses[0, 1] = p.masses[0, 0]
    p.concentration[0, 1] = 3.0
    before = inventory(aerosol)
    config = replace(
        config, exhaustion_controls=ExhaustionControls(True, False)
    )
    result = source.commit_particle_source(
        *records(aerosol), p, aerosol.gas, config
    )
    np.testing.assert_array_equal(result.released_slot_count, [1])
    np.testing.assert_allclose(
        inventory(aerosol), before, rtol=1e-12, atol=1e-30
    )
    np.testing.assert_allclose(p.concentration[0, 0], 5.0)


def test_later_substep_failure_keeps_earlier_commit(monkeypatch):
    """Per-attempt atomicity retains the first successful source transaction."""
    import particula.dynamics.particle_process as process_module

    aerosol, config = fixture()
    expected, _ = fixture()
    process = runnable(aerosol, config)
    process.execute(expected, 0.5)
    commit = process_module._commit_particle_source
    attempts = 0

    def fail_second(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("second source attempt")
        return commit(*args)

    monkeypatch.setattr(process_module, "_commit_particle_source", fail_second)
    with pytest.raises(RuntimeError, match="second source attempt"):
        process.execute(aerosol, 1.0, 2)
    for name in ("masses", "concentration", "charge", "volume"):
        np.testing.assert_array_equal(
            getattr(aerosol.particles, name), getattr(expected.particles, name)
        )
    np.testing.assert_array_equal(
        aerosol.gas.concentration, expected.gas.concentration
    )


@pytest.mark.parametrize(
    "pairs",
    [((2, 0), (2, 1)), ((1, 0), (2, 0)), ((True, 0),), ((4, 0),), ((1, -1),)],
)
def test_species_map_rejects_ambiguous_pairs(pairs):
    """Duplicate, Boolean and out-of-range indices never acquire meaning."""
    with pytest.raises(ValueError):
        NucleationSpeciesMap(("inert", "B", "A", "spectator"), pairs)


def test_native_builder_preserves_explicit_map():
    """Fluent and dictionary configuration preserve the native lane map."""
    from particula.dynamics.nucleation.nucleation_builders import (
        NucleationSourceConfigBuilder,
    )

    aerosol, config = fixture()
    selected = runnable(aerosol, config).source_config
    builder = NucleationSourceConfigBuilder().set_parameters(
        {
            "strategy": selected.strategy,
            "precursor_index": 2,
            "species_map": config.species_map,
        }
    )
    assert builder.build() == selected
    with pytest.raises(TypeError, match="species_map"):
        builder.set_species_map(None)
    assert builder.build() == selected
