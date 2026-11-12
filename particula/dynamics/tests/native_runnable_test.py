"""Native runnable ownership, composition and scientific usage regressions."""

from types import SimpleNamespace

import numpy as np
import pytest
from particula.aerosol import Aerosol
from particula.dynamics import Coagulation, MassCondensation, WallLoss
from particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy import (
    BrownianCoagulationStrategy,
)
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
)
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.dynamics.wall_loss import ChargedWallLossStrategy
from particula.gas import EnvironmentData, GasData
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)
from particula.particles import ParticleData, get_slot_diagnostics
from particula.particles.activity_strategies import ActivityIdealMass
from particula.particles.surface_strategies import SurfaceStrategyVolume


def state(volume=0.25, kind="particle_resolved"):
    """Use unequal particle/gas widths and distinguishable gas order."""
    return Aerosol(
        particles=ParticleData(
            masses=np.array([[[2e-18, 3e-18], [4e-18, 5e-18]]]),
            concentration=np.array([[2.0, 2.0]]),
            charge=np.array([[1.0, -2.0]]),
            density=np.array([1000.0, 1200.0]),
            volume=np.array([volume]),
            distribution_type=kind,
            radius_grid=np.array([1e-8, 2e-8])
            if kind == "continuous_pdf"
            else None,
        ),
        gas=GasData(
            ["inert", "B", "A"],
            np.array([0.028, 0.046, 0.018]),
            np.array([[9e-15, 2e-15, 3e-15]]),
            np.array([False, True, True]),
        ),
        environment=EnvironmentData([298.15], [101325.0], [[0.0, 1.0, 1.0]]),
    )


def condensation(gas):
    """Return a real M3 strategy with explicit reordered lane provenance."""
    return CondensationIsothermal(
        molar_mass=gas.molar_mass.copy(),
        activity_strategy=ActivityIdealMass(),
        surface_strategy=SurfaceStrategyVolume(density=1000.0),
        vapor_pressure_strategy=ConstantVaporPressureStrategy(0.0),
        species_map=CondensationSpeciesMap(
            ("inert", "B", "A"), ((2, 0), (1, 1))
        ),
    )


@pytest.fixture(autouse=True)
def forbid_facades(forbid_legacy_facades):
    """Keep the native execution guard valid after facade modules disappear."""


@pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
def test_real_condensation_substeps_close_each_species_and_retain_objects(
    volume,
):
    """Real native usage couples all equal substeps to the current gas."""
    aerosol = state(volume)
    particles, gas, environment = (
        aerosol.particles,
        aerosol.gas,
        aerosol.environment,
    )
    arrays = (particles.masses, particles.concentration, gas.concentration)
    before_mass, before_gas = particles.masses.copy(), gas.concentration.copy()
    process = MassCondensation(condensation(gas))
    rates = process.rate(aerosol)
    np.testing.assert_array_equal(particles.masses, before_mass)
    np.testing.assert_array_equal(gas.concentration, before_gas)
    assert np.all(np.isfinite(rates))
    assert process.execute(aerosol, 4e8, 4) is aerosol
    assert aerosol.particles is particles and aerosol.gas is gas
    assert aerosol.environment is environment
    for actual, expected in zip(
        (particles.masses, particles.concentration, gas.concentration),
        arrays,
        strict=True,
    ):
        assert actual is expected
    gain = np.sum(
        (particles.masses - before_mass) * particles.concentration[..., None],
        axis=1,
    )
    loss = (before_gas[:, [2, 1]] - gas.concentration[:, [2, 1]]) * volume
    assert np.all(gain > 0)
    np.testing.assert_allclose(gain, loss, rtol=1e-12, atol=1e-30)
    assert gas.concentration[0, 0] == before_gas[0, 0]


def test_full_cycle_order_and_current_environment_replacement():
    """Each sequence substep reads state left by its preceding processes."""
    aerosol = state()
    events = []

    def condense(**kwargs):
        events.append(("condense", kwargs["time_step"], kwargs["temperature"]))
        aerosol.environment = EnvironmentData([310.0], [90000.0], [[0, 1, 1]])
        return kwargs["particle"], kwargs["gas_species"]

    def coagulate(**kwargs):
        events.append(("coagulate", kwargs["time_step"], kwargs["temperature"]))
        assert kwargs["pressure"] == 90000.0
        return kwargs["particle"]

    sequence = MassCondensation(SimpleNamespace(step=condense)) | Coagulation(
        SimpleNamespace(step=coagulate)
    )
    assert sequence.execute(aerosol, 2.0, 2) is aerosol
    assert events == [
        ("condense", 1.0, 298.15),
        ("coagulate", 1.0, 310.0),
        ("condense", 1.0, 310.0),
        ("coagulate", 1.0, 310.0),
    ]


@pytest.mark.parametrize("kind", ["discrete", "continuous_pdf"])
def test_density_coagulation_rate_keeps_measure_and_does_not_mutate(kind):
    """The runnable forwards M3 PMF/PDF rates without volume renormalization."""
    aerosol = state(1.0, kind)
    aerosol.particles = ParticleData(
        masses=np.array([[[2, 3], [4, 5], [8, 9], [16, 17]]], dtype=float)
        * 1e-18,
        concentration=np.array([[2.0, 3.0, 4.0, 5.0]]),
        charge=np.zeros((1, 4)),
        density=np.array([1000.0, 1200.0]),
        volume=np.ones(1),
        distribution_type=kind,
        radius_grid=np.array([1e-8, 2e-8, 3e-8, 4e-8])
        if kind == "continuous_pdf"
        else None,
    )
    strategy = BrownianCoagulationStrategy(kind)
    before = aerosol.particles.concentration.copy()
    expected = strategy.net_rate(aerosol.particles, 298.15, 101325.0)
    np.testing.assert_array_equal(Coagulation(strategy).rate(aerosol), expected)
    np.testing.assert_array_equal(aerosol.particles.concentration, before)


def test_wall_loss_retirement_makes_slots_reusable():
    """A removed charged representative becomes an exact canonical free slot."""
    aerosol = state()
    strategy = ChargedWallLossStrategy(
        1e-3,
        "spherical",
        chamber_radius=0.5,
        wall_potential=0,
        distribution_type="particle_resolved",
    )
    gas = aerosol.gas.concentration.copy()
    assert WallLoss(strategy).execute(aerosol, 1e30) is aerosol
    free, active_counts, free_counts = get_slot_diagnostics(aerosol.particles)
    np.testing.assert_array_equal(free, [[0, 1]])
    np.testing.assert_array_equal(active_counts, [0])
    np.testing.assert_array_equal(free_counts, [2])
    np.testing.assert_array_equal(aerosol.gas.concentration, gas)


@pytest.mark.parametrize("family", [MassCondensation, Coagulation, WallLoss])
@pytest.mark.parametrize("substeps", [0, -1, True, 1.5])
def test_invalid_substeps_reject_before_dispatch(family, substeps):
    """Native controls cannot silently skip processing or partially execute."""

    def forbidden(**kwargs):
        pytest.fail("invalid controls reached strategy")

    with pytest.raises(ValueError, match="sub_steps"):
        family(SimpleNamespace(step=forbidden)).execute(state(), 1.0, substeps)


@pytest.mark.parametrize("family", [MassCondensation, Coagulation, WallLoss])
def test_current_invalid_environment_rejects_before_dispatch(family):
    """Writable environment edits are validated before any process writer."""
    aerosol = state()
    aerosol.environment.temperature[0] = np.nan

    def forbidden(**kwargs):
        pytest.fail("invalid environment reached strategy")

    with pytest.raises(ValueError):
        family(SimpleNamespace(step=forbidden)).execute(aerosol, 0.0)


@pytest.mark.parametrize("operation", ["rate", "execute"])
@pytest.mark.parametrize("candidate", [None, object(), SimpleNamespace()])
def test_coagulation_rejects_non_native_aggregate(operation, candidate):
    """Reject obsolete/duck-typed aggregates before strategy access."""

    def forbidden(**kwargs):
        pytest.fail("non-native aggregate reached strategy")

    process = Coagulation(SimpleNamespace(step=forbidden, net_rate=forbidden))
    with pytest.raises(
        TypeError, match="Coagulation requires a native Aerosol"
    ):
        if operation == "rate":
            process.rate(candidate)
        else:
            process.execute(candidate, 1.0)


def test_coagulation_rejects_replacement_particle_result():
    """A strategy cannot replace the held population through its return value."""
    aerosol = state()
    held = aerosol.particles
    before = held.masses.copy()
    process = Coagulation(SimpleNamespace(step=lambda **kwargs: held.copy()))
    with pytest.raises(ValueError, match="retain its container"):
        process.execute(aerosol, 0.1)
    assert aerosol.particles is held
    np.testing.assert_array_equal(held.masses, before)
