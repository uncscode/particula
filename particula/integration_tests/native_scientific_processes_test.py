"""v0.3.0 construction through v0.3.0 science, without runnable or facade
adapters.
"""

import numpy as np
import numpy.testing as npt
import pytest
from particula import AerosolBuilder
from particula.dynamics import (
    BrownianCoagulationStrategy,
    ChargedWallLossStrategy,
    CondensationIsothermal,
    CondensationIsothermalStaggered,
    CondensationLatentHeat,
    SphericalWallLossStrategy,
)
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.gas import EnvironmentData, GasDataBuilder
from particula.gas.latent_heat_strategies import ConstantLatentHeat
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)
from particula.particles import (
    ParticleDataBuilder,
    create_resolved_particle_data,
)
from particula.particles.activity_strategies import ActivityIdealMass
from particula.particles.surface_strategies import SurfaceStrategyVolume

CONDENSATION = [
    CondensationIsothermal,
    CondensationIsothermalStaggered,
    CondensationLatentHeat,
]


@pytest.fixture(autouse=True)
def forbid_facades(forbid_legacy_facades):
    """Apply the removal-safe native execution boundary guard."""


def _aggregate(particles):
    """Build full gas order with a distinct, unmapped inert lane."""
    scale = (
        1e-15 if particles.distribution_type == "particle_resolved" else 1e-9
    )
    gas = (
        GasDataBuilder()
        .set_names(["inert", "B", "A"])
        .set_molar_mass([0.028, 0.046, 0.018])
        .set_concentration(np.array([9.0, 2.0, 3.0]) * scale)
        .set_partitioning([False, True, True])
        .build()
    )
    return (
        AerosolBuilder()
        .set_particles(particles)
        .set_gas(gas)
        .set_environment(EnvironmentData([298.15], [101325.0], [[0, 1, 1]]))
        .build()
    )


def _integrate(particle, values):
    """Sum raw counts or integrate PDF products independently of containers."""
    if particle.distribution_type != "continuous_pdf":
        return np.sum(values, axis=0)
    width = np.diff(particle.radius_grid)
    if values.ndim == 2:
        width = width[:, None]
    return np.sum((values[1:] + values[:-1]) * width / 2, axis=0)


def _inventory(particle):
    return _integrate(
        particle, particle.masses[0] * particle.concentration[0, :, None]
    )


def _condense(aerosol, strategy_type):
    """Check particle/gas deltas separately at the explicit strategy boundary."""
    particle, gas = aerosol.particles, aerosol.gas
    extra = {}
    if strategy_type is CondensationIsothermalStaggered:
        extra = dict(theta_mode="random", random_state=123, num_batches=4)
    elif strategy_type is CondensationLatentHeat:
        extra = dict(latent_heat_strategy=ConstantLatentHeat(2e6))
    strategy = strategy_type(
        molar_mass=gas.molar_mass.copy(),
        diffusion_coefficient=np.array([3e-5, 1e-5, 2e-5]),
        activity_strategy=ActivityIdealMass(),
        surface_strategy=SurfaceStrategyVolume(density=1000.0),
        vapor_pressure_strategy=ConstantVaporPressureStrategy(0.0),
        species_map=CondensationSpeciesMap(
            ("inert", "B", "A"), ((2, 0), (1, 1))
        ),
        **extra,
    )
    old_mass = particle.masses.copy()
    old_gas = gas.concentration.copy()
    result = strategy.step(
        particle,
        gas,
        float(aerosol.environment.temperature[0]),
        float(aerosol.environment.pressure[0]),
        0.1,
    )
    assert result[0] is particle and result[1] is gas
    gain = _integrate(
        particle,
        (particle.masses[0] - old_mass[0]) * particle.concentration[0, :, None],
    )
    loss = (
        old_gas[0, [2, 1]] - gas.concentration[0, [2, 1]]
    ) * particle.volume[0]
    assert np.all(gain > 0) and np.all(loss > 0)
    npt.assert_allclose(
        gain,
        loss,
        rtol=1e-12,
        atol=8 * np.finfo(float).eps * np.max(old_gas) * particle.volume[0],
    )
    assert gas.concentration[0, 0] == old_gas[0, 0]
    if strategy_type is CondensationLatentHeat:
        npt.assert_allclose(strategy.last_latent_heat_energy, gain.sum() * 2e6)


@pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("weight", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("strategy_type", CONDENSATION)
def test_resolved_native_aggregate_through_three_processes(
    volume, weight, strategy_type
):
    """Actual selection conserves mapped species before an explicit wall sink."""
    masses = np.tile([[2e-18, 3e-18], [5e-18, 7e-18]], (8, 1))
    particle = create_resolved_particle_data(
        masses=masses,
        density=[1000.0, 1200.0],
        concentration=weight,
        charge=np.tile([2.0, -1.0], 8),
        volume=volume,
    )
    aerosol = _aggregate(particle)
    arrays = [particle.masses, particle.concentration, particle.charge]
    protected = [
        particle.density,
        particle.volume,
        aerosol.gas.molar_mass,
        aerosol.gas.partitioning,
        aerosol.environment.temperature,
        aerosol.environment.pressure,
        aerosol.environment.saturation_ratio,
    ]
    snapshots = [x.copy() for x in protected]
    _condense(aerosol, strategy_type)
    gas_before = aerosol.gas.concentration.copy()
    mass_before = _inventory(particle)
    charge_before = np.dot(particle.charge[0], particle.concentration[0])
    coagulation = BrownianCoagulationStrategy("particle_resolved")
    coagulation.random_generator = np.random.default_rng(123)
    assert (
        coagulation.step(particle, 298.15, 101325.0, 1e18 * volume / weight)
        is particle
    )
    assert np.count_nonzero(particle.concentration) < len(masses)
    npt.assert_allclose(_inventory(particle), mass_before, rtol=2e-15, atol=0)
    assert (
        np.dot(particle.charge[0], particle.concentration[0]) == charge_before
    )

    before_wall_mass = particle.masses.copy()
    before_wall_count = particle.concentration.copy()
    before_wall_charge = particle.charge.copy()
    wall = ChargedWallLossStrategy(
        wall_eddy_diffusivity=1e-3,
        chamber_geometry="spherical",
        chamber_radius=0.5,
        wall_potential=0.0,
        distribution_type="particle_resolved",
    )
    wall.random_generator = np.random.default_rng(456)
    assert wall.step(particle, 298.15, 101325.0, 1e30) is particle
    removed = before_wall_count[0] > particle.concentration[0]
    assert np.any(removed)
    sink = np.sum(
        before_wall_mass[0, removed] * before_wall_count[0, removed, None],
        axis=0,
    )
    npt.assert_allclose(
        _inventory(particle) + sink, mass_before, rtol=2e-15, atol=0
    )
    npt.assert_array_equal(particle.charge[0, removed], 0)
    npt.assert_array_equal(
        particle.charge[0, ~removed], before_wall_charge[0, ~removed]
    )
    charge_sink = np.dot(
        before_wall_charge[0, removed], before_wall_count[0, removed]
    )
    npt.assert_allclose(
        np.dot(particle.charge[0], particle.concentration[0]) + charge_sink,
        charge_before,
        rtol=2e-15,
        atol=0,
    )
    npt.assert_array_equal(aerosol.gas.concentration, gas_before)
    assert aerosol.particles is particle
    for actual, original in zip(
        (particle.masses, particle.concentration, particle.charge),
        arrays,
        strict=True,
    ):
        assert actual is original
    for actual, original in zip(protected, snapshots, strict=True):
        npt.assert_array_equal(actual, original)


@pytest.mark.parametrize("kind", ["discrete", "continuous_pdf"])
@pytest.mark.parametrize("strategy_type", CONDENSATION)
def test_density_native_aggregate_through_three_processes(kind, strategy_type):
    """Retain explicit PDF quadrature through condensation, collisions and loss."""
    radius = np.geomspace(1e-9, 1e-6, 512)
    fraction = np.clip(0.5 + 0.3 * np.log(radius / 1e-7), 0.05, 0.95)
    mass = 4 * np.pi * radius**3 / 3 / (fraction / 1000 + (1 - fraction) / 1200)
    pdf = 1e9 * np.exp(-0.5 * (np.log(radius / 1e-7) / 0.35) ** 2)
    pdf /= radius * 0.35 * np.sqrt(2 * np.pi)
    counts = (
        pdf
        if kind == "continuous_pdf"
        else pdf * np.diff(radius, append=2 * radius[-1] - radius[-2])
    )
    builder = (
        ParticleDataBuilder()
        .set_distribution_type(kind)
        .set_masses(np.column_stack([mass * fraction, mass * (1 - fraction)]))
        .set_density(np.array([1000.0, 1200.0]))
        .set_concentration(
            counts, units="1/m^4" if kind == "continuous_pdf" else "1/m^3"
        )
        .set_charge(np.zeros(512))
        .set_volume(np.array(1.0))
    )
    if kind == "continuous_pdf":
        builder.set_radius_grid(radius)
    particle = builder.build()
    aerosol = _aggregate(particle)
    _condense(aerosol, strategy_type)
    gas_before = aerosol.gas.concentration.copy()
    coagulation = BrownianCoagulationStrategy(kind)
    old_count = particle.concentration.copy()
    old_mass = particle.masses.copy()
    old_charge = particle.charge.copy()
    expected_count = (
        old_count + 10 * coagulation.net_rate(particle, 298.15, 101325.0)[None]
    )
    # Preserve the accepted fixed-composition method. Species closure is not
    # promised here; the independent refinement characterization documents why.
    assert coagulation.step(particle, 298.15, 101325.0, 10.0) is particle
    assert not np.array_equal(particle.concentration, old_count)
    npt.assert_array_equal(particle.concentration, expected_count)
    npt.assert_array_equal(particle.masses, old_mass)
    npt.assert_array_equal(particle.charge, old_charge)
    npt.assert_allclose(
        _inventory(particle),
        _integrate(particle, old_mass[0] * expected_count[0, :, None]),
        rtol=2e-15,
        atol=0,
    )
    before_wall = _inventory(particle)
    old_count = particle.concentration.copy()
    old_mass = particle.masses.copy()
    wall = SphericalWallLossStrategy(1e-3, 0.5, kind)
    assert wall.step(particle, 298.15, 101325.0, 0.1) is particle
    sink = _integrate(
        particle,
        old_mass[0] * (old_count[0] - particle.concentration[0])[:, None],
    )
    assert np.all(sink > 0)
    npt.assert_allclose(
        _inventory(particle) + sink, before_wall, rtol=2e-15, atol=0
    )
    npt.assert_array_equal(particle.masses, old_mass)
    npt.assert_array_equal(aerosol.gas.concentration, gas_before)
    if kind == "continuous_pdf":
        npt.assert_array_equal(particle.radius_grid, radius)


@pytest.mark.parametrize("strategy_type", CONDENSATION)
@pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("solid_core", [False, True])
def test_complete_evaporation_remains_consumable_by_later_processes(
    strategy_type, volume, solid_core
):
    """Retire absent particles while preserving a surviving unmapped core."""
    from particula.dynamics.condensation.tests.species_mapping_test import (
        native_strategy,
    )
    from particula.particles import get_slot_diagnostics

    masses = np.array([[2e-18, 3e-18, 0], [5e-18, 7e-18, 0]])
    if solid_core:
        masses[1, 2] = 11e-18
    particle = create_resolved_particle_data(
        masses=masses,
        density=[1000.0, 1200.0, 1500.0],
        concentration=2.0,
        charge=[1.0, -2.0],
        volume=volume,
    )
    aerosol = _aggregate(particle)
    gas = aerosol.gas
    gas.concentration[0, 1:] = 0
    mapping = CondensationSpeciesMap(("inert", "B", "A"), ((2, 0), (1, 1)))
    extra = {}
    if strategy_type is CondensationIsothermalStaggered:
        extra = dict(theta_mode="half", num_batches=2, shuffle_each_step=False)
    elif strategy_type is CondensationLatentHeat:
        extra = dict(latent_heat_strategy=ConstantLatentHeat(2e6))
    strategy = native_strategy(strategy_type, gas, mapping, **extra)
    arrays = (particle.masses, particle.concentration, particle.charge)
    strategy.step(particle, gas, 298, 101325, 1e6, 1.8e-5)
    assert all(
        a is b
        for a, b in zip(
            arrays,
            (particle.masses, particle.concentration, particle.charge),
            strict=True,
        )
    )
    npt.assert_array_equal(particle.masses[0, :, :2], 0)
    npt.assert_array_equal(
        particle.concentration, [[0, 2 if solid_core else 0]]
    )
    npt.assert_array_equal(particle.charge, [[0, -2 if solid_core else 0]])
    npt.assert_allclose(
        gas.concentration[0, [2, 1]] * volume,
        2 * masses[:, :2].sum(axis=0),
        rtol=1e-12,
        atol=1e-30,
    )
    get_slot_diagnostics(particle)
    if strategy_type is CondensationLatentHeat:
        npt.assert_allclose(
            strategy.last_latent_heat_energy,
            -2 * masses[:, :2].sum() * 2e6,
            rtol=1e-12,
            atol=1e-30,
        )
    BrownianCoagulationStrategy("particle_resolved").step(
        particle,
        298,
        101325,
        0.01,
    )
    SphericalWallLossStrategy(
        wall_eddy_diffusivity=1e-3,
        chamber_radius=1.0,
        distribution_type="particle_resolved",
    ).step(particle, 298, 101325, 0.01)
