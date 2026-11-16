"""Native continuous condensation against independent growth and budgets."""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.integrate import quad

from particula.aerosol import Aerosol
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
)
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.execution.condensation_ode import CondensationODE
from particula.gas import EnvironmentData, GasData
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)
from particula.particles import ParticleData
from particula.particles.activity_strategies import ActivityIdealMass
from particula.particles.surface_strategies import SurfaceStrategyVolume


def case(boxes=1):
    """Create two slots, non-unit volume and reversed full-gas map."""
    mass = 4 * np.pi * 1000 * 1e-18 / 3
    volume = np.linspace(0.3, 0.8, boxes)
    p = ParticleData(
        masses=np.tile(np.array([[[mass], [0.0]]]), (boxes, 1, 1)),
        concentration=np.column_stack([1e9 * volume, np.zeros(boxes)]),
        charge=np.zeros((boxes, 2)),
        density=np.array([1000.0]),
        volume=volume,
        distribution_type="particle_resolved",
    )
    g = GasData(
        name=["inert", "vapor"],
        molar_mass=np.array([0.028, 0.018]),
        concentration=np.tile([[0.1, 1e-5]], (boxes, 1)),
        partitioning=np.array([False, True]),
    )
    e = EnvironmentData(
        temperature=np.full(boxes, 298.15),
        pressure=np.full(boxes, 101325.0),
        saturation_ratio=np.zeros((boxes, 2)),
    )
    strategy = CondensationIsothermal(
        molar_mass=np.array([0.028, 0.018]),
        diffusion_coefficient=2e-5,
        activity_strategy=ActivityIdealMass(),
        surface_strategy=SurfaceStrategyVolume(density=1000.0),
        vapor_pressure_strategy=ConstantVaporPressureStrategy(0.0),
        species_map=CondensationSpeciesMap(("inert", "vapor"), ((1, 0),)),
    )
    return Aerosol(particles=p, gas=g, environment=e), strategy


def reference_duration():
    """Integrate dt/d(r/r0) directly to target r/r0=1.2."""
    mean_free_path = (2 * 1.8e-5 / 101325) / np.sqrt(
        8 * 0.018 / (np.pi * 8.31446261815324 * 298.15)
    )
    mass = 4 * np.pi * 1000 * 1e-18 / 3

    def inverse(x):
        kn = mean_free_path / (1e-6 * x)
        correction = 0.75 * (1 + kn) / (kn**2 + 1.283 * kn + 0.75)
        gas = 1e-5 - 1e9 * mass * (x**3 - 1)
        return 1000 * 1e-12 * x / (2e-5 * correction * gas)

    return quad(inverse, 1, 1.2, epsabs=1e-12, epsrel=1e-12)[0]


@pytest.mark.parametrize("method", ["rk23", "ros2"])
def test_native_growth_reference_and_all_species_budget(method):
    """Both methods reach independent radius while preserving inert/empty lanes."""
    aerosol, strategy = case()
    initial = aerosol.particles.masses.copy()
    total = (
        aerosol.particles.species_mass_density[0, 0]
        + aerosol.gas.concentration[0, 1]
    )
    owner = CondensationODE(
        aerosol,
        strategy,
        method=method,
        rtol=1e-5,
        particle_atol=1e-23,
        gas_atol=1e-13,
        dynamic_viscosity=1.8e-5,
    )
    duration = reference_duration()
    while owner.time < duration:
        owner.step(duration)
    result = owner.snapshot()
    npt.assert_allclose(result.particles.radii[0, 0] / 1e-6, 1.2, rtol=2e-5)
    npt.assert_allclose(
        result.particles.species_mass_density[0, 0]
        + result.gas.concentration[0, 1],
        total,
        rtol=3e-13,
    )
    npt.assert_array_equal(result.particles.masses[0, 1], [0])
    npt.assert_array_equal(result.gas.concentration[:, 0], [0.1])
    npt.assert_array_equal(aerosol.particles.masses, initial)
    # Snapshots are detached from accepted state and each other.
    result.particles.masses[:] = 99
    assert owner.snapshot().particles.masses.max() < 1e-12


def test_batched_nonunit_volumes_match_individual_box():
    """Shared controller retains native counts/volume and independent boxes."""
    aerosol, strategy = case(2)
    owner = CondensationODE(aerosol, strategy)
    while owner.time < 0.1:
        owner.step(0.1)
    output = owner.snapshot()
    npt.assert_allclose(
        output.particles.masses[0], output.particles.masses[1], rtol=1e-14
    )
    npt.assert_allclose(
        output.gas.concentration[0], output.gas.concentration[1], rtol=1e-14
    )
    npt.assert_array_equal(
        output.particles.concentration, aerosol.particles.concentration
    )


def test_forcing_uses_absolute_time_and_owner_configuration_is_detached():
    """Forcing and caller edits cannot leak into retained process configuration."""
    aerosol, strategy = case()
    owner = CondensationODE(
        aerosol, strategy, time=2, forcing=lambda t: (298.15 + t, 101325.0)
    )
    strategy.diffusion_coefficient = 999
    aerosol.gas.concentration[:] = 0
    while owner.time < 2.01:
        owner.step(2.01)
    result = owner.snapshot()
    npt.assert_allclose(result.environment.temperature, [300.16])
    assert result.gas.concentration[0, 0] == 0.1


def test_failed_trial_retains_native_snapshot_and_time():
    """Exhausted trial budget cannot publish partial masses or gas state."""
    aerosol, strategy = case()
    owner = CondensationODE(
        aerosol,
        strategy,
        first_step=100,
        max_step=100,
        max_trials=1,
        dynamic_viscosity=1.8e-5,
    )
    with pytest.raises(RuntimeError):
        owner.step(100)
    npt.assert_array_equal(
        owner.snapshot().particles.masses, aerosol.particles.masses
    )
    npt.assert_array_equal(
        owner.snapshot().gas.concentration, aerosol.gas.concentration
    )
    assert owner.time == 0
