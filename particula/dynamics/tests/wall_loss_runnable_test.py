"""Tests for the WallLoss runnable process."""

# ruff: noqa: D100,D101,D102,D103,D107

import numpy as np
import pytest
from particula.aerosol import Aerosol
from particula.dynamics import (
    Coagulation,
    MassCondensation,
    WallLoss,
)
from particula.dynamics.coagulation.coagulation_strategy import (
    coagulation_strategy_abc,
)
from particula.dynamics.condensation.condensation_strategies import (
    CondensationStrategy,
)
from particula.dynamics.wall_loss.wall_loss_strategies import (
    RectangularWallLossStrategy,
    SphericalWallLossStrategy,
    WallLossStrategy,
)
from particula.gas import EnvironmentData, GasData
from particula.particles import (
    ParticleData,
)

CoagulationStrategyABC = coagulation_strategy_abc.CoagulationStrategyABC


def _native_aerosol(kind):
    """Build explicit size lanes with independent inert gas state."""
    radius = np.geomspace(1e-8, 1e-6, 20)
    volume = 1e-6 if kind == "particle_resolved" else 1.0
    particle = ParticleData(
        masses=(4 / 3 * np.pi * radius**3 * 1000)[None, :, None],
        concentration=np.full(
            (1, radius.size), 1.0 if kind == "particle_resolved" else 1e6
        ),
        charge=np.zeros((1, radius.size)),
        density=np.array([1000.0]),
        volume=np.array([volume]),
        distribution_type=kind,
    )
    return Aerosol(
        particles=particle,
        gas=GasData(
            name=["inert"],
            molar_mass=np.array([0.029]),
            concentration=np.zeros((1, 1)),
            partitioning=np.array([False]),
        ),
        environment=EnvironmentData(
            temperature=np.array([298.15]),
            pressure=np.array([101325.0]),
            saturation_ratio=np.ones((1, 1)),
        ),
    )


@pytest.fixture()
def aerosol():
    """Return a discrete native aerosol."""
    return _native_aerosol("discrete")


@pytest.fixture()
def aerosol_resolved():
    """Return a resolved native aerosol with explicit representative volume."""
    return _native_aerosol("particle_resolved")


@pytest.fixture()
def spherical_strategy() -> SphericalWallLossStrategy:
    """Return a discrete spherical wall loss strategy."""
    return SphericalWallLossStrategy(
        wall_eddy_diffusivity=1e-3,
        chamber_radius=0.5,
        distribution_type="discrete",
    )


@pytest.fixture()
def rectangular_strategy() -> RectangularWallLossStrategy:
    """Return a discrete rectangular wall loss strategy."""
    return RectangularWallLossStrategy(
        wall_eddy_diffusivity=1e-4,
        chamber_dimensions=(1.0, 0.5, 0.5),
        distribution_type="discrete",
    )


class DummyCoagulationStrategy(CoagulationStrategyABC):
    """Minimal coagulation strategy returning zero rates."""

    def __init__(self):
        super().__init__(distribution_type="discrete")

    def dimensionless_kernel(self, diffusive_knudsen, coulomb_potential_ratio):
        return np.zeros_like(diffusive_knudsen)

    def kernel(self, particle, temperature, pressure):
        return np.zeros((particle.n_particles, particle.n_particles))


class DummyCondensationStrategy(CondensationStrategy):
    """Minimal condensation strategy that leaves state unchanged."""

    def __init__(self):
        super().__init__(
            molar_mass=0.018,
            diffusion_coefficient=1.0,
            accommodation_coefficient=0.0,
            update_gases=False,
        )

    def mass_transfer_rate(self, *_, **__):
        return 0.0

    def rate(self, particle, *_, **__):
        return np.zeros_like(particle.concentration[0])

    def step(self, particle, gas_species, *_, **__):
        return particle, gas_species


class SpyWallLossStrategy(WallLossStrategy):
    """Spy strategy to confirm sub-step time splitting."""

    def __init__(self):
        super().__init__(
            wall_eddy_diffusivity=1e-3,
            distribution_type="discrete",
        )
        self.time_steps: list[float] = []

    def loss_coefficient(self, particle, temperature, pressure):
        return np.zeros_like(particle.concentration[0])

    def loss_coefficient_for_particles(
        self,
        particle_radius,
        particle_density,
        temperature,
        pressure,
    ):
        return np.zeros_like(particle_radius)

    def step(self, particle, temperature, pressure, time_step):
        self.time_steps.append(time_step)
        return particle


def test_init_with_spherical_strategy(spherical_strategy):
    wall_loss = WallLoss(wall_loss_strategy=spherical_strategy)
    assert wall_loss.wall_loss_strategy is spherical_strategy


def test_init_with_rectangular_strategy(rectangular_strategy):
    wall_loss = WallLoss(wall_loss_strategy=rectangular_strategy)
    assert wall_loss.wall_loss_strategy is rectangular_strategy


def test_execute_reduces_concentration(aerosol, spherical_strategy):
    wall_loss = WallLoss(wall_loss_strategy=spherical_strategy)
    initial = aerosol.particles.concentration[0].copy()
    updated = wall_loss.execute(aerosol, time_step=1.0)
    assert np.all(updated.particles.concentration[0] <= initial)


def test_execute_zero_time_step_no_change(aerosol, spherical_strategy):
    wall_loss = WallLoss(wall_loss_strategy=spherical_strategy)
    initial = aerosol.particles.concentration[0].copy()
    updated = wall_loss.execute(aerosol, time_step=0.0)
    np.testing.assert_array_equal(updated.particles.concentration[0], initial)


def test_execute_clamps_non_negative(aerosol, spherical_strategy, monkeypatch):
    """The runnable clamps successful custom strategy output in place."""
    array = aerosol.particles.concentration

    def negative_output(particle, **_):
        particle.concentration[0, 0] = -1.0
        return particle

    monkeypatch.setattr(spherical_strategy, "step", negative_output)
    wall_loss = WallLoss(wall_loss_strategy=spherical_strategy)
    assert wall_loss.execute(aerosol, time_step=1.0) is aerosol
    assert aerosol.particles.concentration is array
    assert np.all(array >= 0.0)
    assert array[0, 0] == 0.0


def test_oversized_native_step_rejects_before_mutation(
    aerosol, spherical_strategy
):
    """Native finite-step admission rejects a negative concentration proposal."""
    original = aerosol.particles.concentration.copy()
    with pytest.raises(ValueError, match="smaller finite time_step"):
        WallLoss(spherical_strategy).execute(aerosol, time_step=1e8)
    np.testing.assert_array_equal(aerosol.particles.concentration, original)


def test_execute_honors_sub_steps(aerosol):
    spy_strategy = SpyWallLossStrategy()
    wall_loss = WallLoss(wall_loss_strategy=spy_strategy)
    wall_loss.execute(aerosol, time_step=3.0, sub_steps=3)
    assert spy_strategy.time_steps == [1.0, 1.0, 1.0]


def test_rate_negative_and_shape(aerosol, spherical_strategy):
    wall_loss = WallLoss(wall_loss_strategy=spherical_strategy)
    rates = wall_loss.rate(aerosol)
    assert rates.shape == aerosol.particles.concentration[0].shape
    assert np.all(rates <= 0.0)


def test_rectangular_strategy_smoke(aerosol, rectangular_strategy):
    wall_loss = WallLoss(wall_loss_strategy=rectangular_strategy)
    updated = wall_loss.execute(aerosol, time_step=1.0)
    rates = wall_loss.rate(updated)
    assert np.all(rates <= 0.0)
    assert updated.particles.concentration.sum() >= 0.0


def test_particle_resolved_distribution_smoke(aerosol_resolved):
    strategy = SphericalWallLossStrategy(
        wall_eddy_diffusivity=1e-3,
        chamber_radius=0.5,
        distribution_type="particle_resolved",
    )
    wall_loss = WallLoss(wall_loss_strategy=strategy)
    updated = wall_loss.execute(aerosol_resolved, time_step=0.5)
    rates = wall_loss.rate(updated)
    assert np.all(rates <= 0.0)
    assert updated.particles.concentration.sum() >= 0.0


def test_process_chaining_with_coagulation(aerosol, spherical_strategy):
    coagulation = Coagulation(coagulation_strategy=DummyCoagulationStrategy())
    wall_loss = WallLoss(wall_loss_strategy=spherical_strategy)
    combined = coagulation | wall_loss
    updated = combined.execute(aerosol, time_step=1.0)
    assert isinstance(updated, Aerosol)
    assert np.all(updated.particles.concentration[0] >= 0.0)


def test_process_chaining_with_condensation(aerosol, spherical_strategy):
    condensation = MassCondensation(
        condensation_strategy=DummyCondensationStrategy()
    )
    wall_loss = WallLoss(wall_loss_strategy=spherical_strategy)
    combined = condensation | wall_loss
    updated = combined.execute(aerosol, time_step=1.0)
    assert isinstance(updated, Aerosol)
    assert np.all(updated.particles.concentration[0] >= 0.0)


@pytest.mark.parametrize("operation", ["rate", "execute"])
def test_non_native_aggregate_rejected_before_strategy(
    operation, spherical_strategy, monkeypatch
):
    """Former aggregate shapes reject before strategy or mutation."""
    runnable = WallLoss(spherical_strategy)
    monkeypatch.setattr(
        spherical_strategy, "step", lambda **_: pytest.fail("step called")
    )
    monkeypatch.setattr(
        spherical_strategy, "rate", lambda **_: pytest.fail("rate called")
    )
    with pytest.raises(TypeError, match="WallLoss requires a native Aerosol"):
        if operation == "rate":
            runnable.rate(object())
        else:
            runnable.execute(object(), 1.0)
