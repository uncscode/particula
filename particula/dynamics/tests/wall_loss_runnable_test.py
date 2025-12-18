"""Tests for the WallLoss runnable process."""

import importlib
import pathlib
import sys

import numpy as np
import pytest

_WORKTREE_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))
for _module in [
    "particula",
    "particula.dynamics",
    "particula.dynamics.wall_loss",
]:
    sys.modules.pop(_module, None)

from particula.aerosol import Aerosol
from particula.dynamics import Coagulation, MassCondensation, WallLoss
from particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc import (
    CoagulationStrategyABC,
)
from particula.dynamics.condensation.condensation_strategies import (
    CondensationStrategy,
)
from particula.dynamics.wall_loss.wall_loss_strategies import (
    RectangularWallLossStrategy,
    SphericalWallLossStrategy,
    WallLossStrategy,
)
from particula.gas.atmosphere import Atmosphere
from particula.particles import (
    PresetParticleRadiusBuilder,
    PresetResolvedParticleMassBuilder,
)


@pytest.fixture()
def atmosphere() -> Atmosphere:
    """Return a simple atmosphere fixture."""
    return Atmosphere(
        temperature=298.15,
        total_pressure=101325.0,
        partitioning_species=[],  # type: ignore[arg-type]
        gas_only_species=[],  # type: ignore[arg-type]
    )


@pytest.fixture()
def particle():
    """Return a preset particle representation."""
    return PresetParticleRadiusBuilder().build()


@pytest.fixture()
def particle_resolved():
    """Return a particle-resolved representation."""
    return PresetResolvedParticleMassBuilder().set_volume(1e-6, "m^3").build()


@pytest.fixture()
def aerosol(atmosphere: Atmosphere, particle) -> Aerosol:
    """Return an aerosol with preset particle representation."""
    return Aerosol(atmosphere=atmosphere, particles=particle)


@pytest.fixture()
def aerosol_resolved(atmosphere: Atmosphere, particle_resolved) -> Aerosol:
    """Return an aerosol with particle-resolved representation."""
    return Aerosol(atmosphere=atmosphere, particles=particle_resolved)


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
        return np.zeros_like(particle.get_concentration())


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
        return np.zeros_like(particle.get_concentration())

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
        return np.zeros_like(particle.get_concentration())

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
    initial = aerosol.particles.get_concentration(clone=True)
    updated = wall_loss.execute(aerosol, time_step=1.0)
    assert np.all(updated.particles.get_concentration() <= initial)


def test_execute_zero_time_step_no_change(aerosol, spherical_strategy):
    wall_loss = WallLoss(wall_loss_strategy=spherical_strategy)
    initial = aerosol.particles.get_concentration(clone=True)
    updated = wall_loss.execute(aerosol, time_step=0.0)
    np.testing.assert_array_equal(
        updated.particles.get_concentration(), initial
    )


def test_execute_clamps_non_negative(aerosol, spherical_strategy):
    wall_loss = WallLoss(wall_loss_strategy=spherical_strategy)
    updated = wall_loss.execute(aerosol, time_step=1e8)
    concentrations = updated.particles.get_concentration()
    assert np.all(concentrations >= 0.0)
    assert np.any(concentrations == 0.0)


def test_execute_honors_sub_steps(aerosol, atmosphere):
    spy_strategy = SpyWallLossStrategy()
    wall_loss = WallLoss(wall_loss_strategy=spy_strategy)
    wall_loss.execute(aerosol, time_step=3.0, sub_steps=3)
    assert spy_strategy.time_steps == [1.0, 1.0, 1.0]


def test_rate_negative_and_shape(aerosol, spherical_strategy):
    wall_loss = WallLoss(wall_loss_strategy=spherical_strategy)
    rates = wall_loss.rate(aerosol)
    assert rates.shape == aerosol.particles.get_concentration().shape
    assert np.all(rates <= 0.0)


def test_rectangular_strategy_smoke(aerosol, rectangular_strategy):
    wall_loss = WallLoss(wall_loss_strategy=rectangular_strategy)
    updated = wall_loss.execute(aerosol, time_step=1.0)
    rates = wall_loss.rate(updated)
    assert np.all(rates <= 0.0)
    assert updated.particles.get_total_concentration() >= 0.0


def test_continuous_distribution_smoke(aerosol_resolved, atmosphere):
    strategy = SphericalWallLossStrategy(
        wall_eddy_diffusivity=1e-3,
        chamber_radius=0.5,
        distribution_type="particle_resolved",
    )
    wall_loss = WallLoss(wall_loss_strategy=strategy)
    updated = wall_loss.execute(aerosol_resolved, time_step=0.5)
    rates = wall_loss.rate(updated)
    assert np.all(rates <= 0.0)
    assert updated.particles.get_total_concentration() >= 0.0


def test_process_chaining_with_coagulation(aerosol, spherical_strategy):
    coagulation = Coagulation(coagulation_strategy=DummyCoagulationStrategy())
    wall_loss = WallLoss(wall_loss_strategy=spherical_strategy)
    combined = coagulation | wall_loss
    updated = combined.execute(aerosol, time_step=1.0)
    assert isinstance(updated, Aerosol)
    assert np.all(updated.particles.get_concentration() >= 0.0)


def test_process_chaining_with_condensation(aerosol, spherical_strategy):
    condensation = MassCondensation(
        condensation_strategy=DummyCondensationStrategy()
    )
    wall_loss = WallLoss(wall_loss_strategy=spherical_strategy)
    combined = condensation | wall_loss
    updated = combined.execute(aerosol, time_step=1.0)
    assert isinstance(updated, Aerosol)
    assert np.all(updated.particles.get_concentration() >= 0.0)
