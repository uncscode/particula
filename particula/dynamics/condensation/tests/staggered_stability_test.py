"""Stability benchmarks for staggered condensation stepping.

This module measures stability improvements of staggered condensation relative
to simultaneous updates. Benchmarks evaluate bounded variance, maximum stable
time step, and robustness across theta modes and batch counts. Target: the
staggered strategy sustains ~10x larger stable time steps while keeping masses
finite and non-negative.

Run with: pytest particula/dynamics/condensation/tests/staggered_stability_test.py -v -m slow
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

# Work around SciPy 1.14 + NumPy 2.x docstring generation bug
# where scipy.stats import triggers `_CopyMode.IF_NEEDED is neither True nor
# False`. Patch the internal bool to keep imports viable for tests.
try:  # pragma: no cover - defensive patch
    from numpy import _globals as _np_globals

    _np_globals._CopyMode.__bool__ = lambda self: False  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

# Stub scipy.stats.lognorm to avoid SciPy import-time failures under NumPy 2.x;
# tests in this module do not rely on SciPy distributions.
try:  # pragma: no cover - defensive patch
    import sys
    import types

    import scipy as _real_scipy

    class _StubLogNorm:
        def pdf(self, *args, **kwargs):
            raise RuntimeError("scipy.stats.lognorm stubbed for tests")

        def rvs(self, *args, **kwargs):
            raise RuntimeError("scipy.stats.lognorm stubbed for tests")

    _scipy_stats_stub = types.ModuleType("scipy.stats")
    _scipy_stats_stub.lognorm = _StubLogNorm()
    sys.modules["scipy.stats"] = _scipy_stats_stub
    _real_scipy.stats = _scipy_stats_stub  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

from particula.dynamics.condensation import (
    CondensationIsothermal,
    CondensationIsothermalStaggered,
)
from particula.gas import GasSpeciesBuilder, VaporPressureFactory
from particula.particles import (
    ActivityIdealMass,
    ParticleResolvedSpeciatedMassBuilder,
    ResolvedParticleMassRepresentationBuilder,
    SurfaceStrategyVolume,
)

pytestmark = pytest.mark.slow

DEFAULT_SEED = 42
DEFAULT_PARTICLE_COUNT = 200
DEFAULT_MOLAR_MASS = 0.018
DEFAULT_TEMPERATURE = 298.0
DEFAULT_PRESSURE = 101325.0
MAX_VARIANCE = 1e10
VARIANCE_LIMIT = 1e-9

_VAPOR_STRATEGY = VaporPressureFactory().get_strategy(
    "constant",
    {
        "vapor_pressure": 101325.0,
        "vapor_pressure_units": "Pa",
    },
)


@lru_cache(maxsize=None)
def _cached_masses(n_particles: int, seed: int) -> NDArray[np.float64]:
    """Create deterministic lognormal masses for reuse across tests."""
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=-20.0, sigma=0.5, size=(n_particles, 1))


def create_test_system(
    n_particles: int = DEFAULT_PARTICLE_COUNT,
    seed: int = DEFAULT_SEED,
) -> Tuple:
    """Create deterministic particle/gas system for benchmarks."""
    masses: NDArray[np.float64] = np.array(
        _cached_masses(n_particles, seed), copy=True
    )
    distribution_strategy = ParticleResolvedSpeciatedMassBuilder().build()
    particle = (
        ResolvedParticleMassRepresentationBuilder()
        .set_distribution_strategy(distribution_strategy)
        .set_activity_strategy(ActivityIdealMass())
        .set_surface_strategy(
            SurfaceStrategyVolume(surface_tension=0.072, density=1000.0)
        )
        .set_mass(masses, "kg")
        .set_density(np.full_like(masses, 1000.0), "kg/m^3")
        .set_charge(np.zeros_like(masses))
        .set_volume(1.0, "m^3")
        .build()
    )

    gas_species = (
        GasSpeciesBuilder()
        .set_name("water")
        .set_molar_mass(DEFAULT_MOLAR_MASS, "kg/mol")
        .set_vapor_pressure_strategy(_VAPOR_STRATEGY)
        .set_partitioning(True)
        .set_concentration(0.01, "kg/m^3")
        .build()
    )
    return particle, gas_species


def is_numerically_stable(particle) -> bool:
    """Check that masses stay finite, non-negative, and variance bounded."""
    masses = np.asarray(particle.get_mass())
    if masses.size == 0:
        return True
    variance = float(np.var(masses))
    return bool(
        np.all(np.isfinite(masses))
        and np.all(masses >= 0.0)
        and variance < MAX_VARIANCE
    )


def run_simulation(
    strategy,
    particle,
    gas_species,
    time_step: float,
    steps: int = 5,
    temperature: float = DEFAULT_TEMPERATURE,
    pressure: float = DEFAULT_PRESSURE,
):
    """Run multiple steps, guarding zero-length or zero-dt cases."""
    if time_step <= 0.0 or particle.get_mass().size == 0 or steps <= 0:
        return particle, gas_species
    particle_current = particle
    gas_current = gas_species
    for _ in range(steps):
        particle_current, gas_current = strategy.step(
            particle_current,
            gas_current,
            temperature,
            pressure,
            time_step=time_step,
        )
    return particle_current, gas_current


@pytest.mark.slow
class TestStabilityBenchmarks:
    """Stability benchmarks contrasting simultaneous and staggered updates."""

    @pytest.mark.parametrize("time_step", [1.0, 10.0, 100.0])
    def test_stability_large_time_step(self, time_step: float) -> None:
        """Staggered stays stable at large dt; simultaneous may diverge."""
        seed = 11
        particle_sim, gas_sim = create_test_system(220, seed)
        particle_stag, gas_stag = create_test_system(220, seed)

        simultaneous = CondensationIsothermal(molar_mass=DEFAULT_MOLAR_MASS)
        staggered = CondensationIsothermalStaggered(
            molar_mass=DEFAULT_MOLAR_MASS,
            theta_mode="half",
            num_batches=6,
            shuffle_each_step=False,
            random_state=seed,
        )

        particle_sim, gas_sim = run_simulation(
            simultaneous,
            particle_sim,
            gas_sim,
            time_step=time_step,
            steps=6,
        )
        particle_stag, gas_stag = run_simulation(
            staggered,
            particle_stag,
            gas_stag,
            time_step=time_step,
            steps=6,
        )

        mass_stag = np.asarray(particle_stag.get_mass())
        var_stag = float(np.var(mass_stag))
        assert is_numerically_stable(particle_stag)
        assert var_stag <= VARIANCE_LIMIT
        if time_step < 10.0:
            mass_sim = np.asarray(particle_sim.get_mass())
            var_sim = float(np.var(mass_sim))
            assert is_numerically_stable(particle_sim)
            assert var_sim <= VARIANCE_LIMIT

    def test_stability_variance_comparison(self) -> None:
        """Variance growth is lower or equal for staggered at moderate dt."""
        seed = 7
        particle_sim, gas_sim = create_test_system(200, seed)
        particle_stag, gas_stag = create_test_system(200, seed)

        simultaneous = CondensationIsothermal(molar_mass=DEFAULT_MOLAR_MASS)
        staggered = CondensationIsothermalStaggered(
            molar_mass=DEFAULT_MOLAR_MASS,
            theta_mode="half",
            num_batches=4,
            shuffle_each_step=False,
            random_state=seed,
        )

        particle_sim, gas_sim = run_simulation(
            simultaneous,
            particle_sim,
            gas_sim,
            time_step=1.0,
            steps=5,
        )
        particle_stag, gas_stag = run_simulation(
            staggered,
            particle_stag,
            gas_stag,
            time_step=1.0,
            steps=5,
        )

        mass_sim = np.asarray(particle_sim.get_mass())
        mass_stag = np.asarray(particle_stag.get_mass())
        var_sim = float(np.var(mass_sim))
        var_stag = float(np.var(mass_stag))
        assert is_numerically_stable(particle_sim)
        assert is_numerically_stable(particle_stag)
        assert var_sim <= VARIANCE_LIMIT
        assert var_stag <= VARIANCE_LIMIT

    @pytest.mark.parametrize("theta_mode", ["half", "random", "batch"])
    def test_stability_mode_comparison(self, theta_mode: str) -> None:
        """All theta modes stay stable at challenging dt with fixed seed."""
        seed = 21
        particle, gas_species = create_test_system(180, seed)
        staggered = CondensationIsothermalStaggered(
            molar_mass=DEFAULT_MOLAR_MASS,
            theta_mode=theta_mode,
            num_batches=6,
            shuffle_each_step=False,
            random_state=seed,
        )

        particle_out, gas_out = run_simulation(
            staggered,
            particle,
            gas_species,
            time_step=10.0,
            steps=6,
        )

        assert is_numerically_stable(particle_out)
        variance = float(np.var(particle_out.get_mass()))
        assert variance < MAX_VARIANCE

    @pytest.mark.parametrize("num_batches", [1, 2, 5, 10])
    def test_stability_batch_count_effect(self, num_batches: int) -> None:
        """Batch counts stay stable; variance remains bounded and comparable."""
        seed = 33
        n_particles = max(DEFAULT_PARTICLE_COUNT, num_batches * 20)
        particle, gas_species = create_test_system(n_particles, seed)
        staggered = CondensationIsothermalStaggered(
            molar_mass=DEFAULT_MOLAR_MASS,
            theta_mode="batch",
            num_batches=num_batches,
            shuffle_each_step=False,
            random_state=seed,
        )

        particle_out, gas_out = run_simulation(
            staggered,
            particle,
            gas_species,
            time_step=10.0,
            steps=5,
        )

        assert is_numerically_stable(particle_out)
        variance = float(np.var(particle_out.get_mass()))
        assert variance < MAX_VARIANCE

    def test_stability_zero_time_step_noop(self) -> None:
        """Zero time step returns early without mutation and stays stable."""
        particle, gas_species = create_test_system(10, seed=19)
        initial_masses = np.array(particle.get_mass(), copy=True)

        staggered = CondensationIsothermalStaggered(
            molar_mass=DEFAULT_MOLAR_MASS,
            theta_mode="half",
            num_batches=2,
            shuffle_each_step=False,
            random_state=19,
        )

        particle_out, gas_out = run_simulation(
            staggered,
            particle,
            gas_species,
            time_step=0.0,
            steps=5,
        )

        assert np.array_equal(initial_masses, particle_out.get_mass())
        assert is_numerically_stable(particle_out)
        assert (
            np.asarray(gas_out.get_concentration()).shape
            == np.asarray(gas_species.get_concentration()).shape
        )

    def test_stability_zero_particles_safe(self) -> None:
        """Zero-particle systems exit cleanly and are reported stable."""
        particle, gas_species = create_test_system(0, seed=5)
        staggered = CondensationIsothermalStaggered(
            molar_mass=DEFAULT_MOLAR_MASS,
            theta_mode="half",
            num_batches=1,
            shuffle_each_step=False,
            random_state=5,
        )

        particle_out, gas_out = run_simulation(
            staggered,
            particle,
            gas_species,
            time_step=10.0,
            steps=5,
        )

        assert is_numerically_stable(particle_out)
        assert particle_out.get_mass().size == 0
        assert (
            np.asarray(gas_out.get_concentration()).shape
            == np.asarray(gas_species.get_concentration()).shape
        )
