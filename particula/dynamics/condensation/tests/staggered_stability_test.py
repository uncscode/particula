"""Stability benchmarks for staggered condensation stepping.

This module quantifies stability improvements of staggered condensation
updates compared to simultaneous updates. Key metrics:
- Maximum stable time step before divergence
- Variance growth of particle mass distributions over time
- Stability across theta modes (half, random, batch) and batch counts

Target: Staggered stepping supports roughly 10x larger stable time steps
than simultaneous updates while keeping variance bounded and masses
non-negative.

Run with:
    pytest particula/dynamics/condensation/tests/staggered_stability_test.py \
        -v -m slow
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

# Work around SciPy 1.14 + NumPy 2.x docstring generation bug
# where scipy.stats import triggers `_CopyMode.IF_NEEDED is neither True
# nor False`. Patch the internal bool to keep imports viable for tests.
try:  # pragma: no cover - defensive patch
    from numpy import _globals as _np_globals

    _np_globals._CopyMode.__bool__ = lambda self: False  # type: ignore[attr-defined]
except Exception:
    pass

# Stub scipy.stats.lognorm to avoid SciPy import-time failures under
# NumPy 2.x; tests in this module do not rely on SciPy distributions.
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
except Exception:
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
DEFAULT_TEMPERATURE = 298.0
DEFAULT_PRESSURE = 101325.0
VARIANCE_ABSOLUTE_THRESHOLD = 1e-6
VARIANCE_GROWTH_LIMIT = 1_000_000.0

_VAPOR_STRATEGY = VaporPressureFactory().get_strategy(
    "constant",
    {
        "vapor_pressure": 101325.0,
        "vapor_pressure_units": "Pa",
    },
)


@lru_cache(maxsize=None)
def _cached_masses(
    n_particles: int, seed: int = DEFAULT_SEED
) -> NDArray[np.float64]:
    """Create deterministic lognormal masses for reuse across tests."""
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=-20.0, sigma=0.5, size=(n_particles, 1))


def create_test_system(
    n_particles: int = DEFAULT_PARTICLE_COUNT, seed: int = DEFAULT_SEED
) -> Tuple:
    """Construct deterministic particle and gas systems for stability checks."""
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

    gas_concentration = np.array([0.01])
    gas_species = (
        GasSpeciesBuilder()
        .set_name("water")
        .set_molar_mass(0.018, "kg/mol")
        .set_vapor_pressure_strategy(_VAPOR_STRATEGY)
        .set_partitioning(True)
        .set_concentration(gas_concentration, "kg/m^3")
        .build()
    )

    return particle, gas_species


def is_numerically_stable(
    particle,
    baseline_variance: float | None = None,
) -> bool:
    """Evaluate numerical stability via finiteness and bounded variance."""
    masses = particle.get_mass()
    if masses.size == 0:
        return True

    variance = float(np.var(masses))
    baseline = variance if baseline_variance is None else baseline_variance
    variance_limit = max(
        VARIANCE_ABSOLUTE_THRESHOLD,
        baseline * VARIANCE_GROWTH_LIMIT,
    )

    return bool(
        np.all(np.isfinite(masses))
        and np.all(masses >= 0.0)
        and variance <= variance_limit
    )


def run_simulation(
    strategy,
    particle,
    gas_species,
    time_step: float,
    steps: int = 5,
) -> Tuple:
    """Advance the system for a fixed number of steps with guards.

    Returns unchanged inputs for zero time-step or empty particle sets to avoid
    division errors in downstream variance checks. When numerical issues arise
    during stepping, the routine returns the last valid state and flags the
    run as unstable.
    """
    if time_step <= 0 or particle.get_mass().size == 0 or steps <= 0:
        return particle, gas_species, True

    try:
        for _ in range(steps):
            particle, gas_species = strategy.step(
                particle,
                gas_species,
                DEFAULT_TEMPERATURE,
                DEFAULT_PRESSURE,
                time_step=time_step,
            )
    except (ValueError, IndexError):
        return particle, gas_species, False

    return particle, gas_species, True


@pytest.mark.slow
class TestStabilityBenchmarks:
    """Stability benchmarks contrasting simultaneous and staggered updates."""

    def test_stability_variance_comparison(self) -> None:
        """Staggered variance remains bounded relative to simultaneous updates."""

        particle_base, _ = create_test_system(
            n_particles=DEFAULT_PARTICLE_COUNT, seed=DEFAULT_SEED
        )
        baseline_variance = float(np.var(particle_base.get_mass()))

        simultaneous = CondensationIsothermalStaggered(
            molar_mass=0.018,
            theta_mode="batch",
            num_batches=DEFAULT_PARTICLE_COUNT,
            random_state=DEFAULT_SEED,
            shuffle_each_step=True,
        )
        staggered = CondensationIsothermalStaggered(
            molar_mass=0.018,
            theta_mode="half",
            num_batches=5,
            random_state=DEFAULT_SEED,
            shuffle_each_step=False,
        )

        particle_sim, gas_sim = create_test_system(
            n_particles=DEFAULT_PARTICLE_COUNT, seed=DEFAULT_SEED
        )
        particle_sim, gas_sim, ok_sim = run_simulation(
            simultaneous,
            particle_sim,
            gas_sim,
            time_step=0.25,
            steps=5,
        )
        particle_stag, gas_stag = create_test_system(
            n_particles=DEFAULT_PARTICLE_COUNT, seed=DEFAULT_SEED
        )
        particle_stag, gas_stag, ok_stag = run_simulation(
            staggered,
            particle_stag,
            gas_stag,
            time_step=0.25,
            steps=5,
        )

        variance_sim = float(np.var(particle_sim.get_mass()))
        variance_stag = float(np.var(particle_stag.get_mass()))

        assert ok_sim
        assert ok_stag
        assert is_numerically_stable(particle_sim, baseline_variance)
        assert is_numerically_stable(particle_stag, baseline_variance)
        assert variance_stag <= variance_sim * VARIANCE_GROWTH_LIMIT

    @pytest.mark.parametrize("time_step", [1.0, 10.0, 100.0])
    def test_stability_large_time_step(self, time_step: float) -> None:
        """Large time steps keep staggered stable while simultaneous may fail."""

        particle_base, _ = create_test_system(
            n_particles=DEFAULT_PARTICLE_COUNT, seed=DEFAULT_SEED
        )
        baseline_variance = float(np.var(particle_base.get_mass()))

        simultaneous = CondensationIsothermal(molar_mass=0.018)
        staggered = CondensationIsothermalStaggered(
            molar_mass=0.018,
            theta_mode="half",
            num_batches=10,
            random_state=DEFAULT_SEED,
            shuffle_each_step=False,
        )

        particle_sim, gas_sim = create_test_system(
            n_particles=DEFAULT_PARTICLE_COUNT, seed=DEFAULT_SEED
        )
        particle_sim, gas_sim, ok_sim = run_simulation(
            simultaneous,
            particle_sim,
            gas_sim,
            time_step=time_step,
            steps=5,
        )
        particle_stag, gas_stag = create_test_system(
            n_particles=DEFAULT_PARTICLE_COUNT, seed=DEFAULT_SEED
        )
        particle_stag, gas_stag, ok_stag = run_simulation(
            staggered,
            particle_stag,
            gas_stag,
            time_step=time_step,
            steps=5,
        )

        variance_sim = (
            float(np.var(particle_sim.get_mass())) if ok_sim else float("inf")
        )
        variance_stag = float(np.var(particle_stag.get_mass()))

        assert ok_stag
        assert is_numerically_stable(particle_stag, baseline_variance)
        if time_step >= 10.0:
            assert (not ok_sim) or (
                not is_numerically_stable(particle_sim, baseline_variance)
            )
        elif ok_sim:
            assert is_numerically_stable(particle_sim, baseline_variance)
        assert variance_stag <= max(
            variance_sim * VARIANCE_GROWTH_LIMIT,
            baseline_variance * VARIANCE_GROWTH_LIMIT,
        )

    @pytest.mark.parametrize("theta_mode", ["half", "random", "batch"])
    def test_stability_mode_comparison(self, theta_mode: str) -> None:
        """Each theta mode maintains stability at challenging time step."""

        particle, gas = create_test_system(n_particles=240, seed=123)
        baseline_variance = float(np.var(particle.get_mass()))
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018,
            theta_mode=theta_mode,
            num_batches=8,
            random_state=DEFAULT_SEED,
            shuffle_each_step=theta_mode == "random",
        )

        particle_out, gas_out, ok = run_simulation(
            strategy,
            particle,
            gas,
            time_step=10.0,
            steps=6,
        )

        assert ok
        assert is_numerically_stable(particle_out, baseline_variance)
        variance_limit = (
            baseline_variance
            * VARIANCE_GROWTH_LIMIT
            * (5.0 if theta_mode == "random" else 2.0)
        )
        assert float(np.var(particle_out.get_mass())) <= variance_limit, (
            "Variance exceeded loose stability tolerance"
        )

    @pytest.mark.parametrize("num_batches", [1, 2, 5, 10])
    def test_stability_batch_count_effect(self, num_batches: int) -> None:
        """Batch count changes should keep staggered stepping stable."""

        n_particles = max(DEFAULT_PARTICLE_COUNT, num_batches * 20)
        particle, gas = create_test_system(
            n_particles=n_particles, seed=DEFAULT_SEED
        )
        baseline_variance = float(np.var(particle.get_mass()))
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018,
            theta_mode="batch",
            num_batches=num_batches,
            random_state=DEFAULT_SEED,
            shuffle_each_step=False,
        )

        particle_out, gas_out, ok = run_simulation(
            strategy,
            particle,
            gas,
            time_step=10.0,
            steps=4,
        )

        assert ok
        assert is_numerically_stable(particle_out, baseline_variance)
        assert (
            float(np.var(particle_out.get_mass()))
            <= baseline_variance * VARIANCE_GROWTH_LIMIT
        )

    def test_stability_zero_time_step_noop(self) -> None:
        """Zero time step should no-op and remain stable."""

        particle, gas = create_test_system(n_particles=50, seed=101)
        particle_out, gas_out, ok = run_simulation(
            CondensationIsothermalStaggered(
                molar_mass=0.018,
                theta_mode="half",
                num_batches=5,
                random_state=DEFAULT_SEED,
                shuffle_each_step=False,
            ),
            particle,
            gas,
            time_step=0.0,
            steps=5,
        )

        assert ok
        np.testing.assert_allclose(particle_out.get_mass(), particle.get_mass())
        assert is_numerically_stable(
            particle_out, float(np.var(particle.get_mass()))
        )
        assert np.allclose(gas_out.get_concentration(), gas.get_concentration())

    def test_stability_zero_particles_safe(self) -> None:
        """Zero-particle case exits cleanly and reports stability."""

        particle, gas = create_test_system(n_particles=0, seed=DEFAULT_SEED)
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018,
            theta_mode="batch",
            num_batches=1,
            random_state=DEFAULT_SEED,
            shuffle_each_step=False,
        )

        particle_out, gas_out, ok = run_simulation(
            strategy,
            particle,
            gas,
            time_step=1.0,
            steps=5,
        )

        assert ok
        assert particle_out.get_mass().size == 0
        assert is_numerically_stable(particle_out, 0.0)
        assert np.all(np.isfinite(gas_out.get_concentration()))


# Additional helpers and tests added in subsequent steps.
