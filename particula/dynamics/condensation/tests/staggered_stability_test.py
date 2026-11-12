"""Stability benchmarks for staggered condensation stepping.

These native fixtures check finite nonnegative masses and bounded variance
at specified timesteps, theta modes and batch counts. They do not establish
accuracy, a maximum stable timestep, or a timestep advantage over simultaneous
stepping. Both processes retain their own inventory-limited update contracts.

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
from particula.dynamics.condensation import (
    CondensationIsothermal,
    CondensationIsothermalStaggered,
)
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.gas import GasDataBuilder, VaporPressureFactory
from particula.particles import (
    ActivityIdealMass,
    SurfaceStrategyVolume,
    create_resolved_particle_data,
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


@pytest.fixture(autouse=True)
def forbid_facades(forbid_legacy_facades):
    """Check stability on supported native execution only."""


def _physics():
    """Keep the original material and gas laws on the process."""
    return dict(
        activity_strategy=ActivityIdealMass(),
        surface_strategy=SurfaceStrategyVolume(
            surface_tension=0.072, density=1000.0
        ),
        vapor_pressure_strategy=_VAPOR_STRATEGY,
        species_map=CondensationSpeciesMap(("water",), ((0, 0),)),
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
    particle = create_resolved_particle_data(
        masses=masses, density=1000.0, volume=1.0
    )

    gas_species = (
        GasDataBuilder()
        .set_names(["water"])
        .set_molar_mass([DEFAULT_MOLAR_MASS])
        .set_partitioning([True])
        .set_concentration([0.01])
        .build()
    )
    return particle, gas_species


def is_numerically_stable(particle) -> bool:
    """Check that masses stay finite, non-negative, and variance bounded."""
    masses = np.asarray(particle.total_mass[0])
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
    if time_step <= 0.0 or particle.total_mass[0].size == 0 or steps <= 0:
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
        """Retain bounded-variance checks at the specified large timesteps."""
        seed = 11
        particle_sim, gas_sim = create_test_system(220, seed)
        particle_stag, gas_stag = create_test_system(220, seed)

        simultaneous = CondensationIsothermal(
            **_physics(), molar_mass=DEFAULT_MOLAR_MASS
        )
        staggered = CondensationIsothermalStaggered(
            **_physics(),
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

        mass_stag = np.asarray(particle_stag.total_mass[0])
        var_stag = float(np.var(mass_stag))
        assert is_numerically_stable(particle_stag)
        assert var_stag <= VARIANCE_LIMIT
        if time_step < 10.0:
            mass_sim = np.asarray(particle_sim.total_mass[0])
            var_sim = float(np.var(mass_sim))
            assert is_numerically_stable(particle_sim)
            assert var_sim <= VARIANCE_LIMIT

    def test_stability_variance_comparison(self) -> None:
        """Both native strategies retain bounded variance at moderate dt."""
        seed = 7
        particle_sim, gas_sim = create_test_system(200, seed)
        particle_stag, gas_stag = create_test_system(200, seed)

        simultaneous = CondensationIsothermal(
            **_physics(), molar_mass=DEFAULT_MOLAR_MASS
        )
        staggered = CondensationIsothermalStaggered(
            **_physics(),
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

        mass_sim = np.asarray(particle_sim.total_mass[0])
        mass_stag = np.asarray(particle_stag.total_mass[0])
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
            **_physics(),
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
        variance = float(np.var(particle_out.total_mass[0]))
        assert variance < MAX_VARIANCE

    @pytest.mark.parametrize("num_batches", [1, 2, 5, 10])
    def test_stability_batch_count_effect(self, num_batches: int) -> None:
        """Batch counts stay stable; variance remains bounded and comparable."""
        seed = 33
        n_particles = max(DEFAULT_PARTICLE_COUNT, num_batches * 20)
        particle, gas_species = create_test_system(n_particles, seed)
        staggered = CondensationIsothermalStaggered(
            **_physics(),
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
        variance = float(np.var(particle_out.total_mass[0]))
        assert variance < MAX_VARIANCE

    def test_stability_zero_time_step_noop(self) -> None:
        """Zero time step returns early without mutation and stays stable."""
        particle, gas_species = create_test_system(10, seed=19)
        initial_masses = np.array(particle.total_mass[0], copy=True)

        staggered = CondensationIsothermalStaggered(
            **_physics(),
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

        assert np.array_equal(initial_masses, particle_out.total_mass[0])
        assert is_numerically_stable(particle_out)
        assert (
            np.asarray(gas_out.concentration).shape
            == np.asarray(gas_species.concentration).shape
        )

    def test_stability_zero_particles_safe(self) -> None:
        """Zero-particle systems exit cleanly and are reported stable."""
        particle, gas_species = create_test_system(0, seed=5)
        staggered = CondensationIsothermalStaggered(
            **_physics(),
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
        assert particle_out.total_mass[0].size == 0
        assert (
            np.asarray(gas_out.concentration).shape
            == np.asarray(gas_species.concentration).shape
        )
