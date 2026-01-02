"""Performance benchmarks for staggered condensation stepping.

These slow+performance tests measure CondensationIsothermalStaggered overhead
against the simultaneous baseline, quantify O(n) scaling with particle count,
and describe theta-mode trade-offs. Overhead near 1x is ideal; values above
2x signal a regression. Run with:
    pytest particula/dynamics/condensation/tests/staggered_performance_test.py -v -m "slow and performance"
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pytest
from numpy.typing import NDArray

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

pytestmark = [
    pytest.mark.slow,
    pytest.mark.performance,
]

DEFAULT_MOLAR_MASS = 0.018
DEFAULT_NUM_BATCHES = 10
ITERATIONS = {1000: 5, 10000: 5, 100000: 3}
N_PARTICLES = (1000, 10000, 100000)
OVERHEAD_TARGET = 2.0
PRESSURE_PA = 101325.0
SEED = 42
TEMP_K = 298.0
TIME_STEP = 0.0002

_VAPOR_STRATEGY = VaporPressureFactory().get_strategy(
    "constant",
    {
        "vapor_pressure": PRESSURE_PA,
        "vapor_pressure_units": "Pa",
    },
)


def create_test_system(n_particles: int, seed: int = SEED):
    """Create deterministic particle/gas pairs for performance benchmarks."""
    if n_particles <= 0:
        raise ValueError("n_particles must be positive")

    rng = np.random.default_rng(seed)
    masses: NDArray[np.float64] = rng.lognormal(
        mean=-20.0,
        sigma=0.5,
        size=(n_particles, 1),
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


def _validate_num_batches(num_batches: int) -> int:
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    return num_batches


def _time_strategy(strategy, particle, gas_species, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        current_particle = copy.deepcopy(particle)
        current_gas = copy.deepcopy(gas_species)
        current_particle, current_gas = strategy.step(
            current_particle,
            current_gas,
            TEMP_K,
            PRESSURE_PA,
            time_step=TIME_STEP,
        )
    return time.perf_counter() - start


def _new_staggered_strategy(theta_mode: str, shuffle_each_step: bool):
    return CondensationIsothermalStaggered(
        molar_mass=DEFAULT_MOLAR_MASS,
        theta_mode=theta_mode,
        num_batches=_validate_num_batches(DEFAULT_NUM_BATCHES),
        random_state=SEED,
        shuffle_each_step=shuffle_each_step,
    )


def _run_scaling_case(n_particles: int) -> tuple[float, float, float]:
    iterations = ITERATIONS[n_particles]
    base_particle, base_gas = create_test_system(n_particles, seed=SEED)

    baseline = CondensationIsothermal(molar_mass=DEFAULT_MOLAR_MASS)
    staggered = _new_staggered_strategy(
        theta_mode="random",
        shuffle_each_step=True,
    )

    baseline_time = _time_strategy(
        baseline, base_particle, base_gas, iterations
    )
    staggered_time = _time_strategy(
        staggered,
        base_particle,
        base_gas,
        iterations,
    )

    if baseline_time == 0.0:
        pytest.skip(
            "Baseline timing is zero; cannot compute overhead reliably "
            "for CondensationIsothermal vs CondensationIsothermalStaggered."
        )

    overhead = staggered_time / baseline_time
    print(
        "Scaling {count} particles: simultaneous={sim:.3f}s, staggered={stag:.3f}s, "
        "overhead={over:.3f}".format(
            count=n_particles,
            sim=baseline_time,
            stag=staggered_time,
            over=overhead,
        )
    )

    assert overhead < OVERHEAD_TARGET, (
        f"Overhead {overhead:.2f}x for {n_particles} particles exceeds target {OVERHEAD_TARGET}x"
    )

    return baseline_time, staggered_time, overhead


def test_performance_1k_particles() -> None:
    """Named 1k-case for acceptance tracking."""
    _run_scaling_case(1000)


def test_performance_10k_particles() -> None:
    """Named 10k-case for acceptance tracking."""
    _run_scaling_case(10000)


def test_performance_100k_particles() -> None:
    """Named 100k-case for acceptance tracking (capped iterations)."""
    _run_scaling_case(100000)


def _run_theta_mode(n_particles: int, theta_mode: str) -> float:
    iterations = ITERATIONS[n_particles]
    strategy = _new_staggered_strategy(
        theta_mode=theta_mode,
        shuffle_each_step=(theta_mode == "random"),
    )
    particle, gas_species = create_test_system(n_particles, seed=SEED)
    return _time_strategy(
        strategy,
        particle,
        gas_species,
        iterations,
    )


def test_performance_mode_comparison() -> None:
    """Compare half/random/batch timings and keep them within a narrow band."""
    n_particles = 10000
    mode_durations: dict[str, float] = {}
    for mode in ("half", "random", "batch"):
        duration = _run_theta_mode(n_particles, mode)
        mode_durations[mode] = duration
        print(
            f"Theta mode {mode}: {duration:.3f}s over {ITERATIONS[n_particles]} iterations"
        )

    fastest = min(mode_durations.values())
    slowest = max(mode_durations.values())
    # Guard against zero-duration measurements caused by insufficient timing resolution.
    assert fastest >= 0.0
    if fastest == 0.0:
        pytest.skip(
            "Theta-mode timings are all zero; timing resolution too low to compare modes."
        )
    assert slowest / fastest <= 1.5, (
        "Theta-mode timings drifted beyond the 1.5x band"
    )


def test_performance_vs_simultaneous() -> None:
    """Smoke benchmark: 10k simultaneous vs staggered overhead remains under target."""
    _, _, overhead = _run_scaling_case(10000)
    assert overhead < OVERHEAD_TARGET


def test_create_test_system_is_deterministic() -> None:
    """Factory recreates identical masses and concentrations for the same seed."""
    particle_a, gas_a = create_test_system(1000, seed=SEED)
    particle_b, gas_b = create_test_system(1000, seed=SEED)

    assert np.array_equal(particle_a.get_mass(), particle_b.get_mass())
    assert np.array_equal(
        np.asarray(gas_a.get_concentration()),
        np.asarray(gas_b.get_concentration()),
    )


def test_invalid_num_batches_or_n_particles_raises_value_error() -> None:
    """Guardrails enforce positive particle counts and batch sizes."""
    with pytest.raises(ValueError):
        create_test_system(0)
    with pytest.raises(ValueError):
        _validate_num_batches(0)
    with pytest.raises(ValueError):
        _validate_num_batches(-4)
