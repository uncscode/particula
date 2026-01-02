"""Performance benchmarks for staggered condensation stepping.

This module measures runtime overhead and scaling of the staggered
condensation strategy against the simultaneous baseline. Targets:
- Runtime overhead <2x vs simultaneous
- Approximately O(n) scaling with particle count
- Optional memory overhead guidance (<1.5x)

Run with:
    pytest particula/dynamics/condensation/tests/staggered_performance_test.py \
        -v -m "slow and performance"

Interpretation guidance:
- Overhead near 1x is ideal; >2x signals a regression.
- Scaling should remain roughly linear with particle count.
- Theta-mode differences highlight trade-offs between determinism and
  batching; use the printed timings to choose a mode.
"""

from __future__ import annotations

import copy
import time
from typing import Dict, Tuple

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

pytestmark = [pytest.mark.slow, pytest.mark.performance]

TEMP_K = 298.0
PRESSURE_PA = 101325.0
TIME_STEP = 0.0002
ITERATIONS: Dict[int, int] = {1000: 5, 10000: 5, 100000: 3}
OVERHEAD_TARGET = 2.0
SEED = 42
N_PARTICLES = (1000, 10000, 100000)
DEFAULT_NUM_BATCHES = 10
DEFAULT_MOLAR_MASS = 0.018

_VAPOR_STRATEGY = VaporPressureFactory().get_strategy(
    "constant",
    {"vapor_pressure": PRESSURE_PA, "vapor_pressure_units": "Pa"},
)


def _validate_num_batches(num_batches: int) -> None:
    """Validate batch count for staggered strategy."""
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")


def create_test_system(n_particles: int, seed: int = SEED):
    """Create deterministic particle/gas system for benchmarks.

    Args:
        n_particles: Number of particles to create (must be >0).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (particle, gas_species) with shapes matching (n, 1).

    Raises:
        ValueError: If n_particles is non-positive.
    """
    if n_particles <= 0:
        raise ValueError("n_particles must be positive")

    rng = np.random.default_rng(seed)
    masses: NDArray[np.float64] = rng.lognormal(
        mean=-20.0, sigma=0.5, size=(n_particles, 1)
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


def _time_strategy(
    strategy,
    particle,
    gas_species,
    iterations: int,
) -> float:
    """Time average step calls for a strategy using deep copies per run."""
    total = 0.0
    for _ in range(iterations):
        particle_copy = copy.deepcopy(particle)
        gas_copy = copy.deepcopy(gas_species)
        start = time.perf_counter()
        strategy.step(
            particle_copy,
            gas_copy,
            TEMP_K,
            PRESSURE_PA,
            time_step=TIME_STEP,
        )
        total += time.perf_counter() - start
    return total / iterations


def _benchmark_overhead(
    n_particles: int,
    iterations: int,
    theta_mode: str = "random",
    num_batches: int = DEFAULT_NUM_BATCHES,
    shuffle_each_step: bool = True,
) -> Tuple[float, float, float]:
    """Benchmark overhead for staggered vs simultaneous strategies."""
    _validate_num_batches(num_batches)
    particle, gas_species = create_test_system(n_particles, seed=SEED)

    baseline = CondensationIsothermal(molar_mass=DEFAULT_MOLAR_MASS)

    staggered = CondensationIsothermalStaggered(
        molar_mass=DEFAULT_MOLAR_MASS,
        theta_mode=theta_mode,
        num_batches=num_batches,
        random_state=SEED,
        shuffle_each_step=shuffle_each_step,
    )

    baseline_time = _time_strategy(baseline, particle, gas_species, iterations)
    staggered_time = _time_strategy(
        staggered, particle, gas_species, iterations
    )
    overhead = staggered_time / baseline_time

    print(
        f"n={n_particles}: simultaneous={baseline_time:.4f}s, "
        f"staggered={staggered_time:.4f}s, overhead={overhead:.2f}x"
    )
    return baseline_time, staggered_time, overhead


def test_performance_scaling() -> None:
    """Scaling benchmark; overhead should stay below 2x across sizes."""
    results = []
    for n_particles in N_PARTICLES:
        iterations = ITERATIONS[n_particles]
        _, _, overhead = _benchmark_overhead(
            n_particles,
            iterations,
            theta_mode="random",
            shuffle_each_step=True,
        )
        results.append((n_particles, overhead))

    worst_overhead = max(overhead for _, overhead in results)
    assert worst_overhead < OVERHEAD_TARGET, (
        f"Overhead exceeded target; worst={worst_overhead:.2f}x "
        f"(target <{OVERHEAD_TARGET}x)"
    )


def test_performance_1k_particles() -> None:
    """Overhead check at 1k particles remains below the 2x target."""
    _, _, overhead = _benchmark_overhead(
        1000, ITERATIONS[1000], theta_mode="random", shuffle_each_step=True
    )
    assert overhead < OVERHEAD_TARGET, (
        f"1k overhead {overhead:.2f}x exceeds {OVERHEAD_TARGET}x target"
    )


def test_performance_10k_particles() -> None:
    """Overhead check at 10k particles remains below the 2x target."""
    _, _, overhead = _benchmark_overhead(
        10000,
        ITERATIONS[10000],
        theta_mode="random",
        shuffle_each_step=True,
    )
    assert overhead < OVERHEAD_TARGET, (
        f"10k overhead {overhead:.2f}x exceeds {OVERHEAD_TARGET}x target"
    )


def test_performance_100k_particles() -> None:
    """Overhead check at 100k particles using capped iterations."""
    _, _, overhead = _benchmark_overhead(
        100000,
        ITERATIONS[100000],
        theta_mode="random",
        shuffle_each_step=True,
    )
    assert overhead < OVERHEAD_TARGET, (
        f"100k overhead {overhead:.2f}x exceeds {OVERHEAD_TARGET}x target"
    )


def test_performance_mode_comparison() -> None:
    """Compare theta modes; fastest vs slowest should stay within ~1.5x."""
    n_particles = 10000
    iterations = ITERATIONS[n_particles]
    times: Dict[str, float] = {}

    for theta_mode in ("half", "random", "batch"):
        _, staggered_time, _ = _benchmark_overhead(
            n_particles,
            iterations,
            theta_mode=theta_mode,
            shuffle_each_step=True,
        )
        times[theta_mode] = staggered_time

    fastest = min(times.values())
    slowest = max(times.values())
    print(
        "theta mode timings: "
        + ", ".join(f"{mode}={elapsed:.4f}s" for mode, elapsed in times.items())
    )
    assert np.isfinite(fastest)
    assert slowest <= 1.5 * fastest, (
        "Theta mode overhead drifted; slowest exceeds 1.5x fastest"
    )


def test_performance_vs_simultaneous() -> None:
    """Smoke benchmark: staggered overhead vs baseline at moderate size."""
    _, _, overhead = _benchmark_overhead(
        10000,
        ITERATIONS[10000],
        theta_mode="random",
        shuffle_each_step=True,
    )
    assert overhead < OVERHEAD_TARGET, (
        f"Overhead {overhead:.2f}x exceeds {OVERHEAD_TARGET}x target"
    )


def test_create_test_system_is_deterministic() -> None:
    """Factory produces identical systems for the same seed and size."""
    particle_a, gas_a = create_test_system(64, seed=99)
    particle_b, gas_b = create_test_system(64, seed=99)

    assert np.allclose(particle_a.get_mass(), particle_b.get_mass())
    assert np.allclose(
        gas_a.get_concentration(),
        gas_b.get_concentration(),
    )


def test_invalid_num_batches_or_n_particles_raises_value_error() -> None:
    """Guards reject invalid particle counts or batch settings."""
    with pytest.raises(ValueError):
        create_test_system(0)

    with pytest.raises(ValueError):
        _validate_num_batches(0)

    with pytest.raises(ValueError):
        _validate_num_batches(-3)
