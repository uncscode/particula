"""Performance benchmarks for staggered condensation stepping.

These slow+performance tests measure CondensationIsothermalStaggered scaling
characteristics, validate O(n) linear scaling with particle count, and compare
theta-mode trade-offs. The staggered algorithm uses Gauss-Seidel style per-
particle updates which are inherently sequential and cannot be vectorized like
the simultaneous baseline. Therefore, staggered vs simultaneous overhead ratios
are expected to be high (O(n) per-particle loops vs O(1) vectorized ops).

Run with:
    pytest particula/dynamics/condensation/tests/staggered_performance_test.py
        -v -m "slow and performance"

Note:
    These tests focus on:
    1. O(n) scaling verification (time scales linearly with particles)
    2. Theta-mode comparison (different modes should perform similarly)
    3. Deterministic behavior validation (same seed = same results)
    4. Timing baseline reporting (informational, not enforced)

    They do NOT enforce overhead targets against simultaneous baseline because
    the algorithms are fundamentally different (sequential vs vectorized).
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
PRESSURE_PA = 101325.0
# Maximum allowed scaling factor between consecutive particle counts.
# Linear O(n) scaling means 10x particles should take ~10x time.
# We allow up to 15x to account for caching, allocation, and noise.
SCALING_TOLERANCE = 15.0
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


def _time_staggered_only(n_particles: int) -> float:
    """Time only the staggered strategy for O(n) scaling tests."""
    iterations = ITERATIONS[n_particles]
    base_particle, base_gas = create_test_system(n_particles, seed=SEED)
    staggered = _new_staggered_strategy(
        theta_mode="random",
        shuffle_each_step=True,
    )
    return _time_strategy(staggered, base_particle, base_gas, iterations)


def _run_timing_report(n_particles: int) -> tuple[float, float, float]:
    """Run both strategies and report timings (informational only).

    This function measures both simultaneous and staggered timing but does NOT
    enforce any overhead target. The overhead ratio is expected to be high
    because staggered uses per-particle Python loops while simultaneous uses
    vectorized NumPy operations.

    Returns:
        Tuple of (baseline_time, staggered_time, overhead_ratio)
    """
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
        "Timing report {count} particles: simultaneous={sim:.3f}s, "
        "staggered={stag:.3f}s, overhead={over:.1f}x".format(
            count=n_particles,
            sim=baseline_time,
            stag=staggered_time,
            over=overhead,
        )
    )

    # No assertion on overhead - this is informational only
    # The staggered algorithm is inherently slower due to per-particle loops

    return baseline_time, staggered_time, overhead


def test_performance_1k_particles() -> None:
    """Report timing for 1k particles (informational benchmark)."""
    _run_timing_report(1000)


def test_performance_10k_particles() -> None:
    """Report timing for 10k particles (informational benchmark)."""
    _run_timing_report(10000)


def test_performance_100k_particles() -> None:
    """Report timing for 100k particles (informational benchmark)."""
    _run_timing_report(100000)


def test_performance_scaling_is_linear() -> None:
    """Verify staggered algorithm scales O(n) with particle count.

    The staggered algorithm should scale linearly with particle count.
    We measure timing at 1k, 10k, and 100k particles and verify that
    the scaling factor is approximately 10x for each 10x increase.
    """
    timings = {}
    for n in N_PARTICLES:
        timings[n] = _time_staggered_only(n)
        print(f"Staggered timing {n} particles: {timings[n]:.3f}s")

    # Check scaling from 1k to 10k (should be ~10x, allow up to 15x)
    scaling_1k_to_10k = timings[10000] / timings[1000]
    print(f"Scaling 1k->10k: {scaling_1k_to_10k:.1f}x (expected ~10x)")

    # Check scaling from 10k to 100k (should be ~10x, allow up to 15x)
    scaling_10k_to_100k = timings[100000] / timings[10000]
    print(f"Scaling 10k->100k: {scaling_10k_to_100k:.1f}x (expected ~10x)")

    # Allow generous tolerance for O(n) scaling
    # Superlinear scaling (>15x per 10x particles) indicates regression
    assert scaling_1k_to_10k <= SCALING_TOLERANCE, (
        f"Scaling 1k->10k is {scaling_1k_to_10k:.1f}x, "
        f"exceeds {SCALING_TOLERANCE}x (expected ~10x for O(n) scaling)"
    )
    assert scaling_10k_to_100k <= SCALING_TOLERANCE, (
        f"Scaling 10k->100k is {scaling_10k_to_100k:.1f}x, "
        f"exceeds {SCALING_TOLERANCE}x (expected ~10x for O(n) scaling)"
    )


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
            f"Theta mode {mode}: {duration:.3f}s over "
            f"{ITERATIONS[n_particles]} iterations"
        )

    fastest = min(mode_durations.values())
    slowest = max(mode_durations.values())
    # Guard against zero-duration measurements
    assert fastest >= 0.0
    if fastest == 0.0:
        pytest.skip(
            "Theta-mode timings are all zero; "
            "timing resolution too low to compare modes."
        )
    ratio = slowest / fastest
    print(f"Theta-mode ratio (slowest/fastest): {ratio:.2f}x")
    # Allow 2.5x tolerance to account for:
    # - batch mode potentially faster due to reduced gas update overhead
    # - system load variations during benchmarks
    assert ratio <= 2.5, (
        f"Theta-mode timings drifted beyond the 2.5x band ({ratio:.2f}x)"
    )


def test_performance_vs_simultaneous() -> None:
    """Report overhead comparison between staggered and simultaneous.

    This test reports the overhead ratio but does NOT enforce a target.
    The staggered algorithm uses Gauss-Seidel style per-particle updates
    which are inherently slower than the vectorized simultaneous approach.
    High overhead ratios (100x-1000x+) are expected and acceptable.
    """
    baseline_time, staggered_time, overhead = _run_timing_report(10000)

    # Report only - no assertion on overhead target
    # The algorithms are fundamentally different
    print(
        f"Note: Overhead {overhead:.1f}x is expected - staggered uses "
        "sequential per-particle updates while simultaneous is vectorized."
    )

    # Basic sanity: both should complete in reasonable time
    assert baseline_time < 60.0, "Baseline took too long (>60s)"
    assert staggered_time < 300.0, "Staggered took too long (>300s)"


def test_create_test_system_is_deterministic() -> None:
    """Factory recreates identical masses and concentrations for same seed."""
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
