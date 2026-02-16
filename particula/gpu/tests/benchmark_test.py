"""GPU benchmark suite for condensation and coagulation kernels.

Run with:
    pytest particula/gpu/tests/benchmark_test.py -v -m "slow and performance" -s

Set ``WARP_PROFILE=1`` to enable Warp capture hooks for Nsight/warp
profiling. When enabled, run Nsight Systems/Compute while the benchmark
executes to inspect memory access patterns and kernel launch metrics.
"""

# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false

from __future__ import annotations

from contextlib import contextmanager
import os
import time
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import pytest

wp = pytest.importorskip("warp")

from particula.dynamics.coagulation.brownian_kernel import (  # noqa: E402
    get_brownian_kernel_via_system_state,
)
from particula.dynamics.coagulation.particle_resolved_step.particle_resolved_method import (  # noqa: E402
    get_particle_resolved_coagulation_step,
    get_particle_resolved_update_step,
)
from particula.dynamics.condensation.mass_transfer import (  # noqa: E402
    get_first_order_mass_transport_k,
    get_mass_transfer_rate,
)
from particula.gas.gas_data import GasData  # noqa: E402
from particula.gas.properties.dynamic_viscosity import (  # noqa: E402
    get_dynamic_viscosity,
)
from particula.gas.properties.mean_free_path import (  # noqa: E402
    get_molecule_mean_free_path,
)
from particula.gas.properties.pressure_function import (  # noqa: E402
    get_partial_pressure,
)
from particula.gpu.conversion import (  # noqa: E402
    to_warp_gas_data,
    to_warp_particle_data,
)
from particula.gpu.dynamics.coagulation_funcs import (  # noqa: E402
    brownian_diffusivity_wp,
    brownian_kernel_pair_wp,
)
from particula.gpu.dynamics.condensation_funcs import (  # noqa: E402
    diffusion_coefficient_wp,
    mass_transfer_rate_wp,
    particle_radius_from_volume_wp,
)
from particula.gpu.kernels.coagulation import (  # noqa: E402
    coagulation_step_gpu,
)
from particula.gpu.kernels.condensation import (  # noqa: E402
    condensation_step_gpu,
)
from particula.particles.particle_data import ParticleData  # noqa: E402
from particula.particles.properties.aerodynamic_mobility_module import (  # noqa: E402
    get_aerodynamic_mobility,
)
from particula.particles.properties.diffusion_coefficient import (  # noqa: E402
    get_diffusion_coefficient,
)
from particula.particles.properties.kelvin_effect_module import (  # noqa: E402
    get_kelvin_radius,
    get_kelvin_term,
)
from particula.particles.properties.knudsen_number_module import (  # noqa: E402
    get_knudsen_number,
)
from particula.particles.properties.partial_pressure_module import (  # noqa: E402
    get_partial_pressure_delta,
)
from particula.particles.properties.slip_correction_module import (  # noqa: E402
    get_cunningham_slip_correction,
)
from particula.particles.properties.vapor_correction_module import (  # noqa: E402
    get_vapor_transition_correction,
)
from particula.util import constants  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.performance]

DEFAULT_TEMPERATURE = 298.15
DEFAULT_PRESSURE = 101325.0
DEFAULT_TIME_STEP = 0.5
DEFAULT_STEPS = 10
DEFAULT_WARMUP = 1
DEFAULT_SURFACE_TENSION = 0.072
DEFAULT_MASS_ACCOMMODATION = 1.0
DEFAULT_DIFFUSION_COEFFICIENT = 2.0e-5
MAX_COLLISIONS = 256


def _skip_if_no_cuda() -> None:
    if not wp.is_cuda_available():
        pytest.skip("CUDA not available")


@contextmanager
def _warp_profiled(tag: str):
    """Optionally enable Warp capture/profiling when WARP_PROFILE=1."""
    if os.getenv("WARP_PROFILE", "0") != "1":
        yield
        return

    if hasattr(wp, "capture_begin") and hasattr(wp, "capture_end"):
        wp.capture_begin(tag)
        try:
            yield
        finally:
            wp.capture_end()
        return

    profiler = getattr(wp, "profiler", None)
    if profiler is not None and hasattr(profiler, "begin"):
        profiler.begin()
        try:
            yield
        finally:
            if hasattr(profiler, "end"):
                profiler.end()
        return

    yield


def _time_gpu_loop(step_fn, steps: int, warmup: int) -> float:
    """Time a GPU loop with a single synchronize before/after."""
    for _ in range(warmup):
        step_fn()
    wp.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        step_fn()
    wp.synchronize()
    return time.perf_counter() - start


def _time_cpu_loop(step_fn, steps: int, warmup: int) -> float:
    """Time a CPU loop with warmup iterations."""
    for _ in range(warmup):
        step_fn()
    start = time.perf_counter()
    for _ in range(steps):
        step_fn()
    return time.perf_counter() - start


def _compute_speedup(cpu_time: float, gpu_time: float) -> float:
    if cpu_time <= 0.0 or gpu_time <= 0.0:
        pytest.skip("Invalid timing data")
    return cpu_time / gpu_time


def _make_particle_data(
    n_boxes: int,
    n_particles: int,
    n_species: int,
    concentration_scale: float = 1.0,
) -> ParticleData:
    """Create deterministic particle data for benchmarks."""
    base_masses = np.linspace(1.0e-18, 3.0e-18, n_species, dtype=np.float64)
    masses = np.empty((n_boxes, n_particles, n_species), dtype=np.float64)
    for box_idx in range(n_boxes):
        for particle_idx in range(n_particles):
            scale = 1.0 + 0.1 * particle_idx + 0.05 * box_idx
            masses[box_idx, particle_idx, :] = base_masses * scale
    concentration = np.full(
        (n_boxes, n_particles), concentration_scale, dtype=np.float64
    )
    charge = np.zeros((n_boxes, n_particles), dtype=np.float64)
    density = np.linspace(1000.0, 1400.0, n_species, dtype=np.float64)
    volume = np.full((n_boxes,), 1.0e-6, dtype=np.float64)
    return ParticleData(
        masses=masses,
        concentration=concentration,
        charge=charge,
        density=density,
        volume=volume,
    )


def _make_gas_data(n_boxes: int, n_species: int) -> GasData:
    """Create deterministic gas data for benchmarks."""
    molar_mass = np.linspace(0.018, 0.05, n_species, dtype=np.float64)
    concentration = np.empty((n_boxes, n_species), dtype=np.float64)
    for box_idx in range(n_boxes):
        concentration[box_idx, :] = 1.0e-6 * (1.0 + 0.2 * box_idx)
    partitioning = np.ones((n_species,), dtype=bool)
    names = [f"species_{idx}" for idx in range(n_species)]
    return GasData(
        name=names,
        molar_mass=molar_mass,
        concentration=concentration,
        partitioning=partitioning,
    )


def _make_vapor_pressure(n_boxes: int, n_species: int) -> np.ndarray:
    """Create deterministic vapor pressure array."""
    vapor_pressure = np.empty((n_boxes, n_species), dtype=np.float64)
    for box_idx in range(n_boxes):
        vapor_pressure[box_idx, :] = 800.0 + 50.0 * box_idx
    return vapor_pressure


def _cpu_mass_transfer(
    particles: ParticleData,
    gas: GasData,
    vapor_pressure: np.ndarray,
    surface_tension: np.ndarray,
    mass_accommodation: np.ndarray,
    diffusion_coefficient_vapor: np.ndarray,
    temperature: float,
    pressure: float,
    time_step: float,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute CPU mass transfer matching GPU kernel physics."""
    n_boxes, n_particles, n_species = particles.masses.shape
    if out is None:
        mass_transfer = np.zeros_like(particles.masses)
    else:
        mass_transfer = out
        mass_transfer.fill(0.0)
    dynamic_viscosity = get_dynamic_viscosity(
        temperature,
        reference_viscosity=constants.REF_VISCOSITY_AIR_STP,
        reference_temperature=constants.REF_TEMPERATURE_STP,
    )
    mean_free_path = get_molecule_mean_free_path(
        molar_mass=constants.MOLECULAR_WEIGHT_AIR,
        temperature=temperature,
        pressure=pressure,
        dynamic_viscosity=dynamic_viscosity,
    )

    for box_idx in range(n_boxes):
        for particle_idx in range(n_particles):
            if particles.concentration[box_idx, particle_idx] == 0.0:
                continue
            total_volume = np.sum(
                particles.masses[box_idx, particle_idx, :] / particles.density
            )
            if total_volume <= 0.0:
                continue
            total_mass = np.sum(particles.masses[box_idx, particle_idx, :])
            radius = np.cbrt(3.0 * total_volume / (4.0 * np.pi))
            effective_density = (
                total_mass / total_volume if total_volume > 0.0 else 0.0
            )
            if effective_density <= 0.0:
                effective_density = particles.density[0]

            knudsen_number = get_knudsen_number(mean_free_path, radius)
            slip_correction = get_cunningham_slip_correction(knudsen_number)
            mobility = get_aerodynamic_mobility(
                particle_radius=radius,
                slip_correction_factor=slip_correction,
                dynamic_viscosity=dynamic_viscosity,
            )
            diffusion_particle = get_diffusion_coefficient(
                temperature=temperature,
                aerodynamic_mobility=mobility,
                boltzmann_constant=constants.BOLTZMANN_CONSTANT,
            )

            for species_idx in range(n_species):
                transition = get_vapor_transition_correction(
                    knudsen_number=knudsen_number,
                    mass_accommodation=mass_accommodation[species_idx],
                )
                diffusion_value = diffusion_coefficient_vapor[species_idx]
                if diffusion_value <= 0.0:
                    diffusion_value = diffusion_particle
                mass_transport = get_first_order_mass_transport_k(
                    particle_radius=radius,
                    vapor_transition=transition,
                    diffusion_coefficient=diffusion_value,
                )
                kelvin_radius = get_kelvin_radius(
                    effective_surface_tension=surface_tension[species_idx],
                    effective_density=effective_density,
                    molar_mass=gas.molar_mass[species_idx],
                    temperature=temperature,
                )
                kelvin_term = get_kelvin_term(radius, kelvin_radius)
                partial_pressure_gas = get_partial_pressure(
                    concentration=gas.concentration[box_idx, species_idx],
                    molar_mass=gas.molar_mass[species_idx],
                    temperature=temperature,
                )
                pressure_delta = get_partial_pressure_delta(
                    partial_pressure_gas=partial_pressure_gas,
                    partial_pressure_particle=vapor_pressure[
                        box_idx, species_idx
                    ],
                    kelvin_term=kelvin_term,
                )
                mass_rate = get_mass_transfer_rate(
                    pressure_delta=pressure_delta,
                    first_order_mass_transport=mass_transport,
                    temperature=temperature,
                    molar_mass=gas.molar_mass[species_idx],
                )
                mass_transfer[box_idx, particle_idx, species_idx] = (
                    mass_rate * time_step
                )
    return mass_transfer


def _cpu_condensation_step(
    particles: ParticleData,
    gas: GasData,
    vapor_pressure: np.ndarray,
    surface_tension: np.ndarray,
    mass_accommodation: np.ndarray,
    diffusion_coefficient_vapor: np.ndarray,
    temperature: float,
    pressure: float,
    time_step: float,
    mass_transfer_buffer: np.ndarray,
) -> None:
    """Update particle masses via CPU mass transfer."""
    mass_transfer = _cpu_mass_transfer(
        particles,
        gas,
        vapor_pressure,
        surface_tension,
        mass_accommodation,
        diffusion_coefficient_vapor,
        temperature,
        pressure,
        time_step,
        out=mass_transfer_buffer,
    )
    particles.masses = np.maximum(0.0, particles.masses + mass_transfer)


def _build_kernel_radius(radii: np.ndarray) -> np.ndarray:
    """Build interpolation radii for particle-resolved coagulation."""
    valid = radii[radii > 0.0]
    if valid.size == 0:
        return np.linspace(1.0e-9, 1.0e-6, 32)
    min_radius = max(valid.min() * 0.8, 1.0e-9)
    max_radius = max(valid.max() * 1.2, min_radius * 10.0)
    return np.linspace(min_radius, max_radius, 64)


def _cpu_coagulation_step(
    particles: ParticleData,
    temperature: float,
    pressure: float,
    time_step: float,
    rng: np.random.Generator,
    kernel_radius: np.ndarray,
) -> None:
    """Update particle masses via particle-resolved CPU coagulation."""
    n_boxes, n_particles, _ = particles.masses.shape
    for box_idx in range(n_boxes):
        masses_box = particles.masses[box_idx]
        concentration_box = particles.concentration[box_idx]
        total_mass = np.sum(masses_box, axis=-1)
        total_volume = np.sum(masses_box / particles.density, axis=-1)
        radii = np.cbrt(3.0 * total_volume / (4.0 * np.pi))
        kernel = cast(
            NDArray[np.float64],
            np.atleast_2d(
                np.asarray(
                    get_brownian_kernel_via_system_state(
                        particle_radius=radii,
                        particle_mass=total_mass,
                        temperature=temperature,
                        pressure=pressure,
                    ),
                    dtype=np.float64,
                )
            ),
        )
        collision_pairs = get_particle_resolved_coagulation_step(
            radii,
            kernel,
            kernel_radius,
            float(particles.volume[box_idx]),
            time_step,
            rng,
        )  # type: ignore[arg-type]
        if collision_pairs.size == 0:
            continue
        small_index = collision_pairs[:, 0]
        large_index = collision_pairs[:, 1]
        radii, _, _ = get_particle_resolved_update_step(
            radii,
            np.zeros(n_particles, dtype=np.float64),
            np.zeros(n_particles, dtype=np.float64),
            small_index,
            large_index,
        )
        concentration_box[small_index] = 0.0
        mass_fractions = np.divide(
            masses_box,
            total_mass[:, np.newaxis],
            where=total_mass[:, np.newaxis] > 0,
            out=np.zeros_like(masses_box),
        )
        effective_density = np.sum(mass_fractions * particles.density, axis=-1)
        new_volume = 4.0 / 3.0 * np.pi * np.power(radii, 3)
        new_total_mass = new_volume * effective_density
        masses_box[:] = mass_fractions * new_total_mass[:, np.newaxis]


def _print_timing(label: str, gpu_time: float, cpu_time: float) -> None:
    """Print timing summary for benchmark output."""
    speedup = cpu_time / gpu_time if gpu_time > 0 else np.nan
    print(
        f"{label}: GPU {gpu_time:.4f}s | CPU {cpu_time:.4f}s | "
        f"speedup {speedup:.2f}x"
    )


def test_condensation_benchmark_large_box() -> None:
    """Benchmark condensation kernel for 1 box x 100k particles."""
    _skip_if_no_cuda()
    n_boxes = 1
    n_particles = 100_000
    n_species = 3
    particles = _make_particle_data(n_boxes, n_particles, n_species)
    gas = _make_gas_data(n_boxes, n_species)
    vapor_pressure = _make_vapor_pressure(n_boxes, n_species)
    surface_tension = np.full(
        n_species, DEFAULT_SURFACE_TENSION, dtype=np.float64
    )
    mass_accommodation = np.full(
        n_species, DEFAULT_MASS_ACCOMMODATION, dtype=np.float64
    )
    diffusion_vapor = np.full(
        n_species, DEFAULT_DIFFUSION_COEFFICIENT, dtype=np.float64
    )

    gpu_particles = to_warp_particle_data(particles, device="cuda")
    gpu_gas = to_warp_gas_data(
        gas,
        device="cuda",
        vapor_pressure=vapor_pressure,
    )
    mass_transfer_buffer = wp.zeros(
        (n_boxes, n_particles, n_species), dtype=wp.float64, device="cuda"
    )
    surface_tension_wp = wp.array(
        surface_tension, dtype=wp.float64, device="cuda"
    )
    mass_accommodation_wp = wp.array(
        mass_accommodation, dtype=wp.float64, device="cuda"
    )
    diffusion_vapor_wp = wp.array(
        diffusion_vapor, dtype=wp.float64, device="cuda"
    )

    def gpu_step() -> None:
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=DEFAULT_TEMPERATURE,
            pressure=DEFAULT_PRESSURE,
            time_step=DEFAULT_TIME_STEP,
            surface_tension=surface_tension_wp,
            mass_accommodation=mass_accommodation_wp,
            diffusion_coefficient_vapor=diffusion_vapor_wp,
            mass_transfer=mass_transfer_buffer,
        )

    with _warp_profiled("condensation_large_box"):
        gpu_time = _time_gpu_loop(gpu_step, DEFAULT_STEPS, DEFAULT_WARMUP)

    cpu_particles = particles.copy()
    cpu_mass_transfer = np.zeros_like(cpu_particles.masses)

    def cpu_step() -> None:
        _cpu_condensation_step(
            cpu_particles,
            gas,
            vapor_pressure,
            surface_tension,
            mass_accommodation,
            diffusion_vapor,
            DEFAULT_TEMPERATURE,
            DEFAULT_PRESSURE,
            DEFAULT_TIME_STEP,
            cpu_mass_transfer,
        )

    cpu_time = _time_cpu_loop(cpu_step, DEFAULT_STEPS, DEFAULT_WARMUP)
    _print_timing("Condensation large box", gpu_time, cpu_time)
    speedup = _compute_speedup(cpu_time, gpu_time)
    assert speedup > 10.0


def test_condensation_benchmark_many_boxes() -> None:
    """Benchmark condensation kernel for 100 boxes x 1k particles."""
    _skip_if_no_cuda()
    n_boxes = 100
    n_particles = 1_000
    n_species = 3
    particles = _make_particle_data(n_boxes, n_particles, n_species)
    gas = _make_gas_data(n_boxes, n_species)
    vapor_pressure = _make_vapor_pressure(n_boxes, n_species)
    surface_tension = np.full(
        n_species, DEFAULT_SURFACE_TENSION, dtype=np.float64
    )
    mass_accommodation = np.full(
        n_species, DEFAULT_MASS_ACCOMMODATION, dtype=np.float64
    )
    diffusion_vapor = np.full(
        n_species, DEFAULT_DIFFUSION_COEFFICIENT, dtype=np.float64
    )

    gpu_particles = to_warp_particle_data(particles, device="cuda")
    gpu_gas = to_warp_gas_data(
        gas,
        device="cuda",
        vapor_pressure=vapor_pressure,
    )
    mass_transfer_buffer = wp.zeros(
        (n_boxes, n_particles, n_species), dtype=wp.float64, device="cuda"
    )
    surface_tension_wp = wp.array(
        surface_tension, dtype=wp.float64, device="cuda"
    )
    mass_accommodation_wp = wp.array(
        mass_accommodation, dtype=wp.float64, device="cuda"
    )
    diffusion_vapor_wp = wp.array(
        diffusion_vapor, dtype=wp.float64, device="cuda"
    )

    def gpu_step() -> None:
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=DEFAULT_TEMPERATURE,
            pressure=DEFAULT_PRESSURE,
            time_step=DEFAULT_TIME_STEP,
            surface_tension=surface_tension_wp,
            mass_accommodation=mass_accommodation_wp,
            diffusion_coefficient_vapor=diffusion_vapor_wp,
            mass_transfer=mass_transfer_buffer,
        )

    with _warp_profiled("condensation_many_boxes"):
        gpu_time = _time_gpu_loop(gpu_step, DEFAULT_STEPS, DEFAULT_WARMUP)

    cpu_particles = particles.copy()
    cpu_mass_transfer = np.zeros_like(cpu_particles.masses)

    def cpu_step() -> None:
        _cpu_condensation_step(
            cpu_particles,
            gas,
            vapor_pressure,
            surface_tension,
            mass_accommodation,
            diffusion_vapor,
            DEFAULT_TEMPERATURE,
            DEFAULT_PRESSURE,
            DEFAULT_TIME_STEP,
            cpu_mass_transfer,
        )

    cpu_time = _time_cpu_loop(cpu_step, DEFAULT_STEPS, DEFAULT_WARMUP)
    _print_timing("Condensation many boxes", gpu_time, cpu_time)


def test_coagulation_benchmark_large_box() -> None:
    """Benchmark coagulation kernel for 1 box x 100k particles."""
    _skip_if_no_cuda()
    n_boxes = 1
    n_particles = 100_000
    n_species = 2
    particles = _make_particle_data(n_boxes, n_particles, n_species)

    gpu_particles = to_warp_particle_data(particles, device="cuda")
    collision_pairs = wp.zeros(
        (n_boxes, MAX_COLLISIONS, 2), dtype=wp.int32, device="cuda"
    )
    n_collisions = wp.zeros((n_boxes,), dtype=wp.int32, device="cuda")
    rng_states = wp.zeros((n_boxes,), dtype=wp.uint32, device="cuda")
    step_counter = {"idx": 0}

    def gpu_step() -> None:
        coagulation_step_gpu(
            gpu_particles,
            temperature=DEFAULT_TEMPERATURE,
            pressure=DEFAULT_PRESSURE,
            time_step=DEFAULT_TIME_STEP,
            rng_seed=42 + step_counter["idx"],
            max_collisions=MAX_COLLISIONS,
            collision_pairs=collision_pairs,
            n_collisions=n_collisions,
            rng_states=rng_states,
        )
        step_counter["idx"] += 1

    with _warp_profiled("coagulation_large_box"):
        gpu_time = _time_gpu_loop(gpu_step, DEFAULT_STEPS, DEFAULT_WARMUP)

    cpu_particles = particles.copy()
    rng = np.random.default_rng(42)
    kernel_radius = _build_kernel_radius(cpu_particles.radii)

    def cpu_step() -> None:
        _cpu_coagulation_step(
            cpu_particles,
            DEFAULT_TEMPERATURE,
            DEFAULT_PRESSURE,
            DEFAULT_TIME_STEP,
            rng,
            kernel_radius,
        )

    cpu_time = _time_cpu_loop(cpu_step, DEFAULT_STEPS, DEFAULT_WARMUP)
    _print_timing("Coagulation large box", gpu_time, cpu_time)
    speedup = _compute_speedup(cpu_time, gpu_time)
    assert speedup > 10.0


def test_coagulation_benchmark_many_boxes() -> None:
    """Benchmark coagulation kernel for 100 boxes x 1k particles."""
    _skip_if_no_cuda()
    n_boxes = 100
    n_particles = 1_000
    n_species = 2
    particles = _make_particle_data(n_boxes, n_particles, n_species)

    gpu_particles = to_warp_particle_data(particles, device="cuda")
    collision_pairs = wp.zeros(
        (n_boxes, MAX_COLLISIONS, 2), dtype=wp.int32, device="cuda"
    )
    n_collisions = wp.zeros((n_boxes,), dtype=wp.int32, device="cuda")
    rng_states = wp.zeros((n_boxes,), dtype=wp.uint32, device="cuda")
    step_counter = {"idx": 0}

    def gpu_step() -> None:
        coagulation_step_gpu(
            gpu_particles,
            temperature=DEFAULT_TEMPERATURE,
            pressure=DEFAULT_PRESSURE,
            time_step=DEFAULT_TIME_STEP,
            rng_seed=100 + step_counter["idx"],
            max_collisions=MAX_COLLISIONS,
            collision_pairs=collision_pairs,
            n_collisions=n_collisions,
            rng_states=rng_states,
        )
        step_counter["idx"] += 1

    with _warp_profiled("coagulation_many_boxes"):
        gpu_time = _time_gpu_loop(gpu_step, DEFAULT_STEPS, DEFAULT_WARMUP)

    cpu_particles = particles.copy()
    rng = np.random.default_rng(7)
    kernel_radius = _build_kernel_radius(cpu_particles.radii)

    def cpu_step() -> None:
        _cpu_coagulation_step(
            cpu_particles,
            DEFAULT_TEMPERATURE,
            DEFAULT_PRESSURE,
            DEFAULT_TIME_STEP,
            rng,
            kernel_radius,
        )

    cpu_time = _time_cpu_loop(cpu_step, DEFAULT_STEPS, DEFAULT_WARMUP)
    _print_timing("Coagulation many boxes", gpu_time, cpu_time)


@wp.kernel
# type: ignore[misc]
def _diffusion_coefficient_kernel(
    temperatures: Any,
    mobilities: Any,
    boltzmann_constant: Any,
    result: Any,
) -> None:
    """Evaluate diffusion_coefficient_wp across an array."""
    tid = wp.tid()  # type: ignore[misc]
    result[tid] = diffusion_coefficient_wp(
        temperatures[tid], mobilities[tid], boltzmann_constant
    )


@wp.kernel
# type: ignore[misc]
def _mass_transfer_rate_kernel(
    pressure_deltas: Any,
    mass_transport: Any,
    temperatures: Any,
    molar_masses: Any,
    gas_constant: Any,
    result: Any,
) -> None:
    """Evaluate mass_transfer_rate_wp across an array."""
    tid = wp.tid()  # type: ignore[misc]
    result[tid] = mass_transfer_rate_wp(
        pressure_deltas[tid],
        mass_transport[tid],
        temperatures[tid],
        molar_masses[tid],
        gas_constant,
    )


@wp.kernel
# type: ignore[misc]
def _brownian_diffusivity_kernel(
    temperatures: Any,
    mobilities: Any,
    boltzmann_constant: Any,
    result: Any,
) -> None:
    """Evaluate brownian_diffusivity_wp across an array."""
    tid = wp.tid()  # type: ignore[misc]
    result[tid] = brownian_diffusivity_wp(
        temperatures[tid], mobilities[tid], boltzmann_constant
    )


@wp.kernel
# type: ignore[misc]
def _brownian_kernel_pair_kernel(
    radii_i: Any,
    radii_j: Any,
    diff_i: Any,
    diff_j: Any,
    g_i: Any,
    g_j: Any,
    speed_i: Any,
    speed_j: Any,
    result: Any,
) -> None:
    """Evaluate brownian_kernel_pair_wp across an array."""
    tid = wp.tid()  # type: ignore[misc]
    result[tid] = brownian_kernel_pair_wp(
        radii_i[tid],
        radii_j[tid],
        diff_i[tid],
        diff_j[tid],
        g_i[tid],
        g_j[tid],
        speed_i[tid],
        speed_j[tid],
        wp.float64(1.0),
    )


@wp.kernel
# type: ignore[misc]
def _particle_radius_kernel(volumes: Any, result: Any) -> None:
    """Evaluate particle_radius_from_volume_wp across an array."""
    tid = wp.tid()  # type: ignore[misc]
    result[tid] = particle_radius_from_volume_wp(volumes[tid])


def test_wp_func_benchmarks() -> None:
    """Benchmark key Warp @wp.func utilities against NumPy equivalents."""
    _skip_if_no_cuda()
    n_evals = 100_000
    rng = np.random.default_rng(123)
    temperatures = rng.uniform(280.0, 320.0, size=n_evals).astype(np.float64)
    mobilities = rng.uniform(1.0e-8, 5.0e-8, size=n_evals).astype(np.float64)
    pressure_deltas = rng.uniform(-5.0, 10.0, size=n_evals).astype(np.float64)
    mass_transport = rng.uniform(1.0e-18, 1.0e-16, size=n_evals).astype(
        np.float64
    )
    molar_masses = rng.uniform(0.018, 0.05, size=n_evals).astype(np.float64)
    total_volumes = rng.uniform(1.0e-21, 1.0e-18, size=n_evals).astype(
        np.float64
    )

    cpu_start = time.perf_counter()
    _ = get_diffusion_coefficient(
        temperature=temperatures,
        aerodynamic_mobility=mobilities,
        boltzmann_constant=constants.BOLTZMANN_CONSTANT,
    )
    cpu_diffusion_time = time.perf_counter() - cpu_start

    cpu_start = time.perf_counter()
    _ = get_mass_transfer_rate(
        pressure_delta=pressure_deltas,
        first_order_mass_transport=mass_transport,
        temperature=temperatures,
        molar_mass=molar_masses,
    )
    cpu_mass_transfer_time = time.perf_counter() - cpu_start

    cpu_start = time.perf_counter()
    _ = np.cbrt(3.0 * total_volumes / (4.0 * np.pi))
    cpu_radius_time = time.perf_counter() - cpu_start

    cpu_start = time.perf_counter()
    _ = constants.BOLTZMANN_CONSTANT * temperatures * mobilities
    cpu_brownian_diffusivity_time = time.perf_counter() - cpu_start

    kernel_sample = 256
    cpu_start = time.perf_counter()
    _ = get_brownian_kernel_via_system_state(
        particle_radius=temperatures[:kernel_sample] * 0.0 + 1.0e-8,
        particle_mass=np.full(kernel_sample, 1.0e-18, dtype=np.float64),
        temperature=DEFAULT_TEMPERATURE,
        pressure=DEFAULT_PRESSURE,
    )
    cpu_brownian_kernel_time = time.perf_counter() - cpu_start

    temperatures_wp = wp.array(temperatures, dtype=wp.float64, device="cuda")
    mobilities_wp = wp.array(mobilities, dtype=wp.float64, device="cuda")
    pressure_wp = wp.array(pressure_deltas, dtype=wp.float64, device="cuda")
    mass_transport_wp = wp.array(
        mass_transport, dtype=wp.float64, device="cuda"
    )
    molar_mass_wp = wp.array(molar_masses, dtype=wp.float64, device="cuda")
    volumes_wp = wp.array(total_volumes, dtype=wp.float64, device="cuda")

    diffusion_out = wp.zeros(n_evals, dtype=wp.float64, device="cuda")
    transfer_out = wp.zeros(n_evals, dtype=wp.float64, device="cuda")
    radius_out = wp.zeros(n_evals, dtype=wp.float64, device="cuda")

    wp.launch(
        _diffusion_coefficient_kernel,
        dim=n_evals,
        inputs=[
            temperatures_wp,
            mobilities_wp,
            wp.float64(constants.BOLTZMANN_CONSTANT),
        ],
        outputs=[diffusion_out],
        device="cuda",
    )
    wp.launch(
        _mass_transfer_rate_kernel,
        dim=n_evals,
        inputs=[
            pressure_wp,
            mass_transport_wp,
            temperatures_wp,
            molar_mass_wp,
            wp.float64(constants.GAS_CONSTANT),
        ],
        outputs=[transfer_out],
        device="cuda",
    )
    wp.launch(
        _particle_radius_kernel,
        dim=n_evals,
        inputs=[volumes_wp],
        outputs=[radius_out],
        device="cuda",
    )
    wp.synchronize()

    start = time.perf_counter()
    wp.launch(
        _diffusion_coefficient_kernel,
        dim=n_evals,
        inputs=[
            temperatures_wp,
            mobilities_wp,
            wp.float64(constants.BOLTZMANN_CONSTANT),
        ],
        outputs=[diffusion_out],
        device="cuda",
    )
    wp.synchronize()
    gpu_diffusion_time = time.perf_counter() - start

    start = time.perf_counter()
    wp.launch(
        _mass_transfer_rate_kernel,
        dim=n_evals,
        inputs=[
            pressure_wp,
            mass_transport_wp,
            temperatures_wp,
            molar_mass_wp,
            wp.float64(constants.GAS_CONSTANT),
        ],
        outputs=[transfer_out],
        device="cuda",
    )
    wp.synchronize()
    gpu_mass_transfer_time = time.perf_counter() - start

    start = time.perf_counter()
    wp.launch(
        _particle_radius_kernel,
        dim=n_evals,
        inputs=[volumes_wp],
        outputs=[radius_out],
        device="cuda",
    )
    wp.synchronize()
    gpu_radius_time = time.perf_counter() - start

    radii_i = rng.uniform(1.0e-9, 1.0e-7, size=n_evals).astype(np.float64)
    radii_j = rng.uniform(1.0e-9, 1.0e-7, size=n_evals).astype(np.float64)
    diff_i = rng.uniform(1.0e-10, 1.0e-9, size=n_evals).astype(np.float64)
    diff_j = rng.uniform(1.0e-10, 1.0e-9, size=n_evals).astype(np.float64)
    g_i = rng.uniform(1.0e-9, 1.0e-8, size=n_evals).astype(np.float64)
    g_j = rng.uniform(1.0e-9, 1.0e-8, size=n_evals).astype(np.float64)
    speed_i = rng.uniform(10.0, 40.0, size=n_evals).astype(np.float64)
    speed_j = rng.uniform(10.0, 40.0, size=n_evals).astype(np.float64)

    radii_i_wp = wp.array(radii_i, dtype=wp.float64, device="cuda")
    radii_j_wp = wp.array(radii_j, dtype=wp.float64, device="cuda")
    diff_i_wp = wp.array(diff_i, dtype=wp.float64, device="cuda")
    diff_j_wp = wp.array(diff_j, dtype=wp.float64, device="cuda")
    g_i_wp = wp.array(g_i, dtype=wp.float64, device="cuda")
    g_j_wp = wp.array(g_j, dtype=wp.float64, device="cuda")
    speed_i_wp = wp.array(speed_i, dtype=wp.float64, device="cuda")
    speed_j_wp = wp.array(speed_j, dtype=wp.float64, device="cuda")
    brownian_out = wp.zeros(n_evals, dtype=wp.float64, device="cuda")

    wp.launch(
        _brownian_diffusivity_kernel,
        dim=n_evals,
        inputs=[
            temperatures_wp,
            mobilities_wp,
            wp.float64(constants.BOLTZMANN_CONSTANT),
        ],
        outputs=[diffusion_out],
        device="cuda",
    )
    wp.launch(
        _brownian_kernel_pair_kernel,
        dim=n_evals,
        inputs=[
            radii_i_wp,
            radii_j_wp,
            diff_i_wp,
            diff_j_wp,
            g_i_wp,
            g_j_wp,
            speed_i_wp,
            speed_j_wp,
        ],
        outputs=[brownian_out],
        device="cuda",
    )
    wp.synchronize()

    start = time.perf_counter()
    wp.launch(
        _brownian_diffusivity_kernel,
        dim=n_evals,
        inputs=[
            temperatures_wp,
            mobilities_wp,
            wp.float64(constants.BOLTZMANN_CONSTANT),
        ],
        outputs=[diffusion_out],
        device="cuda",
    )
    wp.synchronize()
    gpu_brownian_diffusivity_time = time.perf_counter() - start

    start = time.perf_counter()
    wp.launch(
        _brownian_kernel_pair_kernel,
        dim=n_evals,
        inputs=[
            radii_i_wp,
            radii_j_wp,
            diff_i_wp,
            diff_j_wp,
            g_i_wp,
            g_j_wp,
            speed_i_wp,
            speed_j_wp,
        ],
        outputs=[brownian_out],
        device="cuda",
    )
    wp.synchronize()
    gpu_brownian_kernel_time = time.perf_counter() - start

    kernel_calls = kernel_sample * kernel_sample
    print(
        "@wp.func timings (per call, microseconds):\n"
        f"  diffusion_coefficient_wp: CPU {cpu_diffusion_time / n_evals * 1e6:.4f} | "
        f"GPU {gpu_diffusion_time / n_evals * 1e6:.4f}\n"
        f"  mass_transfer_rate_wp: CPU {cpu_mass_transfer_time / n_evals * 1e6:.4f} | "
        f"GPU {gpu_mass_transfer_time / n_evals * 1e6:.4f}\n"
        f"  particle_radius_from_volume_wp: CPU {cpu_radius_time / n_evals * 1e6:.4f} | "
        f"GPU {gpu_radius_time / n_evals * 1e6:.4f}\n"
        f"  brownian_diffusivity_wp: CPU {cpu_brownian_diffusivity_time / n_evals * 1e6:.4f} | "
        f"GPU {gpu_brownian_diffusivity_time / n_evals * 1e6:.4f}\n"
        f"  brownian_kernel_pair_wp: CPU {cpu_brownian_kernel_time / kernel_calls * 1e6:.4f} | "
        f"GPU {gpu_brownian_kernel_time / n_evals * 1e6:.4f}"
    )
