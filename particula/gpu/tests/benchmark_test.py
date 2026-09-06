"""Opt-in GPU benchmarks and resident profiling evidence.

Run with:
    pytest particula/gpu/tests/benchmark_test.py --benchmark -v -s

Set ``WARP_PROFILE=1`` to enable Warp capture hooks for Nsight/warp
profiling. When enabled, run Nsight Systems/Compute while the benchmark
executes to inspect memory access patterns and kernel launch metrics.

Condensation benchmarks use a generic deterministic particle fixture.
Coagulation benchmarks use a dedicated mixed-scale fixture so the timed
path reflects the shipped NPF/droplet regression baseline without
changing condensation setup.

Resident launch profiling uses native CUDA capture only and publishes
separate host-launch and synchronized-completion evidence. It does not
provide a CPU or Warp-CPU fallback.
"""

# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false

from __future__ import annotations

import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from particula._pytest_support import benchmark_option_enabled_from_env
from particula.execution.tests.resident_benchmark_cuda_support import (
    ResidentBenchmarkUnavailableError,
    cuda_capture_availability,
    qualified_cuda_resident_benchmark,
    resident_benchmark_provenance,
)
from particula.execution.tests.resident_benchmark_support import (
    ResidentBenchmarkArtifact,
    ResidentBenchmarkResult,
    ResidentBenchmarkStatus,
    ResidentMemoryObservation,
    build_default_resident_benchmark_matrix,
    build_resident_benchmark_metadata,
    build_resident_case_provenance_digest,
    build_resident_memory_comparison,
    build_resident_memory_model,
    collect_paired_device_timings,
    preflight_resident_benchmark_case,
    summarize_timing_samples,
    write_resident_capture_comparison_artifact,
)
from particula.gpu.tests.cuda_availability import (
    CUDA_SKIP_REASON,
    cuda_available,
)
from particula.gpu.tests.mass_precision_study_support import (
    _build_mass_precision_cases,
    _project_candidate,
)
from particula.gpu.tests.profiling_support import (
    REPLAY_COUNTS,
    ExecutedEvidence,
    MachineProvenance,
    MeasurementMethod,
    NormalizedMetric,
    ProfilingArtifact,
    RawDurationSample,
    UnavailableEvidence,
    build_default_profiling_workload_matrix,
    build_raw_report_provenance,
    ensure_profiling_raw_root,
    serialize_profiling_artifact,
)


def _benchmark_enabled() -> bool:
    """Check whether pytest resolved benchmark mode for this process."""
    return benchmark_option_enabled_from_env()


if not _benchmark_enabled():
    pytest.skip(
        "GPU benchmarks skipped (pass --benchmark to enable)",
        allow_module_level=True,
    )

wp: Any
try:
    import warp as wp
except ImportError:
    wp = None

from particula.dynamics.coagulation.brownian_kernel import (  # noqa: E402
    get_brownian_kernel_via_system_state,
)
from particula.dynamics.coagulation.particle_resolved_step import (  # noqa: E402
    particle_resolved_method as pr_method,
)

get_particle_resolved_coagulation_step = (
    pr_method.get_particle_resolved_coagulation_step
)
get_particle_resolved_update_step = pr_method.get_particle_resolved_update_step
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

if wp is not None:
    from particula.gpu.conversion import to_warp_gas_data, to_warp_particle_data
    from particula.gpu.dynamics.coagulation_funcs import (
        brownian_diffusivity_wp,
        brownian_kernel_pair_wp,
    )
    from particula.gpu.dynamics.condensation_funcs import mass_transfer_rate_wp
    from particula.gpu.kernels.coagulation import (
        coagulation_step_gpu,
        initialize_coagulation_rng_states,
    )
    from particula.gpu.kernels.condensation import condensation_step_gpu
    from particula.gpu.kernels.thermodynamics import ThermodynamicsConfig
    from particula.gpu.properties.particle_properties import (
        diffusion_coefficient_wp,
        particle_radius_from_volume_wp,
    )

pytestmark = [pytest.mark.slow, pytest.mark.performance, pytest.mark.benchmark]

DEFAULT_TEMPERATURE = 298.15
DEFAULT_PRESSURE = 101325.0
DEFAULT_TIME_STEP = 0.5
DEFAULT_STEPS = 5
DEFAULT_WARMUP = 1
DEFAULT_SURFACE_TENSION = 0.072
DEFAULT_MASS_ACCOMMODATION = 1.0
DEFAULT_DIFFUSION_COEFFICIENT = 2.0e-5
MAX_COLLISIONS = 256
DEFAULT_BENCHMARK_MAX_BYTES = 2 * 1024 * 1024 * 1024
# P1/P2 configured conservative requested-case estimates. These are explicit
# matrix inputs, not P3 byte formulas or allocator observations.
RESIDENT_BENCHMARK_REQUESTED_BYTES_BY_SHAPE = {
    (1, 16, 2): 64 * 1024 * 1024,
    (10, 16, 2): 256 * 1024 * 1024,
    (100, 16, 2): 1024 * 1024 * 1024,
    (1000, 16, 2): 4 * 1024 * 1024 * 1024,
}
BENCHMARK_ARTIFACT_DIR = Path(".artifacts") / "benchmarks"
DEFAULT_BENCHMARK_OUTPUT_NAME = "gpu_benchmark_results.json"
PROFILING_ARTIFACT_NAMES = {
    (
        "prepared_uncaptured",
        "host_launch",
    ): "prepared_uncaptured_host_launch.json",
    ("prepared_uncaptured", "synchronized_elapsed"): (
        "prepared_uncaptured_synchronized_elapsed.json"
    ),
    ("captured_replay", "host_launch"): "captured_replay_host_launch.json",
    ("captured_replay", "synchronized_elapsed"): (
        "captured_replay_synchronized_elapsed.json"
    ),
}

# ---------------------------------------------------------------------------
# Scaling configurations
# ---------------------------------------------------------------------------
# Each tuple: (label, n_boxes, n_particles, n_species, run_cpu)
#
# Budget targets (~10 min total for full suite):
#   Condensation GPU is O(n_boxes * n_particles) and very fast (<1s even
#   at 1M particles), so we scale aggressively.  CPU is O(n_boxes *
#   n_particles * n_species) with Python loops, so we cap at 10k.
#
#   Coagulation GPU is O(n_boxes * n_particles) after the k_max fix but
#   the sequential pair-selection loop still dominates.  CPU uses the
#   bin-pair method which scales well to ~5k particles per box.

CONDENSATION_CONFIGS: list[tuple[str, int, int, int, bool]] = [
    # label               boxes  particles species  cpu?
    # --- CPU + GPU comparison (light CPU) ---
    ("1x1k", 1, 1_000, 3, True),
    ("1x10k", 1, 10_000, 3, True),
    ("10x1k", 10, 1_000, 3, True),
    ("100x1k", 100, 1_000, 3, True),
    # --- GPU-only scaling ---
    ("1x100k", 1, 100_000, 3, False),
    ("1x500k", 1, 500_000, 3, False),
    ("1x1M", 1, 1_000_000, 3, False),
    ("1x2M", 1, 2_000_000, 3, False),
    ("10x10k", 10, 10_000, 3, False),
    ("10x100k", 10, 100_000, 3, False),
    ("100x10k", 100, 10_000, 3, False),
    ("100x100k", 100, 100_000, 3, False),
]

COAGULATION_CONFIGS: list[tuple[str, int, int, int, bool]] = [
    # label               boxes  particles species  cpu?
    # --- CPU + GPU comparison (light CPU) ---
    ("1x500", 1, 500, 2, True),
    ("1x2k", 1, 2_000, 2, True),
    ("1x5k", 1, 5_000, 2, True),
    ("10x500", 10, 500, 2, True),
    ("10x1k", 10, 1_000, 2, True),
    # --- GPU-only scaling ---
    ("1x10k", 1, 10_000, 2, False),
    ("1x20k", 1, 20_000, 2, False),
    ("1x50k", 1, 50_000, 2, False),
    ("10x5k", 10, 5_000, 2, False),
    ("10x10k", 10, 10_000, 2, False),
    ("50x1k", 50, 1_000, 2, False),
    ("50x5k", 50, 5_000, 2, False),
    ("100x1k", 100, 1_000, 2, False),
]


@dataclass(frozen=True)
class BenchmarkMemoryBudget:
    """Estimated benchmark allocation footprint."""

    label: str
    cpu_bytes: int
    gpu_bytes: int

    @property
    def total_bytes(self) -> int:
        """Return the combined CPU and GPU allocation estimate."""
        return self.cpu_bytes + self.gpu_bytes


def _parse_positive_int_env(name: str, default: int) -> int:
    """Parse a positive integer environment override."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a positive integer, got {raw_value!r}"
        ) from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _sanitize_benchmark_output_name(raw_value: str) -> str:
    """Normalize an output override to a safe artifact filename."""
    candidate = Path(raw_value.strip()).name
    if candidate in {"", ".", ".."}:
        raise ValueError(
            "BENCHMARK_OUTPUT must resolve to a non-empty filename"
        )
    return candidate


def _get_benchmark_output_path() -> Path:
    """Resolve the benchmark JSON output inside the controlled artifact root."""
    file_name = _sanitize_benchmark_output_name(
        os.getenv("BENCHMARK_OUTPUT", DEFAULT_BENCHMARK_OUTPUT_NAME)
    )
    return BENCHMARK_ARTIFACT_DIR / file_name


def _summarize_warp_device(device: Any) -> dict[str, Any]:
    """Return a bounded JSON-safe summary for a Warp device object."""
    summary: dict[str, Any] = {}
    for attr in (
        "alias",
        "name",
        "ordinal",
        "arch",
        "is_cuda",
        "is_cpu",
        "is_uva",
    ):
        value = getattr(device, attr, None)
        if isinstance(value, str | int | bool):
            summary[attr] = value

    total_memory = getattr(device, "total_memory", None)
    if isinstance(total_memory, int | float):
        summary["total_memory_bytes"] = int(total_memory)
    return summary


def _build_benchmark_metadata() -> dict[str, Any]:
    """Capture traceable command and runtime context for benchmark output."""
    metadata: dict[str, Any] = {
        "command": " ".join(sys.argv),
        "artifact_path": str(BENCHMARK_OUTPUT),
    }
    warp_version = getattr(wp, "__version__", None)
    if warp_version is not None:
        metadata["warp_version"] = str(warp_version)

    if wp is None:
        metadata["warp_available"] = False
        return metadata

    metadata["warp_available"] = True
    cuda_ready = cuda_available(wp)
    metadata["cuda_available"] = cuda_ready
    if not cuda_ready:
        return metadata

    try:
        metadata["device"] = _summarize_warp_device(wp.get_device("cuda"))
    except Exception as exc:  # pragma: no cover - defensive metadata path
        metadata["device_probe_error"] = str(exc)
    return metadata


BENCHMARK_OUTPUT = _get_benchmark_output_path()
WARP_FLOAT64: Any
WARP_FLOAT32: Any
WARP_INT32: Any
WARP_UINT32: Any
if wp is None:
    WARP_FLOAT64 = np.float64
    WARP_FLOAT32 = np.float32
    WARP_INT32 = np.int32
    WARP_UINT32 = np.uint32
else:
    WARP_FLOAT64 = wp.float64
    WARP_FLOAT32 = wp.float32
    WARP_INT32 = wp.int32
    WARP_UINT32 = wp.uint32


def _warp_dtype_nbytes(dtype: Any) -> int:
    """Return bytes per element for the supported Warp benchmark dtypes."""
    warp_item_sizes = {
        WARP_FLOAT64: 8,
        WARP_FLOAT32: 4,
        WARP_INT32: 4,
        WARP_UINT32: 4,
    }
    try:
        return warp_item_sizes[dtype]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported Warp dtype for sizing: {dtype!r}"
        ) from exc


def _array_nbytes(shape: tuple[int, ...], itemsize: int) -> int:
    """Return the byte footprint for a dense array shape."""
    element_count = int(np.prod(np.asarray(shape, dtype=np.int64)))
    return element_count * itemsize


def _numpy_nbytes(shape: tuple[int, ...], dtype: Any) -> int:
    """Return the byte footprint for a NumPy array shape and dtype."""
    return _array_nbytes(shape, np.dtype(dtype).itemsize)


def _warp_nbytes(shape: tuple[int, ...], dtype: Any) -> int:
    """Return the byte footprint for a Warp array shape and dtype."""
    return _array_nbytes(shape, _warp_dtype_nbytes(dtype))


def _estimate_condensation_budget(
    label: str,
    n_boxes: int,
    n_particles: int,
    n_species: int,
    run_cpu: bool,
) -> BenchmarkMemoryBudget:
    """Estimate cumulative large-array allocations for condensation setup."""
    particle_shape = (n_boxes, n_particles, n_species)
    box_particle_shape = (n_boxes, n_particles)
    box_species_shape = (n_boxes, n_species)

    cpu_bytes = 0
    cpu_bytes += _numpy_nbytes(particle_shape, np.float64)  # masses
    cpu_bytes += _numpy_nbytes(box_particle_shape, np.float64)  # concentration
    cpu_bytes += _numpy_nbytes(box_particle_shape, np.float64)  # charge
    cpu_bytes += _numpy_nbytes((n_species,), np.float64)  # density
    cpu_bytes += _numpy_nbytes((n_boxes,), np.float64)  # volume
    cpu_bytes += _numpy_nbytes((n_species,), np.float64)  # molar mass
    cpu_bytes += _numpy_nbytes(box_species_shape, np.float64)  # gas conc
    cpu_bytes += _numpy_nbytes((n_species,), np.bool_)  # partitioning
    cpu_bytes += _numpy_nbytes(box_species_shape, np.float64)  # vapor pressure
    cpu_bytes += _numpy_nbytes((n_species,), np.float64) * 3  # species vectors
    if run_cpu:
        cpu_bytes += _numpy_nbytes(particle_shape, np.float64)  # copy masses
        cpu_bytes += _numpy_nbytes(box_particle_shape, np.float64) * 2
        cpu_bytes += _numpy_nbytes((n_species,), np.float64)
        cpu_bytes += _numpy_nbytes((n_boxes,), np.float64)
        cpu_bytes += _numpy_nbytes(particle_shape, np.float64)  # mass transfer

    gpu_bytes = 0
    gpu_bytes += _warp_nbytes(particle_shape, WARP_FLOAT64) * 2
    gpu_bytes += _warp_nbytes(box_particle_shape, WARP_FLOAT64) * 2
    gpu_bytes += _warp_nbytes((n_species,), WARP_FLOAT64) * 4
    gpu_bytes += _warp_nbytes((n_boxes,), WARP_FLOAT64)
    gpu_bytes += _warp_nbytes(box_species_shape, WARP_FLOAT64) * 2
    gpu_bytes += _warp_nbytes((n_species,), WARP_INT32)

    return BenchmarkMemoryBudget(
        label=label, cpu_bytes=cpu_bytes, gpu_bytes=gpu_bytes
    )


def _estimate_coagulation_budget(
    label: str,
    n_boxes: int,
    n_particles: int,
    n_species: int,
    run_cpu: bool,
) -> BenchmarkMemoryBudget:
    """Estimate cumulative large-array allocations for coagulation setup."""
    particle_shape = (n_boxes, n_particles, n_species)
    box_particle_shape = (n_boxes, n_particles)

    cpu_bytes = 0
    cpu_bytes += _numpy_nbytes(particle_shape, np.float64)
    cpu_bytes += _numpy_nbytes(box_particle_shape, np.float64) * 2
    cpu_bytes += _numpy_nbytes((n_species,), np.float64)
    cpu_bytes += _numpy_nbytes((n_boxes,), np.float64)
    if run_cpu:
        cpu_bytes += _numpy_nbytes(particle_shape, np.float64)
        cpu_bytes += _numpy_nbytes(box_particle_shape, np.float64) * 2
        cpu_bytes += _numpy_nbytes((n_species,), np.float64)
        cpu_bytes += _numpy_nbytes((n_boxes,), np.float64)
        cpu_bytes += _numpy_nbytes((n_particles,), np.float64) * 7
        cpu_bytes += _numpy_nbytes((64,), np.float64)
        cpu_bytes += _numpy_nbytes((64, 64), np.float64)

    gpu_bytes = 0
    gpu_bytes += _warp_nbytes(particle_shape, WARP_FLOAT64)
    gpu_bytes += _warp_nbytes(box_particle_shape, WARP_FLOAT64) * 2
    gpu_bytes += _warp_nbytes((n_species,), WARP_FLOAT64)
    gpu_bytes += _warp_nbytes((n_boxes,), WARP_FLOAT64)
    gpu_bytes += _warp_nbytes((n_boxes, MAX_COLLISIONS, 2), WARP_INT32)
    gpu_bytes += _warp_nbytes((n_boxes,), WARP_INT32)
    gpu_bytes += _warp_nbytes((n_boxes,), WARP_UINT32)

    return BenchmarkMemoryBudget(
        label=label, cpu_bytes=cpu_bytes, gpu_bytes=gpu_bytes
    )


def _validate_benchmark_budget(budget: BenchmarkMemoryBudget) -> None:
    """Skip oversized benchmark cases before allocating large buffers."""
    max_bytes = _parse_positive_int_env(
        "BENCHMARK_MAX_BYTES", DEFAULT_BENCHMARK_MAX_BYTES
    )
    if budget.total_bytes > max_bytes:
        pytest.skip(
            f"{budget.label} requires ~{budget.total_bytes:,} bytes, "
            f"exceeding BENCHMARK_MAX_BYTES={max_bytes:,}"
        )


_benchmark_results: dict[str, Any] = {
    "started_at": datetime.now(timezone.utc).isoformat(),
    "benchmark_metadata": _build_benchmark_metadata(),
    "benchmarks": {},
}
_MASS_PRECISION_BENCHMARK_CONFIGS: list[tuple[str, int, str]] = [
    ("npf_cluster", 0, "fp32_absolute_mass"),
    ("accumulation_mode", 2, "fp32_total_mass_fp32_mass_fraction"),
    ("cloud_droplet", 3, "mixed_precision_mass_plus_density"),
]


def _save_results() -> None:
    """Flush current benchmark results to disk as JSON.

    Called after each meaningful timing measurement so that partial
    results survive if the process is interrupted (Ctrl+C, timeout,
    etc.). The file is overwritten each time with the full dict.
    """
    _benchmark_results["updated_at"] = datetime.now(timezone.utc).isoformat()
    output_path = BENCHMARK_OUTPUT
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(_benchmark_results, indent=2) + "\n")
    except OSError as exc:
        raise RuntimeError(
            f"Failed to write benchmark results to {output_path}: {exc}"
        ) from exc
    print(f"  [save] Results written to {output_path}")


def _skip_if_no_cuda() -> None:
    """Skip benchmarks when CUDA is unavailable.

    Raises:
        pytest.SkipTest: With the shared ``CUDA_SKIP_REASON`` message when Warp
            is missing or no CUDA device is present.
    """
    if wp is None or not cuda_available(wp):
        pytest.skip(CUDA_SKIP_REASON)


def _allocation_itemsize(itemsize: int | None, dtype: Any) -> int:
    """Return an allocation itemsize for preflight checks."""
    if itemsize is not None:
        return itemsize
    return int(np.dtype(dtype).itemsize)


def _estimate_allocation_nbytes(shape: tuple[int, ...], itemsize: int) -> int:
    """Estimate the bytes required for one array allocation."""
    total_items = 1
    for value in shape:
        total_items *= value
    return total_items * itemsize


def _preflight_large_allocation(
    shape: tuple[int, ...],
    *,
    label: str,
    itemsize: int,
) -> None:
    """Skip oversized opt-in allocations before attempting device allocation."""
    max_bytes = os.getenv("BENCHMARK_MAX_ALLOC_BYTES")
    if max_bytes is None:
        return
    required_bytes = _estimate_allocation_nbytes(shape, itemsize)
    if required_bytes > int(max_bytes):
        pytest.skip(
            f"Skipping {label}: requires {required_bytes} bytes, limit is "
            f"{max_bytes}"
        )


def _wp_zeros_with_guard(
    shape: tuple[int, ...],
    *,
    dtype: Any,
    device: str,
    label: str,
    itemsize: int | None = None,
):
    """Allocate a Warp array with preflight and graceful failure handling."""
    _preflight_large_allocation(
        shape,
        label=label,
        itemsize=_allocation_itemsize(itemsize, dtype),
    )
    try:
        return wp.zeros(shape, dtype=dtype, device=device)
    except (MemoryError, RuntimeError, ValueError) as exc:
        pytest.skip(f"Skipping {label} allocation on {device}: {exc}")


def _seed_coagulation_rng_states_once(
    *,
    rng_seed: int,
    rng_states: Any,
    n_boxes: int,
    device: str,
) -> None:
    """Seed a caller-owned coagulation RNG buffer exactly once."""
    initialized_states = initialize_coagulation_rng_states(
        rng_seed=rng_seed,
        rng_states=rng_states,
        device=device,
    )
    if getattr(initialized_states, "shape", None) != (n_boxes,):
        raise ValueError("rng_states must keep shape (n_boxes,) after seeding")


@contextmanager
def _warp_profiled(tag: str):
    """Optionally enable Warp capture/profiling when WARP_PROFILE=1.

    Args:
        tag: Label to identify the profiled region.

    Yields:
        Iterator context that wraps optional profiling.
    """
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
    """Time a GPU loop with a single synchronize before/after.

    Args:
        step_fn: Callable that performs one GPU step.
        steps: Number of timed iterations to execute.
        warmup: Number of warmup iterations executed before timing.

    Returns:
        Elapsed time in seconds for the timed iterations.
    """
    for _ in range(warmup):
        step_fn()
    wp.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        step_fn()
    wp.synchronize()
    return time.perf_counter() - start


def _time_cpu_loop(step_fn, steps: int, warmup: int) -> float:
    """Time a CPU loop with warmup iterations.

    Args:
        step_fn: Callable that performs one CPU step.
        steps: Number of timed iterations to execute.
        warmup: Number of warmup iterations executed before timing.

    Returns:
        Elapsed time in seconds for the timed iterations.
    """
    for _ in range(warmup):
        step_fn()
    start = time.perf_counter()
    for _ in range(steps):
        step_fn()
    return time.perf_counter() - start


def _compute_speedup(cpu_time: float, gpu_time: float) -> float:
    """Compute CPU/GPU speedup ratio.

    Args:
        cpu_time: Elapsed CPU time in seconds.
        gpu_time: Elapsed GPU time in seconds.

    Returns:
        Ratio of CPU time to GPU time.

    Raises:
        pytest.SkipTest: If either timing is non-positive.
    """
    if cpu_time <= 0.0 or gpu_time <= 0.0:
        pytest.skip("Invalid timing data")
    return cpu_time / gpu_time


def _preflight_benchmark_array(
    shape: tuple[int, ...],
    *,
    label: str,
    dtype: Any = np.float64,
    itemsize: int | None = None,
) -> None:
    """Preflight one large benchmark array before allocating it."""
    _preflight_large_allocation(
        shape,
        label=label,
        itemsize=_allocation_itemsize(itemsize, dtype),
    )


def _preflight_condensation_case_allocations(
    label: str,
    n_boxes: int,
    n_particles: int,
    n_species: int,
) -> None:
    """Preflight the largest arrays used by a condensation benchmark case."""
    for shape, allocation_label in (
        ((n_boxes, n_particles, n_species), f"cond-{label} particle masses"),
        ((n_boxes, n_particles), f"cond-{label} particle concentration"),
        ((n_boxes, n_species), f"cond-{label} gas concentration"),
        ((n_boxes, n_species), f"cond-{label} vapor pressure"),
        (
            (n_boxes, n_particles, n_species),
            f"cond-{label} gpu mass transfer buffer",
        ),
    ):
        _preflight_benchmark_array(shape, label=allocation_label)


def _preflight_coagulation_case_allocations(
    label: str,
    n_boxes: int,
    n_particles: int,
    n_species: int,
) -> None:
    """Preflight the largest arrays used by a coagulation benchmark case."""
    for shape, allocation_label, itemsize in (
        ((n_boxes, n_particles, n_species), f"coag-{label} particle masses", 8),
        ((n_boxes, n_particles), f"coag-{label} particle concentration", 8),
        ((n_boxes, MAX_COLLISIONS, 2), f"coag-{label} collision pairs", 4),
        ((n_boxes,), f"coag-{label} collision counts", 4),
        ((n_boxes,), f"coag-{label} RNG state", 4),
    ):
        _preflight_benchmark_array(
            shape,
            label=allocation_label,
            itemsize=itemsize,
        )


def _make_particle_data(
    n_boxes: int,
    n_particles: int,
    n_species: int,
    concentration_scale: float = 1.0,
) -> ParticleData:
    """Create deterministic particle data for benchmarks.

    Args:
        n_boxes: Number of spatial boxes to populate.
        n_particles: Number of particles per box.
        n_species: Number of condensible species per particle.
        concentration_scale: Scale factor applied to concentrations.

    Returns:
        ParticleData with masses, concentration, charge, density, and volume.
    """
    base_masses = np.linspace(1.0e-18, 3.0e-18, n_species, dtype=np.float64)
    particle_scale = 0.1 * np.arange(n_particles, dtype=np.float64)
    box_scale = 0.05 * np.arange(n_boxes, dtype=np.float64)
    scales = 1.0 + box_scale[:, np.newaxis] + particle_scale[np.newaxis, :]
    masses = scales[..., np.newaxis] * base_masses[np.newaxis, np.newaxis, :]
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


def _make_coagulation_particle_data(
    n_boxes: int,
    n_particles: int,
    n_species: int,
) -> ParticleData:
    """Create deterministic mixed-scale particle data for coagulation.

    The fixture intentionally combines nanometer-scale and droplet-scale
    particles so opt-in coagulation timings exercise the same broad mass
    spread covered by the mixed-scale regression tests.

    Args:
        n_boxes: Number of spatial boxes to populate.
        n_particles: Number of particles per box.
        n_species: Number of species carried by each particle.

    Returns:
        ParticleData with mixed NPF/droplet radii, active concentrations, and
        deterministic ``np.float64`` masses for coagulation benchmarks only.
    """
    density = np.linspace(950.0, 1250.0, n_species, dtype=np.float64)
    species_fractions = np.linspace(1.0, 2.0, n_species, dtype=np.float64)
    species_fractions /= np.sum(species_fractions)

    small_count = max(1, n_particles // 2)
    large_count = max(1, n_particles - small_count)

    small_radii = np.geomspace(1.5e-9, 2.5e-8, small_count, dtype=np.float64)
    large_radii = np.geomspace(1.0e-6, 1.5e-5, large_count, dtype=np.float64)
    base_radii = np.concatenate((small_radii, large_radii))[:n_particles]

    box_scale = 1.0 + 0.01 * np.arange(n_boxes, dtype=np.float64)
    radii = box_scale[:, np.newaxis] * base_radii[np.newaxis, :]
    total_volume = (4.0 / 3.0) * np.pi * radii**3
    masses = (
        total_volume[..., np.newaxis]
        * density[np.newaxis, np.newaxis, :]
        * species_fractions[np.newaxis, np.newaxis, :]
    )

    base_concentration = np.concatenate(
        (
            np.linspace(150.0, 220.0, small_count, dtype=np.float64),
            np.linspace(0.8, 1.6, large_count, dtype=np.float64),
        )
    )[:n_particles]
    concentration = (
        1.0 + 0.02 * np.arange(n_boxes, dtype=np.float64)[:, np.newaxis]
    ) * base_concentration[np.newaxis, :]
    charge = np.zeros((n_boxes, n_particles), dtype=np.float64)
    volume = np.full((n_boxes,), 2.0e-6, dtype=np.float64)
    return ParticleData(
        masses=masses,
        concentration=concentration,
        charge=charge,
        density=density,
        volume=volume,
    )


def _make_gas_data(n_boxes: int, n_species: int) -> GasData:
    """Create deterministic gas data for benchmarks.

    Args:
        n_boxes: Number of spatial boxes.
        n_species: Number of condensible species.

    Returns:
        GasData with molar mass, concentration, partitioning, and names.
    """
    molar_mass = np.linspace(0.018, 0.05, n_species, dtype=np.float64)
    concentration = (
        1.0e-6
        * (1.0 + 0.2 * np.arange(n_boxes, dtype=np.float64)[:, np.newaxis])
        * np.ones((1, n_species), dtype=np.float64)
    )
    partitioning = np.ones((n_species,), dtype=bool)
    names = [f"species_{idx}" for idx in range(n_species)]
    return GasData(
        name=names,
        molar_mass=molar_mass,
        concentration=concentration,
        partitioning=partitioning,
    )


def _make_vapor_pressure(n_boxes: int, n_species: int) -> np.ndarray:
    """Create deterministic vapor pressure array.

    Args:
        n_boxes: Number of spatial boxes.
        n_species: Number of condensible species.

    Returns:
        Vapor pressure array shaped (n_boxes, n_species).
    """
    return 800.0 + 50.0 * np.arange(n_boxes, dtype=np.float64)[
        :, np.newaxis
    ] * np.ones(
        (1, n_species),
        dtype=np.float64,
    )


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
    """Compute CPU mass transfer matching GPU kernel physics.

    Args:
        particles: Particle state with masses, density, charge, and volume.
        gas: Gas state with molar masses and concentrations.
        vapor_pressure: Vapor pressure per box and species.
        surface_tension: Surface tension per species.
        mass_accommodation: Mass accommodation coefficient per species.
        diffusion_coefficient_vapor: Vapor diffusion coefficient per species.
        temperature: Gas temperature in kelvin.
        pressure: Gas pressure in pascals.
        time_step: Time step duration in seconds.
        out: Optional buffer to reuse for mass transfer outputs.

    Returns:
        Mass transfer array shaped like particle masses.
    """
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
    """Update particle masses via CPU mass transfer.

    Args:
        particles: Particle state to update in-place.
        gas: Gas state corresponding to each box.
        vapor_pressure: Vapor pressure per box and species.
        surface_tension: Surface tension per species.
        mass_accommodation: Mass accommodation coefficient per species.
        diffusion_coefficient_vapor: Vapor diffusion coefficient per species.
        temperature: Gas temperature in kelvin.
        pressure: Gas pressure in pascals.
        time_step: Time step duration in seconds.
        mass_transfer_buffer: Scratch buffer reused across iterations.
    """
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
    """Build interpolation radii for particle-resolved coagulation.

    Args:
        radii: Particle radii array.

    Returns:
        Interpolation radii grid spanning the observed radius range.
    """
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
    """Update particle masses via particle-resolved CPU coagulation.

    Args:
        particles: Particle state to update in-place.
        temperature: Gas temperature in kelvin.
        pressure: Gas pressure in pascals.
        time_step: Time step duration in seconds.
        rng: Random number generator for collision sampling.
        kernel_radius: Interpolation radii grid for kernel evaluation.
    """
    n_boxes, n_particles, _ = particles.masses.shape
    # Build a (len(kernel_radius), len(kernel_radius)) kernel matrix
    # evaluated on the interpolation grid, not on per-particle radii.
    avg_density = float(np.mean(particles.density))
    grid_volume = (4.0 / 3.0) * np.pi * np.power(kernel_radius, 3)
    grid_mass = grid_volume * avg_density
    kernel = cast(
        NDArray[np.float64],
        np.atleast_2d(
            np.asarray(
                get_brownian_kernel_via_system_state(
                    particle_radius=kernel_radius,
                    particle_mass=grid_mass,
                    temperature=temperature,
                    pressure=pressure,
                ),
                dtype=np.float64,
            )
        ),
    )
    for box_idx in range(n_boxes):
        masses_box = particles.masses[box_idx]
        concentration_box = particles.concentration[box_idx]
        total_mass = np.sum(masses_box, axis=-1)
        total_volume = np.sum(masses_box / particles.density, axis=-1)
        radii = np.cbrt(3.0 * total_volume / (4.0 * np.pi))
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
    """Print timing summary for benchmark output.

    Args:
        label: Description of the benchmark case.
        gpu_time: GPU elapsed time in seconds.
        cpu_time: CPU elapsed time in seconds.
    """
    speedup = cpu_time / gpu_time if gpu_time > 0 else np.nan
    print(
        f"{label}: GPU {gpu_time:.4f}s | CPU {cpu_time:.4f}s | "
        f"speedup {speedup:.2f}x"
    )


def _build_wp_func_benchmark_inputs(
    n_evals: int,
    seed: int = 123,
) -> dict[str, np.ndarray]:
    """Build deterministic NumPy inputs for wp.func timing checks."""
    rng = np.random.default_rng(seed)
    return {
        "temperatures": rng.uniform(280.0, 320.0, size=n_evals).astype(
            np.float64
        ),
        "mobilities": rng.uniform(1.0e-8, 5.0e-8, size=n_evals).astype(
            np.float64
        ),
        "pressure_deltas": rng.uniform(-5.0, 10.0, size=n_evals).astype(
            np.float64
        ),
        "mass_transport": rng.uniform(1.0e-18, 1.0e-16, size=n_evals).astype(
            np.float64
        ),
        "molar_masses": rng.uniform(0.018, 0.05, size=n_evals).astype(
            np.float64
        ),
        "total_volumes": rng.uniform(1.0e-21, 1.0e-18, size=n_evals).astype(
            np.float64
        ),
        "radii_i": rng.uniform(1.0e-9, 1.0e-7, size=n_evals).astype(np.float64),
        "radii_j": rng.uniform(1.0e-9, 1.0e-7, size=n_evals).astype(np.float64),
        "diff_i": rng.uniform(1.0e-10, 1.0e-9, size=n_evals).astype(np.float64),
        "diff_j": rng.uniform(1.0e-10, 1.0e-9, size=n_evals).astype(np.float64),
        "g_i": rng.uniform(1.0e-9, 1.0e-8, size=n_evals).astype(np.float64),
        "g_j": rng.uniform(1.0e-9, 1.0e-8, size=n_evals).astype(np.float64),
        "speed_i": rng.uniform(10.0, 40.0, size=n_evals).astype(np.float64),
        "speed_j": rng.uniform(10.0, 40.0, size=n_evals).astype(np.float64),
    }


def _benchmark_cpu_wp_funcs(
    inputs: dict[str, np.ndarray],
    *,
    kernel_sample: int = 256,
) -> dict[str, float]:
    """Time CPU-side reference calculations for wp.func comparisons."""
    cpu_start = time.perf_counter()
    _ = get_diffusion_coefficient(
        temperature=inputs["temperatures"],
        aerodynamic_mobility=inputs["mobilities"],
        boltzmann_constant=constants.BOLTZMANN_CONSTANT,
    )
    cpu_diffusion_time = time.perf_counter() - cpu_start

    cpu_start = time.perf_counter()
    _ = get_mass_transfer_rate(
        pressure_delta=inputs["pressure_deltas"],
        first_order_mass_transport=inputs["mass_transport"],
        temperature=inputs["temperatures"],
        molar_mass=inputs["molar_masses"],
    )
    cpu_mass_transfer_time = time.perf_counter() - cpu_start

    cpu_start = time.perf_counter()
    _ = np.cbrt(3.0 * inputs["total_volumes"] / (4.0 * np.pi))
    cpu_radius_time = time.perf_counter() - cpu_start

    cpu_start = time.perf_counter()
    _ = (
        constants.BOLTZMANN_CONSTANT
        * inputs["temperatures"]
        * inputs["mobilities"]
    )
    cpu_brownian_diffusivity_time = time.perf_counter() - cpu_start

    cpu_start = time.perf_counter()
    _ = get_brownian_kernel_via_system_state(
        particle_radius=inputs["temperatures"][:kernel_sample] * 0.0 + 1.0e-8,
        particle_mass=np.full(kernel_sample, 1.0e-18, dtype=np.float64),
        temperature=DEFAULT_TEMPERATURE,
        pressure=DEFAULT_PRESSURE,
    )
    cpu_brownian_kernel_time = time.perf_counter() - cpu_start

    return {
        "diffusion_coefficient": cpu_diffusion_time,
        "mass_transfer_rate": cpu_mass_transfer_time,
        "particle_radius_from_volume": cpu_radius_time,
        "brownian_diffusivity": cpu_brownian_diffusivity_time,
        "brownian_kernel_pair": cpu_brownian_kernel_time,
    }


def _benchmark_gpu_wp_funcs(
    inputs: dict[str, np.ndarray],
) -> dict[str, float]:
    """Time GPU-side wp.func kernel wrappers."""
    n_evals = inputs["temperatures"].size
    temperatures_wp: Any = wp.array(
        inputs["temperatures"], dtype=wp.float64, device="cuda"
    )
    mobilities_wp: Any = wp.array(
        inputs["mobilities"], dtype=wp.float64, device="cuda"
    )
    pressure_wp: Any = wp.array(
        inputs["pressure_deltas"], dtype=wp.float64, device="cuda"
    )
    mass_transport_wp: Any = wp.array(
        inputs["mass_transport"],
        dtype=wp.float64,
        device="cuda",
    )
    molar_mass_wp: Any = wp.array(
        inputs["molar_masses"], dtype=wp.float64, device="cuda"
    )
    volumes_wp: Any = wp.array(
        inputs["total_volumes"], dtype=wp.float64, device="cuda"
    )
    radii_i_wp: Any = wp.array(
        inputs["radii_i"], dtype=wp.float64, device="cuda"
    )
    radii_j_wp: Any = wp.array(
        inputs["radii_j"], dtype=wp.float64, device="cuda"
    )
    diff_i_wp: Any = wp.array(inputs["diff_i"], dtype=wp.float64, device="cuda")
    diff_j_wp: Any = wp.array(inputs["diff_j"], dtype=wp.float64, device="cuda")
    g_i_wp: Any = wp.array(inputs["g_i"], dtype=wp.float64, device="cuda")
    g_j_wp: Any = wp.array(inputs["g_j"], dtype=wp.float64, device="cuda")
    speed_i_wp: Any = wp.array(
        inputs["speed_i"], dtype=wp.float64, device="cuda"
    )
    speed_j_wp: Any = wp.array(
        inputs["speed_j"], dtype=wp.float64, device="cuda"
    )

    diffusion_out = wp.zeros(n_evals, dtype=wp.float64, device="cuda")
    transfer_out = wp.zeros(n_evals, dtype=wp.float64, device="cuda")
    radius_out = wp.zeros(n_evals, dtype=wp.float64, device="cuda")
    brownian_out = wp.zeros(n_evals, dtype=wp.float64, device="cuda")

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

    return {
        "diffusion_coefficient": gpu_diffusion_time,
        "mass_transfer_rate": gpu_mass_transfer_time,
        "particle_radius_from_volume": gpu_radius_time,
        "brownian_diffusivity": gpu_brownian_diffusivity_time,
        "brownian_kernel_pair": gpu_brownian_kernel_time,
    }


@pytest.mark.parametrize(
    "label,n_boxes,n_particles,n_species,run_cpu",
    CONDENSATION_CONFIGS,
    ids=[c[0] for c in CONDENSATION_CONFIGS],
)
def test_condensation_scaling(
    label: str,
    n_boxes: int,
    n_particles: int,
    n_species: int,
    run_cpu: bool,
) -> None:
    """Parametrized condensation benchmark across particle counts."""
    _skip_if_no_cuda()
    tag = f"cond-{label}"
    print(
        f"\n  [{tag}] Setup: {n_boxes} box(es) x "
        f"{n_particles:,} particles x {n_species} species"
    )
    _preflight_condensation_case_allocations(
        label,
        n_boxes,
        n_particles,
        n_species,
    )
    _validate_benchmark_budget(
        _estimate_condensation_budget(
            label, n_boxes, n_particles, n_species, run_cpu
        )
    )
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
        gas, device="cuda", vapor_pressure=vapor_pressure
    )
    mass_transfer_buffer = _wp_zeros_with_guard(
        (n_boxes, n_particles, n_species),
        dtype=wp.float64,
        device="cuda",
        label=f"{tag} gpu mass transfer buffer",
    )
    surface_tension_wp: Any = wp.array(
        surface_tension, dtype=wp.float64, device="cuda"
    )
    mass_accommodation_wp: Any = wp.array(
        mass_accommodation, dtype=wp.float64, device="cuda"
    )
    diffusion_vapor_wp: Any = wp.array(
        diffusion_vapor, dtype=wp.float64, device="cuda"
    )
    thermodynamics = ThermodynamicsConfig(
        modes=_wp_zeros_with_guard(
            (n_species,),
            dtype=wp.int32,
            device="cuda",
            label=f"{tag} thermodynamics modes",
        ),
        parameters=_wp_zeros_with_guard(
            (n_species, 4),
            dtype=wp.float64,
            device="cuda",
            label=f"{tag} thermodynamics parameters",
        ),
        molar_mass_reference=wp.array(
            gas.molar_mass, dtype=wp.float64, device="cuda"
        ),
    )

    def gpu_step() -> None:
        """Execute one GPU condensation step for the timed loop."""
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
            thermodynamics=thermodynamics,
        )

    print(
        f"  [{tag}] Running GPU: {DEFAULT_WARMUP} warmup + "
        f"{DEFAULT_STEPS} timed steps ..."
    )
    with _warp_profiled(f"condensation_{label}"):
        gpu_time = _time_gpu_loop(gpu_step, DEFAULT_STEPS, DEFAULT_WARMUP)
    print(f"  [{tag}] GPU done: {gpu_time:.4f}s")

    entry_key = f"condensation_{label}"
    entry = _benchmark_results["benchmarks"].setdefault(
        entry_key,
        {
            "n_boxes": n_boxes,
            "n_particles": n_particles,
            "n_species": n_species,
            "steps": DEFAULT_STEPS,
        },
    )
    entry["gpu_time_s"] = gpu_time
    _save_results()

    if run_cpu:
        cpu_particles = particles.copy()
        cpu_mass_transfer = np.zeros_like(cpu_particles.masses)

        def cpu_step() -> None:
            """Execute one CPU condensation step for the timed loop."""
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

        print(
            f"  [{tag}] Running CPU: {DEFAULT_WARMUP} warmup + "
            f"{DEFAULT_STEPS} timed steps ..."
        )
        cpu_time = _time_cpu_loop(cpu_step, DEFAULT_STEPS, DEFAULT_WARMUP)
        print(f"  [{tag}] CPU done: {cpu_time:.4f}s")
        _print_timing(f"Condensation {label}", gpu_time, cpu_time)
        speedup = _compute_speedup(cpu_time, gpu_time)
        entry["cpu_time_s"] = cpu_time
        entry["speedup"] = speedup
    else:
        total_particles = n_boxes * n_particles
        per_step = gpu_time / DEFAULT_STEPS
        print(
            f"  [{tag}] GPU-only | {total_particles:,} total particles | "
            f"{per_step:.4f}s/step"
        )
    _save_results()


@pytest.mark.parametrize(
    "label,n_boxes,n_particles,n_species,run_cpu",
    COAGULATION_CONFIGS,
    ids=[c[0] for c in COAGULATION_CONFIGS],
)
def test_coagulation_scaling(
    label: str,
    n_boxes: int,
    n_particles: int,
    n_species: int,
    run_cpu: bool,
) -> None:
    """Benchmark coagulation scaling with the mixed-scale fixture path."""
    _skip_if_no_cuda()
    tag = f"coag-{label}"
    print(
        f"\n  [{tag}] Setup: {n_boxes} box(es) x "
        f"{n_particles:,} particles x {n_species} species"
    )
    _preflight_coagulation_case_allocations(
        label,
        n_boxes,
        n_particles,
        n_species,
    )
    _validate_benchmark_budget(
        _estimate_coagulation_budget(
            label, n_boxes, n_particles, n_species, run_cpu
        )
    )
    particles = _make_coagulation_particle_data(n_boxes, n_particles, n_species)

    gpu_particles = to_warp_particle_data(particles, device="cuda")
    collision_pairs_buf = _wp_zeros_with_guard(
        (n_boxes, MAX_COLLISIONS, 2),
        dtype=wp.int32,
        device="cuda",
        label=f"{tag} collision pairs",
        itemsize=4,
    )
    n_collisions_buf = _wp_zeros_with_guard(
        (n_boxes,),
        dtype=wp.int32,
        device="cuda",
        label=f"{tag} collision counts",
        itemsize=4,
    )
    rng_states_buf = _wp_zeros_with_guard(
        (n_boxes,),
        dtype=wp.uint32,
        device="cuda",
        label=f"{tag} RNG state",
        itemsize=4,
    )
    _seed_coagulation_rng_states_once(
        rng_seed=42,
        rng_states=rng_states_buf,
        n_boxes=n_boxes,
        device="cuda",
    )

    def gpu_step() -> None:
        """Execute one GPU coagulation step for the timed loop."""
        coagulation_step_gpu(
            gpu_particles,
            temperature=DEFAULT_TEMPERATURE,
            pressure=DEFAULT_PRESSURE,
            time_step=DEFAULT_TIME_STEP,
            # Keep the seed fixed when reusing caller-owned RNG state so the
            # benchmark matches persisted-state runtime semantics.
            rng_seed=42,
            max_collisions=MAX_COLLISIONS,
            collision_pairs=collision_pairs_buf,
            n_collisions=n_collisions_buf,
            rng_states=rng_states_buf,
            initialize_rng=False,
        )

    print(
        f"  [{tag}] Running GPU: {DEFAULT_WARMUP} warmup + "
        f"{DEFAULT_STEPS} timed steps ..."
    )
    with _warp_profiled(f"coagulation_{label}"):
        gpu_time = _time_gpu_loop(gpu_step, DEFAULT_STEPS, DEFAULT_WARMUP)
    print(f"  [{tag}] GPU done: {gpu_time:.4f}s")

    entry_key = f"coagulation_{label}"
    entry = _benchmark_results["benchmarks"].setdefault(
        entry_key,
        {
            "n_boxes": n_boxes,
            "n_particles": n_particles,
            "n_species": n_species,
            "steps": DEFAULT_STEPS,
        },
    )
    entry["gpu_time_s"] = gpu_time
    _save_results()

    if run_cpu:
        cpu_particles = particles.copy()
        rng = np.random.default_rng(42)
        kernel_radius = _build_kernel_radius(cpu_particles.radii)

        def cpu_step() -> None:
            """Execute one CPU coagulation step for the timed loop."""
            _cpu_coagulation_step(
                cpu_particles,
                DEFAULT_TEMPERATURE,
                DEFAULT_PRESSURE,
                DEFAULT_TIME_STEP,
                rng,
                kernel_radius,
            )

        print(
            f"  [{tag}] Running CPU: {DEFAULT_WARMUP} warmup + "
            f"{DEFAULT_STEPS} timed steps ..."
        )
        cpu_time = _time_cpu_loop(cpu_step, DEFAULT_STEPS, DEFAULT_WARMUP)
        print(f"  [{tag}] CPU done: {cpu_time:.4f}s")
        _print_timing(f"Coagulation {label}", gpu_time, cpu_time)
        speedup = _compute_speedup(cpu_time, gpu_time)
        entry["cpu_time_s"] = cpu_time
        entry["speedup"] = speedup
    else:
        total_particles = n_boxes * n_particles
        per_step = gpu_time / DEFAULT_STEPS
        print(
            f"  [{tag}] GPU-only | {total_particles:,} total particles | "
            f"{per_step:.4f}s/step"
        )
    _save_results()


if wp is not None:

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
    kernel_sample = 256
    inputs = _build_wp_func_benchmark_inputs(n_evals)
    cpu_timings = _benchmark_cpu_wp_funcs(inputs, kernel_sample=kernel_sample)
    gpu_timings = _benchmark_gpu_wp_funcs(inputs)

    kernel_calls = kernel_sample * kernel_sample
    timing_lines = [
        "@wp.func timings (per call, microseconds):",
        (
            "  diffusion_coefficient_wp: "
            f"CPU {cpu_timings['diffusion_coefficient'] / n_evals * 1e6:.4f} | "
            f"GPU {gpu_timings['diffusion_coefficient'] / n_evals * 1e6:.4f}"
        ),
        (
            "  mass_transfer_rate_wp: "
            f"CPU {cpu_timings['mass_transfer_rate'] / n_evals * 1e6:.4f} | "
            f"GPU {gpu_timings['mass_transfer_rate'] / n_evals * 1e6:.4f}"
        ),
        (
            "  particle_radius_from_volume_wp: "
            f"CPU {cpu_timings['particle_radius_from_volume'] / n_evals * 1e6:.4f} | "
            f"GPU {gpu_timings['particle_radius_from_volume'] / n_evals * 1e6:.4f}"
        ),
        (
            "  brownian_diffusivity_wp: "
            f"CPU {cpu_timings['brownian_diffusivity'] / n_evals * 1e6:.4f} | "
            f"GPU {gpu_timings['brownian_diffusivity'] / n_evals * 1e6:.4f}"
        ),
        (
            "  brownian_kernel_pair_wp: "
            f"CPU {cpu_timings['brownian_kernel_pair'] / kernel_calls * 1e6:.4f} | "
            f"GPU {gpu_timings['brownian_kernel_pair'] / n_evals * 1e6:.4f}"
        ),
    ]
    print("\n".join(timing_lines))

    _benchmark_results["benchmarks"]["wp_func"] = {
        "n_evals": n_evals,
        "diffusion_coefficient": {
            "cpu_us": cpu_timings["diffusion_coefficient"] / n_evals * 1e6,
            "gpu_us": gpu_timings["diffusion_coefficient"] / n_evals * 1e6,
        },
        "mass_transfer_rate": {
            "cpu_us": cpu_timings["mass_transfer_rate"] / n_evals * 1e6,
            "gpu_us": gpu_timings["mass_transfer_rate"] / n_evals * 1e6,
        },
        "particle_radius_from_volume": {
            "cpu_us": cpu_timings["particle_radius_from_volume"]
            / n_evals
            * 1e6,
            "gpu_us": gpu_timings["particle_radius_from_volume"]
            / n_evals
            * 1e6,
        },
        "brownian_diffusivity": {
            "cpu_us": cpu_timings["brownian_diffusivity"] / n_evals * 1e6,
            "gpu_us": gpu_timings["brownian_diffusivity"] / n_evals * 1e6,
        },
        "brownian_kernel_pair": {
            "cpu_us": cpu_timings["brownian_kernel_pair"] / kernel_calls * 1e6,
            "gpu_us": gpu_timings["brownian_kernel_pair"] / n_evals * 1e6,
        },
    }
    _save_results()


@pytest.mark.parametrize(
    ("label", "case_index", "candidate_id"),
    _MASS_PRECISION_BENCHMARK_CONFIGS,
)
def test_mass_precision_projection_benchmark(
    label: str,
    case_index: int,
    candidate_id: str,
) -> None:
    """Record optional bounded projection timings for P3 study candidates."""
    case = _build_mass_precision_cases()[case_index]
    start = time.perf_counter()
    reconstructed = case.masses
    repeats = 5_000
    for _ in range(repeats):
        reconstructed = _project_candidate(
            case,
            candidate_id,
        )["reconstructed_masses"]
    elapsed = time.perf_counter() - start

    assert reconstructed.shape == case.masses.shape
    entry_key = f"mass_precision_candidate_payload_{label}_{candidate_id}"
    _benchmark_results["benchmarks"][entry_key] = {
        "case_name": case.case_name,
        "candidate_id": candidate_id,
        "repeats": repeats,
        "elapsed_s": elapsed,
        "mean_us": elapsed / repeats * 1e6,
        "n_boxes": case.masses.shape[0],
        "n_particles": case.masses.shape[1],
        "n_species": case.masses.shape[2],
    }
    _save_results()


def _estimate_resident_requested_bytes(case: Any) -> int:
    """Return the configured conservative estimate for one exact row."""
    try:
        return RESIDENT_BENCHMARK_REQUESTED_BYTES_BY_SHAPE[case.requested_shape]
    except KeyError as error:
        raise ValueError(
            "resident benchmark case has no configured requested estimate."
        ) from error


def _profiling_machine(binding: Any) -> MachineProvenance:
    """Build bounded provenance from an already-qualified CUDA binding."""
    device = binding.selected_device
    identity = device.get("identity")
    if device.get("status") != "available" or not isinstance(identity, str):
        raise ValueError("qualified profiling binding lacks device provenance.")
    return MachineProvenance(
        machine_id=platform.node() or "unknown",
        platform=sys.platform,
        python_version=platform.python_version(),
        cuda_version=str(getattr(wp, "__version__", "unknown")),
        driver_version="qualified",
        device=identity,
        source_revision="working-tree",
    )


def _profiling_method(source: str) -> MeasurementMethod:
    """Build one P1 measurement-method record."""
    return MeasurementMethod(
        method_id=source,
        source=source,
        command=" ".join(sys.argv) or "pytest",
        version="perf_counter_ns",
        duration_unit="ns",
    )


def _positive_duration_ns(start: int, end: int) -> int:
    """Require a strictly positive integer nanosecond timing delta."""
    if type(start) is not int or type(end) is not int or end <= start:
        raise ValueError("benchmark clock must produce increasing integer ns.")
    return end - start


def _validate_profiling_binding(binding: Any, workload: Any) -> None:
    """Reject a fixture that is not the exact frozen workload binding."""
    binding.validate_identities()
    dimensions = binding.loop.prepared.signature.dimensions
    if (
        dimensions.n_boxes,
        dimensions.n_particles,
        dimensions.n_species,
    ) != workload.shape or binding.duration != workload.duration_seconds:
        raise ValueError("profiling fixture does not match frozen workload.")
    if not callable(binding.reset) or not callable(binding.synchronize):
        raise TypeError("profiling fixture callbacks are invalid.")


def _collect_profile_row(
    binding: Any,
    workload: Any,
    *,
    mode: str,
    source: str,
    clock: Any = time.perf_counter_ns,
) -> tuple[RawDurationSample, ...]:
    """Collect one replay-count-major method row with reset outside clocks."""
    operation = (
        binding.enqueue if mode == "prepared_uncaptured" else binding.replay
    )
    if not callable(operation) or source not in {
        "host_launch",
        "synchronized_elapsed",
    }:
        raise ValueError("profiling mode or method is invalid.")
    for _ in range(workload.warmup):
        binding.reset()
        binding.validate_identities()
        operation()
    binding.synchronize()
    samples: list[RawDurationSample] = []
    for replay_count in workload.replay_counts:
        if replay_count not in REPLAY_COUNTS:
            raise ValueError("profiling replay count is invalid.")
        for _ in range(workload.sample_count):
            binding.reset()
            binding.validate_identities()
            start = clock()
            for _ in range(replay_count):
                operation()
            if source == "synchronized_elapsed":
                binding.synchronize()
            end = clock()
            samples.append(
                RawDurationSample(
                    replay_count, _positive_duration_ns(start, end)
                )
            )
            if source == "host_launch":
                binding.synchronize()
    return tuple(samples)


def _stage_profile_artifacts(
    profiling_root: Path,
    artifacts: dict[tuple[str, str], ProfilingArtifact],
) -> list[tuple[Path, Path]]:
    """Serialize and stage every profiling artifact and its manifest."""
    serialized = {
        key: serialize_profiling_artifact(value) + "\n"
        for key, value in artifacts.items()
    }
    manifest = (
        json.dumps(
            {
                f"{mode}/{method}": PROFILING_ARTIFACT_NAMES[(mode, method)]
                for mode, method in PROFILING_ARTIFACT_NAMES
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    staged: list[tuple[Path, Path]] = []
    for key, payload in serialized.items():
        final = profiling_root / PROFILING_ARTIFACT_NAMES[key]
        temporary = final.with_suffix(final.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        staged.append((temporary, final))
    manifest_final = profiling_root / "manifest.json"
    manifest_temporary = manifest_final.with_suffix(".json.tmp")
    manifest_temporary.write_text(manifest, encoding="utf-8")
    staged.append((manifest_temporary, manifest_final))
    return staged


def _restore_profile_artifacts(
    staged: list[tuple[Path, Path]],
    backups: list[tuple[Path, Path]],
    published: list[Path],
) -> None:
    """Remove partial output and restore the previous complete publication."""
    for final in published:
        final.unlink(missing_ok=True)
    for backup, final in backups:
        if backup.exists():
            os.replace(backup, final)
    for temporary, _ in staged:
        temporary.unlink(missing_ok=True)


def _write_profile_artifacts(
    artifact_root: Path,
    artifacts: dict[tuple[str, str], ProfilingArtifact],
) -> None:
    """Publish the complete profiling association or restore prior output."""
    artifact_root.mkdir(parents=True, exist_ok=True)
    profiling_root = artifact_root / "benchmarks" / "profiling"
    profiling_root.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        staged = _stage_profile_artifacts(profiling_root, artifacts)
        for _, final in staged:
            if final.exists():
                backup = final.with_suffix(final.suffix + ".bak")
                os.replace(final, backup)
                backups.append((backup, final))
        for temporary, final in staged:
            os.replace(temporary, final)
            published.append(final)
    except BaseException:
        _restore_profile_artifacts(staged, backups, published)
        raise
    for backup, _ in backups:
        backup.unlink(missing_ok=True)


def _collect_resident_launch_profile_artifacts(
    artifact_root: Path = Path(".artifacts"),
    *,
    clock: Any = time.perf_counter_ns,
) -> dict[tuple[str, str], ProfilingArtifact]:
    """Collect native-CUDA-only launch evidence or complete unavailable rows."""
    artifact_root.mkdir(parents=True, exist_ok=True)
    workloads = build_default_profiling_workload_matrix()
    pairs = tuple(PROFILING_ARTIFACT_NAMES)
    try:
        availability = cuda_capture_availability()
        if not availability.available:
            raise ResidentBenchmarkUnavailableError(availability.reason)
        rows: dict[tuple[str, str], list[Any]] = {pair: [] for pair in pairs}
        for workload in workloads:
            with qualified_cuda_resident_benchmark(
                duration=workload.duration_seconds,
                n_boxes=workload.shape[0],
                n_particles=workload.shape[1],
                n_species=workload.shape[2],
                root_seed=workload.seed,
                case_id=workload.workload_id,
                availability=availability,
            ) as binding:
                _validate_profiling_binding(binding, workload)
                try:
                    binding.reset()
                    binding.validate_identities()
                except BaseException as error:
                    raise ResidentBenchmarkUnavailableError(
                        "identity-preserving fixture reset is unavailable: "
                        f"{error}"
                    ) from error
                machine = _profiling_machine(binding)
                raw_root = ensure_profiling_raw_root(artifact_root)
                for mode, source in pairs:
                    samples = _collect_profile_row(
                        binding, workload, mode=mode, source=source, clock=clock
                    )
                    raw_filename = f"{workload.label}_{mode}_{source}.json"
                    (raw_root / raw_filename).write_text(
                        json.dumps(
                            {
                                "capture_elapsed_seconds": binding.capture_elapsed_seconds,
                                "mode": mode,
                                "raw_samples": [
                                    {
                                        "duration_ns": sample.duration_ns,
                                        "replay_count": sample.replay_count,
                                    }
                                    for sample in samples
                                ],
                                "setup_elapsed_seconds": binding.setup_elapsed_seconds,
                                "workload_id": workload.workload_id,
                            },
                            sort_keys=True,
                        ),
                        encoding="utf-8",
                    )
                    provenance = build_raw_report_provenance(
                        artifact_root, raw_filename
                    )
                    metric_name = f"{source}_duration"
                    metric = NormalizedMetric(
                        metric_name,
                        sum(
                            sample.duration_ns / sample.replay_count
                            for sample in samples
                        )
                        / len(samples),
                        "ns",
                    )
                    rows[(mode, source)].append(
                        ExecutedEvidence(
                            "executed",
                            workload,
                            machine,
                            _profiling_method(source),
                            samples,
                            (metric,),
                            (provenance,),
                        )
                    )
        artifacts = {
            key: ProfilingArtifact(tuple(value)) for key, value in rows.items()
        }
    except (ResidentBenchmarkUnavailableError, pytest.skip.Exception) as error:
        artifacts = {
            pair: ProfilingArtifact(
                tuple(
                    UnavailableEvidence("unavailable", workload, str(error))
                    for workload in workloads
                )
            )
            for pair in pairs
        }
    _write_profile_artifacts(artifact_root, artifacts)
    return artifacts


@pytest.mark.warp
@pytest.mark.cuda
def test_resident_launch_profile_evidence() -> None:
    """Publish separate native-CUDA launch and completion evidence artifacts."""
    artifacts = _collect_resident_launch_profile_artifacts()
    assert set(artifacts) == set(PROFILING_ARTIFACT_NAMES)


def _collect_resident_capture_matrix() -> ResidentBenchmarkArtifact:  # noqa: C901
    """Collect the exact P3 matrix, with no downscale, fallback, or partial write."""
    cases = build_default_resident_benchmark_matrix()
    results: list[ResidentBenchmarkResult] = []
    memory_observations: list[ResidentMemoryObservation] = []
    availability_result = None

    def availability():
        """Lazily cache the CUDA capture availability result."""
        nonlocal availability_result
        if availability_result is None:
            availability_result = cuda_capture_availability()
        return availability_result

    case_provenance: dict[str, dict[str, Any]] = {}
    executed_devices: list[dict[str, Any]] = []

    for case in cases:
        preflight = preflight_resident_benchmark_case(
            case,
            budget_bytes=_parse_positive_int_env(
                "BENCHMARK_MAX_BYTES", DEFAULT_BENCHMARK_MAX_BYTES
            ),
            estimate_requested_bytes=_estimate_resident_requested_bytes,
            availability=availability,
        )
        if preflight.status is not ResidentBenchmarkStatus.EXECUTED:
            case_provenance[case.case_id] = {
                "status": preflight.status.value,
                "duration_seconds": case.duration,
            }
            results.append(
                ResidentBenchmarkResult(
                    case_id=case.case_id,
                    timing_mode=None,
                    requested_shape=case.requested_shape,
                    status=preflight.status,
                    reason=preflight.reason,
                    samples=(),
                    summary=None,
                    provenance={
                        "binding": "preconstruction",
                        "case_id": case.case_id,
                        "duration_seconds": case.duration,
                    },
                )
            )
            continue
        try:
            context = qualified_cuda_resident_benchmark(
                duration=case.duration,
                n_boxes=case.requested_shape[0],
                n_particles=case.requested_shape[1],
                n_species=case.requested_shape[2],
                root_seed=case.seed,
                case_id=case.case_id,
                availability=availability(),
            )
            with context as binding:
                binding.validate_identities()
                dimensions = binding.loop.prepared.signature.dimensions
                if (
                    dimensions.n_boxes,
                    dimensions.n_particles,
                    dimensions.n_species,
                ) != case.requested_shape:
                    raise ValueError(
                        "resident benchmark fixture dimensions do not match "
                        "the exact requested shape."
                    )
                prepared_signature_digest, selected_device = (
                    resident_benchmark_provenance(binding)
                )
                case_provenance[case.case_id] = {
                    "status": "executed",
                    "prepared_signature_digest": prepared_signature_digest,
                    "device": selected_device,
                    "duration_seconds": case.duration,
                }
                executed_devices.append(selected_device)
                active_slots = case.active_fraction * dimensions.n_particles
                if not active_slots.is_integer():
                    raise ValueError(
                        "resident benchmark active slots must be integral."
                    )
                diagnostics = tuple(
                    {
                        "gas": "gas_concentration_snapshot",
                        "saturation": "saturation_ratio_snapshot",
                    }[name]
                    for name in case.diagnostics
                )
                model = build_resident_memory_model(
                    n_boxes=dimensions.n_boxes,
                    n_particles=dimensions.n_particles,
                    n_species=dimensions.n_species,
                    active_slots_per_box=int(active_slots),
                    registry_logical_byte_count=binding.capture_set.report.logical_byte_count,
                    diagnostics=diagnostics,
                    communication=case.communication,
                    checkpoint_sidecar_copy_bytes=0,
                    checkpoint_inspection_copy_bytes=0,
                )
                uncaptured, replay = collect_paired_device_timings(
                    uncaptured_operation=binding.enqueue,
                    replay_operation=binding.replay,
                    synchronize=binding.synchronize,
                    clock=time.perf_counter,
                    warmup_count=case.warmup,
                    sample_count=case.timestep_count,
                )
        except (
            ResidentBenchmarkUnavailableError,
            pytest.skip.Exception,
        ) as error:
            case_provenance[case.case_id] = {
                "status": "unavailable",
                "duration_seconds": case.duration,
                "reason": str(error),
            }
            results.append(
                ResidentBenchmarkResult(
                    case_id=case.case_id,
                    timing_mode=None,
                    requested_shape=case.requested_shape,
                    status=ResidentBenchmarkStatus.UNAVAILABLE,
                    reason=str(error),
                    samples=(),
                    summary=None,
                    provenance={
                        "binding": "preconstruction",
                        "case_id": case.case_id,
                        "duration_seconds": case.duration,
                    },
                )
            )
            continue
        monitor = getattr(binding, "memory_monitor", None)
        if monitor is None:
            observation = ResidentMemoryObservation(
                case.case_id,
                False,
                "CUDA allocator monitor unavailable",
                "cuda_runtime.default_pool.used_mem_high.v1",
                {"coverage_complete": False},
                {"warp_device": "unknown"},
                None,
                None,
                None,
                None,
            )
        else:
            observation = monitor.finalize()
        memory_observations.append(
            build_resident_memory_comparison(observation, model)
        )
        provenance = {
            "binding": "native_cuda_capture",
            "case_id": case.case_id,
            "duration_seconds": case.duration,
            "prepared_signature_digest": prepared_signature_digest,
            "device": selected_device,
            "processes": case.processes,
            "sampling_order_policy": "balanced_ab_ba",
        }
        for timing_mode, samples in (
            ("prepared_uncaptured_device_synchronized", uncaptured),
            ("captured_replay_device_synchronized", replay),
        ):
            results.append(
                ResidentBenchmarkResult(
                    case_id=case.case_id,
                    timing_mode=timing_mode,
                    samples=samples,
                    requested_shape=case.requested_shape,
                    status=ResidentBenchmarkStatus.EXECUTED,
                    reason=None,
                    provenance=provenance,
                    setup_elapsed_seconds=binding.setup_elapsed_seconds,
                    capture_elapsed_seconds=binding.capture_elapsed_seconds,
                    summary=summarize_timing_samples(samples),
                )
            )
    digest = build_resident_case_provenance_digest(case_provenance)
    device = {"status": "unavailable", "identity": None, "memory": None}
    if executed_devices and all(
        candidate == executed_devices[0] for candidate in executed_devices
    ):
        device = executed_devices[0]
    return ResidentBenchmarkArtifact(
        metadata=build_resident_benchmark_metadata(
            timestamp_utc=datetime.now(timezone.utc),
            command=" ".join(sys.argv),
            synchronization_method="warp.synchronize",
            warmup=cases[0].warmup,
            timestep_count=cases[0].timestep_count,
            seed=cases[0].seed,
            prepared_signature_digest=digest,
            warp_version={
                "status": "available" if wp is not None else "unavailable",
                "value": str(wp.__version__) if wp is not None else None,
            },
            device=device,
        ),
        cases=cases,
        results=tuple(results),
        memory_observations=tuple(memory_observations),
    )


@pytest.mark.warp
@pytest.mark.cuda
def test_resident_scaling_memory_captured_replay_comparison() -> None:
    """Publish one aggregate opt-in matrix artifact after every row completes."""
    artifact = _collect_resident_capture_matrix()
    root = Path(".artifacts")
    root.mkdir(exist_ok=True)
    destination = write_resident_capture_comparison_artifact(root, artifact)
    print(f"resident capture comparison: {destination}")
    for result in artifact.results:
        print(f"{result.timing_mode}: {result.summary}")
