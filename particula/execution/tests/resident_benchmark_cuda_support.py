"""Lazy CUDA-only support for resident capture timing evidence.

Importing this test-only module neither imports Warp nor probes CUDA. Builder
calls qualify one exact prepared/captured binding and release graph provenance
before closing the resident session; timing callers own synchronization.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from typing import Any, Callable, Iterator

from particula.execution.tests.resident_benchmark_support import (
    CUDA_USED_MEM_HIGH_METHOD,
    CudaDefaultPoolHighWater,
    ResidentBenchmarkAvailability,
    ResidentMemoryObservation,
)


class ResidentBenchmarkUnavailableError(RuntimeError):
    """Report a legal preconstruction CUDA/native-capture absence."""


@dataclass(frozen=True, slots=True)
class ResidentCaptureBenchmarkBinding:
    """Retain one qualified CUDA binding and provenance-only setup timings.

    Attributes:
        loop: Prepared resident loop whose identities must remain unchanged.
        captured: Native captured graph record used for replay.
        duration: Frozen physical timestep duration in seconds.
        setup_elapsed_seconds: One-time setup duration, excluded from samples.
        capture_elapsed_seconds: One-time capture duration, excluded from
            samples.
        synchronize: Device-completion callback owned by the benchmark caller.
        enqueue_operation: Prepared uncaptured operation callback.
        replay_operation: Captured replay operation callback.
        capture_set: Published capture-resource set retained by identity.
    """

    loop: Any
    captured: Any
    duration: float
    setup_elapsed_seconds: float
    capture_elapsed_seconds: float
    synchronize: Callable[[], None]
    enqueue_operation: Callable[[], None]
    replay_operation: Callable[[], None]
    capture_set: Any
    prepared_signature_digest: str
    selected_device: dict[str, Any]
    memory_monitor: Any = None

    def enqueue(self) -> None:
        """Enqueue one prepared uncaptured timestep without synchronization."""
        self.enqueue_operation()

    def replay(self) -> None:
        """Replay one captured timestep without lifecycle work or sync."""
        self.replay_operation()

    def validate_identities(self) -> None:
        """Reject binding drift before any timing callback can run.

        Raises:
            ValueError: If the prepared loop no longer owns the exact request,
                resident resources, guard, or published capture set retained
                during qualification.
        """
        binding = self.loop.binding
        if (
            binding.request is not self.loop.request
            or binding.session is not self.loop.session
            or binding.registry is not self.loop.registry
            or binding.guard is not self.loop.guard
            or self.loop.request.capture_resource_requirements is None
            or self.capture_set is None
        ):
            raise ValueError("resident benchmark binding identity drifted.")


@dataclass(slots=True)
class CudaFixtureMemoryMonitor:
    """Measure one exact CUDA fixture outside its timing collector."""

    case_id: str
    native: object
    device_ordinal: int
    adapter: Any
    synchronize: Callable[[], None]
    sentinel_allocate: Callable[[], Any]
    before: int | None = None
    peak: int | None = None
    reason: str | None = None

    def _unavailable(self, reason: object) -> None:
        """Mark this monitor unavailable with a deterministic reason.

        Args:
            reason: Failure value converted to the structured reason text.
        """
        self.reason = str(reason) or "incomplete CUDA allocator coverage"

    def begin(self) -> None:
        """Prove pool coverage and take the pre-allocation snapshot."""
        if self.reason is not None:
            return
        try:
            self.synchronize()
            self.adapter.reset(self.device_ordinal)
            self.synchronize()
            sentinel_before = self.adapter.read(self.device_ordinal)
            sentinel = self.sentinel_allocate()
            self.synchronize()
            sentinel_peak = self.adapter.read(self.device_ordinal)
            del sentinel
            self.synchronize()
            if sentinel_peak <= sentinel_before:
                raise RuntimeError(
                    "default-pool sentinel did not change UsedMemHigh"
                )
            self.adapter.reset(self.device_ordinal)
            self.synchronize()
            self.before = self.adapter.read(self.device_ordinal)
        except Exception as error:  # monitor failure must not block timing
            self._unavailable(error)

    def snapshot_peak(self) -> None:
        """Read the high-water mark after build, prepare, and capture."""
        if self.reason is not None:
            return
        try:
            self.synchronize()
            self.peak = self.adapter.read(self.device_ordinal)
        except Exception as error:
            self._unavailable(error)

    def finalize(self) -> ResidentMemoryObservation:
        """Read after cleanup and return scalar evidence or unavailability."""
        coverage = {
            "began_before_fixture_allocation": True,
            "default_pool_exact_device": str(self.native),
            "ended_after_binding_cleanup": True,
            "coverage_complete": self.reason is None,
        }
        version: dict[str, Any] = {"warp_device": str(self.native)}
        try:
            version.update(dict(self.adapter.metadata))
            self.synchronize()
            after = self.adapter.read(self.device_ordinal)
            if (
                self.reason is not None
                or self.before is None
                or self.peak is None
            ):
                raise RuntimeError(
                    self.reason or "incomplete CUDA allocator coverage"
                )
            return ResidentMemoryObservation(
                self.case_id,
                True,
                None,
                CUDA_USED_MEM_HIGH_METHOD,
                coverage,
                version,
                self.before,
                self.peak,
                after,
                self.peak - self.before,
            )
        except Exception as error:
            coverage["coverage_complete"] = False
            return ResidentMemoryObservation(
                self.case_id,
                False,
                str(error) or "incomplete CUDA allocator coverage",
                CUDA_USED_MEM_HIGH_METHOD,
                coverage,
                version,
                None,
                None,
                None,
                None,
            )


def _cuda_device_ordinal(native: object) -> int:
    """Return the exact CUDA ordinal accepted by the CUDA Runtime adapter."""
    value = str(native)
    if not value.startswith("cuda:") or not value[5:].isdigit():
        raise ValueError("selected native device is not an exact CUDA ordinal.")
    return int(value[5:])


def _build_memory_monitor(
    *,
    case_id: str,
    wp: Any,
    native: object,
    adapter_factory: Any = CudaDefaultPoolHighWater,
) -> CudaFixtureMemoryMonitor:
    """Build an uncached per-fixture monitor with a bounded Warp sentinel."""
    reason: str | None
    try:
        ordinal = _cuda_device_ordinal(native)
    except ValueError as error:
        ordinal = 0
        reason = str(error)
    else:
        reason = None
    monitor = CudaFixtureMemoryMonitor(
        case_id=case_id,
        native=native,
        device_ordinal=ordinal,
        adapter=adapter_factory(),
        synchronize=wp.synchronize,
        sentinel_allocate=lambda: wp.zeros(1, dtype=wp.uint8, device=native),
    )
    if reason is not None:
        monitor._unavailable(reason)
    return monitor


def _prepared_signature_digest(loop: Any) -> str:
    """Return a stable digest for the actual retained prepared signature."""
    signature = loop.prepared.signature
    device = signature.device
    dimensions = signature.dimensions
    payload = {
        "device_backend": str(device.backend),
        "device_native": str(device.native),
        "dimensions": {
            name: getattr(dimensions, name)
            for name in ("n_boxes", "n_particles", "n_species")
        },
        "signature_type": (
            f"{type(signature).__module__}.{type(signature).__qualname__}"
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _selected_device_metadata(wp: Any, native: object) -> dict[str, Any]:
    """Return identity and memory for the selected native CUDA device."""
    selected = wp.get_device(native)
    total_memory = getattr(selected, "total_memory", None)
    if (
        isinstance(total_memory, bool)
        or not isinstance(total_memory, int | float)
        or not math.isfinite(float(total_memory))
        or total_memory < 0
    ):
        return {"status": "unavailable", "identity": None, "memory": None}
    metadata: dict[str, Any] = {
        "status": "available",
        "identity": str(getattr(selected, "alias", native)),
    }
    metadata["memory"] = int(total_memory)
    return metadata


def resident_benchmark_provenance(
    binding: ResidentCaptureBenchmarkBinding,
) -> tuple[str, dict[str, Any]]:
    """Return fixture-authoritative signature and device provenance."""
    return binding.prepared_signature_digest, binding.selected_device


def cuda_capture_availability() -> ResidentBenchmarkAvailability:
    """Probe P2's CUDA/capture gate and normalize only its absence result."""
    import pytest

    from particula.execution.tests.captured_full_loop_test import (
        _require_native_cuda_capture,
    )

    try:
        _require_native_cuda_capture()
    except pytest.skip.Exception as error:
        return ResidentBenchmarkAvailability(False, str(error))
    return ResidentBenchmarkAvailability(True)


def _require_positive_dimension(value: object, name: str) -> int:
    """Validate a preconstruction non-boolean positive dimension."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-bool integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _close_loop_preserving_error(
    close_loop: Callable[[Any], None], loop: Any, error: BaseException | None
) -> None:
    """Close a loop without allowing teardown to mask a primary failure."""
    if error is None:
        close_loop(loop)
        return
    try:
        close_loop(loop)
    except BaseException as cleanup_error:
        raise error from cleanup_error
    raise error


@contextmanager
def qualified_cuda_resident_benchmark(
    *,
    duration: float = 0.0,
    n_boxes: int = 1,
    n_particles: int = 16,
    n_species: int = 2,
    root_seed: int = 1582,
    case_id: str = "resident-fixture",
    availability: ResidentBenchmarkAvailability | None = None,
) -> Iterator[ResidentCaptureBenchmarkBinding]:
    """Build, qualify, capture, and clean up one real native CUDA binding.

    CUDA/native-capture unavailability raises a structured preconstruction
    exception. Callers may provide the already-memoized preflight availability
    result; setup and capture failures remain errors. Setup and capture
    durations are provenance only and each completion synchronization is outside
    sampling. Cleanup preserves an initiating writer failure while attempting
    release before exact session closure.

    Args:
        duration: Physical resident timestep duration in seconds.
        n_boxes: Number of resident simulation boxes to construct.
        n_particles: Particle capacity per resident box.
        n_species: Species capacity per resident box.
        root_seed: Root seed for resident random-number streams.
        availability: Optional validated preflight availability result.

    Yields:
        One binding exposing paired uncaptured and captured operations.

    Raises:
        ValueError: If identity validation fails or qualification rejects the
            prepared resident loop.
        RuntimeError: If native CUDA capture setup fails after qualification.
    """
    n_boxes = _require_positive_dimension(n_boxes, "n_boxes")
    n_particles = _require_positive_dimension(n_particles, "n_particles")
    n_species = _require_positive_dimension(n_species, "n_species")
    resolved_availability = (
        cuda_capture_availability() if availability is None else availability
    )
    if not isinstance(resolved_availability, ResidentBenchmarkAvailability):
        raise TypeError("availability must be a ResidentBenchmarkAvailability.")
    if not resolved_availability.available:
        raise ResidentBenchmarkUnavailableError(resolved_availability.reason)

    import pytest

    from particula.execution.graph_capture import (
        capture_prepared_resident_graph,
        qualify_prepared_resident_graph_capture,
        replay_captured_resident_graph,
    )
    from particula.execution.resident_scheduler import (
        enqueue_prepared_resident_simulation,
    )
    from particula.execution.tests.captured_full_loop_test import (
        _build_prepared_loop,
        _close_prepared_loop,
        _qualification_is_explicitly_unavailable,
        _require_native_cuda_capture,
        _WarpNativeCaptureAdapter,
    )

    try:
        wp, candidates = _require_native_cuda_capture()
    except pytest.skip.Exception as error:
        raise ResidentBenchmarkUnavailableError(str(error)) from error
    device = candidates[0]
    monitor = _build_memory_monitor(
        case_id=case_id,
        wp=wp,
        native=device.native,
    )
    monitor.begin()
    setup_start = perf_counter()
    loop = _build_prepared_loop(
        device.native,
        n_boxes,
        duration,
        root_seed,
        n_particles=n_particles,
        n_species=n_species,
    )
    try:
        wp.synchronize()
        setup_elapsed = perf_counter() - setup_start
        adapter = _WarpNativeCaptureAdapter(wp, device.native)
        capture_set = loop.registry.validate_capture_resource_set(
            loop.request.capture_resource_requirements
        )
        try:
            qualification = qualify_prepared_resident_graph_capture(
                loop.binding, loop.prepared, capture_set, adapter
            )
        except ValueError as error:
            if _qualification_is_explicitly_unavailable(error):
                raise ResidentBenchmarkUnavailableError(str(error)) from error
            raise
        capture_start = perf_counter()
        captured = capture_prepared_resident_graph(qualification)
        wp.synchronize()
        monitor.snapshot_peak()
        capture_elapsed = perf_counter() - capture_start
        benchmark_binding = ResidentCaptureBenchmarkBinding(
            loop=loop,
            captured=captured,
            duration=duration,
            setup_elapsed_seconds=setup_elapsed,
            capture_elapsed_seconds=capture_elapsed,
            synchronize=wp.synchronize,
            enqueue_operation=lambda: enqueue_prepared_resident_simulation(
                loop.prepared
            ),
            replay_operation=lambda: replay_captured_resident_graph(
                captured, duration
            ),
            capture_set=capture_set,
            prepared_signature_digest=_prepared_signature_digest(loop),
            selected_device=_selected_device_metadata(wp, device.native),
            memory_monitor=monitor,
        )
        benchmark_binding.validate_identities()
        yield benchmark_binding
    except BaseException as error:
        _close_loop_preserving_error(_close_prepared_loop, loop, error)
    else:
        _close_loop_preserving_error(_close_prepared_loop, loop, None)
