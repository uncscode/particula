"""Lazy CUDA-only support for resident capture benchmark evidence.

Importing this test-only module neither imports Warp nor probes CUDA. Builder
calls qualify one exact prepared/captured binding and release graph provenance
before closing the resident session. A private monitor records one
case-scoped CUDA default-pool high-water observation outside unchanged timing
loops; benchmark callers own timing synchronization.
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
        prepared_signature_digest: Digest of the retained prepared signature.
        selected_device: Qualified CUDA device metadata for provenance.
        reset_fixture: Callback that restores state without replacing objects.
        memory_monitor: Optional fixture-scoped CUDA memory monitor.
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
    reset_fixture: Callable[[], None] = lambda: None
    memory_monitor: Any = None
    _identity_snapshot: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Capture every qualified object and mutable-array identity."""
        arrays = _mutable_resident_arrays(self.loop)
        object.__setattr__(
            self,
            "_identity_snapshot",
            (
                id(self.loop),
                id(self.captured),
                id(self.capture_set),
                id(getattr(self.loop, "request", None)),
                id(getattr(self.loop, "session", None)),
                id(getattr(self.loop, "registry", None)),
                id(getattr(self.loop, "guard", None)),
                *(id(array) for array in arrays),
            ),
        )

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
        arrays = _mutable_resident_arrays(self.loop)
        current_snapshot = (
            id(self.loop),
            id(self.captured),
            id(self.capture_set),
            id(getattr(self.loop, "request", None)),
            id(getattr(self.loop, "session", None)),
            id(getattr(self.loop, "registry", None)),
            id(getattr(self.loop, "guard", None)),
            *(id(array) for array in arrays),
        )
        if (
            binding.request is not self.loop.request
            or binding.session is not self.loop.session
            or binding.registry is not self.loop.registry
            or binding.guard is not self.loop.guard
            or self.loop.request.capture_resource_requirements is None
            or self.capture_set is None
            or current_snapshot != self._identity_snapshot
        ):
            raise ValueError("resident benchmark binding identity drifted.")

    def reset(self) -> None:
        """Restore the qualified fixture outside a timing interval.

        The callback restores mutable resident payloads and validates that the
        qualified binding identities remain unchanged.
        """
        self.reset_fixture()


def _is_device_array(value: Any) -> bool:
    """Check whether a value has the minimum Warp-array metadata.

    Args:
        value: Object to inspect for array-like device attributes.

    Returns:
        ``True`` when the object exposes shape, dtype, and device attributes.
    """
    return all(
        hasattr(value, attribute) for attribute in ("shape", "dtype", "device")
    )


def _nested_values(value: Any) -> tuple[Any, ...]:
    """Extract direct nested values without invoking arbitrary properties.

    Args:
        value: Container or object whose direct values should be traversed.

    Returns:
        Tuple of directly stored child values, or an empty tuple for leaves.
    """
    if isinstance(value, dict):
        return tuple(value.values())
    if isinstance(value, (tuple, list)):
        return tuple(value)
    if callable(value):
        return ()
    if hasattr(value, "__dict__"):
        return tuple(vars(value).values())
    return tuple(
        getattr(value, name)
        for name in getattr(type(value), "__slots__", ())
        if hasattr(value, name)
    )


def _mutable_resident_arrays(loop: Any) -> tuple[Any, ...]:
    """Collect mutable primaries and sidecars without duplicate identities.

    Args:
        loop: Prepared resident loop whose retained state is traversed.

    Returns:
        Tuple of unique device-array objects reachable from the loop state.
    """
    arrays: list[Any] = []
    pending = [
        getattr(loop, "request", None),
        getattr(loop, "session", None),
        getattr(loop, "coagulation_resources", None),
        getattr(loop, "wall_loss_resources", None),
    ]
    visited: set[int] = set()
    while pending:
        value = pending.pop()
        if id(value) in visited:
            continue
        visited.add(id(value))
        if _is_device_array(value):
            arrays.append(value)
            continue
        pending.extend(_nested_values(value))
    unique: list[Any] = []
    seen: set[int] = set()
    for array in arrays:
        if id(array) not in seen:
            unique.append(array)
            seen.add(id(array))
    return tuple(unique)


def _build_fixture_reset(
    wp: Any,
    loop: Any,
    validate_identities: Callable[[], None] | None = None,
) -> Callable[[], None]:
    """Build a reset callback that preserves mutable-state identities.

    Args:
        wp: Warp module used for device allocation, copying, and
            synchronization.
        loop: Prepared resident loop whose mutable arrays are snapshotted.
        validate_identities: Optional callback to run after each restoration.

    Returns:
        Callback that restores all snapshotted arrays in place.
    """
    arrays = _mutable_resident_arrays(loop)
    snapshots = tuple(
        wp.zeros(array.shape, dtype=array.dtype, device=array.device)
        for array in arrays
    )
    for snapshot, array in zip(snapshots, arrays, strict=True):
        wp.copy(snapshot, array)
    if arrays:
        wp.synchronize()

    def reset() -> None:
        """Drain, restore pre-bound arrays, drain, then validate identities."""
        wp.synchronize()
        for array, snapshot in zip(arrays, snapshots, strict=True):
            wp.copy(array, snapshot)
        wp.synchronize()
        if validate_identities is not None:
            validate_identities()

    return reset


@dataclass(slots=True)
class CudaFixtureMemoryMonitor:
    """Measure one exact CUDA fixture outside its timing collector.

    The monitor proves default-pool coverage with a bounded same-device
    sentinel, resets the high-water counter, and records pre-build,
    post-capture, and post-cleanup readings. Monitoring failures produce
    structured unavailable evidence rather than timing failures or deltas.

    Attributes:
        case_id: Canonical identifier for the measured benchmark fixture.
        native: Exact native CUDA device selected for the fixture.
        device_ordinal: CUDA Runtime ordinal for ``native``.
        adapter: Lazy default-pool high-water counter adapter.
        synchronize: Callback that completes work on the selected device.
        sentinel_allocate: Bounded same-device allocation used for coverage.
        before: Used-high bytes recorded before fixture construction.
        peak: Used-high bytes recorded after fixture capture.
        reason: Structured unavailability reason, if monitoring failed.
    """

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
    """Extract the exact CUDA ordinal accepted by the runtime adapter.

    Args:
        native: Opaque native device value expected in ``cuda:<ordinal>`` form.

    Returns:
        Integer CUDA runtime ordinal.

    Raises:
        ValueError: If ``native`` is not an exact CUDA ordinal string.
    """
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
    """Build an uncached per-fixture monitor with a bounded Warp sentinel.

    Args:
        case_id: Canonical identifier for the fixture being monitored.
        wp: Lazily imported Warp module used only for device operations.
        native: Exact native device selected by capture qualification.
        adapter_factory: Factory for the lazy CUDA Runtime counter adapter.

    Returns:
        A fresh monitor whose per-fixture coverage has not yet begun.
    """
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
        synchronize=lambda: wp.synchronize_device(native),
        sentinel_allocate=lambda: wp.zeros(1, dtype=wp.uint8, device=native),
    )
    if reason is not None:
        monitor._unavailable(reason)
    return monitor


def _prepared_signature_digest(loop: Any) -> str:
    """Compute a stable digest for the retained prepared signature.

    Args:
        loop: Prepared resident loop containing the signature to identify.

    Returns:
        Hexadecimal SHA-256 digest of signature identity metadata.
    """
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
    """Collect identity and memory metadata for the selected CUDA device.

    Args:
        wp: Warp module used to resolve the selected device.
        native: Opaque native device identifier.

    Returns:
        JSON-compatible device identity and total-memory metadata.
    """
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
    """Return fixture-authoritative signature and device provenance.

    Args:
        binding: Qualified benchmark binding to describe.

    Returns:
        The retained prepared-signature digest and selected-device metadata.
    """
    return binding.prepared_signature_digest, binding.selected_device


def cuda_capture_availability() -> ResidentBenchmarkAvailability:
    """Probe the CUDA/capture gate and normalize only absence results.

    Returns:
        Structured availability metadata for native CUDA capture.
    """
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
    """Validate and return a non-boolean positive dimension.

    Args:
        value: Candidate dimension value.
        name: Dimension name used in validation errors.

    Returns:
        Validated positive integer dimension.

    Raises:
        TypeError: If ``value`` is not an integer or is boolean.
        ValueError: If ``value`` is not positive.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-bool integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _require_positive_duration(value: object) -> float:
    """Validate and return a finite, non-boolean positive duration.

    Args:
        value: Candidate duration in seconds.

    Returns:
        Validated duration in seconds as a float.

    Raises:
        TypeError: If ``value`` is not a real number or is boolean.
        ValueError: If the duration is non-finite or not positive.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("duration must be a non-bool real number.")
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be positive and finite.")
    return duration


def _close_loop_preserving_error(
    close_loop: Callable[[Any], None], loop: Any, error: BaseException | None
) -> None:
    """Close a loop while preserving any initiating failure.

    Args:
        close_loop: Teardown callback for the prepared loop.
        loop: Loop instance to close.
        error: Original failure, if teardown follows an unsuccessful operation.

    Raises:
        BaseException: The initiating error, or the cleanup error when no
            initiating error exists.
    """
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
    duration: float = 0.5,
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
        case_id: Canonical identifier attached to allocator evidence.
        availability: Optional validated preflight availability result.

    Yields:
        One binding exposing paired uncaptured and captured operations.

    Raises:
        ValueError: If identity validation fails or qualification rejects the
            prepared resident loop.
        RuntimeError: If native CUDA capture setup fails after qualification.
    """
    duration = _require_positive_duration(duration)
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
    loop = None
    try:
        monitor.begin()
        setup_start = perf_counter()
        loop = _build_prepared_loop(
            device.native,
            n_boxes,
            duration,
            root_seed,
            n_particles=n_particles,
            n_species=n_species,
            full_activity=True,
        )
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
            reset_fixture=lambda: None,
            memory_monitor=monitor,
        )
        validate_identities = benchmark_binding.validate_identities
        object.__setattr__(
            benchmark_binding,
            "reset_fixture",
            _build_fixture_reset(wp, loop, validate_identities),
        )
        # Reset snapshots are resident allocations, so construct and drain them
        # before recording the fixture high-water mark.  Reset calls remain
        # outside the measured dispatch and replay intervals.
        monitor.snapshot_peak()
        benchmark_binding.validate_identities()
        yield benchmark_binding
    except BaseException as error:
        if loop is None:
            raise
        _close_loop_preserving_error(_close_prepared_loop, loop, error)
    else:
        if loop is None:
            raise RuntimeError("benchmark loop was not created.")
        _close_loop_preserving_error(_close_prepared_loop, loop, None)
