"""Lazy CUDA-only support for resident capture timing evidence.

Importing this test-only module neither imports Warp nor probes CUDA. Builder
calls qualify one exact prepared/captured binding and release graph provenance
before closing the resident session; timing callers own synchronization.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from typing import Any, Callable, Iterator


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
    metadata: dict[str, Any] = {
        "status": "available",
        "identity": str(getattr(selected, "alias", native)),
    }
    if isinstance(total_memory, int | float):
        metadata["memory"] = int(total_memory)
    else:
        metadata["memory"] = {"status": "unavailable"}
    return metadata


def resident_benchmark_provenance(
    binding: ResidentCaptureBenchmarkBinding,
) -> tuple[str, dict[str, Any]]:
    """Return fixture-authoritative signature and device provenance."""
    return binding.prepared_signature_digest, binding.selected_device


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
    *, duration: float = 0.0, n_boxes: int = 1, root_seed: int = 1582
) -> Iterator[ResidentCaptureBenchmarkBinding]:
    """Build, qualify, capture, and clean up one real native CUDA binding.

    CUDA/native-capture unavailability skips before construction. Setup and
    capture durations are provenance only and each completion synchronization is
    outside sampling. Cleanup preserves an initiating writer failure while
    attempting release before exact session closure.

    Args:
        duration: Physical resident timestep duration in seconds.
        n_boxes: Number of resident simulation boxes to construct.
        root_seed: Root seed for resident random-number streams.

    Yields:
        One binding exposing paired uncaptured and captured operations.

    Raises:
        ValueError: If identity validation fails or qualification rejects the
            prepared resident loop.
        RuntimeError: If native CUDA capture setup fails after qualification.
    """
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

    wp, candidates = _require_native_cuda_capture()
    device = candidates[0]
    setup_start = perf_counter()
    loop = _build_prepared_loop(device.native, n_boxes, duration, root_seed)
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
                import pytest

                pytest.skip(str(error))
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
        )
        benchmark_binding.validate_identities()
        yield benchmark_binding
    except BaseException as error:
        _close_loop_preserving_error(_close_prepared_loop, loop, error)
    else:
        _close_loop_preserving_error(_close_prepared_loop, loop, None)
