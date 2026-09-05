"""Lazy CUDA-only support for resident capture timing evidence.

Importing this test-only module neither imports Warp nor probes CUDA. Builder
calls qualify one exact prepared/captured binding and release graph provenance
before closing the resident session; timing callers own synchronization.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Iterator


@dataclass(frozen=True, slots=True)
class ResidentCaptureBenchmarkBinding:
    """Retain one qualified CUDA binding and provenance-only setup timings."""

    loop: Any
    captured: Any
    duration: float
    setup_elapsed_seconds: float
    capture_elapsed_seconds: float
    synchronize: Callable[[], None]
    enqueue_operation: Callable[[], None]
    replay_operation: Callable[[], None]
    capture_set: Any

    def enqueue(self) -> None:
        """Enqueue the already-prepared uncaptured timestep without syncing."""
        self.enqueue_operation()

    def replay(self) -> None:
        """Replay the captured timestep without lifecycle work or syncing."""
        self.replay_operation()

    def validate_identities(self) -> None:
        """Reject capture-binding drift before any timing callback can run."""
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


@contextmanager
def qualified_cuda_resident_benchmark(
    *, duration: float = 0.0, n_boxes: int = 1, root_seed: int = 1582
) -> Iterator[ResidentCaptureBenchmarkBinding]:
    """Build, qualify, capture, and clean up one real native CUDA binding.

    CUDA/native-capture unavailability skips before construction. Setup and
    capture durations are provenance only and each completion synchronization is
    outside sampling. Cleanup preserves an initiating writer failure while
    attempting release before exact session closure.
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
    wp.synchronize()
    setup_elapsed = perf_counter() - setup_start
    try:
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
        )
        benchmark_binding.validate_identities()
        yield benchmark_binding
    finally:
        _close_prepared_loop(loop)
