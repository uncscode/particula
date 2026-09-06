"""Hardware-free tests for lazy resident CUDA benchmark support."""

import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from particula.execution.tests import resident_benchmark_cuda_support
from particula.execution.tests.resident_benchmark_cuda_support import (
    CudaFixtureMemoryMonitor,
    ResidentBenchmarkUnavailableError,
    ResidentCaptureBenchmarkBinding,
    _build_fixture_reset,
    _close_loop_preserving_error,
    qualified_cuda_resident_benchmark,
    resident_benchmark_provenance,
)


def test_cuda_support_import_does_not_import_warp() -> None:
    """Keep the host import boundary free of Warp and CUDA probing."""
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            "import sys; "
            "import particula.execution.tests.resident_benchmark_cuda_support; "
            "assert 'warp' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_memory_monitor_records_only_case_scoped_scalar_snapshots() -> None:
    """Require sentinel coverage before emitting valid high-water evidence."""
    calls: list[str] = []
    readings = iter((0, 8, 2, 12, 3))

    def record_reset(device: Any) -> None:
        calls.append(f"reset:{device}")

    def record_read(device: Any) -> int:
        calls.append(f"read:{device}")
        return next(readings)

    def record_sync() -> None:
        calls.append("sync")

    def allocate_sentinel() -> object:
        calls.append("sentinel")
        return object()

    adapter = SimpleNamespace(
        reset=record_reset,
        read=record_read,
        metadata={"runtime_version": 12000},
    )
    monitor = CudaFixtureMemoryMonitor(
        "case",
        "cuda:7",
        7,
        adapter,
        record_sync,
        allocate_sentinel,
    )
    monitor.begin()
    monitor.snapshot_peak()
    observation = monitor.finalize()
    assert observation.available
    assert observation.before_bytes == 2
    assert observation.peak_bytes == 12
    assert observation.after_bytes == 3
    assert calls == [
        "sync",
        "reset:7",
        "sync",
        "read:7",
        "sentinel",
        "sync",
        "read:7",
        "sync",
        "reset:7",
        "sync",
        "read:7",
        "sync",
        "read:7",
        "sync",
        "read:7",
    ]


def test_memory_monitor_synchronizes_the_exact_selected_cuda_device() -> None:
    """Bind every monitor synchronization to its fixture's native device."""
    calls: list[object] = []
    fake_warp = SimpleNamespace(
        synchronize_device=lambda native: calls.append(native),
        zeros=lambda *_args, **_kwargs: object(),
        uint8="uint8",
    )

    monitor = resident_benchmark_cuda_support._build_memory_monitor(
        case_id="case",
        wp=fake_warp,
        native="cuda:7",
        adapter_factory=lambda: SimpleNamespace(),
    )

    monitor.synchronize()

    assert calls == ["cuda:7"]


def test_memory_monitor_returns_unavailable_when_sentinel_does_not_change() -> (
    None
):
    """Never infer default-pool coverage from an unchanged counter."""
    adapter = SimpleNamespace(
        reset=lambda _device: None,
        read=lambda _device: 0,
        metadata={"runtime_version": 12000},
    )
    monitor = CudaFixtureMemoryMonitor(
        "case", "cuda:0", 0, adapter, lambda: None, lambda: object()
    )
    monitor.begin()
    observation = monitor.finalize()
    assert not observation.available
    assert observation.before_bytes is observation.peak_bytes is None


def test_fixture_reset_restores_mutable_arrays_before_each_sample() -> None:
    """Restore primary and RNG state without replacing resident identities."""

    class Array:
        """Minimal device-array double with observable storage identity."""

        def __init__(self, values: list[int]) -> None:
            self.values = values
            self.shape = (len(values),)
            self.dtype = "uint32"
            self.device = "cuda:0"

    calls: list[str] = []

    def zeros(shape, *, dtype, device):
        assert shape == (2,)
        assert dtype == "uint32"
        assert device == "cuda:0"
        return Array([0, 0])

    def copy(destination: Array, source: Array) -> None:
        calls.append("copy")
        destination.values[:] = source.values

    wp = SimpleNamespace(
        zeros=zeros,
        copy=copy,
        synchronize=lambda: calls.append("synchronize"),
    )
    primary = Array([1, 2])
    rng = Array([3, 4])
    loop = SimpleNamespace(
        request=SimpleNamespace(primary=primary),
        session=SimpleNamespace(),
        coagulation_resources=SimpleNamespace(rng=rng),
        wall_loss_resources=SimpleNamespace(),
    )
    validations: list[str] = []
    reset = _build_fixture_reset(
        wp,
        loop,
        lambda: validations.append("validated"),
    )
    primary.values[:] = [9, 9]
    rng.values[:] = [8, 8]
    reset()

    assert primary.values == [1, 2]
    assert rng.values == [3, 4]
    assert calls == [
        "copy",
        "copy",
        "synchronize",
        "synchronize",
        "copy",
        "copy",
        "synchronize",
    ]
    assert validations == ["validated"]


def test_fixture_reset_handles_empty_mutable_array_registry() -> None:
    """Drain and validate an empty fixture without allocating snapshots."""
    calls: list[str] = []
    wp = SimpleNamespace(
        zeros=lambda *_args, **_kwargs: pytest.fail("unexpected snapshot"),
        copy=lambda *_args, **_kwargs: pytest.fail("unexpected restore"),
        synchronize=lambda: calls.append("synchronize"),
    )
    loop = SimpleNamespace(
        request=SimpleNamespace(),
        session=SimpleNamespace(),
        coagulation_resources=SimpleNamespace(),
        wall_loss_resources=SimpleNamespace(),
    )

    reset = _build_fixture_reset(wp, loop, lambda: calls.append("validated"))
    reset()

    assert calls == ["synchronize", "synchronize", "validated"]


def test_fixture_reset_propagates_final_identity_validation_error() -> None:
    """Preserve validation failures after restoring the captured state."""
    calls: list[str] = []
    wp = SimpleNamespace(
        zeros=lambda *_args, **_kwargs: pytest.fail("unexpected snapshot"),
        copy=lambda *_args, **_kwargs: pytest.fail("unexpected restore"),
        synchronize=lambda: calls.append("synchronize"),
    )
    loop = SimpleNamespace(
        request=SimpleNamespace(),
        session=SimpleNamespace(),
        coagulation_resources=SimpleNamespace(),
        wall_loss_resources=SimpleNamespace(),
    )

    reset = _build_fixture_reset(
        wp,
        loop,
        lambda: (_ for _ in ()).throw(ValueError("identity drift")),
    )

    with pytest.raises(ValueError, match="identity drift"):
        reset()

    assert calls == ["synchronize", "synchronize"]


def test_binding_delegates_prepared_enqueue_and_captured_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegate only through the retained prepared and captured bindings."""
    calls: list[tuple[str, object, object | None]] = []
    scheduler = ModuleType("particula.execution.resident_scheduler")
    scheduler.enqueue_prepared_resident_simulation = (  # type: ignore[attr-defined]
        lambda prepared: calls.append(("enqueue", prepared, None))
    )
    graph_capture = ModuleType("particula.execution.graph_capture")
    graph_capture.replay_captured_resident_graph = (  # type: ignore[attr-defined]
        lambda captured, duration: calls.append(("replay", captured, duration))
    )
    monkeypatch.setitem(sys.modules, scheduler.__name__, scheduler)
    monkeypatch.setitem(sys.modules, graph_capture.__name__, graph_capture)

    binding = ResidentCaptureBenchmarkBinding(
        loop=SimpleNamespace(prepared="prepared"),
        captured="captured",
        duration=0.25,
        setup_elapsed_seconds=1.0,
        capture_elapsed_seconds=2.0,
        synchronize=lambda: None,
        enqueue_operation=lambda: calls.append(("enqueue", "prepared", None)),
        replay_operation=lambda: calls.append(("replay", "captured", 0.25)),
        capture_set="capture-set",
        prepared_signature_digest="signature",
        selected_device={"status": "available", "identity": "cuda:0"},
    )

    binding.enqueue()
    binding.replay()

    assert calls == [
        ("enqueue", "prepared", None),
        ("replay", "captured", 0.25),
    ]


def test_binding_identity_gate_rejects_drift_before_timing_callbacks() -> None:
    """Reject an exact resident-binding identity mismatch without operations."""
    calls: list[str] = []
    loop = SimpleNamespace(
        binding=SimpleNamespace(
            request=object(),
            session=object(),
            registry=object(),
            guard=object(),
        ),
        request=SimpleNamespace(capture_resource_requirements=object()),
        session=object(),
        registry=object(),
        guard=object(),
    )
    binding = ResidentCaptureBenchmarkBinding(
        loop=loop,
        captured=object(),
        duration=0.0,
        setup_elapsed_seconds=0.0,
        capture_elapsed_seconds=0.0,
        synchronize=lambda: calls.append("sync"),
        enqueue_operation=lambda: calls.append("enqueue"),
        replay_operation=lambda: calls.append("replay"),
        capture_set=object(),
        prepared_signature_digest="signature",
        selected_device={"status": "available", "identity": "cuda:0"},
    )
    with pytest.raises(ValueError, match="identity drifted"):
        binding.validate_identities()
    assert calls == []


def test_cleanup_runs_in_release_then_close_order_after_setup_failure() -> None:
    """Clean up acquired capture resources after a setup failure."""
    calls: list[str] = []

    def close_loop(_loop: object) -> None:
        calls.extend(("release", "close"))

    with pytest.raises(RuntimeError, match="setup failed"):
        _close_loop_preserving_error(
            close_loop,
            object(),
            RuntimeError("setup failed"),
        )

    assert calls == ["release", "close"]


def test_cleanup_failure_is_chained_behind_callback_failure() -> None:
    """Keep the benchmark callback failure primary when teardown also fails."""
    primary = RuntimeError("callback failed")

    def fail_close(_loop: object) -> None:
        raise ValueError("cleanup failed")

    with pytest.raises(RuntimeError, match="callback failed") as caught:
        _close_loop_preserving_error(fail_close, object(), primary)

    assert isinstance(caught.value.__cause__, ValueError)
    assert str(caught.value.__cause__) == "cleanup failed"


def test_qualified_cuda_binding_rejects_unavailable_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use supplied unavailable preflight without importing the CUDA helper."""
    calls: list[str] = []
    monkeypatch.setattr(
        resident_benchmark_cuda_support,
        "cuda_capture_availability",
        lambda: calls.append("probe"),
    )

    with pytest.raises(ResidentBenchmarkUnavailableError, match="no CUDA"):
        with qualified_cuda_resident_benchmark(
            availability=resident_benchmark_cuda_support.ResidentBenchmarkAvailability(
                False, "no CUDA"
            )
        ):
            pass

    assert calls == []


@pytest.mark.parametrize("duration", (0.0, float("nan"), True))
def test_qualified_cuda_binding_validates_duration_before_cuda_preflight(
    duration: object,
) -> None:
    """Reject invalid timing inputs before any CUDA availability handling."""
    with pytest.raises((TypeError, ValueError), match="duration"):
        with qualified_cuda_resident_benchmark(
            duration=duration,
            availability=resident_benchmark_cuda_support.ResidentBenchmarkAvailability(
                False, "no CUDA"
            ),
        ):
            pass


@pytest.mark.parametrize("dimension", (0, -1, True))
def test_qualified_cuda_binding_validates_dimensions_before_cuda_preflight(
    dimension: object,
) -> None:
    """Reject invalid fixture shapes before any CUDA availability handling."""
    with pytest.raises((TypeError, ValueError), match="n_boxes"):
        with qualified_cuda_resident_benchmark(
            n_boxes=dimension,
            availability=resident_benchmark_cuda_support.ResidentBenchmarkAvailability(
                False, "no CUDA"
            ),
        ):
            pass


def test_provenance_uses_real_signature_and_nondefault_device() -> None:
    """Expose fixture-owned signature and selected nondefault CUDA metadata."""
    device = {"status": "available", "identity": "cuda:7", "memory": 42}
    binding = SimpleNamespace(
        prepared_signature_digest="real-prepared-signature",
        selected_device=device,
    )

    signature, selected_device = resident_benchmark_provenance(
        cast(ResidentCaptureBenchmarkBinding, binding)
    )

    assert signature == "real-prepared-signature"
    assert selected_device is device


def _install_qualification_failure_modules(
    monkeypatch: pytest.MonkeyPatch, error: ValueError
) -> None:
    """Install minimal CUDA fixture doubles that fail during qualification."""
    request = SimpleNamespace(capture_resource_requirements=object())
    registry = SimpleNamespace(
        validate_capture_resource_set=lambda _requirements: object()
    )
    loop = SimpleNamespace(
        request=request,
        session=object(),
        registry=registry,
        guard=object(),
        prepared=SimpleNamespace(
            signature=SimpleNamespace(
                device=SimpleNamespace(backend="warp", native="cuda:0"),
                dimensions=SimpleNamespace(
                    n_boxes=1, n_particles=16, n_species=2
                ),
            )
        ),
    )
    loop.binding = SimpleNamespace(
        request=loop.request,
        session=loop.session,
        registry=loop.registry,
        guard=loop.guard,
    )
    graph_capture = ModuleType("particula.execution.graph_capture")
    graph_capture.qualify_prepared_resident_graph_capture = (  # type: ignore[attr-defined]
        lambda *_args: (_ for _ in ()).throw(error)
    )
    graph_capture.capture_prepared_resident_graph = lambda _value: object()  # type: ignore[attr-defined]
    graph_capture.replay_captured_resident_graph = lambda *_args: None  # type: ignore[attr-defined]
    scheduler = ModuleType("particula.execution.resident_scheduler")
    scheduler.enqueue_prepared_resident_simulation = lambda _value: None  # type: ignore[attr-defined]
    helpers = ModuleType("particula.execution.tests.captured_full_loop_test")
    helpers._require_native_cuda_capture = lambda: (  # type: ignore[attr-defined]
        SimpleNamespace(
            synchronize=lambda: None,
            get_device=lambda native: SimpleNamespace(
                alias=native, total_memory=1024
            ),
        ),
        [SimpleNamespace(native="cuda:0")],
    )
    helpers._build_prepared_loop = lambda *_args, **_kwargs: loop  # type: ignore[attr-defined]
    helpers._close_prepared_loop = lambda _loop: None  # type: ignore[attr-defined]
    helpers._WarpNativeCaptureAdapter = lambda *_args: object()  # type: ignore[attr-defined]
    helpers._qualification_is_explicitly_unavailable = (  # type: ignore[attr-defined]
        lambda value: str(value)
        in {
            "graph capture runtime is unavailable.",
            "graph capture device is unavailable.",
            "graph capture API is unsupported.",
        }
    )
    monkeypatch.setitem(sys.modules, graph_capture.__name__, graph_capture)
    monkeypatch.setitem(sys.modules, scheduler.__name__, scheduler)
    monkeypatch.setitem(sys.modules, helpers.__name__, helpers)


@pytest.mark.parametrize("total_memory", (None, "unknown", float("nan")))
def test_selected_device_missing_or_nonnumeric_memory_is_unavailable(
    total_memory: object,
) -> None:
    """Keep absent or invalid CUDA memory metadata schema-valid."""
    fake_warp = SimpleNamespace(
        get_device=lambda _native: SimpleNamespace(
            alias="cuda:0", total_memory=total_memory
        )
    )

    metadata = resident_benchmark_cuda_support._selected_device_metadata(
        fake_warp, "cuda:0"
    )

    assert metadata == {
        "status": "unavailable",
        "identity": None,
        "memory": None,
    }


@pytest.mark.parametrize(
    "message",
    (
        "graph capture runtime is unavailable.",
        "graph capture device is unavailable.",
        "graph capture API is unsupported.",
    ),
)
def test_qualified_cuda_binding_normalizes_known_qualification_absence(
    monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    """Translate only documented qualification absence into unavailability."""
    _install_qualification_failure_modules(monkeypatch, ValueError(message))

    with pytest.raises(ResidentBenchmarkUnavailableError, match=message):
        with qualified_cuda_resident_benchmark():
            pass


def test_qualified_cuda_binding_propagates_unknown_qualification_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve unexpected qualification errors as implementation failures."""
    _install_qualification_failure_modules(
        monkeypatch, ValueError("qualification invariant failed")
    )

    with pytest.raises(ValueError, match="qualification invariant failed"):
        with qualified_cuda_resident_benchmark():
            pass


def test_qualified_cuda_binding_closes_once_after_capture_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release the constructed loop once when native capture fails."""
    _install_qualification_failure_modules(
        monkeypatch, ValueError("qualification invariant failed")
    )
    closed: list[object] = []
    graph_capture = sys.modules["particula.execution.graph_capture"]
    graph_capture.qualify_prepared_resident_graph_capture = (  # type: ignore[attr-defined]
        lambda *_args: object()
    )
    graph_capture.capture_prepared_resident_graph = (  # type: ignore[attr-defined]
        lambda _qualification: (_ for _ in ()).throw(
            RuntimeError("capture failed")
        )
    )
    helpers = sys.modules["particula.execution.tests.captured_full_loop_test"]
    helpers._close_prepared_loop = closed.append  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="capture failed"):
        with qualified_cuda_resident_benchmark():
            pass

    assert len(closed) == 1


def test_qualified_cuda_binding_captures_and_closes_after_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build one qualified capture and close its exact loop after use."""
    calls: list[tuple[str, object]] = []
    request = SimpleNamespace(capture_resource_requirements="requirements")
    session = object()

    def validate_capture_resource_set(requirements: object) -> str:
        calls.append(("validate", requirements))
        return "capture-set"

    registry = SimpleNamespace(
        validate_capture_resource_set=validate_capture_resource_set
    )
    guard = object()
    binding_identity = SimpleNamespace(
        request=request,
        session=session,
        registry=registry,
        guard=guard,
    )
    loop = SimpleNamespace(
        registry=registry,
        request=request,
        session=session,
        guard=guard,
        binding=binding_identity,
        prepared=SimpleNamespace(
            signature=SimpleNamespace(
                device=SimpleNamespace(backend="warp", native="cuda:7"),
                dimensions=SimpleNamespace(
                    n_boxes=2, n_particles=3, n_species=2
                ),
            )
        ),
    )
    graph_capture = ModuleType("particula.execution.graph_capture")

    def qualify_capture(
        binding: object,
        prepared: object,
        capture_set: object,
        adapter: object,
    ) -> str:
        calls.append(("qualify", (binding, prepared, capture_set, adapter)))
        return "qualification"

    def capture_graph(qualification: object) -> str:
        calls.append(("capture", qualification))
        return "captured"

    graph_capture.qualify_prepared_resident_graph_capture = (  # type: ignore[attr-defined]
        qualify_capture
    )
    graph_capture.capture_prepared_resident_graph = (  # type: ignore[attr-defined]
        capture_graph
    )
    graph_capture.replay_captured_resident_graph = (  # type: ignore[attr-defined]
        lambda captured, duration: calls.append(
            ("replay", (captured, duration))
        )
    )
    scheduler = ModuleType("particula.execution.resident_scheduler")
    scheduler.enqueue_prepared_resident_simulation = (  # type: ignore[attr-defined]
        lambda prepared: calls.append(("enqueue", prepared))
    )
    helpers = ModuleType("particula.execution.tests.captured_full_loop_test")
    fake_warp = SimpleNamespace(
        synchronize=lambda: calls.append(("sync", None)),
        synchronize_device=lambda native: calls.append(
            ("monitor_sync", native)
        ),
        get_device=lambda native: SimpleNamespace(
            alias=native, total_memory=1024
        ),
    )
    helpers._require_native_cuda_capture = lambda: (  # type: ignore[attr-defined]
        fake_warp,
        [SimpleNamespace(native="cuda:7")],
    )

    def build_prepared_loop(*args: object, **kwargs: object) -> Any:
        calls.append(("build", (args, kwargs)))
        return loop

    helpers._build_prepared_loop = lambda *args, **kwargs: (  # type: ignore[attr-defined]
        build_prepared_loop(*args, **kwargs)
    )
    helpers._close_prepared_loop = lambda value: calls.append(  # type: ignore[attr-defined]
        ("close", value)
    )
    helpers._WarpNativeCaptureAdapter = lambda *args: (  # type: ignore[attr-defined]
        "adapter",
        args,
    )
    helpers._qualification_is_explicitly_unavailable = (  # type: ignore[attr-defined]
        lambda error: False
    )
    monkeypatch.setitem(sys.modules, graph_capture.__name__, graph_capture)
    monkeypatch.setitem(sys.modules, scheduler.__name__, scheduler)
    monkeypatch.setitem(sys.modules, helpers.__name__, helpers)

    class Monitor:
        """Record snapshot ordering without affecting benchmark timing."""

        def begin(self) -> None:
            calls.append(("monitor_begin", None))

        def snapshot_peak(self) -> None:
            calls.append(("monitor_snapshot", None))

    monkeypatch.setattr(
        resident_benchmark_cuda_support,
        "_build_memory_monitor",
        lambda **_kwargs: Monitor(),
    )
    ticks = iter((1.0, 3.0, 10.0, 15.0))

    def clock() -> float:
        """Record timing boundaries around the monitor snapshot seam."""
        calls.append(("clock", None))
        return next(ticks)

    monkeypatch.setattr(resident_benchmark_cuda_support, "perf_counter", clock)

    with qualified_cuda_resident_benchmark(
        duration=0.5, n_boxes=2, root_seed=3
    ) as binding:
        assert binding.loop is loop
        assert binding.captured == "captured"
        assert binding.duration == 0.5
        assert binding.setup_elapsed_seconds == 2.0
        assert binding.capture_elapsed_seconds == 5.0
        assert binding.prepared_signature_digest
        assert binding.selected_device == {
            "status": "available",
            "identity": "cuda:7",
            "memory": 1024,
        }
        binding.synchronize()

    assert calls == [
        ("monitor_begin", None),
        ("clock", None),
        (
            "build",
            (
                ("cuda:7", 2, 0.5, 3),
                {
                    "n_particles": 16,
                    "n_species": 2,
                    "full_activity": True,
                },
            ),
        ),
        ("sync", None),
        ("clock", None),
        ("validate", "requirements"),
        (
            "qualify",
            (
                binding_identity,
                loop.prepared,
                "capture-set",
                ("adapter", (fake_warp, "cuda:7")),
            ),
        ),
        ("clock", None),
        ("capture", "qualification"),
        ("sync", None),
        ("clock", None),
        ("monitor_snapshot", None),
        ("sync", None),
        ("close", loop),
    ]
