"""Hardware-free tests for lazy resident CUDA benchmark support."""

import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from particula.execution.tests import resident_benchmark_cuda_support
from particula.execution.tests.resident_benchmark_cuda_support import (
    ResidentCaptureBenchmarkBinding,
    qualified_cuda_resident_benchmark,
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
    )
    with pytest.raises(ValueError, match="identity drifted"):
        binding.validate_identities()
    assert calls == []


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
        prepared="prepared",
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
        synchronize=lambda: calls.append(("sync", None))
    )
    helpers._require_native_cuda_capture = lambda: (  # type: ignore[attr-defined]
        fake_warp,
        [SimpleNamespace(native="cuda:0")],
    )

    def build_prepared_loop(*args: object) -> Any:
        calls.append(("build", args))
        return loop

    helpers._build_prepared_loop = lambda *args: (  # type: ignore[attr-defined]
        build_prepared_loop(*args)
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
    ticks = iter((1.0, 3.0, 10.0, 15.0))
    monkeypatch.setattr(
        resident_benchmark_cuda_support, "perf_counter", lambda: next(ticks)
    )

    with qualified_cuda_resident_benchmark(
        duration=0.5, n_boxes=2, root_seed=3
    ) as binding:
        assert binding.loop is loop
        assert binding.captured == "captured"
        assert binding.duration == 0.5
        assert binding.setup_elapsed_seconds == 2.0
        assert binding.capture_elapsed_seconds == 5.0
        binding.synchronize()

    assert calls == [
        ("build", ("cuda:0", 2, 0.5, 3)),
        ("sync", None),
        ("validate", "requirements"),
        (
            "qualify",
            (
                binding_identity,
                "prepared",
                "capture-set",
                ("adapter", (fake_warp, "cuda:0")),
            ),
        ),
        ("capture", "qualification"),
        ("sync", None),
        ("sync", None),
        ("close", loop),
    ]
