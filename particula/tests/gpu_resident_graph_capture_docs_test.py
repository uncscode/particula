"""Contract tests for the native-CUDA resident graph-capture example."""

from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "docs/Examples/gpu_resident_graph_capture.py"
MODULE_NAME = "docs.Examples.gpu_resident_graph_capture"


class _NativeDevice:
    """Provide a hardware-free Warp-like device descriptor."""

    def __init__(self, name: str, *, is_cuda: bool) -> None:
        self.name = name
        self.is_cuda = is_cuda

    def __str__(self) -> str:
        return self.name


def _fresh_example() -> Any:
    """Import a fresh example module without retaining prior test patches."""
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


def test_import_is_lazy_about_warp_and_concrete_capture_modules() -> None:
    """Keep optional capture and concrete composition out of import time."""
    sys.modules.pop(MODULE_NAME, None)
    before = set(sys.modules)
    example = importlib.import_module(MODULE_NAME)
    loaded = set(sys.modules) - before
    assert example is not None
    assert "warp" not in loaded
    assert "particula.execution.graph_capture" not in loaded
    assert "particula.execution.resident_scheduler" not in loaded
    assert "particula.gpu" not in loaded


def test_force_disabled_path_is_deterministic_and_has_no_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force disabling avoids Warp loading, fixture creation, and fallback."""
    example = _fresh_example()
    original_import = example.importlib.import_module

    def fail_warp_import(name: str) -> Any:
        if name == "warp":
            pytest.fail("force-disabled path imported Warp")
        return original_import(name)

    monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_NATIVE_CAPTURE", "1")
    monkeypatch.setattr(example.importlib, "import_module", fail_warp_import)
    monkeypatch.setattr(
        example,
        "_load_enabled_runtime",
        lambda: pytest.fail("enabled runtime loaded"),
    )
    monkeypatch.setattr(
        example,
        "_build_cpu_state",
        lambda: pytest.fail("CPU fixture created"),
    )
    result = example.run_example()
    assert result.output == list(example._UNAVAILABLE_OUTPUT)
    assert result.session is None
    assert result.gas_snapshot is None
    assert result.replay_count == 0


def test_force_disabled_subprocess_has_exact_address_free_output() -> None:
    """The unavailable branch exits normally without a device."""
    environment = os.environ | {
        "PARTICULA_EXAMPLE_FORCE_NO_NATIVE_CAPTURE": "1"
    }
    result = subprocess.run(  # noqa: S603 - fixed repository-owned script
        [sys.executable, str(SOURCE)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.splitlines() == [
        "Canonical path: docs/Examples/gpu_resident_graph_capture.py",
        "Native CUDA graph capture is unavailable or disabled.",
        "No CPU or Warp-CPU fallback ran; no fixture, upload, or capture ran.",
    ]


@pytest.mark.parametrize(
    "warp, expected",
    [
        (None, None),
        (
            SimpleNamespace(
                get_devices=lambda: [_NativeDevice("cpu", is_cuda=False)]
            ),
            None,
        ),
        (
            SimpleNamespace(
                get_devices=lambda: [_NativeDevice("cuda:1", is_cuda=True)],
                capture_begin=lambda: None,
                capture_end=lambda: None,
            ),
            "cuda:1",
        ),
        (
            SimpleNamespace(
                get_devices=lambda: [
                    _NativeDevice("cpu", is_cuda=False),
                    _NativeDevice("cuda:1", is_cuda=True),
                    _NativeDevice("cuda:2", is_cuda=True),
                ],
                capture_begin=lambda: None,
                capture_end=lambda: None,
                capture_launch=lambda: None,
            ),
            "cuda:1",
        ),
    ],
)
def test_preflight_handles_only_explicit_unavailable_cases(
    monkeypatch: pytest.MonkeyPatch, warp: Any, expected: str | None
) -> None:
    """Missing Warp, CUDA, or capture callables return the unavailable result."""
    example = _fresh_example()

    def fake_import(name: str) -> Any:
        assert name == "warp"
        if warp is None:
            raise ModuleNotFoundError("warp", name="warp")
        return warp

    monkeypatch.setattr(example.importlib, "import_module", fake_import)
    assert example._qualified_native_cuda() == expected


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("loader failure"), RuntimeError("device failure")],
)
def test_preflight_propagates_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch, failure: RuntimeError
) -> None:
    """Unexpected optional-runtime failures do not become fallbacks."""
    example = _fresh_example()
    monkeypatch.setattr(
        example.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(failure),
    )
    with pytest.raises(RuntimeError, match=str(failure)):
        example._qualified_native_cuda()


def test_source_contract_preserves_lazy_native_capture_lifecycle() -> None:
    """Check import policy, replay ordering, limits, and the published link."""
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imports == {"importlib", "os", "numpy"}
    assert "from particula.execution import" not in source
    assert "from particula import" not in source
    assert "_step_gpu" not in source
    assert source.index("native = _qualified_native_cuda()") < source.index(
        "particles, gas, environment = _build_cpu_state()"
    )
    assert source.index("session.initialize_streams") < source.index(
        "prepare_resident_simulation"
    )
    assert source.count("replay_captured_resident_graph(captured, 1.0)") == 2
    assert source.index("for _ in range(2):") < source.index(
        "synchronize_device"
    )
    assert source.index("synchronize_device") < source.index(
        "gas_output.numpy()"
    )
    assert "session.particles.numpy" not in source
    assert "session.gas.concentration.numpy" not in source
    assert 'configuration.communication_map,\n        "form"' not in source
    assert source.index("object.__setattr__") < source.index(
        "retire_resident_graph_capture"
    )
    assert source.index("retire_resident_graph_capture") < source.index(
        "renew_resident_graph_capture"
    )
    normalized_source = " ".join(source.split())
    for phrase in (
        "CPU and Warp-CPU are not",
        "automatic recapture",
        "migration",
        "resize/compaction",
        "hidden transfer or synchronization",
        "retry/rollback",
        "checkpointed or serialized opaque handles",
        "performance claims",
    ):
        assert phrase in normalized_source
    index = (ROOT / "docs/Examples/index.md").read_text(encoding="utf-8")
    assert "gpu_resident_graph_capture.py" in index
    assert "python docs/Examples/gpu_resident_graph_capture.py" in index
    assert "no CPU or Warp-CPU fallback" in " ".join(index.split())


def test_enabled_lifecycle_retires_and_closes_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the documented capture lifecycle and deterministic teardown."""
    example = _fresh_example()
    events: list[str] = []
    original_update = object()
    request = SimpleNamespace(
        environment_update=original_update,
        capture_resource_requirements=object(),
    )
    gas_snapshot = np.ones((3, 1), dtype=np.float64)
    saturation_snapshot = np.full((3, 1), 2.0, dtype=np.float64)
    gas_output = SimpleNamespace(numpy=lambda: gas_snapshot)
    saturation_output = SimpleNamespace(numpy=lambda: saturation_snapshot)
    session = SimpleNamespace(
        particles=SimpleNamespace(masses=SimpleNamespace(device="cuda:1")),
        initialize_streams=lambda *_args: events.append("initialize"),
    )
    session.close = lambda *_args: events.append("session-close")
    registry = SimpleNamespace(
        validate_capture_resource_set=lambda _requirements: "capture-set",
        prepare_capture_resources=lambda requirements: events.append(
            f"publish-{requirements}"
        ),
    )
    guard = object()
    resident_runtime = SimpleNamespace(
        gpu_session=SimpleNamespace(
            setup_resident_session=lambda *_args: session,
            ResidentStepGuard=lambda *_args: guard,
        ),
        gpu_resources=SimpleNamespace(
            GPUResourceRegistry=lambda _session: registry
        ),
    )

    def prepare(candidate: Any, _duration: float) -> object:
        if candidate.environment_update is not original_update:
            events.append("invalidated")
            raise ValueError("structural drift")
        events.append("prepare")
        return object()

    captured_state = object()
    invalidated_state = object()
    first_capture = SimpleNamespace(
        lifecycle=SimpleNamespace(state=captured_state), name="captured"
    )
    renewed_capture = SimpleNamespace(
        lifecycle=SimpleNamespace(state=captured_state), name="renewed-capture"
    )
    captures = iter((first_capture, renewed_capture))
    binding = SimpleNamespace(lifecycle=SimpleNamespace(state=captured_state))

    def replay(captured: Any, _duration: float) -> None:
        events.append(f"replay-{captured.name}")
        if request.environment_update is not original_update:
            binding.lifecycle.state = invalidated_state
            events.append("invalidated")
            raise ValueError("structural drift")

    graph_capture = SimpleNamespace(
        GraphCaptureAvailability=SimpleNamespace(AVAILABLE="available"),
        GraphCaptureLifecycleState=SimpleNamespace(
            INVALIDATED=invalidated_state
        ),
        ResidentGraphCaptureBinding=lambda *_args: binding,
        _attach_resident_graph_capture_binding=lambda *_args: events.append(
            "attach"
        ),
        create_resident_graph_capture_signature=lambda _request: object(),
        resolve_graph_capture_capability=lambda device, adapter: (
            events.append(f"resolve-{device.native}")
            or SimpleNamespace(device=device, availability="available")
        ),
        create_graph_capture_lifecycle=lambda *_args: object(),
        qualify_prepared_resident_graph_capture=lambda *_args: object(),
        capture_prepared_resident_graph=lambda _qualification: next(captures),
        replay_captured_resident_graph=replay,
        retire_resident_graph_capture=lambda _binding: events.append("retire"),
        renew_resident_graph_capture=lambda *_args: object(),
        close_resident_graph_capture=lambda _binding: events.append(
            "capture-close"
        ),
    )
    communication = SimpleNamespace(
        CommunicationMapForm=SimpleNamespace(ONE_DIMENSIONAL="one-dimensional")
    )

    def communication_map(*args: Any) -> tuple[Any, ...]:
        events.append("communication-map")
        return args

    communication.CommunicationMap = communication_map

    def compose_request(
        composed_runtime: Any,
        _session: Any,
        composed_registry: Any,
        _guard: Any,
        *_args: Any,
    ) -> tuple[Any, Any, Any]:
        mapped = composed_runtime.communication.CommunicationMap(
            "ignored",
            "gas",
            1,
            object(),
            object(),
            object(),
            object(),
        )
        assert mapped[0] == "one-dimensional"
        composed_registry.prepare_capture_resources(
            request.capture_resource_requirements
        )
        return request, gas_output, saturation_output

    resident_example = SimpleNamespace(
        _load_enabled_runtime=lambda: SimpleNamespace(
            **vars(resident_runtime), communication=communication
        ),
        _request=compose_request,
    )
    runtime = SimpleNamespace(
        execution=SimpleNamespace(
            Backend=SimpleNamespace(WARP="warp"),
            Device=lambda _backend, native: SimpleNamespace(native=native),
        ),
        graph_capture=graph_capture,
        resident_scheduler=SimpleNamespace(prepare_resident_simulation=prepare),
        resident_example=resident_example,
        warp=SimpleNamespace(
            synchronize_device=lambda _device: events.append("synchronize")
        ),
    )
    monkeypatch.setattr(example, "_qualified_native_cuda", lambda: "cuda:1")
    monkeypatch.setattr(example, "_load_capability_runtime", lambda: runtime)
    monkeypatch.setattr(example, "_load_enabled_runtime", lambda: runtime)
    monkeypatch.setattr(
        example,
        "_build_cpu_state",
        lambda: (
            SimpleNamespace(
                volume=np.ones(1, dtype=np.float64),
                masses=np.ones((1, 1, 1), dtype=np.float64),
                concentration=np.ones((1, 1), dtype=np.float64),
            ),
            SimpleNamespace(concentration=np.ones((1, 1), dtype=np.float64)),
            object(),
        ),
    )

    result = example.run_example()

    assert result.replay_count == 2
    assert result.invalidated and result.retired and result.renewed
    assert result.synchronized
    assert result.captured is first_capture
    assert result.renewed_capture is renewed_capture
    assert result.captured is not result.renewed_capture
    np.testing.assert_array_equal(result.gas_snapshot, gas_snapshot)
    np.testing.assert_array_equal(
        result.saturation_snapshot, saturation_snapshot
    )
    assert events == [
        "resolve-cuda:1",
        "communication-map",
        f"publish-{request.capture_resource_requirements}",
        "attach",
        "initialize",
        "prepare",
        "replay-captured",
        "replay-captured",
        "synchronize",
        "replay-captured",
        "invalidated",
        "retire",
        "prepare",
        "capture-close",
        "session-close",
    ]


def test_native_adapter_aborts_and_releases_post_begin_failure() -> None:
    """End and release an incomplete native capture through the adapter."""
    example = _fresh_example()
    events: list[str] = []
    handle = SimpleNamespace(destroy=lambda: events.append("release"))

    def capture_end() -> SimpleNamespace:
        events.append("end")
        return handle

    warp = SimpleNamespace(
        capture_begin=lambda **_kwargs: events.append("begin"),
        capture_end=capture_end,
        capture_launch=lambda *_args: events.append("launch"),
    )
    graph_capture = SimpleNamespace(
        GraphCaptureNativeCallables=lambda *args: SimpleNamespace(
            capture_begin=args[0],
            capture_end=args[1],
            capture_instantiate=args[2],
            capture_launch=args[3],
            capture_release=args[4],
            capture_abort=args[5],
        )
    )
    device = SimpleNamespace(native="cuda:1")
    adapter = example._WarpNativeCaptureAdapter(warp, device, graph_capture)
    callables = adapter.capture_callables(device)

    callables.capture_begin()
    aborted = callables.capture_abort()
    callables.capture_release(aborted)

    assert events == ["begin", "end", "release"]


def test_native_adapter_rejects_unsupported_handle_release() -> None:
    """Never silently retain a native graph lacking destruction support."""
    example = _fresh_example()
    graph_capture = SimpleNamespace(
        GraphCaptureNativeCallables=lambda *args: SimpleNamespace(
            capture_release=args[4]
        )
    )
    device = SimpleNamespace(native="cuda:1")
    adapter = example._WarpNativeCaptureAdapter(
        SimpleNamespace(
            capture_begin=lambda **_kwargs: None,
            capture_end=lambda: object(),
            capture_launch=lambda *_args: None,
        ),
        device,
        graph_capture,
    )
    with pytest.raises(TypeError, match="callable destroy"):
        adapter.capture_callables(device).capture_release(object())


def test_native_adapter_requires_exact_device_identity() -> None:
    """Reject an equal-looking device that is not the selected declaration."""
    example = _fresh_example()
    selected = SimpleNamespace(native="cuda:1")
    adapter = example._WarpNativeCaptureAdapter(object(), selected, object())
    assert adapter.device_available(selected)
    assert adapter.capture_api_available(selected)
    assert not adapter.device_available(SimpleNamespace(native="cuda:1"))
    assert not adapter.capture_api_available(SimpleNamespace(native="cuda:1"))


def test_capability_resolution_rejects_device_mismatch() -> None:
    """Reject a canonical resolver result bound to another device identity."""
    example = _fresh_example()
    selected = object()
    graph_capture = SimpleNamespace(
        GraphCaptureAvailability=SimpleNamespace(AVAILABLE=object()),
        resolve_graph_capture_capability=lambda *_args: SimpleNamespace(
            device=object(), availability=object()
        ),
    )
    with pytest.raises(RuntimeError, match="different device"):
        example._resolve_exact_capability(graph_capture, selected, object())


def test_teardown_attempts_session_close_after_graph_close_failure() -> None:
    """Attempt both teardown operations and chain a second teardown failure."""
    example = _fresh_example()
    events: list[str] = []

    def fail_graph_close(_binding: Any) -> None:
        events.append("graph-close")
        raise RuntimeError("graph close failed")

    def fail_session_close(_registry: Any, _guard: Any) -> None:
        events.append("session-close")
        raise RuntimeError("session close failed")

    with pytest.raises(RuntimeError, match="graph close failed") as error:
        example._close_enabled_binding(
            SimpleNamespace(close_resident_graph_capture=fail_graph_close),
            object(),
            SimpleNamespace(close=fail_session_close),
            object(),
            object(),
        )

    assert events == ["graph-close", "session-close"]
    assert error.value.__cause__ is not None
    assert str(error.value.__cause__) == "session close failed"


def test_operation_failure_remains_primary_when_teardown_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chain cleanup failure without replacing the enabled-path failure."""
    example = _fresh_example()
    device = SimpleNamespace(native="cuda:1")
    session = SimpleNamespace(
        close=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("cleanup failed")
        )
    )
    registry = object()
    guard = object()
    resident_runtime = SimpleNamespace(
        gpu_session=SimpleNamespace(
            setup_resident_session=lambda *_args: session,
            ResidentStepGuard=lambda *_args: guard,
        ),
        gpu_resources=SimpleNamespace(
            GPUResourceRegistry=lambda _session: registry
        ),
    )
    runtime = SimpleNamespace(
        execution=SimpleNamespace(
            Backend=SimpleNamespace(WARP="warp"),
            Device=lambda *_args: device,
        ),
        graph_capture=SimpleNamespace(),
        warp=SimpleNamespace(),
        resident_example=SimpleNamespace(
            _load_enabled_runtime=lambda: resident_runtime
        ),
    )
    monkeypatch.setattr(example, "_qualified_native_cuda", lambda: "cuda:1")
    monkeypatch.setattr(example, "_load_capability_runtime", lambda: runtime)
    monkeypatch.setattr(example, "_load_enabled_runtime", lambda: runtime)
    monkeypatch.setattr(
        example, "_resolve_exact_capability", lambda *_args: object()
    )
    monkeypatch.setattr(
        example,
        "_build_cpu_state",
        lambda: (
            SimpleNamespace(
                volume=np.ones(1),
                masses=np.ones((1, 1, 1)),
                concentration=np.ones((1, 1)),
            ),
            SimpleNamespace(concentration=np.ones((1, 1))),
            object(),
        ),
    )
    monkeypatch.setattr(
        example,
        "_compose_request",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("operation failed")),
    )

    with pytest.raises(RuntimeError, match="operation failed") as error:
        example.run_example()
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "cleanup failed"


@pytest.mark.warp
@pytest.mark.cuda
def test_native_cuda_example_smoke_is_explicitly_capability_gated() -> None:
    """Run only a qualified native-CUDA example; Warp CPU never emulates it."""
    if os.getenv("PARTICULA_RUN_NATIVE_CAPTURE_EXAMPLE") != "1":
        pytest.skip("native graph-capture example smoke is opt-in")
    example = _fresh_example()
    native = example._qualified_native_cuda()
    if native is None:
        pytest.skip("native CUDA graph capture is explicitly unavailable")
    result = example.run_example()
    assert result.replay_count == 2
    assert result.invalidated and result.retired and result.renewed
    assert result.synchronized
    assert result.captured is not result.renewed_capture
    for observation in (result.gas_snapshot, result.saturation_snapshot):
        assert observation is not None
        assert observation.shape == (3, 1)
        assert observation.dtype == np.float64
        assert np.all(np.isfinite(observation))
