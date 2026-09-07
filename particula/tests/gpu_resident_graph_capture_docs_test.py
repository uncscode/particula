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


def _fresh_example() -> Any:
    """Import a fresh example module without retaining prior test patches."""
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


def test_import_is_lazy_about_warp_and_concrete_capture_modules() -> None:
    """Keep optional native capture and concrete composition out of import time."""
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
    """Force disabling capture avoids Warp loading, fixture creation, and fallback."""
    example = _fresh_example()
    monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_NATIVE_CAPTURE", "1")
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
    """The script's explicit unavailable branch exits normally without a device."""
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
        (SimpleNamespace(get_devices=lambda: ["cpu"]), None),
        (
            SimpleNamespace(
                get_devices=lambda: ["cpu", "cuda:1"],
                capture_begin=lambda: None,
                capture_end=lambda: None,
            ),
            None,
        ),
        (
            SimpleNamespace(
                get_devices=lambda: ["cpu", "cuda:1", "cuda:2"],
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
    """Missing Warp, CUDA, or capture callables return the one unavailable result."""
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
    """Unexpected optional-runtime failures do not become unavailable fallbacks."""
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
    assert source.count("replay_captured_resident_graph(captured, 1.0)") == 1
    assert source.index("for _ in range(2):") < source.index(
        "synchronize_device"
    )
    assert source.index("synchronize_device") < source.index(
        "gas_output.numpy()"
    )
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
    assert "no CPU or Warp-CPU fallback" in index


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
        validate_capture_resource_set=lambda _requirements: "capture-set"
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

    captures = iter(("captured", "renewed-capture"))
    graph_capture = SimpleNamespace(
        GraphCaptureCapability=lambda *_args: object(),
        GraphCaptureAvailability=SimpleNamespace(AVAILABLE="available"),
        ResidentGraphCaptureBinding=lambda *_args: object(),
        _attach_resident_graph_capture_binding=lambda *_args: events.append(
            "attach"
        ),
        create_resident_graph_capture_signature=lambda _request: object(),
        create_graph_capture_lifecycle=lambda *_args: object(),
        qualify_prepared_resident_graph_capture=lambda *_args: object(),
        capture_prepared_resident_graph=lambda _qualification: next(captures),
        replay_captured_resident_graph=lambda captured,
        _duration: events.append(f"replay-{captured}"),
        retire_resident_graph_capture=lambda _binding: events.append("retire"),
        renew_resident_graph_capture=lambda *_args: object(),
        close_resident_graph_capture=lambda _binding: events.append(
            "capture-close"
        ),
    )
    runtime = SimpleNamespace(
        execution=SimpleNamespace(
            Backend=SimpleNamespace(WARP="warp"),
            Device=lambda _backend, native: SimpleNamespace(native=native),
        ),
        graph_capture=graph_capture,
        resident_scheduler=SimpleNamespace(prepare_resident_simulation=prepare),
        resident_example=SimpleNamespace(
            _load_enabled_runtime=lambda: resident_runtime
        ),
        warp=SimpleNamespace(
            synchronize_device=lambda _device: events.append("synchronize")
        ),
    )
    monkeypatch.setattr(example, "_qualified_native_cuda", lambda: "cuda:1")
    monkeypatch.setattr(example, "_load_enabled_runtime", lambda: runtime)
    monkeypatch.setattr(
        example,
        "_build_cpu_state",
        lambda: (object(), object(), object()),
    )
    monkeypatch.setattr(
        example,
        "_compose_request",
        lambda *_args: (request, gas_output, saturation_output),
    )
    monkeypatch.setattr(
        example,
        "_WarpNativeCaptureAdapter",
        lambda *_args: object(),
    )

    result = example.run_example()

    assert result.replay_count == 2
    assert result.invalidated and result.retired and result.renewed
    assert result.synchronized
    assert result.captured == "captured"
    assert result.renewed_capture == "renewed-capture"
    assert result.captured is not result.renewed_capture
    np.testing.assert_array_equal(result.gas_snapshot, gas_snapshot)
    np.testing.assert_array_equal(
        result.saturation_snapshot, saturation_snapshot
    )
    assert events == [
        "attach",
        "initialize",
        "prepare",
        "replay-captured",
        "replay-captured",
        "synchronize",
        "invalidated",
        "retire",
        "prepare",
        "capture-close",
        "session-close",
    ]


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
