"""Regression coverage for the lazy multi-timestep resident-loop example."""

from __future__ import annotations

import builtins
import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE = _ROOT / "docs" / "Examples" / "gpu_resident_multi_timestep.py"
_DISABLED = [
    "Canonical path: docs/Examples/gpu_resident_multi_timestep.py",
    "Warp is unavailable or disabled; install warp-lang or enable Warp.",
    "No CPU fallback ran; no fixture, upload, diagnostics, or restart ran.",
]


@pytest.fixture
def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import a fresh example while proving its module scope is Warp-free."""
    blocked = {
        "warp",
        "particula.gpu",
        "particula.execution.gpu_session",
        "particula.execution.gpu_resources",
        "particula.execution.checkpoint",
        "particula.execution.resident_scheduler",
    }
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        if name in blocked:
            pytest.fail(f"example imported {name} eagerly")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.syspath_prepend(str(_EXAMPLE.parent))
    sys.modules.pop("gpu_resident_multi_timestep", None)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = importlib.import_module("gpu_resident_multi_timestep")
    yield module
    sys.modules.pop("gpu_resident_multi_timestep", None)


def test_forced_disable_runs_no_enabled_work(
    example_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forced disable produces actionable no-work guidance."""
    monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", "1")
    monkeypatch.setattr(
        example_module, "_load_enabled_runtime", lambda: pytest.fail("loader")
    )
    monkeypatch.setattr(
        example_module, "_build_cpu_state", lambda: pytest.fail("fixture")
    )

    result = example_module.run_example()

    assert result.output == _DISABLED
    assert result.session is result.registry is result.guard is None
    assert (
        result.checkpoint
        is result.restarted
        is result.terminal_checkpoint
        is None
    )
    assert result.gas_snapshot is result.saturation_snapshot is None
    assert result.source_steps == result.restarted_steps == 0


def test_missing_top_level_warp_runs_no_enabled_work(
    example_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a missing top-level Warp module selects the disabled path."""
    monkeypatch.delenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", raising=False)
    monkeypatch.setattr(
        example_module.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError(name="warp")),
    )
    monkeypatch.setattr(
        example_module, "_load_enabled_runtime", lambda: pytest.fail("loader")
    )
    assert example_module.run_example().output == _DISABLED


@pytest.mark.parametrize(
    "error",
    [ImportError("enabled Warp failed"), ModuleNotFoundError(name="other")],
)
def test_broken_warp_import_propagates(
    example_module: Any, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """Enabled import errors never become fallback output."""
    monkeypatch.delenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", raising=False)
    monkeypatch.setattr(
        example_module.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(error),
    )
    with pytest.raises(type(error)) as raised:
        example_module.run_example()
    assert raised.value is error


def test_forced_disabled_script_has_exact_output() -> None:
    """The standalone disabled script prints deterministic guidance."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(_EXAMPLE)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PARTICULA_EXAMPLE_FORCE_NO_WARP": "1"},
        timeout=10,
    )
    assert result.stdout == "\n".join(_DISABLED) + "\n"


def test_loader_requests_only_concrete_resident_seams(
    example_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enabled loading requests the documented seams, never direct kernels."""
    names = (
        "warp",
        "particula.execution",
        "particula.execution.availability",
        "particula.execution.gpu_session",
        "particula.execution.gpu_resources",
        "particula.execution.checkpoint",
        "particula.execution.process_graph",
        "particula.execution.scheduler",
        "particula.execution.resident_scheduler",
        "particula.execution.diagnostics",
        "particula.execution.state_updates",
        "particula.execution.process_adapters",
        "particula.execution.communication",
        "particula.execution.resident_communication",
        "particula.execution.adapters.condensation",
        "particula.execution.adapters.coagulation",
        "particula.gpu.kernels.thermodynamics",
        "particula.gpu.kernels.wall_loss",
        "particula.gpu.kernels.nucleation",
    )
    calls: list[str] = []

    def load(name: str) -> Any:
        calls.append(name)
        return SimpleNamespace()

    monkeypatch.setattr(example_module.importlib, "import_module", load)

    example_module._load_enabled_runtime()

    assert calls == list(names)
    assert "particula.gpu" not in calls
    assert not any(name.endswith("_step_gpu") for name in calls)


@pytest.mark.parametrize(
    "error", [ImportError("resident"), RuntimeError("bad")]
)
def test_enabled_loader_error_propagates_without_fixture(
    example_module: Any, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """An enabled loader failure cannot become a disabled fallback result."""
    monkeypatch.setattr(example_module, "_warp_enabled", lambda: True)
    monkeypatch.setattr(
        example_module,
        "_load_enabled_runtime",
        lambda: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        example_module, "_build_cpu_state", lambda: pytest.fail("fixture")
    )

    with pytest.raises(type(error)) as raised:
        example_module.run_example()

    assert raised.value is error


def _availability_runtime(error: Exception) -> SimpleNamespace:
    """Build the minimal enabled runtime needed to reach availability."""
    execution = SimpleNamespace(
        Backend=SimpleNamespace(WARP="warp"),
        Device=lambda backend, device: (backend, device),
        CapabilityRequirements=lambda values: values,
        Process=lambda name: name,
        ExecutionRequest=lambda *values: values,
        CapabilityDeclaration=lambda *values: values,
        CapabilityMatrix=lambda values: values,
    )
    return SimpleNamespace(
        execution=execution,
        availability=SimpleNamespace(
            resolve_availability=lambda *_args: (_ for _ in ()).throw(error)
        ),
    )


def test_availability_failure_precedes_fixture_and_setup(
    example_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolver failure propagates before constructing or uploading CPU state."""
    error = RuntimeError("device unavailable")
    monkeypatch.setattr(example_module, "_warp_enabled", lambda: True)
    monkeypatch.setattr(
        example_module,
        "_load_enabled_runtime",
        lambda: _availability_runtime(error),
    )
    monkeypatch.setattr(
        example_module, "_build_cpu_state", lambda: pytest.fail("fixture")
    )

    with pytest.raises(RuntimeError) as raised:
        example_module.run_example()

    assert raised.value is error


def test_setup_failure_propagates_without_checkpoint_or_restart(
    example_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setup failure has no fallback, checkpoint, restart, or finalization."""
    error = RuntimeError("upload failed")
    runtime = _availability_runtime(error)
    runtime.availability = SimpleNamespace(
        resolve_availability=lambda request, _matrix: SimpleNamespace(
            request=request
        )
    )
    runtime.gpu_session = SimpleNamespace(
        setup_resident_session=lambda *_args: (_ for _ in ()).throw(error)
    )
    monkeypatch.setattr(example_module, "_warp_enabled", lambda: True)
    monkeypatch.setattr(
        example_module, "_load_enabled_runtime", lambda: runtime
    )

    with pytest.raises(RuntimeError) as raised:
        example_module.run_example()

    assert raised.value is error


def test_writer_dispatch_failure_propagates_after_guard_close(
    example_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post-writer failure closes its token and faults rather than retries."""
    error = RuntimeError("writer failed")

    class Session:
        """Provide the lifecycle field observed by the test scheduler."""

        lifecycle = "active"

    class Guard:
        """Record the closed guard state after the failed dispatch."""

        closed = False

    session = Session()
    guard = Guard()
    runtime = _availability_runtime(error)
    runtime.availability = SimpleNamespace(
        resolve_availability=lambda request, _matrix: SimpleNamespace(
            request=request
        )
    )
    runtime.gpu_session = SimpleNamespace(
        setup_resident_session=lambda *_args: session,
        ResidentStepGuard=lambda *_args: guard,
    )
    runtime.gpu_resources = SimpleNamespace(
        GPUResourceRegistry=lambda _: object()
    )

    class Scheduler:
        """Model the documented post-writer resident failure boundary."""

        def __init__(self, _request: object) -> None:
            """Retain no state for the failing scheduler double."""

        def execute(self, _duration: float) -> None:
            """Fault the session and close the guard before propagation."""
            session.lifecycle = "faulted"
            guard.closed = True
            raise error

    runtime.resident_scheduler = SimpleNamespace(
        ResidentSimulationScheduler=Scheduler
    )
    monkeypatch.setattr(example_module, "_warp_enabled", lambda: True)
    monkeypatch.setattr(
        example_module, "_load_enabled_runtime", lambda: runtime
    )
    monkeypatch.setattr(
        example_module, "_request", lambda *_args: (object(), None, None)
    )

    with pytest.raises(RuntimeError) as raised:
        example_module.run_example()

    assert raised.value is error
    assert session.lifecycle == "faulted"
    assert guard.closed


@pytest.mark.warp
def test_real_warp_cpu_example_has_resident_lifecycle_observations() -> None:
    """The enabled example performs its documented multi-step lifecycle."""
    pytest.importorskip("warp")
    spec = importlib.util.spec_from_file_location(
        "resident_multistep_real", _EXAMPLE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        result = module.run_example()
    finally:
        sys.modules.pop(spec.name, None)

    restarted_session, restarted_registry, restarted_guard = result.restarted
    assert result.source_steps == 2
    assert result.restarted_steps == 1
    assert result.gas_snapshot.shape == (3, 1)
    assert result.saturation_snapshot.shape == (3, 1)
    assert result.checkpoint.lifecycle.value == "active"
    assert result.session.lifecycle.value == "finalized"
    assert result.terminal_checkpoint is result.session.finalize(
        result.registry, result.guard
    )
    result.guard.assert_step_closed()
    restarted_guard.assert_step_closed()
    assert restarted_session is not result.session
    assert restarted_registry is not result.registry
    assert restarted_guard is not result.guard


def test_example_documents_resident_ownership_limits() -> None:
    """The published example retains its no-fallback ownership language."""
    text = _EXAMPLE.read_text(encoding="utf-8")
    for phrase in (
        "caller-owned diagnostics",
        "manual, exact-device",
        "No CPU fallback",
        "automatic restart",
        "graph capture",
        "exact cross-backend RNG replay",
    ):
        assert phrase in text
    assert "_step_gpu" not in text
