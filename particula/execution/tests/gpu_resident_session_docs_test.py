"""Regression coverage for the lazy resident-session documentation example."""

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

_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLE = _ROOT / "docs" / "Examples" / "gpu_resident_session.py"
_DISABLED = [
    "Canonical path: docs/Examples/gpu_resident_session.py",
    "CPU fixture: not constructed because Warp is unavailable or disabled.",
    "Warp is unavailable or disabled; no resident session was created.",
]


@pytest.fixture
def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import a fresh example while proving import-time GPU safety."""
    blocked = {
        "warp",
        "particula.gpu",
        "particula.execution.gpu_session",
        "particula.execution.gpu_resources",
        "particula.execution.checkpoint",
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
    sys.modules.pop("gpu_resident_session", None)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = importlib.import_module("gpu_resident_session")
    yield module
    sys.modules.pop("gpu_resident_session", None)


def test_forced_disable_skips_loader_and_fixture(
    example_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forced disable is a deterministic no-work path."""
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


def test_missing_warp_skips_fixture(
    example_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a missing Warp import selects the no-work route."""
    monkeypatch.delenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", raising=False)
    monkeypatch.setattr(
        example_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )
    monkeypatch.setattr(
        example_module, "_build_cpu_state", lambda: pytest.fail("fixture")
    )

    assert example_module.run_example().output == _DISABLED


def test_forced_disabled_script_has_exact_stdout() -> None:
    """The standalone forced-disabled command exits successfully."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(_EXAMPLE)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PARTICULA_EXAMPLE_FORCE_NO_WARP": "1"},
        timeout=10,
    )
    assert result.stdout == "\n".join(_DISABLED) + "\n"


def test_loader_orders_concrete_imports_without_gpu_package(
    example_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The enabled loader requests only the documented concrete seams."""
    modules = {
        "warp": object(),
        "particula.execution": SimpleNamespace(
            Backend=object(), Device=object()
        ),
        "particula.execution.gpu_session": SimpleNamespace(
            ResidentLifecycle=object(),
            setup_resident_session=object(),
            ResidentStepGuard=object(),
        ),
        "particula.execution.gpu_resources": SimpleNamespace(
            GPUResourceRegistry=object()
        ),
        "particula.execution.checkpoint": SimpleNamespace(
            restart_resident_session=object()
        ),
    }
    calls: list[str] = []

    def load(name: str) -> object:
        calls.append(name)
        return modules[name]

    monkeypatch.setattr(example_module.importlib, "import_module", load)
    example_module._load_enabled_runtime()
    assert calls == list(modules)
    assert "particula.gpu" not in calls


@pytest.mark.parametrize(
    "error", [ImportError("resident"), RuntimeError("bad")]
)
def test_enabled_loader_errors_propagate_without_fixture_or_output(
    example_module: Any, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """Selected enabled failures never become disabled-path success output."""
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


def test_main_propagates_an_enabled_loader_error(
    example_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command entry point does not hide selected enabled-path failures."""
    error = RuntimeError("resident runtime failed")
    monkeypatch.setattr(example_module, "_warp_enabled", lambda: True)
    monkeypatch.setattr(
        example_module,
        "_load_enabled_runtime",
        lambda: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        example_module, "_build_cpu_state", lambda: pytest.fail("fixture")
    )

    with pytest.raises(RuntimeError) as raised:
        example_module.main()

    assert raised.value is error


def test_lifecycle_documentation_preserves_published_boundaries() -> None:
    """Feature and roadmap text retain the concrete-only lifecycle contract."""
    feature = (
        _ROOT / "docs" / "Features" / "data-containers-and-gpu-foundations.md"
    ).read_text()
    checkpoint = (
        _ROOT / "docs" / "Features" / "gpu_resident_checkpoints.md"
    ).read_text()
    roadmap = (
        _ROOT / "docs" / "Features" / "Roadmap" / "data-oriented-gpu.md"
    ).read_text()
    architecture = (
        _ROOT / ".opencode" / "guides" / "architecture_reference.md"
    ).read_text()
    agents = (_ROOT / "AGENTS.md").read_text()
    combined = "\n".join((feature, checkpoint, roadmap, architecture, agents))
    for phrase in (
        "particula.execution.gpu_session",
        "particula.execution.gpu_resources",
        "particula.execution.checkpoint",
        "nonterminal and returns",
        "canonical bytes",
        "schema version `1`",
        "exactly equal `Device`",
        "E7-F5",
        "E7-F7",
        "E7-F8",
        "gpu_resident_session.py",
    ):
        assert phrase in combined
    assert "defers E7-F4 resident sessions" not in roadmap


@pytest.mark.warp
def test_real_warp_cpu_lifecycle_example() -> None:
    """The real Warp CPU route preserves the published identity lifecycle."""
    pytest.importorskip("warp")
    spec = importlib.util.spec_from_file_location(
        "resident_example_real", _EXAMPLE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)

    result = module.run_example()
    restarted_session, restarted_registry, restarted_guard = result.restarted
    assert result.session.lifecycle.value == "finalized"
    assert result.checkpoint.lifecycle.value == "active"
    assert result.terminal_checkpoint is result.session.finalize(
        result.registry, result.guard
    )
    result.guard.assert_step_closed()
    restarted_guard.assert_step_closed()
    assert restarted_session is not result.session
    assert restarted_registry is not result.registry
    assert restarted_guard is not result.guard
    assert restarted_session.particles is not result.session.particles
    assert restarted_session.gas is not result.session.gas
    assert restarted_session.environment is not result.session.environment
    first_view = restarted_registry.acquire_wall_loss()
    second_view = restarted_registry.acquire_wall_loss()
    assert first_view is second_view
    assert first_view.rng_states is second_view.rng_states
    assert all(
        phrase in result.output
        for phrase in (
            "Checkpoint is nonterminal; finalization is terminal and cached.",
            "Restart is explicit, same-device, and never automatic.",
            "Inspection is lossy; canonical checkpoint bytes are restart authority.",
            "Checkpoint schema version 1 compatibility is exact and fail-closed.",
            "Exclusions: no scheduling, transport, fallback, or physics orchestration.",
        )
    )
