"""Demonstrate the bounded GPU-resident session lifecycle and checkpoints.

The optional Warp runtime and every concrete resident seam are imported only on
the enabled path. The enabled path uploads deterministic CPU carriers once,
then demonstrates guard ownership, a nonterminal checkpoint, an explicit
same-device restart, and cached terminal finalization. This example does not
schedule processes, launch physics, select a backend, transport state, or
provide fallback.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData

_FORCE_NO_WARP_ENV = "PARTICULA_EXAMPLE_FORCE_NO_WARP"


@dataclass
class ExampleRun:
    """Retain stable observations from the optional lifecycle demonstration.

    All enabled-only fields remain ``None`` when Warp is unavailable or disabled.

    Attributes:
        output: Deterministic, address-free status lines.
        session: Source resident session after the lifecycle demonstration.
        registry: Resource registry bound to ``session``.
        guard: Closed source step guard bound to ``session`` and ``registry``.
        checkpoint: Nonterminal checkpoint captured while ``session`` is active.
        restarted: Fresh ``(session, registry, guard)`` restarted binding.
        terminal_checkpoint: Cached terminal checkpoint from source finalization.
    """

    output: list[str]
    session: Any | None = None
    registry: Any | None = None
    guard: Any | None = None
    checkpoint: Any | None = None
    restarted: tuple[Any, Any, Any] | None = None
    terminal_checkpoint: Any | None = None


def _warp_enabled() -> bool:
    """Return whether the optional Warp runtime is enabled and importable.

    A force-disable environment variable short-circuits before the Warp probe.
    Only a missing top-level Warp installation is treated as an unavailable
    optional runtime. Broken enabled imports propagate to make their failure
    visible.

    Returns:
        ``True`` when Warp is enabled and imports successfully; otherwise,
        ``False``.
    """
    if os.getenv(_FORCE_NO_WARP_ENV) == "1":
        return False
    try:
        importlib.import_module("warp")
    except ModuleNotFoundError as error:
        if error.name == "warp":
            return False
        raise
    return True


def _load_enabled_runtime() -> SimpleNamespace:
    """Load the enabled-only Warp and concrete resident-session modules.

    This example does not directly import ``particula.gpu``. Its concrete
    ``gpu_resources`` dependency may import that package transitively.

    Returns:
        Namespace containing Warp plus the resident-session types and helpers.

    Raises:
        ImportError: If an enabled-only runtime dependency cannot be imported.
    """
    wp = importlib.import_module("warp")
    execution = importlib.import_module("particula.execution")
    session = importlib.import_module("particula.execution.gpu_session")
    resources = importlib.import_module("particula.execution.gpu_resources")
    checkpoint = importlib.import_module("particula.execution.checkpoint")
    return SimpleNamespace(
        wp=wp,
        Backend=execution.Backend,
        Device=execution.Device,
        ResidentLifecycle=session.ResidentLifecycle,
        setup_resident_session=session.setup_resident_session,
        ResidentStepGuard=session.ResidentStepGuard,
        GPUResourceRegistry=resources.GPUResourceRegistry,
        restart_resident_session=checkpoint.restart_resident_session,
    )


def _build_cpu_state() -> tuple[ParticleData, GasData, EnvironmentData]:
    """Build deterministic one-box CPU carriers for the single upload.

    Returns:
        Particle, gas, and environment carriers in that setup order.
    """
    particles = ParticleData(
        masses=np.array([[[1.0e-18]]], dtype=np.float64),
        concentration=np.array([[1.0]], dtype=np.float64),
        charge=np.array([[0.0]], dtype=np.float64),
        density=np.array([1000.0], dtype=np.float64),
        volume=np.array([1.0e-6], dtype=np.float64),
    )
    gas = GasData(
        name=["water"],
        molar_mass=np.array([0.018], dtype=np.float64),
        concentration=np.array([[1.0e-9]], dtype=np.float64),
        partitioning=np.array([True], dtype=np.bool_),
    )
    environment = EnvironmentData(
        temperature=np.array([298.15], dtype=np.float64),
        pressure=np.array([101325.0], dtype=np.float64),
        saturation_ratio=np.array([[1.0]], dtype=np.float64),
    )
    return particles, gas, environment


def _disabled_output() -> list[str]:
    """Return deterministic, address-free status for the no-Warp path.

    Returns:
        Status lines confirming that no CPU fixture or resident session exists.
    """
    return [
        "Canonical path: docs/Examples/gpu_resident_session.py",
        "CPU fixture: not constructed because Warp is unavailable or disabled.",
        "Warp is unavailable or disabled; no resident session was created.",
    ]


def run_example(device: str = "cpu") -> ExampleRun:
    """Run lifecycle bookkeeping without executing a physical process.

    Enabled-loader errors intentionally propagate. The two zero-duration guard
    tokens demonstrate ownership only, not scheduling or process ordering.

    Args:
        device: Warp device identifier for the explicit restart target. Defaults
            to Warp CPU.

    Returns:
        Deterministic disabled-path status when Warp is unavailable or disabled;
        otherwise, lifecycle observations for the source and restarted bindings.

    Raises:
        ImportError: If an enabled-only runtime dependency cannot be imported.
        RuntimeError: If the resident lifecycle rejects an invalid transition.
        ValueError: If the selected device or resident state is invalid.
    """
    if not _warp_enabled():
        return ExampleRun(output=_disabled_output())
    runtime = _load_enabled_runtime()
    particles, gas, environment = _build_cpu_state()
    selected_device = runtime.Device(runtime.Backend.WARP, device)
    session = runtime.setup_resident_session(
        particles, gas, environment, selected_device
    )
    registry = runtime.GPUResourceRegistry(session)
    guard = runtime.ResidentStepGuard(session, registry)
    for _ in range(2):
        token = guard.begin_step(0.0)
        guard.complete_step(token)
        guard.assert_step_closed()

    checkpoint = session.checkpoint(registry, guard)
    assert session.lifecycle is runtime.ResidentLifecycle.ACTIVE
    restarted = runtime.restart_resident_session(checkpoint, selected_device)
    restarted_session, restarted_registry, restarted_guard = restarted
    restarted_guard.assert_step_closed()
    assert restarted_session is not session
    assert restarted_registry is not registry
    assert restarted_guard is not guard
    assert restarted_session.particles is not session.particles
    assert restarted_session.gas is not session.gas
    assert restarted_session.environment is not session.environment
    terminal_checkpoint = session.finalize(registry, guard)
    assert session.lifecycle is runtime.ResidentLifecycle.FINALIZED
    assert session.finalize(registry, guard) is terminal_checkpoint
    return ExampleRun(
        output=[
            "Canonical path: docs/Examples/gpu_resident_session.py",
            "Finalization terminalizes its source; returned checkpoint is ACTIVE "
            "and cached.",
            "Restart is explicit, same-device, and never automatic.",
            "Inspection is lossy; canonical checkpoint bytes are restart authority.",
            "Checkpoint schema version 1 compatibility is exact and fail-closed.",
            "Exclusions: no scheduling, transport, fallback, or physics orchestration.",
        ],
        session=session,
        registry=registry,
        guard=guard,
        checkpoint=checkpoint,
        restarted=restarted,
        terminal_checkpoint=terminal_checkpoint,
    )


def main() -> None:
    """Run the example and print its deterministic status lines.

    Raises:
        ImportError: If an enabled-only runtime dependency cannot be imported.
        RuntimeError: If the resident lifecycle rejects an invalid transition.
        ValueError: If the selected device or resident state is invalid.
    """
    for line in run_example().output:
        print(line)


if __name__ == "__main__":
    main()
