"""Typed, setup-only dependencies for built-in resident process adapters.

The context contains resources and preparation results, not simulation lifecycle
methods. Built-ins cannot step, close, snapshot or restart their owning runner.
Warp buffers remain runtime-typed at the direct-kernel boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from particula.execution import Device
    from particula.execution.field_buffers import FieldBuffers
    from particula.execution.gpu_resources import GPUResourceRegistry
    from particula.execution.gpu_session import ResidentSession
    from particula.execution.prepared_processes import _ProcessPlan
    from particula.execution.process_extensions import _RegisteredPlan
    from particula.gpu.warp_types import (
        WarpEnvironmentData,
        WarpGasData,
        WarpParticleData,
    )


@dataclass
class BuiltinPreparationContext:
    """Explicit native resources and shared setup outputs for one preparation.

    Binding and auxiliary maps are shared with the owner to preserve resource
    identity. Only setup mutates these maps; retained operations capture their
    fixed buffers. ``refresh`` publishes the derived-state refresh operation.
    """

    wp: ModuleType
    session: ResidentSession
    registry: GPUResourceRegistry
    device: Device
    time_step: float
    failure: Any
    plans: tuple[_ProcessPlan | _RegisteredPlan, ...]
    fields: FieldBuffers
    bindings: dict[str, Any]
    auxiliary: dict[str, Any]
    observers: list[Callable[[], object]]
    refresh: Callable[[], None] | None = None

    @property
    def particles(self) -> WarpParticleData:
        """Expose the already validated native particle buffer layout."""
        return cast("WarpParticleData", self.session.particles)

    @property
    def gas(self) -> WarpGasData:
        """Expose the already validated full-gas buffer layout."""
        return cast("WarpGasData", self.session.gas)

    @property
    def environment(self) -> WarpEnvironmentData:
        """Expose the already validated box environment buffer layout."""
        return cast("WarpEnvironmentData", self.session.environment)
