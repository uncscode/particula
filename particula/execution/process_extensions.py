"""Typed public preparation contracts for explicit resident process adapters.

Adapters are trusted numerical code. Declarations constrain exposed resources;
Python objects and device pointers are not a security boundary. Preparation may
allocate scratch. The retained operation must only enqueue work, check the
shared device failure flag, and preserve the fixed storage schema.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from particula.execution.extension_registry import (
    PreparedOperation,
    ProcessCapabilities,
)

if TYPE_CHECKING:
    from particula.aerosol import Aerosol


def _immutable(value: object) -> bool:
    """Admit immutable scalar/tuple configuration without CPU arrays."""
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return isfinite(value)
    return type(value) is tuple and all(_immutable(item) for item in value)


@dataclass(frozen=True)
class GPUProcessPlan:
    """Freeze one named implementation and its declared array requirements.

    Parameters contain only finite Python scalars and nested tuples. Persistent
    continuation belongs in prepared device state, never in these parameters.
    Reads may include native metadata; writes must name evolving native or
    registered fields. Instance names must be unique within one simulation.
    """

    name: str
    extension_id: str
    version: str
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    parameters: tuple[tuple[str, object], ...] = ()
    capabilities: ProcessCapabilities = ProcessCapabilities(resident=True)

    def __post_init__(self) -> None:
        """Reject mutable configuration and ambiguous resource declarations."""
        for value in (self.name, self.extension_id, self.version):
            if type(value) is not str or not value.strip():
                raise ValueError(
                    "extension identity and version must be nonempty"
                )
        for names in (self.reads, self.writes):
            if type(names) is not tuple or any(
                type(name) is not str or not name for name in names
            ):
                raise TypeError("array declarations must be tuples of names")
            if len(set(names)) != len(names):
                raise ValueError("duplicate array declaration")
        if type(self.parameters) is not tuple or any(
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not str
            or not pair[0]
            or not _immutable(pair[1])
            for pair in self.parameters
        ):
            raise ValueError("parameters must contain immutable named values")
        if len(dict(self.parameters)) != len(self.parameters):
            raise ValueError("duplicate parameter name")
        if type(self.capabilities) is not ProcessCapabilities:
            raise TypeError("capabilities must be ProcessCapabilities")
        if not self.capabilities.resident:
            raise ValueError("GPU adapters require resident capability")

    @property
    def values(self) -> Mapping[str, object]:
        """Return an immutable setup-only parameter mapping."""
        return MappingProxyType(dict(self.parameters))


@dataclass(frozen=True)
class GPUPreparationContext:
    """Expose only declared buffers plus explicit resident execution resources.

    ``clock[0]`` is the absolute start time for the current timestep in seconds.
    ``failure[0]`` is zero while healthy. Kernels must suppress work
    after any failure; extensions report their own failures with code 100 or
    above. Arrays have fixed shapes/dtypes and retain their device identities.
    """

    arrays: Mapping[str, object]
    device: str
    stream: object
    time_step: float
    clock: object
    failure: object
    logical_box_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Detach the map while retaining exact device array objects."""
        object.__setattr__(self, "arrays", MappingProxyType(dict(self.arrays)))


class GPUProcessAdapter(Protocol):
    """Validate and freeze on CPU, then prepare retained resident operations."""

    def plan(self, process: object, aerosol: Aerosol) -> GPUProcessPlan:
        """Validate requirements on detached CPU state without mutation."""
        ...

    def prepare(
        self, plan: GPUProcessPlan, context: GPUPreparationContext
    ) -> PreparedOperation:
        """Allocate once and return an operation retaining its resources."""
        ...


@dataclass(frozen=True)
class _RegisteredPlan:
    """Retain the implementation separately from frozen configuration."""

    configuration: GPUProcessPlan
    adapter: GPUProcessAdapter

    @property
    def family(self) -> str:
        """Use a distinct scheduling key for each custom instance."""
        return "extension:" + self.configuration.name

    @property
    def values(self) -> Mapping[str, object]:
        """Expose immutable parameters to shared admission checks."""
        return self.configuration.values
