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
class GPUStateSpec:
    """Declare a fixed-shape persistent adapter array included in checkpoints.

    Scratch allocated by an adapter is not continuation state. Counters, random
    words and evolving parameters must use these declared arrays instead.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str = "float64"
    initial_value: float | int = 0

    def __post_init__(self) -> None:
        """Validate a portable primitive dtype and fixed nonempty shape."""
        import numpy as np

        if (
            type(self.name) is not str
            or not self.name.isidentifier()
            or self.name.startswith("__")
        ):
            raise ValueError("state name must be an identifier")
        if (
            type(self.shape) is not tuple
            or not 1 <= len(self.shape) <= 4
            or any(type(n) is not int or n < 1 for n in self.shape)
        ):
            raise ValueError(
                "state shape must contain positive integer lengths"
            )
        if self.dtype not in ("float64", "int32", "int64", "uint32"):
            raise ValueError("unsupported persistent state dtype")
        if type(self.initial_value) not in (float, int) or not isfinite(
            self.initial_value
        ):
            raise ValueError("state initial value must be finite")
        if self.dtype != "float64":
            bounds = np.iinfo(self.dtype)
            if (
                int(self.initial_value) != self.initial_value
                or not bounds.min <= self.initial_value <= bounds.max
            ):
                raise ValueError("state initial value exceeds integer dtype")


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
    state: tuple[GPUStateSpec, ...] = ()
    execution_mode: str = "resident"
    stochastic: bool = False

    def __post_init__(self) -> None:  # noqa: C901
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
        if type(self.state) is not tuple or any(
            type(spec) is not GPUStateSpec for spec in self.state
        ):
            raise TypeError("state must contain GPUStateSpec descriptors")
        if len({spec.name for spec in self.state}) != len(self.state):
            raise ValueError("duplicate state name")
        if type(self.stochastic) is not bool:
            raise TypeError("stochastic must be a bool")
        if self.execution_mode == "resident":
            if not self.capabilities.resident:
                raise ValueError("GPU adapters require resident capability")
        elif self.execution_mode == "cpu_transfer":
            if (
                self.capabilities.resident
                or self.capabilities.capture
                or self.state
                or self.stochastic
            ):
                raise ValueError(
                    "CPU transfers cannot declare resident resources"
                )
        else:
            raise ValueError("unsupported process execution mode")

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
    For a stochastic plan, ``rng`` holds one uint32 Warp random-state word per
    logical box. Kernels load a word, draw from it, and store the advanced word.
    One thread per box owns that update; CPU/GPU bitwise parity is not implied.
    """

    arrays: Mapping[str, object]
    device: str
    stream: object
    time_step: float
    clock: object
    failure: object
    logical_box_ids: tuple[str, ...]
    state: Mapping[str, object]
    rng: object | None = None

    def __post_init__(self) -> None:
        """Detach the map while retaining exact device array objects."""
        object.__setattr__(self, "arrays", MappingProxyType(dict(self.arrays)))
        object.__setattr__(self, "state", MappingProxyType(dict(self.state)))


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
