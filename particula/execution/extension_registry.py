"""Explicit, simulation-local registration of process implementations.

Registration performs no runtime/device initialization. Exact type matching
prevents an adapter from silently accepting changed subclass semantics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

AdapterT = TypeVar("AdapterT")


@dataclass(frozen=True)
class ProcessCapabilities:
    """Declare independently supported extension execution boundaries."""

    resident: bool = False
    capture: bool = False
    checkpoint: bool = False

    def __post_init__(self) -> None:
        """Reject contradictory or ambiguous declarations."""
        for value in (self.resident, self.capture, self.checkpoint):
            if type(value) is not bool:
                raise TypeError("capabilities must be Python bool values")
        if self.capture and not self.resident:
            raise ValueError("capture requires resident execution")


class ProcessRegistry(Generic[AdapterT]):
    """Own exact-type adapter registrations without global side effects.

    A prepared simulation takes a snapshot to isolate its registrations from
    subsequent edits to the caller's registry. Adapter configuration itself
    must be frozen separately when creating a process plan.
    """

    def __init__(self) -> None:
        """Create an empty caller-owned registry."""
        self._adapters: dict[type, AdapterT] = {}
        self._frozen = False

    def register(self, process_type: type, adapter: AdapterT) -> None:
        """Register one implementation, rejecting implicit replacement.

        Args:
            process_type: Exact concrete process class.
            adapter: Implementation retained by identity.

        Raises:
            TypeError: If the key is not a class or the adapter is None.
            ValueError: If the class already has a registration.
            RuntimeError: If this is an immutable preparation snapshot.
        """
        if self._frozen:
            raise RuntimeError("prepared process registry is frozen")
        if not isinstance(process_type, type):
            raise TypeError("process_type must be a class")
        if adapter is None:
            raise TypeError("adapter must not be None")
        if process_type in self._adapters:
            raise ValueError("process type already registered")
        self._adapters[process_type] = adapter

    def has_type(self, process_type: type) -> bool:
        """Report exact registration without invoking user adapter code."""
        return process_type in self._adapters

    def resolve(self, process: object) -> AdapterT:
        """Resolve the exact type without inheriting parent registrations."""
        try:
            return self._adapters[type(process)]
        except KeyError:
            raise ValueError(
                f"{type(process).__name__} has no registered implementation"
            ) from None

    def snapshot(self) -> ProcessRegistry[AdapterT]:
        """Retain an immutable copy of the registration map."""
        result: ProcessRegistry[AdapterT] = ProcessRegistry()
        result._adapters = self._adapters.copy()
        result._frozen = True
        return result


@dataclass(frozen=True)
class PreparedOperation:
    """Retain a named operation and the resources required for its lifetime.

    The handler enqueues already prepared work. Resource descriptors remain
    visible to capture identity validation rather than hidden in closures.
    """

    name: str
    handler: Callable[[], object]
    resources: tuple[object, ...] = ()
    capabilities: ProcessCapabilities = ProcessCapabilities()

    def __post_init__(self) -> None:
        """Validate descriptors before any execution can occur."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("operation name must be nonempty")
        if not callable(self.handler):
            raise TypeError("operation handler must be callable")
        if type(self.resources) is not tuple:
            raise TypeError("operation resources must be a tuple")
        if type(self.capabilities) is not ProcessCapabilities:
            raise TypeError(
                "operation capabilities must be ProcessCapabilities"
            )

    def __call__(self) -> None:
        """Delegate exactly once without recovery, transfer or allocation."""
        self.handler()
