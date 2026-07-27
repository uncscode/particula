"""Declare dependency-neutral execution capability metadata and selection.

This module validates immutable backend, device, process, and capability
declarations. An ``ExecutionContext`` capability-validates a typed request and
returns one exact context-local adapter. Selection does not invoke adapters,
load or probe optional backends, resolve native devices, transfer state, or
define execution state and result contracts.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class Backend(str, Enum):
    """Identify a declared execution backend without loading it.

    Attributes:
        CPU: The declared CPU backend.
        WARP: The declared Warp backend.
    """

    CPU = "cpu"
    WARP = "warp"


@dataclass(frozen=True)
class Device:
    """Declare an immutable backend and opaque native device identifier.

    The native identifier is retained verbatim and is never parsed, resolved,
    or checked against backend availability.

    Args:
        backend: Declared backend for the device.
        native: Nonempty native identifier without surrounding whitespace.

    Raises:
        TypeError: If ``backend`` is not a Backend or ``native`` is not a str.
        ValueError: If ``native`` is empty or has surrounding whitespace.
    """

    backend: Backend
    native: str

    def __post_init__(self) -> None:
        """Validate the declared backend and opaque native identifier.

        Raises:
            TypeError: If the backend or native identifier has the wrong type.
            ValueError: If the native identifier is empty or padded.
        """
        if not isinstance(self.backend, Backend):
            raise TypeError("Device.backend must be a Backend.")
        if not isinstance(self.native, str):
            raise TypeError("Device.native must be a str.")
        if not self.native or self.native != self.native.strip():
            raise ValueError(
                "Device.native must be a nonempty str without surrounding "
                "whitespace."
            )


@dataclass(frozen=True)
class Process:
    """Declare an immutable, validated process name.

    Args:
        name: Lowercase identifier-style process name.

    Raises:
        TypeError: If ``name`` is not a str.
        ValueError: If ``name`` is not a lowercase identifier-style name.
    """

    name: str

    def __post_init__(self) -> None:
        """Validate the process name.

        Raises:
            TypeError: If the name is not a str.
            ValueError: If the name is not a lowercase identifier-style name.
        """
        _validate_name(self.name, "Process.name")


@dataclass(frozen=True)
class Capability:
    """Declare an immutable, validated capability name.

    Args:
        name: Lowercase identifier-style capability name.

    Raises:
        TypeError: If ``name`` is not a str.
        ValueError: If ``name`` is not a lowercase identifier-style name.
    """

    name: str

    def __post_init__(self) -> None:
        """Validate the capability name.

        Raises:
            TypeError: If the name is not a str.
            ValueError: If the name is not a lowercase identifier-style name.
        """
        _validate_name(self.name, "Capability.name")


def _validate_name(value: object, field_name: str) -> None:
    """Validate a lowercase identifier-style metadata name.

    Args:
        value: Candidate name to validate.
        field_name: Qualified field name used in error messages.

    Raises:
        TypeError: If ``value`` is not a str.
        ValueError: If ``value`` does not match the supported name pattern.
    """
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a str.")
    if not _NAME_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must match ^[a-z][a-z0-9_]*$.")


@dataclass(frozen=True)
class CapabilityRequirements:
    """Declare an immutable exact set of required capabilities.

    The empty set is valid. Requirements are declarations, not composable
    features inferred by the matrix.

    Args:
        values: Exact frozenset of capability declarations.

    Raises:
        TypeError: If ``values`` is not a frozenset of Capability instances.
    """

    values: frozenset[Capability]

    def __post_init__(self) -> None:
        """Validate the exact immutable capability collection type.

        Raises:
            TypeError: If the collection or one of its members is invalid.
        """
        if type(self.values) is not frozenset:
            raise TypeError(
                "CapabilityRequirements.values must be a frozenset."
            )
        if not all(isinstance(value, Capability) for value in self.values):
            raise TypeError(
                "CapabilityRequirements.values must contain only Capability "
                "instances."
            )


@dataclass(frozen=True)
class CapabilityDeclaration:
    """Declare immutable capability support for one device and process.

    Args:
        device: Device receiving the declaration.
        process: Process supported by the device.
        requirements: Exact required capability set for the support entry.

    Raises:
        TypeError: If any field is not its declared metadata type.
    """

    device: Device
    process: Process
    requirements: CapabilityRequirements

    def __post_init__(self) -> None:
        """Validate the declaration's typed value objects.

        Raises:
            TypeError: If any declaration field has the wrong type.
        """
        if not isinstance(self.device, Device):
            raise TypeError("CapabilityDeclaration.device must be a Device.")
        if not isinstance(self.process, Process):
            raise TypeError("CapabilityDeclaration.process must be a Process.")
        if not isinstance(self.requirements, CapabilityRequirements):
            raise TypeError(
                "CapabilityDeclaration.requirements must be a "
                "CapabilityRequirements."
            )


@dataclass(frozen=True)
class CapabilityMatrix:
    """Provide immutable, pure lookup of exact capability declarations.

    Nonempty requirements must match one complete declaration. Empty
    requirements are supported when at least one declaration has the requested
    device and process. Lookup neither probes backend availability nor selects
    an execution adapter.

    Args:
        declarations: Exact frozenset of capability declarations.

    Raises:
        TypeError: If ``declarations`` is not a frozenset of declarations.
    """

    declarations: frozenset[CapabilityDeclaration]

    def __post_init__(self) -> None:
        """Validate the exact immutable declaration collection type.

        Raises:
            TypeError: If the collection or one of its members is invalid.
        """
        if type(self.declarations) is not frozenset:
            raise TypeError(
                "CapabilityMatrix.declarations must be a frozenset."
            )
        if not all(
            isinstance(declaration, CapabilityDeclaration)
            for declaration in self.declarations
        ):
            raise TypeError(
                "CapabilityMatrix.declarations must contain only "
                "CapabilityDeclaration instances."
            )

    def supports(
        self,
        device: Device,
        process: Process,
        requirements: CapabilityRequirements,
    ) -> bool:
        """Return whether the matrix exactly declares a capability request.

        Args:
            device: Device requested for the process.
            process: Process requested on the device.
            requirements: Exact capability requirements for the request.

        Returns:
            True if the request is declared by this matrix; otherwise, False.

        Raises:
            TypeError: If an argument is not its declared metadata type.
        """
        _validate_request(device, process, requirements)
        if requirements.values:
            return (
                CapabilityDeclaration(device, process, requirements)
                in self.declarations
            )
        return any(
            declaration.device == device and declaration.process == process
            for declaration in self.declarations
        )

    def require(
        self,
        device: Device,
        process: Process,
        requirements: CapabilityRequirements,
    ) -> None:
        """Require a declared capability request without executing it.

        Args:
            device: Device requested for the process.
            process: Process requested on the device.
            requirements: Exact capability requirements for the request.

        Raises:
            TypeError: If an argument is not its declared metadata type.
            ValueError: If the valid request is not declared by this matrix.
        """
        _validate_request(device, process, requirements)
        if self.supports(device, process, requirements):
            return
        raise ValueError(
            "Unsupported capability declaration: "
            + repr(CapabilityDeclaration(device, process, requirements))
        )


def _validate_request(
    device: object,
    process: object,
    requirements: object,
) -> None:
    """Validate matrix request arguments in their required order.

    Args:
        device: Candidate device metadata.
        process: Candidate process metadata.
        requirements: Candidate capability requirements metadata.

    Raises:
        TypeError: If an argument has the wrong type, checked in argument order.
    """
    if not isinstance(device, Device):
        raise TypeError("device must be a Device.")
    if not isinstance(process, Process):
        raise TypeError("process must be a Process.")
    if not isinstance(requirements, CapabilityRequirements):
        raise TypeError("requirements must be a CapabilityRequirements.")


@dataclass(frozen=True)
class ExecutionRequest:
    """Declare a typed request for adapter selection without execution.

    This value only describes selection. It neither probes a backend, transfers
    state, nor establishes execution state or result contracts.

    Args:
        backend: Backend requested for selection.
        device: Device requested on the backend.
        process: Process requested on the device.
        requirements: Exact capability requirements for the process.

    Raises:
        TypeError: If a field is not its declared metadata type.
        ValueError: If the backend does not match the device backend.
    """

    backend: Backend
    device: Device
    process: Process
    requirements: CapabilityRequirements

    def __post_init__(self) -> None:
        """Validate request metadata without resolving or executing it.

        Raises:
            TypeError: If a field is not its declared metadata type.
            ValueError: If backend and device backend differ.
        """
        if not isinstance(self.backend, Backend):
            raise TypeError("ExecutionRequest.backend must be a Backend.")
        if not isinstance(self.device, Device):
            raise TypeError("ExecutionRequest.device must be a Device.")
        if not isinstance(self.process, Process):
            raise TypeError("ExecutionRequest.process must be a Process.")
        if not isinstance(self.requirements, CapabilityRequirements):
            raise TypeError(
                "ExecutionRequest.requirements must be a "
                "CapabilityRequirements."
            )
        if self.backend != self.device.backend:
            raise ValueError(
                "ExecutionRequest.backend must match device.backend."
            )


@runtime_checkable
class _ExecutionAdapter(Protocol):
    """Describe the minimal private adapter shape for selection only.

    P2 validates this shape but does not call it, probe optional dependencies,
    or transfer data.
    """

    def execute(self, *args: object, **kwargs: object) -> object:
        """Declare a future execution seam without invoking it.

        Args:
            *args: Positional arguments for a future execution contract.
            **kwargs: Keyword arguments for a future execution contract.

        Returns:
            A future execution result whose contract is not defined by P2.
        """


class _AdapterRegistry:
    """Keep exact context-local adapter registrations without execution.

    The registry validates only an adapter's callable ``execute`` shape. It
    neither invokes adapters nor loads, probes, or transfers backend state.
    """

    def __init__(self) -> None:
        """Create an empty registry without loading or probing backends."""
        self._adapters: dict[tuple[Process, Backend], _ExecutionAdapter] = {}

    def _register_adapter(
        self,
        process: object,
        backend: object,
        adapter: object,
    ) -> None:
        """Register one shaped adapter without invoking or inspecting it.

        Args:
            process: Process to associate with the adapter.
            backend: Backend to associate with the adapter.
            adapter: Object with a callable ``execute`` attribute.

        Raises:
            TypeError: If arguments are invalid in process, backend, adapter
                order.
            ValueError: If the process/backend pair is already registered.
        """
        if not isinstance(process, Process):
            raise TypeError("process must be a Process.")
        if not isinstance(backend, Backend):
            raise TypeError("backend must be a Backend.")
        if not isinstance(adapter, _ExecutionAdapter) or not callable(
            getattr(adapter, "execute", None)
        ):
            raise TypeError("adapter must have a callable execute attribute.")
        key = (process, backend)
        if key in self._adapters:
            raise ValueError(
                "Adapter already registered for process and backend."
            )
        self._adapters[key] = adapter

    def _lookup(self, process: Process, backend: Backend) -> _ExecutionAdapter:
        """Return one exact adapter without fallback or adapter execution.

        Args:
            process: Process associated with the requested adapter.
            backend: Backend associated with the requested adapter.

        Returns:
            The adapter registered under the exact process/backend pair.

        Raises:
            LookupError: If no adapter has the exact process/backend key.
        """
        try:
            return self._adapters[(process, backend)]
        except KeyError as error:
            raise LookupError(
                "No adapter registered for process and backend."
            ) from error

    def _snapshot(self) -> dict[tuple[Process, Backend], _ExecutionAdapter]:
        """Return a fresh private-registration snapshot for focused tests.

        Returns:
            A copy of the registry mapping that cannot mutate registrations.
        """
        return dict(self._adapters)


class ExecutionContext:
    """Select declared adapters without executing, probing, or transferring.

    Each context owns its private registry, preventing mutable module-global
    registration state. It performs selection only: it never invokes an
    adapter, probes availability, or transfers data.
    """

    def __init__(self, matrix: CapabilityMatrix) -> None:
        """Store a matrix and create an empty private adapter registry.

        Args:
            matrix: Immutable capability declarations used for selection.

        Raises:
            TypeError: If ``matrix`` is not a CapabilityMatrix.
        """
        if not isinstance(matrix, CapabilityMatrix):
            raise TypeError("matrix must be a CapabilityMatrix.")
        self._matrix = matrix
        self._registry = _AdapterRegistry()

    def resolve(self, request: ExecutionRequest) -> _ExecutionAdapter:
        """Return one exact adapter after validation without invoking it.

        Capability support is checked before the single registry lookup. This
        method does not catch adapter errors because it never executes adapters.

        Args:
            request: Typed selection request.

        Returns:
            The registered adapter object by identity.

        Raises:
            TypeError: If request is not an ExecutionRequest.
            ValueError: If CPU device spelling is invalid or support is absent.
            LookupError: If supported request has no exact registered adapter.
        """
        if not isinstance(request, ExecutionRequest):
            raise TypeError("request must be an ExecutionRequest.")
        device = _normalize_request_device(request)
        self._matrix.require(device, request.process, request.requirements)
        return self._registry._lookup(request.process, request.backend)


def _normalize_request_device(request: ExecutionRequest) -> Device:
    """Normalize CPU selection without parsing, probing, or transferring.

    CPU requests must use the canonical ``Device(Backend.CPU, "cpu")``
    spelling. Warp native identifiers remain opaque and are returned verbatim.

    Args:
        request: Typed selection request whose device requires normalization.

    Returns:
        Canonical CPU device metadata or the request's unchanged Warp device.

    Raises:
        ValueError: If the CPU spelling or backend/device pairing is invalid.
    """
    if request.backend != request.device.backend:
        raise ValueError("ExecutionRequest.backend must match device.backend.")
    if request.backend is Backend.CPU:
        if request.device.native != "cpu":
            raise ValueError(
                "CPU execution requires Device(Backend.CPU, 'cpu')."
            )
        return Device(Backend.CPU, "cpu")
    if request.backend is Backend.WARP:
        return request.device
    raise ValueError("ExecutionRequest.backend must match device.backend.")
