"""Declare dependency-neutral, immutable execution capability metadata.

This module models declared backend, device, process, and capability support.
It only performs structural validation and read-only capability lookup; it does
not load optional backends, resolve devices, transfer state, choose adapters,
or execute processes.
"""

import re
from dataclasses import dataclass
from enum import Enum

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
