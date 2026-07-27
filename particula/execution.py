"""Immutable execution capability metadata.

This module declares metadata only. It does not load backends, resolve devices,
transfer state, choose adapters, or execute processes.
"""

import re
from dataclasses import dataclass
from enum import Enum

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class Backend(str, Enum):
    """Closed set of declared execution backends."""

    CPU = "cpu"
    WARP = "warp"


@dataclass(frozen=True)
class Device:
    """Immutable backend and opaque native device identifier.

    The native identifier is retained verbatim and is never resolved here.
    """

    backend: Backend
    native: str

    def __post_init__(self) -> None:
        """Validate the declared backend and opaque native identifier."""
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
    """Immutable, validated process name declaration."""

    name: str

    def __post_init__(self) -> None:
        """Validate the process name."""
        _validate_name(self.name, "Process.name")


@dataclass(frozen=True)
class Capability:
    """Immutable, validated capability name declaration."""

    name: str

    def __post_init__(self) -> None:
        """Validate the capability name."""
        _validate_name(self.name, "Capability.name")


def _validate_name(value: object, field_name: str) -> None:
    """Validate a lowercase identifier-style metadata name."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a str.")
    if not _NAME_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must match ^[a-z][a-z0-9_]*$.")


@dataclass(frozen=True)
class CapabilityRequirements:
    """Immutable exact set of capability declarations."""

    values: frozenset[Capability]

    def __post_init__(self) -> None:
        """Validate the exact immutable capability collection type."""
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
    """Immutable capability support declaration for one device and process."""

    device: Device
    process: Process
    requirements: CapabilityRequirements

    def __post_init__(self) -> None:
        """Validate the declaration's typed value objects."""
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
    """Immutable, pure lookup table for exact capability declarations."""

    declarations: frozenset[CapabilityDeclaration]

    def __post_init__(self) -> None:
        """Validate the exact immutable declaration collection type."""
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
        """Return whether this matrix exactly declares the request."""
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
        """Require an exact capability declaration or raise ValueError."""
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
    """Validate matrix request arguments in their required order."""
    if not isinstance(device, Device):
        raise TypeError("device must be a Device.")
    if not isinstance(process, Process):
        raise TypeError("process must be a Process.")
    if not isinstance(requirements, CapabilityRequirements):
        raise TypeError("requirements must be a CapabilityRequirements.")
