"""Shared fail-closed validation for execution declaration carriers."""

import re

from particula.execution import (
    Backend,
    Capability,
    CapabilityDeclaration,
    CapabilityMatrix,
    CapabilityRequirements,
    Device,
    ExecutionRequest,
    Process,
)

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def validate_execution_request(request: object, field_name: str) -> None:
    """Deeply validate an execution request and its nested declarations.

    Args:
        request: Candidate execution request carrier.
        field_name: Qualified public name used in error messages.

    Raises:
        TypeError: If a carrier or nested declaration has an invalid type.
        ValueError: If a nested value violates its closed declaration contract.
    """
    if not isinstance(request, ExecutionRequest):
        raise TypeError(f"{field_name} must be an ExecutionRequest.")
    if not isinstance(request.backend, Backend):
        raise TypeError(f"{field_name}.backend must be a Backend.")
    _validate_device(request.device, f"{field_name}.device")
    _validate_process(request.process, f"{field_name}.process")
    _validate_requirements(request.requirements, f"{field_name}.requirements")
    if request.backend is not request.device.backend:
        raise ValueError(f"{field_name} backend must match device.backend.")


def validate_capability_matrix(matrix: object, field_name: str) -> None:
    """Deeply validate a capability matrix and every declaration it retains.

    Args:
        matrix: Candidate capability matrix carrier.
        field_name: Qualified public name used in error messages.

    Raises:
        TypeError: If a carrier or nested declaration has an invalid type.
    """
    if not isinstance(matrix, CapabilityMatrix):
        raise TypeError(f"{field_name} must be a CapabilityMatrix.")
    if type(matrix.declarations) is not frozenset:
        raise TypeError(f"{field_name}.declarations must be a frozenset.")
    for declaration in matrix.declarations:
        if not isinstance(declaration, CapabilityDeclaration):
            raise TypeError(
                f"{field_name}.declarations must contain only "
                "CapabilityDeclaration instances."
            )
        _validate_device(declaration.device, f"{field_name}.declaration.device")
        _validate_process(
            declaration.process,
            f"{field_name}.declaration.process",
        )
        _validate_requirements(
            declaration.requirements,
            f"{field_name}.declaration.requirements",
        )


def _validate_device(device: object, field_name: str) -> None:
    """Validate a device value and its backend/native fields."""
    if not isinstance(device, Device):
        raise TypeError(f"{field_name} must be a Device.")
    if not isinstance(device.backend, Backend):
        raise TypeError(f"{field_name}.backend must be a Backend.")
    if not isinstance(device.native, str):
        raise TypeError(f"{field_name}.native must be a str.")
    if not device.native or device.native != device.native.strip():
        raise ValueError(
            f"{field_name}.native must be a nonempty str without surrounding "
            "whitespace."
        )


def _validate_process(process: object, field_name: str) -> None:
    """Validate a process value and its identifier-style name."""
    if not isinstance(process, Process):
        raise TypeError(f"{field_name} must be a Process.")
    if not isinstance(process.name, str):
        raise TypeError(f"{field_name}.name must be a str.")
    if not _NAME_PATTERN.fullmatch(process.name):
        raise ValueError(f"{field_name}.name must match ^[a-z][a-z0-9_]*$.")


def _validate_requirements(requirements: object, field_name: str) -> None:
    """Validate an exact immutable capability-requirements declaration."""
    if not isinstance(requirements, CapabilityRequirements):
        raise TypeError(f"{field_name} must be a CapabilityRequirements.")
    if type(requirements.values) is not frozenset:
        raise TypeError(f"{field_name}.values must be a frozenset.")
    for capability in requirements.values:
        if not isinstance(capability, Capability):
            raise TypeError(
                f"{field_name}.values must contain only Capability instances."
            )
        if not isinstance(capability.name, str):
            raise TypeError(f"{field_name}.capability.name must be a str.")
        if not _NAME_PATTERN.fullmatch(capability.name):
            raise ValueError(
                f"{field_name}.capability.name must match ^[a-z][a-z0-9_]*$."
            )
