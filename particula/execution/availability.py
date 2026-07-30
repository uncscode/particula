"""Resolve concrete execution availability without selecting execution work.

This direct-import-only boundary validates declared request metadata, capability
support, optional runtime availability, and request-associated state in a fixed
order.  It does not select adapters, allocate data, or invoke execution.
"""

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, cast

from particula.execution import (
    Backend,
    CapabilityMatrix,
    Device,
    ExecutionRequest,
)
from particula.execution.errors import (
    InvalidExecutionStateError,
    UnavailableDeviceError,
    UnavailableRuntimeError,
    UnknownDeviceError,
    UnsupportedCapabilityError,
    UnsupportedProcessError,
)


class AvailabilityProvider(Protocol):
    """Describe pure and lazy availability checks for one backend."""

    def recognizes(self, device: Device) -> bool:
        """Return whether the declared device is structurally recognized."""

    def runtime_available(self) -> bool:
        """Return whether the optional backend runtime is available."""

    def device_available(self, device: Device) -> bool:
        """Return whether a recognized runtime device is available."""


StateValidator = Callable[[ExecutionRequest], bool]


@dataclass(frozen=True)
class AvailabilityDecision:
    """Record an availability-validated request without runtime state."""

    request: ExecutionRequest


class _CPUAvailabilityProvider:
    """Provide availability checks for the one canonical CPU device."""

    def recognizes(self, device: Device) -> bool:
        """Return whether the device is the canonical CPU declaration."""
        return device == Device(Backend.CPU, "cpu")

    def runtime_available(self) -> bool:
        """Return CPU runtime availability without probing hardware."""
        return True

    def device_available(self, device: Device) -> bool:
        """Return CPU device availability without probing hardware."""
        del device
        return True


class _WarpAvailabilityProvider:
    """Provide lazy runtime and device checks for opaque Warp devices."""

    def recognizes(self, device: Device) -> bool:
        """Return whether the device declares the Warp backend."""
        return device.backend is Backend.WARP

    def runtime_available(self) -> bool:
        """Return whether Warp can be imported without retaining its module."""
        try:
            importlib.import_module("warp")
        except ImportError:
            return False
        return True

    def device_available(self, device: Device) -> bool:
        """Return whether Warp resolves the opaque native device identifier."""
        try:
            runtime = importlib.import_module("warp")
            runtime.get_device(device.native)
        except Exception:
            return False
        return True


_DEFAULT_PROVIDERS: Mapping[Backend, AvailabilityProvider] = {
    Backend.CPU: _CPUAvailabilityProvider(),
    Backend.WARP: _WarpAvailabilityProvider(),
}


def _always_valid(_: ExecutionRequest) -> bool:
    """Return the default successful request-associated state validation."""
    return True


def _context(request: ExecutionRequest) -> dict[str, str]:
    """Return shared structured error context for a valid request."""
    return {
        "backend": request.backend.value,
        "device": request.device.native,
        "process": request.process.name,
        "capability": repr(request.requirements),
    }


def _validate_providers(
    providers: object,
    request: ExecutionRequest,
) -> AvailabilityProvider:
    """Return the selected provider after fail-closed registry validation."""
    if not isinstance(providers, Mapping):
        raise UnavailableRuntimeError(request.backend.value)
    try:
        keys = tuple(providers)
    except Exception as error:
        raise UnavailableRuntimeError(request.backend.value) from error
    if not all(type(key) is Backend for key in keys) or set(keys) != {
        Backend.CPU,
        Backend.WARP,
    }:
        raise UnavailableRuntimeError(request.backend.value)
    try:
        registry = cast(Mapping[Backend, object], providers)
        validated_providers: dict[Backend, AvailabilityProvider] = {}
        for backend in (Backend.CPU, Backend.WARP):
            provider = cast(AvailabilityProvider, registry[backend])
            methods = (
                provider.recognizes,
                provider.runtime_available,
                provider.device_available,
            )
            if not all(callable(method) for method in methods):
                raise UnavailableRuntimeError(request.backend.value)
            validated_providers[backend] = provider
    except UnavailableRuntimeError:
        raise
    except Exception as error:
        raise UnavailableRuntimeError(request.backend.value) from error
    return validated_providers[request.backend]


def _provider_result(
    method: Callable[..., object],
    error_factory: Callable[[], Exception],
    *args: object,
) -> bool:
    """Call one provider phase and map exceptions or malformed results."""
    try:
        result = method(*args)
    except Exception as error:
        raise error_factory() from error
    if type(result) is not bool:
        raise error_factory()
    return result


def resolve_availability(
    request: ExecutionRequest,
    matrix: CapabilityMatrix,
    *,
    providers: Mapping[Backend, AvailabilityProvider] | None = None,
    state_validator: StateValidator | None = None,
) -> AvailabilityDecision:
    """Resolve availability in declaration, runtime, device, and state order.

    Args:
        request: Validated P1 request declaration.
        matrix: Capability declarations applicable to the exact device.
        providers: Optional complete CPU/Warp availability registry.
        state_validator: Optional request-associated state validator.

    Returns:
        An immutable decision retaining the exact request by identity.

    Raises:
        TypeError: If a P1 carrier has an invalid type.
        ExecutionCapabilityError: If any availability phase fails.
    """
    if not isinstance(request, ExecutionRequest):
        raise TypeError("request must be an ExecutionRequest.")
    if not isinstance(matrix, CapabilityMatrix):
        raise TypeError("matrix must be a CapabilityMatrix.")

    provider = _validate_providers(
        _DEFAULT_PROVIDERS if providers is None else providers,
        request,
    )
    context = _context(request)
    if not _provider_result(
        provider.recognizes,
        lambda: UnknownDeviceError(
            request.device.native,
            backend=request.backend.value,
        ),
        request.device,
    ):
        raise UnknownDeviceError(
            request.device.native, backend=request.backend.value
        )

    has_process = any(
        declaration.device == request.device
        and declaration.process == request.process
        for declaration in matrix.declarations
    )
    if not has_process:
        raise UnsupportedProcessError(
            request.process.name,
            backend=request.backend.value,
            device=request.device.native,
        )
    if not matrix.supports(
        request.device, request.process, request.requirements
    ):
        raise UnsupportedCapabilityError(
            repr(request.requirements),
            backend=request.backend.value,
            device=request.device.native,
            process=request.process.name,
        )
    if not _provider_result(
        provider.runtime_available,
        lambda: UnavailableRuntimeError(request.backend.value),
    ):
        raise UnavailableRuntimeError(request.backend.value)
    if not _provider_result(
        provider.device_available,
        lambda: UnavailableDeviceError(
            request.device.native,
            backend=request.backend.value,
        ),
        request.device,
    ):
        raise UnavailableDeviceError(
            request.device.native,
            backend=request.backend.value,
        )

    validator = _always_valid if state_validator is None else state_validator
    try:
        state_valid = validator(request)
    except Exception as error:
        raise InvalidExecutionStateError(
            "validation_failed", **context
        ) from error
    if type(state_valid) is not bool or not state_valid:
        raise InvalidExecutionStateError("validation_failed", **context)
    return AvailabilityDecision(request)
