"""Resolve declared execution availability without selecting execution work.

This concrete, direct-import-only boundary validates P1 request metadata,
capability declarations, optional runtime and device status, and
request-associated state in a fixed order. It neither selects adapters nor
allocates execution resources, transfers, synchronizes, mutates, or executes
work.
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
    """Describe pure recognition and lazy status checks for one backend.

    Implementations must not execute work. Runtime and device checks run only
    after the resolver has accepted the request declaration and capabilities.
    """

    def recognizes(self, device: Device) -> bool:
        """Return whether a declared device is structurally recognized.

        Args:
            device: Validated device declaration to recognize without probing.

        Returns:
            ``True`` when the provider recognizes the declaration.
        """

    def runtime_available(self) -> bool:
        """Return whether the optional backend runtime is available.

        Returns:
            ``True`` when the backend runtime is available for device checking.
        """

    def device_available(self, device: Device) -> bool:
        """Return whether a recognized device is available at runtime.

        Args:
            device: Structurally recognized device declaration to check.

        Returns:
            ``True`` when the runtime can resolve the declared device.
        """


StateValidator = Callable[[ExecutionRequest], bool]


@dataclass(frozen=True)
class AvailabilityDecision:
    """Record an availability-validated request without runtime ownership.

    Attributes:
        request: The exact validated request retained by identity. This record
            owns no adapter, runtime handle, device object, or execution state.
    """

    request: ExecutionRequest


class _CPUAvailabilityProvider:
    """Provide no-probe availability checks for the canonical CPU device."""

    def recognizes(self, device: Device) -> bool:
        """Return whether the device is the canonical CPU declaration.

        Args:
            device: Validated device declaration to compare.

        Returns:
            ``True`` only for ``Device(Backend.CPU, "cpu")``.
        """
        return device == Device(Backend.CPU, "cpu")

    def runtime_available(self) -> bool:
        """Return CPU runtime availability without probing hardware.

        Returns:
            ``True``, because the CPU provider has no optional runtime.
        """
        return True

    def device_available(self, device: Device) -> bool:
        """Return CPU device availability without probing hardware.

        Args:
            device: Recognized canonical CPU declaration, unused by this check.

        Returns:
            ``True`` without inspecting hardware.
        """
        del device
        return True


class _WarpAvailabilityProvider:
    """Provide lazy runtime and device checks for opaque Warp devices.

    Native identifiers are passed unchanged to Warp after lazy runtime loading;
    this provider neither parses nor normalizes them.
    """

    def recognizes(self, device: Device) -> bool:
        """Return whether the device declares the Warp backend.

        Args:
            device: Validated device declaration with an opaque native value.

        Returns:
            ``True`` when the declaration uses ``Backend.WARP``.
        """
        return device.backend is Backend.WARP

    def runtime_available(self) -> bool:
        """Return whether Warp can be imported lazily.

        Returns:
            ``True`` when the optional Warp runtime imports successfully.
        """
        try:
            importlib.import_module("warp")
        except ImportError:
            return False
        return True

    def device_available(self, device: Device) -> bool:
        """Return whether Warp resolves the opaque native device identifier.

        Args:
            device: Recognized Warp declaration whose native value is opaque.

        Returns:
            ``True`` when Warp resolves the original native identifier.
        """
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
    """Return the default successful request-associated state validation.

    Args:
        _: Validated request intentionally ignored by the default validator.

    Returns:
        ``True`` for every valid request.
    """
    return True


def _context(request: ExecutionRequest) -> dict[str, str]:
    """Build structured error context from a validated request.

    Args:
        request: Validated request supplying backend, device, process, and
            capability metadata.

    Returns:
        Error-context fields for availability failures.
    """
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
    """Return a selected provider after fail-closed registry validation.

    Args:
        providers: Candidate registry expected to contain exactly CPU and Warp
            providers with callable availability methods.
        request: Validated request selecting one provider.

    Returns:
        The validated provider for the request backend.

    Raises:
        UnavailableRuntimeError: If the registry is malformed or inaccessible.
    """
    if not isinstance(providers, Mapping):
        raise UnavailableRuntimeError(request.backend.value)
    try:
        keys = tuple(providers)
    except Exception as error:
        raise UnavailableRuntimeError(request.backend.value) from error
    if (
        len(keys) != 2
        or not all(type(key) is Backend for key in keys)
        or set(keys)
        != {
            Backend.CPU,
            Backend.WARP,
        }
    ):
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
    """Call a provider phase and map exceptions or malformed results.

    Args:
        method: Provider status method for one availability phase.
        error_factory: Creates the typed error for a failed phase.
        *args: Positional arguments forwarded to ``method``.

    Returns:
        The validated boolean status returned by ``method``.

    Raises:
        Exception: The phase-specific error when the method raises or returns
            a value other than an exact boolean.
    """
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

    The resolver is pre-execution only. It validates a provider registry before
    provider calls, then short-circuits through recognition, capability
    declarations, runtime status, device status, and request-associated state.

    Args:
        request: Validated P1 request declaration.
        matrix: Capability declarations applicable to the exact device.
        providers: Optional complete CPU/Warp availability registry.
        state_validator: Optional request-associated state validator. Omitting
            it selects an internal validator that accepts every request, so
            state validation is substantive only when a validator is supplied.

    Returns:
        An immutable decision retaining the exact request by identity.

    Raises:
        TypeError: If a P1 carrier has an invalid type.
        UnknownDeviceError: If the selected provider rejects the device.
        UnsupportedProcessError: If no declaration matches the device/process.
        UnsupportedCapabilityError: If no declaration satisfies requirements.
        UnavailableRuntimeError: If a registry or runtime is unavailable.
        UnavailableDeviceError: If the runtime cannot resolve the device.
        InvalidExecutionStateError: If request-associated state is invalid.
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
