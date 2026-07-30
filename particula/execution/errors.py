"""Dependency-neutral execution capability error taxonomy.

This module defines direct-import-only errors for execution availability and
fallback boundaries. It intentionally imports no execution providers or
optional runtimes.
"""

from enum import Enum


class ExecutionCapabilityReason(str, Enum):
    """Closed reason codes for execution capability errors."""

    UNKNOWN_BACKEND = "unknown_backend"
    UNKNOWN_DEVICE = "unknown_device"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    DEVICE_UNAVAILABLE = "device_unavailable"
    PROCESS_UNSUPPORTED = "process_unsupported"
    CAPABILITY_UNSUPPORTED = "capability_unsupported"
    INVALID_STATE = "invalid_state"
    FALLBACK_DISALLOWED = "fallback_disallowed"


class ExecutionCapabilityError(Exception):
    """Describe an unavailable, unsupported, or invalid execution request.

    Args:
        reason: Stable reason code for the error.
        backend: Optional execution backend context.
        device: Optional execution device context.
        process: Optional process context.
        capability: Optional capability context.
        state: Optional execution-state context.
        fallback_boundary: Optional fallback-boundary context.

    Raises:
        TypeError: If a reason or context field has an unsupported type.
    """

    reason: ExecutionCapabilityReason
    backend: str | None
    device: str | None
    process: str | None
    capability: str | None
    state: str | None
    fallback_boundary: str | None

    def __init__(
        self,
        reason: ExecutionCapabilityReason,
        *,
        backend: str | None = None,
        device: str | None = None,
        process: str | None = None,
        capability: str | None = None,
        state: str | None = None,
        fallback_boundary: str | None = None,
    ) -> None:
        """Initialize the structured capability error."""
        if not isinstance(reason, ExecutionCapabilityReason):
            raise TypeError("reason must be an ExecutionCapabilityReason.")

        context = {
            "backend": backend,
            "device": device,
            "process": process,
            "capability": capability,
            "state": state,
            "fallback_boundary": fallback_boundary,
        }
        for name, value in context.items():
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a str or None.")

        self.reason = reason
        self.backend = backend
        self.device = device
        self.process = process
        self.capability = capability
        self.state = state
        self.fallback_boundary = fallback_boundary
        super().__init__(reason)

    def __str__(self) -> str:
        """Render deterministic structured error context."""
        return (
            f"{type(self).__name__}(reason={self.reason.value}, "
            f"backend={self.backend!r}, device={self.device!r}, "
            f"process={self.process!r}, capability={self.capability!r}, "
            f"state={self.state!r}, "
            f"fallback_boundary={self.fallback_boundary!r})"
        )


class UnknownExecutionTargetError(ExecutionCapabilityError):
    """Base category for unknown execution targets."""


class UnavailableExecutionTargetError(ExecutionCapabilityError):
    """Base category for unavailable execution targets."""


class UnsupportedExecutionRequestError(ExecutionCapabilityError):
    """Base category for unsupported execution requests."""


class UnknownBackendError(UnknownExecutionTargetError):
    """Report a backend identifier that is not recognized."""

    def __init__(self, backend: str) -> None:
        """Initialize an unknown-backend error."""
        super().__init__(
            ExecutionCapabilityReason.UNKNOWN_BACKEND,
            backend=backend,
        )


class UnknownDeviceError(UnknownExecutionTargetError):
    """Report a device identifier that is not recognized."""

    def __init__(self, device: str, *, backend: str | None = None) -> None:
        """Initialize an unknown-device error."""
        super().__init__(
            ExecutionCapabilityReason.UNKNOWN_DEVICE,
            backend=backend,
            device=device,
        )


class UnavailableRuntimeError(UnavailableExecutionTargetError):
    """Report an execution backend whose runtime is unavailable."""

    def __init__(self, backend: str) -> None:
        """Initialize an unavailable-runtime error."""
        super().__init__(
            ExecutionCapabilityReason.RUNTIME_UNAVAILABLE,
            backend=backend,
        )


class UnavailableDeviceError(UnavailableExecutionTargetError):
    """Report a device that is unavailable for execution."""

    def __init__(self, device: str, *, backend: str | None = None) -> None:
        """Initialize an unavailable-device error."""
        super().__init__(
            ExecutionCapabilityReason.DEVICE_UNAVAILABLE,
            backend=backend,
            device=device,
        )


class UnsupportedProcessError(UnsupportedExecutionRequestError):
    """Report an execution process that is not supported."""

    def __init__(
        self,
        process: str,
        *,
        backend: str | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize an unsupported-process error."""
        super().__init__(
            ExecutionCapabilityReason.PROCESS_UNSUPPORTED,
            backend=backend,
            device=device,
            process=process,
        )


class UnsupportedCapabilityError(UnsupportedExecutionRequestError):
    """Report an execution capability that is not supported."""

    def __init__(
        self,
        capability: str,
        *,
        backend: str | None = None,
        device: str | None = None,
        process: str | None = None,
    ) -> None:
        """Initialize an unsupported-capability error."""
        super().__init__(
            ExecutionCapabilityReason.CAPABILITY_UNSUPPORTED,
            backend=backend,
            device=device,
            process=process,
            capability=capability,
        )


class InvalidExecutionStateError(ExecutionCapabilityError):
    """Report an invalid state for an execution request."""

    def __init__(
        self,
        state: str,
        *,
        backend: str | None = None,
        device: str | None = None,
        process: str | None = None,
        capability: str | None = None,
    ) -> None:
        """Initialize an invalid-execution-state error."""
        super().__init__(
            ExecutionCapabilityReason.INVALID_STATE,
            backend=backend,
            device=device,
            process=process,
            capability=capability,
            state=state,
        )


class FallbackDisallowedError(ExecutionCapabilityError):
    """Report a fallback request outside its permitted boundary."""

    def __init__(
        self,
        fallback_boundary: str,
        *,
        backend: str | None = None,
        device: str | None = None,
        process: str | None = None,
        capability: str | None = None,
        state: str | None = None,
    ) -> None:
        """Initialize a fallback-disallowed error."""
        super().__init__(
            ExecutionCapabilityReason.FALLBACK_DISALLOWED,
            backend=backend,
            device=device,
            process=process,
            capability=capability,
            state=state,
            fallback_boundary=fallback_boundary,
        )


__all__ = [
    "ExecutionCapabilityReason",
    "ExecutionCapabilityError",
    "UnknownExecutionTargetError",
    "UnavailableExecutionTargetError",
    "UnsupportedExecutionRequestError",
    "UnknownBackendError",
    "UnknownDeviceError",
    "UnavailableRuntimeError",
    "UnavailableDeviceError",
    "UnsupportedProcessError",
    "UnsupportedCapabilityError",
    "InvalidExecutionStateError",
    "FallbackDisallowedError",
]
