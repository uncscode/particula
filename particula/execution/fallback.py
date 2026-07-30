"""Provide a direct-import-only, explicit CPU fallback boundary.

This module does not import optional runtimes or perform transfers, lifecycle
operations, restoration, recovery, or retries. Callers retain responsibility
for providing CPU-authoritative state at a visible boundary.
"""

import inspect
from dataclasses import dataclass
from enum import Enum

from particula.execution import (
    Backend,
    CapabilityRequirements,
    CPUExecutionState,
    Device,
    ExecutionAdapter,
    ExecutionContext,
    ExecutionRequest,
    ExecutionResult,
    Process,
    validate_execution_result,
)
from particula.execution.errors import (
    ExecutionCapabilityError,
    ExecutionCapabilityReason,
    FallbackDisallowedError,
)


class FallbackPolicy(str, Enum):
    """Declare whether an eligible capability error is re-raised or retried.

    Retrying occurs on CPU.
    """

    RAISE = "raise"
    CPU = "cpu"


class FallbackBoundary(str, Enum):
    """Declare the caller-visible boundary at which fallback is requested."""

    PRE_UPLOAD = "pre_upload"
    RESTORED = "restored"


class CPUStateAuthority(str, Enum):
    """Declare the authority of supplied CPU state for a fallback request."""

    CPU_AUTHORITATIVE = "cpu_authoritative"
    RESIDENT = "resident"
    UPLOADED = "uploaded"
    MUTATED = "mutated"


@dataclass(frozen=True, eq=False)
class FallbackRequest:
    """Carry one explicit fallback request without inspecting state payloads.

    Args:
        original_request: Original non-CPU selection request.
        original_error: Typed capability error produced for that request.
        context: Context owning the registered CPU adapter.
        cpu_state: Caller-owned CPU state, if fallback is enabled.
        policy: Default-deny or explicit CPU fallback policy.
        boundary: Caller-visible fallback boundary.
        state_authority: Caller assertion about CPU-state authority.
    """

    original_request: ExecutionRequest
    original_error: ExecutionCapabilityError
    context: ExecutionContext
    cpu_state: CPUExecutionState | None
    policy: FallbackPolicy = FallbackPolicy.RAISE
    boundary: FallbackBoundary = FallbackBoundary.PRE_UPLOAD
    state_authority: CPUStateAuthority = CPUStateAuthority.CPU_AUTHORITATIVE

    def __post_init__(self) -> None:
        """Validate carrier types without selecting or reading payloads."""
        if not isinstance(self.original_request, ExecutionRequest):
            raise TypeError("original_request must be an ExecutionRequest.")
        if not isinstance(self.original_error, ExecutionCapabilityError):
            raise TypeError(
                "original_error must be an ExecutionCapabilityError."
            )
        if not isinstance(self.context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext.")
        if (
            self.cpu_state is not None
            and type(self.cpu_state) is not CPUExecutionState
        ):
            raise TypeError("cpu_state must be a CPUExecutionState or None.")
        _validate_fallback_enums(
            self.policy, self.boundary, self.state_authority
        )


@dataclass(frozen=True, eq=False)
class FallbackResolution:
    """Retain the one CPU selection made for an eligible fallback request."""

    original_request: ExecutionRequest
    cpu_request: ExecutionRequest
    original_error: ExecutionCapabilityError
    cpu_state: CPUExecutionState
    adapter: ExecutionAdapter
    policy: FallbackPolicy
    boundary: FallbackBoundary
    requested_backend: Backend
    selected_backend: Backend
    capability_reason: ExecutionCapabilityReason

    def __post_init__(self) -> None:
        """Validate resolution metadata without reselecting an adapter."""
        if not isinstance(self.original_request, ExecutionRequest):
            raise TypeError("original_request must be an ExecutionRequest.")
        if not isinstance(self.cpu_request, ExecutionRequest):
            raise TypeError("cpu_request must be an ExecutionRequest.")
        if not isinstance(self.original_error, ExecutionCapabilityError):
            raise TypeError(
                "original_error must be an ExecutionCapabilityError."
            )
        if type(self.cpu_state) is not CPUExecutionState:
            raise TypeError("cpu_state must be a CPUExecutionState.")
        if not _is_static_execution_adapter(self.adapter):
            raise TypeError("adapter must have a callable execute attribute.")
        _validate_resolution_request(self)
        _validate_fallback_enums(
            self.policy, self.boundary, CPUStateAuthority.CPU_AUTHORITATIVE
        )
        _validate_resolution_reason(self)


@dataclass(frozen=True, eq=False)
class FallbackDispatchResult:
    """Retain a native result and immutable fallback provenance by identity."""

    resolution: FallbackResolution
    result: ExecutionResult
    metadata: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        """Validate exact fallback metadata without changing native metadata."""
        if type(self.resolution) is not FallbackResolution:
            raise TypeError("resolution must be a FallbackResolution.")
        if type(self.result) is not ExecutionResult:
            raise TypeError("result must be an ExecutionResult.")
        validate_execution_result(self.resolution.cpu_state, self.result)
        if type(self.metadata) is not tuple:
            raise TypeError("metadata must be a tuple.")
        expected = _fallback_metadata(self.resolution)
        if self.metadata != expected:
            raise ValueError("metadata must contain exact fallback provenance.")


_ELIGIBLE_REASONS = frozenset(
    {
        ExecutionCapabilityReason.UNKNOWN_DEVICE,
        ExecutionCapabilityReason.RUNTIME_UNAVAILABLE,
        ExecutionCapabilityReason.DEVICE_UNAVAILABLE,
        ExecutionCapabilityReason.PROCESS_UNSUPPORTED,
        ExecutionCapabilityReason.CAPABILITY_UNSUPPORTED,
    }
)


def _is_static_execution_adapter(adapter: object) -> bool:
    """Return whether an adapter statically exposes a callable execute seam."""
    execute = inspect.getattr_static(adapter, "execute", None)
    if isinstance(execute, classmethod):
        return callable(execute.__func__)
    return callable(execute)


def _validate_resolution_request(resolution: FallbackResolution) -> None:
    """Validate that a resolution retains canonical CPU request metadata."""
    if not isinstance(resolution.selected_backend, Backend):
        raise TypeError("selected_backend must be a Backend.")
    if resolution.selected_backend is not Backend.CPU:
        raise ValueError("selected_backend must be Backend.CPU.")
    if resolution.cpu_request.backend is not Backend.CPU:
        raise ValueError("cpu_request.backend must be Backend.CPU.")
    if not isinstance(resolution.requested_backend, Backend):
        raise TypeError("requested_backend must be a Backend.")
    if resolution.requested_backend is not resolution.original_request.backend:
        raise ValueError("requested_backend must match original_request.")
    if resolution.cpu_request.device != Device(Backend.CPU, "cpu"):
        raise ValueError("cpu_request.device must be canonical CPU.")
    if (
        resolution.cpu_request.process
        is not resolution.original_request.process
    ):
        raise ValueError("cpu_request.process must be original process.")
    if (
        resolution.cpu_request.requirements
        is not resolution.original_request.requirements
    ):
        raise ValueError(
            "cpu_request.requirements must be original requirements."
        )


def _validate_resolution_reason(resolution: FallbackResolution) -> None:
    """Validate that resolution provenance retains its original error reason."""
    if not isinstance(resolution.capability_reason, ExecutionCapabilityReason):
        raise TypeError(
            "capability_reason must be an ExecutionCapabilityReason."
        )
    if resolution.capability_reason is not resolution.original_error.reason:
        raise ValueError("capability_reason must match original_error.")


def _validate_fallback_enums(
    policy: object,
    boundary: object,
    state_authority: object,
) -> None:
    """Validate the closed fallback enum fields."""
    if not isinstance(policy, FallbackPolicy):
        raise TypeError("policy must be a FallbackPolicy.")
    if not isinstance(boundary, FallbackBoundary):
        raise TypeError("boundary must be a FallbackBoundary.")
    if not isinstance(state_authority, CPUStateAuthority):
        raise TypeError("state_authority must be a CPUStateAuthority.")


def _disallow(
    fallback: FallbackRequest,
    *,
    state: str,
    boundary: str,
) -> FallbackDisallowedError:
    """Create a deterministic disallowed error chained by the caller."""
    request = fallback.original_request
    return FallbackDisallowedError(
        boundary,
        backend=request.backend.value,
        device=request.device.native,
        process=request.process.name,
        capability=repr(request.requirements),
        state=state,
    )


def _validate_request_and_error(fallback: FallbackRequest) -> None:
    """Revalidate forged request and error carriers before context selection."""
    request = fallback.original_request
    error = fallback.original_error
    if not isinstance(request, ExecutionRequest):
        raise TypeError("original_request must be an ExecutionRequest.")
    if not isinstance(error, ExecutionCapabilityError):
        raise TypeError("original_error must be an ExecutionCapabilityError.")
    if not isinstance(fallback.context, ExecutionContext):
        raise TypeError("context must be an ExecutionContext.")
    _validate_request_fields(request)
    _validate_error_fields(error)


def _validate_request_fields(request: ExecutionRequest) -> None:
    """Validate the original request's closed carrier fields."""
    if not isinstance(request.backend, Backend):
        raise TypeError("original_request.backend must be a Backend.")
    if not isinstance(request.device, Device):
        raise TypeError("original_request.device must be a Device.")
    if not isinstance(request.process, Process):
        raise TypeError("original_request.process must be a Process.")
    if not isinstance(request.requirements, CapabilityRequirements):
        raise TypeError(
            "original_request.requirements must be a CapabilityRequirements."
        )
    if request.backend is not request.device.backend:
        raise ValueError("original_request backend must match device.backend.")


def _validate_error_fields(error: ExecutionCapabilityError) -> None:
    """Validate the original capability error's closed carrier fields."""
    if not isinstance(error.reason, ExecutionCapabilityReason):
        raise TypeError(
            "original_error.reason must be an ExecutionCapabilityReason."
        )
    for field in (
        "backend",
        "device",
        "process",
        "capability",
        "state",
        "fallback_boundary",
    ):
        value = getattr(error, field, None)
        if value is not None and not isinstance(value, str):
            raise TypeError(f"original_error.{field} must be a str or None.")


def _validate_error_context(fallback: FallbackRequest) -> None:
    """Require any supplied error context to describe the original request."""
    request = fallback.original_request
    error = fallback.original_error
    expected = {
        "backend": request.backend.value,
        "device": request.device.native,
        "process": request.process.name,
        "capability": repr(request.requirements),
    }
    if (
        any(
            getattr(error, field, None) is not None
            and getattr(error, field, None) != value
            for field, value in expected.items()
        )
        or getattr(error, "state", None) is not None
        or getattr(error, "fallback_boundary", None) is not None
    ):
        raise _disallow(
            fallback,
            state="error_context",
            boundary="error_context",
        ) from error


def _fallback_metadata(
    resolution: FallbackResolution,
) -> tuple[tuple[str, str], ...]:
    """Return ordered immutable fallback provenance without copying results."""
    return (
        ("requested_backend", resolution.requested_backend.value),
        ("selected_backend", resolution.selected_backend.value),
        ("capability_reason", resolution.capability_reason.value),
    )


def resolve_cpu_fallback(fallback: FallbackRequest) -> FallbackResolution:
    """Select the registered CPU adapter once for an eligible explicit request.

    Args:
        fallback: Exact explicit fallback carrier.

    Returns:
        Immutable resolution retaining the selected adapter by identity.

    Raises:
        ExecutionCapabilityError: The original error for default-deny policy.
        FallbackDisallowedError: If fallback fails closed before selection.
    """
    if type(fallback) is not FallbackRequest:
        raise TypeError("fallback must be a FallbackRequest.")
    _validate_request_and_error(fallback)
    try:
        _validate_fallback_enums(
            fallback.policy,
            fallback.boundary,
            fallback.state_authority,
        )
    except TypeError:
        raise _disallow(
            fallback,
            state="invalid_fallback_carrier",
            boundary="invalid_fallback_carrier",
        ) from fallback.original_error

    request = fallback.original_request
    error = fallback.original_error
    if request.backend is Backend.CPU:
        raise _disallow(
            fallback, state="cpu_request", boundary="cpu_request"
        ) from error
    if error.reason not in _ELIGIBLE_REASONS:
        raise _disallow(
            fallback, state="ineligible_reason", boundary="ineligible_reason"
        ) from error
    _validate_error_context(fallback)
    if fallback.policy is FallbackPolicy.RAISE:
        raise error
    if type(fallback.cpu_state) is not CPUExecutionState:
        raise _disallow(
            fallback,
            state="missing_cpu_state",
            boundary=fallback.boundary.value,
        ) from error
    if fallback.state_authority is not CPUStateAuthority.CPU_AUTHORITATIVE:
        raise _disallow(
            fallback,
            state=fallback.state_authority.value,
            boundary=fallback.boundary.value,
        ) from error

    cpu_request = ExecutionRequest(
        Backend.CPU,
        Device(Backend.CPU, "cpu"),
        request.process,
        request.requirements,
    )
    adapter = fallback.context.resolve(cpu_request)
    return FallbackResolution(
        request,
        cpu_request,
        error,
        fallback.cpu_state,
        adapter,
        fallback.policy,
        fallback.boundary,
        request.backend,
        Backend.CPU,
        error.reason,
    )


def dispatch_cpu_fallback(fallback: FallbackRequest) -> FallbackDispatchResult:
    """Resolve and execute one CPU adapter without recovery or retry."""
    resolution = resolve_cpu_fallback(fallback)
    native_result = resolution.adapter.execute(resolution.cpu_state)
    result = validate_execution_result(resolution.cpu_state, native_result)
    return FallbackDispatchResult(
        resolution, result, _fallback_metadata(resolution)
    )
