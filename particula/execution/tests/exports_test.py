"""Regression tests for the frozen execution package boundary."""

import particula.execution as execution
from particula.execution import errors, fallback

EXPECTED_EXPORTS = (
    "Backend",
    "Device",
    "Process",
    "Capability",
    "CapabilityRequirements",
    "CapabilityDeclaration",
    "CapabilityMatrix",
    "ExecutionRequest",
    "ExecutionAdapter",
    "ExecutionContext",
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
    "FallbackPolicy",
    "FallbackBoundary",
    "CPUStateAuthority",
)


def test_execution_exports_are_exact_and_identity_preserving() -> None:
    """Test the public names match their concrete value definitions."""
    assert tuple(execution.__all__) == EXPECTED_EXPORTS
    for name in errors.__all__:
        assert getattr(execution, name) is getattr(errors, name)
    for name in ("FallbackPolicy", "FallbackBoundary", "CPUStateAuthority"):
        assert getattr(execution, name) is getattr(fallback, name)


def test_execution_keeps_concrete_fallback_mechanics_private() -> None:
    """Test fallback operations and carriers remain outside the public surface."""
    for name in (
        "errors",
        "fallback",
        "FallbackRequest",
        "FallbackResolution",
        "FallbackDispatchResult",
        "resolve_cpu_fallback",
        "dispatch_cpu_fallback",
        "CPUExecutionState",
        "CPUExecutionAdapter",
        "ExecutionResult",
    ):
        assert name not in execution.__all__
