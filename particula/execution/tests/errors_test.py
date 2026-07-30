"""Tests for the direct execution capability error taxonomy."""

import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from particula.execution.errors import (
    ExecutionCapabilityError,
    ExecutionCapabilityReason,
    FallbackDisallowedError,
    InvalidExecutionStateError,
    UnavailableDeviceError,
    UnavailableExecutionTargetError,
    UnavailableRuntimeError,
    UnknownBackendError,
    UnknownDeviceError,
    UnknownExecutionTargetError,
    UnsupportedCapabilityError,
    UnsupportedExecutionRequestError,
    UnsupportedProcessError,
)


def _message(
    class_name: str,
    reason: ExecutionCapabilityReason,
    *,
    backend: str | None = None,
    device: str | None = None,
    process: str | None = None,
    capability: str | None = None,
    state: str | None = None,
    fallback_boundary: str | None = None,
) -> str:
    """Build the stable public error message expected by the contract."""
    return (
        f"{class_name}(reason={reason.value}, backend={backend!r}, "
        f"device={device!r}, process={process!r}, capability={capability!r}, "
        f"state={state!r}, fallback_boundary={fallback_boundary!r})"
    )


def test_direct_module_exports_closed_taxonomy_and_reason_codes() -> None:
    """Test the direct-module public contract is exact and ordered."""
    import particula.execution.errors as errors

    assert errors.__all__ == [
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
    assert {
        name: reason.value
        for name, reason in ExecutionCapabilityReason.__members__.items()
    } == {
        "UNKNOWN_BACKEND": "unknown_backend",
        "UNKNOWN_DEVICE": "unknown_device",
        "RUNTIME_UNAVAILABLE": "runtime_unavailable",
        "DEVICE_UNAVAILABLE": "device_unavailable",
        "PROCESS_UNSUPPORTED": "process_unsupported",
        "CAPABILITY_UNSUPPORTED": "capability_unsupported",
        "INVALID_STATE": "invalid_state",
        "FALLBACK_DISALLOWED": "fallback_disallowed",
    }


def test_root_error_stores_context_and_renders_deterministically() -> None:
    """Test direct root construction preserves all structured context."""
    error = ExecutionCapabilityError(
        ExecutionCapabilityReason.CAPABILITY_UNSUPPORTED,
        backend="warp",
        device="cuda:0",
        process="condensation",
        capability="graph_capture",
        state="active",
        fallback_boundary="resident",
    )

    assert error.reason is ExecutionCapabilityReason.CAPABILITY_UNSUPPORTED
    assert error.backend == "warp"
    assert error.device == "cuda:0"
    assert error.process == "condensation"
    assert error.capability == "graph_capture"
    assert error.state == "active"
    assert error.fallback_boundary == "resident"
    assert str(error) == _message(
        "ExecutionCapabilityError",
        ExecutionCapabilityReason.CAPABILITY_UNSUPPORTED,
        backend="warp",
        device="cuda:0",
        process="condensation",
        capability="graph_capture",
        state="active",
        fallback_boundary="resident",
    )


def test_concrete_error_hierarchy_has_stable_categories() -> None:
    """Test concrete errors use their declared category bases only."""
    assert UnknownExecutionTargetError.__bases__ == (ExecutionCapabilityError,)
    assert UnavailableExecutionTargetError.__bases__ == (
        ExecutionCapabilityError,
    )
    assert UnsupportedExecutionRequestError.__bases__ == (
        ExecutionCapabilityError,
    )

    concrete_errors = (
        UnknownBackendError,
        UnknownDeviceError,
        UnavailableRuntimeError,
        UnavailableDeviceError,
        UnsupportedProcessError,
        UnsupportedCapabilityError,
        InvalidExecutionStateError,
        FallbackDisallowedError,
    )
    for error_type in concrete_errors:
        assert issubclass(error_type, ExecutionCapabilityError)

    assert issubclass(UnknownBackendError, UnknownExecutionTargetError)
    assert issubclass(UnknownDeviceError, UnknownExecutionTargetError)
    assert issubclass(UnavailableRuntimeError, UnavailableExecutionTargetError)
    assert issubclass(UnavailableDeviceError, UnavailableExecutionTargetError)
    assert issubclass(UnsupportedProcessError, UnsupportedExecutionRequestError)
    assert issubclass(
        UnsupportedCapabilityError,
        UnsupportedExecutionRequestError,
    )
    assert not issubclass(
        InvalidExecutionStateError, UnknownExecutionTargetError
    )
    assert not issubclass(
        InvalidExecutionStateError,
        UnavailableExecutionTargetError,
    )
    assert not issubclass(
        InvalidExecutionStateError,
        UnsupportedExecutionRequestError,
    )
    assert not issubclass(FallbackDisallowedError, UnknownExecutionTargetError)
    assert not issubclass(
        FallbackDisallowedError,
        UnavailableExecutionTargetError,
    )
    assert not issubclass(
        FallbackDisallowedError,
        UnsupportedExecutionRequestError,
    )


@pytest.mark.parametrize(
    ("error", "reason", "context"),
    [
        (
            UnknownBackendError("cpu"),
            ExecutionCapabilityReason.UNKNOWN_BACKEND,
            {"backend": "cpu"},
        ),
        (
            UnknownDeviceError("cuda:0"),
            ExecutionCapabilityReason.UNKNOWN_DEVICE,
            {"device": "cuda:0"},
        ),
        (
            UnavailableRuntimeError("warp"),
            ExecutionCapabilityReason.RUNTIME_UNAVAILABLE,
            {"backend": "warp"},
        ),
        (
            UnavailableDeviceError("cuda:0"),
            ExecutionCapabilityReason.DEVICE_UNAVAILABLE,
            {"device": "cuda:0"},
        ),
        (
            UnsupportedProcessError("condensation"),
            ExecutionCapabilityReason.PROCESS_UNSUPPORTED,
            {"process": "condensation"},
        ),
        (
            UnsupportedCapabilityError("graph_capture"),
            ExecutionCapabilityReason.CAPABILITY_UNSUPPORTED,
            {"capability": "graph_capture"},
        ),
        (
            InvalidExecutionStateError("finalized"),
            ExecutionCapabilityReason.INVALID_STATE,
            {"state": "finalized"},
        ),
        (
            FallbackDisallowedError("resident"),
            ExecutionCapabilityReason.FALLBACK_DISALLOWED,
            {"fallback_boundary": "resident"},
        ),
    ],
)
def test_concrete_errors_render_omitted_context_as_none(
    error: ExecutionCapabilityError,
    reason: ExecutionCapabilityReason,
    context: dict[str, str],
) -> None:
    """Test required-only constructors retain omitted context as ``None``."""
    fields = (
        "backend",
        "device",
        "process",
        "capability",
        "state",
        "fallback_boundary",
    )

    assert error.reason is reason
    for field in fields:
        assert getattr(error, field) == context.get(field)
    assert str(error) == _message(type(error).__name__, reason, **context)


@pytest.mark.parametrize(
    ("error", "reason", "context"),
    [
        (
            UnknownBackendError("cpu"),
            ExecutionCapabilityReason.UNKNOWN_BACKEND,
            {"backend": "cpu"},
        ),
        (
            UnknownDeviceError("cuda:0", backend="warp"),
            ExecutionCapabilityReason.UNKNOWN_DEVICE,
            {"backend": "warp", "device": "cuda:0"},
        ),
        (
            UnavailableRuntimeError("warp"),
            ExecutionCapabilityReason.RUNTIME_UNAVAILABLE,
            {"backend": "warp"},
        ),
        (
            UnavailableDeviceError("cuda:0", backend="warp"),
            ExecutionCapabilityReason.DEVICE_UNAVAILABLE,
            {"backend": "warp", "device": "cuda:0"},
        ),
        (
            UnsupportedProcessError(
                "condensation",
                backend="warp",
                device="cuda:0",
            ),
            ExecutionCapabilityReason.PROCESS_UNSUPPORTED,
            {
                "backend": "warp",
                "device": "cuda:0",
                "process": "condensation",
            },
        ),
        (
            UnsupportedCapabilityError(
                "graph_capture",
                backend="warp",
                device="cuda:0",
                process="condensation",
            ),
            ExecutionCapabilityReason.CAPABILITY_UNSUPPORTED,
            {
                "backend": "warp",
                "device": "cuda:0",
                "process": "condensation",
                "capability": "graph_capture",
            },
        ),
        (
            InvalidExecutionStateError(
                "finalized",
                backend="warp",
                device="cuda:0",
                process="condensation",
                capability="graph_capture",
            ),
            ExecutionCapabilityReason.INVALID_STATE,
            {
                "backend": "warp",
                "device": "cuda:0",
                "process": "condensation",
                "capability": "graph_capture",
                "state": "finalized",
            },
        ),
        (
            FallbackDisallowedError(
                "resident",
                backend="warp",
                device="cuda:0",
                process="condensation",
                capability="graph_capture",
                state="active",
            ),
            ExecutionCapabilityReason.FALLBACK_DISALLOWED,
            {
                "backend": "warp",
                "device": "cuda:0",
                "process": "condensation",
                "capability": "graph_capture",
                "state": "active",
                "fallback_boundary": "resident",
            },
        ),
    ],
)
def test_concrete_errors_store_fixed_reason_and_context(
    error: ExecutionCapabilityError,
    reason: ExecutionCapabilityReason,
    context: dict[str, str],
) -> None:
    """Test each concrete error has fixed reasons and exact rendering."""
    fields = (
        "backend",
        "device",
        "process",
        "capability",
        "state",
        "fallback_boundary",
    )

    assert error.reason is reason
    for field in fields:
        assert getattr(error, field) == context.get(field)
    assert str(error) == _message(type(error).__name__, reason, **context)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: ExecutionCapabilityError(
            cast(ExecutionCapabilityReason, "unknown_backend")
        ),
        lambda: ExecutionCapabilityError(
            ExecutionCapabilityReason.UNKNOWN_BACKEND,
            backend=cast(str, 1),
        ),
        lambda: ExecutionCapabilityError(
            ExecutionCapabilityReason.UNKNOWN_BACKEND,
            device=cast(str | None, []),
        ),
        lambda: UnknownBackendError(cast(str, 1)),
    ],
)
def test_invalid_reason_or_context_type_raises_type_error(constructor) -> None:
    """Test structured error fields accept only their documented types."""
    with pytest.raises(TypeError):
        constructor()


def test_standard_exception_chaining_does_not_change_message() -> None:
    """Test ordinary chaining preserves causes outside deterministic rendering."""
    sentinel = RuntimeError("provider detail")
    try:
        raise UnavailableRuntimeError("warp") from sentinel
    except UnavailableRuntimeError as error:
        assert error.__cause__ is sentinel
        assert str(error) == _message(
            "UnavailableRuntimeError",
            ExecutionCapabilityReason.RUNTIME_UNAVAILABLE,
            backend="warp",
        )
        assert str(sentinel) not in str(error)
        assert "provider detail" not in str(error)


def test_importing_errors_does_not_load_optional_backend() -> None:
    """Test the taxonomy imports without optional runtime or GPU probing."""
    root = Path(__file__).parents[3]
    environment = os.environ | {"PYTHONPATH": str(root)}
    script = """
import builtins
import sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'warp' or name.startswith('warp.') or name == 'particula.gpu' or name.startswith('particula.gpu.'):
        raise AssertionError(name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from particula.execution.errors import UnknownBackendError
assert UnknownBackendError('cpu').backend == 'cpu'
assert not any(name == 'warp' or name.startswith('warp.') or name == 'particula.gpu' or name.startswith('particula.gpu.') for name in sys.modules)
"""
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
