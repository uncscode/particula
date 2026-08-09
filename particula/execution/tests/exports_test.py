"""Regression tests for the frozen execution package boundary."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import particula
import particula.execution as execution
from particula.execution import errors, fallback, values

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

EXPECTED_ERROR_EXPORTS = (
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
)

DENIED_PUBLIC_NAMES = (
    "errors",
    "fallback",
    "AvailabilityDecision",
    "AvailabilityProvider",
    "_CPUAvailabilityProvider",
    "_WarpAvailabilityProvider",
    "resolve_availability",
    "FallbackRequest",
    "FallbackResolution",
    "FallbackDispatchResult",
    "resolve_cpu_fallback",
    "dispatch_cpu_fallback",
    "ExecutionState",
    "MutationScope",
    "MutationDeclaration",
    "BackendResult",
    "ExecutionResult",
    "validate_execution_result",
    "CPUExecutionState",
    "CPUExecutionAdapter",
    "ResidentSession",
    "ResidentStepGuard",
    "ResidentStepToken",
    "ResidentCheckpointController",
    "CheckpointPayload",
    "GPUResourceRegistry",
    "BrownianCoagulationConfig",
    "CPUCoagulationState",
    "CPUCoagulationResult",
    "WarpBrownianCoagulationState",
    "WarpBrownianCoagulationResult",
    "CondensationExecutionConfig",
    "CPUCondensationState",
    "WarpCondensationState",
    "CPUCondensationExecutionState",
    "WarpCondensationExecutionState",
    "CPUCondensationExecutionAdapter",
    "WarpCondensationExecutionAdapter",
    "STREAM_SCHEMA_VERSION",
    "MAX_LOGICAL_BOX_ID_BYTES",
    "MAX_ROOT_SEED",
    "PROCESS_IDS",
    "SUPPORTED_PROCESS_IDS",
    "StreamKey",
    "StreamDescriptor",
    "StreamRegistry",
)

FORBIDDEN_MODULE_PREFIXES = (
    "warp",
    "particula.gpu",
    "particula.execution.availability",
    "particula.execution.adapters",
    "particula.execution.gpu_session",
    "particula.execution.gpu_resources",
    "particula.execution.checkpoint",
    "particula.execution.state_updates",
    "particula.execution.scheduler",
    "particula.execution.resident_scheduler",
    "particula.execution.diagnostics",
    "particula.execution.process_graph",
    "particula.execution.thermodynamic_updates",
    "particula.execution.rng",
)


def test_execution_exports_are_exact_and_identity_preserving() -> None:
    """Test the public names match their concrete value definitions."""
    assert tuple(execution.__all__) == EXPECTED_EXPORTS
    assert tuple(errors.__all__) == EXPECTED_ERROR_EXPORTS
    for name in errors.__all__:
        assert getattr(execution, name) is getattr(errors, name)
    for name in ("FallbackPolicy", "FallbackBoundary", "CPUStateAuthority"):
        assert getattr(execution, name) is getattr(fallback, name)
        assert getattr(execution, name) is getattr(values, name)
    for name in EXPECTED_EXPORTS:
        assert getattr(particula, name) is getattr(execution, name)


def test_execution_keeps_concrete_boundary_names_private() -> None:
    """Test fallback mechanics and related concrete names stay private."""
    for name in DENIED_PUBLIC_NAMES:
        assert name not in execution.__all__
        assert not hasattr(particula, name)


def test_public_execution_import_is_cpu_only_in_a_fresh_guarded_process() -> (
    None
):
    """Test package initialization never imports or promotes GPU dependencies."""
    root = Path(__file__).parents[3]
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join(
            filter(None, (str(root), os.environ.get("PYTHONPATH")))
        )
    }
    script = f"""
import builtins
import sys

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if (
        name == "warp"
        or name.startswith("warp.")
        or name == "particula.gpu"
        or name.startswith("particula.gpu.")
    ):
        raise AssertionError(f"Unexpected optional backend import: {{name}}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import

import particula
import particula.execution as execution
from particula.execution import errors, fallback

expected = {EXPECTED_EXPORTS!r}
assert tuple(execution.__all__) == expected
for name in expected:
    assert getattr(particula, name) is getattr(execution, name)

for name in {EXPECTED_ERROR_EXPORTS!r}:
    assert getattr(particula, name) is getattr(execution, name)

assert particula.FallbackPolicy is fallback.FallbackPolicy
assert particula.FallbackBoundary is fallback.FallbackBoundary
assert particula.CPUStateAuthority is fallback.CPUStateAuthority
assert tuple(errors.__all__) == {EXPECTED_ERROR_EXPORTS!r}
assert not hasattr(particula, "errors")
assert not hasattr(particula, "fallback")
assert not any(
    module == prefix or module.startswith(prefix + ".")
    for prefix in {FORBIDDEN_MODULE_PREFIXES!r}
    for module in sys.modules
)
"""

    completed = subprocess.run(  # noqa: S603 -- fixed interpreter and script
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
