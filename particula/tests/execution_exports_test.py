"""Tests for the public execution selection and registration surface."""

import os
import re
import subprocess
import sys
from pathlib import Path

import particula
import particula.execution as execution
import pytest
from particula.execution import (
    Backend,
    CapabilityDeclaration,
    CapabilityMatrix,
    CapabilityRequirements,
    Device,
    ExecutionAdapter,
    ExecutionContext,
    ExecutionRequest,
    Process,
)

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
)

EXCLUDED_EXPORTS = (
    "ExecutionState",
    "MutationScope",
    "MutationDeclaration",
    "BackendResult",
    "ExecutionResult",
    "validate_execution_result",
    "CPUExecutionState",
    "CPUExecutionAdapter",
    "CondensationExecutionConfig",
    "CPUCondensationState",
    "WarpCondensationState",
    "CPUCondensationExecutionState",
    "WarpCondensationExecutionState",
    "CPUCondensationExecutionAdapter",
    "WarpCondensationExecutionAdapter",
)

DIRECT_IMPORT_ONLY_ERRORS = (
    "errors",
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


class _Adapter:
    """Record calls that selection and registration must not make."""

    def __init__(self) -> None:
        """Initialize the execution-call counter."""
        self.calls = 0

    def execute(self, *args: object, **kwargs: object) -> object:
        """Record an invocation that this selection test forbids."""
        del args, kwargs
        self.calls += 1
        return None


def _process() -> Process:
    """Create the standard public process declaration."""
    return Process("condensation")


def _requirements() -> CapabilityRequirements:
    """Create the empty public capability requirements declaration."""
    return CapabilityRequirements(frozenset())


def _matrix() -> CapabilityMatrix:
    """Create a matrix declaring the standard CPU process."""
    device = Device(Backend.CPU, "cpu")
    return CapabilityMatrix(
        frozenset(
            {
                CapabilityDeclaration(device, _process(), _requirements()),
            }
        )
    )


def _context() -> ExecutionContext:
    """Create a context declaring the standard CPU process."""
    return ExecutionContext(_matrix())


def _request() -> ExecutionRequest:
    """Create the standard CPU selection request."""
    return ExecutionRequest(
        Backend.CPU,
        Device(Backend.CPU, "cpu"),
        _process(),
        _requirements(),
    )


def test_execution_exports_are_exact_and_identical_at_package_boundary() -> (
    None
):
    """Test approved selection names are the complete public execution surface."""
    assert tuple(execution.__all__) == EXPECTED_EXPORTS
    for name in EXPECTED_EXPORTS:
        assert getattr(particula, name) is getattr(execution, name)

    assert isinstance(_Adapter(), ExecutionAdapter)
    assert hasattr(execution, "__path__")


def test_public_registration_resolves_by_identity_without_execution() -> None:
    """Test the public registration seam only stores and selects an adapter."""
    context = _context()
    adapter = _Adapter()

    context.register_adapter(_process(), Backend.CPU, adapter)

    assert context.resolve(_request()) is adapter
    assert adapter.calls == 0


@pytest.mark.parametrize(
    ("process", "backend", "adapter", "exception", "message"),
    [
        (
            "process",
            "backend",
            object(),
            TypeError,
            "process must be a Process.",
        ),
        (
            _process(),
            "backend",
            object(),
            TypeError,
            "backend must be a Backend.",
        ),
        (
            _process(),
            Backend.CPU,
            object(),
            TypeError,
            "adapter must have a callable execute attribute.",
        ),
    ],
)
def test_public_registration_rejects_invalid_entries_without_replacing_adapter(
    process: object,
    backend: object,
    adapter: object,
    exception: type[Exception],
    message: str,
) -> None:
    """Test public registration preserves process/backend/adapter validation."""
    context = _context()
    original = _Adapter()
    context.register_adapter(_process(), Backend.CPU, original)

    with pytest.raises(exception, match=f"^{re.escape(message)}$"):
        context.register_adapter(process, backend, adapter)  # type: ignore[arg-type]

    assert context.resolve(_request()) is original
    assert original.calls == 0


def test_public_registration_rejects_duplicate_without_replacing_adapter() -> (
    None
):
    """Test a duplicate public registration retains the first adapter."""
    context = _context()
    original = _Adapter()
    duplicate = _Adapter()
    context.register_adapter(_process(), Backend.CPU, original)

    with pytest.raises(
        ValueError,
        match="^Adapter already registered for process and backend.$",
    ):
        context.register_adapter(_process(), Backend.CPU, duplicate)

    assert context.resolve(_request()) is original
    assert original.calls == duplicate.calls == 0


def test_public_contexts_are_isolated_when_sharing_one_matrix() -> None:
    """Test adapters registered in one context remain unavailable in another."""
    matrix = _matrix()
    registered = ExecutionContext(matrix)
    unregistered = ExecutionContext(matrix)
    adapter = _Adapter()
    registered.register_adapter(_process(), Backend.CPU, adapter)

    assert registered.resolve(_request()) is adapter
    with pytest.raises(
        LookupError,
        match="^No adapter registered for process and backend.$",
    ):
        unregistered.resolve(_request())
    assert adapter.calls == 0


def test_public_registration_uses_static_execute_inspection() -> None:
    """Test public registration avoids dynamic adapter execute lookup."""

    class StaticAdapter:
        """Expose execute statically while rejecting dynamic lookup."""

        def __init__(self) -> None:
            """Initialize the dynamic-lookup counter."""
            self.lookups = 0
            self.calls = 0

        def __getattribute__(self, name: str) -> object:
            """Reject dynamic access to the execution seam."""
            if name == "execute":
                object.__setattr__(
                    self,
                    "lookups",
                    object.__getattribute__(self, "lookups") + 1,
                )
                raise AssertionError("Registration must not inspect execute.")
            return object.__getattribute__(self, name)

        def execute(self, *args: object, **kwargs: object) -> object:
            """Provide the statically discoverable execution seam."""
            del args, kwargs
            self.calls += 1
            return None

    context = _context()
    adapter = StaticAdapter()
    context.register_adapter(_process(), Backend.CPU, adapter)

    assert context.resolve(_request()) is adapter
    assert adapter.lookups == adapter.calls == 0


def test_result_and_cpu_types_remain_off_public_export_boundaries() -> None:
    """Test P3/P4 implementation types remain direct-module-only."""
    for name in EXCLUDED_EXPORTS + DIRECT_IMPORT_ONLY_ERRORS:
        assert name not in execution.__all__
        assert not hasattr(particula, name)


def test_error_taxonomy_remains_direct_import_only_in_fresh_process() -> None:
    """Test errors stay off package attributes until directly imported."""
    root = Path(__file__).parents[2]
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join(
            filter(None, (str(root), os.environ.get("PYTHONPATH")))
        )
    }
    script = f"""
import particula
import particula.execution as execution

names = {DIRECT_IMPORT_ONLY_ERRORS!r}
for name in names:
    assert name not in execution.__all__
    assert not hasattr(execution, name)
    assert not hasattr(particula, name)

from particula.execution.errors import UnknownBackendError
assert UnknownBackendError("cpu").backend == "cpu"
"""

    completed = subprocess.run(  # noqa: S603 -- fixed interpreter and script
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_public_execution_import_is_cpu_only_in_a_fresh_guarded_process() -> (
    None
):
    """Test package initialization never imports or promotes GPU dependencies."""
    root = Path(__file__).parents[2]
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join(
            filter(None, (str(root), os.environ.get("PYTHONPATH")))
        )
    }
    script = """
import builtins
import sys

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "warp" or name.startswith("warp.") or name == "particula.gpu" or name.startswith("particula.gpu."):
        raise AssertionError(f"Unexpected optional backend import: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from particula import (
    Backend, Capability, CapabilityDeclaration, CapabilityMatrix,
    CapabilityRequirements, Device, ExecutionAdapter, ExecutionContext,
    ExecutionRequest, Process,
)
import particula

class Adapter:
    def __init__(self):
        self.calls = 0
    def execute(self, *args, **kwargs):
        self.calls += 1

device = Device(Backend.CPU, "cpu")
process = Process("condensation")
requirements = CapabilityRequirements(frozenset())
matrix = CapabilityMatrix(
    frozenset({CapabilityDeclaration(device, process, requirements)})
)
context = ExecutionContext(matrix)
adapter = Adapter()
assert isinstance(adapter, ExecutionAdapter)
context.register_adapter(process, Backend.CPU, adapter)
assert context.resolve(
    ExecutionRequest(Backend.CPU, device, process, requirements)
) is adapter
assert adapter.calls == 0
assert not any(
    name == "warp" or name.startswith("warp.") or name == "particula.gpu"
    or name.startswith("particula.gpu.") for name in sys.modules
)
assert "gpu" not in particula.__dict__
assert not any(name.endswith("_gpu") for name in particula.__dict__)
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


def test_runnable_and_direct_gpu_kernel_import_boundaries_remain_compatible() -> (
    None
):
    """Test runnable and direct-kernel APIs retain their established paths."""
    from particula import RunnableSequence
    from particula.runnable import RunnableABC
    from particula.runnable import RunnableSequence as DirectSequence

    assert RunnableABC is not None
    assert RunnableSequence is DirectSequence

    pytest.importorskip("warp")
    import particula.gpu.kernels as kernels

    assert callable(kernels.coagulation_step_gpu)
    assert not hasattr(particula, "coagulation_step_gpu")
