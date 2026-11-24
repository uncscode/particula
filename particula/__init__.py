"""Top-level package for the particula particle simulator.

This package exposes the primary aerosol, execution, runnable, and namespace
modules used by the public API. Its frozen execution subset provides
dependency-neutral backend-selection metadata, context-local adapter
registration, the closed capability-error taxonomy, and fallback-policy enums.
Concrete execution mechanics and optional backends remain outside the import
path.
"""

# pylint: disable=unused-import
# flake8: noqa
# pyright: basic

from particula import (
    gas,
    particles,
    util,
    dynamics,
    activity,
    equilibria,
)
from particula.extension_fields import ExtensionFields, FieldSpec
from particula.aerosol import Aerosol
from particula.aerosol_builder import AerosolBuilder
from particula.execution import (
    CPUExecutionContext,
    Backend,
    Capability,
    CapabilityDeclaration,
    CapabilityMatrix,
    CapabilityRequirements,
    Device,
    ExecutionAdapter,
    ExecutionCapabilityError,
    ExecutionCapabilityReason,
    ExecutionContext,
    ExecutionRequest,
    FallbackBoundary,
    FallbackDisallowedError,
    FallbackPolicy,
    CPUStateAuthority,
    InvalidExecutionStateError,
    Process,
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
from particula.runnable import RunnableSequence
from particula.box_process import BoxProcess, FunctionProcess, BoxContext

import logging
from particula.logger_setup import setup

__version__ = "0.3.6"

# Libraries leave logging configuration and output destinations to callers.
logger = logging.getLogger("particula")
logger.addHandler(logging.NullHandler())


def __getattr__(name: str):
    """Expose prepared simulation types without loading optional runtimes."""
    if name in (
        "PreparedSimulation",
        "SimulationDiagnostics",
        "SimulationCheckpoint",
    ):
        from particula import execution

        return getattr(execution, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from particula.execution.extension_registry import (
    PreparedOperation,
    ProcessCapabilities,
    ProcessRegistry,
)
from particula.execution.process_extensions import (
    GPUPreparationContext,
    GPUProcessAdapter,
    GPUProcessPlan,
    GPUStateSpec,
)

from particula.cpu_transfer import CPUTransfer
