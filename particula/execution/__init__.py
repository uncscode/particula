"""Declare dependency-neutral public execution selection and error contracts.

This module validates immutable backend, device, process, and capability
declarations. An ``ExecutionContext`` capability-validates a typed request and
returns one exact context-local adapter. Selection does not invoke adapters,
load or probe optional backends, resolve native devices, transfer state, or
validate P3 state or result contracts.

P3 result validation retains caller-owned state by identity, keeps backend
payloads and results opaque, and records explicit in-place mutation permission.
It does not import a backend, provide fallback behavior, transfer state, or
execute adapters. P2 selection remains context-local and callable-only.

The direct-module-only CPU adapter is a narrow P4 dispatch boundary. It
preflights concrete CPU state and execution controls, delegates exactly once to
a caller-supplied CPU runnable, and retains the original state and aerosol by
identity. It does not select adapters, validate process physics, transfer or
convert data, or load optional backends.

The direct-module-only condensation profile catalogue declares semantic support
only. It neither selects an adapter nor validates runtime or native-device
availability, imports optional backends, allocates state, or exposes a
user-facing API.

The public API is a frozen, dependency-neutral 26-name selection, capability
error, and fallback-policy surface. Its three public fallback policy enums are
``FallbackPolicy``, ``FallbackBoundary``, and ``CPUStateAuthority``. Concrete
fallback mechanics and their carriers remain direct-module-only. Importing this
package does not import optional backends, Warp, or ``particula.gpu``.
"""

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from itertools import product
from math import isfinite
from numbers import Integral, Rational, Real
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

__all__ = [
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
]

if TYPE_CHECKING:
    from particula.aerosol import Aerosol
    from particula.runnable import RunnableABC

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class Backend(str, Enum):
    """Identify a declared execution backend without loading it.

    Attributes:
        CPU: The declared CPU backend.
        WARP: The declared Warp backend.
    """

    CPU = "cpu"
    WARP = "warp"


@dataclass(frozen=True)
class Device:
    """Declare an immutable backend and opaque native device identifier.

    The native identifier is retained verbatim and is never parsed, resolved,
    or checked against backend availability.

    Args:
        backend: Declared backend for the device.
        native: Nonempty native identifier without surrounding whitespace.

    Raises:
        TypeError: If ``backend`` is not a Backend or ``native`` is not a str.
        ValueError: If ``native`` is empty or has surrounding whitespace.
    """

    backend: Backend
    native: str

    def __post_init__(self) -> None:
        """Validate the declared backend and opaque native identifier.

        Raises:
            TypeError: If the backend or native identifier has the wrong type.
            ValueError: If the native identifier is empty or padded.
        """
        if not isinstance(self.backend, Backend):
            raise TypeError("Device.backend must be a Backend.")
        if not isinstance(self.native, str):
            raise TypeError("Device.native must be a str.")
        if not self.native or self.native != self.native.strip():
            raise ValueError(
                "Device.native must be a nonempty str without surrounding "
                "whitespace."
            )


@dataclass(frozen=True)
class Process:
    """Declare an immutable, validated process name.

    Args:
        name: Lowercase identifier-style process name.

    Raises:
        TypeError: If ``name`` is not a str.
        ValueError: If ``name`` is not a lowercase identifier-style name.
    """

    name: str

    def __post_init__(self) -> None:
        """Validate the process name.

        Raises:
            TypeError: If the name is not a str.
            ValueError: If the name is not a lowercase identifier-style name.
        """
        _validate_name(self.name, "Process.name")


@dataclass(frozen=True)
class Capability:
    """Declare an immutable, validated capability name.

    Args:
        name: Lowercase identifier-style capability name.

    Raises:
        TypeError: If ``name`` is not a str.
        ValueError: If ``name`` is not a lowercase identifier-style name.
    """

    name: str

    def __post_init__(self) -> None:
        """Validate the capability name.

        Raises:
            TypeError: If the name is not a str.
            ValueError: If the name is not a lowercase identifier-style name.
        """
        _validate_name(self.name, "Capability.name")


def _validate_name(value: object, field_name: str) -> None:
    """Validate a lowercase identifier-style metadata name.

    Args:
        value: Candidate name to validate.
        field_name: Qualified field name used in error messages.

    Raises:
        TypeError: If ``value`` is not a str.
        ValueError: If ``value`` does not match the supported name pattern.
    """
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a str.")
    if not _NAME_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must match ^[a-z][a-z0-9_]*$.")


def _isfinite_real(value: Real) -> bool:
    """Return whether a real scalar is finite without coercing rationals.

    ``math.isfinite`` converts its input to ``float``. That conversion can
    overflow for finite rational values such as large ``Fraction`` instances.

    Args:
        value: Real scalar already validated by the caller.

    Returns:
        True when the scalar is finite.
    """
    return isinstance(value, Rational) or isfinite(value)


@dataclass(frozen=True)
class CapabilityRequirements:
    """Declare an immutable exact set of required capabilities.

    The empty set is valid. Requirements are declarations, not composable
    features inferred by the matrix.

    Args:
        values: Exact frozenset of capability declarations.

    Raises:
        TypeError: If ``values`` is not a frozenset of Capability instances.
    """

    values: frozenset[Capability]

    def __post_init__(self) -> None:
        """Validate the exact immutable capability collection type.

        Raises:
            TypeError: If the collection or one of its members is invalid.
        """
        if type(self.values) is not frozenset:
            raise TypeError(
                "CapabilityRequirements.values must be a frozenset."
            )
        if not all(isinstance(value, Capability) for value in self.values):
            raise TypeError(
                "CapabilityRequirements.values must contain only Capability "
                "instances."
            )


@dataclass(frozen=True)
class CapabilityDeclaration:
    """Declare immutable capability support for one device and process.

    Args:
        device: Device receiving the declaration.
        process: Process supported by the device.
        requirements: Exact required capability set for the support entry.

    Raises:
        TypeError: If any field is not its declared metadata type.
    """

    device: Device
    process: Process
    requirements: CapabilityRequirements

    def __post_init__(self) -> None:
        """Validate the declaration's typed value objects.

        Raises:
            TypeError: If any declaration field has the wrong type.
        """
        if not isinstance(self.device, Device):
            raise TypeError("CapabilityDeclaration.device must be a Device.")
        if not isinstance(self.process, Process):
            raise TypeError("CapabilityDeclaration.process must be a Process.")
        if not isinstance(self.requirements, CapabilityRequirements):
            raise TypeError(
                "CapabilityDeclaration.requirements must be a "
                "CapabilityRequirements."
            )


@dataclass(frozen=True)
class CapabilityMatrix:
    """Provide immutable, pure lookup of exact capability declarations.

    Nonempty requirements must match one complete declaration. Empty
    requirements are supported when at least one declaration has the requested
    device and process. Lookup neither probes backend availability nor selects
    an execution adapter.

    Args:
        declarations: Exact frozenset of capability declarations.

    Raises:
        TypeError: If ``declarations`` is not a frozenset of declarations.
    """

    declarations: frozenset[CapabilityDeclaration]

    def __post_init__(self) -> None:
        """Validate the exact immutable declaration collection type.

        Raises:
            TypeError: If the collection or one of its members is invalid.
        """
        if type(self.declarations) is not frozenset:
            raise TypeError(
                "CapabilityMatrix.declarations must be a frozenset."
            )
        if not all(
            isinstance(declaration, CapabilityDeclaration)
            for declaration in self.declarations
        ):
            raise TypeError(
                "CapabilityMatrix.declarations must contain only "
                "CapabilityDeclaration instances."
            )

    def supports(
        self,
        device: Device,
        process: Process,
        requirements: CapabilityRequirements,
    ) -> bool:
        """Return whether the matrix exactly declares a capability request.

        Args:
            device: Device requested for the process.
            process: Process requested on the device.
            requirements: Exact capability requirements for the request.

        Returns:
            True if the request is declared by this matrix; otherwise, False.

        Raises:
            TypeError: If an argument is not its declared metadata type.
        """
        _validate_request(device, process, requirements)
        if requirements.values:
            return (
                CapabilityDeclaration(device, process, requirements)
                in self.declarations
            )
        return any(
            declaration.device == device and declaration.process == process
            for declaration in self.declarations
        )

    def require(
        self,
        device: Device,
        process: Process,
        requirements: CapabilityRequirements,
    ) -> None:
        """Require a declared capability request without executing it.

        Args:
            device: Device requested for the process.
            process: Process requested on the device.
            requirements: Exact capability requirements for the request.

        Raises:
            TypeError: If an argument is not its declared metadata type.
            ValueError: If the valid request is not declared by this matrix.
        """
        _validate_request(device, process, requirements)
        if self.supports(device, process, requirements):
            return
        raise ValueError(
            "Unsupported capability declaration: "
            + repr(CapabilityDeclaration(device, process, requirements))
        )


class CondensationExecutionMode(str, Enum):
    """Declare the semantic condensation time-stepping mode.

    This vocabulary describes configuration semantics only; it does not select
    an execution implementation.

    Attributes:
        EQUAL_STEP: Equal-step condensation semantics.
        STAGGERED: Staggered condensation semantics.
    """

    EQUAL_STEP = "equal_step"
    STAGGERED = "staggered"


class CondensationActivityMode(str, Enum):
    """Declare the semantic condensation activity-coefficient mode.

    ``NONREPRESENTABLE`` is a valid semantic category that a profile may not
    support. It does not inspect or detect concrete activity implementations.

    Attributes:
        IDEAL: Ideal activity-coefficient semantics.
        KAPPA: Kappa activity-coefficient semantics.
        NONREPRESENTABLE: Semantics not representable by a profile.
    """

    IDEAL = "ideal"
    KAPPA = "kappa"
    NONREPRESENTABLE = "nonrepresentable"


class CondensationSurfaceMode(str, Enum):
    """Declare the semantic condensation surface-tension mode.

    ``NONREPRESENTABLE`` is a valid semantic category that a profile may not
    support. It does not inspect or detect concrete surface implementations.

    Attributes:
        STATIC: Static surface-tension semantics.
        COMPOSITION_WEIGHTED: Composition-weighted surface-tension semantics.
        NONREPRESENTABLE: Semantics not representable by a profile.
    """

    STATIC = "static"
    COMPOSITION_WEIGHTED = "composition_weighted"
    NONREPRESENTABLE = "nonrepresentable"


@dataclass(frozen=True)
class CondensationConfiguration:
    """Declare immutable semantic condensation configuration metadata.

    This metadata does not inspect or translate strategy objects, device names,
    optional-backend sidecars, or concrete activity and surface implementations.
    It also does not select an adapter or determine runtime availability.

    Args:
        execution_mode: Semantic time-stepping mode.
        latent_heat: Whether latent-heat semantics are required.
        activity_mode: Semantic activity-coefficient mode.
        surface_mode: Semantic surface-tension mode.

    Raises:
        TypeError: If a field is not its exact declared metadata type.
    """

    execution_mode: CondensationExecutionMode
    latent_heat: bool
    activity_mode: CondensationActivityMode
    surface_mode: CondensationSurfaceMode

    def __post_init__(self) -> None:
        """Validate the exact semantic configuration field types."""
        if not isinstance(self.execution_mode, CondensationExecutionMode):
            raise TypeError(
                "CondensationConfiguration.execution_mode must be a "
                "CondensationExecutionMode."
            )
        if type(self.latent_heat) is not bool:
            raise TypeError(
                "CondensationConfiguration.latent_heat must be a bool."
            )
        if not isinstance(self.activity_mode, CondensationActivityMode):
            raise TypeError(
                "CondensationConfiguration.activity_mode must be a "
                "CondensationActivityMode."
            )
        if not isinstance(self.surface_mode, CondensationSurfaceMode):
            raise TypeError(
                "CondensationConfiguration.surface_mode must be a "
                "CondensationSurfaceMode."
            )


CONDENSATION_PROCESS = Process("condensation")
CPU_CONDENSATION_PROFILE_DEVICE = Device(Backend.CPU, "cpu")
# "profile" is an opaque catalogue-only identifier, never a native-device
# claim or a device accepted or normalized by ``ExecutionContext.resolve()``.
WARP_CONDENSATION_PROFILE_DEVICE = Device(Backend.WARP, "profile")

CONDENSATION_EQUAL_STEP_CAPABILITY = Capability("condensation_equal_step")
CONDENSATION_STAGGERED_CAPABILITY = Capability("condensation_staggered")
CONDENSATION_LATENT_HEAT_CAPABILITY = Capability("condensation_latent_heat")
CONDENSATION_NO_LATENT_HEAT_CAPABILITY = Capability(
    "condensation_no_latent_heat"
)
CONDENSATION_IDEAL_ACTIVITY_CAPABILITY = Capability(
    "condensation_ideal_activity"
)
CONDENSATION_KAPPA_ACTIVITY_CAPABILITY = Capability(
    "condensation_kappa_activity"
)
CONDENSATION_NONREPRESENTABLE_ACTIVITY_CAPABILITY = Capability(
    "condensation_nonrepresentable_activity"
)
CONDENSATION_STATIC_SURFACE_CAPABILITY = Capability(
    "condensation_static_surface"
)
CONDENSATION_COMPOSITION_WEIGHTED_SURFACE_CAPABILITY = Capability(
    "condensation_composition_weighted_surface"
)
CONDENSATION_NONREPRESENTABLE_SURFACE_CAPABILITY = Capability(
    "condensation_nonrepresentable_surface"
)

_CONDENSATION_EXECUTION_CAPABILITIES = {
    CondensationExecutionMode.EQUAL_STEP: CONDENSATION_EQUAL_STEP_CAPABILITY,
    CondensationExecutionMode.STAGGERED: CONDENSATION_STAGGERED_CAPABILITY,
}
_CONDENSATION_LATENT_HEAT_CAPABILITIES = {
    True: CONDENSATION_LATENT_HEAT_CAPABILITY,
    False: CONDENSATION_NO_LATENT_HEAT_CAPABILITY,
}
_CONDENSATION_ACTIVITY_CAPABILITIES = {
    CondensationActivityMode.IDEAL: CONDENSATION_IDEAL_ACTIVITY_CAPABILITY,
    CondensationActivityMode.KAPPA: CONDENSATION_KAPPA_ACTIVITY_CAPABILITY,
    CondensationActivityMode.NONREPRESENTABLE: (
        CONDENSATION_NONREPRESENTABLE_ACTIVITY_CAPABILITY
    ),
}
_CONDENSATION_SURFACE_CAPABILITIES = {
    CondensationSurfaceMode.STATIC: CONDENSATION_STATIC_SURFACE_CAPABILITY,
    CondensationSurfaceMode.COMPOSITION_WEIGHTED: (
        CONDENSATION_COMPOSITION_WEIGHTED_SURFACE_CAPABILITY
    ),
    CondensationSurfaceMode.NONREPRESENTABLE: (
        CONDENSATION_NONREPRESENTABLE_SURFACE_CAPABILITY
    ),
}


def get_condensation_requirements(
    configuration: CondensationConfiguration,
) -> CapabilityRequirements:
    """Return the exact four-axis semantic requirements for a configuration.

    The pure mapping does not compose declarations, select an adapter, or
    inspect runtime, device, or optional-backend state.

    Args:
        configuration: Validated semantic condensation configuration.

    Returns:
        A new requirements value containing exactly one capability per axis.

    Raises:
        TypeError: If ``configuration`` is not a CondensationConfiguration.
    """
    if not isinstance(configuration, CondensationConfiguration):
        raise TypeError("configuration must be a CondensationConfiguration.")
    return CapabilityRequirements(
        frozenset(
            {
                _CONDENSATION_EXECUTION_CAPABILITIES[
                    configuration.execution_mode
                ],
                _CONDENSATION_LATENT_HEAT_CAPABILITIES[
                    configuration.latent_heat
                ],
                _CONDENSATION_ACTIVITY_CAPABILITIES[
                    configuration.activity_mode
                ],
                _CONDENSATION_SURFACE_CAPABILITIES[configuration.surface_mode],
            }
        )
    )


def _condensation_declarations() -> frozenset[CapabilityDeclaration]:
    """Build the immutable CPU and declarative Warp profile catalogue.

    The Warp rows describe semantics only and make no native-device or runtime
    availability claim.
    """
    all_configurations = tuple(
        CondensationConfiguration(*values)
        for values in product(
            CondensationExecutionMode,
            (True, False),
            CondensationActivityMode,
            CondensationSurfaceMode,
        )
    )
    warp_configurations = (
        configuration
        for configuration in all_configurations
        if configuration.execution_mode is CondensationExecutionMode.EQUAL_STEP
        and configuration.activity_mode
        is not CondensationActivityMode.NONREPRESENTABLE
        and configuration.surface_mode
        is not CondensationSurfaceMode.NONREPRESENTABLE
    )
    return frozenset(
        CapabilityDeclaration(
            device,
            CONDENSATION_PROCESS,
            get_condensation_requirements(configuration),
        )
        for device, configurations in (
            (CPU_CONDENSATION_PROFILE_DEVICE, all_configurations),
            (WARP_CONDENSATION_PROFILE_DEVICE, warp_configurations),
        )
        for configuration in configurations
    )


CONDENSATION_CAPABILITY_MATRIX = CapabilityMatrix(_condensation_declarations())


def _condensation_profile_device(backend: Backend) -> Device:
    """Return the fixed catalogue-only device for a validated backend.

    This helper intentionally accepts no native-device identifier and cannot
    resolve or validate a runtime device.

    Args:
        backend: Backend whose fixed profile device is requested.

    Returns:
        The backend's opaque catalogue-only profile device.

    Raises:
        TypeError: If ``backend`` is not a Backend.
    """
    if not isinstance(backend, Backend):
        raise TypeError("backend must be a Backend.")
    return {
        Backend.CPU: CPU_CONDENSATION_PROFILE_DEVICE,
        Backend.WARP: WARP_CONDENSATION_PROFILE_DEVICE,
    }[backend]


def condensation_profile_supports(
    backend: Backend,
    configuration: CondensationConfiguration,
) -> bool:
    """Return whether a fixed backend profile declares a configuration.

    This pure catalogue query does not accept or parse a native device, select
    an adapter, import an optional backend, or determine runtime availability.

    Args:
        backend: Catalogue backend profile to query.
        configuration: Semantic condensation configuration to query.

    Returns:
        True when the exact configuration is declared by the backend profile.
    """
    device = _condensation_profile_device(backend)
    requirements = get_condensation_requirements(configuration)
    return CONDENSATION_CAPABILITY_MATRIX.supports(
        device,
        CONDENSATION_PROCESS,
        requirements,
    )


def require_condensation_profile(
    backend: Backend,
    configuration: CondensationConfiguration,
) -> None:
    """Require that a fixed backend profile declares a configuration.

    This pure catalogue check preserves the matrix's unsupported-declaration
    error. It does not accept or parse a native device, select an adapter,
    import an optional backend, or determine runtime availability.

    Args:
        backend: Catalogue backend profile to query.
        configuration: Semantic condensation configuration to require.

    Raises:
        TypeError: If either argument is not its declared metadata type.
        ValueError: If the configuration is not declared by the profile.
    """
    device = _condensation_profile_device(backend)
    requirements = get_condensation_requirements(configuration)
    CONDENSATION_CAPABILITY_MATRIX.require(
        device,
        CONDENSATION_PROCESS,
        requirements,
    )


def _validate_request(
    device: object,
    process: object,
    requirements: object,
) -> None:
    """Validate matrix request arguments in their required order.

    Args:
        device: Candidate device metadata.
        process: Candidate process metadata.
        requirements: Candidate capability requirements metadata.

    Raises:
        TypeError: If an argument has the wrong type, checked in argument order.
    """
    if not isinstance(device, Device):
        raise TypeError("device must be a Device.")
    if not isinstance(process, Process):
        raise TypeError("process must be a Process.")
    if not isinstance(requirements, CapabilityRequirements):
        raise TypeError("requirements must be a CapabilityRequirements.")


@dataclass(frozen=True)
class ExecutionRequest:
    """Declare a typed request for adapter selection without execution.

    This value only describes selection. It neither probes a backend, transfers
    state, nor establishes execution state or result contracts.

    Args:
        backend: Backend requested for selection.
        device: Device requested on the backend.
        process: Process requested on the device.
        requirements: Exact capability requirements for the process.

    Raises:
        TypeError: If a field is not its declared metadata type.
        ValueError: If the backend does not match the device backend.
    """

    backend: Backend
    device: Device
    process: Process
    requirements: CapabilityRequirements

    def __post_init__(self) -> None:
        """Validate request metadata without resolving or executing it.

        Raises:
            TypeError: If a field is not its declared metadata type.
            ValueError: If backend and device backend differ.
        """
        if not isinstance(self.backend, Backend):
            raise TypeError("ExecutionRequest.backend must be a Backend.")
        if not isinstance(self.device, Device):
            raise TypeError("ExecutionRequest.device must be a Device.")
        if not isinstance(self.process, Process):
            raise TypeError("ExecutionRequest.process must be a Process.")
        if not isinstance(self.requirements, CapabilityRequirements):
            raise TypeError(
                "ExecutionRequest.requirements must be a "
                "CapabilityRequirements."
            )
        if self.backend != self.device.backend:
            raise ValueError(
                "ExecutionRequest.backend must match device.backend."
            )


@runtime_checkable
class ExecutionState(Protocol):
    """Describe retained caller-owned state with an opaque backend payload.

    The payload is not inspected by this module. Boundary validation requires
    the same state object supplied by the caller and retains it by identity.
    """

    @property
    def backend_payload(self) -> object:
        """Return opaque backend-owned payload state."""


@runtime_checkable
class ExecutionAdapter(Protocol):
    """Declare the structural registration and selection seam.

    Registration validates that an adapter exposes a callable ``execute``
    attribute without invoking or probing it. Selection returns the registered
    adapter by identity and does not establish an execution, state, or result
    contract.
    """

    @property
    def execute(self) -> Callable[..., object]:
        """Return the callable execution seam without fixing its signature."""


class MutationScope(str, Enum):
    """Declare the one allowed mutation scope for an execution result.

    ``STATE`` permits in-place mutation of retained state. P3 does not infer or
    verify that a declared mutation occurred.
    """

    NONE = "none"
    STATE = "state"


@dataclass(frozen=True)
class MutationDeclaration:
    """Declare exactly one explicit state-mutation permission.

    Args:
        scopes: Exact frozenset containing either ``NONE`` or ``STATE``.

    Raises:
        TypeError: If scopes is not an exact frozenset of MutationScope values.
        ValueError: If scopes is empty or combines mutation scopes.
    """

    scopes: frozenset[MutationScope]

    def __post_init__(self) -> None:
        """Validate the closed, immutable mutation declaration."""
        _validate_mutation_scopes(self.scopes)


def _validate_mutation_scopes(value: object) -> None:
    """Validate one exact immutable mutation scope declaration.

    This helper is also called at the result boundary so declarations fabricated
    by bypassing frozen dataclass construction cannot bypass the P3 contract.

    Args:
        value: Candidate exact frozenset of mutation scopes.

    Raises:
        TypeError: If the collection or one of its members has an invalid type.
        ValueError: If the declaration is empty or contains multiple scopes.
    """
    if type(value) is not frozenset:
        raise TypeError("MutationDeclaration.scopes must be a frozenset.")
    if not all(isinstance(scope, MutationScope) for scope in value):
        raise TypeError(
            "MutationDeclaration.scopes must contain only MutationScope "
            "instances."
        )
    if value not in (
        frozenset({MutationScope.NONE}),
        frozenset({MutationScope.STATE}),
    ):
        raise ValueError(
            "MutationDeclaration.scopes must contain exactly one mutation "
            "scope."
        )


def _is_static_execute_callable(adapter: object) -> bool:
    """Return whether an adapter statically exposes a callable execute seam.

    Args:
        adapter: Candidate adapter to inspect without dynamic attribute access.

    Returns:
        True when a statically discoverable execute member is callable.
    """
    execute = inspect.getattr_static(adapter, "execute", None)
    if isinstance(execute, classmethod):
        return callable(execute.__func__)
    return callable(execute)


@dataclass(frozen=True, eq=False)
class BackendResult:
    """Retain opaque backend result values by identity without inspection.

    Args:
        value: Opaque backend-owned result value.
        diagnostics: Optional opaque backend-owned diagnostic value.
    """

    value: object
    diagnostics: object | None = None


@dataclass(frozen=True, eq=False)
class ExecutionResult:
    """Declare immutable P3 result ownership and mutation metadata.

    ``validate_execution_result`` verifies this carrier at the execution
    boundary, including that ``state`` is the original caller-owned object.

    Args:
        state: Execution state expected to be the original caller-owned object.
        metadata: Ordered immutable key-value result metadata for validation.
        mutation: Explicit permission for in-place state mutation.
        backend_result: Optional opaque backend result retained by identity.
    """

    state: ExecutionState
    metadata: tuple[tuple[str, str], ...]
    mutation: MutationDeclaration
    backend_result: BackendResult | None = None


def _validate_execution_state(value: object, field_name: str) -> None:
    """Validate state structure without reading its opaque payload.

    Args:
        value: Candidate state object.
        field_name: Qualified name used in the error message.

    Raises:
        TypeError: If value does not structurally satisfy ExecutionState.
    """
    if not isinstance(value, ExecutionState):
        raise TypeError(f"{field_name} must be an ExecutionState.")


def _validate_execution_metadata(value: object) -> None:
    """Validate ordered immutable execution metadata without copying it.

    Args:
        value: Candidate metadata collection.

    Raises:
        TypeError: If collection, entries, keys, or values have invalid types.
        ValueError: If a key is invalid or appears more than once.
    """
    if type(value) is not tuple:
        raise TypeError("ExecutionResult.metadata must be a tuple.")
    names: set[str] = set()
    for entry in value:
        if type(entry) is not tuple or len(entry) != 2:
            raise TypeError(
                "ExecutionResult.metadata entries must be (str, str) tuples."
            )
        key, item = entry
        if type(key) is not str or type(item) is not str:
            raise TypeError(
                "ExecutionResult.metadata entries must be (str, str) tuples."
            )
        _validate_name(key, "ExecutionResult.metadata key")
        if key in names:
            raise ValueError("ExecutionResult.metadata keys must be unique.")
        names.add(key)


def validate_execution_result(
    original_state: object,
    result: object,
) -> ExecutionResult:
    """Validate and return an immutable P3 result without executing an adapter.

    Args:
        original_state: Caller-owned state supplied before execution.
        result: Candidate result that retains the original state by identity.

    Returns:
        The exact validated result object.

    Raises:
        TypeError: If state, result layout, metadata, or declarations are
            invalid.
        ValueError: If state was replaced or metadata is otherwise invalid.
    """
    _validate_execution_state(original_state, "original_state")
    if type(result) is not ExecutionResult:
        raise TypeError("result must be an ExecutionResult.")
    if result.state is not original_state:
        raise ValueError("ExecutionResult.state must be original_state.")
    _validate_execution_metadata(result.metadata)
    if not isinstance(result.mutation, MutationDeclaration):
        raise TypeError(
            "ExecutionResult.mutation must be a MutationDeclaration."
        )
    _validate_mutation_scopes(result.mutation.scopes)
    if result.backend_result is not None and not isinstance(
        result.backend_result, BackendResult
    ):
        raise TypeError(
            "ExecutionResult.backend_result must be a BackendResult."
        )
    return result


class _CPURunnable(Protocol):
    """Describe the CPU runnable seam without importing runtime physics.

    Implementations own aerosol and process validation. This structural
    protocol permits the direct CPU boundary to retain a runnable without a
    runtime dependency on the runnable implementation hierarchy.
    """

    def execute(
        self,
        aerosol: "Aerosol",
        time_step: float,
        sub_steps: int = 1,
    ) -> "Aerosol":
        """Execute one CPU process over the supplied controls.

        Args:
            aerosol: Caller-owned aerosol to mutate and return by identity.
            time_step: Total process duration forwarded without coercion.
            sub_steps: Positive process substep count forwarded without
                coercion.

        Returns:
            The original aerosol object after process execution.
        """


@dataclass(frozen=True, eq=False)
class CPUExecutionState:
    """Retain CPU execution inputs for the direct CPU adapter.

    Construction is intentionally side-effect-free: control validation occurs
    at ``CPUExecutionAdapter.execute`` so invalid states can reach that
    boundary. The carrier neither copies nor validates the aerosol or controls.

    Args:
        aerosol: Caller-owned aerosol supplied to the CPU runnable.
        time_step: Caller-supplied total execution duration.
        sub_steps: Caller-supplied number of runnable substeps.
    """

    aerosol: object
    time_step: object
    sub_steps: object

    @property
    def backend_payload(self) -> object:
        """Return the caller-owned aerosol as the opaque CPU payload."""
        return self.aerosol


class CPUExecutionAdapter:
    """Dispatch one validated CPU state to one caller-supplied runnable.

    The adapter accepts only an exact ``CPUExecutionState``, locally validates
    finite nonnegative duration and a positive integral substep count, then
    delegates once using those state-held objects. It requires the runnable to
    return the original aerosol and reports ``MutationScope.STATE``. This
    direct-module-only boundary neither selects adapters nor transfers,
    converts, or otherwise validates process state. The stored runnable owns
    aerosol and process-physics validation.

    Args:
        runnable: Runnable invoked once for every valid execution state.
    """

    def __init__(self, runnable: "_CPURunnable | RunnableABC") -> None:
        """Retain the runnable by identity without inspecting it.

        Args:
            runnable: CPU runnable invoked once by :meth:`execute` for each
                valid state.
        """
        self._runnable = runnable

    def execute(self, state: ExecutionState) -> ExecutionResult:
        """Dispatch one concrete CPU state and retain its identity.

        This method performs local state and control preflight before one
        positional runnable call. It does not split time steps, catch runnable
        exceptions, inspect aerosol physics, select adapters, or use a
        conversion or fallback path.

        Args:
            state: Exact ``CPUExecutionState`` containing the aerosol and
                execution controls.

        Returns:
            An ``ExecutionResult`` retaining the original state and aerosol by
            identity, with ``MutationScope.STATE`` and no metadata.

        Raises:
            TypeError: If state is not an exact CPU state or time_step is not a
                non-boolean real scalar.
            ValueError: If controls are invalid or the runnable returns a
                different aerosol object.
        """
        if type(state) is not CPUExecutionState:
            raise TypeError("state must be a CPUExecutionState.")
        if isinstance(state.time_step, bool) or not isinstance(
            state.time_step, Real
        ):
            raise TypeError("time_step must be a real scalar.")
        if not _isfinite_real(state.time_step) or state.time_step < 0:
            raise ValueError("time_step must be finite and nonnegative.")
        if (
            isinstance(state.sub_steps, bool)
            or not isinstance(state.sub_steps, Integral)
            or state.sub_steps <= 0
        ):
            raise ValueError("sub_steps must be a positive integer.")

        returned_aerosol = self._runnable.execute(
            cast("Aerosol", state.aerosol),
            cast(float, state.time_step),
            cast(int, state.sub_steps),
        )
        if returned_aerosol is not state.aerosol:
            raise ValueError("CPU runnable must return the original aerosol.")
        return ExecutionResult(
            state,
            (),
            MutationDeclaration(frozenset({MutationScope.STATE})),
            BackendResult(returned_aerosol),
        )


class _AdapterRegistry:
    """Keep exact context-local adapter registrations without execution.

    The registry validates only an adapter's callable ``execute`` shape. It
    neither invokes adapters nor loads, probes, or transfers backend state.
    """

    def __init__(self) -> None:
        """Create an empty registry without loading or probing backends."""
        self._adapters: dict[tuple[Process, Backend], ExecutionAdapter] = {}

    def _register_adapter(
        self,
        process: object,
        backend: object,
        adapter: object,
    ) -> None:
        """Register one shaped adapter without invoking or inspecting it.

        Args:
            process: Process to associate with the adapter.
            backend: Backend to associate with the adapter.
            adapter: Object with a callable ``execute`` attribute.

        Raises:
            TypeError: If arguments are invalid in process, backend, adapter
                order.
            ValueError: If the process/backend pair is already registered.
        """
        if not isinstance(process, Process):
            raise TypeError("process must be a Process.")
        if not isinstance(backend, Backend):
            raise TypeError("backend must be a Backend.")
        if not _is_static_execute_callable(adapter):
            raise TypeError("adapter must have a callable execute attribute.")
        typed_adapter = cast(ExecutionAdapter, adapter)
        key = (process, backend)
        if key in self._adapters:
            raise ValueError(
                "Adapter already registered for process and backend."
            )
        self._adapters[key] = typed_adapter

    def _lookup(self, process: Process, backend: Backend) -> ExecutionAdapter:
        """Return one exact adapter without fallback or adapter execution.

        Args:
            process: Process associated with the requested adapter.
            backend: Backend associated with the requested adapter.

        Returns:
            The adapter registered under the exact process/backend pair.

        Raises:
            LookupError: If no adapter has the exact process/backend key.
        """
        try:
            return self._adapters[(process, backend)]
        except KeyError as error:
            raise LookupError(
                "No adapter registered for process and backend."
            ) from error

    def _snapshot(self) -> dict[tuple[Process, Backend], ExecutionAdapter]:
        """Return a fresh private-registration snapshot for focused tests.

        Returns:
            A copy of the registry mapping that cannot mutate registrations.
        """
        return dict(self._adapters)


class ExecutionContext:
    """Register and select declared adapters without executing or transferring.

    Each context owns its private registry, preventing mutable module-global
    registration state. Registration and selection are context-local only:
    neither invokes an adapter, probes availability, loads a backend, nor
    transfers data.
    """

    def __init__(self, matrix: CapabilityMatrix) -> None:
        """Store a matrix and create an empty private adapter registry.

        Args:
            matrix: Immutable capability declarations used for selection.

        Raises:
            TypeError: If ``matrix`` is not a CapabilityMatrix.
        """
        if not isinstance(matrix, CapabilityMatrix):
            raise TypeError("matrix must be a CapabilityMatrix.")
        self._matrix = matrix
        self._registry = _AdapterRegistry()

    def register_adapter(
        self,
        process: Process,
        backend: Backend,
        adapter: ExecutionAdapter,
    ) -> None:
        """Register one context-local adapter for selection only.

        Registration never invokes or probes the adapter, transfers data, or
        loads an optional backend. The adapter is available only from this
        context and is selected later by its exact process/backend pair.

        Args:
            process: Process to associate with the adapter.
            backend: Backend to associate with the adapter.
            adapter: Structurally callable adapter to register.

        Raises:
            TypeError: If arguments are invalid in process, backend, adapter
                order.
            ValueError: If the process/backend pair is already registered.
        """
        self._registry._register_adapter(process, backend, adapter)

    def resolve(self, request: ExecutionRequest) -> ExecutionAdapter:
        """Return one exact adapter after validation without invoking it.

        Capability support is checked before the single registry lookup. This
        method does not catch adapter errors because it never executes adapters.

        Args:
            request: Typed selection request.

        Returns:
            The registered adapter object by identity.

        Raises:
            TypeError: If request is not an ExecutionRequest.
            ValueError: If CPU device spelling is invalid or support is absent.
            LookupError: If supported request has no exact registered adapter.
        """
        if not isinstance(request, ExecutionRequest):
            raise TypeError("request must be an ExecutionRequest.")
        device = _normalize_request_device(request)
        self._matrix.require(device, request.process, request.requirements)
        return self._registry._lookup(request.process, request.backend)


def _normalize_request_device(request: ExecutionRequest) -> Device:
    """Normalize CPU selection without parsing, probing, or transferring.

    CPU requests must use the canonical ``Device(Backend.CPU, "cpu")``
    spelling. Warp native identifiers remain opaque and are returned verbatim.

    Args:
        request: Typed selection request whose device requires normalization.

    Returns:
        Canonical CPU device metadata or the request's unchanged Warp device.

    Raises:
        ValueError: If the CPU spelling or backend/device pairing is invalid.
    """
    if request.backend != request.device.backend:
        raise ValueError("ExecutionRequest.backend must match device.backend.")
    if request.backend is Backend.CPU:
        if request.device.native != "cpu":
            raise ValueError(
                "CPU execution requires Device(Backend.CPU, 'cpu')."
            )
        return Device(Backend.CPU, "cpu")
    if request.backend is Backend.WARP:
        return request.device
    raise ValueError("ExecutionRequest.backend must match device.backend.")


from particula.execution.errors import (  # noqa: E402
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
from particula.execution.values import (  # noqa: E402
    CPUStateAuthority,
    FallbackBoundary,
    FallbackPolicy,
)
