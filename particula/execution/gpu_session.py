"""Define concrete-only GPU-resident session carriers and setup.

The immutable carriers retain caller-owned Warp containers and CPU gas-name
metadata. ``setup_resident_session`` snapshots validated ordered CPU gas names
into immutable tuple metadata and is P2's sole CPU-to-Warp upload point: local
CPU preflight precedes conversion imports, then particles, gas, and environment
convert once in that order before one complete ACTIVE session is published. Gas
names are not stored on the Warp gas container. This module provides no public
export, lifecycle operation, fallback, synchronization, restore, or process
sidecar. Its concrete-only P4 ``ResidentStepGuard`` and identity-only
``ResidentStepToken`` track one open timestep for one pinned active
session/registry binding. They execute no adapters, transfer no data,
synchronize no Warp work, allocate no resources, and provide no fallback.
Future checkpoint, restore, finalize, close, fault, conversion, and
resize/rebind boundaries must call the guard's ``assert_step_closed`` gate
before their own work; raw low-level helpers remain outside that gate's
interception.
Selected native Warp-device availability remains an upstream E7-F6
precondition.
"""

from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, cast

from particula.execution import Backend, Device, _isfinite_real

if TYPE_CHECKING:
    from particula.execution.gpu_resources import GPUResourceRegistry
    from particula.gas import EnvironmentData, GasData
    from particula.particles import ParticleData


def _validate_dimension(value: object, name: str, *, positive: bool) -> None:
    """Validate one immutable resident dimension.

    Args:
        value: Candidate dimension value.
        name: Dimension name used in validation errors.
        positive: Whether the value must be greater than zero instead of
            nonnegative.

    Raises:
        TypeError: If ``value`` is not an integral value or is a boolean.
        ValueError: If ``value`` violates its required lower bound.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integral, not bool.")
    if positive and value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    if not positive and value < 0:
        raise ValueError(f"{name} must be nonnegative.")


def _validate_gas_names(gas_names: object) -> None:
    """Validate an exact tuple of exact string gas names.

    Args:
        gas_names: Candidate ordered gas-name metadata.

    Raises:
        TypeError: If the metadata is not an exact tuple of exact strings.
    """
    if type(gas_names) is not tuple:
        raise TypeError("gas_names must be an exact tuple.")
    if not all(type(name) is str for name in gas_names):
        raise TypeError("gas_names entries must be exact str instances.")


def _validate_cpu_shape(
    value: object,
    name: str,
    expected_shape: tuple[int, ...],
) -> None:
    """Validate one CPU field shape without reading or copying its values."""
    if getattr(value, "shape", None) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}.")


def _preflight_cpu_session(
    particles: object,
    gas: object,
    environment: object,
    device: object,
) -> tuple[
    "ParticleData",
    "GasData",
    "EnvironmentData",
    "ResidentDimensions",
    tuple[str, ...],
]:
    """Validate local CPU session inputs before importing conversion helpers.

    This function intentionally does not probe native-device availability. That
    policy is an upstream E7-F6 responsibility and the selected device is
    assumed to have been capability-approved before this boundary.
    """
    if type(device) is not Device:
        raise TypeError("device must be an exact Device.")
    if device.backend is not Backend.WARP:
        raise ValueError("device.backend must be Backend.WARP.")

    from particula.gas import EnvironmentData, GasData
    from particula.particles import ParticleData

    if not isinstance(particles, ParticleData):
        raise TypeError("particles must be a ParticleData.")
    if not isinstance(gas, GasData):
        raise TypeError("gas must be a GasData.")
    if not isinstance(environment, EnvironmentData):
        raise TypeError("environment must be an EnvironmentData.")

    masses_shape = getattr(particles.masses, "shape", None)
    if not isinstance(masses_shape, tuple) or len(masses_shape) != 3:
        raise ValueError("particles.masses must have rank 3.")
    dimensions = ResidentDimensions(*masses_shape)
    boxes = dimensions.n_boxes
    particle_count = dimensions.n_particles
    species = dimensions.n_species

    _validate_cpu_shape(
        particles.concentration,
        "particles.concentration",
        (boxes, particle_count),
    )
    _validate_cpu_shape(
        particles.charge,
        "particles.charge",
        (boxes, particle_count),
    )
    _validate_cpu_shape(particles.density, "particles.density", (species,))
    _validate_cpu_shape(particles.volume, "particles.volume", (boxes,))
    _validate_cpu_shape(gas.molar_mass, "gas.molar_mass", (species,))
    _validate_cpu_shape(gas.partitioning, "gas.partitioning", (species,))
    _validate_cpu_shape(
        gas.concentration,
        "gas.concentration",
        (boxes, species),
    )
    if isinstance(gas.name, str):
        raise TypeError("gas.name must be an ordered collection of strings.")
    if len(gas.name) != species:
        raise ValueError("gas.name length must match n_species.")
    _validate_cpu_shape(
        environment.temperature,
        "environment.temperature",
        (boxes,),
    )
    _validate_cpu_shape(
        environment.pressure,
        "environment.pressure",
        (boxes,),
    )
    _validate_cpu_shape(
        environment.saturation_ratio,
        "environment.saturation_ratio",
        (boxes, species),
    )
    gas_names = tuple(gas.name)
    _validate_gas_names(gas_names)
    return particles, gas, environment, dimensions, gas_names


@dataclass(frozen=True)
class ResidentDimensions:
    """Declare immutable dimensions for caller-owned resident state.

    Attributes:
        n_boxes: Positive number of resident boxes.
        n_particles: Nonnegative fixed particle capacity per box.
        n_species: Nonnegative number of gas species.
    """

    n_boxes: int
    n_particles: int
    n_species: int

    def __post_init__(self) -> None:
        """Validate the immutable resident dimensions.

        Raises:
            TypeError: If a dimension is not an integral value or is boolean.
            ValueError: If a dimension violates its required lower bound.
        """
        _validate_dimension(self.n_boxes, "n_boxes", positive=True)
        _validate_dimension(self.n_particles, "n_particles", positive=False)
        _validate_dimension(self.n_species, "n_species", positive=False)


class ResidentLifecycle(Enum):
    """Declare P1 immutable lifecycle vocabulary without transition logic.

    This vocabulary records a supplied state only. Lifecycle transitions,
    recovery, finalization, and close operations are outside this module.
    """

    ACTIVE = "active"
    FAULTED = "faulted"
    FINALIZED = "finalized"
    CLOSED = "closed"


@dataclass(frozen=True)
class ResidentMetadata:
    """Retain declared Warp-device and ordered gas-name metadata by identity.

    Attributes:
        device: Exact Warp-backed execution device declaration.
        gas_names: Ordered caller-declared gas names; entries are not inspected
            or normalized at this boundary.
    """

    device: Device
    gas_names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate fixed-cost metadata declarations without normalizing names.

        Raises:
            TypeError: If the device or name tuple has the wrong type.
            ValueError: If the device does not declare the Warp backend.
        """
        if type(self.device) is not Device:
            raise TypeError("device must be an exact Device.")
        if self.device.backend is not Backend.WARP:
            raise ValueError("device.backend must be Backend.WARP.")
        _validate_gas_names(self.gas_names)


def _validate_array(
    name: str,
    array: Any,
    dtype: Any,
    shape: tuple[int, ...],
    device: Any,
) -> None:
    """Validate only one Warp primary-array metadata declaration.

    Args:
        name: Fully qualified primary-array name for validation errors.
        array: Candidate Warp array whose metadata is inspected.
        dtype: Required Warp dtype.
        shape: Required immutable array shape.
        device: Required shared Warp device.

    Raises:
        ValueError: If the array metadata is absent or differs from the schema.
    """
    if not all(
        hasattr(array, attribute) for attribute in ("dtype", "shape", "device")
    ):
        raise ValueError(f"{name} must be a Warp array.")
    if array.dtype != dtype:
        raise ValueError(f"{name} must use dtype {dtype}.")
    if type(array.shape) is not tuple:
        raise ValueError(f"{name} must provide a tuple shape.")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if array.device != device:
        raise ValueError(f"{name} device must match particles.masses device.")
    if not (
        type(array).__module__.startswith("warp")
        and type(array).__name__ == "array"
    ):
        raise ValueError(f"{name} must be a Warp array.")
    _validate_contiguous_array(
        name,
        array,
        shape,
        8 if getattr(dtype, "__name__", None) == "float64" else 4,
    )


def _validate_contiguous_array(
    name: str,
    array: Any,
    shape: tuple[int, ...],
    item_size: int,
) -> None:
    """Validate pointer and contiguous-stride primary metadata."""
    expected: list[int] = []
    stride = item_size
    for length in reversed(shape):
        expected.insert(0, stride)
        stride *= length
    if getattr(array, "strides", None) != tuple(expected):
        raise ValueError(f"{name} must be contiguous.")
    if all(shape) and (
        not isinstance(getattr(array, "ptr", None), Integral) or array.ptr < 0
    ):
        raise ValueError(f"{name} must have a valid pointer.")


def _validate_generated_containers(
    particles: object,
    gas: object,
    environment: object,
) -> None:
    """Validate generated Warp structs without reading their payloads.

    Args:
        particles: Candidate generated particle container.
        gas: Candidate generated gas container.
        environment: Candidate generated environment container.

    Raises:
        TypeError: If a container is not its required generated Warp struct.
    """
    from particula.gpu.warp_types import (
        WarpEnvironmentData,
        WarpGasData,
        WarpParticleData,
    )

    particle_type = cast(Any, WarpParticleData).cls
    gas_type = cast(Any, WarpGasData).cls
    environment_type = cast(Any, WarpEnvironmentData).cls
    if not isinstance(particles, particle_type):
        raise TypeError("particles must be a WarpParticleData.")
    if not isinstance(gas, gas_type):
        raise TypeError("gas must be a WarpGasData.")
    if not isinstance(environment, environment_type):
        raise TypeError("environment must be a WarpEnvironmentData.")


def _validate_schema(
    particles: object,
    gas: object,
    environment: object,
    dimensions: ResidentDimensions,
    metadata: ResidentMetadata,
    wp: Any,
) -> None:
    """Validate fixed primary schemas and shared Warp-device metadata.

    Args:
        particles: Validated generated particle container.
        gas: Validated generated gas container.
        environment: Validated generated environment container.
        dimensions: Declared resident dimensions to match the particle schema.
        metadata: Declared device metadata to match the shared Warp device.
        wp: Lazily imported Warp module.

    Raises:
        ValueError: If a primary array has incompatible metadata or device.
    """
    masses = cast(Any, particles).masses
    if not all(
        hasattr(masses, attribute) for attribute in ("dtype", "shape", "device")
    ):
        raise ValueError("particles.masses must be a Warp array.")
    if masses.dtype != wp.float64:
        raise ValueError(f"particles.masses must use dtype {wp.float64}.")
    if type(masses.shape) is not tuple:
        raise ValueError("particles.masses must provide a tuple shape.")
    if len(masses.shape) != 3:
        raise ValueError("particles.masses must have rank 3.")
    dimensions_from_masses = ResidentDimensions(*masses.shape)
    if dimensions_from_masses != dimensions:
        raise ValueError(
            "particles.masses shape must match ResidentDimensions."
        )
    device = masses.device
    boxes = dimensions_from_masses.n_boxes
    particle_count = dimensions_from_masses.n_particles
    species = dimensions_from_masses.n_species
    _validate_array(
        "particles.masses",
        masses,
        wp.float64,
        (boxes, particle_count, species),
        device,
    )
    _validate_array(
        "particles.concentration",
        cast(Any, particles).concentration,
        wp.float64,
        (boxes, particle_count),
        device,
    )
    _validate_array(
        "particles.charge",
        cast(Any, particles).charge,
        wp.float64,
        (boxes, particle_count),
        device,
    )
    _validate_array(
        "particles.density",
        cast(Any, particles).density,
        wp.float64,
        (species,),
        device,
    )
    _validate_array(
        "particles.volume",
        cast(Any, particles).volume,
        wp.float64,
        (boxes,),
        device,
    )
    _validate_array(
        "gas.molar_mass",
        cast(Any, gas).molar_mass,
        wp.float64,
        (species,),
        device,
    )
    _validate_array(
        "gas.concentration",
        cast(Any, gas).concentration,
        wp.float64,
        (boxes, species),
        device,
    )
    _validate_array(
        "gas.vapor_pressure",
        cast(Any, gas).vapor_pressure,
        wp.float64,
        (boxes, species),
        device,
    )
    _validate_array(
        "gas.partitioning",
        cast(Any, gas).partitioning,
        wp.int32,
        (boxes, species),
        device,
    )
    _validate_array(
        "environment.temperature",
        cast(Any, environment).temperature,
        wp.float64,
        (boxes,),
        device,
    )
    _validate_array(
        "environment.pressure",
        cast(Any, environment).pressure,
        wp.float64,
        (boxes,),
        device,
    )
    _validate_array(
        "environment.saturation_ratio",
        cast(Any, environment).saturation_ratio,
        wp.float64,
        (boxes, species),
        device,
    )
    if str(device) != metadata.device.native:
        raise ValueError(
            "particles.masses device must match metadata.device.native."
        )


def _validate_resident_carriers(
    dimensions: object,
    metadata: object,
    lifecycle: object,
) -> tuple[ResidentDimensions, ResidentMetadata, ResidentLifecycle]:
    """Validate exact resident carrier declarations.

    Args:
        dimensions: Declared resident dimensions.
        metadata: Declared resident metadata.
        lifecycle: Declared lifecycle value.

    Returns:
        The validated resident carriers.
    """
    if type(dimensions) is not ResidentDimensions:
        raise TypeError("dimensions must be an exact ResidentDimensions.")
    dimensions.__post_init__()
    if type(metadata) is not ResidentMetadata:
        raise TypeError("metadata must be an exact ResidentMetadata.")
    if type(metadata.device) is not Device:
        raise TypeError("device must be an exact Device.")
    if metadata.device.backend is not Backend.WARP:
        raise ValueError("device.backend must be Backend.WARP.")
    _validate_gas_names(metadata.gas_names)
    if len(metadata.gas_names) != dimensions.n_species:
        raise ValueError("metadata.gas_names length must match n_species.")
    if type(lifecycle) is not ResidentLifecycle:
        raise TypeError("lifecycle must be an exact ResidentLifecycle.")
    return (
        cast(ResidentDimensions, dimensions),
        cast(ResidentMetadata, metadata),
        cast(ResidentLifecycle, lifecycle),
    )


def _validate_resident_imports() -> Any:
    """Import Warp for resident checks and preserve missing-runtime errors."""
    try:
        import warp as wp
    except ModuleNotFoundError as error:
        if error.name != "warp":
            raise
        raise RuntimeError(
            "ResidentSession requires the optional Warp runtime."
        ) from error
    return wp


@dataclass(frozen=True, eq=False)
class ResidentSession:
    """Retain one validated caller-owned GPU-resident session by identity.

    Construction performs only O(1) schema and device metadata validation. It
    neither reads resident payloads nor transfers, allocates, synchronizes,
    schedules, or operates on the retained lifecycle state.

    Attributes:
        particles: Caller-owned generated Warp particle container.
        gas: Caller-owned generated Warp gas container.
        environment: Caller-owned generated Warp environment container.
        dimensions: Immutable dimensions that describe the retained containers.
        metadata: Immutable Warp-device and ordered gas-name metadata.
        lifecycle: Immutable supplied lifecycle vocabulary value.
    """

    particles: object
    gas: object
    environment: object
    dimensions: ResidentDimensions
    metadata: ResidentMetadata
    lifecycle: ResidentLifecycle

    def __post_init__(self) -> None:
        """Perform ordered O(1) metadata-only resident-session preflight.

        Raises:
            TypeError: If carrier, lifecycle, or generated container types are
                invalid.
            ValueError: If declared metadata, schema, or device values disagree.
            RuntimeError: If resident validation requires an unavailable Warp
                runtime.
        """
        (
            dimensions,
            metadata,
            lifecycle,
        ) = _validate_resident_carriers(
            self.dimensions, self.metadata, self.lifecycle
        )
        if self.particles is self.gas:
            raise ValueError("particles and gas must not be identical.")
        if self.particles is self.environment:
            raise ValueError("particles and environment must not be identical.")
        if self.gas is self.environment:
            raise ValueError("gas and environment must not be identical.")
        wp = _validate_resident_imports()
        _validate_generated_containers(
            self.particles, self.gas, self.environment
        )
        _validate_schema(
            self.particles,
            self.gas,
            self.environment,
            dimensions,
            metadata,
            wp,
        )


@dataclass(frozen=True, eq=False)
class ResidentStepToken:
    """Retain one guard-owned timestep identity and validated duration.

    A token is opaque to scheduler callers. ``complete_step`` accepts only the
    exact outstanding token by identity. Its private origin and duration fields
    are guard bookkeeping, not a process-execution interface.
    This concrete-only token does not execute work, transfer data, synchronize,
    allocate resources, or provide a fallback.

    Attributes:
        _guard: Guard that created this token.
        _duration: Validated nonnegative finite duration retained without
            coercion.
    """

    _guard: object
    _duration: Any


class ResidentStepGuard:
    """Track one open timestep for one pinned active resident session.

    The guard retains one exact :class:`ResidentSession` and
    :class:`GPUResourceRegistry` binding by identity. A successful
    :meth:`begin_step` creates the sole open token; only successful matching
    :meth:`complete_step` advances completed-step and simulated-time metadata.

    This concrete-only scheduler bookkeeping does not execute adapters,
    transfer or restore data, synchronize Warp, acquire sidecars, resize
    resident state, evolve the environment, or provide a CPU fallback. Future
    checkpoint, finalize, close, conversion, resize, rebind, and fault
    boundaries must call :meth:`assert_step_closed` before their own work. The
    gate does not intercept callers that bypass those lifecycle boundaries with
    raw low-level helpers.
    """

    def __init__(
        self, session: ResidentSession, registry: "GPUResourceRegistry"
    ) -> None:
        """Create a closed guard for one exact active session-registry binding.

        Construction validates the pinned binding without acquiring sidecars or
        executing, transferring, synchronizing, or mutating resident state.

        Args:
            session: Exact active resident session retained by ``registry``.
            registry: Exact resource registry pinned to ``session``.

        Raises:
            TypeError: If either argument is not its exact concrete type.
            ValueError: If the registry does not retain ``session`` or its
                active metadata-only signature is invalid.
        """
        from particula.execution.gpu_resources import GPUResourceRegistry

        if type(session) is not ResidentSession:
            raise TypeError("session must be an exact ResidentSession.")
        if type(registry) is not GPUResourceRegistry:
            raise TypeError("registry must be an exact GPUResourceRegistry.")
        registry.validate_pinned_session(session)
        self._session = session
        self._registry = registry
        self._open_token: ResidentStepToken | None = None
        self._completed_steps = 0
        # ``numbers.Real`` lacks static numeric-tower support in mypy. Values
        # enter only through ``_validate_duration``, which enforces the runtime
        # Real invariant while preserving finite Rational values without casts.
        self._simulated_time: Any = 0

    @property
    def completed_steps(self) -> int:
        """Return the number of successfully completed guarded timesteps.

        Returns:
            Count advanced only after successful matching completion.
        """
        return self._completed_steps

    @property
    def simulated_time(self) -> Real:
        """Return the sum of successfully completed timestep durations.

        Returns:
            Uncoerced-real sum advanced only after successful matching
            completion.
        """
        return cast(Real, self._simulated_time)

    @staticmethod
    def _validate_duration(duration: object) -> Any:
        """Validate one nonnegative finite real duration without coercion.

        Args:
            duration: Candidate timestep duration.

        Returns:
            The validated duration unchanged.

        Raises:
            TypeError: If the duration is boolean or not a real number.
            ValueError: If the duration is negative or non-finite.
        """
        if isinstance(duration, bool) or not isinstance(duration, Real):
            raise TypeError("duration must be a non-boolean real.")
        if not _isfinite_real(duration) or duration < 0:
            raise ValueError("duration must be finite and nonnegative.")
        return duration

    def begin_step(self, duration: object) -> ResidentStepToken:
        """Open and return the sole token for a validated timestep duration.

        The guard validates duration before its pinned active session-registry
        binding. It performs bookkeeping only; callers execute any process work
        outside this method and must complete the returned exact token only
        after that work succeeds.

        Args:
            duration: Nonnegative finite real timestep duration.

        Returns:
            The new opaque token for the sole open timestep.

        Raises:
            TypeError: If ``duration`` is boolean or not a real number.
            ValueError: If ``duration`` is invalid or the pinned session binding
                is no longer valid and active.
            RuntimeError: If another timestep is already open.
        """
        validated_duration = self._validate_duration(duration)
        self._registry.validate_pinned_session(self._session)
        if self._open_token is not None:
            raise RuntimeError("A resident timestep is already open.")
        token = ResidentStepToken(self, validated_duration)
        self._open_token = token
        return token

    def complete_step(self, token: ResidentStepToken) -> None:
        """Complete the exact outstanding token and advance bookkeeping.

        Completed-step count and simulated time change only after the current
        pinned binding validates and ``token`` is the exact open token. This
        method executes no adapter and supplies no failure recovery or rollback
        for external process work.

        Args:
            token: Exact opaque token returned by this guard's open step.

        Raises:
            ValueError: If the pinned session binding is invalid or ``token``
                does not match the open token by identity.
            RuntimeError: If no timestep is open.
        """
        self._registry.validate_pinned_session(self._session)
        if self._open_token is None:
            raise RuntimeError("No resident timestep is open.")
        if token is not self._open_token:
            raise ValueError("token does not match the open resident timestep.")
        self._completed_steps += 1
        self._simulated_time += self._open_token._duration
        self._open_token = None

    def assert_step_closed(self) -> None:
        """Reject a lifecycle boundary while a guarded timestep remains open.

        Future checkpoint, finalize, close, conversion, resize, rebind, and
        fault boundaries must call this side-effect-free gate before their own
        transfer, synchronization, mutation, or state transition. It does not
        globally intercept direct raw helper calls.

        Raises:
            RuntimeError: If this guard has an outstanding timestep token.
        """
        if self._open_token is not None:
            raise RuntimeError("A resident timestep is open.")


def setup_resident_session(
    particles: "ParticleData",
    gas: "GasData",
    environment: "EnvironmentData",
    device: Device,
) -> ResidentSession:
    """Upload CPU carriers once and publish one active resident session.

    This concrete-only, unexported factory is P2's only CPU-to-Warp upload
    point. Local CPU preflight completes before conversion helpers are imported.
    It then converts particles, gas, and environment once in that order,
    snapshots ordered gas names into immutable CPU-owned tuple metadata, and
    publishes only one complete, validated ACTIVE session retaining converted
    containers by identity. It
    provides no lifecycle operations, fallback, synchronization, restore, or
    process sidecars. The selected native Warp device must already be
    availability-approved by upstream E7-F6; this revision does not probe it.

    Args:
        particles: CPU particle carrier to upload.
        gas: CPU gas carrier to upload; its ordered names remain CPU metadata.
        environment: CPU environment carrier to upload.
        device: Exact, upstream-approved Warp device declaration.

    Returns:
        One complete ACTIVE session retaining conversion results by identity.

    Raises:
        TypeError: If a local device, carrier, or gas-name declaration is
            invalid.
        ValueError: If the backend or cross-container CPU schema is invalid.
        RuntimeError: If optional Warp conversion or session validation requires
            an unavailable runtime.
    """
    (
        validated_particles,
        validated_gas,
        validated_environment,
        dimensions,
        gas_names,
    ) = _preflight_cpu_session(particles, gas, environment, device)

    from particula.gpu.conversion import (
        to_warp_environment_data,
        to_warp_gas_data,
        to_warp_particle_data,
    )

    warp_particles = to_warp_particle_data(
        validated_particles,
        device=device.native,
    )
    warp_gas = to_warp_gas_data(validated_gas, device=device.native)
    warp_environment = to_warp_environment_data(
        validated_environment,
        device=device.native,
    )
    metadata = ResidentMetadata(device, gas_names)
    return ResidentSession(
        warp_particles,
        warp_gas,
        warp_environment,
        dimensions,
        metadata,
        ResidentLifecycle.ACTIVE,
    )
