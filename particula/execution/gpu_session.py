"""Define concrete-only GPU-resident session ownership carriers.

These immutable carriers retain caller-owned Warp containers and CPU gas-name
metadata by identity. They perform construction-time metadata validation only.
They do not convert, allocate, schedule, synchronize, restore, fall back,
resize, migrate devices, operate on lifecycle state, or change package exports.
"""

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Any, cast

from particula.execution import Backend, Device


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
