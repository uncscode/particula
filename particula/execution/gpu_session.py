"""Define concrete-only GPU-resident session ownership carriers.

These immutable carriers retain caller-owned Warp containers and CPU gas-name
metadata by identity. They perform construction-time metadata validation only;
they do not convert, allocate, schedule, synchronize, restore, fall back,
resize, migrate devices, operate on lifecycle state, or change package exports.
"""

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Any, cast

from particula.execution import Backend, Device


def _validate_dimension(value: object, name: str, *, positive: bool) -> None:
    """Validate one immutable resident dimension."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integral, not bool.")
    if positive and value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    if not positive and value < 0:
        raise ValueError(f"{name} must be nonnegative.")


@dataclass(frozen=True)
class ResidentDimensions:
    """Declare immutable dimensions for caller-owned resident state."""

    n_boxes: int
    n_particles: int
    n_species: int

    def __post_init__(self) -> None:
        """Validate resident dimensions."""
        _validate_dimension(self.n_boxes, "n_boxes", positive=True)
        _validate_dimension(self.n_particles, "n_particles", positive=False)
        _validate_dimension(self.n_species, "n_species", positive=False)


class ResidentLifecycle(Enum):
    """Declare P1 immutable lifecycle vocabulary without transition logic."""

    ACTIVE = "active"
    FAULTED = "faulted"
    FINALIZED = "finalized"
    CLOSED = "closed"


@dataclass(frozen=True)
class ResidentMetadata:
    """Retain declared Warp-device and ordered gas-name metadata by identity."""

    device: Device
    gas_names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate exact metadata contents without normalizing names."""
        if type(self.device) is not Device:
            raise TypeError("device must be an exact Device.")
        if self.device.backend is not Backend.WARP:
            raise ValueError("device.backend must be Backend.WARP.")
        if type(self.gas_names) is not tuple:
            raise TypeError("gas_names must be an exact tuple.")
        if any(type(name) is not str for name in self.gas_names):
            raise TypeError("gas_names must contain only exact str values.")


def _validate_array(
    name: str,
    array: Any,
    dtype: Any,
    shape: tuple[int, ...],
    device: Any,
) -> None:
    """Validate only one Warp primary-array metadata declaration."""
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
    """Validate generated Warp structs without reading their payloads."""
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
    """Validate fixed primary schemas and shared Warp-device metadata."""
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


@dataclass(frozen=True, eq=False)
class ResidentSession:
    """Retain one validated caller-owned GPU-resident session by identity."""

    particles: object
    gas: object
    environment: object
    dimensions: ResidentDimensions
    metadata: ResidentMetadata
    lifecycle: ResidentLifecycle

    def __post_init__(self) -> None:
        """Perform ordered O(1) metadata-only resident-session preflight."""
        if type(self.dimensions) is not ResidentDimensions:
            raise TypeError("dimensions must be an exact ResidentDimensions.")
        self.dimensions.__post_init__()
        if type(self.metadata) is not ResidentMetadata:
            raise TypeError("metadata must be an exact ResidentMetadata.")
        self.metadata.__post_init__()
        if len(self.metadata.gas_names) != self.dimensions.n_species:
            raise ValueError("metadata.gas_names length must match n_species.")
        if type(self.lifecycle) is not ResidentLifecycle:
            raise TypeError("lifecycle must be an exact ResidentLifecycle.")
        if self.particles is self.gas:
            raise ValueError("particles and gas must not be identical.")
        if self.particles is self.environment:
            raise ValueError("particles and environment must not be identical.")
        if self.gas is self.environment:
            raise ValueError("gas and environment must not be identical.")
        try:
            import warp as wp
        except ModuleNotFoundError as error:
            if error.name != "warp":
                raise
            raise RuntimeError(
                "ResidentSession requires the optional Warp runtime."
            ) from error
        _validate_generated_containers(
            self.particles, self.gas, self.environment
        )
        _validate_schema(
            self.particles,
            self.gas,
            self.environment,
            self.dimensions,
            self.metadata,
            wp,
        )
