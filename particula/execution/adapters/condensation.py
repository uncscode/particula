"""Concrete, non-executing condensation state carriers.

These frozen carriers retain caller-owned resources by identity and perform
only read-only construction-time metadata checks.  They neither select nor run
an adapter, transfer, allocate, or synchronize resources.  Frozen fields only
prevent rebinding: retained resources remain mutable and caller-owned.  A
future adapter may mutate particle and gas fields and writable sidecars, so
callers retain lifetime, synchronization, and concurrency responsibility until
that future launch completes.
"""

from dataclasses import dataclass
from math import prod
from typing import Any, cast

from particula.aerosol import Aerosol
from particula.execution import CondensationConfiguration


@dataclass(frozen=True, eq=False)
class CondensationExecutionConfig:
    """Retain exact semantic condensation configuration by identity.

    Profile selection remains the responsibility of a future selection caller.

    Args:
        configuration: Exact concrete condensation configuration.
    """

    configuration: CondensationConfiguration

    def __post_init__(self) -> None:
        """Validate the exact configuration carrier type."""
        if type(self.configuration) is not CondensationConfiguration:
            raise TypeError(
                "configuration must be an exact CondensationConfiguration."
            )


@dataclass(frozen=True, eq=False)
class CPUCondensationState:
    """Retain caller-owned CPU condensation state without executing it.

    Args:
        config: Exact P2 condensation execution configuration.
        aerosol: Caller-owned aerosol retained by identity.
    """

    config: CondensationExecutionConfig
    aerosol: Aerosol

    def __post_init__(self) -> None:
        """Validate state inputs in configuration then aerosol order."""
        if type(self.config) is not CondensationExecutionConfig:
            raise TypeError(
                "config must be an exact CondensationExecutionConfig."
            )
        if not isinstance(self.aerosol, Aerosol):
            raise TypeError("aerosol must be an Aerosol.")

    @property
    def backend_payload(self) -> Aerosol:
        """Return the caller-owned aerosol without inspecting it."""
        return self.aerosol


def _validate_array(
    name: str,
    array: Any,
    dtype: Any,
    shape: tuple[int, ...],
    device: Any | None = None,
) -> None:
    """Validate Warp-array metadata only, without device operations."""
    attributes = ("dtype", "shape", "device")
    if not all(hasattr(array, attribute) for attribute in attributes):
        raise ValueError(f"{name} must be a Warp array.")
    if array.dtype != dtype:
        raise ValueError(f"{name} must use dtype {dtype}.")
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if device is not None and str(array.device) != str(device):
        raise ValueError(f"{name} device must match particles.masses device.")


def _memory_range(
    name: str,
    array: Any,
    itemsize: int,
    *,
    require_contiguous: bool = True,
) -> tuple[int, int] | None:
    """Return the metadata-only storage range, or None for an empty array."""
    strides = getattr(array, "strides", None)
    if not hasattr(array, "ptr"):
        raise ValueError(
            f"{name} must be a contiguous Warp array for ownership."
        )
    expected: list[int] = []
    stride = itemsize
    for dimension in reversed(array.shape):
        expected.insert(0, stride)
        stride *= dimension
    if require_contiguous and (
        strides is None or tuple(strides) != tuple(expected)
    ):
        raise ValueError(f"{name} must be contiguous for ownership checks.")
    if prod(array.shape) == 0:
        return None
    start = int(array.ptr)
    if strides is None:
        raise ValueError(f"{name} must provide strides for ownership checks.")
    offsets = tuple(
        (dimension - 1) * stride
        for dimension, stride in zip(array.shape, strides, strict=True)
    )
    return (
        start + sum(min(0, offset) for offset in offsets),
        start + sum(max(0, offset) for offset in offsets) + itemsize,
    )


def _validate_output_ownership(
    mass_transfer: Any | None,
    energy_transfer: Any | None,
    primary_fields: tuple[Any, ...],
) -> None:
    """Reject writable outputs overlapping primary fields or each other."""
    outputs = (
        ("mass_transfer", mass_transfer),
        ("energy_transfer", energy_transfer),
    )
    output_ranges = [
        (name, array, _memory_range(name, array, 8))
        for name, array in outputs
        if array is not None
    ]
    primary_ranges = [
        _memory_range(
            "primary state",
            array,
            _dtype_itemsize(array.dtype),
            require_contiguous=False,
        )
        for array in primary_fields
    ]
    for name, array, output_range in output_ranges:
        if output_range is None:
            continue
        for primary, primary_range in zip(
            primary_fields, primary_ranges, strict=True
        ):
            if array is primary or _overlaps(output_range, primary_range):
                raise ValueError(f"{name} must not overlap primary state.")
        for other_name, other_array, other_range in output_ranges:
            if name != other_name and (
                array is other_array or _overlaps(output_range, other_range)
            ):
                raise ValueError(f"{name} must not overlap {other_name}.")


def _dtype_itemsize(dtype: Any) -> int:
    """Return the fixed byte size of P2 primary Warp dtypes."""
    return 4 if str(dtype) == "int32" else 8


def _overlaps(first: tuple[int, int], second: tuple[int, int] | None) -> bool:
    """Return whether two nonempty contiguous byte ranges overlap."""
    return second is not None and first[0] < second[1] and second[0] < first[1]


@dataclass(frozen=True, eq=False)
class WarpCondensationState:
    """Retain validated caller-owned resident Warp condensation resources.

    P2 validates primary container metadata and writable output ownership only.
    Thermodynamics and all other sidecars are opaque except that thermodynamics
    must be non-None.  Rejection occurs before a writer launch and mutates
    nothing; it does not promise post-launch rollback for a future adapter.
    """

    config: CondensationExecutionConfig
    particles: object
    gas: object
    environment: object
    thermodynamics: object
    activity_surface: object | None = None
    scratch_buffers: object | None = None
    mass_transfer: object | None = None
    latent_heat: object | None = None
    energy_transfer: object | None = None
    thermal_work: object | None = None

    def __post_init__(self) -> None:
        """Perform ordered metadata and writable-output ownership validation."""
        if type(self.config) is not CondensationExecutionConfig:
            raise TypeError(
                "config must be an exact CondensationExecutionConfig."
            )

        try:
            import warp as wp

            from particula.gpu.warp_types import (
                WarpEnvironmentData,
                WarpGasData,
                WarpParticleData,
            )
        except ImportError as error:
            raise RuntimeError(
                "WarpCondensationState requires the optional Warp runtime."
            ) from error

        # ``@wp.struct`` exposes a Warp ``Struct`` descriptor.  Its callable
        # creates a generated instance subclass, so validate against the
        # descriptor's underlying Python class instead of exact type identity.
        particle_type = cast(Any, WarpParticleData).cls
        gas_type = cast(Any, WarpGasData).cls
        environment_type = cast(Any, WarpEnvironmentData).cls
        if not isinstance(self.particles, particle_type):
            raise TypeError("particles must be a WarpParticleData.")
        if not isinstance(self.gas, gas_type):
            raise TypeError("gas must be a WarpGasData.")
        if not isinstance(self.environment, environment_type):
            raise TypeError("environment must be a WarpEnvironmentData.")

        masses = self.particles.masses
        _validate_array(
            "particles.masses",
            masses,
            wp.float64,
            tuple(masses.shape),
        )
        if len(masses.shape) != 3:
            raise ValueError("particles.masses must have shape (B, N, S).")
        boxes, particles, species = masses.shape
        device = masses.device
        _validate_array(
            "particles.concentration",
            self.particles.concentration,
            wp.float64,
            (boxes, particles),
            device,
        )
        _validate_array(
            "particles.charge",
            self.particles.charge,
            wp.float64,
            (boxes, particles),
            device,
        )
        _validate_array(
            "particles.density",
            self.particles.density,
            wp.float64,
            (species,),
            device,
        )
        _validate_array(
            "particles.volume",
            self.particles.volume,
            wp.float64,
            (boxes,),
            device,
        )
        _validate_array(
            "gas.molar_mass",
            self.gas.molar_mass,
            wp.float64,
            (species,),
            device,
        )
        _validate_array(
            "gas.concentration",
            self.gas.concentration,
            wp.float64,
            (boxes, species),
            device,
        )
        _validate_array(
            "gas.vapor_pressure",
            self.gas.vapor_pressure,
            wp.float64,
            (boxes, species),
            device,
        )
        _validate_array(
            "gas.partitioning",
            self.gas.partitioning,
            wp.int32,
            (boxes, species),
            device,
        )
        _validate_array(
            "environment.temperature",
            self.environment.temperature,
            wp.float64,
            (boxes,),
            device,
        )
        _validate_array(
            "environment.pressure",
            self.environment.pressure,
            wp.float64,
            (boxes,),
            device,
        )
        _validate_array(
            "environment.saturation_ratio",
            self.environment.saturation_ratio,
            wp.float64,
            (boxes, species),
            device,
        )
        if self.thermodynamics is None:
            raise ValueError("thermodynamics must not be None.")
        if self.mass_transfer is not None:
            _validate_array(
                "mass_transfer",
                self.mass_transfer,
                wp.float64,
                (boxes, particles, species),
                device,
            )
        if self.energy_transfer is not None:
            _validate_array(
                "energy_transfer",
                self.energy_transfer,
                wp.float64,
                (boxes, species),
                device,
            )

        _validate_output_ownership(
            self.mass_transfer,
            self.energy_transfer,
            (
                masses,
                self.particles.concentration,
                self.particles.charge,
                self.particles.density,
                self.particles.volume,
                self.gas.molar_mass,
                self.gas.concentration,
                self.gas.vapor_pressure,
                self.gas.partitioning,
                self.environment.temperature,
                self.environment.pressure,
                self.environment.saturation_ratio,
            ),
        )

    @property
    def backend_payload(self) -> tuple[object, object, object]:
        """Return the three caller-owned primary containers by identity."""
        return self.particles, self.gas, self.environment
