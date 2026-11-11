"""Read-only metadata, schema and storage checks for native population users."""

from typing import Any, no_type_check

import warp as wp

from particula.gpu.kernels.environment import _is_warp_array_like
from particula.gpu.population_metadata import (
    DISCRETE,
    particle_distribution_type,
)


def array_field(
    container: Any, name: str, shape: tuple[int, ...], dtype: Any, device: Any
) -> Any:
    """Require a contiguous same-device field without reading its payload."""
    value = getattr(container, name, None)
    if (
        not _is_warp_array_like(value)
        or value.dtype != dtype
        or value.shape != shape
        or str(value.device) != str(device)
    ):
        raise ValueError(
            f"native {name} must have matching shape, dtype and device"
        )
    memory_range(value)
    return value


def memory_range(array: Any) -> tuple[int, int]:
    """Return a checked contiguous array interval for all supported dtypes."""
    size = {wp.float64: 8, wp.float32: 4, wp.int32: 4, wp.uint32: 4}.get(
        array.dtype
    )
    if size is None:
        raise ValueError("unsupported native storage dtype")
    strides = []
    length = size
    for dimension in reversed(array.shape):
        strides.insert(0, length)
        length *= dimension
    if tuple(array.strides) != tuple(strides):
        raise ValueError("native arrays must be contiguous")
    if length == 0:
        return 0, 0
    pointer = int(array.ptr)
    return pointer, pointer + length


def reject_writer_aliases(
    writers: tuple[Any, ...], protected: tuple[Any, ...]
) -> None:
    """Reject writers overlapping other writers or protected state."""
    for index, writer in enumerate(writers):
        start, end = memory_range(writer)
        for other in (*writers[index + 1 :], *protected):
            other_start, other_end = memory_range(other)
            if writer is other or (start < other_end and other_start < end):
                raise ValueError(
                    "native writers must not overlap protected state"
                )


@no_type_check
@wp.kernel
def _scan_population(
    masses: Any,
    counts: Any,
    charge: Any,
    density: Any,
    volume: Any,
    code: int,
    invalid: Any,
) -> None:
    """Check representation values on device without a caller-array readback."""
    box = wp.tid()
    box_volume = volume[box]
    if not wp.isfinite(box_volume) or box_volume <= wp.float64(0):
        wp.atomic_max(invalid, 0, 1)
        return
    if code == 1 and box_volume != wp.float64(1):
        wp.atomic_max(invalid, 0, 1)
    for species in range(masses.shape[2]):
        if not wp.isfinite(density[species]) or density[species] <= wp.float64(
            0
        ):
            wp.atomic_max(invalid, 0, 1)
    for slot in range(masses.shape[1]):
        count = counts[box, slot]
        if (
            not wp.isfinite(count)
            or count < wp.float64(0)
            or not wp.isfinite(count / box_volume)
        ):
            wp.atomic_max(invalid, 0, 1)
        if not wp.isfinite(charge[box, slot]):
            wp.atomic_max(invalid, 0, 1)
        for species in range(masses.shape[2]):
            mass = masses[box, slot, species]
            if not wp.isfinite(mass) or mass < wp.float64(0):
                wp.atomic_max(invalid, 0, 1)


def validate_population(particles: Any) -> tuple[Any, ...]:
    """Validate native PMF/resolved representation and return primary fields.

    This establishes storage interpretation only, not algorithm admission or
    active-slot geometry. Callers retain those process-specific checks.
    """
    kind = particle_distribution_type(particles)
    if kind is None:
        raise ValueError(
            "native validation requires explicit population metadata"
        )
    masses = getattr(particles, "masses", None)
    if not _is_warp_array_like(masses) or masses.ndim != 3:
        raise ValueError("native masses must be rank-three Warp storage")
    boxes, slots, species = masses.shape
    device = masses.device
    fields = (
        array_field(
            particles, "masses", (boxes, slots, species), wp.float64, device
        ),
        array_field(
            particles, "concentration", (boxes, slots), wp.float64, device
        ),
        array_field(particles, "charge", (boxes, slots), wp.float64, device),
        array_field(particles, "density", (species,), wp.float64, device),
        array_field(particles, "volume", (boxes,), wp.float64, device),
    )
    if boxes:
        invalid = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(
            _scan_population,
            dim=boxes,
            inputs=[*fields, DISCRETE if kind == "discrete" else 3, invalid],
            device=device,
        )
        if invalid.numpy()[0]:
            raise ValueError("invalid native population values or volume")
    return fields
