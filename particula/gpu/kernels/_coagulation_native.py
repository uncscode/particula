"""Native resolved coagulation preflight; no caller-owned storage is written.

Population counts retain their physical volume authority. Equal positive
counts are necessary for the existing donor/recipient merge. Validation uses
private device flags only; preparation may synchronize, enqueue must not.
"""

from typing import Any, no_type_check

import numpy as np
import warp as wp


def validate_schema(particles: Any) -> None:
    """Require native fp64 primary storage before payload access."""
    shape: Any = getattr(particles.masses, "shape", ())
    if len(shape) != 3 or shape[0] < 1 or shape[2] < 1:
        raise ValueError(
            "native coagulation masses require (boxes, slots, species)"
        )
    boxes, slots, species = shape
    for name, expected in (
        ("masses", shape),
        ("concentration", (boxes, slots)),
        ("charge", (boxes, slots)),
        ("density", (species,)),
        ("volume", (boxes,)),
    ):
        array = getattr(particles, name)
        if (
            getattr(array, "shape", None) != expected
            or getattr(array, "dtype", None) != wp.float64
        ):
            raise ValueError(
                f"native coagulation {name} requires fp64 {expected}"
            )


@no_type_check
@wp.kernel
def _validate_volume_and_derived(
    masses: Any,
    concentration: Any,
    density: Any,
    volume: Any,
    requested_volume: Any,
    invalid: Any,
) -> None:
    """Check volume authority and finite derived population quantities."""
    box = wp.tid()  # type: ignore[misc]
    v = volume[box]
    if not wp.isfinite(v) or v <= wp.float64(0.0):
        wp.atomic_max(invalid, 0, 1)
    if requested_volume[box] != v:
        wp.atomic_max(invalid, 0, 1)
    for slot in range(masses.shape[1]):
        if not wp.isfinite(concentration[box, slot] / v):
            wp.atomic_max(invalid, 1, 1)
        mass = wp.float64(0.0)
        material_volume = wp.float64(0.0)
        for species in range(masses.shape[2]):
            mass += masses[box, slot, species]
            material_volume += masses[box, slot, species] / density[species]
        if not wp.isfinite(mass) or not wp.isfinite(material_volume):
            wp.atomic_max(invalid, 1, 1)
        if mass > wp.float64(0.0) and material_volume <= wp.float64(0.0):
            wp.atomic_max(invalid, 1, 1)


def validate_physics(particles: Any, requested_volume: Any) -> None:
    """Admit equal active counts and physical state before any state writer."""
    from .coagulation import _validate_sedimentation_particle_physics_kernel

    device = particles.masses.device
    boxes, slots, _ = particles.masses.shape
    invalid = wp.zeros(7, dtype=wp.int32, device=device)
    # Reuse the established field/weight reduction without importing the
    # sedimentation-specific active-count cap into unrelated mechanisms.
    wp.launch(
        _validate_sedimentation_particle_physics_kernel,
        dim=boxes,
        inputs=[
            particles.masses,
            particles.concentration,
            particles.density,
            particles.charge,
            wp.int32(slots),
            invalid,
        ],
        device=device,
    )
    values = invalid.numpy()
    if values[4]:
        raise ValueError(
            "native coagulation requires equal positive active counts"
        )
    if np.any(values):
        raise ValueError("invalid native coagulation physical state")
    volume_flags = wp.zeros(2, dtype=wp.int32, device=device)
    wp.launch(
        _validate_volume_and_derived,
        dim=boxes,
        inputs=[
            particles.masses,
            particles.concentration,
            particles.density,
            particles.volume,
            requested_volume,
            volume_flags,
        ],
        device=device,
    )
    values = volume_flags.numpy()
    if values[0]:
        raise ValueError(
            "native coagulation volume must match positive particle volume"
        )
    if values[1]:
        raise ValueError("native coagulation derived quantities must be finite")


def _memory_range(array: Any) -> tuple[int, int]:
    """Check contiguous supported storage and return its byte range."""
    sizes = {wp.float64: 8, wp.float32: 4, wp.int32: 4, wp.uint32: 4}
    itemsize = sizes.get(array.dtype)
    if itemsize is None:
        raise ValueError("unsupported native coagulation storage dtype")
    strides: list[int] = []
    stride = itemsize
    for dimension in reversed(array.shape):
        strides.insert(0, stride)
        stride *= dimension
    if tuple(array.strides) != tuple(strides):
        raise ValueError("native coagulation arrays must be contiguous")
    if np.prod(array.shape) == 0:
        return 0, 0
    start = int(array.ptr)
    return start, start + stride


def validate_ownership(
    particles: Any,
    pairs: Any,
    counts: Any,
    states: Any,
    protected_inputs: tuple[Any, ...],
) -> None:
    """Reject every native writer overlapping another writer or read source."""
    writers = [
        particles.masses,
        particles.concentration,
        particles.charge,
        pairs,
        counts,
        states,
    ]
    protected = [particles.density, particles.volume, *protected_inputs]
    # Validate storage metadata for all arrays, even if a writer is empty.
    ranges = {
        id(array): _memory_range(array)
        for array in (*writers, *protected)
        if array is not None and hasattr(array, "ptr")
    }
    for index, writer in enumerate(writers):
        if writer is None:
            continue
        start, end = ranges[id(writer)]
        if start == end:
            continue
        for other in (*writers[index + 1 :], *protected):
            if other is None or not hasattr(other, "ptr"):
                continue
            other_start, other_end = ranges[id(other)]
            if start < other_end and other_start < end:
                raise ValueError(
                    "native coagulation writers must not overlap state"
                )
