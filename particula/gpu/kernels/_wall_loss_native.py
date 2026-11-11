"""Read-only native resolved wall-loss preflight before caller-owned writers."""

from typing import Any, no_type_check

import numpy as np
import warp as wp

from particula.gpu.properties import (
    effective_density_wp,
    particle_radius_from_volume_wp,
)


@no_type_check
@wp.kernel
def _scan_population(
    masses: Any,
    concentration: Any,
    density: Any,
    volume: Any,
    invalid: Any,
) -> None:
    """Validate count density and representable material geometry per box."""
    box = wp.tid()
    for slot in range(masses.shape[1]):
        count = concentration[box, slot]
        if not wp.isfinite(count / volume[box]):
            wp.atomic_max(invalid, 0, 1)
        mass = wp.float64(0.0)
        material_volume = wp.float64(0.0)
        for species in range(masses.shape[2]):
            mass += masses[box, slot, species]
            material_volume += masses[box, slot, species] / density[species]
        if not wp.isfinite(mass) or not wp.isfinite(material_volume):
            wp.atomic_max(invalid, 0, 1)
        if count > wp.float64(0.0) and mass <= wp.float64(0.0):
            wp.atomic_max(invalid, 0, 1)
        if mass > wp.float64(0.0):
            if material_volume <= wp.float64(0.0):
                wp.atomic_max(invalid, 0, 1)
            else:
                radius = particle_radius_from_volume_wp(material_volume)
                material_density = effective_density_wp(mass, material_volume)
                if (
                    not wp.isfinite(radius)
                    or radius <= wp.float64(0.0)
                    or not wp.isfinite(material_density)
                    or material_density <= wp.float64(0.0)
                ):
                    wp.atomic_max(invalid, 0, 1)


def validate_physics(particles: Any) -> None:
    """Admit current raw-count physics without imposing equal weights."""
    boxes, _, species = particles.masses.shape
    if boxes < 1 or species < 1:
        raise ValueError(
            "native wall loss requires positive box/species counts"
        )
    device = particles.masses.device
    invalid = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _scan_population,
        dim=boxes,
        inputs=[
            particles.masses,
            particles.concentration,
            particles.density,
            particles.volume,
            invalid,
        ],
        device=device,
    )
    if invalid.numpy()[0]:
        raise ValueError(
            "invalid native wall-loss population or derived geometry"
        )


def _memory_range(array: Any) -> tuple[int, int]:
    """Return a contiguous array's byte range without payload access."""
    itemsize = {wp.float64: 8, wp.float32: 4, wp.int32: 4, wp.uint32: 4}.get(
        array.dtype
    )
    if itemsize is None:
        raise ValueError("unsupported native wall-loss storage dtype")
    strides: list[int] = []
    stride = itemsize
    for dimension in reversed(array.shape):
        strides.insert(0, stride)
        stride *= dimension
    if tuple(array.strides) != tuple(strides):
        raise ValueError("native wall-loss arrays must be contiguous")
    if np.prod(array.shape) == 0:
        return 0, 0
    start = int(array.ptr)
    return start, start + stride


def validate_ownership(
    particles: Any, states: Any, protected_inputs: tuple[Any, ...]
) -> None:
    """Reject writer overlap with another writer or protected input storage."""
    writers = [
        particles.masses,
        particles.concentration,
        particles.charge,
        states,
    ]
    protected = [particles.density, particles.volume, *protected_inputs]
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
                    "native wall-loss writers must not overlap state"
                )
