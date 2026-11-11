"""Explicit preparation of process-owned condensation lane maps on a device.

Gas names remain CPU-owned provenance. Callers supply their actual full order
at this boundary; it must match the process expectation. The returned binding
pins exact carriers and arrays and owns fresh device maps. These maps are
configuration, not physical state. They must remain unmodified while bound.
No name, lane, distribution, or device is inferred from equal array widths.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Sequence

import numpy as np

from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.gpu.population_metadata import (
    DISCRETE,
    PARTICLE_RESOLVED,
    particle_distribution_code,
)

_PARTICLE_FIELDS = ("masses", "concentration", "charge", "density", "volume")
_GAS_FIELDS = ("concentration", "molar_mass", "partitioning", "vapor_pressure")


def _layout(array: Any) -> tuple[Any, ...]:
    """Freeze metadata that could otherwise change behind an array identity."""
    return (array.dtype, array.shape, array.strides, array.device, array.ptr)


def _binding_schema(particles: Any, gas: Any) -> tuple[tuple[Any, ...], Any]:
    """Check complete carrier schemas and a common device without readback."""
    import warp as wp

    fields: tuple[Any, ...] = tuple(
        getattr(particles, name, None) for name in _PARTICLE_FIELDS
    )
    fields += tuple(getattr(gas, name, None) for name in _GAS_FIELDS)
    if any(not isinstance(field, wp.array) for field in fields):
        raise ValueError("species binding requires complete Warp array storage")
    masses, concentration, charge, density, volume, gas_c, molar, flags, vp = (
        fields
    )
    if masses.ndim != 3 or gas_c.ndim != 2:
        raise ValueError("species binding requires particle/gas rank 3/2")
    boxes, slots, particle_width = masses.shape
    gas_width = gas_c.shape[1]
    shapes = (
        (boxes, slots, particle_width),
        (boxes, slots),
        (boxes, slots),
        (particle_width,),
        (boxes,),
        (boxes, gas_width),
        (gas_width,),
        (boxes, gas_width),
        (boxes, gas_width),
    )
    for field, shape in zip(fields, shapes, strict=True):
        dtype = wp.int32 if field is flags else wp.float64
        if field.shape != shape or field.dtype != dtype:
            raise ValueError(
                "species binding has incompatible shapes or dtypes"
            )
        if field.device != masses.device:
            raise ValueError("species binding arrays must share one device")
    return fields, masses.device


@dataclass(frozen=True, eq=False)
class CondensationSpeciesBinding:
    """Fixed explicit lane-map configuration for exact Warp particle/gas state.

    Create with ``prepare_condensation_species_map``. Particle and gas arrays
    remain caller-owned. ``particle_to_gas`` and ``gas_to_particle`` are fresh
    device int32 arrays containing -1 for unmapped/skipped lanes. Partitioning
    is not baked into these maps: dispatch reads current full gas flags.
    Replacing storage, metadata, names or mapping requires fresh preparation.
    """

    particles: Any
    gas: Any
    species_map: CondensationSpeciesMap
    gas_names: tuple[str, ...]
    skipped_gas_indices: tuple[int, ...]
    distribution_kind: int
    arrays: tuple[Any, ...]
    array_layouts: tuple[tuple[Any, ...], ...]
    device: Any
    particle_to_gas: Any
    gas_to_particle: Any

    def validate(self, particles: Any, gas: Any) -> None:
        """Reject incompatible binding reuse before numerical preparation."""
        if particles is not self.particles or gas is not self.gas:
            raise ValueError("species binding belongs to different carriers")
        if particle_distribution_code(particles) != self.distribution_kind:
            raise ValueError("species binding population metadata changed")
        arrays, device = _binding_schema(particles, gas)
        if device != self.device or any(
            current is not pinned
            for current, pinned in zip(arrays, self.arrays, strict=True)
        ):
            raise ValueError("species binding storage changed; prepare again")
        current_layouts = tuple(
            _layout(array)
            for array in (*arrays, self.particle_to_gas, self.gas_to_particle)
        )
        if current_layouts != self.array_layouts:
            raise ValueError(
                "species binding array layout changed; prepare again"
            )


def prepare_condensation_species_map(
    particles: Any,
    gas: Any,
    *,
    gas_names: Sequence[str],
    species_map: CondensationSpeciesMap,
    skip_partitioning_indices: Sequence[int] = (),
) -> CondensationSpeciesBinding:
    """Explicitly upload validated lane configuration, never simulation state.

    Preparation validates host provenance and storage metadata before the two
    configuration uploads. It performs no physical validation, kernel launch,
    synchronization, payload readback, or mutation of caller-owned arrays.
    Direct scientific admission owns physical checks and algorithm support.

    Args:
        particles: Tagged discrete or resolved Warp particle carrier.
        gas: Complete Warp gas carrier, independently sized from particles.
        gas_names: Actual full gas order retained by the caller on transfer.
        species_map: Immutable process-owned expected order and lane pairs.
        skip_partitioning_indices: Optional full gas indices to exclude.

    Returns:
        A fixed binding with newly owned device int32 forward/reverse maps.

    Raises:
        TypeError: If the process map has the wrong type.
        ValueError: If provenance, schema, indices or devices are incompatible.
    """
    if not isinstance(species_map, CondensationSpeciesMap):
        raise TypeError("species_map must be CondensationSpeciesMap")
    if isinstance(gas_names, (str, bytes)):
        raise ValueError("gas names must be the complete ordered sequence")
    actual_names = tuple(gas_names)
    if actual_names != species_map.gas_names:
        raise ValueError("gas names must match the expected complete gas order")
    code = particle_distribution_code(particles)
    if code not in (DISCRETE, PARTICLE_RESOLVED):
        raise ValueError(
            "species binding requires discrete or resolved metadata"
        )
    arrays, device = _binding_schema(particles, gas)
    particle_width = particles.masses.shape[2]
    gas_width = gas.concentration.shape[1]
    if gas_width != len(actual_names):
        raise ValueError("full gas name count must match gas storage")
    if any(
        target >= particle_width for _, target in species_map.gas_to_particle
    ):
        raise ValueError("particle species index is out of range")
    skipped = tuple(skip_partitioning_indices)
    if any(
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, Integral)
        or index < 0
        or index >= gas_width
        for index in skipped
    ):
        raise ValueError("skip indices must be valid full-gas indices")
    particle_to_gas = np.full(particle_width, -1, dtype=np.int32)
    gas_to_particle = np.full(gas_width, -1, dtype=np.int32)
    for source, target in species_map.gas_to_particle:
        if source not in skipped:
            particle_to_gas[target] = source
            gas_to_particle[source] = target
    import warp as wp

    forward: Any = wp.array(particle_to_gas, dtype=wp.int32, device=device)
    reverse: Any = wp.array(gas_to_particle, dtype=wp.int32, device=device)
    return CondensationSpeciesBinding(
        particles=particles,
        gas=gas,
        species_map=species_map,
        gas_names=actual_names,
        skipped_gas_indices=tuple(int(index) for index in skipped),
        distribution_kind=code,
        arrays=arrays,
        array_layouts=tuple(
            _layout(array) for array in (*arrays, forward, reverse)
        ),
        device=device,
        particle_to_gas=forward,
        gas_to_particle=reverse,
    )
