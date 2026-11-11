"""Explicit device configuration binding for native nucleation injection."""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from particula.dynamics.nucleation.species_mapping import NucleationSpeciesMap
from particula.gpu.population_metadata import (
    PARTICLE_RESOLVED,
    particle_distribution_code,
)
from particula.gpu.species_mapping import _binding_schema, _layout


@dataclass(frozen=True, eq=False)
class NucleationSpeciesBinding:
    """Fixed explicit lane-map configuration for exact Warp particle/gas state.

    Create with ``prepare_nucleation_species_map``. Particle and gas arrays
    remain caller-owned. ``particle_to_gas`` and ``gas_to_particle`` are fresh
    device int32 arrays containing -1 for unmapped lanes. Partitioning
    is not baked into these maps: dispatch reads current full gas flags.
    Replacing storage, metadata, names or mapping requires fresh preparation.
    """

    particles: Any
    gas: Any
    species_map: NucleationSpeciesMap
    gas_names: tuple[str, ...]
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


def prepare_nucleation_species_map(
    particles: Any,
    gas: Any,
    *,
    gas_names: Sequence[str],
    species_map: NucleationSpeciesMap,
) -> NucleationSpeciesBinding:
    """Explicitly upload validated lane configuration, never simulation state.

    Preparation validates host provenance and storage metadata before the two
    configuration uploads. It performs no physical validation, kernel launch,
    synchronization, payload readback, or mutation of caller-owned arrays.
    Direct scientific admission owns physical checks and algorithm support.

    Args:
        particles: Tagged resolved Warp particle carrier.
        gas: Complete Warp gas carrier, independently sized from particles.
        gas_names: Actual full gas order retained by the caller on transfer.
        species_map: Immutable process-owned expected order and lane pairs.

    Returns:
        A fixed binding with newly owned device int32 forward/reverse maps.

    Raises:
        TypeError: If the process map has the wrong type.
        ValueError: If provenance, schema, indices or devices are incompatible.
    """
    if not isinstance(species_map, NucleationSpeciesMap):
        raise TypeError("species_map must be NucleationSpeciesMap")
    if isinstance(gas_names, (str, bytes)):
        raise ValueError("gas names must be the complete ordered sequence")
    actual_names = tuple(gas_names)
    if actual_names != species_map.gas_names:
        raise ValueError("gas names must match the expected complete gas order")
    code = particle_distribution_code(particles)
    if code != PARTICLE_RESOLVED:
        raise ValueError("nucleation binding requires resolved metadata")
    arrays, device = _binding_schema(particles, gas)
    particle_width = particles.masses.shape[2]
    gas_width = gas.concentration.shape[1]
    if gas_width != len(actual_names):
        raise ValueError("full gas name count must match gas storage")
    if any(
        target >= particle_width for _, target in species_map.gas_to_particle
    ):
        raise ValueError("particle species index is out of range")
    particle_to_gas = np.full(particle_width, -1, dtype=np.int32)
    gas_to_particle = np.full(gas_width, -1, dtype=np.int32)
    for source, target in species_map.gas_to_particle:
        particle_to_gas[target] = source
        gas_to_particle[source] = target
    import warp as wp

    forward: Any = wp.array(particle_to_gas, dtype=wp.int32, device=device)
    reverse: Any = wp.array(gas_to_particle, dtype=wp.int32, device=device)
    return NucleationSpeciesBinding(
        particles=particles,
        gas=gas,
        species_map=species_map,
        gas_names=actual_names,
        distribution_kind=code,
        arrays=arrays,
        array_layouts=tuple(
            _layout(array) for array in (*arrays, forward, reverse)
        ),
        device=device,
        particle_to_gas=forward,
        gas_to_particle=reverse,
    )
