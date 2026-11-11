"""Explicit full-gas order and injection alignment for native nucleation."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from particula.gas.gas_data import GasData
from particula.particles.particle_data import ParticleData


@dataclass(frozen=True)
class NucleationSpeciesMap:
    """Own unique gas names and one-to-one gas-to-particle injection pairs.

    Unmapped gas is preserved. Every positive injection mass must have a pair
    and a partitioning-enabled gas lane. Particle-only lanes receive no source.
    """

    gas_names: tuple[str, ...]
    gas_to_particle: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        """Detach input sequences and reject ambiguous lane identities."""
        if isinstance(self.gas_names, (str, bytes)):
            raise ValueError("gas_names must be an ordered sequence")
        names = tuple(self.gas_names)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("gas_names must be nonempty strings")
        if len(set(names)) != len(names):
            raise ValueError("gas_names must be unique")
        try:
            pairs = tuple(tuple(pair) for pair in self.gas_to_particle)
            if any(len(pair) != 2 for pair in pairs) or any(
                isinstance(index, (bool, np.bool_))
                or not isinstance(index, Integral)
                or index < 0
                for pair in pairs
                for index in pair
            ):
                raise ValueError("species pairs require nonnegative integers")
        except TypeError as error:
            raise ValueError(
                "gas_to_particle must contain index pairs"
            ) from error
        if any(source >= len(names) for source, _ in pairs):
            raise ValueError("gas species index is out of range")
        if len({source for source, _ in pairs}) != len(pairs):
            raise ValueError("duplicate gas species index")
        if len({target for _, target in pairs}) != len(pairs):
            raise ValueError("duplicate particle species index")
        object.__setattr__(self, "gas_names", names)
        object.__setattr__(self, "gas_to_particle", pairs)


def validate_nucleation_mapping(
    particles: ParticleData,
    gas: GasData,
    species_map: NucleationSpeciesMap | None,
    injection_mass: np.ndarray,
) -> np.ndarray:
    """Validate alignment and return particle-lane source mass."""
    particles.validate_representation()
    if particles.distribution_type != "particle_resolved":
        raise ValueError("native nucleation requires particle_resolved storage")
    if particles.n_boxes != 1:
        raise ValueError("native CPU nucleation requires exactly one box")
    if not isinstance(species_map, NucleationSpeciesMap):
        raise TypeError("native nucleation requires NucleationSpeciesMap")
    if tuple(gas.name) != species_map.gas_names:
        raise ValueError("gas names must match the expected complete gas order")
    width = len(species_map.gas_names)
    if (
        gas.concentration.shape != (1, width)
        or gas.molar_mass.shape != (width,)
        or gas.partitioning.shape != (width,)
        or gas.partitioning.dtype != np.bool_
        or injection_mass.shape != (width,)
    ):
        raise ValueError(
            "gas and injection shapes must match the complete order"
        )
    result = np.zeros(particles.n_species, dtype=np.float64)
    mapped = set()
    for source, target in species_map.gas_to_particle:
        if target >= particles.n_species:
            raise ValueError("particle species index is out of range")
        mapped.add(source)
        result[target] = injection_mass[source]
    if any(
        value > 0 and (index not in mapped or not gas.partitioning[index])
        for index, value in enumerate(injection_mass)
    ):
        raise ValueError("positive injection requires mapped partitioning gas")
    return result
