"""Explicit species provenance and read-only native condensation admission.

The map belongs to the process, not to particle or gas storage. Gas names
describe the complete gas ordering; pairs declare ordinary one-to-one mass
transfer. An omitted pair is explicitly nonparticipating. No chemical identity
is inferred from array widths or partitioning flags.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from particula.gas.gas_data import GasData
from particula.particles.particle_data import ParticleData


@dataclass(frozen=True)
class CondensationSpeciesMap:
    """Immutable process-owned gas order and gas-to-particle lane pairs.

    Args:
        gas_names: Complete expected gas order, including inert/unmapped gas.
        gas_to_particle: Ordered ``(gas_index, particle_index)`` pairs. Indices
            must be nonnegative integers, with unique sources and targets.
            Empty pairs explicitly disable all transfer. Particle upper bounds
            are checked against current storage on every native admission.

    Raises:
        ValueError: If names or pairs are ambiguous or malformed.
    """

    gas_names: tuple[str, ...]
    gas_to_particle: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        """Detach caller sequences and reject ambiguous alignment."""
        if isinstance(self.gas_names, (str, bytes)):
            raise ValueError("gas_names must be an ordered sequence of names")
        names = tuple(self.gas_names)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("gas_names must be nonempty strings")
        if len(set(names)) != len(names):
            raise ValueError("gas_names must be unique")
        pairs = _validate_pairs(self.gas_to_particle, len(names))
        if len({source for source, _ in pairs}) != len(pairs):
            raise ValueError("duplicate gas species index")
        if len({target for _, target in pairs}) != len(pairs):
            raise ValueError("duplicate particle species index")
        object.__setattr__(self, "gas_names", names)
        object.__setattr__(self, "gas_to_particle", tuple(pairs))


def _validate_pairs(
    values: Sequence[tuple[int, int]],
    width: int,
) -> list[tuple[int, int]]:
    """Validate integer lane pairs and detach caller-owned sequences."""
    pairs: list[tuple[int, int]] = []
    try:
        for pair in values:
            if len(pair) != 2:
                raise ValueError("each species pair must have two indices")
            if any(
                isinstance(index, (bool, np.bool_))
                or not isinstance(index, Integral)
                or index < 0
                for index in pair
            ):
                raise ValueError("species indices must be nonnegative integers")
            source, target = int(pair[0]), int(pair[1])
            if source >= width:
                raise ValueError("gas species index is out of range")
            pairs.append((source, target))
    except TypeError as error:
        raise ValueError("gas_to_particle must contain index pairs") from error
    return pairs


def _validate_gas_state(
    gas: GasData,
    species_map: CondensationSpeciesMap,
) -> None:
    """Validate full gas schema, physical state and ordered provenance."""
    if (
        not isinstance(gas.name, (list, tuple))
        or tuple(gas.name) != species_map.gas_names
    ):
        raise ValueError("gas names must match the expected complete gas order")
    width = len(species_map.gas_names)
    for field, shape, dtype in (
        (gas.concentration, (1, width), np.dtype(np.float64)),
        (gas.molar_mass, (width,), np.dtype(np.float64)),
        (gas.partitioning, (width,), np.dtype(np.bool_)),
    ):
        if (
            not isinstance(field, np.ndarray)
            or field.shape != shape
            or field.dtype != dtype
        ):
            raise ValueError(
                "gas data must have single-box float64/bool schema"
            )
    if (
        not np.all(np.isfinite(gas.concentration))
        or np.any(gas.concentration < 0)
        or not np.all(np.isfinite(gas.molar_mass))
        or np.any(gas.molar_mass <= 0)
    ):
        raise ValueError("gas concentration and molar mass must be physical")


def validate_condensation_state(
    particle: ParticleData,
    gas: GasData,
    species_map: CondensationSpeciesMap | None,
    temperature: float,
    pressure: float,
    skip_partitioning_indices: Sequence[int] | None = None,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Validate current native state and return participating lane indices.

    Particle representation validation supplies physical mass/count/density,
    volume and PDF-grid checks. Gas storage is checked without calling its
    coercing constructor. Returned arrays are fresh; all input storage and
    metadata remain untouched, including on failure. Skip indices follow the
    full gas order. Nonpartitioning or unmapped gas never transfers.

    Args:
        particle: Explicitly tagged particle data, in SI units, with one box.
        gas: Full ordered gas state; concentration in kg/m3, molar mass kg/mol.
        species_map: Explicit expected gas order and transfer pairs.
        temperature: Positive finite scalar temperature in K.
        pressure: Positive finite scalar pressure in Pa.
        skip_partitioning_indices: Additional full-gas lanes to exclude.

    Returns:
        Participating gas and particle indices in declared pair order.

    Raises:
        TypeError: If containers or the process map have incorrect types.
        ValueError: If current state, configuration or physical inputs fail.
    """
    if not isinstance(particle, ParticleData) or not isinstance(gas, GasData):
        raise TypeError("native condensation requires ParticleData and GasData")
    if not isinstance(species_map, CondensationSpeciesMap):
        raise TypeError("native condensation requires CondensationSpeciesMap")
    particle.validate_representation()
    if particle.n_boxes != 1:
        raise ValueError("ParticleData must have n_boxes=1 for condensation")
    _validate_gas_state(gas, species_map)
    width = len(species_map.gas_names)
    for value, label in ((temperature, "temperature"), (pressure, "pressure")):
        if np.ndim(value) != 0 or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be a positive finite scalar")
    if any(
        target >= particle.n_species
        for _, target in species_map.gas_to_particle
    ):
        raise ValueError("particle species index is out of range")
    skipped: set[int] = set()
    if skip_partitioning_indices is not None:
        for index in skip_partitioning_indices:
            if (
                isinstance(index, (bool, np.bool_))
                or not isinstance(index, Integral)
                or index < 0
                or index >= width
            ):
                raise ValueError("skip indices must be valid full-gas indices")
            skipped.add(int(index))
    pairs = [
        (source, target)
        for source, target in species_map.gas_to_particle
        if gas.partitioning[source] and source not in skipped
    ]
    return (
        np.array([source for source, _ in pairs], dtype=np.intp),
        np.array([target for _, target in pairs], dtype=np.intp),
    )


def condensation_population_weights(
    particle: ParticleData,
) -> NDArray[np.float64]:
    """Return fresh per-node inventory weights in m^-3 for one CPU box.

    Resolved weights are counts/V; PMF weights are stored bin densities.
    PDF weights multiply dn/dr by trapezoidal radius-node widths, so a sum of
    weight times species mass equals the integral of the product. Neither the
    radius grid nor raw concentration is modified.

    Args:
        particle: Explicitly tagged particle data with SI storage.

    Returns:
        One density weight per particle/bin/PDF node.

    Raises:
        ValueError: If representation validation fails or boxes differ from 1.
    """
    density = particle.slot_concentration_density
    if particle.n_boxes != 1:
        raise ValueError("ParticleData must have n_boxes=1 for condensation")
    weights = density[0].copy()
    if particle.distribution_type == "continuous_pdf":
        grid = particle.radius_grid
        if grid is None:
            raise ValueError("PDF radius grid is required")
        intervals = np.diff(grid)
        widths = np.empty_like(grid)
        widths[0], widths[-1] = intervals[0] / 2, intervals[-1] / 2
        widths[1:-1] = (intervals[:-1] + intervals[1:]) / 2
        weights *= widths
    return weights
