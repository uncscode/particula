"""Discover and activate fixed particle slots in CPU particle data.

This module provides the CPU reference classification for active and free
particle-resolved slots. Its discovery API returns fixed-shape, newly
allocated diagnostics without changing particle storage. Its activation API
maps validated request prefixes into the ascending free slots after complete
read-only preflight.
"""

from typing import cast

import numpy as np
from numpy.typing import NDArray

from particula.particles.particle_data import ParticleData


def get_slot_diagnostics(
    data: ParticleData,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
    """Return read-only fixed-shape diagnostics for particle slots.

    A slot is active when its concentration is positive and finite, each mass
    lane is finite and nonnegative, and its total mass is positive and finite.
    A slot is free when its concentration, all mass lanes, and its charge are
    exactly zero. All other slot states are invalid. This function does not
    modify ``data`` and allocates fresh output arrays for every call.

    Args:
        data: Fixed-shape particle data to inspect without mutation.

    Returns:
        A tuple ``(free_indices, active_counts, free_counts)`` of newly
        allocated ``np.int32`` arrays. ``free_indices`` has shape
        ``(n_boxes, n_particles)``; each row contains ascending free-slot
        indices followed by ``-1``. The count arrays have shape ``(n_boxes,)``.

    Raises:
        ValueError: If any particle slot is neither active nor free. The error
            message is ``"Invalid particle slot state."``.
    """
    masses = data.masses
    concentration = data.concentration
    charge = data.charge

    expected_shape = masses.shape[:2] if masses.ndim == 3 else None
    if (
        expected_shape is None
        or concentration.shape != expected_shape
        or charge.shape != expected_shape
    ):
        raise ValueError("Invalid particle slot state.")

    mass_valid = np.all(np.isfinite(masses) & (masses >= 0.0), axis=-1)
    mass_zero = np.all(masses == 0.0, axis=-1)
    with np.errstate(over="ignore", invalid="ignore"):
        total_mass = np.sum(masses, axis=-1, dtype=np.float64)

    active = (
        np.isfinite(concentration)
        & (concentration > 0.0)
        & mass_valid
        & np.isfinite(total_mass)
        & (total_mass > 0.0)
        & np.isfinite(charge)
    )
    free = (
        (concentration == 0.0)
        & mass_zero
        & (charge == 0.0)
        & np.isfinite(charge)
    )

    if np.any(~(active | free)):
        raise ValueError("Invalid particle slot state.")

    free_indices = np.full((data.n_boxes, data.n_particles), -1, dtype=np.int32)
    free_counts = np.sum(free, axis=1, dtype=np.int32)
    active_counts = np.sum(active, axis=1, dtype=np.int32)
    particle_indices = np.arange(data.n_particles, dtype=np.int32)

    for box_index in range(data.n_boxes):
        count = free_counts[box_index]
        free_indices[box_index, :count] = particle_indices[free[box_index]]

    return free_indices, active_counts, free_counts


def _validate_destinations(
    data: ParticleData,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Validate and return writable activation destination arrays.

    Args:
        data: Particle data whose mutable mass, concentration, and charge
            storage is validated.

    Returns:
        The validated mass, concentration, and charge arrays, in that order.

    Raises:
        ValueError: If a mutable destination is not a writable ``float64``
            NumPy array, storage overlaps another mutable or protected field,
            or the destination shapes are incompatible.
    """
    masses = getattr(data, "masses", None)
    concentration = getattr(data, "concentration", None)
    charge = getattr(data, "charge", None)
    fields = (masses, concentration, charge)
    if any(
        not isinstance(field, np.ndarray)
        or field.dtype != np.float64
        or not field.flags.writeable
        for field in fields
    ):
        raise ValueError(
            "Particle destination fields must be writable float64 arrays."
        )
    masses = cast(NDArray[np.float64], masses)
    concentration = cast(NDArray[np.float64], concentration)
    charge = cast(NDArray[np.float64], charge)
    if (
        masses.ndim != 3
        or concentration.shape != masses.shape[:2]
        or charge.shape != masses.shape[:2]
    ):
        raise ValueError("Particle destination fields have invalid shapes.")
    protected_fields = (data.density, data.volume)
    if any(
        np.shares_memory(first, second)
        for index, first in enumerate((masses, concentration, charge))
        for second in (masses, concentration, charge)[index + 1 :]
    ) or any(
        np.shares_memory(destination, protected)
        for destination in (masses, concentration, charge)
        for protected in protected_fields
    ):
        raise ValueError("Particle destination fields must not share storage.")
    return masses, concentration, charge


def _validate_requests(
    request_masses: NDArray[np.float64],
    request_concentration: NDArray[np.float64],
    request_charge: NDArray[np.float64],
    requested_counts: NDArray[np.integer],
    n_boxes: int,
    n_species: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Validate request schemas and return their fields.

    Args:
        request_masses: Requested per-species mass records.
        request_concentration: Requested concentration records.
        request_charge: Requested charge records.
        requested_counts: Per-box prefix lengths to activate.
        n_boxes: Required number of request boxes.
        n_species: Required number of mass lanes in each request record.

    Returns:
        The validated request mass, concentration, and charge arrays, in that
        order.

    Raises:
        ValueError: If request fields are not shaped ``float64`` NumPy arrays,
            counts are not a valid integer vector, or a count exceeds request
            capacity.
    """
    fields = (request_masses, request_concentration, request_charge)
    if any(
        not isinstance(field, np.ndarray) or field.dtype != np.float64
        for field in fields
    ):
        raise ValueError("Request fields must be float64 NumPy arrays.")
    if (
        request_masses.ndim != 3
        or request_concentration.ndim != 2
        or request_charge.ndim != 2
        or request_masses.shape[0] != n_boxes
        or request_masses.shape[2] != n_species
        or request_concentration.shape != request_masses.shape[:2]
        or request_charge.shape != request_masses.shape[:2]
    ):
        raise ValueError("Request fields have invalid shapes.")
    if (
        not isinstance(requested_counts, np.ndarray)
        or requested_counts.dtype.kind not in "iu"
        or requested_counts.ndim != 1
        or requested_counts.shape != (n_boxes,)
    ):
        raise ValueError(
            "requested_counts must be a one-dimensional integer array."
        )
    request_capacity = request_masses.shape[1]
    if np.any(requested_counts < 0) or np.any(
        requested_counts > request_capacity
    ):
        raise ValueError("requested_counts exceeds request capacity.")
    return fields


def _validate_preflight(
    request_fields: tuple[
        NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
    ],
    destination_fields: tuple[
        NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
    ],
    requested_counts: NDArray[np.integer],
    free_counts: NDArray[np.int32],
) -> None:
    """Validate storage isolation, capacity, and selected request records.

    Only request records within each box's declared prefix are inspected.
    This validation performs no writes so that activation remains atomic when
    a later box or record is invalid.

    Args:
        request_fields: Validated request mass, concentration, and charge
            arrays.
        destination_fields: Validated mutable mass, concentration, and charge
            destination arrays.
        requested_counts: Per-box request prefix lengths.
        free_counts: Number of currently free slots in each box.

    Raises:
        ValueError: If request storage aliases a destination, a requested
            prefix exceeds free capacity, or a selected record is invalid.
    """
    for request in request_fields:
        if any(
            np.shares_memory(request, destination)
            for destination in destination_fields
        ):
            raise ValueError(
                "Request fields must not share destination storage."
            )
    request_masses, request_concentration, request_charge = request_fields
    for box_index, count in enumerate(requested_counts):
        count_as_int = int(count)
        if count_as_int > int(free_counts[box_index]):
            raise ValueError("Requested activation exceeds free slot capacity.")
        masses = request_masses[box_index, :count_as_int]
        concentration = request_concentration[box_index, :count_as_int]
        charge = request_charge[box_index, :count_as_int]
        if (
            not np.all(np.isfinite(masses) & (masses >= 0.0))
            or not np.all(np.isfinite(concentration) & (concentration > 0.0))
            or not np.all(np.isfinite(charge))
        ):
            raise ValueError("Requested activation record is invalid.")
        with np.errstate(over="ignore", invalid="ignore"):
            total_mass = np.sum(masses, axis=-1, dtype=np.float64)
        if not np.all(np.isfinite(total_mass) & (total_mass > 0.0)):
            raise ValueError("Requested activation record is invalid.")


def activate_slots(
    data: ParticleData,
    request_masses: NDArray[np.float64],
    request_concentration: NDArray[np.float64],
    request_charge: NDArray[np.float64],
    requested_counts: NDArray[np.integer],
) -> NDArray[np.int32]:
    """Activate ascending free slots from per-box request prefixes.

    Request rank ``r`` in each box is copied into that box's ``r``-th
    ascending free slot. Only records below the corresponding requested count
    are validated or read. The destination fields mutate in place after a
    complete global preflight, and a newly allocated ``np.int32`` count array
    is returned.

    Args:
        data: Particle storage with writable float64 mass, concentration, and
            charge arrays.
        request_masses: Requested per-species masses with shape
            ``(n_boxes, request_capacity, n_species)``.
        request_concentration: Requested concentrations with shape
            ``(n_boxes, request_capacity)``.
        request_charge: Requested charges with shape
            ``(n_boxes, request_capacity)``.
        requested_counts: Integer prefix counts with shape ``(n_boxes,)``.

    Returns:
        A fresh ``np.int32`` array containing the activated count per box.

    Raises:
        ValueError: If an input schema, destination, slot state, requested
            prefix, or available free capacity is invalid.
    """
    if not isinstance(data, ParticleData):
        raise ValueError("data must be a ParticleData instance.")

    masses, concentration, charge = _validate_destinations(data)
    destination_fields = (masses, concentration, charge)

    n_boxes, _, n_species = masses.shape
    free_indices, _, free_counts = get_slot_diagnostics(data)
    request_fields = _validate_requests(
        request_masses,
        request_concentration,
        request_charge,
        requested_counts,
        n_boxes,
        n_species,
    )
    _validate_preflight(
        request_fields,
        destination_fields,
        requested_counts,
        free_counts,
    )

    for box_index, count in enumerate(requested_counts):
        count_as_int = int(count)
        if count_as_int == 0:
            continue
        slots = free_indices[box_index, :count_as_int]
        masses[box_index, slots] = request_masses[box_index, :count_as_int]
        concentration[box_index, slots] = request_concentration[
            box_index, :count_as_int
        ]
        charge[box_index, slots] = request_charge[box_index, :count_as_int]

    return requested_counts.astype(np.int32, copy=True)
