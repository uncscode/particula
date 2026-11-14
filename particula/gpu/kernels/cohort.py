"""Resident original-seed lineage bookkeeping, separate from particle counts."""

from typing import no_type_check

import warp as wp


@no_type_check
@wp.kernel
def initialize_cohort(
    masses: wp.array3d(dtype=wp.float64),
    counts: wp.array2d(dtype=wp.float64),
    seed_species: wp.int32,
    background_species: wp.int32,
    pending: wp.array2d(dtype=wp.int64),
    background: wp.array2d(dtype=wp.int32),
    statistics: wp.array2d(dtype=wp.int64),
):
    """Assign one original lineage per initial seed-containing particle."""
    box = wp.tid()
    for slot in range(counts.shape[1]):
        if counts[box, slot] == wp.float64(1.0):
            if masses[box, slot, seed_species] > wp.float64(0.0):
                pending[box, slot] = wp.int64(1)
                statistics[box, 0] += wp.int64(1)
            if masses[box, slot, background_species] > wp.float64(0.0):
                background[box, slot] = wp.int32(1)


@no_type_check
@wp.kernel
def merge_cohort(
    pairs: wp.array3d(dtype=wp.int32),
    n_collisions: wp.array(dtype=wp.int32),
    pending: wp.array2d(dtype=wp.int64),
    background: wp.array2d(dtype=wp.int32),
    failure: wp.array(dtype=wp.int32),
):
    """Follow applied compacted pairs and clear donor labels for reuse."""
    if failure[0] != 0:
        return
    box = wp.tid()
    for index in range(n_collisions[box]):
        recipient = pairs[box, index, 0]
        donor = pairs[box, index, 1]
        pending[box, recipient] += pending[box, donor]
        background[box, recipient] = wp.max(
            background[box, recipient], background[box, donor]
        )
        pending[box, donor] = wp.int64(0)
        background[box, donor] = wp.int32(0)


@no_type_check
@wp.kernel
def observe_cohort(
    masses: wp.array3d(dtype=wp.float64),
    counts: wp.array2d(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    target_volume: wp.float64,
    pending: wp.array2d(dtype=wp.int64),
    background: wp.array2d(dtype=wp.int32),
    statistics: wp.array2d(dtype=wp.int64),
    failure: wp.array(dtype=wp.int32),
):
    """Count capture before size crossing, then enforce lineage partition."""
    if failure[0] != 0:
        return
    box = wp.tid()
    unresolved = wp.int64(0)
    for slot in range(counts.shape[1]):
        count = counts[box, slot]
        if count != wp.float64(0.0) and count != wp.float64(1.0):
            wp.atomic_max(failure, 0, 6)
            return
        lineages = pending[box, slot]
        if lineages > wp.int64(0):
            if count == wp.float64(0.0):
                wp.atomic_max(failure, 0, 6)
                return
            if background[box, slot] != 0:
                statistics[box, 2] += lineages
                pending[box, slot] = wp.int64(0)
            else:
                volume = wp.float64(0.0)
                for species in range(masses.shape[2]):
                    volume += masses[box, slot, species] / density[species]
                if volume >= target_volume:
                    statistics[box, 1] += lineages
                    pending[box, slot] = wp.int64(0)
                else:
                    unresolved += lineages
    statistics[box, 3] = unresolved
    if statistics[box, 0] != (
        statistics[box, 1] + statistics[box, 2] + unresolved
    ):
        wp.atomic_max(failure, 0, 6)
