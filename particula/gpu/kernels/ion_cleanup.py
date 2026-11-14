"""Resident neutral ion-product export with deterministic per-box ledgers."""

from typing import no_type_check

import warp as wp


@no_type_check
@wp.kernel
def remove_neutral_ion_products(
    masses: wp.array3d(dtype=wp.float64),
    counts: wp.array2d(dtype=wp.float64),
    charges: wp.array2d(dtype=wp.float64),
    positive_species: wp.int32,
    negative_species: wp.int32,
    exported_mass: wp.array2d(dtype=wp.float64),
    exported_particles: wp.array(dtype=wp.int64),
    failure: wp.array(dtype=wp.int32),
):
    """Export only neutral products containing both reserved ion materials."""
    if failure[0] != 0:
        return
    box = wp.tid()
    for slot in range(counts.shape[1]):
        count = counts[box, slot]
        if count != wp.float64(0.0) and count != wp.float64(1.0):
            wp.atomic_max(failure, 0, 5)
            return
    for slot in range(counts.shape[1]):
        eligible = (
            counts[box, slot] == wp.float64(1.0)
            and charges[box, slot] == wp.float64(0.0)
            and masses[box, slot, positive_species] > wp.float64(0.0)
            and masses[box, slot, negative_species] > wp.float64(0.0)
        )
        for species in range(masses.shape[2]):
            if species != positive_species and species != negative_species:
                if masses[box, slot, species] != wp.float64(0.0):
                    eligible = False
        if eligible:
            for species in range(masses.shape[2]):
                exported_mass[box, species] += masses[box, slot, species]
                masses[box, slot, species] = wp.float64(0.0)
            counts[box, slot] = wp.float64(0.0)
            charges[box, slot] = wp.float64(0.0)
            exported_particles[box] += wp.int64(1)
