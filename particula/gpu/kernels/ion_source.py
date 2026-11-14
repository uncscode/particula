"""Fixed-capacity paired ion source; the caller owns arrays and RNG words."""

from typing import no_type_check

import warp as wp


@no_type_check
@wp.kernel
def inject_ion_pairs(  # noqa: C901
    masses: wp.array3d(dtype=wp.float64),
    counts: wp.array2d(dtype=wp.float64),
    charge: wp.array2d(dtype=wp.float64),
    volume: wp.array(dtype=wp.float64),
    rate: wp.float64,
    dt: wp.float64,
    positive_species: wp.int32,
    negative_species: wp.int32,
    positive_mass: wp.float64,
    negative_mass: wp.float64,
    rng: wp.array(dtype=wp.uint32),
    injected_pairs: wp.array(dtype=wp.int64),
    failure: wp.array(dtype=wp.int32),
):
    """Sample one pair count per box and reject insufficient capacity.

    No fractional ions or resampling. The local RNG word is published only
    after capacity admission. A fault in another box does not roll back boxes
    that already completed; the owner must stop dependent processes.
    """
    if failure[0] != 0:
        return
    box = wp.tid()
    mean = rate * volume[box] * dt
    if mean <= wp.float64(0.0):
        return
    # Warp's Poisson primitive takes float32 lambda. A bounded mean avoids
    # integer overflow; sampling accuracy is checked by ensemble tests.
    if not wp.isfinite(mean) or mean > wp.float64(1000000.0):
        wp.atomic_max(failure, 0, 4)
        return
    state = rng[box]
    pairs = wp.int32(wp.poisson(state, wp.float32(mean)))
    free = wp.int32(0)
    for slot in range(counts.shape[1]):
        if counts[box, slot] == wp.float64(0.0):
            free += 1
    if pairs > free // 2:
        wp.atomic_max(failure, 0, 3)
        return
    inserted = wp.int32(0)
    for slot in range(counts.shape[1]):
        if inserted < 2 * pairs and counts[box, slot] == wp.float64(0.0):
            for species in range(masses.shape[2]):
                masses[box, slot, species] = wp.float64(0.0)
            if inserted % 2 == 0:
                masses[box, slot, positive_species] = positive_mass
                charge[box, slot] = wp.float64(1.0)
            else:
                masses[box, slot, negative_species] = negative_mass
                charge[box, slot] = wp.float64(-1.0)
            counts[box, slot] = wp.float64(1.0)
            inserted += 1
    injected_pairs[box] += wp.int64(pairs)
    rng[box] = state
