"""Device preparation for live fixed-capacity nucleation admission."""

from typing import no_type_check

import warp as wp


@no_type_check
@wp.kernel
def _releasable_slots(
    counts: wp.array2d(dtype=wp.float64),
    budget: wp.array(dtype=wp.int32),
):
    """Resampling can retain one active representative and release the rest."""
    box = wp.tid()
    active = int(0)
    for slot in range(counts.shape[1]):
        if counts[box, slot] > 0.0:
            active += 1
    budget[box] = wp.max(active - 1, 0)
