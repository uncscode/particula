"""Resident particle-field lifecycle kernels using native event decisions."""

from typing import Any, no_type_check

import warp as wp


@no_type_check
@wp.kernel
def merge_fields(
    values: wp.array3d(dtype=wp.float64),
    pairs: wp.array3d(dtype=wp.int32),
    collisions: wp.array(dtype=wp.int32),
    failure: wp.array(dtype=wp.int32),
):
    """Sum extensive per-particle quantities along applied compacted pairs."""
    box, channel = wp.tid()
    if failure[0] != 0:
        return
    for index in range(collisions[box]):
        recipient = pairs[box, index, 0]
        donor = pairs[box, index, 1]
        values[box, recipient, channel] += values[box, donor, channel]
        values[box, donor, channel] = wp.float64(0)


@no_type_check
@wp.kernel
def retire_fields(
    values: wp.array3d(dtype=Any),
    zero: Any,
    before: wp.array2d(dtype=wp.float64),
    after: wp.array2d(dtype=wp.float64),
    removed: wp.array2d(dtype=wp.float64),
    additive: int,
    failure: wp.array(dtype=wp.int32),
):
    """Preserve per-particle values under count loss and clear retired slots."""
    box, channel = wp.tid()
    if failure[0] != 0:
        return
    inventory = wp.float64(0)
    for slot in range(after.shape[1]):
        if after[box, slot] > before[box, slot]:
            wp.atomic_max(failure, 0, 91)
        if additive != 0:
            inventory += (before[box, slot] - after[box, slot]) * wp.float64(
                values[box, slot, channel]
            )
        if after[box, slot] == wp.float64(0):
            values[box, slot, channel] = zero
    if additive != 0:
        removed[box, channel] += inventory


@no_type_check
@wp.kernel
def birth_fields(
    values: wp.array3d(dtype=Any),
    initial: Any,
    counts: wp.array2d(dtype=wp.float64),
    selected: wp.array2d(dtype=wp.int32),
    selected_counts: wp.array(dtype=wp.int32),
    source: wp.array2d(dtype=wp.float64),
    additive: int,
    failure: wp.array(dtype=wp.int32),
):
    """Initialize actual activated slots, including reused resampling slots."""
    box, channel = wp.tid()
    if failure[0] != 0:
        return
    inventory = wp.float64(0)
    for rank in range(selected_counts[box]):
        slot = selected[box, rank]
        values[box, slot, channel] = initial
        inventory += counts[box, slot] * wp.float64(initial)
    if additive != 0:
        source[box, channel] += inventory


@no_type_check
@wp.kernel
def remap_fields(
    values: wp.array3d(dtype=wp.float64),
    original: wp.array3d(dtype=wp.float64),
    entries: wp.array(dtype=wp.int32),
    targets: wp.array2d(dtype=wp.int32),
    donors: wp.array2d(dtype=wp.int32),
    overlaps: wp.array2d(dtype=wp.float64),
    retained: wp.array2d(dtype=wp.int32),
    replacement_counts: wp.array2d(dtype=wp.float64),
    failure: wp.array(dtype=wp.int32),
):
    """Apply the exact native mass remap's recorded interval overlaps."""
    box, channel = wp.tid()
    if failure[0] != 0 or entries[box] == 0:
        return
    for slot in range(values.shape[1]):
        values[box, slot, channel] = wp.float64(0)
    for entry in range(entries[box]):
        rank = targets[box, entry]
        slot = retained[box, rank]
        donor = donors[box, entry]
        values[box, slot, channel] += (
            overlaps[box, entry]
            * original[box, donor, channel]
            / replacement_counts[box, rank]
        )


@no_type_check
@wp.kernel
def validate_fields(
    values: wp.array3d(dtype=Any),
    lower: Any,
    upper: Any,
    zero: Any,
    counts: wp.array2d(dtype=wp.float64),
    source: wp.array2d(dtype=wp.float64),
    removed: wp.array2d(dtype=wp.float64),
    additive: int,
    failure: wp.array(dtype=wp.int32),
):
    """Check active bounds, inactive clearing and finite inventory ledgers."""
    box, channel = wp.tid()
    if failure[0] != 0:
        return
    for slot in range(values.shape[1]):
        value = values[box, slot, channel]
        if not wp.isfinite(wp.float64(value)):
            wp.atomic_max(failure, 0, 90)
        if counts[box, slot] > wp.float64(0):
            if value < lower or value > upper:
                wp.atomic_max(failure, 0, 90)
        elif value != zero:
            wp.atomic_max(failure, 0, 90)
    if additive != 0:
        if not wp.isfinite(source[box, channel]) or not wp.isfinite(
            removed[box, channel]
        ):
            wp.atomic_max(failure, 0, 90)


@no_type_check
@wp.kernel
def preserve_activity(
    before: wp.array2d(dtype=wp.float64),
    after: wp.array2d(dtype=wp.float64),
    failure: wp.array(dtype=wp.int32),
):
    """Custom value updates must preserve particle slot activity."""
    box, slot = wp.tid()
    if failure[0] == 0:
        if (before[box, slot] > wp.float64(0)) != (
            after[box, slot] > wp.float64(0)
        ):
            wp.atomic_max(failure, 0, 91)
