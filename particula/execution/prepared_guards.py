"""Persistent device failure admission for prepared scientific sequences."""

from typing import no_type_check

import warp as wp

from particula.util.constants import GAS_CONSTANT

_GAS_CONSTANT = wp.constant(wp.float64(GAS_CONSTANT))


@no_type_check
@wp.kernel
def equal_active_counts(
    counts: wp.array2d(dtype=wp.float64),
    failure: wp.array(dtype=wp.int32),
):
    """Reject an evolving unequal-weight cohort before any collision writer."""
    box = wp.tid()
    if failure[0] != 0:
        return
    first = wp.float64(0.0)
    for slot in range(counts.shape[1]):
        value = counts[box, slot]
        if not wp.isfinite(value) or value < 0.0:
            wp.atomic_max(failure, 0, 2)
        if value > 0.0:
            if first == 0.0:
                first = value
            elif value != first:
                wp.atomic_max(failure, 0, 2)


@no_type_check
@wp.kernel
def latch_nucleation_failure(
    pipeline: wp.array(dtype=wp.int32),
    p1: wp.array(dtype=wp.int32),
    p2: wp.array(dtype=wp.int32),
    p3: wp.array(dtype=wp.int32),
    molecule: wp.array(dtype=wp.int32),
    p4: wp.array(dtype=wp.int32),
    resampling_invalid: wp.array(dtype=wp.int32),
    resampling_failed: wp.array(dtype=wp.int32),
    scaling: wp.array(dtype=wp.int32),
    final: wp.array(dtype=wp.int32),
    commit: wp.array(dtype=wp.int32),
    saved: wp.array(dtype=wp.int32),
    failure: wp.array(dtype=wp.int32),
):
    """Retain the first source failure and its phase diagnostics forever."""
    if failure[0] != 0 or (pipeline[0] == 0 and (commit[0] & 1) == 0):
        return
    for index in range(14):
        saved[index] = p1[index]
    for index in range(2):
        saved[14 + index] = p2[index]
        saved[16 + index] = p3[index]
    saved[18] = molecule[0]
    saved[19] = p4[0]
    if resampling_invalid:
        saved[20] = resampling_invalid[0]
        saved[21] = resampling_failed[0]
    if scaling:
        saved[22] = scaling[0]
    saved[23] = final[0]
    saved[24] = commit[0]
    failure[0] = 1


@no_type_check
@wp.kernel
def refresh_saturation(
    concentration: wp.array2d(dtype=wp.float64),
    molar_mass: wp.array(dtype=wp.float64),
    vapor_pressure: wp.array2d(dtype=wp.float64),
    temperature: wp.array(dtype=wp.float64),
    saturation: wp.array2d(dtype=wp.float64),
    failure: wp.array(dtype=wp.int32),
):
    """Refresh full-gas ratios only while the sequence remains valid."""
    if failure[0] != 0:
        return
    box, species = wp.tid()
    saturation[box, species] = (
        concentration[box, species]
        * _GAS_CONSTANT
        * temperature[box]
        / (molar_mass[species] * vapor_pressure[box, species])
    )
