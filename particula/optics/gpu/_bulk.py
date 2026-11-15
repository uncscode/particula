# Warp array annotations and vector operations are compiled by Warp.
# mypy: disable-error-code="valid-type,operator,index"
"""Chunk gathering and accumulation for bounded-memory population optics."""

import math

import warp as wp

_PI = wp.constant(wp.float64(math.pi))


@wp.kernel
def gather_cases(
    indices: wp.array2d(dtype=wp.vec2d),
    sizes: wp.array2d(dtype=wp.float64),
    radii: wp.array2d(dtype=wp.float64),
    weights: wp.array2d(dtype=wp.float64),
    group: int,
    start: int,
    valid_count: int,
    index: wp.array(dtype=wp.vec2d),
    size: wp.array(dtype=wp.float64),
    radius: wp.array(dtype=wp.float64),
    weight: wp.array(dtype=wp.float64),
    scale: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    """Fill a fixed-size chunk; padding is an exact index-matched zero case."""
    i = wp.tid()
    zero = wp.float64(0.0)
    index[i] = wp.vec2d(wp.float64(1.0), zero)
    size[i] = zero
    radius[i] = zero
    weight[i] = zero
    scale[i] = zero
    status[i] = 0
    if i >= valid_count:
        return
    r = radii[group, start + i]
    w = weights[group, start + i]
    x = sizes[group, start + i]
    if (
        not wp.isfinite(r)
        or r < zero
        or not wp.isfinite(w)
        or w < zero
        or not wp.isfinite(x)
        or x < zero
        or (r > zero and x == zero)
        or (r == zero and x > zero)
    ):
        status[i] = 1
        return
    index[i] = indices[group, start + i]
    size[i] = x
    radius[i] = r
    weight[i] = w
    if x > zero:
        scale[i] = (r / x) * (r / x)


@wp.kernel
def clear_bulk(
    totals: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    matrices: wp.array3d(dtype=wp.float64),
):
    """Reset accumulated state once at the beginning of each evaluation."""
    group = wp.tid()
    status[group] = 0
    for k in range(9):
        totals[group, k] = wp.float64(0.0)
    for angle in range(matrices.shape[1]):
        for element in range(16):
            matrices[group, angle, element] = wp.float64(0.0)


@wp.kernel
def accumulate_scalar(
    efficiencies: wp.array2d(dtype=wp.float64),
    radius: wp.array(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    gather_status: wp.array(dtype=wp.int32),
    optical_status: wp.array(dtype=wp.int32),
    group: int,
    totals: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    """Accumulate cross sections and scattering-weighted g in fixed order."""
    if status[group] != 0:
        return
    for i in range(radius.shape[0]):
        if gather_status[i] != 0 or optical_status[i] != 0:
            status[group] = 2
            return
        factor = _PI * radius[i] * radius[i] * weights[i]
        for k in range(3):
            totals[group, k] += efficiencies[i, k] * factor
        totals[group, 3] += efficiencies[i, 4] * factor
        totals[group, 4] += efficiencies[i, 5] * factor
        totals[group, 6] += efficiencies[i, 1] * efficiencies[i, 3] * factor


@wp.kernel
def accumulate_angular(
    matrices: wp.array4d(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    input_status: wp.array2d(dtype=wp.int32),
    group: int,
    totals: wp.array3d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    """Accumulate scaled chunk matrices without full-population storage."""
    angle, element = wp.tid()  # type: ignore[misc]
    if status[group] != 0:
        return
    value = wp.float64(0.0)
    for i in range(weights.shape[0]):
        if input_status[i, angle] != 0:
            wp.atomic_max(status, group, 2)
            return
        value += matrices[i, angle, element // 4, element % 4] * weights[i]
    totals[group, angle, element] += value
    if not wp.isfinite(totals[group, angle, element]):
        wp.atomic_max(status, group, 3)


@wp.kernel
def finalize_bulk(
    totals: wp.array2d(dtype=wp.float64), status: wp.array(dtype=wp.int32)
):
    """Convert accumulated moments to SSA/g and validate finite results."""
    group = wp.tid()
    if status[group] != 0:
        return
    zero = wp.float64(0.0)
    if totals[group, 0] > zero:
        totals[group, 5] = totals[group, 1] / totals[group, 0]
        totals[group, 7] = wp.float64(1.0)
    if totals[group, 1] > zero:
        totals[group, 6] /= totals[group, 1]
        totals[group, 8] = wp.float64(1.0)
    else:
        totals[group, 6] = zero
    for k in range(9):
        if not wp.isfinite(totals[group, k]):
            status[group] = 3
