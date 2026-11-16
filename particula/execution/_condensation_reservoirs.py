"""Deterministic parallel reservoir reductions with bounded device scratch."""

import warp as wp

from particula.execution._condensation_ode_kernels import (
    FloatMatrix,
    FloatTensor,
    FloatVector,
    IntVector,
)
from particula.execution._condensation_ros3_kernels import GAMMA


@wp.kernel
def gas_parts(
    rate: FloatTensor,
    counts: FloatMatrix,
    volume: FloatVector,
    mapping: IntVector,
    resolved: int,
    parts: FloatTensor,
):
    """Retain native slot weights and reduce independent fixed slot groups."""
    box, gas, lane = wp.tid()  # type: ignore[misc]
    material = mapping[gas]
    value = wp.float64(0.0)
    if material >= 0:
        for slot in range(lane, rate.shape[1], parts.shape[2]):
            weight = counts[box, slot]
            if resolved == 1:
                weight /= volume[box]
            value -= weight * rate[box, slot, material]
    parts[box, gas, lane] = value


@wp.kernel
def finish_gas(
    parts: FloatTensor,
    mass_size: int,
    scale: FloatVector,
    derivative: FloatVector,
    invalid: IntVector,
):
    """Finish in fixed lane order and write scaled gas derivatives."""
    box, gas = wp.tid()  # type: ignore[misc]
    value = wp.float64(0.0)
    for lane in range(parts.shape[2]):
        value += parts[box, gas, lane]
    index = mass_size + box * parts.shape[1] + gas
    value /= scale[index]
    derivative[index] = value
    if not wp.isfinite(value):
        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def assemble_parts(
    local: FloatTensor,
    response: FloatTensor,
    particular: FloatMatrix,
    counts: FloatMatrix,
    volume: FloatVector,
    scale: FloatVector,
    rhs: FloatVector,
    targets: IntVector,
    sources: IntVector,
    materials: int,
    gases: int,
    resolved: int,
    h: FloatVector,
    parts: FloatTensor,
):
    """Sum fixed strided slot groups without atomic floating-point additions."""
    box, i, index = wp.tid()  # type: ignore[misc]
    lanes = parts.shape[2]
    column, lane = index // lanes, index % lanes
    n = targets.shape[0]
    mass_size = counts.shape[0] * counts.shape[1] * materials
    gi = mass_size + box * gases + sources[i]
    value = wp.float64(0.0)
    for slot in range(lane, counts.shape[1], lanes):
        p = box * counts.shape[1] + slot
        weight = (
            counts[box, slot] * scale[p * materials + targets[i]] / scale[gi]
        )
        if resolved == 1:
            weight /= volume[box]
        if column == n:
            product = wp.float64(0.0)
            for j in range(n):
                product += local[p, i, j] * particular[p, j]
            value -= weight * product
        else:
            value += weight * response[p, i, column]
    parts[box * n + i, column, lane] = value


@wp.kernel
def finish_schur(
    parts: FloatTensor,
    rhs: FloatVector,
    sources: IntVector,
    mass_size: int,
    gases: int,
    h: FloatVector,
    schur: FloatTensor,
):
    """Add the diagonal/RHS once after deterministic partial reduction."""
    box, row, column = wp.tid()  # type: ignore[misc]
    n = sources.shape[0]
    value = wp.float64(0.0)
    for lane in range(parts.shape[2]):
        value += parts[box * n + row, column, lane]
    if column == n:
        index = mass_size + box * gases + sources[row]
        q = wp.float64(1.0) / (h[0] * GAMMA)
        value = (rhs[index] + value) / q
    elif column == row:
        value += wp.float64(1.0)
    schur[box, row, column] = value
