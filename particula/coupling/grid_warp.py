"""Float64 Warp operations for prescribed field sampling and transport."""

from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from particula.execution._condensation_ode_kernels import (
    FloatMatrix,
    FloatTensor,
    FloatVector,
    IntVector,
)

if TYPE_CHECKING:
    IntMatrix = Any
    Index3 = Any
    Position3 = Any
    F = Any
else:
    IntMatrix = wp.array2d(dtype=wp.int32)
    Index3 = wp.vec3i
    Position3 = wp.vec3d
    F = wp.float64


def _array(values, device, dtype=F):
    return wp.array(np.ascontiguousarray(values), dtype=dtype, device=device)


@wp.kernel
def _uniform(row: FloatVector, out: FloatMatrix):
    i, j = wp.tid()  # type: ignore[misc]  # two-dimensional launch
    out[i, j] = row[j]


def uniform_sample(n, row, device):
    """Sample uniform forcing on the selected device."""
    out = wp.empty((n, 6), dtype=F, device=device)
    wp.launch(_uniform, (n, 6), [_array(row, device), out], device=device)
    return out.numpy()


@wp.func
def _interp(table: FloatMatrix, z: F, col: int):
    hi = table.shape[0] - 1
    lo = int(0)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if table[mid, 0] <= z:
            lo = mid
        else:
            hi = mid
    f = wp.clamp(
        (z - table[lo, 0]) / (table[hi, 0] - table[lo, 0]),
        F(0.0),
        F(1.0),
    )
    return table[lo, col] + f * (table[hi, col] - table[lo, col])


@wp.kernel
def _profile(
    table: FloatMatrix,
    query: FloatMatrix,
    out: FloatMatrix,
):
    i = wp.tid()
    u = _interp(table, query[i, 1], 1) / query[i, 2]
    out[i, 0] = u
    out[i, 1] = _interp(table, query[i, 1], 2)
    out[i, 2] = u * query[i, 3]
    for j in range(3, 6):
        out[i, j] = _interp(table, query[i, 0], j)


def profile_sample(s, z, eta, jac, gradient, device):
    """Interpolate wind and thermodynamics using the explicit device."""
    table = np.column_stack(
        (s.height, s.east, s.north, s.temperature, s.pressure, s.vapor)
    )
    query = np.column_stack((z, eta, jac, gradient))
    out = wp.empty((len(z), 6), dtype=F, device=device)
    wp.launch(
        _profile,
        len(z),
        [_array(table, device), _array(query, device), out],
        device=device,
    )
    return out.numpy()


@wp.kernel
def transport(
    nb: IntMatrix,
    incoming: FloatMatrix,
    outgoing: FloatMatrix,
    volume: FloatVector,
    inlet: FloatTensor,
    old: FloatMatrix,
    updated: FloatMatrix,
    imports: FloatMatrix,
    exports: FloatMatrix,
):
    """Gather one conservative substep per cell and inventory lane."""
    i, j = wp.tid()  # type: ignore[misc]  # two-dimensional launch
    retain = F(1.0)
    gain = F(0.0)
    for side in range(6):
        neighbor = nb[i, side]
        frac = outgoing[i, side] / volume[i]
        retain -= frac
        if neighbor >= 0:
            gain += incoming[i, side] * (old[neighbor, j] / volume[neighbor])
        else:
            added = incoming[i, side] * inlet[i, side, j]
            gain += added
            imports[i, j] += added
            exports[i, j] += frac * old[i, j]
    updated[i, j] = old[i, j] * retain + gain


@wp.kernel
def uptake(
    state: FloatMatrix,
    gas: int,
    number: int,
    material: int,
    fraction: F,
):
    """Transfer gas once into an occupied particle bin."""
    i = wp.tid()
    if state[i, number] > F(0.0):
        amount = fraction * state[i, gas]
        state[i, gas] -= amount
        state[i, material] += amount


@wp.kernel
def move_particles(  # noqa: C901 - single-device face traversal
    interval: int,
    start: F,
    dt: F,
    shape: Index3,
    lo: Position3,
    width: Position3,
    bc: IntVector,
    jac: FloatVector,
    speed: FloatMatrix,
    slip: FloatVector,
    release: IntVector,
    pos: FloatMatrix,
    cell: IntVector,
    status: IntVector,
    event_time: FloatVector,
):
    """Advance each stable record through mapped cell faces."""
    p = wp.tid()
    if status[p] == 0 and release[p] == interval:
        status[p] = 1
    if status[p] != 1:
        return
    old = cell[p]
    idx = Index3(
        old // (shape[1] * shape[2]),
        (old // shape[2]) % shape[1],
        old % shape[2],
    )
    vel = Position3(
        speed[old, 0], speed[old, 1], speed[old, 2] - slip[p] / jac[old]
    )
    point = Position3(pos[p, 0], pos[p, 1], pos[p, 2])
    # Frozen cell velocity for a bounded substep. Traverse every crossed
    # face in reference coordinates; the physical bottom is the mapped bed.
    remaining = dt
    while remaining > F(0.0):
        axis = int(-1)
        sign = int(0)
        travel = remaining
        for a in range(3):
            v = vel[a]
            if v != F(0.0):
                direction = int(1)
                edge = idx[a] + 1
                if v < F(0.0):
                    edge = idx[a]
                    direction = -1
                lag = wp.max(
                    F(0.0),
                    (lo[a] + F(edge) * width[a] - point[a]) / v,
                )
                if lag <= travel:
                    travel = lag
                    sign = direction
                    axis = a
        point += vel * travel
        remaining -= travel
        if axis < 0:
            break
        edge_index = idx[axis]
        if sign > 0:
            edge_index += 1
        point[axis] = lo[axis] + F(edge_index) * width[axis]
        dest = idx[axis] + sign
        if dest >= 0 and dest < shape[axis]:
            idx[axis] = dest
        else:
            side = 2 * axis
            if sign > 0:
                side += 1
            mode = bc[side]
            if mode == 1:
                idx[axis] = (dest + shape[axis]) % shape[axis]
                point[axis] -= F(sign * shape[axis]) * width[axis]
            elif mode == 0:
                vel[axis] = F(0.0)
            else:
                status[p] = mode
                event_time[p] = start + (dt - remaining)
                break
    cell[p] = (idx[0] * shape[1] + idx[1]) * shape[2] + idx[2]
    for a in range(3):
        pos[p, a] = point[a]


@wp.kernel
def field_totals(state: FloatMatrix, total: FloatVector):
    """Reduce extensive fields at an observation boundary."""
    i, j = wp.tid()  # type: ignore[misc]  # two-dimensional launch
    wp.atomic_add(total, j, state[i, j])


@wp.kernel
def particle_totals(
    state: IntVector,
    inventory: FloatMatrix,
    total: FloatMatrix,
):
    """Reduce particle payloads by transport fate."""
    i, j = wp.tid()  # type: ignore[misc]  # two-dimensional launch
    wp.atomic_add(total, state[i], j, inventory[i, j])


@wp.kernel
def accumulate(ledger: FloatMatrix, increment: FloatMatrix):
    """Accumulate host-reported boundary totals on device."""
    i, j = wp.tid()  # type: ignore[misc]  # two-dimensional launch
    ledger[i, j] += increment[i, j]
