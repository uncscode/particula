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
    Keys = Any
    Index3 = Any
    Position3 = Any
    F = Any
else:
    IntMatrix = wp.array2d(dtype=wp.int32)
    Keys = wp.array(dtype=wp.uint64)
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


# Typed constants prevent Warp from narrowing Python literals in device code.
_MIX_FIRST = wp.constant(wp.uint64(0xBF58476D1CE4E5B9))
_MIX_SECOND = wp.constant(wp.uint64(0x94D049BB133111EB))
_TWO_PI = wp.constant(wp.float64(2.0 * np.pi))


@wp.func
def _mix64(x: wp.uint64):
    # SplitMix64 finalizer; unsigned arithmetic wraps on both devices.
    x = (wp.bit_xor(x, wp.rshift(x, wp.uint64(30)))) * _MIX_FIRST
    x = (wp.bit_xor(x, wp.rshift(x, wp.uint64(27)))) * _MIX_SECOND
    return wp.bit_xor(x, wp.rshift(x, wp.uint64(31)))


@wp.func
def _uniform_draw(key: wp.uint64, interval: int, step: int, lane: int):
    counter = wp.bit_or(
        wp.lshift(wp.uint64(interval), wp.uint64(32)), wp.uint64(step)
    )
    bits = _mix64(_mix64(wp.bit_xor(key, counter)) + wp.uint64(lane))
    # Midpoints of equal 52-bit bins are strictly inside (0,1).
    return (F(wp.rshift(bits, wp.uint64(12))) + F(0.5)) * F(
        2.220446049250313e-16
    )


@wp.func
def _normal_draw(key: wp.uint64, interval: int, step: int, axis: int):
    u = _uniform_draw(key, interval, step, 2 * axis)
    v = _uniform_draw(key, interval, step, 2 * axis + 1)
    return wp.sqrt(F(-2.0) * wp.log(u)) * wp.cos(_TWO_PI * v)


@wp.kernel
def move_particles(  # noqa: C901 - single-device face traversal
    interval: int,
    step: int,
    keys: Keys,
    diffusivity: Position3,
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
    for a in range(3):
        if diffusivity[a] > F(0.0):
            vel[a] += wp.sqrt(F(2.0) * diffusivity[a] / dt) * _normal_draw(
                keys[p], interval, step, a
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
                if diffusivity[axis] > F(0.0):
                    vel[axis] = -vel[axis]
                else:
                    vel[axis] = F(0.0)
            else:
                status[p] = mode
                event_time[p] = start + (dt - remaining)
                break
    cell[p] = (idx[0] * shape[1] + idx[1]) * shape[2] + idx[2]
    for a in range(3):
        pos[p, a] = point[a]


@wp.kernel
def diffuse_cells(
    interval: int,
    step: int,
    end: F,
    keys: Keys,
    fraction: FloatMatrix,
    neighbors: IntMatrix,
    shape: Index3,
    lo: Position3,
    width: Position3,
    jac_edge: FloatVector,
    bc: IntVector,
    pos: FloatMatrix,
    cells: IntVector,
    status: IntVector,
    event_time: FloatVector,
):
    """Sample conservative face exchanges on variable-K or mapped cells."""
    p = wp.tid()
    if status[p] != 1:
        return
    old = cells[p]
    draw = _uniform_draw(keys[p], interval, step, 6)
    for side in range(6):
        draw -= fraction[old, side]
        if draw < F(0.0):
            axis = side // 2
            stride = int(1)
            if axis == 0:
                stride = shape[1] * shape[2]
            elif axis == 1:
                stride = shape[2]
            index = (old // stride) % shape[axis]
            dest = neighbors[old, side]
            if dest < 0:
                pos[p, axis] = lo[axis] + F(index + side % 2) * width[axis]
                status[p] = bc[side]
                event_time[p] = end
            else:
                new_index = (dest // stride) % shape[axis]
                location = (pos[p, axis] - lo[axis]) / width[axis] - F(index)
                if axis == 0:
                    location = wp.clamp(location, F(0.0), F(1.0))
                    # Carry the physical-volume quantile across the sloped cell.
                    a = jac_edge[index]
                    b = jac_edge[index + 1]
                    area_rank = (
                        location * (F(2.0) * a + (b - a) * location) / (a + b)
                    )
                    c = jac_edge[new_index]
                    d = jac_edge[new_index + 1]
                    location = (
                        area_rank
                        * (c + d)
                        / (
                            c
                            + wp.sqrt(
                                c * c * (F(1.0) - area_rank) + d * d * area_rank
                            )
                        )
                    )
                pos[p, axis] = (
                    lo[axis] + (F(new_index) + location) * width[axis]
                )
                cells[p] = dest
            break


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
