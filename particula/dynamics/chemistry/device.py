"""Double-precision resident mass-action RK23 and ROS3 box integrators."""

# Warp struct fields use runtime array constructors as annotations.
# mypy: disable-error-code="valid-type"
from typing import no_type_check

import warp as wp

from particula.execution._ros3_coefficients import (
    C21,
    C31,
    C32,
    E2,
    E3,
    GAMMA,
    M2,
    M3,
)
from particula.util.constants import GAS_CONSTANT


@wp.struct
class DeviceMechanism:
    """Immutable resident configuration for all boxes."""

    lanes: wp.array(dtype=int)
    masses: wp.array(dtype=wp.float64)
    active: wp.array(dtype=int)
    net: wp.array2d(dtype=int)
    orders: wp.array2d(dtype=int)
    coefficients: wp.array2d(dtype=wp.float64)
    atol: wp.array(dtype=wp.float64)
    rtol: wp.float64
    first_step: wp.float64
    min_step: wp.float64
    max_step: wp.float64
    trials: int
    stiff: int


@wp.struct
class Work:
    """Fixed scratch indexed by box, plus interval work counters."""

    vectors: wp.array3d(dtype=wp.float64)
    matrix: wp.array3d(dtype=wp.float64)
    lu: wp.array3d(dtype=wp.float64)
    pivots: wp.array2d(dtype=int)
    rates: wp.array2d(dtype=wp.float64)
    counts: wp.array2d(dtype=wp.int64)


# Named vector slots. Each thread owns one independent box.
Y = wp.constant(0)
STAGE = wp.constant(1)
K1 = wp.constant(2)
K2 = wp.constant(3)
K3 = wp.constant(4)
K4 = wp.constant(5)
ERROR = wp.constant(6)
RHS = wp.constant(7)
DELTA = wp.constant(8)
MOLAR_GAS_CONSTANT = wp.constant(wp.float64(GAS_CONSTANT))
SAFETY = wp.constant(wp.float64(0.9))
POWER = wp.constant(wp.float64(-1.0 / 3.0))


@wp.func
def _product(
    m: DeviceMechanism,
    w: Work,
    box: int,
    slot: int,
    reaction: int,
    derivative: int,
):
    """Evaluate a molar rate or a partial without division by concentration."""
    value = w.rates[box, reaction]
    for j in range(m.lanes.shape[0]):
        order = m.orders[reaction, j]
        if j == derivative:
            if order == 0:
                return wp.float64(0)
            value *= wp.float64(order)
            order -= 1
        if order > 0:
            value *= wp.pow(w.vectors[box, slot, j], wp.float64(order))
    return value


@wp.func
def _rate(m: DeviceMechanism, w: Work, box: int, slot: int, target: int):
    """Sum all molar reaction extents into one species tendency."""
    for i in range(m.lanes.shape[0]):
        x = w.vectors[box, slot, i]
        if not wp.isfinite(x) or x < 0:
            return False
        w.vectors[box, target, i] = wp.float64(0)
    for r in range(m.net.shape[0]):
        flux = _product(m, w, box, slot, r, -1)
        if not wp.isfinite(flux):
            return False
        for i in range(m.lanes.shape[0]):
            w.vectors[box, target, i] += flux * wp.float64(m.net[r, i])
    return True


@wp.func
def _rk23(m: DeviceMechanism, w: Work, b: int, h: wp.float64):
    """Bogacki–Shampine embedded 3(2) trial, matching the CPU tableau."""
    if not _rate(m, w, b, Y, K1):
        return False
    n = m.lanes.shape[0]
    for i in range(n):
        w.vectors[b, STAGE, i] = (
            w.vectors[b, Y, i] + h * wp.float64(0.5) * w.vectors[b, K1, i]
        )
    if not _rate(m, w, b, STAGE, K2):
        return False
    for i in range(n):
        w.vectors[b, STAGE, i] = (
            w.vectors[b, Y, i] + h * wp.float64(0.75) * w.vectors[b, K2, i]
        )
    if not _rate(m, w, b, STAGE, K3):
        return False
    for i in range(n):
        w.vectors[b, STAGE, i] = w.vectors[b, Y, i] + h * (
            wp.float64(2.0) * w.vectors[b, K1, i] / wp.float64(9.0)
            + w.vectors[b, K2, i] / wp.float64(3.0)
            + wp.float64(4.0) * w.vectors[b, K3, i] / wp.float64(9.0)
        )
    if not _rate(m, w, b, STAGE, K4):
        return False
    for i in range(n):
        w.vectors[b, ERROR, i] = h * (
            -wp.float64(5.0) * w.vectors[b, K1, i] / wp.float64(72.0)
            + w.vectors[b, K2, i] / wp.float64(12.0)
            + w.vectors[b, K3, i] / wp.float64(9.0)
            - w.vectors[b, K4, i] / wp.float64(8.0)
        )
    return True


@wp.func
def _factor(m: DeviceMechanism, w: Work, b: int, h: wp.float64):  # noqa: C901
    """Form I/(gamma*h) - J and factor it with partial pivoting."""
    n = m.lanes.shape[0]
    for j in range(n):
        for i in range(n):
            w.matrix[b, i, j] = wp.float64(0)
        w.matrix[b, j, j] = wp.float64(1) / (GAMMA * h)
        for r in range(m.net.shape[0]):
            partial = _product(m, w, b, Y, r, j)
            for i in range(n):
                w.matrix[b, i, j] -= partial * wp.float64(m.net[r, i])
        for i in range(n):
            w.lu[b, i, j] = w.matrix[b, i, j]
    for k in range(n):
        pivot = int(k)
        for i in range(k + 1, n):
            if wp.abs(w.lu[b, i, k]) > wp.abs(w.lu[b, pivot, k]):
                pivot = i
        p = w.lu[b, pivot, k]
        if not wp.isfinite(p) or p == 0:
            return False
        w.pivots[b, k] = pivot
        for j in range(n):
            tmp = w.lu[b, k, j]
            w.lu[b, k, j] = w.lu[b, pivot, j]
            w.lu[b, pivot, j] = tmp
        for i in range(k + 1, n):
            ratio = w.lu[b, i, k] / p
            w.lu[b, i, k] = ratio
            for j in range(k + 1, n):
                w.lu[b, i, j] -= ratio * w.lu[b, k, j]
    return True


@wp.func
def _solve(m: DeviceMechanism, w: Work, b: int, target: int):  # noqa: C901
    """Solve a stage and perform the CPU method's two residual corrections."""
    n = m.lanes.shape[0]
    for i in range(n):
        w.vectors[b, target, i] = wp.float64(0)
    for correction in range(3):
        for i in range(n):
            residual = w.vectors[b, RHS, i]
            if correction > 0:
                for j in range(n):
                    residual -= w.matrix[b, i, j] * w.vectors[b, target, j]
            w.vectors[b, DELTA, i] = residual
        for i in range(n):
            pivot = w.pivots[b, i]
            tmp = w.vectors[b, DELTA, i]
            w.vectors[b, DELTA, i] = w.vectors[b, DELTA, pivot]
            w.vectors[b, DELTA, pivot] = tmp
        for i in range(n):
            for j in range(i):
                w.vectors[b, DELTA, i] -= w.lu[b, i, j] * w.vectors[b, DELTA, j]
        for reverse in range(n):
            i = n - 1 - reverse
            for j in range(i + 1, n):
                w.vectors[b, DELTA, i] -= w.lu[b, i, j] * w.vectors[b, DELTA, j]
            w.vectors[b, DELTA, i] = w.vectors[b, DELTA, i] / w.lu[b, i, i]
            w.vectors[b, target, i] += w.vectors[b, DELTA, i]
        w.counts[b, 3] += wp.int64(1)


@wp.func
def _ros3(m: DeviceMechanism, w: Work, b: int, h: wp.float64):
    """Autonomous ROS3 step at fixed temperature, sharing one factorization."""
    if not _rate(m, w, b, Y, RHS) or not _factor(m, w, b, h):
        return False
    w.counts[b, 2] += wp.int64(1)
    _solve(m, w, b, K1)
    n = m.lanes.shape[0]
    for i in range(n):
        w.vectors[b, STAGE, i] = w.vectors[b, Y, i] + w.vectors[b, K1, i]
    if not _rate(m, w, b, STAGE, K4):
        return False
    for i in range(n):
        w.vectors[b, RHS, i] = w.vectors[b, K4, i] - (
            C21 * w.vectors[b, K1, i] / h
        )
    _solve(m, w, b, K2)
    for i in range(n):
        w.vectors[b, RHS, i] = (
            w.vectors[b, K4, i]
            + (C31 * w.vectors[b, K1, i] + C32 * w.vectors[b, K2, i]) / h
        )
    _solve(m, w, b, K3)
    for i in range(n):
        w.vectors[b, STAGE, i] = (
            w.vectors[b, Y, i]
            + w.vectors[b, K1, i]
            + M2 * w.vectors[b, K2, i]
            - M3 * w.vectors[b, K3, i]
        )
        w.vectors[b, ERROR, i] = (
            wp.float64(0.5) * w.vectors[b, K1, i]
            - E2 * w.vectors[b, K2, i]
            + E3 * w.vectors[b, K3, i]
        )
    return True


@no_type_check
@wp.kernel
def integrate_boxes(  # noqa: C901
    m: DeviceMechanism,
    w: Work,
    gas: wp.array2d(dtype=wp.float64),
    temperature: wp.array(dtype=wp.float64),
    dt: wp.float64,
    failure: wp.array(dtype=int),
):
    """Stage each box independently and signal a failed interval on device."""
    b = wp.tid()
    if failure[0] != 0:
        return
    temp = temperature[b]
    for r in range(m.net.shape[0]):
        k = (
            m.coefficients[r, 0]
            * wp.pow(temp / m.coefficients[r, 3], m.coefficients[r, 1])
            * wp.exp(-m.coefficients[r, 2] / (MOLAR_GAS_CONSTANT * temp))
        )
        if not wp.isfinite(k):
            wp.atomic_max(failure, 0, 101)
            return
        w.rates[b, r] = k
    for i in range(m.lanes.shape[0]):
        w.vectors[b, Y, i] = gas[b, m.lanes[i]] / m.masses[i]
    elapsed = wp.float64(0)
    step = m.first_step
    for _trial in range(m.trials):
        remaining = dt - elapsed
        h = wp.min(step, remaining)
        if h < wp.min(m.min_step, remaining) or elapsed + h == elapsed:
            break
        valid = bool(False)
        if m.stiff == 1:
            valid = _ros3(m, w, b, h)
        else:
            valid = _rk23(m, w, b, h)
        norm = wp.float64(0)
        for i in range(m.lanes.shape[0]):
            y = w.vectors[b, STAGE, i]
            error = w.vectors[b, ERROR, i]
            if not wp.isfinite(y) or y < 0 or not wp.isfinite(error):
                valid = False
            scale = m.atol[i] + m.rtol * wp.max(
                wp.abs(y), wp.abs(w.vectors[b, Y, i])
            )
            norm = wp.max(norm, wp.abs(error) / scale)
        if not valid or not wp.isfinite(norm):
            step = h * wp.float64(0.5)
            w.counts[b, 1] += wp.int64(1)
        elif norm > 1:
            step = h * wp.clamp(
                SAFETY * wp.pow(norm, POWER),
                wp.float64(0.1),
                wp.float64(0.5),
            )
            w.counts[b, 1] += wp.int64(1)
        else:
            for i in range(m.lanes.shape[0]):
                w.vectors[b, Y, i] = w.vectors[b, STAGE, i]
            w.counts[b, 0] += wp.int64(1)
            if h == remaining:
                return
            elapsed += h
            factor = wp.float64(5)
            if norm > 0:
                factor = wp.clamp(
                    SAFETY * wp.pow(norm, POWER),
                    wp.float64(0.2),
                    wp.float64(5),
                )
            step = wp.clamp(h * factor, m.min_step, m.max_step)
    wp.atomic_max(failure, 0, 100)


@no_type_check
@wp.kernel
def publish_boxes(
    m: DeviceMechanism,
    w: Work,
    gas: wp.array2d(dtype=wp.float64),
    failure: wp.array(dtype=int),
):
    """Publish only after every chemistry box reached the interval boundary."""
    b, i = wp.tid()
    if failure[0] == 0 and m.active[i] != 0:
        initial = gas[b, m.lanes[i]] / m.masses[i]
        if initial != w.vectors[b, Y, i]:
            gas[b, m.lanes[i]] = w.vectors[b, Y, i] * m.masses[i]
