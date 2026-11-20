"""Resident RK23 for resolved gas and particle material balance."""

# Warp array descriptors use runtime dimensionality in annotations.
# mypy: disable-error-code="valid-type"
from typing import no_type_check

import warp as wp

from particula.util.constants import GAS_CONSTANT

R = wp.constant(wp.float64(GAS_CONSTANT))
PI = wp.constant(wp.float64(3.141592653589793))
THIRD = wp.constant(wp.float64(1 / 3))
NEGATIVE_THIRD = wp.constant(wp.float64(-1 / 3))
SAFETY = wp.constant(wp.float64(0.9))


@wp.struct
class Network:
    """Immutable reaction and solver configuration shared by boxes."""

    gm: wp.array(dtype=wp.float64)
    pm: wp.array(dtype=wp.float64)
    density: wp.array(dtype=wp.float64)
    offsets: wp.array(dtype=int)
    columns: wp.array(dtype=int)
    coefficients_net: wp.array(dtype=int)
    orders: wp.array2d(dtype=int)
    params: wp.array2d(dtype=wp.float64)
    kinds: wp.array(dtype=int)
    sources: wp.array(dtype=int)
    atol: wp.float64
    rtol: wp.float64
    first: wp.float64
    minimum: wp.float64
    maximum: wp.float64
    limit: int


@wp.struct
class Stages:
    """Work vectors hold accepted state, trial state and four derivatives."""

    v: wp.array3d(dtype=wp.float64)
    reference: wp.array2d(dtype=wp.float64)
    weights: wp.array2d(dtype=wp.float64)
    rates: wp.array2d(dtype=wp.float64)
    counts: wp.array2d(dtype=wp.int64)


@wp.func
def _evaluate(m: Network, w: Stages, b: int, target: int):  # noqa: C901
    ng = m.gm.shape[0]
    ns = m.pm.shape[0]
    for i in range(w.v.shape[2]):
        x = w.v[b, 1, i]
        if not wp.isfinite(x) or x < 0:
            return False
        w.v[b, target, i] = wp.float64(0)
    for r in range(m.kinds.shape[0]):
        if m.kinds[r] == 0:
            flux = w.rates[b, r]
            for i in range(ng):
                if m.orders[r, i] > 0:
                    flux *= wp.pow(w.v[b, 1, i], wp.float64(m.orders[r, i]))
            for term in range(m.offsets[r], m.offsets[r + 1]):
                w.v[b, target, m.columns[term]] += (
                    wp.float64(m.coefficients_net[term]) * flux
                )
    for slot in range(w.reference.shape[1]):
        ref = w.reference[b, slot]
        if ref > 0:
            start = ng + slot * ns
            ratio = wp.float64(0)
            for j in range(ns):
                ratio += w.v[b, 1, start + j] * m.pm[j] / m.density[j]
            if not wp.isfinite(ratio) or ratio <= wp.float64(0):  # type: ignore[operator]
                return False
            radius = wp.pow(
                wp.float64(3) / (wp.float64(4) * PI) * ref * ratio, THIRD
            )
            area = wp.float64(4) * PI * radius * radius
            for r in range(m.kinds.shape[0]):
                kind = m.kinds[r]
                flux = wp.float64(0)
                if kind == 1:
                    flux = w.rates[b, r] * ratio
                    for j in range(ns):
                        order = m.orders[r, ng + j]
                        if order > 0:
                            flux *= wp.pow(
                                w.v[b, 1, start + j] / ratio, wp.float64(order)
                            )
                elif kind == 2:
                    source = m.sources[r]
                    extent = w.rates[b, r] * area * w.v[b, 1, source]
                    w.v[b, target, source] -= w.weights[b, slot] * extent
                    flux = extent / ref
                if kind != 0:
                    for term in range(m.offsets[r], m.offsets[r + 1]):
                        j = m.columns[term] - ng
                        if j >= 0:
                            w.v[b, target, start + j] += (
                                wp.float64(m.coefficients_net[term]) * flux
                            )
    return True


@wp.func
def _trial(m: Network, w: Stages, b: int, h: wp.float64):
    """Bogacki–Shampine stages with a common step for both reservoirs."""
    for i in range(w.v.shape[2]):
        w.v[b, 1, i] = w.v[b, 0, i]
    if not _evaluate(m, w, b, 2):
        return False
    for i in range(w.v.shape[2]):
        w.v[b, 1, i] = w.v[b, 0, i] + h * wp.float64(0.5) * w.v[b, 2, i]
    if not _evaluate(m, w, b, 3):
        return False
    for i in range(w.v.shape[2]):
        w.v[b, 1, i] = w.v[b, 0, i] + h * wp.float64(0.75) * w.v[b, 3, i]
    if not _evaluate(m, w, b, 4):
        return False
    for i in range(w.v.shape[2]):
        w.v[b, 1, i] = w.v[b, 0, i] + h * (
            wp.float64(2) * w.v[b, 2, i] / wp.float64(9)
            + w.v[b, 3, i] / wp.float64(3)
            + wp.float64(4) * w.v[b, 4, i] / wp.float64(9)
        )
    return _evaluate(m, w, b, 5)


@wp.func
def _integrate(m: Network, w: Stages, b: int, dt: wp.float64):  # noqa: C901
    elapsed = wp.float64(0)
    step = m.first
    for _attempt in range(m.limit):
        remaining = dt - elapsed
        h = wp.min(step, remaining)
        if h < wp.min(m.minimum, remaining) or elapsed + h == elapsed:
            return False
        valid = _trial(m, w, b, h)
        norm = wp.float64(0)
        if valid:
            for i in range(w.v.shape[2]):
                error = h * (
                    -wp.float64(5) * w.v[b, 2, i] / wp.float64(72)
                    + w.v[b, 3, i] / wp.float64(12)
                    + w.v[b, 4, i] / wp.float64(9)
                    - w.v[b, 5, i] / wp.float64(8)
                )
                scale = m.atol + m.rtol * wp.max(w.v[b, 0, i], w.v[b, 1, i])
                if not wp.isfinite(error):
                    valid = False
                norm = wp.max(norm, wp.abs(error) / scale)
        if not valid or norm > wp.float64(1):  # type: ignore[operator]
            w.counts[b, 1] += wp.int64(1)
            step = h * wp.float64(0.5)
            if valid:
                step = h * wp.clamp(
                    SAFETY * wp.pow(norm, NEGATIVE_THIRD),
                    wp.float64(0.1),
                    wp.float64(0.5),
                )
        else:
            for i in range(w.v.shape[2]):
                w.v[b, 0, i] = w.v[b, 1, i]
            w.counts[b, 0] += wp.int64(1)
            if h == remaining:
                return True
            elapsed += h
            factor = wp.float64(5)
            if norm > wp.float64(0):  # type: ignore[operator]
                factor = wp.clamp(
                    SAFETY * wp.pow(norm, NEGATIVE_THIRD),
                    wp.float64(0.2),
                    wp.float64(5),
                )
            step = wp.clamp(h * factor, m.minimum, m.maximum)
    return False


@no_type_check
@wp.kernel
def stage(  # noqa: C901
    m: Network,
    w: Stages,
    gas: wp.array2d(dtype=wp.float64),
    particles: wp.array3d(dtype=wp.float64),
    counts: wp.array2d(dtype=wp.float64),
    volume: wp.array(dtype=wp.float64),
    temp: wp.array(dtype=wp.float64),
    dt: wp.float64,
    failure: wp.array(dtype=int),
):
    """Advance one thread per box into unpublished scratch."""
    b = wp.tid()
    if failure[0] != 0:
        return
    ng = m.gm.shape[0]
    ns = m.pm.shape[0]
    for r in range(m.kinds.shape[0]):
        k = m.params[r, 0]
        if m.kinds[r] == 2:
            k *= wp.sqrt(
                wp.float64(8) * R * temp[b] / (PI * m.gm[m.sources[r]])
            ) / wp.float64(4)
        else:
            k *= wp.pow(temp[b] / m.params[r, 3], m.params[r, 1]) * wp.exp(
                -m.params[r, 2] / (R * temp[b])
            )
        if not wp.isfinite(k):
            wp.atomic_max(failure, 0, 111)
            return
        w.rates[b, r] = k
    for i in range(ng):
        w.v[b, 0, i] = gas[b, i] / m.gm[i]
    for slot in range(particles.shape[1]):
        ref = wp.float64(0)
        for j in range(ns):
            ref += particles[b, slot, j] / m.density[j]
        if (ref > 0) != (counts[b, slot] > 0) or not wp.isfinite(ref):
            wp.atomic_max(failure, 0, 111)
            return
        w.reference[b, slot] = ref
        w.weights[b, slot] = counts[b, slot] / volume[b]
        for j in range(ns):
            value = wp.float64(0)
            if ref > 0:
                value = particles[b, slot, j] / m.pm[j] / ref
            w.v[b, 0, ng + slot * ns + j] = value
    if not _integrate(m, w, b, dt):
        wp.atomic_max(failure, 0, 110)


@no_type_check
@wp.kernel
def publish(
    m: Network,
    w: Stages,
    gas: wp.array2d(dtype=wp.float64),
    particles: wp.array3d(dtype=wp.float64),
    failure: wp.array(dtype=int),
):
    """Commit each changed lane only after all boxes succeeded."""
    b, i = wp.tid()
    if failure[0] != 0:
        return
    ng = m.gm.shape[0]
    if i < ng:
        if gas[b, i] / m.gm[i] != w.v[b, 0, i]:
            gas[b, i] = w.v[b, 0, i] * m.gm[i]
    else:
        j = (i - ng) % m.pm.shape[0]
        slot = (i - ng) // m.pm.shape[0]
        ref = w.reference[b, slot]
        if ref > 0 and particles[b, slot, j] / m.pm[j] / ref != w.v[b, 0, i]:
            particles[b, slot, j] = w.v[b, 0, i] * m.pm[j] * ref
