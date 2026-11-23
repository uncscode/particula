"""Float64 multivapor parcel kernels and resident trial controller."""

from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    Vector = Any
    Matrix = Any
    Cube = Any
    Ints = Any
    IntMatrix = Any
else:
    Vector = wp.array(dtype=wp.float64)
    Matrix = wp.array2d(dtype=wp.float64)
    Cube = wp.array3d(dtype=wp.float64)
    Ints = wp.array(dtype=int)
    IntMatrix = wp.array2d(dtype=int)

# y contains scaled pressure, temperature and the exchangeable material masses
# in class-major order. Gas inventories follow from exact species conservation.
# clock contains accepted time, proposed h, segment end and current trial h.
# counters track accepted/rejected steps, segment trials, fault and RHS calls.
ONE = wp.constant(wp.float64(1.0))
ZERO = wp.constant(wp.float64(0.0))
PI4 = wp.constant(wp.float64(12.56637061435917295385))


@wp.kernel(enable_backward=False)
def start(clock: Vector, counters: Ints, stop: wp.float64, condition: Ints):
    clock[2] = stop
    counters[2] = 0
    condition[0] = int(stop > clock[0] and counters[3] == 0)


@wp.kernel(enable_backward=False)
def trial(clock: Vector, counters: Ints, invalid: Ints, detail: Ints):
    clock[3] = wp.min(clock[1], clock[2] - clock[0])
    counters[2] += 1
    invalid[0] = 0
    detail[0] = 0


@wp.kernel(enable_backward=False)
def unpack(
    y: Vector,
    m0: Matrix,
    q0: Vector,
    scale: Matrix,
    lanes: Ints,
    counts: Matrix,
    p0: wp.float64,
    t0: wp.float64,
    air: wp.float64,
    pressure: Vector,
    temperature: Vector,
    vapor: Vector,
    masses: Cube,
    invalid: Ints,
    detail: Ints,
    stage: int,
):
    pressure[0] = y[0] * p0
    temperature[0] = y[1] * t0
    bad = int(0)
    bad_slot = int(-1)
    if (
        not wp.isfinite(pressure[0])
        or pressure[0] <= ZERO
        or not wp.isfinite(temperature[0])
        or temperature[0] <= ZERO
    ):
        bad = 1
        pressure[0] = p0
        temperature[0] = t0
    for c in range(m0.shape[0]):
        for m in range(m0.shape[1]):
            masses[0, c, m] = m0[c, m]
    for j in range(lanes.shape[0]):
        q = q0[j]
        for c in range(m0.shape[0]):
            mass = y[2 + c * lanes.shape[0] + j] * scale[c, j]
            if not wp.isfinite(mass) or mass < ZERO:
                bad = 1
                bad_slot = c
                mass = m0[c, lanes[j]]
            masses[0, c, lanes[j]] = mass
            q += counts[0, c] * (m0[c, lanes[j]] - mass) / air
        if not wp.isfinite(q) or q < ZERO:
            bad = 1
            q = q0[j]
        vapor[j] = q
    if bad != 0:
        invalid[0] = 1
        if detail[0] == 0:
            detail[0] = 1
            detail[1] = 0
            detail[2] = bad_slot
            detail[3] = stage
            detail[4] = -100


@wp.kernel(enable_backward=False)
def hydro(
    pressure: Vector,
    temperature: Vector,
    vapor: Vector,
    props: Matrix,
    rd: wp.float64,
    loading_gw: wp.float64,
    volume: Vector,
    dp: Vector,
):
    r = rd
    for j in range(vapor.shape[0]):
        r += vapor[j] * props[j, 0]
    volume[0] = r * temperature[0] / pressure[0]
    dp[0] = -loading_gw / volume[0]


@wp.kernel(enable_backward=False)
def fluxes(
    pressure: Vector,
    temperature: Vector,
    vapor: Vector,
    volume: Vector,
    masses: Cube,
    counts: Matrix,
    density: Vector,
    lanes: Ints,
    props: Matrix,
    sigma: wp.float64,
    activities: Cube,
    free: Matrix,
    slopes: Matrix,
    flux: Matrix,
):
    c = wp.tid()
    t = temperature[0]
    size = wp.float64(0.0)
    for k in range(masses.shape[2]):
        size += masses[0, c, k] / density[k]
    if counts[0, c] <= ZERO or size <= ZERO:
        for j in range(lanes.shape[0]):
            flux[c, j] = ZERO
        return
    r = wp.pow(wp.float64(3.0) * size / PI4, ONE / wp.float64(3.0))
    heat = wp.float64(0.0)
    sink = (
        PI4 * r * wp.float64(0.001) * (wp.float64(4.39) + wp.float64(0.071) * t)
    )
    for j in range(lanes.shape[0]):
        gas_r = props[j, 0]
        latent = props[j, 1]
        curve = wp.float64(2.0) * sigma / (gas_r * t * density[lanes[j]] * r)
        peq = (
            activities[0, c, lanes[j]]
            * props[j, 2]
            * wp.exp(latent / gas_r * (ONE / props[j, 3] - ONE / t) + curve)
        )
        mobility = (
            PI4
            * r
            * props[j, 4]
            * wp.pow(t / wp.float64(273.15), wp.float64(1.94))
            * wp.float64(101325.0)
            / (pressure[0] * gas_r * t)
        )
        f = mobility * (vapor[j] * gas_r * t / volume[0] - peq)
        b = mobility * peq * (latent / (gas_r * t) - ONE - curve) / t
        free[c, j] = f
        slopes[c, j] = b
        heat += latent * f
        sink += latent * b
    for j in range(lanes.shape[0]):
        flux[c, j] = free[c, j] - slopes[c, j] * heat / sink


@wp.kernel(enable_backward=False)
def accumulate(
    volume: Vector,
    dp: Vector,
    flux: Matrix,
    counts: Matrix,
    props: Matrix,
    scale: Matrix,
    air: wp.float64,
    p0: wp.float64,
    t0: wp.float64,
    cp: wp.float64,
    result: Vector,
    counters: Ints,
    invalid: Ints,
):
    heat = wp.float64(0.0)
    for c in range(flux.shape[0]):
        for j in range(flux.shape[1]):
            result[2 + c * flux.shape[1] + j] = flux[c, j] / scale[c, j]
            heat += props[j, 1] * counts[0, c] * flux[c, j] / air
    result[0] = dp[0] / p0
    result[1] = (volume[0] * dp[0] + heat) / (t0 * cp)
    if not wp.isfinite(result[1]):
        invalid[0] = 1
    counters[4] += 1


@wp.kernel(enable_backward=False)
def mark_base(invalid: Ints, counters: Ints):
    if invalid[0] != 0:
        counters[3] = 1


@wp.kernel(enable_backward=False)
def rk_stage(
    y: Vector,
    clock: Vector,
    k1: Vector,
    k2: Vector,
    k3: Vector,
    stage: int,
    output: Vector,
):
    i = wp.tid()
    x = wp.float64(0.5) * k1[i]
    if stage == 2:
        x = wp.float64(0.75) * k2[i]
    elif stage == 3:
        x = (
            wp.float64(2.0) * k1[i]
            + wp.float64(3.0) * k2[i]
            + wp.float64(4.0) * k3[i]
        ) / wp.float64(9.0)
    output[i] = y[i] + clock[3] * x


@wp.kernel(enable_backward=False)
def rk_error(
    clock: Vector, k1: Vector, k2: Vector, k3: Vector, k4: Vector, error: Vector
):
    i = wp.tid()
    error[i] = (
        clock[3]
        * (
            -wp.float64(5.0) * k1[i]
            + wp.float64(6.0) * k2[i]
            + wp.float64(8.0) * k3[i]
            - wp.float64(9.0) * k4[i]
        )
        / wp.float64(72.0)
    )


@wp.kernel(enable_backward=False)
def perturb(
    y: Vector,
    ytrial: Vector,
    column: int,
    delta: Vector,
    q0: Vector,
    m0: Matrix,
    lanes: Ints,
    scale: Matrix,
    counts: Matrix,
    air: wp.float64,
):
    i = wp.tid()
    value = y[i]
    if i == column:
        step = wp.float64(1.4901161193847656e-8) * wp.max(ONE, wp.abs(value))
        if column >= 2:
            # Respect both liquid and vapor inventory boundaries.
            c = (column - 2) // lanes.shape[0]
            gas = (column - 2) % lanes.shape[0]
            q = q0[gas]
            for slot in range(scale.shape[0]):
                q += (
                    counts[0, slot]
                    * (
                        m0[slot, lanes[gas]]
                        - y[2 + slot * lanes.shape[0] + gas] * scale[slot, gas]
                    )
                    / air
                )
            uptake = counts[0, c] * scale[c, gas] / air
            if uptake > ZERO and step * uptake > wp.float64(0.5) * q:
                if value >= step:
                    step = -step
                else:
                    step = wp.max(ZERO, wp.float64(0.5) * q / uptake)
        value += step
        delta[0] = value - y[i]
    ytrial[i] = value


@wp.kernel(enable_backward=False)
def linearize(
    base: Vector,
    f: Vector,
    delta: Vector,
    clock: Vector,
    jac: Matrix,
    column: int,
):
    i = wp.tid()
    value = wp.float64(0.0)
    if delta[0] != ZERO:
        value = -(f[i] - base[i]) / delta[0]
    if i == column:
        value += ONE / (clock[3] * GAMMA)
    jac[i, column] = value


@wp.kernel(enable_backward=False)
def factor(jac: Matrix, pivots: Ints, invalid: Ints):
    # Small dense pivoted LU, reused for all three Rosenbrock stages.
    for j in range(jac.shape[0]):
        pivot = j
        for i in range(j + 1, jac.shape[0]):
            if wp.abs(jac[i, j]) > wp.abs(jac[pivot, j]):
                pivot = i
        pivots[j] = pivot
        if wp.abs(jac[pivot, j]) <= wp.float64(1e-280) or not wp.isfinite(
            jac[pivot, j]
        ):
            invalid[0] = 1
            return
        if pivot != j:
            for k in range(jac.shape[0]):
                tmp = jac[j, k]
                jac[j, k] = jac[pivot, k]
                jac[pivot, k] = tmp
        for i in range(j + 1, jac.shape[0]):
            jac[i, j] /= jac[j, j]
            for k in range(j + 1, jac.shape[0]):
                jac[i, k] -= jac[i, j] * jac[j, k]


@wp.kernel(enable_backward=False)
def solve(
    jac: Matrix,
    pivots: Ints,
    clock: Vector,
    rhs: Vector,
    k1: Vector,
    k2: Vector,
    stage: int,
    out: Vector,
):
    for i in range(rhs.shape[0]):
        value = rhs[i]
        if stage == 2:
            value -= C21 * k1[i] / clock[3]
        elif stage == 3:
            value += (C31 * k1[i] + C32 * k2[i]) / clock[3]
        out[i] = value
    for j in range(rhs.shape[0]):
        p = pivots[j]
        tmp = out[p]
        out[p] = out[j]
        out[j] = tmp
    for j in range(rhs.shape[0]):
        for i in range(j + 1, rhs.shape[0]):
            out[i] -= jac[i, j] * out[j]
    for k in range(rhs.shape[0]):
        j = rhs.shape[0] - 1 - k
        for i in range(j + 1, rhs.shape[0]):
            out[j] -= jac[j, i] * out[i]
        out[j] /= jac[j, j]


@wp.kernel(enable_backward=False)
def ros_stage(y: Vector, k1: Vector, ytrial: Vector):
    i = wp.tid()
    ytrial[i] = y[i] + k1[i]


@wp.kernel(enable_backward=False)
def ros_finish(
    y: Vector, k1: Vector, k2: Vector, k3: Vector, ytrial: Vector, error: Vector
):
    i = wp.tid()
    ytrial[i] = y[i] + k1[i] + M2 * k2[i] - M3 * k3[i]
    error[i] = wp.float64(0.5) * k1[i] - E2 * k2[i] + E3 * k3[i]


@wp.kernel(enable_backward=False)
def decide(
    y: Vector,
    ytrial: Vector,
    error: Vector,
    clock: Vector,
    invalid: Ints,
    counters: Ints,
    condition: Ints,
    rtol: wp.float64,
    minimum: wp.float64,
    limit: int,
):
    norm = wp.float64(1e-16)
    for i in range(y.shape[0]):
        rel = wp.abs(error[i]) / (
            rtol * (wp.float64(1e-8) + wp.max(wp.abs(y[i]), wp.abs(ytrial[i])))
        )
        if not wp.isfinite(rel):
            invalid[0] = 1
        norm = wp.max(norm, rel)
    h = clock[3]
    factor = wp.clamp(
        wp.float64(0.9) * wp.pow(norm, wp.float64(-1.0) / wp.float64(3.0)),
        wp.float64(0.1),
        wp.float64(4.0),
    )
    if invalid[0] != 0:
        factor = wp.float64(0.25)
    if counters[3] == 0 and norm <= ONE and invalid[0] == 0:
        for i in range(y.shape[0]):
            y[i] = ytrial[i]
        counters[0] += 1
        clock[0] += h
        clock[1] = wp.max(minimum, h * factor)
    else:
        counters[1] += 1
        clock[1] = h * wp.min(wp.float64(0.8), factor)
        if clock[1] < minimum and counters[3] == 0:
            counters[3] = 2
    if counters[2] >= limit and counters[3] == 0 and clock[0] < clock[2]:
        counters[3] = 3
    condition[0] = int(counters[3] == 0 and clock[0] < clock[2])


@wp.kernel(enable_backward=False)
def flux_guard(
    pressure: Vector,
    temperature: Vector,
    vapor: Vector,
    volume: Vector,
    masses: Cube,
    density: Vector,
    lanes: Ints,
    props: Matrix,
    free: Matrix,
    slopes: Matrix,
    flux: Matrix,
    errors: Cube,
    uncertain: IntMatrix,
):
    """Bound heat and mass flux errors about the trial surface solution."""
    c = wp.tid()
    size = wp.float64(0.0)
    for j in range(density.shape[0]):
        size += masses[0, c, j] / density[j]
    r = wp.pow(wp.float64(3.0) * size / PI4, ONE / wp.float64(3.0))
    t = temperature[0]
    sink = (
        PI4 * r * wp.float64(0.001) * (wp.float64(4.39) + wp.float64(0.071) * t)
    )
    heat = wp.float64(0.0)
    dsink = wp.float64(0.0)
    dheat = wp.float64(0.0)
    uncertain[0, c] = 0
    active = int(0)
    for j in range(lanes.shape[0]):
        eps = wp.exp(errors[0, c, lanes[j]]) - ONE
        if eps > ZERO:
            active = 1
        mobility = (
            PI4
            * r
            * props[j, 4]
            * wp.pow(t / wp.float64(273.15), wp.float64(1.94))
            * wp.float64(101325.0)
            / (pressure[0] * props[j, 0] * t)
        )
        driving = mobility * vapor[j] * props[j, 0] * t / volume[0] - free[c, j]
        heat += free[c, j] * props[j, 1]
        sink += slopes[c, j] * props[j, 1]
        dsink += wp.abs(slopes[c, j]) * eps * props[j, 1]
        dheat += wp.abs(driving) * eps * props[j, 1]
    if active == 0:
        return
    if not (sink > dsink):
        uncertain[0, c] = 1
        return
    offset = heat / sink
    doffset = (dheat + wp.abs(offset) * dsink) / (sink - dsink)
    for j in range(lanes.shape[0]):
        if props[j, 4] > ZERO:
            eps = wp.exp(errors[0, c, lanes[j]]) - ONE
            mobility = (
                PI4
                * r
                * props[j, 4]
                * wp.pow(t / wp.float64(273.15), wp.float64(1.94))
                * wp.float64(101325.0)
                / (pressure[0] * props[j, 0] * t)
            )
            driving = (
                mobility * vapor[j] * props[j, 0] * t / volume[0] - free[c, j]
            )
            bound = (
                eps * (wp.abs(driving) + wp.abs(slopes[c, j] * offset))
                + wp.abs(slopes[c, j]) * (ONE + eps) * doffset
            )
            if wp.abs(flux[c, j]) <= bound:
                uncertain[0, c] = 1
