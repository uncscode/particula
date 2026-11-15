# Warp DSL vector operations/array annotations are compiled by Warp.
# mypy: disable-error-code="index,operator,valid-type"
"""Direct float64 Warp Mie kernels with caller-owned buffers.

Equations and SI/index conventions match particula.optics.mie. Inputs here
are flattened independent cases of relative n+ik and dimensionless x.
Status codes: 0 success, 1 invalid input/capacity, 2 convergence failure,
3 nonfinite results, 4 unphysical efficiencies beyond roundoff. Failed
results must not be consumed.
"""

import warp as wp

from particula.optics.gpu._complex import (
    divide,
    multiply,
    norm_squared,
    real_product_conjugate,
)

# Typed constants preserve double precision before Warp compiles expressions.
_CF_TINY = wp.constant(wp.float64(1e-300))
_CF_TOLERANCE_SQUARED = wp.constant(wp.float64(4e-28))
_SMALL_TOLERANCE = wp.constant(wp.float64(2e-16))
_ORDER_GROWTH = wp.constant(wp.float64(4.05))
_ROUNDOFF = wp.constant(wp.float64(64 * 2.220446049250313e-16))


@wp.func
def regular_small(x: wp.float64, order: int) -> wp.float64:
    """Evaluate x*j_order(x) using the convergent small-argument series."""
    leading = x
    for n in range(1, order + 1):
        leading *= x / wp.float64(2 * n + 1)
    term = leading
    value = term
    for k in range(1, 100):
        term *= -x * x / wp.float64(2 * k * (2 * order + 2 * k + 1))
        value += term
        if wp.abs(term) <= wp.abs(value) * _SMALL_TOLERANCE:
            break
    return value


@wp.func
def logarithmic_derivative(z: wp.vec2d, count: int):
    """Return the continued-fraction derivative and convergence flag."""
    zero = wp.float64(0.0)
    one = wp.float64(1.0)
    c_one = wp.vec2d(one, zero)
    c_zero = wp.vec2d(zero, zero)
    value = divide(wp.vec2d(wp.float64(2 * count + 1), zero), z)
    c = value
    d = c_zero
    converged = int(0)
    for j in range(1, 10001):
        denominator = divide(wp.vec2d(wp.float64(2 * (count + j) + 1), zero), z)
        d = denominator - d
        c = denominator - divide(c_one, c)
        if wp.max(wp.abs(d[0]), wp.abs(d[1])) < _CF_TINY:
            d = wp.vec2d(_CF_TINY, zero)
        if wp.max(wp.abs(c[0]), wp.abs(c[1])) < _CF_TINY:
            c = wp.vec2d(_CF_TINY, zero)
        d = divide(c_one, d)
        delta = multiply(c, d)
        value = multiply(value, delta)
        if norm_squared(delta - c_one) < _CF_TOLERANCE_SQUARED:
            converged = 1
            break
    return value - divide(wp.vec2d(wp.float64(count), zero), z), converged


@wp.kernel
def mie_coefficients_kernel(
    relative_index: wp.array(dtype=wp.vec2d),
    size_parameter: wp.array(dtype=wp.float64),
    derivatives: wp.array2d(dtype=wp.vec2d),
    electric: wp.array2d(dtype=wp.vec2d),
    magnetic: wp.array2d(dtype=wp.vec2d),
    term_count: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
):
    """Evaluate one sphere per thread; overwrite counts/status every call.

    Workspace derivatives has coefficient width+1 columns. Electric and
    magnetic arrays share shape (cases, capacity). Trailing coefficients are
    zeroed. No host transfers, allocations or synchronization occur here.
    """
    i = wp.tid()
    zero = wp.float64(0.0)
    one = wp.float64(1.0)
    c_one = wp.vec2d(one, zero)
    c_zero = wp.vec2d(zero, zero)
    x = size_parameter[i]
    m = relative_index[i]
    capacity = electric.shape[1]
    status[i] = 0
    term_count[i] = 0
    for j in range(capacity):
        electric[i, j] = c_zero
        magnetic[i, j] = c_zero
    if (
        not wp.isfinite(x)
        or not wp.isfinite(m[0])
        or not wp.isfinite(m[1])
        or x < zero
        or x > wp.float64(10000.0)
        or m[0] <= zero
        or m[1] < zero
    ):
        status[i] = 1
        return
    if x == zero or (m[0] == one and m[1] == zero):
        return
    count = int(
        wp.ceil(
            x
            + _ORDER_GROWTH * wp.pow(x, one / wp.float64(3.0))
            + wp.float64(2.0)
        )
    )
    if count > capacity:
        status[i] = 1
        return
    z = m * x
    derivative, converged = logarithmic_derivative(z, count)
    if converged == 0:
        status[i] = 2
        return
    derivatives[i, count] = derivative
    for back in range(count):
        n = count - back
        ratio = divide(wp.vec2d(wp.float64(n), zero), z)
        derivatives[i, n - 1] = ratio - divide(c_one, derivatives[i, n] + ratio)
    previous_psi = wp.sin(x)
    psi = wp.sin(x) / x - wp.cos(x)
    previous_chi = wp.cos(x)
    chi = wp.cos(x) / x + wp.sin(x)
    for n in range(1, count + 1):
        if x < one:
            previous_psi = regular_small(x, n - 1)
            psi = regular_small(x, n)
        previous_xi = wp.vec2d(previous_psi, -previous_chi)
        xi = wp.vec2d(psi, -chi)
        shift = wp.vec2d(wp.float64(n) / x, zero)
        da = divide(derivatives[i, n], m) + shift
        db = multiply(derivatives[i, n], m) + shift
        a = divide(
            da * psi - wp.vec2d(previous_psi, zero),
            multiply(da, xi) - previous_xi,
        )
        b = divide(
            db * psi - wp.vec2d(previous_psi, zero),
            multiply(db, xi) - previous_xi,
        )
        if (
            not wp.isfinite(a[0])
            or not wp.isfinite(a[1])
            or not wp.isfinite(b[0])
            or not wp.isfinite(b[1])
        ):
            status[i] = 3
            return
        electric[i, n - 1] = a
        magnetic[i, n - 1] = b
        next_psi = wp.float64(2 * n + 1) * psi / x - previous_psi
        next_chi = wp.float64(2 * n + 1) * chi / x - previous_chi
        previous_psi = psi
        psi = next_psi
        previous_chi = chi
        chi = next_chi
    term_count[i] = count


@wp.kernel
def mie_efficiencies_kernel(
    relative_index: wp.array(dtype=wp.vec2d),
    size_parameter: wp.array(dtype=wp.float64),
    electric: wp.array2d(dtype=wp.vec2d),
    magnetic: wp.array2d(dtype=wp.vec2d),
    term_count: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
    efficiencies: wp.array2d(dtype=wp.float64),
):
    """Write Qext,Qsca,Qabs,g,Qback,Qpr and a scattering-valid flag.

    Shape is (cases,7). Failed upstream cases produce zero outputs but retain
    nonzero status; status must be checked before results are used.
    """
    i = wp.tid()
    zero = wp.float64(0.0)
    for j in range(7):
        efficiencies[i, j] = zero
    if status[i] != 0 or term_count[i] == 0:
        return
    x = size_parameter[i]
    ext = zero
    sca = zero
    adjacent = zero
    cross = zero
    backward = wp.vec2d(zero, zero)
    sign = wp.float64(-1.0)
    for j in range(term_count[i]):
        n = wp.float64(j + 1)
        a = electric[i, j]
        b = magnetic[i, j]
        weight = wp.float64(2.0) * n + wp.float64(1.0)
        ext += weight * (a[0] + b[0])
        sca += weight * (norm_squared(a) + norm_squared(b))
        cross += (
            weight / (n * (n + wp.float64(1.0))) * real_product_conjugate(a, b)
        )
        if j + 1 < term_count[i]:
            adjacent += (
                n
                * (n + wp.float64(2.0))
                / (n + wp.float64(1.0))
                * (
                    real_product_conjugate(a, electric[i, j + 1])
                    + real_product_conjugate(b, magnetic[i, j + 1])
                )
            )
        backward += (a - b) * weight * sign
        sign = -sign
    factor = wp.float64(2.0) / (x * x)
    ext *= factor
    sca *= factor
    absorption = ext - sca
    if relative_index[i][1] == zero:
        absorption = zero
        ext = sca
    g = zero
    if sca > zero:
        g = wp.float64(2.0) * factor * (adjacent + cross) / sca
        efficiencies[i, 6] = wp.float64(1.0)
    efficiencies[i, 0] = ext
    efficiencies[i, 1] = sca
    efficiencies[i, 2] = absorption
    efficiencies[i, 3] = g
    efficiencies[i, 4] = norm_squared(backward) / (x * x)
    efficiencies[i, 5] = ext - g * sca
    for j in range(6):
        if not wp.isfinite(efficiencies[i, j]):
            status[i] = 3
            return
    tolerance = _ROUNDOFF * wp.max(wp.abs(ext), sca)
    if (
        absorption < -tolerance
        or ext < zero
        or wp.abs(g) > wp.float64(1.0) + _ROUNDOFF
    ):
        status[i] = 4
        return
    # Only negative cancellation within roundoff is zeroed, preserving the
    # extinction = scattering + absorption identity for downstream consumers.
    absorption = wp.max(absorption, zero)
    ext = wp.max(ext, sca)
    g = wp.clamp(g, wp.float64(-1.0), wp.float64(1.0))
    efficiencies[i, 0] = ext
    efficiencies[i, 2] = absorption
    efficiencies[i, 3] = g
    efficiencies[i, 5] = ext - g * sca
