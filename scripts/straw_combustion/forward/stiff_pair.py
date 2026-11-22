"""Exact constant-bath two-reservoir kinetics with finite-bath coupling."""

import warp as wp

from .common import F, k


@wp.func
def weights(rate: F, dt: F):
    """Return the exponential weights for the coupled precursor step."""
    x = rate * dt
    phi = F(1) - x / F(2) + x * x / F(6)
    psi = F(0.5) - x / F(6) + x * x / F(24)
    if x > F(1e-4):
        phi = k.fraction(x) / x
        psi = (F(1) - phi) / x
    return wp.vec3d(wp.exp(-x), dt * phi, dt * dt * psi)


@wp.func
def evolve(
    h: F, b: F, src_h: F, src_b: F, kr: F, kb: F, kd: F, ko: F, ka: F, dt: F
):
    """Return KOH, KHSO3 and their time integrals for affine mass action."""
    diag_h = kr + kb + ka
    diag_b = kd + ko
    det = (kr + ka) * (kd + ko) + kb * ko
    gap = wp.sqrt((diag_h - diag_b) * (diag_h - diag_b) + F(4) * kb * kd)
    fast = F(0.5) * (diag_h + diag_b + gap)
    slow = det / wp.max(fast, F(1e-100))
    xf = weights(fast, dt)
    xs = weights(slow, dt)
    after_h = xf[0] * h + xf[1] * src_h
    after_b = xf[0] * b + xf[1] * src_b
    int_h = xf[1] * h + xf[2] * src_h
    int_b = xf[1] * b + xf[2] * src_b
    if gap > F(1e-9) * wp.max(F(1), fast):
        # Spectral projectors of A constructed without a matrix inverse.
        ph = ((fast - diag_h) * h + kd * b) / gap
        pb = (kb * h + (fast - diag_b) * b) / gap
        sh = ((fast - diag_h) * src_h + kd * src_b) / gap
        sb = (kb * src_h + (fast - diag_b) * src_b) / gap
        after_h += (xs[0] - xf[0]) * ph + (xs[1] - xf[1]) * sh
        after_b += (xs[0] - xf[0]) * pb + (xs[1] - xf[1]) * sb
        int_h += (xs[1] - xf[1]) * ph + (xs[2] - xf[2]) * sh
        int_b += (xs[1] - xf[1]) * pb + (xs[2] - xf[2]) * sb
    else:
        # Coincident eigenvalues, exp(A t)=exp(-f t)[I+t(A+fI)].
        x = fast * dt
        first = dt * dt * (F(0.5) - x / F(3) + x * x / F(8))
        second = dt * dt * dt * (F(1) / F(6) - x / F(12) + x * x / F(40))
        if x > F(0.001):
            first = (xf[1] - dt * xf[0]) / fast
            second = (xf[2] - first) / fast
        nh = (fast - diag_h) * h + kd * b
        nb = kb * h + (fast - diag_b) * b
        ns_h = (fast - diag_h) * src_h + kd * src_b
        ns_b = kb * src_h + (fast - diag_b) * src_b
        after_h += xf[0] * dt * nh + first * ns_h
        after_b += xf[0] * dt * nb + first * ns_b
        int_h += first * nh + second * ns_h
        int_b += first * nb + second * ns_b
    return wp.vec4d(
        wp.max(F(0), after_h),
        wp.max(F(0), after_b),
        wp.max(F(0), int_h),
        wp.max(F(0), int_b),
    )


Ledger = wp.types.vector(8, F)


@wp.func
def coupled_pair(
    oh: F,
    bis: F,
    source_b: F,
    salt: F,
    so2: F,
    so3: F,
    hcl: F,
    hydro: F,
    chlorinate: F,
    bind: F,
    associate: F,
    diss: F,
    oxidize: F,
    dt: F,
):
    """Use two midpoint updates with conservative gas limits."""
    avg_salt = salt
    avg_acid = hcl
    avg_so2 = so2
    avg_so3 = so3
    result = Ledger()
    for _ in range(3):
        kf = bind * avg_so2
        k12 = associate * avg_so3
        z = evolve(
            oh,
            bis,
            hydro * avg_salt,
            source_b / dt,
            chlorinate * avg_acid,
            kf,
            diss,
            oxidize,
            k12,
            dt,
        )
        oxidation = oxidize * z[3]
        capture = k12 * z[2]
        # Recover net source fluxes from stable inventories, avoiding
        # cancellation of very large, nearly equal forward/reverse rates.
        net_so2 = z[1] - bis - source_b + oxidation
        net_h = z[0] - oh + net_so2 + capture
        use = F(1)
        if net_h > salt:
            use = wp.min(use, salt / net_h)
        if -net_h > hcl:
            use = wp.min(use, hcl / (-net_h))
        if net_so2 > so2:
            use = wp.min(use, so2 / net_so2)
        if capture > so3:
            use = wp.min(use, so3 / capture)
        result = Ledger(
            oh + use * (z[0] - oh),
            bis + source_b + use * (z[1] - bis - source_b),
            use * net_h,
            use * net_so2,
            use * capture,
            use * oxidation,
            use * kf * z[2],
            use * diss * z[3],
        )
        avg_salt = wp.max(F(0), salt - F(0.5) * result[2])
        avg_acid = wp.max(F(0), hcl + F(0.5) * result[2])
        avg_so2 = wp.max(F(0), so2 - F(0.5) * result[3])
        avg_so3 = wp.max(F(0), so3 - F(0.5) * result[4])
    return result
