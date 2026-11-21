# Warp DSL vector operations and annotations are compiled by Warp.
# mypy: disable-error-code="index,operator,valid-type"
"""Analytic derivatives of the smooth Mie cross-section channels.

These local derivatives use the same evaluated coefficients as the forward
Mie calculation. They avoid differentiating through convergence-controlled
loops and can seed a Warp tape for upstream size and composition physics.
"""

import warp as wp

from particula.optics.gpu._complex import (
    divide,
    multiply,
    real_product_conjugate,
)
from particula.optics.gpu.mie import regular_small


@wp.func
def _coefficient_slopes(
    d: wp.vec2d,
    dm: wp.vec2d,
    dx: wp.vec2d,
    a: wp.vec2d,
    psi: wp.float64,
    psi_derivative: wp.float64,
    xi: wp.vec2d,
    xi_derivative: wp.vec2d,
    previous_xi: wp.vec2d,
    previous_xi_derivative: wp.vec2d,
    previous_psi_derivative: wp.float64,
):
    denominator = multiply(d, xi) - previous_xi
    index_slope = divide(
        multiply(wp.vec2d(psi, wp.float64(0.0)) - multiply(a, xi), dm),
        denominator,
    )
    numerator_slope = (
        d * psi_derivative
        + dx * psi
        - wp.vec2d(previous_psi_derivative, wp.float64(0.0))
    )
    denominator_slope = (
        multiply(d, xi_derivative) + multiply(dx, xi) - previous_xi_derivative
    )
    size_slope = divide(
        numerator_slope - multiply(a, denominator_slope), denominator
    )
    return index_slope, size_slope


@wp.func
def _slopes(a: wp.vec2d, index_slope: wp.vec2d, size_slope: wp.vec2d):
    k_slope = wp.vec2d(-index_slope[1], index_slope[0])
    extinction = wp.vec3d(index_slope[0], k_slope[0], size_slope[0])
    scattering = wp.float64(2.0) * wp.vec3d(
        real_product_conjugate(index_slope, a),
        real_product_conjugate(k_slope, a),
        real_product_conjugate(size_slope, a),
    )
    return extinction, scattering


@wp.kernel
def mie_cross_section_sums_jacobian(
    relative_index: wp.array(dtype=wp.vec2d),
    size_parameter: wp.array(dtype=wp.float64),
    derivatives: wp.array2d(dtype=wp.vec2d),
    electric: wp.array2d(dtype=wp.vec2d),
    magnetic: wp.array2d(dtype=wp.vec2d),
    term_count: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
    extinction: wp.array(dtype=wp.vec3d),
    scattering: wp.array(dtype=wp.vec3d),
):
    """Differentiate the dimensionless Mie sums before the 2/x² factor.

    Each output vector is ordered (relative n, relative k, x). Physical cross
    sections multiply the sums by 2*pi*(lambda/(2*pi*n_medium))². At k=0
    the index derivative is the right derivative of the unrounded solution.
    Inputs require successful coefficients and positive x in the qualified
    gradient domain. A failed case retains or sets nonzero status.
    """
    case = wp.tid()
    zero = wp.float64(0.0)
    one = wp.float64(1.0)
    ext = wp.vec3d(zero, zero, zero)
    sca = wp.vec3d(zero, zero, zero)
    extinction[case] = ext
    scattering[case] = sca
    if status[case] != 0:
        return
    m = relative_index[case]
    x = size_parameter[case]
    if term_count[case] == 0:
        # At index match, the right derivative of absorption is the
        # Beer–Lambert coefficient per k times the sphere volume.
        if m[0] == one and m[1] == zero and x > zero:
            extinction[case] = wp.vec3d(
                zero, wp.float64(4.0) / wp.float64(3.0) * x * x * x, zero
            )
        return
    z = m * x
    z_squared = multiply(z, z)
    m_squared = multiply(m, m)
    previous_psi = wp.sin(x)
    psi = wp.sin(x) / x - wp.cos(x)
    previous_chi = wp.cos(x)
    chi = wp.cos(x) / x + wp.sin(x)
    for order in range(1, term_count[case] + 1):
        n = wp.float64(order)
        if x < one:
            previous_psi = regular_small(x, order - 1)
            psi = regular_small(x, order)
        previous_xi = wp.vec2d(previous_psi, -previous_chi)
        xi = wp.vec2d(psi, -chi)
        shift = n / x
        previous_psi_prime = shift * previous_psi - psi
        psi_prime = previous_psi - shift * psi
        previous_xi_prime = shift * previous_xi - xi
        xi_prime = previous_xi - shift * xi
        d = derivatives[case, order]
        # D = psi'/psi obeys the spherical Riccati–Bessel equation.
        d_prime = (
            divide(wp.vec2d(n * (n + one), zero), z_squared)
            - wp.vec2d(one, zero)
            - multiply(d, d)
        )
        offset = wp.vec2d(shift, zero)
        offset_prime = wp.vec2d(-shift / x, zero)
        d_a = divide(d, m)
        a = electric[case, order - 1]
        b = magnetic[case, order - 1]
        a_index, a_size = _coefficient_slopes(
            d_a + offset,
            divide(d_prime * x - d_a, m),
            d_prime + offset_prime,
            a,
            psi,
            psi_prime,
            xi,
            xi_prime,
            previous_xi,
            previous_xi_prime,
            previous_psi_prime,
        )
        b_index, b_size = _coefficient_slopes(
            multiply(d, m) + offset,
            multiply(d_prime, z) + d,
            multiply(d_prime, m_squared) + offset_prime,
            b,
            psi,
            psi_prime,
            xi,
            xi_prime,
            previous_xi,
            previous_xi_prime,
            previous_psi_prime,
        )
        ext_a, sca_a = _slopes(a, a_index, a_size)
        ext_b, sca_b = _slopes(b, b_index, b_size)
        factor = wp.float64(2.0) * n + one
        ext = ext + factor * (ext_a + ext_b)
        sca = sca + factor * (sca_a + sca_b)
        next_order = wp.float64(2.0) * shift + one / x
        next_psi = next_order * psi - previous_psi
        next_chi = next_order * chi - previous_chi
        previous_psi = psi
        psi = next_psi
        previous_chi = chi
        chi = next_chi
    if not wp.isfinite(ext) or not wp.isfinite(sca):
        status[case] = 3
        return
    extinction[case] = ext
    scattering[case] = sca
