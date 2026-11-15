# Warp DSL vector operations/array annotations are compiled by Warp.
# mypy: disable-error-code="index"
"""Float64 complex arithmetic for optical Warp kernels.

Complex values are represented by (real, imaginary) wp.vec2d components.
"""

import warp as wp


@wp.func
def multiply(a: wp.vec2d, b: wp.vec2d) -> wp.vec2d:
    """Return the complex product."""
    return wp.vec2d(a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


@wp.func
def divide(a: wp.vec2d, b: wp.vec2d) -> wp.vec2d:
    """Return a/b using scaled division to avoid squaring large components."""
    result = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    if wp.abs(b[0]) >= wp.abs(b[1]):
        ratio = b[1] / b[0]
        denominator = b[0] + b[1] * ratio
        result = wp.vec2d(
            (a[0] + a[1] * ratio) / denominator,
            (a[1] - a[0] * ratio) / denominator,
        )
    else:
        ratio = b[0] / b[1]
        denominator = b[1] + b[0] * ratio
        result = wp.vec2d(
            (a[0] * ratio + a[1]) / denominator,
            (a[1] * ratio - a[0]) / denominator,
        )
    return result


@wp.func
def norm_squared(a: wp.vec2d) -> wp.float64:
    """Return the squared complex modulus."""
    return a[0] * a[0] + a[1] * a[1]


@wp.func
def real_product_conjugate(a: wp.vec2d, b: wp.vec2d) -> wp.float64:
    """Return Re(a conjugate(b))."""
    return a[0] * b[0] + a[1] * b[1]
