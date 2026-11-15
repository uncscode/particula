# Warp compiles these vector operations and array annotations.
# mypy: disable-error-code="index,operator,valid-type"
"""Direct device angular scattering and polarized optical operations.

Conventions match the CPU optics modules: S1 perpendicular, S2 parallel,
Stokes I,Q,U,V with V=-2 Im(Eparallel Eperpendicular*). Every output is
caller-owned; kernels allocate and transfer no arrays.
"""

import warp as wp

from particula.optics.gpu._complex import norm_squared


@wp.kernel
def scattering_amplitudes_kernel(
    electric: wp.array2d(dtype=wp.vec2d),
    magnetic: wp.array2d(dtype=wp.vec2d),
    term_count: wp.array(dtype=wp.int32),
    coefficient_status: wp.array(dtype=wp.int32),
    angles: wp.array(dtype=wp.float64),
    perpendicular: wp.array2d(dtype=wp.vec2d),
    parallel: wp.array2d(dtype=wp.vec2d),
    status: wp.array2d(dtype=wp.int32),
):
    """Evaluate one (case,angle) per thread using cached coefficients.

    Outputs have shape (cases,angles). Status 0 is success, 1 invalid angle,
    2 failed coefficients, 3 numerical failure. Zero outputs on failure
    are not valid results.
    """
    i, j = wp.tid()  # type: ignore[misc]
    zero = wp.float64(0.0)
    one = wp.float64(1.0)
    perpendicular[i, j] = wp.vec2d(zero, zero)
    parallel[i, j] = wp.vec2d(zero, zero)
    status[i, j] = 0
    if coefficient_status[i] != 0:
        status[i, j] = 2
        return
    angle = angles[j]
    if (
        not wp.isfinite(angle)
        or angle < zero
        or angle > wp.float64(3.141592653589793)
    ):
        status[i, j] = 1
        return
    mu = wp.cos(angle)
    previous = zero
    current = one
    s1 = wp.vec2d(zero, zero)
    s2 = wp.vec2d(zero, zero)
    for order in range(1, term_count[i] + 1):
        n = wp.float64(order)
        tau = n * mu * current - (n + one) * previous
        weight = (wp.float64(2.0) * n + one) / (n * (n + one))
        a = electric[i, order - 1]
        b = magnetic[i, order - 1]
        s1 += (a * current + b * tau) * weight
        s2 += (a * tau + b * current) * weight
        following = (
            (wp.float64(2.0) * n + one) * mu * current - (n + one) * previous
        ) / n
        previous = current
        current = following
    if (
        not wp.isfinite(s1[0])
        or not wp.isfinite(s1[1])
        or not wp.isfinite(s2[0])
        or not wp.isfinite(s2[1])
    ):
        status[i, j] = 3
        return
    perpendicular[i, j] = s1
    parallel[i, j] = s2


@wp.kernel
def mueller_kernel(
    perpendicular: wp.array2d(dtype=wp.vec2d),
    parallel: wp.array2d(dtype=wp.vec2d),
    amplitude_status: wp.array2d(dtype=wp.int32),
    scale: wp.array(dtype=wp.float64),
    matrix: wp.array4d(dtype=wp.float64),
    status: wp.array2d(dtype=wp.int32),
):
    """Construct full Mueller matrices with an explicit per-case scale.

    Use scale=1 for raw amplitudes; (lambda/(2*pi*n_medium))² for m²/sr;
    divide that factor by scattering cross section for normalized phase.
    Zero scattering uses scale=0 and an externally retained validity mask.
    Shape is (cases,angles,4,4); status 1 invalid scale, 2 failed amplitudes,
    3 numerical failure.
    """
    i, j = wp.tid()  # type: ignore[misc]
    zero = wp.float64(0.0)
    for row in range(4):
        for col in range(4):
            matrix[i, j, row, col] = zero
    status[i, j] = 0
    if amplitude_status[i, j] != 0:
        status[i, j] = 2
        return
    factor = scale[i]
    if not wp.isfinite(factor) or factor < zero:
        status[i, j] = 1
        return
    s1 = perpendicular[i, j]
    s2 = parallel[i, j]
    s11 = (norm_squared(s2) + norm_squared(s1)) * wp.float64(0.5) * factor
    s12 = (norm_squared(s2) - norm_squared(s1)) * wp.float64(0.5) * factor
    real = (s2[0] * s1[0] + s2[1] * s1[1]) * factor
    imaginary = (s2[1] * s1[0] - s2[0] * s1[1]) * factor
    if (
        not wp.isfinite(s11)
        or not wp.isfinite(s12)
        or not wp.isfinite(real)
        or not wp.isfinite(imaginary)
    ):
        status[i, j] = 3
        return
    matrix[i, j, 0, 0] = s11
    matrix[i, j, 1, 1] = s11
    matrix[i, j, 0, 1] = s12
    matrix[i, j, 1, 0] = s12
    matrix[i, j, 2, 2] = real
    matrix[i, j, 3, 3] = real
    matrix[i, j, 2, 3] = imaginary
    matrix[i, j, 3, 2] = -imaginary


@wp.func
def rotated_stokes(vector: wp.vec4d, angle: wp.float64) -> wp.vec4d:
    """Rotate basis axes by angle, with Q'=cos(2a)Q+sin(2a)U."""
    cosine = wp.cos(wp.float64(2.0) * angle)
    sine = wp.sin(wp.float64(2.0) * angle)
    return wp.vec4d(
        vector[0],
        cosine * vector[1] + sine * vector[2],
        -sine * vector[1] + cosine * vector[2],
        vector[3],
    )


@wp.kernel
def apply_mueller_kernel(
    matrix: wp.array4d(dtype=wp.float64),
    matrix_status: wp.array2d(dtype=wp.int32),
    incident: wp.array2d(dtype=wp.vec4d),
    input_rotation: wp.array2d(dtype=wp.float64),
    output_rotation: wp.array2d(dtype=wp.float64),
    outgoing: wp.array2d(dtype=wp.vec4d),
    status: wp.array2d(dtype=wp.int32),
):
    """Apply R(output) M R(input) on-device to each (case,angle).

    Status 1 marks nonfinite vector/rotation/matrix, 2 upstream failure,
    3 numerical failure.
    No distance scaling or intensity normalization is applied.
    """
    i, j = wp.tid()  # type: ignore[misc]
    zero = wp.float64(0.0)
    outgoing[i, j] = wp.vec4d(zero)
    status[i, j] = 0
    if matrix_status[i, j] != 0:
        status[i, j] = 2
        return
    incoming = incident[i, j]
    angle_in = input_rotation[i, j]
    angle_out = output_rotation[i, j]
    valid = wp.isfinite(angle_in) and wp.isfinite(angle_out)
    for k in range(4):
        valid = valid and wp.isfinite(incoming[k])
    if not valid:
        status[i, j] = 1
        return
    vector = rotated_stokes(incoming, angle_in)
    scattered = wp.vec4d(zero)
    for row in range(4):
        component = zero
        for col in range(4):
            value = matrix[i, j, row, col]
            valid = valid and wp.isfinite(value)
            component += value * vector[col]
        scattered[row] = component
    if not valid:
        status[i, j] = 1
        return
    result = rotated_stokes(scattered, angle_out)
    for k in range(4):
        valid = valid and wp.isfinite(result[k])
    if not valid:
        status[i, j] = 3
        return
    outgoing[i, j] = result
