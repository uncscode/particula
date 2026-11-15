# Warp array annotation syntax is interpreted by its compiler.
# mypy: disable-error-code="valid-type,operator"
"""Device population weighting and incoherent optical reductions.

Caller-provided arrays retain stable shapes. Status 0 means success, 1 invalid
input, 2 upstream failure, 3 nonfinite result. Failed output must not be used.
"""

import warp as wp


@wp.kernel
def population_weights_kernel(
    concentration: wp.array2d(dtype=wp.float64),
    volume: wp.array(dtype=wp.float64),
    radius_grid: wp.array(dtype=wp.float64),
    measure: int,
    weights: wp.array2d(dtype=wp.float64),
    status: wp.array2d(dtype=wp.int32),
):
    """Produce (boxes,particles) number weights [m⁻³].

    Measure 0 is discrete density, 1 represented count divided by box volume,
    2 radius PDF times trapezoidal widths. PDF grid length matches particles
    and must be at least two. Volume/grid arrays are only read when needed.
    """
    box, particle = wp.tid()  # type: ignore[misc]
    zero = wp.float64(0.0)
    value = concentration[box, particle]
    weights[box, particle] = zero
    status[box, particle] = 0
    if not wp.isfinite(value) or value < zero or measure < 0 or measure > 2:
        status[box, particle] = 1
        return
    if measure == 1:
        v = volume[box]
        if not wp.isfinite(v) or v <= zero:
            status[box, particle] = 1
            return
        value /= v
    elif measure == 2:
        count = concentration.shape[1]
        if count < 2:
            status[box, particle] = 1
            return
        r = radius_grid[particle]
        width = zero
        valid = wp.isfinite(r) and r > zero
        if particle > 0:
            left = radius_grid[particle - 1]
            valid = valid and wp.isfinite(left) and left > zero and left < r
            width += (r - left) * wp.float64(0.5)
        if particle + 1 < count:
            right = radius_grid[particle + 1]
            valid = valid and wp.isfinite(right) and right > r
            width += (right - r) * wp.float64(0.5)
        if not valid:
            status[box, particle] = 1
            return
        value *= width
    if not wp.isfinite(value):
        status[box, particle] = 3
        return
    weights[box, particle] = value


@wp.func
def finite_output(output: wp.array2d(dtype=wp.float64), group: int) -> bool:
    """Check a nine-column optical result for finite values."""
    finite = True
    for column in range(9):
        finite = finite and wp.isfinite(output[group, column])  # type: ignore[assignment]
    return finite


@wp.kernel
def population_optics_kernel(
    cross_sections: wp.array3d(dtype=wp.float64),
    asymmetry: wp.array2d(dtype=wp.float64),
    weights: wp.array2d(dtype=wp.float64),
    input_status: wp.array2d(dtype=wp.int32),
    output: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    """Reduce (groups,particles,5) cross sections to (groups,9) outputs.

    Groups can flatten box/wavelength axes. Cross-section columns are
    extinction, scattering, absorption, backscatter, radiation pressure [m²].
    Outputs are those five coefficients [m⁻¹], SSA, g, extinction-valid,
    scattering-valid. input_status combines optical and weight admission.
    """
    group = wp.tid()
    zero = wp.float64(0.0)
    status[group] = 0
    for column in range(9):
        output[group, column] = zero
    g_sum = zero
    for particle in range(weights.shape[1]):
        if input_status[group, particle] != 0:
            status[group] = 2
            return
        weight = weights[group, particle]
        g = asymmetry[group, particle]
        if (
            not wp.isfinite(weight)
            or weight < zero
            or not wp.isfinite(g)
            or wp.abs(g) > wp.float64(1.0)
        ):
            status[group] = 1
            return
        for column in range(5):
            value = cross_sections[group, particle, column]
            if not wp.isfinite(value) or value < zero:
                status[group] = 1
                return
            output[group, column] += value * weight
        g_sum += cross_sections[group, particle, 1] * weight * g
    extinction = output[group, 0]
    scattering = output[group, 1]
    if extinction > zero:
        output[group, 5] = scattering / extinction
        output[group, 7] = wp.float64(1.0)
    if scattering > zero:
        output[group, 6] = g_sum / scattering
        output[group, 8] = wp.float64(1.0)
    if not finite_output(output, group):  # type: ignore[arg-type]
        status[group] = 3


@wp.kernel
def population_mueller_kernel(
    matrices: wp.array4d(dtype=wp.float64),
    weights: wp.array2d(dtype=wp.float64),
    input_status: wp.array3d(dtype=wp.int32),
    output: wp.array3d(dtype=wp.float64),
    status: wp.array3d(dtype=wp.int32),
):
    """Reduce (groups,particles,angles,16) differential Mueller matrices.

    Flatten the final 4x4 in row-major order. Output is (groups,angles,16)
    [m⁻¹/sr]; signed polarized elements are preserved. Status is per element.
    """
    group, angle, element = wp.tid()  # type: ignore[misc]
    zero = wp.float64(0.0)
    output[group, angle, element] = zero
    status[group, angle, element] = 0
    total = zero
    for particle in range(weights.shape[1]):
        if input_status[group, particle, angle] != 0:
            status[group, angle, element] = 2
            return
        weight = weights[group, particle]
        value = matrices[group, particle, angle, element]
        if not wp.isfinite(weight) or weight < zero or not wp.isfinite(value):
            status[group, angle, element] = 1
            return
        total += weight * value
    if not wp.isfinite(total):
        status[group, angle, element] = 3
        return
    output[group, angle, element] = total
