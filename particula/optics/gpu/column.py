# Warp array annotation syntax is interpreted by its compiler.
# mypy: disable-error-code="valid-type,operator"
"""Device optical depth, direct-beam transmission and mass opacity."""

import warp as wp


@wp.kernel
def column_optics_kernel(
    extinction: wp.array2d(dtype=wp.float64),
    path_length: wp.array2d(dtype=wp.float64),
    input_status: wp.array2d(dtype=wp.int32),
    layer_depth: wp.array2d(dtype=wp.float64),
    total_depth: wp.array(dtype=wp.float64),
    transmission: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    """Integrate (columns,layers) coefficients along explicit lengths.

    Extinction [m⁻¹], lengths [m], and outputs [1]. Status codes 1 invalid,
    2 upstream failure, 3 overflow; failed partial results are not usable.
    Empty columns have zero depth and unit transmission.
    """
    column = wp.tid()
    zero = wp.float64(0.0)
    total = zero
    status[column] = 0
    total_depth[column] = zero
    transmission[column] = zero
    for layer in range(extinction.shape[1]):
        layer_depth[column, layer] = zero
    for layer in range(extinction.shape[1]):
        if input_status[column, layer] != 0:
            status[column] = 2
            return
        coefficient = extinction[column, layer]
        length = path_length[column, layer]
        if (
            not wp.isfinite(coefficient)
            or coefficient < zero
            or not wp.isfinite(length)
            or length < zero
        ):
            status[column] = 1
            return
        depth = coefficient * length
        total += depth
        if not wp.isfinite(total):
            status[column] = 3
            return
        layer_depth[column, layer] = depth
    total_depth[column] = total
    transmission[column] = wp.exp(-total)


@wp.kernel
def mass_opacity_kernel(
    coefficient: wp.array(dtype=wp.float64),
    mass_density: wp.array(dtype=wp.float64),
    input_status: wp.array(dtype=wp.int32),
    opacity: wp.array(dtype=wp.float64),
    valid: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
):
    """Compute mass-specific opacity [m²/kg] from m⁻¹ and kg/m³.

    Zero/zero returns zero with valid=0; nonzero/zero is invalid. Status codes
    1 invalid input, 2 upstream failure, 3 nonfinite result.
    """
    i = wp.tid()
    zero = wp.float64(0.0)
    opacity[i] = zero
    valid[i] = 0
    status[i] = 0
    if input_status[i] != 0:
        status[i] = 2
        return
    optical = coefficient[i]
    density = mass_density[i]
    if (
        not wp.isfinite(optical)
        or not wp.isfinite(density)
        or optical < zero
        or density < zero
        or (density == zero and optical > zero)
    ):
        status[i] = 1
        return
    if density > zero:
        result = optical / density
        if not wp.isfinite(result):
            status[i] = 3
            return
        opacity[i] = result
        valid[i] = 1
