# Warp array annotation syntax is interpreted by its compiler.
# mypy: disable-error-code="valid-type,operator"
"""Per-particle dimensional optical cross sections with explicit status.

Cross sections are Q*pi*r². Efficiencies follow Bohren and Huffman (1983),
https://doi.org/10.1002/9783527618156. Particle absorption remains separate
from any later energy transfer or aerosol/environment state mutation.
"""

import math

import warp as wp

_PI = wp.constant(wp.float64(math.pi))


@wp.kernel
def cross_sections_kernel(
    efficiencies: wp.array2d(dtype=wp.float64),
    radius: wp.array(dtype=wp.float64),
    input_status: wp.array(dtype=wp.int32),
    cross_sections: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    """Convert (cases,7) efficiencies to (cases,5) cross sections [m²].

    Args:
        efficiencies: Standard Mie order: ext, sca, abs, g, back, pr, valid.
        radius: Nonnegative finite SI radii [m], shape (cases,).
        input_status: Upstream coefficient/efficiency status, shape (cases,).
        cross_sections: Caller-owned output (cases,5): ext, sca, abs, back, pr.
        status: Caller-owned (cases,) int32: 0 success, 1 invalid input,
            2 upstream failure, 3 nonfinite result.

    Launch one thread per case, with all arrays on the same device. Shapes
    and float64/int32 dtypes are the caller's responsibility. Observe status
    before using outputs; a failed case can contain partial results. No
    synchronization, transfer, allocation or source mutation occurs here.
    """
    i = wp.tid()
    zero = wp.float64(0.0)
    status[i] = 0
    for k in range(5):
        cross_sections[i, k] = zero
    if input_status[i] != 0:
        status[i] = 2
        return
    r = radius[i]
    if not wp.isfinite(r) or r < zero:
        status[i] = 1
        return
    for k in range(5):
        column = k
        if k >= 3:
            column = k + 1
        efficiency = efficiencies[i, column]
        if not wp.isfinite(efficiency) or efficiency < zero:
            status[i] = 1
            return
        value = efficiency * r * r * _PI
        if not wp.isfinite(value):
            status[i] = 3
            return
        cross_sections[i, k] = value
