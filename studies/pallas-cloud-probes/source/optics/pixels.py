"""Pixel integration of a focused radial intensity-contrast lookup table."""

import warp as wp


@wp.kernel
def pixel_area(
    profile: wp.array3d(dtype=wp.float32),
    diameter: wp.array(dtype=wp.float32),
    pitch: wp.array(dtype=wp.float32),
    draws: wp.array2d(dtype=wp.float32),
    z_index: wp.array(dtype=wp.int32),
    dr: float,
    threshold: float,
    quadrature: int,
    sizes: wp.array2d(dtype=wp.float32),
):
    d, trial = wp.tid()
    z = z_index[trial]
    p = pitch[z]
    ox = draws[trial, 0] - 0.5
    oy = draws[trial, 1] - 0.5
    # Region of interest encloses the geometrical shadow plus a diffraction halo.
    # Each output counts segmented area belonging to the isolated test target.
    half = int(wp.ceil((diameter[d] * 0.5 + 6.0) / p)) + 1
    area = int(0)
    adjacent = bool(False)
    previous_row = wp.uint64(0)
    for y in range(-half, half + 1):
        row = wp.uint64(0)
        for x in range(-half, half + 1):
            darkness = float(0.0)
            for sy in range(quadrature):
                yy = (
                    float(y) - oy + (float(sy) + 0.5) / float(quadrature) - 0.5
                ) * p
                for sx in range(quadrature):
                    xx = (
                        float(x)
                        - ox
                        + (float(sx) + 0.5) / float(quadrature)
                        - 0.5
                    ) * p
                    ri = wp.sqrt(xx * xx + yy * yy) / dr
                    k = int(wp.floor(ri))
                    if k + 1 < profile.shape[2]:
                        darkness += wp.lerp(
                            profile[d, z, k],
                            profile[d, z, k + 1],
                            ri - float(k),
                        )
            if darkness >= threshold * float(quadrature * quadrature):
                area += 1
                row = row | (wp.uint64(1) << wp.uint64(x + half))
        if (row & (row << wp.uint64(1))) != wp.uint64(0) or (
            row & previous_row
        ) != wp.uint64(0):
            adjacent = True
        previous_row = row
    sizes[d, trial] = float(0.0)
    if adjacent:
        sizes[d, trial] = 2.0 * wp.sqrt(float(area) / 3.14159265359) * p
