"""Higher-precision conservative transport for the one-second refinement."""

import warp as wp


@wp.kernel
def particle_mix(
    original: wp.array3d(dtype=float),
    mixed: wp.array3d(dtype=float),
    weights: wp.array3d(dtype=float),
    nl: int,
    ni: int,
    cells: int,
):
    job, slot = wp.tid()
    cell = job % cells
    offset = job - cell
    case = job // (cells * 6)
    if slot < nl:
        count = wp.float64(original[job, 0, slot])
        vol = count * wp.float64(original[job, 1, slot])
        dry = count * wp.float64(original[job, 2, slot])
        act = count * wp.float64(original[job, 3, slot])
        n = count
        v = vol
        c = dry
        a = act
        for j in range(cells):
            w = wp.float64(weights[case, cell, j])
            other = wp.float64(original[offset + j, 0, slot])
            n += w * (other - count)
            v += w * (other * wp.float64(original[offset + j, 1, slot]) - vol)
            c += w * (other * wp.float64(original[offset + j, 2, slot]) - dry)
            a += w * (other * wp.float64(original[offset + j, 3, slot]) - act)
        mixed[job, 0, slot] = float(n)
        if n > wp.float64(0.0):
            mixed[job, 1, slot] = float(v / n)
            mixed[job, 2, slot] = float(c / n)
            mixed[job, 3, slot] = float(a / n)
        else:
            for q in range(1, 4):
                mixed[job, q, slot] = original[job, q, slot]
    if slot < ni:
        initial = wp.float64(original[job, 4, slot])
        ice = initial
        for j in range(cells):
            ice += wp.float64(weights[case, cell, j]) * (
                wp.float64(original[offset + j, 4, slot]) - initial
            )
        mixed[job, 4, slot] = float(ice)


@wp.kernel
def air_mix(
    original: wp.array2d(dtype=float),
    mixed: wp.array2d(dtype=float),
    weights: wp.array3d(dtype=float),
    cells: int,
):
    job, field = wp.tid()
    cell = job % cells
    base = wp.float64(original[job, field])
    total = base
    if field != 7:
        for j in range(cells):
            total += wp.float64(weights[job // (6 * cells), cell, j]) * (
                wp.float64(original[job - cell + j, field]) - base
            )
    mixed[job, field] = float(total)
