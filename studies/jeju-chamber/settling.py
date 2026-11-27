"""Optional vertical sedimentation and an absorbing physical floor."""

import warp as wp


@wp.func
def terminal(d: float, stokes: float, reynolds: float):
    """Use the archived low-Re and drag-corrected terminal velocity."""
    v = stokes * d * d
    for _ in range(3):
        v = (
            stokes
            * d
            * d
            / (1.0 + 0.15 * wp.pow(wp.max(0.0, reynolds * v * d), 0.687))
        )
    return v


@wp.func
def poisson(mean: float, moves: int):
    """Probability of exactly the specified number of vertical cell moves."""
    probability = wp.exp(-mean)
    for i in range(moves):
        probability *= mean / float(i + 1)
    return probability


@wp.func
def transit(
    d: float,
    phase: int,
    case: int,
    run: int,
    k: int,
    dt: int,
    dz: float,
    params: wp.array2d(dtype=float),
    forc: wp.array3d(dtype=float),
):
    factor = params[case, 10]
    if phase == 1:
        factor = 0.917 * params[case, 22]
    stokes = forc[run, k, 7]
    return wp.clamp(
        float(dt)
        / dz
        * factor
        * terminal(
            d,
            stokes,
            forc[run, k, 8],
        ),
        0.0,
        80.0,
    )


@wp.kernel
def sediment(
    state: wp.array3d(dtype=float),
    result: wp.array3d(dtype=float),
    params: wp.array2d(dtype=float),
    forc: wp.array3d(dtype=float),
    k: int,
    dt: int,
    nx: int,
    nz: int,
    nl: int,
    ni: int,
    dz: float,
):
    """Gather surviving sedimenting particles and their conserved moments."""
    job, slot = wp.tid()
    cells = nx * nz
    case = job // (6 * cells)
    run = (job // cells) % 6
    cell = job % cells
    z = cell // nx
    if forc[run, k, 12] < 0.5:
        return
    if slot < nl:
        n = float(0.0)
        v = float(0.0)
        dry = float(0.0)
        activity = float(0.0)
        for move in range(nz - z):
            above = job + move * nx
            wet = state[above, 1, slot]
            fall = transit(
                wp.pow(wet, 0.3333333333), 0, case, run, k, dt, dz, params, forc
            )
            take = state[above, 0, slot] * poisson(fall, move)
            n += take
            v += take * wet
            dry += take * state[above, 2, slot]
            activity += take * state[above, 3, slot]
        result[job, 0, slot] = n
        result[job, 1, slot] = state[job, 1, slot]
        result[job, 2, slot] = state[job, 2, slot]
        result[job, 3, slot] = state[job, 3, slot]
        if n > 0.0:
            result[job, 1, slot] = v / n
            result[job, 2, slot] = dry / n
            result[job, 3, slot] = activity / n
    if slot < ni:
        volume = 0.001 * wp.exp(wp.log(5.12e11) * float(slot) / float(ni - 1))
        fall = transit(
            wp.pow(volume, 0.3333333333), 1, case, run, k, dt, dz, params, forc
        )
        n = float(0.0)
        for move in range(nz - z):
            n += state[job + move * nx, 4, slot] * poisson(fall, move)
        result[job, 4, slot] = n


@wp.kernel
def floor_flux(
    state: wp.array3d(dtype=float),
    book: wp.array2d(dtype=float),
    floor: wp.array2d(dtype=float),
    params: wp.array2d(dtype=float),
    forc: wp.array3d(dtype=float),
    k: int,
    dt: int,
    nx: int,
    nz: int,
    nl: int,
    ni: int,
    dz: float,
):
    """Accumulate the absorbing-floor water and number transfers."""
    job = wp.tid()
    cells = nx * nz
    case = job // (6 * cells)
    run = (job // cells) % 6
    z = (job % cells) // nx
    if forc[run, k, 12] < 0.5:
        return
    water = float(0.0)
    number = float(0.0)
    for j in range(nl):
        wet = state[job, 1, j]
        fall = transit(
            wp.pow(wet, 0.3333333333), 0, case, run, k, dt, dz, params, forc
        )
        survive = float(0.0)
        for move in range(z + 1):
            survive += poisson(fall, move)
        lost = state[job, 0, j] * wp.clamp(1.0 - survive, 0.0, 1.0)
        number += lost
        water += (
            lost * wp.max(0.0, wet - state[job, 2, j]) * 5.235987755982988e-7
        )
    for j in range(ni):
        volume = 0.001 * wp.exp(wp.log(5.12e11) * float(j) / float(ni - 1))
        fall = transit(
            wp.pow(volume, 0.3333333333), 1, case, run, k, dt, dz, params, forc
        )
        survive = float(0.0)
        for move in range(z + 1):
            survive += poisson(fall, move)
        lost = state[job, 4, j] * wp.clamp(1.0 - survive, 0.0, 1.0)
        number += lost
        water += lost * volume * 0.917 * 5.235987755982988e-7
    book[job, 5] += water
    book[job, 7] += number
    floor[job, 0] += water
    floor[job, 1] += number
