"""Couple native thermal encounter tables to moving cohorts and ice bins."""

import warp as wp
from warp.types import vector

Counts = vector(length=10, dtype=float)


@wp.func
def lookup(d: float, lower: float, step: float, size: int):
    return wp.clamp(
        (wp.log(wp.max(d, 1.0e-12)) - lower) / step,
        0.0,
        float(size - 1) - 1.0e-5,
    )


@wp.func
def interpolate(
    table: wp.array4d(dtype=float),
    run: int,
    minute: int,
    channel: int,
    x: float,
):
    i = int(x)
    return wp.lerp(
        table[run, minute, channel, i],
        table[run, minute, channel, i + 1],
        x - float(i),
    )


@wp.func
def wall_rate(
    table: wp.array4d(dtype=float),
    run: int,
    minute: int,
    d: float,
    material: int,
    eddy: float,
    lower: float,
    bin_step: float,
    size: int,
):
    x = lookup(d, lower, bin_step, size)
    diffusion = interpolate(table, run, minute, 0, x)
    velocity = interpolate(table, run, minute, material + 1, x)
    scale = wp.sqrt(eddy * diffusion)
    # Crump-Seinfeld rectangular coefficient. Equal horizontal sides preserve
    # the 22.4 m³ inner volume at the documented 3 m height.
    side = float(2.7325202042558927)
    x = 0.7853981633974483 * velocity / scale
    settling = 4.0 * scale / 3.141592653589793
    if x > 1.0e-3:
        settling = velocity / wp.tanh(x)
    return (8.0 * scale / (3.141592653589793 * side)) + settling / 3.0


@wp.func
def pair_lookup(
    table: wp.array4d(dtype=float),
    state: int,
    term: int,
    i: int,
    j: int,
    fx: float,
    fy: float,
):
    a = wp.lerp(table[state, term, i, j], table[state, term, i, j + 1], fy)
    b = wp.lerp(
        table[state, term, i + 1, j], table[state, term, i + 1, j + 1], fy
    )
    return wp.lerp(a, b, fx)


@wp.func
def liquid_product(
    scratch: wp.array3d(dtype=float),
    job: int,
    idx: int,
    count: float,
    volume: float,
    core: float,
):
    scratch[job, 6, idx] += count
    scratch[job, 7, idx] += count * volume
    scratch[job, 8, idx] += count * core


@wp.func
def merge_pairs(
    state: wp.array3d(dtype=float),
    scratch: wp.array3d(dtype=float),
    wall: wp.array4d(dtype=float),
    thermal: wp.array4d(dtype=float),
    job: int,
    run: int,
    m: int,
    temperature: float,
    pressure: float,
    re: float,
    therm_mode: int,
    grav_switch: int,
    shear_switch: int,
    ham: float,
    efficiency: float,
    epsilon: float,
    lower: float,
    dlog: float,
    size: int,
    nl: int,
    salt_end: int,
    nc: int,
    ni: int,
    ice_lo: float,
    ice_step: float,
    interval: float,
):
    nslots = nl + ni
    nprod = nl - salt_end
    high = ice_lo * wp.exp(ice_step * float(ni - 1))
    pair_t = wp.clamp(ham + 21.0, 0.0, 1.999999)
    ham_i = int(pair_t) + 1
    ham_f = pair_t - float(int(pair_t))
    tstate = run * wall.shape[1] + m
    # For cumulatively estimated air/liquid-water properties used by the shear kernel.
    shear_coef = (
        wp.sqrt(3.141592653589793 * epsilon * re * 1.0e6 / 120.0) * 1.0e-12
    )
    for i in range(nslots):
        mat = int(0)
        n = float(0.0)
        v = float(0.0)
        core = float(0.0)
        if i < nl:
            n = state[job, 0, i]
            v = state[job, 1, i]
            core = state[job, 2, i]
            if v <= 1.25 * core and i >= nc:
                mat = 1
        else:
            n = state[job, 4, i - nl]
            v = ice_lo * wp.exp(ice_step * float(i - nl))
            mat = 2
        d = wp.pow(v, 0.3333333333333)
        q = lookup(d, lower, dlog, size)
        scratch[job, 0, i] = n
        scratch[job, 1, i] = v
        scratch[job, 2, i] = core
        scratch[job, 3, i] = q + float(mat * size)
        scratch[job, 4, i] = interpolate(wall, run, m, mat + 4, q)
        scratch[job, 5, i] = 0.0
        scratch[job, 6, i] = 0.0
        scratch[job, 7, i] = 0.0
        scratch[job, 8, i] = 0.0
        scratch[job, 9, i] = d
    n_merge = float(0.0)
    water_to_ice = float(0.0)
    core_to_ice = float(0.0)
    clipped_volume = float(0.0)
    demand_max = float(0.0)
    limited = float(0.0)
    sum_thermal = float(0.0)
    sum_gravity = float(0.0)
    sum_shear = float(0.0)
    outside_thermal = float(0.0)
    for i in range(nslots):
        n1 = scratch[job, 0, i]
        if n1 <= 0.0:
            continue
        vol1 = scratch[job, 1, i]
        dry1 = scratch[job, 2, i]
        diam1 = scratch[job, 9, i]
        speed1 = scratch[job, 4, i]
        pos1 = scratch[job, 3, i]
        ti = int(pos1)
        fx = pos1 - float(ti)
        for j in range(i, nslots):
            n2 = scratch[job, 0, j]
            if n2 <= 0.0:
                continue
            diam2 = scratch[job, 9, j]
            pos2 = scratch[job, 3, j]
            tj = int(pos2)
            fy = pos2 - float(tj)
            kt = float(0.0)
            if therm_mode == 1:
                kt = pair_lookup(thermal, tstate, 0, ti, tj, fx, fy)
            outside = wp.max(diam1, diam2) > 32.0 * wp.min(diam1, diam2)
            if therm_mode == 2 and not outside:
                a = pair_lookup(thermal, tstate, ham_i, ti, tj, fx, fy)
                b = pair_lookup(thermal, tstate, ham_i + 1, ti, tj, fx, fy)
                kt = wp.lerp(a, b, ham_f)
            if therm_mode == 2 and outside:
                kt = pair_lookup(thermal, tstate, 0, ti, tj, fx, fy)
            if therm_mode == 3:
                kt = pair_lookup(
                    thermal, tstate, 0 if outside else 4, ti, tj, fx, fy
                )
            width = diam1 + diam2
            kg = (
                0.7853981633974483e-6
                * width
                * width
                * wp.abs(speed1 - scratch[job, 4, j])
                * efficiency
                * float(grav_switch)
            )
            ks = (
                shear_coef
                * width
                * width
                * width
                * efficiency
                * float(shear_switch)
            )
            rate = kt + kg + ks
            if rate <= 0.0:
                continue
            proposed = n1 * n2 * rate * interval
            cap = wp.min(
                0.20 * n1 - scratch[job, 5, i], 0.20 * n2 - scratch[job, 5, j]
            )
            if i == j:
                cap *= 0.5
                proposed *= 0.5
            events = wp.min(proposed, wp.max(cap, 0.0))
            if events < proposed:
                limited = 1.0
            if events <= 0.0:
                continue
            scratch[job, 5, i] += events
            scratch[job, 5, j] += events
            n_merge += events
            sum_thermal += events * kt / rate
            if outside:
                outside_thermal += events * kt / rate
            sum_gravity += events * kg / rate
            sum_shear += events * ks / rate
            vol2 = scratch[job, 1, j]
            dry2 = scratch[job, 2, j]
            if j < nl:
                mixed = vol1 + vol2
                mixed_d = wp.pow(mixed, 0.3333333333333)
                idx = wp.clamp(
                    int(
                        lookup(mixed_d, lower, dlog, size)
                        * float(nprod)
                        / float(size - 1)
                    ),
                    0,
                    nprod - 1,
                )
                liquid_product(
                    scratch, job, salt_end + idx, events, mixed, dry1 + dry2
                )
            else:
                volume = vol1 + vol2
                if i < nl:
                    moved = vol1 - dry1
                    water_to_ice += events * moved
                    core_to_ice += events * dry1
                    volume = moved / 0.917 + vol2
                clamped = wp.clamp(volume, ice_lo, high)
                clipped_volume += events * (volume - clamped)
                index = wp.clamp(
                    int(wp.log(clamped / ice_lo) / ice_step), 0, ni - 2
                )
                left = ice_lo * wp.exp(float(index) * ice_step)
                fraction = wp.clamp(
                    (clamped - left) / (left * (wp.exp(ice_step) - 1.0)),
                    0.0,
                    1.0,
                )
                scratch[job, 6, nl + index] += events * (1.0 - fraction)
                scratch[job, 6, nl + index + 1] += events * fraction
    for i in range(nslots):
        n = scratch[job, 0, i]
        gone = scratch[job, 5, i]
        new = scratch[job, 6, i]
        remaining = wp.max(n - gone, 0.0)
        if n > 0.0:
            demand_max = wp.max(demand_max, gone / n)
        if i < nl:
            final = remaining + new
            state[job, 0, i] = final
            if new > 0.0:
                v = (
                    remaining * scratch[job, 1, i] + scratch[job, 7, i]
                ) / final
                core = (
                    remaining * scratch[job, 2, i] + scratch[job, 8, i]
                ) / final
                state[job, 1, i] = wp.max(v, core)
                state[job, 2, i] = core
                if v > 1.25 * core:
                    state[job, 3, i] = 1.0
        else:
            state[job, 4, i - nl] = remaining + new
    return Counts(
        n_merge,
        water_to_ice,
        core_to_ice,
        clipped_volume,
        demand_max,
        limited,
        sum_thermal,
        sum_gravity,
        sum_shear,
        outside_thermal,
    )
