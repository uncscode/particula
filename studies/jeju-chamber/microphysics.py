"""Size-resolved, pumped chamber with finite vapor and linked boundary budgets."""

import warp as wp
from pairs import merge_pairs, wall_rate

# Parameters shared across the six fills, followed by six CCN scales and six
# potential ice-active abundances at the beginning of the pumpdown.
BOUNDS = {
    "ccn_mean": (350, 2250),
    "ccn_gmd": (0.055, 0.26),
    "ccn_gsd": (1.25, 2.4),
    "exposure_spread": (0.65, 1.35),
    "water_response_s": (18, 270),
    "warm_sensor_bias": (-0.08, 0.18),
    "warm_bias_slope": (0, 0.17),
    "cold_initial_rh": (0.55, 0.83),
    "cold_vapor_excess": (0.45, 1.30),
    "wall_loss_min": (0, 0.05),
    "liquid_settling": (0.1, 1.2),
    "salt_number": (0.08, 9),
    "salt_gmd": (12, 35),
    "salt_gsd": (1.15, 1.8),
    "salt_clock_e5_s": (-30, 60),
    "background_mid": (12, 26),
    "background_width": (0.75, 5.5),
    "salt_mid": (5, 21),
    "salt_width": (0.75, 6),
    "delay_min": (1, 24),
    "delay_shape": (1, 8),
    "ice_growth": (0.20, 1.65),
    "ice_settling": (0.18, 1.35),
    "ice_optical": (0.65, 1.35),
    "ice_length": (0.95, 2.2),
    "collection": (0.12, 1.5),
    "riming": (0.12, 1.5),
    "cpi_selection": (0, 1.6),
    **{f"ccn_scale_{i}": (0.5, 1.7) for i in range(1, 7)},
    **{f"active_{i}": (0.02, 80) for i in range(1, 7)},
    "particle_exhaust_fraction": (0.25, 1.0),
}
KEYS = list(BOUNDS)
# Process switches: salt addition; background freeze; enhancement; Weibull delay;
# ice deposition; gravitational settling; warm collection; riming; pressure proxy.
FULL = (1, 1, 1, 1, 1, 1, 1, 1, 0, 0)
MODES = {
    "full": FULL,
    "immediate": (1, 1, 1, 0, 1, 1, 1, 1, 0, 0),
    "no_salt": (0, 1, 0, 0, 1, 1, 1, 1, 0, 0),
    "no_salt_liquid": (0, 1, 1, 1, 1, 1, 1, 1, 0, 0),
    "no_background": (1, 0, 1, 1, 1, 1, 1, 1, 0, 0),
    "background_only": (1, 1, 0, 0, 1, 1, 1, 1, 0, 0),
    "no_deposition": (1, 1, 1, 1, 0, 1, 1, 1, 0, 0),
    "no_settling": (1, 1, 1, 1, 1, 0, 1, 1, 0, 0),
    "no_coalescence": (1, 1, 1, 1, 1, 1, 0, 1, 0, 0),
    "no_riming": (1, 1, 1, 1, 1, 1, 1, 0, 0, 0),
    "no_collection": (1, 1, 1, 1, 1, 1, 0, 0, 0, 0),
    "no_deposition_or_settling": (1, 1, 1, 1, 0, 0, 1, 1, 0, 0),
    "pressure_diagnostic": (1, 1, 1, 0, 1, 1, 1, 1, 1, 0),
    "well_mixed_pump": (1, 1, 1, 1, 1, 1, 1, 1, 0, 1),
    "no_ice_growth": (1, 1, 1, 1, 0, 1, 1, 0, 0, 0),
}


@wp.func
def power3(x: float):
    return x * x * x


@wp.func
def cdf(x: float):
    # Approximate normal CDF used by the fitting kernel, <8e-8 absolute.
    t = 1.0 / (1.0 + 0.2316419 * wp.abs(x))
    p = (
        0.39894228
        * wp.exp(-0.5 * x * x)
        * t
        * (
            0.31938153
            + t
            * (
                -0.356563782
                + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))
            )
        )
    )
    if x >= 0.0:
        return 1.0 - p
    return p


@wp.func
def collect_index(d: float):
    j = int(0)
    if d >= 15.0:
        j = 1
    if d >= 25.0:
        j = 2
    if d >= 40.0:
        j = 3
    if d >= 60.0:
        j = 4
    if d >= 80.0:
        j = 5
    if d >= 120.0:
        j = 6
    if d >= 200.0:
        j = 7
    return j


@wp.func
def place_ice(
    number: float,
    volume: float,
    job: int,
    col: int,
    size: int,
    lo: float,
    step: float,
    dst: wp.array3d(dtype=float),
):
    # Remap via particle volume. Linear weights conserve mass and number.
    h = wp.log(wp.max(lo, volume) / lo) / step
    j = wp.clamp(int(h), 0, size - 2)
    low = lo * wp.exp(step * float(j))
    high = low * wp.exp(step)
    f = wp.clamp((volume - low) / (high - low), 0.0, 1.0)
    dst[job, col, j] += number * (1.0 - f)
    dst[job, col, j + 1] += number * f


@wp.func
def fall(d: float, stokes: float, dens: float):
    # Native Stokes prefactor with low-Re to inertial drag correction.
    v = stokes * d * d
    for i in range(3):
        v = (
            stokes
            * d
            * d
            / (1.0 + 0.15 * wp.pow(wp.max(0.0, dens * v * d), 0.687))
        )
    return v


@wp.func
def transfer(old: float, dry: float, delta: float):
    return wp.max(
        dry, wp.pow(wp.max(0.0, wp.pow(old, 0.6666666667) + delta), 1.5)
    )


@wp.func
def damp(x: float):
    return (1.0 - wp.exp(-x)) / wp.max(x, 1.0e-18)


@wp.func
def bucket(d: float, lo: float, di: float, nb: int):
    return wp.clamp(int(wp.floor(wp.log(wp.max(d, 1.0e-10) / lo) / di)), -1, nb)


@wp.kernel
def solve(
    p: wp.array2d(dtype=float),
    modes: wp.array2d(dtype=int),
    forc: wp.array3d(dtype=float),
    observed_initial: wp.array2d(dtype=float),
    sizes: wp.array(dtype=float),
    ccn: wp.array(dtype=float),
    saltq: wp.array(dtype=float),
    wall_table: wp.array4d(dtype=float),
    thermal: wp.array4d(dtype=float),
    pair_work: wp.array3d(dtype=float),
    bins: int,
    nproducts: int,
    interval: int,
    log_d_min: float,
    log_d_step: float,
    workspace: wp.array3d(dtype=float),
    out: wp.array3d(dtype=float),
    diag: wp.array3d(dtype=float),
    book: wp.array2d(dtype=float),
    dt: int,
    k: int,
    cells: int,
    config: wp.array2d(dtype=float),
    layout: wp.array2d(dtype=float),
    history: wp.array2d(dtype=float),
    nice: int,
    nb: int,
    optic_lo: float,
    optic_di: float,
    wall_history: wp.array4d(dtype=float),
    uptake: wp.array(dtype=float),
    wall_vapor: wp.array(dtype=float),
    material_input: wp.array3d(dtype=float),
    material_flux: wp.array2d(dtype=float),
):
    job = wp.tid()
    cell = job % cells
    run = (job // cells) % 6
    case = job // (6 * cells)
    if forc[run, k, 12] < 0.5:
        return
    nc = ccn.shape[0]
    a2_start = nc + 3 * nb
    nbig = a2_start + nb + 9
    corrected = run == 4 and p[case, 62] > 0.5
    salt_end = nbig + saltq.shape[0]
    nl = salt_end + nproducts
    lo = float(0.001)
    step = wp.log(800.0 * 800.0 * 800.0 / lo) / float(nice - 1)
    g0 = float(5.235987755982988e-7)
    gi = g0 * 0.917
    vapor = forc[run, 0, 4]
    if run == 5:
        vapor = forc[run, 0, 2] * p[case, 7]
    initial_vapor = vapor
    initial_number = p[case, 0] * p[case, 28 + run]
    source_n = float(0.0)
    source_core = float(0.0)
    pumped_n = float(0.0)
    deposited_n = float(0.0)
    merged_n = float(0.0)
    pumped_w = float(0.0)
    deposited_w = float(0.0)
    external_w = float(0.0)
    ice_cut_error = float(0.0)
    bgprev = float(0.0)
    saltprev = float(0.0)
    drop_lost = float(0.0)
    ice_lost = float(0.0)
    for j in range(nb):
        if corrected:
            initial_number += material_input[case, 0, j]
        else:
            initial_number += observed_initial[run, j]
    if corrected:
        for j in range(nb + 9):
            initial_number += material_input[case, 1, j]
    if k == 0:
        for j in range(nl):
            d = float(1.0)
            n = float(0.0)
            if j < nc:
                d = p[case, 1] * wp.exp(wp.log(p[case, 2]) * ccn[j])
                if p[case, 47] > 0.5:
                    d *= 2.2 if j % 4 == 0 else 0.7
                n = p[case, 0] * p[case, 28 + run] / float(nc)
            elif j < nc + nb:
                d = sizes[j - nc]
                n = observed_initial[run, j - nc]
                if corrected:
                    n = material_input[case, 0, j - nc]
            elif j >= a2_start and j < nbig:
                if corrected:
                    d = material_input[case, 2, j - a2_start]
                    n = material_input[case, 1, j - a2_start]
                    material_flux[job, 0] += n * power3(d)
                    material_flux[job, 5] += n
            elif j < nbig:
                d = sizes[(j - nc - nb) % nb]
            elif j < salt_end:
                d = p[case, 12] * wp.exp(wp.log(p[case, 13]) * saltq[j - nbig])
            workspace[job, 0, j] = n
            workspace[job, 1, j] = power3(d)
            workspace[job, 2, j] = power3(d)
            workspace[job, 3, j] = float(0.0)
            if j >= nc and j < nc + nb and d > 4.0 and p[case, 48] > 0.0:
                dry_volume = workspace[job, 2, j] * (1.0 - p[case, 48])
                external_w += g0 * n * (workspace[job, 1, j] - dry_volume)
                workspace[job, 2, j] = dry_volume
            if j >= nc:
                workspace[job, 3, j] = 1.0 if d > 4.0 else 0.0
    if k == 0 and p[case, 54] > 0.0:
        for j in range(nb):
            if sizes[j] >= 2.0 and sizes[j] < 10.0:
                moved = p[case, 54] * workspace[job, 0, nc + j]
                workspace[job, 0, nc + j] -= moved
                workspace[job, 0, nc + 2 * nb + j] = moved
    inj = float(960.0)
    if run == 4:
        inj += p[case, 14]
    if run == 0:
        inj = 1.0e9
    pop_prev = initial_number
    cold_mix = initial_vapor / forc[run, 0, 1]
    min_temp = float(0.0)
    native_eddy = wp.pow(10.0, p[case, 41])
    epsilon = wp.pow(10.0, p[case, 42])
    dry_transferred = float(0.0)
    limited_steps = float(0.0)
    collision_thermal = float(0.0)
    collision_sedimentation = float(0.0)
    collision_shear = float(0.0)
    max_demand = float(0.0)
    outside_collisions = float(0.0)
    local_dose = 1.0 + config[case, 3] * (layout[cell, 1] - 1.0)
    tracer = float(0.0)
    contact = float(0.0)
    if k > 0:
        vapor = history[job, 0]
        min_temp = history[job, 2]
        bgprev = history[job, 3]
        saltprev = history[job, 4]
        tracer = history[job, 5]
        contact = history[job, 6]
        external_w = history[job, 7]
        source_n = book[job, 9]
        source_core = book[job, 10] / (g0 * 2.16)
        pumped_w = book[job, 4]
        deposited_w = book[job, 5]
        pumped_n = book[job, 6]
        drop_lost = float(0.0)
        merged_n = book[job, 8]
        ice_cut_error = book[job, 11]
        dry_transferred = book[job, 12] / g0
        max_demand = book[job, 13]
        limited_steps = book[job, 14]
        collision_thermal = book[job, 15]
        collision_sedimentation = book[job, 16]
        collision_shear = book[job, 17]
        outside_collisions = book[job, 18]
    pop_prev = float(0.0)
    for j in range(nl):
        pop_prev += workspace[job, 0, j]
    for j in range(nice):
        pop_prev += workspace[job, 4, j]
    minute = k // 60
    T = forc[run, k, 0]
    density = forc[run, k, 1]
    vw = forc[run, k, 2]
    vi = forc[run, k, 3]
    raw = forc[run, k, 4]
    co = forc[run, k, 5]
    ico = forc[run, k, 6]
    stokes = forc[run, k, 7]
    reynolds = forc[run, k, 8]
    pump = forc[run, k, 9]
    cutoff = forc[run, k, 10]
    T_ref = T
    vw_ref = vw
    vi_ref = vi
    T = T_ref + wall_history[case, run, k // dt, cell]
    reciprocal = 1.0 / T_ref - 1.0 / T
    vw *= wp.exp(5418.0 * reciprocal) * T_ref / T
    vi *= wp.exp(6140.0 * reciprocal) * T_ref / T
    co *= vw / vw_ref
    co *= p[case, 45]
    ico *= vi / vi_ref
    cutoff = 273.15 - T
    # Ratio of dry-air mass between the two accepted states. Air leaves the
    # pumped volume carrying its current concentrations.
    if k > 0:
        pump = wp.clamp(density / forc[run, k - dt, 1], 0.0, 1.0)
    else:
        pump = 1.0
    particle_pump = wp.pow(pump, p[case, 40])
    if modes[case, 9] != 0:
        particle_pump = pump
    pumped_n += (1.0 - particle_pump) * pop_prev
    pumped_w += (1.0 - pump) * vapor
    vapor *= pump
    tracer *= particle_pump
    if float(k) >= inj and float(k - dt) < inj:
        tracer += local_dose
    contact += float(dt) * wp.clamp(tracer, 0.0, 1.0)
    if run == 5:
        raw = wp.min(cold_mix * density, vw_ref)
        if T_ref < 273.15 and cold_mix * density >= vw_ref:
            raw = vi_ref + p[case, 8] * (vw_ref - vi_ref)
            raw = wp.max(raw, 0.995 * vw_ref)
    else:
        bias = p[case, 5] + p[case, 6] * (1.0 - density / forc[run, 0, 1])
        raw = raw / (1.0 + bias)
    add = (raw - vapor) * (1.0 - wp.exp(-float(dt) / p[case, 4]))
    active = float(k) >= p[case, 49]
    if active and (
        p[case, 57] != 0.0 or p[case, 58] != 0.0 or p[case, 61] != 1.0
    ):
        pattern = 1.0 if cell < 6 else -0.5
        target = raw * (1.0 + p[case, 57] * pattern + p[case, 58])
        add = (
            (target - vapor)
            * (1.0 - wp.exp(-float(dt) / p[case, 4]))
            * p[case, 61]
        )
    external_w += add
    vapor += add
    saturation_wall = wall_history[case, run, k // dt, cells + cell]
    transfer_rate = wall_history[case, run, k // dt, 2 * cells + cell]
    efficiency = uptake[case]
    if active:
        efficiency = p[case, 59]
    wall_water = wp.max(0.0, vapor - saturation_wall) * (
        1.0 - wp.exp(-float(dt) * transfer_rate * efficiency)
    )
    release = float(0.0)
    if active and p[case, 60] > 0.0:
        release = wp.min(
            wall_vapor[job],
            wp.max(0.0, saturation_wall - vapor)
            * (1.0 - wp.exp(-float(dt) * transfer_rate * efficiency)),
        )
    wall_water -= release
    vapor -= wall_water
    deposited_w += wall_water
    wall_vapor[job] += wall_water
    if float(k) >= p[case, 49]:
        co *= p[case, 50]
    lsum = float(0.0)
    isum = float(0.0)
    pump_liq = float(0.0)
    pump_ice = float(0.0)
    total_num = float(0.0)
    for j in range(nl):
        n = workspace[job, 0, j]
        v = workspace[job, 1, j]
        dry = workspace[job, 2, j]
        pump_liq += g0 * n * wp.max(0.0, v - dry) * (1.0 - particle_pump)
        if j >= a2_start and j < nbig:
            material_flux[job, 1] += n * (1.0 - particle_pump) * dry
        n *= particle_pump
        if (
            modes[case, 0] != 0
            and (j >= nbig and j < salt_end)
            and float(k) >= inj
            and float(k - dt) < inj
        ):
            added = local_dose * p[case, 11] / float(saltq.shape[0])
            n += added
            source_n += added
            source_core += added * dry
        workspace[job, 0, j] = n
        d = wp.pow(v, 0.33333333333)
        enabled = workspace[job, 3, j]
        if cells > 1 and v > 1.25 * dry:
            enabled = 1.0
        if enabled < 0.5 and n > 0.0:
            dry_d = wp.pow(dry, 0.33333333333)
            crit = p[case, 46] * 0.0038 * wp.pow(0.10 / dry_d, 1.5)
            if j >= nbig and j < salt_end:
                crit = -0.10
            if (vapor / vw - 1.0) >= crit:
                enabled = 1.0
        if j >= nc + 2 * nb and (
            j < a2_start or (j < nbig and p[case, 63] < 0.5)
        ):
            enabled = 0.0
        workspace[job, 3, j] = enabled
        lsum += n * d * enabled
    for j in range(nice):
        n = workspace[job, 4, j]
        v = lo * wp.exp(step * float(j))
        pump_ice += gi * n * v * (1.0 - particle_pump)
        n *= particle_pump
        workspace[job, 4, j] = n
        isum += n * wp.pow(v, 0.33333333333)
        workspace[job, 5, j] = 0.0
    pumped_w += pump_liq + pump_ice
    # Effective integration of a well-mixed vapor relaxation within the
    # step, with a common supply cap and closed interphase transfers.
    kw = 1.5 * g0 * co * lsum / vw
    if modes[case, 4] == 0:
        ico = 0.0
    ico *= p[case, 21]
    ki = 1.5 * gi * ico * isum / vi
    time = float(dt)
    factor = damp(wp.max((kw + ki) * time, 0.00001))
    dew = (vapor - vw) * factor
    frost = (vapor - vi) * factor
    dw = co * dew / vw * time
    di = ico * frost / vi * time
    if kw + ki > 1.0e-10:
        dmean = (kw * vw + ki * vi) / (kw + ki)
        correction = (dmean - vw) * (1.0 - factor)
        dw += co * correction / vw * time
        di += ico * (dmean - vi) * (1.0 - factor) / vi * time
    # Effective saturation offset for small soluble drops and giant salt.
    salt_collision = float(0.0)
    ice_collision = float(0.0)
    for j in range(nl):
        n = workspace[job, 0, j]
        vol = workspace[job, 1, j]
        d = wp.pow(vol, 0.33333333333)
        if j >= nbig and j < salt_end:
            salt_collision += (
                0.78539816e-6 * n * d * d * fall(d, stokes, reynolds)
            )
    for j in range(nice):
        d = wp.pow(lo * wp.exp(step * float(j)), 0.33333333333)
        ice_collision += (
            0.78539816e-6
            * workspace[job, 4, j]
            * d
            * d
            * fall(d, stokes, reynolds)
        )
    salt_collision *= float(modes[case, 6]) * p[case, 25]
    ice_collision *= float(modes[case, 7]) * p[case, 26]
    grow_w = float(0.0)
    grow_i = float(0.0)
    for j in range(nl):
        v = workspace[job, 1, j]
        dry = workspace[job, 2, j]
        nudged = dw
        if j < nc:
            spread = wp.max(0.3, 1.0 + (p[case, 3] - 1.0) * ccn[j] * 0.55)
            if dw > 0.0:
                nudged *= spread
        elif (j >= nbig and j < salt_end) and float(k) >= inj:
            nudged += (
                co
                * wp.min(0.04, 0.65 * dry / wp.max(v, 1.0e-9))
                * time
                * wp.max(0.0, (vapor / vw - 0.75) / 0.25)
            )
        if workspace[job, 3, j] > 0.5 and workspace[job, 0, j] > 0.0:
            if float(k) >= p[case, 49]:
                nudged *= p[case, 55] if nudged > 0.0 else p[case, 56]
            target = transfer(v, dry, nudged)
            workspace[job, 6, j] = target
            grow_w += g0 * workspace[job, 0, j] * (target - v)
        else:
            workspace[job, 6, j] = v
    for j in range(nice):
        v = lo * wp.exp(step * float(j))
        target = wp.min(
            transfer(v, lo, di), lo * wp.exp(step * float(nice - 1))
        )
        workspace[job, 7, j] = target
        grow_i += gi * workspace[job, 4, j] * (target - v)
    use = wp.max(0.0, grow_w) + wp.max(0.0, grow_i)
    cap = wp.min(1.0, vapor / wp.max(use, 1.0e-15))
    vapor = wp.max(
        0.0,
        vapor
        - (
            wp.min(grow_w, 0.0)
            + wp.max(grow_w, 0.0) * cap
            + wp.min(grow_i, 0.0)
            + wp.max(grow_i, 0.0) * cap
        ),
    )
    min_temp = wp.max(min_temp, wp.min(cutoff, 38.0))
    cold = wp.max(0.0, cutoff)
    B = float(0.0)
    S = float(0.0)
    if cold > 0.0:
        if modes[case, 1] != 0:
            B = cdf((min_temp - p[case, 15]) / p[case, 16])
        if modes[case, 2] != 0 and float(k) >= inj:
            # Only the ambient-air experiments share the salt response.
            salt_temp = min_temp
            if modes[case, 8] != 0:
                salt_temp = wp.max(
                    0.0, (1.0 - density / forc[run, 960, 1]) * 25.0
                )
            S = cdf((salt_temp - p[case, 17]) / p[case, 18])
            if modes[case, 3] != 0:
                age = (float(k) - inj) / 60.0
                age = wp.lerp(age, contact / 60.0, config[case, 3])
                S *= 1.0 - wp.exp(
                    -0.69314718 * wp.pow(age / p[case, 19], p[case, 20])
                )
            access = wp.clamp(
                1.15 * tracer / wp.max(0.15, tracer + 0.15), 0.0, 1.0
            )
            S *= wp.lerp(1.0, access, config[case, 3])
            if run == 3 or run == 4:
                S *= 0.01
    frac = p[case, 34 + run] / (p[case, 0] * p[case, 28 + run])
    # Competing recruitment applied to the current eligible background
    # droplets. The amplitude is bounded by the available population.
    df = wp.max(0.0, B - bgprev) * frac + wp.max(0.0, S - saltprev) * frac
    bgprev = wp.max(B, bgprev)
    saltprev = wp.max(S, saltprev)
    coalesced = float(0.0)
    rimed = float(0.0)
    salt_total = float(0.0)
    for j in range(nl):
        if j >= nbig and j < salt_end:
            salt_total += workspace[job, 0, j]
    # First compute collision water available to each collector class.
    for j in range(nbig):
        v0 = workspace[job, 1, j]
        vc = workspace[job, 6, j]
        v = v0 + (vc - v0) * (cap if grow_w > 0.0 else 1.0)
        d = wp.pow(v, 0.33333333333)
        dry = workspace[job, 2, j]
        if d >= 3.0 and v > 1.25 * dry:
            n = workspace[job, 0, j]
            loss = 1.0 - wp.exp(-time * (salt_collision + ice_collision))
            dn = n * loss
            hit = (
                dn
                * salt_collision
                / wp.max(1.0e-30, salt_collision + ice_collision)
            )
            if j >= a2_start:
                material_flux[job, 3] += dn * dry
            coalesced += hit * (v - dry)
            rimed += (dn - hit) * (v - dry)
            merged_n += dn
            workspace[job, 0, j] = n - dn
    for j in range(nl):
        n = workspace[job, 0, j]
        old = workspace[job, 1, j]
        next = workspace[job, 6, j]
        dry = workspace[job, 2, j]
        v = wp.max(dry, old + (next - old) * (cap if grow_w > 0.0 else 1.0))
        if j >= nbig and j < salt_end:
            v += coalesced / wp.max(1.0e-20, salt_total)
        d = wp.pow(v, 0.33333333333)
        frozen = float(0.0)
        if (
            (j < nc or (j >= nc + nb and j < nbig) or j >= salt_end)
            and d > 2.5
            and v > 1.25 * dry
        ):
            frozen = n * wp.min(df, 1.0)
            if frozen > 0.0:
                place_ice(
                    frozen,
                    (v - dry) / 0.917,
                    job,
                    5,
                    nice,
                    lo,
                    step,
                    workspace,
                )
        if j >= a2_start and j < nbig:
            material_flux[job, 4] += frozen * dry
        n -= frozen
        loss = p[case, 9] / 60.0
        if modes[case, 5] != 0:
            loss += p[case, 10] * fall(d, stokes, reynolds) / 3.0
        if modes[case, 10] != 0:
            material = 0
            if v <= 1.25 * dry and j >= nc:
                material = 1
            loss = wall_rate(
                wall_table,
                run,
                minute,
                d,
                material,
                native_eddy,
                log_d_min,
                log_d_step,
                bins,
            )
        if float(k) >= p[case, 49]:
            loss *= p[case, 51]
        remaining = n * wp.exp(-time * loss)
        if j >= a2_start and j < nbig:
            material_flux[job, 2] += (n - remaining) * dry
        drop_lost += n - remaining
        deposited_w += g0 * (n - remaining) * (v - dry)
        workspace[job, 1, j] = v
        workspace[job, 0, j] = remaining
        total_num += remaining
        optical = d
        if j >= a2_start and j < nbig and corrected:
            optical = wp.pow(
                wp.max(
                    0.0,
                    v
                    - dry
                    + dry * power3(material_input[case, 3, j - a2_start]),
                ),
                0.33333333333,
            )
        jbin = bucket(optical, optic_lo, optic_di, nb)
        if j >= a2_start and j < nbig and jbin < 11:
            jbin = -1
        if jbin >= 0 and jbin < nb:
            channel = int(0)
            if v > 1.25 * dry:
                channel = 1
            if j >= nbig and j < salt_end:
                channel = 2
            out[job, minute, jbin + channel * nb] += remaining * time / 60.0
        if d >= 10.0:
            c = collect_index(d)
            selection = wp.pow(wp.clamp(d / 20.0, 0.45, 4.0), p[case, 27])
            diag[job, minute, c] += remaining * selection * time / 60.0
    for j in range(nice):
        n = workspace[job, 4, j]
        old = lo * wp.exp(step * float(j))
        v = old + (workspace[job, 7, j] - old) * (cap if grow_i > 0.0 else 1.0)
        if rimed > 0.0:
            d = wp.pow(old, 0.33333333333)
            speed = fall(d, stokes, reynolds)
            # Fixed collecting area × speed is the same weighting used
            # for removing droplets. Liquid mass becomes ice volume.
            v += (
                rimed
                / 0.917
                * 0.78539816e-6
                * d
                * d
                * speed
                * float(modes[case, 7])
                * p[case, 26]
                / wp.max(1.0e-20, ice_collision)
            )
        d = wp.pow(v, 0.33333333333)
        loss = p[case, 9] / 60.0
        if modes[case, 5] != 0:
            loss += p[case, 22] * fall(d, stokes, reynolds) * 0.917 / 3.0
        if modes[case, 10] != 0:
            loss = wall_rate(
                wall_table,
                run,
                minute,
                d,
                2,
                native_eddy,
                log_d_min,
                log_d_step,
                bins,
            )
        if float(k) >= p[case, 49]:
            loss *= p[case, 51]
        rem = n * wp.exp(-time * loss)
        ice_lost += n - rem
        deposited_w += gi * (n - rem) * v
        if rem > 0.0:
            limited = wp.clamp(v, lo, lo * wp.exp(step * float(nice - 1)))
            ice_cut_error += gi * rem * (v - limited)
            place_ice(rem, limited, job, 5, nice, lo, step, workspace)
    lwater = float(0.0)
    iwater = float(0.0)
    for j in range(nl):
        lwater += (
            g0
            * workspace[job, 0, j]
            * (workspace[job, 1, j] - workspace[job, 2, j])
        )
    for j in range(nice):
        n = workspace[job, 5, j]
        v = lo * wp.exp(step * float(j))
        workspace[job, 4, j] = n
        d = wp.pow(v, 0.33333333333)
        total_num += n
        iwater += n * v * gi
        jbin = bucket(d * p[case, 23], optic_lo, optic_di, nb)
        if jbin >= 0 and jbin < nb:
            out[job, minute, jbin + 3 * nb] += n * time / 60.0
        dp = d * p[case, 24]
        if dp >= 10.0:
            c = collect_index(dp)
            selection = wp.pow(wp.clamp(dp / 20.0, 0.45, 4.0), p[case, 27])
            diag[job, minute, c] += n * selection * time / 60.0
            diag[job, minute, 8 + c] += n * selection * time / 60.0
    # Unresolved cores left by complete sublimation remain in the model
    # particle count. Their small-volume grid floor is tiny.
    f = time / 60.0
    diag[job, minute, 16] += vapor * f
    diag[job, minute, 17] += lwater * f
    diag[job, minute, 18] += iwater * f
    diag[job, minute, 19] += total_num * f
    if (
        nproducts > 0
        and (k + dt) % interval == 0
        and (float(k) < p[case, 49] or p[case, 52] > 0.5)
    ):
        pre = float(0.0)
        for j in range(a2_start, nbig):
            pre += workspace[job, 0, j] * workspace[job, 2, j]
        values = merge_pairs(
            workspace,
            pair_work,
            wall_table,
            thermal,
            job,
            run,
            minute,
            T,
            density * 287.05 * T,
            reynolds,
            modes[case, 11],
            modes[case, 12],
            modes[case, 13],
            p[case, 44],
            p[case, 43],
            epsilon,
            log_d_min,
            log_d_step,
            bins,
            nl,
            salt_end,
            nc,
            nice,
            lo,
            step,
            float(interval),
        )
        post = float(0.0)
        for j in range(a2_start, nbig):
            post += workspace[job, 0, j] * workspace[job, 2, j]
        material_flux[job, 3] += pre - post
        merged_n += values[0]
        total_num -= values[0]
        lwater -= values[1] * g0
        iwater += values[1] * g0 - values[3] * gi
        dry_transferred += values[2]
        ice_cut_error += values[3] * gi
        max_demand = wp.max(max_demand, values[4])
        limited_steps += values[5]
        collision_thermal += values[6]
        collision_sedimentation += values[7]
        collision_shear += values[8]
        outside_collisions += values[9]
    material_flux[job, 6] = 0.0
    material_flux[job, 7] = 0.0
    for j in range(a2_start, nbig):
        material_flux[job, 6] += workspace[job, 0, j] * workspace[job, 2, j]
        material_flux[job, 7] += workspace[job, 0, j]
    pop_prev = total_num
    deposited_n = drop_lost + ice_lost
    if k > 0:
        deposited_n += book[job, 7]
    # Current-state inventories and fluxes. Last accepted step is retained.
    book[job, 0] = initial_vapor + external_w
    book[job, 1] = (
        vapor + lwater + iwater + pumped_w + deposited_w + ice_cut_error
    )
    book[job, 2] = initial_number + source_n
    book[job, 3] = total_num + pumped_n + deposited_n + merged_n
    book[job, 4] = pumped_w
    book[job, 5] = deposited_w
    book[job, 6] = pumped_n
    book[job, 7] = deposited_n
    book[job, 8] = merged_n
    book[job, 9] = source_n
    book[job, 10] = source_core * g0 * 2.16
    book[job, 11] = ice_cut_error
    book[job, 12] = dry_transferred * g0
    book[job, 13] = max_demand
    book[job, 14] = limited_steps
    book[job, 15] = collision_thermal
    book[job, 16] = collision_sedimentation
    book[job, 17] = collision_shear
    book[job, 18] = outside_collisions
    history[job, 0] = vapor
    history[job, 2] = min_temp
    history[job, 3] = bgprev
    history[job, 4] = saltprev
    history[job, 5] = tracer
    history[job, 6] = contact
    history[job, 7] = external_w
    diag[job, minute, 20] += (vapor / vw - 1.0) * f
    diag[job, minute, 21] += wp.max(0.0, wall_water)
    diag[job, minute, 22] += release
    diag[job, minute, 23] += add
