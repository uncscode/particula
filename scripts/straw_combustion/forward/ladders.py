"""Reversible cluster ladders coupled to bulk birth and salt growth."""

import warp as wp

from .common import F, Vec, k
from .gas import monomer_fraction
from .particle_rates import phase_step
from .wet_network import Net, Pool
from .wet_stiff import stiff_gas

# A fixed maximum of six reversible states per core, with a run-time size
# boundary.
Ladder = wp.types.vector(length=6, dtype=F)
Pars = wp.types.vector(length=9, dtype=F)
Ledger = wp.types.vector(length=12, dtype=F)


@wp.func
def number(population: Ladder):
    """Sum mole ppm of cluster molecules over all occupied sizes."""
    count = F(0)
    for h in range(6):
        count += population[h]
    return count


@wp.func
def units(population: Ladder):
    """Sum the sulfate-equivalent repeat units stored across sizes."""
    total = F(0)
    for h in range(6):
        total += F(h + 1) * population[h]
    return total


@wp.func
def availability(stock: F, delta: F):
    """Limit a proposed transfer by the available stock."""
    fraction = F(1)
    if delta > stock and delta > F(0):
        fraction = wp.max(F(0), stock / delta)
    return fraction


@wp.func
def evolve(
    old: Ladder,
    source: F,
    frequency: F,
    evap: F,
    first_off: F,
    surface: F,
    bind: F,
    gate: F,
    temp: F,
    dt: F,
    limit: int,
):
    """Solve one implicit birth, evaporation and growth step for a core."""
    # Finite-volume birth-death balance with a nonnegative implicit tridiagonal
    # step. Reverse rates yield the specified size-dependent saturated weights.
    low = Ladder()
    diag = Ladder()
    upper = Ladder()
    rhs = Ladder()
    escape = F(0)
    for j in range(limit):
        n = F(j + 1)
        area = wp.pow(n + F(1), F(2) / F(3))
        back = (
            evap
            * wp.pow(n, F(2) / F(3))
            * wp.exp(
                wp.min(
                    F(50),
                    surface
                    * (wp.pow(n + F(1), F(2) / F(3)) - wp.pow(n, F(2) / F(3)))
                    - bind / (temp * n),
                )
            )
        )
        if j == 0:
            back = first_off
        forward = frequency * area
        if j == limit - 1:
            forward *= gate
            escape = forward
        diag[j] = F(1) + dt * (back + forward)
        if j < limit - 1:
            low[j + 1] = -dt * forward
        if j > 0:
            upper[j - 1] = -dt * back
        rhs[j] = old[j]
    rhs[0] += dt * source
    for j in range(1, limit):
        ratio = low[j] / diag[j - 1]
        diag[j] -= ratio * upper[j - 1]
        rhs[j] -= ratio * rhs[j - 1]
    updated = Ladder()
    updated[limit - 1] = rhs[limit - 1] / diag[limit - 1]
    for h in range(1, limit):
        j = limit - 1 - h
        updated[j] = (rhs[j] - upper[j] * updated[j + 1]) / diag[j]
    return (
        updated,
        dt * escape * updated[limit - 1],
        dt * first_off * updated[0],
    )


@wp.func
def gas(
    y: k.State,
    q: Pool,
    p: k.Params,
    a: Vec,
    b: Net,
    feed: k.Feed,
    t: F,
    dt: F,
    dry: F,
    stored_s: F,
    stored_hs: F,
):
    """Advance the gas while retaining HCl released by stored salts."""
    # The legacy gas-only function reads deposited sulfate only in its HCl
    # balance. Transient clusters retain sulfate (2 HCl equivalents) and
    # bisulfate (1 equivalent); leave physical deposition untouched.
    wall = y[9]
    y[9] += stored_s + F(0.5) * stored_hs
    z, pools, flux = stiff_gas(y, q, p, a, b, feed, t, dt, 40, 0, dry)
    z[9] = wall
    return z, pools


@wp.func
def supply(free: F, prev: F, gas_cl: F, t: F, dt: F, source_scale: F):
    """Advance charge-pair formation and recombination in mole ppm."""
    density = F(101325e-6) / (F(1.380649e-23) * t)
    # Number concentrations expressed as mole ppm, per polarity.
    create = (
        source_scale
        * F(2.2e7)
        * wp.pow(t, F(-3.5))
        * wp.exp(-F(60800) / t)
        * density
        * gas_cl
    )
    loss = F(1.1e-17) * wp.pow(t, F(-3)) * density * density * F(1e-6)
    arg = wp.sqrt(create * loss) * dt
    step = dt
    if arg > F(1e-8):
        step *= wp.tanh(arg) / arg
    out = (free + create * step) / (F(1) + loss * free * step)
    return wp.min(out, free + prev), create * dt


@wp.kernel(enable_backward=False)
def parcels(  # noqa: C901 - coupled reaction branches
    px: wp.array(dtype=k.Params),
    aa: wp.array(dtype=Vec),
    bb: wp.array(dtype=Net),
    pars: wp.array(dtype=Pars),
    feedarr: wp.array(dtype=k.Feed),
    props: wp.array2d(dtype=k.Props),
    ticks: wp.array(dtype=F),
    dryarr: wp.array(dtype=F),
    limit: int,
    output: wp.array3d(dtype=k.State),
    audit: wp.array3d(dtype=Ledger),
):
    """Integrate one fit, feed and trajectory and audit elemental balances."""
    i, c, h = wp.tid()
    feed = feedarr[c]
    p = px[i]
    a = aa[i]
    b = bb[i]
    u = pars[i]
    dt = ticks[h]
    dry = dryarr[c]
    y = k.State()
    y[0] = feed[1]
    y[2] = feed[0]
    q = Pool()
    m = Ladder()
    v = Ladder()
    w = Ladder()
    ions = F(0)
    stats = Ledger()
    ss = F(0)
    hs = F(0)
    first = F(0)
    for j in range(props.shape[1]):
        pr = props[h, j]
        t = pr[0]
        z, q1 = gas(y, q, p, a, b, feed, t, F(0.5) * dt, dry, ss, hs)
        mon = z[2] * monomer_fraction(t, z[2])
        if u[2] > F(0):
            updated, created = supply(ions, z[2], mon, t, dt, u[7])
            z[2] += ions - updated
            ions = updated
            stats[0] += created
        # Positive rates are independent of measurement labels. Mixed and
        # ionic subcritical cores can evaporate above the birth window.
        if t < F(1300) and (
            z[3] > F(0) or ss > F(0) or hs > F(0) or q1[1] > F(0)
        ):
            nppm = k.PRESSURE * F(1e-12) / (k.KB * t)

            attach = F(6e-10) * nppm * wp.sqrt(t / F(900)) * u[6]
            sulfate_bar = wp.max(F(1e-16), pr[3] / k.PRESSURE)
            frequency = attach * z[3]
            eq_off = attach * sulfate_bar * F(1e6)
            # Bulk surface term for a sulfate mantle. Core binding adjusts
            # first contact and decays with size, in Kelvin (energy/kB).
            sigma = wp.max(F(0.05), F(0.6557) - F(2.942e-4) * t)
            surface = (
                sigma
                * wp.pow(F(36) * k.PI * k.SALT_V_S * k.SALT_V_S, F(1) / F(3))
                / (k.KB * t)
            )
            log_saturation = wp.log(
                wp.max(F(1e-50), z[3] / (sulfate_bar * F(1e6)))
            )
            gate = F(0)
            if log_saturation > F(0) and t >= F(600) and t <= F(1044):
                n = F(limit + 1)
                critical = wp.pow(
                    F(2) * surface / (F(3) * log_saturation), F(3)
                )
                barrier = F(0)
                if critical > n:
                    barrier = (
                        surface
                        * (
                            wp.pow(critical, F(2) / F(3))
                            - wp.pow(n, F(2) / F(3))
                        )
                        - (critical - n) * log_saturation
                    )
                gate = wp.exp(-wp.min(F(120), wp.max(F(0), barrier)))
            # First association ratio uses 1-bar standard state, and trial
            # standard binding free energies expressed as E/kB in Kelvin.
            bound0 = u[3]
            bound1 = u[4]
            bound2 = u[5]
            used_m = m
            used_v = v
            used_w = w
            exit_m = F(0)
            exit_v = F(0)
            exit_w = F(0)
            off_m = F(0)
            off_v = F(0)
            off_w = F(0)
            if u[0] > F(0):
                rate = attach * u[0]
                used_m, exit_m, off_m = evolve(
                    m,
                    rate * mon * z[3],
                    frequency,
                    eq_off,
                    rate * F(1e6) * wp.exp(-bound0 / t),
                    surface,
                    bound0,
                    gate,
                    t,
                    dt,
                    limit,
                )
            if u[1] > F(0):
                rate = attach * u[1]
                used_v, exit_v, off_v = evolve(
                    v,
                    rate * mon * q1[1],
                    frequency,
                    eq_off,
                    rate * F(1e6) * wp.exp(-bound1 / t),
                    surface,
                    bound1,
                    gate,
                    t,
                    dt,
                    limit,
                )
            if u[2] > F(0):
                rate = attach * u[2]
                used_w, exit_w, off_w = evolve(
                    w,
                    rate * F(2) * ions * z[3],
                    frequency,
                    eq_off,
                    rate * F(1e6) * wp.exp(-bound2 / t),
                    surface,
                    bound2,
                    gate,
                    t,
                    dt,
                    limit,
                )
            dn_m = number(used_m) - number(m)
            dn_v = number(used_v) - number(v)
            dn_w = number(used_w) - number(w)
            # Terminal ternary adduct releases the third molecule to the hot
            # bath.
            cl = dn_m + exit_m + dn_v + exit_v
            delta_hs = dn_v
            delta_ions = F(0.5) * (dn_w + exit_w)
            sulf = (
                units(used_m)
                - units(m)
                + units(used_v)
                - units(v)
                - dn_v
                + units(used_w)
                - units(w)
                + F(limit + 1) * (exit_m + exit_w)
                + F(limit) * exit_v
            )
            frac = wp.min(
                wp.min(availability(z[2], cl), availability(z[3], sulf)),
                wp.min(
                    availability(q1[1], delta_hs),
                    availability(ions, delta_ions),
                ),
            )
            z[2] -= frac * cl
            z[3] -= frac * sulf
            q1[1] -= frac * delta_hs
            ions -= frac * delta_ions
            m += frac * (used_m - m)
            v += frac * (used_v - v)
            w += frac * (used_w - w)
            hs = number(v)
            ss = units(m) + units(v) - hs + units(w)
            out_m = frac * exit_m
            out_v = frac * exit_v
            out_w = frac * exit_w
            z[4] += out_m + out_v + F(0.5) * out_w
            z[5] += F(limit + 1) * (out_m + out_w) + F(limit) * out_v
            inc = (
                (out_m + out_v + out_w)
                * k.PRESSURE
                / (k.KB * k.TREF_K)
                * F(1e-18)
            )
            z[7] += inc
            stats[1] += out_m
            stats[2] += out_v
            stats[3] += out_w
            stats[4] += frac * (off_m + off_v + off_w)
            stats[5] = wp.max(stats[5], ss)
            stats[6] = wp.max(stats[6], hs)
            if first == F(0) and (
                stats[1] + stats[2] + stats[3]
            ) * k.PRESSURE / (k.KB * k.TREF_K) * F(1e-18) > F(0.001):
                first = t
            stats[10] = wp.max(stats[10], F(1) - frac)
        mid = phase_step(z, z, p, feed, pr, dt * F(0.5), a, 20, 0, dry)
        aeros = phase_step(z, mid, p, feed, pr, dt, a, 20, 0, dry)
        y, q = gas(aeros, q1, p, a, b, feed, t, F(0.5) * dt, dry, ss, hs)
        remain = ions + F(0.5) * number(w) + number(m) + number(v)
        kk = (
            y[2]
            + y[4]
            + y[8]
            + F(2) * (y[3] + y[5] + y[9] + ss)
            + q[0]
            + q[1]
            + q[2]
            + q[3]
            + remain
            + hs
        )
        su = y[0] + y[1] + y[3] + y[5] + y[9] + q[0] + q[1] + q[2] + ss + hs
        stats[7] = wp.max(
            stats[7], wp.max(wp.abs(kk - feed[0]), wp.abs(su - feed[1]))
        )
        stats[8] = wp.min(
            stats[8], wp.min(ions, wp.min(y[2], wp.min(y[3], q[1])))
        )
        for size_index in range(limit):
            stats[8] = wp.min(
                stats[8],
                wp.min(m[size_index], wp.min(v[size_index], w[size_index])),
            )
    stats[9] = first
    stats[11] = ions
    output[i, c, h] = y
    audit[i, c, h] = stats
