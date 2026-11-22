"""Stiff KOH–KHSO3 chemistry coupled to the other sulfur reservoirs."""

import warp as wp

from .common import F, Vec, k
from .stiff_pair import coupled_pair
from .wet_network import (
    Flux,
    Net,
    Pool,
    _rise,
    conditional,
    eq,
    exchange,
    falloff,
    generic,
    monomer_fraction,
    parameter,
    prompt,
    turnover,
    v,
    wet_gas_audit,
)


@wp.func
def stiff_gas(  # noqa: C901 - coupled reaction branches
    y: k.State,
    q: Pool,
    p: k.Params,
    a: Vec,
    b: Net,
    feed: k.Feed,
    t: F,
    dt: F,
    stage: int,
    trial: int,
    dry: F,
):
    """Advance gas chemistry with implicit precursor binding competition."""
    if dry > F(0.5) or (v(b, 16, stage) <= F(0) and v(b, 17, stage) <= F(0)):
        return wet_gas_audit(y, q, p, a, b, feed, t, dt, stage, trial, dry)
    c = k.PRESSURE / (k.GAS_R * t)
    rt = k.GAS_R * t
    temp = t / F(1000)
    ly = wp.log(temp)
    acid_extra = q[0] + q[1] + q[3]
    share = v(b, 40, stage)
    delta = F(0)
    if stage >= 40:
        if generic(stage, trial, 40):
            # Bounded equilibrium relaxation at 1/s for the free supplier.
            ratio = wp.exp(
                F(2.302585092994046) * (F(4.678) - F(4956) / t)
            ) / wp.sqrt(feed[2] * F(0.01) * k.PRESSURE / F(1e5))
            delta = ((y[0] + y[1]) / (F(1) + ratio) - y[1]) * k.fraction(dt)
        else:
            water = F(0.05)
            if dry > F(0.5):
                water = F(0)
            gate = _rise((t - F(1000)) / F(50)) * _rise((F(1423) - t) / F(23))
            share *= gate
            if gate > F(0):
                # The supplier applies the same taper internally to the
                # hazard. Compute its unrestricted extent then mix once.
                delta = exchange(
                    y[0], y[1], feed[2], t, dt / gate, b, stage, trial, water
                )
    z, de, we = prompt(
        y, p, feed, t, dt, a, 20, 0, F(0.05), acid_extra, share, delta
    )

    # No measured water in the pure-chloride series. Preserve its legacy
    # zero-sulfur chemistry and exclude all added hydrolysis paths.
    flux = Flux()
    flux[0] = we
    flux[1] = de
    flux[20] = share * delta
    if dry > F(0.5):
        flux[21] = we + de
        return z, q, flux

    # Route some of the legacy flux through real one-K intermediates.
    # Each rerouted formula unit returns the second KCl to the gas.
    bsrc = v(b, 13, stage) * we
    csrc = v(b, 14, stage) * de
    asrc = v(b, 21, stage) * (we - bsrc)
    flux[2] = bsrc
    flux[3] = csrc
    flux[4] = asrc
    flux[21] = we + de - bsrc - csrc - asrc
    delay = bsrc + csrc + asrc
    so2 = z[0]
    so3 = z[1]
    salt = z[2] + delay
    product = wp.max(F(0), z[3] - delay)
    oh = q[3]

    # Perturbing one formation enthalpy changes all reversible pathways that
    # contain its species. 3 kcal/mol is a sensitivity scale, not a fitted Hf.
    stabilize_b = parameter(b, 36, stage, trial, F(1), F(12552) / rt)
    stabilize_a = parameter(b, 37, stage, trial, F(1), F(12552) / rt)
    stabilize_c = parameter(b, 38, stage, trial, F(1), F(12552) / rt)
    k11 = falloff(c, t, 33, v(b, 33, stage), generic(stage, trial, 33))
    k12 = falloff(c, t, 34, v(b, 34, stage), generic(stage, trial, 34))
    k13 = falloff(c, t, 35, v(b, 35, stage), generic(stage, trial, 35))
    # Updated state chlorine balance (KClSO3 retains one Cl).
    hcl = F(2) * (product + z[5] + z[9]) + q[0] + q[1] + oh + bsrc + asrc
    rate16 = F(8.4e3) * wp.pow(t, F(1.22)) * wp.exp(-F(93303.2) / rt)
    # R17b oxidation of actual KHSO3 with competition from R11 breakup.
    ox = (
        F(2e6)
        * eq.k17b(temp, ly)
        * stabilize_a
        / stabilize_b
        * c
        * feed[2]
        * F(0.01)
    )
    if generic(stage, trial, 13):
        ox = F(1)
    ox *= parameter(b, 30, stage, trial, F(2), F(6000) / rt)
    ox += conditional(
        b,
        28,
        stage,
        trial,
        F(1),
        rate16 * c * F(1e-9) * stabilize_a / (eq.k16(temp, ly) * stabilize_b),
    )
    r11b = conditional(
        b,
        18,
        stage,
        trial,
        F(1),
        k11 * F(1e5) / (rt * eq.k11(temp, ly) * stabilize_b),
    )
    acid_scale = F(1.7e8) * c * v(b, 15, stage)
    bind = conditional(b, 16, stage, trial, F(0.01), k11 * c * F(1e-6))
    assoc = conditional(b, 17, stage, trial, F(0.01), k12 * c * F(1e-6))
    fast = coupled_pair(
        oh,
        q[0],
        bsrc,
        salt,
        so2,
        so3,
        hcl,
        acid_scale * eq.kh(temp, ly) * F(0.05),
        acid_scale * F(1e-6),
        bind,
        assoc,
        r11b,
        ox,
        dt,
    )
    oh = fast[0]
    bnew = fast[1]
    salt -= fast[2]
    so2 -= fast[3]
    so3 -= fast[4]
    asrc += fast[4] + fast[5]
    flux[5] = fast[2]
    flux[6] = fast[6]
    flux[7] = fast[4]
    flux[10] = fast[5]
    flux[11] = fast[7]

    # Independently adjustable product return reactions R19b and R18b.
    # Cap R19b by gas HCl and share a finite sulfate product.
    hcl = F(2) * (product + z[5] + z[9]) + bnew + q[1] + oh + asrc
    rev19 = conditional(
        b,
        25,
        stage,
        trial,
        hcl / F(100),
        F(3e7) * c * hcl * F(1e-6) * stabilize_a / eq.k19(temp, ly),
    )
    rev18 = conditional(
        b,
        26,
        stage,
        trial,
        F(1),
        F(1e8) * c * F(0.05) * stabilize_a / eq.k18(temp, ly),
    )
    rr = rev19 + rev18
    if rr > F(0):
        total = product * k.fraction(dt * rr)
        e19 = wp.min(hcl, total * rev19 / rr)
        e18 = total * rev18 / rr
        flux[8] = e19
        flux[9] = e18
        product = wp.max(F(0), product - e19 - e18)
        asrc += e19 + e18
        salt += e19
        oh += e18
        hcl -= e19

    # Exchange with a prescribed radical bath in sensitivity-only R27–29.
    # O, OH, H are 0–1 ppb equivalent concentrations (unmeasured).
    rred = conditional(b, 27, stage, trial, F(1), F(2e6) * c * F(1e-9))
    rate16 = F(8.4e3) * wp.pow(t, F(1.22)) * wp.exp(-F(93303.2) / rt)
    rred += conditional(b, 29, stage, trial, F(1), rate16 * c * F(1e-9))
    # R20b rebinds HCl on bisulfate. R12b decomposes to SO3 and KOH.
    r20b = conditional(
        b,
        24,
        stage,
        trial,
        hcl / F(100),
        F(1e8)
        * c
        * hcl
        * F(1e-6)
        * stabilize_c
        / (eq.k20(temp, ly) * stabilize_a),
    )
    r12b = conditional(
        b,
        20,
        stage,
        trial,
        F(1),
        k12 * F(1e5) / (rt * eq.k12(temp, ly) * stabilize_a),
    )
    # R13 breakup competes with R20 hydrolysis and R21 attack by KOH.
    hydro = (
        F(1e8)
        * c
        * F(0.05)
        * parameter(b, 31, stage, trial, F(2), F(6000) / rt)
    )
    if generic(stage, trial, 14):
        hydro = F(1)
    r13b = conditional(
        b,
        19,
        stage,
        trial,
        F(1),
        k13 * F(1e5) / (rt * eq.k13(temp, ly) * stabilize_c),
    )
    r21 = conditional(
        b, 23, stage, trial, oh / F(10), F(1e8) * c * oh * F(1e-6)
    )
    rr = hydro + r13b + r21
    total = turnover(q[2], csrc, rr, dt)
    hyd = total * hydro / rr
    breakup = total * r13b / rr
    consumed_oh = wp.min(oh, total * r21 / rr)
    flux[12] = hyd
    flux[13] = breakup
    flux[14] = consumed_oh
    cnew = wp.max(F(0), q[2] + csrc - hyd - breakup - consumed_oh)
    oh -= consumed_oh
    product += consumed_oh
    asrc += hyd
    salt += breakup
    so3 += breakup

    # R19 terminal KCl capture competes with R18 consumption of free KOH.
    mono = monomer_fraction(t, salt)
    usable = F(1) + v(b, 39, stage) * (mono - F(1))
    if generic(stage, trial, 39):
        usable = F(1) - F(0.9) * v(b, 39, stage)
    r19 = (
        F(3e7)
        * c
        * salt
        * F(1e-6)
        * usable
        * parameter(b, 32, stage, trial, F(2), F(6000) / rt)
    )
    if generic(stage, trial, 21):
        r19 = salt / F(200)
    r18 = conditional(
        b, 22, stage, trial, oh / F(10), F(1e8) * c * oh * F(1e-6)
    )
    # All five bisulfate outlets share the same inventory and compete
    # on their rates, even when one association is very fast. Products
    # that re-enter an upstream intermediate react in the next half-step.
    rr = r18 + r19 + r20b + r12b + rred
    anew = q[1] + asrc
    if rr > F(0):
        total = turnover(q[1], asrc, rr, dt)
        e19 = wp.min(salt, total * r19 / rr)
        e18 = wp.min(oh, total * r18 / rr)
        back_c = wp.min(hcl, total * r20b / rr)
        back_b = total * rred / rr
        breakup = total * r12b / rr
        flux[15] = e19
        flux[16] = e18
        flux[17] = back_c
        flux[18] = back_b
        flux[19] = breakup
        anew = wp.max(F(0), q[1] + asrc - e19 - e18 - back_c - back_b - breakup)
        product += e19 + e18
        salt -= e19
        oh = oh - e18 + breakup
        cnew += back_c
        bnew += back_b
        so3 += breakup
    return (
        k.State(so2, so3, salt, product, z[4], z[5], z[6], z[7], z[8], z[9]),
        Pool(bnew, anew, cnew, oh),
        flux,
    )
