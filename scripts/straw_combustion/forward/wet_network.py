"""Explicit finite-inventory gas subnetwork embedded in the frozen salt reactor.

Pool order is KHSO3, KHSO4, KClSO3, KOH (gas ppm). The two old sulfate
sinks and both original particle birth laws still compete for free vapor.
The oxygen and 5% water baths are prescribed. The input coordinate map
identifies the retained pathways.
"""

import warp as wp

from . import wet_equilibrium as eq
from .common import F, Vec, k
from .drylink import _rise, exchange
from .gas import monomer_fraction
from .wet_literature import _hydrolyzed
from .wet_prompt import prompt

Pool = wp.vec4d
Net = wp.types.vector(40, F)
Flux = wp.types.vector(22, F)
FLUX_NAMES = [
    "wet_prompt_sulfate",
    "dry_prompt_sulfate",
    "wet_divert_KHSO3",
    "dry_divert_KClSO3",
    "wet_divert_KHSO4",
    "KOH_hydrolysis_signed",
    "R11_KOH_SO2_to_KHSO3",
    "R12_KOH_SO3_to_KHSO4",
    "R19b_sulfate_HCl_return",
    "R18b_sulfate_water_return",
    "R17b_R16b_KHSO3_to_KHSO4",
    "R11b_KHSO3_to_SO2",
    "R20_KClSO3_to_KHSO4",
    "R13b_KClSO3_to_SO3",
    "R21_KClSO3_KOH_to_sulfate",
    "R19_KHSO4_KCl_to_sulfate",
    "R18_KHSO4_KOH_to_sulfate",
    "R20b_bisulfate_HCl_to_KClSO3",
    "R17_R16_KHSO4_to_KHSO3",
    "R12b_KHSO4_to_SO3",
    "substituted_shared_SO3_signed",
    "direct_prompt_sulfate_after_diversion",
]


@wp.func
def v(b: Net, n: int, stage: int):
    """Read a modifier when its stage is active."""
    if stage >= n:
        return b[n - 1]
    return F(0)


@wp.func
def generic(stage: int, trial: int, n: int):
    """Identify the generalized comparison for a single stage."""
    return stage == n and trial == 1


@wp.func
def hydro_hcl(y: k.State, q: Pool):
    """Recover hydrolysis-generated hydrogen chloride in mole ppm."""
    return F(2) * (y[3] + y[5] + y[9]) + q[0] + q[1] + q[3]


@wp.func
def parameter(b: Net, n: int, stage: int, trial: int, general: F, ground: F):
    """Return the positive multiplier for the selected reaction."""
    factor = ground
    if generic(stage, trial, n):
        factor = general
    return wp.exp(v(b, n, stage) * factor)


@wp.func
def conditional(b: Net, n: int, stage: int, trial: int, general: F, ground: F):
    """Scale an additive reaction extent or branching coefficient."""
    target = ground
    if generic(stage, trial, n):
        target = general
    return v(b, n, stage) * target


@wp.func
def turnover(amount: F, source: F, rate: F, dt: F):
    """Initial stock + uniform input; solve first-order depletion exactly."""
    x = rate * dt
    frac = k.fraction(x)
    released = F(0.5) * x * (F(1) - x / F(3) + x * x / F(12))
    if x > F(1e-4):
        released = F(1) - frac / x
    return amount * frac + source * released


@wp.func
def falloff(c: F, t: F, n: int, change: F, is_general: bool):
    """Evaluate the pressure-dependent association rate."""
    kinf = F(1e8)
    low = F(2.6e30) * wp.pow(t, F(-7.6)) * c
    if n == 35:
        low = F(1.9e29) * wp.pow(t, F(-7.8)) * c
    # General multiplier changes the effective rate at any pressure;
    # sourced multiplier changes only its low-pressure association limit.
    mult = wp.exp(F(2.302585092994046) * change)
    if is_general:
        return mult * low / (F(1) + low / kinf)
    low = low * mult
    return low / (F(1) + low / kinf)


@wp.func
def wet_gas_audit(  # noqa: C901 - coupled reaction branches
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
    """Advance the wet network and record signed reaction extents."""
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
    defkh = eq.kh(temp, ly)
    if stage >= 15:
        # Relax R10 to its finite-pool equilibrium. Background HCl excludes
        # acid paired with the already explicit free KOH.
        bg = wp.max(F(0), hcl - oh)
        target = F(1e6) * _hydrolyzed(
            defkh, F(0), (salt + oh) * F(1e-6), F(0.05), bg * F(1e-6), F(0)
        )
        hydro_hz = (
            F(1.7e8) * c * ((bg + oh + target) * F(1e-6) + defkh * F(0.05))
        )
        if generic(stage, trial, 15):
            hydro_hz = F(1)
        extent = (target - oh) * k.fraction(v(b, 15, stage) * hydro_hz * dt)
        flux[5] = extent
        salt -= extent
        oh += extent
        hcl += extent

    # R11/R12 explicitly compete for the same KOH. Reverse paths below
    # return the matching sulfur oxide and hydroxide.
    r11 = conditional(
        b, 16, stage, trial, so2 / F(100), k11 * c * so2 * F(1e-6)
    )
    r12 = conditional(
        b, 17, stage, trial, so3 / F(100), k12 * c * so3 * F(1e-6)
    )
    rr = r11 + r12
    if rr > F(0):
        total = oh * k.fraction(rr * dt)
        fb = wp.min(so2, total * r11 / rr)
        fa = wp.min(so3, total * r12 / rr)
        flux[6] = fb
        flux[7] = fa
        so2 -= fb
        so3 -= fa
        oh = wp.max(F(0), oh - fb - fa)
        bsrc += fb
        asrc += fa

    # Independently adjustable product return reactions R19b and R18b.
    # Cap R19b by gas HCl and share a finite sulfate product.
    hcl = F(2) * (product + z[5] + z[9]) + q[0] + q[1] + oh + bsrc + asrc
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
    total = turnover(q[0], bsrc, ox + r11b, dt)
    frac_b = r11b / wp.max(F(1e-50), ox + r11b)
    breakup = total * frac_b
    bnew = wp.max(F(0), q[0] + bsrc - total)
    flux[10] = total - breakup
    flux[11] = breakup
    asrc += total - breakup
    oh += breakup
    so2 += breakup

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
