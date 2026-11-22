"""Conservative particle and surface candidate rates for rounds 11–20.

Inputs use the coupled kernel's ppm of formula units and million particles
per reference cm³. Stage is one-based. A trial of 1 changes only the current
round to its general form. All retained earlier rounds use the grounded law.
Surface and particle parameters in [0,1] are shared across every feed.
"""

import warp as wp

from .common import F, Vec, k


@wp.func
def condensed_sulfation(t: F, so2: F, o2: F):
    """Sengeløv et al. initial chloride conversion rate, 53.2 µm radius."""
    sulfur = so2 * F(1e-6)
    # Hold the law at its highest measured T; suppress sub-range cold
    # reaction by the same Arrhenius tail. Conversion requires water.
    return (
        F(2.7e7)
        * wp.exp(-F(18900) / wp.min(t, F(1023)))
        * sulfur
        / (F(1) + F(708) * sulfur)
        * wp.pow(o2 * F(0.01), F(0.37))
    )


@wp.func
def accommodation(t: F, radius: F, diff: F, mass: F, amplitude: F):
    """Add the missing interfacial resistance at alpha below one."""
    kn = F(2) * diff / radius * wp.sqrt(k.PI * mass / (F(8) * k.GAS_R * t))
    den = F(1) + F(2) * kn * (F(1) + kn)
    alpha = wp.exp(-F(2.302585092994046) * amplitude)
    # Dahneke base kernel, normalized to exactly one for alpha = 1.
    return den / (den + F(2) * kn * (F(1) + kn) * (F(1) / alpha - F(1)))


@wp.func
def curved_growth(pressure: F, flat: F, curved: F):
    """Positive-only flux at curved vs flat equilibrium pressure."""
    if pressure > flat:
        return wp.max(F(0), pressure - curved) / (pressure - flat)
    return F(1)


@wp.func
def phase_step(  # noqa: C901 - stage-specific comparisons share a vapor pool
    old: k.State,
    geo: k.State,
    p: k.Params,
    feed: k.Feed,
    props: k.Props,
    dt: F,
    a: Vec,
    stage: int,
    trial: int,
    dry: F,
):
    """Single midpoint phase evaluation, retaining one shared population."""
    flags = k.Switch(1, 1, 1, 1, 1, 0)
    if stage < 11:
        return k.phase(old, geo, p, props, dt, flags)
    t = props[0]
    geom = k.geometry(geo, t)
    r = geom[0]
    cl = k.exchange_rates(geo, p, props, flags, 0, geom)
    su = k.exchange_rates(geo, p, props, flags, 1, geom)
    flat_cl = props[1] + F(2) * props[2]
    pp = k.PRESSURE * F(1e-6)
    growth_cl = F(1)
    growth_s = F(1)
    birth_cl = F(1)
    birth_s = F(1)

    if stage >= 13 and r > F(0):
        # Source doesn't resolve surface coverage; use bulk sulfate volume
        # as a proxy for foreign area and retain pure-KCl surface growth.
        vcl = geo[4] * k.M_CL / F(1980)
        vs = geo[5] * k.M_S / F(2660)
        foreign = vs / (vcl + vs)
        blocked = F(1)
        if not (stage == 13 and trial == 1):
            # Shared activation supersaturation scale ln(S)=1. At larger
            # S an increasing fraction of foreign surface permits KCl growth.
            s = wp.max(F(1), geo[2] * pp / flat_cl)
            blocked = F(1) / s
        growth_cl *= F(1) - a[12] * foreign * blocked

    if stage >= 14 and r > F(0):
        if stage == 14 and trial == 1:
            factor = wp.exp(-F(2.302585092994046) * a[13])
            growth_cl *= factor
            growth_s *= factor
        else:
            growth_cl *= accommodation(
                t, r, F(1.15 * 1.632e-10) * wp.pow(t, F(1.88)), k.M_CL, a[13]
            )
            growth_s *= accommodation(t, r, F(5.66e-11) * t * t, k.M_S, a[13])

    if stage >= 15 and r > F(0):
        if stage == 15 and trial == 1:
            growth_cl *= F(1) - F(0.9) * a[14]
        else:
            sigma = F(0.2824) - F(7.867e-5) * t - F(4.37e-8) * t * t
            log_ratio = F(2) * sigma * k.SALT_V_CL / (r * k.KB * t)
            kelvin = wp.exp(wp.min(F(40), log_ratio))
            # Equilibrium dimer pressure follows square of monomer activity.
            star = props[1] * kelvin + F(2) * props[2] * kelvin * kelvin
            growth_cl *= F(1) - a[14] * (
                F(1) - curved_growth(geo[2] * pp, flat_cl, star)
            )

    if stage >= 16 and r > F(0):
        if stage == 16 and trial == 1:
            growth_s *= F(1) - F(0.9) * a[15]
        else:
            sigma = F(0.6557) - F(2.942e-4) * t
            # Bulk molar volume for grown sulfate, consistent with geometry.
            volume = k.M_S / (F(2660) * k.NA)
            kelvin = wp.exp(
                wp.min(F(40), F(2) * sigma * volume / (r * k.KB * t))
            )
            growth_s *= F(1) - a[15] * (
                F(1) - curved_growth(geo[3] * pp, props[3], props[3] * kelvin)
            )

    if stage >= 17 and cl[0] > F(0):
        unavailable = F(0.9)
        if not (stage == 17 and trial == 1):
            assoc = props[2] / (props[1] * props[1])
            pressure = geo[2] * pp
            pm = (
                F(2)
                * pressure
                / (F(1) + wp.sqrt(F(1) + F(8) * assoc * pressure))
            )
            dimer_flux = F(1.4142135623730951) * assoc * pm * pm
            unavailable = dimer_flux / (pm + dimer_flux)
        birth_cl *= F(1) - a[16] * unavailable

    if stage >= 20 and su[0] > F(0):
        suppressed = F(0.9)
        if not (stage == 20 and trial == 1):
            vapor = geo[3] * F(1e-6)
            b2 = F(2) * (
                F(1) + (F(1) - vapor) / vapor * wp.sqrt(k.M_S / F(0.028))
            )
            # Derivative of ln(p*) gives latent heat divided by kT. At
            # the critical capillary saddle d(surface energy)/dn ~ kT ln S.
            latent = F(44407.65610348317) / t - F(7.15)
            q = latent - F(0.5) - wp.log(geo[3] * pp / props[3])
            suppressed = q * q / (b2 + q * q)
        birth_s *= F(1) - a[19] * suppressed

    chloride = k.transfer(
        old[2],
        flat_cl,
        dt,
        wp.vec4d(cl[0] * birth_cl, cl[1] * growth_cl, cl[2], cl[3]),
    )
    sulfate = k.transfer(
        old[3],
        props[3],
        dt,
        wp.vec4d(su[0] * birth_s, su[1] * growth_s, su[2], su[3]),
    )

    # Two condensed chlorides and one SO2 yield one sulfate in the same
    # reservoir and two inferred HCl. No HCl is produced by wall deposition.
    air = F(0)
    wall = F(0)
    if dry < F(0.5) and geo[0] > F(0):
        reference = condensed_sulfation(t, geo[0], feed[2])
        if r > F(0):
            freq = reference * F(53.2e-6) / r
            if stage == 11 and trial == 1:
                freq = F(1) * geo[0] / F(500)
            air = F(0.5) * old[4] * k.fraction(a[10] * freq * dt)
        if stage >= 12:
            # Unresolved exposed deposit grains assigned the original
            # fixed-bed reference size; this is a parcel-local wall ledger.
            freq_wall = reference
            if stage == 12 and trial == 1:
                freq_wall = F(1) * geo[0] / F(500)
            wall = F(0.5) * old[8] * k.fraction(a[11] * freq_wall * dt)
    scale = F(1)
    if air + wall > old[0]:
        scale = old[0] / (air + wall)
    air *= scale
    wall *= scale

    ncl = old[6] + chloride[3]
    ns = old[7] + sulfate[3]
    cp = old[4] - F(2) * air + chloride[1]
    sp = old[5] + air + sulfate[1]
    cw = old[8] - F(2) * wall + chloride[2]
    sw = old[9] + wall + sulfate[2]
    # Exchange preserves the existing gas equilibrium. Catalytic activity is
    # an effective wall dose bounded by access to a perfectly absorbing wall.
    so2 = wp.max(F(0), old[0] - air - wall)
    so3 = old[1]
    if stage >= 18 and so2 + so3 > F(0):
        activity = F(1)
        if not (stage == 18 and trial == 1):
            activity = F(0)
            if t >= F(873):
                diffusion = F(1.2e-5) * wp.pow(t / F(298.15), F(1.75))
                activity = F(23424) * diffusion
        ratio = wp.exp(
            F(2.302585092994046) * (F(4.678) - F(4956) / t)
        ) / wp.sqrt(feed[2] * F(0.01) * k.PRESSURE / F(1e5))
        equilibrium_so3 = (so2 + so3) / (F(1) + ratio)
        relax = k.fraction(dt * a[17] * activity)
        so2 = (F(1) - relax) * so2 + relax * equilibrium_so3 * ratio
        so3 = (F(1) - relax) * so3 + relax * equilibrium_so3
    if (
        stage >= 19
        and dry < F(0.5)
        and r > F(0)
        and so3 > F(0)
        and geo[4] > F(0)
    ):
        # Gas impingement on the exposed chloride portion of existing grains.
        # Restrict uptake to SO3 reaching the particle by diffusion/collision.
        freq = F(1)
        if not (stage == 19 and trial == 1):
            diffusion = F(1.0e-5) * wp.pow(t / F(298.15), F(1.75))
            speed = wp.sqrt(F(8) * k.GAS_R * t / (k.PI * F(0.080063)))
            transport = (
                F(4)
                * k.PI
                * r
                * geom[1]
                * diffusion
                / (F(1) + F(4) * diffusion / (r * speed))
            )
            vcl = geo[4] * k.M_CL / F(1980)
            vs = geo[5] * k.M_S / F(2660)
            freq = transport * vcl / (vcl + vs)
        extent = wp.min(F(0.5) * cp, so3 * k.fraction(dt * a[18] * freq))
        so3 -= extent
        cp -= F(2) * extent
        sp += extent
    return k.State(
        so2,
        so3,
        chloride[0],
        sulfate[0],
        cp,
        sp,
        ncl,
        ns,
        cw,
        sw,
    )
