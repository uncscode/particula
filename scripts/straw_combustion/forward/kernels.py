"""Vapor transfer, pure salt birth and common surface geometry."""

import numpy as np
import warp as wp
from scipy.constants import Avogadro, Boltzmann, R

F = wp.float64
State = wp.types.vector(10, F)
Params = wp.types.vector(7, F)
Switch = wp.types.vector(6, wp.int32)
Props = wp.vec4d
Feed = wp.vec3d
PI = wp.constant(F(np.pi))
KB = wp.constant(F(Boltzmann))
GAS_R = wp.constant(F(R))
NA = wp.constant(F(Avogadro))
PRESSURE = wp.constant(F(101325.0))
TREF_K = wp.constant(F(293.15))
M_CL = wp.constant(F(0.0745513))
M_S = wp.constant(F(0.174259))
SALT_V_CL = wp.constant(F(0.0745513 / (1980.0 * Avogadro)))
SALT_V_S = wp.constant(F(0.174259 / (1870.0 * Avogadro)))


@wp.func
def fraction(x: F):
    """Evaluate exponential conversion accurately at small exposure."""
    if x < F(1e-5):
        return x * (F(1.0) - x * (F(0.5) - x / F(6.0)))
    return F(1.0) - wp.exp(-x)


@wp.func
def cnt(
    t: F,
    log_s: F,
    delta: F,
    sigma: F,
    unit_volume: F,
    molar_mass: F,
    p_m: F,
    p_d: F,
):
    """Compute critical-cluster flux and monomer-equivalent size."""
    # Christensen & Livbjerg (2000), Eqs. 14–20. For sulfate p_d=0.
    j = F(0.0)
    q = F(0.0)
    if log_s > F(1e-6):
        kt = KB * t
        surface = F(3.0) * sigma * unit_volume
        h = kt * log_s / surface
        c = F(2.0) * delta
        rcrit = (
            F(1.0 / 3.0) - h * c + wp.sqrt(F(1.0 / 9.0) + h * c / F(3.0))
        ) / h
        # At extreme supersaturations the continuum saddle can fall below
        # molecular size. Bound the kinetic continuation at a dimer.
        r = wp.max(rcrit, wp.pow(F(1.5) * unit_volume / PI, F(1.0 / 3.0)))
        q = F(4.0 / 3.0) * PI * r * r * r / unit_volume
        barrier = wp.max(F(0.0), q * (surface / ((r + c) * kt) - log_s))
        if barrier < F(135.0):
            z = wp.sqrt(
                surface
                * r
                * (r + F(2.0) * c)
                / (F(9.0) * PI * kt * q * wp.pow(r + c, F(3.0)))
            )
            arrival = (
                F(4.0)
                * PI
                * r
                * r
                * (p_m + F(1.4142135623730951) * p_d)
                / kt
                * wp.sqrt(GAS_R * t / (F(2.0) * PI * molar_mass))
            )
            j = arrival * z * (p_m + p_d) / kt * wp.exp(-barrier)
    return wp.vec2d(j, q)


@wp.func
def geometry(y: State, t: F):
    """Return radius and number per actual cubic metre."""
    # Physical volume per reference m³, and number density per actual m³.
    volume = (
        (y[4] * M_CL / F(1980.0) + y[5] * M_S / F(2660.0))
        * (PRESSURE / (GAS_R * TREF_K))
        * F(1e-6)
    )
    n = (y[6] + y[7]) * F(1e12)
    radius = F(0.0)
    if n > F(1e-12) and volume > F(1e-45):
        radius = wp.pow(F(3.0) * volume / (F(4.0) * PI * n), F(1.0 / 3.0))
    return wp.vec2d(radius, n * TREF_K / t)


@wp.func
def exchange_rates(
    y: State,
    p: Params,
    props: Props,
    switches: Switch,
    kind: int,
    geom: wp.vec2d,
):
    """Compute birth, shared-surface growth and wall frequencies."""
    # Return frequency for birth mass, deposition, wall loss; number per
    # ppm of birth mass. All three processes draw from one supersaturation.
    t = props[0]
    pp = PRESSURE * F(1e-6)
    vapor = y[3]
    ps = props[3]
    mass = M_S
    diff = F(5.66e-11) * t * t
    vd = SALT_V_S
    sigma = F(0.6557) - F(2.942e-4) * t
    delta = F(9.5e-11) * p[5]
    pm = vapor * pp
    pd = F(0.0)
    star_m = ps
    enable = switches[1] != 0 and t > F(600.0)
    if kind == 0:
        vapor = y[2]
        star_m = props[1]
        star_d = props[2] * F(switches[4])
        ps = star_m + F(2.0) * star_d
        mass = M_CL
        diff = F(1.15 * 1.632e-10) * wp.pow(t, F(1.88))
        vd = SALT_V_CL
        sigma = F(0.2824) - F(7.867e-5) * t - F(4.37e-8) * t * t
        delta = F(1e-10) * p[4]
        pressure = vapor * pp
        assoc = star_d / (star_m * star_m)
        pm = (
            F(2.0)
            * pressure
            / (F(1.0) + wp.sqrt(F(1.0) + F(8.0) * assoc * pressure))
        )
        pd = assoc * pm * pm
        enable = switches[0] != 0 and t < F(1044.0)
    excess = wp.max(vapor - ps / pp, F(0.0))
    birth = F(0.0)
    growth = F(0.0)
    wall = F(0.0)
    number_per_ppm = F(0.0)
    if excess > F(0.0):
        if enable and pm > star_m:
            source = cnt(t, wp.log(pm / star_m), delta, sigma, vd, mass, pm, pd)
            if source[0] > F(0.0):
                actual_mol_per_ppm = pp / (GAS_R * t)
                birth = (
                    source[0] * source[1] / (NA * actual_mol_per_ppm * excess)
                )
                number_per_ppm = (
                    NA * pp / (GAS_R * TREF_K * F(1e12) * source[1])
                )
        if geom[0] > F(0.0):
            kn = (
                F(2.0)
                * diff
                / geom[0]
                * wp.sqrt(PI * mass / (F(8.0) * GAS_R * t))
            )
            transition = (F(1.0) + kn) / (F(1.0) + F(2.0) * kn * (F(1.0) + kn))
            growth = F(4.0) * PI * geom[0] * diff * geom[1] * transition
        if switches[2] != 0:
            # Fully developed laminar constant-concentration wall (Sh=3.66).
            # Wall T equals imposed parcel T; only vapor above p* deposits.
            wall = wp.exp(p[3]) * F(23424.0) * diff
    return wp.vec4d(birth, growth, wall, number_per_ppm)


@wp.func
def transfer(vapor: F, pstar: F, dt: F, rates: wp.vec4d):
    """Partition finite vapor supersaturation among competing sinks."""
    # Competitive finite-reservoir exponential integrates frozen rates.
    # Births consume their critical salt mass; counting uses that transfer.
    k = rates[0] + rates[1] + rates[2]
    salt = F(0.0)
    on_wall = F(0.0)
    count = F(0.0)
    if k > F(0.0):
        excess = wp.max(F(0.0), vapor - pstar / (PRESSURE * F(1e-6)))
        pool = excess * fraction(k * dt)
        salt = pool * ((rates[0] + rates[1]) / k)
        on_wall = pool * (rates[2] / k)
        count = pool * (rates[0] / k) * rates[3]
    return wp.vec4d(
        wp.max(F(0.0), vapor - salt - on_wall), salt, on_wall, count
    )


@wp.func
def phase(
    old: State,
    evaluation: State,
    p: Params,
    props: Props,
    dt: F,
    switches: Switch,
):
    """Advance the two salts with one common particle area."""
    geom = geometry(evaluation, props[0])
    cl = exchange_rates(evaluation, p, props, switches, 0, geom)
    su = exchange_rates(evaluation, p, props, switches, 1, geom)
    chloride = transfer(
        old[2], props[1] + F(2.0) * props[2] * F(switches[4]), dt, cl
    )
    sulfate = transfer(old[3], props[3], dt, su)
    return State(
        old[0],
        old[1],
        chloride[0],
        sulfate[0],
        old[4] + chloride[1],
        old[5] + sulfate[1],
        old[6] + chloride[3],
        old[7] + sulfate[3],
        old[8] + chloride[2],
        old[9] + sulfate[2],
    )
