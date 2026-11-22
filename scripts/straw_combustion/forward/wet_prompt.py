"""Frozen iter20 gas_step_bath, annotated to expose both prompt fluxes.

The extra supplier replaces a fraction of the native SO2/SO3 exchange
before KCl capture. Setting its weight to zero restores the native operator.
"""

import warp as wp

from .common import F, Vec, k
from .gas import monomer_fraction, wet_rates


@wp.func
def prompt(
    y: k.State,
    p: k.Params,
    feed: k.Feed,
    t: F,
    dt: F,
    a: Vec,
    stage: int,
    trial: int,
    water_fraction: F,
    external_hcl_ppm: F,
    supplier_mix: F,
    shared_extent: F,
):
    """Advance SO₂/SO₃ and both KCl conversion branches, conserving K and S."""
    correction = wp.exp(
        F(2.302585092994046) * (F(4.678) - F(4956) / t)
    ) / wp.sqrt(feed[2] * F(0.01) * k.PRESSURE / F(1e5))
    pivot = p[0]
    energy = p[1]
    if stage == 6 and trial == 1:
        pivot = pivot + a[5]
    else:
        pivot = pivot + F(1000) * a[5] * (F(1) / F(1273) - F(1) / F(1100))
        energy = energy + F(8.31446261815324) * a[5]
    dry_k = F(69) * wp.exp(
        -F(63000) / (k.GAS_R * t)
        + pivot
        - F(1000) * energy / k.GAS_R * (F(1) / t - F(1) / F(1100))
    )
    exchange = k.fraction(dry_k * (F(1) + correction) * dt)
    eq3 = (y[0] + y[1]) / (F(1) + correction)
    so2 = y[0] * (F(1) - exchange) + exchange * eq3 * correction
    so3 = y[1] * (F(1) - exchange) + exchange * eq3
    so2 = so2 + supplier_mix * ((y[0] - shared_extent) - so2)
    so3 = so3 + supplier_mix * ((y[1] + shared_extent) - so3)
    salt = y[2]
    if stage == 10 and trial == 1:
        salt = salt * wp.exp(-a[9] * salt / (F(200) + salt))
    else:
        salt = salt * wp.exp(F(0.5) * a[9] * wp.log(monomer_fraction(t, salt)))
    c = k.PRESSURE / (k.GAS_R * t)
    hcl = external_hcl_ppm + F(2) * (y[3] + y[5] + y[9])
    wet = wet_rates(t, so2, salt, feed[2], hcl, a, stage, trial, water_fraction)
    hydroxide = wet[1]
    # R13 and R12 use high-/low-pressure limits, with Lindemann interpolation.
    # For R13 the hazard relative to the screening 100/s law is independent
    # of salt at fixed T, so this ratio remains well defined as salt tends to 0.
    base_rapidity = F(100) * salt / F(200)
    rapidity = base_rapidity
    if stage == 7 and trial == 1:
        rapidity = rapidity * wp.exp(a[6])
    else:
        low13 = F(1.9e29) * wp.exp(-F(7.8) * wp.log(t)) * c
        k13 = low13 / (F(1) + low13 / F(1e8))
        ratio = k13 * c * F(2e-4) / F(100)
        rapidity = rapidity * wp.exp(F(0.5) * a[6] * wp.log(ratio))
    if stage == 8 and trial == 1:
        rapidity = rapidity + a[7] * base_rapidity * feed[2] / F(5)
    else:
        low12 = F(2.6e30) * wp.exp(-F(7.6) * wp.log(t)) * c
        k12 = low12 / (F(1) + low12 / F(1e8))
        rapidity = rapidity + F(0.5) * a[7] * k12 * c * hydroxide
    dry_extent = so3 * k.fraction(rapidity * dt)
    wet_freq = wp.exp(p[2]) * wet[0]
    if stage == 9 and trial == 1:
        wet_freq = wet_freq / (F(1) + a[8] * F(0.1) * wet_freq)
    else:
        capture = F(3e7) * c * salt * F(1e-6) + F(1e8) * c * hydroxide
        wet_freq = wet_freq / (
            F(1) + a[8] * wet_freq / wp.max(capture, F(1e-30))
        )
    wet_extent = so2 * k.fraction(wet_freq * dt)
    demand = dry_extent + wet_extent
    limit = F(1)
    if demand * F(2) > y[2] and demand > F(0):
        limit = y[2] / (F(2) * demand)
    dry = dry_extent * limit
    wet_use = wet_extent * limit
    used = wp.min(y[2], F(2) * (dry + wet_use))
    z = k.State(
        so2 - wet_use,
        so3 - dry,
        y[2] - used,
        y[3] + F(0.5) * used,
        y[4],
        y[5],
        y[6],
        y[7],
        y[8],
        y[9],
    )

    return z, dry, wet_use
