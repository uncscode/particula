"""Reversible sulfur exchange and ten bounded LFAC gas-chemistry hypotheses.

All kinetic extents use analytical KCl/SOx mole ppm. Partial pressures and
molar concentrations below use Pa and mol/m³. The bounded increments describe
tests of reaction laws or sensitivity weights.
"""

import warp as wp

from .common import F, Vec, k
from .wet_literature import _hydrolyzed

# DIMER - 2*MONOMER (log10 Kp at 1 bar, evaluated every 100 K).
# Same JANAF tables, interpolation and pure-gas speciation used by the parent
# salt-vapor inputs used in the reactor thermal calculation.
Table = wp.types.vector(10, F)
LOG_ASSOC = wp.constant(
    Table(
        18.058, 13.169, 9.926, 7.618, 5.892, 4.556, 3.491, 2.624, 1.902, 1.295
    )
)


@wp.func
def monomer_fraction(t: F, salt_ppm: F) -> F:
    """Fraction of analytical gas KCl carried in monomeric KCl."""
    tt = wp.clamp(t, F(400), F(1300))
    j = wp.min(int((tt - F(400)) / F(100)), 8)
    t0 = F(400) + F(100) * F(j)
    t1 = t0 + F(100)
    w = (F(1) / tt - F(1) / t0) / (F(1) / t1 - F(1) / t0)
    log_k = LOG_ASSOC[j] + w * (LOG_ASSOC[j + 1] - LOG_ASSOC[j])
    loading = (
        wp.exp(F(2.302585092994046) * log_k)
        * salt_ppm
        * F(1e-6)
        * k.PRESSURE
        / F(1e5)
    )
    return F(2) / (F(1) + wp.sqrt(F(1) + F(8) * loading))


@wp.func
def wet_rates(
    t: F,
    so2: F,
    salt: F,
    oxygen_pct: F,
    hcl_ppm: F,
    a: Vec,
    stage: int,
    trial: int,
    water_fraction: F,
) -> wp.vec2d:
    """Return SO₂ conversion hazard (s⁻¹) and KOH mole fraction."""
    if salt <= F(0) or oxygen_pct <= F(0):
        return wp.vec2d(F(0), F(0))
    y = t / F(1000)
    inv = F(1) / y
    ly = wp.log(y)
    kh = wp.exp(
        F(0.1281423431)
        - F(16.8251584274) * inv
        + F(1.4042287932) * ly
        - F(0.3740561356) * y
        + F(0.0298796942) * y * y
    )
    ka = wp.exp(
        -F(18.5493290417)
        + F(20.6649999910) * inv
        + F(0.0289047448) * ly
        + F(1.2481596862) * y
        - F(0.2016503037) * y * y
    )
    ko = wp.exp(
        -F(4.7672762997)
        + F(0.7006360520) * inv
        - F(0.4378263697) * ly
        + F(1.3622351514) * y
        - F(0.2304255656) * y * y
    )
    common_scale = F(1)
    k10 = F(1.7e8)
    k17 = F(2e6) * ko
    low_scale = F(1)
    acid = hcl_ppm
    if stage == 1 and trial == 1:
        common_scale = common_scale * wp.exp(a[0] * wp.log(oxygen_pct / F(5)))
    else:
        k17 = k17 * wp.exp(a[0])
    if stage == 2 and trial == 1:
        common_scale = common_scale * wp.exp(
            -a[1] * hcl_ppm / (F(20) + hcl_ppm)
        )
    else:
        k10 = k10 * wp.exp(a[1])
    if stage == 3 and trial == 1:
        common_scale = common_scale * wp.exp(a[2] * wp.log(t / F(1100)))
    else:
        low_scale = wp.exp(a[2])
    if stage == 4 and trial == 1:
        common_scale = common_scale * wp.exp(a[3] * so2 / (F(50) + so2))
    else:
        # Lowering Hf(KHSO3) favors R11 and disfavors R17b by the same energy.
        stability = wp.exp(F(5000) * a[3] / (k.GAS_R * t))
        ka = ka * stability
        k17 = k17 / stability
    if stage == 5 and trial == 1:
        common_scale = common_scale * wp.exp(-a[4])
    else:
        acid = acid + F(7.5) * a[4]
    c = F(101325) / (F(8.31446261815324) * t)
    s = so2 * F(1e-6)
    potassium = salt * F(1e-6)
    oxygen = oxygen_pct * F(0.01)
    low = F(2.6e30) * wp.exp(-F(7.6) * wp.log(t)) * c * low_scale
    k11 = low / (F(1) + low / F(1e8))
    diss = k11 * F(1e5) / (F(8.31446261815324) * t * ka)
    assoc = k11 * c / (diss + k17 * c * oxygen)
    free_s = s
    sulfite = F(0)
    hydroxide = F(0)
    for _ in range(4):
        binding = assoc * free_s
        hydro = _hydrolyzed(
            kh,
            binding,
            potassium,
            water_fraction,
            acid * F(1e-6),
            k17 * oxygen / k10,
        )
        sulfite = hydro * binding / (F(1) + binding)
        hydroxide = hydro / (F(1) + binding)
        free_s = s - sulfite
    frequency = F(0)
    if s > F(0):
        frequency = k17 * c * oxygen * sulfite / s * common_scale
    return wp.vec2d(frequency, hydroxide)
