"""Local O steady-state bridge to the wet 1 atm parcel; no external O or O3.

The Yilmaz core shares rounds 1–4 with dry40. A smooth taper is zero at
1000/1423 K and unity over 1050–1400 K. The elementary N2 dissociation
estimate comes from 1273–1348 K; use outside that interval is extrapolation.
The N2 collider excludes water, oxygen and tracked sulfur gases. Unknown
remaining traces are treated as N2. Wet radicals outside the two-reaction
cycle are omitted. This is a transfer sensitivity, not a wet validation.
"""

import warp as wp

F = wp.float64
Net = wp.types.vector(40, F)
C7 = wp.types.vector(7, F)
DISS_G = wp.constant(
    C7(
        F(-0.7130357699999998),
        F(0.0012332345486),
        F(-5.456280148000001e-07),
        F(9.985461154999997e-11),
        F(-6.898486306e-15),
        F(-41879.16099999999),
        F(-15.03576325),
    )
)
ABST_G = wp.constant(
    C7(
        F(0.7132773399999999),
        F(0.0005222365284000001),
        F(-4.128591202000001e-07),
        F(8.918448264999997e-11),
        F(-6.558461214e-15),
        F(17788.84025),
        F(-8.60653595),
    )
)


@wp.func
def _dg(t: F, a: C7):
    return (
        a[0] * (F(1) - wp.log(t))
        - a[1] * t / F(2)
        - a[2] * t * t / F(6)
        - a[3] * t * t * t / F(12)
        - a[4] * t * t * t * t / F(20)
        + a[5] / t
        - a[6]
    )


@wp.func
def _rise(x: F):
    z = wp.clamp(x, F(0), F(1))
    return z * z * (F(3) - F(2) * z)


@wp.func
def exchange(  # noqa: C901 - coupled reaction branches
    so2: F,
    so3: F,
    o2_pct: F,
    t: F,
    dt: F,
    b: Net,
    stage: int,
    trial: int,
    water_fraction: F,
):
    """Advance reversible oxygen exchange between SO2 and SO3, in ppm."""
    ans = F(0)
    if t > F(1000) and t < F(1423) and so2 + so3 > F(0):
        rt = wp.static(F(8.31446261815324)) * t
        cm = F(101325) / rt
        oxy = wp.static(F(0.01)) * o2_pct
        n2 = cm * wp.max(
            wp.static(F(1e-8)),
            F(1) - oxy - water_fraction - (so2 + so3) * wp.static(F(1e-6)),
        )
        # NASA7 differences for the same reaction equilibrium used by the dry
        # operator.
        eq1 = wp.exp(_dg(t, DISS_G)) * wp.static(F(1e5)) / rt
        eq2 = wp.exp(_dg(t, ABST_G))
        kd = wp.static(F(5.7e11)) * wp.exp(-F(40000) / t) * n2
        k2 = wp.static(F(7.8e5)) * wp.exp(-F(3065) / t)
        if stage >= 3:
            ab = (
                wp.static(F(0.028))
                * wp.pow(t, wp.static(F(2.57)))
                * wp.exp(-F(122300) / rt)
            )
            shift = wp.log(ab / k2)
            if stage == 3 and trial == 1:
                shift = F(5000) * (F(1) / F(1273) - F(1) / t)
            k2 = k2 * wp.exp(b[2] * shift)
        if stage >= 4:
            hi = wp.static(F(3.7e5)) * wp.exp(-F(850) / t)
            pr = (
                wp.static(F(2.9e15))
                * wp.pow(t, wp.static(F(-3.58)))
                * wp.exp(-F(2620) / t)
                * n2
                / hi
            )
            lf = wp.log10(
                wp.static(F(0.57)) * wp.exp(-t / F(371))
                + wp.static(F(0.43)) * wp.exp(-t / F(7442))
            )
            u = wp.log10(pr) - wp.static(F(0.4)) - wp.static(F(0.67)) * lf
            v = u / (
                wp.static(F(0.75))
                - wp.static(F(1.27)) * lf
                - wp.static(F(0.14)) * u
            )
            curve = (
                hi * pr / (F(1) + pr) * wp.pow(F(10), lf / (F(1) + v * v)) * eq1
            )
            shift = wp.log(curve / kd)
            if stage == 4 and trial == 1:
                shift = F(1000) * (F(1) / F(1273) - F(1) / t)
            kd = kd * wp.exp(b[3] * shift)
        if stage >= 1:
            fac = wp.static(F(0.6931471805599453))
            if stage == 1 and trial == 1:
                fac = F(2)
            kd = kd * wp.exp(b[0] * fac)
        if stage >= 2:
            fac = wp.static(F(1.0986122886681098))
            if stage == 2 and trial == 1:
                fac = fac * oxy / wp.static(F(0.04))
            k2 = k2 * wp.exp(b[1] * fac)
        c2 = so2 * wp.static(F(1e-6)) * cm
        c3 = so3 * wp.static(F(1e-6)) * cm
        ox = (kd * c3 + k2 / eq2 * oxy * cm * c2) / wp.max(
            wp.static(F(1e-30)), kd / eq1 * c2 + k2 * c3
        )
        forward = kd / eq1 * ox + k2 / eq2 * oxy * cm
        reverse = kd + k2 * ox
        tot = forward + reverse
        expo = (
            dt
            * tot
            * _rise((t - F(1000)) / F(50))
            * _rise((F(1423) - t) / F(23))
        )
        # (1-exp(-expo)) evaluated without cancellation at small steps.
        gain = F(1) - wp.exp(-expo)
        if expo < wp.static(F(1e-5)):
            gain = expo * (F(1) - expo / F(2) + expo * expo / F(6))
        ans = gain * (forward * so2 - reverse * so3) / tot
    return ans
