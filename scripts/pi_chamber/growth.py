"""Implicit aqueous NaCl growth tables for prescribed chamber humidity."""

import numpy as np
import particula as par
import warp as wp
from particula.gpu.dynamics.condensation_funcs import (
    _mass_transfer_rate_latent_heat_wp,
    first_order_mass_transport_k_wp,
)
from particula.gpu.properties.particle_properties import (
    kelvin_term_wp,
    vapor_transition_correction_wp,
)

KELVIN = wp.constant(
    wp.float64(2 * 0.0728 * 0.01801528 / (8.314462618 * 293.15 * 998.2))
)
F = wp.float64
T = wp.constant(F(293.15))
ONE = wp.constant(F(1))
DV = wp.constant(F(2.5e-5))
ES = wp.constant(F(2338.3))
MW = wp.constant(F(0.01801528))
RG = wp.constant(F(8.314462618))
LAM = wp.constant(
    F(par.gas.get_molecule_mean_free_path(0.01801528, 293.15, 101325.0))
)
PI = wp.constant(F(np.pi))
SMAX = wp.constant(
    F(0.12)
)  # allows +/-12 percentage points, checked at runtime


@wp.func
def eqsat(wet: F, dry: F, k: F):
    """Return the κ–Köhler equilibrium saturation ratio."""
    v = (wet / dry) ** F(3.0) - ONE
    return v / (v + k) * kelvin_term_wp(wet * F(0.5e-6), KELVIN)


@wp.kernel
def build_steps(
    drys: wp.array(dtype=F),
    kappas: wp.array(dtype=F),
    critical: wp.array(dtype=F),
    steps: wp.array3d(dtype=wp.float32),
    dt: F,
):
    """Tabulate the implicit change in squared wet diameter (µm²)."""
    kind, j, humidity_index = wp.tid()
    d = drys[kind]
    k = kappas[kind]
    wet = d * wp.exp(wp.log(F(120) / d) * F(j) / F(steps.shape[1] - 1))
    target = (
        ONE - SMAX + F(2) * SMAX * F(humidity_index) / F(steps.shape[2] - 1)
    )
    surface = eqsat(wet, d, k)
    radius = wet * F(0.5e-6)
    beta = vapor_transition_correction_wp(LAM / radius, ONE)
    conductance = first_order_mass_transport_k_wp(radius, beta, DV)
    # Positive coefficient d(D_um^2)/dt per unit saturation difference.
    rate = _mass_transfer_rate_latent_heat_wp(
        ES, conductance, T, MW, F(2.45e6), F(0.02520365), ES * surface, DV, RG
    )
    scale = F(2e12) * rate / (PI * F(998.2) * radius)
    old = wet * wet
    peak = critical[kind]
    lo = d * d
    hi = old + dt * scale * target
    if target >= surface:
        lo = old
        if wet < peak and target < eqsat(peak, d, k):
            hi = wp.min(hi, peak * peak)
    else:
        hi = old
        if wet > peak and target >= eqsat(peak, d, k):
            lo = peak * peak
    # Solve equilibrium implicitly on the accessible branch.
    for _ in range(38):
        mid = (lo + hi) * F(0.5)
        residual = mid - old - dt * scale * (target - eqsat(wp.sqrt(mid), d, k))
        if residual > F(0):
            hi = mid
        else:
            lo = mid
    steps[kind, j, humidity_index] = wp.float32((lo + hi) * F(0.5) - old)


@wp.func
def grow(y: F, dry: F, kind: int, s: F, lut: wp.array3d(dtype=wp.float32)):
    """Interpolate one growth step in squared diameter and saturation."""
    a = wp.max(
        F(0),
        wp.log(wp.sqrt(y) / dry) / wp.log(F(120) / dry) * F(lut.shape[1] - 1),
    )
    b = (s + SMAX) / (F(2) * SMAX) * F(lut.shape[2] - 1)
    j = wp.clamp(int(a), 0, lut.shape[1] - 2)
    humidity_index = wp.clamp(int(b), 0, lut.shape[2] - 2)
    ratio = wp.exp(F(2) * wp.log(F(120) / dry) / F(lut.shape[1] - 1))
    yleft = dry * dry * wp.pow(ratio, F(j))
    u = (y / yleft - ONE) / (ratio - ONE)
    v = b - F(humidity_index)
    dy = (ONE - v) * (
        (ONE - u) * F(lut[kind, j, humidity_index])
        + u * F(lut[kind, j + 1, humidity_index])
    ) + v * (
        (ONE - u) * F(lut[kind, j, humidity_index + 1])
        + u * F(lut[kind, j + 1, humidity_index + 1])
    )
    return y + dy
