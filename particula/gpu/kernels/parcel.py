"""Float64 Warp integration for the warm parcel in dynamics.parcel.

Each time level owns disjoint rows of the trajectory and adjoint arrays.
Kernels read the previous level and write the next. A linearly implicit
Köhler relaxation stabilizes the steep subsaturated equilibrium branch.
"""

from contextlib import nullcontext

import numpy as np
import warp as wp

from particula.dynamics.parcel import (
    CP,
    LATENT,
    RD,
    RHO_W,
    RV,
    SIGMA,
    WATER_FACTOR,
    ParcelTrajectory,
    _saturation_pressure,
)
from particula.execution._condensation_ode_kernels import FloatVector
from particula.gpu.dynamics.condensation_funcs import _thermal_conductivity_wp
from particula.util.constants import STANDARD_GRAVITY

# Thermodynamic row layout, shared by kernels and the one final readback.
HEIGHT, PRESSURE, TEMP, VOLUME, VAPOR, LIQUID, SUPERSAT = range(7)
F = wp.float64
KELVIN_T = wp.constant(F(2 * SIGMA / (RV * RHO_W)))
LATENT_OVER_RV = wp.constant(F(LATENT / RV))
EPSILON = wp.constant(F(RD / RV))
R_DRY = wp.constant(F(RD))
R_VAPOR = wp.constant(F(RV))
GRAVITY = wp.constant(F(STANDARD_GRAVITY))
T_REF = wp.constant(F(273.15))
ES_REF = wp.constant(F(611.2))
D_REF = wp.constant(F(2.11e-5))
D_EXPONENT = wp.constant(F(1.94))


@wp.func
def _es(t: F):
    return ES_REF * wp.exp(LATENT_OVER_RV * (F(1.0) / T_REF - F(1.0) / t))


@wp.kernel
def _grow(
    previous: FloatVector,
    radius: FloatVector,
    dry: FloatVector,
    kappa: F,
    dt: F,
    active: bool,
    result: FloatVector,
):
    j = wp.tid()
    r = radius[j]
    new_radius = r
    if active:
        t = previous[TEMP]
        p = previous[PRESSURE]
        es = _es(t)
        d = dry[j]
        water = (r - d) * (r * r + r * d + d * d)
        solute = kappa * d * d * d
        denom = water + solute
        activity = water / denom
        kelvin_length = KELVIN_T / t
        curvature = wp.exp(kelvin_length / r)
        eq = activity * curvature
        diffusivity = D_REF * wp.pow(t / T_REF, D_EXPONENT) * F(101325.0) / p
        conduction = _thermal_conductivity_wp(t)
        resistance = F(RHO_W) * (
            R_VAPOR * t / (es * diffusivity)
            + F(LATENT) / (conduction * t) * (LATENT_OVER_RV / t - F(1.0))
        )
        # Linearize equilibrium about the old r² on its restoring branch.
        deq_dr = curvature * (
            F(3.0) * r * r * solute / (denom * denom)
            - activity * kelvin_length / (r * r)
        )
        relaxation = wp.max(deq_dr / (r * resistance), F(0.0))
        delta_r2 = (
            F(2.0)
            * dt
            * (previous[SUPERSAT] + F(1.0) - eq)
            / (resistance * (F(1.0) + dt * relaxation))
        )
        new_radius = wp.sqrt(r * r + delta_r2)
    result[j] = new_radius


@wp.kernel
def _liquid_water(
    wet: FloatVector,
    dry: FloatVector,
    weights: FloatVector,
    result: FloatVector,
):
    # Save the reduction as a tape intermediate. Later nonlinear kernels
    # need its actual forward value when Warp replays their adjoints.
    ql = F(0.0)
    for j in range(wet.shape[0]):
        r = wet[j]
        d = dry[j]
        ql += weights[j] * (r - d) * (r * r + r * d + d * d)
    result[0] = ql


@wp.kernel
def _advance(
    previous: FloatVector,
    liquid: FloatVector,
    speed: FloatVector,
    total_water: F,
    dry_mass: F,
    dt: F,
    result: FloatVector,
):
    ql = liquid[0]
    qv = total_water - ql
    dz = speed[0] * dt
    weight = GRAVITY * (F(1.0) + total_water)
    t = previous[TEMP] + (
        F(LATENT) * (ql - previous[LIQUID]) - weight * dz
    ) / F(CP)
    # Trapezoidal integration of d log(p) / dz = -g(1+qt)/[(Rd+qv Rv)T].
    gas_t = (R_DRY + qv * R_VAPOR) * t
    old_gas_t = (R_DRY + previous[VAPOR] * R_VAPOR) * previous[TEMP]
    p = previous[PRESSURE] * wp.exp(
        -F(0.5) * weight * dz * (F(1.0) / old_gas_t + F(1.0) / gas_t)
    )
    result[HEIGHT] = previous[HEIGHT] + dz
    result[PRESSURE] = p
    result[TEMP] = t
    result[VOLUME] = dry_mass * gas_t / p
    result[VAPOR] = qv
    result[LIQUID] = ql
    result[SUPERSAT] = p * qv / (EPSILON + qv) / _es(t) - F(1.0)


@wp.kernel
def _observe(state: FloatVector, loss: FloatVector):
    loss[0] = state[LIQUID]


def run_warp_parcel(
    initial, wet, time, updraft, device, gradient, phase_exchange
):
    """Advance from equilibrated radii, reading host outputs after the tape."""
    dry = np.asarray(initial.dry_radius)
    weights = WATER_FACTOR * np.asarray(initial.number) / initial.dry_air_mass
    liquid = weights @ ((wet - dry) * (wet * wet + wet * dry + dry * dry))
    vapor = initial.vapor
    temp = initial.temperature
    p = initial.pressure
    volume = initial.dry_air_mass * (RD + vapor * RV) * temp / p
    s = p * vapor / (RD / RV + vapor) / _saturation_pressure(temp) - 1
    options = dict(dtype=F, device=device)
    state = wp.empty((len(time), 7), requires_grad=gradient, **options)
    radius = wp.empty((len(time), len(dry)), requires_grad=gradient, **options)
    condensate = wp.empty((len(time), 1), requires_grad=gradient, **options)
    state[0].assign(
        np.array([initial.height, p, temp, volume, vapor, liquid, s])
    )
    radius[0].assign(wet)
    dry_device = wp.array(dry, **options)
    weights_device = wp.array(weights, **options)
    speed = wp.array([updraft], requires_grad=gradient, **options)
    tape = wp.Tape() if gradient else None
    sensitivity = None
    dt = time[1]
    with tape if tape is not None else nullcontext():
        for i in range(1, len(time)):
            previous = state[i - 1]
            next_radius = radius[i]
            wp.launch(
                _grow,
                len(dry),
                inputs=[
                    previous,
                    radius[i - 1],
                    dry_device,
                    initial.kappa,
                    dt,
                    phase_exchange,
                ],
                outputs=[next_radius],
                device=device,
            )
            liquid_row = condensate[i]
            wp.launch(
                _liquid_water,
                1,
                inputs=[next_radius, dry_device, weights_device],
                outputs=[liquid_row],
                device=device,
            )
            wp.launch(
                _advance,
                1,
                inputs=[
                    previous,
                    liquid_row,
                    speed,
                    vapor + liquid,
                    initial.dry_air_mass,
                    dt,
                ],
                outputs=[state[i]],
                device=device,
            )
        if tape is not None:
            loss = wp.empty(1, requires_grad=True, **options)
            wp.launch(
                _observe,
                1,
                inputs=[state[len(time) - 1]],
                outputs=[loss],
                device=device,
            )
    if tape is not None:
        tape.backward(loss)
        sensitivity = float(speed.grad.numpy()[0])
    h, p, temp, vol, qv, ql, s = state.numpy().T
    return ParcelTrajectory(
        time, h, p, temp, vol, qv, ql, s, radius.numpy(), sensitivity, initial
    )
