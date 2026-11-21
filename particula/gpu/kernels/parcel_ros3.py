"""Linearly implicit ROS3(2) for the closed warm parcel.

In normalized coordinates each row depends on local water, pressure and the
weighted sum of class water. Eliminating pressure and solving for that sum
gives an O(classes) solve for each of the three Rosenbrock increments. Warp
differentiates the factorization as well as the stages on accepted-step replay.
"""

from dataclasses import dataclass

import warp as wp

from particula.dynamics.parcel import CP, LATENT, RHO_W
from particula.execution import _condensation_ros3_kernels as algebra
from particula.execution._condensation_ode_kernels import FloatVector
from particula.execution._ros3_coefficients import E2, E3, GAMMA, M2, M3
from particula.gpu.dynamics.condensation_funcs import (
    _THERMAL_CONDUCTIVITY_GRADIENT,
    _THERMAL_CONDUCTIVITY_SCALE,
)
from particula.gpu.kernels.parcel import (
    D_EXPONENT,
    EPSILON,
    GRAVITY,
    KELVIN_T,
    LATENT_OVER_RV,
    R_DRY,
    R_VAPOR,
    F,
    _es,
    _resistance_terms,
)
from particula.gpu.kernels.parcel_rk23 import (
    _Parcel,
    _ParcelStepper,
    _sum_water,
    _temperature,
    _TrialBatch,
)


@wp.kernel
def _linearize(
    state: FloatVector,
    liquid: FloatVector,
    rate: FloatVector,
    dry: FloatVector,
    speed: FloatVector,
    parcel: _Parcel,
    time: F,
    h: F,
    inverse_diagonal: FloatVector,
    pressure_column: FloatVector,
    liquid_column: FloatVector,
    partial_time: FloatVector,
):
    """Form the local, pressure, shared-water and explicit time derivatives."""
    i = wp.tid()
    y = state[0]
    p = parcel.pressure * y
    vapor = parcel.vapor + (parcel.liquid - liquid[0])
    temp = _temperature(parcel, liquid[0], speed[0] * time)
    loading = F(1.0) + parcel.vapor + parcel.liquid
    force_t = -loading * GRAVITY * speed[0] / F(CP)
    temp_slope = F(0.0)
    local = F(0.0)
    dp = F(0.0)
    dl = F(0.0)
    if i == 0:
        local = rate[i] / y
        temp_slope = -rate[i] / temp
        dl = rate[i] * R_VAPOR / (R_DRY + vapor * R_VAPOR) + temp_slope * F(
            LATENT
        ) / F(CP)
    elif parcel.exchange:
        x = state[i]
        d = dry[i - 1]  # type: ignore[operator]
        r = d * wp.cbrt(F(1.0) + x)
        es = _es(temp)
        rh = p * vapor / ((EPSILON + vapor) * es)
        exponent = KELVIN_T / (temp * r)
        curvature = wp.exp(exponent)
        eq = curvature * x / (x + parcel.kappa)
        mass_term, heat_term, conductivity = _resistance_terms(temp, p, es)
        mass = F(RHO_W) * mass_term
        heat_scale = F(RHO_W) * heat_term
        heat = heat_scale * (LATENT_OVER_RV / temp - F(1.0))
        resistance = mass + heat
        growth = F(3.0) * r / (resistance * d * d * d)
        es_log_slope = LATENT_OVER_RV / (temp * temp)
        resistance_t = (
            mass * ((F(1.0) - D_EXPONENT) / temp - es_log_slope)
            - heat
            * (
                F(1.0) / temp
                + _THERMAL_CONDUCTIVITY_SCALE
                * _THERMAL_CONDUCTIVITY_GRADIENT
                / conductivity
            )
            - heat_scale * es_log_slope
        )
        radius_log_slope = F(1.0) / (F(3.0) * (F(1.0) + x))
        eq_slope = (
            curvature * parcel.kappa / ((x + parcel.kappa) * (x + parcel.kappa))
            - eq * exponent * radius_log_slope
        )
        local = rate[i] * radius_log_slope - growth * eq_slope
        dp = (growth * rh - rate[i] * mass / resistance) / y
        temp_slope = (
            growth * (eq * exponent / temp - rh * es_log_slope)
            - rate[i] * resistance_t / resistance
        )
        rh_vapor = p * EPSILON / (es * (EPSILON + vapor) * (EPSILON + vapor))
        dl = temp_slope * F(LATENT) / F(CP) - growth * rh_vapor
    factor = F(1.0) / (F(1.0) / (GAMMA * h) - local)
    inverse_diagonal[i] = factor
    pressure_column[i] = factor * dp
    liquid_column[i] = factor * dl
    partial_time[i] = temp_slope * force_t


@wp.kernel
def _water_increment(
    rhs: FloatVector,
    inv: FloatVector,
    dp: FloatVector,
    dl: FloatVector,
    weights: FloatVector,
    result: FloatVector,
):
    # k_i = inv_i rhs_i + dp_i k_p + dl_i k_L, k_L = sum(w_i k_i).
    base_p = inv[0] * rhs[0]
    base = F(0.0)
    feedback = F(0.0)
    for j in range(weights.shape[0]):
        i = j + 1
        base += weights[j] * (inv[i] * rhs[i] + dp[i] * base_p)
        feedback += weights[j] * (dl[i] + dp[i] * dl[0])
    # Keep the reduced numerator and feedback for nonlinear adjoint replay.
    result[0] = base
    result[1] = feedback


@wp.kernel
def _back_substitute(
    rhs: FloatVector,
    inv: FloatVector,
    dp: FloatVector,
    dl: FloatVector,
    liquid: FloatVector,
    result: FloatVector,
):
    i = wp.tid()
    increment = liquid[0] / (F(1.0) - liquid[1])
    pressure = inv[0] * rhs[0] + dl[0] * increment
    result[i] = inv[i] * rhs[i] + dp[i] * pressure + dl[i] * increment


@wp.kernel
def _finish(
    state: FloatVector,
    first: FloatVector,
    second: FloatVector,
    third: FloatVector,
    out: FloatVector,
):
    i = wp.tid()
    out[i] = state[i] + first[i] + M2 * second[i] - M3 * third[i]


@wp.kernel
def _error(
    state: FloatVector,
    candidate: FloatVector,
    first: FloatVector,
    second: FloatVector,
    third: FloatVector,
    rtol: F,
    embedded: FloatVector,
    norm: FloatVector,
):
    i = wp.tid()
    diff = F(0.5) * first[i] - E2 * second[i] + E3 * third[i]
    embedded[i] = diff
    value = wp.abs(diff) / (
        rtol * (F(1e-8) + wp.max(wp.abs(state[i]), wp.abs(candidate[i])))
    )
    if (
        not wp.isfinite(value)
        or not wp.isfinite(candidate[i])
        or candidate[i] < F(0.0)
        or (i == 0 and candidate[i] <= F(0.0))
    ):
        value = F(1e300)
    wp.atomic_max(norm, 0, value)


@wp.kernel
def _reservoir_norm(
    old: FloatVector,
    new_liquid: FloatVector,
    error: FloatVector,
    speed: FloatVector,
    parcel: _Parcel,
    t: F,
    h: F,
    rtol: F,
    norm: FloatVector,
):
    old_v = parcel.vapor + (parcel.liquid - old[0])
    new_v = parcel.vapor + (parcel.liquid - new_liquid[0])
    old_t = _temperature(parcel, old[0], speed[0] * t)
    new_t = _temperature(parcel, new_liquid[0], speed[0] * (t + h))
    delta = wp.abs(error[0])
    value = wp.max(
        delta / (rtol * (F(1e-12) + wp.max(old_v, new_v))),
        delta * F(LATENT) / (F(CP) * rtol * wp.max(old_t, new_t)),
    )
    if (
        not wp.isfinite(value)
        or not wp.isfinite(new_t)
        or not wp.isfinite(new_v)
        or new_t <= F(0.0)
        or new_v < F(0.0)
    ):
        value = F(1e300)
    wp.atomic_max(norm, 0, value)


@dataclass
class _ROS3Stages:
    inverse: FloatVector
    dp: FloatVector
    dl: FloatVector
    ft: FloatVector
    base_rate: FloatVector
    stage_rate: FloatVector
    stage: FloatVector
    old_liquid: FloatVector
    stage_liquid: FloatVector
    rhs: list
    water: list
    increments: list


class _ROS3Stepper(_ParcelStepper):
    """Three solves with one parcel linearization per trial or replay."""

    def scratch(self, grad=False):
        return _ROS3Stages(
            *(self.empty(self.width, grad) for _ in range(7)),
            self.empty(1, grad),
            self.empty(1, grad),
            [self.empty(self.width, grad) for _ in range(3)],
            [self.empty(2, grad) for _ in range(3)],
            [self.empty(self.width, grad) for _ in range(3)],
        )

    def advance(self, state, t, h, out, work):
        self.rate(state, t, work.base_rate, work.old_liquid)
        wp.launch(
            _linearize,
            self.width,
            inputs=[
                state,
                work.old_liquid,
                work.base_rate,
                self.dry,
                self.speed,
                self.parcel,
                t,
                h,
            ],
            outputs=[work.inverse, work.dp, work.dl, work.ft],
            device=self.device,
        )
        rate = work.base_rate
        first, second, third = work.increments
        for j, increment in enumerate(work.increments):
            # Stages 2 and 3 share their rate and forcing time.
            rhs = work.rhs[j]
            self.launch(
                algebra.stage_rhs,
                self.width,
                [
                    rate,
                    first if j else rate,
                    second if j == 2 else rate,
                    work.ft,
                    h,
                    j + 1,
                ],
                rhs,
            )
            factors = [rhs, work.inverse, work.dp, work.dl]
            self.launch(
                _water_increment, 1, [*factors, self.weights], work.water[j]
            )
            self.launch(
                _back_substitute,
                self.width,
                [*factors, work.water[j]],
                increment,
            )
            if j == 0:
                self.launch(
                    algebra.stage_state, self.width, [state, first], work.stage
                )
                self.rate(
                    work.stage,
                    t + float(GAMMA) * h,
                    work.stage_rate,
                    work.stage_liquid,
                )
                rate = work.stage_rate
        self.launch(_finish, self.width, [state, first, second, third], out)


class _ROS3Trials(_TrialBatch):
    def __init__(self, solver, tolerance):
        super().__init__(solver, tolerance)
        self.work = solver.scratch()
        self.embedded = solver.empty(solver.width)
        self.error_water = solver.empty(1)
        self.new_water = solver.empty(1)

    def trial(self, state, t, h, out, error):
        solver = self.solver
        solver.advance(state, t, h, out, self.work)
        wp.launch(
            _error,
            solver.width,
            inputs=[state, out, *self.work.increments, self.tolerance],
            outputs=[self.embedded, error],
            device=solver.device,
        )
        solver.launch(
            _sum_water, 1, [self.embedded, solver.weights], self.error_water
        )
        solver.launch(_sum_water, 1, [out, solver.weights], self.new_water)
        solver.launch(
            _reservoir_norm,
            1,
            [
                self.work.old_liquid,
                self.new_water,
                self.error_water,
                solver.speed,
                solver.parcel,
                t,
                h,
                self.tolerance,
            ],
            error,
        )
