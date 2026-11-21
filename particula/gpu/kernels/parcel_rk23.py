"""Shared adaptive warm-parcel integration and explicit RK3(2) stages.

The independent coordinates are p/p0 and liquid volume / dry volume per
class. At every stage total water and the parcel first law determine vapor
and temperature. Only scalar trial errors cross to the host during integration.
The accepted trajectory uses the selected third-order method on replay.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from particula.dynamics.parcel import (
    CP,
    LATENT,
    WATER_FACTOR,
    ParcelROS3,
    ParcelStepStatistics,
    ParcelTrajectory,
)
from particula.execution._condensation_ode_kernels import (
    FloatVector,
    error_norm,
    stage,
)
from particula.gpu.kernels.parcel import (
    EPSILON,
    GRAVITY,
    KELVIN_T,
    LATENT_OVER_RV,
    R_DRY,
    R_VAPOR,
    F,
    _es,
    _growth_resistance,
)

_BATCH_CAPACITY = 32


@wp.struct
class _Parcel:
    temperature: F
    pressure: F
    vapor: F
    liquid: F
    kappa: F
    dry_mass: F
    height: F
    exchange: bool


@wp.func
def _temperature(parcel: _Parcel, liquid: F, dz: F):
    loading = F(1.0) + parcel.vapor + parcel.liquid
    return parcel.temperature + (
        F(LATENT) * (liquid - parcel.liquid) - GRAVITY * loading * dz
    ) / F(CP)


@wp.kernel
def _sum_water(state: FloatVector, weights: FloatVector, liquid: FloatVector):
    # Keep this reduction available to the adjoint of each nonlinear stage.
    total = F(0.0)
    for j in range(weights.shape[0]):
        total += state[j + 1] * weights[j]
    liquid[0] = total


@wp.kernel
def _rate(
    state: FloatVector,
    liquid: FloatVector,
    dry: FloatVector,
    speed: FloatVector,
    parcel: _Parcel,
    time: F,
    result: FloatVector,
):
    i = wp.tid()
    ql = liquid[0]
    qv = parcel.vapor + (parcel.liquid - ql)
    temp = _temperature(parcel, ql, speed[0] * time)
    p = parcel.pressure * state[0]
    # A nonphysical stage invalidates the whole embedded trial. The controller
    # discards its private workspace; the accepted state is untouched.
    value = F(wp.nan)
    if (
        wp.isfinite(p)
        and wp.isfinite(qv)
        and wp.isfinite(temp)
        and p > F(0.0)
        and qv >= F(0.0)
        and temp > F(0.0)
    ):
        if i == 0:
            value = (
                -state[0]
                * GRAVITY
                * (F(1.0) + parcel.vapor + parcel.liquid)
                * speed[0]
                / ((R_DRY + qv * R_VAPOR) * temp)
            )
        elif state[i] >= F(0.0):
            value = F(0.0)
            if parcel.exchange:
                d = dry[i - 1]  # type: ignore[operator]  # 1-D launch
                water = state[i]
                r = d * wp.cbrt(F(1.0) + water)
                es = _es(temp)
                rh = p * qv / (EPSILON + qv) / es
                eq = (
                    water
                    / (water + parcel.kappa)
                    * wp.exp(KELVIN_T / (temp * r))
                )
                value = (
                    F(3.0)
                    * r
                    * (rh - eq)
                    / (_growth_resistance(temp, p, es) * d * d * d)
                )
    result[i] = value


@wp.func
def _relaxation_rate(
    state: FloatVector,
    dry: FloatVector,
    weights: FloatVector,
    vapor: F,
    temp: F,
    pressure: F,
    kappa: F,
):
    # Near equilibrium a zero forward residual can hide a fast tangent mode.
    # Linearize the growth drive in class liquid/dry volume and total liquid.
    # The latter affects every class via vapor depletion and latent heating.
    es = _es(temp)
    resistance = _growth_resistance(temp, pressure, es)
    rh = pressure * vapor / (EPSILON + vapor) / es
    rh_vapor = pressure * EPSILON / (es * (EPSILON + vapor) * (EPSILON + vapor))
    fastest = F(0.0)
    collective = F(0.0)
    for j in range(weights.shape[0]):
        x = state[j + 1]
        d = dry[j]
        r = d * wp.cbrt(F(1.0) + x)
        exponent = KELVIN_T / (temp * r)
        kelvin = wp.exp(exponent)
        eq = kelvin * x / (x + kappa)
        growth = F(3.0) * r / (resistance * d * d * d)
        slope = kelvin * kappa / ((x + kappa) * (x + kappa)) - eq * exponent / (
            F(3.0) * (F(1.0) + x)
        )
        fastest = wp.max(fastest, growth * slope)
        thermal = (rh * LATENT_OVER_RV / temp - eq * exponent) / temp
        collective += (
            weights[j] * growth * wp.abs(rh_vapor + F(LATENT) / F(CP) * thermal)
        )
    return fastest + collective


@wp.kernel
def _reservoir_error(
    state: FloatVector,
    dry: FloatVector,
    weights: FloatVector,
    k1: FloatVector,
    k2: FloatVector,
    k3: FloatVector,
    k4: FloatVector,
    old_liquid: FloatVector,
    new_liquid: FloatVector,
    speed: FloatVector,
    parcel: _Parcel,
    time: F,
    h: F,
    rtol: F,
    norm: FloatVector,
):
    # Project the same embedded third-minus-second-order difference into
    # vapor and temperature. This matters when very little vapor remains.
    delta = F(0.0)
    for j in range(weights.shape[0]):
        i = j + 1
        delta += (
            weights[j]
            * h
            * (
                -F(5.0) / F(72.0) * k1[i]
                + k2[i] / F(12.0)
                + k3[i] / F(9.0)
                - k4[i] / F(8.0)
            )
        )
    vapor_start = parcel.vapor + (parcel.liquid - old_liquid[0])
    vapor_end = parcel.vapor + (parcel.liquid - new_liquid[0])
    temp_start = _temperature(parcel, old_liquid[0], speed[0] * time)
    temp_end = _temperature(parcel, new_liquid[0], speed[0] * (time + h))
    vapor_error = wp.abs(delta) / (
        rtol * (F(1e-12) + wp.max(vapor_start, vapor_end))
    )
    temp_error = wp.abs(F(LATENT) / F(CP) * delta) / (
        rtol * wp.max(temp_start, temp_end)
    )
    value = wp.max(vapor_error, temp_error)
    if parcel.exchange and parcel.vapor + parcel.liquid > F(0.0):
        rate = _relaxation_rate(
            state,
            dry,
            weights,
            vapor_start,
            temp_start,
            state[0] * parcel.pressure,
            parcel.kappa,
        )
        # h*rate <= 2 lies inside RK3's negative-real stability interval.
        # Cube this ratio to use the controller's existing 1/8 doubling test.
        ratio = F(0.5) * h * rate
        value = wp.max(value, ratio * ratio * ratio)
    if not wp.isfinite(value):
        value = F(1e300)
    wp.atomic_max(norm, 0, value)


@wp.kernel
def _sample(
    state: FloatVector,
    liquid: FloatVector,
    dry: FloatVector,
    speed: FloatVector,
    parcel: _Parcel,
    time: F,
    air: FloatVector,
    wet: FloatVector,
):
    i = wp.tid()
    if i == 0:
        dz = time * speed[0]
        ql = liquid[0]
        qv = parcel.vapor + (parcel.liquid - ql)
        temp = _temperature(parcel, ql, dz)
        p = state[0] * parcel.pressure
        air[0] = parcel.height + dz
        air[1] = p
        air[2] = temp
        air[3] = parcel.dry_mass * (R_DRY + qv * R_VAPOR) * temp / p
        air[4] = qv
        air[5] = ql
        air[6] = p * qv / (EPSILON + qv) / _es(temp) - F(1.0)
    j = i + 1  # type: ignore[operator]  # 1-D launch
    wet[i] = dry[i] * wp.cbrt(F(1.0) + state[j])


@dataclass
class _Stages:
    """Three slope vectors, two stage states and their saved water sums."""

    k1: FloatVector
    k2: FloatVector
    k3: FloatVector
    s2: FloatVector
    s3: FloatVector
    q1: FloatVector
    q2: FloatVector
    q3: FloatVector


class _ParcelStepper:
    """Shared physical rates and accepted-step reverse replay."""

    def __init__(self, initial, wet, speed, device, gradient, exchange):
        self.device = device
        self.width = len(wet) + 1
        self.speed = self.array([speed], grad=gradient)
        self.dry = self.array(initial.dry_radius)
        d = initial.dry_radius
        weights = WATER_FACTOR * initial.number * d**3 / initial.dry_air_mass
        self.weights = self.array(weights)
        water = (wet - d) * (wet * wet + wet * d + d * d) / d**3
        self.initial_state = self.array(np.r_[1.0, water])
        self.seed = self.array(np.r_[0.0, weights]) if gradient else None
        self.parcel = _Parcel()
        self.parcel.temperature = initial.temperature
        self.parcel.pressure = initial.pressure
        self.parcel.vapor = initial.vapor
        self.parcel.kappa = initial.kappa
        self.parcel.height = initial.height
        self.parcel.dry_mass = initial.dry_air_mass
        self.parcel.exchange = exchange
        # Obtain exactly the same initial sum as the device stage reduction.
        temporary = self.empty(1)
        self.launch(
            _sum_water, 1, [self.initial_state, self.weights], temporary
        )
        self.parcel.liquid = float(temporary.numpy()[0])

    def array(self, values, grad=False):
        return wp.array(values, dtype=F, device=self.device, requires_grad=grad)

    def empty(self, shape, grad=False):
        return wp.empty(shape, dtype=F, device=self.device, requires_grad=grad)

    def launch(self, kernel, dim, args, out):
        wp.launch(
            kernel, dim=dim, inputs=args, outputs=[out], device=self.device
        )

    def rate(self, state, t, derivative, liquid):
        self.launch(_sum_water, 1, [state, self.weights], liquid)
        self.launch(
            _rate,
            self.width,
            [state, liquid, self.dry, self.speed, self.parcel, t],
            derivative,
        )

    def advance(self, state, t, h, out, work):
        raise NotImplementedError

    def scratch(self, grad=False):
        raise NotImplementedError

    @staticmethod
    def record_step(checkpoints, accepted, start, t, h):
        """Keep one device checkpoint for each block of accepted intervals."""
        if accepted % 32 == 0:
            checkpoints.append((wp.clone(start, requires_grad=True), []))
        checkpoints[-1][1].append((t, h))

    def reverse(self, checkpoints):
        """Bound tape storage by replaying at most 32 accepted steps at once."""
        seed = self.seed
        for checkpoint, intervals in reversed(checkpoints):
            state = checkpoint
            with wp.Tape() as tape:
                for t, h in intervals:
                    out = self.empty(self.width, True)
                    self.advance(state, t, h, out, self.scratch(True))
                    state = out
            tape.backward(grads={state: seed})
            seed = checkpoint.grad
        return float(self.speed.grad.numpy()[0])


class _RK23Stepper(_ParcelStepper):
    def advance(self, state, t, h, out, work, *, reuse_first=False):
        """Write third-order result, retaining distinct stage values for AD."""
        k1, k2, k3 = work.k1, work.k2, work.k3
        if not reuse_first:
            self.rate(state, t, k1, work.q1)
        self.launch(
            stage, self.width, [state, k1, k1, k1, h, 0.5, 0.0, 0.0], work.s2
        )
        self.rate(work.s2, t + 0.5 * h, k2, work.q2)
        self.launch(
            stage, self.width, [state, k2, k2, k2, h, 0.75, 0.0, 0.0], work.s3
        )
        self.rate(work.s3, t + 0.75 * h, k3, work.q3)
        self.launch(
            stage, self.width, [state, k1, k2, k3, h, 2 / 9, 1 / 3, 4 / 9], out
        )

    def scratch(self, grad=False):
        return _Stages(
            *(self.empty(self.width, grad) for _ in range(5)),
            *(self.empty(1, grad) for _ in range(3)),
        )


class _TrialBatch:
    """Stage speculative candidates in disjoint rows until the next decision."""

    def __init__(self, solver, tolerance):
        self.solver = solver
        self.tolerance = tolerance
        # A single-box CUDA trial is small. Amortize device/host waits when
        # several consecutive steps take the same size on a shared GPU.
        self.capacity = _BATCH_CAPACITY
        self.states = [solver.empty(solver.width) for _ in range(self.capacity)]
        self.errors = solver.empty((self.capacity, 1))

    def evaluate(self, initial, start, target, proposed, count):
        if count <= 0:
            raise ValueError(
                f"adaptive parcel max_trials exhausted at {start:g} s"
            )
        t, state = start, initial
        steps = []
        self.errors.zero_()
        self.start(state, t)
        for j in range(count):
            remaining = target - t
            # Snap a roundoff-sized remainder to the requested observation.
            h = (
                remaining
                if remaining <= proposed + 2 * np.spacing(target)
                else proposed
            )
            if t + h <= t:
                raise ValueError(
                    f"adaptive parcel cannot advance time at {t:g} s"
                )
            out, error = self.states[j], self.errors[j]
            self.trial(state, t, h, out, error)
            steps.append(h)
            if h == remaining:
                break
            self.reuse_endpoint()
            t, state = t + h, out
        return steps, self.errors.numpy()[: len(steps), 0]

    def start(self, state, t):
        """Prepare any reusable first-stage derivative."""

    def trial(self, state, t, h, out, error):
        raise NotImplementedError

    def reuse_endpoint(self):
        """Carry endpoint derivatives into the next candidate if available."""


class _RK23Trials(_TrialBatch):
    def __init__(self, solver, tolerance):
        super().__init__(solver, tolerance)
        self.rtol = solver.array(np.full(solver.width, tolerance))
        self.atol = solver.array(np.full(solver.width, tolerance * 1e-8))
        self.work = solver.scratch()
        self.k4 = solver.empty(solver.width)
        self.ql = solver.empty(1)

    def start(self, state, t):
        self.solver.rate(state, t, self.work.k1, self.work.q1)

    def trial(self, state, t, h, out, error):
        solver, work = self.solver, self.work
        solver.advance(state, t, h, out, work, reuse_first=True)
        solver.rate(out, t + h, self.k4, self.ql)
        rates = [work.k1, work.k2, work.k3, self.k4]
        solver.launch(
            error_norm,
            solver.width,
            [state, out, *rates, self.atol, self.rtol, h],
            error,
        )
        solver.launch(
            _reservoir_error,
            1,
            [
                state,
                solver.dry,
                solver.weights,
                *rates,
                work.q1,
                self.ql,
                solver.speed,
                solver.parcel,
                t,
                h,
                self.tolerance,
            ],
            error,
        )

    def reuse_endpoint(self):
        work = self.work
        # The last slope of a trial is the first slope of its successor.
        work.k1, self.k4 = self.k4, work.k1
        work.q1, self.ql = self.ql, work.q1


def _adaptive_method(controls):
    """Select the concrete Warp stepper and its error estimate."""
    if isinstance(controls, ParcelROS3):
        from particula.gpu.kernels.parcel_ros3 import _ROS3Stepper, _ROS3Trials

        return _ROS3Stepper, _ROS3Trials
    return _RK23Stepper, _RK23Trials


def run_adaptive_parcel(
    initial, wet, time, updraft, device, gradient, phase_exchange, controls
):
    """Integrate to fixed observations with an embedded componentwise error."""
    solver_type, trial_type = _adaptive_method(controls)
    solver = solver_type(
        initial, wet, updraft, device, gradient, phase_exchange
    )
    state = solver.initial_state
    trials = trial_type(solver, controls.rtol)
    liquid = solver.empty(1)
    air = solver.empty((len(time), 7))
    radii = solver.empty((len(time), len(wet)))
    accepted = rejected = evaluated = 0
    smallest, largest = np.inf, 0.0
    t = 0.0
    proposed = time[1]
    batch_size = 1
    checkpoints = []
    for i, target in enumerate(time):
        while t < target:
            budget = controls.max_trials - evaluated
            steps, errors = trials.evaluate(
                state, t, target, proposed, min(budget, batch_size)
            )
            evaluated += len(steps)
            last = state
            for j, (h, norm) in enumerate(zip(steps, errors, strict=True)):
                if norm > 1 or not np.isfinite(norm):
                    rejected += 1
                    if h <= controls.min_step:
                        raise ValueError(
                            f"adaptive parcel cannot meet tolerance at {t:g} s"
                        )
                    proposed = max(controls.min_step, h / 2)
                    batch_size = 1
                    break
                if gradient:
                    solver.record_step(checkpoints, accepted, last, t, h)
                accepted += 1
                smallest, largest = min(smallest, h), max(largest, h)
                t = target if h == target - t else t + h
                last = trials.states[j]
                # The embedded difference is O(h³). Keep the dyadic mesh
                # locally constant in speed between controller decisions.
                if norm <= 0.125 and h == proposed:
                    proposed = h * 2
                    batch_size = 1
                    break
                if norm > 0.125:
                    proposed = h
                batch_size = trials.capacity
            if last is not state:
                wp.copy(state, last)
        solver.launch(_sum_water, 1, [state, solver.weights], liquid)
        wp.launch(
            _sample,
            len(wet),
            inputs=[
                state,
                liquid,
                solver.dry,
                solver.speed,
                solver.parcel,
                target,
            ],
            outputs=[air[i], radii[i]],
            device=device,
        )
    sensitivity = solver.reverse(checkpoints) if gradient else None
    z, p, temp, volume, qv, ql, s = air.numpy().T
    return ParcelTrajectory(
        time,
        z,
        p,
        temp,
        volume,
        qv,
        ql,
        s,
        radii.numpy(),
        sensitivity,
        initial,
        ParcelStepStatistics(accepted, rejected, smallest, largest, evaluated),
    )
