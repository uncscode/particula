"""GPU-owned adaptive clock, acceptance and step-size control.

The host supplies an observation bound. A native CUDA conditional graph runs
trials to that bound without control readbacks. Float control entries are
h, time, next_step, bound, min_step, max_step, last_error. Integer counters
are accepted, rejected, domain_rejected, failure_code, consecutive_rejections.
"""

import warp as wp

from particula.execution._condensation_ode_kernels import (
    FloatMatrix,
    FloatTensor,
    FloatVector,
    IntVector,
)
from particula.execution._condensation_ros3_kernels import C21, C31, C32


@wp.kernel
def set_bound(
    control: FloatVector,
    condition: IntVector,
    counters: IntVector,
    bound: wp.float64,
):
    control[3] = bound
    condition[0] = int(control[1] < bound and counters[3] == 0)


@wp.kernel
def begin_trial(
    control: FloatVector,
    invalid: IntVector,
    norm: FloatVector,
    accepted: IntVector,
):
    control[0] = wp.min(wp.min(control[2], control[5]), control[3] - control[1])
    invalid[0] = 0
    norm[0] = wp.float64(0.0)
    accepted[0] = 0


@wp.kernel
def accumulate_invalid(current: IntVector, sticky: IntVector):
    if current[0] != 0:
        sticky[0] = 1


@wp.kernel
def decide(
    control: FloatVector,
    norm: FloatVector,
    invalid: IntVector,
    accepted: IntVector,
    condition: IntVector,
    counters: IntVector,
    max_trials: int,
):
    h = control[0]
    floor = wp.min(control[4], control[3] - control[1])
    if h < floor or control[1] + h == control[1]:
        counters[3] = 1
        condition[0] = 0
        return
    error = norm[0]
    domain = invalid[0] != 0 or not wp.isfinite(error)
    if not domain and error <= wp.float64(1.0):
        accepted[0] = 1
        counters[0] += 1
        counters[4] = 0
        control[6] = error
        if h == control[3] - control[1]:
            control[1] = control[3]
        else:
            control[1] += h
        factor = wp.float64(5.0)
        if error > wp.float64(0.0):
            factor = wp.min(
                wp.float64(5.0),
                wp.max(
                    wp.float64(0.2),
                    wp.float64(0.9)
                    * wp.pow(error, -wp.float64(1.0) / wp.float64(3.0)),
                ),
            )
        control[2] = wp.min(control[5], wp.max(control[4], h * factor))
    else:
        counters[1] += 1
        counters[2] += int(domain)
        counters[4] += 1
        factor = wp.float64(0.5)
        if not domain:
            factor = wp.max(
                wp.float64(0.1),
                wp.min(
                    wp.float64(0.5),
                    wp.float64(0.9)
                    * wp.pow(error, -wp.float64(1.0) / wp.float64(3.0)),
                ),
            )
        control[2] = h * factor
        if counters[4] >= max_trials:
            counters[3] = 2
    condition[0] = int(control[1] < control[3] and counters[3] == 0)


@wp.kernel
def commit_candidate(
    state: FloatVector, candidate: FloatVector, accepted: IntVector
):
    i = wp.tid()
    if accepted[0] != 0:
        state[i] = candidate[i]


@wp.kernel
def reset_clock(
    control: FloatVector,
    counters: IntVector,
    time: wp.float64,
    first_step: wp.float64,
):
    control[0] = first_step
    control[1] = time
    control[2] = first_step
    control[3] = time
    control[6] = wp.float64(0.0)
    for i in range(5):
        counters[i] = 0


@wp.kernel
def import_physical(
    masses: FloatTensor,
    gas: FloatMatrix,
    physical_atol: FloatVector,
    state: FloatVector,
    scale: FloatVector,
    initial_scaled: FloatVector,
    initial_physical: FloatVector,
    atol: FloatVector,
):
    """Reset scaled coordinates at a split boundary without leaving device."""
    i = wp.tid()
    mass_size = masses.shape[0] * masses.shape[1] * masses.shape[2]
    value = wp.float64(0.0)
    unit = wp.float64(0.0)
    if i < mass_size:
        material = i % masses.shape[2]
        slot = (i // masses.shape[2]) % masses.shape[1]
        box = i // (masses.shape[1] * masses.shape[2])
        value = masses[box, slot, material]
        for m in range(masses.shape[2]):
            unit += masses[box, slot, m]
        unit = wp.max(unit, wp.float64(1e-30))
    else:
        index = i - mass_size
        value = gas[index // gas.shape[1], index % gas.shape[1]]
        unit = wp.max(value, wp.float64(1e-12))
    scale[i] = unit
    state[i] = value / unit
    initial_scaled[i] = state[i]
    initial_physical[i] = value
    atol[i] = physical_atol[i] / unit


@wp.kernel
def rk_stage(
    state: FloatVector,
    k1: FloatVector,
    k2: FloatVector,
    k3: FloatVector,
    control: FloatVector,
    a: wp.float64,
    b: wp.float64,
    c: wp.float64,
    output: FloatVector,
):
    i = wp.tid()
    value = wp.float64(0.0)
    if a != wp.float64(0.0):
        value += a * k1[i]
    if b != wp.float64(0.0):
        value += b * k2[i]
    if c != wp.float64(0.0):
        value += c * k3[i]
    output[i] = state[i] + control[0] * value


@wp.kernel
def rk_error(
    state: FloatVector,
    candidate: FloatVector,
    k1: FloatVector,
    k2: FloatVector,
    k3: FloatVector,
    k4: FloatVector,
    atol: FloatVector,
    rtol: FloatVector,
    control: FloatVector,
    norm: FloatVector,
):
    i = wp.tid()
    error = control[0] * (
        -wp.float64(5.0 / 72.0) * k1[i]
        + wp.float64(1.0 / 12.0) * k2[i]
        + wp.float64(1.0 / 9.0) * k3[i]
        - wp.float64(1.0 / 8.0) * k4[i]
    )
    value = wp.abs(error) / (
        atol[i] + rtol[i] * wp.max(wp.abs(state[i]), wp.abs(candidate[i]))
    )
    if not wp.isfinite(value):
        value = wp.float64(1e300)
    wp.atomic_max(norm, 0, value)


@wp.kernel
def ros_rhs(
    rate: FloatVector,
    first: FloatVector,
    second: FloatVector,
    control: FloatVector,
    stage: int,
    rhs: FloatVector,
):
    """Autonomous ROS3 stage RHS; environment is fixed inside each segment."""
    i = wp.tid()
    value = rate[i]
    if stage == 2:
        value -= C21 * first[i] / control[0]
    elif stage == 3:
        value += (C31 * first[i] + C32 * second[i]) / control[0]
    rhs[i] = value
