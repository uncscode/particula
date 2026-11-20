"""Resident single-exchange Rosenbrock algebra and conservative linear solves.

For one exchanging material per particle, the Jacobian has independent particle
blocks coupled through one gas reservoir per box. A Schur complement solves
this arrowhead system in linear storage. Other material rates are zero in this
profile; their stage increments are zero, so composition cross derivatives
into those fixed coordinates need not be stored. No physical rate is evaluated
here: the native mapped condensation kernel supplies every derivative.
"""

import warp as wp

from particula.execution._condensation_ode_kernels import (
    FloatMatrix,
    FloatVector,
    IntVector,
)
from particula.execution._ros3_coefficients import (
    C21,
    C31,
    C32,
    E2,
    E3,
    GAMMA,
    GAMMA2,
    GAMMA3,
    M2,
    M3,
)


@wp.kernel
def perturb(
    state: FloatVector,
    scale: FloatVector,
    counts: FloatMatrix,
    materials: wp.int32,
    gases: wp.int32,
    target: wp.int32,
    source: wp.int32,
    gas_columns: wp.int32,
    plus: FloatVector,
    minus: FloatVector,
    span: FloatVector,
):
    """Perturb independent local mass or per-box gas columns simultaneously."""
    i = wp.tid()
    mass_size = counts.shape[0] * counts.shape[1] * materials
    delta = wp.float64(0.0)
    if i < mass_size:
        material = i % materials
        slot = (i // materials) % counts.shape[1]  # type: ignore[operator]
        box = i // (counts.shape[1] * materials)
        if (
            gas_columns == 0
            and material == target
            and counts[box, slot] > wp.float64(0.0)
        ):
            total = wp.float64(0.0)
            offset = (box * counts.shape[1] + slot) * materials
            for m in range(materials):
                total += state[offset + m] * scale[offset + m]
            delta = wp.float64(1.4901161193847656e-8) * total / scale[i]
    elif gas_columns == 1 and (i - mass_size) % gases == source:
        delta = wp.float64(1.4901161193847656e-8) * wp.max(
            wp.abs(state[i]), wp.float64(1.0)
        )
    plus[i] = state[i] + delta
    minus[i] = state[i]
    if state[i] >= delta:
        minus[i] = state[i] - delta
    span[i] = plus[i] - minus[i]


@wp.kernel
def extract_derivative(
    plus: FloatVector,
    minus: FloatVector,
    span: FloatVector,
    counts: FloatMatrix,
    materials: wp.int32,
    gases: wp.int32,
    target: wp.int32,
    source: wp.int32,
    gas_columns: wp.int32,
    result: FloatMatrix,
    invalid: IntVector,
):
    """Extract local derivatives from colored physical-rate evaluations."""
    box, slot = wp.tid()  # type: ignore[misc]
    i = (box * counts.shape[1] + slot) * materials + target
    column = i
    if gas_columns == 1:
        column = (
            counts.shape[0] * counts.shape[1] * materials + box * gases + source
        )
    result[box, slot] = wp.float64(0.0)
    if counts[box, slot] > wp.float64(0.0):
        divisor = span[column]
        if divisor <= wp.float64(0.0) or not wp.isfinite(divisor):
            wp.atomic_max(invalid, 0, 1)
        else:
            value = (plus[i] - minus[i]) / divisor
            result[box, slot] = value
            if not wp.isfinite(value):
                wp.atomic_max(invalid, 0, 1)


@wp.kernel
def solve_gas(
    local: FloatMatrix,
    gas_column: FloatMatrix,
    counts: FloatMatrix,
    volume: FloatVector,
    scale: FloatVector,
    rhs: FloatVector,
    materials: wp.int32,
    gases: wp.int32,
    target: wp.int32,
    source: wp.int32,
    resolved: wp.int32,
    inverse_step_gamma: wp.float64,
    solution: FloatVector,
    invalid: IntVector,
):
    """Eliminate particle coordinates with conservative weighted gas rows."""
    box = wp.tid()
    mass_size = counts.shape[0] * counts.shape[1] * materials
    gas_index = mass_size + box * gases + source
    numerator = rhs[gas_index]
    correction = wp.float64(0.0)
    for slot in range(counts.shape[1]):
        i = (box * counts.shape[1] + slot) * materials + target
        weight = counts[box, slot]
        if resolved == 1:
            weight /= volume[box]
        weight *= scale[i] / scale[gas_index]
        diagonal = inverse_step_gamma - local[box, slot]
        if diagonal == wp.float64(0.0) or not wp.isfinite(diagonal):
            wp.atomic_max(invalid, 0, 1)
        else:
            numerator -= weight * local[box, slot] * rhs[i] / diagonal
            correction += weight * gas_column[box, slot] / diagonal
    # q + sum(c*b) + sum(c*a*b/(q-a)) = q*(1 + sum(c*b/(q-a))).
    denominator = inverse_step_gamma * (wp.float64(1.0) + correction)
    if denominator == wp.float64(0.0) or not wp.isfinite(denominator):
        wp.atomic_max(invalid, 0, 1)
    else:
        value = numerator / denominator
        solution[gas_index] = value
        if not wp.isfinite(value):
            wp.atomic_max(invalid, 0, 1)


@wp.kernel
def solve_remaining(
    local: FloatMatrix,
    gas_column: FloatMatrix,
    rhs: FloatVector,
    materials: wp.int32,
    gases: wp.int32,
    target: wp.int32,
    source: wp.int32,
    inverse_step_gamma: wp.float64,
    solution: FloatVector,
    invalid: IntVector,
):
    """Back-substitute all local coordinates without a dense factorization."""
    i = wp.tid()
    mass_size = local.shape[0] * local.shape[1] * materials
    if i < mass_size:
        if i % materials == target:
            box = i // (local.shape[1] * materials)
            slot = (i // materials) % local.shape[1]  # type: ignore[operator]
            gas_index = mass_size + box * gases + source
            diagonal = inverse_step_gamma - local[box, slot]
            if diagonal != wp.float64(0.0) and wp.isfinite(diagonal):
                solution[i] = (
                    rhs[i] + gas_column[box, slot] * solution[gas_index]
                ) / diagonal
            else:
                wp.atomic_max(invalid, 0, 1)
        else:
            solution[i] = rhs[i] / inverse_step_gamma
    elif (i - mass_size) % gases != source:
        solution[i] = rhs[i] / inverse_step_gamma
    if not wp.isfinite(solution[i]):
        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def stage_rhs(
    rate: FloatVector,
    first: FloatVector,
    second: FloatVector,
    partial: FloatVector,
    h: wp.float64,
    stage: wp.int32,
    rhs: FloatVector,
):
    i = wp.tid()
    gamma = GAMMA
    if stage == 2:
        gamma = GAMMA2
    elif stage == 3:
        gamma = GAMMA3
    value = rate[i] + h * gamma * partial[i]
    if stage == 2:
        value -= C21 * first[i] / h
    elif stage == 3:
        value += (C31 * first[i] + C32 * second[i]) / h
    rhs[i] = value


@wp.kernel
def stage_state(
    state: FloatVector, increment: FloatVector, result: FloatVector
):
    i = wp.tid()
    result[i] = state[i] + increment[i]


@wp.kernel
def finish(
    state: FloatVector,
    first: FloatVector,
    second: FloatVector,
    third: FloatVector,
    atol: FloatVector,
    rtol: FloatVector,
    candidate: FloatVector,
    norm: FloatVector,
):
    i = wp.tid()
    candidate[i] = state[i] + first[i] + M2 * second[i] - M3 * third[i]
    error = wp.float64(0.5) * first[i] - E2 * second[i] + E3 * third[i]
    denominator = atol[i] + rtol[i] * wp.max(
        wp.abs(state[i]), wp.abs(candidate[i])
    )
    value = wp.abs(error) / denominator
    if not wp.isfinite(value):
        value = wp.float64(1.0e300)
    wp.atomic_max(norm, 0, value)


@wp.kernel
def partial_time(
    base: FloatVector,
    shifted: FloatVector,
    delta: wp.float64,
    result: FloatVector,
):
    """Differentiate forcing at fixed state inside one smooth time segment."""
    i = wp.tid()
    result[i] = (shifted[i] - base[i]) / delta
