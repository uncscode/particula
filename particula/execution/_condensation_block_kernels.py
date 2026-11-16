"""Conservative block-arrowhead linear algebra for material-coupled ROS3.

Each particle has a dense local material Jacobian A and gas derivative B.
Inventory conservation fixes the gas rows to minus the weighted particle
rows. Eliminate local blocks M=qI-A, then solve the small per-box Schur
system q*(I+sum(W*M^-1*B)). No population-sized dense matrix is allocated.
"""

import warp as wp

from particula.execution._condensation_ode_kernels import (
    FloatMatrix,
    FloatTensor,
    FloatVector,
    IntVector,
)
from particula.execution._condensation_ros3_kernels import GAMMA


@wp.kernel
def extract(
    plus: FloatVector,
    minus: FloatVector,
    span: FloatVector,
    counts: FloatMatrix,
    targets: IntVector,
    sources: IntVector,
    materials: int,
    gases: int,
    column: int,
    gas_column: int,
    derivative: FloatTensor,
    invalid: IntVector,
):
    particle, row = wp.tid()  # type: ignore[misc]
    box = particle // counts.shape[1]
    slot = particle % counts.shape[1]
    derivative[particle, row, column] = wp.float64(0.0)
    if counts[box, slot] == wp.float64(0.0):
        return
    i = particle * materials + targets[row]
    j = particle * materials + targets[column]
    if gas_column == 1:
        j = (
            counts.shape[0] * counts.shape[1] * materials
            + box * gases
            + sources[column]
        )
    divisor = span[j]
    if divisor <= wp.float64(0.0) or not wp.isfinite(divisor):
        wp.atomic_max(invalid, 0, 1)
        return
    value = (plus[i] - minus[i]) / divisor
    derivative[particle, row, column] = value
    if not wp.isfinite(value):
        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def invert_local(  # noqa: C901
    local: FloatTensor,
    h: FloatVector,
    augmented: FloatTensor,
    inverse: FloatTensor,
    invalid: IntVector,
):
    """Invert each small local block with partial pivoting in float64."""
    p = wp.tid()  # type: ignore[misc]
    n = local.shape[1]
    q = wp.float64(1.0) / (h[0] * GAMMA)
    for i in range(n):
        for j in range(n):
            value = -local[p, i, j]
            identity = wp.float64(0.0)
            if i == j:
                value += q
                identity = wp.float64(1.0)
            augmented[p, i, j] = value
            augmented[p, i, n + j] = identity
    for k in range(n):
        pivot = k
        largest = wp.abs(augmented[p, k, k])
        for i in range(k + 1, n):
            value = wp.abs(augmented[p, i, k])
            if value > largest:
                largest = value
                pivot = i
        if largest == wp.float64(0.0) or not wp.isfinite(largest):
            wp.atomic_max(invalid, 0, 1)
            return
        for j in range(2 * n):
            value = augmented[p, k, j]
            augmented[p, k, j] = augmented[p, pivot, j]
            augmented[p, pivot, j] = value
        diagonal = augmented[p, k, k]
        for j in range(2 * n):
            augmented[p, k, j] /= diagonal
        for i in range(n):
            if i != k:
                factor = augmented[p, i, k]
                for j in range(2 * n):
                    augmented[p, i, j] -= factor * augmented[p, k, j]
    for i in range(n):
        for j in range(n):
            inverse[p, i, j] = augmented[p, i, n + j]


@wp.kernel
def eliminate_local(
    inverse: FloatTensor,
    gas_columns: FloatTensor,
    rhs: FloatVector,
    targets: IntVector,
    materials: int,
    response: FloatTensor,
    particular: FloatMatrix,
):
    p, i = wp.tid()  # type: ignore[misc]
    n = targets.shape[0]
    value = wp.float64(0.0)
    for k in range(n):
        value += inverse[p, i, k] * rhs[p * materials + targets[k]]
    particular[p, i] = value
    for j in range(n):
        value = wp.float64(0.0)
        for k in range(n):
            value += inverse[p, i, k] * gas_columns[p, k, j]
        response[p, i, j] = value


@wp.kernel
def assemble_reservoirs(
    local: FloatTensor,
    response: FloatTensor,
    particular: FloatMatrix,
    counts: FloatMatrix,
    volume: FloatVector,
    scale: FloatVector,
    rhs: FloatVector,
    targets: IntVector,
    sources: IntVector,
    materials: int,
    gases: int,
    resolved: int,
    h: FloatVector,
    schur: FloatTensor,
):
    """Assemble independent Schur entries in parallel, in fixed slot order."""
    box, i, column = wp.tid()  # type: ignore[misc]
    n = targets.shape[0]
    mass_size = counts.shape[0] * counts.shape[1] * materials
    q = wp.float64(1.0) / (h[0] * GAMMA)
    gi = mass_size + box * gases + sources[i]
    value = wp.float64(0.0)
    if column == i:
        value = wp.float64(1.0)
    if column == n:
        value = rhs[gi]
    for slot in range(counts.shape[1]):
        p = box * counts.shape[1] + slot
        weight = (
            counts[box, slot] * scale[p * materials + targets[i]] / scale[gi]
        )
        if resolved == 1:
            weight /= volume[box]
        if column == n:
            product = wp.float64(0.0)
            for j in range(n):
                product += local[p, i, j] * particular[p, j]
            value -= weight * product
        else:
            value += weight * response[p, i, column]
    if column == n:
        value /= q
    schur[box, i, column] = value


@wp.kernel
def solve_reservoirs(  # noqa: C901
    schur: FloatTensor,
    sources: IntVector,
    mass_size: int,
    gases: int,
    solution: FloatVector,
    invalid: IntVector,
):
    """Pivot the small per-box Schur matrix."""
    box = wp.tid()
    n = sources.shape[0]
    for k in range(n):
        pivot = k
        largest = wp.abs(schur[box, k, k])
        for i in range(k + 1, n):
            value = wp.abs(schur[box, i, k])
            if value > largest:
                largest = value
                pivot = i
        if largest == wp.float64(0.0) or not wp.isfinite(largest):
            wp.atomic_max(invalid, 0, 1)
            return
        for j in range(n + 1):
            value = schur[box, k, j]
            schur[box, k, j] = schur[box, pivot, j]
            schur[box, pivot, j] = value
        diagonal = schur[box, k, k]
        for j in range(n + 1):
            schur[box, k, j] /= diagonal
        for i in range(n):
            if i != k:
                factor = schur[box, i, k]
                for j in range(n + 1):
                    schur[box, i, j] -= factor * schur[box, k, j]
    for i in range(n):
        value = schur[box, i, n]
        solution[mass_size + box * gases + sources[i]] = value  # type: ignore[operator]
        if not wp.isfinite(value):
            wp.atomic_max(invalid, 0, 1)


@wp.kernel
def back_substitute(
    response: FloatTensor,
    particular: FloatMatrix,
    rhs: FloatVector,
    targets: IntVector,
    sources: IntVector,
    reverse_target: IntVector,
    reverse_source: IntVector,
    slots: int,
    materials: int,
    gases: int,
    h: FloatVector,
    solution: FloatVector,
    invalid: IntVector,
):
    i = wp.tid()  # type: ignore[misc]
    mass_size = response.shape[0] * materials
    if i < mass_size:
        row = reverse_target[i % materials]  # type: ignore[operator]
        if row >= 0:
            p = i // materials  # type: ignore[operator]
            box = p // slots
            value = particular[p, row]
            for j in range(targets.shape[0]):
                value += (
                    response[p, row, j]
                    * solution[mass_size + box * gases + sources[j]]
                )
            solution[i] = value
        else:
            solution[i] = rhs[i] * h[0] * GAMMA
    elif reverse_source[(i - mass_size) % gases] < 0:
        solution[i] = rhs[i] * h[0] * GAMMA
    if not wp.isfinite(solution[i]):
        wp.atomic_max(invalid, 0, 1)
