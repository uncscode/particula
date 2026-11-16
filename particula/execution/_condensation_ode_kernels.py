"""Device state algebra for native deterministic condensation RK3(2).

The existing mapped condensation kernel supplies the physical mass rates.
These kernels only unpack scaled coordinates, couple conservative reservoirs,
check physical domains and evaluate the embedded max-component error estimate.
"""

from typing import TYPE_CHECKING, Any

import warp as wp

if TYPE_CHECKING:
    FloatVector = Any
    FloatMatrix = Any
    FloatTensor = Any
    IntVector = Any
else:
    FloatVector = wp.array(dtype=wp.float64)
    FloatMatrix = wp.array2d(dtype=wp.float64)
    FloatTensor = wp.array3d(dtype=wp.float64)
    IntVector = wp.array(dtype=wp.int32)


@wp.kernel
def unpack(
    state: FloatVector,
    scale: FloatVector,
    initial_scaled: FloatVector,
    initial_physical: FloatVector,
    masses: FloatTensor,
    gas: FloatMatrix,
    invalid: IntVector,
):
    """Unpack a trial and flag nonfinite/negative physical inventories."""
    i = wp.tid()
    value = state[i] * scale[i]
    if state[i] == initial_scaled[i]:
        value = initial_physical[i]
    if not wp.isfinite(value) or value < wp.float64(0.0):
        wp.atomic_max(invalid, 0, 1)
    mass_size = masses.shape[0] * masses.shape[1] * masses.shape[2]
    if i < mass_size:
        material = i % masses.shape[2]
        slot = (i // masses.shape[2]) % masses.shape[1]
        box = i // (masses.shape[1] * masses.shape[2])
        masses[box, slot, material] = value
    else:
        index = i - mass_size
        gas[index // gas.shape[1], index % gas.shape[1]] = value


@wp.kernel
def check_slots(
    masses: FloatTensor,
    counts: FloatMatrix,
    invalid: IntVector,
):
    """Reject active-slot evaporation and any occupied formerly empty slot."""
    box, slot = wp.tid()  # type: ignore[misc]
    total = wp.float64(0.0)
    for material in range(masses.shape[2]):
        total += masses[box, slot, material]
    if counts[box, slot] > wp.float64(0.0):
        if total <= wp.float64(0.0) or not wp.isfinite(total):  # type: ignore[operator]
            wp.atomic_max(invalid, 0, 1)
    elif total != wp.float64(0.0):
        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def pack_rate(
    rate: FloatTensor,
    counts: FloatMatrix,
    volume: FloatVector,
    gas_to_particle: IntVector,
    scale: FloatVector,
    derivative: FloatVector,
    invalid: IntVector,
    resolved: wp.int32,
):
    """Couple gas to particle flux with native population weights."""
    i = wp.tid()
    mass_size = rate.shape[0] * rate.shape[1] * rate.shape[2]
    value = wp.float64(0.0)
    if i < mass_size:
        material = i % rate.shape[2]
        slot = (i // rate.shape[2]) % rate.shape[1]
        box = i // (rate.shape[1] * rate.shape[2])
        value = rate[box, slot, material]
    else:
        index = i - mass_size
        box = index // gas_to_particle.shape[0]
        gas = index % gas_to_particle.shape[0]
        material = gas_to_particle[gas]
        if material >= 0:
            for slot in range(rate.shape[1]):
                weight = counts[box, slot]
                if resolved == 1:
                    weight /= volume[box]
                value -= weight * rate[box, slot, material]
    value /= scale[i]
    derivative[i] = value
    if not wp.isfinite(value):
        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def stage(
    state: FloatVector,
    k1: FloatVector,
    k2: FloatVector,
    k3: FloatVector,
    h: wp.float64,
    a: wp.float64,
    b: wp.float64,
    c: wp.float64,
    result: FloatVector,
):
    """Form one out-of-place stage without touching accepted state."""
    i = wp.tid()
    value = wp.float64(0.0)
    if a != wp.float64(0.0):
        value += a * k1[i]
    if b != wp.float64(0.0):
        value += b * k2[i]
    if c != wp.float64(0.0):
        value += c * k3[i]
    result[i] = state[i] + h * value


@wp.kernel
def error_norm(
    old: FloatVector,
    candidate: FloatVector,
    k1: FloatVector,
    k2: FloatVector,
    k3: FloatVector,
    k4: FloatVector,
    atol: FloatVector,
    rtol: FloatVector,
    h: wp.float64,
    norm: FloatVector,
):
    """Reduce the shared Bogacki-Shampine local estimate across every lane."""
    i = wp.tid()
    error = h * (
        -wp.float64(5.0 / 72.0) * k1[i]
        + wp.float64(1.0 / 12.0) * k2[i]
        + wp.float64(1.0 / 9.0) * k3[i]
        - wp.float64(1.0 / 8.0) * k4[i]
    )
    denominator = atol[i] + rtol[i] * wp.max(
        wp.abs(old[i]), wp.abs(candidate[i])
    )
    value = wp.abs(error) / denominator
    if not wp.isfinite(value):
        value = wp.float64(1.0e300)
    wp.atomic_max(norm, 0, value)
