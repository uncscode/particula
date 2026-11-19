"""Resident fp64 capture rates for compact polarizable spheres."""

from typing import no_type_check

import numpy as np
import warp as wp

from particula.dynamics.coagulation.van_der_waals_kernel import (
    COULOMB_LENGTH_T,
    DISTANCES,
    LOG_RATIO_STEP,
    MAX_RADIUS_RATIO,
    RADIAL_INTERVALS,
    RATIO_INTERVALS,
    dielectric_row,
)
from particula.gpu.dynamics.coagulation_funcs import hard_sphere_transition_wp
from particula.util.constants import BOLTZMANN_CONSTANT

_LENGTH_T = wp.constant(wp.float64(COULOMB_LENGTH_T))
_LOG_STEP = wp.constant(wp.float64(LOG_RATIO_STEP))
_RATIO_LIMIT = wp.constant(wp.float64(MAX_RADIUS_RATIO))
_KB = wp.constant(wp.float64(BOLTZMANN_CONSTANT))
_Profile = wp.types.vector(RADIAL_INTERVALS + 1, wp.float64)
_Indices = wp.types.vector(RADIAL_INTERVALS + 1, wp.int32)


@wp.struct
class CaptureData:
    """Immutable geometry and material for one prepared mechanism."""

    hamaker: wp.float64
    polarizable: wp.int32
    x: wp.array(dtype=wp.float64)  # type: ignore[valid-type]
    response: wp.array3d(dtype=wp.float64)  # type: ignore[valid-type]


def prepare_capture(hamaker, epsilon, device):
    """Transfer the dielectric response once, outside native graph capture."""
    data = CaptureData()
    data.hamaker = hamaker
    data.polarizable = int(epsilon != 1)
    data.x = wp.array(DISTANCES, dtype=wp.float64, device=device)
    if epsilon != 1:
        response = np.stack(
            [dielectric_row(epsilon, row) for row in range(RATIO_INTERVALS + 1)]
        )
        data.response = wp.array(response, dtype=wp.float64, device=device)
    return data


@no_type_check
@wp.func
def _expm1(x: wp.float64):
    if wp.abs(x) < wp.float64(1e-5):
        return x * (
            wp.float64(1.0) + x * (wp.float64(0.5) + x / wp.float64(6.0))
        )
    return wp.exp(x) - wp.float64(1.0)


@wp.func
def _hamaker(x: wp.float64, fraction: wp.float64):
    one = wp.float64(1.0)
    u = (
        wp.float64(4.0)
        * fraction
        * (one - fraction)
        * x
        * x
        / ((one - x) * (one + x))
    )
    if u < wp.float64(0.001):
        bracket = (
            u
            * u
            * u
            * (
                one / wp.float64(6.0)
                - u
                * (
                    wp.float64(0.25)
                    - u * (wp.float64(0.3) - u / wp.float64(3.0))
                )
            )
        )
        return -bracket / wp.float64(6.0)
    return (
        wp.log(one + u) - wp.float64(0.5) * (u + u / (one + u))
    ) / wp.float64(6.0)


@no_type_check
@wp.func
def capture_pair_wp(  # noqa: C901 -- fused device orbit and flux quadrature
    data: CaptureData,
    radius_i: wp.float64,
    radius_j: wp.float64,
    mass_i: wp.float64,
    mass_j: wp.float64,
    z_i: wp.float64,
    z_j: wp.float64,
    diffusion_sum: wp.float64,
    temperature: wp.float64,
):
    """Evaluate the actual pair; negative return signals domain rejection."""
    one = wp.float64(1.0)
    zero = wp.float64(0.0)
    contact = radius_i + radius_j
    ratio = wp.max(radius_i, radius_j) / wp.min(radius_i, radius_j)
    fraction = wp.min(radius_i, radius_j) / contact
    left, right = z_i, z_j
    if radius_i > radius_j:
        left, right = z_j, z_i
    coupling = _LENGTH_T / (temperature * contact)
    attraction = data.hamaker / (_KB * temperature)
    if (
        coupling * left * right < wp.float64(-0.5)
        or ratio > _RATIO_LIMIT
        or attraction > wp.float64(200.0)
        or coupling * wp.max(left * left, right * right) > wp.float64(60.0)
    ):
        return wp.float64(-1.0)
    bare = coupling * left * right
    ec, ef = one, one
    if attraction == zero and (
        data.polarizable == 0 or (left == zero and right == zero)
    ):
        if bare != zero:
            ec = bare / _expm1(bare)
            ef = one - bare
            if bare > zero:
                ef = wp.exp(-bare)
    else:
        profile = _Profile()
        slopes = _Profile()
        intercept = _Profile()
        starts = _Profile()
        indices = _Indices()
        coordinate = wp.log(ratio) / _LOG_STEP
        row = wp.min(wp.int32(coordinate), RATIO_INTERVALS - 1)
        weight = coordinate - wp.float64(row)
        count = int(0)
        barrier = wp.float64(0.0)
        last = int(RADIAL_INTERVALS)
        endpoint = one
        if attraction > zero:
            gap = wp.min(
                (one - data.x[last - 1]) * wp.float64(0.5), wp.sqrt(attraction)
            )
            endpoint = wp.min(wp.float64(0.9999999999999999), one - gap)
        previous_induced = wp.float64(0)
        for i in range(1, last + 1):
            x = data.x[i]
            if i == last:
                x = endpoint
            energy = bare * x
            if data.polarizable != 0 and (left != zero or right != zero):
                ll = (one - weight) * data.response[
                    row, i, 0
                ] + weight * data.response[row + 1, i, 0]
                lr = (one - weight) * data.response[
                    row, i, 1
                ] + weight * data.response[row + 1, i, 1]
                rr = (one - weight) * data.response[
                    row, i, 2
                ] + weight * data.response[row + 1, i, 2]
                induced = (
                    ll * left * left + lr * left * right + rr * right * right
                )
                if i == last and attraction > zero:
                    induced += (
                        (one - endpoint)
                        / (one - data.x[i - 1])
                        * (previous_induced - induced)
                    )
                previous_induced = induced
                energy += coupling * induced
            if attraction > zero:
                energy += attraction * _hamaker(x, fraction)
            profile[i] = energy
            barrier = wp.max(barrier, energy)
            slopes[i] = one / (x * x)
            intercept[i] = -energy * slopes[i]
            start = wp.float64(-wp.inf)
            while count > 0:
                j = indices[count - 1]
                start = (intercept[i] - intercept[j]) / (slopes[j] - slopes[i])
                if start > starts[count - 1]:
                    break
                count -= 1
            if count == 0:
                start = wp.float64(-wp.inf)
            indices[count] = i
            starts[count] = start
            count += 1
        resistance = wp.float64(0.0)
        previous = wp.exp(-barrier)
        for i in range(1, RADIAL_INTERVALS + 1):
            value = wp.float64(0.0)
            if i <= last:
                value = wp.exp(profile[i] - barrier)
            location = data.x[i]
            if i == last:
                location = endpoint
            resistance += (
                (location - data.x[i - 1])
                * (value + previous)
                * wp.float64(0.5)
            )
            previous = value
        resistance += (one - endpoint) * previous * wp.float64(0.5)
        ec = wp.exp(-barrier) / resistance
        ef = wp.float64(0.0)
        initial = wp.float64(wp.inf)
        for k in range(count):
            j = indices[k]
            low = wp.max(barrier, starts[k])
            high = wp.float64(wp.inf)
            if k + 1 < count:
                high = wp.max(barrier, starts[k + 1])
            initial = wp.min(initial, barrier * slopes[j] + intercept[j])
            ef += slopes[j] * wp.exp(-low) * (-_expm1(low - high))
        ef += wp.max(zero, initial) * wp.exp(-barrier)
    mu = mass_i / (one + mass_i / mass_j)
    drag = _KB * temperature / diffusion_sum
    kn = wp.sqrt(_KB * temperature * mu) / (drag * contact) * ec / ef
    h = hard_sphere_transition_wp(kn)
    rate = h * drag * contact * contact * contact / mu * (ef * ef / ec)
    if not wp.isfinite(rate) or rate < zero:
        return wp.float64(-1.0)
    return rate
