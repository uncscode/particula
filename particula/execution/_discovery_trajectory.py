"""Warp kernels for deterministic gas chemistry and seeded chamber growth."""

import warp as wp

from particula.execution._condensation_ode_kernels import (
    FloatMatrix,
    FloatVector,
    IntVector,
)
from particula.gpu.dynamics.condensation_funcs import (
    first_order_mass_transport_k_wp,
)
from particula.gpu.properties.particle_properties import (
    vapor_transition_correction_wp,
)

PI = wp.constant(wp.float64(3.141592653589793))
THIRD = wp.constant(wp.float64(1 / 3))


@wp.kernel
def initialize(
    dose: FloatVector,
    seed: FloatVector,
    quantile: FloatVector,
    fractions: FloatVector,
    molar_volume: FloatVector,
    state: FloatMatrix,
):
    """Initialize gas and equal-number lognormal seed cohorts in mol/m³ air."""
    b, j = wp.tid()  # type: ignore[misc]
    ns = dose.shape[0]
    value = wp.float64(0)
    if b == 0 and j < ns:
        value = dose[j]
    if b > 0:
        number = seed[0] / wp.float64(quantile.shape[0])
        if j < ns and fractions[j] > wp.float64(0):
            diameter = seed[1] * wp.exp(wp.log(seed[2]) * quantile[b - 1])
            value = (
                number
                * fractions[j]
                * PI
                / wp.float64(6)
                * diameter
                * diameter
                * diameter
                / molar_volume[j]
            )
        if j == 2 * ns:
            value = number
    state[b, j] = value


@wp.kernel
def inventory(state: FloatMatrix, molar_volume: FloatVector, bulk: FloatMatrix):
    """Return suspended volume fraction, number, diameter and total moles."""
    j = wp.tid()  # Scalar launch over occupied cohorts.
    ns = molar_volume.shape[0]
    vol = wp.float64(0)
    moles = wp.float64(0)
    for s in range(ns):
        vol += state[j + 1, s] * molar_volume[s]  # type: ignore[operator]
        moles += state[j + 1, s]  # type: ignore[operator]
    number = state[j + 1, 2 * ns]  # type: ignore[operator]
    bulk[j, 0] = vol
    bulk[j, 1] = number
    bulk[j, 2] = wp.float64(0)
    bulk[j, 3] = moles


@wp.kernel
def geometry(inventory: FloatMatrix, bulk: FloatMatrix):
    """Use the saved volume reduction when differentiating the cube root."""
    j = wp.tid()
    volume = inventory[j, 0]
    number = inventory[j, 1]
    bulk[j, 0] = volume
    bulk[j, 1] = number
    bulk[j, 2] = wp.pow(wp.float64(6) * volume / (PI * number), THIRD)
    bulk[j, 3] = inventory[j, 3]


@wp.kernel
def exchange(
    state: FloatMatrix,
    bulk: FloatMatrix,
    gas: IntVector,
    particle: IntVector,
    diffusion: FloatVector,
    mean_path: FloatVector,
    saturation: FloatVector,
    flux: FloatMatrix,
):
    """Evaluates reversible ideal-solution transfer at unit accommodation."""
    j, p = wp.tid()  # type: ignore[misc]
    radius = wp.float64(0.5) * bulk[j, 2]
    correction = vapor_transition_correction_wp(
        mean_path[p] / radius, wp.float64(1)
    )
    conductance = first_order_mass_transport_k_wp(
        radius, correction, diffusion[p]
    )
    eq = saturation[p] * state[j + 1, particle[p]] / bulk[j, 3]
    flux[j, p] = bulk[j, 1] * conductance * (state[0, gas[p]] - eq)


@wp.func
def _chemistry(
    b: int,
    s: int,
    kind: int,
    volume: wp.float64,
    state: FloatMatrix,
    rates: FloatVector,
    phase: IntVector,
    first: IntVector,
    second: IntVector,
    change: wp.array2d(dtype=int),  # type: ignore[valid-type]
):
    value = wp.float64(0)
    for r in range(rates.shape[0]):
        if kind == phase[r] and change[r, s] != 0:
            extent = rates[r] * state[b, first[r]]
            if second[r] >= 0:
                extent *= state[b, second[r]] / volume
            value += wp.float64(change[r, s]) * extent
    return value


@wp.kernel
def derivative(
    state: FloatMatrix,
    bulk: FloatMatrix,
    flux: FloatMatrix,
    gas: IntVector,
    particle: IntVector,
    rates: FloatVector,
    phase: IntVector,
    first: IntVector,
    second: IntVector,
    change: wp.array2d(dtype=int),  # type: ignore[valid-type]
    seed: FloatVector,
    result: FloatMatrix,
):
    """Compute molar tendencies and accumulate deposited inventories."""
    b, s = wp.tid()  # type: ignore[misc]
    ns = change.shape[1]
    volume = wp.float64(1)
    kind = int(0)
    wall = wp.float64(0)
    if b > 0:
        kind = 1
        volume = bulk[b - 1, 0]
        # Synthetic deposition law: inverse diameter, referenced to 100 nm.
        wall = seed[3] * wp.float64(1e-7) / bulk[b - 1, 2]
    value = wp.float64(0)
    if s < ns:
        value = _chemistry(
            b, s, kind, volume, state, rates, phase, first, second, change
        )
        for p in range(gas.shape[0]):
            if b == 0 and gas[p] == s:
                for j in range(bulk.shape[0]):
                    value -= flux[j, p]
            if b > 0 and particle[p] == s:
                value += flux[b - 1, p]
        value -= wall * state[b, s]
    elif b > 0 and s < 2 * ns:
        value = wall * state[b, s - ns]
    elif b > 0:
        value = -wall * state[b, s]
    result[b, s] = value


@wp.kernel
def advance(
    state: FloatMatrix,
    slope: FloatMatrix,
    dt: wp.float64,
    result: FloatMatrix,
    invalid: IntVector,
):
    """Advance one explicit stage and flag failed domain checks."""
    b, s = wp.tid()  # type: ignore[misc]
    value = state[b, s] + dt * slope[b, s]
    result[b, s] = value
    if not wp.isfinite(value) or value < wp.float64(0):
        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def cohort_observables(
    bulk: FloatMatrix, diam: FloatMatrix, count: FloatMatrix
):
    j = wp.tid()
    diam[0, j] = bulk[j, 2]
    count[0, j] = bulk[j, 1]


@wp.kernel
def observe(
    state: FloatMatrix,
    spectra: FloatMatrix,
    masses: FloatVector,
    organic: IntVector,
    gases: IntVector,
    output: FloatVector,
):
    """Observe gas and organic mass in kg/m³ and number per instrument bin."""
    i = wp.tid()
    value = wp.float64(0)
    if i < gases.shape[0]:
        s = gases[i]
        value = state[0, s] * masses[s]
    elif i == gases.shape[0]:
        for j in range(1, state.shape[0]):
            for s in range(masses.shape[0]):
                if organic[s] != 0:
                    value += state[j, s] * masses[s]
    else:
        value = spectra[0, i - gases.shape[0] - 1]
    output[i] = value
