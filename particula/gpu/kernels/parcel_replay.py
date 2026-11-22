"""Batched float64 replay on a fixed schedule using extensive inventories."""

from typing import TYPE_CHECKING, Any

import warp as wp

from particula.dynamics.parcel import WATER_FACTOR
from particula.execution._condensation_ode_kernels import FloatMatrix as Matrix
from particula.execution._condensation_ode_kernels import FloatVector as Vector
from particula.gpu.kernels.parcel import R_VAPOR, F, _es, _grown_radius

WATER_MASS_FACTOR = wp.constant(F(WATER_FACTOR))

if TYPE_CHECKING:
    Indices = Any
else:
    Indices = wp.array2d(dtype=wp.int32)

# State lanes are organic vapor, integrated source, signed local condensate,
# net imported vapor water, followed by count/water kg/organic kg per bin.
GAS, ADDED, CONDENSED, WATER_IN = range(4)
FIRST_BIN = 4


@wp.kernel
def exchange(
    previous: Matrix,
    volume: Vector,
    edges: Indices,
    flows: Vector,
    enabled: Vector,
    vapor: Vector,
    dt: F,
    flux: Matrix,
):
    """Compute each transfer from the same unmodified donor inventory."""
    edge, lane = wp.tid()  # type: ignore[misc]  # 2-D launch
    donor = edges[edge, 0]
    inventory = previous[donor, lane]
    allowed = (lane == GAS and edges[edge, 2] != 0) or (
        lane >= FIRST_BIN and edges[edge, 3] != 0
    )
    amount = F(0.0)
    if allowed:
        amount = inventory * (flows[edge] * enabled[edge] * dt / volume[donor])
    if lane == WATER_IN and edges[edge, 4] != 0:
        amount = vapor[donor] * (
            flows[edge] * enabled[edge] * dt / volume[donor]
        )
    flux[edge, lane] = amount


@wp.kernel
def gather(previous: Matrix, edges: Indices, flux: Matrix, result: Matrix):
    """Apply signed incoming and outgoing transfers to each lane."""
    cell, lane = wp.tid()  # type: ignore[misc]  # 2-D launch
    value = previous[cell, lane]
    for e in range(edges.shape[0]):
        if edges[e, 0] == cell:
            value -= flux[e, lane]
        if edges[e, 1] == cell:
            value += flux[e, lane]
    result[cell, lane] = value


@wp.kernel
def population_count(previous: Matrix, count: Vector):
    # Store this reduction on the tape before using it in the uptake branch.
    """Save total represented count for uptake and its reverse pass."""
    cell = wp.tid()
    total = F(0.0)
    for j in range((previous.shape[1] - FIRST_BIN) // 3):
        total += previous[cell, FIRST_BIN + 3 * j]
    count[cell] = total


@wp.kernel
def local(
    previous: Matrix,
    forcing: Matrix,
    dry: Matrix,
    kappa: Vector,
    uptake: Vector,
    counts: Vector,
    dt: F,
    amplitude: Vector,
    result: Matrix,
):
    """Advance water growth, organic uptake and source ledgers in each cell."""
    cell = wp.tid()
    # Forcing columns: T, p, base vapor density, ramp, rate and endpoint total.
    t = forcing[cell, 0]
    rh = (
        forcing[cell, 2]
        * (F(1.0) + amplitude[cell] * forcing[cell, 3])
        * R_VAPOR
        * t
        / _es(t)
    )
    available = previous[cell, GAS] + forcing[cell, 4] * dt
    population = counts[cell]
    if available < F(0.0):
        available = F(wp.nan)
    gain = F(0.0)
    if population > F(0.0):
        gain = available * (F(1.0) - wp.exp(-uptake[cell] * dt))
    change = F(0.0)
    for j in range(dry.shape[1]):
        start = FIRST_BIN + 3 * j
        n = previous[cell, start]
        water = previous[cell, start + 1]
        organic = previous[cell, start + 2]
        if n > F(0.0):
            d = dry[cell, j]
            radius = wp.cbrt(d * d * d + water / (n * WATER_MASS_FACTOR))
            r = _grown_radius(
                radius, d, kappa[cell], t, forcing[cell, 1], rh, dt, True
            )
            new_water = (
                n * WATER_MASS_FACTOR * (r - d) * (r * r + r * d + d * d)
            )
            change += new_water - water
            water = new_water
            organic += gain * (n / population)
        result[cell, start] = n
        result[cell, start + 1] = water
        result[cell, start + 2] = organic
    result[cell, GAS] = available - gain + forcing[cell, 5]
    result[cell, ADDED] = (
        previous[cell, ADDED] + forcing[cell, 4] * dt + forcing[cell, 5]
    )
    result[cell, CONDENSED] = previous[cell, CONDENSED] + change
    result[cell, WATER_IN] = previous[cell, WATER_IN]


@wp.kernel
def observe(state: Matrix, cell: int, lane: int, result: Vector):
    """Reduce the selected terminal inventory to a scalar Warp loss."""
    amount = F(0.0)
    if lane == GAS:
        amount = state[cell, GAS]
    else:
        for i in range((state.shape[1] - FIRST_BIN) // 3):
            amount += state[cell, FIRST_BIN + 3 * i + lane]
    result[0] = amount
