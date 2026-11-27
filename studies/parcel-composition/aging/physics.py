"""Synthetic equal-volume parcel exchange, dry aging and response operators."""

import numpy as np
from particula.optics import get_mie_efficiencies
from particula.particles import get_kelvin_radius
from scipy.interpolate import RegularGridInterpolator

# Species amount per particle is 6/pi times dry volume in nm^3.
DENSITY = np.array([1770.0, 1300.0, 1300.0, 1800.0])
KAPPA = np.array([0.60, 0.025, 0.18, 0.0])
SS = np.geomspace(0.07, 1.2, 31)
EDGES = np.geomspace(30, 450, 13)
KELVIN = 2e9 * get_kelvin_radius(0.072, 997.0, 0.01801528, 298.15)


def make_optics():
    """Tabulate Mie efficiency by log size and black carbon fraction."""
    diam = np.geomspace(8, 3000, 301)
    bc = np.linspace(0, 1, 151)
    refractive = 1.50 + (0.45 + 0.79j) * bc
    q = get_mie_efficiencies(
        diam[:, None] * 5e-10, 550e-9, refractive
    ).absorption
    q[:, 0] = 0.0
    return RegularGridInterpolator((np.log(diam), bc), q, bounds_error=True)


def responses(vol, optics):
    """Return critical supersaturation in percent and absorption area in nm²."""
    total = vol.sum(axis=1)
    diameter = np.cbrt(total)
    # kappa times cube of the diameter equals sum of species volume times kappa.
    kappa_volume = vol @ KAPPA
    critical = np.full(len(vol), np.inf)
    wettable = kappa_volume > 0
    critical[wettable] = 100 * np.expm1(
        np.sqrt(4 * KELVIN**3 / (27 * kappa_volume[wettable]))
    )
    fraction = vol[:, 3] / total
    efficiency = optics((np.log(diameter), fraction))
    return critical, np.pi / 4 * diameter**2 * efficiency


def initial_parcels(seed, size):
    """Draw source mixtures and finite condensable reservoirs per parcel."""
    param, rng, swaps = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    ]
    # Initial sources: salt-rich, fresh organic-rich, combustion-rich.
    endmembers = np.array(
        [
            [0.83, 0.03, 0.14, 0.0],
            [0.03, 0.88, 0.09, 0.0],
            [0.035, 0.295, 0.12, 0.55],
        ]
    )
    centers = np.array([120.0, 72.0, 105.0]) * param.uniform(0.70, 1.40, (2, 3))
    probs = np.stack([param.dirichlet([5, 2, 1]), param.dirichlet([1, 3, 4])])
    parcels = []
    for weights, modes in zip(probs, centers, strict=True):
        source = rng.choice(3, size, p=weights)
        diam = modes[source] * rng.lognormal(0, 0.35, size)
        # Continuous class compositions; preserve structural absence of BC.
        v = endmembers[source] * rng.lognormal(0, 0.55, (size, 4))
        parcels.append(diam[:, None] ** 3 * v / v.sum(axis=1, keepdims=True))
    state = np.stack(parcels)
    stock = state.sum(axis=(1, 2))[:, None] * param.uniform(0.08, 0.42, (2, 2))
    rate = param.uniform(0.08, 0.45, (2, 3))
    exchange = param.uniform(0.10, 0.45)
    return state, stock, rate, exchange, swaps


def evolve(seed, size, substeps=2):
    """Yield hourly states; exchange every 10 minutes, subdividing aging."""
    state, stock, rate, exchange, swaps = initial_parcels(seed, size)
    initial = state.sum(axis=(0, 1))
    initial_stock = stock.sum(axis=0)
    stock0 = stock.copy()
    probability = -np.expm1(-2 * exchange / 6) / 2
    decay = -np.expm1(-rate / (6 * substeps))
    yield 0, state, rate, exchange, stock0
    for step in range(1, 37):
        mask = swaps.random(size) < probability
        state[:, mask] = state[::-1, mask]
        stock += probability * (stock[::-1] - stock)
        for _ in range(substeps):
            # Convert primary organic to oxidized organic at equal density.
            converted = state[:, :, 1] * decay[:, 2, None]
            state[:, :, 1] -= converted
            state[:, :, 2] += converted
            surface = state.sum(axis=2) ** (2 / 3)
            share = surface / surface.sum(axis=1, keepdims=True)
            added = stock * decay[:, :2]
            stock -= added
            state[:, :, 0] += share * added[:, 0, None]
            state[:, :, 2] += share * added[:, 1, None]
        if step % 6 == 0:
            if not np.isfinite(state).all() or not np.isfinite(stock).all():
                raise ValueError("Nonfinite reservoir or particle inventory")
            if state.min() < 0 or stock.min() < 0:
                raise ValueError("Negative reservoir or particle inventory")
            total = state.sum(axis=(0, 1))
            np.testing.assert_allclose(total[3], initial[3], rtol=1e-11)
            np.testing.assert_allclose(
                total[0] + stock[:, 0].sum(),
                initial[0] + initial_stock[0],
                rtol=1e-11,
            )
            np.testing.assert_allclose(
                total[1:3].sum() + stock[:, 1].sum(),
                initial[1:3].sum() + initial_stock[1],
                rtol=1e-11,
            )
            yield step // 6, state, rate, exchange, stock0
