"""Equal-number, particle-resolved two-box boreal challenge in SI and nm."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from particula.dynamics.coagulation.brownian_kernel import (
    get_brownian_kernel_via_system_state,
)

SEEDS = (181, 302, 463, 594, 715, 826, 947, 1068, 1199, 1320, 1441, 1562)
# Acid, ammonium, background sulfate salt, BC, POA, SVOC, LVOC, HOM.
RHO = np.array([1830.0, 1760.0, 1770.0, 1800.0, 1250.0, 1300.0, 1400.0, 1500.0])
KAPPA = np.array([0.90, 0.65, 0.61, 0.0, 0.025, 0.065, 0.14, 0.22])
MW = np.array([98.079, 18.039, 132.14, 12.0, 200.0, 180.0, 220.0, 390.0])
MOL_VOL = 6 / np.pi * 1e27 * MW * 1e-3 / (6.02214076e23 * RHO)
SS = np.geomspace(0.07, 1.2, 31)
REGIMES = {
    f"c{c}b{b}n{n}": (c, b, n, 1)
    for c in (0, 1)
    for b in (0, 1)
    for n in (0, 1)
}
REGIMES.update(inert=(0, 0, 0, 0), coupled_dark=(1, 1, 1, 0))


def sat(t):
    """Saturated vapor density (kg/m³), liquid-water Magnus relation."""
    return 610.94 * np.exp(17.625 * (t - 273.15) / (t - 30.11)) / (461.5 * t)


@dataclass
class Box:
    # Species and water expressed as 6/pi times physical volume in nm³.
    """Particle, gas, thermodynamic and lineage state in one equal volume."""

    dry: np.ndarray
    water: np.ndarray
    old: np.ndarray
    pending: np.ndarray
    gas: np.ndarray
    qt: float
    born: int = 0
    reached: int = 0
    captured: int = 0
    mergers: int = 0
    cloud_peak: float = 0.0
    lwc_peak: float = 0.0
    temperature: float = 278.0
    liquid: float = 0.0
    droplets: float = 0.0
    jrate: float = 0.0
    mt: float = 0.0


def init(seed, n):
    """Equal-volume parcels, 1,000 cm⁻³ initial concentration in each box."""
    params, initial, events = [
        np.random.default_rng(x) for x in np.random.SeedSequence(seed).spawn(3)
    ]
    mix = params.uniform(0.10, 0.36)
    lift = params.uniform(310, 590)
    acid = params.uniform(6.2e6, 1.15e7, 2)
    nh3 = params.uniform(22, 86, 2)
    mt = params.uniform(360, 660, 2)  # pptv at 278 K and 1 atm
    centers = params.uniform(0.8, 1.25, (2, 3)) * [105, 71, 123]
    weights = [params.dirichlet([6, 3, 1]), params.dirichlet([1, 3, 5])]
    types = np.array(
        [
            [0.02, 0.055, 0.61, 0, 0.025, 0.05, 0.16, 0.08],
            [0.006, 0.014, 0.025, 0, 0.62, 0.23, 0.085, 0.02],
            [0.008, 0.012, 0.04, 0.52, 0.30, 0.065, 0.04, 0.015],
        ]
    )
    volume = n / 1000  # cm³
    boxes = []
    for i in range(2):
        mode = initial.choice(3, n, p=weights[i])
        d = centers[i, mode] * initial.lognormal(0, 0.34, n)
        frac = types[mode] * initial.lognormal(0, 0.45, (n, 8))
        dry = d[:, None] ** 3 * frac / frac.sum(1)[:, None]
        gas = np.zeros(8)
        gas[[0, 1, 7]] = (
            np.array([acid[i], nh3[i] * 2.64e7, 1.1e5])
            * volume
            * MOL_VOL[[0, 1, 7]]
        )
        w = (dry @ KAPPA) * 0.38 / 0.62
        box = Box(
            dry,
            w,
            np.ones(n, bool),
            np.zeros(n, np.int32),
            gas,
            sat(278) * 0.38 + w.sum() * np.pi / 6 * 1e-18 / volume,
            mt=mt[i] * 2.64e7 * volume,
        )
        boxes.append(box)
    return (
        boxes,
        events,
        dict(volume=volume, exchange=mix, lift=lift, acid=acid, nh3=nh3),
    )


@lru_cache
def brownian_table():
    """Tabulate the transition-regime kernel in cubic centimetres per second."""
    d = np.geomspace(1.25, 40000, 112)
    r = d * 5e-10
    kernel = (
        get_brownian_kernel_via_system_state(
            r, 4 * np.pi / 3 * r**3 * 1400, 278.0, 101325.0
        )
        * 1e6
    )
    return np.sqrt(d[:-1] * d[1:]), kernel


def cohort(box):
    """Credit pending lineages the first time their particle reaches 10 nm."""
    grown = (box.pending > 0) & (box.dry.sum(1) >= 1000.0)
    box.reached += int(box.pending[grown].sum())
    box.pending[grown] = 0


def collide(box, dt, volume, rng):
    """Frozen Brownian kernel, Poisson pairs by wet diameter, one merger per
    slot/step.
    """
    bins, kernel = brownian_table()
    group = np.searchsorted(bins, np.cbrt(box.dry.sum(1) + box.water))
    order = np.argsort(group, kind="stable")
    count = np.bincount(group, minlength=len(bins) + 1)
    start = np.r_[0, count.cumsum()[:-1]]
    pop = np.flatnonzero(count)
    a, b = np.triu_indices(len(pop))
    a, b = pop[a], pop[b]
    pairs = count[a] * (count[b] - (a == b)) / np.where(a == b, 2, 1)
    draws = rng.poisson(pairs * kernel[a, b] * dt / volume)
    a, b = np.repeat(a, draws), np.repeat(b, draws)
    if not len(a):
        return
    left = order[start[a] + rng.integers(count[a])]
    pos = np.empty(len(order), int)
    pos[order] = np.arange(len(order))
    j = rng.integers(count[b] - (a == b))
    j += (a == b) & (j >= pos[left] - start[a])
    right = order[start[b] + j]
    used = set()
    keep = []
    for k in rng.permutation(len(a)):
        survivor, consumed = int(left[k]), int(right[k])
        if survivor not in used and consumed not in used:
            used.update((survivor, consumed))
            keep.append(k)
    survivor, consumed = left[keep], right[keep]
    box.dry[survivor] += box.dry[consumed]
    box.water[survivor] += box.water[consumed]
    box.old[survivor] |= box.old[consumed]
    box.pending[survivor] += box.pending[consumed]
    lost = survivor[box.old[survivor]]
    box.captured += int(box.pending[lost].sum())
    box.pending[lost] = 0
    alive = np.ones(len(box.dry), bool)
    alive[consumed] = False
    for field in ("dry", "water", "old", "pending"):
        setattr(box, field, getattr(box, field)[alive])
    box.mergers += len(consumed)
    cohort(box)


def exchange(boxes, p, rng):
    """Exchange particles, gas and water between equal volumes."""
    a, b = boxes
    ma = rng.random(len(a.dry)) < p
    mb = rng.random(len(b.dry)) < p
    for field in ("dry", "water", "old", "pending"):
        x, y = getattr(a, field), getattr(b, field)
        setattr(a, field, np.concatenate([x[~ma], y[mb]]))
        setattr(b, field, np.concatenate([y[~mb], x[ma]]))
    for field in ("gas", "mt", "qt"):
        x, y = getattr(a, field), getattr(b, field)
        setattr(a, field, (1 - p) * x + p * y)
        setattr(b, field, p * x + (1 - p) * y)


def met(t, i, lift):
    """Prescribed moist ascent/descent of one box through an entraining edge."""
    if i:
        return 278.0, 0.38
    humid = np.clip((t - 7200) / 1800, 0, 1) * np.clip((16200 - t) / 1800, 0, 1)
    height = (
        lift
        * np.clip((t - 9000) / 1800, 0, 1)
        * np.clip((14400 - t) / 1800, 0, 1)
    )
    return 278.0 - 0.007 * height, 0.38 + 0.635 * humid


def water_step(box, t0, t1, i, par, cloud):
    """Implicit competition for finite water, bulk air forcing & haze
    equilibrium.
    """
    volume = par["volume"]
    factor = np.pi / 6 * 1e-18 / volume  # d³ nm³ to kg water per m³ of air
    temp, rh = met(t1, i, par["lift"]) if cloud else (278.0, 0.38)
    _, oldrh = met(t0, i, par["lift"]) if cloud else (278.0, 0.38)
    box.qt += sat(278.0) * (rh - oldrh)
    ws = sat(temp)
    dry = box.dry.sum(1)
    kd = box.dry @ KAPPA
    # Relax surface haze when unsaturated. Solve the shared vapor balance.
    if box.qt / ws < 0.99:
        k = factor * kd.sum() / ws
        # q/w_s = r + k*r/(1-r), smaller quadratic root.
        ratio = box.qt / ws
        v = (
            2
            * ratio
            / (1 + k + ratio + np.sqrt((1 + k + ratio) ** 2 - 4 * ratio))
        )
        box.water = kd * v / (1 - v)
        liquid = box.water.sum() * factor
        vapor = v * ws
        active = 0
    else:
        critical = np.sqrt(4 * 2.25**3 / (27 * kd))
        wet2 = (dry + box.water) ** (2 / 3)
        haze = kd * 49
        active0 = box.water > haze * 1.05 + 100
        # Finite-step diffusional growth from the haze branch or existing drop.
        base = np.maximum(wet2, (dry + haze) ** (2 / 3))
        g = 8 * 6e-11 * 1e18 * (t1 - t0)

        def liquid_at(s):
            increment = g * np.where(active0, s, np.maximum(s - critical, 0))
            return np.maximum(
                haze, np.maximum(base + increment, 0) ** 1.5 - dry
            )

        lo, hi = -0.10, max(0.001, box.qt / ws - 1)
        for _ in range(30):
            mid = (lo + hi) / 2
            if ws * (1 + mid) + factor * liquid_at(mid).sum() > box.qt:
                hi = mid
            else:
                lo = mid
        box.water = liquid_at((lo + hi) / 2)
        liquid = box.water.sum() * factor
        vapor = (1 + (lo + hi) / 2) * ws
        active = np.count_nonzero(
            (dry + box.water > 1000.0**3) & (box.water > haze * 1.05)
        )
    np.testing.assert_allclose(liquid + vapor, box.qt, rtol=2e-5, atol=1e-10)
    box.temperature = temp
    box.liquid = liquid * 1000  # g/m³
    box.droplets = active / volume
    box.cloud_peak = max(box.cloud_peak, box.droplets)
    box.lwc_peak = max(box.lwc_peak, box.liquid)


def chemistry(box, params, i, t, dt, on, nuc, rng):
    """Prescribed acid supply, alpha-pinene + ozone, finite-vapor uptake &
    J1.7.
    """
    volume = params["volume"]
    gas = box.gas
    source = np.zeros(8)
    # External sulfuric acid/NH3 replenishment during the clear-air pulse.
    pulse = 1800 <= t < 5400
    if pulse:
        for j, c in [(0, params["acid"][i]), (1, params["nh3"][i] * 2.64e7)]:
            source[j] = (c * volume * MOL_VOL[j] - gas[j]) * (
                -np.expm1(-dt / 120)
            )
    if on:
        reacted = box.mt * (-np.expm1(-8.7e-17 * 40e-9 * 2.64e19 * dt))
        box.mt -= reacted
        # Product molecules per reacted alpha-pinene (dimer consumes two units).
        source[5:] += reacted * np.array([0.18, 0.035, 0.0015]) * MOL_VOL[5:]
    gas += source
    c = gas / (volume * MOL_VOL)
    mt = box.mt / volume / 2.64e7
    nh = c[1] / 2.64e7
    box.jrate = 0.0
    # Fixed chamber backgrounds during the precursor pulse: 40 ppbv O3,
    # 1.2 ppbv NOx. The latter bounds applicability, but is absent from Eq. 1.
    if nuc and pulse:
        rh = (box.qt - box.water.sum() * np.pi / 6 * 1e-18 / volume) / sat(
            box.temperature
        )
        if box.temperature != 278.0 or abs(rh - 0.38) > 0.001:
            raise ValueError(
                "Formation outside the chamber temperature or humidity"
            )
    if (
        nuc
        and pulse
        and 5e6 < c[0] < 6e7
        and 10 < nh < 3000
        and 100 < mt < 1200
    ):
        box.jrate = 7.4e-23 * c[0] ** 2 * nh * c[7]
        n = rng.poisson(box.jrate * dt * volume)
        # Explicit assumed dry embryo fractions; J does not provide
        # stoichiometry.
        embryo = 1.7**3 * np.array([0.28, 0.07, 0, 0, 0, 0, 0, 0.65])
        if np.any(n * embryo > gas):
            raise ValueError("Embryo demand exceeds measured vapor inventory")
        if n:
            box.dry = np.vstack([box.dry, np.broadcast_to(embryo, (n, 8))])
            box.water = np.r_[box.water, np.zeros(n)]
            box.pending = np.r_[box.pending, np.ones(n, np.int32)]
            box.old = np.r_[box.old, np.zeros(n, bool)]
            gas -= n * embryo
            box.born += n
    if on:
        d = np.cbrt(box.dry.sum(1))
        # Transition from kinetic surface weighting to diffusion weighting.
        uptake = d**2 / (1 + d / 120)
        share = uptake / uptake.sum()
        rate = np.array([0.0011, 0.00002, 0, 0, 0, 0.00013, 0.00045, 0.00085])
        amount = gas * (-np.expm1(-rate * dt * uptake.sum() / (volume * 5e6)))
        box.dry += share[:, None] * amount
        gas -= amount
    cohort(box)
    return source


def step(boxes, rng, params, regime, t, dt):
    """Exchange, chemical production/formation, Brownian collisions & water."""
    cloud, brown, nuc, chem = REGIMES[regime]
    p = -np.expm1(-2 * params["exchange"] * dt / 3600) / 2
    exchange(boxes, p, rng)
    source = np.zeros(8)
    for i, box in enumerate(boxes):
        source += chemistry(box, params, i, t, dt, chem, nuc, rng)
        if brown:
            collide(box, dt, params["volume"], rng)
        water_step(box, t, t + dt, i, params, cloud)
    return source
