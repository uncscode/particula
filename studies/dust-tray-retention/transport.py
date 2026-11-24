"""Dust population balance in vertically exchanging equal-volume air parcels."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import csv
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile
import numpy as np
from scipy.linalg import expm
from study import (
    BOOK,
    DENSITY,
    HEIGHT,
    PRESSURE,
    ROOT,
    TEMP,
    WINDS,
    median,
    normalize,
    par,
    save,
    tv,
    velocities,
)

AIR = PRESSURE / (287.05 * TEMP)
NU = par.gas.get_dynamic_viscosity(TEMP) / AIR
# Table S1, mg/m³, for locating deposition time zero in the archived bulk trace.
C0_S1 = (191.35, 178.20, 186.36, 189.44)


@lru_cache
def inputs(wind):
    with (ROOT / "channels.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream) if int(r["wind_ms"]) == wind]
    raw_d, early, late = (
        np.asarray([float(r[key]) for r in rows])
        for key in ("channel_um", "early_pct", "measured_pct")
    )
    # Extend the dry grid for aggregate births above the instrument range.
    d = np.r_[raw_d, raw_d[-1] * (raw_d[-1] / raw_d[-2]) ** np.arange(1, 8)]
    with ZipFile(BOOK) as book:
        strings = [
            "".join(v.itertext())
            for v in ET.fromstring(book.read("xl/sharedStrings.xml"))
        ]
        records = []
        for row in ET.fromstring(book.read("xl/worksheets/sheet3.xml")).findall(
            ".//{*}row"
        ):
            record = {}
            for c in row:
                val = c.findtext("{*}v")
                if val is not None:
                    record[re.sub(r"\d+", "", c.get("r"))] = (
                        strings[int(val)] if c.get("t") == "s" else float(val)
                    )
            records.append(record)
    tc, vc = (("H", "I"), ("K", "L"), ("N", "O"), ("Q", "R"))[wind // 2]
    air = {
        int(r[tc]): r[vc]
        for r in records
        if isinstance(r.get(tc), float) and isinstance(r.get(vc), float)
    }
    start = next(
        t for t, c in air.items() if abs(c - C0_S1[wind // 2]) < 0.0051
    )
    ustar = 0.0
    if wind:
        col = ("U", "V", "W")[wind // 2 - 1]
        prof = {
            float(r["T"][:-2]) / 100: r[col]
            for r in records
            if isinstance(r.get("T"), str) and r["T"].endswith("cm")
        }
        near = [z for z in sorted(prof) if 0.05 <= z <= 0.4]
        ustar = 0.4 * np.polyfit(np.log(near), [prof[z] for z in near], 1)[0]
    return (
        d,
        np.pad(early / 100, (0, 7)),
        np.pad(late / 100, (0, 7)),
        ustar,
        air[start],
        air[start + 30],
    )


@dataclass(frozen=True)
class Case:
    name: str
    n: int = 12
    dt: float = 120.0
    wall: bool = True
    mechanism: str = "all"
    rh: float = 0.0
    charge: str = "neutral"
    wind_scale: float = 1.0
    wall_scale: float = 1.0
    sticking: float = 1.0
    monitor: bool = False
    air_shape: str = "depleted"
    from_release: bool = False
    history: tuple = ()
    refine_size: bool = False


def properties(d, rh):
    growth = (1 + 0.05 * rh / (1 - rh)) ** (1 / 3)
    wet = d * growth
    density = (DENSITY + 1000 * (growth**3 - 1)) / growth**3
    v = velocities(wet, density)
    radius = wet * 0.5e-6
    diff = par.particles.get_diffusion_coefficient_via_system_state(
        radius, TEMP, PRESSURE
    )
    return radius, density, v, diff


def transport(d, n, u, rh, wall_scale, time):
    """Each column gives a conservative transition from a layer to air, floor or sides."""
    radius, density, v, diff = properties(d, rh)
    dz = HEIGHT / n
    z = dz * np.arange(1, n)
    exchange = (0.4 * u * z * (1 - z / HEIGHT)) / dz**2
    g = np.zeros((len(d), n + 2, n + 2))
    for j, k in enumerate(exchange):
        g[:, j + 1, j] += k
        g[:, j, j + 1] += k
    side_rate = 4 / (np.pi * 2.0) * np.sqrt(wall_scale * u * u / NU * diff)
    for j in range(n):
        g[:, j - 1 if j else n, j] += v / dz
        g[:, n + 1, j] += side_rate
        g[:, j, j] = -g[:, :, j].sum(axis=1)
    out = expm(g * time)
    assert np.min(out) > -1e-15 and np.max(abs(out.sum(axis=1) - 1)) < 2e-9
    return out


def apply(matrix, state):
    return np.einsum("sij,js->is", matrix, state, optimize=False)


def size_grid(d, refine):
    if not refine:
        return d
    grid = np.empty(2 * len(d) - 1)
    grid[::2] = d
    grid[1::2] = np.sqrt(d[:-1] * d[1:])
    return grid


@lru_cache
def initial(wind, n, refine):
    d, p, q, u, c0, c30 = inputs(wind)
    if refine:
        d = size_grid(d, True)
        on_grid = np.zeros(len(d))
        on_grid[::2] = p
        p = on_grid
    m = DENSITY * np.pi / 6 * (d * 1e-6) ** 3
    # Unit initial concentration in every equal-volume layer; sinks sum layer concentrations.
    tracers = np.zeros((n + 2, len(d)))
    tracers[:n] = 1.0
    frac = apply(transport(d, n, u, 0, 0, 1800), tracers)
    initial_mass = normalize(p / (frac[n] / n)) * c0 * 1e-6
    state = frac * (initial_mass / m)[None, :]
    assert np.allclose(normalize(state[n] * m), p)
    return state, m


def kernel(d, n, u, rh, charge, mechanism):
    radius, rho, v, diff = properties(d, rh)
    brown = par.dynamics.get_brownian_kernel_via_system_state(
        radius, 4 * np.pi / 3 * rho * radius**3, TEMP, PRESSURE
    )
    if charge != "neutral":
        phi = par.particles.get_coulomb_enhancement_ratio(
            radius, np.full(len(d), 5), TEMP
        )
        factor = par.particles.get_coulomb_continuum_limit(phi)
        if charge == "bipolar":
            factor = (
                factor + par.particles.get_coulomb_continuum_limit(-phi)
            ) / 2
        brown = brown * factor
    gravity = (
        np.pi * np.add.outer(radius, radius) ** 2 * abs(np.subtract.outer(v, v))
    )
    # Local-equilibrium dissipation, limited to nearest parcel midpoint at the wall.
    z = (np.arange(n) + 0.5) * HEIGHT / n
    eps = u**3 / (0.4 * z)
    k = np.zeros((n, len(d), len(d)))
    if mechanism in ("brownian", "all"):
        k += brown
    if mechanism in ("differential", "all"):
        k += gravity
    if mechanism in ("shear", "all") and u > 0:
        for j, e in enumerate(eps):
            k[j] += (
                par.dynamics.get_turbulent_shear_kernel_st1956_via_system_state(
                    radius, float(e), TEMP, AIR
                )
            )
    return k


def collision_operator(m, k):
    i, j = np.triu_indices(len(m))
    weights = k[:, i, j] * np.where(i == j, 0.5, 1.0)
    s = np.zeros((len(i), len(m)))
    for row, (a, b) in enumerate(zip(i, j)):
        product = m[a] + m[b]
        s[row, a] -= 1
        s[row, b] -= 1
        hi = np.searchsorted(m, product)
        if hi == len(m):
            s[row, -1] += product / m[-1]
        else:
            frac = (product - m[hi - 1]) / (m[hi] - m[hi - 1])
            s[row, hi - 1] += 1 - frac
            s[row, hi] += frac
    assert np.max(abs(s @ m) / (m[i] + m[j])) < 2e-14

    def rate(x):
        return ((x[:, i] * x[:, j]) * weights) @ s

    return rate


def collide(x, rate, seconds):
    """Second-order positive explicit integration, with local step rejection."""
    rem = seconds
    while rem > 1e-8:
        dx = rate(x)
        loss = dx < 0
        h = (
            min(rem, 0.65 * np.min(x[loss] / -dx[loss]))
            if np.any(loss)
            else rem
        )
        while True:
            pred = x + h * dx
            new = x + 0.5 * h * (dx + rate(pred))
            if new.min() >= 0:
                break
            h *= 0.5
        x = new
        rem -= h
    return x


def advance(x, d, m, u, case, seconds):
    n = case.n
    wall = case.wall_scale if case.wall else 0.0
    if case.mechanism == "off":
        return apply(transport(d, n, u, case.rh, wall, seconds), x)
    steps = int(np.ceil(seconds / case.dt))
    dt = seconds / steps
    move = transport(d, n, u, case.rh, wall, dt / 2)
    both = move @ move
    rate = collision_operator(
        m, case.sticking * kernel(d, n, u, case.rh, case.charge, case.mechanism)
    )
    x = apply(move, x)
    for step in range(steps):
        x[:n] = collide(x[:n], rate, dt)
        x = apply(move if step == steps - 1 else both, x)
    return x


def original_channels(mass_profile, mass_pivots, refined):
    if not refined:
        return mass_profile
    # Map newborn intermediate pivots to their bounding original pivots by
    # number interpolation in particle mass, then return the dry mass shares.
    out = mass_profile[::2].copy()
    original = mass_pivots[::2]
    mid = mass_pivots[1::2]
    high = (mid - original[:-1]) / np.diff(original)
    out[:-1] += mass_profile[1::2] / mid * (1 - high) * original[:-1]
    out[1:] += mass_profile[1::2] / mid * high * original[1:]
    assert np.isclose(out.sum(), mass_profile.sum(), atol=0, rtol=1e-12)
    return out


def simulate(wind, case):
    channels, early, late, ustar, c0, c30 = inputs(wind)
    d = size_grid(channels, case.refine_size)
    base, m = initial(wind, case.n, case.refine_size)
    x = base.copy()
    n = case.n
    if case.from_release:
        x[:n] = base.sum(axis=0) / n
        x[n:] = 0.0
        initial_budget = x.sum(axis=0) @ m
        initial_number = x.sum()
        x = advance(x, d, m, ustar, case, 1800)
    air30 = x[:n].mean(axis=0) @ m * 1e6
    if case.monitor:
        x[:n] *= c30 / air30
    if case.air_shape == "tray":
        layer_totals = x[:n] @ m
        x[:n] = layer_totals[:, None] * early[None, :] / m[None, :]
    if not case.from_release:
        initial_budget = x.sum(axis=0) @ m
        initial_number = x.sum()
    first = original_channels(x[n] * m, m, case.refine_size)
    initial_floor = first.sum()
    early_d50 = median(channels, first)
    early_tv = tv(normalize(first), early)
    wall = case.wall_scale if case.wall else 0.0
    scales = case.history or (case.wind_scale,)
    for scale in scales:
        x = advance(x, d, m, ustar * scale, case, 9000.0 / len(scales))
    budget = abs(x.sum(axis=0) @ m / initial_budget - 1)
    assert (
        budget < 3e-9
        and x.min() >= 0
        and x.sum() <= initial_number * (1 + 2e-9)
    )
    floor = original_channels(x[n] * m, m, case.refine_size)
    outside = floor[len(channels) - 7 :].sum() / floor.sum()
    # Compare only published size channels. Report any dry mass outside separately.
    p = normalize(floor[:-7])
    obs = normalize(late[:-7])
    band = (channels[:-7] > 10) & (channels[:-7] <= 20)
    score = dict(
        wind_ms=wind,
        case=case.name,
        parcels=n,
        step_s=case.dt,
        wind_scale=case.wind_scale,
        ustar_ms=ustar,
        wall_scale=wall,
        mechanism=case.mechanism,
        sticking=case.sticking,
        rh_pct=100 * case.rh,
        charge=case.charge,
        match_air30=case.monitor,
        bulk_C0_mgm3=c0,
        measured_air30_mgm3=c30,
        reconstructed_air30_mgm3=air30,
        d50_um=median(channels[:-7], p),
        observed_d50_um=median(channels[:-7], obs),
        early_pred_d50_um=early_d50,
        early_tv_pp=early_tv,
        from_release=case.from_release,
        size_pivots=len(d),
        air_shape=case.air_shape,
        wind_history=",".join(str(a) for a in scales),
        tv_pp=tv(p, obs),
        middle_pct=100 * p[band].sum(),
        observed_middle_pct=100 * obs[band].sum(),
        retained_ratio=floor.sum() / initial_floor,
        sidewall_dry_mass_pct=100 * x[n + 1] @ m / initial_budget,
        number_removed_by_mergers_pct=100 * (1 - x.sum() / initial_number),
        outside_channels_pct=100 * outside,
        budget_rel_error=budget,
        top_bin_budget_fraction=x[:, -1].sum() * m[-1] / initial_budget,
    )

    wall_bins = original_channels(x[n + 1] * m, m, case.refine_size)
    air_bins = original_channels(x[:n].sum(axis=0) * m, m, case.refine_size)
    bins = [
        dict(
            wind_ms=wind,
            case=case.name,
            channel_um=size,
            within_instrument=i < len(p),
            floor_pct_of_inventory=100 * floor[i] / initial_budget,
            sidewall_pct_of_inventory=100 * wall_bins[i] / initial_budget,
            airborne_pct_of_inventory=100 * air_bins[i] / initial_budget,
        )
        for i, size in enumerate(channels)
    ]
    return score, bins


def main():
    nominal = Case("all")
    cases = [
        Case("one_box_floor", n=1, wall=False, mechanism="off"),
        Case("multi_floor", wall=False, mechanism="off"),
        Case("multi_wall", mechanism="off"),
        *[Case(k, mechanism=k) for k in ("brownian", "shear", "differential")],
        nominal,
        Case("all_RH50", rh=0.5),
        Case("all_RH85", rh=0.85),
        Case("all_like5", charge="like"),
        Case("all_bipolar5", charge="bipolar"),
        Case("all_RH85_bipolar5", rh=0.85, charge="bipolar"),
    ]
    rows, distributions = [], []
    for wind in WINDS:
        tests = (
            cases
            + [
                Case("wind_low", wind_scale=0.75),
                Case("wind_high", wind_scale=1.25),
            ]
            if wind
            else cases
        )
        tests = tests + [
            Case("monitor", monitor=True),
            Case("sticking_0.1", sticking=0.1),
            Case("wall_0.1", wall_scale=0.1),
            Case("wall_10", wall_scale=10),
            Case("time_refined", dt=60),
            Case("height_refined", n=24, dt=60),
        ]
        tests.extend(
            [
                Case("from_release", from_release=True),
                Case("monitor_tray_shape", monitor=True, air_shape="tray"),
            ]
        )
        if wind:
            tests.extend(
                [
                    Case("low_then_high", history=(0.75, 1.25)),
                    Case("high_then_low", history=(1.25, 0.75)),
                ]
            )
        if wind == 2:
            tests.extend(
                [
                    Case("size_refined", refine_size=True, dt=60),
                    Case(
                        "monitor_no_collisions", mechanism="off", monitor=True
                    ),
                ]
            )
        if wind == 2:
            tests.extend(
                [
                    Case("monitor_no_wall", monitor=True, wall=False),
                    Case("monitor_RH50", monitor=True, rh=0.5),
                    Case("monitor_RH85", monitor=True, rh=0.85),
                    Case("monitor_like5", monitor=True, charge="like"),
                    Case("monitor_bipolar5", monitor=True, charge="bipolar"),
                    Case("monitor_weak_wind", monitor=True, wind_scale=0.75),
                    Case("monitor_strong_wind", monitor=True, wind_scale=1.25),
                    Case(
                        "monitor_finer",
                        monitor=True,
                        n=24,
                        dt=60,
                        refine_size=True,
                    ),
                ]
            )
        for case in tests:
            score, bins = simulate(wind, case)
            rows.append(score)
            distributions.extend(bins)
        print(wind, rows[-1]["d50_um"], flush=True)
    save("transport-results.csv", rows)
    save("transport-channels.csv", distributions)
    rates = []
    for wind in WINDS:
        u = inputs(wind)[3]
        sizes = np.array([0.5, 2.0, 10.0, 20.0])
        _, _, _, diff = properties(sizes, 0.0)
        beta = 4 / (np.pi * 2.0) * np.sqrt(u**2 / NU * diff)
        for diameter, rate in zip(sizes, beta):
            rates.append(
                dict(
                    wind_ms=wind,
                    dry_diameter_um=diameter,
                    side_rate_per_h=3600 * rate,
                )
            )
    save("sidewall-rates.csv", rates)


if __name__ == "__main__":
    main()
