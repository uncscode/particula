"""Size-resolved CUDA cohort replay with measured forcing and optional S(IV) chemistry."""

from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile
import sys
import numpy as np
import warp as wp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
from warm_chamber import TEMP, PRESSURE, transport, acid_pair, par
from extract_run1833 import spectrum, curves, integrate, median_samples, ARCHIVE

F = wp.float64
wp.config.kernel_cache_dir = str(HERE.parent / "results/warp-cache")
KD = 0.0001754037
AIR = PRESSURE / (1.380649e-23 * TEMP) / 1e6
GRID = np.geomspace(2.9, 500, 350)
HIGH = None


def speed(molar):
    return np.sqrt(8 * 8.314462618 * TEMP / (np.pi * molar))


@lru_cache(None)
def measurements():
    with ZipFile(ARCHIVE) as z:
        t, db, edges, density = spectrum(z)
        so2 = median_samples(curves(z, "04b")[0], t)
        ozone = median_samples(curves(z, "04a")[0], t)
        acid = median_samples(curves(z, "02b")[1], t)
        oh, _, ho2, *_ = curves(z, "12c")
    valid = np.isfinite(acid)
    acid = np.interp(t, t[valid], acid[valid])
    # Reconstruct peroxide with authors' gas-phase radical forcing (Fig 12c, R27/28/31).
    # The dilution-only and OH sinks plus assumed zero initial peroxide are explicit.
    k = (3e-13 * np.exp(460 / TEMP) + 2.1e-33 * np.exp(920 / TEMP) * AIR) * (
        1 + 2.8e-18 * AIR * 0.58 * 0.0314
    )
    hp = np.zeros(len(ho2))
    for j, dt in enumerate(np.diff(ho2[:, 0]) * 3600):
        loss = KD + 1.8e-12 * 0.5 * (oh[j, 1] + oh[j + 1, 1])
        src = k * (0.5 * (ho2[j, 1] + ho2[j + 1, 1])) ** 2
        hp[j + 1] = (
            hp[j] * np.exp(-loss * dt) + src * (-np.expm1(-loss * dt)) / loss
        )
    peroxide = np.interp(t, ho2[:, 0], hp)
    so = so2 * AIR * 1e-9
    oz = ozone * AIR * 1e-9
    # Species-specific incident number flux per projected particle area (m^-2 s^-1).
    # Bound by incident substrate and oxidant, with a bounded cosubstrate availability.
    ozone_flux = (
        1e6
        * np.minimum(so * speed(0.064066), oz * speed(0.047998))
        * np.minimum(ozone / 42.0, 1.0)
    )
    peroxide_flux = (
        1e6
        * np.minimum(so * speed(0.064066), peroxide * speed(0.034015))
        * np.minimum(so2 / 2.0, 1.0)
    )
    observed = np.column_stack(
        [integrate(density, edges, d) for d in [5.0, 9.7, 15.0]]
    )
    observed[:, 0] -= observed[:, 1]
    return (
        t,
        db,
        edges,
        density,
        acid,
        ozone_flux,
        peroxide_flux,
        observed,
        peroxide,
        valid,
        so,
    )


@lru_cache(None)
def geometry():
    d = GRID
    diffusion = 8.0e-6 * (TEMP / 293.15) ** 1.75
    k = transport(d, TEMP, 0.098079, diffusion)
    # Retain identical tabulation below the original splice for both branches.
    a, p = acid_pair(TEMP, PRESSURE, 6.4e-20, 1.0, 1500.0, 10.0, 141)
    mask = d <= 9.7
    k[mask] = np.interp(d[mask], a, p)
    extended = k.copy()
    a, p = acid_pair(TEMP, PRESSURE, 6.4e-20, 1.0, 1500.0, 19.5, 151)
    middle = (~mask) & (d <= 19.5)
    extended[middle] = np.interp(d[middle], a, p)
    x = 0.58 * np.exp(
        -4 * 0.079 * 0.01801528 / (1000 * 8.314462618 * TEMP * d * 1e-9)
    )
    unit = (
        98.079 + 2 * 17.03052 + 18.01528 * 3 * x / (1 - x)
    ) * 1.66053906660e-27
    inv = 1350 * np.pi / 6 * (d * 1e-9) ** 3 / unit
    deriv = np.gradient(inv, d, edge_order=2)
    kw = 0.002 * np.sqrt(
        par.particles.get_diffusion_coefficient_via_system_state(
            d * 0.5e-9, TEMP, PRESSURE
        )
        / diffusion
    )
    area = np.pi / 4 * (d * 1e-9) ** 2
    return np.column_stack(
        [d, k * 1e6 / deriv, extended * 1e6 / deriv, area / deriv, kw, area]
    )


@wp.func
def index_at(d: F):
    return wp.clamp(
        int(
            wp.log(wp.max(d, F(2.9)) / F(2.9))
            * F(349.0)
            / wp.log(F(500.0 / 2.9))
        ),
        0,
        348,
    )


@wp.func
def value(table: wp.array2d(dtype=F), d: F, col: int):
    i = index_at(d)
    x = wp.clamp(
        (d - table[i, 0]) / (table[i + 1, 0] - table[i, 0]), F(0.0), F(1.0)
    )
    return (F(1.0) - x) * table[i, col] + x * table[i + 1, col]


@wp.kernel
def advance(
    cfg: wp.array2d(dtype=F),
    grid: wp.array2d(dtype=F),
    force: wp.array2d(dtype=F),
    initial: wp.array2d(dtype=F),
    endobs: wp.array(dtype=wp.int32),
    high: F,
    low: F,
    out: wp.array3d(dtype=F),
    budget: wp.array2d(dtype=F),
):
    case, track = wp.tid()
    ninit = initial.shape[0]
    acid_scale = cfg[case, 0]
    wall = cfg[case, 1]
    col = 1 + int(cfg[case, 2])
    go = cfg[case, 3]
    gp = cfg[case, 4]
    start = int(0)
    d = F(0.0)
    n = F(0.0)
    if track < ninit:
        d = initial[track, 0]
        n = initial[track, 1]
        if d <= high:
            if d >= F(5.0) and d < F(9.7):
                wp.atomic_add(out, case, 0, 0, n)
            if d >= F(9.7):
                wp.atomic_add(out, case, 0, 1, n)
            if d >= F(15.0):
                wp.atomic_add(out, case, 0, 2, n)
    else:
        start = track - ninit
        dt = force[start, 0]
        velocity = grid[0, col] * force[start, 1] * acid_scale + grid[0, 3] * (
            go * force[start, 2] + gp * force[start, 3]
        )
        inlet = velocity
        if cfg[case, 5] >= 0.0:
            inlet = grid[0, 1] * force[start, 1] * cfg[case, 5]
        n = dt * inlet * force[start, 4] / (wp.log(F(10.0)) * low)
        d = low
    used_o = F(0.0)
    used_p = F(0.0)
    for s in range(start, force.shape[0]):
        h = force[s, 0]
        if track >= ninit and s == start:
            h = h * F(0.5)
        gas = force[s, 1] * acid_scale
        ox = go * force[s, 2]
        px = gp * force[s, 3]
        v = value(grid, d, col) * gas + value(grid, d, 3) * (ox + px)
        mid = d + F(0.5) * h * v
        dn = h * (value(grid, mid, col) * gas + value(grid, mid, 3) * (ox + px))
        hz = wall * value(grid, d + F(0.5) * dn, 4) + F(0.0001754037)
        exposure = n * (F(1.0) - wp.exp(-hz * h)) / hz
        used_o += exposure * value(grid, mid, 5) * ox
        used_p += exposure * value(grid, mid, 5) * px
        n = n * wp.exp(-hz * h)
        d = d + dn
        obs = endobs[s]
        if obs > 0 and d <= high:
            if d >= F(5.0) and d < F(9.7):
                wp.atomic_add(out, case, obs, 0, n)
            if d >= F(9.7):
                wp.atomic_add(out, case, obs, 1, n)
            if d >= F(15.0):
                wp.atomic_add(out, case, obs, 2, n)
    wp.atomic_add(budget, case, 0, used_o)
    wp.atomic_add(budget, case, 1, used_p)


@lru_cache(None)
def arrays(count, step, device):
    t, db, edges, m, acid, oz, px, _, _, _, _ = measurements()
    t = t[:count]
    low = 2.9
    boundary = np.array(
        [np.interp(np.log(low), np.log(db), r) for r in m[:, :count].T]
    )
    times = (t - t[0]) * 3600
    times = np.round(times / 3600, 10) * 3600
    forcing = np.column_stack((acid[:count], oz[:count], px[:count], boundary))
    f = []
    out = []
    for i, delta in enumerate(np.diff(times)):
        ns = int(np.ceil(delta / step))
        h = delta / ns
        for j in range(ns):
            w = (j + 0.5) / ns
            f.append([h, *((1 - w) * forcing[i] + w * forcing[i + 1])])
            out.append(i + 1 if j + 1 == ns else -1)
    ini = []
    for i, row in enumerate(m[:, 0]):
        l = max(edges[i], np.log10(low))
        u = edges[i + 1]
        if l < u:
            e = np.linspace(l, u, 17)
            ini.extend(zip(10 ** ((e[:-1] + e[1:]) * 0.5), np.diff(e) * row))
    return [
        wp.array(np.array(x), dtype=d, device=device)
        for x, d in [(geometry(), F), (f, F), (ini, F), (out, wp.int32)]
    ], 10 ** edges[-1]


def simulate(config, count=181, step=30, device="cuda:0"):
    (g, f, i, o), high = arrays(count, step, device)
    cfg = wp.array(np.array(config, dtype=float), dtype=F, device=device)
    out = wp.zeros((len(config), count, 3), dtype=F, device=device)
    budget = wp.zeros((len(config), 2), dtype=F, device=device)
    wp.launch(
        advance,
        dim=(len(config), len(i) + len(f)),
        inputs=[cfg, g, f, i, o, high, 2.9, out, budget],
        device=device,
    )
    return out.numpy(), budget.numpy()
