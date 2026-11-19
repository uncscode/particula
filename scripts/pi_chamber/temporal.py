"""Particle-resolved time snapshots with explicit injection and removal."""

import json
from pathlib import Path

import numpy as np
import warp as wp
from scipy.optimize import minimize_scalar
from scripts.pi_chamber.growth import KELVIN, SMAX, F, build_steps, grow

# Include dry seeds and both optical thresholds.
EDGES = np.array(
    [
        0.12,
        0.16,
        0.22,
        0.30,
        0.42,
        0.6,
        0.8,
        1,
        1.3,
        1.8,
        2.5,
        3.5,
        5,
        7,
        10,
        15,
        25,
        40,
        120,
    ]
)
TIMES = [
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
]
OBS = {"FW": (1073, 771), "SW": (1327, 1038)}
SOURCE_RATE = 33 / 60
FLOOR = 0.15
DRY = 0.13


@wp.kernel
def advance(
    sizes: wp.array2d(dtype=F),
    noise: wp.array2d(dtype=F),
    zones: wp.array2d(dtype=int),
    births: wp.array(dtype=int),
    table: wp.array3d(dtype=wp.float32),
    errors: wp.array(dtype=int),
    core_mean: F,
    start: int,
    end: int,
    dt: F,
    seed: int,
):
    """Advance a living particle from birth until the snapshot or loss."""
    case, p = wp.tid()
    zone = zones[case, p]  # -1 means deposited or lost.
    birth = births[p]
    if zone < 0 or birth >= end:
        return
    rng = wp.rand_init(seed, p)
    y = sizes[case, p]
    z = noise[case, p]
    saline = case >= 2
    for _ in range(wp.max(start, birth), end):
        next_z = z
        normal = F(wp.randn(rng))
        if F(wp.randf(rng)) < F(1) - wp.exp(-dt):
            next_z = normal
        mean = core_mean
        sigma = wp.where(saline, F(0.01), F(0.0075))
        if zone == 1:
            mean = wp.where(saline, F(-0.015), F(-0.0025))
            sigma *= F(0.25)
        s = mean + sigma * (z + next_z) * F(0.5)
        z = next_z
        y = grow(y, F(0.13), 0, s, table)
        wet = wp.sqrt(y)
        if (
            not wp.isfinite(y)
            or wet < F(0.129999)
            or wet >= F(120)
            or wp.abs(s) >= SMAX
        ):
            wp.atomic_add(errors, 0, 1)
            break
        kn = F(0.132) / wet
        slip = F(1) + kn * (F(1.257) + F(0.4) * wp.exp(-F(1.1) / kn))
        speed = F(998.2 * 9.80665e-12 / (18.0 * 1.813e-5)) * y * slip
        dep = F(0)
        transfer = F(0.15 / 4.0) + speed / F(0.85)
        if zone == 1:
            dep = speed / F(0.15)
            transfer = F(0.85 / 4.0)
        loss_draw = F(wp.randf(rng))
        exchange_draw = F(wp.randf(rng))
        if loss_draw < F(1) - wp.exp(-(F(1.0 / 7200.0) + dep) * dt):
            zone = -1
            break
        both = F(0.25) + speed / F(0.85)
        if exchange_draw < transfer / both * (F(1) - wp.exp(-both * dt)):
            zone = 1 - zone
    sizes[case, p] = y
    zones[case, p] = zone
    noise[case, p] = z


def _population(rng, dt, restart):
    """Build stratified resident and arrival cohorts with physical weights."""
    groups, n0, nin = 8, 1024, 2048
    n = n0 + nin
    tmax = TIMES[-1]
    zones = np.zeros((groups, n), dtype=np.int32)
    # Assign representative diameters separately in each region.
    zones[:, :n0] = rng.random((groups, n0)) < FLOOR
    is_resident = np.arange(n) < n0
    births = np.zeros((groups, n), int)
    for g in range(groups):
        # Jitter one injection time in every equal interval, in steps.
        births[g, n0:] = np.floor(
            (np.arange(nin) + rng.random(nin)) * tmax / (nin * dt)
        ).astype(int)
    initial = np.full((4, groups, n), DRY)
    weights = np.zeros_like(initial)
    for i, case in enumerate(["FW", "FW", "SW", "SW"]):
        haze, cloud = OBS[case]
        for g in range(groups):
            for zone, volume in [(0, 1 - FLOOR), (1, FLOOR)]:
                idx = np.flatnonzero(is_resident & (zones[g] == zone))
                weights[i, g, idx] = (haze + cloud) * volume / len(idx)
                if i % 2 == 1:
                    q = (rng.permutation(len(idx)) + 0.5) / len(idx)
                    frac = haze / (haze + cloud)
                    for flag, local, group in [
                        (q < frac, q / frac, "haze_um"),
                        (q >= frac, (q - frac) / (1 - frac), "cloud_um"),
                    ]:
                        quantiles = np.array(restart[case][group])
                        initial[i, g, idx[flag]] = quantiles[
                            (local[flag] * len(quantiles)).astype(int)
                        ]
            weights[i, g, n0:] = SOURCE_RATE * tmax / nin
    return initial, weights, zones, births, n0


def simulate_history(dt, seed, mean_pct):
    """Return per-interior-volume spectra and paired sampling diagnostics."""
    device = wp.get_device("cuda:0")
    if not device.is_cuda:
        raise RuntimeError("The Pi Chamber particle model requires CUDA.")
    root = Path(__file__).resolve().parents[2]
    restart = json.loads(
        (root / "docs/data/pi-chamber/restart.json").read_text()
    )

    def equilibrium(w):
        v = (w / DRY) ** 3 - 1
        return -v / (v + 1.28) * np.exp(2e6 * KELVIN.value / w)

    peak = minimize_scalar(equilibrium, bounds=(0.14, 8), method="bounded").x

    def upload(a, dtype):
        return wp.array(np.ascontiguousarray(a), dtype=dtype, device=device)

    table = wp.empty((1, 768, 961), dtype=wp.float32, device=device)
    wp.launch(
        build_steps,
        table.shape,
        [upload([DRY], F), upload([1.28], F), upload([peak], F), table, dt],
        device=device,
    )
    rng = np.random.default_rng(seed)
    initial, weights, zones, births, n0 = _population(rng, dt, restart)
    _, groups, n = initial.shape
    nin = n - n0
    is_resident = np.arange(n) < n0
    shape = (4, groups * n)
    sizes = upload(initial.reshape(shape) ** 2, F)
    z = upload(np.tile(rng.normal(size=groups * n), (4, 1)), F)
    regions = upload(np.tile(zones.ravel(), (4, 1)), int)
    arrivals = upload(births.ravel(), int)
    err = wp.zeros(1, dtype=int, device=device)
    rows = []
    prev = 0
    for j, t in enumerate(TIMES):
        end = round(t / dt)
        if end > prev:
            # Short kernels also bound device watchdog time.
            for start in range(prev, end, round(64 / dt)):
                wp.launch(
                    advance,
                    shape,
                    [
                        sizes,
                        z,
                        regions,
                        arrivals,
                        table,
                        err,
                        mean_pct / 100,
                        start,
                        min(end, start + round(64 / dt)),
                        dt,
                        seed + 137 * start + 17 * j,
                    ],
                    device=device,
                )
        prev = end
        diam = np.sqrt(sizes.numpy()).reshape(initial.shape)
        region = regions.numpy().reshape(initial.shape)
        entered = (births < end) | is_resident
        counts = np.empty((4, groups, len(EDGES) - 1))
        for i in range(4):
            for g in range(groups):
                present = entered[g] & (region[i, g] == 0)
                w = weights[i, g, present] / (1 - FLOOR)
                counts[i, g] = np.histogram(
                    diam[i, g, present], EDGES, weights=w
                )[0]
                np.testing.assert_allclose(
                    counts[i, g].sum(), w.sum(), rtol=1e-10
                )
        rows.append(counts)
    if err.numpy()[0]:
        raise ArithmeticError("Growth left the validated interpolation range.")
    counts = np.array(rows)
    means = counts.mean(axis=2)
    cloud_grp = counts[
        ..., np.searchsorted(EDGES, 1.8) : np.searchsorted(EDGES, 40)
    ].sum(axis=-1)
    # Half-L1 difference in absolute spectra divided by mean total number.
    # Paired normalizations include all tracked interior sizes.
    diff = np.abs(means[:, ::2] - means[:, 1::2]).sum(axis=-1) / (
        means[:, ::2] + means[:, 1::2]
    ).sum(axis=-1)
    return dict(
        seconds=TIMES,
        edges_um=EDGES.tolist(),
        cases=["FW dry", "FW wet", "SW dry", "SW wet"],
        bin_cm3=means.tolist(),
        cloud_mc95=(
            2.365 * cloud_grp.std(axis=2, ddof=1) / np.sqrt(groups)
        ).tolist(),
        relative_spectrum_difference=diff.tolist(),
        core_mean_pct=mean_pct,
        dt=dt,
        seed=seed,
        groups=groups,
        residents_per_group=n0,
        injections_per_group=nin,
        device=device.name,
    )
