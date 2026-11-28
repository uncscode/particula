"""Axisymmetric finite volumes, optical chord and Particula deposition rates."""

import numpy as np
from workspace import measurement
from observations import D, load, average
from simulate import DIMS, RHO, par
from particula.dynamics.properties.wall_loss_coefficient import (
    get_rectangle_wall_loss_coefficient_via_system_state as wall_coefficient,
)

H = DIMS[2]
RR = 0.254
BANDS = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 65), (65, 80)]
PARAMS = [
    "temperature_offset_K",
    "thermal_contrast",
    "log_exchange",
    "log_circulation",
    "log_outer_density_ratio",
]
BOUNDS = [(-3, 3), (0, 1.5), (-3, 2), (-3, 2), (-2.3, 2.3)]


def geom(mode):
    nr, nz = (1, 1) if mode == "one" else (6, 3)
    radial = (np.arange(nr) + 0.5) / nr
    rings = np.diff((np.arange(nr + 1) / nr) ** 2)
    vol = np.tile(rings / nz, nz)
    diff = np.zeros((nr * nz, nr * nz))
    circ = diff.copy()

    def link(mat, a, b, flux):
        mat[b, a] += flux / vol[a]
        mat[a, a] -= flux / vol[a]

    for z in range(nz):
        for r in range(nr):
            a = z * nr + r
            for b in ([a + 1] if r + 1 < nr else []) + ([a + nr] if z + 1 < nz else []):
                q = min(vol[a], vol[b])
                link(diff, a, b, q)
                link(diff, b, a, q)
    for z in range(nz - 1):
        for r in range(nr - 1):
            a = z * nr + r
            cycle = [a, a + nr, a + nr + 1, a + 1]
            flux = min(vol[cycle]) * np.sin(np.pi * (z + 1) / nz) * np.sin(np.pi * (r + 1) / nr)
            for a, b in zip(cycle, cycle[1:] + cycle[:1]):
                link(circ, a, b, flux)
    for mat in (diff, circ):
        assert np.allclose(mat.sum(0), 0) and np.allclose(mat @ vol, 0)
    return nr, nz, radial, vol, diff, circ


def optical(mode, d, offset=0.0):
    nr, nz, _, v, *_ = geom(mode)
    if mode in ("one", "global"):
        return np.broadcast_to(v[:, None], (len(v), len(d))).copy()
    half = 0.00331 * np.asarray(d) / 2
    lo = np.maximum(-RR, offset - half)
    hi = np.minimum(RR, offset + half)
    w = np.zeros((len(v), len(d)))
    for i in range(nr):
        a = RR * i / nr
        b = RR * (i + 1) / nr
        for left, right in [(-b, -a), (a, b)]:
            w[(nz // 2) * nr + i] += np.maximum(0, np.minimum(hi, right) - np.maximum(lo, left)) / (
                hi - lo
            )
    assert np.allclose(w.sum(0), 1)
    return w


def inputs(run, mode, d, dt):
    nr, nz, r, v, *_ = geom(mode)
    p = 66800 if run == "forced" else 63700
    times = 2.75 + (np.arange(round(4.25 / dt)) + 0.5) * dt
    f = measurement("forcing/figure6.csv").query("run==@run")
    base = np.interp(times, f.time_s, f.temperature_K)
    chi = np.interp(times, f.time_s, f.vapor_mol_mol)
    s = (
        measurement("forcing/figureS2.csv")
        .query("run==@run")
        .pivot(index="time_s", columns="sensor_from_center", values="temperature_K")
    )
    sites = np.array([np.interp(times, s.index, s[i]) for i in range(5)]).T
    tr = np.array([np.interp(r * RR, np.arange(5) * 0.06, row) for row in sites])
    anomaly = np.tile(tr, (1, nz))
    anomaly -= np.sum(anomaly * v, axis=1)[:, None]
    radius = d * 0.5e-6
    tref = float(base.mean())
    speed = par.particles.get_particle_settling_velocity_via_system_state(radius, RHO, tref, p)
    wall = wall_coefficient(3.6, radius, RHO, tref, p, DIMS)
    vertical = wall_coefficient(3.6, radius, RHO, tref, p, (1e12, 1e12, H))
    side = np.maximum(0, wall - vertical)
    extra = np.maximum(0, vertical - speed / H) / 2
    loss = np.zeros((len(v), len(d)))
    for z in range(nz):
        for i in range(nr):
            loss[z * nr + i] = (side / (v[i] * nz) if i == nr - 1 else 0) + nz * extra * (
                (z == 0) + (z == nz - 1)
            )
    if nr == 1:
        assert np.allclose(loss[0] + speed / H, wall)
    else:
        assert np.allclose(v @ loss + speed / H, wall)
    tobs, obs = load(run)
    initial = average(tobs, obs, 2.5, 3.0)
    edges = np.arange(0, len(d) + 1) * (d[1] - d[0])
    init = np.diff(np.interp(edges, np.r_[D - 0.5, D[-1] + 0.5], np.r_[0.0, initial.cumsum()]))
    bands = np.array(
        [
            np.maximum(0, np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo)) / (d[1] - d[0])
            for lo, hi in BANDS
        ]
    )
    obsrows = [average(tobs, obs, t, t + 0.25) for t in np.arange(2.75, 7, 0.25)]
    targets = np.array([[y[(D >= lo) & (D < hi)].sum() for lo, hi in BANDS] for y in obsrows])
    return times, p, base, chi, anomaly, speed, loss, init, bands, targets


def airflow(run):
    return 1.0 if run == "forced" else 0.1


def vapor_chord(mode):
    nr, nz, *_ = geom(mode)
    weights = np.zeros(nr * nz)
    weights[(nz // 2) * nr : (nz // 2 + 1) * nr] = 1 / nr
    return weights
