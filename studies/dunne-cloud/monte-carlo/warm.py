"""Paired joint sampling for the 297.8 K companion chamber experiment."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import ctypes
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import brentq
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [
    str(ROOT / "simba/wall-sulfur"),
    str(ROOT / "simba"),
    str(ROOT / "analysis"),
]
from balance import wall
from experiment import forcing, META
from replay import GRID, KD, TEMP, PRESSURE, par, measurements
from sectional import setup, simulate

FAMILIES = (
    "measured",
    "gas_matched",
    "gas_absolute",
    "ozone",
    "peroxide",
    "bulk",
    "dimer",
    "both",
    "measured_clear",
    "gas_matched_clear",
)
BANDS = ("5–9.7 nm", "≥9.7 nm", "≥15 nm")
LEGACY = (
    "gas_transfer",
    "finite_hard",
    "vdw_fixed",
    "empirical",
    "ammoniation",
    "extended_attraction",
)
FAMILIES += LEGACY
SEEDS = (183301, 183302, 183303, 183304)


def write(name, rows):
    with (HERE / name).open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def draws(m):
    result = []
    for replicate, seed in enumerate(SEEDS):
        uniform = qmc.Sobol(10, scramble=True, seed=seed).random_base2(m)
        extra = qmc.Sobol(3, scramble=True, seed=seed + 800).random_base2(m)
        for i, u in enumerate(uniform):
            coag = 0.5 + u[0]
            # Slope from the zero/double-coag dark fits. Add window variation
            # spanning the observed trimmed-to-earlier fits at the nominal kernel.
            scale = (
                0.4755959064
                + 0.2416414595 * (1 - coag)
                + (-0.021171614 + 0.149367692 * u[1])
            )
            result.append(
                dict(
                    draw=len(result),
                    replicate=replicate,
                    seed=seed,
                    member=i,
                    coag=coag,
                    wall_at_10=scale,
                    acid=1.05 + 0.20 * u[2],
                    gas_absolute=0.9 + 0.20 * u[2],
                    boundary=0.85 + 0.30 * u[3],
                    ozone=0.0008 * u[4],
                    peroxide=0.16 * u[5],
                    pH=(2, 4, 6)[int(3 * u[6])],
                    thermo=("median", "optimized")[int(2 * u[7])],
                    accounting=("pooled", "monomer")[int(2 * u[8])],
                    led_lifetime_s=3600 * u[9],
                    ammoniation_amplitude=0.5 + 0.5 * extra[i, 0],
                    midpoint_nm=3 + 12 * extra[i, 1],
                    power=2 + 2 * extra[i, 2],
                )
            )
    return result


@lru_cache(None)
def grid(bins):
    edges, centers, dx, initial, boundary, weights, _ = setup(bins)
    volume = centers**3
    masses = 1350 * np.pi / 6 * (centers * 1e-9) ** 3
    K = (
        par.dynamics.get_brownian_kernel_via_system_state(
            centers * 0.5e-9, masses, TEMP, PRESSURE
        )
        * 1e6
    )
    i, j = np.triu_indices(bins)
    kernels = K[i, j] * np.where(i == j, 0.5, 1.0)
    sums = volume[i] + volume[j]
    upper = np.searchsorted(volume, sums).clip(1, bins)
    f = (sums - volume[upper - 1]) / (
        np.r_[volume, 2 * volume[-1]][upper] - volume[upper - 1]
    )
    f = np.where(upper == bins, 1.0, f)
    return (
        edges,
        centers,
        dx,
        initial,
        boundary,
        weights,
        kernels,
        upper.astype("i4"),
        f,
    )


@lru_cache(None)
def runlib():
    src, lib = HERE / "warm_solver.cpp", HERE / "warm_solver.so"
    if not lib.exists() or lib.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(
            [
                "g++",
                "-O3",
                "-std=c++17",
                "-fPIC",
                "-shared",
                str(src),
                "-o",
                str(lib),
            ],
            check=True,
        )
    dll = ctypes.CDLL(str(lib))
    d = np.ctypeslib.ndpointer(dtype="f8", flags="C_CONTIGUOUS")
    i = np.ctypeslib.ndpointer(dtype="i4", flags="C_CONTIGUOUS")
    dll.replay.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        d,
        d,
        d,
        d,
        d,
        d,
        d,
        i,
        d,
        ctypes.c_double,
        ctypes.c_int,
        d,
        d,
        d,
    ]
    dll.replay.restype = ctypes.c_int
    return dll


@lru_cache(None)
def hazards(scale, bins):
    nominal = float(wall(10.0, META["eddy_nominal_s"]))
    eddy = brentq(lambda x: float(wall(10.0, x)) - nominal * scale, 1e-6, 500)
    return np.ascontiguousarray(wall(grid(bins)[1], eddy) + KD)


@lru_cache(None)
def field_inputs(bins):
    from unittest.mock import patch
    import field

    # Read the published charge histories without re-exporting the archived table.
    with patch.object(field, "write"):
        inventory = field.charges()
    centers = grid(bins)[1]
    event_time, removed = [], []
    for h, d, frac in inventory:
        event_time.append(h * 3600)
        removed.append(
            np.interp(np.log(centers), np.log(d), frac, left=0, right=0)
        )
    return np.asarray(event_time), np.asarray(removed)


@lru_cache(None)
def legacy_growth(name):
    from warm_chamber import RHO, DMASS, transport, acid_pair

    diff = 8e-6 * (TEMP / 293.15) ** 1.75
    k = transport(GRID, TEMP, 0.098079, diff)
    below = GRID <= 9.7
    if name in ("finite_hard", "vdw_fixed"):
        d, pair = acid_pair(
            TEMP,
            PRESSURE,
            6.4e-20 if name == "vdw_fixed" else 0.0,
            1.0,
            RHO,
            10.0,
            141,
        )
        k[below] = np.interp(GRID[below], d, pair)
    gr = k * 1e6 * DMASS / (np.pi / 2 * RHO * (GRID * 1e-9) ** 2) * 1e9
    if name == "empirical":
        gr[below] = (2.68 * (GRID[below] + 0.3) ** -1.27 + 0.81) / 3.6e10
    return gr


def velocity(family, p, z):
    family = family.removesuffix("_clear")
    if family in LEGACY:
        from neutralization import network_grid

        geo, inp = forcing("effective", 1.0)
        if family == "ammoniation":
            gr = network_grid(
                geo, p["ammoniation_amplitude"], p["midpoint_nm"], p["power"]
            )[:, 1]
        elif family == "extended_attraction":
            gr = geo[:, 2]
        else:
            gr = legacy_growth(family)
        speed = np.interp(z, GRID, gr) / z
        return np.ascontiguousarray(inp[:, 0, None] * p["acid"] * speed)
    lookup = {
        "measured": "effective",
        "ozone": "effective",
        "peroxide": "effective",
        "gas_matched": "gas_equal_training_mean",
        "gas_absolute": "gas_oh",
        "both": "effective",
        "bulk": f"bulk{p['pH']}",
        "dimer": f"dimer_{p['thermo']}_{p['accounting']}",
    }
    gas = p["gas_absolute"] if family == "gas_absolute" else p["acid"]
    if family == "dimer":
        geo, inp = forcing(lookup[family], gas)
        monomer, other = inp[:, 0], inp[:, 1]
    else:
        geo, inp = forcing(lookup[family], 1.0)
        monomer = inp[:, 0] * gas
        a, b = {
            "ozone": (p["ozone"], 0),
            "peroxide": (0, p["peroxide"]),
            "bulk": (1, 1),
            "both": (p["ozone"], p["peroxide"]),
        }.get(family, (0, 0))
        other = a * inp[:, 1] + b * inp[:, 2]
    va = np.interp(z, GRID, geo[:, 1]) / z
    vb = np.interp(z, GRID, geo[:, 3]) / z
    return np.ascontiguousarray(monomer[:, None] * va + other[:, None] * vb)


def replay_one(family, p, bins=80, step=30.0, count=None):
    z, _, dx, initial, boundary, weights, kernel, upper, f = grid(bins)
    t = measurements()[0][:count]
    sec = (t - t[0]) * 3600.0
    speed = velocity(family, p, z)
    out = np.empty((len(t), 3))
    event_time, removed = field_inputs(bins)
    status = runlib().replay(
        bins,
        len(t),
        step,
        dx,
        sec,
        initial,
        boundary * p["boundary"],
        weights,
        speed,
        hazards(p["wall_at_10"], bins),
        kernel,
        upper,
        f,
        p["coag"],
        len(event_time) if family.endswith("_clear") else 0,
        event_time,
        removed,
        out,
    )
    if status:
        raise ArithmeticError((family, p, status))
    return out


def masks():
    t = measurements()[0]
    h = t - t[0]
    return {
        "training": h < 3.8,
        "peak": (h >= 3.8) & (h <= 6.0001),
        "later": h >= 6.36,
    }


def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2, axis=-2))


def main(m, workers):
    t0 = time.monotonic()
    ps = draws(m)
    refs = []
    p = dict(
        ps[0],
        acid=1.25,
        gas_absolute=1.0,
        boundary=1.0,
        wall_at_10=META["scale_10nm"],
        coag=1.0,
        ozone=0.0004,
    )
    for name, spec in [
        ("measured", ("effective", 1.25, 0, 0)),
        ("ozone", ("effective", 1.25, 0.0004, 0)),
        ("measured_clear", ("effective", 1.25, 0, 0)),
    ]:
        got = replay_one(name, p, count=181)
        ref = simulate(*spec, count=181, clear=name.endswith("_clear"))
        err = np.max(abs(got - ref))
        assert err < 1e-6, err
        refs.append(
            dict(family=name, reference="sectional.py", max_abs_cm3=err)
        )
    write("warm-parity.csv", refs)
    write("warm-parameters.csv", ps)

    def screen(p):
        return [replay_one(name, p) for name in FAMILIES]

    with ThreadPoolExecutor(workers) as pool:
        result = []
        for i, y in enumerate(pool.map(screen, ps)):
            result.append(y)
            if (i + 1) % 16 == 0:
                print(
                    i + 1,
                    "of",
                    len(ps),
                    int(time.monotonic() - t0),
                    "seconds",
                    flush=True,
                )
    result = np.array(result)
    assert np.isfinite(result).all() and result.min() >= 0
    t, *_, obs, _, _, _ = measurements()
    output = HERE / "warm-trajectories.tmp.npz"
    np.savez_compressed(
        output,
        hours=t - t[0],
        observed=obs,
        prediction=result,
        family=FAMILIES,
        band=BANDS,
    )
    output.replace(HERE / "warm-trajectories.npz")
    period = masks()
    ref_rows = []
    for idx in (0, len(ps) // 2):
        for j, name in enumerate(FAMILIES):
            better = replay_one(name, ps[idx], bins=140, step=15.0)
            for k, mask in period.items():
                coarse_rmse = rmse(result[idx, j, mask], obs[mask])
                fine_rmse = rmse(better[mask], obs[mask])
                ref_rows.append(
                    dict(
                        draw=idx,
                        family=name,
                        period=k,
                        coarse_rmse=float(coarse_rmse[1]),
                        fine_rmse=float(fine_rmse[1]),
                    )
                )
    write("warm-refinement.csv", ref_rows)
    print("all done", time.monotonic() - t0, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--power", type=int, default=7)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    main(args.power, args.workers)
