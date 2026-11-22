"""Finite-time interacting cold chambers across the joint draws."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import lru_cache
import numpy as np
from transport import (
    HERE,
    LINES,
    LAWS,
    P,
    RADIUS,
    NATIVE,
    acid_pair,
    conditions,
    draws,
    par,
)
from population import simulate, transport

GRID = np.geomspace(1.39, 1000.0, 256)
PRESSURE = P


@lru_cache(None)
def sinks(temp):
    radius = np.r_[GRID, 50.0] * 0.5e-9
    return (
        par.dynamics.get_brownian_kernel_via_system_state(
            radius, 4 * np.pi / 3 * radius**3 * 1500, temp, P
        )[:-1, -1]
        * 1e6
    )


@lru_cache(None)
def forcing(line, draw_id, law):
    d = draws()[draw_id]
    c = conditions(line, d, law)
    temp = c["temp"]
    if law == "vapor":
        k = transport(GRID, temp, 0.098079, 8e-6 * (temp / 293.15) ** 1.75)
    else:
        radii, kernels = acid_pair(temp, P, 6.4e-20, c["water_per_acid"])
        k = np.interp(GRID, radii, kernels)
        # Continue beyond the small-particle kernel tabulation with a vapor
        # transfer curve matched in value at 12 nm, above both target sizes.
        large = GRID > radii[-1]
        vapor = transport(
            np.r_[radii[-1], GRID[large]],
            temp,
            0.098079,
            8e-6 * (temp / 293.15) ** 1.75,
        )
        k[large] = kernels[-1] * vapor[1:] / vapor[0]
    return dict(
        acid_mass_amu=98.079 + c["water_per_acid"] * 18.01528,
        dilution=c["dilution"],
        acid_uptake_s=k * c["acid"] * 1e6 * c["sticking"],
        base_scale=d["base0"],
        background_hz=sinks(temp) * c["background_cm3"],
        wall_hz=NATIVE(c["eddy_s"], GRID * 0.5e-9, 1500.0, temp, P, RADIUS),
    )


def run(job):
    draw_id, line, law, source, j, step, resolution, seed = job
    data = forcing(line, draw_id, law)
    rows = simulate(
        (line, source, True, step, resolution, seed), {**data, "J17": j}
    )
    return [dict(id=draw_id, block=draw_id // 32, law=law, **r) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    with (HERE / "formation.csv").open() as f:
        rates = {
            (int(r["id"]), int(r["line"]), r["topology"]): float(
                r["predicted_J"]
            )
            for r in csv.DictReader(f)
            if r["split"] in ("223_low", "248_low")
            and r["shape"] == "linear"
            and r["sample"] == "neutral"
        }
    jobs = [
        (i, line, law, src, rates[i, line, src], dt, res, seed)
        for i in range(128)
        for line in LINES
        for law in LAWS
        for src in ("full", "assisted")
        for dt, res, seed in [(30, 256, 9031 + i)]
        + ([(15, 512, 14971 + i)] if i % 16 == 0 and line in (336, 332) else [])
    ]
    if args.pilot:
        print(run(jobs[0]), flush=True)
        return
    path = HERE / "transport-population.csv"
    with (
        path.with_suffix(".tmp").open("w", newline="") as f,
        ProcessPoolExecutor(args.workers) as pool,
    ):
        for i, rows in enumerate(pool.map(run, jobs, chunksize=4)):
            if i == 0:
                writer = csv.DictWriter(f, fieldnames=rows[0])
                writer.writeheader()
            writer.writerows(rows)
            if i % 48 == 0:
                f.flush()
                print(i, len(jobs), flush=True)
    path.with_suffix(".tmp").replace(path)


if __name__ == "__main__":
    main()
