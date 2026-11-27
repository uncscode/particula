"""Compare exact-count restarts with the continuation of their own random
stream.
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from campaign import audit, inventory, read
from metrics import compress, critical
from model import REGIMES, SEEDS, SS, init, step

from optics import Optics


def activation(ref, test):
    """Compare two normalized CDFs at channels and all jumps."""
    r, x = np.sort(critical(ref)), np.sort(critical(test))

    def diff(at, side):
        return 100 * (
            np.searchsorted(x, at, side) / len(x)
            - np.searchsorted(r, at, side) / len(r)
        )

    channel = diff(SS, "right")
    jumps = np.unique(x[(x >= SS[0]) & (x <= SS[-1])])
    # The reference CDF is monotone between consecutive reconstructed jumps.
    return np.sqrt(np.mean(channel**2)), max(
        abs(channel).max(),
        abs(diff(jumps[jumps > SS[0]], "left")).max(initial=0),
        abs(diff(jumps, "right")).max(initial=0),
    )


def relative(value, ref):
    """Return percent error, or absolute error for a zero reference."""
    return abs(100 * (value / ref - 1)) if ref else abs(value)


def _check_identity(boxes, full, rng, rr):
    np.testing.assert_equal(rng.bit_generator.state, rr.bit_generator.state)
    for b, f in zip(boxes, full, strict=True):
        for k in vars(b):
            np.testing.assert_array_equal(getattr(b, k), getattr(f, k))


def branch(job, allocate=None):
    """Advance three exact-count restarts and score future observables."""
    base, out, seed, regime, grid, cap, rule, optic = job
    folder = base / str(seed) / regime
    n, dt, _ = map(int, (folder / "done").read_text().split())
    _, _, params = init(seed, n)
    opt = Optics(optic)
    paths = {}
    for path in sorted(folder.glob("*.npz")):
        with np.load(path) as z:
            paths[int(z["t"])] = path
    truth = np.loadtxt(folder / "trajectory.csv", delimiter=",", skiprows=1)
    rows = []
    for start in (3600, 10800, 18000):
        boxes, rng, t = read(paths[start])
        initial = inventory(boxes)
        totalwater = sum(b.water.sum() for b in boxes)
        if cap:
            for b in boxes:
                local = allocate(b) if allocate is not None else cap
                compress(b, grid, local, rule)
        np.testing.assert_allclose(inventory(boxes), initial, rtol=1e-10)
        np.testing.assert_allclose(
            sum(b.water.sum() for b in boxes), totalwater, rtol=1e-11
        )
        source = np.zeros(8)
        cloud = np.zeros(2)
        baseline_born = sum(b.born for b in boxes)
        for end in range(start + dt, 21601, dt):
            source += step(boxes, rng, params, regime, end - dt, dt)
            cloud = np.maximum(
                cloud,
                [max(b.droplets for b in boxes), max(b.liquid for b in boxes)],
            )
            if end not in paths:
                continue
            full, rr, _ = read(paths[end])
            if not cap:
                _check_identity(boxes, full, rng, rr)
            audit(boxes, initial, n, source)
            select = truth[(truth[:, 0] > start) & (truth[:, 0] <= end)]
            peak = [select[:, [5, 12]].max(), select[:, [6, 13]].max()]
            born = sum(b.born for b in boxes)
            births = sum(b.born for b in full)
            f = sum(b.reached for b in full) / max(1, births)
            survival = 100 * abs(
                sum(b.reached for b in boxes) / max(1, born) - f
            )
            for i, (b, r) in enumerate(zip(boxes, full, strict=True)):
                rmse, sup = activation(r.dry, b.dry)
                refarea = opt.areas(r.dry, r.water).sum(0)
                area = opt.areas(b.dry, b.water).sum(0)
                rows.append(
                    [
                        start,
                        end,
                        i,
                        rmse,
                        sup,
                        *[
                            relative(x, y)
                            for x, y in zip(area, refarea, strict=True)
                        ],
                        relative(cloud[0], peak[0]),
                        relative(cloud[1], peak[1]),
                        relative(born - baseline_born, births - baseline_born),
                        survival,
                        cloud[0] - peak[0],
                        born - births,
                    ]
                )
    dest = out / str(seed) / regime
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"g{grid}-k{cap}-r{rule}.csv"
    with path.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            "start,end,parcel,rmse,sup,coated,wet,homogeneous,droplets,liquid,births,survival,drop_difference_cm3,birth_difference".split(
                ","
            )
        )
        writer.writerows(rows)
    print(seed, regime, grid, cap, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--regimes", nargs="+", default=list(REGIMES))
    p.add_argument("--grid", type=int, default=10)
    p.add_argument("--cap", type=int, default=16)
    p.add_argument("--rule", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--optics", default=".artifacts/boreal/optics.npz")
    a = p.parse_args()
    jobs = [
        (a.base, a.out, s, r, a.grid, a.cap, a.rule, a.optics)
        for s in a.seeds
        for r in a.regimes
        if not (
            a.out / str(s) / r / f"g{a.grid}-k{a.cap}-r{a.rule}.csv"
        ).exists()
    ]
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(branch, jobs))
