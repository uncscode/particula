"""Freeze uniform snapshot caps on eight histories, then forecast from these
budgets.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from model import REGIMES, SEEDS
from restart import branch

TOL = np.array([2, 5, 5])


def load(root, seed, reg):
    """Read scores, boundary metadata and grid/grouping ordering."""
    with np.load(root / str(seed) / reg / "snapshot.npz") as z:
        return (
            z["errors"],
            z["checkpoints"],
            list(product(z["grids"], z["rules"])),
        )


def select(root, out):
    """Freeze the first passing integer cap separately for every regime."""
    rows = []
    for reg in REGIMES:
        cal = np.stack(
            [load(root, s, reg)[0][..., :3].max(0) for s in SEEDS[:8]]
        ).max(0)
        _, _, choices = load(root, SEEDS[0], reg)
        for i, (grid, rule) in enumerate(choices):
            ok = np.flatnonzero((cal[i] <= TOL).all(1))
            cap = int(ok[0]) + 1 if len(ok) else 0
            rows.append((reg, grid, rule, cap, *cal[i, 15]))
    frame = pd.DataFrame(
        rows,
        columns="regime grid rule cap cal16_rmse cal16_max cal16_abs".split(),
    )
    frame.to_csv(out, index=False)
    return frame


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scores", type=Path)
    p.add_argument("caps", type=Path)
    p.add_argument("--base", type=Path)
    p.add_argument("--restarts", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--optics", default=".artifacts/boreal/optics.npz")
    a = p.parse_args()
    frame = select(a.scores, a.caps)
    if a.base:
        jobs = [
            (a.base, a.restarts, s, r.regime, 10, int(r.cap), 2, a.optics)
            for r in frame.query("grid==10 and rule==2").itertuples()
            for s in SEEDS
            if r.cap
            and not (
                a.restarts / str(s) / r.regime / f"g10-k{r.cap}-r2.csv"
            ).exists()
        ]
        with ProcessPoolExecutor(a.workers) as pool:
            list(pool.map(branch, jobs))
