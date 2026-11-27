"""Test the frozen snapshot allocation in held-out forecasts at equal bytes."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from allocation import DEC, uniform
from metrics import prepare
from model import REGIMES, SEEDS
from restart import branch


def forecast(job):
    """Rebuild the frozen allocation at each of the three restart times."""
    base, out, seed, regime, scheme, decade_caps, optics = job

    def allocate(box):
        count = np.diff(prepare(box, 10)[3])
        caps = decade_caps[DEC]
        rows = np.minimum(count, caps).sum()
        return caps if scheme == "adaptive" else uniform(count, rows)

    dest = out / scheme
    path = dest / str(seed) / regime / "g10-k1-r2.csv"
    if not path.exists():
        branch((base, dest, seed, regime, 10, 1, 2, optics), allocate)
    result = pd.read_csv(path)
    result["seed"] = seed
    result["regime"] = regime
    result["allocation"] = scheme
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("study", type=Path)
    p.add_argument("--workers", default=4, type=int)
    p.add_argument("--optics", default=".artifacts/boreal/optics.npz")
    a = p.parse_args()
    levels = (
        pd.read_csv(a.study / "particle-allocation.csv")
        .iloc[-1, :3]
        .to_numpy(int)
    )
    jobs = [
        (a.root / "base", a.root / "allocated", s, r, scheme, levels, a.optics)
        for s in SEEDS[8:]
        for r in REGIMES
        for scheme in ("adaptive", "uniform")
    ]
    with ProcessPoolExecutor(a.workers) as pool:
        data = pd.concat(pool.map(forecast, jobs))
    data.to_csv(
        a.study / "allocated-restarts.csv.gz",
        index=False,
        float_format="%.7g",
        compression={"method": "gzip", "mtime": 0},
    )
