"""Replicate the limiting burst-onset window in independent air masses."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from campaign import audit, inventory, write
from model import step
from redraw_design import DT, FORCING
from redraw_run import worker
from redraw_scaling import RECIPES
from stage_two_design import REGIME, Case, initialize

SEEDS = (10111, 10247, 10369, 10487, 10601, 10723, 10853, 10973)
START = 1800


def burst_cases():
    """Use two fresh seeds per forcing pattern at both population sizes."""
    return [
        Case(seed, *level, n=n, dt=DT)
        for n in (25000, 50000)
        for seed, level in zip(SEEDS, FORCING * 2, strict=True)
    ]


def run(job):
    """Evolve to onset, then run the usual 30-minute forecast from disk."""
    root, case, table = job
    folder = root / case.key
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{START}.npz"
    if not path.exists():
        boxes, rng, forcing = initialize(case)
        initial = inventory(boxes)
        source = np.zeros(8)
        for t in range(0, START, DT):
            source += step(boxes, rng, forcing, REGIME, t, DT)
        audit(boxes, initial, case.n, source)
        write(path, boxes, rng, START)
    worker((root, root / "burst", case, START, RECIPES, table))


def main():
    """Execute the additional burst forecasts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--optics", default=".artifacts/boreal/optics.npz")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    with ProcessPoolExecutor(args.workers) as pool:
        list(
            pool.map(run, [(args.root, c, args.optics) for c in burst_cases()])
        )


if __name__ == "__main__":
    main()
