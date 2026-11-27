"""Prepare the same 16 air masses for multi-observable checkpoint tests."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from burst_counts import FRESH, SEEDS
from campaign import audit, inventory, write
from model import step
from redraw_design import DT, FORCING
from stage_two_design import REGIME, Case, initialize

CASES = [
    Case(s, *f, n=50000, dt=DT)
    for s, f in zip(SEEDS + FRESH, FORCING * 4, strict=True)
]
SHORT = (3600, 10800)
ASCENT = 9000


def prepare(job):
    """Recreate the history from its seed and audit each saved state."""
    root, c = job
    folder = root / c.key
    folder.mkdir(parents=True, exist_ok=True)
    starts = (*SHORT, ASCENT)
    if all((folder / f"{s}.npz").is_file() for s in starts):
        return
    boxes, rng, forcing = initialize(c)
    initial = inventory(boxes)
    source = np.zeros(8)
    for t in range(0, max(starts), DT):
        source += step(boxes, rng, forcing, REGIME, t, DT)
        if t + DT in starts:
            audit(boxes, initial, c.n, source)
            write(folder / f"{t + DT}.npz", boxes, rng, t + DT)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(prepare, [(a.root, c) for c in CASES]))
