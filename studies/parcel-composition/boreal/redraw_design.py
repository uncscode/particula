"""Fresh, blocked six-hour trajectories and event windows for redraw
qualification.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from campaign import audit, inventory, write
from model import step
from stage_two_design import REGIME, Case, initialize

TRAIN = (7187, 7309, 7451, 7583)
TEST = (8329, 8467, 8599, 8719)
FORCING = ((-1, -1, 1), (1, -1, -1), (-1, 1, -1), (1, 1, 1))
SIZES = (5000, 10000)
WINDOWS = {
    1800: "burst onset",
    3600: "burst growth",
    9000: "ascent",
    10800: "in cloud",
    14400: "evaporation",
}
DT = 75
LEADS = (0, DT, 600, 1200, 1800)
# Grid bins per decade, row cap per bin, store one joint direction per row.
CANDIDATES = (
    (4, 8, 1),
    (10, 8, 1),
    (10, 16, 0),
    (10, 16, 1),
    (16, 16, 1),
    (10, 32, 1),
)
DRAWS = 10


def cases(seeds):
    """Cross both particle counts with each independently initialized
    history.
    """
    levels = dict(zip(TRAIN + TEST, FORCING * 2, strict=True))
    return [Case(s, *levels[s], n=n, dt=DT) for s in seeds for n in SIZES]


def reference(job):
    """Save the resolved model at event entry; later observations are
    replayed.
    """
    root, c = job
    folder = root / c.key
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "reference.done").exists():
        return
    boxes, rng, forcing = initialize(c)
    initial = inventory(boxes)
    source = np.zeros(8)
    for t in range(0, 21600, c.dt):
        source += step(boxes, rng, forcing, REGIME, t, c.dt)
        if t + c.dt in WINDOWS or t + c.dt == 21600:
            audit(boxes, initial, c.n, source)
            write(folder / f"{t + c.dt}.npz", boxes, rng, t + c.dt)
    (folder / "reference.done").write_text("complete\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(reference, [(a.root, c) for c in cases(TRAIN + TEST)]))
