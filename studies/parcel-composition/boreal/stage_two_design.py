"""Paired forcing and independent stage-two reference trajectories."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from campaign import audit, inventory, write
from model import init, step

TIMES = (2700, 3600, 5400, 10800, 14400, 21600)
SEEDS = (2039, 3499, 5171)
REGIME = "c1b1n1"


@dataclass(frozen=True)
class Case:
    """One forcing cell, particle sample and time discretization."""

    seed: int
    acid: int
    exchange: int
    lift: int
    n: int = 25000
    dt: int = 75
    stream: int = 0

    @property
    def primary(self):
        """Identify the nominal time step, population and event stream."""
        return (self.n, self.dt, self.stream) == (25000, 75, 0)

    @property
    def key(self):
        """Stable, unique cache key for all settings."""
        return "_".join(str(x) for x in asdict(self).values())


def design():
    """Cross three forcing factors within each initial parcel pair."""
    factors = [*product((-1, 1), repeat=3), (0, 0, 0)]
    main = [Case(s, *levels) for s in SEEDS for levels in factors]
    controls = [
        replace(case, **change)
        for case in [
            Case(SEEDS[0], -1, -1, -1),
            Case(SEEDS[0], 0, 0, 0),
            Case(SEEDS[0], 1, 1, 1),
            # Follow up two pre-cloud peak errors after the factorial sweep.
            Case(3499, 1, 1, 1),
            Case(5171, -1, 1, -1),
        ]
        for change in [dict(n=100000), dict(dt=30), dict(stream=1)]
    ]
    return main + controls


def initialize(case):
    """Perturb both reservoirs and scheduled acid supply by the same factor."""
    boxes, rng, par = init(case.seed, case.n)
    factor = 1 + 0.15 * case.acid
    for box in boxes:
        box.gas[0] *= factor
    par["acid"] *= factor
    par["exchange"] *= 1 + 0.35 * case.exchange
    par["lift"] *= 1 + 0.20 * case.lift
    if case.stream:
        rng = np.random.default_rng(
            np.random.SeedSequence([case.seed, case.stream])
        )
    return boxes, rng, par


def reference(job):
    """Evolve each reference and record closure and checkpoints."""
    root, case = job
    folder = root / case.key
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "trace.csv").exists():
        return
    boxes, rng, par = initialize(case)
    initial = inventory(boxes)
    source = np.zeros(8)
    trace = []
    for t in range(0, 21600, case.dt):
        source += step(boxes, rng, par, REGIME, t, case.dt)
        end = t + case.dt
        if end in TIMES:
            audit(boxes, initial, case.n, source)
            write(folder / f"{end}.npz", boxes, rng, end)
        trace.append(
            [
                end,
                boxes[0].droplets,
                boxes[0].liquid,
                sum(b.born for b in boxes),
                sum(b.reached for b in boxes),
                sum(b.captured for b in boxes),
            ]
        )
    path = folder / "trace.tmp"
    pd.DataFrame(
        trace,
        columns=[
            "seconds",
            "drops_cm3",
            "water_g_m3",
            "born",
            "reached",
            "captured",
        ],
    ).to_csv(path, index=False)
    path.replace(folder / "trace.csv")
    print(case.key, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(reference, [(args.root, case) for case in design()]))
