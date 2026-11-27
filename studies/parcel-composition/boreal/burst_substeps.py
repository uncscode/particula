"""Trace when formation turns on during a fixed 75-second forecast."""

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from burst_counts import FINE, FRESH, SPEC, START, N
from model import step
from redraw_checkpoint import draw_box, fit_box
from redraw_design import DT, FORCING
from stage_two_design import REGIME, Case, initialize


def trace(job):
    """Subdivide the same physical window for each particle layout."""
    seed, levels, dt = job
    case = Case(seed, *levels, n=N, dt=DT)
    boxes, rng, forcing = initialize(case)
    for t in range(0, START, DT):
        step(boxes, rng, forcing, REGIME, t, DT)
    packed = [fit_box(b, *SPEC) for b in boxes]
    rows = []
    for draw in range(8):
        key = [seed, N, START, 7521, draw]
        sample = np.random.default_rng([*key, 2])
        for method in ("full", "redraw"):
            event = np.random.default_rng([*key, 1])
            state = (
                copy.deepcopy(boxes)
                if method == "full"
                else [draw_box(b, sample) for b in packed]
            )
            for tick in range(round(DT / dt)):
                lead = tick * dt
                before = [b.born for b in state]
                step(state, event, forcing, REGIME, START + lead, dt)
                for i, b in enumerate(state):
                    rows.append(
                        dict(
                            seed=seed,
                            draw=draw,
                            method=method,
                            dt=dt,
                            end_s=lead + dt,
                            parcel=i,
                            expected=b.jrate * dt * forcing["volume"],
                            births=b.born - before[i],
                        )
                    )
    return rows


def main():
    """Write the observed and expected count in each substep."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--workers", type=int, default=5)
    args = p.parse_args()
    jobs = [
        (seed, level, dt)
        for seed, level in zip(
            (*FINE, FRESH[1]), (FORCING[1], FORCING[0], FORCING[1]), strict=True
        )
        for dt in (75, 25, 15, 5, 1)
    ]
    jobs.append((FINE[0], FORCING[1], 0.5))
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [r for group in pool.map(trace, jobs) for r in group]
    pd.DataFrame(rows).to_csv(
        args.output,
        index=False,
        float_format="%.12g",
        compression={"method": "gzip", "mtime": 0},
    )


if __name__ == "__main__":
    main()
