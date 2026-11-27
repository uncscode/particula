"""Check counter sensitivity to shorter steps from identical states."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from adaptive_trials import ARMS, channel, pack
from campaign import audit, inventory, read
from model import step
from redraw_checkpoint import DEFAULT_RECIPE, draw_box, serialize
from redraw_run import percentage
from response_course import CASES, SHORT
from stage_two_design import REGIME, initialize


def forecast_counts(boxes, par, event, start, dt, n):
    """Advance one 75 s window, validating inventories at its endpoint."""
    state = deepcopy(boxes)
    initial = inventory(state)
    source = np.zeros(8)
    for t in range(start, start + 75, dt):
        source += step(state, event, par, REGIME, t, dt)
    audit(state, initial, n, source)
    return channel(state)


def compare(job):
    """Retain each paired count at 75 s under both discrete time grids."""
    root, out, c, t = job
    boxes, saved, _ = read(root / "response-course" / c.key / f"{t}.npz")
    _, _, par = initialize(c)
    packed = {}
    for arm in ("fixed", *ARMS):
        data, _, _ = pack(boxes, saved, c, par, t, arm)
        packed[arm] = data
        if c == CASES[0] and t == SHORT[0] and arm == "pinned":
            payload = serialize(
                data, c, par, t, saved, dict(base=DEFAULT_RECIPE, edges=arm)
            )
            (out / "redraw-adaptive-pinned.npz").write_bytes(payload)
    records = []
    for trial in range(4):
        stream = [20261127, c.seed, t, trial]
        draws = {}
        for arm, data in packed.items():
            rng = np.random.default_rng([*stream, 1])
            draws[arm] = [draw_box(b, rng) for b in data]
        for dt in (25, 75):
            target = forecast_counts(
                boxes, par, np.random.default_rng([*stream, 0]), t, dt, c.n
            )
            for arm, draw in draws.items():
                num = forecast_counts(
                    draw, par, np.random.default_rng([*stream, 0]), t, dt, c.n
                )
                records.append(
                    dict(
                        seed=c.seed,
                        start=t,
                        trial=trial,
                        dt=dt,
                        method=arm,
                        a=num[0],
                        b=num[1],
                        full_a=target[0],
                        full_b=target[1],
                        counter_pct=percentage(num, target).max(),
                    )
                )
    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    jobs = [(args.root, args.out, c, t) for c in CASES[:4] for t in SHORT]
    with ProcessPoolExecutor(args.workers) as pool:
        batches = pool.map(compare, jobs)
        rows = pd.DataFrame(row for batch in batches for row in batch)
    rows.to_csv(
        args.out / "adaptive-steps.csv", index=False, float_format="%.12g"
    )
