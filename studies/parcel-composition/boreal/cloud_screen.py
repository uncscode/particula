"""Exclude integer restart caps when the very next step exceeds the future
peak.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
from campaign import read
from metrics import make_tree, prepare
from model import REGIMES, init, step


def screen(job):
    """Test all 256 caps against a future peak lower bound."""
    base, seed, regime, out = job
    folder = base / str(seed) / regime
    n, dt, _ = map(int, (folder / "done").read_text().split())
    boxes, rng, _ = read(folder / "10800.npz")
    state = rng.bit_generator.state
    _, _, params = init(seed, n)
    truth = pd.read_csv(folder / "trajectory.csv")
    peak = (
        truth[truth.seconds > 10800][["drops_cm3_A", "drops_cm3_B"]]
        .to_numpy()
        .max()
    )
    trees = [make_tree(prepare(b, 10), 256, 2) for b in boxes]
    rows = []
    for cap in range(1, 257):
        subset = []
        for orig, (means, meta, index) in zip(boxes, trees, strict=True):
            b = copy(orig)
            b.gas = orig.gas.copy()
            which = (meta[:, 1] <= cap) & (meta[:, 2] > cap)
            assignment = np.empty(len(index), int)
            for j in np.flatnonzero(which):
                _, _, _, pos, count = meta[j]
                assignment[index[pos : pos + count]] = j
            b.dry = means[assignment, :8].copy()
            b.water = means[assignment, 8]
            subset.append(b)
        rng.bit_generator.state = state
        step(subset, rng, params, regime, 10800, dt)
        drops = max(b.droplets for b in subset)
        rows.append([seed, regime, cap, drops, peak, drops > peak * 1.10])
    pd.DataFrame(
        rows,
        columns="seed regime cap next_drops ref_future_peak excluded".split(),
    ).to_csv(out / f"screen-{regime}.csv", index=False)
    print(regime, sum(r[-1] for r in rows), "excluded", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    with ProcessPoolExecutor(3) as pool:
        list(
            pool.map(
                screen,
                [
                    (a.base, 594, r, a.out)
                    for r, flags in REGIMES.items()
                    if flags[0]
                ],
            )
        )
