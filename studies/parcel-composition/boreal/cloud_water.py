"""Ablate particle-water and dry-composition averaging at the cloudy restart."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from campaign import read
from metrics import compress
from model import init, step


def run(job):
    """Separate changes caused by averaging water and dry composition."""
    folder, seed, reg = job
    f = folder / str(seed) / reg
    n, dt, _ = map(int, (f / "done").read_text().split())
    _, _, params = init(seed, n)
    rows = []
    for treatment in ("exact", "dry only", "water only", "both"):
        boxes, rng, t = read(f / "10800.npz")
        for b in boxes:
            if treatment == "exact":
                continue
            dry = b.dry.copy()
            water = b.water.copy()
            compress(b, 10, 16, 2)
            if treatment == "dry only":
                b.water = water
            if treatment == "water only":
                b.dry = dry
        peak = 0
        for s in range(t, 21600, dt):
            step(boxes, rng, params, reg, s, dt)
            peak = max(peak, *(b.droplets for b in boxes))
        rows.append([seed, reg, treatment, peak])
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    with ProcessPoolExecutor(2) as pool:
        rows = [
            r
            for result in pool.map(
                run,
                [
                    (a.base, s, reg)
                    for s in [594, 1562]
                    for reg in ["c1b1n0", "c1b1n1"]
                ],
            )
            for r in result
        ]
    pd.DataFrame(
        rows, columns=["seed", "regime", "averaged", "peak_cm3"]
    ).to_csv(a.out, index=False)
