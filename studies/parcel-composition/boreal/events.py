"""Recover cloud entry & exit from the last preceding hourly reference state."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from campaign import read, write
from model import REGIMES, init, step


def extract(folder):
    """Save the first cloudy and first subsequent cloud-free step."""
    n, dt, _ = map(int, (folder / "done").read_text().split())
    regime = folder.name
    if not REGIMES[regime][0]:
        return
    a = np.loadtxt(folder / "trajectory.csv", skiprows=1, delimiter=",")
    cloud = np.flatnonzero(a[:, 5] > 0)
    if not cloud.size:
        return
    _, _, params = init(int(folder.parent.name), n)
    for label, index in [
        ("cloud-entry", cloud[0]),
        ("cloud-exit", cloud[-1] + 1),
    ]:
        end = int(a[index, 0])
        dest = folder / (label + ".npz")
        if dest.exists():
            continue
        candidates = sorted(folder.glob("[0-9]*.npz"))
        path = max(
            (p for p in candidates if int(p.stem) <= end),
            key=lambda p: int(p.stem),
        )
        boxes, rng, t = read(path)
        for s in range(t, end, dt):
            step(boxes, rng, params, regime, s, dt)
        np.testing.assert_equal(boxes[0].droplets, a[index, 5])
        write(dest, boxes, rng, end)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    a = parser.parse_args()
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(extract, [p.parent for p in a.root.glob("*/*/done")]))
