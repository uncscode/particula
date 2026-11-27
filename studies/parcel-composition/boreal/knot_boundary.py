"""Refine the calibration snapshot lying nearest the six-movable-knot gate."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from metrics import prepare, profile
from model import init, step
from shape import criteria, movable


def check(n, dt, stream):
    """Recompute hour 0.5 in history 826 at specified physical resolution."""
    b, rng, par = init(826, n)
    if stream:
        rng = np.random.default_rng(np.random.SeedSequence([826, stream]))
    for t in range(0, 1800, dt):
        step(b, rng, par, "c0b0n0", t, dt)
    p = b[1]
    chi, count, _ = profile(p, prepare(p, 16), np.zeros((len(p.dry), 1)))
    y = chi[1:-1]
    w = count[1:-1] * np.isfinite(y)
    y = np.nan_to_num(y)
    _, curve = movable(y, w)
    return [n, dt, stream, *criteria(y, curve, w)]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    a = p.parse_args()
    rows = [
        check(n, dt, s)
        for n, dt, s in [
            (100000, 300, 0),
            (100000, 300, 1),
            (200000, 300, 0),
            (100000, 75, 0),
            (200000, 75, 1),
        ]
    ]
    pd.DataFrame(
        rows, columns="particles dt stream overall small middle large".split()
    ).to_csv(a.out, index=False, float_format="%.7g")
