"""Measure sampling & step sensitivity of the recomputed χ bins."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from campaign import read
from metrics import prepare, profile
from model import REGIMES, SEEDS
from shape import criteria


def compare(job):
    """Compare independently resolved profiles on common occupied bins."""
    root, variant, seed, regime = job
    rows = []
    for seconds in range(0, 21601, 3600):
        states = [
            read(root / v / str(seed) / regime / f"{seconds:05d}.npz")[0]
            for v in ("base", variant)
        ]
        for i, (original, replay) in enumerate(zip(*states, strict=True)):
            profiles = []
            for box in (original, replay):
                chi, count, _ = profile(
                    box, prepare(box, 16), np.zeros((len(box.dry), 1))
                )
                profiles.append((chi[1:-1], count[1:-1]))
            (x, n), (y, m) = profiles
            live = np.isfinite(x) & np.isfinite(y)
            err = criteria(np.nan_to_num(x), np.nan_to_num(y), n * live)
            rows.append(
                [
                    seed,
                    regime,
                    variant,
                    seconds,
                    i,
                    *err,
                    *n.reshape(3, 16).sum(1),
                    *(n * ~live).reshape(3, 16).sum(1),
                    *m.reshape(3, 16).sum(1),
                ]
            )
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    jobs = [
        (a.root, v, s, r)
        for v in ("repeat", "double", "15sec")
        for s in SEEDS[8:]
        for r in REGIMES
        if (v == "repeat" or r == "c1b1n1")
        and (a.root / v / str(s) / r / "done").exists()
    ]
    with ProcessPoolExecutor(3) as pool:
        rows = [r for block in pool.map(compare, jobs) for r in block]
    pd.DataFrame(
        rows,
        columns=(
            "seed regime variant time parcel overall small middle "
            "large small_n middle_n large_n missing_s missing_m "
            "missing_l comparison_s comparison_m comparison_l"
        ).split(),
    ).to_csv(a.out, index=False, float_format="%.7g")
