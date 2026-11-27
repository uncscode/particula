"""Repeat the specific held-out witnesses with more particles & finer steps."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from campaign import read, reference
from metrics import compress
from restart import activation, branch, relative

from optics import Optics


def repeat(job):
    """Return the signed-boundary observables for a matched case."""
    root, variant, seed, reg, cap, optic = job
    folder = root / variant / str(seed) / reg
    dt = {"finer": 75, "15sec": 15}.get(variant, 300)
    n = 200000 if variant == "double" else 100000
    if not (folder / "done").exists():
        reference((root / variant, seed, reg, n, dt, 0))
    if reg == "c0b0n0":
        box = read(folder / "07200.npz")[0][0]
        original = box.dry.copy()
        opt = Optics(optic)
        area = opt.areas(box.dry, box.water)[:, 0].sum()
        compress(box, 10, cap, 2)
        rmse, sup = activation(original, box.dry)
        return [
            variant,
            seed,
            reg,
            cap,
            n,
            dt,
            rmse,
            sup,
            relative(opt.areas(box.dry, box.water)[:, 0].sum(), area),
            0.0,
        ]
    dest = root / "limiting" / variant
    path = dest / str(seed) / reg / f"g10-k{cap}-r2.csv"
    if not path.exists():
        branch((root / variant, dest, seed, reg, 10, cap, 2, optic))
    d = pd.read_csv(path)
    row = d[
        (d.start == 10800)
        & (d.end == (21600 if reg == "c1b1n0" else 14400))
        & (d.parcel == (0 if reg == "c1b1n0" else 1))
    ].iloc[0]
    return [
        variant,
        seed,
        reg,
        cap,
        n,
        dt,
        row.rmse,
        row.sup,
        row.coated,
        row.droplets,
    ]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--optics", default=".artifacts/boreal/optics.npz")
    a = p.parse_args()
    jobs = [
        (a.root, v, 1320, r, k, a.optics)
        for v in ["base", "repeat", "double", "finer"]
        for r, k in [("c0b0n0", 16), ("c0b0n1", 7), ("c0b1n1", 7)]
    ]
    jobs.extend(
        (a.root, v, 1562, "c1b1n0", 16, a.optics)
        for v in ["base", "repeat", "double", "15sec"]
    )
    with ProcessPoolExecutor(4) as pool:
        rows = list(pool.map(repeat, jobs))
    pd.DataFrame(
        rows,
        columns="variant seed regime cap n dt rmse sup coated droplets".split(),
    ).to_csv(a.out, index=False, float_format="%.8g")
