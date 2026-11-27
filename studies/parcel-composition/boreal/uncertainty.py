"""Measure reference sensitivity to sampling and time step."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from campaign import read
from model import REGIMES, SEEDS
from restart import activation, relative

from optics import Optics


def compare(job):
    """Compare matched full-resolution histories at equal physical times."""
    base, variant, seed, reg, optic = job
    refdir = base / str(seed) / reg
    testdir = variant / str(seed) / reg
    nr = int((refdir / "done").read_text().split()[0])
    nt = int((testdir / "done").read_text().split()[0])
    r = pd.read_csv(refdir / "trajectory.csv")
    v = pd.read_csv(testdir / "trajectory.csv")
    drops = relative(v.drops_cm3_A.max(), r.drops_cm3_A.max())
    lw = relative(v.water_g_m3_A.max(), r.water_g_m3_A.max())

    def births(d):
        return d.births_A.iloc[-1] + d.births_B.iloc[-1]

    born = relative(births(v) * nr / nt, births(r))

    def survival(d):
        return (d.reached_A.iloc[-1] + d.reached_B.iloc[-1]) / max(1, births(d))

    surv = 100 * abs(survival(v) - survival(r))
    opt = Optics(optic)
    rows = []
    for t in range(0, 21601, 3600):
        full, _, _ = read(refdir / f"{t:05d}.npz")
        test, _, _ = read(testdir / f"{t:05d}.npz")
        for i, (b, f) in enumerate(zip(test, full, strict=True)):
            a, s = activation(f.dry, b.dry)
            area = opt.areas(b.dry, b.water).sum(0) * nr / nt
            ref = opt.areas(f.dry, f.water).sum(0)
            rows.append(
                [
                    seed,
                    reg,
                    variant.name,
                    t,
                    i,
                    a,
                    s,
                    *[relative(x, y) for x, y in zip(area, ref, strict=True)],
                    drops,
                    lw,
                    born,
                    surv,
                    v.drops_cm3_A.max(),
                    births(v) / (2 * nt / 1000),
                ]
            )
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument(
        "--variants",
        nargs="+",
        default=[
            "repeat",
            "double",
            "fine",
            "finer",
            "finest",
            "15sec",
            "5sec",
        ],
    )
    p.add_argument("--optics", default=".artifacts/boreal/optics.npz")
    a = p.parse_args()
    jobs = [
        (a.root / "base", a.root / v, s, r, a.optics)
        for v in a.variants
        for s in SEEDS[8:]
        for r in REGIMES
        if (v == "repeat" or r == "c1b1n1")
        and (a.root / v / str(s) / r / "done").exists()
    ]
    with ProcessPoolExecutor(3) as pool:
        rows = [row for results in pool.map(compare, jobs) for row in results]
    pd.DataFrame(
        rows,
        columns=(
            "seed regime variant t parcel rmse sup dry wet hom "
            "droplets lwc formation survival peak_cm3 births_cm3"
        ).split(),
    ).to_csv(a.out, index=False, float_format="%.8g")
