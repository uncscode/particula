"""Compare smaller checkpoints on the frozen held-out draw streams."""

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from redraw_design import TEST, WINDOWS, cases
from redraw_results import reliability, trials
from redraw_run import worker

COMPARATORS = ((4, 8, 1), (10, 8, 1), (10, 16, 0), (10, 16, 1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--optics", default=".artifacts/boreal/optics.npz")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    jobs = [
        (a.root, a.root / "comparators", c, t, COMPARATORS, a.optics)
        for c in cases(TEST)
        for t in WINDOWS
    ]
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(worker, jobs))
    d = pd.concat(
        [
            pd.read_csv(a.root / "comparators" / c.key / f"{t}.csv.gz")
            for c in cases(TEST)
            for t in WINDOWS
        ]
    )
    ref = pd.read_csv(a.out / "redraw-trajectories.csv.gz").query(
        "recipe=='full'"
    )
    key = ["seed", "n", "start", "draw", "lead"]
    cols = [
        "dry_abs_pct",
        "wet_abs_pct",
        "ccn_max_pp",
        "transfer_pct",
        "new_cdf_pp",
    ]

    np.testing.assert_allclose(
        d[d.recipe == "full"].sort_values(key)[cols],
        ref.sort_values(key)[cols],
        rtol=1e-8,
        atol=1e-10,
    )
    tbl = reliability(trials(d)).merge(
        d.groupby(["recipe", "n"])
        .stored_bytes.median()
        .rename("median_bytes")
        .reset_index(),
        on=["recipe", "n"],
    )
    tbl.to_csv(
        a.out / "redraw-comparators.csv", index=False, float_format="%.8g"
    )
    sample_case = next(c for c in cases(TEST[:1]) if c.n == 10000)
    source = a.root / "comparators" / sample_case.key / "10-16-1.npz"
    dest = a.out / "redraw-cloud-16.npz"
    shutil.copyfile(source, dest)
    from redraw_replay import replay

    check = replay(dest, 0, a.optics)
    expected = d.query(
        "seed==@sample_case.seed & n==10000 & start==10800 "
        "& draw==0 & recipe=='10-16-1'"
    ).sort_values("lead")
    np.testing.assert_allclose(
        check.to_numpy(),
        expected[check.columns].to_numpy(),
        rtol=1e-8,
        atol=1e-10,
    )
