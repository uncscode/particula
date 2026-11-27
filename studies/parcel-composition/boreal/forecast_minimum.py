"""Select a forecast-only cap by rejecting smaller budgets on calibration
replays.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from model import REGIMES, SEEDS
from restart import branch


def run(job):
    """Reject each smaller cap on a calibration history before testing."""
    root, out, reg, optic = job
    records = []
    # Optimize search order using only the eight calibration seeds.
    histories = [594, 463, 181, 302, 715, 826, 947, 1068]
    for cap in range(1, 257):
        passed = True
        for seed in histories:
            dest = root / "restarts" / str(seed) / reg / f"g10-k{cap}-r2.csv"
            if not dest.exists():
                branch(
                    (
                        root / "base",
                        root / "restarts",
                        seed,
                        reg,
                        10,
                        cap,
                        2,
                        optic,
                    )
                )
            d = pd.read_csv(dest)
            d = d[(d.end % 3600 == 0) | d.end.isin([5400, 9000])]
            e = np.max(d[["rmse", "sup", "coated"]].to_numpy() / [2, 5, 5])
            end = d[d.end == 21600]
            proc = np.max(
                end[["droplets", "liquid", "births", "survival"]].to_numpy()
                / [10, 10, 10, 5]
            )
            records.append([reg, cap, seed, max(e, proc)])
            if max(e, proc) > 1:
                passed = False
                break
        if passed:
            for base, sub in [("base", "restarts"), ("repeat", "paired")]:
                for seed in SEEDS[8:]:
                    dest = root / sub / str(seed) / reg / f"g10-k{cap}-r2.csv"
                    if not dest.exists():
                        branch(
                            (
                                root / base,
                                root / sub,
                                seed,
                                reg,
                                10,
                                cap,
                                2,
                                optic,
                            )
                        )
            break
    pd.DataFrame(records, columns="regime cap seed worst".split()).to_csv(
        out / f"forecast-{reg}.csv", index=False
    )
    print(reg, "forecast minimum", cap if passed else "unresolved", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--optics", default=".artifacts/boreal/optics.npz")
    a = p.parse_args()
    with ProcessPoolExecutor(a.workers) as pool:
        list(
            pool.map(
                run,
                [
                    (a.root, a.out, r, a.optics)
                    for r, v in REGIMES.items()
                    if not v[0]
                ],
            )
        )
