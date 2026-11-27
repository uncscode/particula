# %% How does the size benefit compare with a measured plant-mixture κ contrast?
"""Import only the magnitude of an independent CCN-derived κ contrast."""

import argparse
import json
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import warp as wp
from parcel import PAIRS, Case, load_survival, simulate, write_csv


def experiment(tables, out):
    """Use each mean and both one-standard-deviation corners from Zhao Fig. 2."""
    start = time.monotonic()
    rows = []

    @lru_cache(None)
    def one(c):
        result = simulate(c, tables, "cuda:0")
        rows.append(dict(**asdict(c), **result))
        return result["fraction_pct"]

    comparisons = []
    for pair, (cyclic, acyclic) in PAIRS.items():
        for rh in ("dry", "wet"):
            for w in (0.2, 1.0):
                for kc, ka, tag in (
                    (0.15, 0.15, "same affinity"),
                    (0.15, 0.07, "reported means"),
                    (0.13, 0.08, "closest 1-SD endpoints"),
                    (0.17, 0.06, "furthest 1-SD endpoints"),
                ):
                    cyc = one(Case(precursor=cyclic, rh=rh, w=w, kappa=kc))
                    acy = one(Case(precursor=acyclic, rh=rh, w=w, kappa=ka))
                    comparisons.append(
                        dict(
                            pair=pair,
                            rh=rh,
                            w=w,
                            scenario=tag,
                            cyclic_kappa=kc,
                            acyclic_kappa=ka,
                            cyclic_pct=cyc,
                            acyclic_pct=acy,
                            difference_pp=acy - cyc,
                        )
                    )
    write_csv(out / "plant.csv", rows)
    write_csv(out / "plant-comparisons.csv", comparisons)
    (out / "plant-runtime.json").write_text(
        json.dumps(
            dict(
                runs=len(rows),
                device=wp.get_device("cuda:0").name,
                wall_seconds=time.monotonic() - start,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--survival", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    if (args.out / "plant.csv").exists():
        raise FileExistsError("Use a new directory")
    experiment(load_survival(args.survival), args.out)
