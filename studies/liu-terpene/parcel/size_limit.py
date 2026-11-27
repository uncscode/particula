# %% Hold volume loss independent of the starting particle diameter.
"""Anchor constant volume survival at the one measured diameter per precursor."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp
from parcel import PAIRS, Case, load_survival, simulate, write_csv


def constant_survival(tables, out):
    """Apply 200-minute fitted survival at the experimental size to all bins."""
    start = time.monotonic()
    dref = {
        "pinene": 165.0,
        "ocimene": 220.0,
        "caryophyllene": 165.0,
        "farnesene": 140.0,
    }
    rows = []
    comparisons = []
    constant = {}
    for (p, rh, model), (d, v) in tables.items():
        if model == "primary":
            value = float(np.interp(np.log(dref[p]), np.log(d), v))
            constant[p, rh, model] = d, np.full_like(v, value)
    for pair, (cyclic, acyclic) in PAIRS.items():
        for rh in ("dry", "wet"):
            for w in (0.2, 1.0):
                fracs = []
                for p in (cyclic, acyclic):
                    case = Case(
                        precursor=p,
                        rh=rh,
                        w=w,
                        kappa=0.10 if pair == "monoterpenes" else 0.05,
                    )
                    result = simulate(case, constant, "cuda:0")
                    rows.append(
                        dict(
                            pair=pair,
                            precursor=p,
                            rh=rh,
                            w=w,
                            initial_observed_d_nm=dref[p],
                            fitted_volume_fraction_200min=constant[
                                p, rh, "primary"
                            ][1][0],
                            **result,
                        )
                    )
                    fracs.append(result["fraction_pct"])
                comparisons.append(
                    dict(
                        pair=pair,
                        rh=rh,
                        w=w,
                        cyclic_pct=fracs[0],
                        acyclic_pct=fracs[1],
                        difference_pp=fracs[1] - fracs[0],
                    )
                )
    write_csv(out / "size-independent.csv", rows)
    write_csv(out / "size-independent-comparisons.csv", comparisons)
    (out / "size-independent-runtime.json").write_text(
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
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--survival", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "size-independent.csv").exists():
        raise FileExistsError("Use a new output directory")
    constant_survival(load_survival(args.survival), args.out)
