"""Rescore frozen snapshot & χ budgets on independent and refined particles."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from shape import basis, criteria, fit, positions


def score(root, out):
    """Apply frozen budgets to each independently resolved population."""
    caps = (
        pd.read_csv(out / "caps.csv")
        .query("grid==10 and rule==2")
        .set_index("regime")
        .cap
    )
    allocations = np.loadtxt(
        out / "knot-allocation.csv", skiprows=1, delimiter=","
    )[-1, :3].astype(int)
    design = basis(positions(allocations))
    rows = []
    for variant in ("snaprepeat", "snapdouble", "snap15sec"):
        for f in sorted((root / variant).glob("*/*/snapshot.npz")):
            reg = f.parent.name
            with np.load(f) as z:
                cp = z["checkpoints"]
                err = z["errors"][:, 0, int(caps[reg]) - 1, :3]
                a = z["profiles"]
                profiles = a[a[:, 2] == 16].reshape(-1, 50, 7)
                for c, e, pr in zip(cp, err, profiles, strict=True):
                    y = pr[1:-1, 5]
                    w = np.where(np.isfinite(y), pr[1:-1, 4], 0)
                    y = np.nan_to_num(y)
                    rows.append(
                        [
                            int(f.parent.parent.name),
                            reg,
                            variant,
                            int(c[0]),
                            int(c[1]),
                            caps[reg],
                            *e,
                            *criteria(y, fit(design, y, w), w),
                        ]
                    )
    pd.DataFrame(
        rows,
        columns=(
            "seed regime variant time parcel cap rmse sup coated chi "
            "small middle large"
        ).split(),
    ).to_csv(out / "frozen-refinements.csv", index=False, float_format="%.7g")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    score(a.root, a.out)
