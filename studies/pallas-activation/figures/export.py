"""Export the small published figure tables from a completed fit replay."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = (
    Path(__file__).resolve().parents[3]
    / "docs/data/pallas-activation"
)


def export(run, comparator, destination):
    """Preserve the matched dates, dry-volume profiles and illustrated scan."""
    fits = (
        pd.read_csv(run / "fits.csv")
        .query("model == 'two organics'")
        .set_index("date")
    )
    baseline = (
        pd.read_csv(comparator)
        .query("model == 'size-dependent kappa'")
        .set_index("date")
    )
    profiles = pd.read_csv(run / "profiles.csv").set_index("date")
    assert fits.index.equals(baseline.index) and fits.index.equals(
        profiles.index
    )
    assert fits.n.sum() == 1823 and len(fits) == 27
    for size in (50, 120):
        components = [
            f"{group}{size}" for group in ("salt", "sorbitol", "squalane")
        ]
        np.testing.assert_allclose(fits[components].sum(axis=1), 1, atol=1e-9)
    shares = pd.read_csv(run / "shares.csv.gz")
    knots = (
        (shares.share * shares.salt)
        .groupby([shares.date, shares.knot])
        .sum()
        .unstack()
    )
    knots.columns = [f"knot_{size:g}" for size in knots.columns]
    daily = (
        fits[["n", "rmse", "salt50", "salt120", "chi"]]
        .join(baseline.rmse.rename("empirical_rmse"))
        .join(profiles[["chi_lo", "chi_hi", "delta_lo", "delta_hi"]])
        .join(knots)
    )
    destination.mkdir(exist_ok=True, parents=True)
    daily.to_csv(destination / "daily.csv", float_format="%.10g")
    example = json.loads((run / "between.json").read_text())
    date = example["date"]
    curves = pd.read_csv(run / "curves.csv")
    illustrated = curves.loc[curves.date == date].sort_values(["SS", "D"])
    illustrated.to_csv(
        destination / "activation.csv", index=False, float_format="%.10g"
    )
    between = pd.read_csv(run / "between-curves.csv").query(
        "component == 'salt'"
    )
    paired = (
        between.pivot(index="D", columns="label", values="fraction")
        .loc[80:165]
        .copy()
    )
    # Mark measured sizes on the exact model mesh, allowing for log/exp rounding.
    paired["measured_size"] = np.isclose(
        paired.index.to_numpy()[:, None], illustrated.D.unique(), rtol=1e-9
    ).any(axis=1)
    paired.to_csv(destination / "between-sizes.csv", float_format="%.10g")
    pd.read_csv(run / "midpoint-check.csv").to_csv(
        destination / "midpoint-check.csv", index=False, float_format="%.10g"
    )
    pd.read_csv(run / "new-size-test.csv").to_csv(
        destination / "new-size-test.csv", index=False, float_format="%.10g"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--comparator", type=Path)
    parser.add_argument("--out", type=Path, default=DATA)
    args = parser.parse_args()
    export(args.run, args.comparator or args.run / "comparator.csv", args.out)
