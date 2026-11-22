"""Rebuild the factorial calibration ranges for Figures 2 and 3."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scripts.antarctic_ccn.analysis import FIT, date_weights
from scripts.antarctic_ccn.model import BANDS, DATA, make_thresholds, responses


def candidates(salt, organics, water, observed, train):
    """Fit one, two or four coating fractions using the training dates."""
    target = (observed - salt)[train][:, FIT].ravel()
    for material, steps, counts in [
        ("film", organics, [1, 2, 4]),
        ("water", water, [1, 2]),
    ]:
        for n in counts:
            groups = np.stack([g.sum(0) for g in np.array_split(steps, n)])
            design = groups[:, train][:, :, FIT].reshape(n, -1).T
            fit = lsq_linear(design, target, bounds=(0, 1), tol=1e-12)
            name = material + "_" + {1: "common", 2: "split", 4: "four"}[n]
            yield name, salt + np.einsum("b,bhs->hs", fit.x, groups)


def sweep(output):
    """Score all instrument settings and write the retained ranges."""
    with np.load(DATA / "inputs.npz") as archive:
        data = dict(archive)
    train, test = [a[0].astype(bool) for a in date_weights(data["dates"])]
    observed = data["measured"]
    sc = data["sc"]
    settings = list(product([-0.05, 0, 0.05], [-0.025, 0, 0.025]))
    ss = np.unique(
        np.round(
            [s * (1 + gain) + offset for gain, offset in settings for s in sc],
            8,
        )
    )
    scores = []
    bands = []
    case = 0
    for gf, rh in product([0.98, 1, 1.02], [0.896, 0.9, 0.904]):
        cuts = make_thresholds(data, ss, gf, rh)
        for diameter in [0.95, 1, 1.05]:
            tables = responses(data, cuts, diameter)
            for gain, offset in settings:
                ix = np.searchsorted(ss, np.round(sc * (1 + gain) + offset, 8))
                for number in [0.9, 1, 1.1]:
                    xu, k, salt, organic, water, _ = [
                        a[..., ix] * number for a in tables
                    ]
                    models = [
                        ("Xu", xu),
                        ("distributed", k),
                        ("salt", salt),
                        ("maximum", salt + organic.sum(0)),
                    ]
                    models.extend(
                        candidates(salt, organic, water, observed, train)
                    )
                    for name, pred in models:
                        loss = np.sqrt(
                            np.mean((pred - observed)[train][:, FIT] ** 2)
                        )
                        uplift = (pred - salt)[test, 1].mean()
                        residual = (observed - pred)[test, 1].mean()
                        scores.append(
                            (
                                case,
                                name,
                                loss,
                                uplift,
                                residual,
                                gf,
                                rh,
                                diameter,
                                number,
                                gain,
                                offset,
                            )
                        )
                    inc = np.diff(organic[:, test].mean(1))
                    for i, (lo, hi) in enumerate(BANDS):
                        bands.append((case, lo, hi, *inc[i, :2]))
                    case += 1
    frame = pd.DataFrame(
        scores,
        columns=[
            "case",
            "model",
            "train_rmse",
            "uplift",
            "unexplained",
            "gf_factor",
            "rh",
            "diameter",
            "number",
            "ss_gain",
            "ss_offset",
        ],
    )
    accepted = frame[frame.train_rmse <= frame.train_rmse.min() + 1]
    common = accepted[accepted.model == "film_common"]
    selected = pd.DataFrame(
        bands, columns=["case", "lo_nm", "hi_nm", "rise24", "rise46"]
    ).merge(common[["case"]])
    env = selected.groupby(["lo_nm", "hi_nm"])[["rise24", "rise46"]].agg(
        ["min", "max"]
    )
    env.columns = ["rise24_lo", "rise24_hi", "rise46_lo", "rise46_hi"]
    output.mkdir(parents=True, exist_ok=True)
    env.to_csv(output / "calibration-ranges.csv")
    frame.to_csv(output / "calibration-cases.csv", index=False)
    effects = pd.read_csv(DATA / "attribution-ranges.csv")
    for i, r in effects.iterrows():
        v = accepted.loc[accepted.model == r.model, r.outcome]
        effects.loc[i, ["compatible_lo", "compatible_hi"]] = [v.min(), v.max()]
    effects.to_csv(output / "attribution-ranges.csv", index=False)
    print(
        f"{case} scenarios, {len(common)} common-film settings, "
        f"cutoff {frame.train_rmse.min() + 1:.5f}"
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("output", type=Path)
    sweep(cli.parse_args().output)
