"""Join strict small-cut windows to chemistry-only allocations and bound molecules."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from recipe import GROUPS, MATRIX, MW, NAMES
from response import kelvin_nm, koehler_threshold
from scipy.optimize import brentq, linprog

HERE = Path(__file__).resolve().parent
CHEM = HERE.parent / "chem"
SALT = ["ammonium", "sulfate", "nitrate", "chloride"]
MW_IONS = np.array([18.038, 96.063, 62.004, 35.453])


def active_component(r):
    # Ideal dilute osmole volume per unit dry volume, neutral molecules counted
    # once and the balancing counterion as one mol per equivalent. The high
    # recipe's soluble unresolved fraction is a glycerol-equivalent surrogate.
    n_ions = np.dot(r[SALT].to_numpy(float), 1 / MW_IONS) + max(r.missing_eq, 0.0)
    n_org = np.dot(r[NAMES[:4]].to_numpy(float), 1 / MW[:4])
    if r.recipe.startswith("high"):
        n_org += max(r.residual - r.amine_ug_m3, 0) / MW[0]
    return 18.01528 * (n_ions + n_org) / r.dry_volume / 0.997


def prepare(mode, output):
    r = pd.read_csv(CHEM / "recipes.csv")
    windows = pd.read_csv(HERE.parent / "observations/filter-windows.csv")
    windows = windows.query("clock=='raw_UTC' and cut_nm==180 and eligible")
    base = r.merge(
        windows[["filter_id", "utc"]],
        on=["filter_id", "utc"],
        validate="many_to_one",
    )
    f = (
        pd.read_csv(CHEM / "filter-fits.csv")
        .query("recipe=='low'")
        .set_index("filter_id")
    )
    maxfr = pd.read_csv(CHEM / "film-bounds.csv").set_index("filter_id")
    rows = []
    for _, r in base.iterrows():
        # Reallocate the existing organic mass among molecular carriers on
        # the independently measured group hyperrectangle; no new mass added.
        c = f.loc[r.filter_id]
        for bound in [0, 2, 3]:
            a = r.copy()
            if bound:
                y = c[["obs_" + g for g in GROUPS]].to_numpy(float)
                sig = c[["sd_" + g for g in GROUPS]].to_numpy(float)
                caps = np.where(y > 0, y + bound * sig, 3 * sig)
                floor = np.maximum(0, y - bound * sig)
                res = linprog(
                    [0, 0, 0, 0, -1],
                    A_ub=np.r_[MATRIX, -MATRIX, np.ones((1, 5))],
                    b_ub=np.r_[caps, -floor, c.ams_om_mean - c.amine],
                    bounds=(0, None),
                )
                if not res.success:
                    raise ValueError("no group-bounded solution")
                a[NAMES] = res.x * r.organic_total / c.ams_om_mean
                a["residual"] = r.organic_total - a[NAMES].sum()
                assert (
                    abs(
                        a.hexadecanol / r.organic_total
                        - maxfr.loc[r.filter_id, f"c16_upper_fraction_{bound}sd"]
                    )
                    < 1e-10
                )
                a["recipe"] = r.recipe + f"_C16_max{bound}"
            a["c16_organic_fraction"] = a.hexadecanol / a.organic_total
            drymass = a.organic_total + a[SALT].sum() + a.missing_eq * 22.99
            a["organic_mass_fraction"] = a.organic_total / drymass
            a["density"] = 1000 * drymass / a.dry_volume
            a["kappa_ideal"] = active_component(a)
            a["kappa08"] = a.kappa_ideal
            a["method"] = "dilute ideal osmoles"
            if a.kappa_ideal <= 0:
                raise ValueError(a)
            rows.append(a)
    d = pd.DataFrame(rows)
    if mode == "anchored":
        t = pd.read_csv(CHEM / "predictions.csv.gz")
        x = t.query(
            "draw==0 and class_id=='uniform' and model=='full' and closure=='sodium'"
        )
        if not len(x):
            raise ValueError(
                "thermodynamic reference selectors: "
                + str(
                    t[["class_id", "model", "closure", "draw"]]
                    .drop_duplicates()
                    .head(30)
                )
            )
        d["base_recipe"] = d.recipe.str.split("_").str[0]
        d = d.merge(
            x[["filter_id", "utc", "recipe", "dcrit08_nm"]].rename(
                columns={"recipe": "base_recipe"}
            ),
            how="left",
            on=["filter_id", "utc", "base_recipe"],
            validate="many_to_one",
        )
        anchor = d[d.recipe == d.base_recipe].copy()
        anchor["direct_k"] = [
            brentq(
                lambda k: koehler_threshold(np.array([k]), s, kelvin_nm(t), 3073)[0]
                - diam,
                0.001,
                4,
            )
            for s, t, diam in zip(
                anchor.ss08_pct,
                anchor.temperature_K,
                anchor.dcrit08_nm,
                strict=True,
            )
        ]
        anchor["offset"] = anchor.direct_k - anchor.kappa_ideal
        d = d.merge(
            anchor[["filter_id", "utc", "recipe", "offset"]].rename(
                columns={"recipe": "base_recipe"}
            ),
            on=["filter_id", "utc", "base_recipe"],
            validate="many_to_one",
        )
        d["kappa08"] = d.kappa_ideal + d.offset
        d["method"] = "0.8% full activity threshold plus differential dilute osmoles"
    d.to_csv(output, index=False, float_format="%.12g")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["ideal", "anchored"])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.mode, args.output)
