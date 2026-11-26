"""Decompose the number median into cohort selection and growth of shared seeds."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from summarize import SEASONS

ROOT = Path(__file__).resolve().parent
TMP = Path(os.environ["SEASONAL_TMP"])


def quantile(d, n, sel):
    vals = d[sel]
    w = n[sel]
    order = np.argsort(vals)
    return np.interp(
        0.5 * w.sum(), np.cumsum(w[order]) - 0.5 * w[order], vals[order]
    )


def main():
    directory = TMP / "columns"
    cache = TMP / "cohort-attribution"
    cache.mkdir(exist_ok=True)
    dark = {}
    for p in directory.glob("resolved_m00_*"):
        with np.load(p) as z:
            key = (z["profile"][0], z["w"][0])
            dark[key] = (
                z["wet_radius_m"][0, -1].astype(float) * 2e6,
                z["number_m3"][0, -1].astype(float) * 1e-6,
            )
    files = []
    for p in sorted(directory.glob("resolved_m??_*")):
        if "_dark_" in p.name or p.suffix != ".npz":
            continue
        dest = cache / (p.stem + ".csv")
        files.append(dest)
        if dest.exists() and dest.stat().st_mtime > p.stat().st_mtime:
            continue
        rows = []
        with np.load(p) as z:
            diam = z["wet_radius_m"][:, -1].astype(float) * 2e6
            nums = z["number_m3"][:, -1].astype(float) * 1e-6
            for j, (d, n) in enumerate(zip(diam, nums)):
                key = (z["profile"][j], z["w"][j])
                d0, n0 = dark[key]
                act0 = (d0 >= 3.5) & (d0 <= 40)
                act = (d >= 3.5) & (d <= 40)
                shared = act & act0
                dark_all = quantile(d0, n0, act0)
                dark_common = quantile(d0, n0, shared)
                lit_common = quantile(d, n, shared)
                lit_all = quantile(d, n, act)
                gone = act0 & ~act
                new = act & ~act0
                tag = np.arange(len(n)) < 768
                rows.append(
                    dict(
                        case_id=z["case_id"][j],
                        month=z["month"][j],
                        hour_in_week=z["hour_in_week"][j],
                        profile=key[0],
                        w=key[1],
                        model=z["model"][j],
                        dark_number_cm3=n0[act0].sum(),
                        sun_number_cm3=n[act].sum(),
                        lost_seed_cm3=n0[gone & tag].sum(),
                        lost_field_cm3=n0[gone & ~tag].sum(),
                        gained_seed_cm3=n[new & tag].sum(),
                        gained_field_cm3=n[new & ~tag].sum(),
                        shared_number_cm3=n[shared].sum(),
                        expansion_term_cm3=n[shared].sum() - n0[shared].sum(),
                        median_delta_um=lit_all - dark_all,
                        exit_selection_um=dark_common - dark_all,
                        shared_growth_um=lit_common - dark_common,
                        entry_selection_um=lit_all - lit_common,
                        dark_median_um=dark_all,
                        sun_median_um=lit_all,
                        common_larger_number_fraction=n[shared & (d > d0)].sum()
                        / n[shared].sum(),
                    )
                )
        pd.DataFrame(rows).to_csv(dest, index=False, float_format="%.10g")
    x = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    assert (
        x.median_delta_um
        - x.exit_selection_um
        - x.shared_growth_um
        - x.entry_selection_um
    ).abs().max() < 1e-7
    err = (
        x.sun_number_cm3
        - x.dark_number_cm3
        + x.lost_seed_cm3
        + x.lost_field_cm3
        - x.gained_seed_cm3
        - x.gained_field_cm3
        - x.expansion_term_cm3
    )
    assert err.abs().max() < 2e-6
    x["season"] = x.month.map(SEASONS)
    keys = ["month", "profile", "w", "model"]
    nums = (
        x.select_dtypes("number")
        .columns.difference(["w", "month", "hour_in_week"])
        .tolist()
    )
    months = []
    for key, g in x.groupby(keys):
        g = g.sort_values("hour_in_week")
        h = g.hour_in_week.to_numpy()
        wt = np.diff(np.r_[-0.5, (h[:-1] + h[1:]) / 2, 167.5])
        row = dict(zip(keys, key))
        row.update(zip(nums, np.average(g[nums], axis=0, weights=wt)))
        months.append(row)
    out = ROOT / "sampling"
    out.mkdir(exist_ok=True)
    x.to_csv(out / "cohort-pairs.csv.gz", index=False, float_format="%.8g")
    pd.DataFrame(months).to_csv(
        out / "cohort-months.csv", index=False, float_format="%.8g"
    )


if __name__ == "__main__":
    main()
