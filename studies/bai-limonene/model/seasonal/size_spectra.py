"""Seasonal differences in absolute particle number per diameter bin."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from particula.particles import get_binned_number_spectrum
from summarize import SEASONS

ROOT = Path(__file__).resolve().parent


def main():
    """Write seasonal sun-minus-dark spectra from time-weighted monthly bins."""
    scratch = Path(os.environ["SEASONAL_TMP"])
    edges = np.arange(3.5, 40.01, 0.25)
    cols = [f"b{i}" for i in range(len(edges) - 1)]
    records = []
    for p in sorted((scratch / "columns").glob("resolved_m??_*.npz")):
        if "_ranked_" not in p.name and "_dark_" not in p.name:
            continue
        with np.load(p) as a:
            d = a["wet_radius_m"][:, -1].astype(float) * 2
            n = a["number_m3"][:, -1].astype(float)
            counts = (
                get_binned_number_spectrum(
                    d, n, edges * 1e-6
                ).bin_number_density
                * 1e-6
            )
            for j, h in enumerate(a["hour_in_week"]):
                records.append(
                    dict(
                        month=a["month"][j],
                        hour=h,
                        profile=a["profile"][j],
                        w=a["w"][j],
                        **dict(zip(cols, counts[j], strict=True)),
                    )
                )
    df = pd.DataFrame(records)
    month = []
    for (m, p, w), g in df.groupby(["month", "profile", "w"]):
        g = g.sort_values("hour")
        h = g.hour.to_numpy()
        wt = np.diff(np.r_[-0.5, (h[:-1] + h[1:]) / 2, 167.5])
        hist = np.average(g[cols], axis=0, weights=wt)
        month.append(
            dict(
                month=m,
                season=SEASONS.get(m, "dark"),
                profile=p,
                w=w,
                **dict(zip(cols, hist, strict=True)),
            )
        )
    x = pd.DataFrame(month)
    if len(x) != 52:
        raise ValueError(f"Expected 52 monthly spectra, found {len(x)}")
    coarse = x.groupby(["season", "profile", "w"], as_index=False)[cols].mean()
    rows = []
    for (p, w), g in coarse.groupby(["profile", "w"]):
        base = g[g.season == "dark"][cols].iloc[0].to_numpy(float)
        for _, r in g[g.season != "dark"].iterrows():
            for j, v in enumerate(r[cols].to_numpy(float)):
                rows.append(
                    dict(
                        season=r.season,
                        profile=p,
                        w=w,
                        lower_um=edges[j],
                        upper_um=edges[j + 1],
                        sun_number_cm3=v,
                        dark_number_cm3=base[j],
                        delta_number_cm3=v - base[j],
                    )
                )
    spectra = pd.DataFrame(rows)
    spectra.to_csv(
        ROOT / "analysis/seasonal-fine-bin-differences.csv.gz",
        index=False,
        float_format="%.8g",
    )


if __name__ == "__main__":
    main()
