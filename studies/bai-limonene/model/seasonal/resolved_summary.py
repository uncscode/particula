"""Interpolate fine size-grid cloud states in time and test the interpolation."""

import numpy as np
import pandas as pd
from summarize import OUT, ROOT, METRICS, SEASONS, pairs, read, save, aggregate


def main():
    OUT.mkdir(exist_ok=True)
    fine = read("resolved")
    x = pairs(fine, fine)
    f = pd.read_csv(ROOT / "forcing/forcing.csv")[
        ["month", "hour_in_week", "local_solar_hour", "ghi_wm2"]
    ]
    rows = []
    for (month, profile, w, mode), g in x.groupby(
        ["month", "profile", "w", "model"]
    ):
        if mode != "ranked":
            continue
        g = g.sort_values("hour_in_week")
        h = g.hour_in_week.to_numpy()
        assert h[0] == 0 and h[-1] == 167
        hi = np.arange(168)
        q = pd.DataFrame(
            {
                "hour_in_week": hi,
                "month": month,
                "profile": profile,
                "w": w,
                "model": mode,
                "season": g.season.iloc[0],
            }
        )
        for s in METRICS:
            y = g["delta_" + s].to_numpy()
            q["delta_" + s] = np.interp(hi, h, y)
            q["dark_" + s] = g["dark_" + s].iloc[0]
            q[s] = q["delta_" + s] + q["dark_" + s]
        for s in [
            "duration_fast_reference_h",
            "organic_retained",
            "secondary_reacted_fraction",
        ]:
            q[s] = np.interp(hi, h, g[s])
        q["number_pct"] = 100 * q.delta_activated_cm3 / q.dark_activated_cm3
        q["median_pct"] = 100 * q.delta_median_um / q.dark_median_um
        q["number_above5"] = (q.delta_activated_cm3.abs() > 5).astype(float)
        q["median_above1"] = (q.delta_median_um.abs() > 1).astype(float)
        q["fine_simulated"] = np.isin(hi, h)
        rows.append(q)
    allq = pd.concat(rows, ignore_index=True).merge(
        f, on=["month", "hour_in_week"], validate="many_to_one"
    )
    assert len(allq) == 8064
    allq["solar_hour"] = np.floor(allq.local_solar_hour).astype(int) % 24
    save(allq, "resolved-hourly-cloud.csv.gz")
    for suf, keys in [
        ("monthly", ["month", "season"]),
        ("seasonal", ["season"]),
        ("diurnal", ["season", "solar_hour"]),
    ]:
        save(
            aggregate(allq, keys + ["profile", "w", "model"]),
            "resolved-" + suf + "-cloud.csv",
        )
    july = []
    for (profile, w, mode), g in x[x.month == 7].groupby(
        ["profile", "w", "model"]
    ):
        g = g.sort_values("hour_in_week")
        h = g.hour_in_week.to_numpy()
        wt = np.diff(np.r_[-0.5, (h[:-1] + h[1:]) / 2, 167.5])
        cols = (
            METRICS
            + ["dark_" + v for v in METRICS]
            + ["delta_" + v for v in METRICS]
        )
        data = dict(zip(cols, np.average(g[cols], axis=0, weights=wt)))
        july.append(
            dict(
                profile=profile,
                w=w,
                model=mode,
                actual_time_points=len(g),
                **data,
            )
        )
    save(pd.DataFrame(july), "july-airborne-uncertainty.csv")
    # A pooled size distribution counts droplets across equally weighted hours.
    spectra = pd.concat(
        [
            pd.read_csv(p)
            for p in (ROOT / "results/resolved").glob("*.spectrum.csv.gz")
        ]
    )
    spec = spectra.merge(
        fine[["case_id", "month", "hour_in_week", "model", "profile", "w"]],
        on="case_id",
        validate="one_to_one",
    )
    sizecols = [c for c in spec if c.startswith("N_")]
    srows = []
    for (m, mode, profile, w), g in spec.groupby(
        ["month", "model", "profile", "w"]
    ):
        if mode not in ("dark", "ranked"):
            continue
        g = g.sort_values("hour_in_week")
        h = g.hour_in_week.to_numpy()
        weights = np.diff(np.r_[-0.5, (h[:-1] + h[1:]) / 2, 167.5])
        b = np.average(g[sizecols], axis=0, weights=weights)
        srows.append(
            dict(
                season=SEASONS.get(m, "dark"),
                model=mode,
                profile=profile,
                w=w,
                **dict(zip(sizecols, b)),
            )
        )
    sm = (
        pd.DataFrame(srows)
        .groupby(["season", "model", "profile", "w"], as_index=False)[sizecols]
        .mean()
    )
    save(sm, "resolved-seasonal-spectra.csv")


if __name__ == "__main__":
    main()
