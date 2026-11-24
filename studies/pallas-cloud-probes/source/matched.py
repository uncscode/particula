# %% Standardize by measurements from a third instrument, independent of the count contrast
import pandas as pd
import numpy as np
from study import ROOT, COUNT, ED


def run():
    path = ROOT / "input/FMI.CDA.b1.20220914.csv"
    c = pd.read_csv(path, encoding="latin1")
    dt = c.iloc[:, :5].copy()
    dt.columns = ["year", "month", "day", "hour", "minute"]
    c.index = pd.to_datetime(dt)
    c = c.loc[~c.index.duplicated(keep=False)]
    results = []
    strata = []
    for conv in ["upper", "center"]:
        counts = pd.read_csv(
            ROOT / f"minutes-{conv}.csv.gz", parse_dates=["utc"]
        ).set_index("utc")
        x = counts.join(c[[COUNT, ED]], how="inner")
        x = x.loc[
            x.index.day.isin([15, 16])
            & x[COUNT].between(0, 200, inclusive="right")
            & (x[ED] > 0)
        ].copy()
        x["day"] = x.index.day
        x["gap"] = x.f_common - x.i_common
        for width_n, width_d, min_n in [
            (20, 0, 10),
            (20, 2, 10),
            (10, 1, 10),
            (40, 4, 10),
            (20, 2, 30),
        ]:
            x["n_band"] = (x[COUNT] // width_n) * width_n
            x["d_band"] = (x[ED] // width_d) * width_d if width_d else 0.0
            t = (
                x.groupby(["n_band", "d_band", "day"])
                .gap.agg(["mean", "count"])
                .unstack("day")
            )
            t = t.dropna()
            t = t[t["count"].min(axis=1) >= min_n]
            if t.empty:
                continue
            w = t["count"].min(axis=1)
            g15 = np.average(t["mean"][15], weights=w)
            g16 = np.average(t["mean"][16], weights=w)
            results.append(
                dict(
                    geometry=conv,
                    n_width=width_n,
                    d_width=width_d,
                    min_support=min_n,
                    strata=len(t),
                    raw15=len(x[x.day == 15]),
                    raw16=len(x[x.day == 16]),
                    eligible15=int(t["count"][15].sum()),
                    eligible16=int(t["count"][16].sum()),
                    effective_minutes_per_date=w.sum(),
                    gap15=g15,
                    gap16=g16,
                    change=g16 - g15,
                )
            )
            for (n, d), row in t.iterrows():
                strata.append(
                    dict(
                        geometry=conv,
                        n_width=width_n,
                        d_width=width_d,
                        min_support=min_n,
                        cda_N_lower=n,
                        cda_ED_lower=d,
                        n15=row["count", 15],
                        n16=row["count", 16],
                        gap15=row["mean", 15],
                        gap16=row["mean", 16],
                    )
                )
    pd.DataFrame(results).to_csv(ROOT / "matched-regimes.csv", index=False)
    pd.DataFrame(strata).to_csv(ROOT / "matched-strata.csv", index=False)


if __name__ == "__main__":
    run()
