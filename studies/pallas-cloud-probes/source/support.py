# %% Sensitivity to a contiguous sampling period among independently matched cloud regimes
import pandas as pd
import numpy as np
from study import ROOT, COUNT, ED


def run():
    obs = pd.read_csv(ROOT / "input/FMI.CDA.b1.20220914.csv", encoding="latin1")
    cols = obs.iloc[:, :5].copy()
    cols.columns = ["year", "month", "day", "hour", "minute"]
    obs.index = pd.to_datetime(cols)
    obs = obs[~obs.index.duplicated(keep=False)]
    m = pd.read_csv(
        ROOT / "minutes-upper.csv.gz", parse_dates=["utc"]
    ).set_index("utc")
    m = m.join(obs[[COUNT, ED]], how="inner")
    m = m.loc[
        m.index.day.isin([15, 16])
        & m[COUNT].between(0, 200, inclusive="right")
        & (m[ED] > 0)
    ].copy()
    m["day"] = m.index.day
    m["n"] = (m[COUNT] // 20) * 20
    m["size"] = (m[ED] // 2) * 2
    m["gap"] = m.f_common - m.i_common
    m["hour"] = m.index.floor("h")

    def score(data):
        g = (
            data.groupby(["n", "size", "day"])
            .gap.agg(["count", "mean"])
            .unstack()
            .dropna()
        )
        g = g[g["count"].min(axis=1) >= 10]
        w = g["count"].min(axis=1)
        return g, dict(
            strata=len(g),
            weight=w.sum(),
            **{
                f"gap{d}": np.average(g["mean"][d], weights=w)
                if len(g)
                else np.nan
                for d in [15, 16]
            },
        )

    g, full = score(m)
    primary = m.set_index(["n", "size"]).loc[g.index]
    hrs = sorted(primary.hour.unique())
    rows = [dict(removed_utc_hour="none", **full)]
    for hour in hrs:
        _, record = score(m[m.hour != hour])
        rows.append(dict(removed_utc_hour=str(hour), **record))
    pd.DataFrame(rows).to_csv(ROOT / "matched-hours.csv", index=False)
    primary.groupby(["day", "hour"]).size().rename("minutes").to_csv(
        ROOT / "matched-coverage.csv"
    )


if __name__ == "__main__":
    run()
