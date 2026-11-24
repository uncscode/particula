# %% Fourth probe and within-date timing
from hashlib import md5
import numpy as np
import pandas as pd
from study import ROOT, inputs, native, window, channels, COUNT


def run():
    f, i = inputs()
    path = ROOT / "input/FMI.CDA.b1.20220914.csv"
    assert (
        md5(path.read_bytes()).hexdigest() == "463d17906aea80fee4406d4663b57b7e"
    )
    df = pd.read_csv(path, encoding="latin1")
    dt = df.iloc[:, :5].copy()
    dt.columns = ["year", "month", "day", "hour", "minute"]
    df.index = pd.to_datetime(dt)
    df = df.loc[~df.index.duplicated(keep=False)]
    ix = f.index.intersection(df.index)
    ix = ix[df.loc[ix, COUNT] > 0]
    a, b, c = f.loc[ix], i.loc[ix], df.loc[ix]
    ef, nf, _ = native(a, "F", "upper")
    ei, ni, _ = native(b, "I", "center")
    u = np.array(
        [
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.9,
            1.1,
            1.3,
            1.5,
            1.8,
            2.1,
            2.4,
            2.8,
            3.3,
            3.8,
            4.5,
            5.2,
            6,
            7,
            8.1,
            9.4,
            10.9,
            12.6,
            14.6,
            16.9,
            19.5,
            22.6,
            26.1,
            30.1,
            34.8,
            40.3,
            46.5,
            53.7,
            62.1,
            71.8,
            82.9,
            95.8,
        ]
    )
    # The first channel's lower limit is unspecified. Exclude it from the
    # common window and report its observed maximum to establish its influence.
    ec = np.r_[0.15, u]
    matrix = channels(c)
    rows = []
    for day in sorted(set(ix.day)):
        for rule in ["geometric", "constant_0.05"]:
            n = matrix * (
                np.diff(np.log10(ec)) if rule == "geometric" else 0.05
            )
            for limit, ok in [
                ("all", ix.day == day),
                (
                    "all_native_le_200",
                    (ix.day == day)
                    & (a[COUNT] <= 200)
                    & (b[COUNT] <= 200)
                    & (c[COUNT] <= 200),
                ),
            ]:
                if not np.any(ok):
                    continue

                def mn(x):
                    return np.mean(x[ok])

                rows.append(
                    dict(
                        day=day,
                        rule=rule,
                        screen=limit,
                        minutes=np.sum(ok),
                        native_median_ratio=np.median(
                            n[ok].sum(1) / c[COUNT][ok]
                        ),
                        first_channel_max=matrix[ok, 0].max(),
                        c_native=mn(c[COUNT]),
                        f_native=mn(a[COUNT]),
                        i_native=mn(b[COUNT]),
                        c_common=mn(n @ window(ec, 6, 45)),
                        f_common=mn(nf @ window(ef, 6, 45)),
                        i_common=mn(ni @ window(ei, 6, 45)),
                    )
                )
    pd.DataFrame(rows).to_csv(ROOT / "vertical-probe.csv", index=False)
    minutes = pd.read_csv(
        ROOT / "minutes-upper.csv.gz", parse_dates=["utc"]
    ).set_index("utc")
    cols = ["f_common", "i_common", "f_native", "i_native"]
    hourly = minutes[cols].resample("h").mean()
    hourly["minutes"] = minutes.f_common.resample("h").count()
    hourly["gap"] = hourly.f_common - hourly.i_common
    hourly[hourly.minutes >= 30].to_csv(ROOT / "hourly.csv", index_label="utc")


if __name__ == "__main__":
    run()
