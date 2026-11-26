"""Paired cloud diagnostics with equal-hour, local-apparent-solar-time summaries."""

from pathlib import Path
import os
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("BAI_SEASONAL_ANALYSIS", ROOT / "analysis"))
SEASONS = {
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
    12: "DJF",
}
METRICS = [
    "activated_cm3",
    "median_um",
    "p10_um",
    "p90_um",
    "width_um",
    "re_all_um",
    "lwp_gm2",
]


def read(label):
    return pd.concat(
        [
            pd.read_csv(p)
            for p in sorted((ROOT / "results" / label).glob("m*.csv"))
        ],
        ignore_index=True,
    )


def pairs(raw, ref):
    raw = raw[raw.month > 0].copy()
    ref = ref[ref.month == 0].copy()
    for df in [raw, ref]:
        df["width_um"] = df.p90_um - df.p10_um
    ref = ref[["profile", "w"] + METRICS].rename(
        columns={s: "dark_" + s for s in METRICS}
    )
    raw = raw.merge(ref, on=["profile", "w"], validate="many_to_one")
    for s in METRICS:
        raw["delta_" + s] = raw[s] - raw["dark_" + s]
    raw["number_pct"] = 100 * raw.delta_activated_cm3 / raw.dark_activated_cm3
    raw["median_pct"] = 100 * raw.delta_median_um / raw.dark_median_um
    raw["number_above5"] = (raw.delta_activated_cm3.abs() > 5).astype(float)
    raw["median_above1"] = (raw.delta_median_um.abs() > 1).astype(float)
    raw["season"] = raw.month.map(SEASONS)
    return raw


def save(df, name):
    df.to_csv(OUT / name, index=False, float_format="%.8g")


def aggregate(x, keys):
    values = (
        METRICS
        + ["dark_" + s for s in METRICS]
        + ["delta_" + s for s in METRICS]
    )
    values += [
        "number_pct",
        "median_pct",
        "number_above5",
        "median_above1",
        "duration_fast_reference_h",
        "organic_retained",
        "secondary_reacted_fraction",
    ]
    group = x.groupby(keys, as_index=False, sort=True)
    tab = group[values].mean()
    tab = tab.merge(group.size().rename(columns={"size": "samples"}), on=keys)
    for v in ["delta_activated_cm3", "delta_median_um"]:
        qs = (
            x.groupby(keys)[v]
            .quantile([0.1, 0.9])
            .unstack()
            .rename(columns={0.1: v + "_p10", 0.9: v + "_p90"})
        )
        tab = tab.merge(qs, on=keys)
    return tab
