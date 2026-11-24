# %% Independent reconstruction from raw September files and checks of saved summaries
import json
import numpy as np
import pandas as pd
import hashlib

from study import ROOT as root


def load(filename):
    data = pd.read_csv(root / "input" / filename, encoding="latin1")
    key = ["year", "month", "day", "hour", "min"]
    return data.drop_duplicates(key, keep=False).set_index(key)


f = load("FMI.FSSP.b1.20220913.csv")
i = load("OULU.ICEMET.b1.20220914.csv")
paired = f.add_suffix("_f").join(i.add_suffix("_i"), how="inner")
c = "number concentration (cm-3)"
paired = paired[
    (paired[c + "_f"] > 0)
    & (paired[c + "_i"] > 0)
    & (paired.flag_i == 0)
    & (paired["Temp 570m (C)_f"] > 0)
    & paired["FD12P visibility (m)_f"].between(1, 999)
]
assert len(paired) == 4283
# Nominal 6 and 45 are exact boundaries in BOTH instruments; no fractional bins.
ff = sum(paired[f"dN/dlogDp (cm-3) b{k}_f"] * 0.05 for k in range(4, 30))
ii = sum(
    paired[f"dN/dlogDp (cm-3) b{k}_i"] * np.log10((5 + k) / (4 + k))
    for k in range(2, 41)
)
res = ff - ii
day = res.groupby("day").mean()
daily = (
    pd.read_csv(root / "dates.csv").query("geometry=='upper'").set_index("day")
)
np.testing.assert_allclose(day.values, daily.gap, atol=1e-10)
native = paired[c + "_f"] - paired[c + "_i"]
mae_change = (abs(res) - abs(native)).groupby("day").mean().drop(14).mean()
assert round(mae_change, 2) == 8.65
np.testing.assert_array_equal(day.loc[[15, 16]].round(2), [-51.10, 23.29])
matched = pd.read_csv(root / "matched-strata.csv").query(
    "geometry=='upper' and n_width==20 and d_width==2 and min_support==10"
)
weight = matched[["n15", "n16"]].min(axis=1)
assert weight.sum() == 62
np.testing.assert_array_equal(
    np.round(
        [np.average(matched[k], weights=weight) for k in ["gap15", "gap16"]], 2
    ),
    [6.26, 10.77],
)
response = pd.read_csv(root / "response-full.csv")
for conv in ["upper", "center"]:
    x = response[response.geometry == conv].pivot(
        index="day", columns="nscale", values="adjusted_gap"
    )
    np.testing.assert_allclose(x[21], x[161], atol=1e-10)
figure = root / "comparison.png"
assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
report = dict(
    paired_minutes=len(paired),
    upper_day15=float(day[15]),
    upper_day16=float(day[16]),
    equal_date_mae_change=mae_change,
    matched_weight_minutes=int(weight.sum()),
    figure_sha256=hashlib.sha256(figure.read_bytes()).hexdigest(),
    all_checks_passed=True,
)
(root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
