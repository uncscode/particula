"""Score chemistry predictions against the same filter, six-hour window and dry size."""

import numpy as np
import pandas as pd
from response import (
    FIELD,
    HERE,
    MAPPING,
    crossings,
    held,
    read_ccn,
    score_response,
)

KEY = ["filter_id", "utc", "recipe", "closure", "model", "population"]
OBS = FIELD / "observations"


def save(df, name):
    df.to_csv(
        HERE / name,
        index=False,
        float_format="%.9g",
        compression={"method": "gzip", "mtime": 0} if name.endswith(".gz") else None,
    )


def filter_matches():
    f = pd.read_csv(OBS / "filter-windows.csv").query("eligible").copy()
    f["date"] = pd.to_datetime(f.utc, unit="s").dt.strftime("%Y-%m-%d")
    f["held"] = held(f.utc)
    if (f.held != held(f.start_utc)).any() or f.groupby(
        ["filter_id", "clock"]
    ).held.nunique().max() != 1:
        raise ValueError("Filter straddles holdout.")
    return f[["clock", "utc", "filter_id", "cut_nm", "date", "held"]]


def predict():
    chemistry = pd.read_csv(FIELD / "chem/predictions.csv.gz")
    inlet = pd.read_csv(FIELD / "chem/inlet-hydration.csv.gz")
    shared = [
        "filter_id",
        "utc",
        "recipe",
        "closure",
        "model",
        "class_id",
        "draw",
    ]
    chemistry = chemistry.merge(
        inlet[shared + ["dcrit04_wet_nm", "dcrit08_wet_nm"]],
        on=shared,
        validate="one_to_one",
    )
    chemistry["population"] = np.where(
        chemistry.class_id == "uniform", "uniform", "two_class"
    )
    standards = (
        pd.read_csv(OBS / "standards.csv").query("salt=='AS'").set_index("size_nm")
    )
    apparent = pd.read_csv(OBS / "salt-effective-rh.csv").query(
        "size_nm==60 and volume_to_mobility_ratio==1 and statistic=='median_gf'"
    )
    boxes = read_ccn()
    fm = filter_matches()
    times = set(zip(fm.filter_id, fm.utc))
    growth, ccn = [], []
    for keys, part in chemistry.groupby(KEY, sort=False):
        fid, utc, *_ = keys
        if (fid, utc) not in times:
            continue
        weights = part.number_fraction.to_numpy()
        if abs(weights.sum() - 1) > 1e-6:
            raise ValueError(keys)
        for size in [30, 60, 90]:
            gf = part[f"gf85_{size}"].to_numpy()
            mean = weights @ gf
            var = weights @ (gf - mean) ** 2
            record = dict(zip(KEY, keys, strict=True))
            record.update(
                size_nm=size,
                pred_mean=mean,
                pred_sd=np.sqrt(var + standards.loc[size, "median_pdf_sd"] ** 2),
                pred_sd_p90=np.sqrt(var + standards.loc[size, "sd_p90"] ** 2),
                pred_gf95_100=weights @ part.gf95_100,
            )
            if size == 60:
                for row in apparent.itertuples():
                    axis = [76, 78, 80, 82, 84, 85]
                    values = [weights @ part[f"gf{rh}_60"] for rh in axis]
                    record["pred_mean_RH_" + row.salt] = np.interp(
                        100 * row.effective_rh, axis, values
                    )
            growth.append(record)
        for ss, tag in [(0.4, "04"), (0.8, "08")]:
            if (utc, ss) not in boxes:
                continue
            data = boxes[utc, ss]
            dry = part["dcrit" + tag + "_nm"].to_numpy()
            obs_mid, obs_width = crossings(data["obs"], data)
            for mapping, scale, spread in [
                ("physical", 0.0, 0.0),
                ("inlet_hydrated", 0.0, 0.0),
                ("scale_only", MAPPING.log_scale, 0.0),
                ("previous_mapping", MAPPING.log_scale, MAPPING.spread),
            ]:
                axis = (
                    part["dcrit" + tag + "_wet_nm"].to_numpy()
                    if mapping == "inlet_hydrated"
                    else dry
                )
                mid, width, rmse = score_response(axis, weights, data, scale, spread)
                row = dict(zip(KEY, keys, strict=True))
                row.update(
                    S=ss,
                    mapping=mapping,
                    mean_dry_threshold_nm=weights @ dry,
                    obs_mid=obs_mid,
                    obs_width=obs_width,
                    pred_mid=mid,
                    pred_width=width,
                    rmse=rmse,
                )
                ccn.append(row)
    g = pd.DataFrame(growth).merge(fm, on=["filter_id", "utc"])
    obs = pd.read_csv(OBS / "windows.csv")[
        [
            "clock",
            "utc",
            "size_nm",
            "gf_mean",
            "gf_sd",
            "gf_within_sd",
            "n_scans",
        ]
    ]
    g = g.merge(obs, on=["clock", "utc", "size_nm"], validate="many_to_one")
    c = pd.DataFrame(ccn).merge(fm, on=["filter_id", "utc"])
    save(g, "growth.csv.gz")
    save(c, "activation.csv.gz")
    return g, c


def score_by_date(g, c):
    group = [
        "clock",
        "cut_nm",
        "held",
        "recipe",
        "closure",
        "model",
        "population",
    ]
    for col, ref in [
        ("pred_mean", "gf_mean"),
        ("pred_sd", "gf_sd"),
        ("pred_mean_RH_AS", "gf_mean"),
        ("pred_mean_RH_NaCl", "gf_mean"),
    ]:
        g["ae_" + col] = (g[col] - g[ref]).abs()
    cols = [
        "gf_mean",
        "gf_sd",
        "gf_within_sd",
        "pred_mean",
        "pred_sd",
        "pred_gf95_100",
        "pred_mean_RH_AS",
        "pred_mean_RH_NaCl",
        *[k for k in g if k.startswith("ae_")],
    ]
    gd = g.groupby(group + ["size_nm", "date"], as_index=False)[cols].mean()
    save(gd, "growth-by-date.csv.gz")
    save(
        gd.groupby(group + ["size_nm"])[cols].mean().reset_index(),
        "growth-scores.csv",
    )
    primary = g.query(
        "clock=='raw_UTC' and cut_nm==180 and recipe=='low' and closure=='sodium' and population=='uniform'"
    )
    detail = primary.pivot_table(
        index=["date", "size_nm"],
        columns="model",
        values=["gf_mean", "gf_sd", "pred_mean"],
    )
    detail.to_csv(HERE / "three-dates.csv", float_format="%.6f")
    c["ae_mid"] = (c.pred_mid - c.obs_mid).abs()
    c["ae_width"] = (c.pred_width - c.obs_width).abs()
    ccols = [
        "mean_dry_threshold_nm",
        "obs_mid",
        "obs_width",
        "pred_mid",
        "pred_width",
        "rmse",
        "ae_mid",
        "ae_width",
    ]
    cd = c.groupby(group + ["S", "mapping", "date"], as_index=False)[ccols].mean()
    save(cd, "activation-by-date.csv.gz")
    save(
        cd.groupby(group + ["S", "mapping"])[ccols].mean().reset_index(),
        "activation-scores.csv",
    )
    k = pd.read_csv(HERE / "kappa-transfer.csv.gz").merge(
        filter_matches(), on=["clock", "utc", "held"]
    )
    k["ae_mid"] = (k.pred_mid - k.obs_mid).abs()
    k["ae_width"] = (k.pred_width - k.obs_width).abs()
    keys = ["clock", "cut_nm", "held", "S", "case", "mapping"]
    cols = [
        "obs_mid",
        "obs_width",
        "pred_mid",
        "pred_width",
        "rmse",
        "ae_mid",
        "ae_width",
    ]
    kd = k.groupby(keys + ["date"], as_index=False)[cols].mean()
    save(kd.groupby(keys)[cols].mean().reset_index(), "kappa-scores.csv")
    records = []
    compare = gd.query("model in ['full','two_pool'] and population=='uniform'")
    rng = np.random.default_rng(512083)
    for keys, d in compare.groupby(
        ["clock", "cut_nm", "held", "recipe", "closure", "size_nm"]
    ):
        for metric in [
            "ae_pred_mean",
            "ae_pred_mean_RH_AS",
            "ae_pred_mean_RH_NaCl",
            "pred_mean",
            "pred_gf95_100",
        ]:
            p = d.pivot(index="date", columns="model", values=metric).dropna()
            if p.empty:
                continue
            delta = (p.full - p.two_pool).to_numpy()
            ci = (
                np.quantile(
                    rng.choice(delta, (5000, len(delta)), replace=True).mean(axis=1),
                    [0.025, 0.975],
                )
                if len(delta) >= 4
                else [np.nan, np.nan]
            )
            records.append([*keys, metric, len(delta), delta.mean(), *ci])
    save(
        pd.DataFrame(
            records,
            columns=[
                "clock",
                "cut_nm",
                "held",
                "recipe",
                "closure",
                "size_nm",
                "metric",
                "dates",
                "difference",
                "lower",
                "upper",
            ],
        ),
        "paired-contrasts.csv",
    )
    return gd, cd


def draw_check():
    u = pd.read_csv(FIELD / "chem/uncertainty-cuda.csv.gz")
    obs = pd.read_csv(OBS / "windows.csv").query("clock=='raw_UTC' and size_nm==60")
    y = u.merge(
        filter_matches().query("clock=='raw_UTC'"), on=["utc", "filter_id"]
    ).merge(obs[["utc", "gf_mean"]], on="utc", validate="many_to_one")
    y["mae60"] = (y.gf85_60 - y.gf_mean).abs()
    keys = ["cut_nm", "held", "recipe", "draw", "date", "model"]
    a = (
        y.groupby(keys)[["mae60", "gf85_60", "dcrit08_nm"]]
        .mean()
        .groupby(keys[:-2] + ["model"])
        .mean()
        .reset_index()
    )
    rows = []
    for keys, part in a.groupby(["cut_nm", "held", "recipe"]):
        for metric in ["mae60", "gf85_60", "dcrit08_nm"]:
            c = part.pivot(index="draw", columns="model", values=metric)
            for n in [32, 64]:
                for label, series in [
                    ("full", c.full),
                    ("full_minus_two_pool", c.full - c.two_pool),
                ]:
                    rows.append(
                        [
                            *keys,
                            metric,
                            label,
                            n,
                            *np.quantile(series.iloc[:n], [0.05, 0.5, 0.95]),
                        ]
                    )
    save(
        pd.DataFrame(
            rows,
            columns=[
                "cut_nm",
                "held",
                "recipe",
                "metric",
                "contrast",
                "draws",
                "q05",
                "median",
                "q95",
            ],
        ),
        "draw-sensitivity.csv",
    )


if __name__ == "__main__":
    g, c = predict()
    score_by_date(g, c)
    draw_check()
