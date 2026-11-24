"""Post hoc negative control: a homogeneous solution mixed with a zero-uptake number class."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import ndtr

BASE = Path(__file__).resolve().parent
APRIL = int(np.datetime64("2023-04-01").astype(int))


def held(t):
    return (np.asarray(t) // 86400 - APRIL) % 3 == 0


def read_data():
    field = pd.read_csv(BASE / "windows.csv").query("clock=='raw_UTC' and n_sizes==3")
    filters = pd.read_csv(BASE / "filter-windows.csv").query("clock=='raw_UTC'").copy()
    filters["held"] = held(filters.utc)
    filters["filter_held"] = held(filters.start_utc)
    # Withhold or train complete filters; remove any filter whose published
    # six-hour model inputs include a date from the opposite split.
    recipes = pd.read_csv(BASE.parent / "chem/recipes.csv").drop_duplicates(
        ["filter_id", "utc"]
    )
    recipes["split_cross"] = held(recipes.utc) != held(recipes.start)
    invalid = recipes.groupby("filter_id").split_cross.any()
    filters = filters[~filters.filter_id.map(invalid)]
    assert (filters.held == filters.filter_held).all()
    field = field.merge(filters[["utc", "filter_id", "cut_nm", "held"]], on="utc")
    predicted = pd.read_csv(BASE.parent / "chem/predictions.csv.gz").query(
        "draw==0 and model=='full' and closure=='sodium' and class_id=='uniform'"
    )
    small = predicted.melt(
        id_vars=["filter_id", "utc", "recipe"],
        value_vars=["gf85_30", "gf85_60", "gf85_90"],
        value_name="active_gf",
    )
    small["size_nm"] = small.variable.str.split("_").str[-1].astype(int)
    field = field.merge(
        small.drop(columns="variable"),
        on=["filter_id", "utc", "size_nm"],
        validate="one_to_many",
    )
    tails = []
    for size in (30, 60, 90):
        with np.load(BASE / f"pdf{size}.npz") as z:
            g = z["gf"]
            mask = g <= 1.1
            assert np.isclose(g[mask][-1], 1.1)
            tails.extend(
                (t, size, v)
                for t, v in zip(
                    z["utc"][z["clock"] == "raw_UTC"],
                    np.trapezoid(
                        z["density"][z["clock"] == "raw_UTC"][:, mask], g[mask], axis=1
                    ),
                )
            )
    return field.merge(
        pd.DataFrame(tails, columns=["utc", "size_nm", "gf_below_110"]),
        on=["utc", "size_nm"],
    )


def main():
    field = read_data()
    train = field.query(
        "not held and size_nm==60 and cut_nm==1000 and recipe=='low'"
    ).copy()
    train["w"] = 1 / train.groupby("filter_id")["utc"].transform("count")
    x = train.active_gf - 1
    y = train.active_gf - train.gf_mean
    p = np.clip((train.w * x * y).sum() / (train.w * x * x).sum(), 0, 1)
    train["numerator"] = train.w * x * y
    train["denominator"] = train.w * x**2
    summands = train.groupby("filter_id")[["numerator", "denominator"]].sum().to_numpy()
    rng = np.random.default_rng(230622)
    idx = rng.integers(len(summands), size=(4096, len(summands)))
    sample = summands[idx].sum(axis=1)
    prob = np.clip(sample[:, 0] / sample[:, 1], 0, 1)
    salts = pd.read_csv(BASE / "standards.csv")
    sigmas = salts.groupby("size_nm").median_pdf_sd.mean()
    field["blur"] = field.size_nm.map(sigmas)
    cases = []
    for label, fraction in [("single", 0), ("partial", p)]:
        one = field.copy()
        one["model"] = label
        one["mean_pred"] = fraction + (1 - fraction) * one.active_gf
        one["sd_pred"] = np.sqrt(
            one.blur**2 + fraction * (1 - fraction) * (one.active_gf - 1) ** 2
        )
        one["tail_pred"] = fraction * ndtr(0.1 / one.blur) + (1 - fraction) * ndtr(
            (1.10 - one.active_gf) / one.blur
        )
        cases.append(one)
    cases = pd.concat(cases, ignore_index=True)
    keys = ["cut_nm", "held", "recipe", "size_nm", "model"]
    cols = [
        "filter_id",
        "utc",
        "cut_nm",
        "held",
        "recipe",
        "size_nm",
        "model",
        "active_gf",
        "blur",
        "gf_mean",
        "mean_pred",
        "gf_sd",
        "sd_pred",
        "gf_below_110",
        "tail_pred",
    ]
    cases[cols].to_csv(
        BASE / "partial-cases.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    rows = []
    for group, sel in cases.groupby(keys):
        item = dict(zip(keys, group))
        item.update(filters=sel.filter_id.nunique(), windows=sel.utc.nunique())
        for short, ob, pr in [
            ("mean", "gf_mean", "mean_pred"),
            ("sd", "gf_sd", "sd_pred"),
            ("tail", "gf_below_110", "tail_pred"),
        ]:
            s = sel.copy()
            s["abs"] = (s[pr] - s[ob]).abs()
            s["signed"] = s[pr] - s[ob]
            m = s.groupby("filter_id")[[ob, pr, "abs", "signed"]].mean().mean()
            for k, v in zip(["observed", "predicted", "mae", "bias"], m):
                item[f"{short}_{k}"] = v
        rows.append(item)
    pd.DataFrame(rows).to_csv(BASE / "partial-scores.csv", index=False)
    intervals = []
    # Jointly resample the training filters that determine p and held filters
    # that determine the held response, retaining every time cell of a filter.
    for size in (30, 60, 90):
        test = field.query("held and cut_nm==1000 and recipe=='low' and size_nm==@size")
        day_stats = []
        for _, day in test.groupby("filter_id"):
            a = day.active_gf.to_numpy()[None, :]
            b = day.blur.to_numpy()[None, :]
            f = prob[:, None]
            mu = f + (1 - f) * a
            sd = np.sqrt(b**2 + f * (1 - f) * (a - 1) ** 2)
            tail = f * ndtr(0.1 / b) + (1 - f) * ndtr((1.1 - a) / b)
            day_stats.append(
                np.stack(
                    [
                        (mu - day.gf_mean.to_numpy()).mean(axis=1),
                        (sd - day.gf_sd.to_numpy()).mean(axis=1),
                        (tail - day.gf_below_110.to_numpy()).mean(axis=1),
                    ],
                    axis=-1,
                )
            )
        span = np.stack(day_stats, axis=1)
        indexes = rng.integers(span.shape[1], size=span.shape[:2])
        dist = span[np.arange(len(span))[:, None], indexes].mean(axis=1)
        for j, metric in enumerate(["mean_bias", "sd_bias", "tail_bias"]):
            lo, hi = np.quantile(dist[:, j], [0.025, 0.975])
            intervals.append(dict(size_nm=size, metric=metric, lo95=lo, hi95=hi))
    pd.DataFrame(intervals).to_csv(BASE / "partial-held-intervals.csv", index=False)
    base = dict(
        dry_number_fraction=p,
        training_filters=len(summands),
        training_windows=len(train),
        bootstrap_replicates=prob.shape[0],
        bootstrap_p05_p95=np.quantile(prob, [0.05, 0.95]).tolist(),
        train_filter_ids=sorted(train.filter_id.unique()),
        held_pm1_ids=sorted(field.query("held and cut_nm==1000").filter_id.unique()),
    )
    (BASE / "partial-fit.json").write_text(json.dumps(base, indent=2) + "\n")


if __name__ == "__main__":
    main()
