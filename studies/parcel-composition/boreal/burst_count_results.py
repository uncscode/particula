"""Check per-history birth counts and their equally weighted mixture."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from burst_counts import DRAWS, DT, FINE, FIVE_SECOND_DRAWS, FRESH, SEEDS
from scipy.stats import poisson

METHODS = ("full", "independent", "redraw")


def load_trials(root, fine=False):
    """Verify every history, draw and parcel before combining the parcels."""
    rows = []
    for seed in SEEDS + FRESH:
        frame = pd.read_csv(root / f"{seed}.csv")
        steps = [5] if fine else ([DT, 25] if seed in FINE else [DT])
        draws = FIVE_SECOND_DRAWS if fine else DRAWS
        expected = pd.MultiIndex.from_product(
            [steps, range(draws), (0, 1)],
            names=["dt", "draw", "parcel"],
        )
        pd.testing.assert_index_equal(
            frame.set_index(expected.names).index.sort_values(),
            expected.sort_values(),
        )
        np.testing.assert_array_equal(frame.seed.unique(), [seed])
        measured = frame[["original", *METHODS, "full_mean", "redraw_mean"]]
        if not np.isfinite(measured).all().all() or (measured < 0).any().any():
            raise ValueError(f"Invalid count or rate for history {seed}")
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def cdf_errors(samples, laws):
    """Largest absolute CDF differences for each history and their mixture."""
    width = laws.shape[1]
    empirical = np.stack(
        [
            np.cumsum(
                np.bincount(np.minimum(x, width), minlength=width + 1)[:width]
            )
            / len(x)
            for x in samples
        ]
    )
    diff = empirical - laws
    return np.abs(diff).max(axis=1), np.abs(diff.mean(axis=0)).max()


def distances(frame):
    """Calibrate discrete CDF tests using the actual equal-history design."""
    ids = np.sort(frame.seed.unique())
    rates = frame.groupby("seed").full_mean.first().loc[ids].to_numpy()
    observed = {
        name: frame.pivot(index="seed", columns="draw", values=name)
        .loc[ids]
        .to_numpy()
        for name in METHODS
    }
    limit = int(
        max(
            poisson.ppf(1 - 1e-13, rates).max(),
            *(x.max() for x in observed.values()),
        )
    )
    grid = np.arange(limit + 1)
    laws = poisson.cdf(grid, rates[:, None])
    rng = np.random.default_rng(923187)
    null = []
    for _ in range(1999):
        sample = rng.poisson(rates[:, None], (len(rates), DRAWS))
        conditional, mixture = cdf_errors(sample, laws)
        null.append([conditional.max(), mixture])
    null = np.array(null)
    rows = []
    for name, arr in observed.items():
        cond, mix = cdf_errors(arr, laws)
        for i, (metric, value) in enumerate(
            [("worst conditional CDF", cond.max()), ("mixture CDF", mix)]
        ):
            rows.append(
                dict(
                    method=name,
                    metric=metric,
                    distance=value,
                    null95=np.quantile(null[:, i], 0.95),
                    mc_p=(1 + (null[:, i] >= value).sum()) / (1 + len(null)),
                )
            )
    return pd.DataFrame(rows)


def _two_sample_scores(left, pooled):
    histories, draws = left.shape
    width = pooled.shape[1]
    lhs = np.stack([np.bincount(x, minlength=width) for x in left])
    diff = np.cumsum(2 * lhs - pooled, axis=1)
    return np.array(
        [
            np.abs(diff).max() / draws,
            np.abs(diff.sum(axis=0)).max() / (draws * histories),
        ]
    )


def fine_distances(frame):
    """Compare independent samples by permuting labels within each history."""
    rows = []
    rng = np.random.default_rng(502197)
    for method in ("full", "redraw"):
        first = frame.pivot(
            index="seed", columns="draw", values=method
        ).to_numpy()
        control = frame.pivot(
            index="seed", columns="draw", values="independent"
        ).to_numpy()
        joined = np.column_stack((first, control))
        draws = first.shape[1]
        width = int(joined.max()) + 1
        pooled = np.stack([np.bincount(x, minlength=width) for x in joined])

        observed = _two_sample_scores(first, pooled)
        null = np.array(
            [
                _two_sample_scores(
                    rng.permuted(joined, axis=1)[:, :draws], pooled
                )
                for _ in range(1999)
            ]
        )
        for i, metric in enumerate(("worst conditional CDF", "mixture CDF")):
            rows.append(
                dict(
                    method=method,
                    metric=metric,
                    distance=observed[i],
                    null95=np.quantile(null[:, i], 0.95),
                    mc_p=(1 + (null[:, i] >= observed[i]).sum())
                    / (len(null) + 1),
                )
            )
    return pd.DataFrame(rows)


def metrics(g):
    """Report spread, tails, paired errors and independent coverage."""
    full, independent, redraw = [g[k].to_numpy() for k in METHODS]
    bounds = np.quantile(redraw, [0.05, 0.95], method="inverted_cdf")
    record = dict(
        original=g.original.iloc[0],
        expected=g.full_mean.mean(),
        redraw_expected=g.redraw_mean.mean(),
        max_rate_shift=np.max(np.abs(g.redraw_mean - g.full_mean)),
        paired_rms=np.sqrt(np.mean((redraw - full) ** 2)),
        full_full_rms=np.sqrt(np.mean((full - independent) ** 2)),
        paired_exact=np.mean(redraw == full),
        interval_coverage=np.mean(
            (independent >= bounds[0]) & (independent <= bounds[1])
        ),
    )
    for k in METHODS:
        x = g[k].to_numpy()
        record[k + "_mean"] = x.mean()
        record[k + "_sd"] = x.std(ddof=1)
        record[k + "_q05"], record[k + "_q95"] = np.quantile(
            x, [0.05, 0.95], method="inverted_cdf"
        )
    return pd.Series(record)


def plot(g, stats, sub, image):
    """Show conditional spreads, an original count and step sensitivity."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    ax = axes[0]
    s = stats[stats.dt == DT].sort_values("expected")
    for key, color, dx, label in [
        ("independent", "#43495e", -0.12, "Full repeat"),
        ("redraw", "#087c78", 0.12, "Redraw"),
    ]:
        x = np.arange(len(s)) + dx
        ax.vlines(x, s[key + "_q05"], s[key + "_q95"], colors=color, lw=2)
        ax.scatter(x, s[key + "_mean"], s=10, color=color, label=label)
    ax.set(
        xlabel="Histories ordered by expected count",
        ylabel="Births in 75 s",
        title="Within-history 5–95% range",
        xticks=[0, 4, 9, 15],
        xticklabels=[1, 5, 10, 16],
    )
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    ex = g[(g.dt == DT) & (g.seed == 10723)]
    rate = ex.full_mean.iloc[0]
    x = np.arange(
        min(ex.original.iloc[0], ex[list(METHODS)].min().min()) - 5,
        ex[list(METHODS)].max().max() + 6,
    )
    ax.plot(x, poisson.cdf(x, rate), color="#bd6924", label="Model law")
    for k, c, label in [
        ("independent", "#43495e", "Full repeat"),
        ("redraw", "#087c78", "Redraw"),
    ]:
        ax.step(
            x,
            np.searchsorted(np.sort(ex[k]), x, side="right") / len(ex),
            where="post",
            label=label,
            color=c,
        )
    ax.axvline(ex.original.iloc[0], ls=":", color="#222222", label="Original")
    ax.set(
        xlabel="Births in 75 s",
        ylabel="Cumulative probability",
        title="History 10723",
    )
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    for seed, c in zip(
        (FINE[0], FRESH[1]), ("#753d82", "#416e91"), strict=True
    ):
        q = (
            sub[(sub.method == "full") & (sub.seed == seed)]
            .groupby(["dt", "draw"])
            .expected.sum()
            .groupby("dt")
            .mean()
        )
        ax.plot(q.index, q / q.loc[75] * 100, "o-", color=c, label=str(seed))
    ax.set(
        xscale="log",
        xticks=[0.5, 1, 5, 15, 25, 75],
        xticklabels=["0.5", "1", "5", "15", "25", "75"],
        xlabel="Integration step (s)",
        ylabel="Expected births (% of 75 s step)",
        title="Pulse-onset time resolution",
    )
    ax.legend(frameon=False, title="History", fontsize=8)
    fig.savefig(image, dpi=170)
    plt.close(fig)


def main():
    """Write replay evidence and reproducible distribution diagnostics."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("image", type=Path)
    a = p.parse_args()
    data = pd.concat(
        [load_trials(a.root), load_trials(a.root / "five-second", fine=True)],
        ignore_index=True,
    )
    total = (
        data.groupby(["seed", "dt", "draw"])[
            ["original", *METHODS, "full_mean", "redraw_mean"]
        ]
        .sum()
        .reset_index()
    )
    coarse = total[total.dt == DT]
    np.testing.assert_equal(coarse.groupby("seed").full_mean.nunique().max(), 1)
    np.testing.assert_array_equal(coarse.full_mean, coarse.redraw_mean)
    prior = pd.read_csv(a.output / "redraw-burst-leads.csv.gz")
    prior = (
        prior[
            (prior.n == 50000) & (prior.lead == DT) & (prior.recipe == "full")
        ]
        .groupby("seed")
        .ref_births.first()
    )
    np.testing.assert_array_equal(
        coarse.groupby("seed").original.first().loc[prior.index], prior
    )
    stats = (
        total.groupby(["seed", "dt"])
        .apply(metrics, include_groups=False)
        .reset_index()
    )
    groups = {
        "all histories": coarse,
        "new histories": coarse[coarse.seed.isin(FRESH)],
        "active histories": coarse[coarse.full_mean > 0],
    }
    scores = pd.concat(
        [
            distances(group).assign(histories=name)
            for name, group in groups.items()
        ],
        ignore_index=True,
    )
    scores = scores.assign(dt=DT, reference="Poisson model")
    fine = total[total.dt == 5]
    active = fine.groupby("seed").full_mean.transform("max") > 0
    for label, group in (
        ("all histories", fine),
        ("active histories", fine[active]),
    ):
        additional = fine_distances(group).assign(
            dt=5, histories=label, reference="independent full repeat"
        )
        scores = pd.concat([scores, additional], ignore_index=True)
    scores.to_csv(
        a.output / "burst-count-cdf.csv", index=False, float_format="%.10g"
    )
    stats.to_csv(
        a.output / "burst-count-histories.csv",
        index=False,
        float_format="%.10g",
    )
    data.to_csv(
        a.output / "burst-count-draws.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
        float_format="%.15g",
    )
    sub = pd.read_csv(a.output / "burst-substeps.csv.gz")
    plot(total, stats, sub, a.image)


if __name__ == "__main__":
    main()
