"""Publish the fixed and moving grid comparison from matched parcel trials."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta

ORDER = ("fixed", "density", "fixed-pinned", "pinned")
NAMES = ("Fixed log", "Density", "Fixed + cutoffs", "Density + cutoffs")
COLORS = ("#495d64", "#397c7b", "#ae702f", "#573f76")
BULK = ["activation", "cloud", "chemistry", "absorption", "scattering"]


def assemble(root, out):
    """Verify trial keys and sizes before publishing the paired measurements."""
    edges = pd.concat(
        [pd.read_csv(p) for p in sorted(root.glob("*-edges.csv.gz"))]
    )
    pairs = pd.concat(
        [pd.read_csv(p) for p in sorted(root.glob("*-pairs.csv.gz"))]
    )
    if pairs.seed.nunique() != 16 or set(pairs.method) != set(ORDER):
        raise ValueError("Incomplete ensemble")
    for (seed, t, arm), g in pairs.groupby(["seed", "start", "method"]):
        n = 6 if t == 9000 else 192
        expected = pd.MultiIndex.from_product(
            [range(n), [0, 1800 if t == 9000 else 75]], names=["trial", "lead"]
        )
        if (
            not pd.MultiIndex.from_frame(g[["trial", "lead"]])
            .sort_values()
            .equals(expected)
        ):
            raise ValueError(f"Missing trials for {seed}, {t}, {arm}")
        if not (
            g[BULK + ["nucleation"]].notna().all(axis=1) == (g.trial < 24)
        ).all():
            raise ValueError("Missing process scores")
    if len(pairs.groupby(["seed", "start", "method"])) != 16 * 3 * 4:
        raise ValueError("Missing design cells")
    if (
        len(edges) != 16 * 3 * 4 * 2 * 31
        or edges.duplicated(
            ["seed", "start", "method", "parcel", "edge_index"]
        ).any()
    ):
        raise ValueError("Missing edge vectors")
    pairs.to_csv(
        out / "adaptive-pairs.csv.gz", index=False, float_format="%.12g"
    )
    edges.to_csv(
        out / "adaptive-edges.csv.gz", index=False, float_format="%.13g"
    )
    return pairs, edges


def qualification(pairs):
    """Simultaneous two-sided binomial results for the short-window counter."""
    alpha = 0.05 / (2 * 16 * 4)
    rows = []
    short = pairs[pairs.start != 9000]
    for (seed, method), g in short.groupby(["seed", "method"]):
        worst = g.groupby("trial").counter_pct.max()
        n = len(worst)
        k = (worst > 10).sum()
        lo = beta.ppf(alpha, k, n - k + 1) if k else 0
        hi = beta.isf(alpha, k + 1, n - k) if k < n else 1
        verdict = (
            "qualified"
            if hi <= 0.05
            else "exceeds"
            if lo > 0.05
            else "unresolved"
        )
        rows.append(
            dict(
                seed=seed,
                method=method,
                trials=n,
                exceedances=k,
                max_error_pct=worst.max(),
                lower=lo,
                upper=hi,
                status=verdict,
            )
        )
    return pd.DataFrame(rows)


def blind_channels(table):
    """Summarize ordinary dry-size channels outside the prescribed passband."""
    take = (table.full_count >= 100) & (
        (table.hi_nm <= 150) | (table.lo_nm >= 160)
    )
    x = table.loc[take].copy()
    x["bias"] = 100 * np.abs(x.expected_count / x.full_count - 1)
    mid = np.sqrt(x.lo_nm * x.hi_nm)
    x["decade"] = pd.cut(
        mid,
        [1.7, 17, 170, 1700],
        right=False,
        labels=["1.7–17", "17–170", "170–1700"],
    )

    def summarize(g):
        return dict(
            cells=len(g),
            median=g.bias.median(),
            p90=g.bias.quantile(0.9),
            over10=(g.bias > 10).mean() * 100,
        )

    rows = []
    for (arm, band), g in x.groupby(["method", "decade"], observed=True):
        rows.append(dict(method=arm, band=band, **summarize(g)))
    for arm, g in x.groupby("method"):
        rows.append(dict(method=arm, band="all", **summarize(g)))
    return pd.DataFrame(rows)


def plot(edges, qual, spectra, path):
    """Relate one grid to conditional count recovery and independent spectra."""
    fig, (a, b, c) = plt.subplots(
        1,
        3,
        figsize=(11.6, 3.35),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.32, 1, 1]},
    )
    sample = edges[
        (edges.seed == 10111) & (edges.start == 3600) & (edges.parcel == 1)
    ]
    for i, (arm, color) in enumerate(zip(ORDER, COLORS, strict=True)):
        v = sample[sample.method == arm].edge_nm
        a.vlines(v, i - 0.21, i + 0.21, color=color, lw=1.6)
        q = qual[qual.method == arm]
        tally = (q.status == "qualified").sum()
        b.barh(i, tally, color=color, height=0.43)
        b.text(tally + 0.25, i, str(tally), va="center")
        row = spectra[(spectra.method == arm) & (spectra.band == "all")].iloc[0]
        c.plot([row["median"], row.p90], [i, i], color=color, lw=2)
        c.plot(row["median"], i, "o", color=color)
        c.plot(row.p90, i, "|", ms=12, color=color)
    a.axvspan(150, 160, color="#d8e6dd", zorder=-1)
    for ax in (a, b, c):
        ax.set_ylim(3.6, -0.6)
        ax.set_yticks(range(4), labels=NAMES if ax is a else [""] * 4)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    a.set(
        xlim=(65, 230),
        xlabel="Dry diameter (nm)",
        title="One parcel’s bin edges",
    )
    b.set(
        xlim=(0, 17.5),
        xticks=[0, 4, 8, 12, 16],
        xlabel="Qualified air masses (of 16)",
        title="150–160 nm counter",
    )
    c.set(
        xlim=(0, None),
        xlabel="Absolute bias (%)\nMedian (dot) to 90th percentile",
        title="Other size channels",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("figure", type=Path)
    p.add_argument(
        "--cache", type=Path, help="Assemble simulation results before plotting"
    )
    a = p.parse_args()
    if a.cache:
        assemble(a.cache, a.out)
    pairs = pd.read_csv(a.out / "adaptive-pairs.csv.gz")
    edges = pd.read_csv(a.out / "adaptive-edges.csv.gz")
    qual = qualification(pairs)
    qual.to_csv(
        a.out / "adaptive-qualification.csv", index=False, float_format="%.10g"
    )
    blind = blind_channels(pd.read_csv(a.out / "adaptive-spectra.csv.gz"))
    blind.to_csv(
        a.out / "adaptive-channel-summary.csv",
        index=False,
        float_format="%.10g",
    )
    plot(edges, qual, blind, a.figure)
