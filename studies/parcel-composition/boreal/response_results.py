"""Publish conditional equivalence scores and measurement distributions."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.stats import beta, ks_2samp

CORE = ("activation", "cloud", "chemistry", "absorption", "scattering")
LABELS = [
    "Activation",
    "Cloud water & number",
    "Chemistry",
    "Absorption",
    "Scattering",
    "Five bulk families",
    "Nucleation & growth",
    "All six together",
    "150–160 nm counter",
]
ALL = (*CORE, "bulk_joint", "nucleation", "joint", "counter")
PHYSICAL = {
    "ccn_a": "% activated at 0.3%",
    "ccn_b": "% activated at 0.3%",
    "drops_cm3": "cm^-3",
    "lwc_g_m3": "g m^-3",
    "peak_drops": "cm^-3",
    "births": "particles",
    "reach": "% of at-risk lineages",
    "organic_transfer": "dry equivalent diameter cubed (nm^3)",
    **{
        s + k: "Mm^-1"
        for s in ("abs_", "wet_abs_", "sca_", "wet_sca_")
        for k in ("a", "b")
    },
    "channel_a": "cm^-3",
    "channel_b": "cm^-3",
}


def limits(k, n, tail):
    """Exact binomial bounds, including the two degenerate outcomes."""
    return (
        beta.ppf(tail, k, n - k + 1) if k else 0.0,
        beta.isf(tail, k + 1, n - k) if k < n else 1.0,
    )


def gather(root):
    """Check the full trial design and score the combined endpoints."""
    from redraw_run import GATES, percentage
    from response_course import ASCENT, CASES, SHORT
    from response_pairs import BUDGET, FAMILIES, LONG_BUDGET

    chunks = []
    for case in CASES:
        for start in (*SHORT, ASCENT):
            d = pd.read_csv(root / f"{case.seed}-{start}.csv.gz")
            n = LONG_BUDGET if start == ASCENT else BUDGET
            if set(d.seed) != {case.seed} or set(d.start) != {start}:
                raise ValueError("Cache belongs to a different history or time")
            for name, expected in (("redraw", n), ("repeat", 12)):
                sample = d[d.method == name]
                indices = pd.MultiIndex.from_product(
                    [range(expected), [0, 1800 if start == ASCENT else 75]],
                    names=["trial", "lead"],
                )
                actual = pd.MultiIndex.from_frame(sample[["trial", "lead"]])
                if not actual.sort_values().equals(indices):
                    raise ValueError(
                        f"Incomplete trials for {case.key}, {start}"
                    )
            chunks.append(d)
    trials = pd.concat(chunks, ignore_index=True)
    for field, gate in (("drops_cm3", 10), ("lwc_g_m3", 10)):
        trials["cloud"] = np.maximum(
            trials.cloud,
            percentage(
                trials[field].to_numpy(), trials["ref_" + field].to_numpy()
            )
            / gate,
        )
    if trials[list(FAMILIES)].isna().any().any():
        raise ValueError("Undefined endpoint score")
    trials["bulk_joint"] = trials[list(CORE)].max(axis=1)
    trials["nucleation"] = np.maximum.reduce(
        [trials[k] / limit for k, limit in GATES["nucleation"].items()]
    )
    trials["joint"] = np.maximum(trials.bulk_joint, trials.nucleation)
    trials["window"] = np.where(
        trials.start == ASCENT, "ascent", "two short windows"
    )
    return trials


def summarize(trials):
    """Bound failure probability separately for every fixed air mass."""
    histories = trials.seed.nunique()
    if histories != 16:
        raise ValueError("All 16 history ensembles are required")
    tail = 0.05 / (2 * histories * len(ALL))
    rows = []
    for (seed, method, window), part in trials.groupby(
        ["seed", "method", "window"]
    ):
        worst = part.groupby("trial")[list(ALL)].max()
        for family in ALL:
            n = len(worst)
            k = int((worst[family] > 1).sum())
            lo, hi = limits(k, n, tail)
            idx = part[family].idxmax()
            row = part.loc[idx]
            qualified = method == "redraw" and window == "two short windows"
            verdict = (
                (
                    "qualified"
                    if hi <= 0.05
                    else "exceeds"
                    if lo > 0.05
                    else "unresolved"
                )
                if qualified
                else "descriptive"
            )
            rows.append(
                dict(
                    seed=seed,
                    method=method,
                    window=window,
                    family=family,
                    trials=n,
                    exceedances=k,
                    lower=lo if qualified else np.nan,
                    upper=hi if qualified else np.nan,
                    status=verdict,
                    max_gate_ratio=row[family],
                    worst_start=row.start,
                    worst_lead=row.lead,
                    worst_trial=row.trial,
                )
            )
    return pd.DataFrame(rows)


def marginals(trials):
    """Compare paired marginal samples in their original units."""
    rows = []
    for keys, group in trials[trials.method == "redraw"].groupby(
        ["seed", "start", "lead"]
    ):
        for name, unit in PHYSICAL.items():
            x, y = group["ref_" + name], group[name]
            # Descriptive distance only: these samples are paired.
            row = dict(zip(("seed", "start", "lead"), keys, strict=True))
            row.update(
                measurement=name,
                unit=unit,
                trials=len(x),
                empirical_ks=ks_2samp(x, y, method="asymp").statistic,
            )
            for label, values in (("full", x), ("redraw", y)):
                for pct in (5, 50, 95):
                    row[f"{label}_q{pct}"] = np.percentile(values, pct)
            rows.append(row)
    return pd.DataFrame(rows)


def figure(scores, output):
    """Show exceedance counts and simultaneous decisions by history."""
    short = scores[
        (scores.method == "redraw") & (scores.window == "two short windows")
    ]
    grid = short.pivot(index="family", columns="seed", values="status").loc[
        list(ALL)
    ]
    colors = {"qualified": 0, "unresolved": 1, "exceeds": 2}
    values = grid.apply(lambda col: col.map(colors)).to_numpy()
    cmap = ListedColormap(["#367763", "#e0bf72", "#ab4938"])
    fig, ax = plt.subplots(figsize=(10.4, 3.7), layout="constrained")
    ax.imshow(values, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set(
        yticks=range(len(ALL)),
        yticklabels=LABELS,
        xticks=range(len(grid.columns)),
        xticklabels=grid.columns,
        xlabel="Synthetic air mass (initialization seed)",
    )
    ax.tick_params(axis="x", rotation=50, length=0)
    ax.tick_params(axis="y", length=0)
    counts = (
        short.pivot(index="family", columns="seed", values="exceedances")
        .loc[list(ALL)]
        .to_numpy()
    )
    for (i, j), k in np.ndenumerate(counts):
        ax.text(
            j,
            i,
            str(k),
            ha="center",
            va="center",
            fontsize=8,
            color="black" if values[i, j] == 1 else "white",
        )
    ax.set_title(
        "Draws outside tolerance out of 192 paired trials per air mass",
        loc="left",
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "root", type=Path, help="Trial folder or published pairs CSV"
    )
    p.add_argument("out", type=Path)
    p.add_argument("figure", type=Path)
    a = p.parse_args()
    pair_file = a.root
    if not pair_file.is_file():
        pair_file = a.out / "response-pairs.csv.gz"
        gather(a.root).to_csv(pair_file, index=False, float_format="%.12g")
    # Score the archived precision so replay reproduces the published table.
    trials = pd.read_csv(pair_file)
    scores = summarize(trials)
    scores.to_csv(
        a.out / "response-qualification.csv", index=False, float_format="%.10g"
    )
    marginals(trials).to_csv(
        a.out / "response-distributions.csv.gz",
        index=False,
        float_format="%.12g",
    )
    figure(scores, a.figure)
