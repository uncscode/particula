"""Summarize paired stage-two errors by history and forcing level."""

import argparse
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stage_two import METHODS
from stage_two_design import design

KEYS = ["seed", "acid", "exchange", "lift", "n", "dt", "stream", "method"]


def collect(root):
    """Load every expected history and verify its checkpoint pairs."""
    scores = []
    for c in design():
        methods = METHODS if c.primary else ("water", "rank")
        for m in methods:
            data = pd.read_csv(root / c.key / f"stage2-{m}.csv")
            if (
                len(data) != 24
                or data.duplicated(["start", "end", "parcel"]).any()
            ):
                raise ValueError(
                    f"Incomplete or duplicate scores for {c.key}, {m}"
                )
            scores.append(data)
    return pd.concat(scores, ignore_index=True)


def histories(s):
    """Take the largest applicable error within each history and method."""
    records = []
    for key, g in s.groupby(KEYS):
        labels = dict(zip(KEYS, key, strict=True))
        for name, group in [
            ("snapshot", g.query("start==end")),
            ("restart", g.query("start<end")),
        ]:
            row = (
                group[
                    [
                        "ks_pp",
                        "ks_raw",
                        "w1_nm",
                        "cut_pp",
                        "act_rmse",
                        "act_max",
                        "dry_abs",
                        "wet_abs",
                        "chi_coverage",
                    ]
                ]
                .max()
                .to_dict()
            )
            row["chi_coverage"] = group.chi_coverage.min()
            row["rows_max"] = group.rows.max()
            for shape in ["nine", "six", "rep"]:
                row[f"chi_{shape}"] = group[f"chi_{shape}_all"].max()
                row[f"dec_{shape}"] = (
                    group[
                        [
                            f"chi_{shape}_{d}"
                            for d in ["small", "middle", "large"]
                        ]
                    ]
                    .max()
                    .max()
                )
            if name == "restart":
                end = group.query("end==21600 & parcel==0")
                row["survival_pp"] = end.survival_pp.max()
                row["birth_abs_n"] = (end.births - end.ref_births).abs().max()
                present = end.ref_births > 0
                row["birth_pct"] = (
                    100
                    * abs(
                        end.loc[present, "births"]
                        / end.loc[present, "ref_births"]
                        - 1
                    ).max()
                )
                row["zero_birth_abs_n"] = (
                    (
                        end.loc[~present, "births"]
                        - end.loc[~present, "ref_births"]
                    )
                    .abs()
                    .max()
                )
                for t in [3600, 10800]:
                    e = end.loc[end.start == t].iloc[0]
                    row[f"drops_{t // 3600}h"] = (
                        100 * abs(e.drops / e.ref_drops - 1)
                        if e.ref_drops
                        else np.nan
                    )
                    row[f"zero_drops_{t // 3600}h_cm3"] = (
                        e.drops if e.ref_drops == 0 else np.nan
                    )
                    row[f"lwc_{t // 3600}h"] = 100 * abs(e.lwc / e.ref_lwc - 1)
            records.append(dict(**labels, phase=name, **row))
    return pd.DataFrame(records)


def plot(summary, out):
    """Display per-history errors on the three independent seed blocks."""
    primary = summary.query("dt==75 & n==25000 & stream==0")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout="constrained")
    panels = [
        ("snapshot", "ks_pp", "Largest cohort CDF error (pp)"),
        ("snapshot", "w1_nm", "Mean diameter displacement (nm)"),
        ("restart", "drops_3h", "Cloud droplet peak error (%)"),
    ]
    for ax, (phase, col, label) in zip(axes, panels, strict=True):
        for i, m in enumerate(METHODS):
            block = primary[(primary.phase == phase) & (primary.method == m)]
            for k, (seed, g) in enumerate(block.groupby("seed")):
                jitter = np.linspace(-0.075, 0.075, len(g))
                ax.scatter(
                    i + (k - 1) * 0.19 + jitter,
                    g[col],
                    s=14,
                    alpha=0.76,
                    color=["#216682", "#c4772a", "#499478"][k],
                    linewidths=0,
                    label=str(seed) if i == 0 else None,
                )
        ax.set_xticks(
            range(4), ["Dry", "Water", "+ origin", "+ rank"], rotation=20
        )
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].axhline(10, ls="--", color="0.45", lw=1)
    axes[0].legend(title="History seed", fontsize=7)
    fig.savefig(out, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("figure", type=Path)
    a = p.parse_args()
    reference_rows = []
    for case in design():
        trace = pd.read_csv(a.root / case.key / "trace.csv")
        final = trace.iloc[-1]
        reference_rows.append(
            dict(
                **asdict(case),
                born_cm3=final.born / (2 * case.n / 1000),
                survival_pct=100 * final.reached / final.born
                if final.born
                else np.nan,
                peak_drops_cm3=trace.drops_cm3.max(),
                peak_lwc_g_m3=trace.water_g_m3.max(),
            )
        )
    pd.DataFrame(reference_rows).to_csv(
        a.out / "stage-two-forcing.csv", index=False, float_format="%.10g"
    )
    scores = collect(a.root)
    scores.to_csv(
        a.out / "stage-two-scores.csv.gz",
        index=False,
        float_format="%.10g",
        compression={"method": "gzip", "mtime": 0},
    )
    summary = histories(scores)
    summary.to_csv(
        a.out / "stage-two-histories.csv", index=False, float_format="%.10g"
    )
    plot(summary, a.figure)
