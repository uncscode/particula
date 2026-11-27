"""Compile paired restart tests and compare optical, CCN and chemical errors."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from coating_replay import failures
from stage_two_design import design

KEYS = ["seed", "acid", "exchange", "lift", "n", "dt", "stream"]
EVENT = KEYS + ["start", "end", "parcel"]


def gather(root):
    """Require every specified history, budget and stochastic control."""
    pieces = []
    failed = failures()
    for c in design():
        methods = (
            ["local-4", "local-8", "shell-8"]
            if c.primary
            else ["local-4", "local-8"]
        )
        if c.primary:
            methods.append("baseline-0")
            if (c.seed, c.acid, c.exchange, c.lift) in failed:
                methods += ["local-16", "outside-8", "exact-0"] + [
                    f"{name}-{k}"
                    for name in ["repeat", "paired", "paired4"]
                    for k in (1, 2, 3)
                ]
        for spec in methods:
            g = pd.read_csv(root / c.key / f"{spec}.csv")
            if len(g) != 16 or g.duplicated(EVENT).any():
                raise ValueError(
                    f"Missing or duplicated records for {c.key}/{spec}"
                )
            pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def stochastic(data):
    """Compare reduced and full runs on each alternate stream."""
    pair = data.query("method in ['paired', 'paired4'] & end>start").merge(
        data.query("method=='repeat' & end>start")[EVENT + ["budget", "area"]],
        on=EVENT + ["budget"],
        validate="many_to_one",
        suffixes=("", "_full"),
    )
    pair["extra_rows"] = pair.method.map({"paired": 8, "paired4": 4})
    pair["paired_error_pct"] = 100 * (pair.area / pair.area_full - 1)
    pair["full_repeat_pct"] = 100 * (pair.area_full / pair.ref_area - 1)
    return pair[
        EVENT
        + [
            "budget",
            "extra_rows",
            "area",
            "area_full",
            "paired_error_pct",
            "full_repeat_pct",
        ]
    ].rename(columns={"budget": "replicate"})


def history(data):
    """Report worst forecast and snapshot errors and row counts."""
    rows = []
    for key, g in data[
        ~data.method.isin(["repeat", "paired", "paired4"])
    ].groupby(KEYS + ["method", "budget"]):
        future = g[g.end > g.start]
        snap = g[g.start == g.end]
        end = future[(future.end == 21600) & (future.parcel == 0)]
        real = end[end.ref_drops > 0]
        burst = end[end.ref_births > 0]
        pair = future.groupby(["start", "end"])[["area", "ref_area"]].sum()
        rows.append(
            dict(zip(KEYS + ["method", "budget"], key, strict=True))
            | dict(
                future_pct=future.signed_pct.abs().max(),
                snapshot_pct=snap.signed_pct.abs().max(),
                wet_pct=future.wet_abs_pct.abs().max(),
                pooled_pct=(100 * (pair.area / pair.ref_area - 1)).abs().max(),
                act_max_pp=future.act_max.max(),
                act_rmse_pp=future.act_rmse.max(),
                survival_pp=end.survival_pp.abs().max(),
                birth_pct=100 * abs(burst.births / burst.ref_births - 1).max(),
                zero_birth_n=end.loc[end.ref_births == 0, "births"].max(),
                drops_pct=100 * abs(real.drops / real.ref_drops - 1).max(),
                zero_drops_cm3=end.loc[end.ref_drops == 0, "drops"].max(),
                lwc_pct=100 * abs(end.lwc / end.ref_lwc - 1).max(),
                rows_min=snap.rows.min(),
                rows_max=snap.rows.max(),
                condensed_pct=future[
                    ["cond_svoc_pct", "cond_lvoc_pct", "cond_hom_pct"]
                ]
                .abs()
                .max()
                .max(),
                gas_pp=future[["gas_svoc_pp", "gas_lvoc_pp", "gas_hom_pp"]]
                .abs()
                .max()
                .max(),
            )
        )
    return pd.DataFrame(rows)


def coating_inventory(root, targets):
    """Measure BC inventory misplaced across coating classes at restart."""
    rows = []
    for prefix in ("baseline-0", "shell-8"):
        for path in sorted(root.glob(f"*/{prefix}-cells.csv.gz")):
            g = pd.read_csv(path)
            g = g[(g.start == g.end) & g.dry_bin.isin(targets.dry_bin)]
            for (start, parcel), s in g.groupby(["start", "parcel"]):
                bins = s.groupby(["dry_bin", "shell_bin"])[
                    ["ref_bc", "pred_bc"]
                ].sum()
                rows.append(
                    dict(
                        case=path.parent.name,
                        method=prefix,
                        start=start,
                        parcel=parcel,
                        misplaced_pct=50
                        * (bins.pred_bc - bins.ref_bc).abs().sum()
                        / bins.ref_bc.sum(),
                    )
                )
    return pd.DataFrame(rows)


def figure(peaks, h, path):
    """Plot optical attribution and refinement errors by history."""
    j = peaks.merge(
        h.query("method=='local' & budget==8 & stream==0 & dt==75 & n==25000"),
        on=KEYS[:4],
        validate="one_to_one",
    )
    four = h.query(
        "method=='local' & budget==4 & stream==0 & dt==75 & n==25000"
    )
    j = j.merge(
        four[KEYS[:4] + ["future_pct"]].rename(columns={"future_pct": "four"}),
        on=KEYS[:4],
        validate="one_to_one",
    )
    y = np.arange(len(j))
    labels = [
        f"{r.seed}   "
        + "/".join(
            {-1: "−", 0: "0", 1: "+"}[v] for v in (r.acid, r.exchange, r.lift)
        )
        for r in j.itertuples()
    ]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(8.6, 3.1), sharey=True, layout="constrained"
    )
    left.barh(y, j.mass_pp, color="#2b687e", label="BC amount in parcel")
    left.barh(
        y,
        j.efficiency_pp,
        left=j.mass_pp,
        color="#d9a256",
        label="Absorption per unit BC",
    )
    left.set_yticks(y, labels)
    left.invert_yaxis()
    left.set_xlabel("Contribution at original worst time (pp)")
    left.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, 1))
    right.hlines(
        y,
        j[["signed_pct", "future_pct", "four"]].min(axis=1),
        j[["signed_pct", "future_pct", "four"]].max(axis=1),
        color=".7",
    )
    right.scatter(
        j.signed_pct,
        y,
        facecolor="white",
        edgecolor="#334855",
        label="Original",
    )
    right.scatter(
        j.future_pct, y, marker="D", color="#33866c", label="8 added rows"
    )
    right.scatter(
        j.four,
        y,
        marker="|",
        s=100,
        color="#a83e38",
        label="4 added rows",
        zorder=4,
    )
    right.axvline(5, color=".45", ls="--", lw=1)
    right.set_xlabel("Worst future absorption error (%)")
    right.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, 1), ncols=3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("figure", type=Path)
    a = p.parse_args()
    data = gather(a.root)
    ordinary = data[~data.method.isin(["repeat", "paired", "paired4"])]
    ordinary.to_csv(
        a.out / "coating-restarts.csv.gz",
        index=False,
        float_format="%.10g",
        compression={"method": "gzip", "mtime": 0},
    )
    check = coating_inventory(
        a.root, pd.read_csv(a.out / "coating-targets.csv")
    )
    if len(check) != 48:
        raise ValueError("Missing coating-inventory comparisons")
    check.to_csv(
        a.out / "coating-inventory.csv", index=False, float_format="%.10g"
    )
    repeats = stochastic(data)
    repeats.to_csv(
        a.out / "coating-repeats.csv.gz",
        index=False,
        float_format="%.10g",
        compression={"method": "gzip", "mtime": 0},
    )
    h = history(ordinary)
    h.to_csv(a.out / "coating-histories.csv", index=False, float_format="%.10g")
    figure(pd.read_csv(a.out / "coating-peaks.csv"), h, a.figure)
