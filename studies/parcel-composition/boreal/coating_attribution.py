"""Identify optically consequential size and coating bins in the six misses."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from coating_replay import OUT

CLASSES = ["dry_bin", "core_bin", "shell_bin"]


def attribute(root, out):
    """Calibrate on seed 2039 and publish diagnostic cells for all six cases."""
    all_cells = []
    all_peaks = []
    target_scores = []
    published = pd.read_csv(OUT / "stage-two-scores.csv.gz").query(
        "method=='rank' & n==25000 & dt==75 & stream==0 & end>start"
    )
    for path in sorted(root.glob("*/baseline-0-cells.csv.gz")):
        g = pd.read_csv(path.parent / "baseline-0.csv")
        future = g[g.end > g.start]
        peak = future.loc[future.signed_pct.abs().idxmax()]
        case = {k: int(peak[k]) for k in ["seed", "acid", "exchange", "lift"]}
        orig = published
        for k, v in case.items():
            orig = orig[orig[k] == v]
        keys = ["start", "end", "parcel"]
        np.testing.assert_allclose(
            future.sort_values(keys).signed_pct.abs(),
            orig.sort_values(keys).dry_abs,
            atol=1e-7,
        )
        peaks = pd.read_csv(path)
        for point in future[future.signed_pct.abs() > 5].itertuples():
            same = peaks[
                (peaks.start == point.start) & (peaks.parcel == point.parcel)
            ]
            for t in [point.start, point.end]:
                subset = same[same.end == t].copy()
                all_cells.append(subset.assign(**case, failure_end=point.end))
        worst = peaks[
            (peaks.start == peak.start)
            & (peaks.end == peak.end)
            & (peaks.parcel == peak.parcel)
        ]
        bin_contrib = worst.groupby("dry_bin").signed_pp.sum()
        dry_id = bin_contrib.abs().idxmax()
        combo = worst.groupby(CLASSES).signed_pp.sum()
        dry, core, shell = combo.abs().idxmax()
        initial = g[
            (g.start == peak.start)
            & (g.end == peak.start)
            & (g.parcel == peak.parcel)
        ].iloc[0]
        pair = g[(g.start == peak.start) & (g.end == peak.end)]
        all_peaks.append(
            dict(
                **case,
                start=int(peak.start),
                end=int(peak.end),
                parcel=int(peak.parcel),
                initial_pct=initial.signed_pct,
                signed_pct=peak.signed_pct,
                mass_pp=peak.bc_mass_pp,
                efficiency_pp=peak.efficiency_pp,
                pair_pct=100 * (pair.area.sum() / pair.ref_area.sum() - 1),
                dry_bin=dry_id,
                dry_pp=bin_contrib[dry_id],
                cell_dry=dry,
                cell_core=core,
                cell_shell=shell,
                cell_pp=combo[dry, core, shell],
            )
        )
        if case["seed"] == 2039:
            target_scores.append(
                worst.groupby(["dry_bin", "shell_bin"]).signed_pp.sum().abs()
            )
    score = pd.concat(target_scores, axis=1).fillna(0).mean(axis=1)
    bins = score.groupby(level=0).sum().nlargest(3).index
    target = (
        pd.concat([score.loc[[b]].nlargest(2) for b in bins])
        .rename("mean_abs_pp")
        .reset_index()
    )
    target.to_csv(out / "coating-targets.csv", index=False, float_format="%.9g")
    pd.DataFrame(all_peaks).to_csv(
        out / "coating-peaks.csv", index=False, float_format="%.9g"
    )
    pd.concat(all_cells).to_csv(
        out / "coating-cells.csv.gz",
        index=False,
        float_format="%.10g",
        compression={"method": "gzip", "mtime": 0},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    attribute(a.root, a.out)
