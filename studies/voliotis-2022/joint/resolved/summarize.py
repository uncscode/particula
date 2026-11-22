"""Plot size observations and summarize stochastic particle replays."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def main():
    """Summarize all three checks and plot the fine particle-resolved runs."""
    fit = json.loads((OUT / "fit.json").read_text())
    chosen = json.loads((OUT / "selected.json").read_text())
    cols = (
        [
            "lag_alpha_h",
            "lag_cresol_h",
            "lag_isoprene_h",
            "yield_pure_alpha",
            "yield_pure_cresol",
            "yield_pure_isoprene",
            "yield_mixed_alpha",
            "yield_mixed_cresol",
            "yield_mixed_isoprene",
            "wall_eddy_s_inv",
            "unused_10",
            "unused_11",
            "isoprene_diversion",
            "unused_13",
            "unused_14",
            "unused_15",
        ]
        + chosen["chemistry_keys"]
        + [f"dose_multiplier_run_{r}" for r in chosen["ids"]]
        + [
            "seed_gsd",
            "volume_allocation_fraction",
            "Jmax_cm3_s",
            "kappa_org_HTDMA",
            "kappa_org_CCN",
        ]
    )
    pd.DataFrame(
        dict(parameter=cols, initial=fit["initial"], refitted=chosen["vector"])
    ).to_csv(OUT / "parameters.csv", index=False)
    specs = pd.read_csv(OUT / "psd.csv.gz")
    modes = pd.read_csv(OUT / "modes.csv.gz")
    water = pd.read_csv(OUT / "water.csv.gz")
    histories = pd.read_csv(OUT / "histories.csv.gz")
    grid = pd.read_csv(HERE / "size_data/run24_gmd.csv")
    metrics = []
    for (label, n, seed), g in specs[
        specs.case.isin(["frozen", "refit"])
    ].groupby(["case", "n", "seed"]):

        def select(f, label=label, n=n, seed=seed):
            return f[(f.case == label) & (f.n == n) & (f.seed == seed)]

        m = select(modes)
        h = select(histories.query("run==24"))
        w = select(water).copy()
        gm = np.interp(grid.time_s, h.time_s, h.run24_gmd_nm)
        metrics.append(
            dict(
                case=label,
                n=n,
                seed=seed,
                psd_rmse_cm3_decade=np.sqrt(
                    np.mean((g.simulated - g.dndlog10d_cm3) ** 2)
                ),
                mode_mae_nm=np.mean(abs(m.predicted_nm - m.mode_nm)),
                gmd_mae_nm=np.mean(abs(gm - grid.gmd_nm)),
                gmd_start=gm[0],
                gmd_end=gm[-1],
                water_matched=w.prediction.notna().sum(),
                water_all=len(w),
                water_matched_rmse=np.sqrt(
                    np.nanmean((w.prediction - w.observed) ** 2)
                ),
            )
        )
    pd.DataFrame(metrics).to_csv(OUT / "size_water_summary.csv", index=False)
    fine = specs[specs.n == 2048]
    best = fine[fine.case == "refit"]
    figs, axes = plt.subplots(2, 2, figsize=(10, 6.5), layout="constrained")
    for t, color in [(0, "#3d5776"), (7200, "#008477"), (18000, "#ba5426")]:
        d = best[best.time_s == t]
        x = np.sqrt(d.low_nm * d.high_nm)
        axes[0, 0].plot(x, d.dndlog10d_cm3 / 1e4, "o", ms=3, color=color)
        axes[0, 0].plot(
            x, d.simulated / 1e4, color=color, label=f"{t / 3600:g} h"
        )
    axes[0, 0].set(
        xscale="log",
        xlabel="Dry mobility diameter (nm)",
        ylabel="dN/dlog₁₀D (10⁴ cm⁻³)",
        title="Ternary run 24 · number spectrum",
    )
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=9)
    axes[0, 1].plot(
        grid.time_s / 3600, grid.gmd_nm, "ko", ms=3, label="Measured"
    )
    h = histories.query("run==24 and n==2048")
    for label, color in [("frozen", "#849099"), ("refit", "#008477")]:
        d = h[h.case == label]
        axes[0, 1].plot(
            d.time_s / 3600,
            d.run24_gmd_nm,
            color=color,
            label=label.capitalize(),
        )
    axes[0, 1].set(
        xlabel="Hours after lights on",
        ylabel="Geometric mean diameter (nm)",
        title="Diameter growth · run 24",
    )
    axes[0, 1].legend(frameon=False, fontsize=8)
    for ax, run in zip(axes[1], (6, 15), strict=True):
        d = water.query("run==@run and case=='refit' and n==2048")
        for ch, color in [
            ("HTDMA_75nm", "#3677a4"),
            ("HTDMA_100nm", "#008477"),
            ("HTDMA_150nm", "#ba5426"),
        ]:
            group = d[d.channel == ch].sort_values("seconds")
            ax.scatter(group.seconds / 3600, group.observed, color=color, s=12)
            ax.plot(
                group.seconds / 3600,
                group.prediction,
                color=color,
                label=ch.split("_")[1].replace("nm", " nm"),
            )
        ax.set(
            xlabel="Hours after lights on",
            ylabel="Effective κ",
            title=f"Size-selected water uptake · run {run}",
        )
        ax.legend(frameon=False, ncol=3, fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.15)
    figs.savefig(OUT / "resolved-comparison.png", dpi=180)
    plt.close(figs)


if __name__ == "__main__":
    main()
