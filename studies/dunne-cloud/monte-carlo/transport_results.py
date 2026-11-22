"""Join common draws, check budgets and summarize cold particle transport."""

from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LEVELS = [0.05, 0.5, 0.95]
QUANTILES = ["p05", "median", "p95"]
LINES = [366, 332, 334, 344, 336, 338]
GROUP = ["id", "block", "line"]
COLORS = {"vapor": "#346997", "attraction": "#af542c"}


def bounds(data, groups, variables):
    q = data.groupby(groups)[variables].quantile(LEVELS).unstack()
    q.columns = [
        f"{name}_{QUANTILES[LEVELS.index(p)]}" for name, p in q.columns
    ]
    return q.reset_index()


def scientific_plot(t, rates, pop):
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.6), sharey=True)
    layout = [
        ("survival10", t, "10 nm survival", True),
        (
            "J10",
            rates[rates.source == "full"],
            "10 nm flux, full source\n(cm⁻³ s⁻¹)",
            True,
        ),
        (
            "zero10",
            pop[pop.source == "full"],
            "Simulations with no particles reaching\n10 nm during hours 12–18 (%)",
            False,
        ),
    ]
    laws = list(COLORS)
    for ax, (key, df, label, log) in zip(axes, layout):
        for j, law in enumerate(laws):
            yy = np.arange(len(LINES)) + (j - 0.5) * 0.27
            for y, line in zip(yy, LINES):
                vals = df[(df.line == line) & (df.law == law)][key].values
                if log:
                    lo, mid, hi = np.quantile(vals, LEVELS)
                    ax.plot([lo, hi], [y, y], color=COLORS[law], lw=2)
                    ax.scatter(mid, y, color=COLORS[law], s=24, zorder=3)
                else:
                    ax.scatter(
                        100 * vals.mean(), y, color=COLORS[law], s=24, zorder=3
                    )
        if log:
            ax.set_xscale("log")
            from matplotlib.ticker import LogLocator

            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        else:
            ax.set_xlim(-4, 105)
            ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel(label, fontsize=10)
        ax.grid(axis="x", alpha=0.23)
        ax.spines[["right", "top"]].set_visible(False)
        ax.axhline(1.5, color=".82", lw=1)
    # Display the archived inputs, not the first calibration draw.
    from sys import path

    path.insert(0, str(HERE.parent / "analysis"))
    from reference import cold_condition_label

    axes[0].set_yticks(
        np.arange(6),
        [cold_condition_label(line) for line in LINES],
    )
    axes[0].invert_yaxis()
    axes[0].set_title("Temperature · H₂SO₄*", loc="left", fontsize=10)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker="o", color=COLORS[l], label=lab)
        for l, lab in zip(
            laws, ["Vapor transfer", "Finite acid size + attraction"]
        )
    ]
    fig.legend(
        handles=handles,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.58, 0.99),
        frameon=False,
    )
    fig.subplots_adjust(
        left=0.20, right=0.98, bottom=0.26, top=0.83, wspace=0.24
    )
    fig.text(
        0.14,
        0.09,
        "* Archived acid in 10⁶ molecules cm⁻³. Points and lines show the median and central 90% of 128 draws in the first two panels.",
        fontsize=9,
    )
    volumes = pop.loc[
        (pop.line == 336) & (pop.source == "full"), "sample_volume_cm3"
    ]
    fig.text(
        0.14,
        0.045,
        f"Panel 3 uses finite samples. At 248.3 K and 18 × 10⁶ acid molecules "
        f"cm⁻³, simulated volumes span {volumes.min():.3f}–{volumes.max():.2f} cm³.",
        fontsize=9,
    )
    fig.savefig(HERE / "transport-bounds.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    t = pd.read_csv(HERE / "transport-tracks.csv")
    r = pd.read_csv(HERE / "transport-replicate.csv")
    f = pd.read_csv(HERE / "formation.csv")
    assert len(t) == 128 * 6 * 2 and len(r) == 16 * 6 * 2
    assert (t.endpoint_nm >= 10).all()
    f = f[
        f["split"].isin(["223_low", "248_low"])
        & f["shape"].eq("linear")
        & f["sample"].eq("neutral")
    ]
    assert len(f) == 128 * 6 * 2
    actual = f[f.topology == "full"][GROUP + ["adjusted_J"]].rename(
        columns={"adjusted_J": "J17"}
    )
    actual["source"] = "archive_adjusted"
    fitted = f[GROUP + ["topology", "predicted_J"]].rename(
        columns={"topology": "source", "predicted_J": "J17"}
    )
    flux = pd.concat([actual, fitted], ignore_index=True).merge(
        t[GROUP + ["law", "survival32", "survival10"]], on=GROUP
    )
    assert len(flux) == 128 * 6 * 2 * 3 and not flux.isna().any().any()
    flux["J32"] = flux.J17 * flux.survival32
    flux["J10"] = flux.J17 * flux.survival10
    flux.to_csv(HERE / "transport-flux.csv", index=False)
    ts = bounds(
        t,
        ["line", "law"],
        [
            "survival32",
            "survival10",
            "time32_s",
            "time10_s",
            "wall_loss",
            "dil_loss",
            "coag_loss",
        ],
    )
    ts.to_csv(HERE / "transport-survival-summary.csv", index=False)
    bounds(flux, ["line", "law", "source"], ["J17", "J32", "J10"]).to_csv(
        HERE / "transport-flux-summary.csv", index=False
    )

    rep = t.merge(r, on=GROUP + ["law"], suffixes=("", "_rep"))
    checks = {}
    for x in ("survival32", "survival10"):
        ratio = np.maximum(rep[x] / rep[x + "_rep"], rep[x + "_rep"] / rep[x])
        checks[x + "_repeat_factor_median_p95_max"] = np.quantile(
            ratio, [0.5, 0.95, 1]
        ).tolist()
    checks["max_single_track_number_error"] = max(
        t.max_number_error.max(), r.max_number_error.max()
    )
    checks["max_single_track_species_error_molecules"] = max(
        t.max_species_error.max(), r.max_species_error.max()
    )
    block = bounds(t, ["line", "law", "block"], ["survival32", "survival10"])
    block.to_csv(HERE / "transport-block-bounds.csv", index=False)

    pop = pd.read_csv(HERE / "transport-population.csv")
    assert len(pop) == (128 * 6 * 2 * 2 + 8 * 2 * 2 * 2) * 3
    p = pop[(pop.dt_s == 30) & (pop.window_end_h == 18)].copy()
    assert len(p) == 128 * 6 * 2 * 2
    p["zero10"] = p.J10 == 0
    p["zero32"] = p.J32 == 0
    p["merger_fraction_32"] = p.merger_J32 / p.J32.replace(0, np.nan)
    ps = bounds(
        p,
        ["line", "law", "source"],
        ["J32", "J10", "mean_N32_cm3", "mean_N10_cm3", "merger_fraction_32"],
    )
    ps = ps.merge(
        p.groupby(["line", "law", "source"])[["zero10", "zero32"]]
        .mean()
        .reset_index()
    )
    ps.to_csv(HERE / "transport-population-summary.csv", index=False)
    paired_sources = p.merge(
        f[GROUP + ["topology", "predicted_J"]],
        left_on=GROUP + ["source"],
        right_on=GROUP + ["topology"],
    )
    checks["max_relative_finite_source_deviation_from_final_refit"] = abs(
        paired_sources.J17 / paired_sources.predicted_J - 1
    ).max()
    assert (
        checks["max_relative_finite_source_deviation_from_final_refit"] < 1e-12
    )
    checks["max_population_relative_species_error"] = (
        pop.relative_species_error.max()
    )
    fine = pop[(pop.dt_s == 15) & (pop.window_end_h == 18)]
    fine = fine.merge(p, on=GROUP + ["law", "source"], suffixes=("_fine", ""))
    for k in ["J32", "J10"]:
        keep = (fine[k] > 0) & (fine[k + "_fine"] > 0)
        checks[k + "_both_detect"] = int(keep.sum())
        checks[k + "_one_detect"] = int(
            ((fine[k] > 0) ^ (fine[k + "_fine"] > 0)).sum()
        )
        checks[k + "_both_zero"] = int(
            ((fine[k] == 0) & (fine[k + "_fine"] == 0)).sum()
        )
        factor = np.maximum(
            fine.loc[keep, k] / fine.loc[keep, k + "_fine"],
            fine.loc[keep, k + "_fine"] / fine.loc[keep, k],
        )
        checks[k + "_fine_vs_main_factor_among_both_detect_p50_p95"] = (
            np.quantile(factor, [0.5, 0.95]).tolist()
        )
    # Pair within each draw to preserve the common physical and gas inputs.
    pivot = t.pivot(index=GROUP, columns="law", values="survival10")
    gain = pivot.attraction / pivot.vapor
    checks["attractive_larger_pairs"] = int((gain > 1).sum())
    checks["paired_attraction_gain_336_p05_p50_p95"] = np.quantile(
        gain[gain.index.get_level_values("line") == 336], LEVELS
    ).tolist()
    corr = []
    for (line, law), g in t.groupby(["line", "law"]):
        for x in [
            "acid",
            "base",
            "wall_scale",
            "water_per_acid",
            "sticking",
            "background_cm3",
        ]:
            corr.append(
                dict(
                    line=line,
                    law=law,
                    input=x,
                    spearman32=g[x].corr(g.survival32, method="spearman"),
                    spearman10=g[x].corr(g.survival10, method="spearman"),
                )
            )
    pd.DataFrame(corr).to_csv(HERE / "transport-sensitivity.csv", index=False)
    (HERE / "transport-checks.json").write_text(json.dumps(checks, indent=2))
    scientific_plot(t, flux, p)


if __name__ == "__main__":
    main()
