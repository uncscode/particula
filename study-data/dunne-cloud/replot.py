"""Redraw the eight Dunne CLOUD study figures from their companion CSVs.

Run with Python, NumPy, pandas and Matplotlib. Each CSV sits beside this script.
Values are in the physical units given in the report and README.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator

DATA = Path(__file__).resolve().parent
QUANTILES = ["p05", "median", "p95"]
RATE_COLORS = {"full": "#18577d", "assisted": "#c65c24"}
LABELS = {"full": "Binary allowed", "assisted": "Ammonia required"}
LINES = [366, 332, 334, 344, 336, 338]


def read(name):
    """Read one figure's plotted values at the precision of the CSV."""
    return pd.read_csv(DATA / (name + ".csv"), float_precision="round_trip")


def overview(output):
    """Plot the two formation-rate ranges at the reference condition."""
    s = read("formation-overview").set_index("topology")
    fig = plt.figure(figsize=(3, 3), dpi=300, facecolor="#fff7dc")
    fig.text(0.085, 0.9, "CERN CLOUD", weight="bold", size=11, color="#234149")
    fig.text(0.085, 0.80, "248.3 K  ·  0.38 pptv NH₃", size=9)
    fig.text(0.085, 0.735, "1.80 × 10⁷ H₂SO₄ molecules cm⁻³", size=8)
    fig.text(0.085, 0.65, "Formation at 1.7 nm", size=10, weight="bold")
    ax = fig.add_axes([0.11, 0.23, 0.79, 0.37], facecolor="none")
    for y, t, color in [(2, "full", "#175671"), (1, "assisted", "#a44d25")]:
        r = s.loc[t]
        ax.hlines(y, r.p05, r.p95, lw=3.2, color=color)
        ax.scatter(r["median"], y, s=24, color=color, zorder=3)
        ax.text(
            0.012,
            y + 0.24,
            "Binary allowed" if t == "full" else "Ammonia required",
            size=7,
        )
    ax.scatter(s.observed.iloc[0], 0, color="#212529", marker="D", s=22)
    ax.text(0.037, 0, "CLOUD rate", va="center", size=7)
    ax.set(
        xscale="log",
        xlim=(0.009, 1.1),
        ylim=(-0.35, 2.62),
        yticks=[],
        xticks=[0.01, 0.1, 1],
        xticklabels=["0.01", "0.1", "1"],
    )
    ax.minorticks_off()
    ax.tick_params(axis="x", labelsize=7)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    fig.text(0.51, 0.10, "particles cm⁻³ s⁻¹", size=8, ha="center")
    fig.savefig(output / "formation-overview.png", dpi=300)
    plt.close(fig)


def formation(output):
    """Compare the two fitted rate laws and the proposed 223 K acid scan."""
    form = read("formation-bounds")
    low = form.loc[form.quantity == "formation_rate_cm-3_s-1"]
    score = form.loc[form.quantity == "holdout_error_factor"]
    design = read("design-and-reference")
    scan = design.loc[design.quantity == "acid_step_rate_ratio"]
    archive = design.loc[design.quantity == "log_rate_R2"].set_index("subset")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "figure.dpi": 155,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.6, 4.7),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.1, 1]},
    )
    order = [366, 332, 334, 344, 336, 338]
    ax = axes[0]
    for idx, topo in enumerate(RATE_COLORS):
        bands = (
            low.loc[low.topology == topo]
            .set_index("line")
            .loc[order, QUANTILES]
            .to_numpy()
        )
        yy = np.arange(6) + (idx - 0.5) * 0.22
        ax.hlines(
            yy, bands[:, 0], bands[:, 2], color=RATE_COLORS[topo], linewidth=2.5
        )
        ax.scatter(
            bands[:, 1], yy, color=RATE_COLORS[topo], s=26, label=LABELS[topo]
        )
    ax.scatter(
        low.drop_duplicates("line").set_index("line").loc[order, "observed"],
        range(6),
        s=37,
        marker="x",
        c="black",
        label="CLOUD central rate",
        zorder=4,
    )
    ax.set(
        yticks=range(6),
        yticklabels=low.drop_duplicates("line")
        .set_index("line")
        .loc[order, "condition"],
        xscale="log",
        xlabel="Formation rate at 1.7 nm (cm⁻³ s⁻¹)",
        title="Rates on withheld conditions",
        ylabel="Temperature · H₂SO₄ (10⁶ molecules cm⁻³)",
    )
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[1]
    order2 = [
        "223_low",
        "248_low",
        "223_temperature",
        "248_temperature",
        "278_temperature",
    ]
    for idx, topo in enumerate(RATE_COLORS):
        vals = (
            score.loc[score.topology == topo]
            .set_index("split")
            .loc[order2, QUANTILES]
            .to_numpy()
        )
        yy = np.arange(5) + (idx - 0.5) * 0.22
        ax.hlines(
            yy, vals[:, 0], vals[:, 2], color=RATE_COLORS[topo], linewidth=2.5
        )
        ax.scatter(vals[:, 1], yy, s=26, c=RATE_COLORS[topo])
    ax.set(
        yticks=range(5),
        yticklabels=[
            "223 K low NH₃",
            "248 K low NH₃",
            "223 K all",
            "248 K all",
            "278 K all",
        ],
        xscale="log",
        xlabel="Median error factor within holdout",
        title="Transfer to a new condition or temperature",
    )
    ax.invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", alpha=0.18)
    fig.savefig(output / "formation-bounds.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4), layout="constrained")
    ax = axes[0]
    for k, g in enumerate(
        [
            scan.query('split=="complete"'),
            scan.query('split=="223_low"'),
        ]
    ):
        for n, sc in enumerate(("new_common", "new_differential")):
            for j, top in enumerate(RATE_COLORS):
                vals = (
                    g.loc[(g.scenario == sc) & (g.topology == top), QUANTILES]
                    .iloc[0]
                    .to_numpy()
                )
                y = 2 * k + n + 0.22 * (j - 0.5)
                ax.hlines(
                    y, vals[0], vals[2], color=RATE_COLORS[top], linewidth=2.4
                )
                ax.scatter(
                    vals[1],
                    y,
                    c=RATE_COLORS[top],
                    s=27,
                    label=LABELS[top] if k + n == 0 else None,
                )
    ax.axvline(
        scan.observed.iloc[0], c="black", ls=":", label="Archived ratio (49.4)"
    )
    ax.set(
        xscale="log",
        yticks=range(4),
        yticklabels=[
            "Complete, shared",
            "Complete, separate",
            "Held out, shared",
            "Held out, separate",
        ],
        xlabel="Rate at 13.05 / rate at 6.64\nH₂SO₄ in 10⁶ molecules cm⁻³",
        title="223 K repeat at 0.06 pptv NH₃",
    )
    ax.set_ylim(4.4, -0.55)
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[1]
    keys = ["figure45", "all", "primary", "neutral", "ionized"]
    vals = archive.loc[keys, QUANTILES].to_numpy()
    ax.hlines(range(5), vals[:, 0], vals[:, 2], color="#41685c", lw=3)
    ax.scatter(
        vals[:, 1], range(5), color="#41685c", label="Joint calibrations"
    )
    ax.scatter(
        archive.loc[keys, "nominal"],
        range(5),
        marker="x",
        color="black",
        label="Nominal calibration",
    )
    ax.set(
        xlabel="R² on log₁₀ formation rates",
        yticks=range(5),
        yticklabels=[
            "Fig. S3 selection",
            "All 377",
            "33–43% RH",
            "Neutral, RH window",
            "Ionized, RH window",
        ],
        title="Published law with fixed coefficients",
    )
    ax.set_ylim(5.45, -0.55)
    ax.legend(fontsize=8, loc="lower left")
    for ax in axes:
        ax.grid(axis="x", alpha=0.18)
    fig.savefig(output / "design-and-reference.png", bbox_inches="tight")
    plt.close(fig)


def clusters(output):
    """Plot molecular survival and neutral dimer comparisons."""
    summaries = read("cluster-bounds")
    data = summaries.loc[summaries.quantity == "pentamer_probability"]
    dimer = summaries.loc[summaries.quantity == "model_over_observation"]
    thermos = ("optimized", "posterior_median")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")

    colors = ["#216675", "#bb5236", "#635482"]
    labs = ["2019 optimized", "2019 marginal medians", "2015 adduct"]
    labels = [
        (
            str(i),
            data.loc[data.case == str(i), "condition"]
            .iloc[0]
            .replace(" · ", "\n"),
        )
        for i in LINES
    ]
    channels = [
        (labels, data, thermos),
        (
            [
                (f"CLOUD5_{i}", s)
                for i, s in enumerate(["7.3 pptv", "4.4 pptv", "2.3 pptv"], 1)
            ]
            + [("flowtube_265K", "265 K")],
            dimer,
            (*thermos, "2015_adduct"),
        ),
    ]
    for ax, (ticks, rows, chems) in zip(axes, channels, strict=True):
        for c, chem in enumerate(chems):
            for j, (key, _) in enumerate(ticks):
                item = rows.loc[
                    (rows.case == key) & (rows.chemical == chem)
                ].iloc[0]
                lo, med, hi = (float(item[k]) for k in ("q05", "q50", "q95"))
                ax.errorbar(
                    j + (c - (len(chems) - 1) / 2) * 0.18,
                    med,
                    yerr=[[med - lo], [hi - med]],
                    fmt="o",
                    ms=5,
                    capsize=3,
                    color=colors[c],
                    lw=1.5,
                )
        ax.set(
            xticks=range(len(ticks)),
            xticklabels=[s for _, s in ticks],
            yscale="log",
        )
        ax.grid(True, axis="y", alpha=0.2)
    axes[0].set(
        title="A   Growth from an acid–ammonia pair",
        xlabel="Temperature and H₂SO₄ (10⁶ molecules cm⁻³)",
        ylabel="Probability of reaching five acids",
    )
    axes[1].set(
        title="B   Neutral dimer abundance",
        xlabel="248 K CLOUD ammonia or flow-tube temperature",
        ylabel="Predicted / measured dimer concentration",
    )
    axes[0].tick_params(axis="x", labelsize=8)
    axes[1].axhline(1, color=".3", ls="--", lw=1)
    fig.legend(
        handles=[
            Line2D([], [], marker="o", color=col, ls="", label=lab)
            for col, lab in zip(colors, labs, strict=True)
        ],
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    fig.savefig(output / "cluster-bounds.png", dpi=180)
    plt.close(fig)


def transport(output):
    """Draw finite-volume crossing frequencies beside survival and flux."""
    frame = read("transport-bounds")
    colors = {"vapor": "#346997", "attraction": "#af542c"}

    fig, axes = plt.subplots(1, 3, figsize=(12, 5.6), sharey=True)
    layout = [
        ("survival10", "10 nm survival", True),
        (
            "J10",
            "10 nm flux, full source\n(cm⁻³ s⁻¹)",
            True,
        ),
        (
            "zero10",
            "Simulations with no particles reaching\n"
            "10 nm during hours 12–18 (%)",
            False,
        ),
    ]
    laws = list(colors)
    for ax, (key, label, log) in zip(axes, layout, strict=True):
        for j, law in enumerate(laws):
            yy = np.arange(len(LINES)) + (j - 0.5) * 0.27
            for y, line in zip(yy, LINES, strict=True):
                row = frame.loc[(frame.line == line) & (frame.law == law)].iloc[
                    0
                ]
                if log:
                    lo, mid, hi = (row[f"{key}_{q}"] for q in QUANTILES)
                    ax.plot([lo, hi], [y, y], color=colors[law], lw=2)
                    ax.scatter(mid, y, color=colors[law], s=24, zorder=3)
                else:
                    ax.scatter(
                        100 * row.zero10, y, color=colors[law], s=24, zorder=3
                    )
        if log:
            ax.set_xscale("log")

            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        else:
            ax.set_xlim(-4, 105)
            ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel(label, fontsize=10)
        ax.grid(axis="x", alpha=0.23)
        ax.spines[["right", "top"]].set_visible(False)
        ax.axhline(1.5, color=".82", lw=1)
    axes[0].set_yticks(
        np.arange(6),
        frame.drop_duplicates("line").set_index("line").loc[LINES, "condition"],
    )
    axes[0].invert_yaxis()
    axes[0].set_title("Temperature · H₂SO₄*", loc="left", fontsize=10)

    handles = [
        Line2D([], [], marker="o", color=colors[law], label=lab)
        for law, lab in zip(
            laws,
            ["Vapor transfer", "Finite acid size + attraction"],
            strict=True,
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
        "* Archived acid in 10⁶ molecules cm⁻³. "
        "Points and lines show the median and central 90% of 128 draws "
        "in the first two panels.",
        fontsize=9,
    )
    volumes = frame.loc[frame.line == 336].iloc[0]
    fig.text(
        0.14,
        0.045,
        f"Panel 3 uses finite samples. At 248.3 K and 18 × 10⁶ acid molecules "
        f"cm⁻³, simulated volumes span {volumes.volume_min_cm3:.3f}–"
        f"{volumes.volume_max_cm3:.2f} cm³.",
        fontsize=9,
    )
    fig.savefig(output / "transport-bounds.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def warm(output):
    """Plot warm particle counts, growth-law controls and paired errors."""
    counts = read("warm-ensemble")
    errors = read("warm-paired-errors").set_index(["period", "family"])
    controls = read("warm-growth-controls").set_index(["quantity", "family"])
    h = counts.hour.to_numpy()

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(
        2, 1, sharey=True, figsize=(8.3, 6.2), layout="constrained"
    )
    curves = [
        ("gas_matched", "#067b75", "-", "Model · reconstructed acid"),
        ("measured", "#361a56", (0, (5, 3)), "Model · measured acid"),
    ]
    for ax, (start, end), title, marker_stride in zip(
        axes,
        [(0, 6.0001), (6.36, h[-1])],
        ["Early growth and peak", "Later chamber operation"],
        [6, 20],
        strict=True,
    ):
        mask = (h >= start) & (h <= end)
        time = h[mask]
        if start == 0:
            ax.axvspan(0, 3.8, color="#efefef", zorder=0)
            ax.text(
                1.9,
                0.92,
                "Training interval",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                color="#505050",
                fontsize=10,
            )
        for family, color, style, label in curves:
            low, median, high = (
                counts.loc[mask, f"{family}_{q}"].to_numpy() / 1000
                for q in QUANTILES
            )
            ax.fill_between(time, low, high, color=color, alpha=0.15, lw=0)
            ax.plot(
                time, median, color=color, linestyle=style, lw=2.2, label=label
            )
        ax.plot(
            time,
            counts.observed.to_numpy()[mask] / 1000,
            color="#222222",
            lw=1.1,
            marker="o",
            markerfacecolor="white",
            markeredgecolor="#151515",
            markeredgewidth=1.05,
            markersize=4.4,
            markevery=marker_stride,
            label="CLOUD particle counts",
            zorder=4,
        )
        ax.set(
            xlim=(time[0], time[-1]),
            ylim=(0, 24),
            yticks=range(0, 25, 6),
            ylabel="Particles ≥9.7 nm\n(10³ cm⁻³)",
        )
        ax.set_title(title, loc="left", fontsize=12, weight="semibold")
        ax.grid(axis="y", color=".87", lw=0.7)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("Hours since first size measurement", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        [handles[i] for i in (2, 1, 0)],
        [labels[i] for i in (2, 1, 0)],
        loc="outside upper center",
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=3,
        columnspacing=1.15,
    )
    fig.savefig(output / "warm-ensemble.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(
        1, 2, figsize=(9, 4.4), layout="constrained", sharey=True
    )
    subset = [
        "gas_matched",
        "gas_absolute",
        "ozone",
        "peroxide",
        "both",
        "bulk",
        "dimer",
        "measured_clear",
        "gas_matched_clear",
    ]
    display = [
        "OH + SO₂, matched mean",
        "OH + SO₂, absolute",
        "Effective ozone",
        "Effective peroxide",
        "Both effective oxidants",
        "Aqueous pH 2–6",
        "Gas dimers",
        "Clearing sensitivity",
        "OH + SO₂, with clearing*",
    ]
    for ax, period in zip(axes, ["peak", "later"], strict=True):
        ax.axvline(0, color="#333333", lw=0.9)
        for i, f in enumerate(subset):
            q = errors.loc[(period, f), QUANTILES].to_numpy() / 1000
            ax.plot([q[0], q[2]], [i, i], lw=2, color="#55819a")
            ax.scatter(q[1], i, s=25, color="#182f47", zorder=4)
        ax.set(
            yticks=range(len(subset)),
            yticklabels=display,
            xlabel="Paired RMSE change (10³ cm⁻³)",
            title="Peak (3.8–6 h)"
            if period == "peak"
            else "Later (6.36–32.97 h)",
        )
        ax.grid(axis="x", alpha=0.2)
    axes[0].invert_yaxis()
    fig.savefig(output / "warm-paired-errors.png", dpi=190)
    plt.close(fig)
    comparisons = [
        ("gas_transfer", "Gas transfer, fixed composition"),
        ("finite_hard", "Hard collisions, fixed composition"),
        ("vdw_fixed", "Attractive collisions, fixed composition"),
        ("empirical", "Empirical small-size growth"),
        ("measured", "Attractive collisions, wet salt"),
        ("extended_attraction", "Wet salt, attraction through 19.5 nm"),
        ("ammoniation", "Wet salt, variable ammoniation"),
    ]
    combine, (peak_ax, later_ax) = plt.subplots(
        1, 2, figsize=(10, 4.3), layout="constrained", sharey=True
    )
    peak_ax.axvline(
        controls.observed_max.max() / 1000,
        color="#101010",
        ls="--",
        lw=1,
        label="CLOUD peak",
    )
    for row, (f, _) in enumerate(comparisons):
        for ax, key in [
            (peak_ax, "peak_count_cm-3"),
            (later_ax, "later_RMSE_cm-3"),
        ]:
            lo, med, hi = controls.loc[(key, f), QUANTILES].to_numpy() / 1000
            c = "#087f6a" if f == "measured" else "#3b5877"
            ax.plot([lo, hi], [row, row], color=c, lw=2.4)
            ax.scatter(med, row, color=c, s=26)
    peak_ax.set(
        yticks=range(len(comparisons)),
        yticklabels=[v for _, v in comparisons],
        title="Peak concentration ≥9.7 nm",
        xlabel="Number (10³ cm⁻³)",
    )
    later_ax.set(title="Later count error at ≥9.7 nm", xlabel="RMSE (10³ cm⁻³)")
    peak_ax.invert_yaxis()
    peak_ax.legend(frameon=False, fontsize=8)
    for ax in (peak_ax, later_ax):
        ax.grid(axis="x", alpha=0.2)
    combine.savefig(output / "warm-growth-controls.png", dpi=190)
    plt.close(combine)


def main():
    """Write the figures to the requested directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dunne-figures"))
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    for render in (overview, formation, clusters, transport, warm):
        with plt.rc_context():
            render(output)


if __name__ == "__main__":
    main()
