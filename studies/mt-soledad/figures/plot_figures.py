"""Rebuild the manuscript's two SVG figures using saved plot tables."""

# %%
import argparse
from importlib.resources import files
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
IMAGES = HERE.parents[2] / "docs/images/mt-soledad"
INK, TEAL, ORANGE, PURPLE, GREY = (
    "#18322d",
    "#08786c",
    "#c2410c",
    "#4338ca",
    "#7c8985",
)
LABEL = {
    "measured": "Measured",
    "scalar": "Size-independent count loss",
    "delay": "Common growth delay",
    "full": "Full AIOMFAC",
    "separate": "Separate pools",
    "fitted": "Fitted dry class",
}
COLORS = {
    "measured": INK,
    "scalar": TEAL,
    "delay": ORANGE,
    "full": TEAL,
    "separate": ORANGE,
    "fitted": PURPLE,
}
MARKS = {
    "measured": "o",
    "scalar": "s",
    "delay": "^",
    "full": "s",
    "separate": "^",
    "fitted": "D",
}


def refresh(study_root):
    """Extract compact plot inputs from completed scientific stage outputs."""
    drops = study_root / "mechanisms/droplets"
    profiles = pd.read_csv(drops / "profiles.csv").query(
        "source=='selected' and model in ['scalar_loss','uniform_amplitude']"
    )
    vals = ["channel", "lower_um", "upper_um", "bypass", "denuded"]
    spectrum = profiles.pivot(
        index=["split", *vals], columns="model", values="predicted"
    ).reset_index()
    spectrum.rename(
        columns={"scalar_loss": "scalar", "uniform_amplitude": "delay"}
    ).to_csv(HERE / "optical-spectrum.csv", index=False, float_format="%.11g")
    src = pd.read_csv(drops / "scores.csv").query(
        "source=='selected' and floor==0.5"
    )
    records = []
    for key, model, metrics in [
        ("measured", "scalar_loss", ["observed_ratio", "obs_mean_shift"]),
        ("scalar", "scalar_loss", ["model_ratio", "model_mean_shift"]),
        ("delay", "uniform_amplitude", ["model_ratio", "model_mean_shift"]),
    ]:
        for split in ["train", "test"]:
            for quantity, metric in zip(
                ["count_ratio", "wet_shift_um"], metrics, strict=True
            ):
                r = src[
                    (src.model == model)
                    & (src.metric == metric)
                    & (src.split == split)
                ].iloc[0]
                records.append(
                    [
                        split,
                        key,
                        quantity,
                        r["mean"],
                        r.low,
                        r.high,
                        r.days,
                        r.blocks,
                        r.pairs,
                    ]
                )
    pd.DataFrame(
        records,
        columns=(
            "split series metric mean low95 high95 days windows cycles"
        ).split(),
    ).to_csv(HERE / "optical-summary.csv", index=False, float_format="%.11g")
    fixed = study_root / "field-mixtures/observations"
    cases = pd.read_csv(fixed / "partial-cases.csv.gz").query(
        "held and cut_nm==1000 and recipe=='low'"
    )
    cmp = pd.read_csv(
        study_root / "field-mixtures/comparison/growth-by-date.csv.gz"
    ).query(
        "held and cut_nm==1000 and clock=='raw_UTC' and closure=='sodium' "
        "and recipe=='low' and population=='uniform' and size_nm==60 "
        "and model in ['full','two_pool']"
    )
    dates = cases[["filter_id", "utc"]].drop_duplicates().copy()
    dates["date"] = pd.to_datetime(dates.utc, unit="s", utc=True).dt.strftime(
        "%Y-%m-%d"
    )
    date_key = dates[["date", "filter_id"]].drop_duplicates()
    if len(date_key) != 8 or date_key.filter_id.nunique() != 8:
        raise ValueError("Expected eight held dates on distinct filters")
    records = []
    for z in cmp.merge(
        date_key, on="date", validate="many_to_one"
    ).itertuples():
        records.append(
            [
                z.filter_id,
                60,
                "mean_GF",
                "full" if z.model == "full" else "separate",
                z.pred_mean,
            ]
        )
    reduced = cases.groupby(["filter_id", "size_nm", "model"], as_index=False)[
        ["gf_mean", "mean_pred", "gf_below_110", "tail_pred"]
    ].mean()
    for r in reduced.itertuples():
        if r.model == "partial":
            if r.size_nm == 60:
                records.extend(
                    [
                        [r.filter_id, 60, "mean_GF", "measured", r.gf_mean],
                        [r.filter_id, 60, "mean_GF", "fitted", r.mean_pred],
                    ]
                )
            records.append(
                [
                    r.filter_id,
                    r.size_nm,
                    "P_GF_le_1.10",
                    "measured",
                    r.gf_below_110,
                ]
            )
        records.append(
            [
                r.filter_id,
                r.size_nm,
                "P_GF_le_1.10",
                "fitted" if r.model == "partial" else "full",
                r.tail_pred,
            ]
        )
    pd.DataFrame(
        records, columns=["filter_id", "size_nm", "quantity", "series", "value"]
    ).to_csv(HERE / "growth-by-filter.csv", index=False, float_format="%.11g")
    intervals = pd.read_csv(fixed / "partial-held-intervals.csv")
    intervals[intervals.metric.isin(["mean_bias", "tail_bias"])].to_csv(
        HERE / "growth-intervals.csv", index=False, float_format="%.11g"
    )
    source = study_root / "field-mixtures/films/final/held-summary.csv"
    fields = ["observed_bypass_wet_um", "observed_wet_um", "observed_ratio"]
    reference = pd.read_csv(source)[fields].drop_duplicates()
    if len(reference) != 1:
        raise ValueError("Expected one common measured held optical reference")
    reference.to_csv(
        HERE / "toc-reference.csv", index=False, float_format="%.15g"
    )


def layout():
    """Apply the Particula style with type sized for a manuscript column."""
    plt.style.use(str(files("particula") / "default.mplstyle"))
    plt.rcParams.update(
        {
            "font.size": 9.2,
            "axes.titlesize": 11.2,
            "axes.labelsize": 9.4,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.3,
            "axes.titlepad": 7,
            "svg.fonttype": "none",
            "svg.hashsalt": "soledad-field-study-figures",
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "lines.linewidth": 1.8,
            "lines.solid_capstyle": "round",
            "figure.dpi": 120,
        }
    )


def save(fig, name):
    """Export the figure as editable, deterministically named SVG elements."""
    output = IMAGES / f"{name}.svg"
    fig.savefig(
        output,
        metadata={"Date": None, "Creator": "Particula literature study"},
        format="svg",
    )
    plt.close(fig)
    output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")


def optical():
    """Show how the calibrated mechanisms transfer to the higher setting."""
    s = pd.read_csv(HERE / "optical-spectrum.csv")
    q = pd.read_csv(HERE / "optical-summary.csv")
    fig = plt.figure(figsize=(7.55, 4.95))
    g = fig.add_gridspec(
        2,
        2,
        left=0.196,
        right=0.978,
        bottom=0.105,
        top=0.88,
        hspace=0.63,
        wspace=0.34,
        height_ratios=[1.50, 0.93],
    )
    axes = np.array(
        [
            [
                fig.add_subplot(cell)
                for cell in g[i, :].subgridspec(1, 2, wspace=0.34)
            ]
            for i in range(2)
        ]
    )
    channels = [
        ("bypass", "Bypass measured", GREY, "-", 1.0),
        ("denuded", "Denuded measured", INK, "-", 2.0),
        ("scalar", "Size-independent count loss", TEAL, "--", 1.6),
        ("delay", "Common growth delay", ORANGE, "-.", 1.6),
    ]
    for i, (split, title) in enumerate(
        [
            ("train", "A   Calibration at 0.4%"),
            ("test", "B   Held transfer at 0.8%"),
        ]
    ):
        a = axes[0, i]
        v = s[s.split == split].sort_values("channel")
        ed = np.r_[v.lower_um.iloc[0], v.upper_um]
        for key, lab, color, ls, lw in channels:
            if key == "bypass":
                a.stairs(v[key], ed, fill=True, color="#e3e9e6", lw=0)
            a.stairs(
                v[key], ed, label=lab, color=color, ls=ls, lw=lw, baseline=None
            )
        a.set(
            xticks=np.arange(2, 11, 2),
            xlim=(0.45, 10.05),
            ylim=(0, 0.60),
            yticks=[0, 0.2, 0.4, 0.6],
            xlabel="Wet outlet diameter (µm)",
        )
        a.set_title(title, loc="left")
    axes[0, 0].set_ylabel("Counts per bin / bypass total")
    handles = [
        Line2D([], [], color=c, ls=ls, lw=lw, label=lab)
        for _, lab, c, ls, lw in channels
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.99),
        ncol=4,
        fontsize=7.9,
        columnspacing=1.05,
        handlelength=2.4,
    )
    for i, (metric, title, xlabel, lim) in enumerate(
        [
            (
                "count_ratio",
                "C   Held count response",
                "Denuded / bypass count ratio",
                (0.72, 1.05),
            ),
            (
                "wet_shift_um",
                "D   Held survivor size",
                "Denuded − bypass mean diameter (µm)",
                (-1.78, 0.27),
            ),
        ]
    ):
        a = axes[1, i]
        v = q[(q.split == "test") & (q.metric == metric)].set_index("series")
        a.grid(False)
        a.grid(axis="x")
        a.set_axisbelow(True)
        for y, key in enumerate(["delay", "scalar", "measured"]):
            r = v.loc[key]
            mean = r["mean"]
            shown = 0 if abs(mean) < 5e-5 else mean
            a.errorbar(
                mean,
                y,
                xerr=[[max(0, mean - r.low95)], [max(0, r.high95 - mean)]],
                fmt=MARKS[key],
                color=COLORS[key],
                ms=5.3,
                capsize=3,
                elinewidth=1.4,
            )
            a.annotate(
                f"{shown:.3f}".replace("-", "−"),
                (mean, y),
                xytext=(-6 if i == 1 or key == "delay" else 9, 5),
                textcoords="offset points",
                ha="right" if i == 1 or key == "delay" else "left",
                va="bottom",
                fontsize=8.2,
                color=COLORS[key],
            )
        a.set(
            yticks=[0, 1, 2],
            yticklabels=[
                "Common growth\ndelay",
                "Size-independent\ncount loss",
                "Measured",
            ]
            if i == 0
            else [],
            ylim=(-0.5, 2.62),
            xlim=lim,
            xlabel=xlabel,
        )
        a.tick_params(axis="y", length=0)
        a.set_title(title, loc="left")
        a.spines["left"].set_visible(False)
    save(fig, "optical")


def growth():
    """Compare mean uptake and low-growth probability across held filters."""
    d = pd.read_csv(HERE / "growth-by-filter.csv")
    ci = pd.read_csv(HERE / "growth-intervals.csv")
    fig = plt.figure(figsize=(7.55, 4.9))
    g = fig.add_gridspec(
        2,
        2,
        left=0.105,
        right=0.978,
        bottom=0.10,
        top=0.92,
        hspace=0.57,
        wspace=0.32,
        height_ratios=[1, 0.64],
    )
    a = fig.add_subplot(g[0, 0])
    b = fig.add_subplot(g[0, 1])
    c = fig.add_subplot(g[1, :])
    v = d.query("quantity=='mean_GF'")
    order = ["measured", "full", "separate", "fitted"]
    a.grid(axis="y")
    a.set(
        ylim=(1.09, 1.67),
        xlim=(-0.53, 3.53),
        xticks=range(4),
        xticklabels=[
            "Measured",
            "Full\nAIOMFAC",
            "Separate\npools",
            "Fitted\ndry class",
        ],
        ylabel="Mean growth factor",
    )
    for x, key in enumerate(order):
        z = v[v.series == key].sort_values("filter_id").value
        offsets = np.linspace(-0.135, 0.135, len(z))
        a.scatter(
            x + offsets,
            z,
            s=14,
            color=COLORS[key],
            alpha=0.32,
            linewidths=0,
            zorder=2,
        )
        mu = z.mean()
        a.plot(
            x,
            mu,
            MARKS[key],
            color=COLORS[key],
            markeredgecolor="white",
            markeredgewidth=0.8,
            ms=7.2,
            zorder=4,
        )
        a.text(
            x,
            1.642,
            f"{mu:.3f}",
            ha="center",
            va="top",
            fontsize=8.4,
            color=COLORS[key],
            weight="bold",
        )
    a.set_title("A   Mean uptake at 60 nm", loc="left")
    v = (
        d.query("quantity=='P_GF_le_1.10'")
        .groupby(["series", "size_nm"])
        .value.mean()
        .unstack(0)
    )
    for key, ls in [("measured", "-"), ("full", "--"), ("fitted", "-.")]:
        b.plot(
            v.index,
            100 * v[key],
            marker=MARKS[key],
            ls=ls,
            color=COLORS[key],
            label=LABEL[key],
            ms=4.8,
        )
    b.set(
        xticks=[30, 60, 90],
        xlim=(24, 96),
        yticks=[0, 10, 20, 30, 40, 50],
        ylim=(-3, 61),
        xlabel="Dry mobility diameter (nm)",
        ylabel="Particles with GF ≤ 1.10 (%)",
    )
    b.legend(
        loc="upper right",
        fontsize=7.6,
        ncol=2,
        columnspacing=0.75,
        handlelength=1.7,
        borderpad=0,
        borderaxespad=0.25,
        labelspacing=0.22,
    )
    b.set_title("B   Frequency of weak growth", loc="left")
    p = 100 * (v["fitted"] - v["measured"])
    unc = ci.query("metric=='tail_bias'").set_index("size_nm")
    c.grid(False)
    c.axhline(0, color="#909f98", linewidth=0.8)
    for size in p.index:
        diff = p.loc[size]
        c.errorbar(
            size,
            diff,
            yerr=[
                [diff - 100 * unc.loc[size, "lo95"]],
                [100 * unc.loc[size, "hi95"] - diff],
            ],
            fmt="D",
            color=PURPLE,
            ms=5.3,
            capsize=3.5,
            lw=1.5,
        )
        c.annotate(
            f"{diff:+.1f} pp",
            (size, diff),
            xytext=(8, 1),
            textcoords="offset points",
            fontsize=8.4,
            va="center",
            color=PURPLE,
        )
    c.set(
        xticks=[30, 60, 90],
        xlim=(20, 103),
        ylim=(-7, 48),
        yticks=[0, 20, 40],
        xlabel="Dry mobility diameter (nm)",
        ylabel="Difference (pp)",
    )
    c.set_title(
        "C   Excess weak-growth probability in the fitted dry class model",
        loc="left",
    )
    save(fig, "growth")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--refresh",
        action="store_true",
        help="refresh compact plot data from the study results",
    )
    p.add_argument(
        "--study-root",
        type=Path,
        default=HERE.parent / "source",
        help="directory of the completed source-stage outputs",
    )
    args = p.parse_args()
    if args.refresh:
        refresh(args.study_root)
    IMAGES.mkdir(parents=True, exist_ok=True)
    layout()
    optical()
    growth()
