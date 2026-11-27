"""Draw the four study figures from the published comparison tables."""
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/xiong-sea-spray"
IMAGES = ROOT / "docs/images/xiong-sea-spray"
MODELS = [
    ("direct", "Direct sulfate", "#3563a3", "-"),
    ("parallel_neutral", "S(IV) + neutral sulfate", "#b34d24", "-"),
    (
        "retention_neutral",
        "Retained chloride + neutral sulfate",
        "#148172",
        "-",
    ),
    ("direct_blank", "Direct sulfate + chloride background", "#71417d", "--"),
]


def save(fig, name, dpi):
    fig.savefig(
        IMAGES / name, dpi=dpi, pil_kwargs={"lossless": True, "method": 6}
    )
    plt.close(fig)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.16)
    ax.set_axisbelow(True)


def plot_pathways():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#263143",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    fig.suptitle(
        "Xiong measurements (black) and Particula predictions (colored lines)\nFilled circles entered the fit; open circles show reserved observations",
        fontsize=12,
    )
    a, b = axes
    a.set_title("100 nm particles · ion chromatography", loc="left")
    b.set_title("200 nm particles · 80% readout humidity", loc="left")
    ions = pd.read_csv(DATA / "ion-time.csv")
    g = pd.read_csv(DATA / "growth-time.csv")
    for name, label, c, ls in MODELS:
        a.plot(ions.time_h, ions[name], color=c, ls=ls, lw=2, label=label)
        b.plot(g.time_h, g[name], color=c, ls=ls, lw=2)
    for ax, df, y, lo, hi in [
        (
            a,
            ions,
            "late_to_early_peak_area_ratio",
            ions.raster_edge_min,
            ions.raster_edge_max,
        ),
        (b, g, "gf", g.gf - g.spread, g.gf + g.spread),
    ]:
        for used in [True, False]:
            take = df.fitted == used
            ax.errorbar(
                df.time_h[take],
                df[y][take],
                yerr=[df[y][take] - lo[take], hi[take] - df[y][take]],
                marker="o",
                mfc="black" if used else "white",
                mec="black",
                color="black",
                capsize=3,
                linestyle="none",
                ms=5,
                zorder=8,
            )
        ax.set_xlabel("SO₂ exposure (hours)")
        style(ax)
    a.set_ylabel("Sulfate / chloride peak area")
    b.set_ylabel("Diameter growth factor")
    a.set_ylim(bottom=0)
    fig.legend(
        *a.get_legend_handles_labels(),
        loc="outside lower center",
        ncols=2,
        frameon=False,
    )
    save(fig, "pathways.webp", 170)


def plot_population():
    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    budgets = pd.read_csv(DATA / "size-budget.csv")
    groups = [
        ("direct", "Direct sulfate"),
        ("neutral", "Direct + neutral sulfate"),
        ("parallel_neutral", "S(IV) + neutral sulfate"),
        ("retention_neutral", "Retained chloride + neutral sulfate"),
    ]
    y = np.arange(len(groups))
    for i, (name, label) in enumerate(groups):
        row = budgets[budgets.network == name].iloc[0]
        v, w = row.sulfate200_ug_m3_decade, row.sulfate300_ug_m3_decade
        ax.plot([v, w], [i, i], lw=2, color="#bac4cd")
        for val, color, mark, diam in [
            (v, "#455478", "o", "200 nm"),
            (w, "#be5528", "s", "300 nm"),
        ]:
            ax.scatter(
                val,
                i,
                c=color,
                s=48,
                marker=mark,
                label=diam if i == 0 else None,
                zorder=3,
            )
            ax.text(
                val,
                i - 0.15,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_title("Particula-predicted sulfate after 3 h", loc="left", pad=26)
    ax.text(
        0,
        1.018,
        "Input: Rosati et al.’s measured mean particle number spectrum",
        transform=ax.transAxes,
        fontsize=9,
    )
    ax.set_xlabel("Predicted sulfate (µg m⁻³ per diameter decade)")
    ax.set_yticks(y, [l for _, l in groups])
    ax.set_xlim(0, 70)
    ax.set_ylim(len(y) - 0.5, -0.58)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, title="Dry particle diameter")
    save(fig, "population.webp", 170)


def plot_growth():
    data = pd.read_csv(DATA / "growth-size.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    fig.subplots_adjust(
        left=0.09, right=0.975, top=0.72, bottom=0.27, wspace=0.26
    )
    fig.text(
        0.09,
        0.95,
        "Measured and predicted water uptake by particle size",
        size=18,
        weight="bold",
    )
    fig.text(
        0.09,
        0.88,
        "NaCl particles aged for 3 h in 200 ppm SO₂ at 85% relative humidity, dried, then rehumidified.",
        size=10,
    )
    for ax, humidity in zip(axes, [60, 80]):
        rows = data[data.rh_pct == humidity].sort_values("diameter_nm")
        assert len(rows) == 7
        x = rows.diameter_nm
        measured = ax.errorbar(
            x,
            rows.gf,
            yerr=rows.spread,
            fmt="o",
            color="#223e54",
            mfc="white",
            mew=1.4,
            capsize=3,
            label="Measured by Xiong et al. (bars show reported spread)",
        )
        (predicted,) = ax.plot(
            x,
            rows.predicted_gf,
            color="#ad4920",
            marker="D",
            ms=4,
            lw=2,
            label="Predicted by Particula (AIOMFAC, S(IV) + neutral sulfate)",
        )
        ax.set_title(
            f"Particles rehumidified to {humidity}% RH", loc="left", size=11
        )
        ax.set_xticks([50, 150, 200, 250, 300, 350, 400])
        ax.set_xlabel("Initial dry particle diameter (nm)")
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Growth factor (wet diameter / dry diameter)")
    fig.legend(
        [measured, predicted],
        [measured.get_label(), predicted.get_label()],
        loc="lower center",
        bbox_to_anchor=(0.52, 0.062),
        ncol=2,
        fontsize=9,
        frameon=False,
    )
    fig.text(
        0.09,
        0.018,
        "Each point is a size-selected growth measurement. All 14 observations were excluded from the fit.",
        size=9,
    )
    save(fig, "growth-by-size.webp", 160)


def plot_phase():
    obs = pd.read_csv(DATA / "phase-onset.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.2), sharey=True)
    fig.subplots_adjust(
        left=0.095, right=0.983, top=0.78, bottom=0.22, wspace=0.12
    )
    fig.text(
        0.095,
        0.937,
        "When does a second liquid first appear during drying?",
        fontsize=17,
        weight="bold",
        color="#142839",
    )
    blue = "#135e77"
    orange = "#a94313"
    for ax, s, title in zip(
        axes,
        ["AS", "NaCl"],
        ["Ammonium sulfate · used in fit", "NaCl · held out of fit"],
    ):
        ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=12)
        p = obs[obs.salt == s]
        model = p.predicted_rh_pct.to_numpy()
        o = p[p.srh_status == "observed"]
        ax.errorbar(
            o.ovf_pct - 1.4,
            o.srh_pct,
            yerr=o.srh_sd_pct,
            fmt="o",
            capsize=4,
            markersize=7,
            color=blue,
            ecolor=blue,
            zorder=5,
        )
        ax.scatter(
            p.ovf_pct + 1.4, model, marker="D", s=44, color=orange, zorder=5
        )
        for x, y in zip(p.ovf_pct + 1.4, model):
            ax.annotate(
                f"{y:.1f}",
                (x, y),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=9,
                color=orange,
            )
        for x, y in zip(o.ovf_pct - 1.4, o.srh_pct):
            ax.annotate(
                f"{y:.1f}",
                (x, y),
                xytext=(-6, -4),
                ha="right",
                textcoords="offset points",
                fontsize=9,
                color=blue,
            )
        ax.axhline(85, color="#7c8790", ls=":", lw=1.1, zorder=0)
        ax.set_xticks([50, 70, 90])
        ax.set_xlim(40, 100)
        ax.set_ylim(58, 103)
        ax.set_xlabel("Dry organic volume fraction (%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e5e9ec")
    ax = axes[1]
    crystallization = obs[obs.srh_status == "not_observed"].iloc[0]
    ax.plot(
        crystallization.ovf_pct,
        crystallization.crystallization_rh_pct,
        "x",
        ms=8,
        mew=1.8,
        color=blue,
    )
    ax.annotate(
        "No split seen before\ncrystallization at 61.3%",
        (crystallization.ovf_pct, crystallization.crystallization_rh_pct),
        xytext=(-9, 2),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9,
        color=blue,
    )
    axes[0].set_ylabel("Relative humidity at onset (%)")
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=blue,
            label="Microscopy mean ± 1 SD",
        ),
        Line2D(
            [],
            [],
            marker="D",
            ls="",
            color=orange,
            label="Trial AIOMFAC prediction",
        ),
        Line2D([], [], ls=":", color="#7c8790", label="85% RH exposure"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.035),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    save(fig, "phase-transfer.webp", 180)


if __name__ == "__main__":
    plot_growth()
    plot_pathways()
    plot_population()
    plot_phase()
