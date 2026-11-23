"""Draw paired date intervals, hourly ranges and calibration cases."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#243e43"
TEAL = "#126878"
RUST = "#aa4e34"
PURPLE = "#806294"
STYLE = [
    ("measured", "Observed", INK, "o"),
    ("Xu", "Xu two-class", RUST, "s"),
    ("distributed", "Distributed κ", TEAL, "^"),
    ("film_common", "Fitted film", PURPLE, "D"),
]
IMAGES = Path(__file__).resolve().parents[2] / "docs/images/antarctic-ccn"


def whisker(ax, x, row, color, marker, label=None):
    """Point estimate and percentile endpoints on one vertical scale."""
    ax.vlines(x, row.lo95, row.hi95, color=color, lw=1.5)
    ax.scatter(
        x, row["mean"], s=27, marker=marker, color=color, label=label, zorder=3
    )


def spectrum(report):
    """Compare matched hours, retrospective dates and hourly spread."""
    data = pd.DataFrame(report["spectra"])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for ax, (sample, title) in zip(
        axes[0],
        [
            ("all", "All 373 matched hours"),
            ("held-out", "78 retrospective test hours"),
        ],
        strict=False,
    ):
        settings = np.arange(1, 6) * 0.2
        panel = data[data["sample"] == sample].pivot(
            index="model", columns="metric", values=["mean", "lo95", "hi95"]
        )
        metrics = ["C2", "C4", "C6", "C8", "C10"]
        observed = panel.loc["measured"]
        measured = observed["mean"][metrics].to_numpy()
        ax.errorbar(
            settings,
            measured,
            yerr=[
                measured - observed["lo95"][metrics].to_numpy(),
                observed["hi95"][metrics].to_numpy() - measured,
            ],
            fmt="o",
            color=INK,
            markerfacecolor="white",
            markeredgewidth=1.5,
            markersize=6,
            elinewidth=1.4,
            capsize=3,
            zorder=5,
            label="Observed",
        )
        for (key, name, color, _), linestyle in zip(
            STYLE[1:], ["--", "-.", "-"], strict=True
        ):
            model = panel.loc[key]
            ax.fill_between(
                settings,
                model["lo95"][metrics],
                model["hi95"][metrics],
                color=color,
                alpha=0.13,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                settings,
                model["mean"][metrics],
                color=color,
                linestyle=linestyle,
                linewidth=2,
                label=name,
                zorder=3,
            )
        ax.set(
            title=title,
            xlabel="Nominal supersaturation (%)",
            ylabel="Mean CCN (cm⁻³)",
            ylim=(0, 240),
            xticks=np.arange(1, 6) * 0.2,
        )
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    ax = axes[1, 0]
    held = data[data["sample"] == "held-out"]
    for j, (key, _name, color, mark) in enumerate(STYLE):
        rows = held[held.model == key].set_index("metric")
        for i, m in enumerate(["rise24", "rise46"]):
            whisker(ax, i + (j - 1.5) * 0.11, rows.loc[m], color, mark)
    ax.set(
        title="Uncertainty in mean rise · held-out dates",
        xticks=[0, 1],
        xticklabels=["0.2 → 0.4%", "0.4 → 0.6%"],
        ylabel="Mean rise (CCN cm⁻³)",
        xlim=(-0.5, 1.5),
    )
    ax = axes[1, 1]
    for j, (key, _name, color, mark) in enumerate(STYLE):
        r = held[(held.model == key) & (held.metric == "C4")].iloc[0]
        ax.plot([r.hourly_p10, r.hourly_p90], [j, j], color=color, lw=3)
        ax.scatter(r["mean"], j, marker=mark, color=color, zorder=3)
    ax.set(
        title="Hourly spread at 0.4% · held-out dates",
        xlabel="CCN (cm⁻³), 10th–90th percentiles",
        yticks=range(4),
        yticklabels=[x[1] for x in STYLE],
    )
    ax.invert_yaxis()
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.18)
    return fig


def size_effects(report):
    """Show paired signed and absolute-bias changes for each band."""
    bands = pd.DataFrame(report["bands"])
    env = pd.DataFrame(report["envelope"]).set_index("lo_nm")
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4.3), layout="constrained", sharex=True
    )
    for ax, key, title in zip(
        axes,
        ["change", "change_abs_bias"],
        ["Signed change in each rise", "Change in absolute mean bias"],
        strict=False,
    ):
        for j, (metric, label, color, mk) in enumerate(
            [
                ("rise24", "0.2 → 0.4%", TEAL, "o"),
                ("rise46", "0.4 → 0.6%", RUST, "D"),
            ]
        ):
            rows = bands[
                (bands.metric == metric) & (bands.contrast == key)
            ].sort_values("lo")
            for i, (_, r) in enumerate(rows.iterrows()):
                x = i + (j - 0.5) * 0.28
                whisker(ax, x, r, color, mk, label if i == 0 else None)
                if key == "change":
                    v = env.loc[r.lo]
                    ax.plot(
                        [x + 0.07] * 2,
                        [v[metric + "_lo"], v[metric + "_hi"]],
                        ls=":",
                        color=color,
                        lw=2.3,
                    )
        ax.axhline(0, color=INK, lw=0.8)
        ax.grid(axis="y", alpha=0.18)
        ax.set(
            title=title,
            xticks=range(4),
            xticklabels=["35–40", "40–60", "60–75", "75–165"],
            xlabel="Fully coated dry-size band (nm)",
            ylabel="Change (CCN cm⁻³)",
        )
    axes[0].plot([], [], ls=":", c=INK, label="Calibration range")
    axes[0].legend(frameon=False, fontsize=8)
    return fig


def attribution(report):
    """Compare field ranges with labelled synthetic recovery."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    rows = pd.DataFrame(report["effects"]).set_index(["model", "outcome"])
    labels = [
        "Salt deficit",
        "Fitted film addition",
        "Full-coating addition",
        "Remaining with fitted film",
    ]
    for i, key in enumerate(
        [
            ("salt", "unexplained"),
            ("film_common", "uplift"),
            ("maximum", "uplift"),
            ("film_common", "unexplained"),
        ]
    ):
        r = rows.loc[key]
        ax.plot(
            [r.sampling_lo, r.sampling_hi],
            [i - 0.09] * 2,
            c=TEAL,
            lw=2,
            label="Paired 95% date interval" if not i else None,
        )
        ax.scatter(r.point, i - 0.09, c=TEAL, s=25)
        ax.plot(
            [r.compatible_lo, r.compatible_hi],
            [i + 0.09] * 2,
            c=RUST,
            ls="--",
            lw=2,
            label="Calibration range" if not i else None,
        )
    ax.set(
        title="Mean effect at 0.4%",
        yticks=range(4),
        yticklabels=labels,
        xlabel="Held-out CCN (cm⁻³)",
        ylim=(3.7, -0.45),
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    recovery = pd.DataFrame(report["recovery"])
    for j, (method, color, marker) in enumerate(
        [("empirical", TEAL, "o"), ("gaussian", RUST, "D")]
    ):
        rr = recovery[recovery.noise == method].sort_values("effect_lo")
        for i, r in enumerate(rr.itertuples()):
            x = i + (j - 0.5) * 0.12
            bx.vlines(x, 100 * r.wilson_lo, 100 * r.wilson_hi, color=color)
            bx.scatter(
                x,
                100 * r.rate,
                color=color,
                marker=marker,
                label=method.capitalize() + " noise" if not i else None,
            )
    bx.axhline(80, c=INK, ls=":", lw=0.8)
    bx.set(
        title="Recovery in synthetic campaigns",
        yticks=[0, 20, 40, 60, 80, 100],
        ylim=(-3, 101),
        ylabel="Film detected (%)",
        xlabel="Imposed full-calendar effect (CCN cm⁻³)",
        xticks=range(3),
        xticklabels=["No film", "3–6", "6–7.9"],
    )
    bx.legend(frameon=False, fontsize=8)
    bx.grid(axis="y", alpha=0.18)
    return fig
