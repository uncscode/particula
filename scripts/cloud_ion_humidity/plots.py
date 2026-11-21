"""Figures for the CLOUD humidity comparison and proposed experiment."""

import matplotlib.pyplot as plt
import numpy as np

INK = "#243849"
BLUE = "#186b80"
RUST = "#af4627"


def style(ax):
    """Keep units and light horizontal guides legible at article width."""
    ax.grid(axis="y", color="#d9e0e4", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def humidity_sweep(report):
    """Overlay digitized chamber observations, balances and CUDA ensembles."""
    sweep = report["humidity_sweep"]
    q = sweep["source_pairs_cm3_s"]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.6, 7.5),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for ax, panel in zip(axes.flat, sweep["panels"], strict=True):
        ax.fill_between(
            q,
            panel["lower_cm3"],
            panel["upper_cm3"],
            color="#b9dce7",
            label="Parameter sensitivity",
        )
        ax.plot(
            q,
            sweep["fixed_cm3"],
            ls="--",
            color="#747474",
            label="Fixed recombination",
        )
        ax.plot(
            q,
            panel["balance_cm3"],
            color=BLUE,
            lw=2,
            label="RH-specific balance",
        )
        points = panel["observations"]
        measured = np.array([p["ions_cm3"] for p in points])
        ax.errorbar(
            [p["q_cm3_s"] for p in points],
            measured,
            yerr=[
                measured - [p["bar_low_cm3"] for p in points],
                [p["bar_high_cm3"] for p in points] - measured,
            ],
            fmt="o",
            ms=5,
            mfc="white",
            mec=INK,
            ecolor=INK,
            capsize=2,
            label="CLOUD measurement",
        )
        runs = panel["ensembles"]
        ax.errorbar(
            [r["q"] for r in runs],
            [r["mean_cm3"] for r in runs],
            yerr=[1.96 * r["sem_cm3"] for r in runs],
            fmt="D",
            ms=4,
            color=RUST,
            capsize=3,
            label="CUDA ensemble",
        )
        ax.set_title(
            f"{panel['rh']}% relative humidity", loc="left", fontweight="bold"
        )
        ax.grid(alpha=0.15)
        ax.set(xlim=(0, 102), ylim=(0, 7600))
    fig.supxlabel("Ion-pair production (pairs cm⁻³ s⁻¹)")
    fig.supylabel("Mean concentration per polarity (ions cm⁻³)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside upper center", ncols=3, frameon=False
    )
    return fig


def comparison(report):
    """Plot chamber scatter, CUDA intervals and the physical ceiling."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), layout="constrained")
    for ax, cases, title in zip(
        axes,
        [report["cases"][:2], report["cases"][2:]],
        ["Near 68 pairs cm⁻³ s⁻¹", "At 2.64 pairs cm⁻³ s⁻¹"],
        strict=True,
    ):
        style(ax)
        for i, case in enumerate(cases):
            point = case["observation"]
            measured = point["ions_cm3"]
            ax.errorbar(
                i - 0.11,
                measured,
                [
                    [measured - point["bar_low_cm3"]],
                    [point["bar_high_cm3"] - measured],
                ],
                fmt="o",
                color=INK,
                capsize=5,
                label="CLOUD" if i == 0 else None,
            )
            ax.errorbar(
                i + 0.11,
                case["mean_cm3"],
                case["sem_cm3"] * 1.96,
                fmt="D",
                color=RUST,
                capsize=5,
                label="Particula" if i == 0 else None,
            )
        ax.set(
            xticks=range(len(cases)),
            xticklabels=[r["label"] + "% RH" for r in cases],
            xlim=(-0.52, len(cases) - 0.48),
            title=title,
            ylim=(0, None),
        )
    axes[0].set_ylabel("Mean ions per polarity (cm⁻³)")
    axes[0].legend(frameon=False, loc="upper left")
    ax = axes[1]
    ax.axhline(report["ceiling"], linestyle="--", color=BLUE)
    ax.annotate(
        "394 ceiling",
        (0.02, report["ceiling"]),
        xycoords=("axes fraction", "data"),
        xytext=(0, 5),
        textcoords="offset points",
        color=BLUE,
    )
    return fig


def recovery_figure(report):
    """Show synthetic response envelopes after verified clearing."""
    fig, ax = plt.subplots(figsize=(8.0, 3.9), layout="constrained")
    style(ax)
    for key, color, label, marker in [
        ("offset", RUST, "Readout offset", "o"),
        ("source", BLUE, "Extra production", "D"),
    ]:
        band = report[key]
        ax.fill_between(
            report["times"], band["low"], band["high"], color=color, alpha=0.18
        )
        for edge in ("low", "high"):
            ax.plot(
                report["times"],
                band[edge],
                color=color,
                lw=1.2,
                label=label if edge == "high" else None,
            )
            ax.plot(30, band[edge][30], marker=marker, ms=4, color=color)
    ax.axvline(30, color=INK, ls=":", lw=1)
    ax.set(
        xlabel="Time after release of the clearing field (s)",
        ylabel="Predicted reading (ions cm⁻³)",
        xlim=(0, 120),
        ylim=(0, 1650),
    )
    ax.legend(frameon=False, loc="upper left", ncol=2)
    return fig


def design_figure(report):
    """Show the least tested exposure contrast that passed each audit."""
    fig, ax = plt.subplots(figsize=(8.0, 3.8), layout="constrained")
    style(ax)
    rows = report["design"]
    gains = sorted({r["gain"] for r in rows})
    for noise, shift, color, marker, label in [
        (1, -0.10, BLUE, "o", "Reference noise"),
        (2, 0.10, RUST, "D", "Doubled noise"),
    ]:
        for i, gain in enumerate(gains):
            passed = [
                r["contrast"]
                for r in rows
                if r["noise"] == noise
                and r["gain"] == gain
                and r["lower95"] > 0.9
            ]
            y = 100 * min(passed) if passed else 119
            ax.scatter(
                i + shift,
                y,
                marker=marker,
                facecolors=color if passed else "white",
                edgecolors=color,
                linewidth=1.5,
                label=label if i == 0 else None,
            )
    ax.axhline(109, color="#a8b4ba", lw=0.8)
    ax.set(
        xticks=np.arange(len(gains)),
        xticklabels=[f"{100 * g:.0f}%" for g in gains],
        yticks=[0, 25, 50, 75, 100, 119],
        yticklabels=["0", "25", "50", "75", "100", "None passed"],
        ylim=(-6, 129),
        xlabel="Gain uncertainty at each inlet (±)",
        ylabel="Required reduction in plume fraction (%)",
    )
    ax.legend(frameon=False, loc="lower right")
    return fig
