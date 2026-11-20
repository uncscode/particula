"""Pure plotting of the retained pentane-drop study report."""

import matplotlib.pyplot as plt
import numpy as np

BLUE, GREEN, ORANGE, RED = "#2463a6", "#198269", "#bd7319", "#ab3747"
STYLE = {
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.labelcolor": "#243344",
    "text.color": "#243344",
}


@plt.rc_context(STYLE)
def plot_curves(report):
    """Render saved numerical results without running physics."""
    t = np.array(report["temperatures_K"])
    curves = np.array(report["curves"]) * 1000
    summary = report["summary"]
    roots = report["intersections"]
    fig, axs = plt.subplots(1, 2, figsize=(11.7, 4.8), layout="constrained")
    for ax in axs:
        for i, color, style, label in [
            (0, BLUE, "-", "Mass transport: 1984 correlation"),
            (1, ORANGE, "--", "Mass transport: Stefan film substitution"),
            (3, GREEN, "-", "Heat supply: 1984 blowing correction"),
            (2, "#727a85", ":", "Heat supply: no blowing correction"),
        ]:
            ax.plot(
                t, curves[0, :, i], color=color, ls=style, lw=2, label=label
            )
            ax.fill_between(
                t,
                curves[:, :, i].min(0),
                curves[:, :, i].max(0),
                color=color,
                alpha=0.12,
            )
        ax.axvline(263.6, color=RED, ls=":", lw=1.5)
        ax.set(
            xlabel="Surface temperature (K)",
            ylabel="Evaporation flux (g m⁻² s⁻¹)",
        )
        ax.grid(alpha=0.18)
    j = np.array(summary["observed_ranges"]["printed_proxy_kg_m2_s"]) * 1000
    axs[0].errorbar(
        263.6,
        j.mean(),
        yerr=[[j.mean() - j[0]], [j[1] - j.mean()]],
        fmt="o",
        color=RED,
        capsize=5,
    )
    axs[0].annotate(
        "Reported-condition flux\nconverted with spherical proxy",
        (263.6, j.mean()),
        xytext=(269, 36),
        arrowprops={"arrowstyle": "-", "color": RED},
        fontsize=9,
    )
    axs[0].set(
        xlim=(250, 284),
        ylim=(0, 47),
        title="Reported flux exceeds both model estimates",
    )
    fig.legend(
        *axs[0].get_legend_handles_labels(),
        loc="outside upper center",
        ncol=2,
        fontsize=8,
        frameon=False,
    )
    selected = [r for r in roots if r["mass"] == r["heat"] == "paper"]
    solutions = axs[1].scatter(
        [float(r["Ts_K"]) for r in selected],
        [1000 * float(r["flux_kg_m2_s"]) for r in selected],
        s=20,
        color="#243344",
        zorder=5,
        label="Coupled solutions: 16 property variants",
    )
    axs[1].text(263.85, 9.4, "Reported\n263.6 K", color=RED, fontsize=9)
    axs[1].set(
        xlim=(256, 267),
        ylim=(9, 18),
        title="The heat–mass intersections imply a cooler drop",
    )
    axs[1].legend(handles=[solutions], loc="upper left", fontsize=8)
    return fig


@plt.rc_context(STYLE)
def plot_sensitivity(report):
    """Render saved numerical results without running physics."""
    roots = report["intersections"]
    pr_roots = report["printed_pr"]
    fig, ax = plt.subplots(figsize=(9.5, 5.3), layout="constrained")
    pairs = [
        (m, h) for m in ["paper", "film"] for h in ["no_blow", "paper", "film"]
    ]
    ml = {"paper": "1984", "film": "Stefan"}
    hl = {"no_blow": "No correction", "paper": "1984", "film": "Stefan"}
    for y, (mass, heat) in enumerate(pairs):
        for rows, offset, color, marker, label in [
            (roots, -0.12, BLUE, "o", "Local mixture Prandtl number"),
            (pr_roots, 0.12, ORANGE, "s", "Reported Prandtl number = 0.86"),
        ]:
            vals = [
                float(r["Ts_K"])
                for r in rows
                if r["mass"] == mass and r["heat"] == heat
            ]
            lo, hi = min(vals), max(vals)
            ax.plot(
                [lo, hi],
                [y + offset] * 2,
                color=color,
                lw=3,
                solid_capstyle="round",
            )
            ax.scatter(
                vals,
                np.full(len(vals), y + offset),
                color=color,
                marker=marker,
                s=14,
                label=label if y == 0 else None,
                zorder=3,
            )
    ax.axvline(263.6, color=RED, ls="--", label="Reported surface temperature")
    ax.set(
        yticks=range(6),
        yticklabels=[f"Mass: {ml[m]}   |   Heat: {hl[h]}" for m, h in pairs],
        xlabel="Predicted equilibrium surface temperature (K)",
        xlim=(257.5, 264.2),
        title="Equilibrium predictions remain cooler across model variants",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=1,
        fontsize=8,
        frameon=False,
    )
    return fig


@plt.rc_context(STYLE)
def plot_heat_budget(report):
    """Render saved numerical results without running physics."""
    summary = report["summary"]
    observed = report["observed"]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8), layout="constrained")

    def values(key):
        return np.array([float(r[key]) for r in observed])

    latent = values("latent_J_kg")
    energy = [
        values("q_convection_no_blow_W_m2"),
        values("mass_paper_kg_m2_s") * latent,
        values("printed_proxy_kg_m2_s") * latent,
        np.array([summary["radiation_blackbody_at_326K_W_m2"]]),
    ]
    labels = [
        "Convection\nno blowing",
        "Latent demand\n1984 mass model",
        "Latent demand\nreported flux proxy",
        "Radiation bound\n326 K walls",
    ]
    for i, (vals, color) in enumerate(
        zip(energy, [GREEN, BLUE, RED, ORANGE], strict=True)
    ):
        lo, hi = vals.min() / 1000, vals.max() / 1000
        axs[0].bar(i, (lo + hi) / 2, color=color, alpha=0.9, width=0.65)
        axs[0].errorbar(
            i,
            (lo + hi) / 2,
            yerr=(hi - lo) / 2,
            fmt="none",
            color="#243344",
            capsize=4,
        )
        label = f"{lo:.2f}–{hi:.2f}" if hi > lo else f"≤ {hi:.2f}"
        axs[0].text(i, hi + 0.35, label, ha="center", fontsize=8)
    axs[0].set(
        xticks=range(4),
        xticklabels=labels,
        ylim=(0, 18),
        ylabel="Heat flux (kW m⁻²)",
        title="At the reported surface temperature",
    )
    axs[0].tick_params(axis="x", labelsize=8)
    for i, (key, color) in enumerate(
        [
            ("extra_for_mass_paper_W_m2", BLUE),
            ("extra_for_printed_proxy_W_m2", RED),
        ]
    ):
        vals = values(key) / 1000
        axs[1].barh(i, vals.mean(), color=color, height=0.42)
        axs[1].errorbar(
            vals.mean(),
            i,
            xerr=[[vals.mean() - vals.min()], [vals.max() - vals.mean()]],
            fmt="none",
            color="#243344",
            capsize=5,
        )
        axs[1].text(
            vals.max() + 0.3,
            i,
            f"{vals.min():.2f}–{vals.max():.2f}",
            va="center",
            fontsize=9,
        )
    axs[1].set(
        yticks=[0, 1],
        yticklabels=["1984 mass\nmodel", "Reported\nflux proxy"],
        xlabel="Additional net heat relative to convection (kW m⁻²)",
        xlim=(0, 13.5),
        title="The reported flux needs much more heat",
    )
    axs[1].invert_yaxis()
    axs[0].grid(axis="y", alpha=0.15)
    axs[1].grid(axis="x", alpha=0.15)
    return fig
