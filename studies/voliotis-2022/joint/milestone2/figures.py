"""Render the campaign scores and late trajectories from saved measurements."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
IMAGES = HERE.parents[3] / "docs/images/manchester-soa"
COLORS = {"incumbent": "#116778", "selected": "#b44c16"}


def main():
    """Plot measured and fitted histories and scores for both frozen fits."""
    obs = pd.read_csv(HERE / "measurements.csv.gz")
    predicted = pd.read_csv(HERE / "trajectories.csv.gz")
    step = json.loads((HERE / "selection.json").read_text())[
        "qualification_step_seconds"
    ]
    setting = f"{step:g}s"
    scores = pd.read_csv(HERE / "scorecard.csv")
    scores = scores[scores.step == setting]
    # Plot coincident trajectories once when the parent survives unchanged.
    show = ["incumbent", "selected"]
    if np.array_equal(
        predicted.query("network=='incumbent'")[["mass", "number"]].values,
        predicted.query("network=='selected'")[["mass", "number"]].values,
    ):
        show = ["selected"]
    predicted = predicted[predicted.network.isin(show)]
    scores = scores[scores.network.isin(show)]
    fig, axs = plt.subplots(2, 2, figsize=(9, 5.7), layout="constrained")
    for j, r in enumerate([16, 24]):
        for i, (key, unit) in enumerate(
            [
                ("mass", "Suspended organic mass (µg m⁻³)"),
                ("number", "Particle number (10³ cm⁻³)"),
            ]
        ):
            ax = axs[i, j]
            data = obs.query("run==@r and observable==@key")
            scale = 1000 if key == "number" else 1
            ax.scatter(
                data.time_h,
                data.value / scale,
                s=11,
                color="#252525",
                label="Measured",
                zorder=3,
            )
            for net, g in predicted.query("run==@r").groupby(
                "network", sort=False
            ):
                ax.plot(
                    g.time_h,
                    g[key] / scale,
                    color=COLORS[net],
                    ls="--" if net == "selected" else "-",
                    label=net.capitalize(),
                )
            if j == 0:
                ax.set_ylabel(unit)
            if i == 1:
                ax.set_xlabel("Hours after lights-on")
            if i == 0:
                ax.set_title(
                    f"Run {r} · "
                    + ("α-pinene + o-cresol" if r == 16 else "ternary mixture")
                )
    axs[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(IMAGES / "milestone2-trajectories.png", dpi=155)
    plt.close(fig)
    fig, axes = plt.subplots(
        2, 1, figsize=(7.6, 4.4), layout="constrained", sharex=True
    )
    for ax, key in zip(axes, ("mass", "count"), strict=True):
        for i, (name, g) in enumerate(scores.groupby("network", sort=False)):
            ax.bar(
                np.arange(len(g)) + 0.37 * (i - 0.5),
                g[key] * 100,
                0.36,
                color=COLORS[name],
                label=name.capitalize(),
            )
        ax.set_ylabel(key.capitalize() + " RMSE (%)")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].set_xticks(np.arange(len(g)), g.run)
    axes[1].set_xlabel("Manchester experiment")
    fig.savefig(IMAGES / "milestone2-scorecard.png", dpi=155)
    plt.close(fig)

    campaign_plots(obs, predicted)
    late_loss_plot()


def campaign_plots(observed, predicted):
    """Plot all measurements against the two calibrations, by experiment."""
    for key, unit, scale in (
        ("mass", "Suspended organic mass (µg m⁻³)", 1),
        ("number", "Particle number (10³ cm⁻³)", 1000),
    ):
        fig, axes = plt.subplots(5, 3, figsize=(9.3, 9.8), sharex=True)
        fig.subplots_adjust(
            left=0.10,
            right=0.98,
            top=0.89,
            bottom=0.07,
            wspace=0.24,
            hspace=0.47,
        )
        subset = observed[observed.observable == key]
        for ax, (run_id, obs) in zip(
            axes.flat, subset.groupby("run"), strict=True
        ):
            for label, fit in predicted[predicted.run == run_id].groupby(
                "network"
            ):
                ax.plot(
                    fit.time_h,
                    fit[key] / scale,
                    color=COLORS[label],
                    ls="--" if label == "selected" else "-",
                    label=label.capitalize(),
                )
            ax.scatter(
                obs.time_h,
                obs.value / scale,
                color="#20282a",
                s=5,
                label="Measured",
                zorder=3,
            )
            ax.set_title(f"Run {run_id}", loc="left", fontsize=10)
            ax.set_xlim(0, 5)
            ax.set_ylim(bottom=0)
            if ax.get_ylim()[1] < 0.1:
                ax.set_ylim(0, 1)
            ax.spines[["top", "right"]].set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            frameon=False,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.96),
        )
        fig.suptitle(unit + " across the 15 Manchester experiments", y=0.985)
        fig.supylabel(unit, fontsize=11)
        fig.supxlabel("Hours after lights-on", y=0.015, fontsize=11)
        fig.savefig(IMAGES / f"milestone2-all-{key}.png", dpi=150)
        plt.close(fig)


def late_loss_plot():
    """Show the signed change over the same measured interval in each run."""
    changes = pd.read_csv(HERE / "peak_timing.csv")
    selected = changes.query(
        "network=='selected' and run in [16,24]"
    ).sort_values("run")
    fig, ax = plt.subplots(figsize=(4.9, 4.9), layout="constrained")
    series = [
        ("Measured", -selected.observed_drop_ug_m3, "#273d51", -0.18),
        ("Modeled", -selected.model_observed_peak_drop_ug_m3, "#ab481a", 0.18),
    ]
    for name, values, color, shift in series:
        bars = ax.bar(
            np.arange(len(values)) + shift,
            values,
            0.33,
            color=color,
            label=name,
        )
        for rect, value in zip(bars, values, strict=True):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                value + np.sign(value) * 0.7,
                f"{value:+.1f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                color=color,
            )
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xticks(
        [0, 1],
        ["Run 16 · binary\n2.88–4.98 h", "Run 24 · ternary\n3.42–4.98 h"],
    )
    ax.set_ylabel("Change in suspended organic mass (µg m⁻³)")
    ax.set_title(
        "Manchester chamber\nFrom measured peak to final sample", loc="left"
    )
    ax.margins(y=0.17)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.savefig(IMAGES / "milestone2-late-loss.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
