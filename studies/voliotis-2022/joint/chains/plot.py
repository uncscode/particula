"""Plot search controls and the observed post-peak chamber trajectories."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bank import JOINT, OUT


def plot_rate_swap():
    """Show how the same bulk response hides different dimer concentrations."""
    data = pd.read_csv(OUT / "rate_swap_trajectories.csv")
    data = data[data.objective == "mass"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for ax, field, title in zip(
        axes,
        ("soa", "oxidized"),
        ("Total suspended organic mass", "Oxidized dimer intermediate"),
        strict=True,
    ):
        for k, line, color in [(1.0, "-", "#086d82"), (0.1, "--", "#b34f19")]:
            block = data[data.oxidation == k]
            ax.plot(
                np.r_[0, block.minutes / 60],
                np.r_[0, block[field]],
                line,
                color=color,
                lw=2.2,
                label=f"{k:g} then {0.1 / k:g} h⁻¹",
            )
        ax.set(
            title=title,
            xlabel="Time after illumination (h)",
            ylabel="Concentration (µg m⁻³)",
        )
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(title="Oxidation then fragmentation", frameon=False)
    fig.suptitle(
        f"Isolated α-pinene dimer chain · experiment {data.run.iloc[0]}"
    )
    fig.savefig(OUT / "dimer-rate-swap.png", dpi=175)
    plt.close(fig)


def main():
    """Plot the measured aerosol alongside sampled CUDA model trajectories."""
    rounds = pd.read_csv(OUT / "rounds.csv")
    states = pd.read_csv(OUT / "trajectories.csv.gz")
    observed = pd.read_csv(JOINT / "data/campaign_observations.csv")
    comp = pd.read_csv(OUT / "sensitivity.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4), layout="constrained")
    ink, blue = "#4b545c", "#086d82"
    for ax, obj, title in zip(
        axes[0],
        ("mass", "joint"),
        ("Organic mass", "Organic mass and particle count"),
        strict=True,
    ):
        r = rounds[rounds.objective == obj]
        start = r.before.iloc[0]
        for key, label, color, line in [
            ("static", "Original 20", ink, "--"),
            ("expanded", "With product chains", blue, "-"),
        ]:
            ax.plot(
                [0, *r.iteration],
                100 * np.r_[start, r[key]],
                line,
                color=color,
                label=label,
                lw=2,
            )
            network = "all" if key == "expanded" else "backbone"
            y = comp.query(
                "network == @network and objective == @obj "
                "and test == 'reference'"
            ).score.iloc[0]
            ax.scatter([44], [100 * y], marker="D", color=color, s=30, zorder=3)
        ax.set(
            title=title,
            xlabel="Reaction-chain trial",
            ylabel="Worst trajectory error (%)",
            xticks=[0, 10, 20, 30, 40, 44],
            xticklabels=["0", "10", "20", "30", "40", "Refit"],
        )
    for ax, run in zip(axes[1], (16, 24), strict=True):
        obs = observed.query(
            "run == @run and variable == 'soa_ug_m3' and time_s <= 18000"
        )
        ax.scatter(
            obs.time_s / 3600,
            obs.value,
            s=13,
            color="#292b2e",
            alpha=0.65,
            label="Measured",
        )
        for network, label, color, line in [
            ("backbone", "Original 20", ink, "--"),
            ("all", "With product chains", blue, "-"),
        ]:
            frame = states[
                (states.objective == "mass")
                & (states.network == network)
                & (states.run == run)
            ]
            ax.plot(
                np.r_[0, frame.minutes / 60],
                np.r_[0, frame.mass],
                line,
                color=color,
                lw=2,
                label=label,
            )
        ax.set(
            title=f"Experiment {run}",
            xlabel="Time after illumination (h)",
            ylabel="Suspended SOA (µg m⁻³)",
        )
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
    axes[0, 0].legend(frameon=False)
    axes[1, 0].legend(frameon=False)
    fig.savefig(OUT / "product-chains.png", dpi=175)
    plt.close(fig)
    plot_rate_swap()


if __name__ == "__main__":
    main()
