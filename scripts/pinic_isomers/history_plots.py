"""Show the pre-specified held-out chamber interval and both channel shapes."""

import matplotlib.pyplot as plt
import numpy as np


def plot(report):
    """Show fits and held-out channel predictions with measured total SOA."""
    t = np.array(report["time"])
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.4, 6.8),
        sharex=True,
        layout="constrained",
        height_ratios=[1, 1, 0.65],
    )
    smps = np.asarray(report["smps"])
    axes[2].plot(*smps.T, color="#444")
    axes[2].set(
        ylabel="SMPS SOA\n(µg/m³)",
        xlabel="Minutes since injection",
        xlim=(0, 80),
    )
    for k, (ax, name) in enumerate(
        zip(axes[:2], ["cis-Pinic acid", "Unidentified isomer"], strict=True)
    ):
        ax.plot(t, report["obs"][k], ".", label="IMS", color="#555")
        for c in report["cases"]:
            if (c["radius"], c["dt"], c["early"]) != (80.0, 0.25, "ramp"):
                continue
            ax.plot(
                t,
                c["pred"][k],
                label="Delayed source" if c["delayed"] else "Common source",
                color="#007a79" if c["delayed"] else "#b45c2f",
            )
        ax.set_ylabel(name + "\n(relative signal)")
        ax.axvspan(0.75 * t[-1], t[-1], color="#7a8296", alpha=0.16)
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[1].text(0.77, 0.04, "Held out", transform=axes[1].transAxes)
    return fig
