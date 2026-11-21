"""Effects on 24-hour ultrafine number retention in three regimes."""

import matplotlib.pyplot as plt
import numpy as np


def plot(report):
    """Show the range across paired environmental and resolution controls."""
    fig, axes = plt.subplots(
        1, 3, figsize=(10, 4.6), sharey=True, layout="constrained"
    )
    labels = [
        "mode width",
        "dispersion",
        "dielectric + dispersion",
        "coarse",
        "charge",
        "ion history",
        "settling",
        "shear",
    ]
    for ax, reg in zip(
        axes, ["marine", "continental", "polluted"], strict=True
    ):
        rows = {
            x["process"]: x for x in report["contrasts"] if x["regime"] == reg
        }
        for y, lab in enumerate(labels):
            x = rows[lab]
            ax.plot([x["low"], x["high"]], [y, y], color="#457c87", lw=3)
            ax.plot(x["effect"], y, "o", color="#173a49")
        ax.axvline(0, color="#888", linestyle=":")
        ax.set(
            title=reg.capitalize(),
            yticks=np.arange(len(labels)),
            yticklabels=labels,
        )
    axes[0].invert_yaxis()
    fig.supxlabel("Change in 24 h sub-100 nm retention (%)")
    return fig
