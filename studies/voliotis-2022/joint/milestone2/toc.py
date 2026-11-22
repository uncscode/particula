"""Render the Manchester network TOC from the archived peak-to-last changes."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from particula.plotting.toc import add_toc_logo

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parents[3] / "docs/images/manchester-soa/network-growth-toc.png"


@plt.style.context("particula.toc")
def render():
    """Compare changes over the same measured interval for each experiment."""
    timing = pd.read_csv(HERE / "peak_timing.csv")
    selected = timing.query("network == 'selected'").set_index("run")
    fig = plt.figure()
    fig.text(0.06, 0.91, "Manchester chamber", size=9, weight="bold")
    fig.text(0.06, 0.79, "After the mass peak", size=14, weight="bold")
    fig.text(0.06, 0.71, "Change in organic aerosol mass (µg m⁻³)", size=8)
    add_toc_logo(fig, corner="top right")

    ax = fig.add_axes([0.08, 0.13, 0.84, 0.53])
    ax.set(
        xlim=(-41, 24), ylim=(-0.5, 5.5), yticks=[], xticks=[-30, -15, 0, 15]
    )
    ax.spines["left"].set_visible(False)
    ax.axvline(0, lw=0.8, color=ax.spines["bottom"].get_edgecolor())
    for run, mixture, y in [(16, "Binary", 4.0), (24, "Ternary", 1.05)]:
        row = selected.loc[run]
        ax.text(
            -41,
            y + 0.7,
            f"{mixture} · {row.observed_peak_h:.2f}–{row.last_sample_h:.2f} h",
            size=8.5,
            weight="bold",
        )
        for label, delta, offset, color in [
            (
                "Measured",
                -row.observed_drop_ug_m3,
                0,
                plt.rcParams["text.color"],
            ),
            ("Fitted", -row.model_observed_peak_drop_ug_m3, -0.78, "C0"),
        ]:
            ax.barh(y + offset, delta, height=0.57, color=color)
            # Label the long bars inside and the short bars at the outer end.
            inside = abs(delta) > 18
            x = delta / 2 if inside else delta + (1 if delta > 0 else -1)
            align = "center" if inside else ("left" if delta > 0 else "right")
            ax.text(
                x,
                y + offset,
                f"{label} {delta:+.1f}",
                ha=align,
                va="center",
                size=7.3,
                color=fig.get_facecolor() if inside else color,
            )
    fig.savefig(OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    render()
