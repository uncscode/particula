"""Render the published CUDA ensemble differences from the saved study JSON."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from particula import plotting


@plt.style.context("particula.default")
def render(report, destination):
    """Show CUDA minus CPU means with two combined standard errors."""
    fig, axes = plotting.subplots(1, 3)
    labels = {
        "number_fraction": "Number fraction",
        "small_particle_fraction": "Small-particle fraction",
        "surface_area_fraction": "Surface area fraction",
    }
    for ax, quantity in zip(axes, report["observables"], strict=True):
        column = report["observables"].index(quantity)
        for case in report["rows"]:
            rows = sorted(
                case["refinements"], key=lambda r: r["nominal_step_s"]
            )
            ax.errorbar(
                [r["nominal_step_s"] for r in rows],
                [r["reference_difference"][column] for r in rows],
                yerr=[2 * r["combined_standard_error"][column] for r in rows],
                marker="o",
                capsize=3,
                label=f"N = {case['particles']}",
            )
        ax.axhline(0, color="0.35", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set(
            xlabel="Nominal step (s)",
            ylabel="CUDA − CPU reference",
            title=labels[quantity],
        )
    plotting.legend(axes[-1])
    fig.suptitle(f"Bimodal coagulation · {report['ensembles']} realizations")
    plotting.fit(fig)
    fig.savefig(destination, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    docs = Path(__file__).resolve().parents[2] / "docs"
    data = docs / "data/gpu/coagulation-convergence-1661.json"
    render(
        json.loads(data.read_text()),
        docs / "images/gpu/coagulation-convergence-1661.png",
    )
