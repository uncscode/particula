"""Compare the two-mode reference and particle-resolved CUDA ensembles."""

import matplotlib.pyplot as plt
import numpy as np


def curves(report):
    """Show number decay over one hour with 95% sampling intervals."""
    high = [c for c in report["cases"] if c["slots"] == 3840]
    fig, axes = plt.subplots(
        3, 3, figsize=(9.5, 7.8), sharex=True, sharey=True, layout="constrained"
    )
    t = np.array(report["times_s"]) / 60
    for ax, gpu, modal in zip(axes.flat, high, report["modal"], strict=True):
        n = np.array(gpu["number_cm3"]) / 1e3
        err = np.array(gpu["ci95_cm3"]) / 1e3
        ax.fill_between(t, n - err, n + err, color="#1a7e85", alpha=0.3)
        ax.plot(t, n, "o-", ms=3, color="#116a72", label="Particula")
        ax.plot(
            t,
            np.array(modal["number_cm3"]) / 1e3,
            "--",
            color="#a74e24",
            label="MAM4xx",
        )
        ax.set_title(
            f"{gpu['diameter_nm']:g} nm, ratio {gpu['ratio']:g}", fontsize=10
        )
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.supxlabel("Elapsed time (min)")
    fig.supylabel("Particle number (thousands per cm³)")
    return fig


def spectra(report):
    """Show final spectra on common bins at equal starting mode counts."""
    fig, axes = plt.subplots(
        1, 3, figsize=(9.2, 3.7), sharey=True, layout="constrained"
    )
    edges = np.asarray(report["edges_nm"])
    mid = np.sqrt(edges[1:] * edges[:-1])
    width = np.diff(np.log10(edges))
    high = [
        c for c in report["cases"] if c["slots"] == 3840 and c["ratio"] == 1.0
    ]
    refs = [c for c in report["modal"] if c["ratio"] == 1.0]
    for ax, gpu, modal in zip(axes, high, refs, strict=True):
        for data, style, color, label in (
            (gpu, "-", "#116a72", "Particula"),
            (modal, "--", "#a74e24", "MAM4xx"),
        ):
            n = np.array(data["spectrum_cm3"][-1])[1:-1] / width
            ax.plot(mid, n, style, color=color, label=label)
        ax.set(xscale="log", xlim=(6, 1200), title=f"{gpu['diameter_nm']:g} nm")
    axes[0].legend(frameon=False)
    fig.supxlabel("Dry diameter (nm)")
    axes[0].set_ylabel("dN/dlog₁₀D (cm⁻³)")
    return fig
