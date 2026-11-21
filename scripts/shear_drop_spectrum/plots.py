"""Render the spectrum and the finite-volume sampling comparison."""

import matplotlib.pyplot as plt
import numpy as np


def spectrum_figure(report):
    """Compare the resolved result with both baselines and printed markers."""
    fig, (spectrum, probability) = plt.subplots(
        1, 2, figsize=(10.5, 4.6), layout="constrained"
    )
    reference = [
        r for r in report["reference"] if r["cutoff"] == 128 and r["K"] == 0.09
    ]
    masses = np.arange(1, 9)
    for column, label, color, style in [
        ("constant", "Constant rate, same time", "#666666", "--"),
        ("matched", "Constant rate, same mean mass", "#be6918", "-."),
        ("shear", "Size-dependent shear", "#236d80", "-"),
    ]:
        spectrum.semilogy(
            masses,
            [r[column] for r in reference],
            style,
            color=color,
            label=label,
        )
    markers = [
        r
        for r in report["markers"]
        if r["K"] == 0.09 and r["symbol"] == "circle"
    ]
    spectrum.scatter(
        [r["mass_class"] for r in markers],
        [r["abundance"] for r in markers],
        facecolors="none",
        edgecolors="#222222",
        s=42,
        label="1956 theoretical points",
        zorder=4,
    )
    spectrum.errorbar(
        masses + 0.06,
        report["mean"],
        yerr=report["halfwidth"],
        fmt=".",
        color="#873f70",
        label="Particula, 95% sampling interval",
        zorder=5,
    )
    spectrum.set(
        xlabel="Drop mass / initial drop mass",
        ylabel="Drops in class / initial drop count",
        title="Size spectrum at K = 0.09",
        xticks=masses,
    )
    spectrum.legend(fontsize=8)
    for case, label, color, marker in [
        ("small", "0.427 mL (256 initial drops)", "#be6918", "s"),
        ("large_fine_pooled", "1.71 mL (1,024 initial drops)", "#236d80", "o"),
    ]:
        rows = [r for r in report["summary"] if r["case"] == case]
        center = np.array([r["detection"] for r in rows])
        error = np.maximum(
            0,
            [
                center - [r["p_low"] for r in rows],
                [r["p_high"] for r in rows] - center,
            ],
        )
        probability.errorbar(
            [r["K"] for r in rows],
            center * 100,
            yerr=error * 100,
            marker=marker,
            color=color,
            capsize=3,
            label=label,
        )
    probability.set(
        title="Sampling the eightfold mass class",
        xlabel="Dimensionless time K",
        ylabel="Boxes with at least one such drop (%)",
        xticks=[0.03, 0.06, 0.09],
        ylim=(-2, 60),
    )
    probability.legend(fontsize=8)
    for axis in (spectrum, probability):
        axis.spines[["right", "top"]].set_visible(False)
        axis.grid(axis="y", alpha=0.16)
    return fig
