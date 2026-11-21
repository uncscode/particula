"""Measured WSIM growth, composition reconstructions, and Figure 8 residuals."""

import matplotlib.pyplot as plt
import numpy as np

GROUPS = (
    ("Summit", True, 0.21),
    ("Summit", False, 0.10),
    ("Foothill", True, 0.11),
    ("Foothill", False, 0.02),
)


def group(report, site, summer):
    """Select one season and collection site."""
    return [
        r
        for r in report["samples"]
        if r["site"] == site and (r["time"][5:7] in ("07", "08")) == summer
    ]


def comparison(report):
    """Plot every observed growth factor against its salt predictions."""
    fig, axes = plt.subplots(
        2, 2, figsize=(8.2, 7.4), sharex=True, sharey=True, layout="constrained"
    )
    for ax, (site, summer, _) in zip(axes.flat, GROUPS, strict=True):
        rows = group(report, site, summer)
        ax.plot(
            [1.0, 2.4], [1.0, 2.4], "--", color=".5", lw=1, label="Equal growth"
        )
        for row in rows:
            g = row["expanded"]
            if g:
                ax.vlines(
                    row["observed"],
                    g["low"],
                    g["high"],
                    color="#559c9d",
                    lw=1.6,
                )
                ax.plot(
                    row["observed"], g["growth"], "o", color="#12676b", ms=4
                )
            if row["reduced"]:
                ax.plot(
                    row["observed"],
                    row["reduced"]["growth"],
                    "x",
                    color="#a74b22",
                    ms=4,
                )
        ax.set_title(f"{site}, {'summer' if summer else 'winter'}")
        ax.set(xlim=(1.0, 2.4), ylim=(1.0, 2.4), aspect="equal")
    fig.supxlabel("Measured diameter growth factor at 90% RH")
    fig.supylabel("Predicted diameter growth factor")
    return fig


def residuals(report):
    """Compare mean signed biases with the authors’ Figure 8."""
    fig, ax = plt.subplots(figsize=(7.7, 3.8), layout="constrained")
    labels = []
    for i, (site, summer, reference) in enumerate(GROUPS):
        rows = [r for r in group(report, site, summer) if r["expanded"]]
        mean = {
            k: np.mean([r["expanded"][k] - r["observed"] for r in rows])
            for k in ("low", "high", "growth")
        }
        ax.vlines(i - 0.12, mean["low"], mean["high"], color="#12676b", lw=3)
        ax.plot(
            i - 0.12,
            mean["growth"],
            "o",
            color="#12676b",
            label="Particula, matched ions" if i == 0 else None,
        )
        ax.plot(
            i + 0.12,
            reference,
            "D",
            color="#773969",
            label="Wang et al., Figure 8" if i == 0 else None,
        )
        labels.append(f"{site}\n{'summer' if summer else 'winter'}")
    ax.axhline(0, color=".55", lw=1)
    ax.set(
        xticks=range(4),
        xticklabels=labels,
        ylabel="Mean predicted − measured growth",
        xlim=(-0.45, 3.45),
    )
    ax.legend(frameon=False, loc="upper right")
    return fig
