"""Compare matched equilibrium growth predictions for each extract."""

import matplotlib.pyplot as plt
import numpy as np


def plot(report):
    """Connect predictions for the same extract and dry composition."""
    fig, axes = plt.subplots(
        2, 2, figsize=(7.4, 5.4), layout="constrained", sharex=True, sharey=True
    )
    for ax, (site, summer) in zip(
        axes.flat,
        [(s, b) for s in ("Summit", "Foothill") for b in (True, False)],
        strict=True,
    ):
        rows = [
            r
            for r in report["rows"]
            if r.get("site") == site
            and (r["date"][5:7] in ("07", "08")) == summer
        ]
        for r in rows:
            ax.plot(
                [0, 1, 2],
                [
                    r["zsr"] - r["observed"],
                    r["zsr_web_pure"] - r["observed"],
                    r["growth"] - r["observed"],
                ],
                color="#789394",
                alpha=0.45,
            )
        mean = [
            np.mean([r[k] - r["observed"] for r in rows])
            for k in ("zsr", "zsr_web_pure", "growth")
        ]
        ax.plot([0, 1, 2], mean, "o-", color="#005b62", linewidth=2)
        ax.axhline(0, color="#333", linewidth=0.7)
        ax.set(
            title=site + (" · summer" if summer else " · winter"),
            xticks=[0, 1, 2],
            xticklabels=["Liu ZSR", "Web ZSR", "Mixture"],
            xlim=(-0.15, 2.15),
        )
    fig.supylabel("Predicted − measured diameter growth factor")
    return fig
