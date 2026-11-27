"""Draw overview figures from the saved checkpoint and paired trials."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from particula.plotting.toc import add_toc_logo, save_toc_svg

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
DATA = HERE / "boreal"
IMAGES = HERE.parents[1] / "docs/images"
INK = "#243E43"
TEAL = "#135963"
RUST = "#A44D21"
PALE = "#CAD9D2"


def toc(scores=None):
    """Show a schematic redraw and the conditional qualification counts."""
    if scores is None:
        from zipfile import ZipFile

        from figure_data import ARCHIVE

        with (
            ZipFile(ARCHIVE) as archive,
            archive.open("response.csv") as stream,
        ):
            scores = pd.read_csv(stream)
    short = scores[
        (scores.method == "redraw") & (scores.window == "two short windows")
    ]
    with plt.style.context("particula.toc"):
        fig = plt.figure()
        ax = fig.add_axes((0, 0, 1, 1), xlim=(0, 100), ylim=(0, 100))
        ax.set_axis_off()
        ax.text(6, 92, "Aerosol checkpoint", color=INK, weight="bold", size=12)
        dots = [
            (0, 8, 2.6),
            (7, 3, 2),
            (0, -3, 3.4),
            (12, 10, 3),
            (12, -4, 2.5),
        ]
        for x, mirror, label in [(9, False, "Original"), (74, True, "Redraw")]:
            for i, (dx, dy, r) in enumerate(dots):
                cx = x + dx
                cy = 72 + (-dy if mirror else dy)
                ax.add_patch(Circle((cx, cy), r, fc=PALE, ec=INK, lw=0.8))
                if i % 2 == 0:
                    ax.add_patch(Circle((cx, cy), r * 0.44, fc=INK))
            ax.text(x + 5, 54, label, ha="center", size=9)
        for y, length in [(77, 11), (71, 7), (65, 14)]:
            ax.plot([43, 57], [y, y], color=PALE, lw=3)
            ax.plot([43, 43 + length], [y, y], color=TEAL, lw=2)
        for x in [28, 61]:
            ax.annotate(
                "",
                (x + 10, 71),
                (x, 71),
                arrowprops={"arrowstyle": "->", "color": INK},
            )
        ax.text(50, 54, "Rows", ha="center", size=9)
        ax.plot([6, 94], [49, 49], lw=0.6, color=INK)
        ax.text(
            6, 42, "Air masses qualified", weight="bold", color=INK, size=10
        )
        for y, family, label, color in [
            (28, "bulk_joint", "Five bulk responses", TEAL),
            (10, "counter", "150–160 nm count", RUST),
        ]:
            cases = short[short.family == family]
            n = (cases.status == "qualified").sum()
            ax.text(6, y + 5, label, size=8.5, color=INK)
            ax.plot([6, 72], [y, y], lw=3, color=PALE, solid_capstyle="butt")
            ax.plot(
                [6, 6 + 66 * n / len(cases)],
                [y, y],
                lw=3,
                color=color,
                solid_capstyle="butt",
            )
            ax.text(94, y, f"{n}/{len(cases)}", ha="right", va="center", size=9)
        add_toc_logo(fig, corner="top right")
        save_toc_svg(
            fig,
            IMAGES / "parcel-checkpoint-toc.svg",
            title=(
                "Regenerating a particle-resolved aerosol "
                "from a compact checkpoint"
            ),
            description="Particle icons trace compression and redraw. "
            "Five bulk responses qualified together in all 16 air "
            "masses. A dry 150–160 nm number channel qualified in two.",
        )
        plt.close(fig)


def row_figure():
    """Plot a real row and its allowed BC and water volumes."""
    with np.load(DATA / "redraw-cloud-16.npz") as archive:
        mean = archive["p0_mean"]
        delta = archive["p0_delta"]
        growing = archive["p0_label"][:, 1].astype(bool)
    row = 67
    volume = mean[:, :8].sum(axis=1)
    diameter = np.cbrt(volume)
    bc = 100 * mean[:, 3] / volume
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.45), layout="constrained")
    for mask, marker, color, label in [
        (~growing, "o", TEAL, "Haze"),
        (growing, "^", RUST, "Growing water"),
    ]:
        a.scatter(
            diameter[mask],
            bc[mask],
            s=15,
            marker=marker,
            color=color,
            alpha=0.7,
            label=label,
        )
    a.scatter(
        diameter[row], bc[row], s=105, facecolor="none", edgecolor=INK, zorder=5
    )
    a.annotate(
        "Row 67",
        (diameter[row], bc[row]),
        xytext=(85, 63),
        arrowprops={"arrowstyle": "->", "color": INK},
        color=INK,
    )
    a.set(
        xscale="log",
        xlabel="Mean-volume dry diameter (nm)",
        ylabel="BC fraction of dry volume (%)",
        title="Rows in the cloudy parcel",
    )
    a.set_xticks(
        [20, 50, 100, 200, 400], labels=["20", "50", "100", "200", "400"]
    )
    a.legend(loc="upper left", frameon=False, fontsize=8)
    u = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    points = (mean[row] + u[:, None] * delta[row]) * np.pi / 6 / 1e6
    x, y = points[:, 3], points[:, 8]
    b.plot(x, y, color=TEAL, lw=2, label="Allowed redraws")
    b.scatter(
        x[[1, 3]],
        y[[1, 3]],
        s=50,
        facecolor="white",
        edgecolor=TEAL,
        zorder=3,
        label="One balanced pair",
    )
    b.scatter(
        x[2], y[2], s=48, color=RUST, marker="D", zorder=3, label="Stored mean"
    )
    b.set(
        xlabel="Black carbon (10⁶ nm³ per particle)",
        ylabel="Liquid water (10⁶ nm³ per particle)",
        title="Inside row 67  ·  270 particles",
    )
    b.legend(loc="upper right", frameon=False, fontsize=8)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(IMAGES / "parcel-checkpoint-row.png", dpi=185)
    plt.close(fig)


def channel_summary():
    """Compare signed, paired measurement errors on the same hour-one draws."""
    trials = pd.read_csv(DATA / "response-pairs.csv.gz")
    g = trials[
        (trials.method == "redraw")
        & (trials.start == 3600)
        & (trials.lead == 75)
    ]
    summaries = []
    for key in ("sca", "channel"):
        for side in ("a", "b"):
            field = f"{key}_{side}"
            error = 100 * (g[field] / g["ref_" + field] - 1)
            q = error.groupby(g.seed).quantile([0.05, 0.5, 0.95]).unstack()
            summaries.append(
                q.set_axis(
                    ["error_p05_pct", "error_p50_pct", "error_p95_pct"], axis=1
                ).assign(parcel=side.upper(), measurement=key)
            )
    summary = pd.concat(summaries).reset_index()
    summary.to_csv(
        DATA / "overview-channel.csv", index=False, float_format="%.10g"
    )
    return summary


def plot_channel(summary):
    """Draw the measurement comparison from the published parcel quantiles."""
    fig, axes = plt.subplots(
        1, 2, figsize=(9, 4.7), sharey=True, layout="constrained"
    )
    for ax, key, title, limit in zip(
        axes,
        ["sca", "channel"],
        ["Dry scattering at 550 nm", "Dry count from 150 to 160 nm"],
        [5, 10],
        strict=True,
    ):
        ax.axvspan(-limit, limit, color="#e5eee9")
        ax.axvline(0, lw=0.8, color=INK)
        for side, offset, marker, color in [
            ("a", -0.17, "o", TEAL),
            ("b", 0.17, "s", RUST),
        ]:
            q = (
                summary[
                    (summary.parcel == side.upper())
                    & (summary.measurement == key)
                ]
                .set_index("seed")
                .sort_index()
            )
            pos = np.arange(len(q)) + offset
            ax.hlines(pos, q.error_p05_pct, q.error_p95_pct, color=color, lw=1)
            ax.scatter(
                q.error_p50_pct,
                pos,
                s=15,
                color=color,
                marker=marker,
                label=f"Parcel {side.upper()}",
            )
        ax.set(title=title, xlabel="Redraw error relative to full (%)")
        ax.set_yticks(np.arange(len(q)), labels=q.index)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Air mass (seed)")
    axes[0].set_xlim(-6, 6)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    fig.savefig(IMAGES / "parcel-checkpoint-channel.png", dpi=185)
    plt.close(fig)


if __name__ == "__main__":
    toc()
    row_figure()
    from zipfile import ZipFile

    from figure_data import ARCHIVE

    with (
        ZipFile(ARCHIVE) as archive,
        archive.open("overview-channel.csv") as stream,
    ):
        plot_channel(pd.read_csv(stream))
