"""Draw the graphical abstract from the three-hour wind-holdout forecasts."""

import csv
from io import StringIO
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon
from particula.plotting.toc import add_toc_logo

if __package__:
    from .study_data import result_text
else:
    from study_data import result_text

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT.parents[1] / "docs" / "images" / "dust-tray-retention"
WINDS = (0, 2, 4, 6)


def forecast_errors():
    """Score early-tray persistence and the model on the same omitted winds."""
    records = list(
        csv.DictReader(StringIO(result_text("global-comparisons.csv")))
    )
    errors = {}
    for case in ("persist", "conditional"):
        values = []
        for wind in WINDS:
            (record,) = [
                row
                for row in records
                if row["case"] == case
                and row["fold"] == f"wind{wind}"
                and int(row["wind"]) == wind
                and float(row["hours"]) == 3
            ]
            values.append(
                abs(float(record["d50_pred_um"]) - float(record["d50_obs_um"]))
            )
        errors[case] = [mean(values), *values]
    return errors


def draw_tunnel(ax):
    """Draw the airflow, suspended dust and shallow circular collector."""
    ink, teal, grain = "#243E43", "#135963", "#A44D21"
    ax.set(xlim=(0, 120), ylim=(0, 64), aspect="equal")
    ax.set_axis_off()

    # Longitudinal cutaway through the test section, with an open inlet.
    ax.add_patch(
        Polygon(
            [(3, 61), (19, 54), (117, 54), (117, 57), (20, 57), (3, 64)],
            color=ink,
            linewidth=0,
        )
    )
    ax.add_patch(
        Polygon(
            [(3, 1), (19, 8), (117, 8), (117, 11), (19, 11), (3, 4)],
            color=ink,
            linewidth=0,
        )
    )
    ax.add_patch(
        Polygon(
            [
                (5, 32),
                (28, 32),
                (28, 26),
                (40, 34),
                (28, 42),
                (28, 36),
                (5, 36),
            ],
            color=teal,
            linewidth=0,
        )
    )

    # The tray is drawn obliquely so the lip and deposited grains are visible.
    ax.add_patch(
        Polygon(
            [(49, 20), (100, 20), (96, 12), (89, 10), (60, 10), (53, 12)],
            color=ink,
            linewidth=0,
        )
    )
    ax.add_patch(Ellipse((74.5, 20), 51, 8, color=ink, linewidth=0))
    ax.add_patch(Ellipse((74.5, 20), 45, 4.8, color="#EFD1A5", linewidth=0))
    for x, y, r in [
        (43, 43, 1.6),
        (52, 32, 2.2),
        (65, 44, 2.3),
        (74, 30, 1.8),
        (84, 38, 2.2),
        (96, 47, 1.6),
        (104, 31, 1.4),
        (64, 20, 1.4),
        (76, 21, 1.7),
        (83, 19.8, 1.25),
    ]:
        ax.add_patch(Circle((x, y), r, color=grain, linewidth=0))


def export_tunnel():
    """Save a transparent vector icon and its matching raster preview."""
    plt.rcParams["svg.hashsalt"] = "dust-tray-retention"
    fig = plt.figure(figsize=(2.4, 1.28))
    draw_tunnel(fig.add_axes([0, 0, 1, 1]))
    description = "Air flows past suspended dust above a floor collection tray."
    for suffix in ("svg", "png"):
        fig.savefig(
            IMAGES / f"wind-tunnel-icon.{suffix}",
            dpi=250,
            transparent=True,
            metadata=(
                {"Description": description, "Date": None}
                if suffix == "svg"
                else {"Description": description}
            ),
        )
    vector = IMAGES / "wind-tunnel-icon.svg"
    vector.write_text(
        "\n".join(line.rstrip() for line in vector.read_text().splitlines())
        + "\n"
    )
    plt.close(fig)


def render():
    """Show the mean improvement and each contributing wind on one scale."""
    errors = forecast_errors()
    early, fitted = errors["persist"], errors["conditional"]
    levels = [4.65, 3.35, 2.35, 1.35, 0.35]
    ink, teal, guide = "#243E43", "#135963", "#9ca9a6"
    with plt.style.context("particula.toc"):
        fig = plt.figure(figsize=(3, 3))
        draw_tunnel(fig.add_axes([0.038, 0.775, 0.35, 0.205]))
        fig.text(
            0.411, 0.861, "Dust retained on trays", fontsize=8.8, weight="bold"
        )
        fig.text(0.411, 0.796, "Median diameter at 3 h", fontsize=8.4)
        fig.text(0.411, 0.743, "(by dust volume)", fontsize=7.8)
        add_toc_logo(fig, corner="top right")
        ax = fig.add_axes([0.27, 0.245, 0.66, 0.505])
        for y, copy, model in zip(levels, early, fitted, strict=True):
            ax.plot([copy, model], [y, y], color=guide, lw=1.8, zorder=1)
        ax.scatter(
            early,
            levels,
            marker="s",
            facecolors="none",
            edgecolors=ink,
            linewidths=1.1,
            s=46,
            zorder=2,
            label="30-minute tray carried forward",
        )
        ax.scatter(
            fitted,
            levels,
            marker="o",
            c=teal,
            s=15,
            zorder=3,
            label="Particula-based transport and retention",
        )
        ax.text(
            early[0] + 0.12,
            levels[0] + 0.35,
            f"{early[0]:.2f} µm",
            ha="left",
            color=ink,
            fontsize=9,
            weight="bold",
        )
        ax.text(
            fitted[0] - 0.12,
            levels[0] + 0.35,
            f"{fitted[0]:.2f} µm",
            ha="right",
            color=teal,
            fontsize=9,
            weight="bold",
        )
        ax.set(
            xlim=(0, 12),
            ylim=(-0.14, 5.66),
            xticks=(0, 4, 8, 12),
            yticks=levels,
            yticklabels=[
                "Mean",
                "Still air",
                "2 m/s",
                "4 m/s",
                "6 m/s",
            ],
            xlabel="Distance from measured median (µm)",
        )
        ax.get_yticklabels()[0].set_weight("bold")
        ax.axhline(4.03, lw=0.55, color=guide)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=5, labelsize=8)
        ax.tick_params(axis="x", length=2.5, labelsize=8, pad=2)
        ax.grid(False)
        ax.xaxis.label.set_size(8)
        fig.legend(
            *ax.get_legend_handles_labels(),
            loc="lower left",
            bbox_to_anchor=(0.048, 0.008),
            fontsize=7.2,
            labelspacing=0.45,
            frameon=False,
            handlelength=1.3,
            handletextpad=0.45,
        )
        fig.savefig(IMAGES / "analysis-toc.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    render()
    export_tunnel()
