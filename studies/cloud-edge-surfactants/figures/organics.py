"""Extract or redraw the organic-vapor droplet-count comparison."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
TABLE = STUDY / "plot_data/organics.csv"
IMAGE = STUDY.parents[1] / "docs/images/cloud-edge-surfactants/organics.png"
BACKGROUNDS = {
    "marine": "Marine",
    "smoke": "Smoke",
    "urban": "Urban",
    "soa": "Boreal SOA",
}
SPEEDS = (0.2, 1.0)
# These controls and the numerical resolution must agree within each triplet.
NUMERIC_CONTROLS = (
    "w",
    "entr",
    "seed",
    "rh_ext",
    "cstar",
    "accommodation",
    "gas_ext",
    "bg_scale",
    "target_total",
    "temp_anomaly",
    "target_dg",
    "target_gsd",
    "bins",
    "nodes",
    "dt",
)
CATEGORICAL_CONTROLS = (
    "regime",
    "scenario",
    "mix",
    "import_aerosol",
    "environment",
)


def read(path):
    """Read a simulator endpoint table or the compact plotting table."""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def changes(source):
    """Pair both vapor settings with the otherwise identical zero-gas parcel."""
    results = source / "cloud/results"
    wet = read(results / "gpu_organic_limits.csv")
    # The focused replay includes zero-gas controls in this same table. The
    # original campaign stored those otherwise identical cases in factorial.
    dry = [r for r in wet if float(r["gas"]) == 0]
    if not dry:
        dry = [
            r
            for r in read(results / "gpu_factorial.csv")
            if float(r["gas"]) == 0
        ]

    def cases(rows, regime, speed, gas, access, nu):
        matches = [
            r
            for r in rows
            if r["regime"] == regime
            and float(r["w"]) == speed
            and r["mix"] == "mixed"
            and r["scenario"] == "central"
            and float(r["entr"]) == 0.05
            and float(r["gas"]) == gas
            and float(r["cstar"]) == 10
            and float(r["access"]) == access
            and float(r["nu"]) == nu
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one {regime}/{speed}/{gas}/{access}/{nu} case"
            )
        return matches[0]

    rows = []
    for regime in BACKGROUNDS:
        for speed in SPEEDS:
            baseline = cases(dry, regime, speed, 0, 1, 1)
            result = dict(background=regime, updraft_m_s=speed)
            for name, access, nu in (
                ("soluble", 1.0, 1.0),
                ("weak", 0.001, 0.1),
            ):
                treatment = cases(wet, regime, speed, 1, access, nu)
                if any(
                    float(treatment[k]) != float(baseline[k])
                    for k in NUMERIC_CONTROLS
                ) or any(
                    treatment[k] != baseline[k] for k in CATEGORICAL_CONTROLS
                ):
                    raise ValueError(
                        f"Unmatched treatments for {regime}/{speed}"
                    )
                if any(
                    abs(float(r["height_m"]) - 100) > 1e-6
                    or r["water_ok"] != "True"
                    for r in (baseline, treatment)
                ):
                    raise ValueError("Expected conserved 100 m endpoint")
                result[name + "_change_cm3"] = float(
                    treatment["activated_cm3"]
                ) - float(baseline["activated_cm3"])
            rows.append(result)
    return rows


def draw(rows):
    """Connect the paired changes in final total droplet concentration."""
    lookup = {(r["background"], float(r["updraft_m_s"])): r for r in rows}
    if len(rows) != len(BACKGROUNDS) * len(SPEEDS) or len(lookup) != len(rows):
        raise ValueError("Expected eight unique background and updraft pairs")
    fig, axes = plt.subplots(
        1, 2, figsize=(8.6, 3.55), sharex=True, sharey=True
    )
    fig.subplots_adjust(
        left=0.15, right=0.965, top=0.87, bottom=0.29, wspace=0.16
    )
    y = np.arange(len(BACKGROUNDS))
    for ax, speed in zip(axes, SPEEDS, strict=True):
        group = [lookup[r, speed] for r in BACKGROUNDS]
        soluble = np.array([float(r["soluble_change_cm3"]) for r in group])
        weak = np.array([float(r["weak_change_cm3"]) for r in group])
        ax.hlines(y, weak, soluble, lw=1.8, color="#a8b7b7")
        ax.scatter(
            weak,
            y,
            marker="D",
            facecolor="white",
            edgecolor="#a64d2a",
            linewidth=1.65,
            s=43,
            zorder=3,
            label="Weak water interaction",
        )
        ax.scatter(
            soluble,
            y,
            marker="o",
            color="#096c70",
            s=50,
            zorder=3,
            label="Soluble organic",
        )
        for position, left, right in zip(y, weak, soluble, strict=True):
            ax.annotate(
                f"{left:.1f}",
                (left, position),
                xytext=(2, 8),
                textcoords="offset points",
                ha="center",
                size=8,
                color="#893c1e",
            )
            ax.annotate(
                f"{right:.1f}",
                (right, position),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                size=8,
                color="#07595c",
            )
        ax.set(
            xlim=(0, 18),
            ylim=(3.55, -0.65),
            xticks=[0, 5, 10, 15],
            yticks=y,
            yticklabels=list(BACKGROUNDS.values()),
        )
        ax.set_title(f"{speed:g} m/s ascent", loc="left", weight="bold")
        ax.tick_params(axis="y", length=0, pad=7)
        ax.grid(axis="x", lw=0.6, color="#d8dede")
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.supxlabel(
        "Additional droplets per cm³ relative to no organic vapor",
        y=0.13,
        fontsize=11,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncols=2,
        frameon=False,
        bbox_to_anchor=(0.54, 0.0),
        fontsize=10,
    )
    IMAGE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGE, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    """Replot locally, optionally refreshing the eight paired graph records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Refresh plot data from locally regenerated parcel results",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=STUDY,
        help="Study replay root containing cloud/results",
    )
    args = parser.parse_args()
    if args.extract:
        rows = changes(args.source)
        TABLE.parent.mkdir(parents=True, exist_ok=True)
        with TABLE.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=rows[0], lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    draw(read(TABLE))


if __name__ == "__main__":
    main()
