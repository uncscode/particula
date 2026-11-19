"""Redraw the Π Chamber study figures from published numerical tables."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from particula import plotting

DOCS = Path(__file__).resolve().parents[2] / "docs"
DATA = DOCS / "data/pi-chamber"
IMAGES = DOCS / "images/thermodynamics"


def rows(name):
    """Read the values and units declared by the CSV header."""
    with (DATA / name).open(encoding="utf-8") as source:
        return list(csv.DictReader(source))


def save(fig, name):
    """Export a full paper-sized canvas."""
    plotting.fit(fig)
    fig.savefig(IMAGES / name, dpi=300)
    plt.close(fig)


@plt.style.context("particula.default")
def humidity():
    """Compare modeled haze, droplets and wet diameter with observations."""
    estimates = {(r["case"], r["label"]): r for r in rows("humidity-fits.csv")}
    # Fresh- and salt-water measurements in the published chamber comparison.
    measured = ((1073, 1327), (771, 1038), (4.0, 5.0))
    conditions = [
        ("previous", "Previous RH", "indigo"),
        ("peak equilibrium", "Peak RH", "rust"),
        ("haze count and shape", "Separate haze fit", "berry"),
        ("shared haze fit", "Shared haze fit", "teal"),
    ]
    fig, axes = plotting.subplots(1, 3)
    centers = np.arange(2)
    for ax, observed, field, label in zip(
        axes,
        measured,
        ["haze", "cloud", "mean_um"],
        ["Haze (cm⁻³)", "Cloud droplets (cm⁻³)", "Mean cloud diameter (µm)"],
        strict=True,
    ):
        ax.bar(
            centers - 0.31, observed, width=0.15, color="0.25", label="Observed"
        )
        for j, (key, legend, color) in enumerate(conditions):
            values = [float(estimates[c, key][field]) for c in ("FW", "SW")]
            ax.bar(
                centers + (j - 1) * 0.155,
                values,
                width=0.15,
                color=plotting.COLORS[color],
                label=legend,
            )
        ax.set(
            xticks=centers,
            xticklabels=["Fresh water", "Salt water"],
            ylabel=label,
        )
    plotting.legend(axes[-1], placement="right")
    save(fig, "pi-chamber-inference.png")


@plt.style.context("particula.default")
def mixtures():
    """Compare the two activation measures across continental mixing states."""
    table = rows("continental-spectra-summary.csv")
    fig, axes = plotting.subplots(2, 3, sharex=True, sharey=True)
    metrics = ["above_own_critical_pct", "detected_cloud_pct"]
    for col, mean in enumerate((-0.5, -0.25, 0.1)):
        for mode, marker in zip(
            ("internal", "partial", "external"), "o^s", strict=True
        ):
            subset = sorted(
                (
                    r
                    for r in table
                    if r["family"] == "continental-" + mode
                    and float(r["mean_pct"]) == mean
                ),
                key=lambda r: float(r["sigma_pct"]),
            )
            x = [float(r["sigma_pct"]) for r in subset]
            for ax, metric in zip(axes[:, col], metrics, strict=True):
                ax.plot(
                    x,
                    [float(r[metric]) for r in subset],
                    marker=marker,
                    label=mode,
                )
        axes[0, col].set(
            title=f"Mean supersaturation\n{mean:+g}%", ylim=(-3, 103)
        )
        axes[1, col].set_xlabel("Supersaturation SD (%)")
    axes[0, 0].set_ylabel("Above own critical size\n(% of all particles)")
    axes[1, 0].set_ylabel("Above 1.8 µm\n(% in WELAS window)")
    plotting.legend(axes[0, 0], placement="inside", loc="upper left")
    fig.suptitle(
        "Continental mixing states at equal bulk composition and dry size"
    )
    save(fig, "pi-chamber-mixtures.png")


@plt.style.context("particula.default")
def organic_vapor():
    """Plot the original fine-tolerance BAT organic mass histories."""
    table = rows("organic-vapor.csv")
    fig, axes = plotting.subplots(1, 2, sharey=True)
    for ax, cstar in zip(axes, (1, 10), strict=True):
        for rh, style in ((90.0, "--"), (99.55, "-")):
            for dose, color in zip(
                (0, 1, 5), ("rust", "indigo", "teal"), strict=True
            ):
                subset = sorted(
                    (
                        r
                        for r in table
                        if r["activity"] == "ActivityBAT"
                        and float(r["relative_tolerance"]) == 1e-6
                        and float(r["cstar_ug_m3"]) == cstar
                        and float(r["initial_vapor_ug_m3"]) == dose
                        and np.isclose(float(r["rh_pct"]), rh)
                    ),
                    key=lambda r: float(r["time_s"]),
                )
                mass = np.array(
                    [float(r["particle_organic_ug_m3"]) for r in subset]
                )
                ax.plot(
                    [float(r["time_s"]) for r in subset],
                    mass - mass[0],
                    style,
                    color=plotting.COLORS[color],
                    marker=".",
                    label=f"{dose} µg/m³ · {rh:g}%",
                )
        ax.axhline(0, lw=0.7, color="0.4")
        ax.set(xlabel="Time (s)", title=f"C* = {cstar} µg/m³")
    axes[0].set_ylabel("Particle organic mass\nchange (µg/m³)")
    plotting.legend(axes[1], placement="right")
    save(fig, "pi-chamber-organic-vapor.png")


if __name__ == "__main__":
    humidity()
    mixtures()
    organic_vapor()
