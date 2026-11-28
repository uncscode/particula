"""Redraw the validation and instrument figures from committed source tables."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT.parents[1] / "docs/images/spectral-radiation"
BLUE, ORANGE, GREEN = "#216486", "#a75124", "#247962"


def table(relative_path):
    """Read the named fields in a study CSV, retaining text categories."""
    return np.genfromtxt(
        ROOT / relative_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )


def surface_errors():
    """Compare two surface models on the same Terra footprints."""
    rows = table("observations/comparisons.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.55), layout="constrained")
    for ax, (field, title, unit) in zip(
        axes,
        (
            ("sw", "CERES shortwave", "Flux RMSE (W/m²)"),
            ("b1", "MODIS red", "Radiance RMSE (W m⁻² sr⁻¹ µm⁻¹)"),
            ("b2", "MODIS near infrared", "Radiance RMSE (W m⁻² sr⁻¹ µm⁻¹)"),
        ),
        strict=True,
    ):
        for i, (key, label, color) in enumerate(
            (
                ("ideal_surface", "Ideal surface", "#a5adb3"),
                ("scene", "Scene surface", BLUE),
            )
        ):
            values = []
            for surface in ("ocean", "bare"):
                sample = rows[rows["surface"] == surface]
                error = sample[f"{key}_{field}"] - sample[f"observed_{field}"]
                values.append(np.sqrt(np.mean(error**2)))
            bars = ax.bar(
                np.arange(2) + (i - 0.5) * 0.37,
                values,
                0.35,
                label=label,
                color=color,
                hatch="//" if i == 0 else None,
            )
            ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
        ax.set(
            xticks=[0, 1],
            xticklabels=["Ocean", "Bare land"],
            ylabel=unit,
            title=title,
        )
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    return fig


def low_sun():
    """Compare plane-parallel and pseudo-spherical DISORT surface irradiance."""
    rows = table("benchmark/low_sun.csv")
    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    for wave, color, marker in (
        (350, BLUE, "o"),
        (550, GREEN, "s"),
        (850, ORANGE, "^"),
    ):
        sample = rows[rows["wavelength_nm"] == wave]
        ax.plot(
            sample["sza_deg"],
            np.abs(sample["relative_total_pct"]),
            color=color,
            marker=marker,
            markersize=4,
            label=f"{wave} nm",
        )
    ax.axhline(1, color="#555", ls="--", lw=1, label="1% difference")
    ax.set(
        yscale="log",
        xlabel="Solar zenith angle (degrees)",
        ylabel="Absolute difference in surface irradiance (%)",
        xticks=[30, 45, 60, 70, 80, 89],
    )
    ax.legend(frameon=False, loc="upper left", ncols=2)
    return fig


def sensor_response():
    """Show the archived in-band Terra MODIS response at its native spacing."""
    rows = table("observations/terra-response.csv")
    fig, ax = plt.subplots(figsize=(7, 3.0), layout="constrained")
    for band, label, color, style in (
        (1, "Band 1 · red", ORANGE, "-"),
        (2, "Band 2 · near infrared", BLUE, "--"),
    ):
        sample = rows[rows["band"] == band]
        ax.fill_between(
            sample["wavelength_nm"],
            sample["relative_response"],
            color=color,
            alpha=0.14,
        )
        ax.plot(
            sample["wavelength_nm"],
            sample["relative_response"],
            color=color,
            ls=style,
            label=label,
        )
    ax.set(
        xlabel="Wavelength (nm)",
        ylabel="Relative response",
        xlim=(600, 915),
        ylim=(0, 1.06),
        yticks=[0, 0.25, 0.5, 0.75, 1],
    )
    ax.legend(frameon=False, loc="upper center")
    return fig


def scaling():
    """Plot synchronized CUDA timings and the single-threaded NumPy loop."""
    rows = table("benchmark/warp_scaling_rtx5060.csv")
    fig, ax = plt.subplots(figsize=(7, 3.55), layout="constrained")
    for field, label, color, marker in (
        ("cpu_loop_s", "NumPy · one OpenBLAS thread", ORANGE, "s"),
        ("warm_gpu_median_s", "Warp CUDA · RTX 5060", BLUE, "o"),
    ):
        ax.plot(
            rows["columns"],
            1000 * rows[field],
            color=color,
            marker=marker,
            label=label,
            lw=2,
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Two-layer columns per call",
        ylabel="Elapsed solve time (ms)",
        xticks=rows["columns"],
        yticks=[10, 30, 100, 300, 1000],
        ylim=(9, 1300),
    )
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ScalarFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    ax.legend(frameon=False, loc="upper left")
    return fig


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
        }
    ):
        for name, draw in (
            ("terra-surface-errors", surface_errors),
            ("low-sun", low_sun),
            ("modis-response", sensor_response),
            ("warp-scaling", scaling),
        ):
            figure = draw()
            figure.savefig(OUTPUT / f"{name}.png", dpi=210, facecolor="white")
            plt.close(figure)
