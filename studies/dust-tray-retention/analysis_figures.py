"""Replot the integrated paper from archived forecasts and tray data."""

import csv
from io import StringIO
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .study_data import result_text
else:
    from study_data import result_text

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT.parents[1] / "docs" / "images" / "dust-tray-retention"
WINDS = (0, 2, 4, 6)
TEAL, ORANGE, PURPLE, GRAY = "#116d76", "#bc6325", "#69539a", "#424b54"


def rows(name):
    """Read a CSV table from the study packet."""
    return list(csv.DictReader(StringIO(result_text(name))))


def save(fig, name):
    """Export a paper figure at its specified resolution."""
    fig.savefig(IMAGES / name, dpi=205, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def forecast_bands():
    """Compare the observed and predicted redistribution by size and wind."""
    data = rows("channels.csv")
    fits = json.loads(result_text("global-core.json"))["results"]
    scores = rows("global-comparisons.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 6.3), sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.115, right=0.985, top=0.89, bottom=0.16, hspace=0.32, wspace=0.12
    )
    for i, (wind, ax) in enumerate(zip(WINDS, axes.flat, strict=True)):
        records = [row for row in data if int(row["wind_ms"]) == wind]
        diameter = np.array([float(row["channel_um"]) for row in records])
        fit = next(
            r
            for r in fits
            if r["case"] == "conditional" and r["fold"] == f"wind{wind}"
        )
        mass = np.asarray(fit["prediction"])[i, -1]
        endpoint = 100 * mass / mass.sum()
        expected = next(
            float(r["d50_pred_um"])
            for r in scores
            if r["case"] == "conditional"
            and r["fold"] == f"wind{wind}"
            and int(r["wind"]) == wind
            and float(r["hours"]) == 3
        )
        median = np.exp(np.interp(50, np.cumsum(endpoint), np.log(diameter)))
        np.testing.assert_allclose(median, expected, rtol=1e-5)
        early, observed, settle = (
            np.array([float(row[field]) for row in records])
            for field in ("early_pct", "measured_pct", "settling_pct")
        )
        group = np.searchsorted([10, 20, 30], diameter)
        for shift, late, color, label in [
            (-0.25, observed, TEAL, "Jia et al. measurements"),
            (0, settle, ORANGE, "Particula settling"),
            (
                0.25,
                endpoint,
                PURPLE,
                "Particula-based transport and retention\n(wind held out)",
            ),
        ]:
            change = np.bincount(group, weights=late - early)
            np.testing.assert_allclose(change.sum(), 0, atol=0.02)
            ax.bar(np.arange(4) + shift, change, 0.23, color=color, label=label)
        ax.axhline(0, lw=0.85, color=GRAY)
        ax.set_title(
            "Still air" if wind == 0 else f"{wind} m/s",
            loc="left",
            weight="bold",
        )
        ax.set_xticks(range(4), ["≤10", "10–20", "20–30", ">30"])
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.19)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.52, 1.045),
    )
    fig.supylabel(
        "Change in retained volume share (percentage points)",
        x=0.021,
        fontsize=11,
    )
    fig.supxlabel("Reported channel diameter (µm)", y=0.058, fontsize=11)
    save(fig, "analysis-size-transfer.png")


def timing_and_capture():
    """Compare one-hour exposure and the final retained mass at 2 m/s."""
    timing = [r for r in rows("rebound-timing.csv") if int(r["wind_ms"]) == 2]
    total, middle = (
        [r for r in timing if r["band"] == band] for band in ("all", "middle")
    )
    t = [float(r["time_h"]) for r in total]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.2, 4.25))
    fig.subplots_adjust(
        bottom=0.17, top=0.88, left=0.075, right=0.98, wspace=0.34
    )
    for records, field, label, color, marker, ls in [
        (total, "exposure_fraction", "Centreline exposure", GRAY, "s", "--"),
        (
            total,
            "observed_fraction_of_increment",
            "Total tray mass",
            TEAL,
            "o",
            "-",
        ),
        (
            middle,
            "observed_fraction_of_increment",
            "10–20 µm tray mass",
            ORANGE,
            "^",
            "-",
        ),
    ]:
        ax.plot(
            t,
            [100 * float(r[field]) for r in records],
            color=color,
            marker=marker,
            ls=ls,
            label=label,
            markersize=5,
            linewidth=1.8,
        )
    ax.set(
        xlabel="Tray exposure time (h)",
        ylabel="Fraction of 0.5–3 h increment (%)",
        xticks=t,
        ylim=(-4, 105),
    )
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title(
        "A   When exposure and mass accumulated",
        loc="left",
        weight="bold",
        fontsize=11,
    )

    surfaces = rows("rebound-changing-surface.csv")
    removal = next(
        r
        for r in rows("rebound-results.csv")
        if int(r["wind_ms"]) == 2
        and float(r["hours"]) == 3
        and r["case"] == "fitted_fine_removal"
    )
    observed = [
        float(part[-1]["observed_mg"]) - float(part[0]["observed_mg"])
        for part in (total, middle)
    ]
    sem = next(
        float(r["delta_total_mass_se_mg"])
        for r in rows("measurement-sensitivity.csv")
        if int(r["wind_ms"]) == 2 and r["case"] == "original"
    )
    bx.errorbar(
        *observed, xerr=sem, fmt="none", ecolor=TEAL, capsize=3, zorder=3
    )
    bx.scatter(*observed, marker="*", color=TEAL, s=160, zorder=4)
    bx.annotate(
        "Observed",
        observed,
        xytext=(-8, 8),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        weight="bold",
        color=TEAL,
    )
    scenarios = [("Fine-particle removal", removal, (6, -11))]
    for alpha, late_speed, charge, label, offset in [
        (1, 0.05, 10, "Charge-assisted sticking", (-6, -16)),
        (1, 0.2, 0, "Stronger sticking", (0, 7)),
        (2, 0.2, 0, "Stronger sticking,\nwider impact speeds", (-4, -26)),
    ]:
        surface = next(
            r
            for r in surfaces
            if float(r["alpha"]) == alpha
            and float(r["early_vc10_ms"]) == 0.05
            and float(r["late_vc10_ms"]) == late_speed
            and float(r["late_charge_scale"]) == charge
        )
        scenarios.append((label, surface, offset))
    for label, case, offset in scenarios:
        xy = (float(case["net_gain_mg"]), float(case["cutoff_middle_gain_mg"]))
        bx.scatter(*xy, marker="D", color=PURPLE, s=35, zorder=3)
        bx.annotate(
            label,
            xy,
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
            ha="right" if offset[0] < 0 else "left",
        )
    bx.set(
        xlabel="Change in total mass (mg/tray)",
        ylabel="Change in 10–20 µm mass (mg/tray)",
        xlim=(-6.2, 20.5),
        ylim=(-0.65, 7.3),
    )
    bx.axvline(0, color=GRAY, lw=0.85)
    bx.set_title(
        "B   What each surface retained by 3 h",
        loc="left",
        weight="bold",
        fontsize=11,
    )
    for panel in (ax, bx):
        panel.grid(alpha=0.18)
        panel.set_axisbelow(True)
    save(fig, "analysis-timing-retention.png")


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.labelcolor": "#25313a",
        }
    )
    forecast_bands()
    timing_and_capture()
