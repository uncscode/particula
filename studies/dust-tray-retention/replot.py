"""Regenerate the spectrum comparison from saved numbers (no model run)."""

from pathlib import Path
import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

root = Path(__file__).resolve().parent
with (root / "channels.csv").open() as f:
    channels = list(csv.DictReader(f))
with (root / "scores.csv").open() as f:
    scores = list(csv.DictReader(f))
fig, axes = plt.subplots(2, 2, figsize=(10.3, 7.4), sharex=True, sharey=False)
series = [
    ("early_pct", "Measured, 30 min", "#6a7079", "--", ""),
    ("measured_pct", "Measured, 3 h", "#195f83", "-", "o"),
    ("settling_pct", "Settling forecast, 3 h", "#be5617", "-", ""),
]
for ax, score in zip(axes.flat, scores, strict=True):
    rows = [r for r in channels if r["wind_ms"] == score["wind_ms"]]
    d = [float(r["channel_um"]) for r in rows]
    for key, label, color, style, marker in series:
        ax.plot(
            d,
            [float(r[key]) for r in rows],
            color=color,
            ls=style,
            marker=marker,
            markersize=2.7,
            linewidth=2 if key != "early_pct" else 1.6,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 20, 50])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticklabels(["0.2", "0.5", "1", "2", "5", "10", "20", "50"])
    ax.set_xlim(0.10, 73)
    peak = max(float(r[key]) for r in rows for key, *_ in series)
    ax.set_ylim(0, 1.14 * peak)
    ax.grid(axis="y", alpha=0.24)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{score['wind_ms']} m/s", loc="left", weight="bold")
    ax.text(
        0.025,
        0.94,
        "3 h median (µm)\n"
        + f"Measured   {float(score['measured_d50_um']):.2f}\nForecast     {float(score['settling_d50_um']):.2f}",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
        linespacing=1.45,
    )
for ax in axes[:, 0]:
    ax.set_ylabel("Retained volume per channel (%)")
for ax in axes[-1]:
    ax.set_xlabel("Volume-equivalent diameter (µm)")
fig.legend(
    *axes[0, 0].get_legend_handles_labels(),
    loc="upper center",
    bbox_to_anchor=(0.5, 0.975),
    ncol=3,
    frameon=False,
)
fig.subplots_adjust(
    wspace=0.14, hspace=0.25, left=0.09, bottom=0.10, right=0.98, top=0.88
)
fig.savefig(root / "tray-spectra.png", dpi=175, facecolor="white")

with (root / "middle-band-growth.csv").open() as f:
    growth = list(csv.DictReader(f))
fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.7), sharey=True)
for ax, wind in zip(axes, ("0", "2"), strict=True):
    rows = [r for r in growth if r["wind_ms"] == wind]
    t = [float(r["time_h"]) for r in rows]
    for key, label, style, color in (
        ("archived_mean_mg", "Measured group mean", "o-", "#195f83"),
        ("model_upper_mg", "Settling upper bound", "-", "#be5617"),
        (
            "slow_settling_upper_mg",
            "Slow-settling upper bound",
            "--",
            "#705293",
        ),
    ):
        ax.plot(
            t,
            [float(r[key]) for r in rows],
            style,
            color=color,
            label=label,
            linewidth=2,
        )
    ax.set_title(f"{wind} m/s", loc="left", weight="bold")
    ax.set_xticks(t)
    ax.set_xlabel("Time since loading (h)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.24)
axes[0].set_ylabel("Retained 10–20 µm mass (mg/tray)")
axes[0].set_ylim(5.4, 13.4)
fig.legend(
    *axes[0].get_legend_handles_labels(),
    loc="upper center",
    ncol=3,
    frameon=False,
    fontsize=9,
)
fig.subplots_adjust(left=0.095, right=0.98, bottom=0.16, top=0.79, wspace=0.10)
fig.savefig(root / "middle-band-mass.png", dpi=175, facecolor="white")
