"""Plot the measured and simulated three-hour endpoints from transport-results.csv."""

import csv
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
with (root / "transport-results.csv").open() as f:
    results = {(int(r["wind_ms"]), r["case"]): r for r in csv.DictReader(f)}
fig, (left, right) = plt.subplots(
    1, 2, figsize=(11, 4.65), gridspec_kw={"width_ratios": [1.05, 1]}
)
winds = [0, 2, 4, 6]
for i, (key, column, label, color, marker) in enumerate(
    (
        ("all", "observed_d50_um", "Tray measurement", "#165d79", "o"),
        ("all", "d50_um", "Coupled model", "#bb511c", "s"),
        ("monitor", "d50_um", "Coupled, monitor-scaled", "#654992", "^"),
    )
):
    vals = [float(results[w, key][column]) for w in winds]
    left.scatter(
        np.arange(4) + 0.14 * (i - 1),
        vals,
        s=54,
        c=color,
        marker=marker,
        label=label,
        zorder=3,
    )
left.set_xticks(np.arange(4), winds)
left.set_xlabel("Wind setting (m/s)")
left.set_ylabel("Three-hour dry-volume median (µm)")
left.legend(loc="upper left", frameon=False, fontsize=9)
left.set_title("Across the four tunnel runs", loc="left", weight="bold")
left.grid(axis="y", alpha=0.22)
tests = [
    ("multi_floor", "Settling + mixing"),
    ("multi_wall", "Add sidewalls"),
    ("all", "Add collisions"),
    ("all_RH85_bipolar5", "85% RH, bipolar"),
    ("monitor", "Use bulk monitor"),
    ("monitor_tray_shape", "Vary airborne sizes"),
]
for y, (key, label) in enumerate(tests):
    x = float(results[2, key]["d50_um"])
    right.scatter(
        x,
        y,
        color="#654992" if key.startswith("monitor") else "#bb511c",
        s=42,
        zorder=3,
    )
    right.text(x + 0.13, y, f"{x:.2f}", va="center", fontsize=9)
right.set_yticks(range(len(tests)), [label for _, label in tests])
right.invert_yaxis()
right.set_xlabel("Three-hour dry-volume median (µm)")
right.set_title("How the 2 m/s forecast changes", loc="left", weight="bold")
right.axvline(
    float(results[2, "all"]["observed_d50_um"]), ls="--", lw=1.5, c="#165d79"
)
right.annotate(
    "Observed 9.80",
    xy=(9.8, 0.97),
    xycoords=("data", "axes fraction"),
    ha="right",
    va="top",
    color="#165d79",
    fontsize=9,
    backgroundcolor="white",
)
right.set_xlim(3.2, 10.2)
right.set_ylim(len(tests) - 0.4, -0.7)
right.grid(axis="x", alpha=0.20)
for ax in (left, right):
    ax.spines[["right", "top"]].set_visible(False)
fig.subplots_adjust(left=0.065, right=0.98, bottom=0.16, top=0.9, wspace=0.46)
fig.savefig(root / "transport-comparison.png", dpi=170)
