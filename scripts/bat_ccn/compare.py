"""Recompute the workbook comparison and draw the CCN activation dataset."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[2] / "docs/data/thermodynamics/bat-ccn"


def main():
    """Compare the 30 published model–measurement pairs."""
    with (BASE / "figure12-measurements.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    x, y = np.array(
        [[float(r["kappa_measured"]), float(r["kappa_bat"])] for r in rows]
    ).T
    validation = np.array([r["validation_flag"] == "x" for r in rows])
    for label, mask in (
        ("All records", np.ones(len(rows), dtype=bool)),
        ("Validation subset", validation),
    ):
        observed, modeled = x[mask], y[mask]
        slope = observed @ modeled / (observed @ observed)
        r2 = 1 - sum((modeled - slope * observed) ** 2) / sum(
            (modeled - modeled.mean()) ** 2
        )
        residual = modeled - observed
        print(
            f"{label} (n={mask.sum()}) RMSE={np.mean(residual**2) ** 0.5:.6f}  "
            f"bias={residual.mean():+.6f}  slope={slope:.6f}  "
            f"centered R²={r2:.6f}"
        )
    fig, ax = plt.subplots(figsize=(6.4, 5.2), layout="constrained")
    for flags, label, color, marker in (
        (~validation, "Other published records", "#466d81", "o"),
        (validation, "Authors’ validation subset", "#aa4b25", "s"),
    ):
        ax.scatter(
            x[flags],
            y[flags],
            s=30,
            color=color,
            marker=marker,
            label=label,
            zorder=3,
        )
    for row, measured, bat in zip(rows, x, y, strict=True):
        if row["error_minus"] and row["error_plus"]:
            ax.errorbar(
                measured,
                bat,
                xerr=[[float(row["error_minus"])], [float(row["error_plus"])]],
                fmt="none",
                color="#677278",
                alpha=0.65,
                capsize=2,
            )
    ax.plot(
        [0, 0.35],
        [0, 0.35],
        "--",
        color="#454545",
        linewidth=1,
        label="1:1 agreement",
    )
    ax.set(
        xlabel="Measured CCN hygroscopicity, κ",
        ylabel="Published BAT CCN hygroscopicity, κ",
        aspect="equal",
    )
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(BASE / "figure12-comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
