"""Replot the matched kernel experiments from the saved CSV measurements."""

import argparse
import csv
from pathlib import Path
from zipfile import Path as ZipPath
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parent.parent
OUTPUT = ROOT / "docs/images/studies/weighted-particle-embedding"
PUBLIC = ROOT / "website/public/study-data/weighted-particle-embedding"
LAWS = ("Brownian", "Charged", "Settling", "Shear", "Combined")
LABELS = (
    "Brownian",
    "Charged Brownian",
    "Settling",
    "Turbulent shear",
    "Brownian + settling + shear",
)
TEAL, RUST, INK = "#12606a", "#a64c25", "#213c40"


def read(path):
    """Read an archived measurement table."""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def render(values, output):
    """Render both kernel panels from a directory or open ZIP."""
    forecast(read(values / "kernel-forecast-trials.csv"), output)
    rate_errors(read(values / "kernel-rate-transfer.csv"), output)


def style(ax):
    """Give each panel the same kernel order and reference grid."""
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_ylim(4.5, -0.5)
    ax.set_yticks(range(5), LABELS)
    ax.grid(axis="x", color="#d4dddd", linewidth=0.7)
    ax.set_axisbelow(True)


def seed_times(group, kind):
    """Index timings by the initial seed for a matched comparison."""
    return {
        r["seed"]: float(r["seconds"]) for r in group if r["method"] == kind
    }


def forecast(rows, output):
    """Plot median and interquartile paired speed ratios by size."""
    sizes = sorted({int(r["n"]) for r in rows})
    fig, axes = plt.subplots(1, len(sizes), sharey=True, figsize=(9.6, 3.9))
    for ax, n in zip(axes, sizes, strict=True):
        style(ax)
        for index, law in enumerate(LAWS):
            group = [r for r in rows if r["kernel"] == law and int(r["n"]) == n]
            for method, offset, marker, color in [
                ("sampled physical", -0.13, "o", TEAL),
                ("sampled neural", 0.13, "s", RUST),
            ]:
                native = seed_times(group, "native")
                candidate = seed_times(group, method)
                ratios = [native[s] / candidate[s] for s in sorted(native)]
                lo, mid, hi = np.percentile(ratios, [25, 50, 75])
                ax.plot([lo, hi], [index + offset] * 2, color=color, lw=2)
                ax.plot(
                    mid,
                    index + offset,
                    marker,
                    color=color,
                    ms=6,
                    label=method.capitalize() if index == 0 else None,
                )
        ax.set_xscale("log")
        ax.axvline(1, ls="--", color=INK, lw=1.2)
        ax.set_title(f"{n:,} particles", fontsize=12, weight="bold")
        ax.set_xlabel("Speedup over full native physics (×)")
    limit = max(ax.get_xlim()[1] for ax in axes) * 1.08
    for ax in axes:
        ax.set_xlim(0.8, limit)
        ax.set_xticks(
            [x for x in [1, 2, 5, 10, 20, 50, 100, 200, 500] if x <= limit]
        )
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.tick_params(axis="x", which="minor", bottom=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(
        left=0.25, right=0.98, top=0.79, bottom=0.19, wspace=0.12
    )
    fig.savefig(output / "kernel-forecast-times.png", dpi=180)
    plt.close(fig)


def rate_errors(rows, output):
    """Compare applying the neutral network unchanged with matching its law."""
    fig, ax = plt.subplots(figsize=(8.4, 3.7))
    style(ax)
    for method, offset, marker, color in [
        ("Unchanged Brownian net", -0.10, "o", RUST),
        ("Kernel-specific neural", 0.10, "D", TEAL),
    ]:
        values = [
            float(
                next(
                    r["rate_error_pct"]
                    for r in rows
                    if r["kernel"] == k and r["method"] == method
                )
            )
            for k in LAWS
        ]
        ax.scatter(
            values,
            np.arange(5) + offset,
            c=color,
            marker=marker,
            s=42,
            label=method,
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_xlabel(
        "Absolute pair-rate error summed over pairs / total physical rate (%)"
    )
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.03), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.28, right=0.97, top=0.84, bottom=0.2)
    fig.savefig(output / "kernel-rate-transfer.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-values", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.from_values:
        render(args.from_values, args.output)
    else:
        with ZipFile(PUBLIC / "figure-data.zip") as archive:
            render(ZipPath(archive), args.output)
