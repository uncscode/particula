# %% Plot saved forward predictions and independent timing constraints.
"""Show measured and predicted volume retention and incompatible mobility bounds."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def plot(results, out):
    draws = read(results / "fractional.csv")
    fit = read(results / "layer-refine.csv")
    bounds = read(results / "constant-D-bounds.csv")
    times = np.array([float(r["time_min"]) for r in bounds])
    measured = np.array([float(r["vfr"]) for r in bounds])
    fig, (ax, br) = plt.subplots(
        1, 2, figsize=(10.0, 4.5), layout="constrained"
    )
    for xi, color, marker, ds, name in [
        ("1.0", "#246583", "s", "-", "Classical Stokes–Einstein"),
        ("0.93", "#a1502e", "^", "--", "Fractional Stokes–Einstein"),
    ]:
        data = [r for r in draws if r["xi"] == xi and r["shells"] == "160"]
        center = [
            float(r["computed"]) for r in data if r["solute_radius_nm"] == "0.4"
        ]
        low, high = np.array(
            [
                [
                    min(
                        float(r["computed"])
                        for r in data
                        if float(r["time_min"]) == t
                    ),
                    max(
                        float(r["computed"])
                        for r in data
                        if float(r["time_min"]) == t
                    ),
                ]
                for t in times
            ]
        ).T
        ax.fill_between(times, low, high, color=color, alpha=0.10)
        ax.plot(
            times,
            center,
            color=color,
            marker=marker,
            mfc="white",
            ls=ds,
            lw=1.5,
            label=name,
        )
    layer = [
        r
        for r in fit
        if r["training_shells"] == "320" and r["forecast_shells"] == "480"
    ]
    ax.plot(
        times,
        [float(r["predicted"]) for r in layer],
        color="#537744",
        ls=":",
        marker="D",
        mfc="white",
        label="Surface-enriched, daytime fit",
    )
    ax.scatter(
        times,
        measured,
        color="black",
        label="Liu et al., Fig. 4b",
        zorder=5,
        s=23,
    )
    ax.set_ylabel("Fraction of initial dry volume remaining")
    ax.set_ylim(0.49, 0.99)
    ax.legend(fontsize=8, loc="lower left")
    med, lo, hi = np.array(
        [
            [
                float(r[k])
                for k in ("D_nominal_m2_s", "D_outer_lower", "D_outer_upper")
            ]
            for r in bounds
        ]
    ).T
    for xi, col, hatch, name in [
        ("1.0", "#246583", "", "Classical range"),
        ("0.93", "#a1502e", "//", "Fractional range"),
    ]:
        mobility = [float(r["D_m2_s"]) for r in draws if r["xi"] == xi]
        br.axhspan(
            min(mobility),
            max(mobility),
            color=col,
            alpha=0.14,
            label=name,
            hatch=hatch,
        )
    br.errorbar(
        times,
        med,
        yerr=[med - lo, hi - med],
        marker="o",
        ls="none",
        color="black",
        label="Required by each observation",
        capsize=4,
        zorder=5,
    )
    br.set_ylabel("Bulk diffusivity (m² s⁻¹)")
    br.set_yscale("log")
    br.set_yticks(
        [2e-21, 5e-21, 1e-20, 2e-20],
        labels=["2 × 10⁻²¹", "5 × 10⁻²¹", "1 × 10⁻²⁰", "2 × 10⁻²⁰"],
    )
    br.legend(fontsize=8, loc="lower left")
    for a in (ax, br):
        a.set_xscale("log")
        a.set_xlim(50, 1550)
        a.set_xticks([60, 130, 210, 1200], labels=["60", "130", "210", "1,200"])
        a.set_xlabel("Evaporation time (min)")
        a.grid(axis="y", alpha=0.18)
        a.spines[["right", "top"]].set_visible(False)
    fig.savefig(out, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    plot(a.results, a.out)
