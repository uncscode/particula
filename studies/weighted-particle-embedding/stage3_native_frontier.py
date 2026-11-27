"""Time native and sampled forecasts on identical neutral boxes."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stage3_evidence import write


def method(row):
    """Select one registered clock configuration for each collision choice."""
    if (
        row["kernel"] == "native-warp"
        and float(row["micro_or_vmax"]) == 128
        and int(row["steps"]) == 1
    ):
        return "Native Warp, 128 vapor substeps"
    if (
        row["panel"] == "native-overlap"
        and row["vapor_method"] == "midpoint"
        and int(row["steps"]) == 1
    ):
        return {
            "physical": "Sampled physical collisions",
            "learned": "Sampled neural collisions",
            "off": "Collisions omitted",
        }.get(row["kernel"])
    if row["panel"] == "screen" and int(row["steps"]) == 0:
        return {
            "off": "Adaptive, collisions omitted",
            "physical": "Adaptive, sampled physical collisions",
        }.get(row["kernel"])
    return None


def table(folder):
    """Keep actual time ratios separate from simultaneous forecast accuracy."""
    with (folder / "matched-frontier.csv").open() as stream:
        rows = []
        for r in csv.DictReader(stream):
            m = method(r)
            if not (
                m
                and r["species"] == "32"
                and r["charge"] == "neutral"
                and r["family"] == "307"
                and float(r["density"]) == 1e12
                and float(r["horizon_s"]) == 1
                and r["reference"].startswith("moving-")
            ):
                continue
            rows.append(
                dict(
                    initial_particles=int(r["n"]),
                    vapors=32,
                    density_m3=1e12,
                    horizon_s=1.0,
                    family=307,
                    initial_charge="neutral",
                    method=m,
                    reference=r["reference"],
                    replicates=int(r["gpu_seeds"]),
                    timing_trials=int(r["timing_repeats"]),
                    setup_s=float(r["median_setup_s"]),
                    advance_s=float(r["median_advance_s"]),
                    total_s=float(r["median_total_s"]),
                    number_observed_pct=float(r["number_mean_pct"]),
                    number_upper_pct=float(r["number_upper_pct"]),
                    phase_observed_pct=float(r["mass_mean_pct"]),
                    phase_upper_pct=float(r["mass_upper_pct"]),
                    limiting_phase=r["limiting_mass_field"],
                    qualification=r["decision"],
                )
            )
    native = {
        r["initial_particles"]: r
        for r in rows
        if r["method"].startswith("Native")
    }
    for r in rows:
        base = native.get(r["initial_particles"])
        for field in ("advance_s", "total_s"):
            r["measured_native_ratio_" + field] = (
                base[field] / r[field] if base else ""
            )
        r["both_pass_reference"] = bool(
            base
            and all(
                x["qualification"] == "passes tested bounds" for x in [base, r]
            )
        )
    return rows


def graph(rows, path):
    """Mark which timing measurements passed the independent two-error gate."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5), layout="constrained")
    for label, marker, color in (
        ("Native Warp, 128 vapor substeps", "s", "#584a9e"),
        ("Sampled physical collisions", "o", "#147d69"),
        ("Sampled neural collisions", "^", "#bb5c23"),
        ("Collisions omitted", "D", "#4b779c"),
        ("Adaptive, sampled physical collisions", "*", "#bd752a"),
        ("Adaptive, collisions omitted", "P", "#a2557e"),
    ):
        vals = sorted(
            [r for r in rows if r["method"] == label],
            key=lambda r: r["initial_particles"],
        )
        if not vals:
            continue
        ax.plot(
            [r["initial_particles"] for r in vals],
            [r["total_s"] for r in vals],
            color=color,
            lw=1,
            alpha=0.7,
        )
        for r in vals:
            ax.scatter(
                r["initial_particles"],
                r["total_s"],
                marker=marker,
                edgecolor=color,
                facecolor=color
                if r["qualification"] == "passes tested bounds"
                else "white",
                s=54,
                zorder=3,
            )
        ax.plot([], [], marker=marker, color=color, label=label)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Explicit particles per box",
        ylabel="Setup + one simulated second (wall seconds)",
    )
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Native and sampled dynamics at 32 vapors and 10¹² particles m⁻³"
    )
    ax.set_title(
        "Filled markers meet both targets against the moving grid",
        fontsize=10,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    a = p.parse_args()
    rows = table(a.folder)
    if rows:
        write(a.folder / "native-scale-comparison.csv", rows)
        graph(rows, a.folder / "native-scale-comparison.png")
