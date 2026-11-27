"""Rebuild the exploration figures and their downloadable numerical evidence."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo
from zipfile import Path as ZipPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STUDY = "weighted-particle-embedding"
HERE = Path(__file__).resolve().parent
TEAL = "#08756d"
RUST = "#a6441b"
BLUE = "#355e92"
GREY = "#636b72"


def read_rows(path, **match):
    """Load the rows matching all supplied fields."""
    with path.open(newline="") as stream:
        return [
            row
            for row in csv.DictReader(stream)
            if all(row[key] == str(value) for key, value in match.items())
        ]


def one(path, **match):
    """Require a unique result for each plotted comparison."""
    (row,) = read_rows(path, **match)
    return row


def scale_tables(results, table):
    """Join each seed to its matched physical clock refinement."""
    scale_costs = []
    exchange = []
    for size in ("small", "million"):
        folder = results / "process-scale" / size
        settings = json.loads(table(folder / "settings.json").read_text())
        refinement = {
            int(r["seed"]): r
            for r in read_rows(table(folder / "vapor-refinement.csv"))
        }
        for seed in settings["seeds"]:
            for row in read_rows(table(folder / f"trials-{seed}.csv")):
                scale_costs.append(
                    dict(
                        **row,
                        physical_refinement_phase_error_pct=refinement[seed][
                            "phase_error_pct"
                        ],
                        physical_refinement_number_error_pct=refinement[seed][
                            "number_error_pct"
                        ],
                        physical_refinement_worst_phase=refinement[seed][
                            "worst_phase"
                        ],
                        physical_refinement_worst_time_s=refinement[seed][
                            "worst_time_s"
                        ],
                    )
                )
            for row in read_rows(table(folder / f"exchange-{seed}.csv")):
                exchange.append(dict(hosts=settings["hosts"], **row))
    return scale_costs, exchange


CONDENSATION_SUPPLEMENTS = (
    "settings.json",
    "fits.csv",
    "coefficient-probe.csv",
    "clock-audit.csv",
    "weights.npz",
    "reproduce.md",
)


def condensation_tables(results, table):
    """Recover the stage profile and matched accuracy comparisons."""
    folder = results / "condensation-options"
    for name in CONDENSATION_SUPPLEMENTS:
        table(folder / name)
    settings = json.loads((folder / "settings.json").read_text())
    rows = []
    for seed in settings["seeds"]:
        rows.extend(read_rows(table(folder / f"trials-{seed}.csv")))
    return {
        "condensation-profile": read_rows(table(folder / "profile.csv")),
        "condensation-forecasts": rows,
    }


def extract(results):
    """Select matched measurements and keep their units."""
    late = results / "stage-3"
    early = results / "final-qualification"
    sources = {}

    def table(path):
        sources[str(path.relative_to(results))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        return path

    mixes = []
    for split in ("test", "challenge"):
        for model, label in (
            ("size-only", "Size only"),
            ("first-full-cycle.pt", "Eight coordinates"),
            ("size-composition", "Size and composition"),
        ):
            if model.endswith(".pt"):
                r = one(
                    table(early / "paired-family-intervals.csv"),
                    split=split,
                    domain="pulse",
                    model=model,
                )
                seconds = r["paired_horizon_s_estimate"]
            else:
                r = one(
                    table(early / "scientific-comparison.csv"),
                    split=split,
                    domain="pulse",
                    source_table="physical-controls",
                    model=model,
                )
                seconds = r["earliest_horizon_s"]
            mixes.append(
                dict(
                    cohort=split,
                    representation=label,
                    qualified_lead_ms=1000 * float(seconds),
                )
            )

    native = []

    def add(label, count, elapsed, observed):
        native.append(
            dict(
                collision_method=label,
                runs=int(count),
                seconds=float(elapsed),
                observed_phase_error_pct=float(observed),
            )
        )

    r = one(
        table(late / "native-scale-comparison.csv"),
        initial_particles=1000000,
        method="Native Warp, 128 vapor substeps",
    )
    add("Native", r["timing_trials"], r["total_s"], r["phase_observed_pct"])
    for kernel, label in (
        ("physical", "Sampled physical"),
        ("learned", "Sampled neural"),
    ):
        r = one(
            table(late / "native-vapor-sampled-collisions.csv"),
            n=1000000,
            charge="neutral",
            sampled_kernel=kernel,
        )
        add(label, r["seeds"], r["median_total_s"], r["mass_mean_pct"])
    r = one(
        table(late / "native-omission-control.csv"),
        n=1000000,
        control="Full physical dynamics",
    )
    add(
        "Collisions off",
        r["timing_trials"],
        r["median_total_s"],
        r["phase_observed_pct"],
    )

    clocks = [
        dict(
            particles=int(r["n"]),
            substeps=int(r["vapor_steps_per_half"]),
            seconds=float(r["median_total_s"]),
            phase_upper_pct=float(r["phase_upper_pct"]),
        )
        for r in read_rows(
            table(late / "native-vapor-clock-refinement.csv"),
            control="Full physical dynamics",
        )
    ]
    scale = [
        dict(
            particles=int(r["explicit_particles"]),
            method=r["method"],
            runs=int(r["completed"]),
            memory_skips=int(r["memory_skips"]),
            seconds=r["setup_and_advance_s"],
            peak_reserved_gib=r["peak_reserved_GiB"],
        )
        for r in read_rows(table(late / "explicit-gpu-headroom.csv"))
    ]
    births = [
        dict(
            ions_per_sign=int(r["initial_ions_each_sign_per_host"]),
            diameter_nm=int(r["diameter_nm"]),
            arrivals=float(r["arrivals_per_1000"]),
            sampling_margin=float(r["simultaneous_sampling_margin"]),
            clock_allowance=float(r["paired_clock_change_upper_per_1000"]),
        )
        for r in read_rows(
            table(late / "newborn-repeated-ion-paths.csv"),
            initial_host_charge="opposite",
            level=2,
            batches=128,
        )
    ]
    widths = [
        dict(
            vapors=int(r["vapors"]),
            runs=int(r["runs"]),
            seconds=float(r["median_total_s"]),
            phase_upper_pct=float(r["mass_upper_pct"]),
            number_upper_pct=float(r["number_upper_pct"]),
        )
        for r in read_rows(
            table(late / "vapor-width-speed-and-error.csv"),
            initial_particles=1000000,
            method="physical",
            vapor_integrator="midpoint",
        )
    ]
    corrections = [
        dict(
            vapors=int(r["species"]),
            seeds=int(r["seeds"]),
            policy=r["policy"],
            number_upper_pct=float(r["simultaneous_number_upper_pct"]),
        )
        for r in read_rows(table(late / "reversible-count-confirmation.csv"))
    ]
    costs = read_rows(table(results / "process-cost/trials.csv"))
    table(results / "process-cost/settings.json")
    scale_costs, exchange = scale_tables(results, table)
    return {
        **condensation_tables(results, table),
        "process-cost": costs,
        "million-process-shares": summarize_costs(
            [r for r in scale_costs if int(r["hosts"]) == 1_000_000]
        ),
        "process-scale": scale_costs,
        "process-scale-exchange": exchange,
        "mixing-state": mixes,
        "native-collisions": native,
        "vapor-clocks": clocks,
        "particle-scaling": scale,
        "newborn-survival": births,
        "vapor-width": widths,
        "physical-correction": corrections,
    }, sources


def mixing(rows):
    """Compare the held-out duration of retained mixing-state contrasts."""
    fig, ax = plt.subplots(figsize=(7.1, 3.7), layout="constrained")
    for offset, (name, color) in enumerate(
        (("test", TEAL), ("challenge", BLUE))
    ):
        group = [r for r in rows if r["cohort"] == name]
        y = np.arange(len(group)) + (offset - 0.5) * 0.20
        x = [float(r["qualified_lead_ms"]) for r in group]
        ax.scatter(
            x,
            y,
            color=color,
            marker=("o", "s")[offset],
            label=name.capitalize(),
            s=48,
        )
    ax.set(
        yticks=range(len(group)),
        yticklabels=[r["representation"] for r in group],
        xlabel="Contiguous qualified forecast lead (ms)",
        xlim=(-3, 84),
        xticks=range(0, 81, 20),
    )
    ax.invert_yaxis()
    ax.legend(loc="upper right")
    return fig


COMPONENTS = (
    ("condensation", "Vapor transfer", TEAL),
    ("collision", "Collisions", BLUE),
    ("nucleation", "Nucleation", RUST),
    ("other", "Checks & other", "#a4a7ab"),
)
METHODS = {
    "physical": "Sampled pairs, physical rates",
    "collision": "Neural jump distance",
    "condensation": "Neural transfer factor",
    "nucleation": "Neural birth rate",
}


def summarize_costs(rows):
    """Compute each process's share of mean measured forecast time."""
    result = []
    for policy in METHODS:
        group = [r for r in rows if r["policy"] == policy]
        total = np.mean([float(r["total_ms"]) for r in group]) / 1000
        row = dict(policy=policy, trials=len(group), total_s=total)
        for key, _, _ in COMPONENTS:
            elapsed = np.mean([float(r[key + "_ms"]) for r in group]) / 1000
            row[key + "_s"] = elapsed
            row[key + "_pct"] = elapsed * 100 / total
        result.append(row)
    return result


def stacked_costs(rows, unit="s"):
    """Show absolute and percentage breakdowns of the same elapsed time."""
    fig, (time, share) = plt.subplots(
        1, 2, figsize=(9.5, 4.15), sharey=True, layout="constrained"
    )
    multiplier = 1000 if unit == "ms" else 1
    y = np.arange(len(rows))
    totals = np.array([float(row["total_s"]) * multiplier for row in rows])
    for ax, suffix, scale in ((time, "_s", multiplier), (share, "_pct", 1)):
        left = np.zeros(len(rows))
        for key, label, color in COMPONENTS:
            width = np.array([float(row[key + suffix]) * scale for row in rows])
            ax.barh(y, width, left=left, height=0.36, color=color, label=label)
            left += width
        ax.grid(axis="y", visible=False)
    for pos, (row, total) in enumerate(zip(rows, totals, strict=True)):
        time.annotate(
            f"{total:.4g}",
            xy=(total, pos),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
        percentages = "  /  ".join(
            f"{float(row[key + '_pct']):.2f}%" for key, _, _ in COMPONENTS
        )
        share.text(0, pos + 0.31, percentages, va="center", fontsize=8)
    time.set(
        yticks=y,
        yticklabels=[METHODS[row["policy"]] for row in rows],
        xlim=(0, max(totals) * 1.28),
        ylim=(len(rows) - 0.4, -0.48),
        title="Whole forecast",
        xlabel=f"Mean wall time ({unit})",
    )
    share.set(
        title="Fraction of the same forecast",
        xlim=(0, 100),
        xlabel="Share of wall time (%)",
    )
    handles, labels = time.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4)
    return fig


def process_cost(rows):
    """Plot the 15 paired small-box trials in milliseconds."""
    return stacked_costs(summarize_costs(rows), unit="ms")


def comparison_scale(rows):
    """Compare whole-forecast time at two particle counts."""
    fig, axes = plt.subplots(
        2, 2, figsize=(9.2, 5.9), sharey=True, layout="constrained"
    )
    methods = ("physical", "collision", "condensation", "nucleation")
    labels = [
        "Sampled pairs, physical rates",
        "Neural jump distance",
        "Neural transfer factor",
        "Neural birth rate",
    ]
    sizes = sorted({int(r["hosts"]) for r in rows})
    for col, hosts in enumerate(sizes):
        ax = axes[0, col]
        times = [
            [
                float(r["total_ms"]) / 1000
                for r in rows
                if int(r["hosts"]) == hosts and r["policy"] == method
            ]
            for method in methods
        ]
        mean = np.mean(times, axis=1)
        ax.barh(labels, mean, height=0.47, color=[TEAL, BLUE, BLUE, BLUE])
        for y, sample in enumerate(times):
            ax.hlines(y, min(sample), max(sample), color="#222", linewidth=1)
            ax.plot(sample, [y] * len(sample), "|", color="#222", markersize=6)
            ax.text(
                max(sample) + max(mean) * 0.02, y, f"{mean[y]:.3g}", va="center"
            )
        ax.set(
            title=f"{hosts:,} particles",
            xlabel="Wall time per 10 s forecast (s)",
            xlim=(0, max(map(max, times)) * 1.22),
        )
        ax.grid(axis="y", visible=False)
        errors = axes[1, col]
        for key, name, offset, color, marker in (
            ("phase_error_pct", "Phase mass", -0.12, RUST, "o"),
            ("number_error_pct", "Particle count", 0.12, BLUE, "s"),
        ):
            values = [
                max(
                    float(r[key])
                    for r in rows
                    if int(r["hosts"]) == hosts and r["policy"] == method
                )
                for method in methods
            ]
            errors.scatter(
                values,
                np.arange(len(methods)) + offset,
                label=name,
                color=color,
                marker=marker,
                s=30,
            )
        errors.set_xscale("symlog", linthresh=1e-4)
        ticks = [0, 1e-4, 1e-3, 0.01, 0.1, 1]
        errors.set(
            xticks=ticks,
            xticklabels=["0", ".0001", ".001", ".01", ".1", "1"],
            xlim=(-1e-5, 1.5),
            xlabel="Worst observed error (%)",
        )
        errors.axvline(0.1, color=RUST, ls=":", lw=1)
        errors.axvline(1, color=BLUE, ls=":", lw=1)
        errors.grid(axis="y", visible=False)
    axes[1, 0].legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1))
    axes[0, 0].invert_yaxis()
    return fig


def scale_exchange(rows):
    """Show net transfer of each species in the million-particle reference."""
    fig, ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
    large = max(int(row["hosts"]) for row in rows)
    for label, color, name in (
        ("condensing", TEAL, "Condensing"),
        ("evaporating", RUST, "Evaporating"),
        ("near-steady", BLUE, "Near equilibrium"),
    ):
        group = [
            r
            for r in rows
            if int(r["hosts"]) == large and r["initial_regime"] == label
        ]
        ids = sorted({int(row["species"]) for row in group})
        values = [
            np.mean(
                [
                    float(r["final_net_condensed_pct"])
                    for r in group
                    if int(r["species"]) == i
                ]
            )
            for i in ids
        ]
        ax.bar(ids, values, color=color, label=name)
        ax.scatter(ids, values, color=color, s=8)
    ax.axhline(0, color="#333", lw=0.9)
    ax.set(
        xlabel="Vapor species",
        ylabel="Net transfer to particles\n(% of initial gas mass)",
        xticks=[1, 4, 8, 12, 16, 20, 24, 28, 32],
        xlim=(0, 33),
    )
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1))
    return fig


def collisions(rows):
    """Keep native vapor transfer fixed across four collision treatments."""
    labels = {
        "Native": "Full native physics",
        "Sampled neural": "Sampled pairs · neural collision term",
        "Sampled physical": "Sampled pairs · physical rate (control)",
        "Collisions off": "Mergers off",
    }
    by_method = {row["collision_method"]: row for row in rows}
    rows = [by_method[method] for method in labels]
    seconds = np.array([float(r["seconds"]) for r in rows])
    baseline = next(
        float(r["seconds"]) for r in rows if r["collision_method"] == "Native"
    )
    y = np.arange(len(rows))
    fig, (ax, stats) = plt.subplots(
        1,
        2,
        figsize=(9.4, 3.0),
        width_ratios=(2.3, 1),
        sharey=True,
        layout="constrained",
    )
    bars = ax.barh(y, seconds, height=0.47, color=[GREY, BLUE, TEAL, GREY])
    ax.bar_label(bars, labels=[f"{t:,.1f} s" for t in seconds], padding=4)
    ax.set(
        yticks=y,
        yticklabels=[labels[r["collision_method"]] for r in rows],
        xlim=(0, baseline * 1.29),
        ylim=(len(rows) - 0.5, -0.5),
        xlabel="Whole-forecast wall time (s)",
    )
    ax.grid(axis="y", visible=False)
    stats.set_xlim(0, 1)
    stats.set_axis_off()
    for x, name in ((0.22, "Speedup"), (0.75, "Phase error")):
        stats.text(
            x,
            1.02,
            name,
            transform=stats.transAxes,
            ha="center",
            weight="bold",
            fontsize=10,
        )
    for i, row in enumerate(rows):
        error = float(row["observed_phase_error_pct"])
        stats.text(
            0.22,
            i,
            f"{baseline / seconds[i]:.2f}×",
            ha="center",
            va="center",
        )
        stats.text(
            0.75,
            i,
            f"{error:.4f}%",
            ha="center",
            va="center",
            color=RUST if error >= 0.1 else "#252525",
        )
    return fig


def clocks(rows):
    """Show the cost and error of refining only the vapor clock."""
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), layout="constrained")
    for n, color, marker in ((4096, TEAL, "o"), (65536, BLUE, "x")):
        group = sorted(
            (r for r in rows if int(r["particles"]) == n),
            key=lambda r: int(r["substeps"]),
        )
        for ax, key in zip(axes, ("phase_upper_pct", "seconds"), strict=True):
            ax.plot(
                [int(r["substeps"]) for r in group],
                [float(r[key]) for r in group],
                color=color,
                marker=marker,
                ls="-" if n == 4096 else "--",
                label=f"{n:,}",
            )
            ax.set(
                xlabel="Vapor substeps per half interval",
                xticks=(128, 256, 512),
            )
    axes[0].axhline(0.1, color=GREY, ls=":", label="0.1% target")
    for ax in axes:
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Upper phase-mass error (%)")
    axes[1].set_ylabel("Setup + forecast (s)")
    axes[1].legend(title="Particles per box")
    return fig


def scaling(rows):
    """Plot successful single-box timings and peak reservations."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), layout="constrained")
    for method, label, color, marker in (
        ("physical", "Physical pairs", TEAL, "o"),
        ("learned", "Neural pairs", BLUE, "s"),
        ("off", "Collisions off", RUST, "^"),
    ):
        group = sorted(
            (r for r in rows if r["method"] == method and int(r["runs"]) > 0),
            key=lambda r: int(r["particles"]),
        )
        for ax, key in zip(axes, ("seconds", "peak_reserved_gib"), strict=True):
            ax.plot(
                [int(r["particles"]) / 1e6 for r in group],
                [float(r[key]) for r in group],
                color=color,
                marker=marker,
                label=label,
                ls="--" if method == "learned" else "-",
            )
            ax.set(
                xticks=(2, 4, 6, 8, 10),
                xlabel="Explicit particles (millions)",
            )
    for ax in axes:
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Setup + forecast (s)")
    axes[1].set_ylabel("Peak reserved GPU memory (GiB)")
    axes[0].legend()
    return fig


def formation(rows):
    """Compare newborn survival at two diameter thresholds."""
    fig, ax = plt.subplots(figsize=(6.8, 3.7), layout="constrained")
    for offset, (d, color) in enumerate(((5, TEAL), (10, BLUE))):
        group = sorted(
            (r for r in rows if int(r["diameter_nm"]) == d),
            key=lambda r: int(r["ions_per_sign"]),
        )
        y = [float(r["arrivals"]) for r in group]
        bars = ax.bar(
            np.arange(len(group)) + (offset - 0.5) * 0.34,
            y,
            width=0.34,
            color=color,
            label=f"{d} nm",
        )
        ax.bar_label(
            bars, labels=[f"{v:.0f}" for v in y], fontsize=9, padding=2
        )
    ax.set(
        xticks=range(len(group)),
        xticklabels=[r["ions_per_sign"] for r in group],
        xlabel="Initial ion pairs per background particle",
        ylabel="Arrivals per 1,000 newborns",
        ylim=(0, 480),
    )
    ax.legend()
    return fig


def width(rows):
    """Show the wall time and limiting phase error as chemistry widens."""
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), layout="constrained")
    group = sorted(rows, key=lambda r: int(r["vapors"]))
    x = np.arange(len(group))
    for ax, (key, ylabel) in zip(
        axes,
        (
            ("seconds", "Setup + forecast (s)"),
            ("phase_upper_pct", "Upper phase-mass error (%)"),
        ),
        strict=True,
    ):
        ax.plot(x, [float(r[key]) for r in group], "o-", color=TEAL)
        ax.set(
            xticks=x,
            xticklabels=[r["vapors"] for r in group],
            xlabel="Distinct vapors",
            ylabel=ylabel,
        )
        ax.set_ylim(bottom=0)
    return fig


def correction(rows):
    """Compare count-error bounds at the registered sample sizes."""
    fig, axes = plt.subplots(
        1, 2, figsize=(8.0, 3.6), sharey=True, layout="constrained"
    )
    for ax, vapors in zip(axes, (8, 32), strict=True):
        for policy, label, color, marker in (
            ("none", "No physical interval", RUST, "s"),
            ("early", "First interval physical", TEAL, "o"),
        ):
            group = sorted(
                (
                    r
                    for r in rows
                    if int(r["vapors"]) == vapors and r["policy"] == policy
                ),
                key=lambda r: int(r["seeds"]),
            )
            ax.plot(
                [int(r["seeds"]) for r in group],
                [float(r["number_upper_pct"]) for r in group],
                marker=marker,
                color=color,
                label=label,
            )
        ax.axhline(1, ls=":", color=GREY)
        ax.set(
            title=f"{vapors} vapors",
            xticks=(1024, 2048),
            xlim=(880, 2190),
            xlabel="Independent seeds",
        )
    axes[0].set_ylabel("Upper number error (%)")
    axes[0].set_ylim(bottom=0)
    axes[1].legend()
    return fig


def condensation_profile(rows):
    """Show mean CUDA event times for a physical exchange half step."""
    keys = ("geometry", "conductance", "gas_solve", "commit")
    labels = (
        "Wet radius and mass",
        "Transfer coefficient",
        "Box gas balance",
        "Write particle and gas masses",
    )
    means = [np.mean([float(r[k]) for r in rows]) for k in keys]
    fig, ax = plt.subplots(figsize=(7.6, 2.9), layout="constrained")
    bars = ax.barh(labels, means, color=[GREY, BLUE, TEAL, RUST], height=0.55)
    ax.bar_label(
        bars,
        labels=[f"{v:.1f} ms  ({100 * v / sum(means):.0f}%)" for v in means],
        padding=4,
    )
    ax.set(
        xlabel="Mean time per physical vapor half step (ms)",
        xlim=(0, max(means) * 1.42),
    )
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    return fig


def condensation_forecasts(rows):
    """Show warm rollout times and worst phase errors on a common reference."""
    methods = {
        "physical": "Original physical solver",
        "constants": "Precomputed constants",
        "reuse": "Reuse after events",
        "physical-grid": "Physical radius table",
        "net8": "8-unit network per particle",
        "net24-grid": "24-unit network table",
        "fused": "Two-pass physics",
        "richardson": "Fixed step correction",
        "learned-step": "Learned step correction",
        "fused-8": "8 physical substeps",
        "fused-16": "16 substeps (reference)",
    }
    fig, (time, error) = plt.subplots(
        1, 2, figsize=(9.2, 5.2), sharey=True, layout="constrained"
    )
    for i, key in enumerate(methods):
        group = [r for r in rows if r["method"] == key]
        elapsed = np.mean([float(r["total_ms"]) / 1000 for r in group])
        color = BLUE if key in ("net8", "net24-grid", "learned-step") else TEAL
        time.barh(i, elapsed, color=color, height=0.55)
        time.annotate(
            f"{elapsed:.2f}",
            (elapsed, i),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
        if key != "fused-16":
            err = max(float(r["refined_phase_pct"]) for r in group)
            error.scatter(err, i, color=color)
    time.margins(x=0.17)
    time.set(
        yticks=range(len(methods)),
        yticklabels=list(methods.values()),
        xlabel="Mean whole-forecast time (s)",
    )
    error.set_xscale("log")
    error.set(
        xticks=[0.05, 0.1, 0.3, 1, 2],
        xticklabels=["0.05", "0.1", "0.3", "1", "2"],
        xlabel="Worst phase-mass error (%)",
    )
    error.axvline(0.1, color=RUST, ls="--", lw=1.2, label="0.1% target")
    error.legend(loc="lower right")
    time.invert_yaxis()
    return fig


FIGURES = {
    "condensation-profile": condensation_profile,
    "condensation-forecasts": condensation_forecasts,
    "process-cost": process_cost,
    "million-process-shares": stacked_costs,
    "process-scale": comparison_scale,
    "process-scale-exchange": scale_exchange,
    "mixing-state": mixing,
    "native-collisions": collisions,
    "vapor-clocks": clocks,
    "particle-scaling": scaling,
    "newborn-survival": formation,
    "vapor-width": width,
    "physical-correction": correction,
}


def publish_tables(results, output):
    """Build the standalone figure download from the source measurements."""
    values, sources = extract(results)
    payload = {}
    for name, rows in values.items():
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        payload[f"{name}.csv"] = stream.getvalue().encode("utf-8")
    kernel_results = results / "kernel-transfer"
    for source, target in (
        ("trials.csv", "kernel-forecast-trials.csv"),
        ("rate-transfer.csv", "kernel-rate-transfer.csv"),
        ("shear-clock.csv", "kernel-shear-clock.csv"),
    ):
        content = (kernel_results / source).read_bytes()
        payload[target] = content
        sources[f"kernel-transfer/{source}"] = hashlib.sha256(
            content
        ).hexdigest()
    payload["kernel_transfer_figures.py"] = (
        HERE / "kernel_transfer_figures.py"
    ).read_bytes()
    for name in CONDENSATION_SUPPLEMENTS:
        payload["condensation/" + name] = (
            results / "condensation-options" / name
        ).read_bytes()
    payload[Path(__file__).name] = Path(__file__).read_bytes()
    payload["provenance.json"] = json.dumps(
        dict(
            historical_results_at="9baa471d13",
            process_cost_method="process_cost.py, matched GPU trials",
            process_cost_settings=json.loads(
                (results / "process-cost/settings.json").read_text()
            ),
            process_scale_settings=[
                json.loads(
                    (
                        results / "process-scale" / size / "settings.json"
                    ).read_text()
                )
                for size in ("small", "million")
            ],
            source_sha256=sources,
        ),
        indent=2,
    ).encode()
    payload["README.txt"] = (
        b"Replot with Python 3.12, NumPy and Matplotlib.\n"
        b"python exploration_figures.py --from-values . --output figures\n"
        b"python kernel_transfer_figures.py --from-values . "
        b"--output figures\n"
        b"CSV files store the plotted values. Source checksums refer to "
        b"paths in measurements.zip.\n"
    )
    with ZipFile(output, "w") as archive:
        for name, content in payload.items():
            archive.writestr(ZipInfo(name), content, compress_type=ZIP_DEFLATED)
    return values


def main():
    """Replot saved results or a downloaded figure bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-values", type=Path, help="Directory of downloaded figure CSVs"
    )
    parser.add_argument("--output", type=Path, help="Figure output directory")
    parser.add_argument(
        "--results",
        type=Path,
        help="Rebuild the download from an unpacked measurement directory",
    )
    args = parser.parse_args()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.16,
            "legend.frameon": False,
        }
    )
    if args.from_values:
        values = {
            name: read_rows(args.from_values / f"{name}.csv")
            for name in FIGURES
        }
        out = args.output or Path.cwd()
    else:
        root = HERE.parents[1]
        data = root / "website/public/study-data" / STUDY
        data.mkdir(parents=True, exist_ok=True)
        out = args.output or root / "docs/images/studies" / STUDY
        if args.results:
            values = publish_tables(args.results, data / "figure-data.zip")
        else:
            with ZipFile(HERE / "measurements.zip") as archive:
                values = publish_tables(
                    ZipPath(archive), data / "figure-data.zip"
                )
    out.mkdir(parents=True, exist_ok=True)
    for name, draw in FIGURES.items():
        fig = draw(values[name])
        fig.savefig(out / f"{name}.png", dpi=300, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
