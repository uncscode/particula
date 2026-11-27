"""Validate the forecast and memory sweeps and plot the results."""

import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from campaign import read
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter
from redraw_burst import START, burst_cases
from redraw_checkpoint import fit_box, serialize
from redraw_design import DRAWS, LEADS, WINDOWS
from redraw_replay import replay
from redraw_results import reliability, trials
from redraw_run import GATES, recipe_name
from redraw_scaling import COURSE_SIZES, RECIPES, SCALE_SIZES, course_cases
from stage_two_design import initialize

METHODS = ("10-16-1", "10-32-1", "full")
NAMES = ("16 rows", "32 rows", "Full repeat")
COLORS = ("#007876", "#ba5a20", "#464959")


def course_data(root, folder, cases, windows):
    """Read every expected draw, method and forecast lead exactly once."""
    pieces = []
    expected = set(product(range(DRAWS), LEADS, METHODS))
    for case in cases:
        for start in windows:
            data = pd.read_csv(root / folder / case.key / f"{start}.csv.gz")
            keys = set(
                data[["draw", "lead", "recipe"]].itertuples(
                    index=False, name=None
                )
            )
            if (
                len(data) != len(expected)
                or keys != expected
                or not data.seed.eq(case.seed).all()
                or not data.n.eq(case.n).all()
                or not data.start.eq(start).all()
            ):
                raise ValueError(f"Incomplete records at {case.key}/{start}")
            pieces.append(data)
    return pd.concat(pieces, ignore_index=True)


def verify_memory(data, forecasts):
    """Cross-check field widths and archive sizes against independent runs."""
    expected = set(product(SCALE_SIZES, WINDOWS, (0, *(s[1] for s in RECIPES))))
    keys = set(
        data[["n", "start", "recipe"]].itertuples(index=False, name=None)
    )
    if len(data) != len(expected) or keys != expected:
        raise ValueError("Incomplete memory measurements")
    for stage in ("encode", "decode"):
        if (
            data[f"{stage}_peak_rss_bytes"] < data[f"{stage}_base_rss_bytes"]
        ).any():
            raise ValueError("Invalid peak RSS")
    np.testing.assert_array_equal(data.resolved_array_bytes, 77 * data.slots)
    np.testing.assert_array_equal(
        data.stored_array_bytes, np.where(data.recipe == 0, 77, 168) * data.rows
    )
    shared = forecasts.query("draw == 0 and lead == 0").copy()
    shared["recipe"] = shared.recipe.map(
        dict(zip(METHODS, (16, 32, 0), strict=True))
    )
    joined = data.merge(
        shared, on=["seed", "n", "start", "recipe"], suffixes=("", "_course")
    )
    np.testing.assert_equal(
        len(joined), len(COURSE_SIZES) * len(WINDOWS) * len(METHODS)
    )
    np.testing.assert_array_equal(joined.archive_bytes, joined.stored_bytes)
    np.testing.assert_array_equal(joined.rows, joined.rows_course)


def reliability_figure(paths, events, images):
    """Plot trajectory reliability and event-specific pass fractions."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.1),
        layout="constrained",
        gridspec_kw={"width_ratios": [1, 1.25]},
    )
    ax = axes[0]
    for i, (recipe, name, color) in enumerate(
        zip(METHODS, NAMES, COLORS, strict=True)
    ):
        q = paths[paths.recipe == recipe].sort_values("n")
        x = np.arange(len(q)) + (i - 1) * 0.12
        ax.errorbar(
            x,
            q.rate,
            yerr=[q.rate - q.block_lo, q.block_hi - q.rate],
            fmt="o-",
            lw=1.5,
            capsize=3,
            label=name,
            color=color,
        )
    ax.axhline(0.9, c="#666666", lw=1, ls="--")
    ax.set(
        xticks=range(len(COURSE_SIZES)),
        xticklabels=[f"{n // 1000}k" for n in COURSE_SIZES],
        xlabel="Starting particles per parcel",
        ylabel="Draws passing all four endpoint families",
        title="Half-hour trajectories",
        ylim=(0, 1.05),
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False, loc="lower right")
    ax = axes[1]
    matrix = (
        events[events.recipe == recipe_name(RECIPES[0])]
        .pivot(index="n", columns="start", values="joint")
        .loc[list(COURSE_SIZES), list(WINDOWS)]
    )
    heat = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    for i, j in np.ndindex(matrix.shape):
        val = matrix.iloc[i, j]
        ax.text(
            j,
            i,
            f"{val * 100:g}%",
            ha="center",
            va="center",
            color="white" if val < 0.5 else "black",
        )
    ax.set(
        xticks=range(5),
        xticklabels=[
            "Burst\nonset",
            "Burst\ngrowth",
            "Ascent",
            "In cloud",
            "Evap.",
        ],
        yticks=range(len(matrix)),
        yticklabels=[f"{n // 1000}k" for n in matrix.index],
        ylabel="Starting particles per parcel",
        title="Percent of forecasts that pass",
    )
    fig.colorbar(
        heat,
        ax=ax,
        shrink=0.75,
        format=PercentFormatter(1),
        label="Forecasts passing (%)",
    )
    fig.savefig(images / "parcel-redraw-reliability.png", dpi=170)
    plt.close(fig)


def property_failures(data, events):
    """Measure each forecast's largest absolute error across positive leads."""
    limits = {
        metric: limit
        for gates in GATES.values()
        for metric, limit in gates.items()
    }
    recipe = recipe_name(RECIPES[0])
    course = data.loc[data.recipe.eq(recipe) & data.lead.isin(LEADS[1:])]
    per_draw = course.groupby(["n", "start", "seed", "draw"])
    expected = (
        events.loc[events.recipe.eq(recipe)]
        .set_index(["n", "start"])
        .sort_index()
    )
    if (
        not per_draw.lead.agg(tuple)
        .map(lambda leads: sorted(leads) == sorted(LEADS[1:]))
        .all()
    ):
        raise ValueError("Incomplete positive leads for a forecast")
    peaks = per_draw[list(limits)].max()
    if course[list(limits)].isna().any().any():
        raise ValueError("Missing forecast error")
    failed = peaks.gt(pd.Series(limits))
    cells = ["n", "start"]
    passed = (~failed.any(axis=1)).groupby(level=cells)
    np.testing.assert_array_equal(
        passed.size().reindex(expected.index), expected.trials
    )
    np.testing.assert_array_equal(
        passed.sum().reindex(expected.index), expected.passed
    )

    rows = []
    for (n, start), group in peaks.groupby(level=cells):
        for family, gates in GATES.items():
            for metric, limit in gates.items():
                misses = group.loc[group[metric] > limit, metric]
                rows.append(
                    dict(
                        n=n,
                        start=start,
                        family=family,
                        metric=metric,
                        unit="pp" if metric.endswith("_pp") else "%",
                        tolerance=limit,
                        trials=len(group),
                        failed=len(misses),
                        median_failed_error=misses.median(),
                        worst_failed_error=misses.max(),
                    )
                )
    return pd.DataFrame(rows)


def failure_figure(summary, images):
    """Show failed forecasts and their worst errors on the pass-rate grid."""
    titles = {
        "organic_pct": "Condensed organic inventory",
        "birth_pct": "New particles formed",
        "new_count_pct": "Surviving new-only particles",
        "new_cdf_pp": "New-only size distribution",
        "dry_abs_pct": "Dry absorption at 550 nm",
        "wet_abs_pct": "Wet absorption at 550 nm",
    }
    present = summary[summary.failed > 0]
    metrics = present.metric.unique()
    norm = Normalize(
        1, np.ceil((present.worst_failed_error / present.tolerance).max())
    )
    cmap = plt.get_cmap("YlOrRd").with_extremes(bad="#eef1f2")
    fig, axes = plt.subplots(
        (len(metrics) + 1) // 2,
        2,
        figsize=(10, 2.6 * ((len(metrics) + 1) // 2) + 0.7),
        layout="constrained",
        squeeze=False,
    )
    for ax, metric in zip(axes.flat, metrics, strict=False):
        sub = summary[summary.metric == metric]
        limit, unit = sub.iloc[0][["tolerance", "unit"]]
        worst = sub.pivot(
            index="n", columns="start", values="worst_failed_error"
        ).loc[list(COURSE_SIZES), list(WINDOWS)]
        counts = sub.pivot(index="n", columns="start", values="failed").loc[
            worst.index, worst.columns
        ]
        heat = ax.imshow(worst / limit, norm=norm, cmap=cmap, aspect="auto")
        for i, j in np.ndindex(worst.shape):
            val = worst.iat[i, j]
            count = counts.iat[i, j]
            label = f"{count}\n{val:.2f}{unit}" if count else "–"
            color = "white" if count and norm(val / limit) > 0.60 else "#243034"
            ax.text(
                j, i, label, ha="center", va="center", color=color, fontsize=8
            )
        ax.set(
            title=f"{titles.get(metric, metric)} (limit {limit}{unit})",
            xticks=range(len(WINDOWS)),
            xticklabels=["Onset", "Growth", "Ascent", "In cloud", "Evap."],
            yticks=range(len(COURSE_SIZES)),
            yticklabels=[f"{n // 1000}k" for n in COURSE_SIZES],
        )
    for ax in list(axes.flat)[len(metrics) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Failed forecasts out of 40 and largest observed error", weight="bold"
    )
    fig.supylabel("Starting particles per parcel")
    fig.colorbar(
        heat,
        ax=axes,
        orientation="horizontal",
        shrink=0.69,
        aspect=45,
        label="Largest error as a multiple of the property’s tolerance",
    )
    fig.savefig(images / "parcel-redraw-failure-matrix.png", dpi=180)
    plt.close(fig)


def save_burst_example(root, output, data, table):
    """Export the limiting burst and replay draw zero from the archive alone."""
    case = next(c for c in burst_cases() if c.seed == 10723 and c.n == 50000)
    boxes, rng, time = read(root / case.key / f"{START}.npz")
    _, _, forcing = initialize(case)
    spec = RECIPES[0]
    fields = [fit_box(b, *spec) for b in boxes]
    example = output / "redraw-burst-16.npz"
    example.write_bytes(serialize(fields, case, forcing, time, rng, spec))
    recovered = replay(example, 0, table)
    expected = data[
        (data.seed == case.seed)
        & (data.n == case.n)
        & (data.draw == 0)
        & (data.recipe == recipe_name(spec))
    ]
    np.testing.assert_allclose(
        recovered,
        expected.sort_values("lead")[recovered.columns],
        rtol=1e-8,
        atol=1e-12,
    )


def burst_figure(summary, data, images):
    """Contrast predicting the original and repeating the same event stream."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    ax = axes[0]
    paths = summary[summary.horizon == "path"]
    for i, (recipe, comparison, name, color) in enumerate(
        [
            (METHODS[0], "original", "16 rows vs original", COLORS[0]),
            ("full", "original", "Full repeat vs original", COLORS[2]),
            (METHODS[0], "paired", "16 rows vs paired full", "#8264a2"),
        ]
    ):
        g = paths[(paths.recipe == recipe) & (paths.comparison == comparison)]
        ax.bar(
            np.arange(2) + (i - 1) * 0.24,
            g.sort_values("n").rate,
            width=0.24,
            label=name,
            color=color,
        )
    ax.axhline(0.9, color="#555555", ls="--", lw=1)
    ax.set(
        xticks=[0, 1],
        xticklabels=["25,000", "50,000"],
        xlabel="Starting particles per parcel",
        ylabel="Joint path pass rate",
        ylim=(0, 1.07),
        title="Eight new burst histories",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="lower left", framealpha=1, fontsize=8)
    ax = axes[1]
    window = data[(data.seed == 10723) & (data.n == 50000) & (data.lead == 75)]
    original = window.ref_births.iloc[0]
    ax.axhspan(
        0.9 * original,
        1.1 * original,
        color="#efdfb6",
        alpha=0.6,
        label="Original ±10%",
    )
    ax.axhline(original, lw=1, color="#88652e")
    for recipe, name, dx, color in [
        (METHODS[0], "16 rows", -0.09, COLORS[0]),
        ("full", "Full repeat", 0.09, COLORS[2]),
    ]:
        g = window[window.recipe == recipe]
        ax.scatter(g.draw + 1 + dx, g.births, s=20, color=color, label=name)
    ax.set(
        xticks=np.arange(1, 11),
        xlabel="Paired realization",
        ylabel="Births in the first 75 s, both parcels",
        title="History 10723 at burst onset",
    )
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.savefig(images / "parcel-redraw-burst.png", dpi=170)
    plt.close(fig)


def scaling_figure(data, images):
    """Plot retained bytes, archive sizes and transient resident memory."""
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 6.8), layout="constrained")
    for ax, field, divisor, title, ylabel in [
        (
            axes[0, 0],
            "archive_bytes",
            1000,
            "Portable checkpoint",
            "Archive size (kB)",
        ),
        (
            axes[0, 1],
            "stored_array_bytes",
            1000,
            "Retained particle fields",
            "Array size (kB)",
        ),
        (
            axes[1, 0],
            "encode_extra",
            2**20,
            "Encode from resolved particles",
            "Additional peak RSS (MiB)",
        ),
        (
            axes[1, 1],
            "decode_extra",
            2**20,
            "Decode to resolved particles",
            "Additional peak RSS (MiB)",
        ),
    ]:
        for value, name, color in zip(
            (16, 32, 0), ("16 rows", "32 rows", "Resolved"), COLORS, strict=True
        ):
            s = data[data.recipe == value].groupby("n")[field]
            mid = s.median() / divisor
            ax.fill_between(
                mid.index,
                s.min() / divisor,
                s.max() / divisor,
                color=color,
                alpha=0.12,
            )
            ax.plot(mid.index, mid, marker="o", ms=4, color=color, label=name)
        ax.set(
            xscale="log",
            xticks=SCALE_SIZES,
            xticklabels=["1k", "5k", "10k", "25k", "50k", "100k"],
            xlabel="Starting particles per parcel",
            ylabel=ylabel,
            title=title,
        )
        ax.minorticks_off()
        if field in ("archive_bytes", "stored_array_bytes"):
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.22)
    axes[0, 0].legend(frameon=False)
    fig.savefig(images / "parcel-redraw-scaling.png", dpi=170)
    plt.close(fig)


def main():
    """Run the command-line study stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("root", "output", "images"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--optics", default=".artifacts/boreal/optics.npz")
    args = parser.parse_args()
    data = course_data(args.root, "course", course_cases(), WINDOWS)
    memory = pd.read_csv(args.output / "redraw-memory.csv")
    verify_memory(memory, data)
    for stage in ("encode", "decode"):
        memory[stage + "_extra"] = (
            memory[stage + "_peak_rss_bytes"]
            - memory[stage + "_base_rss_bytes"]
        )
    all_trials = trials(data)
    path = all_trials[all_trials.horizon == "path"]
    rel = reliability(all_trials)
    events = reliability(path.assign(horizon=path.start)).rename(
        columns={"horizon": "start", "rate": "joint"}
    )
    rel.to_csv(
        args.output / "redraw-independent.csv", index=False, float_format="%.8g"
    )
    events.to_csv(
        args.output / "redraw-independent-events.csv",
        index=False,
        float_format="%.8g",
    )
    data.to_csv(
        args.output / "redraw-independent-leads.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
        float_format="%.10g",
    )
    burst = course_data(args.root, "burst", burst_cases(), [START])
    replicated = pd.concat(
        [
            reliability(trials(burst, paired=paired)).assign(comparison=name)
            for name, paired in (("original", False), ("paired", True))
        ],
        ignore_index=True,
    )
    save_burst_example(args.root, args.output, burst, args.optics)
    burst_figure(replicated, burst, args.images)
    replicated.to_csv(
        args.output / "redraw-burst-replication.csv",
        index=False,
        float_format="%.8g",
    )
    burst.to_csv(
        args.output / "redraw-burst-leads.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
        float_format="%.10g",
    )
    reliability_figure(rel[rel.horizon == "path"], events, args.images)
    failures = property_failures(data, events)
    failures.to_csv(
        args.output / "redraw-failure-matrix.csv",
        index=False,
        float_format="%.10g",
    )
    failure_figure(failures, args.images)
    scaling_figure(memory, args.images)


if __name__ == "__main__":
    main()
