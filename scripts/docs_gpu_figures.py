"""CPU plotting helpers for the manually refreshed CUDA scientific examples."""

import numpy as np
from matplotlib import style as mplstyle
from matplotlib.colors import LogNorm
from particula import plotting


@mplstyle.context("particula.default")
def history_figures(report, title, *, forcing):
    """Plot recorded box histories and particle-resolved size distributions."""
    frames = report["frames"]
    time = np.array([f["time_s"] for f in frames])
    temperature = np.array([f["temperature_K"] for f in frames])
    saturation = np.array([f["water_saturation"] for f in frames])
    mass = np.array([f["particle_mass_kg"] for f in frames])
    histograms = np.array([f["histogram_number_m3"] for f in frames])
    edges = np.array(frames[0]["radius_edges_m"])
    widths = np.diff(np.log10(edges))
    distribution = histograms / widths
    boxes = temperature.shape[1]
    selected = sorted({0, boxes // 2, boxes - 1})
    fig, axes = plotting.subplots(2, 2, size="wide")
    if forcing:
        for box in selected:
            axes[0, 0].plot(time, temperature[:, box], label=f"Box {box + 1}")
        axes[0, 0].set(xlabel="Time (s)", ylabel="Prescribed temperature (K)")
        plotting.legend(axes[0, 0])
    else:
        scatter = axes[0, 0].scatter(
            temperature[0],
            mass[-1, :, 0] / mass[0, :, 0],
            c=np.arange(boxes) + 1,
            cmap="viridis",
            s=25,
        )
        axes[0, 0].set(
            xlabel="Initial box temperature (K)",
            ylabel="Final / initial particle water mass",
        )
        fig.colorbar(scatter, ax=axes[0, 0], label="Box")
    for box in selected:
        axes[0, 1].plot(time, saturation[:, box], label=f"Box {box + 1}")
    axes[0, 1].axhline(1, color="0.5", linestyle="--")
    axes[0, 1].set(xlabel="Time (s)", ylabel="Water saturation ratio")
    plotting.legend(axes[0, 1])
    colors = (plotting.COLORS["teal"], plotting.COLORS["rust"], "#684a8c")
    for box, color in zip(selected, colors, strict=True):
        for index, style, label in ((0, "--", "initial"), (-1, "-", "final")):
            axes[1, 0].stairs(
                distribution[index, box],
                edges * 1e9,
                color=color,
                linestyle=style,
                label=f"Box {box + 1}, {label}",
            )
    axes[1, 0].set(
        xscale="log", xlabel="Radius (nm)", ylabel="dN / dlog₁₀ r (m$^{-3}$)"
    )
    plotting.legend(axes[1, 0])
    axes[1, 1].grid(False)
    mesh = axes[1, 1].pcolormesh(
        edges * 1e9,
        np.arange(boxes + 1) + 0.5,
        np.ma.masked_less_equal(distribution[-1], 0),
        norm=LogNorm(),
        cmap=plotting.colormap("magnitude"),
        shading="flat",
    )
    axes[1, 1].set(xscale="log", xlabel="Radius (nm)", ylabel="Box")
    fig.colorbar(mesh, ax=axes[1, 1], label="Final dN / dlog₁₀ r (m$^{-3}$)")
    fig.suptitle(title)
    print(
        "Maximum per-box, per-material inventory relative error:",
        report["max_box_species_relative_error"],
    )
    if forcing:
        source = np.array(report["external_mass_kg_by_box_material"])
        print(
            "Net prescribed water addition across boxes (kg):",
            source[:, 0].sum(),
        )
    plotting.fit(fig)
    return fig


@mplstyle.context("particula.default")
def stochastic_figures(report, initial_histogram, title):
    """Show seed spread with identical initial state across realizations."""
    time = np.array(report["time_s"])
    number = np.array(report["number_m3_by_seed_time_box"])
    histogram = np.array(report["final_histogram_by_seed_box"])
    edges = np.array(report["radius_edges_m"])
    selected = [0, number.shape[2] // 2, number.shape[2] - 1]
    fig, axes = plotting.subplots(1, 2, size="wide")
    for box, color in zip(
        selected,
        (plotting.COLORS["teal"], plotting.COLORS["rust"], "#684a8c"),
        strict=True,
    ):
        fraction = number[:, :, box] / number[:, :1, box]
        low, high = np.percentile(fraction, [5, 95], axis=0)
        axes[0].plot(
            time, fraction.mean(axis=0), color=color, label=f"Box {box + 1}"
        )
        axes[0].fill_between(time, low, high, color=color, alpha=0.2)
    axes[0].set(xlabel="Time (s)", ylabel="Number concentration / initial")
    plotting.legend(axes[0])
    box = selected[1]
    width = np.diff(np.log10(edges))
    distribution = histogram[:, box] / width
    low, high = np.percentile(distribution, [5, 95], axis=0)
    axes[1].stairs(
        initial_histogram[box] / width,
        edges * 1e9,
        linestyle="--",
        label="Initial",
    )
    axes[1].stairs(
        distribution.mean(axis=0), edges * 1e9, label="Final seed mean"
    )
    centres = np.sqrt(edges[:-1] * edges[1:]) * 1e9
    axes[1].fill_between(
        centres, low, high, alpha=0.2, label="5–95% seed percentiles"
    )
    axes[1].set(
        xscale="log",
        xlabel="Radius (nm)",
        ylabel="dN / dlog₁₀ r (m$^{-3}$)",
        title=f"Box {box + 1}",
    )
    plotting.legend(axes[1])
    fig.suptitle(
        title + "\n16 stream seeds · shaded spread is not a confidence interval"
    )
    print(report["band_definition"])
    print("Distinct final trajectories:", len(set(report["result_hashes"])))
    plotting.fit(fig)
    return fig


@mplstyle.context("particula.default")
def lifecycle_figures(report, initial_histogram, title):
    """Show population changes and the independently checked process ledger."""
    edges = np.array(report["radius_edges_m"])
    width = np.diff(np.log10(edges))
    fig, axes = plotting.subplots(1, 2, size="wide")
    axes[0].stairs(
        initial_histogram.mean(axis=0) / width,
        edges * 1e9,
        linestyle="--",
        label="Initial",
    )
    axes[0].stairs(
        np.array(report["mean_final_histogram_number_m3"]) / width,
        edges * 1e9,
        label="Final",
    )
    axes[0].set(
        xscale="log",
        xlabel="Radius (nm)",
        ylabel="Mean dN / dlog₁₀ r (m$^{-3}$)",
    )
    plotting.legend(axes[0])
    ledger = report["process_ledger"]
    change = np.array(
        [row["number_after"] - row["number_before"] for row in ledger]
    )
    axes[1].bar(
        np.arange(len(ledger)),
        change,
        color=np.where(
            change >= 0, plotting.COLORS["teal"], plotting.COLORS["rust"]
        ),
    )
    axes[1].set(
        xticks=np.arange(len(ledger)),
        xticklabels=[row["process"] for row in ledger],
        ylabel="First-cycle represented-number change",
        yscale="symlog",
    )
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].axhline(0, color="0.5", linewidth=0.8)
    fig.suptitle(title)
    for row in ledger:
        print(row)
    print(
        "Restart exact:",
        report["restart_exact"],
        "Capture exact:",
        report["capture_exact"],
        "Snapshot isolated:",
        report["snapshot_isolated"],
    )
    print("Explicit precursor removal (kg):", report["removed_precursor_kg"])
    plotting.fit(fig)
    return fig
