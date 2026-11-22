"""Plot the frozen campaign predictions against reserved size observations."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_spectrum_history(spectra, path):
    """Compare digitized and modeled run 24 spectra on the instrument grid."""
    selected = spectra.query("case == 'selected'")
    if selected.empty:
        selected = spectra.query("case == 'incumbent'")
    cells = selected.groupby(
        ["time_s", "low_nm", "high_nm"], as_index=False
    ).agg(
        observed=("dndlog10d_cm3", "first"),
        modeled=("predicted", "mean"),
        used=("used", "first"),
    )
    # The supplied cells are point samples in time and intervals in diameter.
    # Extend each time sample halfway to its neighbor and clip at 0 and 5 h.
    times = np.sort(cells.time_s.unique()) / 3600
    time_edges = np.r_[times[0], (times[:-1] + times[1:]) / 2, times[-1]]
    bins = cells[["low_nm", "high_nm"]].drop_duplicates().sort_values("low_nm")
    np.testing.assert_allclose(bins.high_nm.iloc[:-1], bins.low_nm.iloc[1:])
    edges = np.r_[bins.low_nm, bins.high_nm.iloc[-1]]

    def matrix(key):
        return cells.pivot(
            index="low_nm", columns="time_s", values=key
        ).to_numpy()

    observed, modeled = matrix("observed") / 1e4, matrix("modeled") / 1e4
    if (
        np.isnan(modeled).any()
        or not np.isfinite(observed[matrix("used")]).all()
    ):
        raise ValueError("Incomplete modeled or scored spectrum")
    difference = modeled - observed
    total_max = max(1, np.ceil(max(np.nanmax(observed), np.max(modeled))))
    delta_max = max(1, np.ceil(np.nanmax(np.abs(difference))))

    fig, axs = plt.subplots(
        1,
        3,
        figsize=(11.4, 4.1),
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    for ax, title, values in zip(
        axs,
        (
            "Manchester measurements",
            "Frozen chemistry",
            "Model minus measurement",
        ),
        (observed, modeled, difference),
        strict=True,
    ):
        residual = ax is axs[-1]
        mesh = ax.pcolormesh(
            time_edges,
            edges,
            values,
            shading="flat",
            vmin=-delta_max if residual else 0,
            vmax=delta_max if residual else total_max,
            cmap="RdBu_r" if residual else "viridis",
            rasterized=True,
        )
        ax.set(xlabel="Hours after lights-on", yscale="log", xlim=(0, 5))
        ax.set_xticks(np.arange(6))
        ax.set_title(title, loc="left", fontsize=11)
        if residual or ax is axs[0]:
            flagged = cells.loc[~cells.used]
            ax.scatter(
                flagged.time_s / 3600,
                np.sqrt(flagged.low_nm * flagged.high_nm),
                marker="s",
                s=28,
                linewidth=1.1,
                edgecolors="white",
                facecolors="none",
            )
        if ax is axs[0]:
            abundance = mesh
    axs[0].set_ylabel("Dry mobility diameter (nm)")
    axs[0].set_yticks(
        [30, 50, 100, 200, 400], ["30", "50", "100", "200", "400"]
    )
    cb = fig.colorbar(
        abundance,
        ax=axs[:2],
        location="bottom",
        shrink=0.76,
        pad=0.04,
        aspect=36,
    )
    cb.set_label("dN/dlog₁₀D (10⁴ particles cm⁻³)")
    cb = fig.colorbar(
        mesh, ax=axs[-1], location="bottom", shrink=0.87, pad=0.04, aspect=22
    )
    cb.set_label("Difference (10⁴ particles cm⁻³)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    """Draw the two-hour spectrum, geometric means and modal errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()

    def table(name):
        frame = pd.read_csv(args.results / name)
        return frame.query("particles == 2048 and step_s == 15").copy()

    full_spectra = table("psd.csv.gz")
    spectra = full_spectra.query("time_s == 7200 and used")
    diameter = table("gmd.csv.gz")
    mode = table("mode_by_run.csv")
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    colors = {"incumbent": "#006d77", "selected": "#a5491a"}
    titles = [
        "Run 24 at two hours",
        "Run 24 over five hours",
        "Seven runs with modal diameter",
    ]
    for ax, title in zip(axs, titles, strict=True):
        ax.set_title(title, loc="left", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    ref = spectra.drop_duplicates(["time_s", "low_nm"]).sort_values("low_nm")
    x = np.sqrt(ref.low_nm * ref.high_nm)
    axs[0].errorbar(
        x,
        ref.dndlog10d_cm3 / 1e4,
        yerr=np.vstack(
            (
                np.minimum(ref.dndlog10d_cm3, ref.reading_band_cm3),
                ref.reading_band_cm3,
            )
        )
        / 1e4,
        fmt="o",
        color="#242424",
        ms=3,
        lw=0.8,
        label="Manchester",
    )
    gm = diameter.drop_duplicates("time_s").sort_values("time_s")
    axs[1].errorbar(
        gm.time_s / 3600,
        gm.gmd_nm,
        yerr=gm.reading_band_nm,
        fmt="o",
        color="#242424",
        ms=2.5,
        lw=0.7,
    )
    summaries = table("summary.csv")
    show_both_as_one = (
        "selected" in summaries.case.unique()
        and summaries.query("case == 'selected'").identical_reused.all()
    )
    for label in ["incumbent"] if show_both_as_one else spectra.case.unique():
        color = colors.get(label)
        g = (
            spectra[spectra.case == label]
            .groupby("low_nm")
            .predicted.agg(["mean", "min", "max"])
        )
        axs[0].plot(
            x,
            g["mean"] / 1e4,
            label="Frozen chemistry"
            if show_both_as_one
            else label.capitalize(),
            color=color,
            lw=1.7,
        )
        axs[0].fill_between(
            x, g["min"] / 1e4, g["max"] / 1e4, color=color, alpha=0.18
        )
        g = (
            diameter[diameter.case == label]
            .groupby("time_s")
            .predicted_nm.agg(["mean", "min", "max"])
        )
        axs[1].plot(g.index / 3600, g["mean"], color=color, lw=1.7)
        axs[1].fill_between(
            g.index / 3600, g["min"], g["max"], color=color, alpha=0.18
        )
        g = (
            mode[mode.case == label]
            .groupby("run")
            .mae_nm.agg(["mean", "min", "max"])
        )
        ofs = (
            0 if show_both_as_one else (-0.12 if label == "incumbent" else 0.12)
        )
        axs[2].errorbar(
            np.arange(len(g)) + ofs,
            g["mean"],
            [g["mean"] - g["min"], g["max"] - g["mean"]],
            color=color,
            fmt="o",
            ms=4,
            capsize=2,
        )
        axs[2].set_xticks(np.arange(len(g)), g.index)
    axs[0].set(
        xscale="log",
        xlabel="Dry mobility diameter (nm)",
        ylabel="dN/dlog₁₀D (10⁴ particles cm⁻³)",
    )
    axs[0].legend(frameon=False, fontsize=8)
    axs[1].set(
        xlabel="Time after lights-on (h)", ylabel="Geometric mean diameter (nm)"
    )
    axs[2].set(xlabel="Manchester run", ylabel="Mean absolute modal error (nm)")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)
    plot_spectrum_history(
        full_spectra, args.figure.with_stem(args.figure.stem + "-history")
    )


if __name__ == "__main__":
    main()
