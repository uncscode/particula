"""Redraw the seven REACh comparison figures directly from the published tables."""

from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "docs/images/reach-droplets"
ARCHIVE = ROOT / "website/public/study-data/reach-droplets/figure-data.zip"
CENTER = np.arange(11.5, 79)
cases = ["import_above", "import_below", "local_growth", "local_mergers"]
names = ["Imported from above", "Imported from below", "Local condensation", "Local mergers"]
colors = ["#195e81", "#647385", "#2c8569", "#b85529"]


def table(name):
    with ZipFile(ARCHIVE) as archive, archive.open(name + ".csv") as fh:
        return pd.read_csv(fh)


def save(fig, name, dpi):
    fig.savefig(HERE / (name + ".webp"), dpi=dpi, pil_kwargs={"lossless": True, "method": 6})


NAMES = ["Hologram", "Forward physics", "Fit before 5 s", "Blocked fit", "Spatial update"]
COLORS = ["#15191b", "#817b72", "#257766", "#326bb0", "#af512b"]
STYLES = ["-", "--", "-.", "-", "-"]
T = np.arange(3.125, 7, 0.25)
FANS = {"forced": "Fans on", "unforced": "Fans off"}


def draw(ax, x, y, i):
    ax.plot(
        x, y, lw=2 if i == 0 else 1.55, alpha=0.94, color=COLORS[i], ls=STYLES[i], label=NAMES[i]
    )


def legend(fig, ax):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", frameon=False, ncol=5, fontsize=9)


def label(ax):
    ax.axvspan(5, 5.5, color="#c38e32", alpha=0.12, lw=0)
    ax.axvspan(6.25, 7, color="#528d9c", alpha=0.11, lw=0)
    ax.set_xlim(3, 7)


def plot_history(vals):
    """Render one standalone figure per fan setting on shared physical scales."""
    density_scale = PowerNorm(0.55, vmin=0, vmax=0.25)
    offset = 0.005
    time_edges = np.arange(3, 7.01, 0.25)
    diameter_edges = np.arange(11, 80)
    for run, fan in FANS.items():
        fig, axes = plt.subplots(5, 2, figsize=(9, 8.8), layout="constrained")
        fig.suptitle(f"{fan}  |  Wet-droplet size distribution", fontsize=16)
        measured = vals[run, "Hologram"]
        for row, name in enumerate(NAMES):
            spectrum = vals[run, name]
            left, right = axes[row]
            density_mesh = left.pcolormesh(
                time_edges, diameter_edges, spectrum.T, cmap="magma", norm=density_scale
            )
            left.set_title("Measured hologram" if row == 0 else name, loc="left", fontsize=10)
            panels = [left]
            if row:
                delta = np.log10((spectrum + offset) / (measured + offset))
                delta_mesh = right.pcolormesh(
                    time_edges, diameter_edges, delta.T, cmap="RdBu_r", vmin=-1, vmax=1
                )
                panels.append(right)
                if row == 1:
                    right.set_title("Model relative to measurement", loc="left", fontsize=10)
            for ax in panels:
                for t in (5, 5.5, 6.25):
                    ax.axvline(
                        t, color="white" if ax is left else "#424242", alpha=0.65, lw=0.85, ls=":"
                    )
                ax.set_yticks([20, 40, 60])
                ax.tick_params(labelbottom=row == len(NAMES) - 1, labelleft=ax is left)
                if row == len(NAMES) - 1:
                    ax.set_xlabel("Recording time (s)")
        fig.supylabel("Wet diameter (µm)", fontsize=11)
        key = axes[0, 1]
        key.set_axis_off()
        for mesh, y, extend, ticks, label in [
            (
                density_mesh,
                0.80,
                "max",
                [0, 0.05, 0.1, 0.15, 0.2, 0.25],
                "Number density (cm⁻³ µm⁻¹)",
            ),
            (
                delta_mesh,
                0.27,
                "both",
                [-1, -0.5, 0, 0.5, 1],
                "Δ log₁₀ density    ·    blue = low    ·    red = high",
            ),
        ]:
            bar = fig.colorbar(
                mesh,
                cax=key.inset_axes([0.025, y, 0.945, 0.085]),
                orientation="horizontal",
                extend=extend,
                ticks=ticks,
            )
            bar.set_label(label, fontsize=9)
            bar.ax.tick_params(labelsize=8)
        suffix = "fan-on" if run == "forced" else "fan-off"
        save(fig, "distribution-history-" + suffix, 150)
        plt.close(fig)


def plot(vals, mom):
    plot_history(vals)
    fig, axes = plt.subplots(
        3, 2, figsize=(11, 8.3), sharex="row", sharey="row", layout="constrained"
    )
    uncert = table("tail-uncertainty")
    real = table("pooled-observed")
    for j, (run, fan) in enumerate(FANS.items()):
        r = real[real.run == run]
        for i, name in enumerate(NAMES):
            draw(
                axes[0, j],
                CENTER,
                r.observed.to_numpy() if i == 0 else vals[run, name][T >= 4].mean(0),
                i,
            )
            q = mom[(mom.run == run) & (mom.series == name)]
            draw(axes[1, j], T, q.mean_diameter_um, i)
            draw(axes[2, j], T, q.large_cm3, i)
        u = uncert[
            (uncert.run == run)
            & (np.isclose(uncert.stop_s - uncert.start_s, 0.25))
            & (uncert.start_s >= 3)
        ]
        axes[2, j].fill_between(
            u.start_s + 0.125, u.block_low_cm3, u.block_high_cm3, color="black", alpha=0.12, lw=0
        )
        for ax in axes[:, j]:
            ax.grid(axis="y", alpha=0.2)
        for ax in axes[1:, j]:
            label(ax)
            ax.set_xlabel("Recording time (s)")
        axes[0, j].set(title=fan, xlabel="Wet droplet diameter (µm)", xlim=(11, 79), ylim=(0, None))
    maximum = max(line.get_ydata().max() for ax in axes[0] for line in ax.lines)
    axes[0, 0].set_ylim(0, 1.05 * maximum)
    axes[0, 0].set_ylabel("4–7 s mean density\n(cm⁻³ µm⁻¹)")
    axes[1, 0].set_ylabel("Number-weighted mean\nwet diameter (µm)")
    axes[2, 0].set_ylabel("50–65 µm droplets\n(cm⁻³)")
    legend(fig, axes[0, 0])
    save(fig, "chamber-comparison", 155)
    plt.close(fig)


def ranked(vals):
    best = table("ranked-configurations").drop_duplicates("kind").iloc[:4]
    data = table("best-four")
    labels = {
        "joint": "Joint size-spectrum fit",
        "redistribute": "Spatial redistribution",
        "smooth-pairs": "Size-dependent pair kernel",
        "entry": "Small-droplet entry",
        "reservoir": "Initial hidden population",
        "transport-time": "Time-varying transport",
        "collisions": "Collision-only adjustment",
        "environment": "Thermodynamic adjustment",
        "standard": "Standard kernels",
    }
    fig, axes = plt.subplots(
        2, 2, figsize=(10, 6.5), layout="constrained", sharex="row", sharey="row"
    )
    for j, (run, fan) in enumerate(FANS.items()):
        o = vals[run, "Hologram"]
        tail = (CENTER >= 50) & (CENTER < 65)
        for i, (_, row) in enumerate(best.iterrows()):
            q = data[(data.run == run) & (data.family == row.kind)]
            y = q.pivot(index="time_s", columns="diameter_um", values="density_cm3_um").to_numpy()
            axes[0, j].plot(
                CENTER,
                y[(T >= 5) & (T < 5.5)].mean(0),
                c=COLORS[i + 1],
                label=labels[row.kind],
                lw=1.6,
            )
            axes[1, j].plot(T, y[:, tail].sum(1), c=COLORS[i + 1], lw=1.6)
        axes[0, j].plot(CENTER, o[(T >= 5) & (T < 5.5)].mean(0), c="black", lw=2, label="Measured")
        axes[1, j].plot(T, o[:, tail].sum(1), c="black", lw=2)
        axes[0, j].set(xlabel="Wet droplet diameter (µm)", title=fan)
        axes[1, j].set(xlabel="Recording time (s)", xlim=(3, 7))
        for lo, hi in [(5, 5.5), (6.25, 7)]:
            axes[1, j].axvspan(lo, hi, alpha=0.13, color="#848484", lw=0)
    axes[0, 0].set_ylabel("5–5.5 s density\n(cm⁻³ µm⁻¹)")
    axes[1, 0].set_ylabel("50–65 µm droplets (cm⁻³)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False)
    save(fig, "best-four", 145)
    plt.close(fig)


def matched_plot():
    tails = table("matched-tails")
    spectra = table("matched-spectra")
    fig, axes = plt.subplots(
        2, 2, figsize=(10, 7), layout="constrained", sharex="col", sharey="col"
    )
    for k, ce in enumerate(["import_above", "local_mergers"]):
        for view, c, lab in [
            ("A", "#222426", "A · original height"),
            ("above55", "#066c8a", "B · 5.5 cm above"),
        ]:
            curve = tails[(tails["case"] == ce) & (tails.view == view)]
            axes[k, 0].plot(curve.time_s, curve.excess_cm3, color=c, label=lab, lw=2)
            group = spectra[(spectra["case"] == ce) & (spectra.view == view)]
            for start, style, window in [(4.75, "--", "4.75–5.00 s"), (5.0, "-", "5.00–5.25 s")]:
                curve = group[group.start_s == start]
                axes[k, 1].plot(
                    curve.diameter_um,
                    curve.excess_density_cm3_um,
                    c=c,
                    ls=style,
                    lw=1.8,
                    label=f"{lab[0]} · {window}",
                )
        for ax in axes[k]:
            ax.axhline(0, color="#999", lw=0.7)
            ax.grid(axis="y", alpha=0.18)
        axes[k, 0].axvspan(5, 5.25, color="#dfbd75", alpha=0.23, lw=0)
        axes[k, 0].set(
            xlim=(4.15, 6.5),
            ylabel=["Imported from above", "Local mergers"][k] + "\nExcess 50–65 µm (cm⁻³)",
        )
        axes[k, 1].set(xlim=(15, 78), ylabel="Change in density\n(cm⁻³ µm⁻¹)")
    axes[0, 0].set_title("Two synchronized cameras")
    axes[0, 1].set_title("Which sizes arrive or disappear?")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=8, ncol=2)
    axes[1, 0].set_xlabel("Recording time (s)")
    axes[1, 1].set_xlabel("Wet diameter (µm)")
    save(fig, "matched-cameras", 150)
    plt.close(fig)


def source_plot():
    tails = table("source-tails")
    spectra = table("source-spectra")
    fig, axes = plt.subplots(
        2, 3, figsize=(13, 7.2), layout="constrained", sharex="row", sharey="row"
    )
    for j, view in enumerate(["below55", "A", "above55"]):
        for ce, name, col in zip(cases, names, colors):
            curve = tails[(tails["case"] == ce) & (tails.view == view)]
            axes[0, j].plot(curve.time_s, curve.excess_cm3, color=col, label=name, lw=1.8)
            curve = spectra[(spectra["case"] == ce) & (spectra.view == view)]
            axes[1, j].plot(curve.diameter_um, curve.excess_density_cm3_um, color=col, lw=1.7)
        axes[0, j].set(
            title=["B · 5.5 cm below", "A · original height", "B · 5.5 cm above"][j],
            xlim=(4.15, 6.3),
            xlabel="Recording time (s)",
        )
        axes[1, j].set(xlim=(11, 79), xlabel="Wet diameter (µm)")
        for ax in axes[:, j]:
            ax.axhline(0, c="#888", lw=0.8)
            ax.grid(axis="y", alpha=0.18)
    axes[0, 0].set_ylabel("Excess 50–65 µm (cm⁻³)")
    axes[1, 0].set_ylabel("4.75–5.25 s change in\ndensity (cm⁻³ µm⁻¹)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4, frameon=False)
    save(fig, "source-comparison", 140)
    plt.close(fig)


def placement_plot(tab):
    from matplotlib.patches import Circle

    fig, (a, b) = plt.subplots(
        1, 2, figsize=(9, 4), layout="constrained", gridspec_kw={"width_ratios": [1, 1.45]}
    )
    a.add_patch(Circle((0, 0), 6.75, color="#7f8988", fill=False, lw=1.3))
    for name, dz, dx in table("view-positions").itertuples(index=False, name=None):
        color = "#066c8a" if name == "above55" else "#1b2628" if name == "A" else "#8c9294"
        a.plot(dx, dz, "s", ms=7, color=color)
    a.text(0.5, 5.55, "B", color="#066c8a", weight="bold")
    a.text(0.5, 0.05, "A", weight="bold")
    a.set(
        xlim=(-7.4, 7.4),
        ylim=(-7.4, 7.4),
        aspect="equal",
        xlabel="Horizontal offset (cm)",
        ylabel="Vertical offset (cm)",
        title="Looking along the optical axis",
    )
    labels = {
        "above55": "5.5 cm above",
        "below55": "5.5 cm below",
        "above25": "2.5 cm above",
        "below25": "2.5 cm below",
        "left40": "4 cm left",
        "right40": "4 cm right",
    }
    order = tab.drop("A").sort_values("min")
    floor = table("placement-floor").set_index("view").loc[order.index]
    y = np.arange(len(order))
    b.barh(y + 0.18, order["min"], height=0.32, color="#066c8a", label="Normalize by peak at A")
    b.barh(y - 0.18, floor["min"], height=0.32, color="#af6337", label="With 0.01 cm⁻³ floor")
    b.set_yticks(y, [labels[v] for v in order.index])
    b.legend(fontsize=8, frameon=False)
    b.set(xlabel="Smallest normalized RMS contrast", title="Separation across tested conditions")
    b.grid(axis="x", alpha=0.2)
    save(fig, "placement", 150)
    plt.close(fig)


if __name__ == "__main__":
    spectra = table("spectra")
    values = {
        (run, series): group.pivot(
            index="t_mid_s", columns="diameter_um", values="density_cm-3_um-1"
        ).to_numpy()
        for (run, series), group in spectra.groupby(["run", "series"])
    }
    plot(values, table("time-series"))
    ranked(values)
    matched_plot()
    source_plot()
    placement_plot(table("placement-summary").set_index("view"))
