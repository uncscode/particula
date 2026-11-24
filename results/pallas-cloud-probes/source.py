# %% [markdown]
# # Compare Pallas cloud-probe spectra and optical sizing
#
# How often would two probes place the same droplet on opposite sides of
# 13 µm? Compare measured spectra at Pallas with a forward simulation of
# scattering and holographic sizing. The deployed FSSP used 633 nm and
# ICEMET used 660 nm. The wavelength swap is a simulation control. This
# CPU example renders the retained field and model tables used in the
# [Pallas study](index.md).
#
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
from particula import plotting

plt.style.use("particula.default")
TEAL = plotting.COLORS["teal"]
RUST = plotting.COLORS["rust"]
INDIGO = plotting.COLORS["indigo"]


# %% [markdown]
# ## Read the comparison tables
#
# The two field spectra use observations matched by minute. The optical
# predictions use the cloud droplet analyzer as the independent input
# distribution. To reproduce the optical calculations that supplied these
# tables, follow the [study specification](specification.md).
#
# %%
if "saved_data" in globals():
    report = globals()["saved_data"]
else:
    names = ("distributions", "method-sizing", "method-budget")
    data_dir = Path(__file__).resolve().parent
    if not (data_dir / "distributions.csv").is_file():
        data_dir = Path.cwd()
    if not (data_dir / "distributions.csv").is_file():
        data_dir = (
            Path(__file__).resolve().parents[3] / "data/pallas-cloud-probes"
        )
    report = {
        name: pd.read_csv(data_dir / f"{name}.csv").to_dict("records")
        for name in names
    }


# %% [markdown]
# ## Compare the field spectra and predicted probe differences
#
# Convert number per interval to number per unit log diameter for the upper
# panels. The lower panels express FSSP minus ICEMET as a percentage of their
# mean, with the paired field uncertainty shaded. Compare the direction and
# size of each predicted difference below 13 µm on the two dates.
#
# %%
def field_figure(field):
    """Compare paired spectra and the corresponding signed model prediction."""
    field = field.query("high_um <= 21")
    fig, panels = plotting.subplots(
        2, 2, size="square", sharex=True, sharey="row"
    )
    ink = plt.rcParams["text.color"]
    for col, (day, group) in enumerate(field.groupby("day")):
        top, bottom = panels[:, col]
        edges = np.r_[group.low_um, group.high_um.iloc[-1]]
        width = np.diff(np.log10(edges))
        for name, color, stroke in [
            ("FSSP", TEAL, "-"),
            ("ICEMET", RUST, "--"),
        ]:
            top.stairs(
                group[name] / width,
                edges,
                baseline=None,
                label=name,
                color=color,
                linewidth=plt.rcParams["lines.linewidth"],
                linestyle=stroke,
            )
        top.set_title(f"{day} September", loc="left")
        bottom.stairs(
            group.percent_p975,
            edges,
            baseline=group.percent_p025,
            fill=True,
            color=TEAL,
            alpha=0.15,
        )
        bottom.stairs(
            group.percent,
            edges,
            baseline=None,
            color=ink,
            linestyle="-",
            linewidth=plt.rcParams["lines.linewidth"],
            label="Measured",
        )
        bottom.stairs(
            group.model_percent,
            edges,
            baseline=None,
            color=INDIGO,
            linestyle="--",
            linewidth=plt.rcParams["lines.linewidth"],
            label="Particula",
        )
        bottom.axhline(
            0, color=plt.rcParams["axes.edgecolor"], linestyle="-", lw=1
        )
        bottom.set_xlabel("Reported diameter (µm)")
        bottom.set_ylim(-125, 125)
        for ax in (top, bottom):
            ax.set_xlim(6, 21)
            ax.set_xscale("log")
            ticks = [6, 9, 13, 18, 21]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(
                FixedFormatter([str(v) for v in ticks])
            )
            ax.xaxis.set_minor_locator(NullLocator())
            ax.axvline(
                13, color=plt.rcParams["axes.edgecolor"], linestyle=":", lw=1.2
            )
    panels[0, 0].set_ylim(bottom=0)
    panels[0, 0].set_ylabel("dN/dlog₁₀D (cm⁻³)")
    panels[1, 0].set_ylabel("FSSP − ICEMET\n(% of mean)")
    plotting.legend(panels[0, 1], placement="inside", loc="upper right")
    plotting.legend(panels[1, 0], placement="inside", loc="upper left")
    plotting.fit(fig)
    return fig


field_figure(pd.DataFrame(report["distributions"]))
plt.show()


# %% [markdown]
# ## Follow a droplet of known size
#
# Set both simulated probes to 633 nm and retain each native calibration.
# Read horizontally from the true diameter to the mean registered diameter.
# The shaded interval covers the middle 68% of simulated registered sizes.
#
# %%
def size_figure(data):
    """Isolate sizing at one common simulated wavelength."""
    data = data.query("nm == 633")
    fig, axes = plotting.subplots(1, 2, size="square", sharex=True, sharey=True)
    for ax, (probe, label, color) in zip(
        axes, [("F", "FSSP", TEAL), ("I", "ICEMET", RUST)]
    ):
        group = data.loc[data.probe == probe]
        ax.set_title(label, loc="left")
        ax.fill_between(
            group.true_um, group.q16_um, group.q84_um, color=color, alpha=0.18
        )
        ax.plot(group.true_um, group.mean_um, color=color, linestyle="-")
        ref = plt.rcParams["axes.edgecolor"]
        ax.plot([6, 13], [6, 13], ls="--", lw=1.1, color=ref)
        ax.axhline(13, ls=":", color=ref, lw=1.2)
        example = group.loc[group.true_um == 11.96875].iloc[0]
        ax.scatter(
            example.true_um,
            example.mean_um,
            s=40,
            color=color,
            zorder=5,
            edgecolor="white",
            linewidth=0.7,
        )
        ax.annotate(
            f"{example.mean_um:.2f} µm",
            xy=(example.true_um, example.mean_um),
            xytext=(8.6, 14.5),
            arrowprops=dict(arrowstyle="-", color=color),
            color=color,
        )
        ax.set_xlabel("True diameter (µm)")
        ax.set_xlim(6, 13)
        ax.set_ylim(4, 15.5)
        ax.set_xticks([6, 8, 10, 12, 13])
        ax.set_yticks([4, 6, 8, 10, 12, 14])
    axes[0].set_ylabel("Reported diameter (µm)")
    plotting.fit(fig)
    return fig


size_figure(pd.DataFrame(report["method-sizing"]))
plt.show()


# %% [markdown]
# ## Reprocess the same images
#
# Compare the two wavelengths within a row. Then compare the ICEMET rows
# at the same wavelength to see how the correction changes the number of
# droplets assigned to the 6–13 µm band on 15 September.
#
# %%
def calibration_figure(scores):
    """Hold the input fixed while changing the imager calibration."""
    scores = scores.query("day == 15")
    fig, ax = plotting.subplots(size="wide")
    cases = [
        ("I", "uncorrected", "ICEMET\nuncorrected"),
        ("I", "nominal", "ICEMET"),
        ("F", "nominal", "FSSP"),
    ]
    ref = scores.true_band_cm3.iloc[0]
    for y, (probe, setting, _) in enumerate(cases):
        group = scores[(scores.probe == probe) & (scores.setting == setting)]
        for nm, dy, color, marker in [
            (633, 0.15, TEAL, "o"),
            (660, -0.15, RUST, "s"),
        ]:
            x = group.loc[group.nm == nm, "reported_band_cm3"].item()
            ax.scatter(
                x,
                y + dy,
                color=color,
                marker=marker,
                s=40,
                label=f"{nm} nm" if y == 0 else None,
            )
            ax.annotate(
                f"{x:.1f}",
                (x, y + dy),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                color=color,
            )
    ax.axvline(
        ref,
        ls=":",
        lw=1.5,
        color=plt.rcParams["axes.edgecolor"],
        label="True input",
    )
    ax.set_yticks(range(len(cases)), [case[2] for case in cases])
    ax.set_xlim(70, 98)
    ax.set_ylim(-0.55, 2.6)
    ax.set_xlabel("6–13 µm concentration (cm⁻³)")
    ax.grid(False, axis="y")
    ax.grid(True, axis="x")
    plotting.legend(ax, placement="right")
    plotting.fit(fig)
    return fig


calibration_figure(pd.DataFrame(report["method-budget"]))
plt.show()

# %% [markdown]
# The two simulated methods can assign one true size to different sides of
# the cutoff even at a common wavelength. The optical model still predicts
# the wrong sign for the integrated difference on 15 September. Independent
# size standards around 13 µm would test the calibration explanation.
