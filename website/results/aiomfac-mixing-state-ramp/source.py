# %% [markdown]
# # Which of 1,000 drying droplets first separate into two liquids?
#
# Two droplets in the same air can have different dissolved compositions.
# As humidity falls, each loses water at its own rate. Which mixing states
# develop a second liquid first, and which still contain one liquid at the
# last observation? This example follows the same 1,000 particle IDs through
# the ramp. Two coexisting liquids occupy the same droplet and retain its ID.
#
# ## Follow the individual inventories
#
# The particles contain water, 2-propanol, sodium chloride and potassium
# chloride. The three solutes are retained in each droplet while water
# exchanges with the vapor at 298.15 K. This controlled experiment isolates
# water-driven liquid separation. Salt crystallization is disabled in this
# liquid-only calculation. A volatile alcohol can also exchange with the
# atmosphere. The finite rate of water transfer and the instantaneous
# equilibrium of up to two internal liquids are separate model assumptions.
# [Activity and phase selection](../../../../guide/processes/activity.md) ·
# [Liquid coexistence equations](../../../../theory/thermodynamics/equilibria-theory.md)
#
# The random seed is 2012. Each particle begins with 7–13 pg of retained
# solute, of which 19–36% is salt. KCl supplies 15–70% of that salt by mass.
# A CPU water-uptake calculation sets the initial 90% RH equilibrium over
# each curved particle. The GPU independently checks that each state has
# one stable liquid before advancing the population.
# The surface tension is 0.072 N/m and the Kelvin density is 1,000 kg/m³.
# The box holds 1,000 unit-weight particles in 10 cm³ at 101,325 Pa.
#
# The simulation helper selects `ActivityFactory().get_strategy("aiomfac", …)`
# with `phase_mode="stable"`, in water, 2-propanol, NaCl, KCl order.
# A water-only `CondensationSpeciesMap` controls vapor exchange. The GPU
# resident condensation process evolves all particle masses for 0.006 s
# after each step change in vapor. Each of the 32 imposed RH steps removes
# 0.25 percentage points, ending at 82% after 0.192 s. Gas water can change
# again within a step. The plots use the RH from the resulting gas mass.
# The helper also checks each retained solute and the water budget across
# particles, vapor and the imposed changes.
#
# Use the complete example download to obtain the simulation helper. A
# source checkout also works. Install from the feature checkout using
# `python -m pip install -e "./packages/particula-aiomfac[cuda]" -e ".[plot]"`.
# On an NVIDIA GPU, run `python example.py --cuda` from the extracted ZIP. The
# calculation returns one row per saved time and one column per particle for
# water mass and phase count. Every comparison below gives each particle ID
# the same weight. To save every ID, RH interval and water trajectory in a
# machine-readable file, run the included helper as
# `python studies/aiomfac-mixing-state/ramp.py --particles 1000 --output ramp.json`.
# The complete download includes the [independent CPU solver](../../../../../studies/aiomfac-mixing-state/equilibrium.py).
# Use `python studies/aiomfac-mixing-state/equilibrium.py --ramp ramp.json --output equilibrium.json`
# to run its checks on an existing trajectory. The [checked numerical data](../../../../../website/results/aiomfac-mixing-state-ramp/data.json) include both calculations.
# %%
import argparse
import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.patches import Patch


def study_helper(filename):
    """Locate a declared helper in the complete download or source checkout."""
    helper = Path("studies/aiomfac-mixing-state") / filename
    if not helper.is_file():
        parents = Path(globals().get("__file__", ".")).resolve().parents
        helper = next(
            (base / helper for base in parents if (base / helper).is_file()),
            None,
        )
        if helper is None:
            raise FileNotFoundError(
                "Extract the complete example ZIP with both study helpers."
            )
    return runpy.run_path(str(helper))


def simulate():
    """Run the GPU trajectories, then independently solve the CPU cloud points."""
    report = study_helper("ramp.py")["run_ramp"](
        n_particles=1000, device="cuda:0"
    )
    report["equilibrium_control"] = study_helper("equilibrium.py")["analyze"](
        report
    )
    return report


# %% [markdown]
# ## Locate the first observed split
#
# If a droplet changes from one liquid at one observation to two at the
# next, those observations bracket the event. A droplet already split at
# the first observation has a left-censored onset in time. A droplet
# unsplit throughout the record has a right-censored onset in time. The
# initialization checks that all 1,000 droplets begin with one liquid.
# The first-split statistic and the fraction currently split answer
# different questions if a droplet subsequently returns to one liquid.
# %%
def first_splits(phase_count):
    """Classify first observation, bracketed onset and unobserved onset."""
    split = np.asarray(phase_count) == 2
    seen = np.any(split, axis=0)
    first = np.argmax(split, axis=0)
    initial = split[0]
    return first, initial, seen & ~initial, ~seen


# %% [markdown]
# ## Relate onset to the starting mixture
#
# The mass distributions show the range of retained inventories for each
# solute. Each point in the composition panel is one particle. Its horizontal
# and vertical coordinates give the total salt fraction of the dry mass
# and the KCl fraction within those salts. 2-propanol makes up the rest
# of the dry mass. Color gives the RH at the
# first **observation** of two liquids. Its preceding observation supplies
# the other endpoint of the onset interval. Distinct symbols identify
# censored particles.
# %%
@plt.style.context("particula.default")
def plot_mixing(report):
    """Compare individual solute masses with onset across the mixing states."""
    from particula import plotting

    dry = np.asarray(report["dry_mass_kg"])
    fractions = dry / dry.sum(axis=1, keepdims=True)
    rh = 100 * np.asarray(report["rh"])
    first, initial, detected, unseen = first_splits(report["phase_count"])
    fig, (left, right) = plotting.subplots(1, 2, size="wide")
    picograms = dry * 1e15
    positive = picograms[picograms > 0]
    bins = np.geomspace(positive.min() * 0.98, positive.max() * 1.02, 24)
    names = ("2-propanol", "NaCl", "KCl")
    for j, (label, color, line) in enumerate(
        zip(
            names,
            (
                plotting.COLORS["rust"],
                plotting.COLORS["indigo"],
                plotting.COLORS["teal"],
            ),
            ("-", "--", ":"),
            strict=True,
        )
    ):
        left.hist(
            picograms[:, j],
            bins=bins,
            histtype="step",
            linewidth=2,
            linestyle=line,
            color=color,
            label=label,
        )
    left.set(
        xscale="log",
        xlabel="Retained mass per droplet (pg)",
        ylabel="Droplets per log-spaced bin",
    )
    plotting.legend(left, placement="inside")
    salt = 100 * (fractions[:, 1] + fractions[:, 2])
    potassium_share = (
        100 * fractions[:, 2] / (fractions[:, 1] + fractions[:, 2])
    )
    if detected.any():
        onsets = rh[first[detected]]
        levels = np.unique(onsets)
        rh_step = np.min(np.abs(np.diff(np.unique(rh))))
        edges = np.r_[
            levels[0] - rh_step / 2,
            (levels[:-1] + levels[1:]) / 2,
            levels[-1] + rh_step / 2,
        ]
        ramp = plotting.colormap("magnitude")
        shades = colors.ListedColormap(ramp(np.linspace(0.32, 1, len(levels))))
        points = right.scatter(
            salt[detected],
            potassium_share[detected],
            c=onsets,
            s=13,
            cmap=shades,
            norm=colors.BoundaryNorm(edges, ncolors=shades.N),
            linewidths=0,
            alpha=0.85,
        )
        ticks = levels[
            np.linspace(0, len(levels) - 1, min(8, len(levels)))
            .round()
            .astype(int)
        ]
        fig.colorbar(
            points,
            ax=right,
            ticks=ticks,
            label="First sampled two-liquid RH (%)",
        )
    if unseen.any():
        right.scatter(
            salt[unseen],
            potassium_share[unseen],
            s=16,
            facecolors="none",
            edgecolors="#444444",
            linewidths=0.65,
            label="No split observed",
        )
    if initial.any():
        right.scatter(
            salt[initial],
            potassium_share[initial],
            s=20,
            marker="x",
            color=plotting.COLORS["rust"],
            linewidths=1,
            label="Split at first sample",
        )
    if unseen.any() or initial.any():
        plotting.legend(right, placement="inside")
    right.set(
        xlabel="Total salt (% of retained mass)",
        ylabel="KCl (% of retained salt mass)",
    )
    plotting.fit(fig)
    return fig


# %% [markdown]
# ## Read the sampled trajectories
#
# The phase map keeps every droplet as a separate row, sorted by increasing
# salt mass fraction. Each column represents a saved state. Color between
# column centers serves only to make the sampled states visible. The second
# panel summarizes the same particles on the time axis. Compare the count
# fraction with two liquids against the median water fraction of individual
# droplets and its 10th–90th percentile range. These percentiles describe
# the simulated population.
# %%
@plt.style.context("particula.default")
def plot_history(report):
    """Show per-ID liquid counts and water histories across the humidity ramp."""
    from particula import plotting

    dry = np.asarray(report["dry_mass_kg"])
    water = np.asarray(report["water_kg"])
    time = np.asarray(report["time_s"])
    rh = 100 * np.asarray(report["rh"])
    split = np.asarray(report["phase_count"]) == 2
    rank = np.argsort(dry[:, 1:].sum(axis=1) / dry.sum(axis=1), kind="stable")
    fig, (left, right) = plotting.subplots(1, 2, size="wide")
    palette = (plotting.colormap("magnitude")(0.12), plotting.COLORS["rust"])
    left.grid(False)
    left.imshow(
        split[:, rank].T,
        cmap=colors.ListedColormap(palette),
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
        origin="lower",
    )
    ticks = np.unique(
        np.linspace(0, len(rh) - 1, min(6, len(rh))).round().astype(int)
    )
    left.set_xticks(ticks, [f"{rh[i]:.1f}" for i in ticks])
    left.set(
        xlabel="Sampled ambient RH (%)",
        ylabel="Droplet rank by dry salt fraction",
    )
    plotting.legend(
        left,
        placement="inside",
        handles=[
            Patch(color=c, label=n)
            for c, n in zip(palette, ("One liquid", "Two liquids"), strict=True)
        ],
        loc="lower left",
        ncols=2,
    )
    water_fraction = water / (water + dry.sum(axis=1))
    lower, middle, upper = 100 * np.quantile(
        water_fraction, [0.1, 0.5, 0.9], axis=1
    )
    right.fill_between(
        time,
        lower,
        upper,
        color=plotting.COLORS["teal"],
        alpha=0.18,
        label="Water, 10th–90th percentiles",
    )
    right.plot(
        time,
        middle,
        color=plotting.COLORS["teal"],
        label="Median water mass fraction",
    )
    right.plot(
        time,
        100 * split.mean(axis=1),
        "o--",
        ms=3,
        color=plotting.COLORS["rust"],
        label="Droplets with two liquids",
    )
    right.set(xlabel="Elapsed time (s)", ylabel="Percent", ylim=(-2, 102))
    plotting.legend(right, placement="inside")
    plotting.fit(fig)
    return fig


# %% [markdown]
# ## At what droplet size and humidity does a second liquid appear?
#
# For each particle, the independent CPU calculation fixes its dry
# composition and solves for the water amount at which a distinct second
# liquid can first coexist. It equates water, alcohol and both independent
# salt chemical potentials across the two liquids and applies the same
# radius-dependent Kelvin correction. The result is the ambient RH for a
# completely equilibrated particle of that dry composition and mass.
# The left panel follows each whole droplet from its initial wet diameter at
# 90% RH to its wet diameter at first detection of two liquids. The dashed
# 1:1 line shows where the two diameters would be equal. At a given initial
# diameter, the vertical distance from the line down to a point is the
# diameter lost during drying. The short gray segment above each point
# reaches back to the preceding one-liquid observation.
# Diameters are volume-equivalent spheres, calculated from the same additive
# material volumes as the kinetic model, $D=(6\sum_i m_i/(π\rho_i))^{1/3}$.
# Color gives the initial salt fraction of the retained dry mass.
# On the right, compare the equilibrium RH for each dry composition with
# the two ambient humidities bracketing its kinetic onset.
# %%
@plt.style.context("particula.default")
def plot_cloud_points(report):
    """Relate the first split to initial size, dry composition and humidity."""
    from particula import plotting

    control = report["equilibrium_control"]["particle_cloud_points"]
    ids = np.asarray(report["particle_id"])
    if not np.array_equal(control["particle_id"], ids):
        raise ValueError(
            "Cloud-point IDs differ from the trajectory column order"
        )
    cloud_rh = 100 * np.asarray(control["equilibrium_rh"])
    first, _, detected, _ = first_splits(report["phase_count"])
    columns = np.flatnonzero(detected)
    end = first[columns]
    rh = 100 * np.asarray(report["rh"])

    dry = np.asarray(report["dry_mass_kg"])
    density = np.asarray(report["material_density_kg_m3"])
    water = np.asarray(report["water_kg"])
    volume = (dry / density[1:]).sum(axis=1) + water / density[0]
    diameter_um = 1e6 * np.cbrt(6 * volume / np.pi)
    initial = diameter_um[0, columns]
    at_split = diameter_um[end, columns]
    salt_percent = 100 * dry[:, 1:].sum(axis=1) / dry.sum(axis=1)

    fig, (size, humidity) = plotting.subplots(1, 2, size="square")
    size.vlines(
        initial,
        at_split,
        diameter_um[end - 1, columns],
        color="0.45",
        lw=0.8,
        alpha=0.45,
    )
    points = size.scatter(
        initial,
        at_split,
        c=salt_percent[columns],
        cmap=plotting.colormap("magnitude"),
        s=12,
        edgecolors=plotting.COLORS["teal"],
        linewidths=0.25,
    )
    fig.colorbar(points, ax=size, label="Salt (% of retained mass)")
    xmin, xmax = size.get_xlim()
    ymin, ymax = size.get_ylim()
    limits = (min(xmin, ymin), max(xmax, ymax))
    size.plot(
        limits,
        limits,
        "--",
        color="0.35",
        lw=1.3,
        zorder=1,
        label="1:1 (equal diameters)",
    )
    size.set(
        xlim=limits,
        ylim=limits,
        xlabel="Initial wet diameter (µm)",
        ylabel="Wet diameter at first split (µm)",
    )
    plotting.legend(size, placement="inside", loc="upper left")

    humidity.vlines(
        salt_percent[columns],
        rh[end],
        rh[end - 1],
        color=plotting.COLORS["rust"],
        alpha=0.35,
        linewidth=0.8,
        label="Kinetic onset interval",
    )
    humidity.scatter(
        salt_percent,
        cloud_rh,
        s=8,
        color=plotting.COLORS["teal"],
        linewidths=0,
        alpha=0.8,
        label="Equilibrium onset",
    )
    humidity.set(xlabel="Salt (% of retained mass)", ylabel="Ambient RH (%)")
    plotting.legend(humidity, placement="inside", loc="lower left")
    plotting.fit(fig)
    return fig


def run_example():
    """Run the complete population, report onset categories and draw all figures."""
    report = simulate()
    dry = np.asarray(report["dry_mass_kg"])
    ids = np.asarray(report["particle_id"])
    if len(np.unique(ids)) != 1000 or dry.shape != (1000, 3):
        raise ValueError(
            "Expected 1,000 distinct particles with three retained solutes"
        )
    if report["materials"] != ["water", "2-propanol", "NaCl", "KCl"]:
        raise ValueError("Native material ordering differs from plot labels")
    fractions = dry / dry.sum(axis=1, keepdims=True)
    if len(np.unique(fractions.round(12), axis=0)) != 1000:
        raise ValueError(
            "The population must contain 1,000 distinct dry mixing states"
        )
    phase = np.asarray(report["phase_count"])
    if (
        phase.shape != (len(report["rh"]), len(ids))
        or not np.isin(phase, (1, 2)).all()
    ):
        raise ValueError(
            "Expected one resolved liquid count per sample and particle"
        )
    first, initial, detected, unseen = first_splits(phase)
    print(
        f"{detected.sum()} onsets bracketed between samples, "
        f"{initial.sum()} already split at the first sample, "
        f"{unseen.sum()} never observed to split."
    )
    rh = 100 * np.asarray(report["rh"])
    if detected.any():
        k = first[detected]
        print(
            f"First detections span {rh[k].min():.2f}–{rh[k].max():.2f}% RH. "
            f"Largest interval width is {np.max(rh[k - 1] - rh[k]):.2f} percentage points."
        )
        for sample in np.unique([k.min(), k.max()]):
            members = np.flatnonzero(detected & (first == sample))
            j = members[0]
            print(
                f"{len(members)} first splits between {rh[sample - 1]:.2f} and "
                f"{rh[sample]:.2f}% RH, including ID {ids[j]} with "
                f"{100 * fractions[j, 1:].sum():.1f}% retained salt."
            )
    initial_water = sum(report["initial_water_kg"]) + report["gas_water_kg"][0]
    print(
        f"Maximum relative water-budget residual {np.max(np.abs(report['balance_residual_kg'])) / initial_water:.2e}."
    )
    control = report["equilibrium_control"]
    lo, hi = 100 * np.asarray(control["equilibrium_onset_rh_range"])
    print(f"Independent equilibrium cloud points span {lo:.2f}–{hi:.2f}% RH.")
    print(
        f"{control['disagreement_count']} phase-count differences in "
        f"{control['particle_sample_comparisons']:,} per-particle sample checks."
    )
    plot_mixing(report)
    plot_history(report)
    plot_cloud_points(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda",
        action="store_true",
        required=True,
        help="run the 1,000-droplet calculation on a real GPU",
    )
    parser.parse_args()
    run_example()
    plt.show()

# %% [markdown]
# ## What the ramp shows
#
# All 1,000 particles developed two liquids and remained split through the
# final sample. First detections occurred in six observations spanning
# **85.75% to 84.50% RH**. The 14 earliest detections followed a one-liquid
# observation at 86.00% RH. The 25 latest followed one at 84.75% RH.
# Median salt content was 19.8% of retained mass among those first 14
# particles and 35.4% among the last 25. Consecutive observations
# differ by 0.25 RH percentage points. No onset was censored in this run.
#
# Initial wet diameters ranged from 3.80 to 4.98 µm. At the first two-liquid
# observation they spanned 3.46 to 4.48 µm, with a median of 3.99 µm.
# All 1,000 points lie below the 1:1 line because every droplet had shrunk
# by the first detection of a second liquid. The median diameter loss was
# 9.3% of the initial wet diameter. Across the shorter interval from the
# last one-liquid sample to the first two-liquid sample, the median
# diameter change was 0.020 µm.
#
# The independent equilibrium cloud points span **85.13% to 86.21% RH**
# across the same 1,000 dry inventories, with curvature included. Water
# loss takes time along the imposed ramp. Every kinetic onset interval
# fell below its particle’s equilibrium cloud point. The median lag was
# between 0.31 and 0.56 RH percentage points. Both calculations agreed on
# the number of liquids at all 33,000 sampled particle compositions.
#
# Among the 250 particles with the least salt (19.0–23.5% of dry mass),
# the median first detection was 85.50% RH, against a median equilibrium
# onset of 86.01%. For the 250 with the most salt (31.5–36.0% of dry mass),
# these values were 84.75% and 85.33% RH. Both the equilibrium and kinetic
# calculations put the salt-rich group’s median onset at lower humidity.
# These are calculated thresholds for a liquid-only model with retained alcohol.
#
# The [equilibrium sweep](aiomfac_humidity_sweep.py) and
# [single evaporating droplet](aiomfac_evaporating_droplet.py) isolate those
# two calculations for a single starting mixture. A repeat ramp with a longer
# residence time at each humidity would test the role of kinetic lag.
# A time-step check kept the same 6 ms plateaus for 16 of the original IDs.
# Dividing each plateau into two steps moved two of their first detections
# one saved interval later. Two and four steps gave the same onset brackets
# for all 16. Their maximum water-mass difference was 0.13%.
