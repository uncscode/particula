# %% [markdown]
# # The lucky few: growth, charging and background capture
#
# A large ensemble makes a rare survival fraction measurable. Each independent
# box begins with one neutral **3 nm seed** and 32 neutral **100 nm background
# particles**. The question is whether the original seed lineage reaches
# **50 nm before a background merger**, under an imposed vapor feed and ion
# production of 0, 2 or 10 pairs cm⁻³ s⁻¹.
#
# This is an illustrative competition between growth and scavenging, not a
# prediction for a measured atmosphere. A 50 nm threshold is a size proxy;
# the example does not calculate cloud activation. Charging is not assumed to
# help: it can also enhance scavenging by background particles.
#
# Development manuscript: the production omission policy and numerical
# qualification are pending. This source deliberately requires an explicit
# policy argument. It is not yet registered as a published example.

# %%
"""Run the survival manuscript only with an explicitly selected rate policy."""

import argparse
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from scripts.gpu_studies.lucky_few import SurvivalConfig, run_survival

# %% [markdown]
# ## What is represented
#
# The observation horizon is 20,000 s (5.56 h).
# Each physical box occupies 3.2 × 10⁻¹⁰ m³; initial background concentration
# is 10¹¹ m⁻³. Increasing the number of boxes adds independent realizations
# without changing particle counts or physical volume within a box. Allocated
# slots are capacity, not particles. There are no statistical particle weights,
# resampling, adaptive volume changes, walls or dilution.
#
# The five material lanes retain condensed vapor, original seed material,
# original background material, positive-ion mass and negative-ion mass.
# Effective ion masses are 150 and 90 u; both free ions have diameter 0.5 nm
# and charge ±1 e. Paired Poisson production adds both signs, with mean qVΔt.
# There are no free ions initially. Attachment changes aerosol charge by
# explicit mergers. Opposite
# free ions recombine at 1.7 × 10⁻¹² m³ s⁻¹; their neutral products are inert
# until periodic cleanup exports their mass. Same-sign free ions do not form
# stable clusters in this model. No additional bulk attachment or recombination
# sink is applied on top of these collisions.
#
# Temperature and pressure are fixed at 298.15 K and 101325 Pa. The illustrative
# vapor has molar mass 0.2 kg/mol, diffusivity 5 × 10⁻⁶ m²/s,
# unit accommodation,
# reference vapor pressure 10⁻¹² Pa and surface tension 0.03 N/m. Condensation
# uses ideal molar activity and Kelvin density 1400 kg/m³. Vapor concentration
# is restored to 2 × 10⁻¹⁰ kg/m³ every 10 s: this is a pulsed external feed,
# not constant concentration between updates. Free ions and neutral products
# are excluded from condensation; aerosols with attached ions can still grow.
#
# Growth is continuous deterministic condensation. Ion production and
# collisions are stochastic. Molecular stochastic growth and ion-induced
# nucleation are outside this model. Background particles also grow and
# coagulate, so they are not an immutable first-order sink.

# %% [markdown]
# ## Attachment uncertainty and the outcome definition
#
# Aerosol–aerosol rates use the hard-sphere charged fit with Coulomb
# interactions. There is no additional image/polarization force between
# aerosols: at fixed masses and sizes, either zero aerosol charge removes the
# Coulomb correction. Ion–aerosol attachment uses LF2013 separately. The
# references motivate the question; reproducing their physical models would
# require matching their interaction laws.
#
# LF2013 supplies ion–aerosol attachment within its tabulated charge, size and
# reference-condition domains. Some repulsive coefficients below 10⁻¹⁵ m³/s
# were omitted. A singly charged 3 nm seed falls outside the same-sign fit's
# lower-radius limit. The library rejects these omissions by default.
#
# `lower` replaces only these documented omissions by zero; `upper` replaces
# them by 10⁻¹⁵ m³/s. Neither is an extrapolated best estimate. Running both is
# a sensitivity comparison between two rate assumptions. Their survival
# probabilities are **not mathematically guaranteed lower and upper bounds**:
# charge history, growth and background capture interact nonlinearly.
# Unsupported conditions outside the documented omission region still fail.
#
# Each original lineage is classified once: reached 50 nm, captured by any
# background merger before crossing, or unresolved at the observation horizon.
# Background capture takes precedence if the merger itself creates a particle
# above the threshold. Later scavenging does not undo an earlier success.
# By default the order is ion injection, condensation, coagulation, then
# cleanup. First passage is checked after physical processes; the ordering
# resolves competing changes within that step, not their exact continuous
# event times. Timestep and process-order sensitivity therefore matter.
# The explicit `coagulation_first` control reverses the two aerosol processes
# while retaining injection first and cleanup last.
#
# Original lineage counts, not the changing number of aggregate particles,
# define the denominator. One initial seed per independent box allows exact
# binomial intervals; unresolved lineages remain visible rather than being
# dropped from the denominator.


# %%
def run_example(config, policies):
    """Run the selected assumptions without choosing a production policy.

    ``config`` contains the already-qualified numerical settings. ``policies``
    must explicitly select one or both documented omitted-rate assumptions.
    The q=0 control is run once with strict lookup and no initial ions.
    """
    policies = tuple(policies)
    if not policies or len(set(policies)) != len(policies):
        raise ValueError("select distinct explicit omission policies")
    if any(policy not in ("lower", "upper") for policy in policies):
        raise ValueError("policies must contain lower and/or upper")
    config.validate()
    if not wp.is_cuda_available():
        raise RuntimeError("This example requires CUDA; no CPU fallback")
    print(f"Device: {wp.get_device('cuda:0').name}")
    print(f"Configuration: {config}")
    cases = [(0, "reject")]
    cases.extend((q, policy) for q in (2, 10) for policy in policies)
    results = []
    for q, policy in cases:
        current = replace(config, ion_pairs_cm3_s=q, omitted=policy)
        result = run_survival(current)
        reached, captured, unresolved = result["outcomes"][-1]
        low, high, zero_upper = result["success_interval95"]
        print(
            f"q={q} pairs cm^-3 s^-1, omitted={policy}: "
            f"reached/captured/unresolved={reached}/{captured}/{unresolved}; "
            f"success={reached / config.boxes:.5f}, "
            f"exact 95% interval=[{low:.5f}, {high:.5f}]; "
            f"maximum mass residual={result['maximum_mass_residual']:.2e}; "
            f"observed peak slots={result['maximum_observed_slots']}/"
            f"{config.capacity}"
        )
        if zero_upper is not None:
            print(f"Zero events: one-sided 95% upper bound={zero_upper:.5f}")
        print(
            f"Preparation/capture={result['preparation_seconds']:.3f} s; "
            f"explicit gas forcing={result['forcing_seconds']:.3f} s; "
            "synchronized stepping="
            f"{result['synchronized_compute_seconds']:.3f} s"
        )
        results.append(result)
    plot_results(results)
    return results


def plot_results(results):
    """Plot reduced lineage summaries; no particle states are downloaded."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    labels, final = [], []
    for index, result in enumerate(results):
        config = result["config"]
        q, policy = config["ion_pairs_cm3_s"], config["omitted"]
        label = "No ions" if q == 0 else f"q={q:g}, {policy}"
        labels.append(label)
        fractions = result["outcomes"] / config["boxes"]
        final.append(fractions[-1])
        low, high, _ = result["success_interval95"]
        value = fractions[-1, 0]
        axes[0].plot(result["time"] / 3600, fractions[:, 0], label=label)
        axes[1].errorbar(
            index,
            value,
            yerr=np.array([[value - low], [high - value]]),
            fmt="o",
            capsize=4,
        )
    final = np.asarray(final)
    bottom = np.zeros(len(labels))
    for index, name in enumerate(("Reached", "Captured", "Unresolved")):
        axes[2].bar(labels, final[:, index], bottom=bottom, label=name)
        bottom += final[:, index]
    axes[0].set(xlabel="Time [h]", ylabel="Fraction reaching 50 nm")
    axes[0].legend(fontsize=8)
    axes[1].set(
        xticks=np.arange(len(labels)),
        xticklabels=labels,
        ylabel="Success fraction with exact 95% interval",
    )
    axes[2].set(ylabel="Fraction of original seed lineages", ylim=(0, 1))
    axes[2].legend(fontsize=8)
    for axis in axes[1:]:
        axis.tick_params(axis="x", rotation=30)
    return figure


# %% [markdown]
# ## Reading the result
#
# The curves show cumulative first passages; they are not instantaneous counts
# of surviving particles. The final intervals quantify finite-ensemble sampling
# uncertainty for each case separately, not parameter uncertainty or a joint
# confidence band. Overlapping intervals alone do not establish equivalence.
# A probability-difference interval and a declared tolerance are used for
# numerical qualification. A zero-success case gives an upper bound rather
# than proving survival impossible.
#
# The runner checks each species' mass inventory, including signed vapor feed,
# ion injection and exported neutral products; it also checks exact net charge
# and the reached/captured/unresolved partition in every box.
# Capacity exhaustion raises an error. Observed occupancy at diagnostic times
# is not a guaranteed instantaneous maximum between observations.
#
# GPU value here is the scale of independent histories and repeated stochastic
# collision work. Report preparation and forcing separately from synchronized
# stepping. These timings exclude diagnostic transfers, plotting and analysis;
# a CPU speedup requires a matched CPU workload and is not inferred here.
# Production conclusions require qualified timestep, cleanup interval, capacity
# and ensemble size, plus sensitivity to the unresolved attachment assumptions.
#
# Primary context:
# - [Gopalakrishnan and Hogan (2012)](https://doi.org/10.1103/PhysRevE.85.026410):
#   Coulombic collision scaling and the hard-sphere fit.
# - [Lehtinen et al. (2007), growth versus scavenging](https://doi.org/10.1016/j.jaerosci.2007.06.009).
# - [Mahfouz and Donahue (2021), charge-enhanced scavenging can reduce survival](https://doi.org/10.1029/2021GL092758).
# - [Olenius et al. (2018), stochastic effects on nanoparticle growth](https://doi.org/10.1038/s41598-018-32610-z);
#   this example uses continuous vapor growth, not their discrete growth model.
# - [López-Yglesias and Flagan (2013), ion attachment](https://doi.org/10.1080/02786826.2013.783684).
# - [Zauner-Wieczorek et al. (2022), ion–ion recombination](https://doi.org/10.5194/acp-22-12443-2022).
# - [Yu (2010), ionization sensitivity](https://doi.org/10.1029/2009JD012630),
#   and [Rose et al. (2013), observed ion sources and sinks](https://doi.org/10.5194/acp-13-11573-2013).
# - [Clopper and Pearson (1934), exact binomial intervals](https://doi.org/10.1093/biomet/26.4.404).

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="CUDA execution is required for this study.",
    )
    parser.add_argument(
        "--omitted", required=True, choices=("lower", "upper", "both")
    )
    parser.add_argument("--boxes", type=int, required=True)
    parser.add_argument("--time-step", type=float, required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--cleanup-seconds", type=float, required=True)
    parser.add_argument("--root-seed", type=int, required=True)
    parser.add_argument("--coagulation-first", action="store_true")
    args = parser.parse_args()
    selected = ("lower", "upper") if args.omitted == "both" else (args.omitted,)
    run_example(
        SurvivalConfig(
            boxes=args.boxes,
            time_step=args.time_step,
            capacity=args.capacity,
            cleanup_seconds=args.cleanup_seconds,
            root_seed=args.root_seed,
            coagulation_first=args.coagulation_first,
        ),
        selected,
    )
