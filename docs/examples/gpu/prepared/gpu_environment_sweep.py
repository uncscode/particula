# %% [markdown]
# # Two million charged particles across sixteen environments
#
# This CUDA example starts with **2,097,152 active resolved particles**:
# 16 conditions × 1,024 independent boxes × 128 particles. All boxes remain
# resident together. Each particle has its own material masses and integer
# charge; its initial multiplicity is one. Boxes are well mixed, with no
# positions, electric fields or exchange between boxes.
#
# The factorial grid contains temperatures of 278.15 and 298.15 K, pressures
# of 70,000 and 101,325 Pa, and initial relative humidities of 40, 60, 80 and
# 90%. These are controlled environments, **not an altitude trajectory**.
# Each condition starts with the same sampled particle population. The box
# volume keeps initial number concentration at 5 × 10¹³ m⁻³, an illustrative
# dense aerosol. All physical parameters here are demonstration assumptions.
#
# Water condensation/evaporation acts first, followed by charge-dependent
# Brownian coagulation. A captured schedule advances 1,000 steps of 1 µs,
# totaling 1 ms. This resolves fast water-growth transients while making
# comparatively rare collisions visible across a large population. It is
# not a long-term atmospheric evolution or a CPU/GPU speed comparison.

# %%
"""Run a manually refreshed two-million-particle charged CUDA sweep."""

import matplotlib.pyplot as plt
import numpy as np
from scripts.gpu_studies.environment_sweep import (
    EnvironmentSweep,
    run_environment_sweep,
)

# %% [markdown]
# ## Physical model and limitations
#
# Dry radii follow a clipped lognormal distribution: geometric mean 35 nm,
# geometric standard deviation 1.6, bounds 15–150 nm. Dry density is
# 1,400 kg m⁻³. Initial water volume is 20% of dry volume. Water density is
# 1,000 kg m⁻³, surface tension 0.072 N m⁻¹ and accommodation coefficient one.
# The κ-water activity model uses illustrative dry κ = 0.3, with Kelvin
# curvature and Buck saturation vapor pressure. Every four initial particles
# carry charges −1, 0, 0, +1 elementary charges. Collisions add charge and
# both material masses; there is no free-ion charging source in this example.
#
# Vapor diffusivity is held at **2 × 10⁻⁵ m² s⁻¹** across conditions. The
# prepared interface accepts a species constant, not a per-box diffusion
# law. Thus this is a controlled transport approximation, not a complete
# prediction of temperature/pressure dependence. The supported formulas
# still use each box's temperature and pressure. Temperature is prescribed;
# latent heat does not change it. Water vapor is finite and unreplenished,
# so the specified RH is initial RH, not a humidity controller.
#
# The charged hard-sphere model uses Coulomb-modified Brownian collision
# rates. No empirical ion-attachment fit is extrapolated to these conditions.
# The separate [Lucky Few example](gpu_lucky_few.py) includes explicit ions
# at its attachment model's fixed reference conditions.


# %%
def run_example():
    """Require CUDA, check closed budgets and plot all sixteen conditions."""
    report, final = run_environment_sweep(EnvironmentSweep())
    del final
    history = report["history"]
    grid = np.array(report["conditions"])
    time_ms = np.array([row["time"] for row in history]) * 1000
    water = np.array([row["water_mass"] for row in history])
    counts = np.array([row["count"] for row in history])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for panel, ax in enumerate(axes.flat):
        for i in range(panel * 4, panel * 4 + 4):
            ax.plot(
                time_ms,
                water[:, i] / water[0, i],
                marker=".",
                label=f"Initial RH {grid[i, 2]:.0%}",
            )
        temperature, pressure, _ = grid[panel * 4]
        ax.set(
            title=f"{temperature:.2f} K · {pressure / 1000:g} kPa",
            xlabel="Time [ms]",
            ylabel="Particle water mass / initial",
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Closed water uptake · 2,097,152 initial charged particles")

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    mergers = counts[0] - counts[-1]
    colors = plt.get_cmap("viridis")(np.tile(np.linspace(0.15, 0.9, 4), 4))
    ax.barh(np.arange(16), mergers, color=colors)
    ax.set_yticks(
        np.arange(16),
        [f"{t:.2f} K · {p / 1000:g} kPa · RH {rh:.0%}" for t, p, rh in grid],
    )
    ax.tick_params(axis="y", labelsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(mergers.max() * 1.1, 1))
    ax.set(
        xlabel="Accepted mergers in 1 ms",
        title="Rare collisions across 131,072 initial particles per condition",
    )
    ax.bar_label(ax.containers[0], fontsize=8)
    print(f"Device: {report['device']}")
    print(f"Initial active particles: {report['initial_particles']:,}")
    print(f"Final active particles: {counts[-1].sum():,}")
    print(f"Total accepted mergers: {mergers.sum():,}")
    print(
        "Maximum per-box species-mass relative residual:",
        f"{report['maximum_mass_relative_error']:.3e}",
    )
    print("Net charge is exactly conserved in every box at every diagnostic.")
    print(f"Preparation/capture: {report['preparation_seconds']:.2f} s")
    print(f"Synchronized stepping: {report['compute_seconds']:.2f} s")
    print("T [K] | P [Pa] | initial RH | final mean RH | mergers")
    for i, (temperature, pressure, rh) in enumerate(grid):
        print(
            f"{temperature:.2f} | {pressure:.0f} | {rh:.2f} | "
            f"{history[-1]['rh'][i]:.6f} | {mergers[i]}"
        )
    print(
        "Counts are one stochastic realization; differences are not significance tests."
    )
    return report


# %% [markdown]
# ## Read the result
#
# The water curves distinguish evaporation, uptake and finite-time response.
# Each bar counts actual accepted mergers, inferred from the loss of resolved
# particles in this closed system. Each condition contains 1,024 independent
# boxes, but the bars are single-realization totals, without confidence
# intervals. They do not establish a statistically resolved environmental
# effect or charge enhancement relative to a neutral control.
#
# Every diagnostic checks water gas + particle water, conserved dry material
# and exact net charge in each individual box. Mass tolerance is 2 × 10⁻¹¹
# relative. Final masses must be finite and nonnegative. Separate small CUDA
# tests halve the water-growth timestep (1% particle mass, 0.1% vapor
# tolerances), compare CPU/CUDA condensation, and check κ-Köhler equilibrium
# against an independently written expression. Those tests qualify water
# integration; they do not establish stochastic convergence of the collision
# counts. The 1 ms endpoint need not be equilibrium for every particle.
#
# Preparation and synchronized stepping timings exclude Python construction,
# diagnostic reductions, state download and plotting. The downloadable ZIP
# includes the helper; run `python example.py` in a compatible CUDA environment.
# The full manual refresh takes minutes and is not executed by routine CI.
#
# ## References
#
# - [Petters and Kreidenweis (2007)](https://doi.org/10.5194/acp-7-1961-2007):
#   κ representation of water activity and hygroscopic growth. The selected
#   κ value here is illustrative, not assigned to a measured material.
# - [Gopalakrishnan and Hogan (2012)](https://doi.org/10.1103/PhysRevE.85.026410):
#   hard-sphere transition-regime fit and charged collision scaling.
# - The [prepared execution guide](../../../guide/gpu/prepared.md) documents
#   resident ownership, CUDA capture and supported process contracts.

# %%
if __name__ == "__main__":
    run_example()
