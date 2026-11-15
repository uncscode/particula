# %% [markdown]
# # Sixty-four heterogeneous aerosol boxes on one GPU
#
# What does an ensemble average hide when every box starts under different
# conditions? Here you will compare per-box inventories and distributions without
# mistaking co-varying inputs for isolated causes. Read the
# [prepared quick start](gpu_prepared_quick_start.py) for state ownership first.
#
# Each box contains 4,096 slots, with 75% initially active: **196,608 resolved
# particles** in total. Temperature varies from 275 to 310 K, pressure from
# 85 to 105 kPa, relative humidity from 0.8 to 1.8, and physical volume by a
# factor of sixteen. Water fraction also varies with box and particle radius.
# These inputs co-vary deliberately; this is a heterogeneous response study,
# not an isolated estimate of temperature sensitivity.
#
# Water condensation and dilution evolve for 6,000 steps of 1 microsecond
# (0.006 s total), with four condensation substeps per step. Two particle
# materials map into three gases, including an inert lane. At twelve observation times,
# detached snapshots check each box's particle-plus-gas water and dry inventory
# against the independent exponential dilution factor. The reusable study
# helpers are included in the downloadable ZIP; they are also used by the
# controlled benchmark campaign. The same trajectory is repeated at half the
# timestep: every particle's final total mass must change by less than 1%, and
# each box's final water vapor by less than 0.1%. Conservation cannot detect an
# unstable integration; the former coarse timestep produced oscillations.

# %%
"""Manually refreshed heterogeneous CUDA condensation study."""

import matplotlib.pyplot as plt
import warp as wp
from scripts.docs_gpu_figures import history_figures
from scripts.gpu_studies.cases import StudyCase, build_case
from scripts.gpu_studies.trajectories import run_trajectory


def run_example():
    """Execute the full declared workload and all inventory assertions."""
    case = StudyCase(
        "heterogeneous-evolution",
        "condensation",
        64,
        4096,
        occupancy=0.75,
        time_step=1e-6,
        steps=6000,
        heterogeneous=True,
        experiment="heterogeneous",
    )
    initial, process, processes = build_case(case)
    report = run_trajectory(case, initial, process, processes)
    history_figures(
        report,
        "64 heterogeneous boxes · " + wp.get_device("cuda:0").name,
        forcing=False,
    )
    print("Active particles:", int(initial.particles.concentration.sum()))
    print("Timestep refinement:", report["refinement"])
    plt.show()
    return report


# %% [markdown]
# ## Read the response
#
# The first panel compares final particle water uptake across the heterogeneous
# boxes. The saturation histories show gas depletion and equilibration; the
# distribution panels expose broadening and growth that a total-mass curve
# would hide. The heatmap retains every box rather than averaging their
# different initial conditions into one apparent population.
#
# This complete CUDA workload is refreshed deliberately. CI checks its source
# and saved outputs without rerunning it. Reproduction requires the matching
# development version of Particula and an available CUDA GPU.

# %%
if __name__ == "__main__":
    run_example()

# %% [markdown]
# ## Preserve the differences that matter to your question
#
# Keeping each box visible lets this calculation expose heterogeneous response
# that a pooled size distribution would obscure. The independent dilution ledger
# checks inventory; the halved timestep checks selected final values. Neither
# turns the co-varying temperature, pressure, water fraction, and volume into a
# causal estimate of any one parameter.
#
# For a temperature-sensitivity study, keep initial number density, composition,
# pressure, and saturation definition controlled while varying temperature.
# Compare the water uptake per box and size-resolved growth, then refine both
# timestep and parameter spacing. The [environment sweep](gpu_environment_sweep.py)
# provides a more controlled route through environmental comparisons.
