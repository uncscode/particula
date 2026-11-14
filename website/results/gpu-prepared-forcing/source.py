# %% [markdown]
# # Cooling, heating and vapor pulses across sixteen boxes
#
# This study uses **49,152 active resolved particles**, four particle materials
# and five gas lanes. Only water exchanges between gas and particle material;
# the other materials and spectator gases provide independent accounting
# checks. The complete trajectory contains 6,000 steps of 1 microsecond
# (0.006 s total). This is an illustrative rapid forcing experiment.
#
# A prescribed temperature cycle cools the boxes toward 275 K, heats them
# toward 310 K, then returns toward 298 K. Each box has a fixed temperature
# offset, updated at 120 fixed control times. Water gas is replaced twice, at nominal saturation ratios 1.6 and
# 0.8. These replacements are external mass sources or sinks, so their actual
# per-box mass changes are added to the conservation ledger before dilution.
#
# `update_environment` and `update_gas` are explicit upload boundaries.
# They change array values in place, so a captured schedule can keep replaying.
# The study helpers and plotting code are included in the downloadable ZIP.
# A second run halves the integration timestep while retaining those exact
# control times. Every particle total mass must change by less than 1%, and
# every box's water vapor by less than 0.1%.

# %%
"""Manually refreshed CUDA forcing and independent-species-width study."""

import matplotlib.pyplot as plt
import warp as wp
from scripts.docs_gpu_figures import history_figures
from scripts.gpu_studies.cases import StudyCase, build_case
from scripts.gpu_studies.trajectories import run_trajectory


def run_example():
    """Run prescribed forcing and require every observed source/sink balance."""
    case = StudyCase(
        "time-varying-16",
        "condensation",
        16,
        4096,
        materials=4,
        gases=5,
        occupancy=0.75,
        time_step=1e-6,
        steps=6000,
        heterogeneous=True,
        experiment="forcing",
    )
    initial, process, processes = build_case(case)
    report = run_trajectory(case, initial, process, processes)
    history_figures(
        report,
        "Cooling and vapor pulses · " + wp.get_device("cuda:0").name,
        forcing=True,
    )
    print(
        "Shapes: particle materials",
        initial.particles.masses.shape,
        "gas species",
        initial.gas.concentration.shape,
    )
    print("Timestep refinement:", report["refinement"])
    plt.show()
    return report


# %% [markdown]
# ## Interpret forcing separately from response
#
# Temperature is prescribed. Saturation is derived from the current gas and
# the water vapor-pressure law. A gas pulse is not spontaneous particle mass
# creation: the displayed inventory residual includes that external exchange.
# Final size distributions can retain different histories even after the
# temperatures approach each other. This illustrates finite-rate response;
# a predictive study should choose its timestep and forcing resolution for
# the physical timescales being investigated.
#
# This manual CUDA refresh is excluded from routine CI execution.

# %%
if __name__ == "__main__":
    run_example()
