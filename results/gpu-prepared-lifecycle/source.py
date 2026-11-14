# %% [markdown]
# # A million-particle source, collision and loss lifecycle
#
# **4,096 boxes × 512 slots × 75% occupancy = 1,572,864 active particles.**
# Each particle has water and dry material; gas has an inert lane, a precursor
# and water. The ordered schedule is nucleation → Brownian coagulation →
# condensation → spherical wall loss → dilution, with four one-second cycles.
#
# The first cycle injects exactly two neutral, unit-count particles per box.
# Remaining precursor is then explicitly removed and its mass recorded. This
# source-off boundary matters: dilution changes surviving multiplicities, so
# later injection of unit-count particles would violate the current Brownian
# requirement for equal active counts within a box.
#
# Before the composed run, each process is observed separately to check its
# per-box, per-material source or sink. The composed first cycle must match
# those results exactly. A checkpoint then continues in a fresh session;
# captured and ordinary execution must produce identical complete state and
# random continuation. Editing detached inspection arrays must change neither.
# The downloadable ZIP includes the full auditable study helpers.

# %%
"""Manually refreshed large native CUDA lifecycle with exact continuation."""

import matplotlib.pyplot as plt
import warp as wp
from scripts.docs_gpu_figures import lifecycle_figures
from scripts.gpu_studies.cases import StudyCase, build_case, radius_histogram
from scripts.gpu_studies.trajectories import run_trajectory


def run_example():
    """Run the full million-particle workload, reference ledger and continuations."""
    case = StudyCase(
        "million-lifecycle",
        "full",
        4096,
        512,
        occupancy=0.75,
        time_step=1.0,
        steps=4,
        experiment="lifecycle",
    )
    initial, process, processes = build_case(case)
    _, histogram = radius_histogram(initial)
    report = run_trajectory(case, initial, process, processes)
    lifecycle_figures(
        report,
        histogram,
        "1.57 million active particles · " + wp.get_device("cuda:0").name,
    )
    print("Initial active count:", int(initial.particles.concentration.sum()))
    print("Slot capacity:", case.boxes * case.slots)
    plt.show()
    return report


# %% [markdown]
# ## Distinguish process effects from conservation
#
# Nucleation increases number, coagulation decreases it, and condensation can
# grow particles without changing their count. Wall loss removes complete
# resolved slots; dilution scales represented counts and gas concentration.
# The signed count panel shows the first cycle only. The final histogram
# includes all four cycles. Neither number nor total aerosol mass is conserved
# across wall loss and dilution; the per-process ledger accounts for them.
#
# This is a fixed-capacity illustration of supported physics, not a calibrated
# chamber. It uses real CUDA execution and float64 state. Refresh it manually;
# its full workload is excluded from routine CI and may use a longer timeout.

# %%
if __name__ == "__main__":
    run_example()
