# %% [markdown]
# # Separate stochastic coagulation from initial-state variability
#
# Sixteen boxes each begin with 1,024 active, mixed-material particles.
# Temperature, pressure and volume differ between boxes. **The initial aerosol
# is identical across all sixteen realizations**; only the process root seed
# changes. Every realization advances forty one-second Brownian steps.
#
# CUDA computes exact Brownian pair maxima using bounded scratch storage, then
# follows the original sequential selection and random-stream semantics.
# Pair work remains quadratic in active particles per box. Each realization
# must conserve species mass and charge, decrease particle number, and produce
# nonzero collisions. The seed ensemble must contain distinct trajectories.
# Download the ZIP to include the shared study and plotting helpers.

# %%
"""Manually refreshed sixteen-seed CUDA particle-resolved coagulation study."""

import matplotlib.pyplot as plt
import warp as wp
from scripts.docs_gpu_figures import stochastic_figures
from scripts.gpu_studies.cases import StudyCase, build_case, radius_histogram
from scripts.gpu_studies.trajectories import run_trajectory


def run_example():
    """Run all sixteen seeds and retain their complete number histories."""
    case = StudyCase(
        "stochastic-16-seeds",
        "brownian",
        16,
        1024,
        time_step=1.0,
        steps=40,
        heterogeneous=True,
        experiment="stochastic",
    )
    initial, process, processes = build_case(case)
    _, histogram = radius_histogram(initial)
    report = run_trajectory(case, initial, process, processes)
    stochastic_figures(
        report,
        histogram,
        "Brownian particle histories · " + wp.get_device("cuda:0").name,
    )
    plt.show()
    return report


# %% [markdown]
# ## Read the shaded bands correctly
#
# Lines show seed means; shading spans the 5th–95th percentiles of these sixteen
# realizations. It describes stochastic spread, not a confidence interval or
# measurement uncertainty. The left panel compares number loss in selected
# boxes, normalized by each box's own initial concentration. The right panel
# shows how collisions move number toward larger radii in the middle box.
#
# Stable logical box identities keep random streams attached to boxes when
# a prepared simulation is segmented or checkpointed. CPU and CUDA algorithms
# need not consume identical random streams, so cross-backend byte equality
# is not claimed. This manual CUDA workload does not run in routine CI.

# %%
if __name__ == "__main__":
    run_example()
