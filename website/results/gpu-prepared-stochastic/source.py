# %% [markdown]
# # Separate stochastic coagulation from initial-state variability
#
# The figure uses a new CUDA run with the archived scientific code. Its
# library, example source and input files matched the original publication
# by SHA-256. With the same one-second interval, current Particula requests
# 6,512 collision trials in box zero and has capacity for 512.
#
# Would a second simulation tell the same physical story if only its collision
# seed changed? This experiment isolates process randomness from initial sampling.
# It teaches you to compare an ensemble spread with a proposed model effect.
# Start with [particle-resolved coagulation](../../../tutorials/coagulation-populations/coagulation_3_particle_resolved_pattern.py)
# for the number-loss and material-conservation distinction.
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
import numpy as np
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
    report["initial_histogram"] = histogram.tolist()
    plot_results(report)
    plt.show()
    return report


def plot_results(report):
    """Draw the measured histories from a completed CUDA run."""
    return stochastic_figures(
        report,
        np.asarray(report["initial_histogram"]),
        "Brownian particle histories",
    )


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

# %% [markdown]
# ## Compare a proposed effect with the spread
#
# Changing only process seeds produces a distribution of number and size outcomes
# from the same starting state. The shaded bands summarize these sixteen runs;
# they do not quantify uncertainty in concentration, material properties, or the
# collision model, and sixteen seeds may poorly resolve rare tails.
#
# To test a small environmental effect, repeat both conditions over a larger seed
# ensemble while holding initial populations controlled. Compare effect size with
# seed-to-seed variation and verify that conclusions persist under timestep and
# sample-size refinement. The [lucky-few study](gpu_lucky_few.py) extends this idea
# to the much more demanding question of uncommon large-particle growth.
