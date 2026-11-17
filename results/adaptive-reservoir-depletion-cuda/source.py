# %% [markdown]
# # Check adaptive CUDA growth against an independent depletion reference
#
# This companion repeats the [finite-reservoir growth calculation](adaptive_reservoir_depletion.py)
# with native CUDA RK23 in addition to the CPU methods. Download the complete
# example ZIP to retain the shared physical setup and scalar reference.
# A real CUDA device is required; this example never substitutes a CPU backend.
#
# Accepted state and trial arrays stay on the device. The controller reads scalar
# error/status reductions, and each requested observation explicitly downloads a
# snapshot. This one-representative-particle case establishes numerical accuracy,
# not useful acceleration or throughput. It cannot be captured as a fixed CUDA
# graph, and it does not qualify stiff wet-to-dry relaxation on the GPU.

# %%
import matplotlib.pyplot as plt
import warp as wp
from docs.examples.simulations.adaptive_reservoir_depletion import (
    plot_results,
    simulate,
)


def run_cuda():
    """Require CUDA and qualify both tolerances against the scalar reference."""
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("This comparison requires a real CUDA device")
    device = wp.get_device("cuda:0")
    print("CUDA device:", device.name)
    report = simulate(device=device.alias)
    report["device"] = {"alias": device.alias, "name": device.name}
    return report


# %% [markdown]
# Every adaptive profile must meet the same predeclared radius and vapor error
# bounds. Each backend also has to improve when its tolerance is tightened.
# Compare these errors with the independent reference rather than treating
# CPU/GPU agreement alone as validation. A heterogeneous chamber adds stiffness,
# shared-controller constraints and stochastic scheduling that this case excludes.
# In the retained RTX 5060 run, tightening CUDA RK23 tolerance from 10⁻⁴ to
# 10⁻⁷ reduces maximum relative radius error from 1.82×10⁻⁵ to 2.57×10⁻⁸.
# The corresponding initial-vapor-normalized errors are 6.74×10⁻⁵ and 9.15×10⁻⁸;
# relative mass residuals remain below 2×10⁻¹⁵. These checks support the stated
# accuracy for this case, without claiming GPU performance from a tiny population.

# %%
if __name__ == "__main__":
    report = run_cuda()
    figure = plot_results(report)
    plt.show()
