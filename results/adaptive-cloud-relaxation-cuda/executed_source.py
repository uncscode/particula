# %% [markdown]
# # Resolve stiff wet–dry droplet relaxation on CUDA
#
# Can a device-resident stiff method resolve rapid evaporation without the
# repeated size oscillations shown in the [fixed-step comparison](adaptive_cloud_relaxation.py)?
# This example repeats that complete 0.4 s humidity-reset experiment with CUDA
# ROS3. The shared setup represents 10⁸ hygroscopic 100 nm dry seeds per m³,
# with kappa 0.5 and a finite water-vapor reservoir at 298.15 K. At 0.2 s the
# external reservoir is reset to 85% saturation. That imposed exchange is
# accounted for separately from closed-phase condensation.
#
# Download the complete ZIP to retain the shared physical setup and independently
# coded scalar reference. A real CUDA device is required. The one-representative-
# particle calculation tests accuracy and error control; it is not a throughput
# benchmark or qualification of stochastic cloud-chamber coupling.

# %%
import numpy as np
import warp as wp
from docs.examples.simulations.adaptive_cloud_relaxation import (
    C_SAT,
    DRY_VOLUME,
    NUMBER,
    R_DRY,
    RHO,
    TIMES,
    build_case,
    plot_results,
    reference,
)
from particula.execution.gpu_condensation_ode import GPUCondensationODE

# %% [markdown]
# ## Keep the humidity event outside adaptive trials
#
# Each output follows accepted condensation steps. Rejected trials cannot reset
# the reservoir or emit observations. Trial masses, derivatives and linear solves
# remain on CUDA; the host controller reads error/status scalars. The requested
# snapshots explicitly download accepted state. The controller restarts only at
# the physical humidity event, where the gas inventory is externally replaced.


# %%
def calculate(device, tolerance):
    """Run both complete phases with separate closed-phase inventory checks."""
    state, strategy = build_case()
    dry = state.particles.masses[:, :, 0].copy()
    owner = None
    rows, work = [], {}
    inventory = (
        NUMBER * state.particles.masses[0, 0, 1] + state.gas.concentration[0, 0]
    )
    budget_error = 0.0

    def retain_work(current):
        for name, value in vars(current.statistics).items():
            work[name] = work.get(name, 0) + value
        work["control_readbacks"] = (
            work.get("control_readbacks", 0) + current.control_readbacks
        )

    for i, time in enumerate(TIMES):
        if i:
            if owner is None:
                owner = GPUCondensationODE(
                    state,
                    strategy,
                    device=device,
                    method="ros3",
                    time=float(TIMES[i - 1]),
                    rtol=tolerance,
                    particle_atol=1e-27,
                    gas_atol=1e-13,
                    first_step=1e-6,
                    max_step=0.005,
                )
            while owner.time < time:
                owner.step(float(time))
            state = owner.snapshot()
        np.testing.assert_array_equal(state.particles.masses[:, :, 0], dry)
        total = (
            NUMBER * state.particles.masses[0, 0, 1]
            + state.gas.concentration[0, 0]
        )
        budget_error = max(
            budget_error, float(abs(total - inventory) / inventory)
        )
        if time == 0.2:
            state.gas.concentration[0, 0] = 0.85 * C_SAT
            inventory = (
                NUMBER * state.particles.masses[0, 0, 1]
                + state.gas.concentration[0, 0]
            )
            retain_work(owner)
            owner = None
        rows.append(
            [
                state.particles.masses[0, 0, 1] / (RHO * DRY_VOLUME),
                state.gas.concentration[0, 0],
            ]
        )
    retain_work(owner)
    assert budget_error < 2e-12
    return np.asarray(rows), work, budget_error


def run_cuda():
    """Cross-check references and require refinement on the actual CUDA device."""
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("This wet–dry comparison requires real CUDA")
    device = wp.get_device("cuda:0")
    reference_water, reference_gas = reference("Radau")
    check_water, check_gas = reference("BDF")
    np.testing.assert_allclose(
        check_water, reference_water, rtol=2e-8, atol=1e-9
    )
    np.testing.assert_allclose(check_gas, reference_gas, rtol=2e-10, atol=1e-12)
    reference_radius = R_DRY * np.cbrt(1 + reference_water)
    results = {}
    for tolerance in (1e-5, 1e-7):
        values, work, budget = calculate(device.alias, tolerance)
        radius = R_DRY * np.cbrt(1 + values[:, 0])
        error = float(np.max(abs(radius / reference_radius - 1)))
        assert error < 2e-5
        results[f"CUDA ROS3 {tolerance:.0e}"] = {
            "radius_nm": (radius * 1e9).tolist(),
            "gas_kg_m3": values[:, 1].tolist(),
            "radius_error": error,
            "budget_error": budget,
            "work": work,
        }
    coarse, tight = results.values()
    assert tight["radius_error"] < coarse["radius_error"] / 10
    print("CUDA device:", device.name)
    return {
        "time_s": TIMES.tolist(),
        "reference_radius_nm": (reference_radius * 1e9).tolist(),
        "results": results,
        "device": {"alias": device.alias, "name": device.name},
        "configuration": {
            "dry_radius_m": R_DRY,
            "number_m3": NUMBER,
            "kappa": 0.5,
        },
    }


# %% [markdown]
# ## Judge accuracy and budgets separately
#
# The plotted error compares every saved radius with the independent reference,
# including the rapid response immediately after the humidity reset. Tightening
# the tolerance must reduce the maximum error by at least a factor of ten, and
# both profiles must satisfy a 2×10⁻⁵ relative radius target. The separate mass
# check does not substitute for that trajectory comparison.
#
# To extend this experiment, vary dry size and composition while retaining an
# independent reference, or resolve a heterogeneous population and record the
# fastest box's effect on the shared controller. Those extensions are not run
# here. Coupling source or collision events also needs a fixed outer schedule,
# explicit transfer accounting and separate stochastic validation.

# %%
if __name__ == "__main__":
    report = run_cuda()
    plot_results(report)
