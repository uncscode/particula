# %% [markdown]
# # Choosing a coagulation timestep from native work diagnostics
#
# Can a bimodal population larger than the former 256-slot limit coagulate
# without losing collision demand? This example combines Brownian motion,
# differential settling and turbulent shear in one physical box. Preflight
# exposes the trial demand before a step; excessive demand raises an error.
#
# The default is explicitly **Warp CPU**, suitable for inspecting the workflow.
# Pass `device="cuda:0"` to `simulate` for native CUDA; it never falls back.
# Exact pair reductions have quadratic work even though their storage is linear.
# The timestep below meets a resource bound, not a numerical accuracy tolerance.

# %%
"""Demonstrate additive native sampling with resource admission and inventories."""

import time

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from matplotlib import style as mplstyle
from particula import plotting
from particula.gpu import from_warp_particle_data, to_warp_particle_data
from particula.gpu.kernels.coagulation import (
    CoagulationMechanismConfig,
    CoagulationWorkConfig,
    coagulation_preflight_gpu,
    coagulation_step_gpu,
)
from particula.particles import ParticleData


def simulate(device="cpu", sizes=(257, 2000, 5000)):
    """Measure one admitted step for each equal-count bimodal population."""
    wp.init()
    if device.startswith("cuda") and not wp.is_cuda_available():
        raise RuntimeError("This requested run requires native CUDA")
    rows = []
    for count in sizes:
        # Ten percent collector droplets deliberately resolve the minority mode.
        small = count - count // 10
        radius = np.concatenate(
            (
                np.geomspace(50e-9, 200e-9, small),
                np.geomspace(2e-6, 8e-6, count - small),
            )
        )
        masses = (4 * np.pi / 3 * 1000 * radius**3)[None, :, None]
        cpu = ParticleData(
            masses=masses,
            concentration=np.ones((1, count)),
            charge=np.zeros((1, count)),
            density=np.array([1000.0]),
            volume=np.array([count / 2e9]),
            distribution_type="particle_resolved",
        )
        particles = to_warp_particle_data(cpu, device=device)
        config = CoagulationMechanismConfig(
            (
                "brownian",
                "sedimentation_sp2016",
                "turbulent_shear_st1956",
            )
        )
        inputs = dict(
            temperature=283.15,
            pressure=101325.0,
            max_collisions=count // 2,
            mechanism_config=config,
            work_config=CoagulationWorkConfig(max_trials_per_box=256),
            turbulent_dissipation=1e-4,
            fluid_density=1.2,
        )
        start = time.perf_counter()
        report = coagulation_preflight_gpu(particles, time_step=1.0, **inputs)
        inspection_seconds = time.perf_counter() - start
        step = min(1000.0, 0.5 * float(report["max_time_step"][0]))
        counts = wp.zeros((1, 2), dtype=wp.int32, device=device)
        _, _, collisions = coagulation_step_gpu(
            particles,
            time_step=step,
            rng_seed=17,
            work_counts=counts,
            **inputs,
        )
        wp.synchronize_device(device)
        after = from_warp_particle_data(particles)
        initial_mass = np.sum(cpu.masses * cpu.concentration[..., None])
        final_mass = np.sum(after.masses * after.concentration[..., None])
        np.testing.assert_allclose(final_mass, initial_mass, rtol=2e-14)
        np.testing.assert_array_equal(after.charge, 0.0)
        accepted = int(collisions.numpy()[0])
        assert np.sum(after.concentration) == count - accepted
        rows.append(
            dict(
                particles=count,
                timestep_s=step,
                collisions=accepted,
                scheduled_trials=int(counts.numpy()[0, 0]),
                attempted_trials=int(counts.numpy()[0, 1]),
                expected_trials_at_1s=float(report["expected_trials"][0]),
                workspace_bytes=report["workspace_and_collision_bytes"],
                inspection_seconds=inspection_seconds,
            )
        )
    return {"device": device, "warp_version": wp.__version__, "rows": rows}


@mplstyle.context("particula.default")
def plot_results(report):
    """Show demand and selected resource-admitted step without timing claims."""
    rows = report["rows"]
    fig, axes = plotting.subplots(1, 2, size="wide")
    sizes = [row["particles"] for row in rows]
    axes[0].plot(sizes, [r["expected_trials_at_1s"] for r in rows], "o-")
    axes[0].set(xlabel="Active particles", ylabel="Expected trials at 1 s")
    axes[1].plot(sizes, [r["timestep_s"] for r in rows], "o-")
    axes[1].set(xlabel="Active particles", ylabel="Admitted timestep (s)")
    fig.suptitle(f"Brownian + settling + shear — {report['device']}")
    plotting.fit(fig)
    return fig


# %% [markdown]
# Higher particle counts can require smaller steps even at fixed number density.
# The outputs verify a complete admitted step and conserved material; they do not
# establish timestep convergence or physical scavenging accuracy. Preflight
# timings include first-use compilation and must not be read as kernel benchmarks.
#
# To study rare collectors, repeat with more particles and inspect ensemble and
# timestep convergence. At a 10^6:1 number ratio, 10,000 equal-count IID samples
# contain only about 0.01 minority particles in expectation. Removing a cap does
# not resolve that sampling problem, and unequal weights require another algorithm.

# %%
if __name__ == "__main__":
    report = simulate()
    print(report)
    figure = plot_results(report)
    plt.show()
