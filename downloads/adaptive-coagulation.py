# %% [markdown]
# # Cloud scavenging over a fixed observation interval
#
# How many native collision passes does a cloud box need between observations?
# Follow two synthetic, bimodal populations for 600 s, observing every 100 s.
# The sparse mode contains 10% water droplets. Smaller aerosol particles and
# droplets each carry one represented particle. The sampler combines Brownian
# motion, differential settling (SP2016) and turbulent shear (ST1956).
# Run on a CUDA device. See the [prepared quick start](gpu_prepared_quick_start.py)
# for the upload and checkpoint lifecycle.

# %%
"""Recover an evolving CUDA coagulation study from a bounded work rejection."""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from particula import Aerosol
from particula.dynamics import (
    BrownianCoagulationStrategy,
    Coagulation,
    CombineCoagulationStrategy,
    SedimentationCoagulationStrategy,
    TurbulentShearCoagulationStrategy,
)
from particula.gas import EnvironmentData, GasData
from particula.gpu.kernels.coagulation import (
    CoagulationSubcyclingConfig,
    CoagulationSubcyclingError,
    CoagulationWorkConfig,
)
from particula.particles import ParticleData


def cloud(slots):
    """Build a one-box mixture at 2,000 particles per cubic centimetre."""
    collectors = slots // 10
    radii = np.r_[
        np.geomspace(50e-9, 200e-9, slots - collectors),
        np.geomspace(2e-6, 8e-6, collectors),
    ]
    particles = ParticleData(
        masses=(4 * np.pi / 3 * 1000 * radii**3)[None, :, None],
        concentration=np.ones((1, slots)),
        charge=np.zeros((1, slots)),
        density=np.array([1000.0]),
        volume=np.array([slots / 2e9]),
        distribution_type="particle_resolved",
    )
    return Aerosol(
        particles=particles,
        gas=GasData(
            ["air"], np.array([0.029]), np.zeros((1, 1)), np.array([False])
        ),
        environment=EnvironmentData(
            np.array([283.15]), np.array([101325.0]), np.zeros((1, 1))
        ),
    )


# %% [markdown]
# Start with a one-pass limit. If a live interval needs more work, checkpoint
# the committed state and raise the pass allowance. Replay the same observation
# interval on a newly captured owner. Inspect the number of accepted passes,
# collision totals and the wall time of the synchronous step call. An extra
# status check on the idle stream measures the observation overhead separately.


# %%
def simulate():
    """Run two physical resolutions at the same density and observation times."""
    model = "particle_resolved"
    process = Coagulation(
        CombineCoagulationStrategy(
            [
                BrownianCoagulationStrategy(model),
                SedimentationCoagulationStrategy(model),
                TurbulentShearCoagulationStrategy(model, 1e-4, 1.2),
            ]
        )
    )
    ceiling = 64 * 1024**2
    rows = []
    for slots in (5000, 10000):
        initial = cloud(slots)
        options = dict(
            time_step=100.0,
            device="cuda:0",
            root_seed=17,
            coagulation_work_config=CoagulationWorkConfig(
                max_trials_per_box=256
            ),
            coagulation_subcycling=CoagulationSubcyclingConfig(
                0.01, 1, ceiling
            ),
        )
        sim = process.prepare(initial, **options)
        retried = False
        events = []
        try:
            sim.capture()
            while sim.completed_steps < 6:
                start = perf_counter()
                try:
                    sim.step()
                except CoagulationSubcyclingError:
                    if retried:
                        raise
                    saved = sim.checkpoint()
                    sim.close()
                    policy = CoagulationSubcyclingConfig(0.01, 128, ceiling)
                    sim = process.resume(saved, retry_subcycling=policy)
                    sim.capture()
                    retried = True
                    continue
                seconds = perf_counter() - start
                start = perf_counter()
                sim.synchronize()
                idle_sync_seconds = perf_counter() - start
                diag = sim.coagulation_diagnostics()
                events.append(
                    dict(
                        time_s=sim.simulated_time,
                        passes=diag["substep_count"],
                        seconds=seconds,
                        idle_sync_seconds=idle_sync_seconds,
                        collisions=int(diag["accepted_collisions"].sum()),
                    )
                )
            final = sim.snapshot().particles
            np.testing.assert_array_equal(
                final.concentration.sum(),
                slots - sum(e["collisions"] for e in events),
            )
            np.testing.assert_array_equal(
                final.concentration[final.concentration > 0], 1.0
            )
            np.testing.assert_array_equal(final.charge, 0.0)
            relative_drift = (
                np.sum(final.masses * final.concentration[..., None])
                / np.sum(initial.particles.masses)
                - 1
            )
            assert (
                abs(relative_drift) < 2e-14
                and sim.simulated_time == 600
                and retried
            )
            rows.append(
                dict(
                    slots=slots,
                    events=events,
                    allocation=diag["allocation_bytes"],
                    relative_mass_drift=float(relative_drift),
                )
            )
        finally:
            sim.close()
    return dict(device=wp.get_device("cuda:0").name, rows=rows)


# %% [markdown]
# A step in the first panel counts how often the live sampler had to recompute
# rates. The second panel tracks the accepted mergers. Compare their shapes:
# trial demand is set by the fastest possible pair in a box, while acceptance
# also depends on the particular pairs drawn.


# %%
def plot_results(report):
    """Plot observed subcycle counts and cumulative accepted mergers."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    for row in report["rows"]:
        events = row["events"]
        times = [e["time_s"] for e in events]
        label = f"{row['slots']:,} initial particles"
        axes[0].plot(times, [e["passes"] for e in events], "o-", label=label)
        axes[1].plot(
            times,
            np.cumsum([e["collisions"] for e in events]),
            "o-",
            label=label,
        )
    for ax in axes:
        ax.set_xlabel("Simulated time (s)")
    axes[0].set_ylabel("Passes per 100 s interval")
    axes[1].set_ylabel("Cumulative mergers")
    axes[0].legend()
    return fig


if __name__ == "__main__":
    report = simulate()
    print(report)
    plot_results(report)
    plt.show()

# %% [markdown]
# Both curves represent one stochastic realization at each resolution. The
# conservation and count checks address bookkeeping. To estimate uncertainty
# in scavenging rates, repeat across independent seeds and refine the observation
# interval while keeping concentration and the size distribution fixed.
