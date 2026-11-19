# %% [markdown]
# # Follow bipolar coagulation through a CUDA restart
#
# How often does an oppositely charged pair merge before a chosen time?
# Start each independent box with the same two salt spheres. A collision
# removes one particle and leaves one neutral survivor. For this special
# case, the continuous-time probability of a merger is known analytically.
#
# ## Prepare the population and inventories
#
# Each box begins with a 150 nm and a 220 nm sphere. They have equal unit
# counts, charges +1 and −1, and volume 10⁻¹³ m³. The two mass channels serve
# as tracers with the same material density and dielectric properties.

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style as mplstyle
from particula import plotting
from particula.aerosol import Aerosol
from particula.dynamics import Coagulation, VanDerWaalsCoagulationStrategy
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData


def initial_state(boxes):
    """Construct independent physical boxes with equal positive counts."""
    density = 2160.0
    mass = 4 * np.pi / 3 * (np.array([150.0, 220.0]) * 0.5e-9) ** 3 * density
    particles = ParticleData(
        masses=np.tile(np.diag(mass), (boxes, 1, 1)),
        concentration=np.ones((boxes, 2)),
        charge=np.tile([1.0, -1.0], (boxes, 1)),
        density=np.array([density, density]),
        volume=np.full(boxes, 1e-13),
        distribution_type="particle_resolved",
    )
    gas = GasData(
        ["air"], np.array([0.029]), np.zeros((boxes, 1)), np.array([False])
    )
    environment = EnvironmentData(
        np.full(boxes, 298.0), np.full(boxes, 101325.0), np.zeros((boxes, 1))
    )
    return Aerosol(particles=particles, gas=gas, environment=environment)


def inventory(state):
    """Count mergers and verify both material channels and signed charge."""
    p = state.particles
    mass = (p.masses * p.concentration[:, :, None]).sum(axis=1)
    charge = (p.charge * p.concentration).sum(axis=1)
    return np.count_nonzero(p.concentration, axis=1) == 1, mass, charge


# %% [markdown]
# ## Select an admitted timestep and replay a checkpoint
#
# First evaluate the CPU pair kernel. Its ratio to box volume is the hazard
# in s⁻¹. Check the same state through direct CUDA preflight, then prepare
# the public runnable. Capture the computation once and retain a checkpoint
# halfway through the run. Resume it and compare the individual boxes.


# %%
def simulate():
    """Run two timestep refinements; report distributions and restart evidence."""
    import warp as wp
    from particula.gpu.conversion import to_warp_particle_data
    from particula.gpu.kernels.coagulation import (
        CoagulationMechanismConfig,
        coagulation_preflight_gpu,
    )

    if not wp.is_cuda_available():
        raise RuntimeError("This experiment requires native CUDA")
    boxes = 2048
    initial = initial_state(boxes)
    _, initial_mass, initial_charge = inventory(initial)
    material = dict(hamaker_constant=8.9e-20, relative_permittivity=5.9)
    strategy = VanDerWaalsCoagulationStrategy("particle_resolved", **material)
    rate = strategy.kernel(initial_state(1).particles, 298.0, 101325.0)[0, 1]
    hazard = rate / initial.particles.volume[0]
    process = Coagulation(strategy)
    curves = []
    device = to_warp_particle_data(initial.particles, device="cuda:0")
    config = CoagulationMechanismConfig(
        mechanisms=("van_der_waals",), **material
    )
    for steps in (8, 32):
        dt = 1 / (hazard * steps)
        report = coagulation_preflight_gpu(
            device,
            298.0,
            101325.0,
            dt,
            max_collisions=1,
            mechanism_config=config,
        )
        assert np.all(report["admitted"])
        np.testing.assert_allclose(
            report["expected_trials"], 1 / steps, rtol=1e-9
        )
        started = time.perf_counter()
        with process.prepare(
            initial, device="cuda:0", time_step=float(dt), root_seed=4161
        ) as sim:
            sim.capture()
            wp.synchronize_device("cuda:0")
            setup = time.perf_counter() - started
            fractions = [0.0]
            run_seconds = 0.0
            for i in range(1, steps + 1):
                started = time.perf_counter()
                sim.step()
                sim.synchronize()
                run_seconds += time.perf_counter() - started
                state = sim.snapshot()
                merged, mass, charge = inventory(state)
                np.testing.assert_allclose(
                    mass, initial_mass, rtol=1e-13, atol=0
                )
                np.testing.assert_array_equal(charge, initial_charge)
                fractions.append(float(merged.mean()))
                if i == steps // 2:
                    checkpoint = sim.checkpoint()
            final = state.particles
        with process.resume(checkpoint) as resumed:
            for _ in range(steps // 2):
                resumed.step()
            replay = resumed.snapshot().particles
            for name in ("masses", "concentration", "charge"):
                np.testing.assert_array_equal(
                    getattr(replay, name), getattr(final, name)
                )
        p = 1 - (1 - 1 / steps) ** steps
        assert abs(fractions[-1] - p) < 4 * np.sqrt(p * (1 - p) / boxes)
        curves.append(
            dict(
                steps=steps,
                dt_s=float(dt),
                merged=fractions,
                setup_s=setup,
                resident_s=run_seconds,
            )
        )
    return dict(
        curves=curves,
        boxes=boxes,
        rate_m3_s=float(rate),
        device=wp.get_device("cuda:0").name,
    )


# %% [markdown]
# ## Inspect number loss
#
# The horizontal axis measures elapsed time in units of the pair collision
# time. Dots show observed merger fractions and the solid line is the
# continuous-time Poisson result. The shaded band is the approximate 95%
# binomial sampling interval for the continuous-time result.


# %%
@mplstyle.context("particula.default")
def plot_results(report):
    """Compare merger fractions with the known two-particle survival law."""
    fig, ax = plotting.subplots(size="wide")
    t = np.linspace(0, 1, 160)
    p = -np.expm1(-t)
    band = 1.96 * np.sqrt(p * (1 - p) / report["boxes"])
    ax.fill_between(
        t, p - band, p + band, color=".88", label="95% sampling interval"
    )
    ax.plot(t, p, color=".25", label="Continuous-time merger probability")
    for c in report["curves"]:
        ax.plot(
            np.linspace(0, 1, c["steps"] + 1),
            c["merged"],
            ".-",
            ms=3,
            label=f"{c['steps']} CUDA steps",
        )
    ax.set(
        xlabel="Time × initial pair hazard",
        ylabel="Fraction of boxes with one survivor",
        ylim=(0, 0.75),
    )
    plotting.legend(ax)
    plotting.fit(fig)
    plt.show()


if __name__ == "__main__":
    plot_results(simulate())


# %% [markdown]
# ## Interpretation
#
# A finite-step simulation makes discrete collision decisions. Refining the
# timestep reduces that bias while the finite ensemble retains sampling
# noise. A trajectory-by-trajectory restart comparison tests continuation
# separately from that statistical result. This example holds charge fixed
# between encounters. Adding ion attachment would require its own physical
# model and validation.
