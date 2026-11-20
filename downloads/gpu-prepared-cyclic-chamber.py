# %% [markdown]
# # Repeated source pulses in a cooling and warming aerosol
#
# The figures use the archived November 16 CUDA run. At its one-second
# coagulation interval, current Particula requests 275,911 collision trials
# in the first box and has capacity for 3,136. The current simulation stops
# at that check. The archived arrays retain the original interval and model.
#
# What must be saved to reproduce a simulation driven by repeated external
# pulses? This example separates the evolving aerosol from the forcing schedule
# and its source ledger. Start with the [prepared quick start](../../../tutorials/working-with-gpu/gpu_prepared_quick_start.py)
# and [refined forcing example](../forcing/gpu_prepared_forcing.py) before using this larger
# composition and restart demonstration.
#
# This longer illustration evolves **196,608 initially active particles** in
# 32 boxes for 600 one-second steps. Each box has 8,192 slots, a polydisperse
# water/dry-material population and three gas species. Nucleation, Brownian
# coagulation retain their fixed one-second event schedule. Water condensation
# uses adaptive ROS3 between those boundaries. Event buffers and random streams
# persist; condensation handoff stays in device memory. Adaptive acceptance,
# rejection and step-size decisions run in a native CUDA conditional graph.
# The host schedules fixed events and observations. Condensation error control does
# not bound splitting or stochastic error, so this remains an illustrative
# chamber calculation rather than a timestep-converged prediction.
#
# Every 50 seconds, a finite precursor pulse creates exactly 128 unit-count,
# neutral particles per box. Remaining precursor is explicitly removed after
# that step. Temperature cycles over a 200-second period, with different box
# offsets; water-vapor replacements at the start of each cycle add or remove
# mass explicitly. The source composition is the existing illustrative
# one-precursor-molecule nucleus, not a calibrated nucleation mechanism.
#
# This schedule intentionally omits dilution: all active multiplicities remain
# one, including later injected cohorts, satisfying native Brownian admission.
# Every observed box must conserve each mapped species after accounting for
# the actual prescribed gas exchanges. Charge must remain unchanged.
# A checkpoint halfway through must reproduce the full final state when the
# same external forcing schedule is continued from its saved step index.
#
# This is a deliberate manual CUDA study, excluded from CI execution. Its
# refresh permits up to four hours; download the ZIP for all study helpers.

# %%
"""Longer coupled CUDA aerosol evolution with cyclic forcing and source pulses."""

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from matplotlib import style as mplstyle
from particula import plotting
from particula.execution.gpu_coupled import GPUCoupledSimulation
from particula.execution.gpu_coupled_restart import resume_coupled
from particula.util.constants import AVOGADRO_NUMBER, GAS_CONSTANT
from scripts.gpu_studies.cases import (
    StudyCase,
    build_case,
    radius_histogram,
    water_pressure,
)
from scripts.gpu_studies.validation import (
    mapped_inventory,
    state_hash,
    valid_state,
)

STEPS = 600
PULSE_EVERY = 50
EVENTS_PER_BOX = 128


def frame(owner, step):
    """Observe a detached distribution and reduced box diagnostics."""
    snapshot = owner.snapshot()
    diagnostics = owner.diagnostics()
    edges, histogram = radius_histogram(snapshot)
    return {
        "time": step,
        "number": diagnostics.number_concentration.tolist(),
        "temperature": diagnostics.temperature.tolist(),
        "saturation": diagnostics.saturation_ratio[:, 2].tolist(),
        "edges": edges.tolist(),
        "histogram": histogram.tolist(),
    }, snapshot


def advance(owner, start, stop, initial, expected, *, observe):
    """Apply the same explicit external schedule on original and resumed owners."""
    boxes = initial.particles.n_boxes
    frames = []
    max_error = 0.0
    initial_charge = np.sum(
        initial.particles.charge * initial.particles.concentration, axis=1
    )
    for step in range(start, stop):
        if step % 10 == 0:
            phase = (step % 200) / 200
            temperature = np.interp(
                phase, [0, 0.3, 0.7, 1], [293, 278, 305, 293]
            )
            owner.update_environment(
                temperature=temperature + np.linspace(-5, 5, boxes)
            )
        pulse = step % PULSE_EVERY == 0
        if pulse:
            data = owner.diagnostics()
            # Prescribed source demand, not a download/re-upload of evolving gas.
            precursor = (
                4
                * EVENTS_PER_BOX
                * initial.gas.molar_mass[1]
                / AVOGADRO_NUMBER
                / initial.particles.volume
            )
            ratios = {}
            if step % 200 == 0:
                ratios[initial.gas.name[2]] = (1.4, 0.7, 1.2)[step // 200]
            owner.update_gas_species(
                {initial.gas.name[1]: precursor}, saturation_ratios=ratios
            )
            forced = owner.diagnostics()
            if ratios:
                # Independent observation audit; this CPU result is never uploaded.
                target = (
                    ratios[initial.gas.name[2]]
                    * water_pressure(data.temperature)
                    * initial.gas.molar_mass[2]
                    / (GAS_CONSTANT * data.temperature)
                )
                np.testing.assert_allclose(
                    forced.gas_concentration[:, 2], target, rtol=2e-14
                )
            for gas_lane, material in ((2, 0), (1, 1)):
                expected[:, material] += (
                    forced.gas_concentration[:, gas_lane]
                    - data.gas_concentration[:, gas_lane]
                ) * data.volume
        owner.step()
        if pulse:
            # Only nucleation consumes this precursor. Its observed depletion
            # independently verifies the unit-event demand through public data.
            before_precursor = precursor.copy()
            data = owner.diagnostics()
            consumed_molecules = (
                (before_precursor - data.gas_concentration[:, 1])
                * data.volume
                * AVOGADRO_NUMBER
                / initial.gas.molar_mass[1]
            )
            np.testing.assert_allclose(
                consumed_molecules, EVENTS_PER_BOX, rtol=2e-12, atol=0
            )
            expected[:, 1] -= data.gas_concentration[:, 1] * data.volume
            owner.update_gas_species({initial.gas.name[1]: 0.0})
        if observe and ((step + 1) % 20 == 0 or step + 1 == stop):
            observation, snapshot = frame(owner, step + 1)
            actual = mapped_inventory(snapshot)
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-30)
            np.testing.assert_allclose(
                np.sum(
                    snapshot.particles.charge
                    * snapshot.particles.concentration,
                    axis=1,
                ),
                initial_charge,
                rtol=0,
                atol=1e-12,
            )
            active = snapshot.particles.concentration
            np.testing.assert_array_equal(active[active > 0], 1.0)
            valid_state(snapshot)
            max_error = max(
                max_error, float(np.max(np.abs(actual / expected - 1)))
            )
            frames.append(observation)
            print(f"Verified CUDA chamber t={step + 1} s", flush=True)
    return frames, max_error


# %% [markdown]
# ## Keep restart state and external control state distinct
#
# The checkpoint retains scientific arrays and random continuation. The
# application also retains its forcing index (`300` here) and external mass
# ledger. Replaying the second half with a different forcing schedule would
# be a different experiment, even with an identical checkpoint.


# %%
def run_example():
    """Run all cycles, accounting gates and the exact second-half restart."""
    if not wp.is_cuda_available():
        raise RuntimeError("This study requires CUDA; no CPU fallback")
    case = StudyCase(
        "cyclic-chamber",
        "full",
        32,
        8192,
        occupancy=0.75,
        time_step=1.0,
        steps=STEPS,
        heterogeneous=True,
    )
    initial, _, processes = build_case(case)
    initial.gas.concentration[:, 1] = 0.0
    source, coagulation, condensation = processes[:3]
    events = (source, coagulation)
    strategy = condensation.condensation_strategy
    expected = mapped_inventory(initial)
    with GPUCoupledSimulation(
        initial,
        events,
        strategy,
        device="cuda:0",
        time_step=1.0,
        root_seed=42,
        resident=True,
    ) as owner:
        owner.capture_events()
        first, _ = frame(owner, 0)
        frames, error_a = advance(
            owner, 0, 300, initial, expected, observe=True
        )
        saved = owner.checkpoint()
        saved_expected = expected.copy()
        later, error_b = advance(
            owner, 300, STEPS, initial, expected, observe=True
        )
        final = owner.snapshot()
        work = {
            **vars(owner.statistics),
            "control_readbacks": owner.control_readbacks,
            "coupling_downloads": owner.coupling_downloads,
            "coupling_uploads": owner.coupling_uploads,
        }
        final_rng = owner.checkpoint().events.resident.rng_continuation
    with resume_coupled(saved, events, strategy, device="cuda:0") as resumed:
        resumed.capture_events()
        advance(resumed, 300, STEPS, initial, saved_expected, observe=False)
        continued = resumed.snapshot()
        if resumed.checkpoint().events.resident.rng_continuation != final_rng:
            raise AssertionError("Cyclic restart changed random continuation")
    if state_hash(final) != state_hash(continued):
        raise AssertionError("Cyclic restart changed the final state")
    np.testing.assert_allclose(
        mapped_inventory(continued), expected, rtol=2e-12, atol=1e-30
    )
    plot_history([first, *frames, *later])
    print("CUDA:", wp.get_device("cuda:0").name)
    print(
        "Initial active particles:", int(initial.particles.concentration.sum())
    )
    print(
        "Physical duration:", STEPS, "s; source pulses:", STEPS // PULSE_EVERY
    )
    print(
        "Final active particles:",
        int(np.count_nonzero(final.particles.concentration)),
    )
    print(
        "Maximum box/species inventory relative error:", max(error_a, error_b)
    )
    print("Second-half checkpoint continuation is byte-exact.")
    print("Adaptive condensation and explicit coupling work:", work)
    plt.show()
    return {
        "frames": [first, *frames, *later],
        "work": work,
        "max_inventory_relative_error": max(error_a, error_b),
        "restart_exact": True,
        "initial_active_particles": int(initial.particles.concentration.sum()),
        "final_active_particles": int(
            np.count_nonzero(final.particles.concentration)
        ),
        "duration_s": STEPS,
    }


# %% [markdown]
# ## Follow cohorts through changing conditions
#
# Source pulses replenish small particles while Brownian merging removes
# number and shifts mass to larger particles. Prescribed vapor and temperature
# changes alter water uptake and evaporation. The radius/time map shows the
# resulting evolving population instead of only a final distribution.
# The point at zero is the supplied initial state before the first forcing
# update. Later observations are 20 seconds apart; they do not resolve every
# forcing jump or transient peak.
# Invariant checks establish accounting and restart correctness; this
# fixed event schedule with adaptive condensation is not a converged prediction
# for a physical chamber.


# %%
@mplstyle.context("particula.default")
def plot_history(frames):
    """Plot CPU observations without any further GPU calculation."""
    from matplotlib.colors import LogNorm

    time = np.array([f["time"] for f in frames])
    number = np.array([f["number"] for f in frames])
    temperature = np.array([f["temperature"] for f in frames])
    saturation = np.array([f["saturation"] for f in frames])
    edges = np.asarray(frames[0]["edges"])
    histogram = np.array(
        [np.asarray(f["histogram"]).mean(axis=0) for f in frames]
    )
    distribution = histogram / np.diff(np.log10(edges))
    fig, axes = plotting.subplots(2, 2, size="wide")
    for box in (0, 15, 31):
        axes[0, 0].plot(time, temperature[:, box], label=f"Box {box + 1}")
        axes[0, 1].plot(time, number[:, box], label=f"Box {box + 1}")
        axes[1, 0].plot(time, saturation[:, box], label=f"Box {box + 1}")
    axes[0, 0].set(xlabel="Time (s)", ylabel="Temperature (K)")
    axes[0, 1].set(xlabel="Time (s)", ylabel="Number concentration (m$^{-3}$)")
    axes[1, 0].set(xlabel="Time (s)", ylabel="Water saturation ratio")
    axes[1, 0].axhline(1, color="0.5", linestyle="--")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        plotting.legend(ax)
    centres = np.sqrt(edges[:-1] * edges[1:]) * 1e9
    axes[1, 1].grid(False)
    mesh = axes[1, 1].pcolormesh(
        centres,
        time,
        np.ma.masked_less_equal(distribution, 0),
        norm=LogNorm(),
        shading="nearest",
        cmap=plotting.colormap("magnitude"),
    )
    axes[1, 1].set(xscale="log", xlabel="Radius (nm)", ylabel="Time (s)")
    fig.colorbar(mesh, ax=axes[1, 1], label="Mean dN / dlog₁₀ r (m$^{-3}$)")
    fig.suptitle(
        "Cyclic CUDA aerosol · source pulses, Brownian collisions and condensation"
    )
    plotting.fit(fig)


if __name__ == "__main__":
    run_example()

# %% [markdown]
# ## Reproduce the experiment, then qualify its response
#
# The exact second-half continuation shows that the checkpoint and restored
# application schedule reproduce this trajectory. The species ledger checks each
# observed external exchange. Neither check establishes that one-second steps
# resolve fast condensation following a source or humidity pulse.
#
# The radius-time panel averages box histograms; it can reveal new small-particle
# cohorts but can hide differences between box temperatures. To investigate pulse
# spacing, retain per-box distributions, hold total supplied material controlled,
# and compare size and number after each cycle. Refine both timestep and forcing
# resolution at identical physical event times before interpreting hysteresis or
# cohort survival as a physical result.
