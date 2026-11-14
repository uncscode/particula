# %% [markdown]
# # Repeated source pulses in a cooling and warming aerosol
#
# This longer illustration evolves **196,608 initially active particles** in
# 32 boxes for 600 one-second steps. Each box has 8,192 slots, a polydisperse
# water/dry-material population and three gas species. Nucleation, Brownian
# coagulation and water condensation share one resident state. This coarse
# fixed-step example stresses process composition, repeated injection and
# restart; it is not a timestep-converged chamber prediction. For condensation
# response with explicit refinement checks, use the multibox and forcing studies.
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
# refresh permits up to two hours; download the ZIP for all study helpers.

# %%
"""Longer coupled CUDA aerosol evolution with cyclic forcing and source pulses."""

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
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
        "number": diagnostics.number_concentration.copy(),
        "temperature": diagnostics.temperature.copy(),
        "saturation": diagnostics.saturation_ratio[:, 2].copy(),
        "edges": edges,
        "histogram": histogram,
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
            gas = data.gas_concentration.copy()
            # Activation coefficient 0.25 s^-1 × 512 molecules × 1 s = 128.
            gas[:, 1] = (
                4
                * EVENTS_PER_BOX
                * initial.gas.molar_mass[1]
                / AVOGADRO_NUMBER
                / data.volume
            )
            if step % 200 == 0:
                humidity = (1.4, 0.7, 1.2)[step // 200]
                gas[:, 2] = (
                    humidity
                    * water_pressure(data.temperature)
                    * initial.gas.molar_mass[2]
                    / (GAS_CONSTANT * data.temperature)
                )
            for gas_lane, material in ((2, 0), (1, 1)):
                expected[:, material] += (
                    gas[:, gas_lane] - data.gas_concentration[:, gas_lane]
                ) * data.volume
            owner.update_gas(gas)
        owner.step()
        if pulse:
            # Only nucleation consumes this precursor. Its observed depletion
            # independently verifies the unit-event demand through public data.
            before_precursor = gas[:, 1].copy()
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
            gas = data.gas_concentration.copy()
            expected[:, 1] -= gas[:, 1] * data.volume
            gas[:, 1] = 0
            owner.update_gas(gas)
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
    process = source | coagulation | condensation
    expected = mapped_inventory(initial)
    with process.prepare(
        initial, device="cuda:0", time_step=1.0, root_seed=42
    ) as owner:
        owner.capture()
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
    with process.resume(saved) as resumed:
        resumed.capture()
        advance(resumed, 300, STEPS, initial, saved_expected, observe=False)
        continued = resumed.snapshot()
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
    plt.show()
    return final


# %% [markdown]
# ## Follow cohorts through changing conditions
#
# Source pulses replenish small particles while Brownian merging removes
# number and shifts mass to larger particles. Prescribed vapor and temperature
# changes alter water uptake and evaporation. The radius/time map shows the
# resulting evolving population instead of only a final distribution.
# Invariant checks establish accounting and restart correctness; this
# illustrative fixed-step calculation is not a timestep-converged prediction
# for a physical chamber.


# %%
def plot_history(frames):
    """Plot CPU observations without any further GPU calculation."""
    from matplotlib.colors import LogNorm

    time = np.array([f["time"] for f in frames])
    number = np.array([f["number"] for f in frames])
    temperature = np.array([f["temperature"] for f in frames])
    saturation = np.array([f["saturation"] for f in frames])
    edges = frames[0]["edges"]
    histogram = np.array([f["histogram"].mean(axis=0) for f in frames])
    distribution = histogram / np.diff(np.log10(edges))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for box in (0, 15, 31):
        axes[0, 0].plot(time, temperature[:, box], label=f"Box {box + 1}")
        axes[0, 1].plot(time, number[:, box], label=f"Box {box + 1}")
        axes[1, 0].plot(time, saturation[:, box], label=f"Box {box + 1}")
    axes[0, 0].set(xlabel="Time (s)", ylabel="Temperature (K)")
    axes[0, 1].set(xlabel="Time (s)", ylabel="Number concentration (m$^{-3}$)")
    axes[1, 0].set(xlabel="Time (s)", ylabel="Water saturation ratio")
    axes[1, 0].axhline(1, color="0.5", linestyle="--")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.legend(fontsize=8)
    centres = np.sqrt(edges[:-1] * edges[1:]) * 1e9
    mesh = axes[1, 1].pcolormesh(
        centres,
        time,
        np.ma.masked_less_equal(distribution, 0),
        norm=LogNorm(),
        shading="nearest",
        cmap="viridis",
    )
    axes[1, 1].set(xscale="log", xlabel="Radius (nm)", ylabel="Time (s)")
    fig.colorbar(mesh, ax=axes[1, 1], label="Mean dN / dlog₁₀ r (m$^{-3}$)")
    fig.suptitle(
        "Cyclic CUDA aerosol · source pulses, Brownian collisions and condensation"
    )


if __name__ == "__main__":
    run_example()
