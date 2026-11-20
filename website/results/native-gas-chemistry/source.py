# %% [markdown]
# # Track a gas reaction through a chamber pulse
#
# A falling precursor concentration in a chamber can reflect chemical
# conversion, dilution, or both. Here the gas concentration of product B
# distinguishes those processes. We inject a known mass of precursor A at
# 4 s and predict both gas trajectories independently. The two boxes have
# different volumes and initial concentrations. All inputs are synthetic.
#
# ## Write the mechanism once
#
# The reaction converts one mole of A to one mole of B at 0.3 s⁻¹. A and B
# have equal molar masses in this case. The same native Python mapping can be
# saved to JSON or YAML with `mechanism.save(path)` and read back with
# `ReactionMechanism.load(path)`. The file contains chemistry, while the
# binding below supplies the gas names used in this particular chamber.

# %%
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style as mplstyle
from particula import Aerosol, FieldSpec, ProcessRegistry, plotting
from particula.dynamics import Dilution, DilutionStrategy
from particula.dynamics.chemistry import ReactionMechanism
from particula.dynamics.chemistry.adapter import GasChemistryAdapter
from particula.dynamics.chemistry.process import ChemistrySolver, GasChemistry
from particula.execution.cpu_simulation import CPUSimulation
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData


def chamber():
    """Return two fixed-volume resolved boxes with an inert gas tracer."""
    state = Aerosol(
        particles=ParticleData(
            masses=np.full((2, 1, 1), 1e-18),
            concentration=np.ones((2, 1)),
            charge=np.zeros((2, 1)),
            density=np.array([1000.0]),
            volume=np.array([1.0, 2.0]),
            distribution_type="particle_resolved",
        ),
        gas=GasData(
            ["inert", "precursor", "product"],
            np.array([0.029, 0.1, 0.1]),
            np.array([[1e-6, 2e-9, 0.0], [1e-6, 5e-9, 0.0]]),
            np.zeros(3, dtype=bool),
        ),
        environment=EnvironmentData(
            temperature=np.array([298.15, 288.15]),
            pressure=np.full(2, 101325.0),
            saturation_ratio=np.zeros((2, 3)),
        ),
    )
    state.fields.add(FieldSpec("demo.injected", ("box",), "kg", minimum=0))
    return state


def chemistry(state):
    """Bind A and B to named gas columns and select physical tolerances."""
    mechanism = ReactionMechanism.from_dict(
        {
            "version": 1,
            "basis": "mol/m^3-air",
            "species": {"gas:A": {}, "gas:B": {}},
            "reactions": [
                {
                    "id": "conversion",
                    "reactants": {"gas:A": 1},
                    "products": {"gas:B": 1},
                    "rate": {"type": "constant", "k": 0.3, "units": "1/s"},
                }
            ],
        }
    )
    bound = mechanism.bind(
        state.gas, {"gas:A": "precursor", "gas:B": "product"}
    )
    return GasChemistry(
        bound, ChemistrySolver(method="rk23", rtol=1e-11, atol=1e-22)
    )


# %% [markdown]
# ## Sample before and after the injection
#
# Each chemistry and dilution step lasts 0.25 s. We observe every 0.5 s.
# The observation at 4 s is taken immediately before injection. The chamber
# protocol adds 3 and 1 µg/m³ of A to boxes of 1 and 2 m³, respectively.
# The source ledger therefore records 3 and 2 µg. Dilution acts at 0.05 s⁻¹
# on all gases and the ledger retains the cumulative injected mass.
#
# The injection is an external intervention between two simulation segments.
# We explicitly take a host snapshot at the boundary, update the gas and
# ledger, and prepare the post-pulse segment at the same physical time.
# Chemistry and dilution run on the selected backend throughout each segment.
# The CUDA run downloads observations and checkpoints for the comparisons.


# %%
def record(owner, ticks, device, histories, ledgers):
    """Advance the segment and sample gas concentrations and added mass."""
    for tick in ticks:
        if device == "numpy":
            owner.step(tick * 0.25)
        else:
            owner.step()
        if tick % 2 == 0:
            sample = owner.snapshot()
            histories.append(sample.gas.concentration.copy())
            ledgers.append(sample.fields["demo.injected"].copy())


def trajectory(device, restart):
    """Optionally resume at the pre-pulse and first post-pulse boundaries."""
    state = chamber()
    process = chemistry(state) | Dilution(DilutionStrategy(0.05))
    registry = ProcessRegistry()
    registry.register(GasChemistry, GasChemistryAdapter())
    histories = [state.gas.concentration.copy()]
    ledgers = [state.fields["demo.injected"].copy()]

    def open_owner(state, time):
        if device == "numpy":
            return nullcontext(
                CPUSimulation(state, process, time_step=0.25, time=time)
            )
        return process.prepare(
            state,
            device=device,
            time_step=0.25,
            time=time,
            process_registry=registry,
        )

    def resume(checkpoint):
        if device == "numpy":
            return nullcontext(CPUSimulation.resume(checkpoint, process))
        return process.resume(checkpoint, process_registry=registry)

    with open_owner(state, 0.0) as owner:
        record(owner, range(1, 17), device, histories, ledgers)
        if restart:
            before = owner.checkpoint()
        state = owner.snapshot()
    if restart:
        with resume(before) as restored:
            state = restored.snapshot()
            clock = (
                restored.time if device == "numpy" else restored.current_time
            )
            assert clock == 4.0
    np.testing.assert_array_equal(state.fields["demo.injected"], 0.0)
    added = np.array([3e-9, 1e-9])
    state.gas.concentration[:, 1] += added
    state.fields["demo.injected"][:] += added * state.particles.volume
    with open_owner(state, 4.0) as owner:
        record(owner, range(17, 19), device, histories, ledgers)
        if restart:
            after = owner.checkpoint()
        else:
            record(owner, range(19, 81), device, histories, ledgers)
    if restart:
        with resume(after) as restored:
            record(restored, range(19, 81), device, histories, ledgers)
    return np.asarray(histories), np.asarray(ledgers)


# %% [markdown]
# Between injections, A decays at the sum of the reaction and dilution rates.
# The A+B inventory loses mass only to dilution. Subtracting the analytical
# precursor concentration from the total gives an independent product
# trajectory at all 41 observation times. The inert tracer decays by dilution
# alone. The checks below also compare each backend with its own restarted run.


# %%
def simulate():
    """Compare both integrations with the pulse solution and mass ledger."""
    t = np.arange(41)[:, None] * 0.5
    initial = np.array([2e-9, 5e-9])
    added = np.where(t > 4.0, [3e-9, 1e-9], 0.0)
    volume = np.array([1.0, 2.0])
    total = initial * np.exp(-0.05 * t) + added * np.exp(-0.05 * (t - 4.0))
    a = initial * np.exp(-0.35 * t) + added * np.exp(-0.35 * (t - 4.0))
    exact = np.stack((a, total - a), axis=-1)
    report = {
        "time_s": t[:, 0].tolist(),
        "reference": exact.tolist(),
        "backends": {},
    }
    for device in ("numpy", "cuda:0"):
        history, ledger = trajectory(device, False)
        repeated, repeated_ledger = trajectory(device, True)
        np.testing.assert_array_equal(repeated, history)
        np.testing.assert_array_equal(repeated_ledger, ledger)
        np.testing.assert_allclose(
            history[:, :, 1:], exact, rtol=2e-9, atol=1e-20
        )
        np.testing.assert_allclose(
            history[:, :, 0],
            np.broadcast_to(1e-6 * np.exp(-0.05 * t), total.shape),
            rtol=1e-13,
        )
        np.testing.assert_allclose(
            history[:, :, 1:].sum(axis=2) * volume,
            total * volume,
            rtol=1e-12,
            atol=1e-22,
        )
        np.testing.assert_array_equal(ledger, added * volume)
        report["backends"][device] = {
            "gas": history[:, :, 1:].tolist(),
            "injected_kg": ledger[-1].tolist(),
            "max_error_kg_m3": float(np.max(np.abs(history[:, :, 1:] - exact))),
        }
    return report


# %% [markdown]
# ## Compare conversion with inventory loss
#
# The vertical line marks the pre-pulse observation. The slope of A steepens
# with chemical loss while B accumulates from conversion. A+B tracks the
# dilution-only prediction. The two initial loads and injected masses
# produce different trajectories even though both boxes use the same rate.
# Each analytical line can be compared with independently integrated points.


# %%
@mplstyle.context("particula.default")
def plot_results(report):
    """Show one panel per chamber with analytical and computed concentrations."""
    time = np.asarray(report["time_s"])
    exact = np.asarray(report["reference"]) * 1e9
    edge = int(np.flatnonzero(time == 4.0)[0])
    line_time = np.insert(time, edge + 1, 4.0)
    post = exact[edge].copy()
    post[:, 0] += (
        np.asarray(report["backends"]["numpy"]["injected_kg"])
        * 1e9
        / [1.0, 2.0]
    )
    line = np.insert(exact, edge + 1, post, axis=0)
    fig, axes = plotting.subplots(1, 2)
    for box, ax in enumerate(axes):
        for j, (name, color) in enumerate(
            (("A", "tab:blue"), ("B", "tab:orange"))
        ):
            ax.plot(
                line_time,
                line[:, box, j],
                color=color,
                label=f"{name} analytical",
            )
            for key, marker, offset in (("numpy", "o", 0), ("cuda:0", "+", 1)):
                y = np.asarray(report["backends"][key]["gas"])[:, box, j] * 1e9
                label = (
                    ("CPU" if key == "numpy" else "CUDA") if j == 0 else None
                )
                ax.plot(
                    time[offset::4],
                    y[offset::4],
                    marker,
                    color=color,
                    fillstyle="none",
                    label=label,
                )
        ax.axvline(4, color="0.5", linestyle=":", linewidth=1)
        ax.set(
            xlabel="Time (s)",
            ylabel="Concentration (µg/m³)",
            title=f"Box {box + 1} · {box + 1} m³",
        )
    axes[0].legend(fontsize="small")
    plotting.fit(fig)
    return fig


if __name__ == "__main__":
    plot_results(simulate())
    plt.show()

# %% [markdown]
# ## Interpreting the two boxes
#
# The faster decline of A relative to total A+B isolates chemical conversion
# from the known ventilation loss. A measurement of A alone would confound
# them. The pulse ledger records the mass added at the boundary even after
# dilution removes most of the injected gas. A next experiment could vary
# temperature for a bimolecular Arrhenius reaction and use the product
# trajectory to test the assumed rate law.
