# %% [markdown]
# # How two particle sizes share a reactive gas
#
# Two populations in a closed 1.5 m³ chamber draw from the same gas. A gas
# molecule that sticks becomes fresh material. Fresh material then ages inside
# its particle. Which population gains more mass, and how much is still fresh
# after eight seconds? The inputs are synthetic. We compare the simulation
# with a separate calculation based on particle radius.
#
# ## Specify materials and reactions
#
# The initial radii are 35 and 80 nm. The chamber contains three times as many
# small particles as large ones. At 298 K and 1000 Pa we prescribe free
# molecular uptake with probability 0.04. Gas diffusion resistance,
# evaporation and heat feedback are outside this example. Both products have
# molar mass 0.04 kg/mol and density 1400 kg/m³. Aging therefore conserves
# particle volume. The core has its own density and contributes to surface area.

# %%
from contextlib import nullcontext
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style as mplstyle
from particula import Aerosol, ProcessRegistry, plotting
from particula.dynamics.chemistry import ReactionMechanism
from particula.dynamics.chemistry.process import ChemistrySolver
from particula.dynamics.chemistry.resolved import ResolvedChemistry
from particula.dynamics.chemistry.resolved_adapter import (
    ResolvedChemistryAdapter,
)
from particula.execution.cpu_simulation import CPUSimulation
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData
from scipy.integrate import solve_ivp

RADII = np.array([35e-9, 80e-9])
COUNTS = np.array([4.5e12, 1.5e12])
VOLUME = 1.5
GAS_INITIAL = 3e-6
TEMPERATURE = 298.0
TIMES = np.linspace(0, 8, 17)


def setup():
    """Build one gas and three material columns plus one unoccupied slot."""
    mass = np.zeros((1, 3, 3))
    mass[0, :2, 0] = (4 * np.pi / 3) * RADII**3 * 1000
    particles = ParticleData(
        masses=mass,
        density=np.array([1000.0, 1400.0, 1400.0]),
        concentration=np.array([[*COUNTS, 0.0]]),
        charge=np.zeros((1, 3)),
        volume=np.array([VOLUME]),
        distribution_type="particle_resolved",
    )
    gas = GasData(
        ["X"], np.array([0.04]), np.array([[GAS_INITIAL]]), np.array([False])
    )
    state = Aerosol(
        particles=particles,
        gas=gas,
        environment=EnvironmentData(
            temperature=np.array([TEMPERATURE]),
            pressure=np.array([1000.0]),
            saturation_ratio=np.zeros((1, 1)),
        ),
    )
    material = {"molar_mass_kg_per_mol": 0.04, "elements": {"X": 1}}
    mechanism = ReactionMechanism.from_dict(
        {
            "version": 2,
            "basis": "mol/m^3-phase",
            "species": {
                "gas:X": {"elements": {"X": 1}},
                "particle:fresh": material,
                "particle:aged": material,
            },
            "reactions": [
                {
                    "id": "stick",
                    "reactants": {"gas:X": 1},
                    "products": {"particle:fresh": 1},
                    "rate": {"type": "reactive_uptake", "gamma": 0.04},
                },
                {
                    "id": "age",
                    "reactants": {"particle:fresh": 1},
                    "products": {"particle:aged": 1},
                    "rate": {"type": "constant", "k": 0.35, "units": "1/s"},
                },
            ],
        }
    )
    bound = mechanism.bind_resolved(
        state, {"particle:fresh": 1, "particle:aged": 2}
    )
    return state, ResolvedChemistry(
        bound, ChemistrySolver(rtol=1e-9, atol=1e-17)
    )


# %% [markdown]
# ## An independent radius calculation
#
# The same gas concentration surrounds each sphere. Its mass flux per unit
# area is $J=0.04\,\bar c\,C_g/4$, where $C_g$ is gas mass per volume of
# air. A sphere grows at $dr/dt=J/1400$, so both radii gain the same increment.
# The shell volume fixes the mass removed from the gas at every instant.
# Integrating the fresh material separately at 0.35 s⁻¹ then gives the aged
# fraction by subtraction. This reference uses a high-order SciPy solver.


# %%
def radial_reference():
    """Integrate radius growth and fresh fractions using the shell inventory."""
    speed = np.sqrt(8 * 8.31446261815324 * TEMPERATURE / (np.pi * 0.04))
    initial_kg = GAS_INITIAL * VOLUME

    def shell(dr):
        return (
            4 * np.pi * dr * (RADII**2 + RADII * dr + dr**2 / 3) * 1400 * COUNTS
        )

    def derivative(time, y):
        dr = y[0] * 1e-9
        gas = (initial_kg - shell(dr).sum()) / VOLUME
        flux = 0.04 * speed * gas / 4
        fresh = (
            flux * 4 * np.pi * (RADII + dr) ** 2 * COUNTS / initial_kg
            - 0.35 * y[1:]
        )
        return np.r_[flux / 1400 * 1e9, fresh]

    solution = solve_ivp(
        derivative,
        (0, 8),
        np.zeros(3),
        t_eval=TIMES,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    total = np.array([shell(dr * 1e-9) / initial_kg for dr in solution.y[0]])
    fresh = solution.y[1:].T
    return np.column_stack((1 - total.sum(axis=1), fresh, total - fresh))


# %% [markdown]
# ## Follow the shared inventory
#
# We record gas and both products every 0.5 s as fractions of the initial
# gas mass. Each particle mass in kg is multiplied by its represented count.
# The measured wall time includes setup, preparation, all observations and
# teardown. The CUDA run also includes recording its fixed launch sequence.


# %%
def run_backend(device):
    """Return five extensive mass fractions at each observation."""
    start = perf_counter()
    state, process = setup()
    dt = float(TIMES[1] - TIMES[0])
    if device == "numpy":
        owner = nullcontext(CPUSimulation(state, process, time_step=dt))
    else:
        registry = ProcessRegistry()
        registry.register(ResolvedChemistry, ResolvedChemistryAdapter())
        owner = process.prepare(
            state, device=device, time_step=dt, process_registry=registry
        )
    history = []
    with owner as simulation:
        if device != "numpy":
            simulation.capture()
        for time in TIMES:
            if time > 0:
                if device == "numpy":
                    simulation.step(float(time))
                else:
                    simulation.step()
            snapshot = simulation.snapshot()
            product = (
                snapshot.particles.masses[0, :2, 1:]
                * COUNTS[:, None]
                / (GAS_INITIAL * VOLUME)
            )
            history.append(
                [
                    snapshot.gas.concentration[0, 0] / GAS_INITIAL,
                    *product[:, 0],
                    *product[:, 1],
                ]
            )
    return np.asarray(history), perf_counter() - start


def simulate():
    """Compare recorded inventories on NumPy and CUDA with the radius model."""
    reference = radial_reference()
    report = {
        "seconds": TIMES.tolist(),
        "reference": reference.tolist(),
        "backends": {},
    }
    for device in ("numpy", "cuda:0"):
        history, elapsed = run_backend(device)
        np.testing.assert_allclose(history, reference, rtol=2e-7, atol=2e-10)
        np.testing.assert_allclose(history.sum(axis=1), 1.0, rtol=1e-12, atol=0)
        report["backends"][device] = {
            "history": history.tolist(),
            "elapsed_s": elapsed,
            "max_fraction_error": float(np.abs(history - reference).max()),
        }
    return report


# %% [markdown]
# ## Compare gas loss and product allocation
#
# The lines in the first panel are the independent reference. Symbols show
# the captured CUDA trajectory. The right panel splits the final particle
# inventory by initial size and chemical state. All fractions use the same
# initial gas inventory, so the two bars and the remaining gas add to one.


# %%
@mplstyle.context("particula.default")
def plot_results(report):
    """Plot the analytical inventory comparison and final material fractions."""
    time = np.asarray(report["seconds"])
    history = np.asarray(report["backends"]["cuda:0"]["history"])
    reference = np.asarray(report["reference"])
    fig, axes = plotting.subplots(1, 2)
    colors = ["#374c67", "#00766b", "#b65b20"]
    for i, (name, columns) in enumerate(
        [("Gas", [0]), ("Fresh", [1, 2]), ("Aged", [3, 4])]
    ):
        axes[0].plot(
            time, reference[:, columns].sum(axis=1), color=colors[i], label=name
        )
        axes[0].plot(
            time[::2],
            history[::2, columns].sum(axis=1),
            "o",
            mfc="white",
            mec=colors[i],
            ms=4,
        )
    axes[0].set(
        xlabel="Time (s)", ylabel="Fraction of initial gas mass", ylim=(0, 1.04)
    )
    axes[0].legend(frameon=False)
    axes[1].bar(
        ["35 nm", "80 nm"], history[-1, 1:3], color=colors[1], label="Fresh"
    )
    axes[1].bar(
        ["35 nm", "80 nm"],
        history[-1, 3:],
        bottom=history[-1, 1:3],
        color=colors[2],
        label="Aged",
    )
    axes[1].set(
        xlabel="Initial particle radius",
        ylabel="Fraction of initial gas mass",
        ylim=(0, 1.04),
    )
    axes[1].legend(frameon=False)
    plotting.fit(fig)
    return fig


if __name__ == "__main__":
    plot_results(simulate())
    plt.show()

# %% [markdown]
# The larger particles collect more mass per particle because each has more
# surface area. Represented counts determine how that advantage translates
# to the population bars. The fresh pool first rises and then declines as the
# shared gas is depleted and aging continues. This synthetic experiment tests
# collision-limited uptake. An application to a measured aerosol would need
# its gas transport regime and a measured reaction probability.
