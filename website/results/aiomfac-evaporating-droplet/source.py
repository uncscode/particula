# %% [markdown]
# # How long does an evaporating droplet take to reach two-liquid equilibrium?
#
# A wet 2-propanol–ammonium sulfate droplet loses water to a finite volume
# of air. Follow the changing phase amounts and compare the terminal water
# mass with a standalone equilibrium solve at the same total water inventory.
# The initial dry mass ratio and temperature match the
# [humidity sweep](aiomfac_humidity_sweep.py). Here the gas humidity evolves
# as water evaporates. A liquid split can occur during a continuous change in
# particle mass.
#
# ## Set up a finite reservoir
#
# One cubic metre contains 10 billion identical droplets at 303.15 K.
# In each droplet 0.1 ng of dry material comprises 70% alcohol and 30% salt
# by mass. Only water exchanges with the gas. The initial water mass fraction
# is 0.61. We choose the total water to place equilibrium near a condensed
# water mass fraction of 0.43, inside the two-liquid region. A near-zero
# prescribed surface tension approximates the flat surface of the standalone
# solver. Check the resulting Kelvin correction before interpreting the
# kinetic–equilibrium comparison.
# %%
import matplotlib.pyplot as plt
import numpy as np
from particula.aerosol import Aerosol
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
)
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.equilibria import EquilibriaFactory
from particula.execution.condensation_ode import CondensationODE
from particula.execution.recording import run_until
from particula.gas import GasData
from particula.gas.environment_data import EnvironmentData
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)
from particula.particles import ParticleData
from particula.particles.activity_factories import ActivityFactory
from particula.particles.surface_strategies import SurfaceStrategyVolume
from particula.util.constants import GAS_CONSTANT
from particula_aiomfac import MixtureSpec

TEMPERATURE = 303.15  # K
WATER_MW = 0.01801528  # kg/mol
VAPOR_PRESSURE = 4246.0  # Pa
WATER_CSTAR = VAPOR_PRESSURE * WATER_MW / (GAS_CONSTANT * TEMPERATURE)
NUMBER = 1e10  # represented particles in 1 m³
DRY_MASS = 1e-13  # kg per particle
DENSITY = 1000.0  # prescribed common material density, kg/m³
SIGMA = 1e-9  # N/m; approach a bulk liquid surface
GROUPS = {
    "water": {16: 1},
    "2-propanol": {141: 2, 151: 1, 153: 1},
    "ammonium sulfate": {204: 2, 261: 1},
}


def prepare_box():
    """Make the initial state, stable activity provider and native process."""
    activity = ActivityFactory().get_strategy(
        "aiomfac",
        {
            "mixture": MixtureSpec(GROUPS),
            "material_mapping": dict(enumerate(GROUPS)),
            "phase_mode": "stable",
        },
    )
    start = DRY_MASS * np.array([0.61 / 0.39, 0.70, 0.30])
    target = DRY_MASS * np.array([0.43 / 0.57, 0.70, 0.30])
    phase = activity.phase_equilibrium(target, TEMPERATURE)
    water_index = phase.alpha.neutral_ids.index("water")
    equilibrium_gas = phase.alpha.neutral_activities[water_index] * WATER_CSTAR
    initial_gas = equilibrium_gas - NUMBER * (start[0] - target[0])
    particles = ParticleData(
        masses=start[None, None, :],
        concentration=np.array([[NUMBER]]),
        charge=np.zeros((1, 1)),
        density=np.full(3, DENSITY),
        volume=np.ones(1),
        distribution_type="particle_resolved",
    )
    box = Aerosol(
        particles=particles,
        gas=GasData(
            ["water"],
            np.array([WATER_MW]),
            np.array([[initial_gas]]),
            np.array([True]),
        ),
        environment=EnvironmentData(
            [TEMPERATURE], [101325.0], [[initial_gas / WATER_CSTAR]]
        ),
    )
    process = CondensationIsothermal(
        molar_mass=np.array([WATER_MW]),
        diffusion_coefficient=2.5e-5,
        accommodation_coefficient=1.0,
        vapor_pressure_strategy=ConstantVaporPressureStrategy(VAPOR_PRESSURE),
        activity_strategy=activity,
        surface_strategy=SurfaceStrategyVolume(
            density=DENSITY, surface_tension=SIGMA
        ),
        species_map=CondensationSpeciesMap(("water",), ((0, 0),)),
        update_gases=True,
    )
    return box, process


# %% [markdown]
# ## Compare an equilibrium inventory with a transport trajectory
#
# Add the gas and particle concentrations before the first step. Passing
# those totals to the equilibrium solver without an RH closes the reservoir.
# Native condensation integrates the flux from each trial droplet composition.
# At each output time, recompute the internal liquid split, reconstruct the
# combined water budget, and record the gas RH. The diagnostic phase masses
# partition the same droplet inventory that drives transport.
# %%
def simulate():
    """Dry a closed box and compare its terminal state with bulk equilibrium."""
    box, process = prepare_box()
    activity = process.activity_strategy
    total = box.particles.species_mass_density[0].copy()
    total[0] += box.gas.concentration[0, 0]
    solver = EquilibriaFactory().get_strategy(
        "liquid_vapor", {"activity_model": activity}
    )
    equilibrium = solver.solve(
        inventories=total * 1e9,
        temperature=TEMPERATURE,
        saturation_concentrations={"water": WATER_CSTAR * 1e9},
    )
    phases = equilibrium.liquids
    target_particle = (phases.alpha_masses + phases.beta_masses) / (
        1e9 * NUMBER
    )
    owner = CondensationODE(
        box,
        process,
        method="rk23",
        rtol=3e-5,
        particle_atol=1e-21,
        gas_atol=1e-10,
        first_step=0.001,
    )
    times = [0, 0.01, 0.03, 0.06, 0.10, 0.17, 0.30, 0.55, 1.0, 1.7, 3.0]
    water, humidity, share, budget = [], [], [], []
    for time in times:
        if time:
            run_until(owner, time)
        state = owner.snapshot()
        masses = state.particles.masses[0, 0]
        gas = state.gas.concentration[0, 0]
        liquids = activity.phase_equilibrium(masses, TEMPERATURE)
        beta = liquids.beta_masses
        share.append(0.0 if beta is None else float(beta.sum() / masses.sum()))
        water.append(float(masses[0] / masses.sum()))
        humidity.append(float(gas / WATER_CSTAR))
        budget.append(float((NUMBER * masses[0] + gas - total[0]) / total[0]))
        np.testing.assert_allclose(masses[1:] * NUMBER, total[1:], rtol=1e-12)
        radius = state.particles.radii[0, 0]
        log_kelvin = (
            2
            * SIGMA
            * WATER_MW
            / (DENSITY * GAS_CONSTANT * TEMPERATURE * radius)
        )
        if log_kelvin > 1e-9:
            raise ArithmeticError(
                "Surface curvature exceeds the bulk approximation"
            )
    if share[0] or share[-1] <= 0:
        raise ArithmeticError(
            "The simulated drying path did not cross the phase boundary"
        )
    target_fraction = float(target_particle[0] / target_particle.sum())
    np.testing.assert_allclose(water[-1], target_fraction, atol=4e-4, rtol=0)
    np.testing.assert_allclose(budget, 0.0, atol=1e-10)
    return {
        "time_s": times,
        "water": water,
        "humidity": humidity,
        "beta_share": share,
        "equilibrium_water": target_fraction,
        "equilibrium_rh": equilibrium.relative_humidity,
        "max_water_budget_error": max(abs(b) for b in budget),
    }


# %% [markdown]
# ## Identify when the second liquid appears
#
# The falling water fraction and rising vapor RH follow the same transfer of
# water. As the droplet approaches its terminal composition, the organic-rich
# share grows from zero. Dashed horizontal references come from the closed
# equilibrium solve, which takes no time steps.
# %%
def plot_results(data):
    """Show vapor uptake of evaporated water and liquid phase appearance."""
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(8.8, 3.7), constrained_layout=True
    )
    t = data["time_s"]
    left.plot(t, data["water"], "o-", color="#17678a", label="Particle water")
    left.plot(
        t, data["beta_share"], "s-", color="#a24426", label="Organic-rich phase"
    )
    left.axhline(
        data["equilibrium_water"],
        ls="--",
        color="#17678a",
        label="Bulk water endpoint",
    )
    left.set_ylabel("Fraction of total particle mass")
    left.legend(frameon=False)
    right.plot(t, 100 * np.array(data["humidity"]), "o-", color="#487b43")
    right.axhline(100 * data["equilibrium_rh"], color="0.35", ls="--")
    right.set_ylabel("Vapor relative humidity (%)")
    for ax in (left, right):
        ax.set_xscale("symlog", linthresh=0.02)
        ax.set_xlim(0, 3.2)
        ax.set_xlabel("Time (s)")
    return fig


def run_example():
    """Report the independent endpoint and normalized water-budget residual."""
    data = simulate()
    print(
        f"Closed equilibrium water mass fraction: {data['equilibrium_water']:.5f}"
    )
    print(f"Final kinetic water mass fraction: {data['water'][-1]:.5f}")
    print(
        f"Maximum relative water-budget error: {data['max_water_budget_error']:.2e}"
    )
    plot_results(data)
    return data


if __name__ == "__main__":
    run_example()
    plt.show()

# %% [markdown]
# Droplet water falls from 61% to 43% of particle mass over the 3 s
# integration. Matching the terminal water mass checks that the standalone equilibrium
# and the finite-reservoir transport calculation describe the same endpoint.
# The model assigns no time to forming an internal interface. The elapsed
# time in the figure belongs to gas–particle water transport. A finite
# interfacial barrier or diffusion through a viscous organic shell could
# delay the transition in a real particle. The predicted drying time also
# depends on the specified particle size, abundance and transport constants.
