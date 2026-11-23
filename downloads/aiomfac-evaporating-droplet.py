# %% [markdown]
# # How long does an evaporating droplet take to reach two-liquid equilibrium?
#
# A wet 2-propanol–ammonium sulfate droplet loses water to a finite volume
# of air. Follow the changing phase amounts and compare the terminal water
# mass with a standalone equilibrium solve at the same total water inventory.
# The initial dry mass ratio and temperature match the
# [humidity sweep](aiomfac_humidity_sweep.py). Here the gas humidity evolves
# as water evaporates. A liquid split can occur during a continuous change in
# particle mass. A second closed-box experiment lets 2-propanol exchange
# with the gas as well and tracks both vapor inventories.
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


def prepare_box(*, exchange_alcohol=False):
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
    names = ("water", "2-propanol") if exchange_alcohol else ("water",)
    count = len(names)
    molar_mass = np.array([WATER_MW, 0.060095])[:count]
    pressures = np.array([VAPOR_PRESSURE, 7859.0])[:count]
    if exchange_alcohol:
        start[1] *= 0.8
    activities = activity.activities_at_temperature(target, TEMPERATURE, names)
    saturation = pressures * molar_mass / (GAS_CONSTANT * TEMPERATURE)
    initial_gas = activities * saturation + NUMBER * (
        target[:count] - start[:count]
    )
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
            list(names),
            molar_mass,
            initial_gas[None, :],
            np.ones(count, dtype=bool),
        ),
        environment=EnvironmentData(
            [TEMPERATURE], [101325.0], (initial_gas / saturation)[None, :]
        ),
    )
    process = CondensationIsothermal(
        molar_mass=molar_mass,
        diffusion_coefficient=np.array([2.5e-5, 1.0e-5])[:count],
        accommodation_coefficient=1.0,
        vapor_pressure_strategy=[
            ConstantVaporPressureStrategy(p) for p in pressures
        ],
        activity_strategy=activity,
        surface_strategy=SurfaceStrategyVolume(
            density=DENSITY, surface_tension=SIGMA
        ),
        species_map=CondensationSpeciesMap(
            names, tuple((i, i) for i in range(count))
        ),
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
# ## Let the alcohol exchange with the gas
#
# Start a second box with 20% less particle alcohol and the same water and
# salt masses. Supply enough vapor to target the same two-liquid endpoint.
# At 303.15 K the measured pure 2-propanol vapor pressure is 7.859 kPa.
# The prescribed gas diffusivities are 2.5 × 10⁻⁵ m²/s for water and
# 1.0 × 10⁻⁵ m²/s for alcohol. Water and alcohol have independently closed
# inventories. Their fluxes are coupled through the evolving liquid
# activities. Ammonium sulfate stays in the droplet.
# [Vapor pressure measurements, Table 4](https://doi.org/10.1021/je034197x)
# %%
def simulate_two_vapors():
    """Track three particle materials and both gas inventories in a closed box."""
    box, process = prepare_box(exchange_alcohol=True)
    activity = process.activity_strategy
    inventory = box.particles.species_mass_density[0].copy()
    inventory[:2] += box.gas.concentration[0]
    names = ("water", "2-propanol")
    saturation = (
        np.array([VAPOR_PRESSURE, 7859.0])
        * box.gas.molar_mass
        / (GAS_CONSTANT * TEMPERATURE)
    )
    bulk = (
        EquilibriaFactory()
        .get_strategy("liquid_vapor", {"activity_model": activity})
        .solve(
            inventories=inventory * 1e9,
            temperature=TEMPERATURE,
            saturation_concentrations=dict(
                zip(names, saturation * 1e9, strict=True)
            ),
        )
    )
    owner = CondensationODE(
        box,
        process,
        method="rk23",
        rtol=5e-5,
        particle_atol=1e-21,
        gas_atol=1e-10,
        first_step=0.0005,
    )
    times = [0, 0.005, 0.015, 0.04, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0]
    particle, vapor, phase_share = [], [], []
    for time in times:
        if time:
            run_until(owner, time)
        state = owner.snapshot()
        mass = state.particles.masses[0, 0]
        split = activity.phase_equilibrium(mass, TEMPERATURE)
        particle.append(mass.copy())
        vapor.append(state.gas.concentration[0].copy())
        phase_share.append(
            float(split.beta_masses.sum() / mass.sum())
            if split.beta is not None
            else 0.0
        )
    particle = np.asarray(particle)
    vapor = np.asarray(vapor)
    np.testing.assert_allclose(
        particle[:, 2], particle[0, 2], rtol=1e-12, atol=0.0
    )
    delta_particle = (particle[:, :2] - particle[0, :2]) * NUMBER
    delta_vapor = vapor - vapor[0]
    closure = np.max(np.abs(delta_particle + delta_vapor), axis=0) / np.max(
        np.abs(delta_particle), axis=0
    )
    np.testing.assert_allclose(closure, 0.0, atol=1e-8)
    assert np.all(np.max(np.abs(delta_particle), axis=0) > 0)
    assert bulk.liquids.beta is not None
    eq_pg = (
        (bulk.liquids.alpha_masses + bulk.liquids.beta_masses) * 1e6 / NUMBER
    )
    np.testing.assert_allclose(particle[-1] * 1e15, eq_pg, rtol=2e-5, atol=0.0)
    if phase_share[0] != 0.0 or not any(phase_share):
        raise ArithmeticError("Expected to cross into the two-liquid region")
    return {
        "time_s": times,
        "particle_pg": (particle * 1e15).tolist(),
        "gas_change_mg_m3": (delta_vapor * 1e6).tolist(),
        "beta_share": phase_share,
        "equilibrium_pg": eq_pg.tolist(),
        "relative_transfer_residual": closure.tolist(),
    }


# %% [markdown]
# ## Identify when the second liquid appears
#
# The falling water fraction and rising vapor RH follow the same transfer of
# water. As the droplet approaches its terminal composition, the organic-rich
# share grows from zero. Dashed horizontal references come from the closed
# equilibrium solve, which takes no time steps.
# %%
@plt.style.context("particula.default")
def plot_results(data):
    """Show vapor uptake of evaporated water and liquid phase appearance."""
    from particula import plotting

    fig, (left, right) = plotting.subplots(1, 2, size="wide")
    t = data["time_s"]
    left.plot(
        t,
        data["water"],
        "o-",
        color=plotting.COLORS["teal"],
        label="Particle water",
    )
    left.plot(
        t,
        data["beta_share"],
        "s-",
        color=plotting.COLORS["rust"],
        label="Organic-rich phase",
    )
    left.axhline(
        data["equilibrium_water"],
        ls="--",
        color=plotting.COLORS["teal"],
        label="Bulk water endpoint",
    )
    left.set_ylabel("Fraction of total particle mass")
    plotting.legend(left, placement="inside")
    right.plot(
        t,
        100 * np.array(data["humidity"]),
        "o-",
        color=plotting.COLORS["indigo"],
    )
    right.axhline(100 * data["equilibrium_rh"], color="0.35", ls="--")
    right.set_ylabel("Vapor relative humidity (%)")
    for ax in (left, right):
        ax.set_xscale("symlog", linthresh=0.02)
        ax.set_xlim(0, 3.2)
        ax.set_xlabel("Time (s)")
    multi = data["two_vapors"]
    transfer, (part, air) = plotting.subplots(1, 2, size="wide")
    colors = (
        plotting.COLORS["teal"],
        plotting.COLORS["rust"],
        plotting.COLORS["indigo"],
    )
    labels = ("Water", "2-propanol", "Ammonium sulfate")
    for i, (label, color) in enumerate(zip(labels, colors, strict=True)):
        part.plot(
            multi["time_s"],
            np.asarray(multi["particle_pg"])[:, i],
            marker=("o", "s", "^")[i],
            color=color,
            label=label,
        )
        part.axhline(multi["equilibrium_pg"][i], ls=":", color=color, lw=1.0)
        if i < 2:
            air.plot(
                multi["time_s"],
                np.asarray(multi["gas_change_mg_m3"])[:, i],
                marker=("o", "s")[i],
                color=color,
                label=label,
            )
    first_split = next(
        i for i, share in enumerate(multi["beta_share"]) if share > 0
    )
    for axis in (part, air):
        axis.axvspan(
            multi["time_s"][first_split - 1],
            multi["time_s"][first_split],
            color="0.35",
            alpha=0.12,
            lw=0,
        )
    part.set_ylabel("Mass per droplet (pg)")
    plotting.legend(part, placement="inside")
    air.axhline(0, color="0.55", lw=0.7)
    air.set_ylabel("Change in vapor mass (mg/m³)")
    for axis in (part, air):
        axis.set_xscale("symlog", linthresh=0.02)
        axis.set_xlim(0, 3.2)
        axis.set_xlabel("Time (s)")
    plotting.fit(transfer)
    plotting.fit(fig)
    return fig, transfer


def run_example():
    """Report the independent endpoint and normalized water-budget residual."""
    data = simulate()
    data["two_vapors"] = simulate_two_vapors()
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
# time in the first figure belongs to gas–particle water transport. A finite
# interfacial barrier or diffusion through a viscous organic shell could
# delay the transition in a real particle. The predicted drying time also
# depends on the specified particle size, abundance and transport constants.

#
# In the two-vapor figure, water falls from 156.4 to 75.4 pg per droplet.
# Alcohol first rises from 56 to 137.5 pg at the 40 ms sample, then falls
# toward 70 pg as the mixture dries. The shaded interval from 5 to 15 ms
# brackets the first detected second liquid. Dotted lines give the particle
# mass endpoints from the bulk equilibrium calculation. The vapor panel measures each gas relative
# to its initial inventory. Opposite signs indicate that one vapor enters the
# droplets while the other leaves. Comparing both changes with their
# respective particle transfers tests conservation even when the background
# vapor mass is large.
# To follow the onset across different dry compositions, see the
# [1,000-droplet humidity ramp](aiomfac_mixing_state_ramp.py).
