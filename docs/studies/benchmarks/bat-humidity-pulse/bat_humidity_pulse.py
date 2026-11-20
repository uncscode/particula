# %% [markdown]
# # Water uptake and evaporation after a humidity pulse
#
# How quickly does a water–organic droplet reverse direction when humid air
# is replaced by dry air? Follow one homogeneous particle population at 293 K.
# Water exchanges kinetically with a finite gas reservoir. The generic organic
# is involatile here. Its BAT descriptors are those of the miscible O:C 0.5
# case in [Gorkowski et al., Figure 9](https://doi.org/10.5194/acp-19-13383-2019).
# The radius, water inventory and gas replacement below define an illustrative
# experiment. CPU evaluates BAT at each substep. CUDA interpolates a resident
# table at the same particle compositions.

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style as mplstyle
from particula import Aerosol, plotting
from particula.dynamics import MassCondensation
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
)
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.gas import EnvironmentData, GasData
from particula.gas.vapor_pressure_strategies import WaterBuckStrategy
from particula.particles import ActivityBAT, ParticleData
from particula.particles.surface_strategies import SurfaceStrategyVolume
from particula.util.constants import GAS_CONSTANT

TEMPERATURE = 293.0
M_WATER = 0.01801528
DENSITY = np.array([1000.0, 1243.4933333333333])
WATER = WaterBuckStrategy()
SAMPLE_INTERVAL = 0.0025  # seconds
SAMPLES_PER_LEG = 32


def gas_water(relative_humidity):
    """Water vapor mass concentration (kg/m³) at the imposed initial RH."""
    pressure = WATER.pure_vapor_pressure(TEMPERATURE)
    return float(
        relative_humidity * pressure * M_WATER / (GAS_CONSTANT * TEMPERATURE)
    )


def chamber():
    """Represent identical droplets with 1 µm dry organic radius."""
    organic = 4 * np.pi / 3 * (1e-6) ** 3 * DENSITY[1]
    particles = ParticleData(
        masses=np.array([[[0.12 * organic, organic]]]),
        concentration=np.array([[1e5]]),
        volume=np.array([1e-6]),
        charge=np.zeros((1, 1)),
        density=DENSITY.copy(),
        distribution_type="particle_resolved",
    )
    gas = GasData(
        ["water"],
        np.array([M_WATER]),
        np.array([[gas_water(0.90)]]),
        np.array([True]),
    )
    environment = EnvironmentData([TEMPERATURE], [101325.0], [[0.90]])
    process = MassCondensation(
        CondensationIsothermal(
            molar_mass=np.array([M_WATER]),
            diffusion_coefficient=2.4e-5,
            accommodation_coefficient=0.2,
            activity_strategy=ActivityBAT(0.2, 0.5, DENSITY[1]),
            surface_strategy=SurfaceStrategyVolume(
                density=1000.0, surface_tension=0.072
            ),
            vapor_pressure_strategy=WATER,
            species_map=CondensationSpeciesMap(("water",), ((0, 0),)),
        )
    )
    return Aerosol(
        particles=particles, gas=gas, environment=environment
    ), process


# %% [markdown]
# ## Account for the gas replacement
#
# Each leg is closed to water exchange with the surroundings. At 80 ms we
# replace the water vapor, setting the gas to 35% RH while retaining the
# particle masses. Two observations at the switch resolve the jump in the gas
# inventory. Within each leg a change in suspended particle water must be
# balanced by the opposite vapor change. Both backends take four substeps per
# outer step. CUDA replays a captured step between observation boundaries.


# %%
def trajectory(device, steps_per_sample):
    """Return time, wet radius and both water inventories at matched times."""
    state, process = chamber()
    dt = SAMPLE_INTERVAL / steps_per_sample
    initial_organic = state.particles.masses[0, 0, 1]
    rows = []
    time = 0.0

    def record():
        snapshot = owner.snapshot() if owner else state
        np.testing.assert_equal(
            snapshot.particles.masses[0, 0, 1], initial_organic
        )
        rows.append(
            [
                time,
                snapshot.particles.radii[0, 0],
                snapshot.gas.concentration[0, 0],
                snapshot.particles.species_mass_density[0, 0],
            ]
        )

    owner = (
        process.prepare(state, time_step=dt, device=device) if device else None
    )
    try:
        if owner:
            owner.capture()
        for leg in range(2):
            if leg:
                dry = np.array([[gas_water(0.35)]])
                if owner:
                    owner.update_gas(dry)
                else:
                    state.gas.concentration[:] = dry
            record()
            for _ in range(SAMPLES_PER_LEG):
                if owner:
                    owner.run(steps_per_sample)
                else:
                    for _ in range(steps_per_sample):
                        process.execute(state, time_step=dt, sub_steps=4)
                time += SAMPLE_INTERVAL
                record()
    finally:
        if owner:
            owner.close()
    data = np.array(rows)
    check_water_budget(data)
    return data


def check_water_budget(data):
    """Close each leg independently of the imposed vapor replacement."""
    for leg_data in np.split(data, 2):
        vapor, condensed = leg_data[:, 2:].T
        np.testing.assert_allclose(
            vapor - vapor[0], condensed[0] - condensed, rtol=1e-9, atol=1e-15
        )
    assert (
        data[SAMPLES_PER_LEG, 1] > data[0, 1]
        and data[-1, 1] < data[SAMPLES_PER_LEG + 1, 1]
    )


def simulate():
    """Compare the CUDA trajectory with direct BAT and a finer CPU run."""
    direct = trajectory(None, 16)
    finer = trajectory(None, 32)
    cuda = trajectory("cuda:0", 16)
    for other in (direct, cuda):
        np.testing.assert_allclose(other[:, 1], finer[:, 1], rtol=1e-3, atol=0)
    error = np.max(np.abs(cuda[:, 1] / finer[:, 1] - 1))
    print(
        f"Maximum CUDA radius difference from refined CPU: {100 * error:.4f}%"
    )
    print(
        f"CUDA radii before wetting, after wetting, after drying (µm): {1e6 * cuda[[0, SAMPLES_PER_LEG, -1], 1]}"
    )
    return {
        "cpu": direct.tolist(),
        "cuda": cuda.tolist(),
        "refined": finer.tolist(),
    }


# %% [markdown]
# ## Read the radius and water budgets together
#
# The radius responds continuously even at the instant of vapor replacement.
# Plot water in each reservoir as a change from the start of its current leg
# to resolve the much smaller particle transfer against the absolute gas mass.


# %%
@mplstyle.context("particula.default")
def plot_results(result):
    """Show radius and the equal and opposite changes in water storage."""
    fig, axes = plotting.subplots(2, 1, sharex=True, size="wide")
    for key, style, label in (
        ("cpu", "-", "Direct CPU"),
        ("cuda", "o", "Tabulated CUDA"),
    ):
        t, r, vapor, particle = np.array(result[key]).T
        axes[0].plot(
            1000 * t,
            1e6 * r,
            style,
            label=label,
            markersize=4,
            fillstyle="none",
        )
        for i, segment in enumerate(np.array_split(np.arange(len(t)), 2)):
            for amounts, name, color in (
                (vapor, "Vapor", plotting.COLORS["teal"]),
                (particle, "Particle", plotting.COLORS["rust"]),
            ):
                axes[1].plot(
                    1000 * t[segment],
                    1e6 * (amounts[segment] - amounts[segment[0]]),
                    style,
                    color=color,
                    markersize=4,
                    fillstyle="none",
                    label=name if key == "cpu" and i == 0 else None,
                )
    for axis in axes:
        axis.axvline(80, ls=":", color="0.45")
        plotting.legend(axis)
    axes[0].set_ylabel("Wet radius (µm)")
    axes[1].set_ylabel("Water change within leg (mg/m³)")
    axes[1].set_xlabel("Time (ms)")
    plotting.fit(fig)
    return fig


if __name__ == "__main__":
    result = simulate()
    plot_results(result)
    plt.show()

# %% [markdown]
# The wetting and drying legs test the sign and magnitude of the surface
# pressure driving kinetic exchange. The time scale depends on particle size,
# gas transport and the available vapor. To explore finite-reservoir effects,
# vary the represented particle count while holding the initial RH fixed.
