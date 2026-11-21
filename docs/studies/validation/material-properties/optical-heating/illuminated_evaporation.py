# %% [markdown]
# # Illumination heats wet aerosol and transfers water to the gas
#
# How does light change droplet temperature and water content in a small
# optically thin chamber? Start two wet droplet sizes at equilibrium with
# humid air. Apply a prescribed visible spectrum, then compare the result
# with a dark control. Follow both water reservoirs and check the energy
# required to explain the temperature and mass changes.
#
# Use a fixed-temperature gas and radiation bath at 298.15 K. The droplets
# start at 4 and 6 µm wet radius with the same nonvolatile absorbing solute
# and hygroscopicity κ = 0.30. A flat solution surface with water activity
# equal to the initial gas RH (88%) sets the dark equilibrium. Water
# transferred to the gas enters the next cycle's vapor balance. The bath
# temperatures stay fixed. These micron-scale particles motivate the
# continuum transport approximation at 101325 Pa.
# The calculation assumes liquid, homogeneous spheres, a one-way
# evaporation law and grey thermal emission.
#
# This illustrative exposure totals 110 kW/m². Each listed irradiance
# is integrated over its own spectral band and applied at that band's
# representative wavelength. The chamber path for the optical depth
# diagnostic is 0.20 m. NumPy and native CUDA each run the light and dark
# cases. See the [optics guide](../../../../guide/optics/index.md) for the
# cross-section, wavelength and refractive-index conventions.

# %%
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from matplotlib import style as mplstyle
from particula import Aerosol, ProcessRegistry, plotting
from particula.dynamics.optical_heating import HeatingPhysics, OpticalHeating
from particula.dynamics.optical_heating_gpu import OpticalHeatingAdapter
from particula.execution.cpu_simulation import CPUSimulation
from particula.execution.optical_recording import (
    OpticalHistory,
    OpticalObserver,
    read_optical_history,
)
from particula.gas import EnvironmentData, GasData
from particula.optics import OpticalConfig
from particula.particles import create_resolved_particle_data
from particula.util.constants import GAS_CONSTANT
from scipy.constants import Stefan_Boltzmann

TEMPERATURE = 298.15
PRESSURE = 101325.0
RH = 0.88
KAPPA = 0.30
WATER_MOLAR_MASS = 0.01801528
PATH_M = 0.20
RADIUS_M = np.array([4e-6, 6e-6])
NUMBER_M3 = np.array([8e6, 12e6])
DENSITY = np.array([997.0, 1600.0])
WAVELENGTH_M = np.array([450, 550, 700]) * 1e-9
IRRADIANCE = np.array([35e3, 50e3, 25e3])
OPTICS = OpticalConfig(
    WAVELENGTH_M,
    species_index=[
        [1.34, 1.73 + 0.085j],
        [1.333, 1.71 + 0.076j],
        [1.33, 1.69 + 0.060j],
    ],
)
PHYSICS = HeatingPhysics(kappa=KAPPA, radiation_temperature=TEMPERATURE)
STEP_S = 2.5e-4
DURATION_S = 0.024
CAPACITY = 192


def dark_equilibrium():
    """Set particle activity and gas RH independently from the heating step."""
    # Activity of a flat solution is V_water/(V_water + kappa*V_solute).
    water_to_solute = KAPPA * RH / (1 - RH)
    wet_volume = 4 * np.pi * RADIUS_M**3 / 3
    solute_volume = wet_volume / (1 + water_to_solute)
    volume = np.column_stack((water_to_solute * solute_volume, solute_volume))
    # Initial gas water density from Buck (Pa), the ideal gas law and RH.
    tc = TEMPERATURE - 273.15
    p_sat = 611.21 * np.exp((18.678 - tc / 234.5) * tc / (257.14 + tc))
    water_vapor = RH * p_sat * WATER_MOLAR_MASS / (GAS_CONSTANT * TEMPERATURE)
    return volume * DENSITY, water_vapor


def chamber(irradiance):
    """Create the same closed water inventory for each light and dark run."""
    masses, water_vapor = dark_equilibrium()
    state = Aerosol(
        particles=create_resolved_particle_data(
            masses=masses,
            density=DENSITY,
            concentration=NUMBER_M3,
            concentration_units="1/m^3",
            volume=0.04,
        ),
        gas=GasData(
            name=["water"],
            molar_mass=[WATER_MOLAR_MASS],
            concentration=[[water_vapor]],
            partitioning=[True],
        ),
        environment=EnvironmentData(
            temperature=[TEMPERATURE],
            pressure=[PRESSURE],
            saturation_ratio=[[RH]],
        ),
    )
    heater = OpticalHeating(
        OPTICS, irradiance, gas_water="water", physics=PHYSICS
    )
    heater.attach_fields(state)
    state.fields[heater.temperature_field][:] = TEMPERATURE
    return state, heater


# %% [markdown]
# ## Sample the process input and its accepted output
#
# Each heating cycle integrates the band irradiances against the particle
# absorption cross sections at the start of that cycle. The process stores
# this incident power on the resulting particle state. Label that power
# with the start time when comparing to the bulk spectrum.
#
# A separate optical observer calculates bulk absorption and extinction
# from the accepted states. These spectra include the water lost during
# the preceding cycle. The two optical calculations answer different
# questions. The first powers evaporation over an interval. The second
# describes the particles present at its end. Resolved particle counts
# are divided by the box volume for comparisons on a per-volume basis.
# The coarse runs retain the two droplets' state at each cycle for the
# balance checks. Refinement runs retain only their endpoints.


# %%
def bind_owner(state, heater, device, interval):
    """Set an explicit backend and register the resident heating adapter."""
    if device == "numpy":
        # Every requested bound sets one CPU cycle of approximately interval.
        return nullcontext(CPUSimulation(state, heater, time_step=2 * interval))
    registry = ProcessRegistry()
    registry.register(
        OpticalHeating, OpticalHeatingAdapter(coefficient_capacity=CAPACITY)
    )
    return heater.prepare(
        state, time_step=interval, device=device, process_registry=registry
    )


def sample(state, heater):
    """Read native state and accumulated energy in the declared ledger order."""
    return {
        "radius": state.particles.radii[0].copy(),
        "mass": state.particles.masses[0].copy(),
        "temperature": state.fields[heater.temperature_field][0].copy(),
        "power": state.fields[heater.power_field][0].copy(),
        "vapor": float(state.gas.concentration[0, 0]),
        "energy": np.array(
            [state.fields[key][0] for key in heater.energy_fields]
        ),
    }


def run_chamber(device, irradiance, interval, path=None):
    """Advance the fixed grid and optionally record the accepted optical trace."""
    state, heater = chamber(irradiance)
    times = interval * np.arange(round(DURATION_S / interval) + 1)
    states = [sample(state, heater)]
    with bind_owner(state, heater, device, interval) as owner:
        observer = None
        if path is not None:
            recorder = OpticalHistory(
                path,
                OPTICS,
                variables=("extinction", "absorption"),
                provenance={
                    "initial_rh_fraction": RH,
                    "optical_path_m": PATH_M,
                    "band_irradiance_w_m2": list(map(float, irradiance)),
                },
            )
            observer = OpticalObserver(
                owner, recorder, batch_size=16, coefficient_capacity=CAPACITY
            )
            observer.record()

        for time in times[1:]:
            if device == "numpy":
                owner.step(float(time))
            else:
                owner.step()
            if observer is not None:
                observer.record()
                states.append(sample(owner.snapshot(), heater))
        if observer is None:
            states.append(sample(owner.snapshot(), heater))
        else:
            observer.flush()

    result = {
        key: np.asarray([row[key] for row in states]) for key in states[0]
    }
    if path is not None:
        chunks = [row for _, row in read_optical_history(path)]
        for key in ("time", "extinction", "absorption"):
            values = np.concatenate([row[key] for row in chunks])
            result[key] = values if key == "time" else values[:, 0]
        np.testing.assert_allclose(result["time"], times, rtol=0, atol=1e-15)
    return result


# %% [markdown]
# ## Check water and energy independently of the ledgers
#
# Gas-water gain and liquid-water loss must agree per cubic metre. For every
# step, recover the heat conducted into the gas, net grey radiation, latent
# energy and change in sensible energy from the saved states. The independent
# calculation uses the cycle-start radius and heat capacity, and it evaluates
# conduction and radiation at the accepted particle temperature. The transfer
# ledger includes energy exchanged with the prescribed baths.
#
# Use optical depth at all three wavelengths to test the thin-path
# assumption. Then halve the timestep twice while holding the same
# exposure duration. Compare successive changes in final wet radius and
# temperature of the initially 6 µm mode, and in the box-wide vapor gain.


# %%
def water_transfer(trace):
    """Return the gas gain and particle loss per volume (kg/m³)."""
    return (
        trace["vapor"] - trace["vapor"][0],
        (trace["mass"][0, :, 0] - trace["mass"][..., 0]) @ NUMBER_M3,
    )


def audit(trace, irradiance):
    """Reconstruct per-volume energy and verify the clock alignment."""
    gas_gain, water_loss = water_transfer(trace)
    np.testing.assert_allclose(gas_gain, water_loss, rtol=2e-8, atol=2e-16)
    np.testing.assert_array_equal(
        trace["mass"][..., 1],
        np.broadcast_to(trace["mass"][0, :, 1], trace["mass"].shape[:2]),
    )
    feedback = trace["power"][1:] @ NUMBER_M3
    accepted_power = trace["absorption"] @ irradiance
    np.testing.assert_allclose(
        feedback, accepted_power[:-1], rtol=5e-10, atol=2e-14
    )

    t = trace["temperature"][1:]
    r = trace["radius"][:-1]
    m = trace["mass"][:-1]
    duration = np.diff(trace["time"])
    area = 4 * np.pi * r**2
    conducting = duration * (
        (4 * np.pi * r * PHYSICS.conductivity * (t - TEMPERATURE)) @ NUMBER_M3
    )
    radiating = duration * (
        (area * PHYSICS.emissivity * Stefan_Boltzmann * (t**4 - TEMPERATURE**4))
        @ NUMBER_M3
    )
    heat_capacity = m @ np.array(
        [PHYSICS.water_heat_capacity, PHYSICS.solute_heat_capacity]
    )
    sensible = (heat_capacity * (t - trace["temperature"][:-1])) @ NUMBER_M3
    budget = np.column_stack(
        (
            duration * feedback,
            conducting,
            radiating,
            PHYSICS.latent_heat * np.diff(water_loss),
            sensible,
        )
    ).cumsum(axis=0)
    np.testing.assert_allclose(
        trace["energy"][1:], budget, rtol=3e-9, atol=2e-14
    )
    energy_error = np.abs(budget[:, 0] - budget[:, 1:].sum(axis=1))
    np.testing.assert_allclose(energy_error, 0.0, rtol=0.0, atol=2e-11)
    depth = float(np.max(PATH_M * trace["extinction"]))
    if depth >= 0.01:
        raise ArithmeticError("Optical depth exceeds the thin-path check")
    return {
        "max_depth": depth,
        "max_water_error_kg_m3": float(np.max(np.abs(gas_gain - water_loss))),
        "max_energy_error_j_m3": float(np.max(energy_error)),
        "max_timing_difference_w_m3": float(
            np.max(np.abs(accepted_power[1:] - feedback))
        ),
    }


def endpoint(trace):
    """Return endpoint changes with explicit units for convergence checks."""
    return np.array(
        [
            1e6 * (trace["radius"][-1, 1] - trace["radius"][0, 1]),
            trace["temperature"][-1, 1] - TEMPERATURE,
            1e6 * (trace["vapor"][-1] - trace["vapor"][0]),
        ]
    )


def simulate():
    """Compare native NumPy/CUDA, controls, budgets and timestep refinement."""
    if not wp.is_cuda_available():
        raise RuntimeError("This example requires a native CUDA device")
    report = {}
    with TemporaryDirectory() as directory:
        for backend, device in (("NumPy", "numpy"), ("CUDA", "cuda:0")):
            light = run_chamber(
                device,
                IRRADIANCE,
                STEP_S,
                Path(directory) / f"{backend}-light.db",
            )
            dark = run_chamber(
                device,
                np.zeros_like(IRRADIANCE),
                STEP_S,
                Path(directory) / f"{backend}-dark.db",
            )
            evidence = audit(light, IRRADIANCE)
            audit(dark, np.zeros_like(IRRADIANCE))
            np.testing.assert_allclose(
                dark["temperature"], TEMPERATURE, atol=1e-12, rtol=0
            )
            np.testing.assert_allclose(water_transfer(dark), 0.0, atol=2e-16)
            np.testing.assert_array_equal(dark["power"], 0.0)
            np.testing.assert_allclose(
                dark["radius"],
                np.broadcast_to(RADIUS_M, dark["radius"].shape),
                atol=1e-20,
                rtol=0,
            )
            if evidence["max_timing_difference_w_m3"] == 0.0:
                raise ArithmeticError(
                    "Cycle-start and accepted optics should differ"
                )
            if not (
                light["vapor"][-1] > light["vapor"][0]
                and np.all(light["radius"][-1] < RADIUS_M)
            ):
                raise ArithmeticError(
                    "Expected illumination to drive evaporation"
                )

            refined = np.asarray(
                [
                    endpoint(light),
                    *(
                        endpoint(
                            run_chamber(device, IRRADIANCE, STEP_S / scale)
                        )
                        for scale in (2, 4)
                    ),
                ]
            )
            difference = np.abs(np.diff(refined, axis=0))
            if not np.all(difference[1] < difference[0]):
                raise ArithmeticError(
                    "Endpoint changes did not decrease under refinement"
                )
            evidence["refinement_delta_6um_k_mg"] = difference.tolist()
            evidence["endpoint_6um_k_mg"] = refined.tolist()
            report[backend] = {
                "illuminated": light,
                "dark": dark,
                "validation": evidence,
            }
    for field in (
        "vapor",
        "mass",
        "temperature",
        "power",
        "absorption",
        "energy",
    ):
        np.testing.assert_allclose(
            report["CUDA"]["illuminated"][field],
            report["NumPy"]["illuminated"][field],
            rtol=6e-9,
            atol=1e-16,
        )
    for case in report.values():
        light = case["illuminated"]
        dark = case["dark"]
        gas_gain, water_loss = water_transfer(light)
        case["illuminated"] = {
            key: light[key].tolist()
            for key in (
                "time",
                "radius",
                "temperature",
                "power",
                "absorption",
                "energy",
            )
        }
        case["illuminated"]["gas_gain"] = gas_gain.tolist()
        case["illuminated"]["water_loss"] = water_loss.tolist()
        case["dark"] = {
            key: dark[key].tolist() for key in ("radius", "temperature")
        }
    return report


# %% [markdown]
# The first panel uses the cycle-start clock for particle power. The next
# panel shows bulk absorption against the accepted-state clock. The inferred
# power from accepted spectra differs from cycle-start feedback by up to
# 0.124 W/m³ in this run. Mode colors carry over to the temperature and
# radius panels. The plots follow the CUDA light and dark cases. Dotted radius
# curves show the dark result. In light, particles warm and shrink
# as water enters the gas. The two curves in the water panel test transfer
# conservation. The energy panel compares the absorbed energy with the
# sum of the four energy terms checked against the snapshots.


# %%
@mplstyle.context("particula.default")
def plot_results(report):
    """Plot optical feedback and show conservation over the exposure."""
    figure, axes = plotting.subplots(2, 3, size="wide")
    light = {k: np.asarray(v) for k, v in report["CUDA"]["illuminated"].items()}
    dark = {k: np.asarray(v) for k, v in report["CUDA"]["dark"].items()}
    time = 1e3 * light["time"]
    colors = ("#00669f", "#b53d29", "#247657")

    for j, label in enumerate(("4 µm initially", "6 µm initially")):
        axes[0, 0].plot(
            time[:-1], 1e6 * light["power"][1:, j], color=colors[j], label=label
        )
        axes[0, 2].plot(
            time, light["temperature"][:, j] - TEMPERATURE, color=colors[j]
        )
        axes[1, 0].plot(time, 1e6 * light["radius"][:, j], color=colors[j])
        axes[1, 0].plot(
            time, 1e6 * dark["radius"][:, j], color=colors[j], linestyle=":"
        )
    axes[0, 2].plot(
        time,
        dark["temperature"][:, 1] - TEMPERATURE,
        color="0.45",
        linestyle=":",
        label="Dark",
    )
    axes[0, 0].legend()
    axes[0, 2].legend()
    for j, wavelength in enumerate(WAVELENGTH_M):
        axes[0, 1].plot(
            time,
            1e6 * light["absorption"][:, j],
            color=colors[j],
            label=f"{1e9 * wavelength:.0f} nm",
        )
    axes[0, 1].legend()
    gas_gain, water_loss = light["gas_gain"], light["water_loss"]
    axes[1, 1].plot(time, 1e6 * gas_gain, label="Vapor gain", color=colors[0])
    axes[1, 1].plot(
        time,
        1e6 * water_loss,
        linestyle="--",
        label="Particle loss",
        color=colors[1],
    )
    axes[1, 1].legend()

    terms = light["energy"]
    axes[1, 2].plot(time, terms[:, 0], label="Absorbed", color=colors[0])
    axes[1, 2].plot(
        time,
        terms[:, 1:].sum(axis=1),
        linestyle="--",
        label="Sum of heat terms",
        color=colors[1],
    )
    axes[1, 2].plot(
        time, terms[:, 3], linestyle=":", label="Latent term", color=colors[2]
    )
    axes[1, 2].legend()

    for axis, title, unit in zip(
        axes.flat,
        (
            "Cycle-start power",
            "Accepted-state absorption",
            "Particle warming",
            "Wet radius",
            "Water transfer",
            "Cumulative energy balance",
        ),
        (
            "Power (µW per particle)",
            "Absorption (Mm⁻¹)",
            "Temperature rise (K)",
            "Radius (µm)",
            "Water transferred (mg m⁻³)",
            "Energy (J m⁻³)",
        ),
        strict=True,
    ):
        axis.set(title=title, xlabel="Time (ms)", ylabel=unit)
        axis.grid(alpha=0.24)

    for backend, case in report.items():
        evidence = case["validation"]
        dw = evidence["max_water_error_kg_m3"]
        de = evidence["max_energy_error_j_m3"]
        timing = evidence["max_timing_difference_w_m3"]
        refinements = np.asarray(evidence["refinement_delta_6um_k_mg"])
        ratios = refinements[1] / refinements[0]
        print(f"{backend} maximum optical depth {evidence['max_depth']:.3e}")
        print(
            f"  Closure residuals {dw:.2e} kg/m³ water and {de:.2e} J/m³ energy"
        )
        print(
            f"  Maximum change in absorbed power within a cycle {timing:.3f} W/m³"
        )
        print(
            "  Fine-to-coarse endpoint changes for radius, temperature and vapor"
        )
        print(f"  {ratios[0]:.3f}, {ratios[1]:.3f}, {ratios[2]:.3f}")
    plotting.fit(figure)
    return figure


if __name__ == "__main__":
    result = simulate()
    figure = plot_results(result)
    plt.show()

# %% [markdown]
# At 24 ms, the initially 6 µm droplet has shrunk to about 5.81 µm
# and warmed by 1.82 K. The illuminated populations put 1.25 mg m⁻³
# of water into the gas while the dark control stays at equilibrium.
# Evaporation takes about 70% of the absorbed energy. Differences
# in the final radius, temperature and vapor gain fall by roughly
# half on the second timestep refinement, consistent with first-order
# convergence over this range. The maximum optical depth is about
# 0.00075 at the prescribed 0.20 m path.
#
# To explore a longer path, recalculate the maximum optical depth and
# compare with a transport model that updates the local spectrum when
# attenuation changes the heating appreciably.
