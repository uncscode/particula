# %% [markdown]
# # Compare drizzle fall speeds and model evaporating drop sizes
#
# Gunn and Kinzer measured the terminal speed of distilled-water drops in
# still air. Two Particula CPU calculations predict speed directly from
# diameter. The finite-Reynolds-number sphere calculation agrees with the
# table to within 4.2% at all nine tested sizes from 0.20 to 1.00 mm.
# Then we ask how a drop's reference speed changes if evaporation separates
# a diameter measurement from a speed measurement by up to one second.
# Particula calculates liquid mass on CUDA. We convert it to wet diameter,
# adjust for ventilation, and assign speed from the published table.
# A two-station experiment could measure both local diameter and actual
# speed. This calculation predicts the terminal speed associated with each
# diameter. The [study report](index.md) discusses the distinction.
#
# ## Start with ten measured speeds
#
# The complete example ZIP includes the mass-equivalent diameters and
# terminal speeds transcribed from [Smithsonian Table 114, p. 396](https://unidata.github.io/MetPy/latest/_static/Smithsonian1951.pdf).
# Its conditions were 20 °C, 760 mmHg, and 50% relative humidity (RH).
# The [original experiment](https://journals.ametsoc.org/view/journals/atsc/6/4/1520-0469_1949_006_0243_ttvoff_2_0_co_2.xml)
# reported better than 0.7% overall measurement accuracy. Each tabulated
# speed is rounded to 0.01 m/s. Use both scales when interpreting a residual.
# The CUDA run requires Particula with Warp, SciPy, NumPy and Matplotlib.
#
# Evaluate Stokes drag and finite-Reynolds-number sphere drag at exactly
# the tabulated diameters. Both use the same water density, moist-air density,
# viscosity and continuum slip factor of one. Tightening the drag solver's
# tolerance checks how much numerical precision contributes to the comparison.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import particula as par
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

REFERENCE_FILE = Path("docs/data/drizzle-settling/reference.csv")
TEMPERATURE_K = 293.15
PRESSURE_PA = 101325.0
WATER_DENSITY = 998.2  # kg/m3
MOLAR_MASS = 0.01801528  # kg/mol
DIFFUSIVITY = 2.5e-5  # m2/s
LATENT_HEAT = 2.45e6  # J/kg
SATURATION_PA = 2338.3
WATER_GAS_CONSTANT = 8.314462618 / MOLAR_MASS  # J/(kg K)
AIR_CONDUCTIVITY = 0.0257  # W/(m K)
AIR_HEAT_CAPACITY = 1006.0  # J/(kg K)
AIR_VISCOSITY = par.gas.get_dynamic_viscosity(TEMPERATURE_K)
EXPOSURES_S = np.array([0.25, 0.5, 1.0])


def read_reference():
    """Load diameter (mm) and speed (m/s) from the checkout or example ZIP."""
    roots = [Path.cwd()]
    if "__file__" in globals():
        roots.insert(0, Path(__file__).resolve().parent)
    for root in roots:
        for parent in (root, *root.parents):
            path = parent / REFERENCE_FILE
            if path.is_file():
                return np.loadtxt(path, delimiter=",", skiprows=1).T
    raise FileNotFoundError(
        f"Extract the example ZIP containing {REFERENCE_FILE}"
    )


def air_density(humidity):
    """Return moist-air density (kg/m3) at the specified RH fraction."""
    vapor_pa = humidity * SATURATION_PA
    return (PRESSURE_PA - vapor_pa) / (287.05 * TEMPERATURE_K) + vapor_pa / (
        461.52 * TEMPERATURE_K
    )


def fixed_size_benchmark(reference):
    diameter, measured = reference
    common = dict(
        particle_radius=diameter * 0.0005,
        particle_density=WATER_DENSITY,
        fluid_density=air_density(0.5),
        dynamic_viscosity=AIR_VISCOSITY,
        slip_correction_factor=1.0,
    )
    stokes = par.particles.get_particle_settling_velocity(**common)
    sphere = par.particles.get_particle_settling_velocity_with_drag(**common)
    refined = par.particles.get_particle_settling_velocity_with_drag(
        **common, tol=1e-10
    )
    solver_change = float(np.max(np.abs(refined - sphere)))
    assert solver_change < 1e-5
    return {
        "diameter_mm": diameter.tolist(),
        "measured_m_s": measured.tolist(),
        "stokes_m_s": stokes.tolist(),
        "sphere_m_s": sphere.tolist(),
        "max_solver_change_m_s": solver_change,
    }


# %% [markdown]
# ## Follow the water in each independent box
#
# Place one resolved, uncharged water drop in each cubic metre of air.
# Its initial diameter fixes its liquid mass, and the prescribed initial
# RH fixes the vapor mass. Particula transfers water between these two
# reservoirs while the air temperature stays at 20 °C. The latent heat
# treatment accounts for cooling during evaporation. Check conservation of
# liquid plus vapor, persistence of exactly one drop, and monotonic shrinkage.
#
# The pilot uses one 0.20 mm drop at 50% RH. The two full runs use identical
# starting boxes and halve the time step from 0.025 to 0.0125 s. Four steps
# separate each pair of saved wet diameters. Six seconds of unventilated
# CUDA history cover the faster ventilated scenarios below.


# %%
def evaporate_on_cuda(diameter_mm, humidity, step_s):
    from particula.dynamics.condensation.species_mapping import (
        CondensationSpeciesMap,
    )

    boxes = diameter_mm.size
    liquid = np.pi / 6 * (diameter_mm * 0.001) ** 3 * WATER_DENSITY
    vapor = humidity * SATURATION_PA / (WATER_GAS_CONSTANT * TEMPERATURE_K)
    inventory = liquid + vapor  # one drop per cubic metre
    state = par.Aerosol(
        particles=par.particles.ParticleData(
            masses=liquid[:, None, None],
            density=np.array([WATER_DENSITY]),
            population=np.ones((boxes, 1)),
            charge=np.zeros((boxes, 1)),
            volume=np.ones(boxes),
            distribution_type="particle_resolved",
        ),
        gas=par.gas.GasData(
            name=["water"],
            molar_mass=np.array([MOLAR_MASS]),
            concentration=vapor[:, None],
            partitioning=np.array([True]),
        ),
        environment=par.gas.EnvironmentData(
            temperature=np.full(boxes, TEMPERATURE_K),
            pressure=np.full(boxes, PRESSURE_PA),
            saturation_ratio=humidity[:, None],
        ),
    )
    strategy = par.dynamics.CondensationLatentHeat(
        molar_mass=np.array([MOLAR_MASS]),
        diffusion_coefficient=DIFFUSIVITY,
        accommodation_coefficient=1.0,
        latent_heat=LATENT_HEAT,
        activity_strategy=par.particles.ActivityIdealMass(),
        surface_strategy=par.particles.SurfaceStrategyVolume(
            surface_tension=0.0728, density=WATER_DENSITY
        ),
        vapor_pressure_strategy=par.gas.ConstantVaporPressureStrategy(
            SATURATION_PA
        ),
        species_map=CondensationSpeciesMap(("water",), ((0, 0),)),
    )
    process = par.dynamics.MassCondensation(strategy)
    interval_s = 4 * step_s
    times_s = np.arange(round(6 / interval_s) + 1) * interval_s
    diameters = [diameter_mm]
    max_inventory_change = 0.0
    max_vapor_change = 0.0
    with process.prepare(state, device="cuda:0", time_step=step_s) as session:
        session.capture()
        for _ in times_s[1:]:
            session.run(steps=4)
            snapshot = session.snapshot()
            wet_mass = snapshot.particles.masses[:, 0, 0]
            gas_mass = snapshot.gas.concentration[:, 0]
            assert np.all(np.isfinite(wet_mass)) and np.all(wet_mass > 0)
            assert np.all(np.isfinite(gas_mass)) and np.all(gas_mass >= 0)
            np.testing.assert_array_equal(
                snapshot.particles.population[:, 0], np.ones(boxes)
            )
            inventory_change = np.max(
                np.abs((wet_mass + gas_mass) / inventory - 1)
            )
            max_inventory_change = max(
                max_inventory_change, float(inventory_change)
            )
            max_vapor_change = max(
                max_vapor_change, float(np.max(abs(gas_mass - vapor)))
            )
            diameters.append(
                1000 * np.cbrt(6 * wet_mass / (np.pi * WATER_DENSITY))
            )
    history = np.array(diameters)
    assert max_inventory_change < 1e-11 and np.all(
        np.diff(history, axis=0) <= 1e-10
    )
    run = {
        "boxes": boxes,
        "step_s": step_s,
        "sample_interval_s": interval_s,
        "duration_s": float(times_s[-1]),
        "max_water_budget_relative": max_inventory_change,
        "max_vapor_change_kg_m3": max_vapor_change,
    }
    return times_s, history, run


# %% [markdown]
# ## Convert changing wet diameter to assigned terminal speed
#
# Ventilation enhances heat and mass transfer as air flows around a falling
# drop. For the first estimate, apply one empirical multiplier to the native
# evaporation rate. The elapsed physical time to reach a given size is the
# integral of its reciprocal along the unventilated trajectory.
# [Loftus and Wordsworth, equations 10–14](https://par.nsf.gov/servlets/purl/10221557)
#
# Interpolate the measured speeds at the initial and final sizes to isolate
# the effect of shrinkage. Hold the experimental speed relation fixed across
# all humidity scenarios. Retain both unventilated and ventilated endpoints
# at each exposure time. Each table has one row per box and one column per
# exposure time.


# %%
def ventilation(transfer_argument):
    return np.where(
        transfer_argument < 1.4,
        1 + 0.108 * transfer_argument**2,
        0.78 + 0.308 * transfer_argument,
    )


def exposure_endpoints(times_s, history, humidity, reference):
    assert history.min() >= reference[0].min()
    assert history.max() <= reference[0].max() + 1e-12
    final_sizes = {"unventilated": [], "ventilated": []}
    for sizes_mm, rh in zip(history.T, humidity, strict=True):
        speed = np.interp(sizes_mm, *reference)
        nu = AIR_VISCOSITY / air_density(rh)
        re_root = np.sqrt(speed * sizes_mm * 0.001 / nu)
        factor = ventilation(re_root * (nu / DIFFUSIVITY) ** (1 / 3))
        ventilated_s = cumulative_trapezoid(1 / factor, times_s, initial=0)
        assert ventilated_s[-1] >= EXPOSURES_S[-1]
        for name, clock in (
            ("unventilated", times_s),
            ("ventilated", ventilated_s),
        ):
            final_sizes[name].append(np.interp(EXPOSURES_S, clock, sizes_mm))
    initial_speed = np.interp(history[0], *reference)[:, None]
    endpoints = {}
    for name, sizes in final_sizes.items():
        final_speed = np.interp(sizes, *reference)
        endpoints[name] = {
            "final_mm": np.asarray(sizes).tolist(),
            "final_speed_m_s": final_speed.tolist(),
            "decrease_pct": (100 * (1 - final_speed / initial_speed)).tolist(),
        }
    return endpoints


# %% [markdown]
# ## Test sensitivity to heat exchange at the drop surface
#
# A second model solves for the surface temperature at which heat supplied
# by the air balances latent heat carried away by water vapor. Airflow
# enhances these fluxes by separate heat and mass transfer factors.
# Use the midpoint rule to update diameter over one second, then repeat
# at twice the step size. This independent CPU sensitivity assumes constant
# ambient humidity, a steady surface temperature, and continuum transfer.


# %%
def heat_balance_rate(diameter_mm, humidity, reference):
    """Return the ventilated change in diameter (mm/s) at steady surface temperature."""
    speed = np.interp(diameter_mm, *reference)
    nu = AIR_VISCOSITY / air_density(humidity)
    re_root = np.sqrt(speed * diameter_mm * 0.001 / nu)
    mass_factor = ventilation(re_root * (nu / DIFFUSIVITY) ** (1 / 3))
    prandtl = AIR_VISCOSITY * AIR_HEAT_CAPACITY / AIR_CONDUCTIVITY
    heat_factor = ventilation(re_root * prandtl ** (1 / 3))
    vapor = humidity * SATURATION_PA / (WATER_GAS_CONSTANT * TEMPERATURE_K)
    lower_k = np.full_like(diameter_mm, 240.0)
    upper_k = np.full_like(diameter_mm, TEMPERATURE_K)
    for _ in range(35):
        surface_k = (upper_k + lower_k) / 2
        saturation = SATURATION_PA * np.exp(
            LATENT_HEAT
            / WATER_GAS_CONSTANT
            * (1 / TEMPERATURE_K - 1 / surface_k)
        )
        excess = saturation / (WATER_GAS_CONSTANT * surface_k) - vapor
        heat_supply = (
            AIR_CONDUCTIVITY * heat_factor * (TEMPERATURE_K - surface_k)
        )
        latent_loss = LATENT_HEAT * DIFFUSIVITY * mass_factor * excess
        lower_k = np.where(heat_supply > latent_loss, surface_k, lower_k)
        upper_k = np.where(heat_supply > latent_loss, upper_k, surface_k)
    return (
        -4e6
        * DIFFUSIVITY
        * mass_factor
        * excess
        / (WATER_DENSITY * diameter_mm)
    )


def heat_balance_endpoint(initial_mm, humidity, reference, steps):
    diameter = initial_mm.copy()
    for _ in range(steps):
        rate = heat_balance_rate(diameter, humidity, reference)
        midpoint = diameter + rate / (2 * steps)
        diameter += heat_balance_rate(midpoint, humidity, reference) / steps
    return diameter


# %% [markdown]
# ## Compare the pilot with continuum theory before running the grid
#
# At 50% RH, the continuum heat and mass transfer resistances give a second
# estimate for the pilot's initial decrease in diameter squared. Proceed if
# the first CUDA interval agrees within 5%. For the full comparison, vary
# starting diameter from 0.20 to 1.00 mm and RH from 30% to 99%.
# Quantify changes from halving the time step, refining the CPU heat-balance
# integration, and using smooth instead of linear interpolation of the speed
# table. The saved report contains the endpoints needed to redraw the figure.


# %%
def simulate():
    """Run both drag laws, the CUDA pilot and two grids, and thermal sensitivity."""
    import warp as wp

    device = wp.get_device("cuda:0")
    if not device.is_cuda:
        raise RuntimeError("This example requires a CUDA device.")
    reference = read_reference()
    fixed = fixed_size_benchmark(reference)
    pilot_rh = np.array([0.5])
    times, history, pilot = evaporate_on_cuda(np.array([0.2]), pilot_rh, 0.025)
    pilot["endpoints"] = exposure_endpoints(times, history, pilot_rh, reference)
    native_rate = -np.diff((history[:2, 0] * 0.001) ** 2).item() / times[1]
    continuum_rate = (
        4
        / WATER_DENSITY
        / (
            WATER_GAS_CONSTANT * TEMPERATURE_K / (DIFFUSIVITY * SATURATION_PA)
            + LATENT_HEAT
            / (AIR_CONDUCTIVITY * TEMPERATURE_K)
            * (LATENT_HEAT / (WATER_GAS_CONSTANT * TEMPERATURE_K) - 1)
        )
    )
    assert abs(native_rate / continuum_rate - 1) < 0.05
    initial, humidity = (
        values.ravel()
        for values in np.meshgrid(
            np.arange(8, 41) / 40, [0.3, 0.5, 0.7, 0.9, 0.99]
        )
    )
    runs = {"pilot": pilot}
    endpoints = {}
    for name, step_s in (("coarse", 0.025), ("fine", 0.0125)):
        times, history, runs[name] = evaporate_on_cuda(
            initial, humidity, step_s
        )
        endpoints[name] = exposure_endpoints(
            times, history, humidity, reference
        )

    heat_final = heat_balance_endpoint(initial, humidity, reference, steps=100)
    heat_coarse = heat_balance_endpoint(initial, humidity, reference, steps=50)
    assert heat_final.min() >= reference[0].min()
    heat_speed = np.interp(heat_final, *reference)
    initial_speed = np.interp(initial, *reference)
    heat_decrease = 100 * (1 - heat_speed / initial_speed)
    ventilated = endpoints["fine"]["ventilated"]
    smooth_speed = PchipInterpolator(*reference)
    smooth_decrease = 100 * (
        1
        - smooth_speed(np.asarray(ventilated["final_mm"])[:, -1])
        / smooth_speed(initial)
    )
    max_change = {}
    for quantity in ("decrease_pct", "final_speed_m_s"):
        max_change[quantity] = max(
            float(
                np.max(
                    np.abs(
                        np.asarray(endpoints["fine"][name][quantity])
                        - np.asarray(endpoints["coarse"][name][quantity])
                    )
                )
            )
            for name in ("ventilated", "unventilated")
        )
    checks = {
        "max_time_refinement_pp": max_change["decrease_pct"],
        "max_speed_refinement_m_s": max_change["final_speed_m_s"],
        "max_heat_balance_midpoint_refinement_um": float(
            1000 * np.max(np.abs(heat_coarse - heat_final))
        ),
        "max_speed_interpolator_sensitivity_pp": float(
            np.max(
                np.abs(
                    smooth_decrease
                    - np.asarray(ventilated["decrease_pct"])[:, -1]
                )
            )
        ),
        "pilot_native_D2_loss_m2_s": native_rate,
        "continuum_D2_loss_m2_s": float(continuum_rate),
        "pilot_slope_difference_pct": float(
            100 * (native_rate / continuum_rate - 1)
        ),
    }
    return {
        "device": device.name,
        "benchmark": fixed,
        "runs": runs,
        "initial_mm": initial.tolist(),
        "humidity_fraction": humidity.tolist(),
        "exposures_s": EXPOSURES_S.tolist(),
        "initial_speed_m_s": initial_speed.tolist(),
        "fine": endpoints["fine"],
        "coarse": endpoints["coarse"],
        "heat_balance_1s": {
            "final_mm": heat_final.tolist(),
            "final_speed_m_s": heat_speed.tolist(),
            "decrease_pct": heat_decrease.tolist(),
        },
        "checks": checks,
    }


# %% [markdown]
# ## Compare a fixed-size drag error with a possible sizing-delay shift
#
# In the left panel, the circles are tabulated observations and the lines
# show the two calculations at fixed diameter. The right panel isolates the
# one-second change in assigned speed due to shrinkage at three humidities.
# Solid and dotted lines show the two thermal treatments. The printed
# values compare the small and large ends of the modeled range at 50% RH.


# %%
def plot_results(report):
    """Render the numerical comparison on CPU from the saved report."""
    bench = report["benchmark"]
    sizes = np.array(report["initial_mm"])
    humidity = np.array(report["humidity_fraction"])
    latent = report["fine"]["ventilated"]
    one_second = report["exposures_s"].index(1.0)
    decreases = np.array(latent["decrease_pct"])[:, one_second]
    heat = np.array(report["heat_balance_1s"]["decrease_pct"])
    measured = np.asarray(bench["measured_m_s"])
    sphere = np.asarray(bench["sphere_m_s"])
    diameter = np.asarray(bench["diameter_mm"])
    differences = 100 * (sphere / measured - 1)
    selected = (diameter >= 0.20) & (diameter <= 1.00)
    print(
        f"Particula sphere drag differs from the table by at most "
        f"{np.max(np.abs(differences[selected])):.2f}% at the "
        f"{np.sum(selected)} sizes from 0.20 to 1.00 mm."
    )
    print(f"CUDA evaporation device: {report['device']}")
    for index in np.flatnonzero((humidity == 0.5) & np.isin(sizes, [0.2, 1.0])):
        print(
            f"{sizes[index]:.2f} mm at 50% RH after 1 s: "
            f"{latent['final_mm'][index][one_second]:.4f} mm, "
            f"assigned {latent['final_speed_m_s'][index][one_second]:.3f} m/s, "
            f"{decreases[index]:.2f}% decrease (surface heat balance {heat[index]:.2f}%)."
        )
    print(
        "Largest time-step effect on assigned-speed decrease: "
        f"{report['checks']['max_time_refinement_pp']:.3g} percentage points."
    )
    figure, (drag, delay) = plt.subplots(
        1, 2, figsize=(10, 4.5), layout="constrained"
    )
    drag.plot(
        bench["diameter_mm"],
        bench["stokes_m_s"],
        "--",
        color="#8a5a2c",
        label="Stokes",
    )
    drag.plot(
        bench["diameter_mm"],
        bench["sphere_m_s"],
        color="#007472",
        label="Finite-Re sphere drag",
    )
    drag.scatter(
        bench["diameter_mm"],
        bench["measured_m_s"],
        facecolor="white",
        edgecolor="#172f42",
        zorder=3,
        label="Gunn and Kinzer table",
    )
    drag.set(
        yscale="log",
        xlabel="Mass-equivalent diameter (mm)",
        ylabel="Terminal speed (m/s)",
        title="Fixed diameters at 20 °C, 50% RH",
    )
    for rh, color in [(0.3, "#973c24"), (0.5, "#007472"), (0.9, "#365a98")]:
        mask = humidity == rh
        delay.plot(
            sizes[mask], decreases[mask], color=color, label=f"{rh:.0%} RH"
        )
        delay.plot(sizes[mask], heat[mask], color=color, linestyle=":")
    delay.plot([], [], color="#343434", label="Latent-rate model")
    delay.plot(
        [], [], color="#343434", linestyle=":", label="Surface heat balance"
    )
    delay.set(
        xlabel="Initial wet diameter (mm)",
        ylabel="Decrease from initial terminal speed (%)",
        title="Predicted sizing-delay shift after 1 s",
    )
    for axis in (drag, delay):
        axis.legend(frameon=False, fontsize=9)
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    return figure


if __name__ == "__main__":
    plot_results(simulate())
    plt.show()

# %% [markdown]
# Finite-Reynolds-number drag captures the 0.30 mm reference speed to within
# one printed increment. At 0.20 mm and 50% RH, both thermal treatments
# predict a decrease above 1% in the assigned speed after one second. At
# 1.00 mm both give a decrease below 0.2%. A paired measurement of wet size
# and speed at each station would test shrinkage and reveal any departure
# from local terminal speed during the fall.
