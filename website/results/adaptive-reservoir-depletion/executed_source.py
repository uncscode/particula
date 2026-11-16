# %% [markdown]
# # Resolve growth as a finite vapor reservoir is depleted
#
# A mass-conserving update can move vapor into particles too quickly. Here a
# closed population of 100 nm organic particles consumes an initially abundant
# vapor reservoir. Compare coarse and refined fixed steps with adaptive RK23
# and ROS2 using an independently written scalar growth equation. Radius and
# remaining vapor test accuracy; their sum tests a separate mass budget.
#
# This idealized isothermal calculation uses one material, unit activity,
# constant vapor pressure, and Kelvin curvature. There is no coagulation,
# wall loss, source, latent heating, or claim of a measured organic compound.
# Number density is 10¹¹ m⁻³ in a 10⁻⁶ m³ representative volume. All methods use
# identical initial conditions and output times. An optional CUDA companion
# exercises the same reference and error targets on the actual device.

# %%
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import particula as par
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.execution.condensation_ode import CondensationODE
from scipy.integrate import solve_ivp

CONFIGURATION = {
    "radius_m": 100e-9,
    "number_m3": 1e11,
    "volume_m3": 1e-6,
    "density_kg_m3": 1400.0,
    "molar_mass_kg_mol": 0.18,
    "temperature_K": 298.15,
    "pressure_Pa": 101325.0,
    "diffusion_m2_s": 5e-6,
    "surface_tension_N_m": 0.03,
    "vapor_pressure_Pa": 1e-5,
    "initial_vapor_kg_m3": 1e-6,
}
TIMES = np.linspace(0, 10, 21)


def build_case():
    """Declare the same closed native state for every numerical method."""
    c = CONFIGURATION
    mass = 4 * np.pi * c["density_kg_m3"] * c["radius_m"] ** 3 / 3
    particles = par.particles.create_resolved_particle_data(
        masses=[mass],
        density=c["density_kg_m3"],
        concentration=[c["number_m3"] * c["volume_m3"]],
        volume=c["volume_m3"],
    )
    gas = par.gas.GasData(
        name=["organic"],
        molar_mass=np.array([c["molar_mass_kg_mol"]]),
        concentration=np.array([[c["initial_vapor_kg_m3"]]]),
        partitioning=np.array([True]),
    )
    environment = par.gas.EnvironmentData(
        temperature=[c["temperature_K"]],
        pressure=[c["pressure_Pa"]],
        saturation_ratio=[[0.0]],
    )
    strategy = par.dynamics.CondensationIsothermal(
        molar_mass=gas.molar_mass,
        diffusion_coefficient=c["diffusion_m2_s"],
        accommodation_coefficient=1.0,
        activity_strategy=par.particles.ActivityIdealMass(),
        surface_strategy=par.particles.SurfaceStrategyVolume(
            surface_tension=c["surface_tension_N_m"],
            density=c["density_kg_m3"],
        ),
        vapor_pressure_strategy=par.gas.ConstantVaporPressureStrategy(
            c["vapor_pressure_Pa"]
        ),
        species_map=CondensationSpeciesMap(("organic",), ((0, 0),)),
    )
    return par.Aerosol(
        particles=particles, gas=gas, environment=environment
    ), strategy


# %% [markdown]
# Let x=(r/r₀)³ be particle mass divided by initial mass m₀. Conservation gives
# C(x)=C₀−N m₀(x−1). The diffusion equation is
# dx/dt=3 D f(Kn) r [C(x)−C_sat exp(A/r)]/(ρ r₀³), with
# A=2 σ M/(R T ρ) and unit-accommodation correction
# f(Kn)=0.75(1+Kn)/(Kn²+1.283 Kn+0.75). The reference below writes these relations
# directly and cross-checks DOP853 against Radau. It does not call Particula's
# rate, activity, radius, or gas-budget helpers. The constants are specified
# explicitly to make this an independent implementation check.


# %%
def reference(method):
    """Solve the independently specified scalar mass-ratio equation."""
    c = CONFIGURATION
    temperature, pressure = c["temperature_K"], c["pressure_Pa"]
    molar, density = c["molar_mass_kg_mol"], c["density_kg_m3"]
    radius0 = c["radius_m"]
    gas_constant = 8.31446261815324
    mass0 = 4 * np.pi * density * radius0**3 / 3
    viscosity = (
        1.716e-5
        * (temperature / 273.15) ** 1.5
        * 383.55
        / (temperature + 110.4)
    )
    path = (2 * viscosity / pressure) / np.sqrt(
        8 * molar / (np.pi * gas_constant * temperature)
    )
    kelvin = (
        2
        * c["surface_tension_N_m"]
        * molar
        / (gas_constant * temperature * density)
    )
    saturated = c["vapor_pressure_Pa"] * molar / (gas_constant * temperature)

    def rate(time, values):
        ratio = values[0]
        radius = radius0 * np.cbrt(ratio)
        vapor = c["initial_vapor_kg_m3"] - c["number_m3"] * mass0 * (ratio - 1)
        kn = path / radius
        correction = 0.75 * (1 + kn) / (kn**2 + 1.283 * kn + 0.75)
        return [
            3
            * c["diffusion_m2_s"]
            * radius
            * correction
            * (vapor - saturated * np.exp(kelvin / radius))
            / (density * radius0**3)
        ]

    solution = solve_ivp(
        rate,
        (0, TIMES[-1]),
        [1.0],
        method=method,
        t_eval=TIMES,
        rtol=1e-11,
        atol=1e-13,
    )
    assert solution.success, solution.message
    ratio = solution.y[0]
    return np.column_stack(
        (
            radius0 * np.cbrt(ratio),
            c["initial_vapor_kg_m3"] - c["number_m3"] * mass0 * (ratio - 1),
        )
    )


def calculate(method, *, tolerance=1e-5, fixed_step=0.5, device=None):
    """Advance full physical intervals; snapshots are explicit GPU downloads."""
    state, strategy = build_case()
    owner = None
    if method != "fixed":
        controls = dict(
            rtol=tolerance,
            particle_atol=1e-28,
            gas_atol=1e-16,
            first_step=0.01,
            max_step=0.5,
        )
        if device is None:
            owner = CondensationODE(state, strategy, method=method, **controls)
        else:
            from particula.execution.gpu_condensation_ode import (
                GPUCondensationODE,
            )

            owner = GPUCondensationODE(
                state, strategy, device=device, **controls
            )
    rows = []
    fixed_steps = 0
    budget_error = 0.0
    c = CONFIGURATION
    total0 = (
        c["number_m3"] * state.particles.masses[0, 0, 0]
        + state.gas.concentration[0, 0]
    )
    for index, time in enumerate(TIMES):
        if index:
            if owner is None:
                duration = float(time - TIMES[index - 1])
                substeps = int(np.ceil(duration / fixed_step))
                par.dynamics.MassCondensation(strategy).execute(
                    state,
                    time_step=duration,
                    sub_steps=substeps,
                )
                fixed_steps += substeps
            else:
                while owner.time < time:
                    owner.step(float(time))
                state = owner.snapshot()
        mass = state.particles.masses[0, 0, 0]
        gas = state.gas.concentration[0, 0]
        radius = np.cbrt(3 * mass / (4 * np.pi * c["density_kg_m3"]))
        rows.append([radius, gas])
        budget_error = max(
            budget_error, abs(c["number_m3"] * mass + gas - total0) / total0
        )
    assert budget_error < 2e-12, budget_error
    work = (
        {"fixed_steps": fixed_steps}
        if owner is None
        else asdict(owner.statistics if device else owner.solver.statistics)
    )
    return np.asarray(rows), budget_error, work


def simulate(device=None):
    """Compare observable errors, reference uncertainty and closed budgets."""
    expected = reference("DOP853")
    independent = reference("Radau")
    np.testing.assert_allclose(independent[:, 0], expected[:, 0], rtol=2e-10)
    np.testing.assert_allclose(
        independent[:, 1], expected[:, 1], rtol=2e-8, atol=1e-15
    )
    profiles = [
        ("Fixed 0.5 s", "fixed", 1e-5, 0.5, None),
        ("Fixed 0.005 s", "fixed", 1e-5, 0.005, None),
        ("CPU RK23 1e-4", "rk23", 1e-4, 0.5, None),
        ("CPU RK23 1e-7", "rk23", 1e-7, 0.5, None),
        ("CPU ROS2 1e-5", "ros2", 1e-5, 0.5, None),
    ]
    if device is not None:
        profiles.extend(
            (
                ("CUDA RK23 1e-4", "rk23", 1e-4, 0.5, device),
                ("CUDA RK23 1e-7", "rk23", 1e-7, 0.5, device),
            )
        )
    results = {}
    for name, method, tolerance, step, backend in profiles:
        values, budget, work = calculate(
            method, tolerance=tolerance, fixed_step=step, device=backend
        )
        radius_error = float(np.max(np.abs(values[:, 0] / expected[:, 0] - 1)))
        vapor_error = float(
            np.max(np.abs(values[:, 1] - expected[:, 1]))
            / CONFIGURATION["initial_vapor_kg_m3"]
        )
        if method != "fixed":
            assert radius_error < 2e-4, (name, radius_error)
            assert vapor_error < 1e-3, (name, vapor_error)
        results[name] = {
            "radius_nm": (values[:, 0] * 1e9).tolist(),
            "vapor_ug_m3": (values[:, 1] * 1e9).tolist(),
            "radius_error": radius_error,
            "vapor_error": vapor_error,
            "budget_error": budget,
            "work": work,
        }
    for coarse, fine in (
        ("Fixed 0.5 s", "Fixed 0.005 s"),
        ("CPU RK23 1e-4", "CPU RK23 1e-7"),
    ):
        for observable in ("radius_error", "vapor_error"):
            assert results[fine][observable] < results[coarse][observable]
    if device is not None:
        for observable in ("radius_error", "vapor_error"):
            assert (
                results["CUDA RK23 1e-7"][observable]
                < results["CUDA RK23 1e-4"][observable]
            )
    return {
        "configuration": CONFIGURATION,
        "time_s": TIMES.tolist(),
        "reference_radius_nm": (expected[:, 0] * 1e9).tolist(),
        "reference_vapor_ug_m3": (expected[:, 1] * 1e9).tolist(),
        "results": results,
    }


def plot_results(report):
    """Render compact saved histories and separate error/work diagnostics."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, key, ylabel in zip(
        axes,
        ("radius_nm", "vapor_ug_m3"),
        ("Radius (nm)", "Vapor (µg m⁻³)"),
        strict=True,
    ):
        axis.plot(
            report["time_s"],
            report[f"reference_{key}"],
            color="black",
            linewidth=2,
            label="Independent reference",
        )
        for name, result in report["results"].items():
            axis.plot(
                report["time_s"],
                result[key],
                label=name,
                linestyle="--",
                alpha=0.8,
            )
        axis.set(xlabel="Time (s)", ylabel=ylabel)
    axes[0].legend(fontsize=7)
    for name, result in report["results"].items():
        print(
            name,
            "radius error:",
            result["radius_error"],
            "vapor error / initial vapor:",
            result["vapor_error"],
            "mass residual:",
            result["budget_error"],
            "work:",
            result["work"],
        )
    return fig


# %% [markdown]
# The plotted transient tests when vapor is transferred, not only the final
# total. Tightening a fixed timestep or adaptive tolerance should reduce both
# radius and vapor errors. The vapor error is normalized by initial vapor so
# an almost depleted denominator cannot exaggerate a tiny absolute difference.
# Work counters explain the accuracy cost without treating unequal CPU/GPU
# timings as a speedup. In the retained CPU comparison, reducing the fixed step
# from 0.5 s to 0.005 s reduces the maximum radius error from 1.31% to 0.0122%.
# RK23 at relative tolerance 10⁻⁷ reaches a radius error of 2.57×10⁻⁸ with
# 430 accepted steps; the initial-vapor-normalized error is 9.15×10⁻⁸. ROS2
# also meets the targets but takes 2,574 accepted steps at tolerance 10⁻⁵.
# This nonstiff case therefore does not justify choosing a stiff method solely
# because it is available. For a new compound, change its declared properties and
# repeat this comparison before adding collisions, forcing, or thermal feedback.

# %%
if __name__ == "__main__":
    report = simulate()
    figure = plot_results(report)
    plt.show()
