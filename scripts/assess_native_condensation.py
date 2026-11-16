"""Compare native condensation steps/rates against independent radius ODEs.

Single material, monodisperse resolved particles, fixed T/P/viscosity and
water-like Kelvin correction. Gas depletion follows closed species accounting.
References use radius rather than native mass coordinates and no production
transport helper.
All values are SI; normalized mass and radius errors are dimensionless.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
    CondensationIsothermalStaggered,
)
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.gas import GasData
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)
from particula.particles import create_resolved_particle_data
from particula.particles.activity_strategies import ActivityIdealMass
from particula.particles.surface_strategies import SurfaceStrategyVolume
from scipy.integrate import solve_ivp

RADIUS = 1e-6
DENSITY = 1000.0
DIFFUSION = 2e-5
TEMPERATURE = 298.15
PRESSURE = 101325.0
VISCOSITY = 1.8e-5
MOLAR_MASS = 0.018
GAS_CONSTANT = 8.31446261815324
MASS = 4 * np.pi * DENSITY * RADIUS**3 / 3
NUMBER = 1e9
VAPOR = 1e-5
VOLUME = 0.37


def initial(strategy_type, equilibrium):
    """Build active/inactive slots and an unmapped inert gas."""
    particle = create_resolved_particle_data(
        masses=np.array([[MASS], [0.0]]),
        density=np.array([DENSITY]),
        concentration=np.array([NUMBER * VOLUME, 0.0]),
        volume=VOLUME,
    )
    gas = GasData(
        name=["inert", "vapor"],
        molar_mass=np.array([0.028, MOLAR_MASS]),
        concentration=np.array([[0.1, VAPOR]]),
        partitioning=np.array([False, True]),
    )
    options = (
        {"shuffle_each_step": False, "theta_mode": "half"}
        if strategy_type is CondensationIsothermalStaggered
        else {}
    )
    strategy = strategy_type(
        molar_mass=np.array([0.028, MOLAR_MASS]),
        diffusion_coefficient=DIFFUSION,
        activity_strategy=ActivityIdealMass(),
        surface_strategy=SurfaceStrategyVolume(
            density=DENSITY, surface_tension=0.072
        ),
        vapor_pressure_strategy=ConstantVaporPressureStrategy(
            equilibrium * GAS_CONSTANT * TEMPERATURE / MOLAR_MASS
        ),
        species_map=CondensationSpeciesMap(("inert", "vapor"), ((1, 0),)),
        **options,
    )
    return particle, gas, strategy


def reference(duration, equilibrium, times):
    """Cross-check radius equations with two tight independent integrations."""
    mean_free_path = (2 * VISCOSITY / PRESSURE) / np.sqrt(
        8 * MOLAR_MASS / (np.pi * GAS_CONSTANT * TEMPERATURE)
    )

    def rate(time, radius):
        knudsen = mean_free_path / (RADIUS * radius[0])
        correction = (
            0.75 * (1 + knudsen) / (knudsen**2 + 1.283 * knudsen + 0.75)
        )
        vapor = VAPOR - NUMBER * MASS * (radius[0] ** 3 - 1)
        surface_vapor = equilibrium * np.exp(
            2
            * 0.072
            * MOLAR_MASS
            / (GAS_CONSTANT * TEMPERATURE * DENSITY * RADIUS * radius[0])
        )
        return np.array(
            [
                DIFFUSION
                * correction
                * (vapor - surface_vapor)
                / (DENSITY * RADIUS**2 * radius[0])
            ]
        )

    results = []
    for method in ("DOP853", "Radau"):
        solution = solve_ivp(
            rate,
            (0, duration),
            [1.0],
            method=method,
            rtol=2e-12,
            atol=2e-14,
            t_eval=times,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        results.append(solution.y[0])
    difference = float(np.max(np.abs(results[0] - results[1])))
    if difference > 1e-9:
        raise RuntimeError("reference disagreement exceeds target")
    return results[0], difference


def metrics(radius, vapor, expected, equilibrium):
    """Keep radius, small gas reservoir and material budget separate."""
    reference_vapor = VAPOR - NUMBER * MASS * (expected**3 - 1)
    return {
        "radius_error": float(np.max(np.abs(radius - expected))),
        "vapor_error_over_initial": float(
            np.max(np.abs(vapor - reference_vapor)) / VAPOR
        ),
        "relative_budget_error": float(
            np.max(
                np.abs(
                    vapor + NUMBER * MASS * radius**3 - (VAPOR + NUMBER * MASS)
                )
            )
            / (VAPOR + NUMBER * MASS)
        ),
        "minimum_vapor": float(np.min(vapor)),
        "equilibrium_vapor": equilibrium,
    }


def fixed_run(kind, duration, equilibrium, count):
    """Assess existing finite-step maps, retaining evidence of clipping."""
    particle, gas, strategy = initial(kind, equilibrium)
    times = np.linspace(0, duration, count + 1)
    expected, uncertainty = reference(duration, equilibrium, times)
    radii, vapors = [1.0], [VAPOR]
    limiting_proposals = 0
    start = perf_counter()
    for _ in range(count):
        proposed = (
            strategy.mass_transfer_rate(
                particle, gas, TEMPERATURE, PRESSURE, VISCOSITY
            )[0, 0]
            * duration
            / count
        )
        # Diagnostic only: staggered internal sweeps have different proposals.
        limiting_proposals += int(
            NUMBER * proposed > gas.concentration[0, 1]
            or proposed < -particle.masses[0, 0, 0]
        )
        strategy.step(
            particle, gas, TEMPERATURE, PRESSURE, duration / count, VISCOSITY
        )
        radii.append(particle.radii[0, 0] / RADIUS)
        vapors.append(gas.concentration[0, 1])
        if gas.concentration[0, 0] != 0.1 or particle.masses[0, 1, 0] != 0:
            raise RuntimeError("inert lane or inactive slot changed")
    return {
        "method": kind.__name__,
        "steps": count,
        "seconds": perf_counter() - start,
        "reference_difference": uncertainty,
        "full_step_limiting_proposals": limiting_proposals,
        **metrics(np.array(radii), np.array(vapors), expected, equilibrium),
    }


def adaptive_run(method, duration, equilibrium, tolerance):
    """Use native continuous mass rates with conservation-derived gas state.

    This is a scalar research prototype, not the supported production adapter.
    Invalid intermediate states are reported as failures, never clipped.
    """
    particle, gas, strategy = initial(CondensationIsothermal, equilibrium)
    times = np.linspace(0, duration, 129)
    expected, uncertainty = reference(duration, equilibrium, times)

    def rate(time, normalized_mass):
        particle.masses[0, 0, 0] = MASS * normalized_mass[0]
        gas.concentration[0, 1] = VAPOR - NUMBER * MASS * (
            normalized_mass[0] - 1
        )
        return np.array(
            [
                strategy.mass_transfer_rate(
                    particle, gas, TEMPERATURE, PRESSURE, VISCOSITY
                )[0, 0]
                / MASS
            ]
        )

    start = perf_counter()
    solution = solve_ivp(
        rate,
        (0, duration),
        [1.0],
        method=method,
        rtol=tolerance,
        atol=tolerance * 1e-3,
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    mass = solution.sol(times)[0]
    vapor = VAPOR - NUMBER * MASS * (mass - 1)
    return {
        "method": method,
        "rtol": tolerance,
        "seconds": perf_counter() - start,
        "accepted_steps": len(solution.t) - 1,
        "nfev": solution.nfev,
        "njev": solution.njev,
        "nlu": solution.nlu,
        "reference_difference": uncertainty,
        **metrics(np.cbrt(mass), vapor, expected, equilibrium),
    }


def assess():
    """Run growth, equilibrium and depletion comparisons without filtering."""
    rows = []
    for name, duration, equilibrium in (
        ("smooth_growth", 1.4, 0.0),
        ("equilibrium", 20.0, 1e-6),
        ("near_depletion", 20.0, 0.0),
    ):
        for kind in (CondensationIsothermal, CondensationIsothermalStaggered):
            for count in (8, 16, 32, 64, 128):
                rows.append(
                    {
                        "case": name,
                        "status": "success",
                        **fixed_run(kind, duration, equilibrium, count),
                    }
                )
        for method in ("RK23", "RK45", "Radau", "BDF"):
            for tolerance in (1e-3, 1e-5, 1e-7):
                row = {"case": name, "method": method, "rtol": tolerance}
                try:
                    row.update(
                        adaptive_run(method, duration, equilibrium, tolerance)
                    )
                    row["status"] = "success"
                except (ValueError, RuntimeError) as exc:
                    row.update(status="failed", reason=str(exc))
                rows.append(row)
    return rows


def main():
    """Write the preliminary native comparison report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "runs": assess(),
        "volume_m3": VOLUME,
        "units": "SI; radius divided by initial radius",
        "scope": "CPU single-box monodisperse research prototype",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
