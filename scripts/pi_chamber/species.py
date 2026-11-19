"""Compare salt activation and organic vapor transfer in closed boxes."""

import csv
from pathlib import Path

import numpy as np
from particula import Aerosol
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
)
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.execution.gpu_resident_condensation import (
    GPUResidentCondensation,
)
from particula.gas import EnvironmentData, GasData
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
    WaterBuckStrategy,
)
from particula.particles import ActivityBAT, ActivityIdealMolar, ParticleData
from particula.particles.surface_strategies import SurfaceStrategyVolume
from particula.util.constants import GAS_CONSTANT
from scipy.optimize import brentq, minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
TEMP = 293.15
DIAMETER = 130e-9
MOLAR = np.array([0.01801528, 0.200])
DENSITY = np.array([1000.0, 1243.4933333333333])
SURFACE = 0.072
KELVIN = 4 * SURFACE * MOLAR / (GAS_CONSTANT * TEMP * 1000)
NUMBER = 2e9  # particles/m³
TIMES = (0, 1, 4, 16, 64)
BAT = ActivityBAT(MOLAR[1], 0.5, DENSITY[1])


def salt_curve(kappa):
    """Solve the subsaturated wet diameter and the κ–Köhler maximum."""

    def ratio(d):
        v = (d / 130) ** 3 - 1
        return v / (v + kappa) * np.exp(KELVIN[0] * 1e9 / d)

    peak = minimize_scalar(
        lambda d: -ratio(d), bounds=(130, 5000), method="bounded"
    ).x
    return (
        brentq(lambda d: ratio(d) - 0.9955, 130, peak),
        peak,
        100 * (ratio(peak) - 1),
    )


def initial_mass(rh, activity):
    """Equilibrate water at the specified RH for a 130 nm organic seed."""
    organic = np.pi / 6 * DIAMETER**3 * DENSITY[1]

    def residual(ratio):
        mass = organic * np.array([ratio, 1])
        wet = (6 / np.pi * np.sum(mass / DENSITY)) ** (1 / 3)
        return activity.activity(mass)[0] * np.exp(KELVIN[0] / wet) - rh

    ratio = brentq(residual, 0, 3000)
    return organic * np.array([ratio, 1])


def run(cstar, rtol, activity):
    """Integrate both volatile species in six independent closed boxes."""
    cases = [(rh, dose) for rh in [0.90, 0.9955] for dose in [0.0, 1.0, 5.0]]
    masses = np.stack([initial_mass(rh, activity) for rh, _ in cases])[
        :, None, :
    ]
    count = len(cases)
    particles = ParticleData(
        masses=masses.copy(),
        concentration=np.full((count, 1), 64.0),
        volume=np.full(count, 64 / NUMBER),
        charge=np.zeros((count, 1)),
        density=DENSITY,
        distribution_type="particle_resolved",
    )
    water_p = float(WaterBuckStrategy().pure_vapor_pressure(TEMP))
    gas = np.array(
        [
            [rh * water_p * MOLAR[0] / (GAS_CONSTANT * TEMP), dose * 1e-9]
            for rh, dose in cases
        ]
    )
    names = ["water", "organic"]
    state = Aerosol(
        particles=particles,
        gas=GasData(names, MOLAR.copy(), gas.copy(), np.ones(2, dtype=bool)),
        environment=EnvironmentData(
            np.full(count, TEMP),
            np.full(count, 101325.0),
            np.array([[rh, dose / cstar] for rh, dose in cases]),
        ),
    )
    strategy = CondensationIsothermal(
        molar_mass=MOLAR.copy(),
        diffusion_coefficient=np.array([2.4e-5, 7e-6]),
        accommodation_coefficient=np.array([0.2, 1.0]),
        activity_strategy=activity,
        surface_strategy=SurfaceStrategyVolume(
            density=1000.0, surface_tension=SURFACE
        ),
        vapor_pressure_strategy=(
            WaterBuckStrategy(),
            ConstantVaporPressureStrategy(
                cstar * 1e-9 * GAS_CONSTANT * TEMP / MOLAR[1]
            ),
        ),
        species_map=CondensationSpeciesMap(tuple(names), ((0, 0), (1, 1))),
    )
    owner = GPUResidentCondensation(
        state,
        strategy,
        device="cuda:0",
        method="ros3",
        rtol=rtol,
        particle_atol=1e-27,
        gas_atol=1e-18,
        max_step=1.0,
        max_trials=10000,
    )
    rows = []
    for t in TIMES:
        if t:
            owner.advance(t)
        snapshot = owner.snapshot()
        pm = snapshot.particles.species_mass_density
        vapor = snapshot.gas.concentration
        closure = pm - masses[:, 0] * NUMBER + (vapor - gas)
        if not (
            np.isfinite(pm).all() and np.all(pm >= 0) and np.all(vapor >= 0)
        ):
            raise ArithmeticError("Invalid gas or particle mass")
        np.testing.assert_allclose(closure[:, 0], 0, atol=3e-14)
        np.testing.assert_allclose(closure[:, 1], 0, atol=2e-19)
        for i, (rh, dose) in enumerate(cases):
            rows.append(
                dict(
                    activity=type(activity).__name__,
                    rh_pct=100 * rh,
                    initial_vapor_ug_m3=dose,
                    cstar_ug_m3=cstar,
                    relative_tolerance=rtol,
                    time_s=t,
                    wet_nm=2e9 * snapshot.particles.radii[i, 0],
                    particle_organic_ug_m3=pm[i, 1] * 1e9,
                    gas_organic_ug_m3=vapor[i, 1] * 1e9,
                    particle_water_ug_m3=pm[i, 0] * 1e9,
                    water_error_kg_m3=closure[i, 0],
                    organic_error_kg_m3=closure[i, 1],
                )
            )
    return rows


def publish():
    """Check the tolerances and save the figure and tabulated evidence."""
    out = ROOT / "docs/data/pi-chamber"
    all_rows = []
    for cstar in (1.0, 10.0):
        coarse = run(cstar, 1e-4, BAT)
        fine = run(cstar, 1e-6, BAT)
        all_rows.extend(coarse + fine)
        all_rows.extend(run(cstar, 1e-6, ActivityIdealMolar(MOLAR)))
        for field in (
            "wet_nm",
            "particle_organic_ug_m3",
            "particle_water_ug_m3",
        ):
            np.testing.assert_allclose(
                [r[field] for r in coarse], [r[field] for r in fine], rtol=1e-4
            )
    with (out / "organic-vapor.csv").open("w") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(all_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(all_rows)
    from scripts.pi_chamber.figures import organic_vapor

    organic_vapor()
    for name, k in [("NaCl", 1.28), ("ammonium sulfate proxy", 0.61)]:
        print(name, salt_curve(k))


if __name__ == "__main__":
    publish()
