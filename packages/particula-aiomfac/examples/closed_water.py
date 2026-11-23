"""Run two finite water reservoirs against the pinned four-component mixture.

Run from an unpacked backend source distribution with particula[aiomfac]
installed. The output is CSV, one line per reservoir and time point.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
)
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.gas.gas_data import GasData
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)
from particula.particles.activity_factories import ActivityFactory
from particula.particles.particle_data import ParticleData
from particula.particles.surface_strategies import SurfaceStrategyVolume
from particula.util.constants import GAS_CONSTANT
from particula_aiomfac import MixtureSpec

REF = Path(__file__).resolve().parents[1] / "reference"
MOLAR_MASS = 0.01801528
TEMPERATURE = 298.0
PURE_PRESSURE = 3167.0
DENSITY = 1000.0
SIGMA = 0.072


def setup(saturation, permuted=False, case_id="mixed"):
    case = json.loads((REF / f"cases/{case_id}.json").read_text())
    groups = json.loads((REF / "cases/components.json").read_text())
    mixture = MixtureSpec(
        {
            k: {int(g): v for g, v in groups[k].items()}
            for k in case["components"]
        }
    )
    names = list(mixture.components)
    if permuted:
        names = ["NaCl", "inert", "butanediol", "water", "vanillic_acid"]
    water = names.index("water")
    provider = ActivityFactory().get_strategy(
        "aiomfac",
        {
            "mixture": mixture,
            "material_mapping": {
                i: k for i, k in enumerate(names) if k != "inert"
            },
            "inert_indices": [i for i, k in enumerate(names) if k == "inert"],
        },
    )
    fractions = dict(zip(case["components"], case["fractions"], strict=True))
    masses = np.array([fractions.get(k, 0.2) for k in names]) * 5e-15
    population = masses[None, :]
    if permuted:
        # A different salt fraction on the second row; the third slot is vacant.
        second = masses.copy()
        second[names.index("NaCl")] *= 2
        population = np.stack([masses, second, np.zeros_like(masses)])
    number = np.where(population.sum(axis=1) > 0, 1e10, 0)
    particle = ParticleData(
        masses=population[None, :, :],
        concentration=number[None, :],
        charge=np.zeros((1, len(number))),
        density=np.full(len(names), DENSITY),
        volume=np.ones(1),
        distribution_type="particle_resolved",
    )
    reference = json.loads((REF / f"expected/{case_id}.json").read_text())
    aw = reference["neutrals"]["water"]["activity"]
    temperature = case["temperature_K"]
    curvature = 2 * SIGMA * MOLAR_MASS / (GAS_CONSTANT * temperature * DENSITY)
    expected = PURE_PRESSURE * aw * np.exp(curvature / particle.radii[0, 0])
    gas_names = ["inert", "water"] if permuted else ["water"]
    source = gas_names.index("water")
    molar = np.array([MOLAR_MASS if k == "water" else 0.028 for k in gas_names])
    gas = GasData(
        name=gas_names,
        molar_mass=molar,
        concentration=np.array(
            [
                [
                    saturation
                    * expected
                    * MOLAR_MASS
                    / (GAS_CONSTANT * temperature)
                    if k == "water"
                    else 0
                    for k in gas_names
                ]
            ]
        ),
        partitioning=np.array([k == "water" for k in gas_names]),
    )
    process = CondensationIsothermal(
        molar_mass=molar,
        diffusion_coefficient=2.5e-5,
        accommodation_coefficient=1.0,
        activity_strategy=provider,
        surface_strategy=SurfaceStrategyVolume(
            surface_tension=SIGMA, density=DENSITY
        ),
        vapor_pressure_strategy=ConstantVaporPressureStrategy(
            vapor_pressure=PURE_PRESSURE
        ),
        species_map=CondensationSpeciesMap(
            tuple(gas_names), ((source, water),)
        ),
        update_gases=True,
    )
    return process, particle, gas, source, water, expected


def trajectory(saturation, permuted=False):
    process, particle, gas, source, water, expected = setup(
        saturation, permuted
    )
    args = particle, gas, TEMPERATURE, 101325.0
    equilibrium = process._native_pressure_terms(
        particle,
        gas,
        TEMPERATURE,
        np.maximum(particle.radii[0], 1e-10),
        [source],
        [water],
    )[1]
    np.testing.assert_allclose(equilibrium[0, water], expected, rtol=1e-10)
    flux = process.mass_transfer_rate(*args)[0, water]
    assert np.sign(flux) == np.sign(saturation - 1)
    dry = np.arange(particle.n_species) != water
    dry_before = particle.masses[:, :, dry].copy()
    rows = []
    for step in range(21):
        gas_water = gas.concentration[0, source] * particle.volume[0]
        water_total = particle.concentration[0] @ particle.masses[0, :, water]
        aw = process.activity_strategy.neutral_activities(
            particle.masses[0],
            TEMPERATURE,
            [water],
            particle.concentration[0] > 0,
        )[0, 0]
        rows.append(
            dict(
                saturation=saturation,
                time_s=step * 1e-4,
                water_activity=aw,
                radius_m=particle.radii[0, 0],
                gas_water_kg=gas_water,
                particle_water_kg=water_total,
                water_budget_kg=gas_water + water_total,
            )
        )
        if step < 20:
            process.step(*args, time_step=1e-4)
    np.testing.assert_array_equal(dry_before, particle.masses[:, :, dry])
    np.testing.assert_allclose(
        [r["water_budget_kg"] for r in rows],
        rows[0]["water_budget_kg"],
        rtol=1e-12,
        atol=1e-30,
    )
    assert (rows[-1]["water_activity"] - rows[0]["water_activity"]) * (
        saturation - 1
    ) > 0
    return rows


if __name__ == "__main__":
    records = trajectory(1.02) + trajectory(0.98)
    writer = csv.DictWriter(sys.stdout, fieldnames=records[0])
    writer.writeheader()
    writer.writerows(records)
