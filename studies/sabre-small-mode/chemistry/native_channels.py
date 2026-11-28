"""Replay two surface populations sharing a trace N2O5 gas reservoir."""

from pathlib import Path

import numpy as np
import pandas as pd
from particula import Aerosol
from particula.dynamics.chemistry import ReactionMechanism
from particula.dynamics.chemistry.process import ChemistrySolver
from particula.dynamics.chemistry.resolved import ResolvedChemistry
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData
from particula.util.constants import GAS_CONSTANT

HERE = Path(__file__).resolve().parent
N2O5, WATER = 0.1080104, 0.01801528
ACID = (N2O5 + WATER) / 2
MW = np.array([1.0, WATER, N2O5, ACID])
RHO = np.array([1500.0, 1000.0, 1500.0, 1500.0])


def channel(row, mode, gamma, pressure, gas_initial, aqueous_rate):
    """Match wet volume, water and area with one weighted sphere."""
    surface = row[f"wet_{mode}_area"] * 1e-6
    volume = row[f"wet_{mode}_volume_um3_cm3"] * 1e-12
    water_v = row[f"water_{mode}_volume_um3_cm3"] * 1e-12
    radius = 3 * volume / surface
    number = surface / (4 * np.pi * radius**2)
    mass = (
        np.array([(volume - water_v) * RHO[0], water_v * RHO[1], 0.0, 0.0])
        / number
    )
    particles = ParticleData(
        masses=mass[None, None, :],
        density=RHO,
        concentration=np.array([[number]]),
        charge=np.zeros((1, 1)),
        volume=np.ones(1),
        distribution_type="particle_resolved",
    )
    env = EnvironmentData(
        temperature=np.array([row.temperature_k]),
        pressure=np.array([pressure]),
        saturation_ratio=np.zeros((1, 1)),
    )
    gas = GasData(
        ["N2O5"],
        np.array([N2O5]),
        np.array([[gas_initial * N2O5]]),
        np.array([False]),
    )
    state = Aerosol(particles=particles, gas=gas, environment=env)
    gas_name = "gas:N2O5"
    water, dissolved, acid = [
        "particle:" + n for n in ("water", "uptake", "HNO3")
    ]
    n2o5_spec = {"molar_mass_kg_per_mol": N2O5, "elements": {"N": 2, "O": 5}}
    species = {
        gas_name: n2o5_spec,
        dissolved: n2o5_spec,
        water: {"molar_mass_kg_per_mol": WATER, "elements": {"H": 2, "O": 1}},
        acid: {
            "molar_mass_kg_per_mol": ACID,
            "elements": {"H": 1, "N": 1, "O": 3},
        },
    }
    reactions = [
        dict(
            id="hydrolysis",
            reactants={dissolved: 1, water: 1},
            products={acid: 2},
            orders={water: 0},
            rate=dict(type="constant", k=aqueous_rate, units="1/s"),
        )
    ]
    if gamma > 0:
        reactions.insert(
            0,
            dict(
                id="surface_uptake",
                reactants={gas_name: 1},
                products={dissolved: 1},
                rate=dict(type="reactive_uptake", gamma=gamma),
            ),
        )
    mechanism = ReactionMechanism.from_dict(
        dict(
            version=2,
            basis="mol/m^3-phase",
            species=species,
            reactions=reactions,
        )
    )
    bound = mechanism.bind_resolved(state, {water: 1, dissolved: 2, acid: 3})
    return (
        state,
        ResolvedChemistry(bound, ChemistrySolver(rtol=1e-8, atol=1e-18)),
        number,
    )


def replay(row, gamma, pressure, dt, aqueous_rate):
    """Share the gas between the two sequential particle updates."""
    init_g = pressure / (GAS_CONSTANT * row.temperature_k) * 1e-12
    states = [
        channel(row, m, g, pressure, init_g, aqueous_rate)
        for m, g in zip(["small", "other"], gamma, strict=True)
    ]
    gas_mass = init_g * N2O5
    inventories0 = np.array(
        [
            state.particles.masses[0, 0] * count / MW
            for state, _, count in states
        ]
    )
    seconds = 600.0
    for start in np.arange(0.0, seconds, dt):
        for state, process, _ in states:
            state.gas.concentration[0, 0] = gas_mass
            process.advance(state, time=start, time_step=dt, context=None)
            gas_mass = state.gas.concentration[0, 0]
    stocks = np.array(
        [
            state.particles.masses[0, 0] * count / MW
            for state, _, count in states
        ]
    )
    uptake = (stocks[:, 2] + stocks[:, 3] / 2).sum()
    acid = stocks[:, 3].sum()
    gas = gas_mass / N2O5
    n_error = abs(gas + uptake - init_g) / init_g
    h_error = abs((inventories0[:, 1] - stocks[:, 1]).sum() - acid / 2) / init_g
    if n_error > 1e-7 or h_error > 1e-7:
        raise ArithmeticError("Elemental budget failed")
    k = (
        np.sqrt(8 * GAS_CONSTANT * row.temperature_k / (np.pi * N2O5))
        / 4
        * 1e-6
        * np.dot(gamma, [row.wet_small_area, row.wet_other_area])
    )
    expected = -np.expm1(-k * seconds)
    actual = 1 - gas / init_g
    np.testing.assert_allclose(actual, expected, rtol=0.003, atol=1e-12)
    return dict(
        step_s=dt,
        duration_s=seconds,
        aqueous_processing_s=aqueous_rate,
        incremental_loss=actual,
        fixed_area_analytic_loss=expected,
        hno3_molecules_per_initial_n2o5=acid / init_g,
        native_small_loss_fraction=(
            (stocks[0, 2] + stocks[0, 3] / 2) / uptake if uptake else 0
        ),
        nitrogen_relative_error=n_error,
        water_budget_error_per_initial_n2o5=h_error,
    )


def main():
    """Check time refinement, hydrolysis and channel deletion in one window."""
    h = pd.read_csv(HERE / "hydration.csv").query("case=='aiomfac'")
    reference = h.iloc[np.argsort(h.wet_total_area.to_numpy())[len(h) // 2]]
    flights = pd.read_csv(HERE.parent / "flight-windows.csv")
    pressure = (
        100
        * flights[
            (flights.flight == reference.flight)
            & (flights.start_s == reference.start_s)
        ].pressure_hpa.item()
    )
    fits = pd.read_csv(HERE / "field-fits.csv").query(
        "case=='aiomfac' and basis=='wet'"
    )
    output = []
    for row in fits.itertuples():
        for step, k in [(60.0, 0.02), (30.0, 0.02), (60.0, 0.04)]:
            result = replay(
                reference, [row.gamma_small, row.gamma_other], pressure, step, k
            )
            result.update(
                model=row.model,
                flight=int(reference.flight),
                start_s=reference.start_s,
            )
            output.append(result)
    zero = replay(reference, [0.0, 0.0], pressure, 60.0, 0.02)
    output.append(
        dict(
            **zero,
            model="both channels off",
            flight=int(reference.flight),
            start_s=reference.start_s,
        )
    )
    pd.DataFrame(output).to_csv(HERE / "native-replay.csv", index=False)


if __name__ == "__main__":
    main()
