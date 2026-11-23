"""Compare two carbonate measurement series at 298.15 K."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from particula_aiomfac import AIOMFACModel, MixtureSpec
from particula_aiomfac.phases import component_molar_masses
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
MIXTURE = MixtureSpec(
    {
        "water": {16: 1},
        "CO2": {173: 1},
        "NaCl": {202: 1, 242: 1},
        "NaHCO3": {202: 1, 250: 1},
    }
)


def _rows(name):
    with (HERE / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _evaluate(model, nacl, carbon):
    # Added H2O sets the experiment's one kg basis. Zero analytical NaHCO3
    # declares the pinned backend's carbonate chemistry on the same mixture.
    mw = component_molar_masses(model)
    masses = [1, carbon * mw[1], nacl * mw[2], 0.0]
    state = model.evaluate(masses, 298.15, diagnostics=True)
    return state, state.species_moles_per_kg_input * sum(masses)


def _conditional_pka(model, salt, co2_molality):
    # Match free CO2 in the cell, which differs from the total titrated carbon.
    def dissolved(total):
        _, species = _evaluate(model, salt, total)
        return species[1] / (species[0] * model.neutral_mw[0])

    feed = brentq(
        lambda total: dissolved(total) - co2_molality,
        0.8 * co2_molality,
        1.2 * co2_molality,
    )
    state, species = _evaluate(model, salt, feed)
    molality = species / (species[0] * model.neutral_mw[0])
    h, hc = [len(model.neutrals) + model.ions.index(j) for j in (205, 250)]
    return -np.log10(molality[h] * molality[hc] / molality[1]), state


def compare():
    """Hold CO2 pressure fixed for solubility and match the ionization cell."""
    model = AIOMFACModel.prepare(MIXTURE, carbonate_model="gibbs_consistent")
    pinned = AIOMFACModel.prepare(MIXTURE, carbonate_model="pinned")
    data = _rows("solubility.csv")
    anchor = 10 ** -float(data[0]["smoothed_minus_log10_s"])
    initial, _ = _evaluate(model, 0.0, anchor)
    activity = initial.neutral_activities[model.co2]
    solubility, ionization = [], []
    max_mass_action = max_charge = max_carbon = max_pressure = 0.0
    for row in data:
        salt = float(row["nacl_mol_per_kg_water"])
        observed = 10 ** -float(row["smoothed_minus_log10_s"])

        def residual(total, salt=salt):
            state, _ = _evaluate(model, salt, total)
            return np.log(state.neutral_activities[model.co2] / activity)

        predicted = brentq(residual, 0.001, 0.07, xtol=1e-13)
        state, species = _evaluate(model, salt, predicted)
        summed = species[1] + sum(
            species[len(model.neutrals) + model.ions.index(j)]
            for j in (250, 262)
        )
        max_carbon = max(max_carbon, abs(summed / predicted - 1))
        max_pressure = max(max_pressure, abs(residual(predicted)))
        max_mass_action = max(
            max_mass_action, state.diagnostics["equilibrium_residuals"][0]
        )
        max_charge = max(
            max_charge,
            abs(state.ion_molalities @ model.charge)
            / (state.ion_molalities @ abs(model.charge)),
        )
        solubility.append(
            dict(
                nacl_mol_kg=salt,
                observed_mol_kg=observed,
                predicted_mol_kg=predicted,
                relative_difference_pct=100 * (predicted / observed - 1),
                dilute_co2_analytical_mol_kg=anchor
                * np.exp(-2 * (0.100 - 0.005) * salt),
            )
        )
    for row in _rows("ionization.csv"):
        salt = float(row["nacl_mol_per_kg_water"])
        # Table I gives -log10(m_CO2) = 1.7440 in the 0.1 mol/kg cell at
        # 25 °C. The authors publish full cell conditions only for that series.
        dissolved = 10**-1.7440
        predicted, state = _conditional_pka(model, salt, dissolved)
        legacy, _ = _conditional_pka(pinned, salt, dissolved)
        sensitivity = [
            _conditional_pka(model, salt, m)[0] for m in (0.005, 0.030)
        ]
        observed = float(row["smoothed_pk1"]) + float(
            row["obs_minus_smoothed_pk1"]
        )
        ionization.append(
            dict(
                nacl_mol_kg=salt,
                observed_pk1=observed,
                matched_cell=salt == 0.1,
                predicted_pk1=predicted,
                pinned_pk1=legacy,
                difference_pk1=predicted - observed,
                sensitivity_pk1=max(abs(x - predicted) for x in sensitivity),
            )
        )
    error = np.array([row["relative_difference_pct"] for row in solubility[1:]])
    return dict(
        temperature_k=298.15,
        co2_pressure_atm=1.0,
        solvent_basis="kg added water",
        pure_water_carbon_anchor_mol_kg=anchor,
        solubility=solubility,
        ionization=ionization,
        held_out_solubility_relative_rmse_pct=np.sqrt(np.mean(error**2)),
        max_log_pressure_residual=max_pressure,
        max_relative_carbon_residual=max_carbon,
        max_normalized_charge_residual=max_charge,
        max_log_mass_action_residual=max_mass_action,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    target = parser.parse_args().output
    report = json.dumps(compare(), indent=2, allow_nan=False) + "\n"
    if target is None:
        print(report, end="")
    else:
        target.write_text(report, encoding="utf-8")
