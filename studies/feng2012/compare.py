"""Compare AIOMFAC coexisting liquids to Feng et al. (2012), without fitting."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from particula.particles import ActivityFactory
from particula_aiomfac import AIOMFACModel, MixtureSpec
from particula_aiomfac._equilibrium import ActivityDomainError

COMPONENTS = ("water", "alcohol", "salt")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/thermodynamics"


def read_observations(kind):
    """Read measured dimensionless mass fractions at each temperature in K."""
    with (DATA / f"feng2012-isopropanol-{kind}.csv").open() as stream:
        return list(csv.DictReader(stream))


def cloud_salt_fraction(points, predicted_alcohol):
    """Interpolate observed cloud salt fraction at a given alcohol fraction."""
    coordinates = {}
    for point in points:
        coordinates.setdefault(float(point["alcohol"]), []).append(
            float(point["salt"])
        )
    x = np.array(sorted(coordinates))
    if not x[0] <= predicted_alcohol <= x[-1]:
        return None
    return float(
        np.interp(predicted_alcohol, x, [np.mean(coordinates[k]) for k in x])
    )


def chemical_potentials(result):
    """Neutral and salt formula-unit potentials divided by RT."""
    return np.log(
        np.concatenate(
            [result.neutral_activities, result.ion_activity_products], axis=-1
        )
    )


def stability_grid(model, temperature):
    """Sample the composition simplex independently of optimizer starts."""
    resolution = 100
    mass = (
        np.array(
            [
                (i, j, resolution - i - j)
                for i in range(1, resolution - 1)
                for j in range(1, resolution - i)
            ],
            dtype=float,
        )
        / resolution
    )
    chemistry = model.evaluate(mass, temperature)
    # One sulfate ion accounts for one analytical ammonium sulfate unit.
    units = chemistry.species_moles_per_kg_input[
        :, [0, 1, 2 + chemistry.ion_ids.index(261)]
    ]
    mole_fractions = units / units.sum(axis=1, keepdims=True)
    return mole_fractions, chemical_potentials(chemistry)


def compare():
    """Return every case, including single-phase predictions and failures."""
    mixture = MixtureSpec(
        {
            "water": {16: 1},
            "alcohol": {141: 2, 151: 1, 153: 1},
            "salt": {204: 2, 261: 1},
        }
    )
    activity = ActivityFactory().get_strategy(
        "aiomfac",
        {
            "mixture": mixture,
            "material_mapping": dict(enumerate(COMPONENTS)),
            "phase_mode": "stable",
        },
    )
    model = AIOMFACModel.prepare(mixture)
    grids = {}
    cloud_points = read_observations("binodal")
    records = []
    for measured in read_observations("tie-lines"):
        temperature = float(measured["temperature_k"])
        endpoints = np.array(
            [
                [float(measured[f"{phase}_{name}"]) for name in COMPONENTS]
                for phase in ("alpha", "beta")
            ]
        )
        feed = (endpoints / endpoints.sum(axis=1, keepdims=True)).mean(axis=0)
        case = {
            "temperature_k": temperature,
            "feed_mass_fractions": feed.tolist(),
            "measured_endpoints": endpoints.tolist(),
        }
        records.append(case)
        try:
            result = activity.phase_equilibrium(feed, temperature)
        except ActivityDomainError as exc:
            case.update(status="unresolved", detail=str(exc))
            continue
        case["minimum_tpd"] = float(result.minimum_tpd)
        if temperature not in grids:
            grids[temperature] = stability_grid(model, temperature)
        grid_x, grid_mu = grids[temperature]
        alpha_mu = chemical_potentials(result.alpha)
        case["min_grid_tpd"] = float(
            (grid_x * (grid_mu - alpha_mu)).sum(axis=1).min()
        )
        if result.beta_masses is None:
            case["status"] = "one_liquid"
            continue
        masses = np.stack([result.alpha_masses, result.beta_masses])
        prediction = masses / masses.sum(axis=1, keepdims=True)
        points = [
            row
            for row in cloud_points
            if float(row["temperature_k"]) == temperature
        ]
        boundary_error = []
        for _, alcohol, salt in prediction:
            measured_salt = cloud_salt_fraction(points, alcohol)
            boundary_error.append(
                None if measured_salt is None else salt - measured_salt
            )
        case.update(
            status="two_liquids",
            predicted_endpoints=prediction.tolist(),
            log_activity_mismatches=(
                alpha_mu - chemical_potentials(result.beta)
            ).tolist(),
            endpoint_mass_fraction_errors=(prediction - endpoints).tolist(),
            binodal_salt_mass_fraction_errors=boundary_error,
            phase_mass_fractions=masses.sum(axis=1).tolist(),
            max_feed_mass_balance_error=float(
                abs(masses.sum(axis=0) - feed).max()
            ),
            max_potential_difference_over_rt=float(
                result.chemical_potential_error
            ),
        )
    return {
        "components": COMPONENTS,
        "phases": ["alpha", "beta"],
        "cases": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    output = parser.parse_args().output
    report = json.dumps(compare(), indent=2, allow_nan=False) + "\n"
    if output is None:
        print(report, end="")
    else:
        output.write_text(report)
