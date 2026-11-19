"""Compare production limits with refined adaptive orbit capture."""

import argparse
import json
from pathlib import Path

import numpy as np
from particula.dynamics.coagulation.van_der_waals_kernel import (
    COULOMB_LENGTH_T,
    pair_enhancements,
)
from particula.util.constants import BOLTZMANN_CONSTANT
from scripts.coagulation_validation.profiles import potential_profile
from scripts.coagulation_validation.reference import reference_capture


def evaluate(case):
    """Compare both transport limits with an adaptive orbital solve."""
    ratio, charges, attraction, coupling, epsilon = case
    temperature = 298.15
    contact = COULOMB_LENGTH_T / (coupling * temperature)
    fraction = 1 / (1 + ratio)
    production = pair_enhancements(
        fraction * contact,
        contact - fraction * contact,
        *charges,
        temperature,
        attraction * BOLTZMANN_CONSTANT * temperature,
        epsilon,
    )
    _, profile = potential_profile(
        fraction, charges, attraction, coupling, epsilon
    )
    reference = reference_capture(profile, resolution=96)
    return dict(
        radius_ratio=ratio,
        charge=charges,
        hamaker_over_kT=attraction,
        coupling=coupling,
        relative_permittivity=epsilon,
        reference=reference,
        production=production,
        relative_error=(np.asarray(production) / reference - 1).tolist(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = []
    for ratio in (1.0, 3.17, 5.0, 31.9):
        for charges, attraction, coupling, epsilon in [
            ((0, 0), 0.0, 1.0, 1.0),
            ((0, 0), 0.001, 1.0, 5.9),
            ((0, 0), 200.0, 1.0, 100.0),
            ((1, 0), 0.0, 6.0, 5.9),
            ((1, 0), 16.0, 3.0, 5.9),
            ((0, 1), 16.0, 3.0, 5.9),
            ((1, 1), 16.0, 2.5, 5.9),
            ((1, -1), 16.0, 0.5, 5.9),
            ((1, 1), 0.0, 20.0, 1.0),
            ((1, 1), 0.01, 59.0, 100.0),
            ((1, 0), 0.0, 0.05, 100.0),
        ]:
            row = evaluate((ratio, charges, attraction, coupling, epsilon))
            records.append(row)
            print(ratio, charges, attraction, row["relative_error"], flush=True)
            args.output.write_text(json.dumps(records, indent=2) + "\n")
