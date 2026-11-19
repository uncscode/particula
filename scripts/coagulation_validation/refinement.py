"""Quantify uncertainty in the reference integrals for difficult pairs."""

import argparse
import json
from pathlib import Path

import numpy as np
from scripts.coagulation_validation.profiles import potential_profile
from scripts.coagulation_validation.reference import reference_capture


def refine(rows):
    """Increase multipole order, radial resolution and quadrature accuracy."""
    checks = []
    for row in rows:
        if row["radius_ratio"] not in (1, 31.9):
            continue
        if row["hamaker_over_kT"] not in (0.01, 16):
            continue
        if row["charge"] not in ([1, 1], [1, -1]):
            continue
        _, potential = potential_profile(
            1 / (1 + row["radius_ratio"]),
            row["charge"],
            row["hamaker_over_kT"],
            row["coupling"],
            row["relative_permittivity"],
            resolution=512,
            order=224,
        )
        result = reference_capture(potential, resolution=192, tolerance=2e-8)
        change = np.array(result) / row["reference"] - 1
        checks.append(
            dict(
                parameters=row, refined=result, relative_change=change.tolist()
            )
        )
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    checks = refine(json.loads(args.reference.read_text()))
    args.output.write_text(json.dumps(checks, indent=2) + "\n")
    if max(abs(x) for r in checks for x in r["relative_change"]) > 0.001:
        raise SystemExit("reference refinement exceeds 0.1%")
