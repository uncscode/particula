"""Reproduce predeclared environmental timestep checks on CUDA.

Run ``python -m scripts.gpu_studies.qualify_environment dry output.json`` or
replace ``dry`` with ``humidity``. These reduced ensembles qualify the timestep,
not the final population size or atmospheric validity. Neither comparison is a
CPU/GPU speed benchmark.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import t
from scripts.gpu_studies.dry_environment_sweep import (
    DryEnvironmentSweep,
    dry_labels,
    run_dry_sweep,
)
from scripts.gpu_studies.environment_sweep import (
    EnvironmentSweep,
    run_environment_sweep,
)


def qualify_dry():
    """Require each individual paired-box 95% interval inside +/-2% of N0."""
    config = DryEnvironmentSweep(boxes_per_condition=64, seed=513)
    coarse, a = run_dry_sweep(config)
    fine, b = run_dry_sweep(replace(config, time_step=0.25))
    differences = (
        b.particles.concentration.sum(axis=1)
        - a.particles.concentration.sum(axis=1)
    ).reshape(19, -1) / config.particles_per_box
    means = differences.mean(axis=1)
    margin = t.ppf(0.975, 63) * differences.std(axis=1, ddof=1) / np.sqrt(64)
    passed = np.abs(means) + margin <= 0.02
    return {
        "criterion": (
            "Individual 95% difference intervals inside +/-0.02 "
            "of N0; not simultaneous coverage"
        ),
        "seed": 513,
        "boxes_per_condition": 64,
        "duration": 1000,
        "timesteps": [0.5, 0.25],
        "passed": bool(passed.all()),
        "rows": [
            {
                "label": label,
                "mean_difference": float(m),
                "low": float(m - e),
                "high": float(m + e),
                "passed": bool(ok),
            }
            for label, m, e, ok in zip(
                dry_labels(), means, margin, passed, strict=True
            )
        ],
        "coarse_compute_seconds": coarse["compute_seconds"],
        "fine_compute_seconds": fine["compute_seconds"],
    }


def qualify_humidity():
    """Check water/vapor and stochastic population sensitivity separately."""
    config = EnvironmentSweep(boxes_per_condition=256, ions=True, seed=413)
    coarse, a = run_environment_sweep(config)
    fine, b = run_environment_sweep(replace(config, time_step=5e-7))
    water = np.max(
        np.abs(
            np.array([r["water_mass"] for r in coarse["history"]])
            / np.array([r["water_mass"] for r in fine["history"]])
            - 1
        )
    )
    vapor = np.max(np.abs(a.gas.concentration / b.gas.concentration - 1))
    errors = {}
    for key in (
        "number",
        "attachment_positive",
        "attachment_negative",
        "recombination",
    ):
        denominator = config.boxes_per_condition * (
            config.particles_per_box
            if key == "number"
            else config.ion_pairs_per_box
        )
        errors[key] = float(
            np.max(
                np.abs(
                    np.array(coarse["history"][-1][key])
                    - np.array(fine["history"][-1][key])
                )
            )
            / denominator
        )
    return {
        "criterion": (
            "Water <=1%, vapor <=0.1%; absolute population differences "
            "<=2% of initial aerosol or per-sign ion count; "
            "sensitivity, not confidence intervals"
        ),
        "seed": 413,
        "boxes_per_condition": 256,
        "duration": config.duration,
        "timesteps": [1e-6, 5e-7],
        "water_relative": float(water),
        "vapor_relative": float(vapor),
        "population_differences_fraction": errors,
        "passed": bool(
            water <= 0.01 and vapor <= 0.001 and max(errors.values()) <= 0.02
        ),
    }


def main():
    """Save the observed result even when a predeclared criterion fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("dry", "humidity"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = qualify_dry() if args.study == "dry" else qualify_humidity()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("Predeclared timestep criterion unresolved")


if __name__ == "__main__":
    main()
