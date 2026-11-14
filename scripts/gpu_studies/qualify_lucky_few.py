"""Run a fixed CUDA numerical qualification of both ion-rate assumptions.

This is a validation campaign, not the published example. Each comparison has
its own conservative 95% interval; no joint 95% claim over the matrix is made.
Run from the repository root with the existing development interpreter.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scripts.gpu_studies.lucky_few import (
    SurvivalConfig,
    difference_interval,
    run_survival,
)


def _json_default(value):
    """Serialize reduced NumPy diagnostics without saving particle states."""
    return value.tolist() if isinstance(value, np.ndarray) else value.item()


def qualify(output):
    """Freeze settings before execution and preserve every completed case."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    base = SurvivalConfig(boxes=8192, root_seed=320)
    tolerance = 0.01
    variants = {
        "baseline": {},
        "half-step": {"time_step": 0.5},
        "reversed-order": {"coagulation_first": True},
        "half-cleanup": {"cleanup_seconds": 5.0},
        "double-capacity": {"capacity": 256},
    }
    specification = {
        "base": asdict(base),
        "ion_production_cm3_s": [2, 10],
        "omission_policies": ["lower", "upper"],
        "variants": variants,
        "absolute_probability_tolerance": tolerance,
        "individual_comparison_confidence": 0.95,
        "maximum_baseline_interval_width": 0.01,
        "stop_when_resolved": True,
        "git_revision": subprocess.check_output(  # noqa: S603 - fixed Git argv
            [shutil.which("git") or "/usr/bin/git", "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "helper_sha256": hashlib.sha256(
            Path("scripts/gpu_studies/lucky_few.py").read_bytes()
        ).hexdigest(),
    }
    (output / "specification.json").write_text(
        json.dumps(specification, indent=2)
    )
    comparisons = []
    for q in (2, 10):
        for policy in ("lower", "upper"):
            reference = None
            for name, changes in variants.items():
                case = f"q{q}-{policy}-{name}"
                config = replace(
                    base, ion_pairs_cm3_s=q, omitted=policy, **changes
                )
                print("START", case, flush=True)
                result = run_survival(config, stop_when_resolved=True)
                (output / f"{case}.json").write_text(
                    json.dumps(result, default=_json_default)
                )
                reached = int(result["outcomes"][-1, 0])
                print(
                    "DONE",
                    case,
                    result["outcomes"][-1].tolist(),
                    "last_time",
                    result["time"][-1],
                    "mass",
                    result["maximum_mass_residual"],
                    "compute",
                    result["synchronized_compute_seconds"],
                    flush=True,
                )
                if reference is None:
                    reference = result
                    low, high, _ = result["success_interval95"]
                    comparisons.append(
                        {
                            "case": case,
                            "baseline_interval_width": high - low,
                            "passed": high - low <= 0.01,
                        }
                    )
                else:
                    interval = difference_interval(
                        int(reference["outcomes"][-1, 0]),
                        base.boxes,
                        reached,
                        base.boxes,
                    )
                    comparisons.append(
                        {
                            "case": case,
                            "difference_interval95": interval,
                            "passed": interval[0] >= -tolerance
                            and interval[1] <= tolerance,
                        }
                    )
                (output / "comparisons.json").write_text(
                    json.dumps(comparisons, indent=2)
                )
                print("CHECK", comparisons[-1], flush=True)
    if not all(row["passed"] for row in comparisons):
        raise RuntimeError("Qualification unresolved; inspect all intervals")
    print("All predeclared numerical checks passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    qualify(parser.parse_args().output)
