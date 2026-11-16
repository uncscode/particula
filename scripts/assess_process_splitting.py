"""Measure splitting order independently of component integration error.

Two normalized reservoirs exchange material (0.7 and 0.2 /s); only the first
reservoir is removed at 0.4 /s. These operators do not commute. All component
maps and the combined reference use matrix exponentials. This tests the
scheduling mathematics; native scheduler qualification is a separate gate.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm


def assess():
    """Return trajectory errors and orders on a fixed refinement grid."""
    exchange = np.array([[-0.7, 0.2], [0.7, -0.2]])
    loss = np.diag([-0.4, 0.0])
    initial = np.array([1.0, 0.0])
    rows = []
    for method in ("sequential", "reversed", "strang"):
        previous = None
        for steps in (8, 16, 32, 64, 128):
            dt = 5.0 / steps
            a = expm(exchange * dt)
            b = expm(loss * dt)
            half = expm(exchange * dt / 2)
            maps = {
                "sequential": b @ a,
                "reversed": a @ b,
                "strang": half @ b @ half,
            }
            value = initial.copy()
            errors = []
            for step in range(steps):
                value = maps[method] @ value
                exact = expm((exchange + loss) * ((step + 1) * dt)) @ initial
                errors.append(np.abs(value - exact))
            error = float(np.max(errors))
            rows.append(
                {
                    "method": method,
                    "steps": steps,
                    "maximum_trajectory_error": error,
                    "observed_order": None
                    if previous is None
                    else float(np.log2(previous / error)),
                    "minimum_final_inventory": float(value.min()),
                }
            )
            previous = error
    return rows


def main():
    """Write all refinements without selecting only favorable results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(assess(), indent=2) + "\n")


if __name__ == "__main__":
    main()
