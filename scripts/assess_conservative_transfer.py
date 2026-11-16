"""Screen MPRK22(1) on closed linear reservoir transfer, including zero lanes.

For p_ij(y)=A_ij*y_j, apply the two linear solves in equation (3) of
Kopecz and Meister, https://arxiv.org/abs/1702.04589. Ratios with zero donors
are evaluated by their linear-rate limit. This extension is specific to linear
transfer; it does not establish arbitrary chemistry invariants or zero handling.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm


def patankar_step(matrix, state, step):
    """Advance a linear closed transfer system by conservative MPRK22(1)."""
    identity = np.eye(state.size)
    stage = np.linalg.solve(identity - step * matrix, state)
    weight = np.ones_like(state)
    nonzero = stage > 0
    weight[nonzero] = (state[nonzero] + stage[nonzero]) / (2 * stage[nonzero])
    if np.any((~nonzero) & (state != 0)):
        raise ValueError("zero denominator has a nonzero original donor")
    return np.linalg.solve(identity - step * matrix * weight[None, :], state)


def assess():
    """Retain errors, zero species, conservation and large-step failures."""
    rows = []
    for name, forward, backward in (
        ("smooth", 0.7, 0.2),
        ("stiff", 1000.0, 1.0),
        ("depletion", 10.0, 0.0),
    ):
        matrix = np.array(
            [
                [-forward, backward, 0.0],
                [forward, -backward, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        initial = np.array([1.0, 0.0, 0.0])
        previous = None
        for steps in (8, 16, 32, 64, 128, 256):
            state = initial.copy()
            errors, budgets, minima = [], [], []
            for i in range(steps):
                state = patankar_step(matrix, state, 1 / steps)
                exact = expm(matrix * ((i + 1) / steps)) @ initial
                errors.append(np.max(np.abs(state - exact)))
                budgets.append(abs(state.sum() - 1))
                minima.append(state.min())
            error = float(max(errors))
            rows.append(
                {
                    "case": name,
                    "steps": steps,
                    "error": error,
                    "budget_error": float(max(budgets)),
                    "minimum": float(min(minima)),
                    "observed_order": None
                    if previous is None
                    else float(np.log2(previous / error)),
                    "linear_solves": 2 * steps,
                }
            )
            previous = error
    return rows


def main():
    """Write every result to the requested report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(assess(), indent=2) + "\n")


if __name__ == "__main__":
    main()
