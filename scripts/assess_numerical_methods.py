"""Reproducible preliminary ODE screening, separate from aerosol qualification.

Linear reservoir transfer has a matrix-exponential reference independent of
integration. Robertson's dimensionless reaction benchmark is cross-checked
with tight Radau and BDF solves. Time is seconds; reservoir components are
normalized inventories. No production solver or GPU claim follows from this
screening alone. SciPy solver definitions:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
"""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.linalg import expm


class EvaluationLimitError(RuntimeError):
    """Keep difficult explicit benchmarks bounded and retain their failure."""


def cases():
    """Yield independent decay/exchange cases and a stiff reaction benchmark."""
    for name, forward, backward, duration in (
        ("smooth_exchange", 0.7, 0.2, 10.0),
        ("stiff_equilibrium", 1000.0, 1.0, 1.0),
        ("near_depletion", 10.0, 1e-8, 10.0),
    ):
        matrix = np.array([[-forward, backward], [forward, -backward]])
        yield {
            "name": name,
            "initial": np.array([1.0, 0.0]),
            "duration": duration,
            "rate": lambda t, y, matrix=matrix: matrix @ y,
            "exact": lambda times, matrix=matrix: np.column_stack(
                [expm(matrix * t) @ np.array([1.0, 0.0]) for t in times]
            ),
            "closed": True,
        }
    yield {
        "name": "decay",
        "initial": np.array([1.0]),
        "duration": 10.0,
        "rate": lambda t, y: -0.7 * y,
        "exact": lambda times: np.exp(-0.7 * times)[None, :],
        "closed": False,
    }

    def robertson(t, y):
        a = 0.04 * y[0]
        b = 1e4 * y[1] * y[2]
        c = 3e7 * y[1] ** 2
        return np.array([-a + b, a - b - c, c])

    yield {
        "name": "robertson",
        "initial": np.array([1.0, 0.0, 0.0]),
        "duration": 100.0,
        "rate": robertson,
        "exact": None,
        "closed": True,
    }


def solve(case, method, tolerance):
    """Measure solver work without extrapolating an incomplete trajectory."""
    evaluations = 0

    def rate(t, y):
        nonlocal evaluations
        evaluations += 1
        if evaluations > 200_000:
            raise EvaluationLimitError("200000 derivative evaluation limit")
        return case["rate"](t, y)

    start = perf_counter()
    solution = solve_ivp(
        rate,
        (0.0, case["duration"]),
        case["initial"],
        method=method,
        rtol=tolerance,
        atol=tolerance * 1e-3,
        dense_output=True,
    )
    elapsed = perf_counter() - start
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution, elapsed, evaluations


def assess():
    """Return screening rows and separately checked reference uncertainty."""
    rows, references = [], []
    for case in cases():
        times = np.r_[0.0, np.geomspace(1e-8, case["duration"], 300)]
        if case["exact"] is not None:
            reference = case["exact"](times)
            discrepancy = 0.0
            source = "analytic matrix exponential or scalar exponential"
        else:
            radau, _, _ = solve(case, "Radau", 1e-11)
            bdf, _, _ = solve(case, "BDF", 1e-12)
            reference = radau.sol(times)
            discrepancy = float(np.max(np.abs(reference - bdf.sol(times))))
            source = "Radau rtol=1e-11 versus BDF rtol=1e-12"
            if discrepancy >= 1e-9:
                raise RuntimeError("reference uncertainty exceeds 1% of 1e-7")
        references.append(
            {"case": case["name"], "source": source, "difference": discrepancy}
        )
        for method in ("RK23", "RK45", "Radau", "BDF"):
            for tolerance in (1e-3, 1e-5, 1e-7):
                row = {
                    "case": case["name"],
                    "method": method,
                    "rtol": tolerance,
                }
                try:
                    solution, elapsed, evaluations = solve(
                        case, method, tolerance
                    )
                    values = solution.sol(times)
                    row.update(
                        status="success",
                        max_error_by_species=np.max(
                            np.abs(values - reference), axis=1
                        ).tolist(),
                        minimum_by_species=values.min(axis=1).tolist(),
                        closed_budget_error=(
                            float(np.max(np.abs(values.sum(axis=0) - 1)))
                            if case["closed"]
                            else None
                        ),
                        accepted_steps=len(solution.t) - 1,
                        derivative_calls=evaluations,
                        jacobian_evaluations=solution.njev,
                        lu_factorizations=solution.nlu,
                        seconds=elapsed,
                        # Public solve_ivp does not expose these counts.
                        rejected_steps=None,
                        peak_scratch_bytes=None,
                    )
                except (
                    EvaluationLimitError,
                    RuntimeError,
                    FloatingPointError,
                ) as exc:
                    row.update(status="failed", reason=str(exc))
                rows.append(row)
    return {"references": references, "runs": rows}


def main():
    """Write a reproducible JSON report to the selected destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = assess()
    report["provenance"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "revision": subprocess.check_output(  # noqa: S603 - fixed Git query
            [shutil.which("git") or "/usr/bin/git", "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "backend": "CPU float64; SciPy screening only",
        "error_scale": "unit initial total; per-species errors retained",
        "missing": [
            "native condensation",
            "CUDA",
            "splitting",
            "Patankar",
            "local estimator reliability",
            "rejection counts",
            "memory",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
