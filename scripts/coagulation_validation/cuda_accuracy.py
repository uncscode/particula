"""Qualify CUDA pair rates against adaptive orbit and continuum integrals."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp
from particula.dynamics.coagulation.van_der_waals_kernel import COULOMB_LENGTH_T
from particula.gpu.dynamics.van_der_waals import (
    CaptureData,
    capture_pair_wp,
    prepare_capture,
)
from particula.util.constants import BOLTZMANN_CONSTANT


@wp.kernel
def _evaluate(
    data: CaptureData,
    inputs: wp.array2d(dtype=wp.float64),
    output: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    output[i] = capture_pair_wp(
        data,
        inputs[i, 0],
        inputs[i, 1],
        inputs[i, 2],
        inputs[i, 3],
        inputs[i, 4],
        inputs[i, 5],
        inputs[i, 6],
        inputs[i, 7],
    )


def run(rows):
    """Compare the dimensional CUDA rate at three diffusive Knudsen numbers."""
    cases = []
    for epsilon, a in sorted(
        {(x["relative_permittivity"], x["hamaker_over_kT"]) for x in rows}
    ):
        t = 298.15
        thermal = BOLTZMANN_CONSTANT * t
        inputs = []
        results = []
        for row in rows:
            if (row["relative_permittivity"], row["hamaker_over_kT"]) != (
                epsilon,
                a,
            ):
                continue
            ec, ef = row["reference"]
            r = COULOMB_LENGTH_T / (row["coupling"] * t)
            ratio = row["radius_ratio"]
            r1 = r / (1 + ratio)
            m1, m2 = np.array([r1, r - r1]) ** 3 * (4 * np.pi / 3 * 2000)
            mu = m1 * m2 / (m1 + m2)
            for q in (1e-5, 1.0, 1e5):
                drag = np.sqrt(thermal * mu) / (q * r)
                inputs.append(
                    [r1, r - r1, m1, m2, *row["charge"], thermal / drag, t]
                )

                # Independently transcribed dimensionless rational law.
                def h(x):
                    return (
                        x
                        * x
                        * (
                            4 * np.pi
                            + 25.836 * x
                            + np.sqrt(8 * np.pi) * 11.211 * x * x
                        )
                        / (1 + 3.502 * x + 7.211 * x * x + 11.211 * x**3)
                    )

                hard = drag * r**3 / mu * h(q)
                expected = drag * r**3 / mu * (ef * ef / ec) * h(q * ec / ef)
                results.append(
                    dict(
                        ratio=ratio,
                        charges=row["charge"],
                        coupling=row["coupling"],
                        epsilon=epsilon,
                        attraction=a,
                        q=q,
                        expected=expected,
                        hard=hard,
                    )
                )
        data = prepare_capture(a * thermal, epsilon, "cuda")
        values = wp.array(np.array(inputs), dtype=wp.float64, device="cuda")
        out = wp.empty(len(inputs), dtype=wp.float64, device="cuda")
        wp.launch(
            _evaluate, len(inputs), inputs=[data, values, out], device="cuda"
        )
        for item, actual in zip(results, out.numpy(), strict=True):
            item["cuda"] = float(actual)
            item["error_relative"] = (actual / item["expected"]) - 1
            item["error_normalized"] = (
                abs(actual - item["expected"]) / item["hard"]
            )
            item["passed"] = bool(
                actual >= 0
                and (
                    abs(item["error_relative"]) <= 0.01
                    if item["expected"] / item["hard"] >= 1e-8
                    else item["error_normalized"] <= 1e-10
                )
            )
            cases.append(item)
    return dict(
        device=wp.get_device("cuda").name, warp=wp.__version__, cases=cases
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    evidence = run(json.loads(args.reference.read_text()))
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    if not all(i["passed"] for i in evidence["cases"]):
        raise SystemExit("CUDA grid contains failed cases")
