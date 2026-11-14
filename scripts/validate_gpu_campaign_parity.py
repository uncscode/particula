"""Check complete CPU/CUDA state parity for every equivalent campaign case."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from particula._cpu_batch import detached
from scripts.gpu_studies.campaign import timing_cases
from scripts.gpu_studies.cases import build_case
from scripts.gpu_studies.validation import native_cpu_steps
from scripts.gpu_studies.worker import executable_hash


def validate_parity(destination):
    """Compare physical arrays after identical complete trajectories on CUDA."""
    import warp as wp

    if not wp.is_cuda_available():
        raise RuntimeError("Parity validation requires actual CUDA")
    rows = []
    for case in timing_cases():
        if case.family not in ("condensation", "dilution"):
            continue
        initial, process, processes = build_case(case)
        cpu = native_cpu_steps(case, detached(initial), processes)
        with process.prepare(
            initial,
            device="cuda:0",
            time_step=case.time_step,
            root_seed=case.seed,
        ) as simulation:
            simulation.capture()
            simulation.run(case.steps)
            gpu = simulation.snapshot()
        errors = {}
        for carrier, fields in (
            ("particles", ("masses", "concentration", "charge", "volume")),
            ("gas", ("concentration", "molar_mass", "partitioning")),
            ("environment", ("temperature", "pressure")),
        ):
            for field in fields:
                left = np.asarray(getattr(getattr(cpu, carrier), field))
                right = np.asarray(getattr(getattr(gpu, carrier), field))
                np.testing.assert_allclose(
                    right,
                    left,
                    rtol=2e-12,
                    atol=1e-30,
                    err_msg=f"{case.name}: {carrier}.{field}",
                )
                if left.dtype == bool:
                    error = float(np.any(left != right))
                else:
                    scale = np.maximum(np.abs(left), 1e-30)
                    error = float(np.max(np.abs(right - left) / scale))
                errors[f"{carrier}.{field}"] = error
        if cpu.gas.name != gpu.gas.name:
            raise AssertionError("Ordered gas names changed")
        rows.append({"case": asdict(case), "max_relative_errors": errors})
        print(f"PASS {case.name}: all physical arrays", flush=True)
    report = {
        "qualified": True,
        "executable_sha256": executable_hash(),
        "gpu": wp.get_device("cuda:0").name,
        "device": "cuda:0",
        "rtol": 2e-12,
        "atol": 1e-30,
        "scope": (
            "Complete particle/gas arrays and prescribed temperature/pressure; "
            "derived saturation is refreshed by CUDA condensation only. "
            "CPU condensation uses four equal substeps, matching CUDA."
        ),
        "cases": rows,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    """Run outside the timing campaign to avoid contention."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    validate_parity(args.output)


if __name__ == "__main__":
    main()
