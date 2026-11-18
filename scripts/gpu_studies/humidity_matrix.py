"""Measure isolated BAT pulse workloads at comparable accuracy."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
from scripts.gpu_studies.humidity_references import errors
from scripts.gpu_studies.humidity_report import metadata


def _execute(command, case, folder, repeats, environment):
    start = perf_counter()
    with (case / f"{folder}.log").open("w") as log:
        subprocess.run(  # noqa: S603 -- fixed local study command
            [
                *command,
                str(case / folder),
                "--repetitions",
                str(repeats),
            ],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return perf_counter() - start


def main():
    """Screen timesteps and time each accepted configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    provenance = output / "environment.json"
    environment = metadata()
    if (
        provenance.exists()
        and json.loads(provenance.read_text()) != environment
    ):
        raise ValueError(
            "Source or environment changed. Use a fresh output directory."
        )
    provenance.write_text(json.dumps(environment, indent=2))
    refs = output / "refs"
    reference_steps = {
        f: n for f, n, _ in json.loads((refs / "references.json").read_text())
    }
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for boxes in (1, 8, 32, 128):
        factors = np.geomspace(0.5, 2, boxes) if boxes > 1 else [1.0]
        reference = np.concatenate(
            [
                np.load(refs / f"{f.hex()}-{reference_steps[f]}.npy")
                for f in factors
            ],
            axis=1,
        )
        for mode in ("cpu", "prepared", "captured"):
            for steps in (8, 16):
                stem = f"{mode}-{boxes}-{steps}"
                case = output / stem
                case.mkdir(exist_ok=True)
                summary = case / "summary.json"
                if summary.exists():
                    continue
                command = [
                    sys.executable,
                    "-m",
                    "scripts.gpu_studies.humidity_campaign",
                    mode,
                    str(boxes),
                    str(steps),
                ]

                with tempfile.TemporaryDirectory(
                    prefix="bat-compiler-"
                ) as cache:
                    settings = dict(
                        env, WARP_CACHE_PATH=cache, CUDA_CACHE_PATH=cache
                    )
                    first = _execute(command, case, "first", 1, settings)
                    second = _execute(command, case, "cached", 1, settings)
                    data = np.load(case / "cached" / f"{stem}.npy")
                    metrics = errors(data, reference)
                    passed = bool(np.all(metrics <= [0.1, 0.5, 0.5]))
                    if passed:
                        _execute(command, case, "warm", 10, settings)
                        np.testing.assert_array_equal(
                            data, np.load(case / "warm" / f"{stem}.npy")
                        )
                summary.write_text(
                    json.dumps(
                        dict(
                            boxes=boxes,
                            mode=mode,
                            steps=steps,
                            passed=passed,
                            errors_percent=metrics.tolist(),
                            cold_process_seconds=first,
                            cached_process_seconds=second,
                        )
                    )
                )
                print(stem, passed, flush=True)
            if not any(
                json.loads(
                    (
                        output / f"{mode}-{boxes}-{n}" / "summary.json"
                    ).read_text()
                )["passed"]
                for n in (8, 16)
            ):
                raise ValueError(
                    f"No accurate configuration for {boxes}, {mode}"
                )


if __name__ == "__main__":
    main()
