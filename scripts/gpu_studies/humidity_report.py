"""Collect the complete measured BAT campaign for publication and replotting."""

import argparse
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scripts.gpu_studies.humidity_references import errors


def _json(path):
    return json.loads(path.read_text())


def collect(output):
    """Recheck histories, reference convergence and required timing records."""
    refs = output / "refs"
    reference_records = _json(refs / "references.json")
    reference_steps = {factor: n for factor, n, _ in reference_records}
    cases = []
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
                folder = output / stem
                case = _json(folder / "summary.json")
                data = np.load(folder / "cached" / f"{stem}.npy")
                if data.shape != (66, boxes, 3) or not np.all(
                    np.isfinite(data)
                ):
                    raise ValueError(f"Invalid history: {stem}")
                measured = errors(data, reference)
                np.testing.assert_array_equal(
                    measured, case.pop("errors_percent")
                )
                passed = bool(np.all(measured <= [0.1, 0.5, 0.5]))
                if case["passed"] != passed or (steps == 16 and not passed):
                    raise ValueError(f"Accuracy gate failed: {stem}")
                case["maximum_errors_percent"] = measured.max(axis=0).tolist()
                if passed:
                    samples = [
                        _json(folder / "warm" / f"{stem}-{i}.json")
                        for i in range(10)
                    ]
                    case["warmups"] = samples[:3]
                    case["repetitions"] = samples[3:]
                cases.append(case)
    return cases, reference_records


def metadata():
    """Record revision, executable hashes and numerical environment."""
    sources = [
        *(
            Path("scripts/gpu_studies") / f"humidity_{name}.py"
            for name in ("campaign", "matrix", "references", "restart")
        ),
        Path("docs/examples/thermodynamics/activity/bat_humidity_pulse.py"),
    ]
    return dict(
        revision=subprocess.check_output(  # noqa: S603, S607
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            text=True,
        ).strip(),
        sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources)
        },
        python=platform.python_version(),
        versions={
            name: version(name)
            for name in ("warp-lang", "numpy", "scipy", "matplotlib")
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    cases, references = collect(args.output)
    rows = np.load(args.output / "restart/restart-128-captured-False.npy")
    report = dict(
        environment=_json(args.output / "environment.json"),
        cases=cases,
        references=references,
        radius_um=(1e6 * rows[:, [0, 63, 127], 0]).tolist(),
        wetting_mg_m3=(1e6 * (rows[32, :, 2] - rows[0, :, 2])).tolist(),
        drying_loss_mg_m3=(1e6 * (rows[33, :, 2] - rows[65, :, 2])).tolist(),
    )
    args.report.write_text(json.dumps(report, indent=1) + "\n")
