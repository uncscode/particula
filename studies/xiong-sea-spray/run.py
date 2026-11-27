"""Replay the Xiong sea-spray study in a new local directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

STUDY = Path(__file__).resolve().parent
CHECKOUT = STUDY.parents[1]
PHASE = "aiomfac/coating/parameter-fit"
THERMO = "aiomfac/thermo"

INPUTS = {
    "reference": {
        "xiong": (
            "endmembers",
            "table-s1",
            "size-rates",
            "fig5-log",
            "fig5-sulfate",
        ),
        "rosati": ("spectra",),
    },
    "chemistry": {
        "xiong": (
            "endmembers",
            "table-s1",
            "fig3-100nm",
            "figs19-ic-peaks",
            "fig4b5c-100nm",
            "fig5-log",
            "fig5-sulfate",
            "text-100nm",
        ),
        "rosati": ("spectra",),
        "aiomfac/low_so2": ("s20",),
    },
    "coating": {"aiomfac/coating": ("figure6a", "figure6b", "figure6-s2")},
    "phase": {PHASE: ("binary-water", "phase-data")},
}


def stage_commands(stages, controls):
    if "reference" in stages:
        for name, device, power in [
            ("reference", "cpu", 4),
            ("pilot", "cuda:0", 8),
            ("cuda-refined", "cuda:0", 15),
        ]:
            yield [
                "inverse.py",
                "--stage",
                name,
                "--device",
                device,
                "--power",
                str(power),
            ]
        for name, slabs in (("joint-pilot", 8), ("joint", 64)):
            yield ["joint.py", "--stage", name, "--slabs", str(slabs)]
        for name in ("population", "challenges", "exposure", "size_bounds"):
            yield [f"{name}.py"]
    if "chemistry" in stages:
        yield [
            f"{THERMO}/solve.py",
            "--output",
            "pilot",
            "--device",
            "cpu",
            "--step",
            ".5",
            "--steps-water",
            "32",
        ]
        full = ["--device", "cuda:0", "--step", ".01", "--trace"]
        yield [f"{THERMO}/solve.py", "--output", "liquid", *full]
        yield [
            f"{THERMO}/solve.py",
            "--output",
            "neutral-liquid",
            "--neutral",
            ".25",
            ".5",
            ".75",
            "1",
            *full,
        ]
        yield [f"{THERMO}/halite.py"]
        yield [
            f"{THERMO}/halite.py",
            "--source",
            "neutral-liquid",
            "--output",
            "neutral-halite",
        ]
        yield [f"{THERMO}/reference.py"]
        yield [f"{THERMO}/verify.py"]
        yield ["aiomfac/pathways.py"]
        manifest = json.loads((STUDY / "models/aiomfac/stages.json").read_text())
        cases = manifest if controls else ("primary", "retention", "blank", "eq8")
        for case in cases:
            yield ["aiomfac/fit.py", *manifest[case].split()]
        for name in ("evaluate", "numerical", "crossdose"):
            yield [f"aiomfac/{name}.py"]
        if controls:
            yield ["aiomfac/size_challenge.py"]
        yield [
            "aiomfac/native_check.py",
            "--device",
            "cpu",
            "--tag",
            "cpu",
            "--names",
            "direct,neutral",
            "--diameters",
            "200",
        ]
        yield [
            "aiomfac/native_check.py",
            "--device",
            "cuda:0",
            "--tag",
            "cuda",
            "--names",
            "direct,neutral,siv_lag,parallel_neutral,retention,retention_neutral",
        ]
    if "coating" in stages:
        yield ["aiomfac/coating/predict.py"]
    if "phase" in stages:
        for command in (
            "calibrate",
            "refine",
            "refine --extend",
            "joint_profile",
            "transfer",
            "chamber_path",
            "challenges",
            "reciprocal_check",
            "audit",
        ):
            name, *flags = command.split()
            yield [f"{PHASE}/{name}.py", *flags]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        required=True,
        help="Observation tree from data/prepare.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for inputs, source, logs and model results",
    )
    parser.add_argument(
        "--stage", choices=(*INPUTS, "all"), nargs="+", default=["chemistry"]
    )
    parser.add_argument(
        "--controls",
        action="store_true",
        help="Run all chemistry robustness profiles",
    )
    args = parser.parse_args()
    stages = set(INPUTS if "all" in args.stage else args.stage)
    if "coating" in stages:
        stages.add("chemistry")
    inputs = args.inputs.resolve()
    output = args.output.resolve()
    files = sorted(
        {
            Path(folder) / f"{name}.csv"
            for stage in stages
            for folder, names in INPUTS[stage].items()
            for name in names
        }
    )
    missing = [str(path) for path in files if not (inputs / path).is_file()]
    if missing:
        parser.error("Missing prepared observations: " + ", ".join(missing))
    if output.is_relative_to(STUDY / "models"):
        parser.error("Choose an output directory outside the model source tree")
    if output.exists():
        parser.error("Select a new --output directory to protect earlier runs")
    output.mkdir(parents=True)
    source_dir = STUDY / "models"
    for source in sorted(
        [*source_dir.rglob("*.py"), source_dir / "aiomfac/stages.json"]
    ):
        destination = output / source.relative_to(source_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    digests = {}
    for path in files:
        (output / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inputs / path, output / path)
        digests[path.as_posix()] = hashlib.sha256(
            (output / path).read_bytes()
        ).hexdigest()
    env = os.environ.copy()
    sources = [str(CHECKOUT), str(CHECKOUT / "packages/particula-aiomfac/src")]
    if env.get("PYTHONPATH"):
        sources.append(env["PYTHONPATH"])
    env.update(
        PYTHONPATH=os.pathsep.join(sources),
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="1",
        JAX_PLATFORMS="cpu",
    )
    logdir = output / "logs"
    logdir.mkdir()
    revision = subprocess.run(
        ["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
    )
    manifest = {
        "stages": sorted(stages),
        "controls": args.controls or "all" in args.stage,
        "checkout": revision.stdout.strip() or None,
        "python": sys.version,
        "input_sha256": digests,
        "runs": [],
        "complete": False,
    }
    record = output / "run-manifest.json"
    record.write_text(json.dumps(manifest, indent=2) + "\n")
    for i, command in enumerate(stage_commands(stages, manifest["controls"]), 1):
        item = {
            "command": command,
            "log": f"logs/{i:02d}-{Path(command[0]).stem}.log",
        }
        print(" ".join(command), flush=True)
        start = time.monotonic()
        with (output / item["log"]).open("w") as log:
            proc = subprocess.run(
                [sys.executable, *command],
                cwd=output,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        item.update(
            seconds=round(time.monotonic() - start, 3),
            exit_code=proc.returncode,
        )
        manifest["runs"].append(item)
        record.write_text(json.dumps(manifest, indent=2) + "\n")
        if proc.returncode:
            raise SystemExit(f"Run failed. See {output / item['log']}")
    manifest["complete"] = True
    record.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
