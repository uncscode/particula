"""Check a fresh public-index installation, or a candidate against its public peer."""

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import time
import tempfile
import venv


def measure(base_version, backend_version):
    """Exercise the installed adapter and the pinned mixed reference state."""
    from particula.particles.activity_factories import ActivityFactory
    from particula_aiomfac import MixtureSpec

    versions = {
        name: importlib.metadata.version(name)
        for name in ("particula", "particula-aiomfac")
    }
    if base_version and versions["particula"] != base_version:
        raise ValueError(f"Unexpected base version: {versions}")
    if backend_version and versions["particula-aiomfac"] != backend_version:
        raise ValueError(f"Unexpected backend version: {versions}")
    if "aiomfac" not in importlib.metadata.metadata("particula").get_all(
        "Provides-Extra", []
    ):
        raise ValueError("Installed base does not declare the aiomfac extra")
    groups = {
        "water": {16: 1},
        "vanillic_acid": {9: 3, 10: 2, 17: 1, 24: 1, 137: 1},
        "butanediol": {142: 2, 150: 2, 153: 2},
        "NaCl": {202: 1, 242: 1},
    }
    activity = ActivityFactory().get_strategy(
        "aiomfac",
        {
            "mixture": MixtureSpec(groups),
            "material_mapping": dict(enumerate(groups)),
        },
    )
    result = activity.prepared.evaluate([0.977, 0.001, 0.002, 0.020], 298.0)
    measured = {
        "water_activity": float(result.neutral_activities[0]),
        "NaCl_mean_molal": float(result.mean_electrolyte_coefficients[0]),
    }
    for value, target in zip(
        measured.values(), [0.9878567852129456, 0.7037048661941817], strict=True
    ):
        if not math.isclose(value, target, abs_tol=1e-10, rel_tol=1e-8):
            raise ValueError(f"Mixed case failed: {measured}")
    print(json.dumps(dict(versions=versions, **measured)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-version")
    parser.add_argument("--backend-version")
    pair = parser.add_mutually_exclusive_group()
    pair.add_argument("--base-wheel", type=Path)
    pair.add_argument("--backend-wheel", type=Path)
    parser.add_argument("--measure", action="store_true")
    args = parser.parse_args()
    if args.measure:
        measure(args.base_version, args.backend_version)
        return
    requirements = [
        str(args.base_wheel.resolve()) + "[aiomfac]"
        if args.base_wheel
        else "particula[aiomfac]"
        + ("==" + args.base_version if args.base_version else "")
    ]
    if args.backend_wheel:
        requirements.append(str(args.backend_wheel.resolve()))
    elif args.backend_version and not args.base_version:
        requirements.append("particula-aiomfac==" + args.backend_version)
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("PIP_", "PYTHON"))
    }
    env["PIP_CONFIG_FILE"] = os.devnull
    with tempfile.TemporaryDirectory() as root:
        venv.EnvBuilder(with_pip=True).create(root)
        python = Path(root) / "bin/python"
        # Simple-index visibility can lag behind version-specific JSON metadata.
        command = [
            python,
            "-I",
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--index-url",
            "https://pypi.org/simple",
            *requirements,
        ]
        for attempt in range(6):
            install = subprocess.run(
                command, cwd=root, env=env, capture_output=True, text=True
            )
            if install.returncode == 0:
                break
            print(install.stderr, flush=True)
            if (
                attempt == 5
                or "No matching distribution found" not in install.stderr
            ):
                install.check_returncode()
            time.sleep(10)
        subprocess.run(
            [python, "-I", "-m", "pip", "check"], cwd=root, env=env, check=True
        )
        command = [python, "-I", Path(__file__).resolve(), "--measure"]
        for flag, value in (
            ("--base-version", args.base_version),
            ("--backend-version", args.backend_version),
        ):
            if value:
                command.extend([flag, value])
        output = subprocess.check_output(command, cwd=root, env=env, text=True)
        print(output)
        if summary := os.getenv("GITHUB_STEP_SUMMARY"):
            with Path(summary).open("a") as f:
                f.write(
                    f"Mixed reference at 298 K, commit {os.environ['GITHUB_SHA']}\n\n```\n{output}```\n"
                )


if __name__ == "__main__":
    main()
