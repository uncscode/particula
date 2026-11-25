"""Run the independent numerical stages of the published surfactant study."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def campaign(stage, bins=48, nodes=1025, dt=0.1):
    """Configure a tagged CUDA ensemble with three activation-cutoff passes."""
    return (
        "cloud/campaign.py",
        stage,
        "--bins",
        str(bins),
        "--nodes",
        str(nodes),
        "--dt",
        str(dt),
        "--rounds",
        "3",
    )


# Each stage can run directly after obtaining the two publisher-derived inputs.
# Separate names keep expensive sensitivity studies optional when redrawing.
STAGES = {
    "equilibrium": [
        ("inventory_check.py",),
        ("dilution_bounds.py",),
        ("analyze_equilibrium.py",),
    ],
    "identity": [campaign("identity", 96, 2049, 0.05)],
    "heights": [("cloud/milestones.py",)],
    "edge": [("cloud/edge_attribution.py", "main")],
    "edge-refinement": [("cloud/edge_attribution.py", "fine")],
    "edge-adsorption": [("cloud/eddy_grid.py", "--stages", "main")],
    "regimes": [
        campaign("factorial"),
        campaign("attribution"),
        campaign("tension"),
        ("cloud/regime_summary.py",),
    ],
    "fluctuations": [campaign("forcing")],
    "loading": [campaign("loading")],
    "organic-limits": [campaign("organic_limits")],
    "smoke-mixing": [
        (
            "cloud/resolved.py",
            "--draws",
            "120",
            "--replicates",
            "32",
            "--nodes",
            "513",
            "--dt",
            "0.25",
        ),
        (
            "cloud/resolved.py",
            "--draws",
            "120",
            "--replicates",
            "32",
            "--nodes",
            "1025",
            "--dt",
            "0.125",
            "--tag",
            "_refined",
        ),
    ],
    "seed-sizes": [
        campaign("spectrum", 64),
        campaign("spectrum_refined", 112, 2049, 0.05),
    ],
    "cpu-check": [
        ("parcel_study.py", "reference"),
        ("parcel_study.py", "steady"),
        ("parcel_study.py", "refine"),
        ("cloud/mixed_cpu.py",),
        ("cloud/mixed_cpu.py", "--crosscheck"),
        ("cloud/mixed_cpu.py", "--audit"),
    ],
    "cuda-reference": [campaign("reference", 64, 2049, 0.05)],
    # Derived only from the immediately preceding scientific stages.
    "edge-summary": [("cloud/edge_summary.py",)],
    "compare-cpu": [("cloud/compare_cpu.py",)],
}
FIGURE_STAGES = ("equilibrium", "identity", "heights", "edge", "organic-limits")


def main():
    """Run a requested numerical stage or show its exact commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(*STAGES, "figures", "all"),
        help="figures regenerates the numerical inputs used in the plots",
    )
    parser.add_argument(
        "--list", action="store_true", help="Print commands without executing"
    )
    args = parser.parse_args()
    if args.stage == "all":
        commands = [cmd for group in STAGES.values() for cmd in group]
    elif args.stage == "figures":
        commands = [cmd for name in FIGURE_STAGES for cmd in STAGES[name]]
    else:
        commands = STAGES[args.stage]

    if not args.list:
        if any(
            not (HERE / name).exists()
            for name in ("figure5.csv", "growth_envelope.csv")
        ):
            parser.error(
                "Run fetch_inputs.py once to obtain the publisher inputs."
            )
        if args.stage in ("compare-cpu", "edge-summary"):
            required = (
                ["cpu_populations.csv", "gpu_reference_populations.csv"]
                if args.stage == "compare-cpu"
                else ["edge-attribution/main.csv", "edge-attribution/fine.csv"]
            )
            if any(
                not (HERE / "cloud/results" / name).exists()
                for name in required
            ):
                depends = (
                    "cpu-check and cuda-reference"
                    if args.stage == "compare-cpu"
                    else "edge and edge-refinement"
                )
                parser.error(f"First run the {depends} stages.")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO), env.get("PYTHONPATH")])
    )
    env["OPENBLAS_NUM_THREADS"] = "1"
    for command in commands:
        print("python", *command, flush=True)
        if not args.list:
            subprocess.run(  # noqa: S603 - fixed study scripts
                [sys.executable, *command], cwd=HERE, env=env, check=True
            )


if __name__ == "__main__":
    main()
