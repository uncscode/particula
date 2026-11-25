"""Replay the smoke cloud edge with matched particle and chemistry controls."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import campaign as gpu
import eddy_grid as eddy
import numpy as np

OUTPUT = gpu.RESULTS / "edge-attribution"
CONTROLS = {
    "finite": ("mixed", "central"),
    "acid": ("acid", "central"),
    "no_targets": ("none", "central"),
    "fixed70": ("mixed", "fixed70"),
}


def baseline(indices):
    """Rebuild the release coordinates and original stage time grids."""
    flows = {}
    selected = set(indices)
    for stage in ("main", *eddy.INLETS):
        if not selected.intersection(
            round(p * 128) for p in eddy.get_ids(stage)
        ):
            continue
        # In the original ensemble all trajectories in a stage share the
        # number of steps required by the longest trajectory in that stage.
        jobs = eddy.make_jobs(stage)
        steps = eddy.step_count(jobs)
        for job in jobs:
            index = round(job["phase"] * 128)
            if (
                index not in selected
                or job["amp"] != 0.8
                or job["env"] not in ("opposed", "patterned")
                or job["ads"] != "central"
            ):
                continue
            job.update(
                steps=steps, entr=job["case"].entr, rh_ext=job["case"].rh_ext
            )
            flows[job["env"], index] = job
    if not (len(flows) == 2 * len(selected)):
        raise ValueError(
            "Missing a paired inlet in the reconstructed velocity field"
        )
    return flows


def record(row, groups, air, control, level, index):
    """Combine tagged populations and check endpoint reservoir closure."""
    total = float(air["activated_cm3"])
    tags = sum(
        float(r["activated_cm3"])
        for r in groups
        if r["mode"] in ("acid", "octyl", "dodecyl")
    )
    incoming = sum(
        float(r["activated_cm3"])
        for r in groups
        if r["mode"].startswith("incoming")
    )
    resident = sum(
        float(r["activated_cm3"])
        for r in groups
        if r["mode"].startswith("resident")
    )
    if not (abs(tags + resident + incoming - total) < 1e-8):
        raise ValueError("Activated tagged populations do not sum to the total")
    if not (
        float(air["water_err_kgkg"]) < 1e-10
        and float(air["organic_err_kgkg"]) < 1e-15
    ):
        raise ValueError("Parcel water or organic balance failed")
    return dict(
        level=level,
        inlet=index,
        placement=row["env"],
        control=control,
        base_speed_ms=float(row["wbase"]),
        transit_s=float(row["time_s"]),
        entrained_fraction=float(row["entr"]),
        ambient_rh=float(row["rh_ext"]),
        total_cm3=total,
        target_cm3=tags,
        incoming_cm3=incoming,
        resident_cm3=resident,
        peak_pct=float(air["peak_pct"]),
        water_closure_kgkg=float(air["water_err_kgkg"]),
        organic_closure_kgkg=float(air["organic_err_kgkg"]),
        cutoff_width_cm3=sum(float(r["uncertainty_width_cm3"]) for r in groups),
    )


def run(level, outdir=OUTPUT, inlet=None):  # noqa: C901
    """Advance each selected trajectory in both placements and all controls."""
    started = time.monotonic()
    fine = level == "fine"
    indices = range(0, 128, 8 if fine else 1) if inlet is None else [inlet]
    flows = baseline(indices)
    run_gpu = gpu.build_cuda()
    outputs = []
    buckets = defaultdict(list)
    for (_, index), row in flows.items():
        steps = int(row["steps"]) * (2 if fine else 1)
        transit = row["time_s"]
        pos = row["position"]
        dt = transit / steps
        heights = pos(np.linspace(0, transit, steps + 1))[1]
        velocity = np.ascontiguousarray(np.diff(heights) / dt)
        for control, (mix, scenario) in CONTROLS.items():
            case = gpu.Case(
                regime="smoke",
                gas=1.0,
                w=0.2,
                entr=float(row["entr"]),
                rh_ext=float(row["rh_ext"]),
                mix=mix,
                scenario=scenario,
            )
            buckets[steps].append((case, row, index, control, dt, velocity))
    bins, nodes = (96, 2049) if level == "fine" else (48, 1025)
    computed = 0
    for steps, jobs in buckets.items():
        brackets = [{} for _ in jobs]
        for iteration in range(3):
            sizes = defaultdict(list)
            for i, (case, *_) in enumerate(jobs):
                sizes[
                    tuple(
                        (m.label, len(brackets[i].get(m.label, [])))
                        for m in gpu.modes(case)
                    )
                ].append(i)
            for ids in sizes.values():
                for first in range(0, len(ids), 10):
                    batch = []
                    selected = ids[first : first + 10]
                    for i in selected:
                        case, _, _, _, dt, velocity = jobs[i]
                        modes, groups, params, _ = gpu.prepare(
                            case, bins, nodes, dt, brackets[i]
                        )
                        batch.append((modes, groups, params, velocity))
                    air, part, _ = gpu.call_gpu(run_gpu, batch, nodes, steps)
                    for j, i in enumerate(selected):
                        case, row, index, control, _, _ = jobs[i]
                        result, groups, brackets[i] = gpu.unpack(
                            case, batch[j], air[j], part[j], bins
                        )
                        if iteration == 2:
                            outputs.append(
                                record(
                                    row, groups, result, control, level, index
                                )
                            )
                            computed += 1
    if not (computed == len(outputs) == len(indices) * 2 * len(CONTROLS)):
        raise ValueError("A treatment or paired placement is missing")
    outdir.mkdir(parents=True, exist_ok=True)
    name = level if inlet is None else f"{level}_inlet_{inlet:03}"
    gpu.write(
        outdir / f"{name}.csv",
        sorted(
            outputs, key=lambda r: (r["placement"], r["inlet"], r["control"])
        ),
    )
    info = dict(
        device=subprocess.check_output(
            [  # noqa: S607
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        simulated=computed,
        inlets=len(indices),
        bins=bins,
        nodes=nodes,
        cutoff_passes=3,
        time_step="matched production grid"
        if level == "main"
        else "half production time step",
        python=platform.python_version(),
        wall_seconds=time.monotonic() - started,
        sha256={
            n: hashlib.sha256((gpu.HERE / n).read_bytes()).hexdigest()
            for n in ("cloud.cu", "organic.cuh", "campaign.py", "eddy_grid.py")
        },
    )
    (outdir / f"{name}.json").write_text(json.dumps(info, indent=2) + "\n")
    print(info, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("level", choices=("main", "fine"))
    cli.add_argument("--out", type=Path, default=OUTPUT)
    cli.add_argument(
        "--inlet",
        type=int,
        choices=range(128),
        metavar="0..127",
        help="Run just this inlet in both placements and all four treatments",
    )
    args = cli.parse_args()
    if args.level == "fine" and args.inlet is not None and args.inlet % 8:
        cli.error("Published fine inlets are multiples of eight")
    run(args.level, args.out, args.inlet)
