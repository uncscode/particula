"""Reduce the paired measurements and render the Milestone 7 scaling study."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _write(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def report(root):
    raw = [
        json.loads(line)
        for file in sorted(root.glob("*.jsonl"))
        for line in file.read_text().splitlines()
        if line.strip()
    ]
    builds = [r for r in raw if r["stage"] == "build"]
    qual = [
        r
        for r in raw
        if r["stage"] == "qualification" and r["device"] == r["build_device"]
    ]
    cost = {
        (r["recipe"], r["device"]): r["total_s"] + r["serialization_s"]
        for r in qual
    }
    for r in builds:
        if r["factor"] == 1:
            cost[r["recipe"], r["device"]] = (
                np.median(r["warmed_build_s"]) + r["serialization_s"]
            )
    build_summary = []
    for r in builds:
        times = np.array(r["warmed_build_s"])
        build_summary.append(
            dict(
                recipe=r["recipe"],
                device=r["device"],
                tolerance_factor=r["factor"],
                states=r["nodes"],
                bytes=r["bytes"],
                build_median_s=np.median(times),
                build_min_s=times.min(),
                build_max_s=times.max(),
                fixed_nodes=r["fixed_node_count"],
                fixed_node_median_s=np.median(r["fixed_node_s"]),
                serialize_s=r["serialization_s"],
                max_log_error=r["holdout_max_log"],
            )
        )
    by_build = {
        (r["recipe"], r["tolerance_factor"], r["device"]): r
        for r in build_summary
    }
    for r in build_summary:
        key = r["recipe"], r["tolerance_factor"]
        cpu = by_build.get((*key, "cpu"))
        gpu = by_build.get((*key, "cuda:0"))
        if cpu and gpu:
            r["cpu_over_gpu_build_time"] = (
                cpu["build_median_s"] / gpu["build_median_s"]
            )
            r["cpu_over_gpu_fixed_node_time"] = (
                cpu["fixed_node_median_s"] / gpu["fixed_node_median_s"]
            )
        else:
            r["cpu_over_gpu_build_time"] = r["cpu_over_gpu_fixed_node_time"] = (
                ""
            )
    _write(root / "build-summary.csv", build_summary)
    misses = []
    for r in raw:
        if r["stage"] == "fallback":
            speedup = np.array(r["direct_s"]) / r["hybrid_s"]
            misses.append(
                dict(
                    recipe=r["recipe"],
                    device=r["device"],
                    slots=r["n"],
                    occupancy=r["occupancy"],
                    direct_fraction=r["fallback"],
                    shuffled=r["shuffled"],
                    direct_rows=r["direct_rows"],
                    speedup=np.median(speedup),
                    min_speedup=speedup.min(),
                    max_speedup=speedup.max(),
                )
            )
    _write(root / "fallback-summary.csv", misses)
    cold_cost = {
        (r["recipe"], r["device"]): r["wall_s"] + r["serialization_s"]
        for r in raw
        if r["stage"] == "cold_build" and r["cold"]
    }
    paired = defaultdict(dict)
    for r in raw:
        if (
            r["stage"] in {"box", "water", "parcel", "activation"}
            and "pair" in r
        ):
            # Restart runs time a second owner's continuation and have their own group.
            key = (
                r["stage"],
                r["n"],
                r["solver"],
                r.get("saturation", 0),
                r.get("restart_compared", False),
                r.get("table_build_device", "cpu"),
            )
            paired[key][r["pair"], r["hybrid"]] = r
    amort = []
    flows = []
    for (stage, n, solver, sat, restart, artifact_device), samples in sorted(
        paired.items()
    ):
        pairs = sorted(i for i, h in samples if h and (i, False) in samples)
        if not pairs:
            continue
        device = samples[pairs[0], True]["device"]
        direct = np.array([samples[i, False]["wall_s"] for i in pairs])
        hybrid = np.array([samples[i, True]["wall_s"] for i in pairs])
        saving = float(np.median(direct - hybrid))
        row = dict(
            workload=stage,
            slots=n,
            solver=solver,
            saturation=sat,
            restart=restart,
            artifact_build_device=artifact_device,
            pairs=len(pairs),
            direct_s=float(np.median(direct)),
            hybrid_s=float(np.median(hybrid)),
            median_paired_saving_s=saving,
            speedup=np.median(direct / hybrid),
            min_speedup=np.min(direct / hybrid),
            max_speedup=np.max(direct / hybrid),
        )
        flows.append(row)
        # Matched-origin trajectories take precedence. With only one measured
        # origin, report the other builder's amortization as an explicit estimate.
        origins = {
            k[-1] for k in paired if k[:-1] == (stage, n, solver, sat, restart)
        }
        for builder in ("cpu", "cuda:0"):
            if builder in origins and builder != artifact_device:
                continue
            key = (
                "mixed-water" if stage == "water" else "mixed-two-vapor",
                builder,
            )
            if key in cost:
                amort.append(
                    dict(
                        **row,
                        run_device=device,
                        build_device=builder,
                        measured_artifact_origin=artifact_device == builder,
                        build_s=cost[key],
                        cold_build_s=cold_cost.get(key, ""),
                        cold_break_even_runs=(
                            math.ceil(cold_cost[key] / saving)
                            if saving > 0
                            else "none"
                        )
                        if key in cold_cost
                        else "",
                        break_even_runs=math.ceil(cost[key] / saving)
                        if saving > 0
                        else "none",
                    )
                )
    _write(root / "workload-summary.csv", flows)
    _write(root / "amortization.csv", amort)
    timings = []
    for r in raw:
        if r["stage"] != "throughput":
            continue
        for kind in ("host", "device"):
            suffix = "_s" if kind == "host" else "_device_s"
            if "direct" + suffix not in r:
                continue
            x = np.array(r["direct" + suffix])
            y = np.array(r["hybrid" + suffix])
            timings.append(
                dict(
                    recipe=r["recipe"],
                    device=r["device"],
                    scope=kind,
                    n=r["n"],
                    direct_ms=1000 * np.median(x),
                    hybrid_ms=1000 * np.median(y),
                    speedup=np.median(x / y),
                    min_speedup=np.min(x / y),
                    max_speedup=np.max(x / y),
                    direct_rows_s=r["n"] / np.median(x),
                    hybrid_rows_s=r["n"] / np.median(y),
                )
            )
    _write(root / "activity-summary.csv", timings)
    operations = []
    for row in raw:
        if row["stage"] != "box-accounting":
            continue
        counts = row.get("lookup_counts", [0] * 7)
        occupied = sum(counts[:6])
        operations.append(
            dict(
                slots=row["n"],
                solver=row["solver"],
                hybrid=row["hybrid"],
                activity_evaluations=row["activity_evaluations"],
                hits=counts[0],
                fallback_fraction=sum(counts[1:6]) / occupied
                if occupied
                else "",
                direct_failures=counts[6],
                control_readbacks=row.get("control_readbacks", 0),
                explicit_snapshots=row.get("explicit_snapshots", 0),
                initial_rate_downloads=row.get("initial_rate_downloads", 0),
                max_species_budget=row["errors"]["budget"],
            )
        )
    _write(root / "box-operations.csv", operations)
    crossings = []
    for recipe in sorted({r["recipe"] for r in timings}):
        rows = [
            r for r in timings if r["recipe"] == recipe and r["scope"] == "host"
        ]
        by = {(r["device"], r["n"]): r for r in rows}
        common = sorted(n for d, n in by if d == "cpu" and ("cuda:0", n) in by)
        for mode in ("direct", "hybrid"):
            eligible = [
                n
                for n in common
                if by["cuda:0", n][mode + "_ms"] < by["cpu", n][mode + "_ms"]
            ]
            crossings.append(
                dict(
                    recipe=recipe,
                    mode=mode,
                    first_tested_gpu_faster=min(eligible)
                    if eligible
                    else "none",
                    paired_sizes=len(common),
                )
            )
    _write(root / "crossovers.csv", crossings)
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    build, activity, memory, fallback, box, parcel = axes.ravel()
    for device, color, marker in (
        ("cpu", "#16697a", "o"),
        ("cuda:0", "#a14411", "^"),
    ):
        subset = [r for r in builds if r["device"] == device]
        if subset:
            build.scatter(
                [r["nodes"] for r in subset],
                [np.median(r["warmed_build_s"]) for r in subset],
                label=device,
                color=color,
                marker=marker,
                s=22,
                alpha=0.75,
            )
        for mode, style in (("direct", "--"), ("hybrid", "-")):
            rows = sorted(
                (
                    r
                    for r in timings
                    if r["recipe"] == "mixed-water"
                    and r["device"] == device
                    and r["scope"] == "host"
                ),
                key=lambda r: r["n"],
            )
            if rows:
                activity.plot(
                    [r["n"] for r in rows],
                    [r[mode + "_rows_s"] for r in rows],
                    style,
                    marker=marker,
                    ms=3,
                    label=f"{device} {mode}",
                    color=color,
                )
        rows = sorted(
            (
                r
                for r in raw
                if r["stage"] == "throughput"
                and r["recipe"] == "mixed-water"
                and r["device"] == device
            ),
            key=lambda r: r["n"],
        )
        if rows:
            mib = [
                r["peak_process_rss_KiB"] / 1024
                if device == "cpu"
                else r["peak_process_device_pool_bytes"] / 2**20
                for r in rows
            ]
            memory.plot(
                [r["n"] for r in rows],
                mib,
                marker=marker,
                label="CPU RSS" if device == "cpu" else "CUDA pool",
                color=color,
            )
        for shuffle, style in ((True, "-"), (False, "--")):
            rows = sorted(
                (
                    r
                    for r in raw
                    if r["stage"] == "fallback"
                    and r["recipe"] == "mixed-water"
                    and r["device"] == device
                    and r["occupancy"] == 1
                    and r["n"] == 4096
                    and r["shuffled"] == shuffle
                ),
                key=lambda r: r["fallback"],
            )
            if rows:
                fallback.plot(
                    [r["fallback"] * 100 for r in rows],
                    [r["speedup"] for r in rows],
                    style,
                    marker=marker,
                    label=f"{device} "
                    + ("shuffled" if shuffle else "clustered"),
                    color=color,
                )
    for ax, stage, solvers in (
        (box, "box", ("native", "fixed", "host-rk23", "resident-rk23")),
        (parcel, "parcel", ("resident-rk23",)),
    ):
        for sol in solvers:
            rows = sorted(
                (
                    r
                    for r in flows
                    if r["workload"] == stage
                    and r["solver"] == sol
                    and not r["restart"]
                ),
                key=lambda r: r["slots"],
            )
            for mode, style in (("direct", "--"), ("hybrid", "-")):
                if rows:
                    ax.plot(
                        [r["slots"] for r in rows],
                        [r[mode + "_s"] for r in rows],
                        style,
                        marker=".",
                        label=f"{sol} {mode}",
                    )
    for ax, title, x, y in (
        (
            build,
            "Table construction",
            "Unique evaluated nodes",
            "Build time (s)",
        ),
        (
            activity,
            "Mixed-water neutral lookup",
            "Occupied rows",
            "Rows per second",
        ),
        (memory, "Process high-water marks", "Occupied rows", "Memory (MiB)"),
        (
            fallback,
            "Direct fallback at 4,096 rows",
            "Fallback (%)",
            "Direct / hybrid time",
        ),
        (box, "Two-vapor closed box", "Occupied slots", "Trajectory time (s)"),
        (parcel, "150 m rising parcel", "Size classes", "Trajectory time (s)"),
    ):
        ax.set(title=title, xlabel=x, ylabel=y)
        if ax.lines or ax.collections:
            if ax != fallback:
                ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fallback.axhline(1, color=".4", linewidth=0.8)
    figure.savefig(root / "scaling.png", dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    report(parser.parse_args().root)
