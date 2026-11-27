"""Compare native and reduced vapor with identical sampled collision laws."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stage3_cross_reference import Moments, group, read_charged, summarize
from stage3_evidence import obs, write


def score(cache, results):
    """Compare forecast times, collision costs and paired trajectories."""
    grids = read_charged(cache / "moving", "moving")
    with (results / "native-scale-comparison.csv").open() as src:
        front = {
            (int(r["initial_particles"]), r["method"]): r
            for r in csv.DictReader(src)
        }
    off = {}
    off_path = results / "native-omission-control.csv"
    if off_path.exists():
        with off_path.open() as src:
            off = {
                int(r["n"]): r
                for r in csv.DictReader(src)
                if r["control"] == "Full physical dynamics"
            }
    groups = defaultdict(list)
    for path in (cache / "native-sampled").glob("*.json"):
        r = json.loads(path.read_text())
        groups[
            group(r),
            r["kernel"].removeprefix("native-vapor-sampled-"),
            r.get("substeps_per_collision_interval", 1),
        ].append(r)
    rows = []
    for (case, method, parts), runs in sorted(groups.items()):
        n, s, density, charge, family, horizon = case
        if case not in grids:
            continue
        label, ref, floor = grids[case]
        moments = Moments(ref)
        pair = []
        for r in runs:
            state = obs(dict(history=r["history"][0]))
            moments.add(state)
            seed = r["first_seed"]
            path = (
                cache
                / "native-overlap"
                / f"match-{n}-{s}-{charge}-midpoint-{method}-1-{seed}.json"
            )
            if parts == 1 and path.exists():
                reduced = json.loads(path.read_text())
                if reduced["status"] == "completed":
                    v = obs(reduced)
                    scale = np.maximum(abs(state), 1e-11)
                    delta = 100 * abs(v - state) / scale
                    pair.append(
                        [
                            np.array_equal(v[:, 0], state[:, 0]),
                            delta[:, 0].max(),
                            delta[:, 1:].max(),
                        ]
                    )
        bounds = summarize(
            moments, ref, np.zeros_like(ref), 10**9, floor, horizon
        )
        full = front.get((n, "Native Warp, 128 vapor substeps"))
        other = front.get(
            (
                n,
                "Sampled physical collisions"
                if method == "physical"
                else "Sampled neural collisions",
            )
        )
        if charge != "neutral":
            full = other = None
        initial = np.median([r["setup_s"] for r in runs])
        warm = np.median([r["advance_and_host_diagnostics_s"] for r in runs])
        whole = np.median(
            [r["setup_s"] + r["advance_and_host_diagnostics_s"] for r in runs]
        )
        rows.append(
            dict(
                n=n,
                species=s,
                density_m3=density,
                charge=charge,
                family=family,
                horizon_s=horizon,
                vapor="Native Warp",
                vapor_substeps_per_half=128,
                sampled_kernel=method,
                substeps_per_collision_interval=parts,
                reference=label,
                seeds=len(runs),
                median_setup_s=initial,
                median_advance_s=warm,
                median_total_s=whole,
                median_collision_only_s=np.median(
                    [r["separate_collision_s"] for r in runs]
                ),
                median_accepted_mergers=np.median(
                    [r["accepted_mergers"] for r in runs]
                ),
                max_ledger=max(r["max_ledger"] for r in runs),
                native_full_s=full["total_s"] if full else "",
                native_full_over_hybrid=float(full["total_s"]) / whole
                if full
                else "",
                native_vapor_only_s=off.get(n, {}).get("median_total_s", "")
                if charge == "neutral"
                else "",
                midpoint_same_law_s=other["total_s"] if other else "",
                hybrid_over_midpoint=float(whole) / float(other["total_s"])
                if other
                else "",
                matching_reduced_seeds=len(pair),
                matching_counts_all_times=sum(p[0] for p in pair),
                paired_max_number_difference_pct=max(
                    (p[1] for p in pair), default=""
                ),
                paired_max_phase_difference_pct=max(
                    (p[2] for p in pair), default=""
                ),
                **bounds,
            )
        )
    return rows


def graph(rows, target):
    """Show the runtime consequence of substituting each individual operator."""
    rows = [
        r
        for r in rows
        if r["charge"] == "neutral"
        and r["substeps_per_collision_interval"] == 1
    ]
    if not rows:
        return
    fig, (ax, mass) = plt.subplots(
        1, 2, figsize=(11, 4.9), layout="constrained"
    )
    for method, color in [("physical", "#187552"), ("learned", "#965727")]:
        v = sorted(
            (r for r in rows if r["sampled_kernel"] == method),
            key=lambda r: r["n"],
        )
        if not v:
            continue
        ax.plot(
            [r["n"] for r in v],
            [r["median_total_s"] for r in v],
            "o-",
            color=color,
            label="Native vapor + sampled " + method,
        )
        lower = [r for r in v if r["midpoint_same_law_s"] != ""]
        ax.plot(
            [r["n"] for r in lower],
            [float(r["midpoint_same_law_s"]) for r in lower],
            "^--",
            color=color,
            label="Midpoint vapor + sampled " + method,
        )
        mass.plot(
            [r["n"] for r in v],
            [r["mass_mean_pct"] for r in v],
            "o-",
            color=color,
            label=method.capitalize() + " pair rate",
        )
    sizes = {r["n"]: r for r in rows}
    for field, lab, col in [
        ("native_full_s", "Native vapor + native collisions", "#654885"),
        ("native_vapor_only_s", "Native vapor alone", "#457d97"),
    ]:
        v = [sizes[n] for n in sorted(sizes) if sizes[n][field] != ""]
        ax.plot(
            [r["n"] for r in v],
            [float(r[field]) for r in v],
            "s-",
            color=col,
            label=lab,
        )
    ax.set(yscale="log", ylabel="Setup + one simulated second (s)")
    mass.axhline(0.1, ls="--", lw=1, color="#555555")
    mass.set(ylabel="Maximum observed phase mass difference (%)")
    for chart in (ax, mass):
        chart.set(xscale="log", xlabel="Explicit particles per box")
        chart.grid(alpha=0.2)
        chart.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Native vapor transfer with measured physical or neural collision rates"
    )
    fig.savefig(target, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    p.add_argument("results", type=Path)
    a = p.parse_args()
    rows = score(a.cache, a.results)
    if rows:
        write(a.results / "native-vapor-sampled-collisions.csv", rows)
        graph(rows, a.results / "native-vapor-sampled-collisions.png")
