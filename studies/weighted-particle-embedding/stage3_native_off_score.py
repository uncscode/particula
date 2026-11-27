"""Isolate native collision cost and bias at unchanged vapor resolution."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stage3_cross_reference import (
    Moments,
    group,
    read_charged,
    read_uptake,
    summarize,
)
from stage3_evidence import obs, write


def score(root, results, panel="native-off"):
    """Use independent coupled and vapor-only references on identical boxes."""
    coupled = read_charged(root / "moving", "moving")
    uptake = read_uptake(root / "uptake")
    groups = defaultdict(list)
    for path in sorted((root / panel).glob("*.json")):
        row = json.loads(path.read_text())
        groups[group(row), row["vapor_micro"]].append(row)
    with (results / "native-scale-comparison.csv").open() as stream:
        native = {
            int(r["initial_particles"]): r
            for r in csv.DictReader(stream)
            if r["method"].startswith("Native Warp")
        }
    out = []
    for (case, micro), runs in sorted(groups.items()):
        n, s, density, charge, family, horizon = case
        advance = [r["advance_and_host_diagnostics_s"] for r in runs]
        total = np.median(
            [r["setup_s"] + a for r, a in zip(runs, advance, strict=True)]
        )
        base = native.get(n)
        for purpose, source in [
            ("Full physical dynamics", coupled.get(case)),
            (
                "Vapor integration alone",
                uptake.get((n, s, density, family, horizon)),
            ),
        ]:
            if source is None:
                continue
            label, ref, floor = source
            # Native atomic reductions can differ at roundoff.
            # Use the worst error across repeated forecasts.
            checks = []
            for r in runs:
                value = obs(dict(history=r["history"][0]))
                moment = Moments(value)
                moment.add(value)
                checks.append(
                    summarize(
                        moment,
                        ref,
                        np.zeros_like(ref),
                        10**9,
                        floor,
                        horizon,
                        deterministic=True,
                    )
                )
            num = max(c["number_upper_pct"] for c in checks)
            mass = max(c["mass_upper_pct"] for c in checks)
            out.append(
                dict(
                    n=n,
                    species=s,
                    charge=charge,
                    family=family,
                    density_m3=density,
                    horizon_s=horizon,
                    vapor_steps_per_half=micro,
                    control=purpose,
                    reference=label,
                    timing_trials=len(runs),
                    median_setup_s=np.median([r["setup_s"] for r in runs]),
                    median_advance_s=np.median(advance),
                    median_total_s=total,
                    native_full_timing_trials=int(base["timing_trials"])
                    if base
                    else "",
                    native_full_total_s=base["total_s"] if base else "",
                    native_full_advance_s=base["advance_s"] if base else "",
                    native_full_over_off_total=float(base["total_s"]) / total
                    if base
                    else "",
                    native_full_over_off_advance=float(base["advance_s"])
                    / np.median(advance)
                    if base
                    else "",
                    number_observed_pct=max(
                        c["number_mean_pct"] for c in checks
                    ),
                    phase_observed_pct=max(c["mass_mean_pct"] for c in checks),
                    number_upper_pct=num,
                    phase_upper_pct=mass,
                    limiting_phase=max(
                        checks, key=lambda c: c["mass_upper_pct"]
                    )["limiting_mass_field"],
                    qualifies=num <= 1 and mass < 0.1,
                    maximum_relative_ledger=max(r["max_ledger"] for r in runs),
                )
            )
    return out


def graph(rows, target):
    """Show native costs and phase errors with identical vapor substeps."""
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.7), layout="constrained")
    coupled = [r for r in rows if r["control"] == "Full physical dynamics"]
    for key, label, col in [
        ("native_full_total_s", "With collisions", "#71488f"),
        ("median_total_s", "Without collisions", "#236875"),
    ]:
        data = [r for r in coupled if r[key] != ""]
        axes[0].plot(
            [r["n"] for r in data],
            [float(r[key]) for r in data],
            "o-",
            label=label,
            color=col,
        )
    axes[0].set(ylabel="Median setup + forecast (s)", yscale="log")
    for label, col in [
        ("Full physical dynamics", "#853f22"),
        ("Vapor integration alone", "#236875"),
    ]:
        data = [r for r in rows if r["control"] == label]
        axes[1].plot(
            [r["n"] for r in data],
            [r["phase_upper_pct"] for r in data],
            "o-",
            label=label,
            color=col,
        )
    axes[1].axhline(0.1, ls="--", color="#555555", lw=1)
    axes[1].set(ylabel="Upper phase mass error without collisions (%)")
    for ax in axes:
        ax.set(xscale="log", xlabel="Explicit particles per box")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
    fig.suptitle("Switching off native coagulation at 32 vapors and one second")
    fig.savefig(target, dpi=155)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("results", type=Path)
    a = p.parse_args()
    rows = score(a.root, a.results)
    if rows:
        write(a.results / "native-omission-control.csv", rows)
        graph(rows, a.results / "native-omission-control.png")
