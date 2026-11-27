"""Estimate repeated-ion newborn survival and numerical clock sensitivity."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import t

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stage3_evidence import write
from stage3_newborn_ion_paths import INITIAL_SEED

LOOKS = (16, 64, 128)
COMPARISONS = 9 * 2 * 3 * len(LOOKS) * 2


def score(cache, results):
    """Use independent stratified batches and common-seed clock differences."""
    envelopes = {}
    with (results / "newborn-finite-ion-survival.csv").open() as f:
        for r in csv.DictReader(f):
            if float(r["background_m3"]) == 1e10:
                envelopes[
                    r["initial_host_charge"],
                    float(r["initial_ions_each_sign_per_host"]),
                    float(r["diameter_nm"]),
                ] = r
    groups = defaultdict(dict)
    for p in (cache / "newborn-ion-paths").glob("*.json"):
        for r in json.loads(p.read_text()):
            groups[
                r["initial_host_charge"],
                r["initial_ions_each_sign_per_host"],
                r["diameter_nm"],
                r["level"],
            ][r["batch_seed"]] = r
    out = []
    for (z, ratio, d, level), g in sorted(groups.items()):
        for n in LOOKS:
            seeds = range(INITIAL_SEED, INITIAL_SEED + n)
            if not all(k in g for k in seeds):
                break
            rs = [g[k] for k in seeds]
            val = np.array([r["arrivals_per_1000"] for r in rs])
            ct = t.ppf(1 - 0.05 / (2 * COMPARISONS), n - 1)
            margin = ct * np.std(val, ddof=1) / np.sqrt(n)
            co = groups.get((z, ratio, d, level - 1), {})
            diff = bound = None
            if all(k in co for k in seeds):
                delta = val - np.array(
                    [co[k]["arrivals_per_1000"] for k in seeds]
                )
                diff = delta.mean()
                bound = abs(diff) + ct * np.std(delta, ddof=1) / np.sqrt(n)
            e = envelopes[z, ratio, d]
            lo, hi = [
                float(e[k])
                for k in (
                    "lower_minus_refinement_per_1000",
                    "upper_plus_refinement_per_1000",
                )
            ]
            fixed = float(e["per_1000_fixed_tracer_live_hosts"])
            ns = rs[0]["tracers_in_batch"]
            row = dict(
                initial_host_charge=z,
                initial_ions_each_sign_per_host=ratio,
                diameter_nm=d,
                level=level,
                time_step_s=rs[0]["time_step_s"],
                batches=n,
                tracer_paths=n * ns,
                arrivals_per_1000=val.mean(),
                simultaneous_sampling_margin=margin,
                change_from_coarser_per_1000=diff,
                paired_clock_change_upper_per_1000=bound,
                first_event_envelope_lower=lo,
                first_event_envelope_upper=hi,
                fixed_tracer_live_hosts_per_1000=fixed,
                observed_change_from_fixed_tracer=val.mean() - fixed,
                frozen_hosts_zero_ions_per_1000=e[
                    "fixed_host_and_tracer_zero_ion_per_1000"
                ],
                sample_interval_overlaps_first_event_envelope=val.mean()
                - margin
                <= hi
                and val.mean() + margin >= lo,
                mean_fraction_tracers_with_ion=np.mean(
                    [r["fraction_with_any_ion"] for r in rs]
                ),
                mean_fraction_with_multiple_ions=np.mean(
                    [r["fraction_with_two_or_more_ions"] for r in rs]
                ),
                maximum_tracer_material_ledger=max(
                    r["max_tracer_relative_ledger"] for r in rs
                ),
                maximum_host_charge_ion_ledger=max(
                    r["max_host_ledger_per_host"] for r in rs
                ),
                maximum_events_per_tracer=max(r["maximum_ions"] for r in rs),
                median_cpu_batch_seconds=np.median([r["wall_s"] for r in rs]),
            )
            out.append(row)
    return out


def graph(rows, target):
    """Compare live tracer charge with the two first-event limits."""
    choices = (
        ("opposite", 0),
        ("opposite", 1),
        ("opposite", 4),
        ("bipolar", 4),
        ("same", 4),
        ("neutral", 4),
    )
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, d in zip(axs, (5, 10), strict=True):
        for i, (z, dose) in enumerate(choices):
            sample = [
                r
                for r in rows
                if r["initial_host_charge"] == z
                and r["initial_ions_each_sign_per_host"] == dose
                and r["diameter_nm"] == d
            ]
            if not sample:
                continue
            r = max(sample, key=lambda k: (k["level"], k["batches"]))
            ax.vlines(
                i,
                r["first_event_envelope_lower"],
                r["first_event_envelope_upper"],
                color="#c8bdde",
                lw=9,
                label="First-ion bounds" if i == 0 else None,
            )
            ax.plot(
                i + 0.08,
                r["fixed_tracer_live_hosts_per_1000"],
                "x",
                color="#98543c",
                label="Hold tracer at +1" if i == 0 else None,
            )
            ax.errorbar(
                i - 0.07,
                r["arrivals_per_1000"],
                yerr=r["simultaneous_sampling_margin"],
                fmt="o",
                color="#126458",
                capsize=3,
                label="Follow ion events" if i == 0 else None,
            )
        ax.set(
            xticks=np.arange(len(choices)),
            xticklabels=[
                "−16\n0",
                "−16\n1",
                "−16\n4",
                "±16\n4",
                "+16\n4",
                "0\n4",
            ],
            ylabel="First arrivals per 1,000 births",
            xlabel="Initial host charge / ion pairs per host",
            title=f"Reach {d} nm by 128 s",
        )
        ax.grid(axis="y", alpha=0.2)
    axs[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Positive newborns acquire real ion mass, volume and charge")
    fig.savefig(target, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    p.add_argument("results", type=Path)
    a = p.parse_args()
    rows = score(a.cache, a.results)
    if rows:
        write(a.results / "newborn-repeated-ion-paths.csv", rows)
        graph(rows, a.results / "newborn-repeated-ion-paths.png")
