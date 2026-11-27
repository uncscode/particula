"""Compare event-order bounds, direct tracer samples and clock refinement."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stage3_evidence import write


def comparison_sources(results):
    """Read first-event bounds and the finest available tracer ensembles."""
    envelopes = {}
    with (results / "newborn-finite-ion-survival.csv").open() as src:
        for r in csv.DictReader(src):
            if float(r["background_m3"]) == 1e10:
                envelopes[
                    r["initial_host_charge"],
                    float(r["initial_ions_each_sign_per_host"]),
                    float(r["diameter_nm"]),
                ] = r
    paths = {}
    source = results / "newborn-repeated-ion-paths.csv"
    if source.exists():
        with source.open() as src:
            for r in csv.DictReader(src):
                key = (
                    r["initial_host_charge"],
                    float(r["initial_ions_each_sign_per_host"]),
                    float(r["diameter_nm"]),
                )
                ranking = int(r["level"]), int(r["batches"])
                if key not in paths or ranking > paths[key][0]:
                    paths[key] = ranking, r
    return envelopes, paths


def score(cache, results):
    """Evaluate deterministic and calibrated arrival estimates independently."""
    envelopes, paths = comparison_sources(results)
    archive = {}
    for p in (cache / "newborn-ion-backward").glob("*.json"):
        for r in json.loads(p.read_text()):
            key = (
                r["initial_host_charge"],
                r["initial_ions_each_sign_per_host"],
                r["diameter_nm"],
            )
            archive[key, r["event_order"], r["level"]] = r
    out = []
    for (key, order, level), r in sorted(archive.items()):
        e = envelopes[key]
        target = float(e["per_1000_lower"])
        change = float(e["lower_adjacent_change_per_1000"])
        correction = target - r["zero_event_per_1000"]
        lo = r["direct_lower_per_1000"] + correction
        hi = r["direct_upper_per_1000"] + correction
        co = archive.get((key, order, level - 1))
        diff = None
        if co:
            d0 = co["zero_event_per_1000"] - r["zero_event_per_1000"]
            diff = max(
                abs(r[k] - co[k] + d0)
                for k in ("direct_lower_per_1000", "direct_upper_per_1000")
            )
        history = paths.get(key)
        monte = mar = clock = None
        if history:
            a = history[1]
            monte = float(a["arrivals_per_1000"])
            mar = float(a["simultaneous_sampling_margin"])
            clock = (
                float(a["paired_clock_change_upper_per_1000"])
                if a["paired_clock_change_upper_per_1000"]
                else None
            )
        out.append(
            dict(
                **r,
                first_event_refined_zero_ion_contact_per_1000=target,
                zero_event_quadrature_correction=correction,
                calibrated_lower_per_1000=lo,
                calibrated_upper_per_1000=hi,
                calibrated_adjacent_clock_change=diff,
                external_first_event_clock_change=change,
                finest_mc_arrivals_per_1000=monte,
                mc_sampling_margin=mar,
                paired_mc_clock_allowance=clock,
                calibrated_interval_overlaps_mc_with_refinements=lo
                - (diff + change)
                <= monte + mar + clock
                and hi + (diff + change) >= monte - mar - clock
                if diff is not None and clock is not None
                else "",
            )
        )
    return out


def graph(rows, output):
    """Show event and time resolution for the strong attractive-host trial."""
    rows = [
        r
        for r in rows
        if r["initial_host_charge"] == "opposite"
        and r["initial_ions_each_sign_per_host"] == 4
        and r["diameter_nm"] == 10
    ]
    if not rows:
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.6, 4.4), layout="constrained")
    for k in (2, 3, 4):
        v = sorted(
            (r for r in rows if r["event_order"] == k),
            key=lambda r: r["time_step_s"],
        )
        if not v:
            continue
        a.plot(
            [r["time_step_s"] for r in v],
            [r["calibrated_lower_per_1000"] for r in v],
            "o-",
            label=f"Up to {k} ions",
        )
    latest = next(
        (r for r in rows if r["finest_mc_arrivals_per_1000"] is not None), None
    )
    if latest:
        m = latest["finest_mc_arrivals_per_1000"]
        margin = latest["mc_sampling_margin"]
        a.axhspan(
            m - margin,
            m + margin,
            color="#8ab8a5",
            alpha=0.35,
            label="Monte Carlo interval",
        )
    a.set(
        xscale="log",
        xlabel="Integration step (s)",
        ylabel="Arrivals per 1,000 births",
        title="Arrival estimate after zero-event correction",
    )
    a.invert_xaxis()
    for level in sorted({r["level"] for r in rows}):
        v = sorted(
            (r for r in rows if r["level"] == level),
            key=lambda r: r["event_order"],
        )
        b.plot(
            [r["event_order"] for r in v],
            [max(r["event_order_width_per_1000"], 1e-12) for r in v],
            "o-",
            label=f"{v[0]['time_step_s']:g} s",
        )
    b.set(
        yscale="log",
        xlabel="Maximum explicit ion attachments",
        ylabel="Residual width per 1,000 births",
        title="Effect of unresolved higher-order events",
        xticks=range(6),
    )
    for ax in (a, b):
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Reach 10 nm with −16 e hosts and four ion pairs per host")
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    p.add_argument("results", type=Path)
    a = p.parse_args()
    rows = score(a.cache, a.results)
    if rows:
        write(a.results / "newborn-backward-ion-survival.csv", rows)
        graph(rows, a.results / "newborn-backward-ion-survival.png")
