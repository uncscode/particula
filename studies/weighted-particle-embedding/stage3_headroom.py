"""Measure remaining explicit-particle capacity after the million-box tests."""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from itertools import count, product
from pathlib import Path
from time import perf_counter, time

import matplotlib
import numpy as np
import torch
from stage3_adaptive_box import advance
from stage3_cross_reference import Moments, group, read_charged, summarize
from stage3_evidence import obs, write
from stage3_midpoint import Midpoint

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run(cache, until, *, extension=False):
    """Check free CUDA memory before every trial on a single large box."""
    out = cache / "headroom"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    sizes = (
        (8000000, 10000000)
        if extension
        else (2000000, 4000000, 6000000, 8000000)
    )
    base = 554001 if extension else 553001
    settings = list(product(sizes, ("physical", "learned", "off")))
    for rep in range(12) if extension else count():
        seed = base + rep
        if rep == 0:
            trials = [
                (n, k) for k in ("physical", "off", "learned") for n in sizes
            ]
        else:
            trials = settings.copy()
            np.random.default_rng(seed).shuffle(trials)
        for n, k in trials:
            if time() >= until:
                return
            path = out / f"headroom-{n}-{k}-{seed}.json"
            if path.exists():
                continue
            start = datetime.now(timezone.utc).isoformat()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            # The matched 1M physical path reached 0.532 GiB allocated.
            need = int((0.66 * n / 1e6 + 0.38) * 2**30)
            if extension:
                # Measured at 2, 4 and 6 million: allocator reserves cover
                # about 0.75 GiB per million with pairs and 0.50 without.
                slope = 0.53 if k == "off" else 0.77
                need = int((slope * n / 1e6 + 0.38) * 2**30)
            row = dict(
                n=n,
                species=32,
                seed=seed,
                kernel=k,
                charge="neutral",
                family=307,
                horizon=1.0,
                density=1e12,
                volume_m3=n / 1e12,
                capacity_window="dedicated" if extension else "initial",
            )
            if free < need:
                row.update(status="skipped", reason="device memory reserve")
            else:
                tic = perf_counter()
                try:
                    row.update(
                        advance(
                            n,
                            32,
                            1e12,
                            "neutral",
                            307,
                            seed,
                            k,
                            1.0,
                            0.075,
                            0.05,
                            0.01,
                            fixed=1,
                            box_class=Midpoint,
                        )
                    )
                    row.update(
                        status="completed",
                        synchronized_call_s=perf_counter() - tic,
                        reserved_bytes=torch.cuda.max_memory_reserved(),
                    )
                except (
                    torch.OutOfMemoryError,
                    ArithmeticError,
                    ValueError,
                    RuntimeError,
                ) as exc:
                    row.update(status="failed", reason=str(exc))
            row.update(
                free_bytes_before=free,
                total_device_bytes=total,
                preflight_bytes=need,
                start=start,
                end=datetime.now(timezone.utc).isoformat(),
            )
            tmp = path.with_suffix(".part")
            tmp.write_text(json.dumps(row) + "\n")
            tmp.replace(path)
            print(n, k, seed, row["status"], row.get("total_s", ""), flush=True)


def score(root, dest):
    """Join actual single-box measurements to same-population references."""
    refs = read_charged(root / "headroom-moving", "moving")
    groups = defaultdict(list)
    for p in (root / "headroom").glob("*.json"):
        row = json.loads(p.read_text())
        groups[group(row), row["kernel"]].append(row)
    rows = []
    for (case, k), records in sorted(groups.items()):
        runs = [r for r in records if r["status"] == "completed"]
        skipped = sum(r["status"] == "skipped" for r in records)
        row = dict(
            explicit_particles=case[0],
            vapors=32,
            horizon_s=1.0,
            density_m3=1e12,
            volume_m3=case[0] / 1e12,
            method=k,
            completed=len(runs),
            memory_skips=skipped,
            failures=len(records) - len(runs) - skipped,
            lowest_before_GiB=min(r["free_bytes_before"] for r in records)
            / 2**30,
        )
        for name, key in (
            ("setup_s", "setup_s"),
            ("advance_s", "advance_s"),
            ("setup_and_advance_s", "total_s"),
            ("complete_call_s", "synchronized_call_s"),
            ("peak_allocated_GiB", "cuda_peak_allocated"),
            ("peak_reserved_GiB", "reserved_bytes"),
        ):
            vals = [r[key] for r in runs]
            row[name] = (
                (
                    np.median(vals)
                    if not name.startswith("peak")
                    else max(vals) / 2**30
                )
                if vals
                else ""
            )
        row.update(
            max_relative_species_ledger=max(
                (h["ledger"] for r in runs for h in r["history"]), default=""
            ),
            reference="",
            number_observed_pct="",
            number_upper_pct="",
            phase_observed_pct="",
            phase_upper_pct="",
            qualification="unmeasured",
        )
        if runs and case in refs:
            label, ref, floor = refs[case]
            moments = Moments(ref)
            for r in runs:
                moments.add(obs(r))
            result = summarize(
                moments,
                ref,
                np.zeros_like(ref),
                10**9,
                floor,
                1.0,
                deterministic=k == "off",
            )
            row.update(
                reference=label,
                number_observed_pct=result["number_mean_pct"],
                number_upper_pct=result["number_upper_pct"],
                phase_observed_pct=result["mass_mean_pct"],
                phase_upper_pct=result["mass_upper_pct"],
                qualification=result["decision"],
            )
        rows.append(row)
    if rows:
        write(dest / "explicit-gpu-headroom.csv", rows)
        graph(rows, dest)


def graph(rows, dest):
    """Display measured extra capacity alongside the earlier million case."""
    fig, axs = plt.subplots(1, 2, figsize=(9.9, 4.2), layout="constrained")
    source = dest / "native-overlap-cost.csv"
    base = []
    if source.exists():
        with source.open() as f:
            base = list(csv.DictReader(f))
    for k, label, color in (
        ("physical", "Physical pairs", "#27745a"),
        ("learned", "Neural pairs", "#8d5789"),
        ("off", "Omit pairs", "#8c6935"),
    ):
        line = [r for r in rows if r["method"] == k and r["completed"]]
        points = [
            (
                r["explicit_particles"] / 1e6,
                r["setup_and_advance_s"],
                r["peak_allocated_GiB"],
            )
            for r in line
        ]
        for r in base:
            if (
                r["n"],
                r["species"],
                r["charge"],
                r["family"],
                r["vapor_method"],
                r["kernel"],
                r["steps"],
            ) == ("1000000", "32", "neutral", "307", "midpoint", k, "1"):
                points.append(
                    (
                        1,
                        float(r["median_total_s"]),
                        float(r["max_cuda_peak_allocated"]) / 2**30,
                    )
                )
        points.sort()
        if points:
            x, cost, memory = zip(*points, strict=True)
            axs[0].plot(x, cost, "o-", color=color, label=label)
            axs[1].plot(x, memory, "o-", color=color, label=label)
    for ax, y in zip(
        axs,
        ("Setup + one simulated second (s)", "Peak allocated GPU memory (GiB)"),
        strict=True,
    ):
        ax.set(xlabel="Explicit particles in one box (millions)", ylabel=y)
        ax.grid(alpha=0.2)
    axs[0].legend(frameon=False)
    fig.suptitle("Measured capacity for one neutral box with 32 vapors")
    fig.savefig(dest / "explicit-gpu-headroom.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("run", "score"))
    p.add_argument("cache", type=Path)
    p.add_argument("--results", type=Path)
    p.add_argument(
        "--until", type=lambda x: datetime.fromisoformat(x).timestamp()
    )
    p.add_argument(
        "--extension",
        action="store_true",
        help="Measure 8M and 10M boxes using calibrated GPU memory reserves",
    )
    a = p.parse_args()
    if a.mode == "run":
        if a.until is None:
            p.error("run requires --until")
        run(a.cache, a.until, extension=a.extension)
    else:
        if not a.results:
            p.error("score requires --results")
        score(a.cache, a.results)
