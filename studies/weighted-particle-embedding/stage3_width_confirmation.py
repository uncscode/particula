"""Confirm error and wall time across five distinct vapor widths."""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from time import time

import matplotlib
import numpy as np
import torch
from stage3_adaptive_box import advance
from stage3_cross_reference import Moments, group, read_charged, summarize
from stage3_evidence import obs, write
from stage3_midpoint import Midpoint

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTHS = (1, 2, 8, 16, 32)
SIZES = (65536, 1000000)
LAWS = ("physical", "learned", "off")


def run(root, until):
    """Randomize simultaneous one-box controls within each event seed."""
    folder = root / "width-confirm"
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    for seed in range(556001, 556257):
        cases = list(product(SIZES, WIDTHS, LAWS))
        np.random.default_rng(seed).shuffle(cases)
        for n, s, k in cases:
            if time() >= until:
                return
            path = folder / f"width-{n}-{s}-{k}-{seed}.json"
            if path.exists():
                continue
            start = datetime.now(timezone.utc).isoformat()
            row = advance(
                n,
                s,
                1e12,
                "neutral",
                101,
                seed,
                k,
                1.0,
                0.075,
                0.05,
                0.01,
                fixed=1,
                box_class=Midpoint,
            )
            row.update(
                status="completed",
                start=start,
                end=datetime.now(timezone.utc).isoformat(),
            )
            tmp = path.with_suffix(".part")
            tmp.write_text(json.dumps(row) + "\n")
            tmp.replace(path)
            print(path.name, row["total_s"], flush=True)


def table(root):
    """Score each width against its own refined gas and aerosol equations."""
    refs = read_charged(root / "width-moving", "moving")
    groups = defaultdict(list)
    for scheme, folder, pattern in (
        ("original", "scale", "scaling-*.json"),
        ("midpoint", "width-confirm", "*.json"),
    ):
        for p in (root / folder).glob(pattern):
            row = json.loads(p.read_text())
            if (
                row.get("status", "completed") == "completed"
                and row.get("charge") == "neutral"
                and row.get("family") == 101
                and group(row) in refs
            ):
                groups[scheme, group(row), row["kernel"]].append(row)
    rows = []
    for (scheme, case, kernel), sample in sorted(groups.items()):
        reference, target, change = refs[case]
        moment = Moments(target)
        for r in sample:
            moment.add(obs(r))
        metric = summarize(
            moment,
            target,
            np.zeros_like(target),
            10**9,
            change,
            1.0,
            deterministic=kernel == "off",
        )
        rows.append(
            dict(
                initial_particles=case[0],
                vapors=case[1],
                charge="neutral",
                number_density_m3=1e12,
                family=101,
                horizon_s=1.0,
                method=kernel,
                vapor_integrator=scheme,
                reference=reference,
                runs=len(sample),
                initial_total_vapor_kg_m3=sum(sample[0]["history"][0]["gas"]),
                median_total_s=np.median([x["total_s"] for x in sample]),
                median_device_advance_s=np.median(
                    [
                        x.get("advance_s", x.get("advance_and_observe_s"))
                        for x in sample
                    ]
                ),
                max_allocated_gpu_gib=max(
                    x["cuda_peak_allocated"] for x in sample
                )
                / 2**30,
                max_species_ledger=max(
                    h["ledger"] for x in sample for h in x["history"]
                ),
                **metric,
            )
        )
    return rows


def graph(rows, target):
    """Show costs and both upper errors using the same independent means."""
    fig, axes = plt.subplots(
        2, 3, figsize=(11.2, 6.1), layout="constrained", sharex=True
    )
    for axs, n in zip(axes, SIZES, strict=True):
        for k, name, color in (
            ("physical", "Physical pairs", "#26756b"),
            ("learned", "Neural pairs", "#9c5a43"),
            ("off", "Collisions off", "#685698"),
        ):
            data = sorted(
                (
                    r
                    for r in rows
                    if r["vapor_integrator"] == "midpoint"
                    and r["initial_particles"] == n
                    and r["method"] == k
                ),
                key=lambda r: r["vapors"],
            )
            for ax, field in zip(
                axs,
                ("median_total_s", "number_upper_pct", "mass_upper_pct"),
                strict=True,
            ):
                values = [
                    (r["vapors"], r[field])
                    for r in data
                    if np.isfinite(r[field]) and r[field] > 0
                ]
                if values:
                    x, y = zip(*values, strict=True)
                    ax.plot(x, y, "o-", color=color, label=name, ms=4)
        axs[1].axhline(1, ls="--", color="#555", lw=1)
        axs[2].axhline(0.1, ls="--", color="#555", lw=1)
        for ax, label in zip(
            axs,
            (
                "Wall time (s)",
                "Upper number error (%)",
                "Upper phase mass error (%)",
            ),
            strict=True,
        ):
            ax.set(xscale="log", yscale="log", ylabel=label)
            ax.set_xticks(WIDTHS, [str(s) for s in WIDTHS])
            ax.grid(alpha=0.2)
        axs[0].set_title(f"{n:,} explicit particles")
    for ax in axes[1]:
        ax.set_xlabel("Distinct vapors")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("A one-second pulse at five vapor widths in a neutral box")
    fig.savefig(target, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument(
        "--until", type=lambda s: datetime.fromisoformat(s).timestamp()
    )
    p.add_argument("--score", type=Path)
    a = p.parse_args()
    if a.score:
        rows = table(a.root)
        if rows:
            write(a.score / "vapor-width-speed-and-error.csv", rows)
            if any(r["vapor_integrator"] == "midpoint" for r in rows):
                graph(rows, a.score / "vapor-width-speed-and-error.png")
    elif a.until:
        run(a.root, a.until)
    else:
        p.error("Provide a run deadline or a results folder")
