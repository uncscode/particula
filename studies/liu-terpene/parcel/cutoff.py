# %% Resolve individual activated classes near the cutoff.
"""Challenge sign and magnitude with 60 dedicated probes across each cutoff."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import warp as wp
from parcel import PAIRS, Case, load_survival, simulate, write_csv


def zoom(tables, folder, out):
    """Find the cutoff in a second integration using each saved coarse cut."""
    start = time.monotonic()
    with (folder / "plant/plant.csv").open() as h:
        plants = list(csv.DictReader(h))
    with (folder / "feedback/feedback.csv").open() as h:
        nominees = [
            dict(
                r,
                precursor=r["carrier"],
                kappa=0.1 if r["pair"] == "monoterpenes" else 0.05,
            )
            for r in csv.DictReader(h)
            if r["refined"] == "True" and r["role"] == "water-coupled"
        ]
    rows = []
    for r in plants + nominees:
        d = float(r["original_cutoff_nm"])
        c = Case(
            precursor=r["precursor"],
            rh=r["rh"],
            kappa=float(r["kappa"]),
            w=float(r["w"]),
            n=96,
            probes=64,
            rtol=1.0e-6,
            dz=0.5,
        )
        near = np.linspace(d * 0.975, d * 1.025, 65)
        result = simulate(c, tables, "cuda:0", near)
        if float(result["max_threshold_interval_pp"]) > 0.15:
            raise RuntimeError(f"Zoom missed {c} cutoff {result}")
        rows.append(
            dict(precursor=c.precursor, rh=c.rh, kappa=c.kappa, w=c.w, **result)
        )
    lookup = {(r["precursor"], r["rh"], r["kappa"], r["w"]): r for r in rows}
    comparisons = []
    for pair, (cyc, acy) in PAIRS.items():
        base = 0.1 if pair == "monoterpenes" else 0.05
        for rh in ("dry", "wet"):
            for w in (0.2, 1.0):
                for kc, ka, scenario in (
                    (base, base, "primary"),
                    (0.15, 0.15, "same affinity"),
                    (0.15, 0.07, "reported means"),
                    (0.13, 0.08, "closest 1-SD endpoints"),
                    (0.17, 0.06, "furthest 1-SD endpoints"),
                ):
                    a, b = (
                        lookup[p, rh, k, w] for p, k in [(cyc, kc), (acy, ka)]
                    )
                    comparisons.append(
                        dict(
                            pair=pair,
                            rh=rh,
                            w=w,
                            case=scenario,
                            cyclic_kappa=kc,
                            acyclic_kappa=ka,
                            cyclic_pct=a["fraction_pct"],
                            acyclic_pct=b["fraction_pct"],
                            gain_pp=b["fraction_pct"] - a["fraction_pct"],
                            conservative_grid_bound_pp=a[
                                "max_threshold_interval_pp"
                            ]
                            + b["max_threshold_interval_pp"],
                            cyclic_peak_pct=a["peak_ss_pct"],
                            acyclic_peak_pct=b["peak_ss_pct"],
                            cyclic_d0_nm=a["original_cutoff_nm"],
                            acyclic_d0_nm=b["original_cutoff_nm"],
                        )
                    )
    write_csv(out / "cutoff.csv", rows)
    write_csv(out / "cutoff-comparisons.csv", comparisons)
    (out / "cutoff-runtime.json").write_text(
        json.dumps(
            dict(
                runs=len(rows),
                device=wp.get_device("cuda:0").name,
                wall_seconds=time.monotonic() - start,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("--survival", type=Path, required=True)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    out = args.out or args.folder / "cutoff"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "cutoff.csv").exists():
        raise FileExistsError("Use a new output directory")
    zoom(load_survival(args.survival), args.folder, out)
