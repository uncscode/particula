"""Tabulate dry, water and lineage diagnostics of reference parcels."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from campaign import read
from metrics import EDGES, compress, prepare, profile
from model import KAPPA, SEEDS
from stage_two import cohort_sizes, spectrum
from stage_two_design import TIMES, design


def diagnostic(job):
    """Tabulate phase water, dry χ and new-particle size by dry bin."""
    root, case = job
    rows = []
    for t in TIMES:
        boxes, _, _ = read(root / case.key / f"{t}.npz")
        for parcel, b in enumerate(boxes):
            prep = prepare(b, 16)
            chi, counts, _ = profile(b, prep, np.zeros((len(b.old), 1)))
            _, _, _, ends, order = prep
            wet = b.water > 1.05 * 49 * (b.dry @ KAPPA) + 100
            d = np.cbrt(b.dry.sum(1))
            edges = np.r_[0, EDGES[16], np.inf]
            for i, count in enumerate(counts):
                idx = order[ends[i] : ends[i + 1]]
                active = wet[idx]
                young = ~b.old[idx]
                quants = (
                    np.quantile(d[idx[young]], [0.1, 0.5, 0.9])
                    if young.any()
                    else [np.nan] * 3
                )
                # d³-equivalent nm³ -> kg per particle (rho_w = 1000 kg/m³).
                water = b.water[idx] * np.pi / 6 * 1e-24
                rows.append(
                    dict(
                        **asdict(case),
                        seconds=t,
                        parcel=parcel,
                        bin=i,
                        lo_nm=edges[i],
                        hi_nm=edges[i + 1],
                        count=count,
                        chi=chi[i],
                        growing_n=active.sum(),
                        growing_water_kg=water[active].sum(),
                        haze_water_kg=water[~active].sum(),
                        bc_d3_nm3=b.dry[idx, 3].sum(),
                        new_n=young.sum(),
                        pending_lineages=b.pending[idx].sum(),
                        new_q10_nm=quants[0],
                        new_q50_nm=quants[1],
                        new_q90_nm=quants[2],
                    )
                )
    return pd.DataFrame(rows)


def audit_bursts(root, output):
    """Re-evaluate the earlier burst discrepancies at fixed diameters."""
    rows = []
    for seed in SEEDS[8:]:
        boxes, _, t = read(root / str(seed) / "c1b1n1" / "burst-peak.npz")
        for i, b in enumerate(boxes):
            x = cohort_sizes(b)
            compress(b, 10, 16, 2)
            rows.append(
                dict(
                    seed=seed,
                    seconds=t,
                    parcel=i,
                    **spectrum(x, cohort_sizes(b)),
                )
            )
    pd.DataFrame(rows).to_csv(output, index=False, float_format="%.10g")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--previous", type=Path)
    a = p.parse_args()
    with ProcessPoolExecutor(a.workers) as pool:
        data = pd.concat(pool.map(diagnostic, [(a.root, c) for c in design()]))
    data.to_csv(
        a.out / "stage-two-profiles.csv.gz",
        index=False,
        float_format="%.10g",
        compression={"method": "gzip", "mtime": 0},
    )
    if a.previous:
        audit_bursts(a.previous, a.out / "stage-two-burst-audit.csv")
